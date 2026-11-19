"""Kite primary data, Angel One supplementation, and optional FYERS backup."""
from __future__ import annotations

from datetime import datetime, timedelta
import math
import threading
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from data.angelone_market_data import ANGELONE_INDEX_TARGETS
from data.angelone_readonly import AngelReadOnlyCapabilities
from data.constituent_volume import CONSTITUENT_SESSION_CLOSE, ConstituentVolumeService, INDEX_VENUES


class KiteAngelMigration:
    _replacement_epoch = staticmethod(AngelReadOnlyCapabilities.exchange_epoch)
    def _initialize_replacement(self, settings):
        self._market_data_route = ("upstox" if getattr(self, "upstox_only", False)
                                   else str(getattr(settings, "supplementary_market_data", "fyers")))
        self._replacement_settings = settings
        self._replacement_lock = threading.RLock()
        self._replacement_history = {}
        self._replacement_history_pending = set()
        self._replacement_history_errors = {}
        self._replacement_history_diagnostics = {}
        self._replacement_stock_cache = {}
        self._replacement_day_oi_cache = {}
        self._replacement_stop = threading.Event()
        self._replacement_closed = False
        self._replacement_auxiliary = None
        self._constituent_volume = ConstituentVolumeService(settings.constituent_membership_file) if self._replacement_enabled() else None

    def _replacement_enabled(self):
        if getattr(self, "upstox_only", False):
            return False
        return getattr(self, "_market_data_route", "fyers") in {"upstox_angel", "kite_angel", "angel_fyers"}

    def _fyers_backup_enabled(self):
        return getattr(self, "_market_data_route", "fyers") == "angel_fyers"

    def _fyers_enabled(self):
        return getattr(self, "_market_data_route", "fyers") in {"fyers", "angel_fyers"}

    def _angel_supplementary_quote(self, symbol):
        source = getattr(self, "angel_index", None)
        if source is None or symbol not in ANGELONE_INDEX_TARGETS:
            return None
        cached = dict((getattr(self, "_ltp_cache", {}) or {}).get(symbol) or {})
        age = self._quote_age_components(cached).get("age")
        if str(cached.get("source", "")).startswith("angel") and age is not None and 0 <= age <= 2:
            return cached
        try:
            quote = source.get_index_quote(symbol)
        except Exception:
            return None
        if not isinstance(quote, dict):
            return None
        try:
            price = float(quote.get("ltp") or 0)
            received = float(quote.get("received_at") or time.time())
        except (ValueError, TypeError):
            return None
        if not math.isfinite(price) or price <= 0:
            return None
        source_name = "angel_ws" if quote.get("source") == "angel_ws" else "angel_rest"
        result = dict(quote, time=received, received_at=received, source=source_name)
        ages = self._quote_age_components(result)
        if ages.get("age") is None or not 0 <= ages["age"] <= 6:
            return None
        with self._index_quote_guard():
            self._ltp_cache[symbol] = result
        previous = self._quote_float(quote, "previous_close")
        if math.isfinite(previous) and previous > 0:
            self._remember_previous_close(symbol, previous, source_name)
        return result

    def _emit_three_broker_tick(self, index, price, volume, volume_source, source, received_at, exchange_at):
        """Keep one price owner while independent FYERS volume continues to flow."""
        now = float(received_at) if received_at is not None else time.time()
        source = str(source).lower()
        try:
            price, volume = float(price), float(volume or 0)
        except (ValueError, TypeError):
            return False
        if not math.isfinite(price) or price <= 0 or not math.isfinite(volume) or volume < 0 or not self.is_market_open():
            return False
        group = next((name for name in ("kite", "angel", "fyers") if source.startswith(name)), "")
        if not group:
            return False
        exchange = self._replacement_epoch(exchange_at)
        limit = self.INDEX_KITE_PRICE_FRESH_SECONDS if group == "kite" else self.INDEX_FYERS_PRICE_FRESH_SECONDS
        if exchange is not None and not -1 <= now-exchange <= limit:
            return False
        if group == "kite":
            chosen, selected_price = "kite_ws", price
        else:
            quote, _ = self._select_index_ws_quote(index, now_ts=now)
            if quote is None:
                return False
            selected_source = str(quote.get("source") or "")
            chosen = next((name+"_ws" for name in ("kite", "angel", "fyers") if selected_source.startswith(name)), "")
            selected_price = float(quote.get("ltp") or 0)
        if not chosen or selected_price <= 0:
            return False
        with self._index_quote_guard():
            routes = getattr(self, "_index_price_route", None)
            if routes is None:
                routes = self._index_price_route = {}
            route = routes.setdefault(index, {})
            day = pd.Timestamp(now, unit="s", tz="UTC").tz_convert("Asia/Kolkata").date().isoformat()
            if route.get("session_date") != day:
                route.clear()
                route["session_date"] = day
            if chosen == "kite_ws":
                route.update(active_source=chosen, candidate_source="", candidate_ticks=0,
                             last_authoritative_price=selected_price)
            elif route.get("active_source") != chosen:
                if group+"_ws" != chosen:
                    return False
                if route.get("candidate_source") != chosen:
                    route.update(candidate_source=chosen, candidate_since=now, candidate_ticks=0)
                route["candidate_ticks"] += 1
                if (route["candidate_ticks"] < self.INDEX_PRICE_FAILOVER_MIN_TICKS
                        or now-route["candidate_since"] < self.INDEX_PRICE_FAILOVER_HYSTERESIS_SECONDS):
                    return False
                route.update(active_source=chosen, last_switch_at=now,
                             last_authoritative_price=selected_price, candidate_source="", candidate_ticks=0)
            else:
                route["last_authoritative_price"] = selected_price
            # Lower-priority price-only packets never change the candle.
            if group+"_ws" != chosen and volume <= 0:
                return False
        self._emit_index_tick(index, selected_price, volume, str(volume_source or ""),
                              timestamp=datetime.fromtimestamp(now), native_price=group+"_ws" == chosen)
        return True

    def _start_replacement_stream(self):
        if (not self._replacement_enabled() or not getattr(self, "start_streams", False)
                or getattr(self, "_replacement_closed", False)):
            return False
        source = self.angel_index
        rows = [dict(exchange=exchange, symbol=symbol, token=token,
                     exchange_type=1 if exchange == "NSE" else 3, index=index)
                for index, (exchange, symbol, token) in ANGELONE_INDEX_TARGETS.items()]
        today = pd.Timestamp.now(tz="Asia/Kolkata").date()
        cash_keys = self._constituent_volume.instruments(today, today)
        resolved = source.resolve_market_instruments([*cash_keys, ("NSE", "INDIA VIX")])
        for row in resolved:
            row["exchange_type"] = 1 if row["exchange"] == "NSE" else 3
            row["cash"] = (row["exchange"], row["canonical_symbol"]) in cash_keys
            if row["canonical_symbol"] == "INDIA VIX":
                row["auxiliary"] = "INDIAVIX"
                self._replacement_auxiliary = row
            rows.append(row)
        # Auxiliary names differ between broker masters; use only a verified match.
        if not any(row.get("auxiliary") for row in rows):
            for row in source.resolve_market_instruments([("NSE", "INDIAVIX")]):
                rows.append(dict(row, exchange_type=1, auxiliary="INDIAVIX"))
                self._replacement_auxiliary = row
        if getattr(self, "_replacement_closed", False):
            return False
        return source.start_stream(rows, self._on_replacement_tick)

    def _on_replacement_tick(self, identity, quote):
        if self._replacement_stop.is_set():
            return
        now = float(quote["received_at"])
        if identity.get("cash"):
            volume = quote.get("volume")
            if volume is not None:
                self._constituent_volume.observe(identity["exchange"], identity["canonical_symbol"],
                    float(volume), float(quote["exchange_at"]), now, "angel_ws")
            return
        index = identity.get("index") or identity.get("auxiliary")
        if not index:
            return
        with self._index_quote_guard():
            self._ws_cache["angel:" + index] = dict(quote, volume=0.0, volume_source="")
            if identity.get("index"):
                self._last_angel_ws_tick = now
        if identity.get("index"):
            self._emit_arbitrated_index_tick(index, quote["ltp"], 0.0, "", "angel_ws", now, quote["exchange_at"])

    def _replacement_expiry(self, index):
        from engine.expiry_manager import expiry_manager
        key = "active_monthly" if index in {"BANKNIFTY", "MIDCPNIFTY"} else "active_weekly"
        text = expiry_manager.effective_expiry(index, key)
        parsed = pd.to_datetime(text, errors="coerce") if text else pd.NaT
        return None if pd.isna(parsed) else parsed.date()

    @staticmethod
    def _angel_contract(contract):
        return {"symbol": contract["tradingsymbol"], "exch_seg": contract["exchange"],
                "expiry": contract["expiry"].strftime("%d%b%Y").upper(),
                "strike": contract["strike"] * 100.0,
                **({"_angel_native": True, "name": contract["name"]} if contract.get("_angel_native") else {})}

    def _independent_angel_contracts(self, index="", expiry=None):
        resolver = getattr(getattr(self, "angel_index", None), "option_contracts", None)
        if not callable(resolver):
            return []
        try:
            return resolver(index, expiry=expiry)
        except Exception:
            return []

    def _supplement_reported_day_oi(self, index, expiry, chain, *, allow_fetch=True):
        """Retain reported day changes separately from measured interval deltas.

        Kite/Angel level snapshots cannot reconstruct this native FYERS field at
        an arbitrary startup time. FYERS backs up that specific capability;
        its day changes never masquerade as the current provider's interval delta.
        """
        result = chain.copy()
        result["call_day_oi_change"] = float("nan")
        result["put_day_oi_change"] = float("nan")
        result["day_oi_change_provider"] = "unavailable"
        result["day_oi_change_received_at"] = float("nan")
        if (not self._fyers_backup_enabled() or getattr(self, "fyers", None) is None
                or getattr(self, "_fyers_auth_required", False)):
            return result
        from data.market_data import FYERS_INDEX_SYMBOLS
        now = time.time()
        session = pd.Timestamp(now, unit="s", tz="UTC").tz_convert("Asia/Kolkata").date().isoformat()
        key = (str(index), expiry.isoformat(), session)
        with self._replacement_lock:
            cache = self._replacement_day_oi_cache
            cached = cache.get(key)
            if cached is not None and now < cached["retry_at"]:
                observations = cached.get("observations", {})
                if observations:
                    for side, column in (("CE", "call_day_oi_change"), ("PE", "put_day_oi_change")):
                        result[column] = [observations.get((float(strike), side), float("nan")) for strike in result["strike"]]
                    quoted = result[["call_day_oi_change", "put_day_oi_change"]].notna().any(axis=1)
                    result.loc[quoted, "day_oi_change_provider"] = "fyers"
                    result.loc[quoted, "day_oi_change_received_at"] = cached["received_at"]
                return result
            if not allow_fetch:
                return result
            # Reserve this capability request before releasing the lock. Other
            # polls see unavailable during recovery rather than queueing another
            # full chain call. A failed endpoint also receives a bounded cooldown.
            cache[key] = {"retry_at": now+60., "received_at": float("nan"), "observations": {}}
            while len(cache) > 32:
                cache.pop(next(iter(cache)))
            last_error = float((getattr(self, "_last_opt_chain_error", {}) or {}).get(f"opt_chain_{index}", 0.) or 0.)
            if last_error and now-last_error < 180.:
                cache[key]["retry_at"] = max(now+60., last_error+180.)
                return result
        request = {"symbol": FYERS_INDEX_SYMBOLS.get(index, ""), "strikecount": 30, "timestamp": ""}
        try:
            ok, response = self._call_fyers_bounded("optionchain", data=request, label=f"FYERS day OI backup {index}")
            if not ok or not isinstance(response, dict) or response.get("s") != "ok":
                return result
            expiries = response.get("data", {}).get("expiryData", []) or []
            label = expiry.strftime("%d-%m-%Y")
            target = next((row for row in expiries if row.get("date") == label), None)
            if target is None or not target.get("expiry"):
                return result
            if not expiries or expiries[0].get("date") != label:
                ok, response = self._call_fyers_bounded("optionchain", data=dict(request, timestamp=str(target["expiry"])),
                    label=f"FYERS expiry-aligned day OI backup {index}")
                if not ok or not isinstance(response, dict) or response.get("s") != "ok":
                    return result
            observations = {}
            for row in response.get("data", {}).get("optionsChain", []) or []:
                if row.get("option_type") not in {"CE", "PE"} or "oich" not in row:
                    continue
                observation_key = (float(row["strike_price"]), row["option_type"])
                value = float(row["oich"])
                if observation_key in observations or not math.isfinite(value):
                    return result
                observations[observation_key] = value
            for side, column in (("CE", "call_day_oi_change"), ("PE", "put_day_oi_change")):
                result[column] = [observations.get((float(strike), side), float("nan")) for strike in result["strike"]]
            received = time.time()
            quoted = result[["call_day_oi_change", "put_day_oi_change"]].notna().any(axis=1)
            result.loc[quoted, "day_oi_change_provider"] = "fyers"
            result.loc[quoted, "day_oi_change_received_at"] = received
            with self._replacement_lock:
                self._replacement_day_oi_cache[key] = {"retry_at": received+60.,
                    "received_at": received, "observations": observations}
        except Exception:
            pass
        return result

    def _queue_reported_day_oi(self, index, expiry, chain):
        """Recover an optional capability without holding up preferred chains."""
        if (not self._fyers_backup_enabled() or getattr(self, "fyers", None) is None
                or getattr(self, "_fyers_auth_required", False) or self._replacement_stop.is_set()):
            return
        now = time.time()
        session = pd.Timestamp(now, unit="s", tz="UTC").tz_convert("Asia/Kolkata").date().isoformat()
        cache_key = (str(index), expiry.isoformat(), session)
        with self._replacement_lock:
            workers = self.__dict__.setdefault("_replacement_day_oi_workers", {})
            worker = workers.get(index)
            if worker is not None and worker.is_alive():
                return
            cached = self._replacement_day_oi_cache.get(cache_key)
            if cached is not None and now < cached["retry_at"]:
                return
            snapshot = chain.copy()
            def recover():
                try:
                    report = self._supplement_reported_day_oi(index, expiry, snapshot)
                    with self._replacement_lock:
                        current = self._opt_chain_cache.get(f"opt_chain_{index}")
                        meta = self._option_chain_meta.get(index, {})
                        if (self._replacement_stop.is_set() or current is None or current.empty
                                or meta.get("source") not in {"kite", "angelone"}
                                or current.attrs.get("expiry") != expiry.isoformat()):
                            return
                        updated = current.copy()
                        evidence = report.set_index("strike")
                        for column in ("call_day_oi_change", "put_day_oi_change", "day_oi_change_provider", "day_oi_change_received_at"):
                            updated[column] = evidence[column].reindex(updated["strike"]).to_numpy()
                        self._opt_chain_cache[f"opt_chain_{index}"] = updated
                        self._option_chain_meta[index] = {**meta, "reported_day_oi_available": bool(
                            updated[["call_day_oi_change", "put_day_oi_change"]].notna().all().all())}
                        quality = dict(self._option_chain_meta[index])
                    if updated[["call_day_oi_change", "put_day_oi_change"]].notna().any().any():
                        # Persist newly received capability evidence at its receipt
                        # time; do not refresh the primary quote's cache timestamp.
                        self._archive_option_chain(index, updated, time.time(),
                            {**quality, "capability_update": "reported_day_oi"})
                except Exception as exc:
                    logger.debug(f"Optional FYERS day OI recovery failed for {index}: {type(exc).__name__}")
            worker = threading.Thread(target=recover, name=f"FyersDayOI-{index}", daemon=True)
            workers[index] = self._replacement_day_oi_thread = worker
            worker.start()

    def _get_replacement_option_chain(self, index, now):
        key = f"opt_chain_{index}"
        expiry = self._replacement_expiry(index)
        kite = self.kite
        spot = self.get_latest_price(index)
        if expiry is None or spot <= 0:
            self._last_opt_chain_error[key] = now
            return pd.DataFrame()
        side_count = max(1, min(100, int(self._replacement_settings.replacement_chain_strikes_each_side)))
        # Preserve missing-side evidence: use CE universe and require both legs.
        try:
            strikes = sorted(set(kite.available_strikes(index, "CE", expiry=expiry)))
        except Exception:
            strikes = []
        independent = None
        if not strikes:
            independent = self._independent_angel_contracts(index, expiry)
            strikes = sorted({row["strike"] for row in independent if row["instrument_type"] == "CE"})
        if not strikes:
            self._last_opt_chain_error[key] = now
            return pd.DataFrame()
        centre = min(range(len(strikes)), key=lambda pos: abs(strikes[pos]-spot))
        wanted = strikes[max(0, centre-side_count):centre+side_count+1]
        try:
            chain = kite.get_option_chain(index, spot=spot, strikes_each_side=side_count, expiry=expiry)
        except Exception:
            chain = pd.DataFrame()
        def complete(frame):
            return (frame is not None and not frame.empty and set(frame["strike"]) == set(wanted)
                    and {"call_ltp", "put_ltp"}.issubset(frame.columns)
                    and all(math.isfinite(float(value)) and float(value) > 0 for column in ("call_ltp", "put_ltp")
                            for value in pd.to_numeric(frame[column], errors="coerce")))
        def fresh(frame):
            if frame is None or frame.empty:
                return False
            clocks = [self._replacement_epoch(value) for column in ("call_quote_at", "put_quote_at")
                      for value in frame.get(column, pd.Series([None]*len(frame))).tolist()]
            return bool(clocks and all(stamp is not None and 0 <= time.time()-stamp <= 90 for stamp in clocks))
        def usable_oi(frame):
            if frame is None or frame.empty or not {"call_oi", "put_oi"}.issubset(frame):
                return False
            return bool((pd.to_numeric(frame["call_oi"], errors="coerce").gt(0)
                         & pd.to_numeric(frame["put_oi"], errors="coerce").gt(0)).sum() >= 3)
        complete_volume = self._chain_volume_is_usable
        provider = "kite"
        if not complete(chain) or not fresh(chain) or not usable_oi(chain) or not complete_volume(chain):
            pairs = []
            for strike in wanted:
                try:
                    ce = kite.resolve_option(index, strike, "CE", expiry=expiry)
                    pe = kite.resolve_option(index, strike, "PE", expiry=expiry)
                except Exception:
                    ce = pe = None
                if not ce or not pe:
                    if independent is None:
                        independent = self._independent_angel_contracts(index, expiry)
                    ce = ce or next((row for row in independent if row["strike"] == strike and row["instrument_type"] == "CE"), None)
                    pe = pe or next((row for row in independent if row["strike"] == strike and row["instrument_type"] == "PE"), None)
                if ce and pe:
                    pairs.append((self._angel_contract(ce), self._angel_contract(pe)))
            try:
                observations = self.angel_index.get_option_pairs(index, pairs)
            except Exception:
                observations = []
            records = []
            for pair in observations:
                record = {"strike": pair["strike"]}
                for side, prefix in (("CE", "call"), ("PE", "put")):
                    row = pair[side]
                    record.update({prefix+"_ltp": row["ltp"], prefix+"_oi": row["oi"],
                                   prefix+"_volume": row["volume"], prefix+"_oi_change": 0.0,
                                   prefix+"_volume_available": row.get("volume_available", math.isfinite(float(row["volume"])) and float(row["volume"]) >= 0),
                                   prefix+"_quote_at": row.get("exchange_at"),
                                   prefix+"_symbol": row["symbol"], prefix+"_token": row["token"]})
                records.append(record)
            chain = pd.DataFrame(records)
            provider = "angelone"
        if chain is None or chain.empty:
            self._last_opt_chain_error[key] = now
            return pd.DataFrame()
        chain.attrs["expiry"] = expiry.isoformat()
        chain.attrs["oi_provider"] = provider
        chain = self._apply_option_oi_snapshot_deltas(index, chain, provider=provider, expiry=expiry.isoformat())
        if complete(chain) and fresh(chain) and usable_oi(chain) and complete_volume(chain):
            chain = self._supplement_reported_day_oi(index, expiry, chain, allow_fetch=False)
        paired_oi = (pd.to_numeric(chain["call_oi"], errors="coerce").gt(0) &
                     pd.to_numeric(chain["put_oi"], errors="coerce").gt(0)).sum()
        breadth_complete = bool(complete(chain))
        clocks_fresh = fresh(chain)
        volumes_complete = complete_volume(chain)
        chain.attrs["volume_complete"] = volumes_complete
        fetched = time.time()
        meta = {"source": provider, "oi_provider": provider, "fetched_at": fetched,
                "strike_count": len(chain), "requested_strike_count": len(wanted),
                "coverage_complete": breadth_complete, "fallback": provider != "kite",
                "active_expiry": expiry.isoformat(), "expiry_aligned": True,
                "auto_certified": bool(breadth_complete and clocks_fresh and paired_oi >= 3 and volumes_complete),
                "volume_complete": volumes_complete,
                "exchange_clocks_fresh": clocks_fresh,
                "reported_day_oi_available": bool({"call_day_oi_change", "put_day_oi_change"}.issubset(chain)
                    and chain[["call_day_oi_change", "put_day_oi_change"]].notna().all().all()),
                "oi_change_source": str(chain["oi_change_source"].iloc[0])}
        self._option_chain_meta[index] = meta
        self._opt_chain_cache[key] = chain
        self._last_opt_chain_fetch[key] = fetched
        self._last_opt_chain_error.pop(key, None)
        self._archive_option_chain(index, chain, fetched, meta)
        if meta["auto_certified"]:
            self._queue_reported_day_oi(index, expiry, chain)
        return chain

    def get_replacement_option_premium(self, index, strike, side, expiry, max_age=10.0):
        """An exact-contract fallback, independent of broad-chain TTL."""
        if not self._replacement_enabled() or side not in {"CE", "PE"}:
            return None
        parsed = pd.to_datetime(expiry, errors="coerce")
        if pd.isna(parsed):
            return None
        try:
            contract = self.kite.resolve_option(index, strike, side, expiry=parsed.date())
        except Exception:
            contract = None
        if contract is None:
            contract = next((row for row in self._independent_angel_contracts(index, parsed.date())
                if row["strike"] == float(strike) and row["instrument_type"] == side), None)
            if contract is None:
                return None
        # Depth supplies an exchange quote clock, unlike an un-timestamped LTP.
        try:
            payload = ({} if contract.get("_angel_native") else
                       self.kite.get_depth(contract["exchange"], contract["tradingsymbol"]))
        except Exception:
            payload = {}
        received = time.time()
        stamp = self._replacement_epoch(payload.get("timestamp") or payload.get("time"))
        value = float(payload.get("ltp") or payload.get("last_price") or 0.0)
        if value > 0 and stamp is not None and 0 <= received-stamp <= max_age:
            return value
        identity = self.angel_index._option_identity(index, self._angel_contract(contract), side)
        if identity is None:
            return None
        master = self.angel_index._load_option_master()
        angel = master.get((identity["exchange"], identity["symbol"]))
        if not angel:
            return None
        rows = self.angel_index.get_full_quotes({angel["exchange"]: [angel["token"]]})
        row = rows.get((angel["exchange"], angel["token"])) or {}
        stamp = self._replacement_epoch(row.get("exchange_at"))
        value = float(row.get("ltp") or 0.0)
        return value if math.isfinite(value) and value > 0 and stamp is not None and 0 <= time.time()-stamp <= max_age else None

    def _replacement_auxiliary_quote(self):
        source = self.angel_index
        identity = self._replacement_auxiliary
        if identity is None:
            # Instrument downloads belong to the off-loop prewarm worker.
            # A quote lookup must never wait 20 seconds for the master endpoint.
            self._prewarm_angelone_fallback()
            return None
        rows = source.get_full_quotes({identity["exchange"]: [identity["token"]]})
        row = rows.get((identity["exchange"], identity["token"])) or {}
        stamp = self._replacement_epoch(row.get("exchange_at"))
        value = float(row.get("ltp") or 0.0)
        if not math.isfinite(value) or value <= 0 or stamp is None or not 0 <= time.time()-stamp <= 20:
            return None
        return {"ltp": value, "time": time.time(), "received_at": time.time(), "exchange_at": stamp,
                "source": "angel_rest"}

    def _replacement_touch(self, exchange, symbol, now, max_age):
        source = self.angel_index
        try:
            with self.kite._lock:
                contract = next((row for row in self.kite._instruments.get(exchange, []) if row.get("tradingsymbol") == symbol), None)
        except Exception:
            contract = None
        if contract is None:
            candidates = [row for row in self._independent_angel_contracts() if
                          row["exchange"] == exchange and row["tradingsymbol"] == symbol]
            if not candidates:
                # A registered execution contract is an exact metadata bridge,
                # including BFO names. Never infer expiry/strike from symbol text.
                registered = [identity for identity in list((getattr(self, "_derivative_token_identity", {}) or {}).values())
                              if str(identity[1]).split("|", 1)[0] == symbol]
                if len(registered) == 1:
                    name, _, expiry, side, strike = registered[0]
                    day = pd.to_datetime(expiry, errors="coerce")
                    if not pd.isna(day):
                        candidates = [row for row in self._independent_angel_contracts(name, day.date()) if
                                      row["exchange"] == exchange and row["strike"] == float(strike)
                                      and row["instrument_type"] == side]
            contract = candidates[0] if len(candidates) == 1 else None
        if not contract or contract.get("instrument_type") not in {"CE", "PE"}:
            return None
        identity = source._option_identity(contract["name"], self._angel_contract(contract), contract["instrument_type"])
        if identity is None:
            return None
        angel = source._load_option_master().get((identity["exchange"], identity["symbol"]))
        if not angel:
            return None
        rows = source.get_full_quotes({exchange: [angel["token"]]})
        row = rows.get((exchange, angel["token"])) or {}
        stamp = self._replacement_epoch(row.get("exchange_at"))
        if stamp is None:
            return None
        depth = row.get("depth") or {}
        bid, ask = (depth.get("buy") or [{}])[0], (depth.get("sell") or [{}])[0]
        payload = {"bid": bid.get("price"), "ask": ask.get("price"),
                   "bid_qty": bid.get("quantity"), "ask_qty": ask.get("quantity"),
                   "ltp": row.get("ltp"), "oi": row.get("opnInterest"),
                   "volume": row.get("tradeVolume"), "time": stamp}
        return self._touch_quote_from_payload(payload, "angel_rest_touch", now, max_age)

    def _request_replacement_history(self, index, interval, days_back, to_date, *, persist_cache=True, force_refresh=False):
        """Return cached reconciled data; broker work never runs on the scan loop."""
        target = pd.Timestamp(to_date) if to_date is not None else pd.Timestamp.now(tz="Asia/Kolkata")
        if target.tzinfo is not None:
            target = target.tz_convert("Asia/Kolkata").tz_localize(None)
        key = (index, int(days_back), target.isoformat() if to_date is not None else "latest", bool(persist_cache))
        with self._replacement_lock:
            observed, base = self._replacement_history.get(key, (0.0, pd.DataFrame()))
            refresh = max(30, float(self._replacement_settings.constituent_history_refresh_seconds))
            can_schedule = (len(self._replacement_history_pending) < 4 and
                            not any(pending[0] == index for pending in self._replacement_history_pending))
            if (force_refresh or time.time()-observed >= refresh) and can_schedule and not self._replacement_stop.is_set():
                self._replacement_history_pending.add(key)
                threading.Thread(target=self._build_replacement_history, args=(key, target),
                                 daemon=True, name=f"ConstituentHistory-{index}").start()
        if force_refresh:
            return pd.DataFrame()
        if base.empty:
            return base
        if interval == "1min":
            return base.copy()
        minutes = {"5min": 5, "15min": 15}.get(interval)
        if minutes is None:
            return pd.DataFrame()
        # Validate every minute before reducing its certificate. Dropping member
        # counts here made all otherwise complete 5m/15m totals fail the downstream
        # preferred-history gate and unnecessarily request FYERS.
        base = base.copy()
        verified = self._merge_preferred_index_history(base, pd.DataFrame())
        base["volume_complete"] = verified["volume_complete"].reindex(base.index, fill_value=False)
        # Exact 09:15 alignment; no partial resample may masquerade as complete.
        aggregation = {"open": "first", "high": "max", "low": "min", "close": "last",
                       "volume": lambda values: values.sum(min_count=minutes),
                       "volume_complete": "all", "volume_basis": "last", "source": "last",
                       "membership_version": "last", "volume_provider": "last",
                       "volume_members_observed": "min", "volume_members_expected": "min"}
        aggregation = {column: operation for column, operation in aggregation.items() if column in base}
        result = base.resample(f"{minutes}min", origin="start_day", offset="15min").agg(aggregation)
        counts = base["close"].resample(f"{minutes}min", origin="start_day", offset="15min").count()
        result = result.loc[counts == minutes].dropna(subset=["close"])
        result["volume_complete"] &= result["volume"].notna()
        for column in ("membership_version", "volume_provider", "volume_members_expected"):
            if column in base:
                consistent = base[column].resample(f"{minutes}min", origin="start_day", offset="15min").nunique().eq(1)
                result["volume_complete"] &= consistent.reindex(result.index, fill_value=False)
        result.loc[~result["volume_complete"], "volume"] = float("nan")
        result.loc[~result["volume_complete"], "volume_basis"] = "UNAVAILABLE"
        return result

    @staticmethod
    def _merge_preferred_index_history(preferred, backup):
        """Prefer Kite/Angel OHLC and certified constituent-share volume.

        FYERS fills only missing certified constituent intervals. Unknown
        membership, incomplete stock coverage and provisional totals never qualify.
        """
        preferred = ConstituentVolumeService.local_frame(preferred)
        columns = ["open", "high", "low", "close"]
        if not set(columns).issubset(preferred):
            return backup
        numeric = preferred[columns].apply(pd.to_numeric, errors="coerce")
        valid = numeric.apply(lambda column: column.map(lambda value: pd.notna(value) and math.isfinite(float(value)) and value > 0)).all(axis=1)
        preferred = preferred.loc[valid]
        if preferred.empty:
            return backup
        if backup is None or backup.empty:
            result = pd.DataFrame(index=preferred.index)
        else:
            backup = ConstituentVolumeService.local_frame(backup)
            result = backup.reindex(backup.index.union(preferred.index))
        # Empty metadata cells reload from CSV as float64. Pandas 3 refuses
        # assigning broker labels/booleans into those numeric columns. Declare
        # metadata storage explicitly without changing numeric OHLC/volume units.
        for column in ("source", "price_provider", "volume_basis", "volume_provider", "membership_version"):
            if column in result:
                result[column] = result[column].astype(object)
            else:
                result[column] = pd.Series(index=result.index, dtype=object)
        if "volume_complete" in result:
            result["volume_complete"] = result["volume_complete"].astype(object)
        for column in columns:
            result.loc[preferred.index, column] = preferred[column]
        if "volume" not in result:
            result["volume"] = float("nan")
        if "volume_basis" not in result:
            result["volume_basis"] = "UNAVAILABLE"
        basis = result["volume_basis"].fillna("UNAVAILABLE")
        measured = basis.eq("fyers_index") & pd.to_numeric(result["volume"], errors="coerce").ge(0)
        if "volume_complete" in result:
            measured &= ~result["volume_complete"].eq(False).fillna(False)
        certified = pd.Series(False, index=preferred.index)
        evidence = {"volume", "volume_complete", "volume_basis", "volume_members_observed",
                    "volume_members_expected", "membership_version", "volume_provider"}
        if evidence.issubset(preferred):
            quantities = pd.to_numeric(preferred["volume"], errors="coerce")
            expected = pd.to_numeric(preferred["volume_members_expected"], errors="coerce")
            observed = pd.to_numeric(preferred["volume_members_observed"], errors="coerce")
            certified = (preferred["volume_complete"].eq(True)
                & preferred["volume_basis"].eq("constituent_shares")
                & quantities.map(lambda value: pd.notna(value) and math.isfinite(float(value)) and value >= 0)
                & expected.gt(0) & observed.eq(expected)
                & preferred["membership_version"].fillna("").ne("")
                & preferred["volume_provider"].isin(["kite", "angelone"]))
        fallback_index = certified[certified].index
        measured &= ~result.index.isin(fallback_index)
        if len(fallback_index):
            for column in evidence:
                result.loc[fallback_index, column] = preferred.loc[fallback_index, column]
        constituent = pd.Series(result.index.isin(fallback_index), index=result.index)
        result["volume_complete"] = measured | constituent
        result["volume_provider"] = "unavailable"
        result.loc[measured, "volume_provider"] = "fyers"
        if len(fallback_index):
            result.loc[fallback_index, "volume_provider"] = preferred.loc[fallback_index, "volume_provider"]
        result.loc[~(measured | constituent), "volume"] = float("nan")
        result.loc[~(measured | constituent), "volume_basis"] = "UNAVAILABLE"
        providers = preferred.get("source", pd.Series("kite_history", index=preferred.index)).map(
            lambda source: "kite" if str(source).startswith("kite") else "angelone")
        result.loc[preferred.index, "price_provider"] = providers
        result.loc[preferred.index, "source"] = providers.map(lambda source: source+"_history")
        result.index.name = "timestamp"
        return result.sort_index()

    def _replacement_rest_pressure(self):
        """Seconds left in Kite's own cooldown for the candle endpoint, or 0.0.

        A throttled ``historical_data`` call answers with an empty frame — the same
        thing a symbol that never traded answers with — so the builder has to read
        the broker's own verdict to tell a refusal from an absence. Sources that do
        not publish that verdict (lightweight probes, fakes) report no pressure, which
        is exactly the previous behaviour.
        """
        probe = getattr(self.kite, "rest_throttle_remaining", None)
        if not callable(probe):
            return 0.0
        try:
            return max(0.0, float(probe("historical_data")))
        except Exception:
            return 0.0

    def _replacement_member_history(self, cache_key, symbol, exchange, stock_days, target):
        """One constituent's 1-minute history, boosted past a broker refusal.

        A failed fetch and a stock that simply did not trade both arrive as an empty
        frame, so caching one as the other silently drops the member: ``aggregate``
        then records ``observed=29`` of ``expected=30``, every interval of that index
        becomes UNAVAILABLE, and the index request falls through to the backup even
        though the data was available. The request's own verdict (``call_outcome``)
        is the authoritative signal — a provider that does not publish one falls back
        to the broker's measured cooldown. Either way a refusal is retried and never
        cached; only a real answer, including a genuine empty one, is.

        Returns ``None`` when the broker kept refusing or the provider is shutting
        down, so the caller can name the member instead of counting it as observed.
        """
        cached_at, frame = self._replacement_stock_cache.get(cache_key, (0.0, None))
        if frame is not None and time.time()-cached_at <= 60:
            return frame
        for attempt in range(3):
            outcome: dict = {}
            try:
                stock_token = self.kite.instrument_token(symbol, exchange) or self.kite.instrument_token(symbol+"-EQ", exchange)
            except Exception:
                stock_token = None
            if not stock_token:
                frame = None      # No instrument resolved: a refusal, not an absence.
            else:
                try:
                    frame = self.kite.get_historical(stock_token, "1min", days_back=stock_days,
                                                     to_date=target, call_outcome=outcome)
                except TypeError:
                    frame, outcome = self._replacement_member_fetch(stock_token, stock_days, target)
                except Exception:
                    frame, outcome = None, {"ok": False, "reason": "fetch raised"}
            refused = (frame is None or outcome.get("ok") is False
                       or (not outcome and self._replacement_rest_pressure() > 0))
            if not refused:
                self._replacement_stock_cache[cache_key] = (time.time(), frame)
                while len(self._replacement_stock_cache) > 512:
                    self._replacement_stock_cache.pop(next(iter(self._replacement_stock_cache)))
                self._replacement_stop.wait(0.4)
                return frame
            # Re-fire only after the window has drained, never into it.
            if attempt < 2 and self._replacement_stop.wait(min(9.0, self._replacement_rest_pressure()+0.25)):
                return None
        return None

    def _replacement_member_fetch(self, stock_token, stock_days, target):
        """Candle fetch for providers whose ``get_historical`` has no ``call_outcome``.

        Lightweight probes expose only the historical call itself, so a raised
        exception is the only refusal they can report.
        """
        try:
            return self.kite.get_historical(stock_token, "1min", days_back=stock_days, to_date=target), {}
        except Exception:
            return None, {"ok": False, "reason": "fetch raised"}

    def _build_replacement_history(self, key, target):
        index, days_back, _, persist_cache = key
        try:
            # This worker is off-loop. Recover missing/expired publisher snapshots
            # before asking constituent history to certify any interval.
            if (getattr(self, "start_streams", False)
                    and target.date() == pd.Timestamp.now(tz="Asia/Kolkata").date()
                    and self._constituent_volume.membership(index, target.date()) is None):
                from data.index_membership import ensure_current_membership
                try:
                    ensure_current_membership(self._constituent_volume.path)
                    self._constituent_volume.load()
                except Exception:
                    # FYERS history remains the established cash-volume backup.
                    logger.warning("Authoritative constituent membership unavailable for {}", index)
            start = target-timedelta(days=max(1, days_back))
            cache_file = Path(self._replacement_settings.constituent_cache_dir)/f"{index}_1min.csv"
            with self._replacement_lock:
                needs_restore = key not in self._replacement_history
            if persist_cache and needs_restore and cache_file.exists():
                try:
                    restored = ConstituentVolumeService.local_frame(pd.read_csv(cache_file, index_col=0, parse_dates=True))
                except (ValueError, TypeError, OSError):
                    restored = pd.DataFrame(index=pd.DatetimeIndex([]))
                restored = restored.loc[(restored.index >= start) & (restored.index < target)]
                # Revalidate membership after restart; expired or revised lists must
                # not inherit a previous process's coverage certificate.
                for stamp, row in restored.iterrows():
                    membership = self._constituent_volume.membership(index, stamp.date())
                    if membership is None or str(row.get("membership_version")) != membership.version:
                        restored.at[stamp, "volume"] = float("nan")
                        restored.at[stamp, "volume_complete"] = False
                        restored.at[stamp, "volume_basis"] = "UNAVAILABLE"
                if not restored.empty:
                    with self._replacement_lock:
                        self._replacement_history[key] = (0.0, restored)
            try:
                token = self._kite_token_for_index(index)
                base = self.kite.get_historical(int(token), "1min", days_back=days_back, to_date=target) if token else pd.DataFrame()
            except Exception:
                base = pd.DataFrame()
            price_provider = "kite"
            if base is None or base.empty:
                exchange, _, angel_token = ANGELONE_INDEX_TARGETS[index]
                base = self.angel_index.get_history(exchange, angel_token, "1min", start, target)
                price_provider = "angel"
            if base is None or base.empty:
                raise ValueError("index_history_unavailable")
            base = ConstituentVolumeService.local_frame(base)
            now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
            base = base.loc[(base.index >= start) & (base.index <= target) &
                            (base.index+pd.Timedelta(minutes=1) <= min(target, now))].between_time("09:15", "15:29")
            service = self._constituent_volume
            dated = pd.Series([service.membership(index, stamp.date()) is not None
                               for stamp in base.index], index=base.index, dtype=bool)
            # Certification only ever covers intervals a cash constituent supplies.
            # The index prints to 15:29 while constituents stop at 15:14, so
            # counting the tail as a coverage failure made every index look
            # permanently uncertified and sent each request to the FYERS backup.
            session_close = base.index.normalize()+pd.Timedelta(hours=CONSTITUENT_SESSION_CLOSE[0],
                                                                minutes=CONSTITUENT_SESSION_CLOSE[1])
            certifiable = dated & pd.Series(base.index+pd.Timedelta(minutes=1) <= session_close,
                                            index=base.index, dtype=bool)
            covered = base.index[dated]
            # Keep the full index-price window, but fetch stock history only where
            # dated membership can certify a constituent interval. Current-day
            # publisher snapshots do not justify downloading 15 days per stock.
            stock_start = covered.min() if len(covered) else target
            stock_days = max(1, math.ceil((target-stock_start).total_seconds()/86400))
            members = sorted({member for day in {stamp.date() for stamp in covered}
                              for member in service.membership(index, day).members})
            frames = {}
            unavailable = {}
            # Serialize all constituent workers. Four indices share one recovery lane.
            with self._replacement_recovery_guard():
                for exchange, symbol in members:
                    if self._replacement_stop.is_set():
                        return
                    cache_key = (exchange, symbol, stock_days, target.isoformat() if key[2] != "latest" else target.date())
                    frame = self._replacement_member_history(cache_key, symbol, exchange, stock_days, target)
                    if frame is None:
                        unavailable[(exchange, symbol)] = "historical REST throttled"
                        continue
                    frames[(exchange, symbol)] = frame
                volume = service.aggregate(index, base.index, frames, "kite", now=min(target, now))
                # An undated historical membership cannot be repaired by asking a
                # second broker. Recover only intervals whose membership is known.
                if members and (certifiable & ~volume["volume_complete"]).any():
                    resolved = self.angel_index.resolve_market_instruments(members)
                    angel_frames = {}
                    for identity in resolved:
                        if self._replacement_stop.is_set():
                            return
                        angel_frames[(identity["exchange"], identity["canonical_symbol"])] = self.angel_index.get_history(
                            identity["exchange"], identity["token"], "1min", stock_start, target)
                    fallback = service.aggregate(index, base.index, angel_frames, "angelone", now=min(target, now))
                    use = ~volume["volume_complete"] & fallback["volume_complete"]
                    volume.loc[use, :] = fallback.loc[use, :]
                # "Coverage incomplete" has always been the message; naming the members
                # is what makes it actionable, and it distinguishes a throttled fetch
                # from a symbol that genuinely traded nothing.
                dated_rows = volume.loc[certifiable] if certifiable.any() else volume.iloc[:0]
                self._replacement_history_diagnostics[index] = {
                    "members_expected": int(pd.to_numeric(dated_rows["volume_members_expected"], errors="coerce").max()) if len(dated_rows) else 0,
                    "members_observed": int(pd.to_numeric(dated_rows["volume_members_observed"], errors="coerce").min()) if len(dated_rows) else 0,
                    "intervals_unavailable": int((certifiable & ~volume["volume_complete"]).sum()),
                    "members_unavailable": {f"{exchange}:{symbol}": reason for (exchange, symbol), reason in sorted(unavailable.items())},
                }
            base = base.drop(columns=[column for column in volume.columns if column in base], errors="ignore").join(volume)
            base["source"] = price_provider+"_constituent_history"
            base.index.name = "timestamp"
            if persist_cache and not self._replacement_stop.is_set():
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_file.with_suffix(".tmp")
                base.to_csv(temporary)
                temporary.replace(cache_file)
            with self._replacement_lock:
                self._replacement_history[key] = (time.time(), base)
                while len(self._replacement_history) > 32:
                    self._replacement_history.pop(next(iter(self._replacement_history)))
                incomplete = int((certifiable & ~base["volume_complete"]).sum())
                self._replacement_history_errors[index] = ("" if certifiable.any() and not incomplete
                                                           else "constituent_coverage_incomplete")
        except Exception:
            with self._replacement_lock:
                self._replacement_history_errors[index] = "history_recovery_failed"
            logger.warning("Kite/Angel history recovery unavailable for {}", index)
        finally:
            with self._replacement_lock:
                self._replacement_history_pending.discard(key)

    def _replacement_recovery_guard(self):
        # Initialized before any worker is launched; lazy only for lightweight probes.
        with self._replacement_lock:
            if not hasattr(self, "_replacement_recovery_lock"):
                self._replacement_recovery_lock = threading.Lock()
            return self._replacement_recovery_lock

    def _replacement_health(self):
        if not self._replacement_enabled():
            return {"route": "fyers", "independent_broker_sources": 3}
        with self._replacement_lock:
            errors = dict(self._replacement_history_errors)
        auxiliary = self._freshest_cached_index_quote(("angel:INDIAVIX",), 20.0, time.time())
        backup = self._fyers_backup_enabled()
        return {"route": "angel_fyers" if backup else "kite_angel", "independent_broker_sources": 3 if backup else 2,
                "source_priority": ["kite", "angelone", "fyers"] if backup else ["kite", "angelone"],
                "backup_provider": "fyers" if backup else None,
                "third_provider_redundancy_preserved": backup,
                "cash_volume_provider": "constituent_shares",
                "cash_volume_source_priority": ["kite_constituent_shares", "angelone_constituent_shares"] + (["fyers_index"] if backup else []),
                "capability_backups": {"cash_index_volume": "fyers", "reported_day_oi_change": "fyers",
                                       "preopen_settlement": "fyers"} if backup else {},
                "membership": self._constituent_volume.health(), "history_errors": errors,
                "history_diagnostics": {index: dict(report)
                                        for index, report in self._replacement_history_diagnostics.items()},
                "vix_fresh": auxiliary is not None,
                "chains": {index: dict((getattr(self, "_option_chain_meta", {}) or {}).get(index) or {}) for index in INDEX_VENUES},
                "parity_verified": False, "cutover_ready": False}
