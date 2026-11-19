"""Pure broker-source health and latency snapshot calculations."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Mapping, Optional


def build_source_health(
    provider: Any,
    now_ts: Optional[float] = None,
    freshness_threshold: float = 45.0,
) -> Dict[str, Any]:
    now = time.time() if now_ts is None else float(now_ts)
    is_live = provider.is_market_open()
    fyers_tick = float(getattr(provider, "_last_fyers_ws_tick", 0.0) or 0.0)
    fyers_age = None if fyers_tick <= 0 else max(0.0, now - fyers_tick)
    fyers_fresh = fyers_age is not None and fyers_age < freshness_threshold

    # Kite health is session-based FIRST: a valid access token is what determines
    # whether orders work, and the stream being quiet must never read as "broker
    # down". Tick age is reported alongside so a stalled socket is still visible.
    # Upstox Primary Broker Health
    upstox_client = getattr(provider, "upstox", None)
    upstox_healthy = False
    upstox_session_live = False
    if upstox_client is not None:
        try:
            from data.upstox_auth import get_upstox_auth_status
            upstox_st = get_upstox_auth_status()
            upstox_healthy = bool(upstox_st.get("authenticated")) and upstox_client.is_healthy()
            upstox_session_live = bool(upstox_st.get("authenticated"))
        except Exception:
            pass

    kite_tick = float(getattr(provider, "_last_kite_ws_tick", 0.0) or 0.0)
    kite_age = None if kite_tick <= 0 else max(0.0, now - kite_tick)
    kite_health = bool(provider.is_connected) or upstox_healthy
    # provider.is_connected is a plain attribute assigned at connect()/rebind_kite() and
    # never revoked, so a token invalidated mid-session leaves it True forever.
    # KiteDataSource.is_connected is a live property (`auth.kite is not None`) and
    # _mark_auth_required sets that to None, so it does track a dead session.
    #
    # Reported ALONGSIDE rather than folded into kite_health on purpose: kite_health
    # feeds all_brokers_unavailable, which gates REAL mode, and tightening a trading
    # gate on an observability defect risks refusing to trade through a transient auth
    # blip. The authoritative recovery path already exists (main.py's RuntimeWatchdog
    # polls kite_auth.status()["auth_required"] and raises a latched alert); this makes
    # the same fact visible in the health snapshot the rest of the system reads.
    kite_session_live = bool(getattr(getattr(provider, "kite", None), "is_connected", False))
    fyers_available = provider.fyers is not None and not provider._fyers_auth_required
    fyers_health = bool(fyers_available and (fyers_fresh if is_live else True))
    route = getattr(provider, "_market_data_route", "fyers")
    replacement = route in {"kite_angel", "angel_fyers"}
    backup_enabled = route == "angel_fyers"
    if backup_enabled:
        fyers_rest_age = _newest_source_age(getattr(provider, "_ltp_cache", None), "fyers", now)
        fyers_health = bool(fyers_health or (fyers_available and fyers_rest_age is not None and fyers_rest_age < freshness_threshold))
    angel_tick = float(getattr(provider, "_last_angel_ws_tick", 0.0) or 0.0)
    angel_age = now - angel_tick if angel_tick > 0 else None
    angel_rest_age = _newest_source_age(getattr(provider, "_ltp_cache", None), "angel_rest", now)
    angel_fresh = bool((angel_age is not None and 0 <= angel_age < freshness_threshold)
                       or (angel_rest_age is not None and angel_rest_age < freshness_threshold))
    supplementary_health = (angel_fresh or fyers_health) if backup_enabled else (angel_fresh if replacement else fyers_health)

    # "Can we place an order" and "are we still receiving prices" are two questions, and
    # only the first one was ever asked. is_connected is set to `auth.kite is not None`
    # at connect time and never revoked, so kite_health is effectively a constant True
    # for the life of the process: broker_degraded could only ever fire on the Fyers
    # half, and all_brokers_unavailable — the preflight BLOCKER that arms the Yahoo
    # emergency path — required `not kite_health` and therefore could not fire at all.
    # kite_age was computed and reported for exactly this and then read by nobody.
    #
    # So the data question is asked separately here. A stalled websocket is not
    # necessarily a dead feed (the REST LTP path keeps serving), which is why a recent
    # REST fill still counts as fresh, and why none of this touches kite_health itself.
    # "No tick has EVER arrived" is not the same claim as "ticks stopped arriving", and
    # conflating them breaks this codebase's central invariant in the worst place: at a
    # cold start inside market hours neither socket has ticked yet, so a first draft of
    # this reported all_brokers_unavailable — a REAL-mode preflight BLOCKER — and the
    # system refused to start at 09:15 until a tick happened to land first.
    #
    # A socket with no tick yet is UNKNOWN until we have actually waited. With no start
    # time to measure that from, it stays unknown rather than becoming an accusation.
    uptime = _uptime_seconds(provider, now)
    kite_stream_stalled = bool(
        is_live
        and (
            kite_age >= freshness_threshold
            if kite_age is not None
            else (uptime is not None and uptime >= freshness_threshold)
        )
    )
    kite_rest_age = _newest_source_age(getattr(provider, "_ltp_cache", None), "kite", now)
    kite_data_fresh = bool(
        not is_live
        or not kite_stream_stalled
        or (kite_rest_age is not None and kite_rest_age < freshness_threshold)
    )
    all_brokers_unavailable = bool(
        is_live and not (upstox_healthy or (kite_health and kite_data_fresh)) and not supplementary_health
    )
    yahoo_mode = (
        "emergency_used"
        if all_brokers_unavailable and provider._yahoo_request_count
        else "idle_standby"
    )

    historical_source = getattr(provider, "kite", None)
    historical_status = {
        "kite_historical_rest_source": "kite_historical_rest",
        "kite_historical_rest_available": False,
        "kite_historical_rest_healthy": False,
        "kite_historical_rest_fresh": False,
        "kite_historical_rest_stale": False,
        "kite_historical_rest_last_call_ok": None,
        "kite_historical_rest_max_age_seconds": None,
        "kite_historical_rest_last_attempt_at": None,
        "kite_historical_rest_last_attempt_age_seconds": None,
        "kite_historical_rest_last_success_at": None,
        "kite_historical_rest_age_seconds": None,
        "kite_historical_rest_last_error_at": None,
        "kite_historical_rest_last_error_age_seconds": None,
        "kite_historical_rest_failure": "not available",
        "kite_historical_rest_last_error": "",
        "kite_historical_rest_failures": 0,
    }
    status_fn = getattr(historical_source, "historical_rest_status", None)
    if callable(status_fn):
        try:
            try:
                observed = status_fn(now_ts=now)
            except TypeError:
                observed = status_fn()
            historical_status.update(dict(observed or {}))
        except Exception:
            historical_status["kite_historical_rest_failure"] = "health unavailable"

    result = {
        "kite": kite_health,
        # Live order capability. `kite` above is the latched connect-time flag and stays
        # True after a mid-session token death; this one goes False with the session.
        "kite_session_live": kite_session_live,
        "kite_tick_age_seconds": round(kite_age, 2) if kite_age is not None else None,
        "kite_stream_stalled": kite_stream_stalled,
        "kite_data_fresh": kite_data_fresh,
        "kite_rest_age_seconds": round(kite_rest_age, 2) if kite_rest_age is not None else None,
        "fyers": fyers_health,
        "fyers_tick_age_seconds": round(fyers_age, 2) if fyers_age is not None else None,
        "angelone": angel_fresh,
        "angelone_tick_age_seconds": round(angel_age, 2) if angel_age is not None else None,
        "supplementary_provider": "angelone" if replacement else "fyers",
        "backup_provider": "fyers" if backup_enabled else ("angelone" if not replacement else None),
        "backup_degraded": bool(is_live and backup_enabled and not fyers_health),
        "broker_degraded": bool(
            is_live and (
                (not upstox_healthy)
                if upstox_client is not None
                else not (kite_health and kite_data_fresh and (angel_fresh if replacement else fyers_health))
            )
        ),
        "all_brokers_unavailable": all_brokers_unavailable,
        "fyers_auth_required": provider._fyers_auth_required,
        "fyers_auth_reason": provider._fyers_auth_reason,
        "fyers_refresh_disabled": provider._fyers_refresh_disabled,
        "yahoo": True,
        "yahoo_mode": yahoo_mode,
        "yahoo_request_count": int(provider._yahoo_request_count),
        "yahoo_last_used_at": (
            datetime.fromtimestamp(provider._yahoo_last_used_at).isoformat()
            if provider._yahoo_last_used_at
            else None
        ),
    }
    # Candle REST is a separate capability. Never let it rewrite websocket/LTP
    # freshness, which is used for live price and order-authority decisions.
    result.update(historical_status)
    return result


def _uptime_seconds(provider: Any, now_ts: float) -> Optional[float]:
    """Seconds since this provider came up, or None if it cannot say.

    Used to tell "we have waited and nothing came" apart from "we only just started".
    """
    for attr in ("_stream_started_at", "_provider_started_at", "_connected_at"):
        try:
            started = float(getattr(provider, attr, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if started > 0:
            return max(0.0, now_ts - started)
    return None


def _newest_source_age(cache: Any, source: str, now_ts: float) -> Optional[float]:
    """Age of the freshest LTP this source filled, or None if it has filled none."""
    if not isinstance(cache, Mapping):
        return None
    youngest: Optional[float] = None
    for value in cache.values():
        if not isinstance(value, dict) or str(value.get("source") or "") != source:
            continue
        age = _age_seconds(value, now_ts)
        if age is not None and (youngest is None or age < youngest):
            youngest = age
    return youngest


def _age_seconds(value: Any, now_ts: float) -> Optional[float]:
    try:
        timestamp = value.get("time") if isinstance(value, dict) else None
        if isinstance(timestamp, datetime):
            # now_ts, not time.time(): every other age here honours the injected clock,
            # and this branch silently ignoring it made the datetime path untestable.
            return max(0.0, now_ts - timestamp.timestamp())
        if timestamp:
            return max(0.0, now_ts - float(timestamp))
    except Exception:
        return None
    return None


def _source_record(
    name: str,
    value: Any,
    latency_class: str,
    entry_eligible: bool,
    now_ts: float,
) -> Dict[str, Any]:
    age = _age_seconds(value, now_ts)
    # Order authority means "this feed comes from the broker we actually trade
    # through", which is Kite now. Leaving it matching "angel" marked every source as
    # non-authoritative once AngelOne was retired.
    has_order_authority = str(name or "").startswith("upstox") or str(name or "").startswith("kite")
    return {
        "source": name,
        "latency_class": latency_class,
        "age_seconds": round(age, 2) if age is not None else None,
        "order_authority": bool(has_order_authority),
        "entry_eligible": bool(entry_eligible and has_order_authority),
        "ltp": float(value.get("ltp", 0.0)) if isinstance(value, dict) else 0.0,
    }


def build_market_source_snapshot(
    provider: Any,
    symbol: str,
    index_spot_tokens: Mapping[str, Any],
    fyers_index_symbols: Mapping[str, str],
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    now = time.time() if now_ts is None else float(now_ts)
    records = []
    if getattr(provider, "upstox_only", False):
        from data.upstox_market_data import UPSTOX_INDEX_KEYS

        key = UPSTOX_INDEX_KEYS.get(symbol)
        for cache, source, latency, max_age in (
            (provider._ws_cache, "upstox_ws", "REALTIME", 5.0),
            (provider._ltp_cache, "upstox_rest", "NEAR_REALTIME", 5.0),
        ):
            row = cache.get(key)
            if isinstance(row, dict) and row.get("source") == source:
                try:
                    age = now - float(row["time"])
                except (KeyError, ValueError, TypeError):
                    age = float("inf")
                records.append(_source_record(source, row, latency,
                                              -1.0 <= age <= max_age, now))
        records.sort(key=lambda row: (not row["entry_eligible"],
                                      float("inf") if row["age_seconds"] is None else row["age_seconds"]))
        return {"instrument": symbol, "sources": records, "freshest": records[0] if records else {},
                "entry_eligible": any(row["entry_eligible"] for row in records),
                "context_only_sources": [row for row in records if not row["entry_eligible"]]}
    ws_keys = []
    token_info = index_spot_tokens.get(symbol)
    if token_info:
        token = token_info[0]
        short_token = token[3:] if token.startswith("999") else token
        ws_keys.extend([token, short_token])

    # index_spot_tokens holds the AngelOne 999-prefixed constants, but the Kite ticker
    # caches under Kite's own instrument_token — as _load_kite_index_tokens puts it, "the
    # AngelOne token constants (26000, 26009...) mean nothing to Kite". That was fixed for
    # index candles and missed here, so this lookup never produced a kite_ws record. Since
    # order authority requires a source named kite* (fyers_ws is passed entry_eligible=
    # False and yahoo is DELAYED), the only realtime feed that can authorise an entry was
    # invisible: entry_eligible could come from nothing but the slower REST path, and
    # _filter_live_candidates_by_source dropped EVERY live entry whenever that path was
    # older than 15s.
    ws_keys.extend(
        str(tok)
        for tok, name in (getattr(provider, "_kite_index_names", None) or {}).items()
        if name == symbol
    )

    seen_keys = set()
    for key in ws_keys:
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        cached = provider._ws_cache.get(key)
        if isinstance(cached, dict):
            source = str(cached.get("source") or "angel_ws")
            age = _age_seconds(cached, now)
            records.append(
                _source_record(
                    source,
                    cached,
                    "REALTIME",
                    age is not None and age <= 6.0,
                    now,
                )
            )

    fyers_symbol = fyers_index_symbols.get(symbol)
    if fyers_symbol and isinstance(provider._ws_cache.get(fyers_symbol), dict):
        cached = provider._ws_cache[fyers_symbol]
        records.append(_source_record("fyers_ws", cached, "REALTIME", False, now))

    angel_cached = provider._ws_cache.get("angel:" + symbol)
    if isinstance(angel_cached, dict):
        records.append(_source_record("angel_ws", angel_cached, "REALTIME", False, now))

    cached_ltp = provider._ltp_cache.get(symbol)
    if isinstance(cached_ltp, dict):
        source = str(cached_ltp.get("source") or "unknown")
        if source == "yahoo":
            records.append(_source_record("yahoo", cached_ltp, "DELAYED", False, now))
        elif source in {"upstox", "kite", "fyers", "angel_rest"}:
            age = _age_seconds(cached_ltp, now)
            records.append(
                _source_record(
                    source if source in {"angel_rest", "upstox"} else f"{source}_rest",
                    cached_ltp,
                    "NEAR_REALTIME",
                    age is not None and age <= 15.0,
                    now,
                )
            )

    if not any(record["source"] == "yahoo" for record in records):
        records.append(
            {
                "source": "yahoo",
                "latency_class": "DELAYED",
                "age_seconds": None,
                "order_authority": False,
                "entry_eligible": False,
                "ltp": 0.0,
            }
        )

    entry_sources = [record for record in records if record.get("entry_eligible")]
    best = (
        min(
            records,
            key=lambda record: (
                0 if record.get("entry_eligible") else 1,
                999999.0
                if record.get("age_seconds") is None
                else float(record.get("age_seconds") or 0.0),
            ),
        )
        if records
        else {}
    )
    return {
        "instrument": symbol,
        "sources": records,
        "freshest": best,
        "entry_eligible": bool(entry_sources),
        "context_only_sources": [
            record for record in records if not record.get("entry_eligible")
        ],
    }
