"""Native UT/EMA admission, shared position accounting, and exit ownership.

Only this scanner bypasses the host's discretionary selection/management stack.
Contract identity, actual prices, broker fill ownership and execution authority
remain required. Paper/manual rows never submit alignment orders.
"""
from __future__ import annotations

import asyncio
import copy
import math
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from engine.ut_ema_engine import ENGINE_KEY, LANE, epoch, initial_stop, market_stamp, minutes


def is_lane(obj, settings=None):
    return bool(obj is not None and (
        str(getattr(obj, "source_engine", "")).lower() == ENGINE_KEY
        or str(getattr(obj, "source_lane", "")).upper() == LANE
        or str(getattr(obj, "exit_policy_owner", "")).upper() == LANE))


def state(obj):
    value = getattr(obj, "utema_state", None)
    if not isinstance(value, dict):
        quality = getattr(obj, "quality_verdict", {}) or {}
        value = (quality.get("checks") or {}).get("utema_state") if isinstance(quality, dict) else None
        value = copy.deepcopy(value) if isinstance(value, dict) else {}
        setattr(obj, "utema_state", value)
    return value


def persist(obj):
    quality = copy.deepcopy(getattr(obj, "quality_verdict", {}) or {})
    checks = dict(quality.get("checks") or {})
    checks["utema_state"] = copy.deepcopy(state(obj))
    quality.update(policy_version="utema_native.v1", checks=checks, allowed=True)
    obj.quality_verdict = quality
    obj.source_lane = obj.exit_policy_owner = LANE
    obj.source_engine = ENGINE_KEY


def selected_rr(settings, timeframe):
    minimum = float(getattr(settings, "dynamic_rr_min", 1.5) or 1.5)
    maximum = float(getattr(settings, "dynamic_rr_max", 5.) or 5.)
    value = min(maximum, float(getattr(settings, "dynamic_rr_5min_max", 2.5) or 2.5)) if timeframe == "5min" else maximum
    return max(minimum, value)


def premium_stop_distance(premium, index_risk, delta=.5):
    """Map the 0.2% underlying risk through ATM delta, capped at 15%; no floor."""
    if not all(math.isfinite(float(v)) for v in (premium, index_risk, delta)) or premium <= 0:
        return 0.
    return min(premium * .15, abs(index_risk) * abs(delta))


def premium_levels(premium, index_risk, delta, rr, previous_stop=0.):
    distance = premium_stop_distance(premium, index_risk, delta)
    stop = max(float(previous_stop or 0.), round(math.ceil((premium-distance)/.05-1e-9)*.05, 2))
    return premium, stop, premium + (premium-stop)*rr


def lane_exit_fields(obj):
    item = state(obj)
    stamp = item.get("exit_time")
    return item.get("exit_reason", ""), market_stamp(stamp).to_pydatetime() if stamp else None, item.get("exit_price", 0.)


def record_for(obj, result):
    if not isinstance(result, dict) or result.get("source_engine") != ENGINE_KEY:
        return None
    return (result.get("utema_records") or {}).get(state(obj).get("lifecycle_id"))


def engine_exit_for(trade, result, now=None):
    item = record_for(trade, result)
    if not item:
        # A restart cannot recover missing intrabar crossings. It can still honor
        # the first newly observed native exit for this held index/timeframe.
        if not isinstance(result, dict) or result.get("source_engine") != ENGINE_KEY:
            return None
        entered = state(trade).get("entry_time")
        if not entered:
            return None
        entry_clock, clock = market_stamp(entered), market_stamp(now)
        observed = sorted((event for event in result.get("utema_events", [])
            if not event.get("entry") and (event.get("realtime") or event.get("confirmed"))
            and entry_clock <= market_stamp(event["observed_at"]) <= clock),
            key=lambda event: (market_stamp(event["observed_at"]), event.get("sequence", 0)))
        if observed:
            event = observed[0]
            return {"reason": event["engine"]+" "+event["reason"],
                    "time": market_stamp(event["observed_at"]).to_pydatetime(),
                    "price": event["price"], "recovered_owner": True}
        return None
    if not item.get("exit_time"):
        return None
    when = market_stamp(item["exit_time"])
    if now is not None and when > market_stamp(now):
        return None
    return {"reason": item["exit_reason"], "time": when.to_pydatetime(), "price": item["exit_price"]}


def closed_bar_state(trade, result):
    item = record_for(trade, result)
    return {"utema_record": copy.deepcopy(item)} if item else None


def execution_levels(scanner, candidate, execution_touch_quote=None):
    premium = float(getattr(candidate, "price", getattr(candidate, "entry_price", 0.)) or 0.)
    touch = execution_touch_quote
    if isinstance(touch, dict) and touch:
        try:
            from config.settings import get_settings
            max_age = max(.05, float(getattr(get_settings(), "execution_touch_quote_max_age_seconds", 1.5) or 1.5))
            if (not touch.get("usable") or float(touch.get("bid", 0)) <= 0
                    or float(touch.get("ask", 0)) < float(touch.get("bid", 0))
                    or not 0 <= float(touch.get("age_seconds", math.inf)) <= max_age):
                return 0., 0., 0.
            premium = float(touch["ask"])
        except (KeyError, ValueError, TypeError):
            return 0., 0., 0.
    item = state(candidate)
    rr = float(item.get("rr", getattr(candidate, "rr", getattr(candidate, "rr_ratio", 2.))) or 2.)
    return premium_levels(premium, abs(float(item.get("entry_spot", 0.)) - float(item.get("initial_stop", 0.))),
                          float(item.get("option_delta", .5)), rr)


def entry_allowed(obj, levels=None):
    item = state(obj)
    if not item.get("lifecycle_id") or not item.get("entry_spot") or not item.get("initial_stop"):
        return False, "UT/EMA native lifecycle or index protection unavailable"
    side = int(item.get("side", 0))
    spot, stop = float(item["entry_spot"]), float(item["initial_stop"])
    tick = .01 if str(getattr(obj, "instrument", "")).upper() == "SENSEX" else .05
    expected = initial_stop(spot, side, .2, tick)
    if side not in (-1, 1) or abs(stop - expected) > tick * .01:
        return False, "UT/EMA requires its native outward-rounded 0.2% index stop"
    entry, premium_stop, target = levels or (
        float(getattr(obj, "price", getattr(obj, "entry_price", 0)) or 0),
        float(getattr(obj, "stop", getattr(obj, "current_stop", 0)) or 0),
        float(getattr(obj, "target", 0) or 0))
    if not all(math.isfinite(v) for v in (entry, premium_stop, target)):
        return False, "UT/EMA requires finite contract prices"
    if entry <= 0 or not 0 < premium_stop < entry or target <= entry:
        return False, "UT/EMA requires valid long-option execution geometry"
    if (entry - premium_stop) / entry > .15 + 1e-8:
        return False, "UT/EMA premium risk exceeds 15%"
    return True, ""


def held_state_valid(obj):
    """Restore owned protection without applying fresh-entry geometry to a winner."""
    try:
        item = state(obj)
        entry = float(getattr(obj, "entry_price", getattr(obj, "price", 0.)) or 0.)
        stop = float(getattr(obj, "current_stop", getattr(obj, "stop", 0.)) or 0.)
        target = float(getattr(obj, "target", 0.) or 0.)
        cap = float(item.get("premium_risk_pct", 0.))
        if not all(math.isfinite(value) for value in (entry, stop, target, cap)) or stop <= 0 or not 0 < cap <= 15:
            return False
        return entry_allowed(obj, (entry, entry*(1-cap/100), target))[0]
    except (ValueError, TypeError, OverflowError):
        return False


def exit_decision(trade, spot, premium, now=None, engine_exit=None, settings=None, market=None, **kwargs):
    from engine.s1_lane import S1ExitDecision
    item = state(trade)
    if not item:
        return S1ExitDecision(False)
    clock = market_stamp(now)
    latest = (market or {}).get("utema_record") or {}
    stop = float(item.get("stop") or item.get("initial_stop") or 0.)
    for update in latest.get("stop_updates", []):
        if market_stamp(update["time"]) <= clock:
            proposed = float(update["stop"])
            stop = max(stop, proposed) if int(item.get("side", 1)) > 0 else min(stop, proposed)
    item["stop"] = stop
    persist(trade)
    entry = float(getattr(trade, "entry_price", getattr(trade, "price", 0.)) or 0.)
    cap = max(0., min(15., float(item.get("premium_risk_pct", 15.))))
    premium_stop = max(entry * (1 - cap / 100),
                       float(getattr(trade, "current_stop", getattr(trade, "stop", 0.)) or 0.))
    target = float(getattr(trade, "target", 0.) or 0.)
    native_clock = market_stamp(engine_exit["time"]) if engine_exit and engine_exit.get("time") else None
    if native_clock is not None and native_clock > clock:
        engine_exit, native_clock = None, None
    for tick in kwargs.get("observed_contract_ticks") or []:
        value = float(tick[0])
        if not math.isfinite(value) or value <= 0:
            continue
        tick_clock = market_stamp(tick[1]) if len(tick) > 1 and tick[1] is not None else None
        if tick_clock is not None and (tick_clock > clock or (native_clock is not None and tick_clock > native_clock)):
            continue
        reason = "UTEMA_PREMIUM_STOP" if value <= premium_stop else (
            "UTEMA_PREMIUM_TARGET" if target > 0 and value >= target else "")
        if reason:
            decision = S1ExitDecision(True, reason, stop, "First observed native premium exit", premium_stop)
            decision.observed_price = value
            decision.observed_at = tick_clock.to_pydatetime() if tick_clock is not None else None
            return decision
    if engine_exit and native_clock is not None:
        return S1ExitDecision(True, "UTEMA_" + str(engine_exit["reason"]).replace(" ", "_"), stop,
                              str(engine_exit["reason"]), premium_stop)
    if premium > 0 and premium <= premium_stop:
        return S1ExitDecision(True, "UTEMA_PREMIUM_STOP", stop, "UT/EMA premium protection", premium_stop)
    if spot > 0 and int(item.get("side", 1)) * (spot - stop) <= 0:
        return S1ExitDecision(True, "UTEMA_INDEX_STOP", stop, "UT/EMA index protection", premium_stop)
    if engine_exit:
        return S1ExitDecision(True, "UTEMA_" + str(engine_exit["reason"]).replace(" ", "_"), stop,
                              str(engine_exit["reason"]), premium_stop)
    if target > 0 and premium >= target:
        return S1ExitDecision(True, "UTEMA_PREMIUM_TARGET", stop, "Selected option RR target", premium_stop)
    return S1ExitDecision(False, index_stop=stop, premium_stop=premium_stop)


def active_rows(scanner, instrument, timeframe):
    rows = []
    getter = getattr(scanner, "_active_session_entry_candidates", None)
    if callable(getter):
        rows.extend(getter(instrument, timeframe=timeframe) or [])
    rows.extend(t for t in (getattr(getattr(scanner, "trades", None), "open_trades", {}) or {}).values()
                if t.instrument == instrument and t.timeframe == timeframe)
    return [r for r in rows if is_lane(r)]


def entry_consumed(scanner, instrument, identity):
    """A premium exit must not re-admit the still-open native index lifecycle."""
    getter = getattr(scanner, "_get_session_trade_candidates", None)
    rows = list(getter(instrument) or []) if callable(getter) else []
    rows.extend(getattr(getattr(scanner, "trades", None), "closed_trades", []) or [])
    return any(is_lane(row) and state(row).get("lifecycle_id") == identity and
               (bool(getattr(row, "accepted_by_gate", False)) or
                str(getattr(row, "action", "")).upper() == "EXIT" or
                str(getattr(row, "status", "")).upper() == "CLOSED") for row in rows)


def result_for_monitor(scanner, trade):
    """Keep the original native event source available after scanner selection changes."""
    service = getattr(scanner, "_utema_tick_service", None)
    if service is not None and str(getattr(scanner, "mode", "")).upper() == "REAL":
        item = state(trade)
        return service.monitor_owner(trade, item.get("lifecycle_id"), item.get("entry_time"))
    mtf = getattr(scanner, "mtf", None)
    key = f"{trade.instrument}_{trade.timeframe}"
    adapter = (getattr(mtf, "engines", {}) or {}).get(key)
    if getattr(adapter, "signal_source", "") != ENGINE_KEY:
        adapter = (getattr(mtf, "_retired_utema_engines", {}) or {}).get(key)
    monitor = getattr(adapter, "monitor_snapshot", None)
    if callable(monitor):
        item = state(trade)
        minimal = monitor(item.get("lifecycle_id"), item.get("entry_time"))
        if minimal is not None:
            return minimal
    snapshot = getattr(adapter, "stream_snapshot", None)
    streamed = snapshot() if callable(snapshot) else getattr(adapter, "_stream_result", None)
    if isinstance(streamed, dict) and streamed.get("source_engine") == ENGINE_KEY:
        return streamed
    cached = (getattr(mtf, "_result_cache", {}) or {}).get(f"{trade.instrument}_{trade.timeframe}")
    if isinstance(cached, dict) and cached.get("source_engine") == ENGINE_KEY:
        return cached
    reader = getattr(mtf, "get_utema_owner_result", None)
    if callable(reader):
        frame = scanner.candles.get_candles(trade.instrument, trade.timeframe)
        return reader(trade.instrument, trade.timeframe, frame, as_of=market_stamp())
    return None


def build_candidates(processor, instrument, mtf_result, cfg, lots, lot_size, settings):
    from config.settings import timeframe_entry_timeframes
    from engine.signal_processor import TradeCandidate
    scanner = processor.scanner
    historical = str(getattr(scanner, "mode", "")).upper() == "HISTORICAL"
    if historical:
        from engine.ut_ema_replay import replay_candidates
        return replay_candidates(processor, instrument, mtf_result, cfg, lots, lot_size, settings)
    clock = market_stamp()
    allowed = timeframe_entry_timeframes(getattr(settings, "ut_timeframe_entry_policy", "INCLUDE_5MIN"))
    output = []
    pending = {}
    for tf in ("5min", "15min"):
        if tf not in allowed:
            continue
        result = getattr(mtf_result, "results_" + tf, None) or {}
        if result.get("source_engine") != ENGINE_KEY:
            continue
        service = getattr(scanner, "_utema_tick_service", None)
        if service is not None and not result.get("utema_entry_eligible", False):
            continue
        health = (getattr(scanner, "_utema_stream_health", {}) or {}).get(f"{instrument}_{tf}", {})
        if health.get("status") in {"ERROR", "OVERFLOW", "WAITING_HISTORY"}:
            continue
        for sig in result.get("signals", []):
            event, native = getattr(sig, "utema_event", {}), getattr(sig, "utema_state", {})
            if not event or not native:
                continue
            from engine.ut_ema_engine import signal_fire_mode
            if signal_fire_mode(settings) == "CANDLE_CLOSE" and event.get("realtime"):
                continue
            if service is not None and getattr(sig, "utema_event_id", "") not in result.get("utema_executable_event_ids", []):
                continue
            when = market_stamp(event["observed_at"])
            if when > clock:
                continue
            action = sig.utema_action
            if action == "EXIT":
                continue  # Exit monitors consume the same immutable native record.
            active = active_rows(scanner, instrument, tf)
            existing = next((r for r in active if state(r).get("lifecycle_id") == native["lifecycle_id"]), None)
            existing = existing or pending.get(native["lifecycle_id"])
            if existing is not None and getattr(existing, "pending_exit", False):
                continue
            if action != "ENTRY" and existing is None:
                continue
            if action == "ENTRY" and (active or when + pd.Timedelta(minutes=minutes(tf)) < clock):
                continue
            if action == "ENTRY" and entry_consumed(scanner, instrument, native["lifecycle_id"]):
                continue
            if action in {"ENTRY", "ADD"} and native.get("exit_time") and market_stamp(native["exit_time"]) <= clock:
                continue
            if action == "ADD" and state(existing).get("alignment_added"):
                continue
            if existing is not None and sig.utema_event_id in state(existing).get("applied_event_ids", []):
                continue
            spot, side = float(native["entry_spot"]), int(native["side"])
            option_type = "CE" if side > 0 else "PE"
            strike = round(spot / float(cfg.get("strike_interval", 50))) * float(cfg.get("strike_interval", 50))
            if existing is not None:
                symbol, token = existing.trading_symbol, existing.symbol_token
                strike = existing.atm_strike
            else:
                contract = scanner.data.get_option_token(instrument, strike, option_type)
                if not contract:
                    continue
                resolved = contract.get("strike_val")
                if resolved is not None and abs(float(resolved) - strike) > 1e-6:
                    continue  # Never silently substitute another strike for requested ATM.
                symbol, token = str(contract.get("symbol", "")), str(contract.get("token", ""))
            probe = SimpleNamespace(instrument=instrument, inst_type="OPT", option_type=option_type,
                atm_strike=strike, trading_symbol=symbol, symbol_token=token, price=spot,
                entry_spot=spot, target=spot, quote_verified=False)
            premium = float(scanner._get_live_premium_cached(probe, allow_blocking=False) or 0.)
            if premium <= 0 or not bool(getattr(probe, "quote_verified", False)):
                continue
            dte = scanner._days_to_expiry_for(instrument, when.to_pydatetime()) if hasattr(scanner, "_days_to_expiry_for") else 1.
            delta = abs(processor._option_signed_delta(spot, strike, option_type, {}, dte, .5))
            rr = selected_rr(settings, tf)
            _, option_stop, option_target = premium_levels(premium, abs(spot - native["initial_stop"]), delta, rr)
            risk = premium-option_stop
            if risk <= 0:
                continue
            current = copy.deepcopy(state(existing)) if existing else {
                key: copy.deepcopy(native[key]) for key in (
                    "lifecycle_id", "side", "entry_engine", "entry_spot", "entry_time", "entry_bar", "initial_stop")}
            current.setdefault("stop", native["initial_stop"])
            current.setdefault("alignment_added", False)
            current.setdefault("applied_event_ids", [])
            current.setdefault("premium_risk_pct", 100 * risk / premium)
            current.setdefault("option_delta", delta)
            current.setdefault("rr", rr)
            candidate = TradeCandidate(instrument=instrument, direction="LONG" if side > 0 else "SHORT",
                price=premium, stop=premium-risk, target=premium+risk*rr, lots=int(lots), lot_size=int(lot_size),
                grade="NATIVE", confidence=1., timeframe=tf, inst_type="OPT", option_type=option_type,
                atm_strike=strike, multiplier=delta if side > 0 else -delta, trading_symbol=symbol,
                symbol_token=token, rr=rr, entry_spot=spot, spot_stop=native["initial_stop"],
                spot_target=spot+side*abs(spot-native["initial_stop"])*2,
                underlying_entry=spot, underlying_stop=native["initial_stop"],
                underlying_target=spot+side*abs(spot-native["initial_stop"])*2,
                premium_entry=premium, premium_stop=premium-risk, premium_target=premium+risk*rr,
                initial_stop=premium-risk,
                signal_timestamp=sig.timestamp, decision_timestamp=clock.to_pydatetime(), current_price=premium,
                action=action, source_engine=ENGINE_KEY, source_lane=LANE, source_engines=["ut", "ema"],
                quote_verified=True, quote_source=str(getattr(probe, "quote_source", "")),
                reasons=[f"{event['engine']} native {action}; shared base+1; first native exit"])
            candidate.utema_state, candidate.utema_event = current, copy.deepcopy(event)
            candidate.utema_event_id = sig.utema_event_id
            candidate.lifecycle_id = current["lifecycle_id"]
            candidate.signal_id = sig.utema_event_id
            candidate.signal_fire_mode = "INTRACANDLE" if event["realtime"] else "CANDLE_CLOSE"
            candidate.entered_intracandle = bool(event["realtime"])
            current["signal_fire_mode"] = candidate.signal_fire_mode
            persist(candidate)
            output.append(candidate)
            if action == "ENTRY":
                pending[native["lifecycle_id"]] = candidate
    return output


def open_kwargs(scanner, candidate, levels, exec_type, on_result=None):
    entry, stop, target = levels
    state(candidate)["premium_risk_pct"] = (entry-stop)/entry*100 if entry > 0 else 0.
    persist(candidate)
    return dict(instrument=candidate.instrument, timeframe=candidate.timeframe, direction=candidate.direction,
        price=entry, trailing_stop=stop, target=target, initial_stop=stop, lots=candidate.lots,
        lot_size=candidate.lot_size, grade="NATIVE", atm_strike=candidate.atm_strike,
        option_type=candidate.option_type, rr_ratio=candidate.rr, confidence=1.,
        instrument_multiplier=candidate.multiplier, trading_symbol=candidate.trading_symbol,
        symbol_token=candidate.symbol_token, inst_type="OPT", source_engine=ENGINE_KEY, source_lane=LANE,
        exit_policy_owner=LANE, quality_verdict=copy.deepcopy(candidate.quality_verdict),
        entry_spot=candidate.entry_spot, spot_stop=candidate.spot_stop, spot_target=candidate.spot_target,
        underlying_entry=candidate.entry_spot, underlying_stop=candidate.spot_stop,
        underlying_target=candidate.spot_target, premium_entry=entry, premium_stop=stop, premium_target=target,
        lifecycle_id=state(candidate)["lifecycle_id"], exec_type=exec_type,
        entry_time=market_stamp().to_pydatetime(), signal_time=candidate.signal_timestamp,
        source_engines=["ut", "ema"], signal_fire_mode=state(candidate).get("signal_fire_mode", "INTRACANDLE"),
        entered_intracandle=bool(getattr(candidate, "entered_intracandle", False)), is_ghost=False,
        on_execution_result=on_result,
        entry_submission_guard=lambda trade: submission_authority(scanner,
            manual=exec_type == "M" and not bool(getattr(trade, "_scale_in_inflight", False)), trade=trade))


def submission_authority(scanner, manual=False, trade=None):
    from engine.ut_ema_engine import signal_fire_mode
    if (trade is not None and signal_fire_mode() == "CANDLE_CLOSE"
            and state(trade).get("signal_fire_mode", "INTRACANDLE") == "INTRACANDLE"):
        return False, "Intracandle entries are disabled"
    if (not getattr(scanner, "is_running", True)
            or str(getattr(scanner, "system_power", "ON")).upper() != "ON"):
        return False, "Native execution stopped or system power is off"
    if str(getattr(scanner, "mode", "")).upper() != "REAL":
        return False, "Native broker entry requires REAL mode"
    if bool(getattr(scanner, "auto_mode", False)) == bool(manual):
        return False, "Execution mode changed before submission"
    if trade is not None:
        status = str(getattr(trade, "status", "OPEN")).upper()
        execution_status = str(getattr(trade, "execution_status", "")).upper()
        if (status == "CLOSED" or getattr(trade, "pending_exit", False) or
                execution_status in {"EXIT_PENDING", "PARTIAL_EXIT", "RECOVERY_REQUIRED", "CLOSED"}):
            return False, "Native parent lifecycle is exiting or closed"
        health = (getattr(scanner, "_utema_stream_health", {}) or {}).get(f"{trade.instrument}_{trade.timeframe}", {})
        if health.get("status") in {"ERROR", "OVERFLOW", "WAITING_HISTORY"}:
            return False, "Native observation stream is unavailable"
        native = record_for(trade, result_for_monitor(scanner, trade))
        if native and native.get("exit_time"):
            return False, "Native lifecycle exited before broker submission"
        adding = bool(getattr(trade, "_scale_in_inflight", False))
        if adding and status != "OPEN":
            return False, "Native alignment requires an open parent"
        if manual and not adding:
            # The virtual option lifecycle may exit while an operator-approved
            # initial entry waits in the broker queue, before the index exits.
            exited = getattr(scanner, "_manual_signal_lifecycle_exited", None)
            if callable(exited):
                try:
                    if exited(trade):
                        return False, "Manual native signal lifecycle exited before broker submission"
                except Exception:
                    return False, "Manual native signal lifecycle cannot be verified"
        if not adding:
            # Quote freshness cannot extend the original native observation's
            # admission window while a first-entry order waits in the queue.
            entered = (native or {}).get("entry_time") or state(trade).get("entry_time")
            try:
                if not entered:
                    raise ValueError("missing native observation time")
                observed, clock = market_stamp(entered), market_stamp()
                period = minutes(trade.timeframe)
                if pd.isna(observed) or pd.isna(clock) or period <= 0:
                    raise ValueError("invalid native observation time")
                if observed > clock:
                    return False, "Native entry observation is in the future"
                if observed + pd.Timedelta(minutes=period) < clock:
                    return False, "Native entry observation expired before broker submission"
            except (TypeError, ValueError, OverflowError):
                return False, "Native entry observation time is invalid or unavailable"
        service = getattr(scanner, "_utema_tick_service", None)
        if service is not None:
            from config.settings import get_settings
            from engine.scanner_modes import configured_mode
            if configured_mode(get_settings()) != ENGINE_KEY:
                return False, "Native entry scanner changed before submission"
            if getattr(service, "_closed", False):
                return False, "Native observation service stopped"
            result = result_for_monitor(scanner, trade) or {}
            permitted = set(result.get("utema_executable_event_ids", []))
            item = state(trade)
            event_ids = ([getattr(trade, "_utema_pending_alignment_event_id", "") or item.get("alignment_event_id", "")]
                         if adding else item.get("applied_event_ids", []))
            if not any(identity in permitted for identity in event_ids):
                return False, "Native entry was replayed or its observation authorization expired"
    return True, ""


async def coordinate(scanner, candidates, is_warmup=False):
    accepted = []
    if (is_warmup or str(getattr(scanner, "mode", "")).upper() == "HISTORICAL"
            or not getattr(scanner, "is_running", True)
            or str(getattr(scanner, "system_power", "ON")).upper() != "ON"):
        return accepted
    for candidate in candidates:
        if not is_lane(candidate):
            continue
        from engine.ut_ema_engine import signal_fire_mode
        if signal_fire_mode() == "CANDLE_CLOSE" and getattr(candidate, "entered_intracandle", False):
            continue
        service = getattr(scanner, "_utema_tick_service", None)
        if service is not None:
            current_result = result_for_monitor(scanner, candidate) or {}
            health = (getattr(scanner, "_utema_stream_health", {}) or {}).get(
                f"{candidate.instrument}_{candidate.timeframe}", {})
            native = record_for(candidate, current_result)
            if (getattr(service, "_closed", False) or health.get("status") in {"ERROR", "OVERFLOW", "WAITING_HISTORY"}
                    or candidate.utema_event_id not in current_result.get("utema_executable_event_ids", [])
                    or (native and native.get("exit_time"))):
                continue
        item = state(candidate)
        existing = next((r for r in active_rows(scanner, candidate.instrument, candidate.timeframe)
                         if state(r).get("lifecycle_id") == item["lifecycle_id"]), None)
        if candidate.action == "EXIT":
            continue  # Native monitors consume the exact same engine record.
        if candidate.action == "ADD":
            if existing is None or getattr(existing, "pending_exit", False) or state(existing).get("alignment_added"):
                continue
            live = next((t for t in (getattr(scanner.trades, "open_trades", {}) or {}).values()
                         if is_lane(t) and state(t).get("lifecycle_id") == item["lifecycle_id"]), None)
            if live is not None:
                if not scanner.auto_mode:
                    continue
                live._utema_pending_alignment_event_id = candidate.utema_event_id
                ok, why = scanner.trades.scale_in_trade(live.id, 1, candidate.price,
                    reason="UTEMA_ALIGNMENT", native_alignment=True)
                if ok:
                    state(live)["alignment_event_id"] = candidate.utema_event_id
                    persist(live)
                continue
            # A manual signal row is a virtual position, even when its market feed
            # is REAL. Never call the real-account TradeManager scale-in here.
            old_lots = int(existing.lots)
            existing.price = (float(existing.price)*old_lots + candidate.price)/(old_lots+1)
            existing.lots = old_lots+1
            current = state(existing)
            current["alignment_added"], current["alignment_event_id"] = True, candidate.utema_event_id
            current.setdefault("applied_event_ids", []).append(candidate.utema_event_id)
            risk = existing.price * float(current["premium_risk_pct"]) / 100
            existing.stop = max(float(existing.stop), round(math.ceil((existing.price-risk)/.05-1e-9)*.05, 2))
            if existing.stop < existing.price:
                existing.target = existing.price+(existing.price-existing.stop)*existing.rr
            existing.premium_entry, existing.premium_stop, existing.premium_target = existing.price, existing.stop, existing.target
            existing.initial_stop = existing.stop
            existing.display_stop, existing.display_target = existing.stop, existing.target
            persist(existing)
            scanner._remember_trade_candidates([existing])
            accepted.append(existing)
            continue
        if (existing is not None or active_rows(scanner, candidate.instrument, candidate.timeframe)
                or entry_consumed(scanner, candidate.instrument, item["lifecycle_id"])):
            continue
        allowed, why = entry_allowed(candidate)
        if not allowed:
            scanner.log_event(f"UT/EMA entry held: {candidate.instrument} {candidate.timeframe} - {why}", "warning")
            continue
        item.setdefault("applied_event_ids", []).append(candidate.utema_event_id)
        persist(candidate)
        if not scanner.auto_mode:
            candidate.accepted_by_gate = True
            candidate.execution_status, candidate.execution_state = "SIGNAL_ONLY", "NOT_PLACED"
            candidate.broker_fill_confirmed = False
            scanner._mark_candidate_accepted_time(candidate)
            scanner._record_accepted_entry(candidate)
            scanner._remember_trade_candidates([candidate])
            scanner.log_event(f"Manual {candidate.direction} Signal: {candidate.instrument} {candidate.timeframe} ut/ema @ {candidate.price:.2f}", "trade")
            accepted.append(candidate)
            continue
        touch = scanner._candidate_execution_touch_quote(candidate)
        levels = execution_levels(scanner, candidate, touch)
        allowed, _ = entry_allowed(candidate, levels)
        if not allowed:
            continue
        loop = asyncio.get_running_loop()
        def completed(trade, ok, sig=candidate):
            asyncio.run_coroutine_threadsafe(scanner._handle_broker_entry_result(sig, trade, ok), loop)
        trade = scanner.trades.open_trade(**open_kwargs(scanner, candidate, levels, "A", completed),
                                         execution_touch_quote=touch)
        if trade:
            # Native state is hydrated by TradeManager before its broker queue.
            # Do not overwrite changes made by a fast fill callback after enqueue.
            candidate.status, candidate.accepted_by_gate = "ORDER PENDING", False
            scanner._latest_trade_candidates[candidate.instrument] = [candidate]
    if candidates:
        await scanner._publish_trade_payload_now()
    return accepted


async def execute_manual(scanner, candidate):
    """Submit an explicitly selected dashboard entry with native geometry."""
    allowed, reason = submission_authority(scanner, manual=True)
    if not allowed:
        return {"status": "blocked", "reason": reason}
    touch = scanner._candidate_execution_touch_quote(candidate)
    levels = execution_levels(scanner, candidate, touch)
    allowed, reason = entry_allowed(candidate, levels)
    if not allowed or not touch:
        return {"status": "blocked", "reason": reason or "quote_unavailable"}
    loop = asyncio.get_running_loop()
    def completed(trade, ok):
        asyncio.run_coroutine_threadsafe(scanner._handle_broker_entry_result(candidate, trade, ok), loop)
    outcome = {}
    trade = scanner.trades.open_trade(**open_kwargs(scanner, candidate, levels, "M", completed),
        execution_touch_quote=touch, entry_refusal_outcome=outcome)
    if not trade:
        return {"status": "blocked", "reason": outcome.get("reason") or "broker_entry_refused"}
    candidate.status = "ORDER PENDING"
    scanner._remember_trade_candidates([candidate])
    await scanner._publish_trade_payload_now()
    return {"status": "submitted", "trade_id": trade.id}


def analysis_view(scanner, instrument, configuration, mtf_result, spot, lots, lot_size,
                  quote, previous_close, change_points, change_pct):
    """Keep native entry independent of intelligence refresh and legacy grading."""
    from config.settings import get_settings
    settings = get_settings()
    rows = scanner._get_session_trade_candidates(instrument)
    scanner._refresh_trade_candidates(rows, configuration, allow_blocking=False)
    historical = str(getattr(scanner, "mode", "")).upper() == "HISTORICAL"
    if not historical:
        scanner._check_session_candidates_breaches(instrument)
    candidates = build_candidates(scanner.signal_processor, instrument, mtf_result,
        configuration, lots, lot_size, settings)
    rows = scanner._get_session_trade_candidates(instrument)
    scanner._refresh_trade_candidates(rows, configuration, allow_blocking=False)
    if historical:
        panel = getattr(scanner, "_historical_panel_intel", None)
        intelligence = panel(instrument) if callable(panel) else {}
    else:
        cache = getattr(scanner, "_intel_cache", {}) or {}
        intelligence = dict(cache.get(f"{instrument}_5min", cache.get(instrument, {})) or {})
        horizons = getattr(scanner, "_live_regime_horizons", None)
        if intelligence and callable(horizons):
            intelligence["regime_horizons"] = horizons(
                instrument, session_context=intelligence.get("session_context"),
                breadth_context=intelligence.get("breadth_context"))
    intelligence["potential_rr"] = selected_rr(settings, "5min")
    label, session, preopen, settlement = scanner._dashboard_quote_session(quote)
    held = {}
    for tf in ("5min", "15min"):
        for row in active_rows(scanner, instrument, tf):
            held[state(row).get("lifecycle_id") or id(row)] = row
    risk = sum(max(0., float(getattr(row, "entry_price", getattr(row, "price", 0.)) or 0.) -
                   float(getattr(row, "current_stop", getattr(row, "stop", 0.)) or 0.)) *
               max(0, int(getattr(row, "lots", 0))) * max(1, int(getattr(row, "lot_size", 1)))
               for row in held.values())
    # Native mode returns before EntryAnalysisService builds legacy trade_state.
    # Surface the engine's own 5m UT ATR instead of presenting a false blank.
    native_5m = (getattr(mtf_result, "results_5min", None) or {}).get("state") or {}
    try:
        atr_value = float(native_5m.get("atr_value") or 0.)
    except (TypeError, ValueError):
        atr_value = 0.
    if not math.isfinite(atr_value) or atr_value <= 0:
        atr_value = 0.
    atr_text = f"ATR {atr_value:.2f} pts" if atr_value else "--"
    return {
        "trade_state": {"next_trigger": "ut/ema native", "open_mtm": sum(float(getattr(row, "pnl", 0) or 0)
            for row in held.values()), "time_in_trade": "--", "risk_exposure": -risk,
            "atr_value": atr_value or None, "atr_timeframe": "5min",
            "atr_reason": native_5m.get("atr_reason") or "native_atr_unavailable",
            "atr_exhaustion": atr_text},
        "mtf": {"confluence_score": getattr(mtf_result, "confluence_score", 0),
                "confluence_signal": getattr(mtf_result, "confluence_signal", ""),
                **{f"state_{tf}": (getattr(mtf_result, f"results_{tf}", None) or {}).get("state")
                   for tf in ("1min", "5min", "15min")}},
        "intelligence": intelligence, "pre_candle_prediction": {},
        "chart": scanner._build_chart_from_cache(instrument) if getattr(scanner, "chart_stream_enabled", True) else {},
        "signals": scanner.signals.get_active_signals(mode=scanner.mode),
        "trade_candidates": [scanner._serialize_trade_candidate(row) for row in rows],
        "simulation_id": getattr(scanner, "simulation_id", ""),
        "atm_strike": round(spot / configuration.get("strike_interval", 50)) * configuration.get("strike_interval", 50),
        "ltp": spot, "spot_price": spot, "change": change_points, "change_points": change_points,
        "change_pct": change_pct, "prev_close": previous_close, "previous_close": previous_close,
        "quote_source": quote.get("source", "unknown"), "quote_label": label, "quote_session": session,
        "quote_age_seconds": quote.get("quote_age_seconds"), "is_preopen": preopen, "preopen_settlement": settlement,
        "market_source": scanner.data.get_market_source_snapshot(instrument),
        "utema_replay_coverage": [{"timeframe": key[1], "lifecycle_id": key[2], "status": value}
            for key, value in (getattr(scanner, "_utema_replay_coverage", {}) or {}).items() if key[0] == instrument],
        "utema_stream": {key: copy.deepcopy(value) for key, value in
            (getattr(scanner, "_utema_stream_health", {}) or {}).items() if key.startswith(instrument+"_")},
    }, candidates
