"""Entry-analysis orchestration extracted from the scanner runtime."""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger

from config.settings import get_settings
from engine.historical_data import unavailable_chain_quality, unavailable_depth
from engine.boundary_failures import AnalysisBoundaryFailure
from engine.intelligence_score import aggregate_intelligence_score
from engine.signal_processor import TradeCandidate
from trading.trade_manager import IST


class EntryAnalysisService:
    """Owns the per-instrument analysis pipeline; scanner remains the state host."""

    def __init__(self, scanner) -> None:
        self.scanner = scanner

    def process(self, name: str, cfg: Dict) -> Tuple[Dict, List[TradeCandidate]]:
        """Main Analysis Pipeline — focused on speed and real-time accuracy"""
        scanner = self.scanner
        self = scanner  # preserve bare-host arguments used by legacy static helpers
        lot_size = cfg.get("lot_size", 25)
        lots = scanner.user_lots.get(name, 1)
        strike_interval = cfg.get("strike_interval", 50)
        combined_cap = scanner.capital_fut + scanner.capital_opt

        # 1. Scan hot path is cache-only. Calling get_index_quote() here can enter Kite,
        # Fyers and Yahoo REST. During the minute history refresh, all four scanner
        # workers then wait behind Fyers' global single-flight for 3-10 seconds. The
        # websocket/background-LTP lanes already populate this coherent snapshot.
        market_quote = {}
        historical_mode = str(getattr(self, "mode", "") or "").upper() == "HISTORICAL"
        if not historical_mode:
            cached_quote = getattr(scanner.data, "get_cached_index_quote_snapshot", None)
            if callable(cached_quote):
                try:
                    market_quote = cached_quote(name) or {}
                except Exception as exc:
                    logger.debug(f"Cached index quote read failed for {name}: {exc}")
            elif hasattr(scanner.data, "get_index_quote"):
                # Compatibility for lightweight/test providers. Production always exposes
                # the cache-only reader above.
                market_quote = scanner.data.get_index_quote(name) or {}

        quote_spot = float(market_quote.get("ltp", 0.0) or 0.0)
        quote_fresh = bool(
            market_quote.get(
                "quote_fresh", market_quote.get("is_fresh", quote_spot > 0)
            )
        )
        if quote_spot > 0 and quote_fresh:
            scanner.candles.update_latest_price(name, quote_spot)

        if historical_mode:
            # Replay may have OHLC without any live quote. Resolve the reference
            # before the native lane's early return, keeping quote and receipt state
            # untouched and never borrowing today's mark for archived bars.
            spot = 0.0
            for timeframe in ("1min", "5min", "15min"):
                history = scanner.candles.get_candles(name, timeframe)
                if history is None or history.empty or "close" not in history.columns:
                    continue
                try:
                    historical_spot = float(history["close"].iloc[-1])
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(historical_spot) and historical_spot > 0:
                    spot = historical_spot
                    break
        else:
            candle_spot = scanner.candles.get_latest_price(name)
            spot = quote_spot if quote_spot > 0 and quote_fresh else candle_spot
            if spot <= 0 and quote_spot > 0:
                # Last-known reference is safer than an artificial zero, but stays stale in
                # market_quote so entry freshness gates cannot treat it as a live tick.
                spot = quote_spot
        if spot <= 0:
            logger.warning(f"âš ï¸  Spot price for {name} is {spot}. Skipping scan.")
            return {"error": f"Invalid spot price: {spot}"}, []

        # â• â•  Daily Change (Optimized: Use Cache) â• â•
        quote_change = float(market_quote.get("change", 0.0) or 0.0)
        quote_change_pct = float(market_quote.get("change_pct", 0.0) or 0.0)
        quote_prev_close = float(
            market_quote.get("previous_close", 0.0)
            or market_quote.get("prev_close", 0.0)
            or 0.0
        )
        prev_close = quote_prev_close
        if quote_change or quote_change_pct:
            if quote_change == 0.0 and quote_prev_close > 0:
                quote_change = spot - quote_prev_close
            if quote_change_pct == 0.0 and quote_prev_close > 0:
                quote_change_pct = quote_change / quote_prev_close * 100.0
            change_points = quote_change
            change_pct = quote_change_pct
            if quote_prev_close > 0:
                if not hasattr(self, "_prev_close_cache"):
                    scanner._prev_close_cache = {}
                scanner._prev_close_cache[name] = quote_prev_close
        else:
            if not hasattr(self, "_prev_close_cache"):
                scanner._prev_close_cache = {}

            prev_close = quote_prev_close or scanner._prev_close_cache.get(name, 0)
            if prev_close <= 0:
                cached_previous_close = getattr(
                    scanner.data, "get_cached_previous_close", None
                )
                prev_close = (
                    cached_previous_close(name)
                    if callable(cached_previous_close)
                    else 0.0
                )
                scanner._prev_close_cache[name] = prev_close

            change_points = (spot - prev_close) if prev_close > 0 else 0
            change_pct = ((change_points) / prev_close * 100) if prev_close > 0 else 0

        # 2. Update MTF and Analysis
        for tf in ["1min", "5min", "15min"]:
            data = scanner.candles.get_candles(name, tf)
            if data is not None:
                data = scanner._prepare_live_signal_data(name, tf, data)
            # Missing/underfilled is a real input state, not permission to keep the
            # previous arrow alive.  MultiTimeframeEngine records and invalidates it.
            scanner.mtf.update_data(name, tf, data)

        # Multi-TF analysis
        mtf_result = scanner.mtf.process_instrument(name, lot_size, lots, combined_cap, scanner.risk_fut_pct)

        # Cache per-TF process results for chart builder (avoid re-processing)
        tf_metadata = dict(getattr(mtf_result, "timeframe_metadata", {}) or {})
        for timeframe, metadata in tf_metadata.items():
            if not isinstance(metadata, dict):
                continue
            source = str(metadata.get("signal_source", "") or "").lower()
            phase = "s1" if source == "s1" else "scanner"
            try:
                duration_ms = float(metadata.get("duration_ms") or 0.0)
            except (TypeError, ValueError):
                duration_ms = 0.0
            scanner._record_scan_phase_ms(
                f"{phase}_{timeframe}_ms",
                duration_ms,
                instrument=name,
            )
        for tf_key in ["1min", "5min", "15min"]:
            key = f"{name}_{tf_key}"
            tf_result = getattr(mtf_result, f"results_{tf_key}", None)
            if tf_result:
                scanner._cached_results[key] = tf_result
                metadata = dict(tf_metadata.get(tf_key) or {})
                if metadata:
                    metadata["result_object"] = tf_result
                    scanner._cached_result_metadata[key] = metadata
                else:
                    # Backward compatibility for pluggable/test MTF implementations:
                    # they may still provide a result, but without provenance it cannot
                    # satisfy the live stabilization freshness contract.
                    scanner._cached_result_metadata.pop(key, None)
                logger.debug(f"Cached results for {key}")
            else:
                # The old result must leave both stores together.  Leaving it in
                # _cached_results is exactly how a short/erroring input used to present
                # yesterday's arrow as this scan's stabilization evidence.
                scanner._cached_results.pop(key, None)
                scanner._cached_result_metadata.pop(key, None)
                logger.debug(f"No results for {key}")

        from engine.scanner_modes import configured_mode
        if configured_mode(get_settings()) == "ut_ema":
            from engine.ut_ema_lane import analysis_view
            return analysis_view(scanner, name, cfg, mtf_result, spot, lots, lot_size,
                                 market_quote, prev_close, change_points, change_pct)

        results = {} # Will be populated with intelligence below

        # â• â•  MASTER SPOT RESOLUTION (Broker Priority Fix) â• â•
        candle_5min = scanner._live_analysis_window(scanner.candles.get_candles(name, "5min"), "5min")

        # â• â•  MASTER SPOT RESOLUTION (Broker Priority Fix) â• â•
        # Priority: Official Broker Spot -> Historical Fallback
        live_spot = (
            0.0
            if historical_mode
            else quote_spot if quote_spot > 0 else scanner.candles.get_latest_price(name)
        )
        hist_spot = candle_5min['close'].iloc[-1] if candle_5min is not None and not candle_5min.empty else 0

        # Use broker price if available, otherwise fallback to candle close
        spot = live_spot if (live_spot and live_spot > 0) else hist_spot

        # Debug Sync
        if scanner.scan_count % 10 == 0:
            logger.info(f"🔍 Price Resolution for {name}: Broker={live_spot}, Hist={hist_spot} -> Final={spot}")

        # simulation_id is only set during actual re-simulations (see configure())

        # â• â•  INTELLIGENCE â• â•
        chain_started = time.perf_counter()
        if historical_mode:
            # Per-decision replay intelligence resolves only timestamped archives in
            # _compute_historical_intel_at. Never borrow current chain/depth here.
            chain = None
            chain_quality = unavailable_chain_quality()
            depth_context = unavailable_depth()
        else:
            allow_chain_fetch = scanner._chain_cache.get(name) is None
            chain, chain_quality = scanner._get_option_chain_cached(
                name, cfg, allow_fetch=allow_chain_fetch
            )
            depth_context = scanner._get_depth_context(name, cfg)
        chain_quality = {**chain_quality, "depth": depth_context}
        scanner._record_scan_phase_ms(
            "chain_depth_ms",
            (time.perf_counter() - chain_started) * 1000.0,
            instrument=name,
        )
        candle_1min = scanner._live_analysis_window(scanner.candles.get_candles(name, "1min"), "1min")
        candle_15min = scanner._live_analysis_window(scanner.candles.get_candles(name, "15min"), "15min")

        # â• â•  Volume Spike Detection â• â•
        now = time.time()
        if candle_1min is not None and len(candle_1min) > 12:
            last_volume = candle_1min['volume'].iloc[-1]
            avg_volume = candle_1min['volume'].iloc[:-1].tail(12).mean()

            if avg_volume > 0 and last_volume >= 2 * avg_volume:
                last_alert = scanner._last_vol_alert_time.get(name, 0)
                if now - last_alert >=  30.0:
                    last_open = float(candle_1min['open'].iloc[-1])
                    last_close = float(candle_1min['close'].iloc[-1])
                    pressure = "BUYING" if last_close > last_open else "SELLING"
                    msg = f"🔈 VOLUME SPIKE ({pressure} PRESSURE) detected for {name}: Current volume ({last_volume:.0f}) is 2x+ the average ({avg_volume:.0f})!"
                    logger.warning(msg)
                    if scanner.on_notification:
                        scanner.on_notification(msg, "warning")
                    scanner._last_vol_alert_time[name] = now

        # Use background-cached intel if fresh enough (avoids redundant heavy computation)
        intelligence_started = time.perf_counter()
        intel_ttl = scanner._intel_analysis_ttl_seconds()
        historical_analysis_ts = None
        if historical_mode and candle_5min is not None and not candle_5min.empty:
            raw_historical_ts = candle_5min.index[-1]
            historical_analysis_ts = (
                raw_historical_ts.to_pydatetime()
                if hasattr(raw_historical_ts, "to_pydatetime")
                else raw_historical_ts
            )
            cached_intel = scanner._get_historical_intel_context_at(
                name, historical_analysis_ts, "5min"
            )
            cached_intel_time = float(pd.Timestamp(historical_analysis_ts).timestamp())
        else:
            cached_intel = scanner._cached_timeframe_intel(name, "5min", allow_fallback=False)
            cached_intel_time = getattr(self, '_intel_cache_time', {}).get(scanner._timeframe_intel_key(name, "5min"), 0) or getattr(self, '_intel_cache_time', {}).get(name, 0)
        regime_history_ready = scanner._regime_history_ready(candle_5min)
        if not regime_history_ready:
            scanner._clear_timeframe_intel(name, "5min")
            cached_intel = None
        intel_is_fresh = (
            bool(cached_intel)
            if historical_mode
            else (
                regime_history_ready
                and scanner._intel_result_cacheable(cached_intel)
                and (now - cached_intel_time) < intel_ttl
                and not scanner._intel_bundle_predates_bar(name, "5min", scanner._intel_bar_start(name, "5min", candle_5min))
            )
        )

        intel_computed_this_pass = False
        if intel_is_fresh:
            intel_result = cached_intel
        else:
            try:
                if scanner.mode == "HISTORICAL":
                    analysis_frame = (
                        candle_5min
                        if candle_5min is not None and not candle_5min.empty
                        else candle_1min
                    )
                    if analysis_frame is not None and not analysis_frame.empty:
                        raw_ts = analysis_frame.index[-1]
                        analysis_ts = (
                            raw_ts.to_pydatetime()
                            if hasattr(raw_ts, "to_pydatetime")
                            else raw_ts
                        )
                    else:
                        analysis_ts = scanner.candles.get_max_timestamp() or datetime.now(IST)
                    historical_vix = scanner._get_historical_vix(analysis_ts)
                    historical_percentile = scanner._vix_percentile(historical_vix, analysis_ts)
                    vol_context = {
                        "value": historical_vix,
                        "percentile": historical_percentile,
                        "source": (
                            "india_vix_daily_percentile"
                            if historical_percentile is not None
                            else "india_vix_insufficient_history"
                        ),
                    }
                else:
                    vol_context = scanner._get_live_volatility_context()
                iv_as_of = analysis_ts if scanner.mode == "HISTORICAL" else datetime.now(IST)
                context_previous_close = scanner._previous_close_for(name)
                dte = scanner._days_to_expiry_for(name, iv_as_of)
                contexts = scanner._build_intelligence_contexts(
                    name,
                    candle_5min,
                    spot=spot,
                    previous_close=context_previous_close,
                    volatility_context=vol_context,
                    as_of=iv_as_of,
                )
                intel_result = scanner.intel.analyze(
                    instrument=name,
                    timeframe="5min",
                    candle_df=candle_5min if candle_5min is not None else pd.DataFrame(),
                    candle_1min_df=candle_1min,
                    options_chain=chain,
                    spot_price=spot,
                    strike_interval=strike_interval,
                    days_to_expiry=dte,
                    price_change_pct=change_pct,
                    chain_quality=chain_quality,
                    volatility_value=vol_context.get("value"),
                    volatility_percentile=vol_context.get("percentile"),
                    volatility_source=vol_context.get("source", ""),
                    instrument_iv_observations=scanner._instrument_iv_observations(name, iv_as_of),
                    session_context=contexts.get("session_context"),
                    anchor_regime=contexts.get("anchor_regime"),
                    breadth_context=contexts.get("breadth_context"),
                )
                intel_computed_this_pass = True
                if scanner._intel_result_cacheable(intel_result):
                    scanner._cache_timeframe_intel(
                        name, "5min", intel_result, at=now, mirror_legacy=True,
                        bar_start=scanner._intel_bar_start(name, "5min", candle_5min),
                    )
                    scanner._record_instrument_iv(name, intel_result, iv_as_of)

                if scanner._regime_history_ready(candle_15min):
                    try:
                        scanner._get_timeframe_intel(
                            name,
                            "15min",
                            chain=chain,
                            chain_quality=chain_quality,
                            depth_context=depth_context,
                            spot=spot,
                            cfg=cfg,
                            price_change_pct=change_pct,
                            volatility_context=vol_context,
                            session_context=contexts.get("session_context"),
                            anchor_regime=contexts.get("anchor_regime"),
                            breadth_context=contexts.get("breadth_context"),
                            now=now,
                            allow_fallback=False,
                        )
                    except Exception as exc:
                        logger.debug(f"15min intelligence warmup skipped for {name}: {exc}")

                # Log for verification
                pcr_val = intel_result.get('pcr', {}).get('pcr_oi', 0)
                call_delta = intel_result.get('greeks', {}).get('call', {}).get('delta', 0)
                logger.info(f"Intelligence for {name}: PCR={pcr_val}, Call Delta={call_delta}")
            except Exception as e:
                import traceback
                failure = AnalysisBoundaryFailure(
                    f"intelligence unavailable for {name}", cause=e, retryable=True
                )
                logger.error(f"{failure}: {e}\n{traceback.format_exc()}")
                # analyze() may have returned and a later cache/telemetry action may
                # have failed. The fallback below is still the old snapshot, so it must
                # never inherit this pass's freshness flag.
                intel_computed_this_pass = False
                intel_result = cached_intel or {
                    "pcr": {"pcr": 1.0, "signal": "NEUTRAL"},
                    "oi": {"signal": "NEUTRAL", "cumulative_analysis": {}},
                    "greeks": {},
                    "volume": {"buy_sell_ratio": 1.0},
                    "order_flow": {"ratio": 1.0},
                    "regime": {"regime": "UNKNOWN"},
                    "abstention": failure.abstention().as_dict(),
                }

        # Multi-horizon live regimes for Regime panel (1–5m / hourly / day session).
        if not historical_mode:
            try:
                _sc = (intel_result or {}).get("session_context") if isinstance(intel_result, dict) else {}
                _br = (intel_result or {}).get("breadth_context") if isinstance(intel_result, dict) else {}
                horizons = scanner._live_regime_horizons(
                    name,
                    session_context=_sc if isinstance(_sc, dict) else None,
                    breadth_context=_br if isinstance(_br, dict) else None,
                )
                if isinstance(intel_result, dict):
                    intel_result = dict(intel_result)
                    intel_result["regime_horizons"] = horizons
                    # Keep aggregate in sync for UI fallbacks.
                    agg = intel_result.get("aggregate") if isinstance(intel_result.get("aggregate"), dict) else {}
                    agg = dict(agg)
                    agg["regime_horizons"] = horizons
                    intel_result["aggregate"] = agg
            except Exception as _rh_exc:
                logger.debug(f"regime_horizons build failed for {name}: {_rh_exc}")

        scanner._record_scan_phase_ms(
            "intelligence_ms",
            (time.perf_counter() - intelligence_started) * 1000.0,
            instrument=name,
        )

        results["intelligence"] = intel_result # Store in results dict
        if not historical_mode:
            scanner._record_bar_aligned_intelligence(name, chain, intel_result, chain_quality, depth_context)
        try:
            pre_candle_prediction = scanner._build_pre_candle_predictions(
                name,
                mtf_result,
                intel_result,
                candle_1min,
                candle_5min,
                candle_15min,
            )
        except Exception as exc:
            logger.debug(f"Pre-candle prediction failed for {name}: {exc}")
            pre_candle_prediction = {}

        # â• â• â•  Record Intelligence Memory â• â• â•
        scan_ts = (
            float(pd.Timestamp(historical_analysis_ts).timestamp())
            if historical_mode and historical_analysis_ts is not None
            else time.time()
        )
        scanner.intel_memory.record(name, scan_ts, intel_result)
        if scanner.scan_count % 300 == 0: scanner._schedule_intel_memory_save() # Periodically persist

        intel_score = aggregate_intelligence_score(intel_result.get("aggregate", {}))
        regime = intel_result.get("regime", {}).get("regime", "UNKNOWN")
        if scanner._intel_result_cacheable(intel_result) and intel_computed_this_pass:
            # `at=now` only when this is a FRESH computation. On the cache-hit path
            # intel_result IS cached_intel, and re-stamping it renewed the TTL on every
            # read — a 90s bound that a 1s scan loop could push out indefinitely, so
            # intel had no enforced age at all. Preserve the original stamp instead:
            # _get_timeframe_intel already returns a hit without re-caching, which is
            # the correct pattern.
            scanner._cache_timeframe_intel(name, "5min", intel_result, at=now, mirror_legacy=True)
        else:
            scanner.latest_regimes[name] = str(regime or "UNKNOWN").upper()

        # â• â• â•  ATM Strike Resolution â• â• â•
        atm_strike = round(spot / strike_interval) * strike_interval if spot > 0 else 0

        # â• â• â•  BEST SIGNAL SELECTION across timeframes â• â• â•
        signal_parts = []
        # The key is the cache's identity. 1min was absent from it, so a brand-new
        # 1-minute signal did not change the key and the 15-second REAL-mode cache
        # kept serving a candidate list computed before that signal existed.
        for tf_name, tf_result in (
            ("1min", getattr(mtf_result, "results_1min", None)),
            ("5min", mtf_result.results_5min),
            ("15min", mtf_result.results_15min),
        ):
            sigs = tf_result.get("signals", []) if tf_result else []
            last_sig = sigs[-1] if sigs else None
            if last_sig:
                signal_parts.append(
                    f"{tf_name}:{last_sig.timestamp}:{last_sig.signal_type}:{round(float(last_sig.price or 0), 2)}"
                )
            else:
                signal_parts.append(f"{tf_name}:none")
        decision_key = "|".join(signal_parts + [str(scanner.inst_pref), str(getattr(get_settings(), "signal_grade_preference", ""))])
        decision_cache = scanner._candidate_process_cache.get(name)
        has_forming_signal = scanner._mtf_has_forming_signal(mtf_result)
        if not regime_history_ready:
            candidates = []
            scanner._candidate_process_cache.pop(name, None)
            scanner.log_signal_decision_once(
                f"{name}|regime-warmup|{len(candle_5min) if candle_5min is not None else 0}",
                (
                    f"Signal grading deferred: {name} regime history warming "
                    f"({len(candle_5min) if candle_5min is not None else 0}/30 five-minute candles)"
                ),
                "data",
            )
        elif (
            scanner.mode == "REAL"
            and not has_forming_signal
            and decision_cache
            and decision_cache.get("key") == decision_key
            and now - decision_cache.get("time", 0.0) <= 15.0
        ):
            candidates = decision_cache.get("candidates", [])
        else:
            if scanner.mode == "HISTORICAL":
                regime_reset = getattr(
                    getattr(getattr(self, "intel", None), "regime", None),
                    "reset_history",
                    None,
                )
                if callable(regime_reset):
                    regime_reset(name)
            candidates = scanner.signal_processor.process_best_signal(
                name, mtf_result, intel_result, intel_score, regime,
                lots, lot_size, cfg, spot, atm_strike,
            )
            # R3: register direction votes as each index grades (same-scan peers)
            observed_at = datetime.now(IST)
            for candidate in candidates or []:
                candidate.analysis_observed_at = observed_at
            try:
                for _vc in list(candidates or []):
                    if getattr(_vc, "action", "ENTRY") != "ENTRY":
                        continue
                    scanner._register_heavy_trend_vote(
                        name,
                        str(getattr(_vc, "direction", "") or ""),
                        str(getattr(_vc, "timeframe", "") or ""),
                        signal_ts=getattr(_vc, "signal_timestamp", None),
                    )
            except Exception:
                pass
            if scanner.mode == "REAL" and not has_forming_signal:
                scanner._candidate_process_cache[name] = {
                    "key": decision_key,
                    "time": now,
                    "candidates": candidates,
                }
        source_snapshot = scanner.data.get_market_source_snapshot(name)
        candidates = scanner._filter_live_candidates_by_source(name, candidates, source_snapshot)
        display_candidates = scanner._get_session_trade_candidates(name)
        if not display_candidates:
            display_candidates = (
                scanner._latest_exit_candidates.get(name, [])
                + scanner._latest_trade_candidates.get(name, [])
            )
        # Display refresh shares the manual-monitor ledger lock. Broker I/O here
        # would hold that lock in this worker while timeout fallbacks wait on it.
        scanner._refresh_trade_candidates(display_candidates, cfg, allow_blocking=False)
        # Use first candidate for RR display if any
        primary_candidate = candidates[0] if candidates else None

        # â• â• â•  Build chart from CACHED results — NO re-processing â• â• â•
        chart_data = scanner._build_chart_from_cache(name) if getattr(self, "chart_stream_enabled", True) else {}

        # Optimization: Strip heavy chart data for instruments NOT in active_indices
        # (Though active_indices filter already applied in _scan_cycle, this is a safety check)

        # Calculate Trade State for UI
        active_trade = next((t for t in getattr(scanner.trades, 'open_trades', {}).values() if getattr(t, 'instrument', '') == name), None) if hasattr(self, 'trades') else None
        open_mtm = 0.0
        time_in_trade = '--'
        risk_exposure = 0.0
        if active_trade:
            open_mtm = (spot - active_trade.entry_price) * (1 if active_trade.direction == 'LONG' else -1) * lot_size
            duration_secs = (datetime.now().astimezone(IST) - active_trade.entry_time).total_seconds() if active_trade.entry_time else 0
            m, s = divmod(int(max(0, duration_secs)), 60)
            h, m = divmod(m, 60)
            time_in_trade = f'{h:02d}:{m:02d}:{s:02d}'
            stop_px = float(
                getattr(active_trade, "stop_loss", 0.0)
                or getattr(active_trade, "current_stop", 0.0)
                or getattr(active_trade, "trailing_stop", 0.0)
                or 0.0
            )
            risk_exposure = abs(active_trade.entry_price - stop_px) * lot_size * -1

        # Calculate ATR exhaustion (approx)
        state_5m = mtf_result.results_5min.get('state', {}) if mtf_result.results_5min else {}
        atr_val = float(state_5m.get('atr_value') or 0)
        try:
            atr_exhaustion_str = f'ATR {int(atr_val)} pts'
        except Exception:
            atr_exhaustion_str = '--'

        quote_label, quote_session, is_preopen, preopen_settlement = scanner._dashboard_quote_session(
            market_quote
        )
        display_intel = intel_result
        panel = getattr(scanner, "_historical_panel_intel", None)
        if historical_mode and callable(panel):
            display_intel = panel(name)

        ui_data = {
            'trade_state': {
                'open_mtm': open_mtm,
                'time_in_trade': time_in_trade,
                'risk_exposure': risk_exposure,
                'next_trigger': 'Ready',
                'overbought_oversold': scanner._overbought_oversold(name),
                'vwap_deviation': scanner._vwap_deviation(name),
                'atr_exhaustion': atr_exhaustion_str,
            },
            "mtf": {
                "confluence_score": mtf_result.confluence_score,
                "confluence_signal": mtf_result.confluence_signal,
                "state_1min": mtf_result.results_1min.get("state") if mtf_result.results_1min else None,
                "state_5min": mtf_result.results_5min.get("state") if mtf_result.results_5min else None,
                "state_15min": mtf_result.results_15min.get("state") if mtf_result.results_15min else None,
            },
            "intelligence": {
                **{key: display_intel.get(key, {}) for key in (
                    "volume", "option_volume", "oi", "pcr", "greeks", "regime",
                    "regime_horizons", "order_flow", "aggregate", "market_structure",
                    "wavetrend", "max_pain", "data_quality", "session_context",
                )},
                "aroon": display_intel.get("aroon_context", {}),
                "as_of": display_intel.get("as_of"),
                "potential_rr": primary_candidate.rr if primary_candidate else 1.5,
            },
            "pre_candle_prediction": pre_candle_prediction,
            "chart": chart_data,
            "signals": scanner.signals.get_active_signals(mode=scanner.mode),
            "trade_candidates": [scanner._serialize_trade_candidate(c) for c in display_candidates],
            "simulation_id": scanner.simulation_id,
            "atm_strike": atm_strike,
            "ltp": spot,
            "change_points": change_points,
            "change_pct": change_pct,
            "change": change_points,
            "prev_close": prev_close,
            "previous_close": prev_close,
            "quote_source": market_quote.get("source", "unknown"),
            "quote_label": quote_label,
            "quote_session": quote_session,
            "quote_age_seconds": market_quote.get("quote_age_seconds"),
            "is_preopen": is_preopen,
            "preopen_settlement": preopen_settlement,
            "market_source": source_snapshot,
            "spot_price": spot,
        }

        # â• â•  BROADCAST OPTIMIZATION â• â•
        # If this is the currently selected instrument in the dashboard, we MUST include chart data.
        # Otherwise, we strip it to save 90% of bandwidth.
        # Note: The main loop in _scan_cycle handles stripping based on is_full_chart_needed.

        return ui_data, candidates
