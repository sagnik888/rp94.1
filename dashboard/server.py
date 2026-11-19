"""
Dashboard Server — FastAPI + WebSocket for Real-Time Dashboard
═══════════════════════════════════════════════════════════════
"""
import asyncio
import base64
import binascii
import hashlib
import inspect
import json
import math
import os
import secrets
import time
import weakref
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from http.cookies import SimpleCookie
from typing import Set, Dict, Any, Optional, Callable
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from config.settings import get_settings
from engine.settings_persistence import schedule_settings_save

try:
    import orjson
except Exception:  # pragma: no cover - optional fast path
    orjson = None

# WebSocket connections
active_connections: Set[WebSocket] = set()
_pending_ws_connections: Set[int] = set()
_ws_command_times: Dict[int, deque] = {}
_ws_rate_violations: Dict[int, int] = {}

# Reference to scanner (set by main.py)
scanner_ref = None

# Periodic dashboard task managed by FastAPI lifespan.
_periodic_trades_task: Optional[asyncio.Task] = None
_broadcast_task: Optional[asyncio.Task] = None
_broadcast_queue: Optional[asyncio.Queue] = None
_startup_task: Optional[asyncio.Future] = None
_ipv6_localhost_proxy: Optional[asyncio.base_events.Server] = None
startup_hook: Optional[Callable[[], Any]] = None
shutdown_hook: Optional[Callable[[], Any]] = None
_dashboard_shutting_down = False
_dashboard_runtime_token = secrets.token_urlsafe(32)
_last_periodic_trades_hash: Optional[str] = None
# Event-driven wake for the trades updater so entry/exit refresh the panel
# immediately instead of waiting for the next 1s tick. Created on the dashboard
# loop; set thread-safely from the scanner via notify_trade_change().
_trade_dirty_event: Optional[asyncio.Event] = None
_trade_dirty_loop: Optional[asyncio.AbstractEventLoop] = None


def notify_trade_change() -> None:
    """Thread-safe poke to push an immediate trade-panel refresh (entry/exit)."""
    loop = _trade_dirty_loop
    event = _trade_dirty_event
    if loop is None or event is None:
        return
    try:
        loop.call_soon_threadsafe(event.set)
    except Exception:
        pass
_DASHBOARD_STATE_TRADE_LIMIT = 80
_HISTORICAL_TRADE_PAGE_SIZE = 100
_latest_full_trades_payload: Dict[str, Any] = {}


def _schedule_on_scanner_loop(coro) -> bool:
    scanner_loop = getattr(scanner_ref, "_event_loop", None)
    if scanner_loop is not None:
        if not scanner_loop.is_running():
            if inspect.iscoroutine(coro):
                coro.close()
            return False
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if scanner_loop is current_loop:
            scanner_loop.create_task(coro)
        else:
            try:
                asyncio.run_coroutine_threadsafe(coro, scanner_loop)
            except Exception:
                if inspect.iscoroutine(coro):
                    coro.close()
                return False
        return True
    try:
        asyncio.create_task(coro)
        return True
    except RuntimeError:
        if inspect.iscoroutine(coro):
            coro.close()
        return False


async def _await_on_scanner_loop(coro, timeout_seconds: float = 8.0):
    scanner_loop = getattr(scanner_ref, "_event_loop", None)
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    if scanner_loop is not None:
        if not scanner_loop.is_running():
            if inspect.iscoroutine(coro):
                coro.close()
            raise RuntimeError("scanner event loop is not running")
        if scanner_loop is not current_loop:
            try:
                fut = asyncio.run_coroutine_threadsafe(coro, scanner_loop)
            except Exception:
                if inspect.iscoroutine(coro):
                    coro.close()
                raise
            return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=timeout_seconds)
    return await coro


# Chart builds are the one heavy, self-contained piece of work this server does, and they
# used to run ON the scanner's own event loop via _call_on_scanner_loop — the same loop
# whose `while self.is_running:` body awaits _scan_cycle(). A rebuild there delays signal
# generation and every other dashboard call marshalled onto that loop.
#
# A DEDICATED pool, not asyncio's default one: every asyncio.to_thread in this file
# (scanner join, trades.save_state, volume_source_health) shares the default executor, so
# a chart build there would queue behind a state save and vice versa. Two workers is
# enough — builds are ~10-20ms and _build_chart_from_cache serialises itself anyway.
_chart_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chart")
_chart_request_locks = weakref.WeakKeyDictionary()


def _chart_request_lock_for_loop() -> asyncio.Lock:
    """Serialize chart selection through build without binding a lock across test/app loops."""
    loop = asyncio.get_running_loop()
    lock = _chart_request_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _chart_request_locks[loop] = lock
    return lock


async def _build_chart_off_loop(instrument: str, timeframe: str):
    """Select the chart on the scanner loop, then BUILD it on the chart pool.

    The selection half (active_chart_instrument / active_chart_tf) has to stay on the
    scanner loop: the scan cycle reads those two attributes to decide which chart to keep
    warm, so setting them from another thread could have it refresh the wrong one. It is
    two attribute writes. The build that follows is the expensive part and is safe off the
    loop because _build_chart_from_cache takes its own lock.
    """
    if not scanner_ref:
        return {}
    # Selection is shared scanner state. Keep it paired with the worker build so
    # an overlapping REST/WebSocket request cannot switch the active instrument
    # or timeframe between selection and _build_chart_from_cache().
    async with _chart_request_lock_for_loop():
        await _call_on_scanner_loop(
            lambda: (
                setattr(scanner_ref, "active_chart_instrument", instrument),
                setattr(scanner_ref, "active_chart_tf", timeframe),
            )
        )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _chart_executor,
            lambda: scanner_ref.get_chart_snapshot(instrument=instrument, timeframe=timeframe),
        )


async def _call_on_scanner_loop(
    func: Callable[[], Any],
    *,
    timeout_seconds: float = 15.0,
) -> Any:
    """Execute one synchronous scanner mutation on the scanner-owned loop."""

    async def _invoke():
        return func()

    return await _await_on_scanner_loop(_invoke(), timeout_seconds=timeout_seconds)


async def _configure_on_scanner_loop(**changes):
    scanner = scanner_ref

    async def apply():
        configure = getattr(scanner, "configure_async", None)
        if callable(configure):
            return await configure(**changes)
        return scanner.configure(**changes)

    return await _await_on_scanner_loop(apply(), timeout_seconds=45.0)


def _start_on_scanner_loop(func: Callable[[], Any]) -> asyncio.Future:
    """Start one synchronous scanner call on the scanner-owned loop; return its future.

    Unlike _call_on_scanner_loop nothing here times out or cancels. The caller decides
    how long to wait, and a result that arrives late is still a good result -- throwing
    it away and asking again is how the trades heartbeat saturated the loop.
    """

    async def _invoke():
        return func()

    coro = _invoke()
    scanner_loop = getattr(scanner_ref, "_event_loop", None)
    if scanner_loop is not None and scanner_loop is not asyncio.get_running_loop():
        if not scanner_loop.is_running():
            coro.close()
            raise RuntimeError("scanner event loop is not running")
        try:
            return asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, scanner_loop))
        except Exception:
            coro.close()
            raise
    return asyncio.ensure_future(coro)


def _schedule_call_on_scanner_loop(func: Callable[[], Any]) -> bool:
    """Schedule a synchronous scanner mutation without crossing loop ownership."""

    async def _invoke():
        return func()

    return _schedule_on_scanner_loop(_invoke())


def _dashboard_ws_limits() -> tuple[int, int, float, int]:
    settings = get_settings()
    max_connections = max(
        1,
        min(64, int(getattr(settings, "dashboard_ws_max_connections", 8) or 8)),
    )
    max_commands = max(
        1,
        min(500, int(getattr(settings, "dashboard_ws_max_commands_per_window", 30) or 30)),
    )
    window_seconds = max(
        0.25,
        min(60.0, float(getattr(settings, "dashboard_ws_command_window_seconds", 1.0) or 1.0)),
    )
    max_violations = max(
        1,
        min(20, int(getattr(settings, "dashboard_ws_max_rate_violations", 3) or 3)),
    )
    return max_connections, max_commands, window_seconds, max_violations


def _record_ws_command(ws: WebSocket) -> tuple[bool, int]:
    """Sliding-window command limiter scoped to one authenticated connection."""
    _, max_commands, window_seconds, _ = _dashboard_ws_limits()
    ws_id = id(ws)
    now = time.monotonic()
    history = _ws_command_times.setdefault(ws_id, deque())
    cutoff = now - window_seconds
    while history and history[0] < cutoff:
        history.popleft()
    history.append(now)
    if len(history) <= max_commands:
        _ws_rate_violations.pop(ws_id, None)
        return True, 0
    violations = int(_ws_rate_violations.get(ws_id, 0) or 0) + 1
    _ws_rate_violations[ws_id] = violations
    return False, violations


def _forget_ws_connection(ws: WebSocket) -> None:
    ws_id = id(ws)
    active_connections.discard(ws)
    _pending_ws_connections.discard(ws_id)
    _ws_command_times.pop(ws_id, None)
    _ws_rate_violations.pop(ws_id, None)
    dashboard_state.last_sent_states.pop(ws_id, None)


async def _close_active_websockets(code: int = 1012) -> None:
    """Close and forget all dashboard clients without serial shutdown waits."""
    clients = list(active_connections)
    if clients:
        async def _close_one(ws: WebSocket) -> None:
            try:
                await asyncio.wait_for(ws.close(code=code), timeout=1.0)
            except Exception:
                pass

        await asyncio.gather(*(_close_one(ws) for ws in clients), return_exceptions=True)
    for ws in clients:
        _forget_ws_connection(ws)


def _dashboard_token() -> str:
    return str(getattr(get_settings(), "dashboard_auth_token", "") or _dashboard_runtime_token)


_WS_APP_PROTOCOL = "ut1.dashboard.v1"
_WS_AUTH_PROTOCOL_PREFIX = "ut1.auth."


def _encode_ws_auth_protocol(token: str) -> str:
    """Encode a credential as an RFC-compatible WebSocket subprotocol token."""
    encoded = base64.urlsafe_b64encode(str(token).encode("utf-8")).decode("ascii")
    return _WS_AUTH_PROTOCOL_PREFIX + encoded.rstrip("=")


def _decode_ws_auth_protocol(protocol: str) -> Optional[str]:
    value = str(protocol or "")
    if not value.startswith(_WS_AUTH_PROTOCOL_PREFIX):
        return None
    encoded = value[len(_WS_AUTH_PROTOCOL_PREFIX):]
    if not encoded or len(encoded) > 4096:
        return None
    try:
        padding = "=" * (-len(encoded) % 4)
        raw = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
        return raw.decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None


def _ws_protocols(ws: WebSocket) -> list[str]:
    header = str(ws.headers.get("sec-websocket-protocol", "") or "")
    return [part.strip() for part in header.split(",") if part.strip()]


def _ws_authentication(ws: WebSocket) -> tuple[bool, Optional[str]]:
    """Authenticate a WS handshake without reading credentials from its URL."""
    protocols = _ws_protocols(ws)
    candidates: list[tuple[str, Optional[str]]] = []

    authorization = str(ws.headers.get("authorization", "") or "").strip()
    scheme, separator, bearer = authorization.partition(" ")
    if separator and scheme.lower() == "bearer" and bearer.strip():
        candidates.append((bearer.strip(), None))

    header_token = ws.headers.get("x-dashboard-token")
    if header_token:
        candidates.append((str(header_token), None))

    for protocol in protocols:
        decoded = _decode_ws_auth_protocol(protocol)
        if decoded is not None:
            candidates.append((decoded, protocol))

    # Cookie auth is useful for non-JS clients and remains protected by the
    # strict Origin check. The app does not put this credential into a URL.
    try:
        cookie_header = str(ws.headers.get("cookie", "") or "")
        cookies = SimpleCookie()
        cookies.load(cookie_header)
        for name in ("ut1_dashboard_token", "dashboard_token"):
            if name in cookies:
                candidates.append((cookies[name].value, None))
    except Exception:
        pass

    for candidate, auth_protocol in candidates:
        if not _token_matches(candidate):
            continue
        if _WS_APP_PROTOCOL in protocols:
            return True, _WS_APP_PROTOCOL
        # If a client offered only the auth carrier, echo it as the selected
        # protocol so strict WebSocket implementations complete negotiation.
        return True, auth_protocol
    return False, None


def _allowed_origins() -> Set[str]:
    settings = get_settings()
    host = str(getattr(settings, "dashboard_host", "127.0.0.1") or "127.0.0.1")
    port = int(getattr(settings, "dashboard_port", 8001) or 8001)
    origins = {
        f"http://{host}:{port}",
        f"http://127.0.0.1:{port}",
        f"http://localhost:{port}",
    }
    origins.update(str(o).rstrip("/") for o in getattr(settings, "dashboard_allowed_origins", []) or [] if o)
    return origins


def _is_allowed_origin(origin: Optional[str]) -> bool:
    if not origin:
        return True
    parsed = urlparse(origin)
    if parsed.hostname in {"127.0.0.1", "localhost"}:
        return True
    return origin.rstrip("/") in _allowed_origins()


def _token_matches(token: Optional[str]) -> bool:
    expected = _dashboard_token()
    return bool(token) and secrets.compare_digest(str(token), expected)


def _request_authorized(request: Request) -> bool:
    origin = request.headers.get("origin") or request.headers.get("referer")
    token = request.headers.get("x-dashboard-token")
    return _is_allowed_origin(origin) and _token_matches(token)


# Strict per-command schema: base fields allowed on any command, plus the explicit
# set of fields each command accepts. Unknown commands AND unknown fields are rejected,
# and known numeric fields are type/range-validated before the scanner sees them.
_BASE_COMMAND_FIELDS = {"cmd", "request_id"}
_ADVANCED_CONFIG_FIELDS = {
    # Peak-anchored give-back cap (2026-08-17). Tunable from the dashboard because the
    # cap is the one number that must be re-validated against replayed pnl_path data
    # before it is moved — below Rs.200 the measured hurt count triples.
    "peak_giveback_cap_enabled",
    "peak_giveback_cap_arm_rs",
    "peak_giveback_cap_rs",
    "peak_giveback_cap_pct",
    "peak_giveback_cap_max_peak_rs",
    "breakeven_stop_enabled",
    "breakeven_stop_arm_rs",
    "breakeven_stop_floor_rs",
    "peak_trail_enabled",
    "peak_trail_above_rs",
    "peak_trail_giveback_pct",
    "backtest_live_exits_enabled",
    "entry_preopen_signal_guard_enabled",
    "entry_mark_touch_validation_enabled",
    "entry_mark_max_ask_multiple",
    "entry_mark_min_bid_multiple",
    "candidate_entry_mark_validation_enabled",
    "candidate_entry_mark_max_divergence_pct",
    "candidate_entry_mark_min_gap_rs",
    "block_entries_during_eod_protect",
    "profit_lock_min_peak_rs",
    "profit_lock_nonscalp_min_peak_rs",
    "profit_lock_min_r_multiple",
    "profit_protect_cost_floor_enabled",
    "profit_protect_min_cost_multiple",
    "first_rung_regime_retention_enabled",
    "first_rung_fast_regime_retention",
    "profit_threshold_scaling_enabled",
    "profit_threshold_regime_scaling_enabled",
    "profit_threshold_reference_risk_rs",
    "profit_threshold_scale_min",
    "profit_threshold_scale_max",
    "pnl_path_logging_enabled",
    "pnl_path_min_interval_seconds",
    "pnl_path_min_move_rs",
    "pnl_path_max_samples",
    "slippage_calibration_max_age_days",
    "stabilization_rtracker_logging_enabled",
    "entry_intel_snapshot_enabled",
    "stabilization_cascade_latch_enabled",
    "stabilization_thrust_scaling_enabled",
    "stabilization_thrust_min_seconds",
    "stabilization_thrust_range_atr",
    "stabilization_thrust_volume_ratio",
    "range_scaled_budget_enabled",
    "range_scaled_budget_reference_pct",
    "range_scaled_budget_min_trades",
    "missed_signal_reentry_scored_mode_enabled",
    "missed_signal_reentry_min_conditions",
    "block_opposite_direction_stack",
    "consecutive_loss_streak_per_index",
    "surge_tick_proxy_enabled",
    "intel_reversal_corroboration_enabled",
    "partial_entry_remainder_cancel_enabled",
    "morning_refresh_catchup_at_live_open",
    "surge_tick_percentile_floor",
    "intel_reversal_corroboration_votes",
    "partial_entry_remainder_timeout_seconds",
    "kite_rest_backoff_seconds",
    "kite_rest_backoff_max_seconds",
    "calendar_coverage_warn_days",
    "lean_routing_to_options",
    "price_action_module_enabled",
    "price_action_block_adverse",
    "price_action_adverse_threshold",
    "price_action_reversal_relief_enabled",
    "regime_veto_enabled",
    "regime_veto_adx_floor",
    "regime_early_trend_er_promotion_enabled",
    "regime_early_trend_er_period",
    "regime_early_trend_er_threshold",
    "regime_early_trend_er_adx_ceiling",
    "regime_early_trend_er_range_day_min_confidence",
    "fut_early_loss_cut_enabled",
    "fut_loss_cut_r",
    "power_move_override_enabled",
    "power_move_late_hard_cutoff",
    "power_move_spike_bypass_15m",
    "power_move_spike_range_atr_mult",
    "live_5min_trend_continuation_enabled",
    "live_5min_trend_continuation_adx",
    "volume_spike_entry_enabled",
    "one_min_scalping_enabled",
    "fifteen_min_one_min_support_exit_enabled",
    "volume_spike_entry_min_volume_ratio",
    "volume_spike_entry_min_range_atr_ratio",
    "volume_spike_entry_min_close_location",
    "volume_spike_entry_futures_sl_pct",
    "volume_spike_entry_options_sl_pct",
    "volume_spike_entry_stabilization_seconds",
    "volume_spike_entry_stabilization_seconds_min",
    "volume_spike_entry_stabilization_seconds_max",
    "volume_spike_depth_confirm_enabled",
    "volume_spike_depth_ofr_bullish",
    "volume_spike_depth_ofr_bearish",
    "stabilization_momentum_override_enabled",
    "live_signal_stabilization_seconds_5min",
    "live_signal_stabilization_seconds_15min",
    "live_15min_momentum_override_enabled",
    "live_15min_momentum_min_confidence",
    "live_15min_momentum_volume_ratio",
    "live_15min_momentum_range_atr_ratio",
    "live_15min_momentum_close_location",
    "live_15min_momentum_adx_threshold",
    "live_15min_momentum_intel_threshold",
    "reversal_entry_relief_enabled",
    "reversal_entry_min_confidence",
    "reversal_entry_thrust_min_range_atr_ratio",
    "reversal_entry_thrust_min_close_location",
    "reversal_entry_thrust_min_volume_ratio",
    "reversal_entry_vwap_min_dev_pct",
    "directional_census_reversal_relief_enabled",
    "market_ltp_shock_enabled",
    "market_ltp_shock_min_move_pct",
    "market_ltp_shock_strong_move_pct",
    "market_ltp_shock_lookback_minutes",
    "market_ltp_shock_min_lookback_minutes",
    "market_ltp_shock_min_indices",
    "market_ltp_shock_min_core_indices",
    "market_ltp_shock_min_volume_ratio",
    "market_ltp_shock_session_start",
    "market_ltp_shock_session_end",
    "market_ltp_shock_session_override_start",
    "market_ltp_shock_opposite_trade_relax",
    "market_ltp_shock_force_exit_opposite",
    "market_ltp_shock_stabilization_seconds",
    "session_mood_dynamic_enabled",
    "session_mood_min_confidence",
    "directional_census_reversal_min_confidence",
    "late_session_choppy_entry_guard_enabled",
    "late_session_choppy_min_runway_minutes",
    "market_wide_reversal_enabled",
    "market_wide_reversal_core_votes",
    "market_wide_reversal_total_votes",
    "market_wide_reversal_min_range_atr_ratio",
    "market_wide_reversal_min_range_atr_ratio_5min",
    "market_wide_reversal_min_range_atr_ratio_15min",
    "market_wide_reversal_midcap_min_range_atr_ratio_5min",
    "market_wide_reversal_midcap_min_range_atr_ratio_15min",
    "market_wide_reversal_min_close_location",
    "market_wide_reversal_max_skew_seconds_5min",
    "market_wide_reversal_max_skew_seconds_15min",
    "market_ltp_shock_outside_window_mode",
    "market_ltp_shock_outside_strong_move_pct",
    "post_exit_reversion_enabled",
    "post_exit_reversion_window_minutes",
    "post_exit_reversion_min_rr",
    "post_exit_reversion_include_stop_loss",
    "post_exit_reversion_include_oi_pcr",
    "post_exit_reversion_open_distress_enabled",
    "post_exit_reversion_loss_band_soft_rs",
    "post_exit_reversion_loss_band_deep_rs",
    "post_exit_reversion_min_move_pct",
    "post_exit_reversion_move_lookback_minutes",
    "post_exit_reversion_fast_move_pct",
    "post_exit_reversion_fast_lookback_minutes",
    "post_exit_reversion_slow_move_pct",
    "post_exit_reversion_slow_lookback_minutes",
    "rare_dual_tf_entry_enabled",
    "rare_dual_tf_min_grade",
    "rare_dual_tf_min_confidence",
    "rare_dual_tf_prefer_opt_second",
    "one_min_scalping_high_burst_only",
    "one_way_cascade_enabled",
    "one_way_cascade_core_votes",
    "one_way_cascade_total_votes",
    "one_way_cascade_volume_votes",
    "one_way_cascade_min_volume_ratio",
    "one_way_cascade_min_range_atr_ratio_1min",
    "one_way_cascade_min_range_atr_ratio_5min",
    "one_way_cascade_min_range_atr_ratio_15min",
    "one_way_cascade_min_close_location",
    "one_way_cascade_runner_lock_pct",
    "one_way_cascade_options_sl_pct",
    "one_way_cascade_futures_sl_pct",
    "one_way_cascade_rr",
    "missed_signal_reentry_enabled",
    "missed_signal_reentry_max_age_minutes",
    "missed_signal_reentry_min_age_seconds",
    "missed_signal_reentry_max_chase_atr",
    "missed_signal_reentry_min_confidence",
    "missed_signal_reentry_min_intel_score",
    "missed_signal_reentry_min_trend_adx",
    "missed_signal_reentry_min_remaining_minutes",
    "missed_signal_reentry_allow_trend_context",
    "missed_signal_reentry_prefer_options",
    "missed_signal_reentry_options_sl_pct",
    "missed_signal_reentry_futures_sl_pct",
    "missed_signal_reentry_rr",
    "late_session_reclaim_override_enabled",
    "late_session_reclaim_start",
    "late_session_reclaim_end",
    "late_session_reclaim_core_votes",
    "late_session_reclaim_total_votes",
    "late_session_reclaim_reclaim_votes",
    "late_session_reclaim_max_skew_seconds",
    "late_session_reclaim_min_close_location",
    "late_session_reclaim_min_volume_ratio",
    "late_session_reclaim_volume_votes",
    "late_session_reclaim_options_sl_pct",
    "late_session_reclaim_futures_sl_pct",
    "late_session_reclaim_rr",
    "fifteen_min_choppy_scalp_guard_enabled",
    "fifteen_min_choppy_scalp_min_intel_score",
    "fifteen_min_choppy_scalp_min_volume_ratio",
    "fifteen_min_choppy_scalp_micro_max_skew_seconds",
    "fifteen_min_choppy_scalp_min_close_location",
    "fifteen_min_choppy_scalp_profit_lock_r_multiple",
    "fifteen_min_choppy_scalp_min_peak_rs",
    "peak_threat_exit_enabled",
    "peak_threat_exit_min_peak_rs",
    "peak_threat_exit_min_peak_r",
    "peak_threat_exit_min_retention",
    "peak_threat_exit_min_score",
    "peak_threat_exit_severe_score",
    "peak_threat_exit_adverse_intel",
    "peak_threat_exit_weak_volume_ratio",
    "peak_threat_exit_loss_enabled",
    "peak_threat_exit_loss_min_score",
    "peak_threat_exit_loss_max_loss_r",
    "option_major_win_trend_lock_ratio_cap",
    "midcap_15min_risky_stabilization_seconds",
    "midcap_15min_risky_stabilization_min_scans",
    "midcap_15min_risky_stabilization_min_volume_ratio",
    "midcap_15min_risky_stabilization_min_intel_score",
    "ut_confidence_calibration_enabled",
    "engine_lab_enabled",
    "engine_lab_real_mode_only",
    "dual_lane_enabled",
    "dual_lane_same_dir_min_bars_gap",
    "dual_lane_same_dir_min_lane_a_r",
    "dual_lane_size_fraction",
    "dual_lane_max_index_exposure_rs",
    "dual_lane_min_lane_b_confidence",
    "dual_lane_jaccard_alert_threshold",
    "dual_clone_reduction_enabled",
    "dual_clone_reduction_window_seconds",
    "cascade_lot_scale_enabled",
    "cascade_lot_multiplier_opt",
    "cascade_lot_multiplier_opt_min",
    "cascade_lot_multiplier_fut",
    "heavy_trend_lot_scale_enabled",
    "heavy_trend_lot_multiplier_opt",
    "heavy_trend_min_aligned_indices",
    "heavy_trend_vote_max_age_seconds",
    "cascade_multi_tf_stack_enabled",
    "cascade_primary_max_tf",
    "cascade_dual_addon_max",
    "cascade_midcap_solo_max_tf",
    "cascade_midcap_dual_main_max",
    "cascade_midcap_dual_addon_max",
    "cascade_core_bucket_max_positions",
    "cascade_midcap_bucket_max_positions",
    "cascade_max_concurrent_positions",
    "multi_signal_mixed_book_enabled",
    "multi_signal_max_fut",
    "multi_signal_fut_lots",
    "multi_signal_max_opt_instruments",
    "execution_use_touch_quote",
    "execution_touch_quote_shadow_enabled",
    "execution_touch_quote_strict",
    "quality_hybrid_min_vote_score",
    "quality_hybrid_broad_agreement",
    "quality_hybrid_chop_min_agreement",
    "quality_hybrid_trend_adx",
    "quality_hybrid_trend_er",
    "quality_hybrid_thrust_volume_ratio",
    "quality_hybrid_thrust_range_atr",
    "quality_hybrid_thrust_close_loc",
    "ut_confidence_calibration_min_samples",
    "ut_no_entry_after",
    "ut_5min_no_entry_after",
    "ut_1min_no_entry_after",
    "ut_force_exit_time",
    "real_mode_premarket_arm_start",
    "real_mode_premarket_settle_end",
    "real_mode_premarket_refresh_end",
    "real_mode_live_start",
    "auto_real_mode_premarket_switch_time",
    # Pullback hedge (A/B): OFF by default — archived-session backtest showed it armed
    # once in 8 sessions and lost at every setting (the 85-90% runner lock already
    # captures the give-back more cheaply). Exposed so the operator can A/B it.
    "pullback_hedge_enabled",
    "stop_out_reentry_enabled",
    "per_timeframe_grading_enabled",
    "stop_out_reentry_window_minutes",
    "stop_out_reentry_max_per_index",
    "stop_out_reentry_min_adx",
    "pullback_hedge_trigger_lock_fraction",
    "pullback_hedge_resume_recovery_pct",
    "pullback_hedge_size_fraction",
    "pullback_hedge_max_hold_minutes",
}
_ADVANCED_BOOLEAN_CONFIG_FIELDS = {
    "peak_giveback_cap_enabled",
    "breakeven_stop_enabled",
    "peak_trail_enabled",
    "backtest_live_exits_enabled",
    "entry_preopen_signal_guard_enabled",
    "entry_mark_touch_validation_enabled",
    "candidate_entry_mark_validation_enabled",
    "block_entries_during_eod_protect",
    "profit_protect_cost_floor_enabled",
    "first_rung_regime_retention_enabled",
    "profit_threshold_scaling_enabled",
    "profit_threshold_regime_scaling_enabled",
    "pnl_path_logging_enabled",
    "stabilization_rtracker_logging_enabled",
    "entry_intel_snapshot_enabled",
    "stabilization_cascade_latch_enabled",
    "stabilization_thrust_scaling_enabled",
    "range_scaled_budget_enabled",
    "missed_signal_reentry_scored_mode_enabled",
    "block_opposite_direction_stack",
    "consecutive_loss_streak_per_index",
    "surge_tick_proxy_enabled",
    "intel_reversal_corroboration_enabled",
    "partial_entry_remainder_cancel_enabled",
    "morning_refresh_catchup_at_live_open",
    "pullback_hedge_enabled",
    "stop_out_reentry_enabled",
    "per_timeframe_grading_enabled",
    "lean_routing_to_options",
    "price_action_module_enabled",
    "price_action_block_adverse",
    "price_action_reversal_relief_enabled",
    "regime_veto_enabled",
    "regime_early_trend_er_promotion_enabled",
    "fut_early_loss_cut_enabled",
    "power_move_override_enabled",
    "power_move_spike_bypass_15m",
    "live_5min_trend_continuation_enabled",
    "volume_spike_entry_enabled",
    "one_min_scalping_enabled",
    "fifteen_min_one_min_support_exit_enabled",
    "volume_spike_depth_confirm_enabled",
    "stabilization_momentum_override_enabled",
    "live_15min_momentum_override_enabled",
    "reversal_entry_relief_enabled",
    "directional_census_reversal_relief_enabled",
    "late_session_choppy_entry_guard_enabled",
    "market_wide_reversal_enabled",
    "market_ltp_shock_enabled",
    "market_ltp_shock_opposite_trade_relax",
    "market_ltp_shock_force_exit_opposite",
    "session_mood_dynamic_enabled",
    "post_exit_reversion_enabled",
    "one_min_scalping_high_burst_only",
    "one_way_cascade_enabled",
    "missed_signal_reentry_enabled",
    "missed_signal_reentry_allow_trend_context",
    "missed_signal_reentry_prefer_options",
    "late_session_reclaim_override_enabled",
    "fifteen_min_choppy_scalp_guard_enabled",
    "peak_threat_exit_enabled",
    "peak_threat_exit_loss_enabled",
    "ut_confidence_calibration_enabled",
    "engine_lab_enabled",
    "engine_lab_real_mode_only",
    "dual_lane_enabled",
    "dual_clone_reduction_enabled",
    "cascade_lot_scale_enabled",
    "heavy_trend_lot_scale_enabled",
    "cascade_multi_tf_stack_enabled",
    "multi_signal_mixed_book_enabled",
    "execution_use_touch_quote",
    "execution_touch_quote_shadow_enabled",
    "execution_touch_quote_strict",
}
_TIME_COMMAND_FIELDS = {
    "power_move_late_hard_cutoff",
    "ut_no_entry_after",
    "ut_5min_no_entry_after",
    "ut_1min_no_entry_after",
    "ut_force_exit_time",
    "late_session_reclaim_start",
    "late_session_reclaim_end",
    "market_ltp_shock_session_start",
    "market_ltp_shock_session_end",
    "market_ltp_shock_session_override_start",
    "real_mode_premarket_arm_start",
    "real_mode_premarket_settle_end",
    "real_mode_premarket_refresh_end",
    "real_mode_live_start",
    "auto_real_mode_premarket_switch_time",
}
_COMMAND_FIELDS: Dict[str, set] = {
    "get_state": set(),
    "configure": {
        "capital_total", "capital_fut", "risk_fut_pct", "capital_opt", "risk_opt_pct",
        "lots", "lots_fut", "futures_sl_pct", "options_sl_pct", "fut_cost", "opt_cost",
        "backtest_days", "auto_mode", "inst_pref", "strike_selection", "grade_preference",
        "ut_preset", "active_indices", "mode", "reset", "confirm_real_mode",
        "real_mode_verification", "timeframe_entry_policy", "max_trades_per_index",
        "max_consecutive_losses", "index_cooldown_minutes", "futures_shadow_enabled",
        "signal_fire_mode",
    } | _ADVANCED_CONFIG_FIELDS,
    "set_mode": {"mode", "reset", "confirm_real_mode", "real_mode_verification"},
    "reset_pnl": set(),
    "set_power": {"state"},
    "set_regime_adaptation": {"state"},
    "set_sl_mode": {"state"},
    "set_scanner_engine": {"state"},
    "set_concurrency_guard": {"state"},
    "set_expiry_gamma": {"state"},
    "set_combo_partner_exit": {"state"},
    "execute_signal": {"signal_id", "lifecycle_id"},
    # ``price`` is an optional accounting mark.  A broker MARKET close has no price
    # parameter, and a stale quote must not prevent an operator from flattening.
    "close_trade": {"trade_id", "price"},
    "kill_all": {"confirm"},
    "update_fyers_token": {"auth_code"},
    "subscribe_chart": {"instrument", "tf"},
    "set_chart_stream": {"enabled"},
    "system_recalibrate": set(),
}
# Known numeric fields that must be finite (and non-negative) when present.
# Note: 'price' is intentionally NOT here — close_trade accepts a zero/omitted mark
# for a MARKET close and validates that contract below.
_NUMERIC_COMMAND_FIELDS = {
    "capital_total", "capital_fut", "risk_fut_pct", "capital_opt", "risk_opt_pct",
    "futures_sl_pct", "options_sl_pct", "fut_cost", "opt_cost", "backtest_days",
    "max_trades_per_index", "max_consecutive_losses", "index_cooldown_minutes",
    "regime_veto_adx_floor", "regime_early_trend_er_period",
    "regime_early_trend_er_threshold", "regime_early_trend_er_adx_ceiling",
    "regime_early_trend_er_range_day_min_confidence",
    "fut_loss_cut_r", "power_move_spike_range_atr_mult",
    "live_5min_trend_continuation_adx", "volume_spike_entry_min_volume_ratio",
    "volume_spike_entry_min_range_atr_ratio", "volume_spike_entry_min_close_location",
    "volume_spike_entry_futures_sl_pct", "volume_spike_entry_options_sl_pct",
    "volume_spike_entry_stabilization_seconds",
    "volume_spike_entry_stabilization_seconds_min",
    "volume_spike_entry_stabilization_seconds_max",
    "volume_spike_depth_ofr_bullish", "volume_spike_depth_ofr_bearish",
    "ut_confidence_calibration_min_samples",
    "live_signal_stabilization_seconds_5min", "live_signal_stabilization_seconds_15min",
    "live_15min_momentum_min_confidence", "live_15min_momentum_volume_ratio",
    "live_15min_momentum_range_atr_ratio", "live_15min_momentum_close_location",
    "live_15min_momentum_adx_threshold", "live_15min_momentum_intel_threshold",
    "reversal_entry_min_confidence", "reversal_entry_thrust_min_range_atr_ratio",
    "reversal_entry_thrust_min_close_location", "reversal_entry_thrust_min_volume_ratio",
    "reversal_entry_vwap_min_dev_pct", "directional_census_reversal_min_confidence",
    "late_session_choppy_min_runway_minutes",
    "market_wide_reversal_core_votes", "market_wide_reversal_total_votes",
    "market_wide_reversal_min_range_atr_ratio", "market_wide_reversal_min_range_atr_ratio_5min",
    "market_wide_reversal_min_range_atr_ratio_15min", "market_wide_reversal_midcap_min_range_atr_ratio_5min",
    "market_wide_reversal_midcap_min_range_atr_ratio_15min", "market_wide_reversal_min_close_location",
    "market_wide_reversal_max_skew_seconds_5min", "market_wide_reversal_max_skew_seconds_15min",
    "one_way_cascade_core_votes", "one_way_cascade_total_votes", "one_way_cascade_volume_votes",
    "one_way_cascade_min_volume_ratio", "one_way_cascade_min_range_atr_ratio_1min",
    "one_way_cascade_min_range_atr_ratio_5min", "one_way_cascade_min_range_atr_ratio_15min",
    "one_way_cascade_min_close_location", "one_way_cascade_runner_lock_pct",
    "one_way_cascade_options_sl_pct", "one_way_cascade_futures_sl_pct", "one_way_cascade_rr",
    "missed_signal_reentry_max_age_minutes", "missed_signal_reentry_min_age_seconds",
    "missed_signal_reentry_max_chase_atr", "missed_signal_reentry_min_confidence",
    "missed_signal_reentry_min_intel_score", "missed_signal_reentry_min_trend_adx",
    "missed_signal_reentry_min_remaining_minutes", "missed_signal_reentry_options_sl_pct",
    "missed_signal_reentry_futures_sl_pct", "missed_signal_reentry_rr",
    "late_session_reclaim_core_votes", "late_session_reclaim_total_votes",
    "late_session_reclaim_reclaim_votes", "late_session_reclaim_max_skew_seconds",
    "late_session_reclaim_min_close_location", "late_session_reclaim_min_volume_ratio",
    "late_session_reclaim_volume_votes", "late_session_reclaim_options_sl_pct",
    "late_session_reclaim_futures_sl_pct", "late_session_reclaim_rr",
    "fifteen_min_choppy_scalp_min_intel_score", "fifteen_min_choppy_scalp_min_volume_ratio",
    "fifteen_min_choppy_scalp_micro_max_skew_seconds", "fifteen_min_choppy_scalp_min_close_location",
    "fifteen_min_choppy_scalp_profit_lock_r_multiple", "fifteen_min_choppy_scalp_min_peak_rs",
    "peak_threat_exit_min_peak_rs", "peak_threat_exit_min_peak_r",
    "peak_threat_exit_min_retention", "peak_threat_exit_min_score",
    "peak_threat_exit_severe_score", "peak_threat_exit_adverse_intel",
    "peak_threat_exit_weak_volume_ratio", "peak_threat_exit_loss_min_score",
    "peak_threat_exit_loss_max_loss_r", "option_major_win_trend_lock_ratio_cap",
    "midcap_15min_risky_stabilization_seconds", "midcap_15min_risky_stabilization_min_scans",
    "midcap_15min_risky_stabilization_min_volume_ratio", "midcap_15min_risky_stabilization_min_intel_score",
    "dual_lane_same_dir_min_bars_gap", "dual_lane_same_dir_min_lane_a_r",
    "dual_lane_size_fraction", "dual_lane_max_index_exposure_rs",
    "dual_lane_min_lane_b_confidence", "dual_lane_jaccard_alert_threshold",
    "dual_clone_reduction_window_seconds",
    "cascade_lot_multiplier_opt", "cascade_lot_multiplier_opt_min",
    "cascade_lot_multiplier_fut",
    "heavy_trend_lot_multiplier_opt", "heavy_trend_min_aligned_indices",
    "heavy_trend_vote_max_age_seconds",
    "cascade_primary_max_tf", "cascade_dual_addon_max",
    "cascade_midcap_solo_max_tf", "cascade_midcap_dual_main_max",
    "cascade_midcap_dual_addon_max",
    "cascade_core_bucket_max_positions", "cascade_midcap_bucket_max_positions",
    "cascade_max_concurrent_positions",
    "multi_signal_max_fut", "multi_signal_fut_lots",
    "multi_signal_max_opt_instruments",
    "quality_hybrid_min_vote_score", "quality_hybrid_broad_agreement",
    "quality_hybrid_chop_min_agreement", "quality_hybrid_trend_adx",
    "quality_hybrid_trend_er", "quality_hybrid_thrust_volume_ratio",
    "quality_hybrid_thrust_range_atr", "quality_hybrid_thrust_close_loc",
}
_SIGNED_NUMERIC_COMMAND_FIELDS = {"price_action_adverse_threshold"}
_BOOLEAN_COMMAND_FIELDS = {
    "auto_mode",
    "confirm_real_mode",
    "enabled",
    "reset",
    "futures_shadow_enabled",
} | _ADVANCED_BOOLEAN_CONFIG_FIELDS


# Hard bounds for risk / capacity configure fields (inclusive). Prevents
# silent persistence of 0%/1000% stops or 1M trades-per-index.
_NUMERIC_FIELD_BOUNDS: Dict[str, tuple] = {
    "futures_sl_pct": (0.01, 5.0),
    "options_sl_pct": (0.5, 50.0),
    "volume_spike_entry_futures_sl_pct": (0.01, 5.0),
    "volume_spike_entry_options_sl_pct": (0.5, 50.0),
    "volume_spike_entry_min_volume_ratio": (1.0, 20.0),
    "volume_spike_entry_min_range_atr_ratio": (0.5, 10.0),
    "volume_spike_entry_min_close_location": (0.5, 1.0),
    "volume_spike_entry_stabilization_seconds": (3.0, 15.0),
    # Floor was 30s while the code default is 15s (config/settings.py), so the shorter
    # clean-trend wait the operator asks for could not be configured at all: anything
    # under 30 was silently raised. 10s keeps a real repaint buffer - the 1m spike lane
    # runs at 3-15s through its own knob above - without dictating the 5m/15m wait.
    "live_signal_stabilization_seconds_5min": (10.0, 120.0),
    "live_signal_stabilization_seconds_15min": (10.0, 120.0),
    "max_trades_per_index": (1, 50),
    "max_consecutive_losses": (1, 20),
    "index_cooldown_minutes": (0, 240),
    "lots": (1, 50),
    "lots_fut": (1, 50),
    "capital_total": (1_000.0, 1_000_000_000.0),
    "capital_fut": (0.0, 1_000_000_000.0),
    "capital_opt": (0.0, 1_000_000_000.0),
    "backtest_days": (1, 60),
    "fut_cost": (0.0, 50_000.0),
    "opt_cost": (0.0, 50_000.0),
}
_INTEGER_NUMERIC_COMMAND_FIELDS = {
    "backtest_days",
    "max_trades_per_index",
    "max_consecutive_losses",
}
# S1 engine / lane settings: one spec (engine/s1_lane.py) drives validation, forwarding
# and the broadcast, so a new S1 knob cannot be accepted here and dropped elsewhere.
from engine.s1_lane import S1_SETTING_SPEC as _S1_SETTING_SPEC
from engine.uts_lane import UTS_SETTING_SPEC as _UTS_SETTING_SPEC
from engine.s12_lane import S12_SETTING_SPEC as _S12_SETTING_SPEC

# UT SMART EXIT settings take the same path as S1's.
_LANE_SETTING_SPEC = {**_S1_SETTING_SPEC, **_UTS_SETTING_SPEC, **_S12_SETTING_SPEC}
_ADVANCED_CONFIG_FIELDS |= set(_LANE_SETTING_SPEC)
_COMMAND_FIELDS["configure"] |= set(_LANE_SETTING_SPEC)
for _s1_name, _s1_spec in _LANE_SETTING_SPEC.items():
    if _s1_spec[0] in {"float", "int"}:
        _NUMERIC_COMMAND_FIELDS.add(_s1_name)
        _NUMERIC_FIELD_BOUNDS[_s1_name] = (_s1_spec[1], _s1_spec[2])
        if _s1_spec[0] == "int":
            _INTEGER_NUMERIC_COMMAND_FIELDS.add(_s1_name)
    elif _s1_spec[0] == "bool":
        _BOOLEAN_COMMAND_FIELDS.add(_s1_name)
    elif _s1_spec[0] == "time":
        _TIME_COMMAND_FIELDS.add(_s1_name)
_S1_ENUM_FIELDS = {k: v[1] for k, v in _LANE_SETTING_SPEC.items() if v[0] == "enum"}

# Unattended morning start: editable any time from the dashboard (the worker re-reads
# them every minute). premarket = switch to REAL/manual at the switch time; startup =
# turn Auto on once the day's full readiness has passed.
for _name in ("auto_real_mode_premarket_enabled", "auto_real_mode_startup_enabled"):
    _ADVANCED_CONFIG_FIELDS.add(_name)
    _COMMAND_FIELDS["configure"].add(_name)
    _BOOLEAN_COMMAND_FIELDS.add(_name)

_ALLOWED_ACTIVE_INDICES = frozenset({"NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"})
# Stay under common 1 MiB WebSocket client frame limits (leave headroom).
_WS_MAX_FRAME_BYTES = 900_000
_WS_TRADE_CANDIDATE_LIMIT = 20
_WS_TRADE_CANDIDATE_FALLBACK_LIMIT = 8
_WS_ACTIVITY_LOG_LIMIT = 50


def _valid_command_payload(msg: Dict[str, Any]) -> Optional[str]:
    if not isinstance(msg, dict):
        return "invalid_payload"
    cmd = msg.get("cmd", "")
    if cmd not in _COMMAND_FIELDS:
        return "unknown_command"

    allowed = _BASE_COMMAND_FIELDS | _COMMAND_FIELDS[cmd]
    unknown = [k for k in msg.keys() if k not in allowed]
    if unknown:
        return f"unknown_field:{sorted(unknown)[0]}"

    for field in _NUMERIC_COMMAND_FIELDS:
        if field in msg and msg[field] is not None:
            try:
                value = float(msg[field])
            except (TypeError, ValueError):
                return f"invalid_number:{field}"
            if not math.isfinite(value) or value < 0:
                return f"invalid_number:{field}"
            if field in _INTEGER_NUMERIC_COMMAND_FIELDS and not value.is_integer():
                return f"invalid_integer:{field}"
            bounds = _NUMERIC_FIELD_BOUNDS.get(field)
            if bounds is not None:
                lo, hi = bounds
                if value < lo or value > hi:
                    return f"out_of_range:{field}"

    for field in _SIGNED_NUMERIC_COMMAND_FIELDS:
        if field in msg and msg[field] is not None:
            try:
                value = float(msg[field])
            except (TypeError, ValueError):
                return f"invalid_number:{field}"
            if not math.isfinite(value):
                return f"invalid_number:{field}"

    for field in _BOOLEAN_COMMAND_FIELDS:
        if field in msg and msg[field] is not None and not isinstance(msg[field], bool):
            return f"invalid_boolean:{field}"

    for field in _TIME_COMMAND_FIELDS:
        if field in msg and msg[field] is not None:
            value = str(msg[field] or "").strip()
            parsed = None
            for fmt in ("%H:%M", "%H:%M:%S"):
                try:
                    parsed = datetime.strptime(value, fmt).time()
                    break
                except (TypeError, ValueError):
                    continue
            if parsed is None:
                return f"invalid_time:{field}"
            if parsed.hour > 23 or parsed.minute > 59:
                return f"invalid_time:{field}"

    if "active_indices" in msg and msg.get("active_indices") is not None:
        raw = msg.get("active_indices")
        if not isinstance(raw, list) or not raw:
            return "invalid_active_indices"
        cleaned: list = []
        for item in raw:
            if not isinstance(item, str):
                return "invalid_active_indices"
            name = item.strip().upper()
            if name not in _ALLOWED_ACTIVE_INDICES:
                return f"invalid_active_index:{name or '?'}"
            if name not in cleaned:
                cleaned.append(name)
        if not cleaned:
            return "invalid_active_indices"
        msg["active_indices"] = cleaned

    for field, choices in _S1_ENUM_FIELDS.items():
        if field in msg and msg[field] is not None and str(msg[field]) not in choices:
            return f"invalid_choice:{field}"

    if "signal_fire_mode" in msg:
        mode = str(msg.get("signal_fire_mode") or "").upper()
        if mode not in {"INTRACANDLE", "CANDLE_CLOSE"}:
            return "invalid_signal_fire_mode"

    if "request_id" in msg and len(str(msg.get("request_id") or "")) > 240:
        return "invalid_request_id"

    if cmd in {"set_power", "set_regime_adaptation", "set_concurrency_guard", "set_expiry_gamma", "set_combo_partner_exit"} and msg.get("state") not in {"ON", "OFF"}:
        return "invalid_state"
    if cmd == "close_trade":
        if not str(msg.get("trade_id", "")).strip() or len(str(msg.get("trade_id", ""))) > 120:
            return "invalid_trade_id"
        try:
            price = float(msg.get("price", 0))
        except (TypeError, ValueError):
            return "invalid_price"
        # MARKET exits do not require a quote.  A non-negative local mark is accepted
        # for UI/accounting continuity; the broker fill remains authoritative.
        if not math.isfinite(price) or price < 0:
            return "invalid_price"
    if cmd == "execute_signal":
        signal_id = str(msg.get("signal_id", "") or "").strip()
        lifecycle_id = str(msg.get("lifecycle_id", "") or "").strip()
        if not signal_id and not lifecycle_id:
            return "invalid_signal_id"
        if len(signal_id) > 240 or len(lifecycle_id) > 240:
            return "invalid_signal_id"
    if cmd == "update_fyers_token" and len(str(msg.get("auth_code", ""))) > 4096:
        return "invalid_auth_code"
    return None


def _command_response(msg: Dict[str, Any], msg_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"type": msg_type, "data": data}
    request_id = msg.get("request_id")
    if request_id is not None:
        payload["request_id"] = request_id
    return payload


async def _send_command_response(ws: WebSocket, msg: Dict[str, Any], msg_type: str, data: Dict[str, Any]) -> None:
    if msg_type == "full_update":
        capped = _cap_ws_payload(make_serializable(data or {}))
        payload = _command_response(msg, msg_type, capped)
        message = json.dumps(payload, default=str)
        if len(message.encode("utf-8")) > _WS_MAX_FRAME_BYTES:
            # Request metadata must never defeat the state cap.
            payload = {
                "type": "full_update",
                "data": {
                    "ws_truncated": True,
                    "ws_payload_omitted": True,
                    "ws_truncation_reason": "frame_limit",
                },
            }
            request_id = msg.get("request_id")
            if request_id is not None:
                payload["request_id"] = str(request_id)[:240]
            message = json.dumps(payload, default=str)
        await ws.send_text(message)
        return
    await ws.send_json(_command_response(msg, msg_type, data))


def _configuration_result_payload(configured: Any, scanner: Any, requested_mode: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": "ok" if configured is not False else "blocked"}
    actual_mode = str(getattr(scanner, "mode", "") or "").upper()
    if requested_mode is not None:
        payload["mode_command"] = True
        payload["requested_mode"] = str(requested_mode or "").upper()
        payload["mode"] = actual_mode
    if configured is False:
        reason = str(getattr(scanner, "_last_configure_block_reason", "") or "").strip()
        payload["reason"] = reason or "configuration rejected"
    return payload


async def shutdown_scanner():
    logger.info("👋 Graceful Shutdown Triggered...")
    scanner = scanner_ref
    if scanner:
        # Out-of-band fail-safe: even if the scanner loop is temporarily busy and
        # cannot service the stop coroutine, its next run-loop check must exit.
        if hasattr(scanner, "is_running"):
            scanner.is_running = False
        scanner_trades = getattr(scanner, "trades", None)
        if scanner_trades is not None and hasattr(scanner_trades, "_sync_active"):
            scanner_trades._sync_active = False
        stop_completed = False

        async def _request_stop() -> None:
            nonlocal stop_completed
            trades = getattr(scanner, "trades", None)
            if trades is not None and hasattr(trades, "_sync_active"):
                trades._sync_active = False
            provider = getattr(scanner, "data", None)
            shutdown_streams = getattr(provider, "shutdown_streams", None)
            if callable(shutdown_streams):
                try:
                    result = shutdown_streams()
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:
                    # A provider close failure must not prevent the scanner stop.
                    logger.error(f"Error stopping provider streams: {exc}")
            result = scanner.stop()
            if inspect.isawaitable(result):
                await result
            flush_session_ledger = getattr(
                scanner,
                "flush_session_trade_candidates",
                None,
            )
            if callable(flush_session_ledger):
                result = flush_session_ledger()
                if inspect.isawaitable(result):
                    await result
            stop_completed = True

        try:
            scanner_loop = getattr(scanner, "_event_loop", None)
            if scanner_loop is not None and not scanner_loop.is_running():
                # The owner loop has already exited, so cleanup is safe locally.
                await _request_stop()
            else:
                await _await_on_scanner_loop(_request_stop(), timeout_seconds=20.0)
            logger.info("✅ Scanner stop completed.")
        except Exception as e:
            logger.error(f"Error stopping scanner: {e}")
            scanner_loop = getattr(scanner, "_event_loop", None)
            if scanner_loop is not None and not scanner_loop.is_running():
                try:
                    await _request_stop()
                except Exception as fallback_exc:
                    logger.error(f"Local scanner cleanup fallback failed: {fallback_exc}")

        raw_join_timeout = getattr(
            get_settings(),
            "scanner_shutdown_join_timeout_seconds",
            120.0,
        )
        try:
            scanner_join_timeout = max(
                5.0,
                min(300.0, float(raw_join_timeout or 120.0)),
            )
        except (TypeError, ValueError):
            scanner_join_timeout = 120.0

        # ScannerEventLoop must fully unwind before the execution queue is closed;
        # otherwise the tail of a scan can submit broker work after queue drain.
        runner_thread = getattr(scanner, "_runner_thread", None)
        if runner_thread is not None and runner_thread.is_alive():
            try:
                await asyncio.to_thread(runner_thread.join, scanner_join_timeout)
            except Exception as exc:
                logger.error(f"Scanner runner join failed: {exc}")
            if runner_thread.is_alive():
                logger.critical(
                    "ScannerEventLoop did not stop before the graceful-shutdown timeout."
                )
        if not stop_completed and (
            runner_thread is None or not runner_thread.is_alive()
        ):
            try:
                await _request_stop()
            except Exception as exc:
                logger.error(f"Post-join scanner cleanup failed: {exc}")

        sync_thread = getattr(getattr(scanner, "trades", None), "_sync_thread", None)
        if sync_thread is not None and sync_thread.is_alive():
            try:
                await asyncio.to_thread(sync_thread.join, min(10.0, scanner_join_timeout))
            except Exception as exc:
                logger.debug(f"Position sync join skipped: {exc}")
            if sync_thread.is_alive():
                logger.warning("Position sync thread did not stop before queue drain.")

    # Entry/exit workers may still be persisting broker acknowledgements. Drain
    # them before the final trade snapshot, without blocking the dashboard loop.
    try:
        from engine.async_queue import execution_queue

        raw_queue_timeout = getattr(
            get_settings(),
            "shutdown_execution_queue_timeout_seconds",
            90.0,
        )
        try:
            parsed_queue_timeout = float(raw_queue_timeout or 90.0)
        except (TypeError, ValueError):
            parsed_queue_timeout = 90.0
        queue_timeout = max(
            5.0,
            min(300.0, parsed_queue_timeout),
        )
        drained = await asyncio.to_thread(
            execution_queue.shutdown,
            drain=True,
            timeout_seconds=queue_timeout,
        )
        if drained:
            logger.info("✅ Broker execution queue drained.")
        else:
            logger.warning("Broker execution queue did not fully drain before timeout.")
    except Exception as exc:
        logger.error(f"Error draining broker execution queue: {exc}")

    if scanner:
        try:
            await asyncio.to_thread(scanner.trades.save_state)
            logger.info("✅ Trade state saved.")
        except Exception as e:
            logger.error(f"Error saving trade state during shutdown: {e}")


async def _relay_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while not reader.at_eof():
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def _handle_ipv6_localhost_proxy(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
) -> None:
    settings = get_settings()
    port = int(getattr(settings, "dashboard_port", 8001) or 8001)
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection("127.0.0.1", port)
    except Exception:
        client_writer.close()
        await client_writer.wait_closed()
        return
    await asyncio.gather(
        _relay_stream(client_reader, upstream_writer),
        _relay_stream(upstream_reader, client_writer),
        return_exceptions=True,
    )


async def start_ipv6_localhost_proxy() -> None:
    """Bridge Windows localhost (::1) to the IPv4 Uvicorn listener."""
    global _ipv6_localhost_proxy
    if _ipv6_localhost_proxy is not None:
        return
    settings = get_settings()
    host = str(getattr(settings, "dashboard_host", "127.0.0.1") or "127.0.0.1")
    if host not in {"127.0.0.1", "localhost"}:
        return
    port = int(getattr(settings, "dashboard_port", 8001) or 8001)
    try:
        _ipv6_localhost_proxy = await asyncio.start_server(
            _handle_ipv6_localhost_proxy,
            host="::1",
            port=port,
        )
        logger.info(f"IPv6 localhost bridge active on [::1]:{port} -> 127.0.0.1:{port}")
    except OSError as exc:
        logger.debug(f"IPv6 localhost bridge not started: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _periodic_trades_task, _ipv6_localhost_proxy, _dashboard_shutting_down, _startup_task
    _dashboard_shutting_down = False
    _startup_task = None
    try:
        from engine.async_queue import execution_queue

        start_queue = getattr(execution_queue, "start", None)
        if callable(start_queue):
            start_queue()
    except Exception as exc:
        logger.error(f"Unable to start broker execution queue: {exc}")
    _ensure_broadcast_worker()
    _periodic_trades_task = asyncio.create_task(periodic_trades_updater())
    await start_ipv6_localhost_proxy()
    if startup_hook:
        result = startup_hook()
        if isinstance(result, asyncio.Future):
            _startup_task = result
        elif inspect.isawaitable(result):
            await result
    try:
        yield
    finally:
        # Close the command ingress first: no new broker/scanner work may race
        # with the shutdown hook or execution-queue drain.
        _dashboard_shutting_down = True
        if _periodic_trades_task:
            _periodic_trades_task.cancel()
            try:
                await _periodic_trades_task
            except asyncio.CancelledError:
                pass
            _periodic_trades_task = None
        await _stop_broadcast_worker()
        await _close_active_websockets(code=1012)
        # The production startup hook intentionally bootstraps the scanner in a
        # background Task. Let it reach a known terminal state before cleanup so
        # resources cannot be created after the shutdown hook has already run.
        startup_task = _startup_task
        _startup_task = None
        if startup_task is not None:
            try:
                await asyncio.shield(startup_task)
            except asyncio.CancelledError:
                # Preserve cleanup even if the lifespan receives cancellation.
                # Shield keeps an in-flight to_thread(create_system) bootstrap
                # alive; wait for it once more so the hook can close its result.
                if not startup_task.cancelled():
                    try:
                        await asyncio.shield(startup_task)
                    except asyncio.CancelledError as exc:
                        logger.error(f"Dashboard startup task did not settle: {exc}")
                    except Exception as exc:
                        logger.error(f"Dashboard startup task did not settle: {exc}")
            except Exception as exc:
                logger.error(f"Dashboard startup task failed during shutdown: {exc}")
        shutdown_hook_completed = False
        if shutdown_hook:
            try:
                result = shutdown_hook()
                if inspect.isawaitable(result):
                    await result
                shutdown_hook_completed = True
            except Exception as exc:
                logger.error(f"Dashboard shutdown hook failed: {exc}")
        if not shutdown_hook_completed:
            await shutdown_scanner()
        if _ipv6_localhost_proxy is not None:
            _ipv6_localhost_proxy.close()
            await _ipv6_localhost_proxy.wait_closed()
            _ipv6_localhost_proxy = None


app = FastAPI(title="UT Bot Index Trading System", version="1.0.0", lifespan=lifespan)


_LOCAL_ONLY_API_PATHS = {
    "/api/status", "/api/state", "/api/diagnostics", "/api/trades",
    "/api/signal-report", "/api/chart", "/api/calibration-status", "/api/quotes",
}

# Audit #10: trade/ops read payloads require the dashboard token even on loopback,
# so a local untrusted process cannot scrape positions/PnL without the token.
# /api/status stays token-optional on loopback for health probes (start scripts).
_TOKEN_REQUIRED_API_PATHS = {
    "/api/state", "/api/diagnostics", "/api/trades",
    "/api/signal-report", "/api/chart", "/api/calibration-status", "/api/quotes",
}


def _read_api_authorized(request: Request) -> bool:
    """Token gate for sensitive read APIs (loopback alone is not enough)."""
    token = request.headers.get("x-dashboard-token")
    return _token_matches(token)


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    """Baseline browser-safety headers + loopback + token gates on read APIs.

    The read APIs expose operational/trade data. Restrict them to local (loopback)
    requests so that if the server is ever bound to 0.0.0.0 a remote client cannot
    pull the data. Sensitive payloads additionally require the dashboard token.
    """
    from fastapi.responses import JSONResponse

    path = request.url.path
    replay_audit = path.startswith("/api/replay-audit/")
    if (path in _LOCAL_ONLY_API_PATHS or replay_audit) and not _is_loopback_request(request):
        return JSONResponse({"error": "local access only"}, status_code=403)
    if (path in _TOKEN_REQUIRED_API_PATHS or replay_audit) and not _read_api_authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    response = await call_next(request)
    # Same-origin scripts only (blocks injected/3rd-party JS) while still allowing the
    # Google Fonts stylesheet + font files the dashboard actually uses.
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "script-src 'self'; connect-src 'self' ws: wss:; "
        "font-src 'self' data: https://fonts.gstatic.com; "
        "object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
    )
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-Frame-Options", "DENY")
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(_allowed_origins()),
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Dashboard-Token"],
)

# Static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = static_dir / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path), headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"})
    return HTMLResponse("<h1>UT Bot Index Trading System</h1><p>Dashboard loading...</p>")


def _is_loopback_request(request: Request) -> bool:
    """True only when the TCP peer is the local machine (loopback)."""
    client = getattr(request, "client", None)
    host = (getattr(client, "host", "") or "").strip().lower()
    return host in {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}


@app.get("/api/client_config")
async def api_client_config(request: Request, response: Response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    # SECURITY: the dashboard token authorizes trading commands (set_mode, kill_all,
    # close_trade, place orders). Only hand it to LOCAL (loopback) requests, so that
    # if the server is ever bound to 0.0.0.0 a remote client cannot fetch it.
    local = _is_loopback_request(request)
    token = _dashboard_token() if local else ""
    if token:
        # Browser WebSocket constructors cannot set Authorization headers. Seed
        # a path-scoped, HttpOnly cookie during the same-origin bootstrap so the
        # WS handshake remains credentialed without putting a secret in its URL.
        response.set_cookie(
            key="ut1_dashboard_token",
            value=token,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
            path="/ws",
        )
    else:
        response.delete_cookie(key="ut1_dashboard_token", path="/ws")
    return {
        "dashboard_token": token,
        "allowed_origins": sorted(_allowed_origins()),
        "local": local,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    if _dashboard_shutting_down:
        await ws.close(code=1012)
        return
    origin = ws.headers.get("origin")
    authenticated, selected_protocol = _ws_authentication(ws)
    if not _is_allowed_origin(origin) or not authenticated:
        await ws.close(code=1008)
        return
    max_connections, _, _, max_violations = _dashboard_ws_limits()
    ws_id = id(ws)
    # Reserve the slot before the accept await, preventing simultaneous handshakes
    # from all observing the same pre-limit connection count.
    if len(active_connections) + len(_pending_ws_connections) >= max_connections:
        await ws.close(code=1013)
        return
    _pending_ws_connections.add(ws_id)
    try:
        await ws.accept(subprotocol=selected_protocol)
    except Exception:
        _pending_ws_connections.discard(ws_id)
        raise
    _pending_ws_connections.discard(ws_id)
    if _dashboard_shutting_down:
        await ws.close(code=1012)
        return
    active_connections.add(ws)
    logger.info(f"📡 Dashboard connected ({len(active_connections)} clients)")
    try:
        while True:
            if _dashboard_shutting_down:
                await ws.close(code=1012)
                return
            # Receive commands from dashboard
            data = await ws.receive_text()
            allowed, violations = _record_ws_command(ws)
            if not allowed:
                await ws.send_json(
                    {
                        "type": "command_rejected",
                        "data": {"reason": "rate_limited"},
                    }
                )
                if violations >= max_violations:
                    await ws.close(code=1008)
                    return
                continue
            try:
                msg = json.loads(data)
                await handle_command(ws, msg)
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        _forget_ws_connection(ws)
        logger.info(f"📡 Dashboard disconnected ({len(active_connections)} clients)")
    except Exception:
        _forget_ws_connection(ws)
    finally:
        _forget_ws_connection(ws)


async def handle_command(ws: WebSocket, msg: Dict):
    """Handle commands from the dashboard"""
    if _dashboard_shutting_down:
        await ws.send_json(
            {
                "type": "command_rejected",
                "data": {"reason": "system_shutting_down"},
            }
        )
        return
    if not isinstance(msg, dict):
        await ws.send_json({"type": "command_rejected", "data": {"reason": "invalid_payload"}})
        return
    cmd = msg.get("cmd", "")
    scanner_required = {
        "get_state",
        "configure",
        "set_mode",
        "reset_pnl",
        "set_power",
        "set_regime_adaptation",
        "set_sl_mode",
        "set_scanner_engine",
        "set_concurrency_guard",
        "set_expiry_gamma",
        "set_combo_partner_exit",
        "execute_signal",
        "close_trade",
        "kill_all",
        "update_fyers_token",
        "subscribe_chart",
        "set_chart_stream",
        "system_recalibrate",
    }
    if cmd in scanner_required and not scanner_ref:
        await _send_command_response(ws, msg, "system_unavailable", {"status": "starting", "cmd": cmd})
        return

    validation_error = _valid_command_payload(msg)
    if validation_error:
        await _send_command_response(ws, msg, "command_rejected", {"cmd": cmd, "reason": validation_error})
        return

    if cmd == "get_state":
        if scanner_ref:
            data = scanner_ref.dashboard_cache.state() if hasattr(scanner_ref, "dashboard_cache") else scanner_ref.latest_results
            if not data and hasattr(scanner_ref, "get_latest_results"):
                data = scanner_ref.get_latest_results(include_full_charts=False) or {}
            payload = dict(data or {})
            payload["dashboard_heartbeat"] = _dashboard_heartbeat()
            await _send_command_response(ws, msg, "full_update", make_serializable(_lighten_state_payload(payload)))

    elif cmd == "configure":
        if scanner_ref:
            configured = await _configure_on_scanner_loop(
                capital_total=msg.get("capital_total"),
                entry_preopen_signal_guard_enabled=msg.get("entry_preopen_signal_guard_enabled"),
                entry_mark_touch_validation_enabled=msg.get("entry_mark_touch_validation_enabled"),
                entry_mark_max_ask_multiple=msg.get("entry_mark_max_ask_multiple"),
                entry_mark_min_bid_multiple=msg.get("entry_mark_min_bid_multiple"),
                candidate_entry_mark_validation_enabled=msg.get("candidate_entry_mark_validation_enabled"),
                candidate_entry_mark_max_divergence_pct=msg.get("candidate_entry_mark_max_divergence_pct"),
                candidate_entry_mark_min_gap_rs=msg.get("candidate_entry_mark_min_gap_rs"),
                block_entries_during_eod_protect=msg.get("block_entries_during_eod_protect"),
                profit_lock_min_peak_rs=msg.get("profit_lock_min_peak_rs"),
                profit_lock_nonscalp_min_peak_rs=msg.get("profit_lock_nonscalp_min_peak_rs"),
                profit_lock_min_r_multiple=msg.get("profit_lock_min_r_multiple"),
                peak_giveback_cap_enabled=msg.get("peak_giveback_cap_enabled"),
                peak_giveback_cap_arm_rs=msg.get("peak_giveback_cap_arm_rs"),
                peak_giveback_cap_rs=msg.get("peak_giveback_cap_rs"),
                peak_giveback_cap_pct=msg.get("peak_giveback_cap_pct"),
                peak_giveback_cap_max_peak_rs=msg.get("peak_giveback_cap_max_peak_rs"),
                breakeven_stop_enabled=msg.get("breakeven_stop_enabled"),
                breakeven_stop_arm_rs=msg.get("breakeven_stop_arm_rs"),
                breakeven_stop_floor_rs=msg.get("breakeven_stop_floor_rs"),
                peak_trail_enabled=msg.get("peak_trail_enabled"),
                peak_trail_above_rs=msg.get("peak_trail_above_rs"),
                peak_trail_giveback_pct=msg.get("peak_trail_giveback_pct"),
                backtest_live_exits_enabled=msg.get("backtest_live_exits_enabled"),
                profit_protect_cost_floor_enabled=msg.get("profit_protect_cost_floor_enabled"),
                profit_protect_min_cost_multiple=msg.get("profit_protect_min_cost_multiple"),
                first_rung_regime_retention_enabled=msg.get("first_rung_regime_retention_enabled"),
                first_rung_fast_regime_retention=msg.get("first_rung_fast_regime_retention"),
                profit_threshold_scaling_enabled=msg.get("profit_threshold_scaling_enabled"),
                profit_threshold_regime_scaling_enabled=msg.get("profit_threshold_regime_scaling_enabled"),
                profit_threshold_reference_risk_rs=msg.get("profit_threshold_reference_risk_rs"),
                profit_threshold_scale_min=msg.get("profit_threshold_scale_min"),
                profit_threshold_scale_max=msg.get("profit_threshold_scale_max"),
                pnl_path_logging_enabled=msg.get("pnl_path_logging_enabled"),
                pnl_path_min_interval_seconds=msg.get("pnl_path_min_interval_seconds"),
                pnl_path_min_move_rs=msg.get("pnl_path_min_move_rs"),
                pnl_path_max_samples=msg.get("pnl_path_max_samples"),
                slippage_calibration_max_age_days=msg.get("slippage_calibration_max_age_days"),
                stabilization_rtracker_logging_enabled=msg.get("stabilization_rtracker_logging_enabled"),
                entry_intel_snapshot_enabled=msg.get("entry_intel_snapshot_enabled"),
                stabilization_cascade_latch_enabled=msg.get("stabilization_cascade_latch_enabled"),
                stabilization_thrust_scaling_enabled=msg.get("stabilization_thrust_scaling_enabled"),
                stabilization_thrust_min_seconds=msg.get("stabilization_thrust_min_seconds"),
                stabilization_thrust_range_atr=msg.get("stabilization_thrust_range_atr"),
                stabilization_thrust_volume_ratio=msg.get("stabilization_thrust_volume_ratio"),
                range_scaled_budget_enabled=msg.get("range_scaled_budget_enabled"),
                range_scaled_budget_reference_pct=msg.get("range_scaled_budget_reference_pct"),
                range_scaled_budget_min_trades=msg.get("range_scaled_budget_min_trades"),
                missed_signal_reentry_scored_mode_enabled=msg.get("missed_signal_reentry_scored_mode_enabled"),
                missed_signal_reentry_min_conditions=msg.get("missed_signal_reentry_min_conditions"),
                capital_fut=msg.get("capital_fut"),
                risk_fut_pct=msg.get("risk_fut_pct"),
                capital_opt=msg.get("capital_opt"),
                risk_opt_pct=msg.get("risk_opt_pct"),
                lots=msg.get("lots"),
                lots_fut=msg.get("lots_fut"),
                futures_sl_pct=msg.get("futures_sl_pct"),
                options_sl_pct=msg.get("options_sl_pct"),
                fut_cost=msg.get("fut_cost"),
                opt_cost=msg.get("opt_cost"),
                backtest_days=msg.get("backtest_days"),
                auto_mode=msg.get("auto_mode"),
                inst_pref=msg.get("inst_pref"),
                strike_selection=msg.get("strike_selection"),
                grade_preference=msg.get("grade_preference"),
                ut_preset=msg.get("ut_preset"),
                active_indices=msg.get("active_indices"),
                mode=msg.get("mode"),
                reset=msg.get("reset", False),
                confirm_real_mode=msg.get("confirm_real_mode", False),
                real_mode_verification=msg.get("real_mode_verification", ""),
                timeframe_entry_policy=msg.get("timeframe_entry_policy"),
                max_trades_per_index=msg.get("max_trades_per_index"),
                max_consecutive_losses=msg.get("max_consecutive_losses"),
                index_cooldown_minutes=msg.get("index_cooldown_minutes"),
                futures_shadow_enabled=msg.get("futures_shadow_enabled"),
                signal_fire_mode=msg.get("signal_fire_mode"),
                lean_routing_to_options=msg.get("lean_routing_to_options"),
                price_action_module_enabled=msg.get("price_action_module_enabled"),
                price_action_block_adverse=msg.get("price_action_block_adverse"),
                price_action_adverse_threshold=msg.get("price_action_adverse_threshold"),
                price_action_reversal_relief_enabled=msg.get("price_action_reversal_relief_enabled"),
                regime_veto_enabled=msg.get("regime_veto_enabled"),
                regime_veto_adx_floor=msg.get("regime_veto_adx_floor"),
                regime_early_trend_er_promotion_enabled=msg.get("regime_early_trend_er_promotion_enabled"),
                regime_early_trend_er_period=msg.get("regime_early_trend_er_period"),
                regime_early_trend_er_threshold=msg.get("regime_early_trend_er_threshold"),
                regime_early_trend_er_adx_ceiling=msg.get("regime_early_trend_er_adx_ceiling"),
                regime_early_trend_er_range_day_min_confidence=msg.get("regime_early_trend_er_range_day_min_confidence"),
                fut_early_loss_cut_enabled=msg.get("fut_early_loss_cut_enabled"),
                fut_loss_cut_r=msg.get("fut_loss_cut_r"),
                power_move_override_enabled=msg.get("power_move_override_enabled"),
                power_move_late_hard_cutoff=msg.get("power_move_late_hard_cutoff"),
                power_move_spike_bypass_15m=msg.get("power_move_spike_bypass_15m"),
                power_move_spike_range_atr_mult=msg.get("power_move_spike_range_atr_mult"),
                live_5min_trend_continuation_enabled=msg.get("live_5min_trend_continuation_enabled"),
                live_5min_trend_continuation_adx=msg.get("live_5min_trend_continuation_adx"),
                quality_hybrid_min_vote_score=msg.get("quality_hybrid_min_vote_score"),
                quality_hybrid_broad_agreement=msg.get("quality_hybrid_broad_agreement"),
                quality_hybrid_chop_min_agreement=msg.get("quality_hybrid_chop_min_agreement"),
                quality_hybrid_trend_adx=msg.get("quality_hybrid_trend_adx"),
                quality_hybrid_trend_er=msg.get("quality_hybrid_trend_er"),
                quality_hybrid_thrust_volume_ratio=msg.get("quality_hybrid_thrust_volume_ratio"),
                quality_hybrid_thrust_range_atr=msg.get("quality_hybrid_thrust_range_atr"),
                quality_hybrid_thrust_close_loc=msg.get("quality_hybrid_thrust_close_loc"),
                volume_spike_entry_enabled=msg.get("volume_spike_entry_enabled"),
                one_min_scalping_enabled=msg.get("one_min_scalping_enabled"),
                fifteen_min_one_min_support_exit_enabled=msg.get("fifteen_min_one_min_support_exit_enabled"),
                volume_spike_entry_min_volume_ratio=msg.get("volume_spike_entry_min_volume_ratio"),
                volume_spike_entry_min_range_atr_ratio=msg.get("volume_spike_entry_min_range_atr_ratio"),
                volume_spike_entry_min_close_location=msg.get("volume_spike_entry_min_close_location"),
                volume_spike_entry_futures_sl_pct=msg.get("volume_spike_entry_futures_sl_pct"),
                volume_spike_entry_options_sl_pct=msg.get("volume_spike_entry_options_sl_pct"),
                volume_spike_entry_stabilization_seconds=msg.get("volume_spike_entry_stabilization_seconds"),
                volume_spike_entry_stabilization_seconds_min=msg.get("volume_spike_entry_stabilization_seconds_min"),
                volume_spike_entry_stabilization_seconds_max=msg.get("volume_spike_entry_stabilization_seconds_max"),
                volume_spike_depth_confirm_enabled=msg.get("volume_spike_depth_confirm_enabled"),
                volume_spike_depth_ofr_bullish=msg.get("volume_spike_depth_ofr_bullish"),
                volume_spike_depth_ofr_bearish=msg.get("volume_spike_depth_ofr_bearish"),
                stabilization_momentum_override_enabled=msg.get("stabilization_momentum_override_enabled"),
                live_signal_stabilization_seconds_5min=msg.get("live_signal_stabilization_seconds_5min"),
                live_signal_stabilization_seconds_15min=msg.get("live_signal_stabilization_seconds_15min"),
                live_15min_momentum_override_enabled=msg.get("live_15min_momentum_override_enabled"),
                live_15min_momentum_min_confidence=msg.get("live_15min_momentum_min_confidence"),
                live_15min_momentum_volume_ratio=msg.get("live_15min_momentum_volume_ratio"),
                live_15min_momentum_range_atr_ratio=msg.get("live_15min_momentum_range_atr_ratio"),
                live_15min_momentum_close_location=msg.get("live_15min_momentum_close_location"),
                live_15min_momentum_adx_threshold=msg.get("live_15min_momentum_adx_threshold"),
                live_15min_momentum_intel_threshold=msg.get("live_15min_momentum_intel_threshold"),
                reversal_entry_relief_enabled=msg.get("reversal_entry_relief_enabled"),
                reversal_entry_min_confidence=msg.get("reversal_entry_min_confidence"),
                reversal_entry_thrust_min_range_atr_ratio=msg.get("reversal_entry_thrust_min_range_atr_ratio"),
                reversal_entry_thrust_min_close_location=msg.get("reversal_entry_thrust_min_close_location"),
                reversal_entry_thrust_min_volume_ratio=msg.get("reversal_entry_thrust_min_volume_ratio"),
                reversal_entry_vwap_min_dev_pct=msg.get("reversal_entry_vwap_min_dev_pct"),
                directional_census_reversal_relief_enabled=msg.get("directional_census_reversal_relief_enabled"),
                directional_census_reversal_min_confidence=msg.get("directional_census_reversal_min_confidence"),
                late_session_choppy_entry_guard_enabled=msg.get("late_session_choppy_entry_guard_enabled"),
                late_session_choppy_min_runway_minutes=msg.get("late_session_choppy_min_runway_minutes"),
                market_wide_reversal_enabled=msg.get("market_wide_reversal_enabled"),
                market_wide_reversal_core_votes=msg.get("market_wide_reversal_core_votes"),
                market_wide_reversal_total_votes=msg.get("market_wide_reversal_total_votes"),
                market_wide_reversal_min_range_atr_ratio=msg.get("market_wide_reversal_min_range_atr_ratio"),
                market_wide_reversal_min_range_atr_ratio_5min=msg.get("market_wide_reversal_min_range_atr_ratio_5min"),
                market_wide_reversal_min_range_atr_ratio_15min=msg.get("market_wide_reversal_min_range_atr_ratio_15min"),
                market_wide_reversal_midcap_min_range_atr_ratio_5min=msg.get("market_wide_reversal_midcap_min_range_atr_ratio_5min"),
                market_wide_reversal_midcap_min_range_atr_ratio_15min=msg.get("market_wide_reversal_midcap_min_range_atr_ratio_15min"),
                market_wide_reversal_min_close_location=msg.get("market_wide_reversal_min_close_location"),
                market_wide_reversal_max_skew_seconds_5min=msg.get("market_wide_reversal_max_skew_seconds_5min"),
                market_wide_reversal_max_skew_seconds_15min=msg.get("market_wide_reversal_max_skew_seconds_15min"),
                one_way_cascade_enabled=msg.get("one_way_cascade_enabled"),
                one_way_cascade_core_votes=msg.get("one_way_cascade_core_votes"),
                one_way_cascade_total_votes=msg.get("one_way_cascade_total_votes"),
                one_way_cascade_volume_votes=msg.get("one_way_cascade_volume_votes"),
                one_way_cascade_min_volume_ratio=msg.get("one_way_cascade_min_volume_ratio"),
                one_way_cascade_min_range_atr_ratio_1min=msg.get("one_way_cascade_min_range_atr_ratio_1min"),
                one_way_cascade_min_range_atr_ratio_5min=msg.get("one_way_cascade_min_range_atr_ratio_5min"),
                one_way_cascade_min_range_atr_ratio_15min=msg.get("one_way_cascade_min_range_atr_ratio_15min"),
                one_way_cascade_min_close_location=msg.get("one_way_cascade_min_close_location"),
                one_way_cascade_runner_lock_pct=msg.get("one_way_cascade_runner_lock_pct"),
                one_way_cascade_options_sl_pct=msg.get("one_way_cascade_options_sl_pct"),
                one_way_cascade_futures_sl_pct=msg.get("one_way_cascade_futures_sl_pct"),
                one_way_cascade_rr=msg.get("one_way_cascade_rr"),
                missed_signal_reentry_enabled=msg.get("missed_signal_reentry_enabled"),
                missed_signal_reentry_max_age_minutes=msg.get("missed_signal_reentry_max_age_minutes"),
                missed_signal_reentry_min_age_seconds=msg.get("missed_signal_reentry_min_age_seconds"),
                missed_signal_reentry_max_chase_atr=msg.get("missed_signal_reentry_max_chase_atr"),
                missed_signal_reentry_min_confidence=msg.get("missed_signal_reentry_min_confidence"),
                missed_signal_reentry_min_intel_score=msg.get("missed_signal_reentry_min_intel_score"),
                missed_signal_reentry_min_trend_adx=msg.get("missed_signal_reentry_min_trend_adx"),
                missed_signal_reentry_min_remaining_minutes=msg.get("missed_signal_reentry_min_remaining_minutes"),
                missed_signal_reentry_allow_trend_context=msg.get("missed_signal_reentry_allow_trend_context"),
                missed_signal_reentry_prefer_options=msg.get("missed_signal_reentry_prefer_options"),
                missed_signal_reentry_options_sl_pct=msg.get("missed_signal_reentry_options_sl_pct"),
                missed_signal_reentry_futures_sl_pct=msg.get("missed_signal_reentry_futures_sl_pct"),
                missed_signal_reentry_rr=msg.get("missed_signal_reentry_rr"),
                late_session_reclaim_override_enabled=msg.get("late_session_reclaim_override_enabled"),
                late_session_reclaim_start=msg.get("late_session_reclaim_start"),
                late_session_reclaim_end=msg.get("late_session_reclaim_end"),
                late_session_reclaim_core_votes=msg.get("late_session_reclaim_core_votes"),
                late_session_reclaim_total_votes=msg.get("late_session_reclaim_total_votes"),
                late_session_reclaim_reclaim_votes=msg.get("late_session_reclaim_reclaim_votes"),
                late_session_reclaim_max_skew_seconds=msg.get("late_session_reclaim_max_skew_seconds"),
                late_session_reclaim_min_close_location=msg.get("late_session_reclaim_min_close_location"),
                late_session_reclaim_min_volume_ratio=msg.get("late_session_reclaim_min_volume_ratio"),
                late_session_reclaim_volume_votes=msg.get("late_session_reclaim_volume_votes"),
                late_session_reclaim_options_sl_pct=msg.get("late_session_reclaim_options_sl_pct"),
                late_session_reclaim_futures_sl_pct=msg.get("late_session_reclaim_futures_sl_pct"),
                late_session_reclaim_rr=msg.get("late_session_reclaim_rr"),
                fifteen_min_choppy_scalp_guard_enabled=msg.get("fifteen_min_choppy_scalp_guard_enabled"),
                fifteen_min_choppy_scalp_min_intel_score=msg.get("fifteen_min_choppy_scalp_min_intel_score"),
                fifteen_min_choppy_scalp_min_volume_ratio=msg.get("fifteen_min_choppy_scalp_min_volume_ratio"),
                fifteen_min_choppy_scalp_micro_max_skew_seconds=msg.get("fifteen_min_choppy_scalp_micro_max_skew_seconds"),
                fifteen_min_choppy_scalp_min_close_location=msg.get("fifteen_min_choppy_scalp_min_close_location"),
                fifteen_min_choppy_scalp_profit_lock_r_multiple=msg.get("fifteen_min_choppy_scalp_profit_lock_r_multiple"),
                fifteen_min_choppy_scalp_min_peak_rs=msg.get("fifteen_min_choppy_scalp_min_peak_rs"),
                midcap_15min_risky_stabilization_seconds=msg.get("midcap_15min_risky_stabilization_seconds"),
                midcap_15min_risky_stabilization_min_scans=msg.get("midcap_15min_risky_stabilization_min_scans"),
                midcap_15min_risky_stabilization_min_volume_ratio=msg.get("midcap_15min_risky_stabilization_min_volume_ratio"),
                midcap_15min_risky_stabilization_min_intel_score=msg.get("midcap_15min_risky_stabilization_min_intel_score"),
                ut_confidence_calibration_enabled=msg.get("ut_confidence_calibration_enabled"),
                engine_lab_enabled=msg.get("engine_lab_enabled"),
                engine_lab_real_mode_only=msg.get("engine_lab_real_mode_only"),
                dual_lane_enabled=msg.get("dual_lane_enabled"),
                dual_lane_same_dir_min_bars_gap=msg.get("dual_lane_same_dir_min_bars_gap"),
                dual_lane_same_dir_min_lane_a_r=msg.get("dual_lane_same_dir_min_lane_a_r"),
                dual_lane_size_fraction=msg.get("dual_lane_size_fraction"),
                dual_lane_max_index_exposure_rs=msg.get("dual_lane_max_index_exposure_rs"),
                dual_lane_min_lane_b_confidence=msg.get("dual_lane_min_lane_b_confidence"),
                dual_lane_jaccard_alert_threshold=msg.get("dual_lane_jaccard_alert_threshold"),
                dual_clone_reduction_enabled=msg.get("dual_clone_reduction_enabled"),
                dual_clone_reduction_window_seconds=msg.get("dual_clone_reduction_window_seconds"),
                cascade_lot_scale_enabled=msg.get("cascade_lot_scale_enabled"),
                cascade_lot_multiplier_opt=msg.get("cascade_lot_multiplier_opt"),
                cascade_lot_multiplier_opt_min=msg.get("cascade_lot_multiplier_opt_min"),
                cascade_lot_multiplier_fut=msg.get("cascade_lot_multiplier_fut"),
                heavy_trend_lot_scale_enabled=msg.get("heavy_trend_lot_scale_enabled"),
                heavy_trend_lot_multiplier_opt=msg.get("heavy_trend_lot_multiplier_opt"),
                heavy_trend_min_aligned_indices=msg.get("heavy_trend_min_aligned_indices"),
                heavy_trend_vote_max_age_seconds=msg.get("heavy_trend_vote_max_age_seconds"),
                cascade_multi_tf_stack_enabled=msg.get("cascade_multi_tf_stack_enabled"),
                cascade_primary_max_tf=msg.get("cascade_primary_max_tf"),
                cascade_dual_addon_max=msg.get("cascade_dual_addon_max"),
                cascade_midcap_solo_max_tf=msg.get("cascade_midcap_solo_max_tf"),
                cascade_midcap_dual_main_max=msg.get("cascade_midcap_dual_main_max"),
                cascade_midcap_dual_addon_max=msg.get("cascade_midcap_dual_addon_max"),
                cascade_core_bucket_max_positions=msg.get("cascade_core_bucket_max_positions"),
                cascade_midcap_bucket_max_positions=msg.get("cascade_midcap_bucket_max_positions"),
                cascade_max_concurrent_positions=msg.get("cascade_max_concurrent_positions"),
                multi_signal_mixed_book_enabled=msg.get("multi_signal_mixed_book_enabled"),
                multi_signal_max_fut=msg.get("multi_signal_max_fut"),
                multi_signal_fut_lots=msg.get("multi_signal_fut_lots"),
                multi_signal_max_opt_instruments=msg.get("multi_signal_max_opt_instruments"),
                execution_use_touch_quote=msg.get("execution_use_touch_quote"),
                execution_touch_quote_shadow_enabled=msg.get("execution_touch_quote_shadow_enabled"),
                execution_touch_quote_strict=msg.get("execution_touch_quote_strict"),
                ut_confidence_calibration_min_samples=msg.get("ut_confidence_calibration_min_samples"),
                **{name: msg.get(name) for name in _S1_SETTING_SPEC},
                **{name: msg.get(name) for name in _UTS_SETTING_SPEC},
                **{name: msg.get(name) for name in _S12_SETTING_SPEC},
                ut_no_entry_after=msg.get("ut_no_entry_after"),
                ut_5min_no_entry_after=msg.get("ut_5min_no_entry_after"),
                ut_1min_no_entry_after=msg.get("ut_1min_no_entry_after"),
                ut_force_exit_time=msg.get("ut_force_exit_time"),
                real_mode_premarket_arm_start=msg.get("real_mode_premarket_arm_start"),
                real_mode_premarket_settle_end=msg.get("real_mode_premarket_settle_end"),
                real_mode_premarket_refresh_end=msg.get("real_mode_premarket_refresh_end"),
                real_mode_live_start=msg.get("real_mode_live_start"),
                auto_real_mode_premarket_switch_time=msg.get("auto_real_mode_premarket_switch_time"),
                auto_real_mode_premarket_enabled=msg.get("auto_real_mode_premarket_enabled"),
                auto_real_mode_startup_enabled=msg.get("auto_real_mode_startup_enabled"),
                # Market LTP shock + rolling mood: whitelisted in
                # _ADVANCED_CONFIG_FIELDS but previously never forwarded here —
                # the dashboard controls were silently dropped (2026-07-10).
                market_ltp_shock_enabled=msg.get("market_ltp_shock_enabled"),
                market_ltp_shock_min_move_pct=msg.get("market_ltp_shock_min_move_pct"),
                market_ltp_shock_strong_move_pct=msg.get("market_ltp_shock_strong_move_pct"),
                market_ltp_shock_lookback_minutes=msg.get("market_ltp_shock_lookback_minutes"),
                market_ltp_shock_min_lookback_minutes=msg.get("market_ltp_shock_min_lookback_minutes"),
                market_ltp_shock_min_indices=msg.get("market_ltp_shock_min_indices"),
                market_ltp_shock_min_core_indices=msg.get("market_ltp_shock_min_core_indices"),
                market_ltp_shock_min_volume_ratio=msg.get("market_ltp_shock_min_volume_ratio"),
                market_ltp_shock_session_start=msg.get("market_ltp_shock_session_start"),
                market_ltp_shock_session_end=msg.get("market_ltp_shock_session_end"),
                market_ltp_shock_session_override_start=msg.get("market_ltp_shock_session_override_start"),
                market_ltp_shock_outside_window_mode=msg.get("market_ltp_shock_outside_window_mode"),
                market_ltp_shock_outside_strong_move_pct=msg.get("market_ltp_shock_outside_strong_move_pct"),
                market_ltp_shock_opposite_trade_relax=msg.get("market_ltp_shock_opposite_trade_relax"),
                market_ltp_shock_force_exit_opposite=msg.get("market_ltp_shock_force_exit_opposite"),
                market_ltp_shock_stabilization_seconds=msg.get("market_ltp_shock_stabilization_seconds"),
                session_mood_dynamic_enabled=msg.get("session_mood_dynamic_enabled"),
                session_mood_min_confidence=msg.get("session_mood_min_confidence"),
                # Post-exit reversion / 1m spike (decoupled toggles).
                post_exit_reversion_enabled=msg.get("post_exit_reversion_enabled"),
                post_exit_reversion_window_minutes=msg.get("post_exit_reversion_window_minutes"),
                post_exit_reversion_min_rr=msg.get("post_exit_reversion_min_rr"),
                post_exit_reversion_include_stop_loss=msg.get("post_exit_reversion_include_stop_loss"),
                post_exit_reversion_include_oi_pcr=msg.get("post_exit_reversion_include_oi_pcr"),
                post_exit_reversion_open_distress_enabled=msg.get("post_exit_reversion_open_distress_enabled"),
                post_exit_reversion_loss_band_soft_rs=msg.get("post_exit_reversion_loss_band_soft_rs"),
                post_exit_reversion_loss_band_deep_rs=msg.get("post_exit_reversion_loss_band_deep_rs"),
                post_exit_reversion_min_move_pct=msg.get("post_exit_reversion_min_move_pct"),
                post_exit_reversion_move_lookback_minutes=msg.get("post_exit_reversion_move_lookback_minutes"),
                post_exit_reversion_fast_move_pct=msg.get("post_exit_reversion_fast_move_pct"),
                post_exit_reversion_fast_lookback_minutes=msg.get("post_exit_reversion_fast_lookback_minutes"),
                post_exit_reversion_slow_move_pct=msg.get("post_exit_reversion_slow_move_pct"),
                post_exit_reversion_slow_lookback_minutes=msg.get("post_exit_reversion_slow_lookback_minutes"),
                rare_dual_tf_entry_enabled=msg.get("rare_dual_tf_entry_enabled"),
                rare_dual_tf_min_grade=msg.get("rare_dual_tf_min_grade"),
                rare_dual_tf_min_confidence=msg.get("rare_dual_tf_min_confidence"),
                rare_dual_tf_prefer_opt_second=msg.get("rare_dual_tf_prefer_opt_second"),
                one_min_scalping_high_burst_only=msg.get("one_min_scalping_high_burst_only"),
                # Added 2026-08-11. Whitelisting alone is not enough — the 2026-07-10
                # regression noted above was exactly this: the control appears in the UI
                # and is silently discarded. These are the A/B levers for the backtest
                # trade-count gap plus the 1min lane and the widened exit vote.
                block_opposite_direction_stack=msg.get("block_opposite_direction_stack"),
                consecutive_loss_streak_per_index=msg.get("consecutive_loss_streak_per_index"),
                surge_tick_proxy_enabled=msg.get("surge_tick_proxy_enabled"),
                intel_reversal_corroboration_enabled=msg.get("intel_reversal_corroboration_enabled"),
                partial_entry_remainder_cancel_enabled=msg.get("partial_entry_remainder_cancel_enabled"),
                morning_refresh_catchup_at_live_open=msg.get("morning_refresh_catchup_at_live_open"),
                surge_tick_percentile_floor=msg.get("surge_tick_percentile_floor"),
                intel_reversal_corroboration_votes=msg.get("intel_reversal_corroboration_votes"),
                partial_entry_remainder_timeout_seconds=msg.get("partial_entry_remainder_timeout_seconds"),
                kite_rest_backoff_seconds=msg.get("kite_rest_backoff_seconds"),
                kite_rest_backoff_max_seconds=msg.get("kite_rest_backoff_max_seconds"),
                calendar_coverage_warn_days=msg.get("calendar_coverage_warn_days"),
                # Pullback hedge (operator A/B; ships OFF — see _ADVANCED_CONFIG_FIELDS).
                # Must be forwarded explicitly: whitelisting alone silently drops the
                # control (2026-07-10 regression noted above).
                per_timeframe_grading_enabled=msg.get("per_timeframe_grading_enabled"),
                stop_out_reentry_enabled=msg.get("stop_out_reentry_enabled"),
                stop_out_reentry_window_minutes=msg.get("stop_out_reentry_window_minutes"),
                stop_out_reentry_max_per_index=msg.get("stop_out_reentry_max_per_index"),
                stop_out_reentry_min_adx=msg.get("stop_out_reentry_min_adx"),
                pullback_hedge_enabled=msg.get("pullback_hedge_enabled"),
                pullback_hedge_trigger_lock_fraction=msg.get("pullback_hedge_trigger_lock_fraction"),
                pullback_hedge_resume_recovery_pct=msg.get("pullback_hedge_resume_recovery_pct"),
                pullback_hedge_size_fraction=msg.get("pullback_hedge_size_fraction"),
                pullback_hedge_max_hold_minutes=msg.get("pullback_hedge_max_hold_minutes"),
            )
            await _send_command_response(
                ws,
                msg,
                "config_ack",
                _configuration_result_payload(configured, scanner_ref, msg.get("mode") if "mode" in msg else None),
            )

    elif cmd == "set_mode":
        if scanner_ref:
            mode = msg.get("mode", "paper")
            reset = msg.get("reset", False)
            configured = await _configure_on_scanner_loop(
                mode=mode,
                reset=reset,
                confirm_real_mode=msg.get("confirm_real_mode", False),
                real_mode_verification=msg.get("real_mode_verification", ""),
            )
            await _send_command_response(
                ws,
                msg,
                "mode_updated",
                _configuration_result_payload(configured, scanner_ref, mode),
            )

    elif cmd == "reset_pnl":
        if scanner_ref:
            configured = await _call_on_scanner_loop(
                lambda: scanner_ref.configure(reset=True)
            )
            await _send_command_response(
                ws,
                msg,
                "pnl_reset",
                {"status": "ok" if configured is not False else "blocked"},
            )

    elif cmd == "set_power":
        if scanner_ref:
            state = msg.get("state", "OFF")
            await _call_on_scanner_loop(
                lambda: setattr(scanner_ref, "system_power", state)
            )
            await _send_command_response(ws, msg, "power_updated", {"state": state})

    elif cmd == "set_regime_adaptation":
        if scanner_ref:
            state = msg.get("state", "OFF")
            is_on = state == "ON"

            def _set_regime_adaptation() -> None:
                settings = get_settings()
                settings.ut_regime_adaptation = is_on
                schedule_settings_save(settings)
                if hasattr(getattr(scanner_ref, "mtf", None), "apply_engine_params"):
                    scanner_ref.mtf.apply_engine_params(settings.get_ut_engine_params())

            await _call_on_scanner_loop(_set_regime_adaptation)
            await _send_command_response(ws, msg, "config_ack", {"status": "ok"})

    elif cmd == "set_sl_mode":
        if scanner_ref:
            state = str(msg.get("state", "NATURAL") or "NATURAL").upper()
            if state not in {"NATURAL", "HARDCODED"}:
                await _send_command_response(ws, msg, "config_ack", {"status": "blocked", "reason": "invalid_sl_mode"})
                return

            def _set_sl_mode() -> None:
                settings = get_settings()
                settings.sl_mode = state
                schedule_settings_save(settings)

            await _call_on_scanner_loop(_set_sl_mode)
            await _send_command_response(ws, msg, "config_ack", {"status": "ok"})

    elif cmd == "set_scanner_engine":
        if scanner_ref:
            state = str(msg.get("state", "utbot") or "utbot").lower()
            if state not in {"utbot", "ema", "consensus", "utbot_consensus", "quad_union", "quad_prime", "adaptive_dual", "s1", "ut_smart", "quad_prime_s1", "ut_smart_s1", "s1s2_dual", "ut_ema"}:
                await _send_command_response(ws, msg, "config_ack", {"status": "blocked", "reason": "invalid_scanner_engine"})
                return

            def _set_scanner_engine() -> Dict[str, Any]:
                if hasattr(scanner_ref, "apply_scanner_engine_switch"):
                    return scanner_ref.apply_scanner_engine_switch(state)
                settings = get_settings()
                settings.scanner_engine = state
                schedule_settings_save(settings)
                return {"scanner_engine": state, "recalculation_queued": False}

            result = await _call_on_scanner_loop(_set_scanner_engine)
            # REAL/live mode only needs a lightweight recalibration; HISTORICAL mode
            # queues a full rebuild inside apply_scanner_engine_switch.
            if not result.get("recalculation_queued") and hasattr(scanner_ref, "perform_system_recalibration"):
                _schedule_on_scanner_loop(scanner_ref.perform_system_recalibration())
            await _send_command_response(ws, msg, "config_ack", {"status": "ok", **result})

    elif cmd == "set_concurrency_guard":
        if scanner_ref:
            state = msg.get("state", "ON")
            is_on = state == "ON"
            await _call_on_scanner_loop(
                lambda: scanner_ref.configure(concurrency_guard=is_on)
            )
            await _send_command_response(ws, msg, "config_ack", {"status": "ok"})

    elif cmd == "set_expiry_gamma":
        if scanner_ref:
            is_on = msg.get("state", "OFF") == "ON"

            def _set_expiry_gamma() -> None:
                settings = get_settings()
                settings.expiry_day_trade_current_week = is_on
                schedule_settings_save(settings)
                # New expiry selection only affects future option resolutions.
                try:
                    from engine.expiry_manager import expiry_manager

                    expiry_manager._token_cache.clear()
                except Exception:
                    pass

            await _call_on_scanner_loop(_set_expiry_gamma)
            await _send_command_response(ws, msg, "config_ack", {"status": "ok", "expiry_day_trade_current_week": is_on})

    elif cmd == "set_combo_partner_exit":
        if scanner_ref:
            is_on = msg.get("state", "ON") == "ON"

            def _set_combo_partner_exit() -> None:
                settings = get_settings()
                settings.combo_partner_confirms_exit = is_on
                schedule_settings_save(settings)

            await _call_on_scanner_loop(_set_combo_partner_exit)
            await _send_command_response(ws, msg, "config_ack", {"status": "ok", "combo_partner_confirms_exit": is_on})

    elif cmd == "execute_signal":
        if scanner_ref:
            try:
                result = await _await_on_scanner_loop(
                    scanner_ref.execute_manual_signal(
                        signal_id=msg.get("signal_id", ""),
                        lifecycle_id=msg.get("lifecycle_id", ""),
                    )
                )
            except Exception as exc:
                logger.error(f"Dashboard execute_signal failed: {exc}")
                result = {"status": "blocked", "reason": "execution_error"}
            await _send_command_response(ws, msg, "signal_executed", make_serializable(result))

    elif cmd == "close_trade":
        if scanner_ref:
            tid = msg.get("trade_id", "")
            price = float(msg.get("price", 0) or 0)
            try:
                if hasattr(scanner_ref, "close_dashboard_trade"):
                    result = await _await_on_scanner_loop(
                        scanner_ref.close_dashboard_trade(tid, price, "MANUAL")
                    )
                else:
                    async def _fallback_close() -> bool:
                        trades = scanner_ref.trades
                        trade = getattr(trades, "open_trades", {}).get(tid)
                        confirm_manual = getattr(
                            trades,
                            "confirm_latched_manual_exit",
                            None,
                        )
                        if bool(
                            getattr(
                                trade,
                                "manual_exit_confirmation_pending",
                                False,
                            )
                        ) and callable(confirm_manual):
                            return bool(
                                await asyncio.to_thread(
                                    confirm_manual,
                                    tid,
                                    price,
                                    "MANUAL",
                                )
                            )
                        latch_exit = getattr(trades, "latch_exit_intent", None)
                        if callable(latch_exit):
                            latch_exit(tid, "MANUAL", price)
                        return bool(
                            await asyncio.to_thread(
                                trades.close_trade,
                                tid,
                                price,
                                "MANUAL",
                            )
                        )

                    accepted = await _await_on_scanner_loop(
                        _fallback_close()
                    )
                    result = {
                        "trade_id": tid,
                        "status": "submitted" if accepted else "blocked",
                    }
            except Exception as exc:
                logger.error(f"Dashboard close_trade failed: {exc}")
                result = {"trade_id": tid, "status": "blocked", "reason": "execution_error"}
            await _send_command_response(
                ws,
                msg,
                "trade_closed",
                make_serializable(result),
            )

    elif cmd == "kill_all":
        if scanner_ref:
            async def _kill_all() -> Dict[str, Any]:
                trades = scanner_ref.trades
                lock = getattr(trades, "_lock", None)
                if lock is None:
                    open_snapshot = list(trades.open_trades.items())
                else:
                    with lock:
                        open_snapshot = list(trades.open_trades.items())
                latch_exit = getattr(trades, "latch_exit_intent", None)
                if callable(latch_exit):
                    for tid, trade in open_snapshot:
                        price = float(
                            getattr(trade, "current_price", 0.0)
                            or getattr(trade, "entry_price", 0.0)
                            or 0.0
                        )
                        latch_exit(tid, "HARDBREAKER", price)
                submitted = 0
                for tid, trade in open_snapshot:
                    price = float(
                        getattr(trade, "current_price", 0.0)
                        or getattr(trade, "entry_price", 0.0)
                        or 0.0
                    )
                    try:
                        confirm_manual = getattr(
                            trades,
                            "confirm_latched_manual_exit",
                            None,
                        )
                        close_fn = trades.close_trade
                        if bool(
                            getattr(
                                trade,
                                "manual_exit_confirmation_pending",
                                False,
                            )
                        ) and callable(confirm_manual):
                            close_fn = confirm_manual
                        submitted += bool(
                            await asyncio.to_thread(
                                close_fn,
                                tid,
                                price,
                                "HARDBREAKER",
                            )
                        )
                    except Exception as exc:
                        logger.error(f"HARDBREAKER exit submission failed for {tid}: {exc}")
                requested = len(open_snapshot)
                log_event = getattr(scanner_ref, "log_event", None)
                if callable(log_event):
                    try:
                        log_event(
                            f"🚨 HARDBREAKER: Exit submitted for {submitted}/{requested} trades.",
                            "warning",
                        )
                    except Exception as exc:
                        logger.debug(f"HARDBREAKER event log failed: {exc}")
                return {
                    "status": "submitted" if submitted == requested else "partial",
                    "submitted": submitted,
                    "requested": requested,
                }

            kill_result = await _await_on_scanner_loop(
                _kill_all(),
                timeout_seconds=30.0,
            )
            await _send_command_response(
                ws,
                msg,
                "all_trades_killed",
                kill_result,
            )

    elif cmd == "update_fyers_token":
        if scanner_ref:
            result = await _await_on_scanner_loop(
                scanner_ref.update_fyers_token_async(
                    msg.get("auth_code", "")
                ),
                timeout_seconds=30.0,
            )
            await _send_command_response(ws, msg, "fyers_auth_updated", make_serializable(result))

    elif cmd == "subscribe_chart":
        if scanner_ref:
            instrument = str(msg.get("instrument") or "NIFTY").upper()
            tf = str(msg.get("tf") or "5min")
            snapshot = await _build_chart_off_loop(instrument, tf)
            await _send_command_response(ws, msg, "chart_snapshot", make_serializable(snapshot))

    elif cmd == "set_chart_stream":
        if scanner_ref:
            enabled = msg.get("enabled", True)

            def _set_chart_stream() -> bool:
                if hasattr(scanner_ref, "set_chart_stream_enabled"):
                    return bool(scanner_ref.set_chart_stream_enabled(enabled))
                scanner_ref.chart_stream_enabled = bool(enabled)
                return bool(enabled)

            enabled = await _call_on_scanner_loop(_set_chart_stream)
            await _send_command_response(ws, msg, "chart_stream_updated", {"enabled": bool(enabled)})

    elif cmd == "system_recalibrate":
        if scanner_ref:
            scheduled = _schedule_on_scanner_loop(scanner_ref.perform_system_recalibration())
            await _send_command_response(
                ws,
                msg,
                "config_ack",
                {"status": "ok" if scheduled else "blocked"},
            )


# Cache to track last sent state to clients
import asyncio

class DashboardState:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.latest_safe_state: Dict[str, Any] = {}
        self.last_sent_states: Dict[int, Any] = {}

dashboard_state = DashboardState()

def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return bool(value)


def _scanner_is_historical() -> bool:
    return bool(scanner_ref and str(getattr(scanner_ref, "mode", "") or "").upper() == "HISTORICAL")


def _scanner_is_historical_building() -> bool:
    if not _scanner_is_historical() or not scanner_ref:
        return False
    recalc = getattr(scanner_ref, "_recalculation_status", {}) or {}
    recalc_status = str(recalc.get("status", "") or "").lower()
    if recalc_status in {
        "queued",
        "waiting_initial_setup",
        "worker_running",
        "applying",
        "running",
        "awaiting_scan",
        "scan_running",
    }:
        return True
    if bool(getattr(scanner_ref, "_historical_rebuild_pending", False)):
        return True
    if bool(getattr(scanner_ref, "_calculation_lock", False)) or bool(getattr(scanner_ref, "is_calculating", False)):
        return True
    return int(getattr(scanner_ref, "scan_count", 0) or 0) == 0


def _current_trade_context() -> Dict[str, Any]:
    settings = get_settings()
    return {
        "mode": str(getattr(scanner_ref, "mode", "") or "").upper() if scanner_ref else "",
        "backtest_days": int(getattr(scanner_ref, "backtest_days", getattr(settings, "default_backtest_days", 1)) or 1),
        "inst_pref": str(getattr(scanner_ref, "inst_pref", getattr(settings, "inst_pref", "AUTO")) or "AUTO").upper(),
        "scanner_engine": str(getattr(settings, "scanner_engine", "utbot") or "utbot").lower(),
        "ut_concurrency_guard": _coerce_bool(getattr(settings, "ut_concurrency_guard", True), True),
        "futures_shadow_enabled": _coerce_bool(
            getattr(settings, "futures_shadow_enabled", False),
            False,
        ),
        "grade_preference": str(getattr(settings, "signal_grade_preference", "auto") or "auto"),
        "timeframe_entry_policy": str(getattr(settings, "ut_timeframe_entry_policy", "PRIMARY_15") or "PRIMARY_15").upper(),
    }


def _cached_trades_match_current_context(cached: Dict[str, Any]) -> bool:
    if not isinstance(cached, dict) or not _payload_matches_scanner_mode({"trades": cached}):
        return False
    if not _scanner_is_historical():
        return True
    if _historical_trade_payload_rows_truncated(cached):
        return False

    meta = cached.get("meta") if isinstance(cached.get("meta"), dict) else {}
    if meta.get("complete") is not True:
        return False
    expected = _current_trade_context()
    if str(meta.get("mode", "") or "").upper() != expected["mode"]:
        return False
    for key in (
        "backtest_days",
        "inst_pref",
        "scanner_engine",
        "grade_preference",
        "timeframe_entry_policy",
        "futures_shadow_enabled",
    ):
        if key not in meta:
            return False
        if str(meta.get(key, "") or "").upper() != str(expected[key]).upper():
            return False
    if "ut_concurrency_guard" not in meta:
        return False
    return _coerce_bool(meta.get("ut_concurrency_guard"), True) == expected["ut_concurrency_guard"]


def _historical_trade_payload_rows_truncated(payload: Dict[str, Any]) -> bool:
    """Detect summary-full/cache-window mismatches in historical trade payloads."""
    if not isinstance(payload, dict):
        return False
    try:
        summary_total = int((payload.get("summary") or {}).get("total_trades") or 0)
    except Exception:
        summary_total = 0
    if summary_total <= 0:
        return False
    row_total = len(payload.get("closed") or []) + len(payload.get("open") or [])
    if row_total >= summary_total:
        return False
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    if str(meta.get("mode", "") or "").upper() != "HISTORICAL":
        return False
    return True


def _trade_payload_is_windowed(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get("window"), dict):
        return False
    for key in ("closed", "signals"):
        total = int(payload.get(f"{key}_total") or 0)
        visible = len(payload.get(key) or [])
        if total > visible:
            return True
    return False


def _building_trades_payload() -> Dict[str, Any]:
    expected = _current_trade_context()
    recalc = getattr(scanner_ref, "_recalculation_status", {}) or {} if scanner_ref else {}
    return {
        "open": [],
        "closed": [],
        "signals": [],
        "summary": {
            "daily_pnl": 0.0,
            "fut_pnl": 0.0,
            "opt_pnl": 0.0,
            "total_trades": 0,
            "open_count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "fut_trades": 0,
            "opt_trades": 0,
        },
        "meta": {
            **expected,
            "simulation_id": int(getattr(scanner_ref, "simulation_id", 0) or 0) if scanner_ref else 0,
            "complete": False,
            "building": True,
            "recalculation_status": str(recalc.get("status", "") or ""),
            "source": "historical_building_guard",
        },
    }


def _historical_building_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay stale dashboard state with an explicit historical-building shell."""
    if not _scanner_is_historical_building() or not scanner_ref:
        return payload
    state = dict(payload or {})
    state["mode"] = getattr(scanner_ref, "mode", "HISTORICAL")
    if hasattr(scanner_ref, "get_broadcast_config"):
        state["config"] = scanner_ref.get_broadcast_config()
    if hasattr(scanner_ref, "_get_diagnostics_payload"):
        state["diagnostics"] = scanner_ref._get_diagnostics_payload()
    state["simulation_id"] = int(getattr(scanner_ref, "simulation_id", 0) or 0)
    state["is_calculating"] = True
    state["trades"] = _building_trades_payload()
    return state


def _build_fresh_trades_payload() -> Dict[str, Any]:
    global _latest_full_trades_payload
    if not scanner_ref:
        return {"open": [], "closed": [], "summary": {}, "signals": []}
    if hasattr(scanner_ref, "_build_dashboard_trade_payload"):
        payload = make_serializable(scanner_ref._build_dashboard_trade_payload())
    else:
        payload = make_serializable(scanner_ref.trades.get_dashboard_payload(
            is_historical=(scanner_ref.mode == "HISTORICAL"),
            backtest_days=scanner_ref.backtest_days,
        ))
    if not _trade_payload_is_windowed(payload):
        _latest_full_trades_payload = payload
    try:
        if hasattr(scanner_ref, "dashboard_cache") and not _republish_cached_trades(payload):
            # No state from this mode/simulation yet (before the first scan cycle, or
            # just after a switch): build the whole thing, as before.
            state = make_serializable(
                scanner_ref.dashboard_cache.state()
                or getattr(scanner_ref, "latest_results", {})
                or {}
            )
            state["mode"] = getattr(scanner_ref, "mode", None)
            # Every container in ``state`` is rebuilt here, so the cache can take it
            # without a second deep copy on the scanner loop. The two scanner-built
            # parts and the returned trades payload are detached explicitly.
            if hasattr(scanner_ref, "get_broadcast_config"):
                state["config"] = make_serializable(scanner_ref.get_broadcast_config())
            if hasattr(scanner_ref, "_get_diagnostics_payload"):
                state["diagnostics"] = make_serializable(scanner_ref._get_diagnostics_payload())
            state["simulation_id"] = int(getattr(scanner_ref, "simulation_id", 0) or 0)
            state["trades"] = make_serializable(payload)
            if hasattr(scanner_ref, "_sync_engine_lab_active_row"):
                scanner_ref._sync_engine_lab_active_row(state, (state["trades"].get("summary") or {}))
            _schedule_call_on_scanner_loop(
                lambda snapshot=state: scanner_ref.dashboard_cache.update(snapshot, detached=True)
            )
    except Exception:
        logger.debug("Unable to refresh dashboard trade cache", exc_info=True)
    return payload


def _republish_cached_trades(payload: Dict[str, Any]) -> bool:
    """Swap fresh trades into the dashboard state the scan cycle last published.

    The scan cycle republishes diagnostics, config and instruments on every cycle, so a
    trades refresh only has to replace the trades and the Engine Lab row derived from
    them. Rebuilding the rest here, once a second on the scanner loop, is what starved
    that loop on 2026-10-06 (see
    tests/test_trades_heartbeat_cannot_starve_the_scanner_loop_2026_10_06.py).
    """
    cache = getattr(scanner_ref, "dashboard_cache", None)
    update_trades = getattr(cache, "update_trades", None)
    if not callable(update_trades):
        return False
    mode = getattr(scanner_ref, "mode", None)
    simulation_id = int(getattr(scanner_ref, "simulation_id", 0) or 0)

    def retarget(state: Dict[str, Any]) -> bool:
        if state.get("mode") != mode or int(state.get("simulation_id") or 0) != simulation_id:
            return False
        if isinstance(state.get("engine_lab"), dict) and hasattr(scanner_ref, "_sync_engine_lab_active_row"):
            # Retargeting edits the lab rows in place, so this generation gets its own
            # copy: a reader may still be serializing the previous one.
            state["engine_lab"] = make_serializable(state["engine_lab"])
            scanner_ref._sync_engine_lab_active_row(state, payload.get("summary") or {})
        return True

    return bool(update_trades(payload, retarget))


def build_trades_payload() -> Dict[str, Any]:
    if not scanner_ref:
        return {"open": [], "closed": [], "summary": {}, "signals": []}
    if not _scanner_is_historical():
        return _build_fresh_trades_payload()
    if (
        dashboard_state.latest_safe_state
        and _payload_matches_scanner_mode(dashboard_state.latest_safe_state)
        and isinstance(dashboard_state.latest_safe_state.get("trades"), dict)
        and not _trade_payload_is_windowed(dashboard_state.latest_safe_state["trades"])
        and _cached_trades_match_current_context(dashboard_state.latest_safe_state["trades"])
    ):
        return dashboard_state.latest_safe_state["trades"]
    if hasattr(scanner_ref, "dashboard_cache"):
        cached = scanner_ref.dashboard_cache.trades()
        if cached and not _trade_payload_is_windowed(cached) and _cached_trades_match_current_context(cached):
            return cached
    if _scanner_is_historical_building():
        return _building_trades_payload()
    return _build_fresh_trades_payload()


def _payload_matches_scanner_mode(payload: Dict[str, Any]) -> bool:
    if not scanner_ref:
        return True
    expected_mode = str(getattr(scanner_ref, "mode", "") or "").upper()
    if not expected_mode:
        return True
    payload_mode = str((payload or {}).get("mode") or "").upper()
    trades = (payload or {}).get("trades")
    trade_meta = trades.get("meta") if isinstance(trades, dict) else {}
    trade_mode = str((trade_meta or {}).get("mode") or "").upper()
    if payload_mode and payload_mode != expected_mode:
        return False
    if trade_mode and trade_mode != expected_mode:
        return False
    return True


def _compact_ws_trade_candidate(candidate: Any) -> Any:
    """Keep live candidate state while dropping bulky nested audit context."""
    if not isinstance(candidate, dict):
        return candidate
    compact: Dict[str, Any] = {}
    for key, value in candidate.items():
        if isinstance(value, (int, float, bool, type(None))):
            compact[key] = value
        elif isinstance(value, str):
            compact[key] = value[:500]
        elif key == "reasons" and isinstance(value, list):
            compact[key] = [str(item)[:240] for item in value[:8]]
        elif key == "regime_conviction_vector" and isinstance(value, dict):
            compact[key] = dict(value)
    return compact


def _window_ws_instrument_candidates(
    instruments: Any,
    limit: int,
    *,
    compact: bool = False,
) -> Any:
    """Window candidate history without removing current instrument truth."""
    if not isinstance(instruments, dict):
        return instruments
    bounded: Dict[str, Any] = {}
    limit = max(1, int(limit))
    for name, ui_data in instruments.items():
        if not isinstance(ui_data, dict):
            bounded[name] = ui_data
            continue
        ui_bounded = dict(ui_data)
        candidates = ui_bounded.get("trade_candidates")
        if isinstance(candidates, list):
            try:
                prior_total = int(ui_bounded.get("trade_candidates_total") or len(candidates))
            except (TypeError, ValueError):
                prior_total = len(candidates)
            visible = list(candidates[:limit])
            if compact:
                visible = [_compact_ws_trade_candidate(row) for row in visible]
            ui_bounded["trade_candidates"] = visible
            ui_bounded["trade_candidates_total"] = prior_total
            ui_bounded["trade_candidates_truncated_for_ws"] = prior_total > len(visible)
        bounded[name] = ui_bounded
    return bounded


def _ws_frame_size(data: Dict[str, Any]) -> int:
    """Byte size of the frame this data would actually be sent as.

    ``data`` has already been through ``make_serializable``, so encode it directly
    with the encoder ``_send_one`` uses rather than framing a stdlib JSON string and
    re-walking the entire payload. On 2026-10-30 ``_ws_frame_size`` alone measured 45
    GIL samples per 15s of broadcast, on top of the send's own serialization.
    """
    return len(_frame_bytes({"type": "full_update", "data": data}))


def _cap_ws_payload(safe_data: Dict[str, Any], *, frame_cache=None) -> Dict[str, Any]:
    """Ensure WS/full_update frames stay under ~1 MiB client limits."""
    first_frame = True
    encoded_fields = {}

    def frame_size(candidate):
        nonlocal first_frame
        if first_frame:
            first_frame = False
            encoded = _frame_bytes({"type": "full_update", "data": candidate})
        else:
            # Oversized snapshots may retain a large unchanged diagnostics/log
            # field across every rescue stage. Encode each such field once;
            # count bytes before assembling another full frame. The initial,
            # normally sized frame keeps the native encoder's single-call path.
            parts = []
            size = len(b'{"type":"full_update","data":{}}')
            for key, value in candidate.items():
                previous = encoded_fields.get(key)
                if previous is not None and previous[0] is value:
                    part = previous[1]
                else:
                    part = _frame_bytes(key) + b":" + _frame_bytes(value)
                    encoded_fields[key] = (value, part)
                size += len(part) + bool(parts)
                if size > _WS_MAX_FRAME_BYTES:
                    return size
                parts.append(part)
            encoded = b'{"type":"full_update","data":{' + b",".join(parts) + b"}}"
        if frame_cache is not None:
            frame_cache.update(payload=candidate, encoded=encoded)
        return len(encoded)
    data = safe_data
    try:
        if frame_size(data) <= _WS_MAX_FRAME_BYTES:
            return data
    except Exception:
        return {
            "ws_truncated": True,
            "ws_payload_omitted": True,
            "ws_truncation_reason": "serialization_error",
        }

    # Progressive strip: historical arrays first, then per-index candidate history.
    # Current instrument fields (LTP, trade/trailing state, active signals and
    # intelligence) remain available to live clients throughout these stages.
    light = dict(data)
    trades = dict(light.get("trades") or {}) if isinstance(light.get("trades"), dict) else {}
    if isinstance(trades.get("signals"), list) and len(trades["signals"]) > 20:
        trades["signals"] = trades["signals"][:20]
    if isinstance(trades.get("closed"), list) and len(trades["closed"]) > 40:
        trades["closed"] = trades["closed"][:40]
        trades["window"] = dict(trades.get("window") or {})
        trades["window"]["truncated_for_ws"] = True
    if "analytics" in light:
        light["analytics"] = {}
    light["trades"] = trades
    light["instruments"] = _window_ws_instrument_candidates(
        light.get("instruments"),
        _WS_TRADE_CANDIDATE_LIMIT,
    )
    light["ws_truncated"] = True
    try:
        if frame_size(light) <= _WS_MAX_FRAME_BYTES:
            return light
    except Exception:
        pass

    # Rescue stage for unusually large candidate context. Compact candidate
    # rows but retain every current scalar (including stop, target, P&L, status,
    # direction and intelligence score), plus the full current instrument state.
    light["instruments"] = _window_ws_instrument_candidates(
        light.get("instruments"),
        _WS_TRADE_CANDIDATE_FALLBACK_LIMIT,
        compact=True,
    )
    activity_log = light.get("activity_log")
    if isinstance(activity_log, list) and len(activity_log) > _WS_ACTIVITY_LOG_LIMIT:
        light["activity_log"] = activity_log[-_WS_ACTIVITY_LOG_LIMIT:]

    try:
        if frame_size(light) <= _WS_MAX_FRAME_BYTES:
            return light
    except Exception:
        pass

    # Hard final guarantee. Preserve only small operational identity plus a
    # compact summary; clients can page trades/charts through REST afterwards.
    trades = light.get("trades") if isinstance(light.get("trades"), dict) else {}
    minimal: Dict[str, Any] = {
        "mode": str(light.get("mode") or "")[:32],
        "scan_count": light.get("scan_count"),
        "simulation_id": light.get("simulation_id"),
        "trades": {"summary": dict((trades or {}).get("summary") or {})},
        "ws_truncated": True,
        "ws_payload_omitted": True,
        "ws_truncation_reason": "frame_limit",
    }
    try:
        minimal_size = frame_size(minimal)
        if minimal_size <= _WS_MAX_FRAME_BYTES:
            return minimal
    except Exception:
        pass
    return {
        "ws_truncated": True,
        "ws_payload_omitted": True,
        "ws_truncation_reason": "frame_limit",
    }


def _ensure_broadcast_worker() -> asyncio.Queue:
    """Return the dashboard-loop latest-wins queue and ensure its pump is alive."""
    global _broadcast_queue, _broadcast_task
    if _broadcast_queue is None or _broadcast_task is None or _broadcast_task.done():
        _broadcast_queue = asyncio.Queue(maxsize=1)
        _broadcast_task = asyncio.create_task(
            _broadcast_worker(_broadcast_queue),
            name="DashboardBroadcastWorker",
        )
    return _broadcast_queue


async def broadcast(data: Dict):
    """Publish a dashboard snapshot without awaiting serialization or client I/O.

    The queue intentionally holds only the newest pending state. A stalled client
    can delay its own delivery worker, but it can no longer delay the scanner loop.
    """
    if _dashboard_shutting_down:
        return False
    queue = _ensure_broadcast_worker()
    payload = dict(data or {})
    if queue.full():
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(payload)
    return True


async def _broadcast_worker(queue: asyncio.Queue) -> None:
    while True:
        payload = await queue.get()
        try:
            await _deliver_broadcast(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"Dashboard broadcast worker error: {exc}")
        finally:
            queue.task_done()


async def _deliver_broadcast(data: Dict) -> None:
    """Serialize and deliver one coalesced snapshot on the dashboard loop."""
    global _latest_full_trades_payload
    payload = dict(data or {})
    payload["dashboard_heartbeat"] = _dashboard_heartbeat()
    payload = _historical_building_state(payload)
    incoming_trades = payload.get("trades")
    if isinstance(incoming_trades, dict):
        serialized_trades = make_serializable(incoming_trades)
        if not _trade_payload_is_windowed(incoming_trades):
            _latest_full_trades_payload = serialized_trades
        payload["trades"] = serialized_trades
        light = _lighten_state_payload(payload)
        safe_data = make_serializable({key: value for key, value in light.items() if key != "trades"})
        safe_data["trades"] = light["trades"]
    else:
        safe_data = make_serializable(_lighten_state_payload(payload))
    frame_cache = {}
    safe_data = _cap_ws_payload(safe_data, frame_cache=frame_cache)
    
    # The lock protects the published reference only. Never hold it while a
    # client socket is awaited.
    async with dashboard_state.lock:
        dashboard_state.latest_safe_state = safe_data

    if not active_connections:
        return

    disconnected = set()
    send_timeout = max(
        0.1,
        min(
            5.0,
            float(getattr(get_settings(), "dashboard_ws_send_timeout_seconds", 1.0) or 1.0),
        ),
    )
    clients = list(active_connections)
    messages = {}

    async def _send_one(ws) -> None:
        try:
            ws_id = id(ws)
            old_state = dashboard_state.last_sent_states.get(ws_id)
            key = id(old_state)
            if key not in messages:
                if old_state is None:
                    encoded = (frame_cache["encoded"] if frame_cache.get("payload") is safe_data else
                               _frame_bytes({"type": "full_update", "data": safe_data}))
                    messages[key] = encoded.decode("utf-8")
                else:
                    delta = compute_delta(old_state, safe_data)
                    messages[key] = (_frame_bytes({"type": "delta_update", "data": delta}).decode("utf-8")
                                     if delta else None)
            message = messages[key]
            if message is not None:
                await asyncio.wait_for(ws.send_text(message), timeout=send_timeout)
                dashboard_state.last_sent_states[ws_id] = safe_data
        except Exception:
            disconnected.add(ws)

    if clients:
        await asyncio.gather(*[_send_one(ws) for ws in clients], return_exceptions=True)

    async def _close_disconnected(ws: WebSocket) -> None:
        try:
            await asyncio.wait_for(ws.close(code=1011), timeout=0.25)
        except Exception:
            pass

    if disconnected:
        await asyncio.gather(
            *(_close_disconnected(ws) for ws in disconnected),
            return_exceptions=True,
        )
    for ws in disconnected:
        _forget_ws_connection(ws)


async def _wait_for_broadcast_idle(timeout_seconds: float = 2.0) -> None:
    """Test/shutdown helper: wait until the latest queued snapshot is delivered."""
    queue = _broadcast_queue
    if queue is None:
        return
    await asyncio.wait_for(queue.join(), timeout=max(0.05, float(timeout_seconds)))


async def _stop_broadcast_worker() -> None:
    global _broadcast_task, _broadcast_queue
    task = _broadcast_task
    _broadcast_task = None
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    queue = _broadcast_queue
    _broadcast_queue = None
    if queue is not None:
        while not queue.empty():
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                break

def compute_delta(old, new):
    """
    Computes a sparse delta dictionary. If applied to 'old' via Object.assign deep-merge, it yields 'new'.
    """
    if not isinstance(old, dict) or not isinstance(new, dict):
        return new if old != new else None

    delta = {}
    for k, v in new.items():
        if k not in old:
            delta[k] = v
        elif k == "instruments":
            inst_delta = {}
            for inst_k, inst_v in v.items():
                if inst_k not in old[k]:
                    inst_delta[inst_k] = inst_v
                else:
                    inst_old = old[k][inst_k]
                    inst_diff = {}
                    for prop_k, prop_v in inst_v.items():
                        if prop_k not in inst_old or inst_old[prop_k] != prop_v:
                            inst_diff[prop_k] = prop_v
                    if inst_diff:
                        inst_delta[inst_k] = inst_diff
            if inst_delta:
                delta[k] = inst_delta
        elif k == "trades":
            trade_delta = {}
            for tk, tv in v.items():
                if tk not in old[k] or old[k][tk] != tv:
                    trade_delta[tk] = tv
            if trade_delta:
                delta[k] = trade_delta
        elif old[k] != v:
            delta[k] = v
            
    return delta if delta else None


def make_serializable(obj: Any) -> Any:
    """High-performance serialization for large trading data"""
    # Most snapshot nodes are builtins. Avoid a NumPy item-attribute probe and
    # repeated isinstance dispatch for every price, key, row and nested context.
    obj_type = type(obj)
    if obj_type is str or obj_type is int or obj_type is bool or obj is None:
        return obj
    if obj_type is float:
        return obj if math.isfinite(obj) else 0
    if obj_type is dict:
        return {str(k): make_serializable(v) for k, v in list(obj.items())}
    if obj_type is list:
        return [make_serializable(i) for i in list(obj)]
    if hasattr(obj, "item") and callable(getattr(obj, "item", None)):
        try:
            return make_serializable(obj.item())
        except Exception:
            pass
    if isinstance(obj, (int, str, bool, type(None))):
        return obj
    if isinstance(obj, Decimal):
        number = float(obj)
        return number if math.isfinite(number) else 0
    if isinstance(obj, float):
        if obj != obj or obj == float('inf') or obj == float('-inf'): return 0
        return obj
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in list(obj.items())}
    if isinstance(obj, (list, tuple, set, deque)):
        return [make_serializable(i) for i in list(obj)]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _frame_bytes(safe_obj: Any) -> bytes:
    """Encode a payload that has already been through ``make_serializable``."""
    if orjson is not None:
        return orjson.dumps(safe_obj)
    return json.dumps(safe_obj, default=str, separators=(",", ":")).encode("utf-8")


def _encode_json_bytes(payload: Dict[str, Any]) -> bytes:
    return _frame_bytes(make_serializable(payload))


def _api_state_json_response(payload: Dict[str, Any], *, lightened: bool = False) -> Response:
    """Serve a pre-lightened JSON response without FastAPI re-walking it."""
    light_payload = payload if lightened else _lighten_state_payload(payload)
    return Response(content=_encode_json_bytes(light_payload), media_type="application/json")


def _prepare_api_state_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Window before conversion; detach the lab table before its active-row edit."""
    light = _lighten_state_payload(_historical_building_state(payload))
    if isinstance(light.get("engine_lab"), dict) and hasattr(scanner_ref, "_sync_engine_lab_active_row"):
        light["engine_lab"] = make_serializable(light["engine_lab"])
    return light


def _trim_chart_payload(payload: Dict[str, Any], requested_bars: int) -> Dict[str, Any]:
    """Trim a single timeframe chart payload to the requested visible window."""
    trimmed = dict(payload)

    for key in ("candles", "trailing_stop"):
        value = trimmed.get(key)
        if isinstance(value, list) and len(value) > requested_bars:
            trimmed[key] = value[-requested_bars:]

    candles = trimmed.get("candles")
    visible_start = None
    visible_end = None
    if isinstance(candles, list) and candles:
        try:
            visible_start = float(candles[0].get("time"))
            visible_end = float(candles[-1].get("time"))
        except (AttributeError, TypeError, ValueError):
            visible_start = visible_end = None

    def trim_points(points):
        if not isinstance(points, list):
            return points
        visible = points
        if visible_start is not None and visible_end is not None:
            filtered = []
            for point in points:
                try:
                    point_time = float(point.get("time"))
                except (AttributeError, TypeError, ValueError):
                    continue
                if visible_start <= point_time <= visible_end:
                    filtered.append(point)
            visible = filtered
        if len(visible) > requested_bars:
            visible = visible[-requested_bars:]
        return visible

    markers = trimmed.get("markers")
    if isinstance(markers, list):
        trimmed["markers"] = trim_points(markers)

    overlays = trimmed.get("indicator_overlays")
    if isinstance(overlays, list):
        trimmed_overlays = []
        for overlay in overlays:
            if isinstance(overlay, dict):
                overlay = dict(overlay)
                overlay["data"] = trim_points(overlay.get("data"))
            trimmed_overlays.append(overlay)
        trimmed["indicator_overlays"] = trimmed_overlays

    intel_history = trimmed.get("intel_history")
    if isinstance(intel_history, dict):
        trimmed_intel = dict(intel_history)
        for key, value in list(trimmed_intel.items()):
            if isinstance(value, list):
                trimmed_intel[key] = trim_points(value)
        trimmed["intel_history"] = trimmed_intel

    return trimmed


@app.get("/api/status")
async def api_status():
    if scanner_ref and hasattr(scanner_ref, "dashboard_cache"):
        return scanner_ref.dashboard_cache.status(scanner_ref, len(active_connections))
    return {"status": "starting", "connected_clients": len(active_connections), "pid": os.getpid()}


@app.get("/api/state")
async def api_state():
    if not scanner_ref:
        return {"status": "starting"}
    if hasattr(scanner_ref, "dashboard_cache"):
        cached_state = scanner_ref.dashboard_cache.state()
        if cached_state and _payload_matches_scanner_mode(cached_state):
            payload = _prepare_api_state_payload(cached_state)
            if not getattr(scanner_ref, "chart_stream_enabled", True) and payload.get("instruments"):
                instruments = {}
                for name, ui_data in payload.get("instruments", {}).items():
                    if isinstance(ui_data, dict):
                        hydrated_ui = dict(ui_data)
                        hydrated_ui["chart"] = {}
                        instruments[name] = hydrated_ui
                    else:
                        instruments[name] = ui_data
                payload["instruments"] = instruments
            if hasattr(scanner_ref, "_sync_engine_lab_active_row"):
                scanner_ref._sync_engine_lab_active_row(
                    payload,
                    ((payload.get("trades") or {}).get("summary") or {}),
                )
            payload["dashboard_heartbeat"] = _dashboard_heartbeat()
            return _api_state_json_response(payload, lightened=True)
    if not getattr(scanner_ref, "chart_stream_enabled", True):
        payload = _prepare_api_state_payload(
            scanner_ref.get_latest_results(include_full_charts=False) or {}
        )
        if hasattr(scanner_ref, "_sync_engine_lab_active_row"):
            scanner_ref._sync_engine_lab_active_row(
                payload,
                ((payload.get("trades") or {}).get("summary") or {}),
            )
        payload["dashboard_heartbeat"] = _dashboard_heartbeat()
        return _api_state_json_response(payload, lightened=True)
    if dashboard_state.latest_safe_state and _payload_matches_scanner_mode(dashboard_state.latest_safe_state):
        payload = _prepare_api_state_payload(dashboard_state.latest_safe_state)
        if hasattr(scanner_ref, "_sync_engine_lab_active_row"):
            scanner_ref._sync_engine_lab_active_row(
                payload,
                ((payload.get("trades") or {}).get("summary") or {}),
            )
        payload["dashboard_heartbeat"] = _dashboard_heartbeat()
        return _api_state_json_response(payload, lightened=True)
    payload = _prepare_api_state_payload(
        scanner_ref.get_latest_results(include_full_charts=False) or {}
    )
    if hasattr(scanner_ref, "_sync_engine_lab_active_row"):
        scanner_ref._sync_engine_lab_active_row(
            payload,
            ((payload.get("trades") or {}).get("summary") or {}),
        )
    payload["dashboard_heartbeat"] = _dashboard_heartbeat()
    return _api_state_json_response(payload, lightened=True)


_DASHBOARD_QUOTE_FIELDS = {
    "ltp", "spot_price", "change", "change_points", "change_pct", "prev_close",
    "quote_source", "quote_label", "quote_session", "session", "entry_eligible",
    "quote_fresh", "quote_stale", "quote_age_seconds", "receipt_age_seconds",
    "exchange_age_seconds", "received_at", "received_at_epoch", "exchange_timestamp",
    "exchange_timestamp_epoch", "preopen_settlement", "selected_price_source",
}


def _dashboard_index_quote_snapshot() -> Dict[str, Any]:
    """Read four index quotes straight from the provider's in-memory WS cache.

    This deliberately never calls a broker REST method or the scanner event loop. The
    full scan can spend tens of seconds in intelligence/chain work; the badge path must
    remain bounded to a few lock-protected dictionary reads.
    """
    now = datetime.now()
    payload: Dict[str, Any] = {
        "status": "ok",
        "quote_snapshot_at": now.isoformat(),
        "quote_snapshot_epoch": time.time(),
        "quote_lane": "provider_cache",
        "instruments": {},
    }
    scanner = scanner_ref
    if scanner is None:
        payload["status"] = "starting"
        return payload

    cached_state: Dict[str, Any] = {}
    try:
        if hasattr(scanner, "dashboard_cache"):
            cached_state = scanner.dashboard_cache.state() or {}
    except Exception:
        cached_state = {}
    for key in ("market_status", "market_status_detail", "mode", "system_power"):
        if key in cached_state:
            payload[key] = cached_state[key]

    configured = list(getattr(scanner, "active_indices", []) or [])
    indices = [str(item or "").strip().upper() for item in configured if str(item or "").strip()]
    if not indices:
        indices = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]

    provider = getattr(scanner, "data", None)
    getter = getattr(provider, "get_cached_index_quote_snapshot", None)
    fallback_instruments = cached_state.get("instruments") if isinstance(cached_state, dict) else {}
    for instrument in indices:
        quote = None
        if callable(getter):
            try:
                quote = getter(instrument)
            except Exception as exc:
                logger.debug(f"Cache-only dashboard quote failed for {instrument}: {exc}")
        if not isinstance(quote, dict):
            quote = (fallback_instruments or {}).get(instrument) if isinstance(fallback_instruments, dict) else None
            payload["quote_lane"] = "scan_cache_fallback"
        if not isinstance(quote, dict):
            continue
        compact = {key: quote[key] for key in _DASHBOARD_QUOTE_FIELDS if key in quote}
        try:
            quote_ltp = float(compact.get("ltp") or compact.get("spot_price") or 0.0)
        except (TypeError, ValueError):
            quote_ltp = 0.0
        if quote_ltp <= 0:
            continue
        compact.setdefault("spot_price", compact.get("ltp"))
        compact.setdefault("ltp", compact.get("spot_price"))
        payload["instruments"][instrument] = compact
    return payload


@app.get("/api/quotes")
async def api_quotes():
    return make_serializable(_dashboard_index_quote_snapshot())


@app.get("/api/diagnostics")
async def api_diagnostics():
    if not scanner_ref:
        return {"status": "starting", "diagnostics": {}}
    if _scanner_is_historical_building():
        return make_serializable(scanner_ref._get_diagnostics_payload())
    if hasattr(scanner_ref, "dashboard_cache"):
        cached = scanner_ref.dashboard_cache.diagnostics()
        if cached:
            return cached
    return make_serializable(scanner_ref._get_diagnostics_payload())


_VOLUME_HEALTH_TTL_SECONDS = 10.0
_volume_health_cache = None


@app.get("/api/volume-health")
async def api_volume_health():
    """Which feed is actually supplying each index's volume, right now.

    Index volume died on 2026-07-27 and stayed dead for days: every volume-gated path
    (the 2.5x surge that is the only 1min entry route, the hybrid thrusts, the cascade,
    and the fast-opposite thrust EXIT guard) reads one ratio, and all of them evaluated
    False in silence. The futures fallback now keeps trading alive when Fyers goes
    quiet — which also means the outage no longer announces itself. This endpoint is
    how it stays visible.
    """
    if not scanner_ref:
        return {"status": "starting", "healthy": None, "instruments": {}}
    if not hasattr(scanner_ref, "volume_source_health"):
        return {"status": "unsupported", "healthy": None, "instruments": {}}
    # This reaches the broker (a Kite quote per index, plus a possible Fyers REST probe)
    # on the same API key that places orders. Two guards: serve a short-lived cache so a
    # dashboard poller cannot turn into an unthrottled quote loop, and run the blocking
    # calls off the event loop so a slow broker cannot stall every other route and the
    # websocket alongside it.
    global _volume_health_cache
    now = time.time()
    cached = _volume_health_cache
    if cached and now - cached[0] < _VOLUME_HEALTH_TTL_SECONDS:
        return cached[1]
    try:
        payload = make_serializable(
            await asyncio.to_thread(scanner_ref.volume_source_health)
        )
    except Exception as exc:
        return {"status": "error", "healthy": None, "instruments": {}, "error": str(exc)}
    _volume_health_cache = (now, payload)
    return payload


@app.get("/api/calibration-status")
async def api_calibration_status():
    if not scanner_ref or not hasattr(scanner_ref, "outcome_calibration"):
        return {"status": "starting", "calibration": {}}
    return {
        "status": "ok",
        "calibration": make_serializable(
            scanner_ref.outcome_calibration.validation_status()
        ),
    }


@app.get("/api/signal-report")
async def api_signal_report(date: str | None = None):
    if not scanner_ref:
        return {"status": "starting", "report": {}}
    loop = asyncio.get_running_loop()
    report = await loop.run_in_executor(
        None,
        lambda: scanner_ref.generate_daily_signal_report(date),
    )
    return make_serializable({"status": "ok", "report": report})


# ── Signal Forensics / Flight Recorder ────────────────────────────────────

_FORENSIC_PHASES = {
    "DETECTED", "STABILIZING", "ACCEPTED", "REJECTED", "DISCARDED",
    "ORDER_PENDING", "ORDER_SUBMITTED", "ORDER_CANCELLED", "ORDER_REJECTED",
    "FILLED", "ENTERED", "STOP_MOVED", "TARGET_MOVED", "TRAILING",
    "MARK_TO_MARKET", "PARTIAL_EXIT", "EXITED", "TIMED_OUT", "MANAGED",
    "PARTIAL",
}
_FORENSIC_DECISIONS = {
    "PENDING", "ACCEPTED", "REJECTED", "SUBMITTED", "FILLED", "OPEN",
    "PARTIAL_EXIT", "CLOSED", "CANCELLED", "FAILED",
}


def _forensic_date(raw):
    """Validate and clean a date string, falling back to latest available."""
    if not raw:
        return None
    import re as _re
    cleaned = str(raw).strip()
    return cleaned if _re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned) else None


# Identifiers that only ever appear in fixtures. The flight-recorder table shipped with
# a single hand-written row -- lifecycle_id "LFC-TEST-12345", signal_id "SIG-TEST-999",
# contract "NIFTY24AUG24500CE", grade A+, pnl +500 -- dated to a real session. Because
# the endpoint falls back to the most recent available date when none is requested, and
# that fixture was the ONLY date present, the panel's default view rendered an invented
# trade as a real lifecycle. A forensics surface that shows fabricated history is worse
# than one that shows nothing, so these are dropped at the read and the panel is allowed
# to be honestly empty.
_FORENSIC_SYNTHETIC_MARKERS = ("TEST-", "-TEST", "FIXTURE", "DUMMY", "SAMPLE-", "LFC-TEST")


def _is_synthetic_forensic_row(row) -> bool:
    if not isinstance(row, dict):
        return False
    for field in ("lifecycle_id", "signal_id", "trade_id"):
        value = str(row.get(field) or "").upper()
        if value and any(marker in value for marker in _FORENSIC_SYNTHETIC_MARKERS):
            return True
    return False


def _drop_synthetic_forensic_rows(result):
    """Strip fixture rows from a query result, in place, and report how many went."""
    if not isinstance(result, dict):
        return result
    items = result.get("items")
    if isinstance(items, list):
        kept = [row for row in items if not _is_synthetic_forensic_row(row)]
        dropped = len(items) - len(kept)
        if dropped:
            result["items"] = kept
            result["synthetic_rows_hidden"] = dropped
    return result


@app.get("/api/signal-forensics")
async def api_signal_forensics(
    date: str | None = None,
    instrument: str | None = None,
    timeframe: str | None = None,
    phase: str | None = None,
    decision: str | None = None,
    search: str | None = None,
    cursor: str | None = None,
    limit: int = 100,
):
    """Query recent signal lifecycle forensic events for session flight recorder."""
    from data.sqlite_db import db
    cleaned_date = _forensic_date(date)
    if not cleaned_date:
        try:
            available = await asyncio.to_thread(db.list_signal_forensics_dates)
            cleaned_date = available[0] if available else None
        except Exception:
            pass
    inst = str(instrument).upper().strip() if instrument else None
    tf = str(timeframe).lower().strip() if timeframe else None
    phase_val = (
        str(phase).upper().strip()
        if phase and str(phase).upper().strip() in _FORENSIC_PHASES
        else None
    )
    decision_val = (
        str(decision).upper().strip()
        if decision and str(decision).upper().strip() in _FORENSIC_DECISIONS
        else None
    )
    clamped_limit = max(1, min(int(limit or 100), 500))

    try:
        result = await asyncio.to_thread(
            db.query_signal_lifecycle_events,
            session_date=cleaned_date,
            cursor=cursor,
            limit=clamped_limit,
            instrument=inst,
            timeframe=tf,
            phase=phase_val,
            decision=decision_val,
            search=search,
        )
        available_dates = await asyncio.to_thread(db.list_signal_forensics_dates)
    except Exception as exc:
        logger.error(f"Signal forensics query failed: {exc}")
        return {
            "status": "error",
            "message": "Failed to load signal forensics",
            "items": [],
            "available_dates": [],
        }

    result = _drop_synthetic_forensic_rows(result)
    result["status"] = "ok"
    result["session_date"] = cleaned_date or (available_dates[0] if available_dates else None)
    result["available_dates"] = available_dates
    return result


@app.get("/api/signal-forensics/{lifecycle_id}")
async def api_signal_forensics_lifecycle(lifecycle_id: str):
    """Retrieve full chronological flight-recorder timeline for a single signal lifecycle."""
    from data.sqlite_db import db
    clean_id = str(lifecycle_id).strip()
    if not clean_id or len(clean_id) > 128:
        return {"status": "error", "message": "Invalid lifecycle_id", "events": []}

    try:
        events = await asyncio.to_thread(db.load_signal_lifecycle, clean_id)
    except Exception as exc:
        logger.error(f"Signal lifecycle detail load failed for {clean_id}: {exc}")
        return {
            "status": "error",
            "message": "Failed to load signal lifecycle",
            "events": [],
        }

    if not events:
        return {
            "status": "error",
            "message": "Lifecycle record not found",
            "events": [],
        }

    return {
        "status": "ok",
        "lifecycle_id": clean_id,
        "count": len(events),
        "events": events,
    }


@app.get("/api/chart")
async def api_chart(instrument: str = "NIFTY", tf: str = "5min", bars: int | None = None):
    if not scanner_ref:
        return {"status": "starting", "instrument": instrument, "tf": tf, "chart": {}}
    settings = get_settings()
    max_bars = max(1, int(getattr(settings, "api_max_chart_bars", 1000) or 1000))
    default_bars = max(1, int(getattr(settings, "api_default_chart_bars", 300) or 300))
    requested_bars = max(1, min(max_bars, int(bars or default_bars)))
    snapshot = await _build_chart_off_loop(instrument, tf)
    chart = snapshot.get("chart") if isinstance(snapshot, dict) else None
    if isinstance(chart, dict):
        for tf_key, payload in list(chart.items()):
            if not isinstance(payload, dict):
                continue
            chart[tf_key] = _trim_chart_payload(payload, requested_bars)
        snapshot["window"] = {"bars": requested_bars, "max_bars": max_bars}
    return make_serializable(snapshot)


@app.post("/api/reset_cache")
async def api_reset_cache(request: Request):
    if not _request_authorized(request):
        return {"status": "error", "message": "Unauthorized"}
    if not scanner_ref:
        return {"status": "error", "message": "Scanner not initialized"}
    if hasattr(scanner_ref, "queue_full_recalculation"):
        await _call_on_scanner_loop(
            lambda: scanner_ref.queue_full_recalculation("api-reset-cache")
        )
    else:
        if not _schedule_on_scanner_loop(scanner_ref._perform_full_recalculation()):
            return {"status": "error", "message": "Unable to schedule cache reset"}
    return {"status": "ok", "message": "Cache reset initiated"}


@app.post("/api/recalibrate")
async def api_recalibrate(request: Request):
    if not _request_authorized(request):
        return {"status": "error", "message": "Unauthorized"}
    if not scanner_ref:
        return {"status": "error", "message": "Scanner not initialized"}
    if not _schedule_on_scanner_loop(scanner_ref.perform_system_recalibration()):
        return {"status": "error", "message": "Unable to schedule recalibration"}
    return {"status": "ok", "message": "System recalibration initiated"}


@app.get("/api/upstox_auth_url")
async def api_upstox_auth_url(request: Request):
    """Return the OAuth login URL for Upstox."""
    if not (_request_authorized(request) or _is_loopback_client(request)):
        return {"status": "error", "message": "Unauthorized"}
    from data.upstox_auth import get_upstox_auth_url
    try:
        return {"status": "ok", "auth_url": get_upstox_auth_url()}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.get("/api/upstox_login")
@app.get("/api/upstox/callback")
async def api_upstox_login(request: Request):
    """Handle Upstox browser login or OAuth code callback."""
    code = str(request.query_params.get("code") or "").strip()
    from data.upstox_auth import get_upstox_auth_url, exchange_code_for_token
    if not code:
        try:
            return RedirectResponse(get_upstox_auth_url())
        except Exception as exc:
            return HTMLResponse(f"Cannot start Upstox login: {exc}", status_code=500)

    try:
        res = exchange_code_for_token(code)
        if res.get("status") == "success":
            return HTMLResponse(
                f"<h3>✅ Upstox Login Successful</h3><p>Connected as {res.get('user_id', 'User')}. Token refreshed for 24h.</p>"
                f"<script>window.opener && window.opener.postMessage('upstox_auth_complete', '*'); setTimeout(() => window.close(), 2500);</script>"
            )
        else:
            return HTMLResponse(
                f"<h3>❌ Upstox Login Failed</h3><p>{res.get('message')}</p>",
                status_code=400,
            )
    except Exception as exc:
        logger.exception(f"Upstox login callback failed: {exc}")
        return HTMLResponse(f"<h3>Upstox login failed</h3><p>{exc}</p>", status_code=500)


@app.post("/api/upstox_auth")
async def api_upstox_auth(request: Request):
    """Update or exchange Upstox token manually or via code."""
    if not (_request_authorized(request) or _is_loopback_client(request)):
        return {"status": "error", "message": "Unauthorized"}
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    code = str(payload.get("code") or payload.get("auth_code") or "").strip()
    token = str(payload.get("token") or payload.get("access_token") or "").strip()
    from data.upstox_auth import exchange_code_for_token, save_upstox_token, get_upstox_auth_status
    if token:
        try:
            save_upstox_token(token)
            return {"status": "ok", "message": "Upstox token saved", "auth": get_upstox_auth_status()}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
    if code:
        try:
            res = exchange_code_for_token(code)
            if res.get("status") == "error":
                return {"status": "error", "message": res.get("message", "Token exchange failed")}
            return {"status": "ok", "data": res, "auth": get_upstox_auth_status()}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
    return {"status": "error", "message": "No code or token provided"}


@app.get("/api/upstox_status")
async def api_upstox_status(request: Request):
    """Return Upstox session and token validity status."""
    if not (_request_authorized(request) or _is_loopback_client(request)):
        return {"status": "error", "message": "Unauthorized"}
    from data.upstox_auth import get_upstox_auth_status
    return make_serializable(get_upstox_auth_status())


@app.get("/api/broker_status")
async def api_broker_status(request: Request):
    """Show independent market-data and execution readiness."""
    if not (_request_authorized(request) or _is_loopback_client(request)):
        return {"status": "error", "message": "Unauthorized"}
    broker = getattr(getattr(scanner_ref, "trades", None), "real_broker", None)
    reader = getattr(broker, "execution_status", None)
    from data.upstox_auth import get_upstox_auth_status
    return {
        "status": "ok",
        "market_data_provider": "upstox",
        "execution": reader() if callable(reader) else {"provider": "angelone", "connected": False},
        "upstox_auth": get_upstox_auth_status(),
    }


@app.post("/api/upstox_postback")
async def api_upstox_postback(request: Request):
    """Upstox is the market-data source in this runtime."""
    return {"status": "ignored", "reason": "Upstox order updates are disabled"}


@app.get("/api/fyers_auth")
async def api_fyers_auth_start(request: Request):
    return {"status": "deprecated", "message": "Fyers removed; use /api/upstox_auth"}


@app.get("/api/fyers_login")
async def api_fyers_login_redirect(request: Request):
    return RedirectResponse("/api/upstox_login")

def _is_loopback_client(request: Request) -> bool:
    """True when the request came from this machine.

    The Kite login legs are browser NAVIGATIONS, not XHR: the first is a new tab and
    the second is Zerodha redirecting back. Neither can carry the x-dashboard-token
    header that _request_authorized() requires, so header auth would reject the login
    permanently. Loopback is the boundary that actually applies here — the dashboard
    binds 127.0.0.1, and a request_token is single-use, short-lived, and worthless
    without the api_secret held in this process.
    """
    # A reverse proxy or tunnel connects to this app FROM localhost, so client.host
    # alone would call a request from the public internet "local" and hand a stranger
    # the Kite login flow. Any forwarding header means the request was relayed, and a
    # relayed request is not local no matter what the socket says.
    for header in ("x-forwarded-for", "x-forwarded-host", "x-real-ip", "cf-connecting-ip",
                   "forwarded", "cf-ray"):
        if request.headers.get(header):
            return False
    host = getattr(getattr(request, "client", None), "host", "") or ""
    return host in {"127.0.0.1", "::1", "localhost"}


@app.get("/api/kite_auth_url")
async def api_kite_auth_url(request: Request):
    """Authenticated: hand the dashboard the Kite login URL to open."""
    if getattr(getattr(scanner_ref, "data", None), "upstox_only", False):
        return {"status": "disabled", "message": "Market data uses Upstox"}
    if not _request_authorized(request):
        return {"status": "error", "message": "Unauthorized"}
    from data.kite_auth import get_kite_auth

    try:
        return {"status": "ok", "auth_url": get_kite_auth().login_url()}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.get("/api/kite_login")
async def api_kite_login(request: Request):
    """Both halves of the Kite daily login.

    Zerodha redirects back to this same URL with ?request_token=..., which is why the
    app's registered redirect must point here. Called with no token it sends the
    operator to Kite; called with one it completes the exchange.

    Kite has no password-based API login, so no account password is accepted, stored,
    or transmitted anywhere in this flow.
    """
    if getattr(getattr(scanner_ref, "data", None), "upstox_only", False):
        return HTMLResponse("Kite login is disabled. Market data uses Upstox.", status_code=410)
    if not _is_loopback_client(request):
        return HTMLResponse("Unauthorized", status_code=401)
    from data.kite_auth import get_kite_auth

    auth = get_kite_auth()
    request_token = str(request.query_params.get("request_token") or "").strip()
    kite_status = str(request.query_params.get("status") or "").strip()

    if not request_token:
        if kite_status and kite_status.lower() != "success":
            return HTMLResponse(
                f"<h3>Kite login was not completed</h3><p>Zerodha reported: {kite_status}</p>",
                status_code=400,
            )
        try:
            return RedirectResponse(auth.login_url())
        except Exception as exc:
            return HTMLResponse(f"Cannot start the Kite login: {exc}", status_code=500)

    try:
        result = auth.complete_login(request_token)
    except Exception as exc:
        logger.exception(f"Kite login exchange failed: {exc}")
        return HTMLResponse(
            f"<h3>Kite login failed</h3><p>{exc}</p>"
            "<p>A request_token is single-use and expires within minutes — "
            "<a href='/api/kite_login'>start again</a>.</p>",
            status_code=500,
        )

    # Hand the live session to anything already constructed with the old one.
    try:
        if scanner_ref is not None:
            # rebind_kite() rebuilds the data source, contract master, tick socket AND
            # the broker. Calling on_kite_reauth directly only rebound the broker, so
            # quotes and ticks kept using the dead token after a re-login.
            data_provider = getattr(scanner_ref, "data", None)
            rebind = getattr(data_provider, "rebind_kite", None)
            if callable(rebind):
                rebind(auth.kite)
            else:
                fallback = getattr(data_provider, "on_kite_reauth", None)
                if callable(fallback):
                    fallback(auth.kite)
    except Exception as exc:
        logger.warning(f"Kite session rebind after login failed: {exc}")

    return HTMLResponse(
        f"<h3>Kite session active</h3><p>Logged in as {result.get('user_id') or 'user'}."
        "</p><p>This token expires tomorrow morning; log in again before the next "
        "session. You can close this tab.</p>"
    )


@app.get("/api/kite_postback")
async def api_kite_postback_probe():
    """Liveness probe for URL validators. Exposes nothing and accepts nothing."""
    return {"status": "ok", "endpoint": "kite_postback"}


@app.post("/api/kite_postback")
async def api_kite_postback(request: Request):
    """Zerodha order postback: authoritative order state, pushed not polled.

    Kite POSTs here on every order state change. That settles the exact question the
    2026-07-31 incident could not answer — whether an order reached the exchange —
    without the order-book scan heuristic, and without waiting for the next poll.

    Authenticity is established by the checksum Zerodha sends: SHA-256 of
    order_id + order_timestamp + api_secret. An unsigned or mismatched payload is
    ignored, because this endpoint must be reachable by Zerodha and therefore cannot
    sit behind the dashboard's header token.
    """
    if getattr(getattr(scanner_ref, "data", None), "upstox_only", False):
        return {"status": "ignored", "reason": "Angel One owns order execution"}
    import hashlib

    from fastapi.responses import JSONResponse

    try:
        payload = await request.json()
    except Exception:
        return {"status": "error", "message": "malformed payload"}
    if not isinstance(payload, dict):
        return {"status": "error", "message": "malformed payload"}

    from data.kite_auth import get_kite_auth

    auth = get_kite_auth()
    secret = getattr(auth, "_KiteAuth__api_secret", None) or getattr(auth, "_api_secret", "")
    order_id = str(payload.get("order_id") or "")
    stamp = str(payload.get("order_timestamp") or "")
    received = str(payload.get("checksum") or "")
    # With no api_secret the "expected" checksum is sha256(order_id + timestamp + "") —
    # a value anyone who can reach this route can compute, so every forged postback
    # would verify. Refuse to serve the route at all rather than authenticate against
    # a secret we do not have (2026-08-16). 503, because this is a server-side
    # misconfiguration and not a bad request from Zerodha.
    if not secret:
        logger.error(
            "Kite postback refused: no api_secret is configured, so no payload can be "
            "authenticated. The route stays closed until a session exists."
        )
        return JSONResponse(
            {"status": "error", "message": "postback not configured"}, status_code=503
        )
    expected = hashlib.sha256(f"{order_id}{stamp}{secret}".encode("utf-8")).hexdigest()
    # compare_digest, not != : a plain comparison short-circuits on the first differing
    # byte and leaks the prefix length through timing. This route is public by
    # necessity, which is exactly the condition that makes that worth closing. The
    # dashboard's own token check at server.py:353 already uses compare_digest.
    if not received or not secrets.compare_digest(received, expected):
        # Never act on an unverified payload: this route is public by necessity, so a
        # forged "COMPLETE" could convince the system a phantom position exists.
        logger.warning(f"Rejected Kite postback with bad checksum for order {order_id}")
        # 403, not a 200 carrying an error body: a 200 tells the sender it succeeded,
        # and it matches what the standalone listener returns for the same payload.
        return JSONResponse({"status": "error", "message": "checksum mismatch"}, status_code=403)

    status = str(payload.get("status") or "").upper()
    logger.info(
        f"Kite postback: order {order_id} -> {status} "
        f"filled={payload.get('filled_quantity')} avg={payload.get('average_price')}"
    )
    try:
        if scanner_ref is not None:
            handler = getattr(getattr(scanner_ref, "trades", None), "on_broker_postback", None)
            if callable(handler):
                handler(dict(payload))
    except Exception as exc:
        logger.exception(f"Kite postback handling failed for {order_id}: {exc}")
    return {"status": "ok"}


@app.get("/api/kite_status")
async def api_kite_status(request: Request):
    if not (_request_authorized(request) or _is_loopback_client(request)):
        return {"status": "error", "message": "Unauthorized"}
    from data.kite_auth import get_kite_auth

    return make_serializable(get_kite_auth().status())


@app.post("/api/fyers_auth")
async def api_fyers_auth_complete(request: Request):
    if not _request_authorized(request):
        return {"status": "error", "message": "Unauthorized"}
    if not scanner_ref:
        return {"status": "error", "message": "Scanner is still starting. Try again in a few seconds."}

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    auth_code = payload.get("auth_code", "")
    try:
        result = await _await_on_scanner_loop(
            scanner_ref.update_fyers_token_async(auth_code),
            timeout_seconds=30.0,
        )
        return make_serializable(result)
    except Exception as exc:
        logger.exception(f"Fyers token update failed: {exc}")
        return {"status": "error", "message": f"Fyers token update failed: {exc}"}


def _window_trades_payload(payload: Dict[str, Any], limit: int | None, offset: int = 0) -> Dict[str, Any]:
    settings = get_settings()
    max_limit = 2000
    default_limit = max(1, int(getattr(settings, "api_default_trade_limit", _HISTORICAL_TRADE_PAGE_SIZE) or _HISTORICAL_TRADE_PAGE_SIZE))
    resolved_limit = max(1, min(max_limit, int(limit or default_limit)))
    resolved_offset = max(0, int(offset or 0))
    windowed = dict(payload or {})
    row_sets = {key: list(windowed.get(key) or []) for key in ("closed", "signals")}
    summary_total = 0
    try:
        summary_total = int((windowed.get("summary") or {}).get("total_trades") or 0)
    except Exception:
        summary_total = 0
    total_rows = max(
        len(row_sets["closed"]),
        len(row_sets["signals"]),
        summary_total,
    )
    page_count = max(1, (total_rows + resolved_limit - 1) // resolved_limit)
    if total_rows > 0 and resolved_offset >= total_rows:
        resolved_offset = (page_count - 1) * resolved_limit
    for key, rows in row_sets.items():
        windowed[key] = rows[resolved_offset: resolved_offset + resolved_limit]
        windowed[f"{key}_total"] = len(rows)
    windowed["window"] = {
        "limit": resolved_limit,
        "offset": resolved_offset,
        "max_limit": max_limit,
        "page": (resolved_offset // resolved_limit) + 1,
        "page_size": resolved_limit,
        "page_count": page_count,
        "total": total_rows,
    }
    return windowed


def _limit_dashboard_trades(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Keep dashboard payloads small; paged history remains available via /api/trades."""
    windowed = dict(payload or {})
    trades = windowed.get("trades")
    if isinstance(trades, dict):
        limit = _HISTORICAL_TRADE_PAGE_SIZE if _scanner_is_historical() else _DASHBOARD_STATE_TRADE_LIMIT
        windowed["trades"] = _window_trades_payload(trades, limit, 0)
    return windowed


def _lighten_state_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove heavy chart arrays from state/WS payloads; /api/chart serves charts."""
    light = _limit_dashboard_trades(payload)
    # Equity curves have been removed to save resources.
    instruments = light.get("instruments")
    if not isinstance(instruments, dict):
        return light

    light_instruments: Dict[str, Any] = {}
    for name, ui_data in list(instruments.items()):
        if not isinstance(ui_data, dict):
            light_instruments[name] = ui_data
            continue

        ui_light = dict(ui_data)
        chart = ui_light.get("chart")
        if isinstance(chart, dict):
            chart_state = {}
            for tf, tf_payload in list(chart.items()):
                if isinstance(tf_payload, dict):
                    chart_state[tf] = {"state": tf_payload.get("state", {})}
            ui_light["chart"] = chart_state
        light_instruments[name] = ui_light

    light["instruments"] = light_instruments
    return light


@app.get("/api/replay-audit/{trade_id}")
async def api_replay_audit(trade_id: str):
    """Return the original contract and guard trace for one current replay row."""
    if not scanner_ref or str(getattr(scanner_ref, "mode", "")).upper() != "HISTORICAL":
        return {"status": "unavailable", "reason": "No historical replay is active"}
    if not trade_id or len(trade_id) > 160:
        return {"status": "error", "reason": "Invalid trade id"}

    def capture():
        # The request can wait behind a mode switch on the scanner-owned loop.
        if not _scanner_is_historical():
            return {"status": "unavailable", "reason": "No historical replay is active"}
        if _scanner_is_historical_building():
            return {"status": "building", "reason": "Historical replay is still calculating"}
        row = next((t for t in scanner_ref.trades.closed_trades if t.id == trade_id), None)
        if row is None or not getattr(row, "replay_exit_trace", None):
            return {"status": "unavailable", "reason": "No original replay trace is available for this row"}
        return make_serializable({
            "status": "ok", "simulation_id": scanner_ref.simulation_id,
            "runtime_build": scanner_ref._runtime_build_status(),
            "historical_intelligence_coverage": getattr(scanner_ref, "_historical_intel_coverage", {}),
            "trade": row.to_dict(),
        })

    return await _call_on_scanner_loop(capture)


@app.get("/api/trades")
async def api_trades(
    limit: int | None = None,
    offset: int = 0,
    page: int | None = None,
    page_size: int | None = None,
):
    if not scanner_ref:
        return {"open": [], "closed": [], "signals": [], "summary": {}}
    requested_limit = limit if limit is not None else page_size
    requested_offset = offset
    if page is not None and offset == 0:
        settings = get_settings()
        default_limit = max(
            1,
            int(
                getattr(settings, "api_default_trade_limit", _HISTORICAL_TRADE_PAGE_SIZE)
                or _HISTORICAL_TRADE_PAGE_SIZE
            ),
        )
        resolved_limit = max(1, min(2000, int(requested_limit or default_limit)))
        requested_offset = (max(1, int(page or 1)) - 1) * resolved_limit
    if (
        _scanner_is_historical()
        and _latest_full_trades_payload
        and not _trade_payload_is_windowed(_latest_full_trades_payload)
        and _cached_trades_match_current_context(_latest_full_trades_payload)
    ):
        return _window_trades_payload(_latest_full_trades_payload, requested_limit, requested_offset)
    if (
        _scanner_is_historical()
        and dashboard_state.latest_safe_state
        and _payload_matches_scanner_mode(dashboard_state.latest_safe_state)
        and isinstance(dashboard_state.latest_safe_state.get("trades"), dict)
        and not _trade_payload_is_windowed(dashboard_state.latest_safe_state["trades"])
        and _cached_trades_match_current_context(dashboard_state.latest_safe_state["trades"])
    ):
        return _window_trades_payload(dashboard_state.latest_safe_state["trades"], requested_limit, requested_offset)
    if _scanner_is_historical() and hasattr(scanner_ref, "dashboard_cache"):
        cached = scanner_ref.dashboard_cache.trades()
        if cached and not _trade_payload_is_windowed(cached) and _cached_trades_match_current_context(cached):
            return _window_trades_payload(cached, requested_limit, requested_offset)
    if _scanner_is_historical_building():
        return _window_trades_payload(_building_trades_payload(), requested_limit, requested_offset)
    trades = await _call_on_scanner_loop(build_trades_payload)
    return _window_trades_payload(trades, requested_limit, requested_offset)


# The trades heartbeat builds ON the scanner loop -- the loop that also runs the scan
# cycle and the trade monitor -- so it must never be able to monopolise it: at most one
# request in flight, and after each build it waits long enough that heartbeat builds
# take at most _PERIODIC_TRADES_MAX_LOOP_SHARE of that loop. A healthy ~50ms build keeps
# the 1s cadence; a contended 2s build refreshes every 8s. Entry/exit pokes still
# refresh immediately, and the scan cycle publishes trades on every cycle regardless.
_PERIODIC_TRADES_POLL_SECONDS = 1.0
_PERIODIC_TRADES_WAIT_SECONDS = 5.0
_PERIODIC_TRADES_MAX_LOOP_SHARE = 0.2
_PERIODIC_TRADES_MAX_PACING_SECONDS = 30.0
_PERIODIC_TRADES_WARNING_INTERVAL_SECONDS = 60.0


def _timed_trades_build() -> tuple[Dict[str, Any], float]:
    """Build the trades payload, reporting how long it held the scanner loop."""
    started = time.perf_counter()
    trades = build_trades_payload()
    return trades, time.perf_counter() - started


async def periodic_trades_updater():
    """Low-cost heartbeat so live open/closed trade rows stay fresh between scans.

    On 2026-10-06 this heartbeat held the scanner loop 96% of the time and stretched a
    scan cycle to 167s: each build overran the 5s wait and was thrown away, and 1s later
    the next request queued behind the one still running, so builds ran back to back.
    A late result is now awaited and used instead of being replaced by another request.
    """
    global _last_periodic_trades_hash, _trade_dirty_event, _trade_dirty_loop
    logger.info("Periodic trades updater background task started (change-driven loop)")
    _trade_dirty_event = asyncio.Event()
    _trade_dirty_loop = asyncio.get_running_loop()
    clock = _trade_dirty_loop.time
    request: Optional[asyncio.Future] = None
    request_loop = None
    requested_at = 0.0
    overdue = False
    hang_reported = False
    next_refresh_at = 0.0
    last_slow_report_at: Optional[float] = None
    unreported_slow = 0

    while True:
        try:
            if request is None:
                # Wake immediately on an entry/exit poke, else at the paced poll.
                try:
                    await asyncio.wait_for(
                        _trade_dirty_event.wait(),
                        timeout=max(_PERIODIC_TRADES_POLL_SECONDS, next_refresh_at - clock()),
                    )
                except asyncio.TimeoutError:
                    pass
                _trade_dirty_event.clear()
                if not (scanner_ref and active_connections):
                    continue
                request_loop = getattr(scanner_ref, "_event_loop", None)
                request = _start_on_scanner_loop(_timed_trades_build)
                requested_at = clock()
                overdue = hang_reported = False
            try:
                # Shielded: running out of patience must not cancel the request.
                trades, build_seconds = await asyncio.wait_for(
                    asyncio.shield(request),
                    timeout=_PERIODIC_TRADES_WAIT_SECONDS,
                )
            except asyncio.TimeoutError:
                overdue = True
                waited = clock() - requested_at
                if request_loop is not None and (
                    not request_loop.is_running()
                    or getattr(scanner_ref, "_event_loop", None) is not request_loop
                ):
                    # That loop has stopped or been replaced; its request can never finish.
                    try:
                        request.cancel()
                    except Exception:
                        pass
                    request = None
                elif not hang_reported and waited >= _PERIODIC_TRADES_WARNING_INTERVAL_SECONDS:
                    hang_reported = True
                    logger.warning(
                        f"Periodic trades updater: still waiting after {waited:.0f}s for the "
                        "scanner loop to serve one trades payload; no further request is queued"
                    )
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                # The build itself failed: report it below and start a fresh request.
                request = None
                raise
            request = None
            pacing = min(
                _PERIODIC_TRADES_MAX_PACING_SECONDS,
                build_seconds * (1.0 / _PERIODIC_TRADES_MAX_LOOP_SHARE - 1.0),
            )
            next_refresh_at = clock() + pacing
            if overdue:
                # One line per minute at most; the old "TimeoutError: (no message)" ERROR
                # fired every ~6s and said neither how long the loop was held nor by what.
                now = clock()
                if (
                    last_slow_report_at is None
                    or now - last_slow_report_at >= _PERIODIC_TRADES_WARNING_INTERVAL_SECONDS
                ):
                    since = f"; {unreported_slow} more since the last report" if unreported_slow else ""
                    logger.warning(
                        f"Periodic trades updater: the scanner loop took {now - requested_at:.1f}s "
                        f"to serve the trades payload ({build_seconds:.2f}s of it building); next "
                        f"unprompted refresh in {max(pacing, _PERIODIC_TRADES_POLL_SECONDS):.1f}s{since}"
                    )
                    last_slow_report_at = now
                    unreported_slow = 0
                else:
                    unreported_slow += 1
            limit =_HISTORICAL_TRADE_PAGE_SIZE if _scanner_is_historical() else _DASHBOARD_STATE_TRADE_LIMIT
            trades = _window_trades_payload(trades, limit, 0)
            trade_hash = hashlib.blake2b(
                json.dumps(make_serializable(trades), sort_keys=True, default=str).encode("utf-8"),
                digest_size=12,
            ).hexdigest()
            if trade_hash == _last_periodic_trades_hash:
                continue
            _last_periodic_trades_hash = trade_hash
            await broadcast({
                "trades": trades,
                "trade_update_id": trade_hash,
                "timestamp": datetime.now().isoformat(),
            })
        except asyncio.CancelledError:
            if request is not None:
                try:
                    request.cancel()
                except Exception:
                    pass
            break
        except Exception as exc:
            # Include the type: some exceptions have an empty str(). A slow scanner loop
            # no longer lands here (see the overdue handling above); a failed build or a
            # stopped scanner loop does.
            logger.error(
                # str(exc), not exc: an exception instance is truthy even when its
                # message is empty, so "exc or ..." always kept the exception and the
                # fallback never rendered -- the live log still read "TimeoutError:"
                # with nothing after the colon.
                f"Error in periodic trades updater: {type(exc).__name__}: "
                f"{str(exc) or '(no message)'}"
            )


def _dashboard_heartbeat() -> Dict[str, Any]:
    now = datetime.now()
    heartbeat: Dict[str, Any] = {"timestamp": now.isoformat()}
    if not scanner_ref:
        return heartbeat
    last_scan = getattr(scanner_ref, "last_scan_time", None)
    if last_scan:
        try:
            last_scan_naive = last_scan.replace(tzinfo=None) if getattr(last_scan, "tzinfo", None) else last_scan
            heartbeat["last_scan"] = last_scan.isoformat()
            heartbeat["scan_age_ms"] = round(max(0.0, (now - last_scan_naive).total_seconds() * 1000.0))
        except Exception:
            heartbeat["scan_age_ms"] = None
    else:
        heartbeat["scan_age_ms"] = None
    heartbeat["scan_count"] = getattr(scanner_ref, "scan_count", 0)
    heartbeat["is_calculating"] = bool(getattr(scanner_ref, "is_calculating", False))
    heartbeat["calculation_lock"] = bool(getattr(scanner_ref, "_calculation_lock", False))
    heartbeat["mode"] = getattr(scanner_ref, "mode", None)
    return heartbeat
