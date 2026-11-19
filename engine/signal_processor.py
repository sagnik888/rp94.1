import pandas as pd
import time
import copy
import math
import re
import hashlib
import json
from datetime import date, datetime, time as dtime, timedelta
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Optional
from loguru import logger
from dataclasses import dataclass, field
from engine.scanner_modes import scanner_mode, configured_mode, COMBINED_MODES, scope_suffix, same_owner, native_result, native_policy
from config.settings import (
    get_settings,
    normalize_timeframe_entry_policy,
    timeframe_entry_timeframes,
)
from engine.intelligence_score import (
    aligned_intelligence_score,
    LEGACY_SCORE_SCALE,
    aggregate_intelligence_score,
    normalize_intelligence_score,
)
from engine.candidate_selection import rank_entry_candidates
from engine.candidate_pipeline import (
    candidate_size_multipliers,
    normalized_direction,
    select_contract_strikes,
)
from engine.build_identity import build_identity
from engine.option_history_store import get_option_history_store
from engine.signal_quality_policy import (
    choppy_confidence_gate,
    dynamic_rr,
    five_min_true_impulse_allowed,
    grade_min_confidence,
    grade_rank,
    is_choppy_regime,
    option_grade_allowed,
    option_index_stop_distance,
    option_premium_health_ok,
    passes_quality_gate,
    power_move_confirmed,
    relaxed_threshold,
    volume_spike_entry_confirmed,
    volume_spike_min_ratio,
)
from engine.market_regime import (
    futures_stop_tolerance_pct,
    is_trending_regime,
    option_stop_tolerance_pct,
    regime_aligns,
)
from engine.entry_policy import effective_underlying_stop_cap_pct
from engine import s1_lane as s1_lane_policy
from engine import profit_lock_hold as profit_lock_hold_policy
from engine import uts_lane as uts_lane_policy
from engine import s12_lane as s12_lane_policy
from engine import entry_window
from engine.position_sizing import resolve_trade_lots, size_lots_for_trade
from engine.trade_accounting import estimate_trade_charges
from engine.signal_manager import GradedSignal
from trading.trade_manager import IST, Trade
from engine.ut_bot_core import UTBotSignal


def _is_bfo_instrument(value) -> bool:
    normalized = str(value or "").upper()
    return "SENSEX" in normalized or "BANKEX" in normalized


def _setting_enabled(value, default: bool = True) -> bool:
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


def _market_naive(value) -> pd.Timestamp:
    """Normalize an instant to the repository's naive IST candle clock."""
    ts = pd.Timestamp(value)
    return ts.tz_convert(IST).tz_localize(None) if ts.tzinfo is not None else ts


def _closed_candles_at(frame, as_of, timeframe):
    if frame is None or getattr(frame, "empty", True):
        return frame
    work = frame.copy()
    work.index = pd.DatetimeIndex([_market_naive(ts) for ts in work.index])
    duration = pd.Timedelta(str(timeframe or "1min"))
    return work.loc[work.index + duration <= _market_naive(as_of)]


class _PointInTimeCandles:
    """Read-only candle view that never exposes rows after replay clock."""

    def __init__(self, source):
        self.source = source
        self.as_of = None

    def get_candles(self, name, timeframe):
        if self.as_of is None or self.source is None:
            return None
        try:
            base = self.source.get_candles(name, "1min")
        except Exception:
            return None
        if base is None or getattr(base, "empty", True):
            return None
        frame = base.copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            return None
        cutoff = pd.Timestamp(self.as_of)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert(IST).tz_localize(None)
        if frame.index.tz is not None:
            frame.index = frame.index.tz_convert(IST).tz_localize(None)
        frame = frame[frame.index + pd.Timedelta(minutes=1) <= cutoff]
        timeframe = str(timeframe or "1min").lower()
        if frame.empty or timeframe == "1min":
            return frame
        if timeframe not in {"5min", "15min"}:
            return None
        aggregations = {
            key: value
            for key, value in {
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }.items()
            if key in frame.columns
        }
        if not all(key in aggregations for key in ("open", "high", "low", "close")):
            return None
        resampled = frame.resample(
            timeframe, label="left", closed="left"
        ).agg(aggregations).dropna(subset=["open", "high", "low", "close"])
        return resampled.loc[resampled.index + pd.Timedelta(timeframe) <= cutoff]

    def get_latest_price(self, name):
        frame = self.get_candles(name, "1min")
        if frame is None or frame.empty:
            return None
        try:
            return float(frame.iloc[-1]["close"])
        except Exception:
            return None

    @staticmethod
    def latest_candle_receipt(_name, _timeframe):
        return None


def directional_fut_veto(
    settings,
    *,
    inst_type: str,
    signal_type: str,
    regime: str,
    adx_value: float,
    power_move: bool = False,
) -> str:
    """Return a rejection reason if a directional FUTURES entry is in trendless
    chop or fights a confirmed opposite trend, else "".

    Narrow veto (relies on the now-honest regime detector). Blocks directional
    FUT only when:
      (a) the regime is trendless chop (RANGING/VOLATILE/UNKNOWN) AND there is no
          ADX momentum (adx < floor) — the genuine negative-expectancy case, or
      (b) it fights a CONFIRMED opposite trend (regime trends against the signal).
    It does NOT block FUT in aligned trends, nor in choppy regimes that carry real
    ADX momentum. This is deliberately less aggressive than the old
    "block-unless-perfect-aligned-trend" rule, which (with the fixed detector)
    strangled legitimate FUT and cratered backtest gains. The low-ADX "breakout"
    case it used to need is now handled at the source: the detector no longer
    mislabels those as BREAKOUT, so they arrive as RANGING/VOLATILE and (a) catches
    them. Options and data-confirmed power-moves are exempt.
    """
    if str(inst_type or "").upper() != "FUT":
        return ""
    if power_move:
        return ""
    if not _setting_enabled(getattr(settings, "regime_veto_enabled", True), True):
        return ""
    veto_adx = float(getattr(settings, "regime_veto_adx_floor", 18.0) or 18.0)
    adx = float(adx_value or 0.0)
    # (a) trendless chop with no momentum
    if is_choppy_regime(regime) and adx < veto_adx:
        return (
            f"Regime veto: directional FUT in trendless {regime} chop "
            f"(ADX {adx:.0f} < {veto_adx:.0f})"
        )
    # (b) fighting a confirmed opposite trend
    if is_trending_regime(regime) and not regime_aligns(regime, signal_type):
        return f"Regime veto: directional FUT counter to confirmed {regime} trend"
    return ""


def session_trend_guard_census_yield(
    settings,
    *,
    census: dict,
    signal_type: str,
    expected: str,
    day_type: str = "",
) -> str:
    """Return a yield reason when the live census contradicts the day-type label.

    2026-07-08: the session-context label stayed TREND_DAY_UP and vetoed six
    midcap SELLs (11:19-12:36) while the session's own directional census was
    already 5L/10S bearish — midcap then fell 1.9%. When the census opposes the
    label by `dominance` with at least `min_total` votes, the label is stale and
    the trend-phase guard must yield instead of vetoing. Returns "" to keep the
    guard's veto in force."""
    if not _setting_enabled(
        getattr(settings, "session_trend_guard_census_yield_enabled", True), True
    ):
        return ""
    signal_side = "LONG" if str(signal_type or "").upper() == "BUY" else "SHORT"
    label_side = "LONG" if str(expected or "").upper() == "BUY" else "SHORT"
    if signal_side == label_side:
        return ""
    signal_votes = int((census or {}).get(signal_side, 0) or 0)
    label_votes = int((census or {}).get(label_side, 0) or 0)
    min_total = int(getattr(settings, "session_trend_guard_census_min_total", 6) or 6)
    dominance = float(getattr(settings, "session_trend_guard_census_dominance", 1.5) or 1.5)
    if (signal_votes + label_votes) < min_total:
        return ""
    if signal_votes < max(1, label_votes) * dominance:
        return ""
    return (
        f"Session trend guard yielded: census {signal_votes}{signal_side[0]}/"
        f"{label_votes}{label_side[0]} contradicts stale {str(day_type or 'trend-day')} label"
    )


def directional_census_block(
    settings,
    census: dict,
    *,
    signal_direction: str,
    confidence: float,
    grade_floor: float,
    power_move: bool = False,
    reversal_context: Optional[dict] = None,
    day_type: str = "",
    day_type_confidence: float = 0.0,
) -> str:
    """Return a rejection reason if a counter-session-bias entry is too weak, else "".

    When the session strongly favours one side, counter-direction entries must
    clear an elevated confidence floor. Shared by live and replay paths.

    The census counts the directions THIS SCANNER emitted, never price, so it can go
    stale against the tape. A CONFIRMED trend day agreeing with the signal outranks it —
    the mirror of session_trend_guard_census_yield above, which exists because a stale
    LABEL once vetoed signals the census had already turned against.
    """
    if power_move:
        return ""
    if not _setting_enabled(getattr(settings, "directional_census_enabled", True), True):
        return ""
    # 2026-08-24: every index fell (BANKNIFTY -582) and the scanner emitted 53 shorts
    # against 45 longs, but a long-heavy morning left the census at 18L/9S, so shorts
    # were refused as "counter-bias" while session context simultaneously read
    # day_type=TREND_DAY_DOWN, bias=BEARISH at 0.92 confidence. Both cannot be right;
    # the label is the one that looks at price.
    if _setting_enabled(
        getattr(settings, "directional_census_trend_day_yield_enabled", True), True
    ):
        _label = str(day_type or "").strip().upper()
        _agrees = (
            (_label == "TREND_DAY_UP" and str(signal_direction or "").upper() == "LONG")
            or (_label == "TREND_DAY_DOWN" and str(signal_direction or "").upper() == "SHORT")
        )
        if _agrees:
            try:
                _floor = float(
                    getattr(settings, "directional_census_trend_day_min_confidence", 0.60)
                    or 0.60
                )
            except (TypeError, ValueError):
                _floor = 0.60
            # A hesitant label must not be able to switch the guard off.
            if float(day_type_confidence or 0.0) >= _floor:
                return ""
    longs = int((census or {}).get("LONG", 0))
    shorts = int((census or {}).get("SHORT", 0))
    total = longs + shorts
    min_n = int(getattr(settings, "directional_census_min_signals", 12) or 12)
    if total < min_n:
        return ""
    ratio_thr = float(getattr(settings, "directional_census_ratio", 1.5) or 1.5)
    dom = "LONG" if longs >= shorts else "SHORT"
    dom_n = max(longs, shorts)
    weak_n = min(longs, shorts)
    skewed = weak_n == 0 or (dom_n / max(1, weak_n)) >= ratio_thr
    if not skewed or signal_direction == dom:
        return ""
    if reversal_context and _setting_enabled(
        getattr(settings, "directional_census_reversal_relief_enabled", True),
        True,
    ):
        relief_floor = float(
            getattr(settings, "directional_census_reversal_min_confidence", 0.20) or 0.20
        )
        if float(confidence or 0.0) >= relief_floor:
            return ""
    penalty = float(getattr(settings, "directional_census_penalty", 0.05) or 0.05)
    floor = float(grade_floor or 0.0) + penalty
    if float(confidence or 0.0) >= floor:
        return ""
    return (
        f"Counter-session-bias {signal_direction} (session {longs}L/{shorts}S) "
        f"needs >= {floor:.0%}, got {float(confidence or 0.0):.0%}"
    )


def price_action_context(scanner, instrument: str, timeframe: str, sig, settings):
    """Structural price-action context for a signal — SEPARATE, gated patch.

    Returns the PriceActionAnalyzer context (or None) using the instrument's
    timeframe candles sliced to the signal's bar (look-ahead safe). Default OFF.
    Self-contained: remove this function + its one call site + the module to drop
    the whole feature.
    """
    if not _setting_enabled(getattr(settings, "price_action_module_enabled", False), False):
        return None
    try:
        from intelligence.price_action import PriceActionAnalyzer
        df = scanner.candles.get_candles(instrument, timeframe)
        if df is None or len(df) < 12:
            return None
        sig_ts = getattr(sig, "timestamp", None)
        if sig_ts is not None and hasattr(df, "index") and isinstance(df.index, pd.DatetimeIndex):
            ts = pd.Timestamp(sig_ts)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
            df = df[df.index <= ts]
        if df is None or len(df) < 12:
            return None
        # Day-type-aware premium/discount: on a confirmed aligned trend day the
        # chase penalty is scaled down (continuation lives at the extreme). Uses
        # the same session-context source and 0.70 confidence bar as the trend
        # guard; instantly reversible via price_action_premium_trend_scaled.
        premium_trend_scale = 1.0
        try:
            if _setting_enabled(
                getattr(settings, "price_action_premium_trend_scaled", True), True
            ):
                base = str(instrument or "").split()[0].upper()
                _intel = scanner._cached_timeframe_intel(base, "5min", allow_fallback=True) or {}
                _sctx = _intel.get("session_context", {}) if isinstance(_intel, dict) else {}
                if isinstance(_sctx, dict):
                    _day = str(
                        _sctx.get("effective_day_type") or _sctx.get("day_type") or ""
                    ).upper()
                    _conf = float(
                        _sctx.get("phase_confidence")
                        if _sctx.get("phase_confidence") is not None
                        else _sctx.get("confidence") or 0.0
                    )
                    _side = str(getattr(sig, "signal_type", "") or "").upper()
                    _aligned = (_day == "TREND_DAY_UP" and _side == "BUY") or (
                        _day == "TREND_DAY_DOWN" and _side == "SELL"
                    )
                    _min_conf = float(
                        getattr(settings, "price_action_premium_trend_min_confidence", 0.70)
                        or 0.70
                    )
                    if _aligned and _conf >= _min_conf:
                        premium_trend_scale = max(
                            0.0,
                            min(
                                1.0,
                                float(
                                    getattr(settings, "price_action_premium_trend_scale", 0.5)
                                    or 0.5
                                ),
                            ),
                        )
        except Exception:
            premium_trend_scale = 1.0
        return PriceActionAnalyzer().analyze(
            df,
            getattr(sig, "signal_type", "BUY"),
            swing_strength=int(getattr(settings, "price_action_swing_strength", 5) or 5),
            use_premium_discount=_setting_enabled(
                getattr(settings, "price_action_premium_discount_enabled", True), True
            ),
            use_liquidity=_setting_enabled(
                getattr(settings, "price_action_liquidity_enabled", True), True
            ),
            premium_trend_scale=premium_trend_scale,
            adverse_threshold=float(
                getattr(settings, "price_action_entry_adverse_threshold", -0.40) or -0.40
            ),
        )
    except Exception:
        return None


def _parse_clock(label: str, default: str):
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(str(label).strip(), fmt).time()
        except ValueError:
            continue
    return datetime.strptime(default, "%H:%M").time()


def _eod_protect_entry_cutoff(settings):
    """When EOD profit-protection engages, or None if it is off/unknowable.

    Entries taken after this point are opened into a regime whose own EOD logic closes
    them immediately, so this is a hard ceiling on any entry cutoff.
    """
    if not _setting_enabled(
        getattr(settings, "block_entries_during_eod_protect", True), True
    ):
        return None
    if not _setting_enabled(getattr(settings, "eod_protect_enabled", True), True):
        return None
    try:
        window = float(getattr(settings, "eod_protect_window_minutes", 30.0) or 0.0)
        if window <= 0:
            return None
        force_exit = _parse_clock(
            getattr(settings, "ut_force_exit_time", "15:22") or "15:22", "15:22"
        )
        start = datetime.combine(date.min, force_exit) - timedelta(minutes=window)
        return start.time()
    except Exception:
        return None


def _entry_cutoff(settings, timeframe: str, s1: bool = False):
    """Return the effective no-entry label and time for a timeframe.

    2026-08-11: ut_5min_no_entry_after ("15:14") sat 22 minutes AFTER EOD protection
    engaged (ut_force_exit_time 15:22 minus a 30-minute window = 14:52), so for 22
    minutes the system opened fresh 5min positions into a regime whose EOD logic closed
    them on the next tick. Four entries at 15:09-15:11 round-tripped in 0.06-0.19s and
    paid ~Rs.80 of round-trip charges each for holds of a fraction of a second.

    The two settings are now reconciled rather than left to contradict each other: the
    effective cutoff is whichever comes first.

    A lane signal (UT SMART EXIT, S1) uses the system-wide last-entry times (2026-09-26
    user: 15m 14:50, 5m 15:10, 1m 15:14) without the EOD-protect clamp: EOD protection is
    another engine's exit guard and never manages a lane trade; the 15:15 force exit still
    closes it. Lane callers test the ENTRY time (entry_window.entry_allowed); the returned
    time is the label's.
    """
    if s1:
        # `s1` is the lane tag ("S1"/"UTS") or a bool (True = S1).
        label = s1_lane_policy.lane_entry_cutoff(s1 if isinstance(s1, str) else "S1", timeframe, settings)
        return label, _parse_clock(label, "15:10")
    if str(timeframe) == "1min":
        label = str(getattr(settings, "ut_1min_no_entry_after", "15:14") or "15:14")
    elif str(timeframe) == "5min":
        label = str(getattr(settings, "ut_5min_no_entry_after", "15:15") or "15:15")
    else:
        label = str(getattr(settings, "ut_no_entry_after", "15:00") or "15:00")
    cutoff = _parse_clock(label, "15:15")
    eod_start = _eod_protect_entry_cutoff(settings)
    if eod_start is not None and eod_start < cutoff:
        return f"{eod_start.strftime('%H:%M')} (EOD protect)", eod_start
    return label, cutoff


#: Grading evidence lives on the UTBotSignal, but TradeCandidate is built field-by-field.
#: Carry the decision-tape fields and the frozen inputs needed for consistent regrading.
GRADING_TELEMETRY_ATTRS = (
    "graded_timeframe",
    "graded_intel_score",
    "graded_regime",
    "grade_pillars",
    "confidence_calibrated",
    "setup_score",
    "calibrated_win_probability",
    "expected_net_r",
    "grading_inputs",
    "grading_inputs",
)

UTS_SETUP_ATTRS = ("uts_setup", "uts_multi_index_alignment")


def carry_grading_telemetry(candidate, signal) -> None:
    """Copy grading evidence from the graded signal onto its candidate.

    `_grade_signal` attaches the pillar breakdown to the signal, and the per-timeframe
    grading context is set on that same object. TradeCandidate does not inherit from it,
    so this hop makes the breakdown visible and preserves inputs for buffer regrading.
    It must never raise into the signal path.
    """
    for name in GRADING_TELEMETRY_ATTRS + UTS_SETUP_ATTRS:
        value = getattr(signal, name, None)
        if value is None and name not in {"setup_score", "calibrated_win_probability", "expected_net_r"}:
            continue
        try:
            setattr(candidate, name, value)
        except Exception:  # pragma: no cover - dataclass always accepts these
            pass


def _intelligence_stop_pct(
    base_pct: float,
    intel_score: float,
    settings,
    direction: str = "",
    score_unit: str = "legacy_0_100",
) -> float:
    """Nudge the option stop percentage by how strong the corroborating evidence is.

    New signal-path callers pass canonical signed -1..+1 scores and a direction. Legacy
    direct callers may still pass 0..100 with 50 neutral during the compatibility window.
    Corroborating evidence widens slightly; contradictory evidence tightens. Adjustment is capped by
    dynamic_stop_intel_max_shift_pct so a stop chosen by ATR and the per-timeframe rules
    is only ever nudged, never redefined, and the result is clamped to a sane band.

    Off unless dynamic_stop_from_intelligence is set: this moves the risk on every option
    entry, which is a strategy decision rather than a defect.
    """
    try:
        base = float(base_pct or 0.0)
    except (TypeError, ValueError):
        return float(base_pct or 0.0)
    if base <= 0:
        return base
    enabled = getattr(settings, "dynamic_stop_from_intelligence", False)
    if not (enabled is True or str(enabled).strip().lower() in {"1", "true", "yes", "on"}):
        return base
    try:
        score = float(intel_score or 0.0)
    except (TypeError, ValueError):
        return base
    unit = str(score_unit or "").strip().lower()
    if unit == "canonical_signed":
        if score == 0.0:
            return base                  # no directional evidence must not move risk
        lean = aligned_intelligence_score(score, direction)
    else:
        if score <= 0:
            return base                  # legacy zero means no reading
        lean = max(-1.0, min(1.0, (score - 50.0) / 50.0))
    try:
        max_shift = float(getattr(settings, "dynamic_stop_intel_max_shift_pct", 20.0) or 0.0)
    except (TypeError, ValueError):
        max_shift = 20.0
    max_shift = max(0.0, min(50.0, max_shift)) / 100.0
    if max_shift <= 0:
        return base
    return max(0.5, min(base * 2.0, base * (1.0 + lean * max_shift)))


@dataclass

class TradeCandidate:
    instrument: str
    direction: str
    price: float
    stop: float
    target: float
    lots: int
    lot_size: int
    grade: str
    confidence: float
    timeframe: str
    inst_type: str
    option_type: str
    atm_strike: float
    multiplier: float
    trading_symbol: str
    symbol_token: str
    rr: float = 1.5
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    # Explicit, non-interchangeable price spaces. ``price/stop/target`` remain the
    # operational traded-contract tuple for compatibility; these fields make the
    # underlying risk and option-premium risk independently auditable.
    entry_spot: float = 0.0
    spot_stop: float = 0.0
    spot_target: float = 0.0
    underlying_entry: float = 0.0
    underlying_stop: float = 0.0
    underlying_target: float = 0.0
    premium_entry: float = 0.0
    premium_stop: float = 0.0
    premium_target: float = 0.0
    geometry_observed_at: str = ""
    geometry_source: str = ""
    geometry_verified: bool = False
    geometry_schema: str = "dual_price_geometry.v1"
    expected_move_forecast: dict[str, Any] = field(default_factory=dict)
    entry_policy_verdict: dict[str, Any] = field(default_factory=dict)
    # Shared live-data verdict for forecasts and exceptional-entry admission.
    entry_data_quality: dict[str, Any] = field(default_factory=dict)
    # Versioned quality decision made before raw/override routing.  Unlike a dynamic
    # attribute, this survives the session ledger and can be joined to rtracker exits.
    quality_verdict: dict[str, Any] = field(default_factory=dict)
    decision_record: dict[str, Any] = field(default_factory=dict)
    signal_timestamp: Optional[datetime] = None
    # Wall-clock snapshot when the candidate was constructed. The signal timestamp is
    # the source candle's left edge and is not a valid cutoff for live location checks.
    decision_timestamp: Optional[datetime] = None
    current_price: float = 0.0
    pnl: float = 0.0
    status: str = "TRADE SIGNAL"
    action: str = "ENTRY"
    exit_reason: str = ""
    adx_value: float = 0.0
    is_explosive_bypass: bool = False
    impulse_confirmed: bool = False
    power_move: bool = False
    volume_spike_entry: bool = False
    # Whether `volume_ratio` represents traded/constituent quantity or a separately
    # calibrated activity proxy.  Consumers must not treat the latter as traded volume.
    volume_basis: str = ""
    market_ltp_shock_entry: bool = False
    late_session_reclaim_entry: bool = False
    one_way_cascade_entry: bool = False
    fifteen_min_choppy_scalp_entry: bool = False
    volume_ratio: float = 0.0
    range_atr_ratio: float = 0.0
    close_location: float = 0.0
    tight_stop_pct: float = 0.0
    initial_stop: float = 0.0
    scalp_lock_mode: bool = False
    quote_verified: bool = False
    quote_source: str = ""
    execution_bid_price: float = 0.0
    execution_ask_price: float = 0.0
    execution_touch_source: str = ""
    execution_touch_age_seconds: float = 0.0
    execution_touch_usable: bool = False
    execution_touch_reason: str = ""
    # Per-contract tradeability evidence carried from the cached option chain to
    # the execution boundary.  Availability is explicit because a measured zero
    # must fail a minimum-liquidity rule while a missing chain is merely unknown.
    strike_oi: float = 0.0
    strike_volume: float = 0.0
    strike_oi_available: bool = False
    strike_volume_available: bool = False
    strike_liquidity_available: bool = False
    source_engine: str = ""
    source_engines: list[str] = field(default_factory=list)
    source_lane: str = ""
    quad_union_entry: bool = False
    raw_union_entry: bool = False
    watch_only: bool = False


def _sig_date(signal) -> Optional[date]:
    """The calendar date of a signal, or None when it carries no usable timestamp.

    Accepts either a signal object with a ``timestamp`` attribute or a bare
    datetime/date. The bare case is checked FIRST because datetime itself has a
    ``timestamp`` *method* -- reading the attribute would hand back a bound method
    and quietly resolve to None.

    Every caller of _estimate_opt_premium used to omit trade_date, which disabled
    both the DTE boost and the historical-VIX lookup. Returning None preserves that
    old behaviour for an unusable timestamp rather than raising on the hot path.
    """
    if isinstance(signal, datetime):
        return signal.date()
    if isinstance(signal, date):
        return signal
    ts = getattr(signal, "timestamp", None)
    if ts is None or callable(ts):
        return None
    try:
        if isinstance(ts, datetime):
            return ts.date()
        if isinstance(ts, date):
            return ts
        parsed = pd.to_datetime(ts)
        # NaN/None parse to NaT rather than raising, and NaT.date() is NaT again --
        # which would sail out of here as a truthy non-date.
        if parsed is None or pd.isna(parsed):
            return None
        return parsed.date()
    except Exception:
        return None


def _estimate_opt_premium(spot_price: float, scanner=None, instrument: str = "", trade_date=None) -> float:
    """Dynamic option premium estimation based on VIX.

    ``instrument`` and ``trade_date`` gate the DTE boost and the historical-VIX
    average. Every call site omitted them for the life of this function, so
    dte_boost was pinned at 1.0 and HISTORICAL runs silently priced against TODAY's
    live VIX -- lookahead bias in every backtest.
    """
    base_pct = 0.012
    vix_val = 15.0
    if scanner and trade_date is not None:
        try:
            # A date-only request has no intraday knowledge. Use midnight as its
            # cutoff, never the mean of the eventual day's VIX observations.
            resolver = getattr(scanner, "_get_historical_vix", None)
            measured = float(resolver(pd.Timestamp(trade_date)) or 0.0) if callable(resolver) else 0.0
            if measured > 0:
                vix_val = measured
        except Exception:
            pass
    elif scanner and hasattr(scanner, "data"):
        try:
            live_vix = scanner.data.get_latest_price("INDIA VIX")
            if live_vix and live_vix > 0:
                vix_val = live_vix
        except Exception:
            pass

    dte_boost = 1.0
    if scanner and instrument and trade_date is not None and hasattr(scanner, "expiry"):
        try:
            dte = max(0.0, float(_scanner_days_to_expiry(scanner, instrument, trade_date)))
            dte_boost = max(0.75, min(1.35, (max(dte, 1.0) / 3.0) ** 0.25))
        except Exception:
            pass

    # Premium roughly scales with VIX. At VIX 15, pct is 1.2%
    scaler = max(10.0, vix_val) / 15.0
    return spot_price * base_pct * scaler * dte_boost

def _scanner_days_to_expiry(scanner, instrument, as_of):
    # Import at call time: Scanner owns mode routing, including replay probes.
    from scanner import Scanner
    return Scanner._days_to_expiry_for(scanner, instrument, as_of)


def _scanner_is_expiry_day(scanner, instrument, as_of):
    from scanner import Scanner
    return Scanner._is_expiry_day_for(scanner, instrument, as_of)


class SignalProcessor:
    """
    Handles signal candidate selection, scoring, and prioritization.
    Extracted from Scanner to reduce file size.
    """
    def __init__(self, scanner):
        self.scanner = scanner
        self._option_history_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}
        self._option_symbol_history_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._option_history_fetch_attempted = set()
        self._option_history_store = get_option_history_store()

    @staticmethod
    def _record_directional_census(scanner, signal, instrument, timeframe,
                                   direction, identity_stamp, settings):
        """Count native selection arrows without mixing in independent own lanes.

        S1 and UTS skip this selection gate. Their arrows must not change the
        census that can reject the companion engine's otherwise eligible setup.
        Live and replay callers retain their existing timestamp identity format.
        """
        if s1_lane_policy.is_lane(signal, settings):
            return
        try:
            day = signal.timestamp.date().isoformat()
            if getattr(scanner, "_dir_census_date", None) != day:
                scanner._dir_census_date = day
                scanner._dir_census = {"LONG": 0, "SHORT": 0}
                scanner._dir_census_seen = set()
            identity = f"{instrument}|{timeframe}|{identity_stamp}|{direction}"
            if identity not in scanner._dir_census_seen:
                scanner._dir_census_seen.add(identity)
                scanner._dir_census[direction] = scanner._dir_census.get(direction, 0) + 1
        except Exception:
            pass

    @staticmethod
    def _historical_signal_for_replay(signal, timeframe):
        """Keep the arrow's bar identity while acting only after its close.

        Raw MTF signals and all live paths retain their original timestamps.
        This copy is used exclusively by the historical reconstruction loop.
        """
        result = copy.copy(signal)
        source = getattr(signal, "source_signal_timestamp", None) or signal.timestamp
        result.source_signal_timestamp = source
        result.signal_available_at = (
            _market_naive(source) + pd.Timedelta(str(timeframe))
        ).to_pydatetime()
        result.timestamp = result.signal_available_at
        return result

    @staticmethod
    def _historical_entry_cutoff_allows(signal, timeframe: str, lane_tag: str, settings) -> bool:
        """Replay entry cutoff.

        Historical signals are shifted to the bar close before execution. A lane signal
        (UT SMART EXIT, S1) enters at its bar's close, which must land by the system-wide
        last entry (entry_window). Other engines retain their established
        availability-time comparison.
        """
        if lane_tag:
            start = getattr(signal, "source_signal_timestamp", None) or getattr(signal, "timestamp", None)
            if start is None:
                return False
            return entry_window.entry_allowed(_market_naive(start), timeframe, settings)
        _, cutoff = _entry_cutoff(settings, timeframe, s1=lane_tag)
        value = getattr(signal, "timestamp", None)
        if value is None:
            return False
        return _market_naive(value).time() < cutoff

    @staticmethod
    def _historical_uts_premium_stop_distance(
        premium: float,
        index_stop_distance: float,
        delta: float,
        settings,
        *,
        regime: str = "UNKNOWN",
        confidence=None,
        grade: str = "",
        setup=None,
        timeframe="",
    ) -> float:
        """UTS replay premium guard, identical to live premium-space geometry."""
        return uts_lane_policy.premium_stop_distance(
            premium,
            index_stop_distance,
            delta,
            settings,
            regime=regime,
            confidence=confidence,
            grade=grade,
            setup=setup,
            timeframe=timeframe,
        )

    @staticmethod
    def _historical_index_price_window(scanner, instrument, start, end):
        """Covered one-minute index proxy prices; not actual futures history."""
        frame = _closed_candles_at(scanner.candles.get_candles(instrument, "1min"), end, "1min")
        if frame is None or frame.empty:
            return pd.DataFrame()
        frame = frame.loc[frame.index >= _market_naive(start)].sort_index()
        if frame.empty or frame.index[0] - _market_naive(start) >= pd.Timedelta(minutes=1):
            return pd.DataFrame()
        if (frame.index.to_series().diff().dropna() > pd.Timedelta(minutes=1)).any():
            return pd.DataFrame()
        return frame

    @staticmethod
    def _historical_price_evidence(inst_type, actual_option=None, *, signal=None, entry_time=None, shadow=False):
        """Describe replay price origin without changing accounting or outcomes.

        BACKTEST remains the compatibility replay-family tag. This trace carries
        the finer price basis so modeled derivatives cannot pass as observations.
        """
        if str(inst_type).upper() == "OPT" and actual_option:
            trace = copy.deepcopy(actual_option.get("exit_trace") or [])
            input_row = next((item for item in trace if item.get("guard") == "replay_input"), None)
            if input_row is None:
                input_row = {"guard": "replay_input", "state": "OBSERVED_OPTION_OHLC"}
                trace.insert(0, input_row)
            input_row.update({"price_basis": "OBSERVED_OPTION_OHLC", "observed_option_prices": True, "actual_futures_prices": False, "synthetic_option_prices": False})
            return trace
        basis = "SYNTHETIC_OPTION_ESTIMATE" if str(inst_type).upper() == "OPT" else "INDEX_PRICE_PROXY"
        source_time = getattr(signal, "source_signal_timestamp", None) or getattr(signal, "timestamp", None)
        return [{
            "guard": "replay_input", "state": basis, "price_basis": basis,
            "time": pd.Timestamp(entry_time).isoformat() if entry_time is not None else "",
            "source_signal_timestamp": pd.Timestamp(source_time).isoformat() if source_time is not None else "",
            "observed_option_prices": False, "actual_futures_prices": False,
            "synthetic_option_prices": basis == "SYNTHETIC_OPTION_ESTIMATE",
            "shadow": bool(shadow), "contract_verified": False,
            "fill_model": "modeled_option_premium_from_index" if basis == "SYNTHETIC_OPTION_ESTIMATE" else "index_ohlc_proxy_not_futures_contract",
        }]

    @staticmethod
    def _historical_futures_fill_evidence(window, *, signal, entry_time, exit_time,
                                         entry_price, exit_price, exit_reason):
        """Bind completed causal index-proxy fills to their accounting record.

        These are modeled futures outcomes, never observed futures contracts. Only
        a continuous, valid path through the selected fill can bypass legacy display
        repairs; malformed or incomplete paths retain ordinary proxy evidence.
        """
        import numpy as np

        trace = SignalProcessor._historical_price_evidence(
            "FUT", signal=signal, entry_time=entry_time,
        )
        if window is None or window.empty:
            return trace
        try:
            start, end = _market_naive(entry_time), _market_naive(exit_time)
            if end < start or not all(math.isfinite(float(value)) and float(value) > 0
                                      for value in (entry_price, exit_price)):
                return trace
            frame = window.copy()
            frame.index = pd.DatetimeIndex([_market_naive(ts) for ts in frame.index])
            # A gap fills at its open; retain that opening row as evidence. A fill
            # at a minute close needs only rows closed by that boundary.
            at_open = end in frame.index and math.isclose(
                float(frame.loc[end, "open"]), float(exit_price), rel_tol=0., abs_tol=1e-8,
            )
            frame = frame.loc[(frame.index >= start) &
                              ((frame.index <= end) if at_open else
                               (frame.index + pd.Timedelta(minutes=1) <= end))]
            if (frame.empty or not frame.index.is_unique or not frame.index.is_monotonic_increasing
                    or frame.index[0] - start >= pd.Timedelta(minutes=1)
                    or (frame.index.to_series().diff().dropna() != pd.Timedelta(minutes=1)).any()
                    or (not at_open and frame.index[-1] + pd.Timedelta(minutes=1) < end)):
                return trace
            values = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
            if (not np.isfinite(values.to_numpy()).all() or (values <= 0).any().any()
                    or (values["low"] > values[["open", "close"]].min(axis=1)).any()
                    or (values["high"] < values[["open", "close"]].max(axis=1)).any()):
                return trace
            trace[0].update({
                "fill_accounting": "CAUSAL_INDEX_PROXY_V1", "complete_path": True,
                "input_sha256": hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
                "rows": len(frame), "entry_time": start.isoformat(), "exit_time": end.isoformat(),
                "entry_price": float(entry_price), "exit_price": float(exit_price),
                "exit_reason": str(exit_reason),
                "fill_model": "confirmed_entry_closed_1m_stop_first_gap_at_open_index_proxy",
            })
        except (KeyError, TypeError, ValueError, AttributeError):
            pass
        return trace

    @staticmethod
    def _bind_historical_intelligence(probe, source, candle_view):
        """Give production exit readers an exact snapshot at the replay clock.

        Some guards call the exact-intelligence method directly rather than the
        attached trade-context fallback. Their live cache and wall-clock access
        must be replaced at this replay adapter boundary, never on the live class.
        """
        def exact(trade_like, *, evaluation_time=None, allow_blocking=True):
            reader = getattr(source, "_get_historical_intel_context_at", None)
            if candle_view.as_of is None or not callable(reader):
                return None
            instrument = str(getattr(trade_like, "instrument", "") or "").split()[0].upper()
            timeframe = str(getattr(trade_like, "timeframe", "5min") or "5min").lower()
            context = reader(instrument, _market_naive(candle_view.as_of).to_pydatetime(), timeframe)
            if not isinstance(context, dict) or not context:
                return None
            result = copy.deepcopy(context)
            result.setdefault("as_of", _market_naive(candle_view.as_of).isoformat())
            result.setdefault("timeframe", timeframe)
            return result
        probe._fresh_exact_trade_intel = exact

    def _historical_audit_log(self, message: str) -> None:
        """Keep bulk backtest reject details out of INFO unless explicitly enabled."""
        scanner = getattr(self, "scanner", None)
        if str(getattr(scanner, "mode", "") or "").upper() == "HISTORICAL":
            settings = get_settings()
            if _setting_enabled(getattr(settings, "ut_historical_verbose_audit_logs", False), False):
                logger.info(message)
            else:
                logger.debug(message)
            return
        logger.info(message)

    @staticmethod
    def _candidate_timeframe_context(
        timeframe: str,
        fallback_intel_result: Dict,
        fallback_intel_score: float,
        fallback_regime: str,
        timeframe_contexts: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[Dict, float, str]:
        """Return intelligence for a candidate's OWN timeframe.

        process_best_signal receives a single intel_result/intel_score/regime (the 5-minute
        snapshot) and grades every candidate with it, so 15-minute candidates were graded
        on 5-minute intelligence and regime. `timeframe_contexts` is optional: when it is
        absent or lacks an entry for `timeframe`, the caller's existing 5-minute values are
        returned unchanged, so behaviour is identical to before.
        """
        fallback_result = fallback_intel_result if isinstance(fallback_intel_result, dict) else {}
        context = (timeframe_contexts or {}).get(str(timeframe or ""))
        if not isinstance(context, dict):
            return fallback_result, float(fallback_intel_score or 0.0), str(fallback_regime or "UNKNOWN")

        nested_result = context.get("intel_result")
        result = nested_result if isinstance(nested_result, dict) else context
        score = context.get("intel_score", context.get("score", fallback_intel_score))
        try:
            score = float(score or 0.0)
        except (TypeError, ValueError):
            score = float(fallback_intel_score or 0.0)
        context_regime = context.get("regime", fallback_regime)
        if isinstance(context_regime, dict):
            context_regime = context_regime.get("regime", fallback_regime)
        if not context_regime and isinstance(result, dict):
            result_regime = result.get("regime", {})
            context_regime = result_regime.get("regime") if isinstance(result_regime, dict) else result_regime
        return result if isinstance(result, dict) else fallback_result, score, str(context_regime or "UNKNOWN")

    @staticmethod
    def _strike_liquidity_snapshot(chain, strike, option_type) -> Dict[str, Any]:
        """Measured OI/volume for one strike, preserving missing-vs-zero.

        The previous tuple-only helper collapsed both states to ``(0, 0)``.  That
        is safe for a ranking tie-break (missing data has no opinion) but unsafe
        for an absolute admission gate, where a broker-measured zero is evidence
        that the contract is not tradeable enough.
        """
        empty = {
            "oi": 0.0,
            "volume": 0.0,
            "oi_available": False,
            "volume_available": False,
            "available": False,
        }
        if chain is None or not len(chain) or "strike" not in getattr(chain, "columns", []):
            return empty
        side = "call" if str(option_type or "").upper() == "CE" else "put"
        oi_col, vol_col = f"{side}_oi", f"{side}_volume"
        try:
            target = float(strike or 0.0)
            if target <= 0:
                return empty
            # Strikes arrive as floats from the broker; match on proximity, not equality.
            row = chain[(chain["strike"] - target).abs() < 0.5]
            if not len(row):
                return empty
            values = row.iloc[0]
        except (TypeError, ValueError, KeyError, IndexError):
            return empty

        result = dict(empty)
        for metric, column in (("oi", oi_col), ("volume", vol_col)):
            if column not in chain.columns:
                continue
            try:
                value = float(values[column])
            except (TypeError, ValueError, KeyError):
                continue
            if value != value:  # NaN is absent evidence, not a measured zero.
                continue
            result[metric] = max(0.0, value)
            result[f"{metric}_available"] = True
        result["available"] = bool(
            result["oi_available"] and result["volume_available"]
        )
        return result

    @staticmethod
    def _strike_liquidity(chain, strike, option_type) -> Tuple[float, float]:
        """(open interest, traded volume) at ONE strike, from the already-cached chain.

        The chain carries these per strike -- call_oi / call_volume / put_oi /
        put_volume -- so nothing here costs a broker call. Returns (0.0, 0.0) whenever
        the answer is not knowable, which the caller treats as "no opinion" rather than
        as "illiquid": an unreadable chain must never move the selection.
        """
        snapshot = SignalProcessor._strike_liquidity_snapshot(
            chain, strike, option_type
        )
        # Backward-compatible ranking API: unavailable remains "no opinion".
        if not snapshot["available"]:
            return 0.0, 0.0
        return float(snapshot["oi"]), float(snapshot["volume"])

    @staticmethod
    def _prefer_liquid_strike(ranked, settings):
        """Break a near-tie between strike candidates on tradeability.

        `ranked` is already sorted by score, so ranked[0] is the incumbent and only a
        candidate within the configured band may displace it -- a materially better
        setup always wins regardless of liquidity.

        The challenger must be better on BOTH open interest and traded volume. A
        weighted blend of the two would need a weight nobody can justify; requiring both
        is scale-free and means "more standing interest AND more activity today".
        """
        if not ranked:
            return None
        leader = ranked[0]
        if len(ranked) < 2:
            return leader
        if not _setting_enabled(
            getattr(settings, "option_strike_liquidity_ranking_enabled", True), True
        ):
            return leader
        try:
            band = float(getattr(settings, "option_strike_liquidity_score_band", 0.05) or 0.0)
            edge = float(getattr(settings, "option_strike_liquidity_min_advantage", 1.25) or 1.25)
        except (TypeError, ValueError):
            return leader
        if band <= 0 or edge <= 1.0:
            return leader

        def _measured(candidate) -> float:
            """Score with the standing ITM preference removed.

            itm_bonus is a flat 0.15 added to the ITM candidate -- a policy preference,
            not a measurement of that contract. Comparing raw scores meant the gap was
            structurally at least 0.15 whenever ITM led, three times the default band,
            so the tie-break could not fire. Netting it out lets liquidity compete with
            the preference instead of being buried by it.
            """
            try:
                return float(candidate.score) - float(
                    getattr(candidate, "strike_preference_bonus", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                return float("nan")

        lead_oi, lead_vol = (
            getattr(leader, "strike_oi", 0.0) or 0.0,
            getattr(leader, "strike_volume", 0.0) or 0.0,
        )
        lead_measured = _measured(leader)
        for challenger in ranked[1:]:
            gap = lead_measured - _measured(challenger)
            if gap != gap:  # NaN: an unscored candidate cannot be compared
                continue
            if gap > band:
                # `continue`, not `break`: ranked is sorted by RAW score, and netting
                # out the preference bonus can reorder it, so a later candidate may
                # still be inside the band.
                continue
            oi = float(getattr(challenger, "strike_oi", 0.0) or 0.0)
            vol = float(getattr(challenger, "strike_volume", 0.0) or 0.0)
            # Absent liquidity is no opinion, not a verdict. Only a strike the chain
            # actually described may displace the incumbent.
            if oi <= 0 or vol <= 0 or lead_oi <= 0 or lead_vol <= 0:
                continue
            if oi >= lead_oi * edge and vol >= lead_vol * edge:
                return challenger
        return leader

    @staticmethod
    def _graded_candidate_context(
        best_sig,
        fallback_regime: str,
        fallback_intel_norm: float,
    ) -> Tuple[str, float]:
        """Return the (regime, normalised intel) a candidate should be STAMPED with.

        _candidate_timeframe_context grades a 15-minute candidate on 15-minute
        intelligence, but the candidate was then stamped with the 5-minute
        `regime`/`intel_norm` that happened to be in scope. Every downstream gate and
        exit reads candidate.market_regime / .intelligence_score, so enabling
        per-timeframe grading changed the grade and nothing else.

        The values recorded on the signal by the grading loop are the ones that
        actually graded it. With per_timeframe_grading_enabled off they resolve to the
        caller's 5-minute values, so this returns the previous pair unchanged.

        Presence, not truthiness: a genuine 0.0 intelligence score is a real reading
        and must not fall back to the 5-minute score.
        """
        regime = str(getattr(best_sig, "graded_regime", "") or "").strip()
        if not regime:
            regime = str(fallback_regime or "UNKNOWN")

        attr = getattr(best_sig, "graded_intel_score", None)
        if isinstance(attr, bool) or not isinstance(attr, (int, float)):
            return regime, float(fallback_intel_norm or 0.0)
        try:
            graded_raw = float(attr)
        except (TypeError, ValueError):
            return regime, float(fallback_intel_norm or 0.0)
        if graded_raw != graded_raw or graded_raw in (float("inf"), float("-inf")):
            return regime, float(fallback_intel_norm or 0.0)
        # Same unit as the caller's intel_score, which process_best_signal converts
        # with normalize_intelligence_score. This is a scalar already extracted by
        # _candidate_timeframe_context, not an aggregate payload -- reading score_norm
        # off a payload belongs there, at the extraction, not here.
        return regime, float(normalize_intelligence_score(graded_raw))

    @staticmethod
    def _graded_candidate_intel_result(best_sig, fallback_intel_result: Dict) -> Dict:
        """Return the full intelligence bundle that graded ``best_sig``.

        The scalar regime/score already crossed the grading→candidate seam, but the
        candidate's pillar/context payload was still built from the function-scope
        five-minute bundle.  Downstream gates and exits therefore saw a mixed object:
        a 15-minute scalar beside 5-minute OI/volume/session context.

        When per-timeframe grading is disabled, the grading loop stores the exact
        fallback object here, so this helper returns that same object and preserves
        flag-off behavior rather than manufacturing a second policy path.
        """
        graded = getattr(best_sig, "graded_intel_result", None)
        if isinstance(graded, dict):
            return graded
        return fallback_intel_result if isinstance(fallback_intel_result, dict) else {}

    @staticmethod
    def _build_timeframe_contexts(scanner, instrument: str, settings) -> Optional[Dict[str, Dict]]:
        """Collect each timeframe's own cached intelligence snapshot.

        Returns None only when the feature is disabled. With the live default enabled,
        callers can distinguish a genuinely missing own-timeframe snapshot from a
        deliberate legacy/fallback mode.
        """
        enabled = getattr(settings, "per_timeframe_grading_enabled", False)
        if not (enabled is True or str(enabled).strip().lower() in {"1", "true", "yes", "on"}):
            return None
        getter = getattr(scanner, "_cached_timeframe_intel", None)
        if not callable(getter):
            return {}
        contexts: Dict[str, Dict] = {}
        for tf in ("1min", "5min", "15min"):
            try:
                intel = getter(instrument, tf, allow_fallback=False)
            except Exception:
                intel = None
            if not isinstance(intel, dict):
                continue
            # Production cache entries carry a timeframe-qualified capture epoch.
            # Do not let an old exact-timeframe bundle grade a new candidate. Small
            # unit-test stubs without cache clocks remain usable; the live Scanner
            # always initializes _intel_cache_time and therefore fails closed.
            cache_times = getattr(scanner, "_intel_cache_time", None)
            if isinstance(cache_times, dict):
                key_builder = getattr(scanner, "_timeframe_intel_key", None)
                key = (
                    key_builder(instrument, tf)
                    if callable(key_builder)
                    else f"{str(instrument or '').upper()}_{tf}"
                )
                captured = cache_times.get(key)
                if captured is None:
                    captured = intel.get("as_of")
                try:
                    captured_ts = float(captured)
                    ttl_getter = getattr(scanner, "_intel_analysis_ttl_seconds", None)
                    ttl = float(ttl_getter() if callable(ttl_getter) else getattr(settings, "intelligence_analysis_ttl_seconds", 30.0))
                    age = time.time() - captured_ts
                except (TypeError, ValueError, OverflowError):
                    continue
                if not (0.0 <= age <= max(0.0, ttl)):
                    continue
            aggregate = intel.get("aggregate") if isinstance(intel.get("aggregate"), dict) else {}
            regime_block = intel.get("regime") if isinstance(intel.get("regime"), dict) else {}
            contexts[tf] = {
                "intel_result": intel,
                # CANONICAL, like every other producer of this value.
                #
                # This read `aggregate.get("score")` raw. That field is always
                # legacy-scaled (-100..100) and was then handed to
                # normalize_intelligence_score, which infers the scale from MAGNITUDE:
                # a legacy 0.9 -- all but neutral -- is under the `abs(score) > 1.0`
                # threshold, never divided, and reads as canonical 0.9, near-maximum
                # conviction. Only SMALL readings were affected, which is what made it
                # quiet; the loud cases (70 -> 0.70) looked perfectly correct.
                #
                # It was left unfixed on the reasoning that storing the canonical value
                # would diverge from the flag-off path "which hands the caller's LEGACY
                # intel_score to the same consumers". That premise was simply wrong.
                # scanner.py:10414 computes the caller's value as
                # `aggregate_intelligence_score(intel_result["aggregate"])`, so it is
                # already canonical and is passed to process_best_signal unchanged.
                # Storing canonical here CONVERGES the two paths rather than splitting
                # them, and normalize_intelligence_score downstream becomes the no-op it
                # is documented to be on values that are already canonical.
                "intel_score": aggregate_intelligence_score(aggregate),
                "regime": regime_block.get("regime") or regime_block.get("hard_label") or "",
            }
        # Empty means "feature enabled, but no own-timeframe evidence". Returning None
        # here made the grading loop interpret a total cache miss as legacy mode and
        # silently borrow the caller's 5m bundle, defeating fail-closed semantics.
        return contexts

    @staticmethod
    def _direction_mode_allows(direction: str, settings) -> Tuple[bool, str]:
        """Apply the configured strategy-side constraint to every entry path.

        `strategy_direction_mode` was declared, persisted and displayed but never
        enforced, so selecting LONG_ONLY / SHORT_ONLY did not actually restrict which
        side could trade. Default LONG_SHORT allows both, so this is a no-op unless the
        operator has deliberately narrowed the side.
        """
        mode = str(getattr(settings, "strategy_direction_mode", "LONG_SHORT") or "LONG_SHORT").upper()
        side = str(direction or "").upper()
        if side in {"BUY", "LONG"}:
            side = "LONG"
        elif side in {"SELL", "SHORT"}:
            side = "SHORT"
        if mode == "LONG_ONLY" and side == "SHORT":
            return False, "Strategy direction mode LONG_ONLY blocks SHORT entries"
        if mode == "SHORT_ONLY" and side == "LONG":
            return False, "Strategy direction mode SHORT_ONLY blocks LONG entries"
        return True, ""

    @staticmethod
    def _live_signal_candle_is_closed(signal_ts: datetime, timeframe: str, now: datetime) -> bool:
        """Return whether a signal's source candle has finalized."""
        tf_minutes = {"1min": 1, "5min": 5, "15min": 15}.get(str(timeframe or ""), 5)
        signal_check = signal_ts.replace(tzinfo=None) if getattr(signal_ts, "tzinfo", None) else signal_ts
        now_check = now.replace(tzinfo=None) if getattr(now, "tzinfo", None) else now
        return now_check >= signal_check + timedelta(minutes=tf_minutes)

    @staticmethod
    def _live_signal_is_new(
        signal_ts: datetime,
        signal_type: str,
        last_ts: datetime,
        last_identity,
    ) -> bool:
        """Return whether this exact timestamp/direction identity is still virgin.

        Union engines can legitimately retain both BUY and SELL at the same timestamp
        (different source families flipped on the same bar).  A single remembered
        identity therefore oscillates BUY -> SELL -> BUY on every scan and makes both
        arrows look new forever.  Accept the legacy scalar as well as the current
        collection so warm/partial scanner objects remain compatible.
        """
        identity = f"{signal_ts.isoformat()}|{signal_type}"
        identities = SignalProcessor._live_signal_identities(last_identity)
        identity_seen = any(
            known == identity or known.startswith(f"{identity}|")
            for known in identities
        )
        return signal_ts > last_ts or (signal_ts == last_ts and not identity_seen)

    @staticmethod
    def _native_entry_has_exit(signal, stream) -> bool:
        """True when this exact scanner-native entry is already terminal."""
        if bool(getattr(signal, "native_exit_only", False)):
            return False
        source = str(getattr(signal, "source_engine", "") or "").lower()
        entry_ts = getattr(signal, "timestamp", None)
        if source not in {"apex_v57", "stock_master_v732"} or entry_ts is None:
            return False
        entry_check = entry_ts.replace(tzinfo=None) if getattr(entry_ts, "tzinfo", None) else entry_ts
        for event in stream or []:
            if not bool(getattr(event, "native_exit_only", False)):
                continue
            if str(getattr(event, "source_engine", "") or "").lower() != source:
                continue
            owner_ts = getattr(event, "native_entry_timestamp", None)
            if owner_ts is None:
                continue
            owner_check = owner_ts.replace(tzinfo=None) if getattr(owner_ts, "tzinfo", None) else owner_ts
            if owner_check == entry_check:
                return True
        return False

    @staticmethod
    def _live_signal_identities(value) -> set[str]:
        """Normalize legacy scalar and current collection identity state."""
        if value is None:
            return set()
        if isinstance(value, str):
            return {value} if value else set()
        try:
            return {str(item) for item in value if str(item)}
        except TypeError:
            text = str(value)
            return {text} if text else set()

    @classmethod
    def _remember_live_signal_identity(
        cls,
        scanner,
        dedup_key: str,
        signal_ts: datetime,
        signal_identity: str,
    ) -> None:
        """Remember every direction seen at the lane's latest timestamp.

        The set is naturally bounded: when the lane advances to a newer candle it is
        replaced, and for the current union engines it contains at most BUY and SELL.
        """
        if not hasattr(scanner, "_last_signal_time"):
            scanner._last_signal_time = {}
        if not hasattr(scanner, "_last_signal_identity"):
            scanner._last_signal_identity = {}

        signal_check = (
            signal_ts.replace(tzinfo=None)
            if getattr(signal_ts, "tzinfo", None)
            else signal_ts
        )
        current_ts = scanner._last_signal_time.get(dedup_key) or datetime.min
        current_check = (
            current_ts.replace(tzinfo=None)
            if getattr(current_ts, "tzinfo", None)
            else current_ts
        )
        identities = cls._live_signal_identities(
            scanner._last_signal_identity.get(dedup_key)
        )
        if signal_check > current_check:
            identities.clear()
        if signal_check >= current_check:
            identities.add(signal_identity)
            scanner._last_signal_identity[dedup_key] = identities
            scanner._last_signal_time[dedup_key] = max(current_check, signal_check)

    @classmethod
    def _forget_live_signal_identity(
        cls,
        scanner,
        dedup_key: str,
        signal_ts: datetime,
        signal_identity: str,
    ) -> bool:
        """Withdraw only the repainted identity without reopening its opposite peer."""
        identity_store = getattr(scanner, "_last_signal_identity", {})
        identities = cls._live_signal_identities(identity_store.get(dedup_key))
        if signal_identity not in identities:
            return False

        identities.discard(signal_identity)
        if identities:
            identity_store[dedup_key] = identities
            return True

        identity_store.pop(dedup_key, None)
        signal_check = (
            signal_ts.replace(tzinfo=None)
            if getattr(signal_ts, "tzinfo", None)
            else signal_ts
        )
        last_times = getattr(scanner, "_last_signal_time", {})
        current_ts = last_times.get(dedup_key) or datetime.min
        current_check = (
            current_ts.replace(tzinfo=None)
            if getattr(current_ts, "tzinfo", None)
            else current_ts
        )
        if current_check == signal_check:
            last_times[dedup_key] = signal_check - timedelta(microseconds=1)
        return True

    @classmethod
    def _live_signal_recheck_due(
        cls,
        signal_ts: datetime,
        timeframe: str,
        now: datetime,
        eligible: bool,
        pending: bool,
        last_recheck: datetime,
        interval_seconds: float = 5.0,
    ) -> bool:
        """Re-evaluate a rejected arrow while forming and once after candle close."""
        if not eligible or pending:
            return False
        now_check = now.replace(tzinfo=None) if getattr(now, "tzinfo", None) else now
        last_check = (
            last_recheck.replace(tzinfo=None)
            if getattr(last_recheck, "tzinfo", None)
            else last_recheck
        )
        return (now_check - last_check).total_seconds() >= max(
            1.0,
            float(interval_seconds or 5.0),
        )

    @staticmethod
    def _futures_stop_distance(
        natural_stop_distance: float,
        reference_price: float,
        futures_sl_pct: float,
        regime: str,
    ) -> float:
        """Cap a natural FUT stop without ever widening it."""
        natural_stop = max(0.0, float(natural_stop_distance or 0.0))
        effective_pct = futures_stop_tolerance_pct(futures_sl_pct, regime)
        regime_cap = max(0.0, float(reference_price or 0.0)) * (effective_pct / 100.0)
        return min(natural_stop, regime_cap)

    @staticmethod
    def _futures_stop_distance_rejection(
        natural_stop_distance: float,
        spot: float,
        futures_sl_pct: float,
        safety_multiplier: float = 1.5,
        regime: str = "UNKNOWN",
    ) -> Optional[str]:
        """Explain when a natural FUT stop exceeds the hard safety envelope.

        The traded stop is separately capped at the regime-selected tolerance,
        with futures_sl_pct as the absolute ceiling (see caller), so
        this gate only rejects when the NATURAL ATR stop is more than
        `safety_multiplier` x that cap — i.e. our forced stop would sit so far
        inside the ATR that whipsaw risk is high. The multiplier is configurable
        (`futures_sl_safety_multiplier`) because a too-tight value rejects every
        FUT entry in higher-volatility regimes where ATR stops naturally run wide.
        """
        mult = max(1.0, float(safety_multiplier or 1.5))
        effective_pct = futures_stop_tolerance_pct(futures_sl_pct, regime)
        allowed_stop = float(spot or 0.0) * (effective_pct / 100.0)
        hard_limit = allowed_stop * mult
        natural_stop = float(natural_stop_distance or 0.0)
        if hard_limit > 0.0 and natural_stop > hard_limit:
            return (
                f"Natural FUT stop {natural_stop:.2f} pts exceeds "
                f"{hard_limit:.2f} pts safety limit "
                f"({str(regime or 'UNKNOWN').upper()} tolerance {effective_pct:.3f}%, "
                f"configured cap {float(futures_sl_pct or 0.0):.3f}% x {mult:g})"
            )
        return None

    @staticmethod
    def _contract_ltp(scanner, exchange: str, symbol: str, token: str) -> float:
        """Use the same quote abstraction for option and futures contracts."""
        if not scanner or not getattr(scanner, "data", None) or not symbol or not token:
            return 0.0
        try:
            price = scanner.data.get_ltp(exchange, symbol, token)
            return float(price or 0.0)
        except Exception as exc:
            logger.debug(f"Contract LTP fetch failed for {symbol}: {exc}")
            return 0.0

    @staticmethod
    def _grade_rank(grade: str) -> int:
        return grade_rank(grade)

    @staticmethod
    def _grade_min_confidence(grade: str) -> float:
        return grade_min_confidence(grade)

    @staticmethod
    def _option_signed_delta(spot, strike, option_type, intel_result, dte, fallback):
        """Black-Scholes signed delta for the ACTUAL strike (fixes ITM/OTM options
        being assigned the flat ATM ~0.5 delta). Falls back to the ATM/±0.5 estimate."""
        if dte is None:
            return fallback
        try:
            from intelligence.greeks_engine import GreeksEngine
            greeks = (intel_result or {}).get("greeks", {}) or {}
            leg = "call" if str(option_type or "").upper() == "CE" else "put"
            iv = float(greeks.get(leg, {}).get("iv", 0.0) or 0.0)
            if iv > 1.5:          # greeks store IV as a percentage
                iv = iv / 100.0
            if iv <= 0:
                iv = 0.15
            days = max(float(dte or 0.0), 0.25)   # keep 0DTE meaningful
            d = float(GreeksEngine().calculate(float(spot), float(strike), days, iv, leg).get("delta", 0.0))
            if leg == "put" and d > 0:
                d = -d
            if abs(d) < 0.05:
                return fallback
            return d
        except Exception:
            return fallback

    @staticmethod
    def _observed_option_stop_context(option_df, signal_time, spot, strike, option_type, underlying_stop_distance):
        """Map a spot stop using the expiry of the observed option, not today's master."""
        from engine.expiry_manager import ArchivedExpiryContext
        attrs = dict(getattr(option_df, "attrs", {}) or {})
        expiry = attrs.get("contract_expiry")
        if not expiry or not attrs.get("contract_verified"):
            return None
        context = ArchivedExpiryContext().resolve(
            "", signal_time, contract_expiry=expiry, contract_verified=True)
        if not context["available"]:
            return None
        delta = SignalProcessor._option_signed_delta(
            spot, strike, option_type, {}, context["days_to_expiry"],
            0.5 if str(option_type).upper() == "CE" else -0.5)
        distance = abs(float(underlying_stop_distance))
        return {"expiry_context": context, "natural_stop_delta": delta,
                "natural_stop_delta_source": "estimated_black_scholes_iv_15pct", "natural_stop_iv": 0.15,
                "underlying_stop_distance": distance, "natural_sl_distance": abs(distance * delta)}

    @staticmethod
    def _power_move_confirmed(*, volume_ratio, range_atr_ratio, close_location, adx_value, intel_score, trend_15m_agrees, settings, signal_type="") -> bool:
        return power_move_confirmed(
            signal_type=signal_type,
            volume_ratio=volume_ratio,
            range_atr_ratio=range_atr_ratio,
            close_location=close_location,
            adx_value=adx_value,
            intel_score=intel_score,
            trend_15m_agrees=trend_15m_agrees,
            settings=settings,
        )

    @staticmethod
    def _volume_spike_entry_confirmed(
        *,
        timeframe,
        volume_ratio,
        range_atr_ratio,
        close_location,
        adx_value=0.0,
        intel_score=0.0,
        power_move=False,
        settings,
        signal_type: str = "",
        depth_usable: bool = False,
        ofr: float = 0.0,
        volume_basis: str = "TRADED",
        surge_percentile: float = -1.0,
        instrument: str = "",
    ) -> bool:
        return volume_spike_entry_confirmed(
            timeframe=timeframe,
            volume_ratio=volume_ratio,
            range_atr_ratio=range_atr_ratio,
            close_location=close_location,
            adx_value=adx_value,
            intel_score=intel_score,
            power_move=power_move,
            settings=settings,
            signal_type=signal_type,
            depth_usable=depth_usable,
            ofr=ofr,
            volume_basis=volume_basis,
            surge_percentile=surge_percentile,
            instrument=instrument,
        )

    @staticmethod
    def _volume_spike_requirement_text(settings, instrument: str = "") -> str:
        """Human-readable fast-entry threshold using the unit actually measured."""
        traded_floor = float(volume_spike_min_ratio(settings, instrument) or 0.0)
        if _setting_enabled(
            getattr(settings, "one_min_scalping_high_burst_only", True), True
        ):
            traded_floor = max(
                traded_floor,
                float(
                    getattr(
                        settings,
                        "volume_spike_entry_strong_1min_min_volume_ratio",
                        traded_floor,
                    )
                    or traded_floor
                ),
            )
        tick_floor = float(
            getattr(settings, "surge_tick_percentile_floor", 97.0) or 97.0
        )
        return (
            f"traded volume >= {traded_floor:.2f}x or "
            f"tick activity >= p{tick_floor:g} (with range/close and burst context)"
        )

    @staticmethod
    def _volume_spike_metric_text(
        metrics: Optional[Dict[str, Any]],
        settings,
        instrument: str = "",
        timeframe: str = "",
    ) -> str:
        """Describe spike evidence without calling tick activity traded volume."""
        evidence = dict(metrics or {})
        basis = str(evidence.get("volume_basis") or "TRADED").upper()
        if basis == "TICK_ACTIVITY":
            percentile = float(evidence.get("surge_percentile", -1.0) or -1.0)
            floor = float(
                getattr(settings, "surge_tick_percentile_floor", 97.0) or 97.0
            )
            return f"tick activity p{percentile:.1f} (floor p{floor:g})"
        ratio = float(evidence.get("volume_ratio", 0.0) or 0.0)
        floor = float(volume_spike_min_ratio(settings, instrument) or 0.0)
        if str(timeframe or "").lower() == "1min" and _setting_enabled(
            getattr(settings, "one_min_scalping_high_burst_only", True), True
        ):
            floor = max(
                floor,
                float(
                    getattr(
                        settings,
                        "volume_spike_entry_strong_1min_min_volume_ratio",
                        floor,
                    )
                    or floor
                ),
            )
        return f"traded volume {ratio:.2f}x (floor {floor:.2f}x)"

    @staticmethod
    def _recheckable_rejection_status(status_msg: str) -> bool:
        """Return whether a forming rejection may improve before candle close."""
        status = str(status_msg or "")
        return status in {
            "REJECTED - Quality Gate",
            "REJECTED - 5min Quality",
            "REJECTED - Low Confidence",
            "REJECTED - 5min Choppy Confirmation",
            "REJECTED - Opt Gate",
            "REJECTED - FUT Stop Distance",
            "REJECTED - Weak 1min",
            # An opposite signal exits the existing trade THIS cycle but is blocked
            # from entering in the same cycle. It must stay recheck-eligible so the
            # NEXT cycle re-evaluates it (the existing trade now gone) and the
            # reversal entry can fire — otherwise the dedup lock orphans it forever.
            "REJECTED - Opposite Signal Wait",
            # Same statement, other path. quad_prime — the SHIPPED engine — refuses the
            # reverse entry while a live-book trade is open on the far side, and it did
            # so with a bare _emit_signal_decision + `return []`, which arms nothing. The
            # dedup lock is already advanced by then, so the flip was presented once and
            # dropped. 56 flips died that way on four sessions (2026-07-09 x24,
            # 07-31 x16, 08-19 x13, 07-10 x3); on 2026-07-09 one NIFTY trade refused
            # every SELL flip from 10:35 onward and no reversal was ever taken.
            "REJECTED - Existing Opposite Trade",
        } or status.startswith("REJECTED - 5Min ")

    @staticmethod
    def _is_live_book_trade(trade) -> bool:
        """True when an open/closed row is real book (not ghost/hist/replay contamination)."""
        if trade is None:
            return False
        if bool(getattr(trade, "is_ghost", False)):
            return False
        tid = str(getattr(trade, "id", "") or "")
        if tid.startswith("H_") or tid.startswith("EOD_"):
            return False
        if str(getattr(trade, "trading_symbol", "") or "").upper() == "FUT_SHADOW":
            return False
        prov = str(getattr(trade, "data_provenance", "") or "").upper()
        if prov in {
            "HISTORICAL",
            "HISTORY",
            "HIST",
            "BACKTEST",
            "REPLAY",
            "SIM",
            "SIMULATION",
            "PAPER_HIST",
        }:
            return False
        return True

    @staticmethod
    def _is_s12_native_burst(signal) -> bool:
        """True only for a native D2 burst; it must not impersonate the host spike stop."""
        return bool(
            s12_lane_policy.lane_of(signal) == "D2"
            and getattr(signal, "s12_burst", False)
        )

    @staticmethod
    def _has_1min_burst_or_spike(signal) -> bool:
        """The host's measured spike or S1+S2's own confirmed D2 burst."""
        return bool(
            getattr(signal, "volume_spike_entry", False)
            or SignalProcessor._is_s12_native_burst(signal)
        )

    @staticmethod
    def _suppress_1min_no_spike_tape(tf, volume_spike_entry, settings, *, native_burst=False) -> bool:
        """Whether to keep a 1min signal off the live tape because it never cleared
        its configured traded-volume or tick-percentile first filter. S1+S2's native D2
        burst is an equivalent explicit fast-entry route, but keeps its own stop policy."""
        return bool(
            str(tf or "") == "1min"
            and not bool(volume_spike_entry)
            and not bool(native_burst)
            and _setting_enabled(getattr(settings, "live_1min_log_only_volume_spike", True), True)
        )

    @staticmethod
    def _should_log_tf_select_diagnostic(best_tf, best_sig, scored, settings, trade_involved: bool = False) -> bool:
        """Log TF selection only when it is useful, not every ordinary 1min churn tick."""
        if trade_involved:
            return True
        if str(best_tf or "") != "1min":
            return len(scored or []) > 1 or str(best_tf or "") == "15min"
        if SignalProcessor._has_1min_burst_or_spike(best_sig):
            return True
        if bool(getattr(best_sig, "power_move", False)):
            return True
        return not _setting_enabled(getattr(settings, "live_1min_log_only_volume_spike", True), True)

    @staticmethod
    def _entry_admissible_raw_candidates(
        candidates_raw,
        *,
        reversion_window_open: bool = False,
        plain_one_min: bool = False,
    ):
        """Keep 1min rows from competing unless an exceptional route exists.

        - ``plain_one_min``: the TF toggle is 1M alone (2026-09-26 user), so a plain
          1m UT flip is an entry path. The 1m combinations take fast 1m rows only.
        - Continuous 1M SPIKE path: requires volume_spike_entry.
        - POST-WIN / distress SNAPBACK path: may admit 1min without spike while the
          reversion window is open (REV-SCLP ON, 1M-SCALP may be OFF).
        - Confirmed shock, reversal, reclaim and cascade flags may enter their own
          downstream state-specific checks.
        """
        allow_reversion_1min = bool(reversion_window_open)
        return [
            c
            for c in list(candidates_raw or [])
            if str(c.get("tf", "")) != "1min"
            or bool(plain_one_min)
            or SignalProcessor._has_1min_burst_or_spike(c.get("sig"))
            or allow_reversion_1min
            or bool(getattr(c.get("sig"), "market_ltp_shock_entry", False))
            or bool(getattr(c.get("sig"), "market_wide_reversal_entry", False))
            or bool(getattr(c.get("sig"), "late_session_reclaim_entry", False))
            or bool(getattr(c.get("sig"), "one_way_cascade_entry", False))
        ]

    @staticmethod
    def _tag_one_min_fast_contexts(scanner, instrument, candidates_raw) -> None:
        """Give an untagged 1m row its reclaim / cascade evidence before admission.

        Unless the TF toggle is 1M alone, a 1m row is admitted only with a burst,
        snapback or reversal tag; the reclaim and cascade tags, though, were computed after
        selection and on the winner only, so a 1m reclaim or cascade could never be
        selected (2026-09-26). The detectors carry their own switches and windows.
        """
        for c in list(candidates_raw or []):
            sig = c.get("sig")
            if str(c.get("tf") or "") != "1min" or sig is None:
                continue
            if bool(getattr(sig, "late_session_reclaim_entry", False)) or bool(
                getattr(sig, "one_way_cascade_entry", False)
            ):
                continue
            side = str(getattr(sig, "signal_type", "") or "").upper()
            for flag, fn_name in (
                ("late_session_reclaim", "_late_session_reclaim_context"),
                ("one_way_cascade", "_one_way_cascade_context"),
            ):
                fn = getattr(scanner, fn_name, None)
                if not callable(fn):
                    continue
                try:
                    ctx = fn(instrument, "1min", side, getattr(sig, "timestamp", None))
                except Exception as exc:
                    logger.debug(f"1m {flag} check skipped for {instrument}: {exc}")
                    ctx = None
                if isinstance(ctx, dict) and ctx:
                    try:
                        setattr(sig, f"{flag}_entry", True)
                        setattr(sig, f"{flag}_context", dict(ctx))
                    except Exception:
                        pass
                    break

    @staticmethod
    def _evaluate_reversion_scalp(
        scanner,
        settings,
        *,
        instrument: str,
        best_tf: str,
        best_sig,
    ):
        """Decide POST-WIN 3M SNAPBACK eligibility for this 1min candidate.

        Window arms only after 5min/15min exits (not 1min scalps). Pure snapback
        does not require the full configured spike threshold but MUST pass bare-min tape data
        (volume/range/close) so entries are data-backed, not hunch-only.

        Returns (is_reversion_scalp: bool, meta: dict).
        """
        meta = {
            "minutes_since": 0.0,
            "view": "",
            "reason": "",
            "move_pct": 0.0,
            "source": "",
            "tape_ok": False,
        }
        if not _setting_enabled(getattr(settings, "post_exit_reversion_enabled", False), False):
            return False, meta
        if str(best_tf or "") != "1min" or best_sig is None:
            return False, meta
        base = str(instrument or "").split()[0].upper()
        _rev_info = (getattr(scanner, "_last_major_exits", {}) or {}).get(base)
        if not isinstance(_rev_info, dict):
            _rev_info = (getattr(scanner, "_last_major_exits", {}) or {}).get(instrument)
        last_exit_time = (
            _rev_info.get("time") if isinstance(_rev_info, dict) else None
        )
        last_exit_view = (
            str(_rev_info.get("direction") or "").upper()
            if isinstance(_rev_info, dict)
            else ""
        )
        _rev_reason = (
            str(_rev_info.get("reason") or "")
            if isinstance(_rev_info, dict)
            else ""
        )
        _rev_source = (
            str(_rev_info.get("source") or "")
            if isinstance(_rev_info, dict)
            else ""
        )
        stored_move = 0.0
        if isinstance(_rev_info, dict):
            try:
                stored_move = float(_rev_info.get("move_pct", 0.0) or 0.0)
            except Exception:
                stored_move = 0.0
        if last_exit_time is None and getattr(
            scanner, "_last_major_exit_instrument", None
        ) == base:
            last_exit_time = getattr(scanner, "_last_major_exit_time", None)
        if last_exit_time is None:
            return False, meta
        _now_ist = datetime.now(IST).replace(tzinfo=None)
        if getattr(last_exit_time, "tzinfo", None) is not None:
            last_exit_time = last_exit_time.astimezone(IST).replace(tzinfo=None)
        _rev_window = float(
            getattr(settings, "post_exit_reversion_window_minutes", 3.0) or 3.0
        )
        minutes_since_exit = (_now_ist - last_exit_time).total_seconds() / 60.0
        counter_trend = True
        if last_exit_view in {"LONG", "SHORT"}:
            _expected_rev = "SELL" if last_exit_view == "LONG" else "BUY"
            counter_trend = str(best_sig.signal_type).upper() == _expected_rev
        if not (0.0 <= minutes_since_exit <= max(0.5, _rev_window) and counter_trend):
            return False, meta

        # Size-of-event: 0.30%+/5m OR 0.40%+/10m (stored at arm, else live).
        qualifies = True
        move_info: dict = {"move_pct": float(stored_move)}
        if callable(getattr(scanner, "_index_event_move_qualifies", None)):
            try:
                qualifies, move_info = scanner._index_event_move_qualifies(
                    base,
                    as_of=_now_ist,
                    stored=_rev_info if isinstance(_rev_info, dict) else None,
                )
            except Exception:
                qualifies = stored_move >= float(
                    getattr(settings, "post_exit_reversion_fast_move_pct", 0.30) or 0.30
                )
                move_info = {"move_pct": float(stored_move)}
        elif stored_move > 0:
            thr = float(getattr(settings, "post_exit_reversion_fast_move_pct", 0.30) or 0.30)
            qualifies = thr <= 0 or stored_move + 1e-9 >= thr
        meta.update(
            {
                "minutes_since": float(minutes_since_exit),
                "view": last_exit_view,
                "reason": _rev_reason,
                "move_pct": float(move_info.get("move_pct", stored_move) or 0.0),
                "move_5m_pct": float(move_info.get("move_5m_pct", 0.0) or 0.0),
                "move_10m_pct": float(move_info.get("move_10m_pct", 0.0) or 0.0),
                "source": _rev_source,
                "band": str(move_info.get("band") or ""),
                "source_tf": str(
                    (_rev_info or {}).get("source_tf") if isinstance(_rev_info, dict) else ""
                ),
            }
        )
        if not qualifies:
            logger.info(
                f"🔄 [REVERSION SCALP] {base} skipped: "
                f"5m={meta.get('move_5m_pct', 0):.2f}% / "
                f"10m={meta.get('move_10m_pct', 0):.2f}% "
                f"(need 0.30%/5m OR 0.40%/10m)"
            )
            return False, meta

        # Bare-min tape confirm: full SPIKE already qualifies; pure SNAPBACK needs
        # directional thrust floors (configurable, below the full spike gate) so we do not
        # fire on a naked UT1 flip with zero volume/range evidence.
        if bool(getattr(best_sig, "volume_spike_entry", False)):
            meta["tape_ok"] = True
            meta["tape_path"] = "volume_spike"
            return True, meta

        min_vol = float(getattr(settings, "post_exit_reversion_min_volume_ratio", 1.5) or 1.5)
        min_rng = float(getattr(settings, "post_exit_reversion_min_range_atr_ratio", 1.0) or 1.0)
        min_close = float(getattr(settings, "post_exit_reversion_min_close_location", 0.55) or 0.55)
        vol_r = rng_r = close_loc = 0.0
        thrust = None
        try:
            thrust_fn = getattr(scanner, "_candle_thrust_metrics", None)
            if callable(thrust_fn):
                thrust = thrust_fn(
                    base,
                    "1min",
                    str(best_sig.signal_type or ""),
                    as_of=getattr(best_sig, "timestamp", None) or _now_ist,
                )
        except Exception:
            thrust = None
        if isinstance(thrust, dict):
            try:
                vol_r = float(thrust.get("volume_ratio", 0.0) or 0.0)
                rng_r = float(thrust.get("range_atr_ratio", 0.0) or 0.0)
                close_loc = float(thrust.get("close_location", 0.0) or 0.0)
            except Exception:
                vol_r = rng_r = close_loc = 0.0
        tape_ok = (
            vol_r + 1e-12 >= min_vol
            and rng_r + 1e-12 >= min_rng
            and close_loc + 1e-12 >= min_close
        )
        meta.update(
            {
                "tape_ok": bool(tape_ok),
                "tape_path": "bare_min" if tape_ok else "fail",
                "tape_vol": float(vol_r),
                "tape_range": float(rng_r),
                "tape_close": float(close_loc),
            }
        )
        if not tape_ok:
            logger.info(
                f"🔄 [REVERSION SCALP] {base} skipped: bare-min tape fail "
                f"vol={vol_r:.2f}x (need>={min_vol:.2f}) "
                f"rng={rng_r:.2f}x (need>={min_rng:.2f}) "
                f"close={close_loc:.0%} (need>={min_close:.0%})"
            )
            return False, meta
        return True, meta

    @staticmethod
    def _select_cascade_wave_scored_rows(
        scored,
        settings,
        scanner_engine_mode: str = "",
        instrument: str = "",
    ):
        """Pick multi-TF cascade stack.

        Core (NIFTY/BN/SENSEX): primary 1m/5m/15m + dual 5m/15m lane-B.
        MIDCAP: solo best 2 of 1/5/15; dual = best 2 main + best 1 dual of 5/15 (2+1).
        """
        if not scored:
            return []
        parts = str(instrument or "").split()
        inst = parts[0].upper() if parts else ""
        core = {
            str(x or "").upper()
            for x in (getattr(settings, "cascade_multi_tf_core_indices", None) or [
                "NIFTY", "BANKNIFTY", "SENSEX"
            ])
        }
        # Empty instrument → core full stack (default / unit tests)
        is_midcap = bool(inst) and (inst == "MIDCPNIFTY" or inst not in core)

        primary_by_tf = {}
        dual_by_tf = {}
        for row in scored:
            score, sig, tf, grade, conf, reasons, after = row
            lane = SignalProcessor._signal_source_lane(sig, scanner_engine_mode)
            tf_key = str(tf or "")
            if lane == "B" and tf_key in {"5min", "15min"}:
                prev = dual_by_tf.get(tf_key)
                if prev is None or score > prev[0]:
                    dual_by_tf[tf_key] = row
                continue
            if tf_key in {"1min", "5min", "15min"}:
                prev = primary_by_tf.get(tf_key)
                if prev is None or score > prev[0]:
                    primary_by_tf[tf_key] = row

        if is_midcap:
            solo_max = max(1, int(getattr(settings, "cascade_midcap_solo_max_tf", 2) or 2))
            dual_main_max = max(1, int(getattr(settings, "cascade_midcap_dual_main_max", 2) or 2))
            dual_addon_max = max(0, int(getattr(settings, "cascade_midcap_dual_addon_max", 1) or 1))
            main_ranked = sorted(primary_by_tf.values(), key=lambda r: r[0], reverse=True)
            if dual_by_tf:
                main = main_ranked[:dual_main_max]
                dual_ranked = sorted(dual_by_tf.values(), key=lambda r: r[0], reverse=True)
                dual = dual_ranked[:dual_addon_max]
                selected = main + dual
                selected = SignalProcessor._apply_scored_dual_clone_reduction(
                    selected, settings
                )
            else:
                selected = main_ranked[:solo_max]
            return selected if selected else [scored[0]]

        primary_max = max(1, int(getattr(settings, "cascade_primary_max_tf", 3) or 3))
        dual_max = max(0, int(getattr(settings, "cascade_dual_addon_max", 2) or 2))
        order = ["15min", "5min", "1min"]
        primary = [primary_by_tf[tf] for tf in order if tf in primary_by_tf][:primary_max]
        dual = [dual_by_tf[tf] for tf in ("5min", "15min") if tf in dual_by_tf][:dual_max]
        selected = primary + dual
        # Dual clone reduction: same TF main+B within 1m → keep first only
        selected = SignalProcessor._apply_scored_dual_clone_reduction(selected, settings)
        return selected if selected else [scored[0]]

    @staticmethod
    def _apply_scored_dual_clone_reduction(selected_rows, settings):
        """Same-TF dual engine rows within 1m → keep earliest only (first wins).

        selected_rows: list of scored tuples (score, sig, tf, grade, conf, reasons, after).
        """
        if not selected_rows:
            return selected_rows
        if not _setting_enabled(
            getattr(settings, "dual_clone_reduction_enabled", True), True
        ):
            return selected_rows
        win = max(
            0.0,
            float(getattr(settings, "dual_clone_reduction_window_seconds", 60) or 60),
        )

        def _ts(row):
            sig = row[1]
            t = getattr(sig, "timestamp", None)
            if t is None:
                return None
            if getattr(t, "tzinfo", None) is not None:
                try:
                    return t.replace(tzinfo=None)
                except Exception:
                    return t
            return t

        by_tf = {}
        for row in selected_rows:
            tf = str(row[2] or "")
            by_tf.setdefault(tf, []).append(row)

        keep = set()
        for tf, rows in by_tf.items():
            if len(rows) <= 1:
                keep.add(id(rows[0]))
                continue
            ordered = sorted(
                rows,
                key=lambda r: (_ts(r) is None, _ts(r) or datetime.max),
            )
            first = ordered[0]
            keep.add(id(first))
            first_ts = _ts(first)
            for other in ordered[1:]:
                other_ts = _ts(other)
                if first_ts is None or other_ts is None:
                    # missing stamps + same TF → clone, drop
                    continue
                if abs((first_ts - other_ts).total_seconds()) <= win:
                    continue  # drop clone
                keep.add(id(other))
        return [r for r in selected_rows if id(r) in keep]

    def _clone_cascade_tf_sibling(
        self,
        winner: "TradeCandidate",
        *,
        sig,
        tf: str,
        grade: str,
        confidence: float,
        reasons,
        scanner_engine_mode: str,
    ) -> Optional["TradeCandidate"]:
        """Materialize an extra cascade TF/lane entry from an already-approved winner.

        Reuses winner book (inst_type/lots/stops policy already cascade-sized) but
        retargets TF, signal stamp, grade, and source lane so concurrency + mixed
        book can keep the multi-TF stack.
        """
        try:
            sibling = copy.copy(winner)
        except Exception:
            return None
        direction = "LONG" if str(getattr(sig, "signal_type", "") or "").upper() == "BUY" else "SHORT"
        if direction != str(getattr(winner, "direction", "") or "").upper():
            return None
        sibling.timeframe = str(tf or winner.timeframe)
        sibling.grade = grade or winner.grade
        sibling.confidence = float(confidence or 0.0)
        sibling.score = float(confidence or 0.0)
        sibling.signal_timestamp = getattr(sig, "timestamp", None) or winner.signal_timestamp
        # The sibling is a different setup. A shallow copy must not regrade it from
        # the winner's raw candle/ADX/timeframe; absent own inputs mean no regrade.
        sibling.grading_inputs = copy.deepcopy(getattr(sig, "grading_inputs", None))
        sibling.reasons = list(reasons or []) + [
            f"Cascade multi-TF stack sibling ({sibling.timeframe})"
        ]
        sibling.one_way_cascade_entry = True
        src_engine = str(getattr(sig, "source_engine", "") or "").lower()
        src_engines = [
            str(src).lower()
            for src in (getattr(sig, "source_engines", None) or [])
            if str(src or "").strip()
        ]
        lane = self._signal_source_lane(sig, scanner_engine_mode)
        setattr(sibling, "source_engine", src_engine or getattr(winner, "source_engine", ""))
        setattr(sibling, "source_engines", src_engines or list(getattr(winner, "source_engines", []) or []))
        setattr(sibling, "source_lane", lane)
        setattr(sibling, "source_signal_type", getattr(sig, "signal_type", ""))
        setattr(sibling, "source_signal_price", float(getattr(sig, "price", 0.0) or 0.0))
        setattr(sibling, "source_signal_bar_index", int(getattr(sig, "bar_index", -1) or -1))
        setattr(sibling, "dual_lane_entry", lane == "B")
        if lane == "B":
            try:
                fraction = max(
                    0.05,
                    min(1.0, float(getattr(get_settings(), "dual_lane_size_fraction", 0.65) or 0.65)),
                )
                base_lots = max(1, int(getattr(winner, "lots", 1) or 1))
                sibling.lots = max(1, int(base_lots * fraction))
            except Exception:
                pass
        # Spot/signal price for FUT siblings: use signal price when available
        if str(getattr(sibling, "inst_type", "") or "").upper() == "FUT":
            px = float(getattr(sig, "price", 0.0) or 0.0)
            if px > 0:
                risk = abs(float(winner.price or 0.0) - float(winner.stop or 0.0))
                rr = float(getattr(winner, "rr", 1.5) or 1.5)
                sibling.price = px
                if direction == "LONG":
                    sibling.stop = px - risk if risk > 0 else float(winner.stop or 0.0)
                    sibling.target = px + risk * rr if risk > 0 else float(winner.target or 0.0)
                else:
                    sibling.stop = px + risk if risk > 0 else float(winner.stop or 0.0)
                    sibling.target = px - risk * rr if risk > 0 else float(winner.target or 0.0)
                sibling.initial_stop = float(sibling.stop or 0.0)
        return sibling

    def _emit_rare_dual_tf_siblings(
        self,
        winner: "TradeCandidate",
        scored,
        settings,
        scanner_engine_mode: str,
    ) -> List["TradeCandidate"]:
        """Rare dual fill: best 1m (spike/snapback) + best 5m/15m when both excellent.

        One sibling max. Prefer OPT for the second leg so we never open 2x FUT
        on the same index. Cascade multi-TF is separate (already stacks more).
        """
        if not _setting_enabled(getattr(settings, "rare_dual_tf_entry_enabled", True), True):
            return []
        if bool(getattr(winner, "one_way_cascade_entry", False)):
            return []  # cascade path owns multi-TF stacking
        min_grade = str(getattr(settings, "rare_dual_tf_min_grade", "A") or "A").upper()
        min_conf = float(getattr(settings, "rare_dual_tf_min_confidence", 0.70) or 0.70)
        min_rank = self._grade_rank(min_grade)
        win_tf = str(getattr(winner, "timeframe", "") or "")
        win_dir = str(getattr(winner, "direction", "") or "").upper()
        win_grade = str(getattr(winner, "grade", "") or "")
        win_conf = float(getattr(winner, "confidence", 0.0) or 0.0)
        if self._grade_rank(win_grade) < min_rank or win_conf + 1e-9 < min_conf:
            return []
        # Need a fast leg (1m spike/snapback) and a structure leg (5/15).
        win_is_fast = win_tf == "1min" and (
            bool(getattr(winner, "volume_spike_entry", False))
            or bool(getattr(winner, "is_reversion_scalp", False))
            or bool(getattr(winner, "market_ltp_shock_entry", False))
        )
        win_is_structure = win_tf in {"5min", "15min"}
        if not (win_is_fast or win_is_structure):
            return []

        best_row = None
        best_score = -1.0
        for row in list(scored or []):
            score, sig, tf, grade, conf, reasons, _after = row
            tf_s = str(tf or "")
            if tf_s == win_tf:
                continue
            sig_dir = (
                "LONG"
                if str(getattr(sig, "signal_type", "") or "").upper() == "BUY"
                else "SHORT"
            )
            if sig_dir != win_dir:
                continue
            if self._grade_rank(grade) < min_rank:
                continue
            if float(conf or 0.0) + 1e-9 < min_conf:
                continue
            if win_is_fast and tf_s not in {"5min", "15min"}:
                continue
            if win_is_structure:
                if tf_s != "1min":
                    continue
                # Companion 1m must be spike / power / shock quality.
                if not (
                    bool(getattr(sig, "volume_spike_entry", False))
                    or bool(getattr(sig, "market_ltp_shock_entry", False))
                    or bool(getattr(sig, "power_move", False))
                ):
                    continue
            if float(score or 0.0) > best_score:
                best_score = float(score or 0.0)
                best_row = row
        if best_row is None:
            return []
        _sc, sig, tf, grade, conf, reasons, _af = best_row
        sib = self._clone_cascade_tf_sibling(
            winner,
            sig=sig,
            tf=str(tf),
            grade=str(grade or ""),
            confidence=float(conf or 0.0),
            reasons=list(reasons or []),
            scanner_engine_mode=scanner_engine_mode,
        )
        if sib is None:
            return []
        # Never dual FUT on same index — second leg becomes OPT when preferred.
        prefer_opt = _setting_enabled(
            getattr(settings, "rare_dual_tf_prefer_opt_second", True), True
        )
        win_type = str(getattr(winner, "inst_type", "FUT") or "FUT").upper()
        if prefer_opt or win_type == "FUT":
            if str(getattr(sib, "inst_type", "") or "").upper() == "FUT":
                # Mark for coordinator; force OPT book on sibling.
                sib.inst_type = "OPT"
                if not str(getattr(sib, "option_type", "") or ""):
                    sib.option_type = "CE" if win_dir == "LONG" else "PE"
        setattr(sib, "rare_dual_tf_entry", True)
        setattr(sib, "one_way_cascade_entry", False)
        # Strip cascade-only reason text if cloned from cascade helper.
        cleaned = [
            r
            for r in list(sib.reasons or [])
            if "Cascade multi-TF stack sibling" not in str(r)
        ]
        sib.reasons = cleaned + [
            f"Rare dual-TF fill: {win_tf}+{sib.timeframe} both ≥{min_grade}/{min_conf:.0%}"
        ]
        logger.info(
            f"🌟 Rare dual-TF: {getattr(winner, 'instrument', '')} "
            f"{win_tf}({win_grade}/{win_conf:.2f}) + "
            f"{sib.timeframe}({grade}/{float(conf or 0):.2f})"
        )
        return [sib]

    def _emit_cascade_tf_siblings(
        self,
        winner: "TradeCandidate",
        scored,
        settings,
        scanner_engine_mode: str,
    ) -> List["TradeCandidate"]:
        """After primary winner, emit remaining cascade multi-TF / dual siblings."""
        if not bool(getattr(winner, "one_way_cascade_entry", False)):
            return []
        if not _setting_enabled(getattr(settings, "cascade_multi_tf_stack_enabled", True), True):
            return []
        wave = self._select_cascade_wave_scored_rows(
            scored,
            settings,
            scanner_engine_mode,
            instrument=str(getattr(winner, "instrument", "") or ""),
        )
        if len(wave) <= 1:
            return []
        win_tf = str(getattr(winner, "timeframe", "") or "")
        win_lane = str(getattr(winner, "source_lane", "") or "").upper() or "A"
        siblings: List[TradeCandidate] = []
        seen = {(win_tf, win_lane)}
        for row in wave:
            _score, sig, tf, grade, conf, reasons, _after = row
            lane = self._signal_source_lane(sig, scanner_engine_mode) or "A"
            key = (str(tf or ""), str(lane or "A").upper())
            if key in seen:
                continue
            # Same TF primary as winner → already materialised
            if str(tf or "") == win_tf and key[1] == win_lane:
                continue
            sib = self._clone_cascade_tf_sibling(
                winner,
                sig=sig,
                tf=str(tf or ""),
                grade=grade,
                confidence=conf,
                reasons=reasons,
                scanner_engine_mode=scanner_engine_mode,
            )
            if sib is None:
                continue
            seen.add(key)
            siblings.append(sib)
        return siblings

    @staticmethod
    def _volume_spike_futures_stop_pct(configured_fut_sl_pct: float, settings) -> float:
        """Use the tighter of the spike cap and the user's global FUT stop setting."""
        spike_cap = max(
            0.0,
            float(
                getattr(settings, "volume_spike_entry_futures_sl_pct", 0.05)
                or 0.05
            ),
        )
        configured_cap = max(0.0, float(configured_fut_sl_pct or 0.0))
        return min(spike_cap, configured_cap)

    @staticmethod
    def _tight_futures_stop_pct(configured_fut_sl_pct: float, settings, attr_name: str, default: float) -> float:
        """Use the tighter of a special scalp cap and the user's global FUT cap."""
        tight_cap = max(0.0, float(getattr(settings, attr_name, default) or default))
        configured_cap = max(0.0, float(configured_fut_sl_pct or 0.0))
        return min(tight_cap, configured_cap)

    @staticmethod
    def _is_choppy_regime(regime: str) -> bool:
        return is_choppy_regime(regime)

    @staticmethod
    def _relaxed_threshold(value: float, settings) -> float:
        return relaxed_threshold(value, settings)

    @staticmethod
    def _choppy_confidence_gate(settings) -> float:
        return choppy_confidence_gate(settings)

    @staticmethod
    def _dynamic_rr(
        grade: str,
        confidence: float,
        intel_score: float = 0.0,
        max_rr: Optional[float] = None,
        timeframe: str = "",
        direction: str = "",
    ) -> float:
        return dynamic_rr(grade, confidence, intel_score, max_rr, timeframe, direction)

    @staticmethod
    def _option_index_stop_distance(
        natural_stop_distance: float,
        est_premium: float,
        options_sl_pct: float,
        instrument_multiplier: float,
        settings,
    ) -> float:
        return option_index_stop_distance(
            natural_stop_distance,
            est_premium,
            options_sl_pct,
            instrument_multiplier,
            settings,
        )

    @staticmethod
    def _option_premium_stop_distance(
        live_trade_price: float,
        scanner,
        regime: str,
        timeframe: str = "",
        atr_premium_stop: float = 0.0,
        settings=None,
        intel_score: float = 0.0,
        direction: str = "",
        score_unit: str = "legacy_0_100",
    ) -> float:
        try:
            opt_sl_pct = float(getattr(scanner, "options_sl_pct", getattr(get_settings(), "options_sl_pct", 15.0)) or 15.0)
        except Exception:
            opt_sl_pct = 15.0
        hard_cap_pct = option_stop_tolerance_pct(opt_sl_pct, regime)
        # `options_sl_pct` is an operator risk cap, not a minimum.  Choppy and
        # unknown regimes may tighten a stop through the ATR/expiry rules below,
        # but must never widen a configured premium loss limit.
        # RR, target and exit all flex with the intelligence score; the STOP was the one
        # of the four that ignored it entirely, moving only with regime and ATR. Strong
        # corroborating evidence earns a little more room to be right; weak evidence
        # gets less. Bounded hard so this can only ever nudge a stop the ATR and
        # per-timeframe rules already chose, never redefine it.
        #
        # Default OFF: this changes the risk on every option entry, which is the
        # operator's decision and not a defect being repaired.
        opt_sl_pct = _intelligence_stop_pct(
            opt_sl_pct, intel_score, settings, direction=direction, score_unit=score_unit,
        )
        opt_sl_pct = min(opt_sl_pct, hard_cap_pct)
        price = float(live_trade_price or 0.0)
        pct_stop = price * (opt_sl_pct / 100.0)

        # Timeframe- and ATR-aware stop: derive the premium stop from the signal's ATR
        # natural stop (already converted to premium points by the caller) bounded by
        # per-timeframe min/max %, so 5min stops sit tighter and 15min wider — driven by
        # real volatility rather than a flat %. options_sl_pct stays the hard ceiling.
        atr_stop = float(atr_premium_stop or 0.0)
        if atr_stop > 0.0 and price > 0.0:
            if settings is None:
                try:
                    settings = get_settings()
                except Exception:
                    settings = None
            if settings is not None and _setting_enabled(
                getattr(settings, "option_stop_tf_atr_aware", True), True
            ):
                tf = str(timeframe or "").lower()
                if tf in {"1min", "5min"}:
                    min_pct = float(getattr(settings, "option_stop_5min_min_pct", 4.0) or 4.0)
                    max_pct = float(getattr(settings, "option_stop_5min_max_pct", 10.0) or 10.0)
                else:
                    min_pct = float(getattr(settings, "option_stop_15min_min_pct", 6.0) or 6.0)
                    max_pct = float(getattr(settings, "option_stop_15min_max_pct", 15.0) or 15.0)
                # Never exceed the configured hard ceiling (options_sl_pct).
                max_pct = min(max_pct, opt_sl_pct)
                floor = price * (min_pct / 100.0)
                ceil = price * (max_pct / 100.0)
                stop = min(max(atr_stop, floor), ceil)
                return min(pct_stop, max(0.5, stop))
        return pct_stop

    def _option_grade_allowed(
        self,
        grade: str,
        confidence: float,
        regime: str,
        adx_value: float = 0.0,
        intel_score: float = 0.0,
        signal_score: float = 0.0,
        five_min_true_impulse: bool = False,
        timeframe: str = "",
    ) -> bool:
        return option_grade_allowed(
            grade,
            confidence,
            regime,
            mode=getattr(getattr(self, "scanner", None), "mode", ""),
            adx_value=adx_value,
            intel_score=intel_score,
            signal_score=signal_score,
            five_min_true_impulse=five_min_true_impulse,
            timeframe=timeframe,
        )

    def _passes_quality_gate(
        self,
        tf: str,
        grade: str,
        confidence: float,
        regime: str,
        inst_type: str,
        settings,
        adx_value: float = 0.0,
        trend_15m_agrees: bool = False,
        intel_score: float = 0.0,
        iv_percentile: Optional[float] = None,
        chain_quality: Optional[Dict] = None,
        option_premium_ok: Optional[bool] = None,
        source_count: int = 1,
        impulse_confirmed: bool = False,
        power_move: bool = False,
        volume_ratio: float = 0.0,
        return_verdict: bool = False,
        confidence_calibrated: bool = False,
    ):
        return passes_quality_gate(
            tf,
            grade,
            confidence,
            regime,
            inst_type,
            settings,
            mode=getattr(getattr(self, "scanner", None), "mode", ""),
            adx_value=adx_value,
            trend_15m_agrees=trend_15m_agrees,
            intel_score=intel_score,
            iv_percentile=iv_percentile,
            chain_quality=chain_quality,
            option_premium_ok=option_premium_ok,
            source_count=source_count,
            impulse_confirmed=impulse_confirmed,
            power_move=power_move,
            volume_ratio=volume_ratio,
            return_verdict=return_verdict,
            confidence_calibrated=confidence_calibrated,
        )

    @staticmethod
    def _explicit_signal_quality_evidence(
        signal,
        *,
        structural_impulse: bool = False,
        structural_power: bool = False,
    ) -> Tuple[bool, bool]:
        """Return explicit impulse/power evidence without inferring either from ADX.

        ADX is already an independent input to the canonical policy.  Treating a high
        ADX (or the absolute value of an intelligence score) as an additional impulse
        leg double-counts the same observation and makes replay/recovery more permissive
        than live.  Callers may provide a separately measured candle-structure result;
        dynamically attached live flags are preserved when a signal already carries one.
        """
        impulse = bool(
            structural_impulse
            or getattr(signal, "impulse_confirmed", False)
            or getattr(signal, "is_explosive_bypass", False)
        )
        power = bool(structural_power or getattr(signal, "power_move", False))
        return impulse, power

    def _replay_candle_volume_evidence(self, signal, tf: str, lookback: int = 20) -> dict:
        """Signal-bar volume against the prior bars' mean, from the replay candle store.

        Replay has no live intel volume context. This measures the same question from
        recorded candles (index candle volume, not the live cash/futures basis) and only
        uses bars up to the signal bar. Returns {} when it cannot be measured.
        """
        try:
            frame = self.scanner.candles.get_candles(str(getattr(signal, "instrument", "") or ""), tf)
            if frame is None or "volume" not in frame:
                return {}
            stamp = pd.Timestamp(getattr(signal, "timestamp")).tz_localize(None)
            index = frame.index.tz_localize(None) if getattr(frame.index, "tz", None) else frame.index
            pos = int(index.searchsorted(stamp, side="right")) - 1
            if pos < lookback or index[pos] != stamp:
                return {}
            volumes = frame["volume"].astype(float)
            base = float(volumes.iloc[pos - lookback:pos].mean())
            if not base > 0:
                return {}
            return dict(volume_ratio=float(volumes.iloc[pos]) / base, volume_available=True,
                        volume_basis="CANDLE_REPLAY")
        except Exception:
            return {}

    def _quality_verdict_for_signal(
        self,
        signal,
        *,
        tf: str,
        grade: str,
        confidence: float,
        regime: str,
        inst_type: str,
        settings,
        trend_15m_agrees: bool,
        source_count: int = 1,
        structural_impulse: bool = False,
        structural_power: bool = False,
    ):
        """Build the canonical call-site inputs shared by replay/recovery/EOD."""
        if s12_lane_policy.is_lane(signal):
            native_signal = copy.copy(signal)
            native_signal.timeframe = tf
            try:
                signal_time = pd.Timestamp(getattr(native_signal, "timestamp", None))
                minutes = {"1min": 1, "5min": 5, "15min": 15}.get(str(tf), 5)
                entry_available_at = signal_time + pd.Timedelta(minutes=minutes)
            except Exception:
                entry_available_at = None
            replay = str(getattr(getattr(self, "scanner", None), "mode", "")).upper() == "HISTORICAL"
            verdict = s12_lane_policy.entry_verdict(
                native_signal,
                settings,
                regime=regime,
                lane="REPLAY" if replay else "LIVE",
                now=entry_available_at,
            )
            signal.quality_verdict = verdict.to_dict()
            return verdict
        if uts_lane_policy.is_uts(signal, settings):
            # Historical/recovery/EOD callers must use the same native admission
            # policy as live. The caller's timeframe is authoritative even when
            # an older signal record omitted its timeframe. Keep measured volume
            # provenance and native RSI/ADX; never turn a bare ratio into volume.
            native_signal = copy.copy(signal)
            native_signal.timeframe = tf
            replay = str(getattr(getattr(self, "scanner", None), "mode", "")).upper() == "HISTORICAL"
            volume = self._replay_candle_volume_evidence(native_signal, tf) if replay else {}
            verdict = uts_lane_policy.entry_verdict(
                native_signal, settings, regime=regime,
                lane=("REPLAY" if replay else "RECOVERY"), **volume,
            )
            signal.quality_verdict = verdict.to_dict()
            return verdict
        impulse_confirmed, power_move = self._explicit_signal_quality_evidence(
            signal,
            structural_impulse=structural_impulse,
            structural_power=structural_power,
        )
        try:
            adx_value = float(getattr(signal, "adx_value", 0.0) or 0.0)
        except (TypeError, ValueError):
            adx_value = 0.0
        try:
            volume_ratio = float(getattr(signal, "volume_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            volume_ratio = 0.0
        return self._passes_quality_gate(
            tf,
            grade,
            confidence,
            regime,
            inst_type,
            settings,
            adx_value=adx_value,
            trend_15m_agrees=bool(trend_15m_agrees),
            source_count=max(1, int(source_count or 1)),
            impulse_confirmed=impulse_confirmed,
            power_move=power_move,
            volume_ratio=volume_ratio,
            return_verdict=True,
            confidence_calibrated=bool(getattr(signal, "confidence_calibrated", False)),
        )

    @staticmethod
    def _trend_15m_agrees_at(mtf_result, signal_ts, signal_type: str, *, closed_only=False) -> bool:
        result = getattr(mtf_result, "results_15min", None) or {}
        signals = result.get("signals", []) or []
        if not signals:
            if closed_only:
                return False
            state = result.get("state", {}) or {}
            position = int(state.get("position") or 0)
            expected = 1 if str(signal_type or "").upper() == "BUY" else -1
            return position == expected

        target_ts = _market_naive(signal_ts) if signal_ts is not None else None
        prior = []
        for signal in signals:
            ts = _market_naive(signal.timestamp)
            if closed_only:
                ts += pd.Timedelta(minutes=15)
            if target_ts is None or ts <= target_ts:
                prior.append(signal)
        if not prior:
            return False
        latest = max(
            prior,
            key=lambda signal: signal.timestamp.replace(tzinfo=None)
            if getattr(signal.timestamp, "tzinfo", None)
            else signal.timestamp,
        )
        return str(getattr(latest, "signal_type", "") or "").upper() == str(signal_type or "").upper()

    def _is_5min_impulse_reversal(
        self,
        sig,
        mtf_result,
        graded_confidence: float,
        settings,
    ) -> bool:
        """Allow a 5m choppy reversal only when the flip is a real impulse."""
        candle = getattr(sig, "raw_candle", {}) or {}
        try:
            open_px = float(candle.get("open") or 0.0)
            high_px = float(candle.get("high") or 0.0)
            low_px = float(candle.get("low") or 0.0)
            close_px = float(candle.get("close") or getattr(sig, "price", 0.0) or 0.0)
            atr = max(float(getattr(sig, "atr_value", 0.0) or 0.0), 1e-9)
            body_atr = abs(close_px - open_px) / atr
            range_atr = abs(high_px - low_px) / atr
            close_near_extreme = (
                close_px >= high_px - (0.25 * atr)
                if getattr(sig, "signal_type", "") == "BUY"
                else close_px <= low_px + (0.25 * atr)
            )
        except Exception:
            return False

        adx_value = float(getattr(sig, "adx_value", 0.0) or 0.0)
        conf = float(graded_confidence or 0.0)
        strong_adx = adx_value >= 28.0
        impulse_body = body_atr >= 0.80 and range_atr >= 1.05 and close_near_extreme
        extreme_range = range_atr >= 1.45 and close_near_extreme and conf >= self._relaxed_threshold(0.78, settings)

        volume_confirmed = False
        try:
            candles = (getattr(mtf_result, "results_5min", None) or {}).get("candles") or []
            idx = int(getattr(sig, "bar_index", -1) or -1)
            if candles and 0 <= idx < len(candles):
                current_vol = float(candles[idx].get("volume") or 0.0)
                lookback = candles[max(0, idx - 12):idx]
                vols = [float(row.get("volume") or 0.0) for row in lookback if float(row.get("volume") or 0.0) > 0]
                avg_vol = sum(vols) / len(vols) if vols else 0.0
                volume_confirmed = avg_vol > 0 and current_vol >= (1.35 * avg_vol)
        except Exception:
            volume_confirmed = False

        quality_confirmed = volume_confirmed or conf >= self._relaxed_threshold(0.82, settings)
        return strong_adx and (impulse_body or extreme_range) and quality_confirmed

    def _historical_option_strike(self, instrument: str, direction: str, price: float, cfg: Dict) -> float:
        interval = cfg.get("strike_interval", 50)
        atm = round(float(price) / interval) * interval
        selection = getattr(get_settings(), "option_strike_selection", "ITM")
        option_type = "CE" if direction == "LONG" else "PE"
        if selection in ("ITM", "BOTH"):
            return atm - interval if option_type == "CE" else atm + interval
        return atm

    @staticmethod
    def _filter_option_history_date(df: pd.DataFrame, trade_date) -> pd.DataFrame:
        return get_option_history_store().filter_day(df, trade_date)

    def _load_cached_historical_option_contract(
        self,
        instrument: str,
        strike: float,
        option_type: str,
        interval: str,
        trade_day,
        now_dt=None,
        expiry_preference: str = "",
    ) -> Tuple[pd.DataFrame, str]:
        """Use an archived contract only when it contains the requested trade day.

        The live instrument master generally contains current and future expiries,
        so resolving a May signal from it can incorrectly produce a June contract.
        Local candle files retain the original expired contract symbol.  This lookup
        is historical-only and returns no token, so it cannot affect live execution.
        """
        if trade_day is None or trade_day >= datetime.now(IST).date():
            return pd.DataFrame(), ""
        try:
            from pathlib import Path

            desired = int(round(float(strike)))
            suffix = f"{str(option_type or '').upper()}_{interval}.csv"
            candidates = []
            # Expiry is part of identity. Monthly Kite symbols encode only the
            # month; resolve those against a local master rather than guessing
            # the exchange calendar. Explicit dated symbols are self-describing.
            from engine.expiry_manager import expiry_policy_signature
            master_paths = (Path("data_store/kite_instruments.json"), Path("data_store/instruments.json"))
            def file_signature(path):
                if not path.exists():
                    return str(path.resolve()), None, None
                stat = path.stat()
                return str(path.resolve()), stat.st_mtime_ns, stat.st_size
            master_signature = tuple(file_signature(path) for path in master_paths)
            snapshot = getattr(self, "_archived_master_snapshot", None)
            if snapshot is None or snapshot[0] != master_signature:
                master = {}
                parsed_dates = {}
                for master_path in master_paths:
                    if not master_path.exists():
                        continue
                    for item in json.loads(master_path.read_text(encoding="utf-8")):
                        symbol_name = str(item.get("symbol") or item.get("tradingsymbol") or "").upper()
                        value = str(item.get("expiry") or "")
                        if value not in parsed_dates:
                            expiry = pd.to_datetime(value, format="%d%b%Y", errors="coerce")
                            if pd.isna(expiry):
                                expiry = pd.to_datetime(value, errors="coerce")
                            parsed_dates[value] = expiry
                        expiry = parsed_dates[value]
                        if symbol_name and not pd.isna(expiry):
                            master[symbol_name] = expiry.date()
                by_index = {name: set() for name in ("NIFTY", "BANKNIFTY", "MIDCPNIFTY", "SENSEX")}
                for symbol, expiry in master.items():
                    if symbol.endswith(("CE", "PE")):
                        for name, values in by_index.items():
                            if symbol.startswith(name):
                                values.add(expiry)
                snapshot = (master_signature, master, by_index)
                self._archived_master_snapshot = snapshot
            _, master, by_index = snapshot
            pattern = re.compile(re.escape(str(instrument).upper()) + r"(.+?)" + str(desired) + re.escape(str(option_type).upper()))
            paths = [path for path in Path("data_store/candles").glob(f"{instrument}*{suffix}")
                     if pattern.fullmatch(path.name[:-len(f"_{interval}.csv")].upper())]
            source_signature = (master_signature, tuple(file_signature(path) for path in paths))
            lookup_key = (str(instrument).upper(), desired, str(option_type).upper(), interval,
                          trade_day, pd.Timestamp(now_dt) if now_dt is not None else None,
                          expiry_preference, expiry_policy_signature(), source_signature)
            cache = self.__dict__.setdefault("_archived_contract_lookup_cache", {})
            cached = cache.get(lookup_key)
            if cached is not None:
                return cached[0].copy(), cached[1]
            for path in paths:
                symbol = path.name[: -len(f"_{interval}.csv")]
                match = re.fullmatch(
                    re.escape(str(instrument).upper()) + r"(.+?)" + str(desired) + re.escape(str(option_type).upper()),
                    symbol.upper(),
                )
                if not match:
                    continue
                expiry = master.get(symbol.upper())
                code = match.group(1)
                if expiry is None and re.fullmatch(r"\d{2}[A-Z]{3}\d{2}", code):
                    expiry = datetime.strptime(code, "%d%b%y").date()
                if expiry is None and re.fullmatch(r"\d{2}[1-9OND]\d{2}", code):
                    month = {"O": 10, "N": 11, "D": 12}.get(code[2], int(code[2]) if code[2].isdigit() else 0)
                    expiry = date(2000 + int(code[:2]), month, int(code[3:]))
                if expiry is None or expiry < trade_day:
                    continue
                day_frame = self._filter_option_history_date(
                    self._option_history_store.load(path), trade_day
                )
                if not day_frame.empty:
                    candidates.append((expiry, symbol, day_frame))
            if candidates:
                # Resolve the configured rollover from the known local expiry set.
                # Never silently relabel a different strike as the requested one.
                from engine.expiry_manager import ExpiryManager
                expiries = {row[0] for row in candidates}
                expiries.update(exp for exp in by_index.get(str(instrument).upper(), ()) if exp >= trade_day)
                resolver = ExpiryManager.__new__(ExpiryManager)
                resolver.df = pd.DataFrame({"name": [instrument] * len(expiries), "instrumenttype": ["OPTIDX"] * len(expiries), "expiry_dt": pd.to_datetime(sorted(expiries))})
                native_monthly = str(expiry_preference or "").upper() == "MONTHLY"
                expected = resolver._select_option_expiry_for_date(
                    instrument, trade_day,
                    (
                        "active_monthly"
                        if native_monthly or instrument in {"BANKNIFTY", "MIDCPNIFTY"}
                        else "active_weekly"
                    ),
                    now_dt=now_dt,
                )
                candidates = [row for row in candidates if expected is not None and row[0] == pd.Timestamp(expected).date()]
                if not candidates:
                    return pd.DataFrame(), ""
                expiry, symbol, frame = min(candidates, key=lambda row: row[1])
                frame = frame.copy()
                frame.attrs.update({"contract_symbol": symbol, "contract_expiry": expiry.isoformat(), "contract_strike": desired, "contract_option_type": str(option_type).upper(), "contract_verified": True})
                if len(cache) >= 128:
                    cache.clear()
                cache[lookup_key] = (frame.copy(), symbol)
                return frame, symbol
        except Exception as exc:
            logger.debug(
                f"Historical option-cache lookup failed for {instrument} {strike} "
                f"{option_type} on {trade_day}: {exc}"
            )
        return pd.DataFrame(), ""

    def _load_option_candles(
        self,
        scanner,
        instrument: str,
        strike: float,
        option_type: str,
        interval: str,
        trade_date,
        now_dt=None,
        expiry_preference: str = "",
    ) -> Tuple[pd.DataFrame, str, str]:
        trade_day = pd.Timestamp(trade_date).date() if trade_date is not None else None
        archived_frame, archived_symbol = self._load_cached_historical_option_contract(
            instrument,
            strike,
            option_type,
            interval,
            trade_day,
            now_dt=now_dt,
            expiry_preference=expiry_preference,
        )
        if not archived_frame.empty:
            diag = getattr(scanner, "_diag_inc", None)
            if diag:
                diag("option_history", "archived_hits")
            return archived_frame, archived_symbol, ""
        option_token_kwargs = {"trade_date": trade_date, "now_dt": now_dt}
        if expiry_preference:
            option_token_kwargs["expiry_preference"] = expiry_preference
        opt_info = scanner.data.get_option_token(
            instrument,
            strike,
            option_type,
            **option_token_kwargs,
        )
        if not opt_info:
            return pd.DataFrame(), "", ""
        symbol = str(opt_info.get("symbol", ""))
        token = str(opt_info.get("token", ""))
        # The live resolver may snap a missing strike. A historical row must not
        # keep the requested strike while replaying that other contract's prices.
        resolved_strike = opt_info.get("strike_val")
        if resolved_strike is None and opt_info.get("strike") is not None:
            resolved_strike = float(opt_info["strike"]) / 100.0
        if resolved_strike is not None and abs(float(resolved_strike) - float(strike)) > 1e-6:
            return pd.DataFrame(), "", ""
        expiry_value = opt_info.get("date_aware_expiry") or opt_info.get("expiry")
        contract_expiry = pd.to_datetime(expiry_value, errors="coerce") if expiry_value else pd.NaT
        if not pd.isna(contract_expiry) and trade_day and contract_expiry.date() < trade_day:
            return pd.DataFrame(), "", ""
        if str(getattr(scanner, "mode", "")).upper() == "HISTORICAL" and (resolved_strike is None or pd.isna(contract_expiry)):
            return pd.DataFrame(), "", ""

        def with_contract(frame):
            if frame is not None and not frame.empty:
                frame = frame.copy()
                frame.attrs.update({
                    "contract_symbol": symbol,
                    "contract_token": token,
                    "contract_expiry": contract_expiry.date().isoformat() if not pd.isna(contract_expiry) else "",
                    "contract_strike": float(resolved_strike) if resolved_strike is not None else float(strike),
                    "contract_option_type": str(option_type).upper(),
                    "contract_verified": resolved_strike is not None and not pd.isna(contract_expiry),
                })
            return frame
        exchange = str(
            opt_info.get("exch_seg")
            or ("BFO" if _is_bfo_instrument(instrument) else "NFO")
        )
        cache_key = (symbol, interval, trade_day.isoformat() if trade_day else "")
        symbol_cache_key = (symbol, interval)
        diag = getattr(scanner, "_diag_inc", None)
        local_cache_missing_day = False
        if cache_key in self._option_history_cache:
            if diag:
                diag("option_history", "hits" if not self._option_history_cache[cache_key].empty else "misses")
            return with_contract(self._option_history_cache[cache_key]), symbol, token
        if symbol_cache_key in self._option_symbol_history_cache:
            day_df = self._filter_option_history_date(
                self._option_symbol_history_cache[symbol_cache_key],
                trade_day,
            )
            if not day_df.empty or symbol_cache_key in self._option_history_fetch_attempted:
                self._option_history_cache[cache_key] = day_df
                if diag:
                    diag("option_history", "hits" if not day_df.empty else "misses")
                return with_contract(day_df), symbol, token
        from pathlib import Path
        cache_path = Path("data_store/candles") / f"{symbol}_{interval}.csv"
        if cache_path.exists():
            try:
                df = self._option_history_store.load(cache_path)
                self._option_symbol_history_cache[symbol_cache_key] = df
                day_df = self._filter_option_history_date(df, trade_day)
                if not day_df.empty:
                    self._option_history_cache[cache_key] = day_df
                    if diag:
                        diag("option_history", "hits")
                    return with_contract(day_df), symbol, token
                self._historical_audit_log(f"Option history cache for {symbol} has no rows for {trade_day}; fetching/using configured fallback.")
                local_cache_missing_day = True
            except Exception:
                pass
        mode = getattr(get_settings(), "ut_option_history_mode", "fetch_or_skip")
        if mode == "stored_or_synthetic":
            self._option_history_cache[cache_key] = pd.DataFrame()
            if diag:
                diag("option_history", "synthetic")
            return pd.DataFrame(), symbol, token
        # ``fetch_or_skip`` must still make one broker attempt for a historical
        # contract/day that is absent from the local cache.  Returning early here
        # turned it into ``stored_or_skip`` and silently downgraded valid option
        # candidates to futures.  The persistent miss store below bounds retries;
        # an unavailable contract still fails closed with no synthetic premium.
        days_back = max(15, int(getattr(scanner, "backtest_days", 7) or 7) * 3)
        miss_ttl = float(
            getattr(get_settings(), "option_history_miss_ttl_seconds", 900.0)
            or 900.0
        )
        if self._option_history_store.has_recent_miss(
            symbol,
            interval,
            days_back,
            miss_ttl,
        ):
            self._option_history_fetch_attempted.add(symbol_cache_key)
            self._option_symbol_history_cache.setdefault(
                symbol_cache_key, pd.DataFrame()
            )
            self._option_history_cache[cache_key] = pd.DataFrame()
            if diag:
                diag("option_history", "misses")
            return pd.DataFrame(), symbol, token
        if diag:
            diag("option_history", "attempts")
        self._option_history_fetch_attempted.add(symbol_cache_key)
        df = scanner.data.get_historical_candles(
            token,
            exchange,
            interval,
            days_back=days_back,
            instrument_name=symbol,
        )
        if df is not None and not df.empty:
            self._option_history_store.clear_miss(symbol, interval, days_back)
            df = df.sort_index()
            existing = self._option_symbol_history_cache.get(symbol_cache_key)
            if existing is not None and not existing.empty:
                df = pd.concat([existing, df]).sort_index()
                df = df[~df.index.duplicated(keep="last")]
            self._option_symbol_history_cache[symbol_cache_key] = df
            day_df = self._filter_option_history_date(df, trade_day)
            if diag and not day_df.empty:
                diag("option_history", "hits")
        else:
            self._option_history_store.record_miss(symbol, interval, days_back)
            self._option_symbol_history_cache.setdefault(symbol_cache_key, pd.DataFrame())
            day_df = pd.DataFrame()
        if day_df.empty:
            if diag:
                diag("option_history", "misses")
        self._option_history_cache[cache_key] = day_df
        return with_contract(day_df), symbol, token

    def _historical_live_management_probe(
        self,
        scanner,
        *,
        instrument: str,
        direction: str,
        option_type: str,
        timeframe: str,
        entry_time: datetime,
        entry_price: float,
        stop_price: float,
        target_price: float,
        qty: int,
        grade: str = "",
        confidence: float = 0.0,
        entry_regime: str = "UNKNOWN",
    ):
        """Point-in-time adapter around the production live exit helpers.

        Missing historical OI/PCR/depth is left unavailable.  It is never filled
        from today's state and never manufactured.  Price/regime guards still run
        against candles visible at the replay clock.
        """
        from scanner import Scanner as _Scanner
        from intelligence.regime_detector import RegimeDetector

        candle_view = _PointInTimeCandles(getattr(scanner, "candles", None))
        probe = _Scanner.__new__(_Scanner)
        probe.mode = "HISTORICAL"
        probe.candles = candle_view
        probe.active_indices = list(
            getattr(scanner, "active_indices", None)
            or ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
        )
        probe.latest_regimes = {}
        probe._timeframe_regimes = {}
        probe._cached_results = {}
        probe._diagnostics = {}
        probe._timeframe_intel_cache = {}
        probe._rupee_loss_band_states = {}
        probe._loss_threat_confirmation = {}
        probe._setting_enabled = _setting_enabled
        self._bind_historical_intelligence(probe, scanner, candle_view)
        regime_detector = RegimeDetector()

        base = str(instrument or "").upper().split()[0]
        tf = str(timeframe or "5min").lower()
        if tf not in {"1min", "5min", "15min"}:
            tf = "5min"
        trade = SimpleNamespace(
            id=f"replay:{base}:{pd.Timestamp(entry_time).isoformat()}",
            instrument=base,
            timeframe=tf,
            direction=str(direction or "").upper(),
            signal_type="BUY" if str(direction or "").upper() == "LONG" else "SELL",
            option_type=str(option_type or "").upper(),
            inst_type="OPT",
            entry_time=entry_time,
            signal_timestamp=entry_time,
            entry_price=float(entry_price),
            price=float(entry_price),
            current_price=float(entry_price),
            initial_stop=float(stop_price),
            trailing_stop=float(stop_price),
            current_stop=float(stop_price),
            target=float(target_price),
            rr_ratio=(float(target_price) - float(entry_price))
            / max(0.05, float(entry_price) - float(stop_price)),
            quantity=max(1, int(qty)),
            pnl_quantity=max(1, int(qty)),
            lots=1,
            lot_size=max(1, int(qty)),
            instrument_multiplier=1.0,
            peak_pnl=0.0,
            max_drawdown=0.0,
            grade=str(grade or ""),
            confidence=float(confidence or 0.0),
            market_regime=str(entry_regime or "UNKNOWN").upper(),
            entry_spot=0.0,
            scalp_lock_mode=(tf == "1min"),
            runner_mode=False,
            original_target_distance=abs(float(target_price) - float(entry_price)),
            market_intelligence_context={},
            decision_intel={},
            entered_intracandle=False,
            repaint_guard_checked=True,
        )
        errors_seen = set()
        trace = []
        evaluated_guards = [
            "broker_stop", "hard_target", "profit_lock", "breakeven_stop",
            "peak_giveback_cap", "early_loss_cut", "rupee_loss_band",
            "peak_threat", "smart_profit_lock", "profit_giveback_or_reversal",
            "eod_loss_minimize", "low_gain_protect", "major_win_guard",
            "stagnation_exit", "15m_fast_reversal",
        ]

        def _record_error(name: str, exc: Exception, ts) -> None:
            key = (name, type(exc).__name__, str(exc))
            if key in errors_seen:
                return
            errors_seen.add(key)
            trace.append({
                "time": pd.Timestamp(ts).isoformat(), "guard": name,
                "state": "ERROR_ABSTAIN", "detail": f"{type(exc).__name__}: {exc}",
            })

        def _recover_or_exit(reason, detail, current_px, pnl_rs, peak_rs, risk_rs, now, regime):
            if pnl_rs >= 0:
                return reason, detail
            try:
                action, recovery_detail = probe._loss_recovery_watch_decision(
                    trade,
                    current_price=float(current_px),
                    current_pnl_rs=float(pnl_rs),
                    peak_pnl=float(peak_rs),
                    initial_risk_rs=float(risk_rs),
                    latest_time=now,
                    regime=regime,
                    proposed_exit_reason=reason,
                )
                if action == "HOLD":
                    trace.append({
                        "time": pd.Timestamp(now).isoformat(), "guard": reason,
                        "state": "RECOVERY_HOLD", "detail": str(recovery_detail or ""),
                    })
                    return "", ""
                if action == "EXIT":
                    return "RECOVERY_WATCH_EXIT", str(recovery_detail or detail or "")
            except Exception as exc:
                _record_error("loss_recovery_watch", exc, now)
            return reason, detail

        def evaluate(ts, close_px: float, high_px: float, peak_pnl_rs: float):
            bar_start = pd.Timestamp(ts)
            if bar_start.tzinfo is not None:
                bar_start = bar_start.tz_convert(IST).tz_localize(None)
            candle_view.as_of = bar_start + pd.Timedelta(minutes=1)
            now = bar_start.to_pydatetime() + timedelta(minutes=1)
            trade.current_price = float(close_px)
            trade.peak_pnl = max(float(trade.peak_pnl or 0.0), float(peak_pnl_rs or 0.0))
            trade.max_drawdown = min(
                float(getattr(trade, "max_drawdown", 0.0) or 0.0),
                (float(close_px) - float(entry_price)) * max(1, int(qty)),
            )

            # Recompute every index regime at this clock. Cross-index guards therefore
            # see contemporaneous peers, never the session's final label.
            for peer in probe.active_indices:
                try:
                    frame = candle_view.get_candles(peer, tf)
                    detected = regime_detector.detect(frame, peer, tf)
                    value = getattr(
                        detected.get("regime"), "value", detected.get("regime", "UNKNOWN")
                    )
                    value = str(value or "UNKNOWN").upper()
                    probe.latest_regimes[peer] = value
                    probe._timeframe_regimes[probe._timeframe_intel_key(peer, tf)] = value
                except Exception as exc:
                    _record_error(f"regime:{peer}", exc, now)
            regime = str(probe.latest_regimes.get(base) or trade.market_regime or "UNKNOWN")
            trade.market_regime = regime
            trade.entry_spot = float(trade.entry_spot or candle_view.get_latest_price(base) or 0.0)

            # Historical intelligence is itself timestamp-sliced. Chain-only fields
            # remain absent, so helpers correctly abstain where they require them.
            try:
                intel_score, intel_regime = scanner._compute_historical_intel_at(base, now, tf)
                if str(intel_regime or "UNKNOWN").upper() != "UNKNOWN":
                    regime = str(intel_regime).upper()
                    trade.market_regime = regime
                    probe.latest_regimes[base] = regime
                trade.intelligence_score = float(intel_score or 0.0)
                trade.market_intelligence_context = {
                    "aggregate": {
                        "score": float(intel_score or 0.0) * 100.0,
                        "score_norm": float(intel_score or 0.0),
                    },
                    "regime": {"regime": regime},
                    "as_of": pd.Timestamp(now).isoformat(),
                    "historical_missing": ["option_chain", "order_flow_depth"],
                }
                historical_context = getattr(scanner, "_get_historical_intel_context_at", None)
                if callable(historical_context):
                    context = historical_context(base, now, tf)
                    if isinstance(context, dict) and context:
                        trade.market_intelligence_context = copy.deepcopy(context)
                        trade.decision_intel = copy.deepcopy(context)
            except Exception as exc:
                _record_error("historical_intelligence", exc, now)

            current_pnl = (float(close_px) - float(entry_price)) * max(1, int(qty))
            risk_rs = max(0.05, float(entry_price) - float(stop_price)) * max(1, int(qty))
            peak = float(trade.peak_pnl or 0.0)
            peak_pct = peak / max(0.05, float(entry_price) * max(1, int(qty))) * 100.0
            current_pct = current_pnl / max(0.05, float(entry_price) * max(1, int(qty))) * 100.0

            # Production runner policy. A close can arm it for the next bar; this avoids
            # claiming knowledge of the intrabar path that produced this OHLC candle.
            try:
                move = max(0.0, float(high_px) - float(entry_price))
                target_dist = max(0.05, float(trade.original_target_distance or 0.0))
                unlock = min(1.0, max(0.50, float(getattr(get_settings(), "dynamic_rr_runner_unlock_ratio", 0.95) or 0.95)))
                big_ok, big_reason = probe._big_move_expected(
                    trade, gain_r=probe._trade_gain_in_r(trade, move), regime=regime
                )
                if not trade.runner_mode and (move >= target_dist * unlock or big_ok):
                    trade.runner_mode = True
                    trade.target = 0.0
                    trace.append({
                        "time": bar_start.isoformat(), "guard": "runner",
                        "state": "ARMED", "detail": "target proximity" if move >= target_dist * unlock else big_reason,
                    })
                if trade.runner_mode:
                    lock_pct = probe._dynamic_runner_lock_pct(
                        float(getattr(get_settings(), "dynamic_rr_runner_lock_pct", 0.90) or 0.90),
                        regime,
                    )
                    trail_dist = target_dist * (1.0 - lock_pct)
                    trade.current_stop = max(
                        float(trade.current_stop or stop_price),
                        float(entry_price) + max(0.0, move - trail_dist),
                    )
            except Exception as exc:
                _record_error("runner", exc, now)

            lock_ratio = 0.0
            lock_stop = 0.0
            lock_breathing_clamped = False
            lock_trend_deferred = False
            try:
                lock_stop, lock_reason, lock_ratio = probe._profit_lock_floor(
                    float(entry_price), "LONG", max(1, int(qty)), peak,
                    bool(trade.scalp_lock_mode), inst_mult=1.0,
                    initial_risk_rs=risk_rs, timeframe=tf, now=now, trade_like=trade,
                )
                # Same HOLD contract as the live and manual paths (engine/
                # profit_lock_hold.py): option breathing room, a lock hit only when the
                # close is at or below the (possibly clamped) lock floor, and a trend
                # deferral that restores the stop this bar started from. Replay used to
                # apply the floor with neither (guard audit 2026-10-06). A structural
                # trail above the floor keeps its own authority through next_stop.
                pre_lock_stop = float(trade.current_stop or stop_price)
                proposed_lock_stop = max(pre_lock_stop, float(lock_stop)) if lock_stop > 0 else pre_lock_stop
                lock_hold = profit_lock_hold_policy.resolve(
                    probe,
                    trade,
                    lock_reason=lock_reason,
                    lock_stop=lock_stop,
                    proposed_stop=proposed_lock_stop,
                    pre_lock_stop=pre_lock_stop,
                    mark=float(close_px),
                    option_space=True,
                    current_pnl_rs=current_pnl,
                    peak_pnl_rs=peak,
                    latest_time=now,
                    regime=regime,
                    apply_breathing=lock_stop > 0,
                )
                trace.extend(dict(item) for item in lock_hold.trace if item.get("state") != "PROPOSED")
                lock_breathing_clamped = lock_hold.breathing_clamped
                lock_trend_deferred = lock_hold.trend_deferred
                lock_stop = lock_hold.lock_stop
                trade.current_stop = float(lock_hold.stop or pre_lock_stop)
                if lock_hold.lock_reason and lock_hold.hit:
                    return (
                        lock_hold.lock_reason,
                        f"production profit-lock floor {lock_hold.effective_lock_stop:.2f}",
                        lock_hold.effective_lock_stop, regime, trace, evaluated_guards,
                    )
            except Exception as exc:
                _record_error("profit_lock", exc, now)

            try:
                verdict = probe._peak_giveback_cap_exit(
                    trade, current_pnl_rs=current_pnl, peak_pnl_rs=peak, evaluated_at=now
                )
                fired, detail = verdict
                if fired:
                    reason = getattr(verdict, "rule", "") or "PROFIT_FLOOR_EXIT"
                    reason, detail = _recover_or_exit(reason, detail, close_px, current_pnl, peak, risk_rs, now, regime)
                    if reason:
                        return reason, detail, close_px, regime, trace, evaluated_guards
            except Exception as exc:
                _record_error("peak_giveback_cap", exc, now)

            try:
                fired, detail, cut_r = probe._loss_guard_decision(
                    trade, current_pnl_rs=current_pnl, initial_risk_rs=risk_rs,
                    latest_time=now, regime=regime, current_price=float(close_px),
                )
                if fired and current_pnl < 0:
                    reason, detail = _recover_or_exit("EARLY_LOSS_CUT", f"{detail} thr={cut_r:.2f}R", close_px, current_pnl, peak, risk_rs, now, regime)
                    if reason:
                        return reason, detail, close_px, regime, trace, evaluated_guards
            except Exception as exc:
                _record_error("early_loss_cut", exc, now)

            try:
                fired, reason, detail = probe._rupee_loss_band_guard_decision(
                    trade, current_pnl_rs=current_pnl, latest_time=now,
                    regime=regime, current_price=float(close_px),
                )
                if fired and current_pnl < 0:
                    reason, detail = _recover_or_exit(reason, detail, close_px, current_pnl, peak, risk_rs, now, regime)
                    if reason:
                        return reason, detail, close_px, regime, trace, evaluated_guards
            except Exception as exc:
                _record_error("rupee_loss_band", exc, now)

            try:
                fired, reason, detail = probe._peak_threat_exit_decision(
                    trade, current_price=float(close_px), current_pnl_rs=current_pnl,
                    peak_pnl=peak, initial_risk_rs=risk_rs, latest_time=now,
                    regime=regime, inst_type="OPT", option_type=trade.option_type,
                    direction=trade.direction,
                )
                if fired:
                    return reason, detail, close_px, regime, trace, evaluated_guards
            except Exception as exc:
                _record_error("peak_threat", exc, now)

            try:
                smart_min = probe._profit_lock_min_peak(
                    bool(trade.scalp_lock_mode), risk_rs, tf, now, trade_like=trade
                )
                ladder_peak = _Scanner._profit_ladder_pnl(probe, trade, peak, max(1, int(qty)))
                backstop = probe._smart_lock_backstop_ratio(lock_ratio, bool(trade.scalp_lock_mode))
                collapse = probe._smart_lock_floor_collapse(
                    peak, current_pnl, backstop, floor_rs=probe._profit_lock_floor_rs(trade)
                )
                if probe._smart_profit_retention_eligible(
                    peak_pnl=peak,
                    minimum_peak=smart_min,
                    ladder_peak=ladder_peak,
                    breathing_clamped=lock_breathing_clamped,
                    trend_deferred=lock_trend_deferred,
                ) and (
                    current_pnl < peak * backstop or collapse
                ):
                    reason, detail = _recover_or_exit("SMART_PROFIT_LOCK", "production retention backstop", close_px, current_pnl, peak, risk_rs, now, regime)
                    if reason:
                        return reason, detail, close_px, regime, trace, evaluated_guards

                fired, reason, detail = probe._peak_giveback_or_reversal_exit(
                    trade, current_pnl_rs=current_pnl, peak_pnl=peak,
                    initial_risk_rs=risk_rs, smart_min_peak=smart_min,
                    latest_time=now, regime=regime, inst_type="OPT",
                    option_type=trade.option_type, direction=trade.direction,
                )
                if fired:
                    reason, detail = _recover_or_exit(reason, detail, close_px, current_pnl, peak, risk_rs, now, regime)
                    if reason:
                        return reason, detail, close_px, regime, trace, evaluated_guards
            except Exception as exc:
                _record_error("smart_or_giveback", exc, now)

            try:
                fired, detail = probe._eod_loss_minimize_exit(
                    trade, current_pnl_rs=current_pnl, initial_risk_rs=risk_rs,
                    latest_time=now, regime=regime,
                )
                if fired:
                    return "EOD_LOSS_MINIMIZE", detail, close_px, regime, trace, evaluated_guards
            except Exception as exc:
                _record_error("eod_loss_minimize", exc, now)

            if peak_pct >= 10.0 and current_pct < peak_pct * 0.40:
                return "LOW_GAIN_PROTECT", f"peak {peak_pct:.1f}% -> {current_pct:.1f}%", close_px, regime, trace, evaluated_guards

            try:
                if _Scanner._profit_ladder_pnl(probe, trade, peak, max(1, int(qty))) >= probe._major_win_guard_ladder_arm_rs(trade, max(1, int(qty))) and current_pnl < peak * 0.75:
                    return "MAJOR_WIN_GUARD", f"peak Rs.{peak:.0f} -> Rs.{current_pnl:.0f}", close_px, regime, trace, evaluated_guards
            except Exception as exc:
                _record_error("major_win_guard", exc, now)

            try:
                fired, detail = probe._stagnation_exit_decision(
                    trade, current_pnl, peak, now, regime=regime
                )
                if fired:
                    return "STAGNATION_EXIT", detail, close_px, regime, trace, evaluated_guards
            except Exception as exc:
                _record_error("stagnation_exit", exc, now)

            return "", "", float(trade.current_stop or stop_price), regime, trace, evaluated_guards

        return evaluate

    def _replay_entry_context(self, scanner, instrument, timeframe, signal, direction, regime, lane):
        """The live market-context gate evaluated on replay data (PCR/OI unavailable).

        Uses only bars closed by the signal candle's close; never raises.
        """
        try:
            from engine import entry_context_gate as ctx_gate

            minutes = {"1min": 1, "5min": 5, "15min": 15}.get(str(timeframe), 5)
            closes_at = _market_naive(signal.timestamp) + pd.Timedelta(minutes=minutes)
            volume = self._replay_candle_volume_evidence(signal, timeframe)
            measures = ctx_gate.market_measures(scanner.candles.get_candles, instrument, timeframe,
                                                closes_at, direction)
            rsi = getattr(signal, "rsi_value", None)
            adx = getattr(signal, "adx_value", None)
            return ctx_gate.evaluate(ctx_gate.ContextInputs(
                lane=str(lane or "QP"), instrument=str(instrument), timeframe=str(timeframe),
                direction=str(direction), regime=str(regime or "UNKNOWN"),
                adx=float(adx) if adx not in (None, "") else None,
                rsi=float(rsi) if rsi not in (None, "") else None,
                volume_ratio=volume.get("volume_ratio"), volume_measured=bool(volume.get("volume_available")),
                **measures,
            ), get_settings())
        except Exception as exc:
            logger.debug(f"Replay entry context unavailable for {instrument} {timeframe}: {exc}")
            return None

    @staticmethod
    def _with_entry_context(verdict: Dict, context) -> Dict:
        if context is not None and isinstance(verdict, dict):
            verdict = dict(verdict)
            verdict["checks"] = {**dict(verdict.get("checks") or {}), "entry_context": context.to_dict()}
        return verdict

    def _s1_replay_bar_states(self, scanner, instrument: str, timeframe: str):
        """Per-bar S1 scores keyed by bar CLOSE time, for replaying native S1 exits.

        Computed once per instrument/timeframe from the replay candle store with the
        live S1 parameters. Indicators are causal, so a bar's row uses only bars up to
        its own close. Returns None when the frame or engine is unavailable.
        """
        cache = self.__dict__.setdefault("_s1_replay_state_cache", {})
        try:
            frame = scanner.candles.get_candles(instrument, timeframe)
            if frame is None or frame.empty:
                return None
            key = (instrument, timeframe, len(frame), frame.index[-1])
            if key in cache:
                return cache[key]
            from engine import s1_engine

            params = s1_engine.params_from_settings(get_settings(), instrument)

            def prepared(tf):
                return s1_engine.drop_off_session_bars(scanner.candles.get_candles(instrument, tf), params.session_end)

            df = prepared(timeframe)
            feats = s1_engine.compute_features(df, timeframe, params, df_5min=prepared("5min"),
                                               df_15min=prepared("15min"))
            minutes = {"1min": 1, "5min": 5, "15min": 15}.get(str(timeframe), 5)
            closes = pd.DatetimeIndex([_market_naive(ts) for ts in df.index]) + pd.Timedelta(minutes=minutes)
            states = pd.DataFrame({
                "bull": feats["bull_score"], "bear": feats["bear_score"], "adx": feats["adx"],
                "rel_vol": feats["rel_vol"], "atr_exp": feats["atr_exp"],
            }, index=closes)
            if len(cache) > 24:
                cache.clear()
            cache[key] = states
            return states
        except Exception as exc:
            logger.debug(f"S1 replay bar states unavailable for {instrument} {timeframe}: {exc}")
            return None

    @staticmethod
    def _closed_candle_replay_window(entry_ts, timeframe, exit_ts, session_exit_ts, crosses_session):
        """Shift a non-UTS replay to when its signal candle (and the next one) has closed.

        ``replay_closed_candle_entries=False`` restores the old open-stamp timing for
        comparison runs only; it books one candle of look-ahead.
        """
        minutes = {"1min": 1, "5min": 5, "15min": 15}.get(str(timeframe or ""), 0)
        if not minutes or not _setting_enabled(
            getattr(get_settings(), "replay_closed_candle_entries", True), True
        ):
            return entry_ts, exit_ts
        step = timedelta(minutes=minutes)
        start = entry_ts + step
        end = exit_ts if crosses_session else min(session_exit_ts, exit_ts + step)
        return start, end

    @staticmethod
    def _uts_replay_window(signal, timeframe, fallback_end, *, next_signal=None):
        """Price a confirmed UTS entry and chart exit only after their bars close."""
        minutes = {"1min": 1, "5min": 5, "15min": 15}[timeframe]
        source = getattr(signal, "source_signal_timestamp", None) or signal.timestamp
        start = _market_naive(source).to_pydatetime() + timedelta(minutes=minutes)
        end = _market_naive(fallback_end).to_pydatetime()
        if next_signal is not None:
            # Native arrows are stamped at their candle opens. An opposite
            # arrow can close this replay only once that candle has closed.
            next_source = getattr(next_signal, "source_signal_timestamp", None) or next_signal.timestamp
            next_available = _market_naive(next_source).to_pydatetime() + timedelta(minutes=minutes)
            end = min(end, next_available)
        exit_bar = getattr(signal, "uts_exit_time", None)
        reason = ""
        if exit_bar is not None:
            chart_end = _market_naive(exit_bar).to_pydatetime() + timedelta(minutes=minutes)
            if start < chart_end <= end:
                end, reason = chart_end, str(getattr(signal, "uts_exit_reason", "") or "")
        return start, end, reason

    def _simulate_actual_option_exit(
        self,
        option_df: pd.DataFrame,
        start_ts: datetime,
        end_ts: datetime,
        sl_price: float,
        rr: float,
        natural_sl_distance: Optional[float] = None,
        max_sl_pct: float = 0.15,
        # Units held, only needed to convert premium points into the rupee P&L the live
        # guards are written in. 0 (the default) leaves the live-exit path inert, so
        # every existing caller keeps the pure stop/target simulation unchanged.
        qty: int = 0,
        tf15_qty: int = 0,
        tf15_exit_scanner=None,
        tf15_instrument: str = "",
        tf15_direction: str = "",
        tf15_option_type: str = "",
        tf15_timeframe: str = "",
        replay_grade: str = "",
        replay_confidence: float = 0.0,
        replay_regime: str = "UNKNOWN",
        replay_stop_context: Optional[Dict] = None,
        live_exits: bool = True,
        replay_source_lane: str = "",
        required_stop_distance: float = 0.0,
        chart_exit_reason: str = "",
        replay_setup=None,
    ) -> Optional[Dict]:
        """``live_exits=False`` keeps the other engines' live guards out of this replay
        (the S1 lane) while ``qty`` still prices the recorded path in rupees."""
        if option_df is None or option_df.empty:
            return None
        start = _market_naive(start_ts).to_pydatetime()
        end = _market_naive(end_ts).to_pydatetime()
        force_exit_time = datetime.strptime(
            str(getattr(get_settings(), "ut_force_exit_time", "15:25") or "15:25"),
            "%H:%M",
        ).time()
        if str(replay_source_lane or "").upper() in {"D1", "D2"} and _setting_enabled(
            getattr(get_settings(), "replay_s12_native_exits", True), True
        ):
            from engine.s12_lane import square_off as s12_square_off

            force_exit_time = datetime.strptime(s12_square_off(get_settings()), "%H:%M").time()
        session_end = datetime.combine(start.date(), force_exit_time)
        end = min(end, session_end)
        frame = option_df.copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            return None
        frame.index = pd.DatetimeIndex([_market_naive(ts) for ts in frame.index])
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        window = frame.loc[(frame.index >= start) & (frame.index + pd.Timedelta(minutes=1) <= end)]
        if window.empty or window.index[0] - pd.Timestamp(start) >= pd.Timedelta(minutes=1):
            return None
        required = ["open", "high", "low", "close"]
        if not all(key in window.columns for key in required):
            return None
        entry_time = window.index[0].to_pydatetime()
        entry_premium = float(window.iloc[0]["open"])
        if entry_premium <= 0:
            return None
        risk_from_price = entry_premium - float(sl_price)
        uts_replay = str(replay_source_lane).upper() == "UTS"
        replay_holder = replay_setup
        if uts_replay:
            # The explicit replay lane is authoritative. Legacy setup objects
            # may lack lane/timeframe metadata, so retain their evidence while
            # filling those identities on a copy rather than changing signals.
            replay_holder = copy.copy(replay_setup) if replay_setup is not None else SimpleNamespace()
            replay_holder.source_lane = "UTS"
            if not str(getattr(replay_holder, 'instrument', '') or '').strip():
                replay_holder.instrument = tf15_instrument
            if not str(getattr(replay_holder, "timeframe", "") or "").strip():
                replay_holder.timeframe = tf15_timeframe or "15min"
        replay_cap_pct = option_stop_tolerance_pct(
            float(max_sl_pct or 0.15) * 100.0,
            replay_regime,
            confidence=replay_confidence if replay_grade else None,
            grade=replay_grade,
            setup=replay_holder,
        )
        max_sl_dist = entry_premium * replay_cap_pct / 100.0
        if natural_sl_distance is not None and float(natural_sl_distance or 0) > 0:
            risk = min(float(natural_sl_distance), max_sl_dist)
        elif risk_from_price >= max(0.5, entry_premium * 0.01):
            risk = min(risk_from_price, max_sl_dist)
        else:
            # The pre-estimated stop can be stale once real option candles supply the entry.
            # Re-anchor to a meaningful premium risk before checking SL/target hits.
            risk = max_sl_dist
        risk = min(max_sl_dist, max(0.5, risk))
        uts_replay = str(replay_source_lane).upper() == "UTS"
        if uts_replay:
            # Re-anchor the UTS guard to the observed option fill, not the
            # provisional premium used before this price path was available.
            risk = min(max_sl_dist, uts_lane_policy.premium_stop_distance(
                entry_premium, required_stop_distance, 1.0, get_settings(),
                regime=replay_regime, confidence=replay_confidence if replay_grade else None,
                grade=replay_grade,
                setup=replay_holder,
            ))
        # Native S1 management (2026-10-08 audit): the replay used to run S1 trades
        # through the other engines' generic guards with a 2.5-3.8R target, so S1
        # replay results said nothing about live S1. Use the lane's own premium risk,
        # target and per-bar exit_decision instead.
        s1_replay = (
            not uts_replay
            and str(replay_source_lane).upper() == "S1"
            and _setting_enabled(getattr(get_settings(), "replay_s1_native_exits", True), True)
        )
        s1_trade = None
        if s1_replay:
            s1_settings = get_settings()
            s1_score = float(getattr(replay_setup, "s1_score", 0.0) or 0.0)
            s1_guard = s1_lane_policy.premium_guard_pct(
                tf15_timeframe or "5min", entry_time, s1_score, s1_settings, regime=replay_regime)
            s1_index_dist = abs(float(getattr(replay_setup, "stop_distance", 0.0) or 0.0))
            mapped = float(natural_sl_distance or 0.0)
            risk = max(0.05, min(entry_premium * s1_guard / 100.0, mapped) if mapped > 0
                       else entry_premium * s1_guard / 100.0)
            weak = s1_lane_policy.entry_is_weak(replay_setup, replay_regime, tf15_direction, s1_settings)
            rr = s1_lane_policy.target_rr(s1_settings, weak=weak)
            s1_dir = 1 if str(tf15_direction or "LONG").upper() in {"LONG", "BUY"} else -1
            s1_spot_entry = float(getattr(replay_setup, "price", 0.0) or 0.0)
            s1_trade = SimpleNamespace(
                direction="LONG" if s1_dir == 1 else "SHORT", inst_type="OPT",
                entry_price=entry_premium, initial_stop=max(0.05, entry_premium - risk),
                timeframe=tf15_timeframe, source_lane="S1", exit_policy_owner="S1",
                underlying_entry=s1_spot_entry,
                underlying_stop=(s1_spot_entry - s1_dir * s1_index_dist) if s1_index_dist > 0 else 0.0,
                s1_score=s1_score, s1_peak_premium=entry_premium, lots=1, lot_size=max(1, int(qty or 1)),
                signal_time=getattr(replay_setup, "timestamp", None),
            )
            s1_states = self._s1_replay_bar_states(tf15_exit_scanner, tf15_instrument, tf15_timeframe)
            s1_spots = None
            try:
                s1_spots = tf15_exit_scanner.candles.get_candles(tf15_instrument, "1min")
                if s1_spots is not None and not s1_spots.empty:
                    s1_spots = s1_spots.copy()
                    s1_spots.index = pd.DatetimeIndex([_market_naive(ts) for ts in s1_spots.index])
                    s1_spots = s1_spots[~s1_spots.index.duplicated(keep="last")].sort_index()
            except Exception:
                s1_spots = None
            s1_tf_min = {"1min": 1, "5min": 5, "15min": 15}.get(str(tf15_timeframe or ""), 5)
            s1_signal_ts = getattr(replay_setup, "timestamp", None)
            s1_signal_close = (_market_naive(s1_signal_ts) + pd.Timedelta(minutes=s1_tf_min)) if s1_signal_ts is not None else None
            s1_exit_bar = getattr(replay_setup, "s1_exit_time", None)
            s1_engine_exit_at = (_market_naive(s1_exit_bar) + pd.Timedelta(minutes=s1_tf_min)) if s1_exit_bar is not None else None
            s1_margin = float(getattr(s1_settings, "s1_lane_trend_margin", 20.0) or 20.0)
        s12_replay = (
            str(replay_source_lane or "").upper() in {"D1", "D2"}
            and _setting_enabled(getattr(get_settings(), "replay_s12_native_exits", True), True)
        )
        s12_trade = None
        s12_spots = None
        s12_market = None
        s12_direction = 1 if str(tf15_direction or "LONG").upper() in {"LONG", "BUY"} else -1
        s12_tf_min = {"1min": 1, "5min": 5, "15min": 15}.get(str(tf15_timeframe or ""), 5)
        if s12_replay:
            from engine import s12_lane as s12_policy

            s12_settings = get_settings()
            replay_holder = copy.copy(replay_setup) if replay_setup is not None else SimpleNamespace()
            replay_holder.source_lane = str(replay_source_lane).upper()
            replay_holder.s12_lane = str(replay_source_lane).upper()
            replay_holder.direction = "LONG" if s12_direction == 1 else "SHORT"
            replay_holder.timeframe = tf15_timeframe or "5min"
            replay_holder.instrument = tf15_instrument
            if tf15_exit_scanner is None:
                return None
            try:
                s12_spots = tf15_exit_scanner.candles.get_candles(tf15_instrument, "1min")
                if s12_spots is not None and not s12_spots.empty:
                    s12_spots = s12_spots.copy()
                    s12_spots.index = pd.DatetimeIndex([_market_naive(ts) for ts in s12_spots.index])
                    s12_spots = s12_spots[~s12_spots.index.duplicated(keep="last")].sort_index()
            except Exception:
                s12_spots = None
            if s12_spots is None or s12_spots.empty or entry_time not in s12_spots.index:
                return None
            entry_spot = float(s12_spots.loc[entry_time].get("open", 0.0) or 0.0)
            if entry_spot <= 0:
                return None
            s12_levels = s12_policy.admission_levels(replay_holder, entry_spot, s12_settings)
            replay_holder.s12_entry_spot = entry_spot
            for key, value in s12_levels.items():
                setattr(replay_holder, key, value)
            replay_cap_pct = s12_policy.premium_cap_pct(
                replay_holder,
                s12_settings,
                operator_cap=float(getattr(s12_settings, "options_sl_pct", max_sl_pct * 100.0) or max_sl_pct * 100.0),
                timeframe=tf15_timeframe,
            )
            max_sl_dist = entry_premium * replay_cap_pct / 100.0
            risk = max(0.05, max_sl_dist)
            stop_updates = []
            for update in getattr(replay_holder, "s12_stop_updates", []) or []:
                try:
                    when, level = update
                    stop_updates.append({"effective_from": _market_naive(when), "stop": float(level)})
                except (TypeError, ValueError):
                    continue
            milestones = {}
            for name, when in (getattr(replay_holder, "s12_milestones", {}) or {}).items():
                try:
                    milestones[str(name)] = {"time": _market_naive(when)}
                except Exception:
                    continue
            s12_trade = SimpleNamespace(
                instrument=tf15_instrument,
                timeframe=tf15_timeframe or "5min",
                direction="LONG" if s12_direction == 1 else "SHORT",
                source_lane=str(replay_source_lane).upper(),
                s12_lane=str(replay_source_lane).upper(),
                s12_trade_id=str(getattr(replay_holder, "s12_trade_id", "") or ""),
                signal_time=getattr(replay_holder, "timestamp", None),
                inst_type="OPT",
                entry_price=entry_premium,
                initial_stop=max(0.05, entry_premium - risk),
                s12_premium_cap_pct=replay_cap_pct,
                **s12_levels,
            )
            for name in ("s12_t1_hit", "s12_t2_hit", "s12_t3_hit"):
                setattr(s12_trade, name, bool(getattr(replay_holder, name, False)))
            s12_market = {
                "lane": str(replay_source_lane).upper(),
                "trade_id": s12_trade.s12_trade_id,
                "stop_updates": stop_updates,
                "milestones": milestones,
            }
            live_exits = False  # S12 uses the engine stop/EXIT and its own premium cap.
            live_qty = fast_exit_qty = 0
        sl_price = max(0.05, entry_premium - risk)
        target_price = 0.0 if s12_trade is not None else entry_premium + (risk * rr)
        exit_price = float(window.iloc[-1]["close"])
        exit_time = window.index[-1] + pd.Timedelta(minutes=1)
        exit_reason = "HISTORICAL"
        exit_detail = ""
        uts_trade = None
        if s1_trade is not None:
            live_exits = False  # S1 has its own lane policy, not the generic guards.
            s1_active_stop = sl_price
            s1_active_target = target_price
        if uts_replay:
            live_exits = False  # UTS has its own lane policy, not the generic guards.
            uts_trade = SimpleNamespace(inst_type="OPT", entry_price=entry_premium,
                                       initial_stop=sl_price, s1_peak_premium=entry_premium)
            uts_active_stop = sl_price

        # Optionally let the replay see the LIVE trade-management guards. Without this
        # the only reachable outcomes are SL HIT / TARGET_HIT / HISTORICAL, so a
        # backtest measures entry selection and a fixed stop/target and nothing else.
        # The guard is CALLED, never copied — one implementation, no second copy to
        # drift. Imported here because scanner imports this module.
        live_qty = 0
        fast_exit_qty = 0
        path_qty = 0 if live_exits else max(0, int(qty or 0))
        try:
            if live_exits and _setting_enabled(
                getattr(get_settings(), "backtest_live_exits_enabled", False), False
            ):
                live_qty = max(0, int(qty or 0))
                fast_exit_qty = max(0, int(tf15_qty or 0))
        except Exception:
            live_qty = 0
        peak_rs = 0.0
        price_path = []
        input_hash = hashlib.sha256(pd.util.hash_pandas_object(window, index=True).values.tobytes()).hexdigest()
        input_meta = {"time": pd.Timestamp(entry_time).isoformat(), "guard": "replay_input", "state": "VERIFIED_PRICE_PATH", "input_sha256": input_hash,
                      "eligible_at": pd.Timestamp(start).isoformat(), "entry_time": pd.Timestamp(entry_time).isoformat(), "entry_price": entry_premium,
                      "fill_model": "first_available_1m_open;stop_first_when_ohlc_order_unknown;gap_stop_at_open;close_guard_at_close",
                      "contract": dict(option_df.attrs), "rows": len(window), "last_closed_at": pd.Timestamp(exit_time).isoformat()}
        if uts_replay:
            input_meta["fill_model"] += (
                ";uts_peak_uses_observed_high;new_uts_lock_cross_at_observed_close"
                ";raised_uts_lock_applies_next_bar"
            )
        if replay_stop_context:
            input_meta.update(copy.deepcopy(replay_stop_context))
        input_meta["option_stop_cap_pct"] = replay_cap_pct
        exit_trace = [input_meta]
        evaluated_guards = []
        live_probe = None
        # Unit-level/premium-only callers have no historical index candle store.
        # Preserve the exact production cap guard for that narrow case; full engine
        # replays use the point-in-time stack below.
        live_guard = None
        if live_qty > 0 and tf15_exit_scanner is None:
            try:
                from scanner import Scanner as _Scanner
                live_guard = _Scanner._peak_giveback_cap_exit
            except Exception:
                live_guard = None
        live_self = SimpleNamespace(_setting_enabled=_setting_enabled)
        live_trade = SimpleNamespace(
            instrument=str(tf15_instrument or ""), inst_type="OPT",
            entry_price=entry_premium, price=entry_premium,
            initial_stop=sl_price, trailing_stop=sl_price,
            quantity=live_qty, pnl_quantity=live_qty, lots=1,
            lot_size=max(1, live_qty), entry_time=entry_time,
        )
        if live_qty > 0 and tf15_exit_scanner is not None:
            try:
                live_probe = self._historical_live_management_probe(
                    tf15_exit_scanner,
                    instrument=tf15_instrument,
                    direction=tf15_direction,
                    option_type=tf15_option_type,
                    timeframe=tf15_timeframe,
                    entry_time=entry_time,
                    entry_price=entry_premium,
                    stop_price=sl_price,
                    target_price=target_price,
                    qty=live_qty,
                    grade=replay_grade,
                    confidence=replay_confidence,
                    entry_regime=replay_regime,
                )
            except Exception as exc:
                exit_trace.append({
                    "time": pd.Timestamp(start).isoformat(),
                    "guard": "live_management_probe",
                    "state": "ERROR_ABSTAIN",
                    "detail": f"{type(exc).__name__}: {exc}",
                })

        # The dashboard's Backtest Live Exits switch is the explicit A/B boundary.
        # Build a point-in-time view only for 15m trades and call the production rule;
        # default/off replays remain byte-for-byte stop/target simulations.
        tf15_probe = None
        if (
            fast_exit_qty > 0
            and str(tf15_timeframe or "").lower() == "15min"
            and tf15_exit_scanner is not None
            and _setting_enabled(
                getattr(get_settings(), "fifteen_min_one_min_support_exit_enabled", True),
                True,
            )
        ):
            try:
                tf15_probe = self._historical_tf15_exit_probe(
                    tf15_exit_scanner,
                    instrument=tf15_instrument,
                    direction=tf15_direction,
                    option_type=tf15_option_type,
                    entry_time=entry_time,
                    entry_price=entry_premium,
                    stop_price=sl_price,
                    qty=fast_exit_qty,
                )
            except Exception:
                tf15_probe = None

        previous_bar = None
        for ts, row in window.iterrows():
            # Validate only the path actually reached. A missing or malformed
            # future bar after an already executed exit cannot change admission.
            if previous_bar is not None and ts - previous_bar > pd.Timedelta(minutes=1):
                return None
            previous_bar = ts
            values = pd.to_numeric(row[required], errors="coerce")
            if not all(math.isfinite(float(value)) for value in values) or (values <= 0).any() or values["low"] > min(values["open"], values["close"]) or values["high"] < max(values["open"], values["close"]):
                return None
            low = float(row.get("low", row.get("close", 0.0)) or 0.0)
            high = float(row.get("high", row.get("close", 0.0)) or 0.0)
            op = float(row.get("open", row.get("close", 0.0)) or 0.0)

            close_px = float(row.get("close", op) or op)
            peak_qty = live_qty or fast_exit_qty or path_qty
            if peak_qty > 0:
                peak_rs = max(peak_rs, (high - entry_premium) * peak_qty)
            active_stop = float(sl_price)
            active_target = 0.0 if uts_trade is not None or s12_trade is not None else float(target_price)
            if uts_trade is not None:
                active_stop = uts_active_stop
            if s1_trade is not None:
                active_stop, active_target = s1_active_stop, s1_active_target
            if live_probe is not None:
                # Stop updates returned by the prior closed candle apply now. Newly
                # computed stops below apply only to the next candle.
                active_stop = float(getattr(live_probe, "active_stop", active_stop) or active_stop)
                active_target = float(getattr(live_probe, "active_target", active_target) or active_target)

            price_path.append({
                "time": pd.Timestamp(ts).isoformat(), "open": op, "high": high,
                "low": low, "close": close_px,
                "pnl": (close_px - entry_premium) * peak_qty if peak_qty else 0.0,
                "peak_pnl": peak_rs, "active_stop": active_stop,
                "active_target": active_target,
            })

            # Resting stop/target orders exist before the candle. A close-only
            # guard cannot override an earlier hit, and OHLC does not identify
            # which level came first if both traded: choose the adverse stop.
            if low <= active_stop:
                exit_price = min(op, active_stop)
                exit_time = ts if op <= active_stop else ts + pd.Timedelta(minutes=1)
                exit_reason = "S12_PREMIUM_CAP" if s12_trade is not None else "SL HIT"
                break
            elif active_target > 0 and high >= active_target:
                exit_price = active_target
                exit_time = ts if op >= active_target else ts + pd.Timedelta(minutes=1)
                exit_reason = "TARGET_HIT"
                break

            if uts_trade is not None:
                # The prior resting stop has already survived this whole bar.
                # Its observed high is therefore eligible to arm or tighten the
                # premium lock. OHLC does not locate the low relative to that
                # high: only a subsequent close below the new floor proves a
                # same-bar crossing, and that observed close prices the exit.
                # Otherwise the raised stop takes effect in the following bar.
                uts_lane_policy.exit_decision(
                    uts_trade, 0.0, high, now=(ts + pd.Timedelta(minutes=1)).to_pydatetime(),
                )
                decision = uts_lane_policy.exit_decision(
                    uts_trade, 0.0, close_px, now=(ts + pd.Timedelta(minutes=1)).to_pydatetime(),
                )
                uts_active_stop = max(active_stop, float(decision.premium_stop or active_stop))
                price_path[-1]["next_stop"] = uts_active_stop
                evaluated_guards = ["UTS_PREMIUM_STOP", "UTS_PROFIT_LOCK", "UTS_CHART_EXIT"]
                if decision.exit:
                    exit_price, exit_time = close_px, ts + pd.Timedelta(minutes=1)
                    exit_reason, exit_detail = decision.reason, decision.detail
                    break

            if s1_trade is not None:
                # Same fill model as UTS: the resting stop survived this bar, so its
                # favourable extremes may arm the T1/premium lock; only the observed
                # close can prove a same-bar crossing; a raised stop applies next bar.
                # Trend/market come from the latest S1 bar CLOSED after the signal bar.
                bar_end = ts + pd.Timedelta(minutes=1)
                spot_fav = spot_adv = 0.0
                if s1_spots is not None:
                    try:
                        spot_row = s1_spots.loc[ts]
                        spot_fav = float(spot_row["high"] if s1_dir == 1 else spot_row["low"])
                        spot_adv = float(spot_row["low"] if s1_dir == 1 else spot_row["high"])
                    except Exception:
                        spot_fav = spot_adv = 0.0
                s1_trend, s1_market = 0, None
                if s1_states is not None and s1_signal_close is not None:
                    closed = s1_states.loc[(s1_states.index <= bar_end) & (s1_states.index > s1_signal_close)]
                    if not closed.empty:
                        last = closed.iloc[-1]
                        gap = (float(last["bull"]) - float(last["bear"])) * s1_dir
                        s1_trend = 1 if gap >= s1_margin else -1 if -gap >= s1_margin else 0
                        s1_market = {"adx_value": float(last["adx"]), "s1_rel_volume": float(last["rel_vol"]),
                                     "s1_atr_expansion": float(last["atr_exp"])}
                s1_engine_exit = ({"reason": str(getattr(replay_setup, "s1_exit_reason", "") or "S1 exit")}
                                  if s1_engine_exit_at is not None and bar_end >= s1_engine_exit_at else None)
                s1_lane_policy.exit_decision(
                    s1_trade, spot_fav, high, now=bar_end.to_pydatetime(), settings=s1_settings,
                    trend=s1_trend, regime=replay_regime, market=s1_market,
                )
                decision = s1_lane_policy.exit_decision(
                    s1_trade, spot_adv, close_px, now=bar_end.to_pydatetime(), engine_exit=s1_engine_exit,
                    settings=s1_settings, trend=s1_trend, regime=replay_regime, market=s1_market,
                )
                if decision.premium_stop and float(decision.premium_stop) > s1_active_stop:
                    s1_active_stop = float(decision.premium_stop)
                if decision.target_premium and float(decision.target_premium) > 0:
                    s1_base_target = float(decision.target_premium)
                    s1_active_target = min(s1_active_target or s1_base_target, s1_base_target)
                if decision.runner:
                    s1_active_target = 0.0
                elif s1_active_target <= 0:
                    s1_active_target = float(target_price)
                price_path[-1]["next_stop"] = s1_active_stop
                if decision.exit:
                    exit_price, exit_time = close_px, bar_end
                    exit_reason, exit_detail = decision.reason, decision.detail
                    break

            if s12_trade is not None:
                # Engine STOP_UPDATED values become resting index protection only at
                # their effective time. Evaluate the bar's adverse extreme against the
                # stop that was already active at its open; a newly raised stop applies
                # to the following minute.
                s12_policy.apply_engine_view(s12_trade, s12_market, now=ts.to_pydatetime())
                try:
                    spot_row = s12_spots.loc[ts]
                    spot_adverse = float(spot_row["low"] if s12_direction == 1 else spot_row["high"])
                    spot_close = float(spot_row["close"])
                except Exception:
                    return None
                bar_end = ts + pd.Timedelta(minutes=1)
                exit_bar = getattr(replay_holder, "s12_exit_time", None)
                engine_exit = None
                if exit_bar is not None and getattr(replay_holder, "s12_exit_reason", ""):
                    engine_exit_at = _market_naive(exit_bar) + pd.Timedelta(minutes=s12_tf_min)
                    if bar_end >= engine_exit_at:
                        engine_exit = {
                            "reason": str(getattr(replay_holder, "s12_exit_reason", "") or ""),
                            "time": exit_bar,
                            "price": getattr(replay_holder, "s12_exit_price", None),
                            "trade_id": s12_trade.s12_trade_id,
                        }
                decision = s12_policy.exit_decision(
                    s12_trade,
                    spot_adverse,
                    close_px,
                    now=bar_end.to_pydatetime(),
                    engine_exit=engine_exit,
                    settings=s12_settings,
                )
                price_path[-1]["s12_stop"] = float(decision.index_stop or 0.0)
                price_path[-1]["s12_spot_close"] = spot_close
                if decision.exit:
                    exit_price, exit_time = close_px, bar_end
                    exit_reason, exit_detail = decision.reason, decision.detail
                    break
                # Closed-bar stop and milestone changes are effective from the next
                # minute, after this bar's adverse excursion has been evaluated.
                s12_policy.apply_engine_view(s12_trade, s12_market, now=bar_end.to_pydatetime())
                price_path[-1]["next_s12_stop"] = float(s12_policy.current_stop(s12_trade) or 0.0)
                price_path[-1]["s12_t1_hit"] = bool(getattr(s12_trade, "s12_t1_hit", False))
                price_path[-1]["s12_t2_hit"] = bool(getattr(s12_trade, "s12_t2_hit", False))
                price_path[-1]["s12_t3_hit"] = bool(getattr(s12_trade, "s12_t3_hit", False))

            if live_guard is not None:
                try:
                    verdict = live_guard(
                        live_self, live_trade,
                        current_pnl_rs=(close_px - entry_premium) * live_qty,
                        peak_pnl_rs=peak_rs,
                        evaluated_at=pd.Timestamp(ts).to_pydatetime() + timedelta(minutes=1),
                    )
                    fired, detail = verdict
                    if fired:
                        exit_price = close_px
                        exit_time = ts + pd.Timedelta(minutes=1)
                        exit_reason = getattr(verdict, "rule", "") or "PROFIT_FLOOR_EXIT"
                        exit_detail = str(detail or "")
                        exit_trace.append({
                            "time": pd.Timestamp(ts).isoformat(), "guard": exit_reason,
                            "state": "EXIT", "detail": exit_detail,
                        })
                        break
                except Exception as exc:
                    exit_trace.append({
                        "time": pd.Timestamp(ts).isoformat(), "guard": "peak_giveback_cap",
                        "state": "ERROR_ABSTAIN", "detail": f"{type(exc).__name__}: {exc}",
                    })

            if live_probe is not None:
                try:
                    reason, detail, next_stop, regime, probe_trace, guards = live_probe(
                        ts, close_px, high, peak_rs
                    )
                    live_probe.active_stop = max(active_stop, float(next_stop or active_stop))
                    # Runner removes the hard target only after a causal close.
                    if any(
                        item.get("guard") == "runner" and item.get("state") == "ARMED"
                        for item in probe_trace
                    ):
                        live_probe.active_target = 0.0
                    else:
                        live_probe.active_target = active_target
                    price_path[-1]["regime"] = regime
                    price_path[-1]["next_stop"] = live_probe.active_stop
                    exit_trace = [input_meta] + list(probe_trace)
                    evaluated_guards = list(guards)
                    if reason:
                        exit_price = close_px
                        exit_time = ts + pd.Timedelta(minutes=1)
                        exit_reason = reason
                        exit_detail = str(detail or "")
                        exit_trace.append({
                            "time": pd.Timestamp(ts).isoformat(), "guard": reason,
                            "state": "EXIT", "detail": exit_detail,
                        })
                        break
                except Exception as exc:
                    exit_trace.append({
                        "time": pd.Timestamp(ts).isoformat(), "guard": "live_management_probe",
                        "state": "ERROR_ABSTAIN", "detail": f"{type(exc).__name__}: {exc}",
                    })

            if tf15_probe is not None:
                close_px = float(row.get("close", op) or op)
                try:
                    fired, _why = tf15_probe(
                        ts,
                        close_px,
                        high,
                        peak_rs,
                    )
                except Exception:
                    fired = False
                if fired:
                    exit_price = close_px
                    exit_time = ts + pd.Timedelta(minutes=1)
                    exit_reason = "TF15_FAST_REVERSAL"
                    exit_detail = str(_why or "")
                    break

        if exit_reason == "HISTORICAL" and pd.Timestamp(exit_time) < pd.Timestamp(end):
            return None
        if uts_replay and exit_reason == "HISTORICAL" and chart_exit_reason:
            exit_reason = uts_lane_policy.EXIT_REASONS.get(chart_exit_reason, "UTS_EXIT")
        exit_trace.append({"time": pd.Timestamp(exit_time).isoformat(), "guard": exit_reason, "state": "FILL", "price": float(exit_price), "detail": exit_detail})
        return {
            "entry_time": entry_time,
            "input_sha256": input_hash,
            "entry": entry_premium,
            "exit": max(0.05, exit_price),
            "stop": sl_price,
            "target": target_price,
            "exit_time": exit_time,
            "reason": exit_reason,
            "detail": exit_detail,
            "price_path": price_path,
            "exit_trace": exit_trace,
            "evaluated_guards": evaluated_guards,
            "s12_state": (
                {key: copy.deepcopy(getattr(s12_trade, key)) for key in s12_lane_policy.SESSION_STATE_FIELDS
                 if hasattr(s12_trade, key)}
                if s12_trade is not None else {}
            ),
        }

    def _historical_tf15_exit_probe(
        self,
        scanner,
        *,
        instrument: str,
        direction: str,
        option_type: str,
        entry_time: datetime,
        entry_price: float,
        stop_price: float,
        qty: int,
    ):
        """Return a no-future-leak adapter around the production 15m/1m exit."""
        from scanner import Scanner as _Scanner
        from intelligence.regime_detector import RegimeDetector

        candle_view = _PointInTimeCandles(getattr(scanner, "candles", None))
        probe = _Scanner.__new__(_Scanner)
        probe.mode = "HISTORICAL"
        probe.candles = candle_view
        probe.active_indices = list(
            getattr(scanner, "active_indices", None)
            or ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
        )
        probe.latest_regimes = {}
        probe._timeframe_regimes = {}
        probe._cached_results = {}
        probe._diagnostics = {}
        probe._timeframe_intel_cache = {}
        regime_detector = RegimeDetector()
        self._bind_historical_intelligence(probe, scanner, candle_view)

        trade = SimpleNamespace(
            instrument=str(instrument or "").upper(),
            timeframe="15min",
            direction=str(direction or "").upper(),
            option_type=str(option_type or "").upper(),
            inst_type="OPT",
            entry_time=entry_time,
            entry_price=float(entry_price),
            current_price=float(entry_price),
            initial_stop=float(stop_price),
            trailing_stop=float(stop_price),
            current_stop=float(stop_price),
            quantity=max(1, int(qty)),
            peak_pnl=0.0,
        )

        def evaluate(ts, close_px: float, high_px: float, peak_pnl_rs: float):
            bar_start = pd.Timestamp(ts)
            if bar_start.tzinfo is not None:
                bar_start = bar_start.tz_convert(IST).tz_localize(None)
            candle_view.as_of = bar_start + pd.Timedelta(minutes=1)
            evaluation_time = bar_start.to_pydatetime() + timedelta(minutes=1)
            current_spot = candle_view.get_latest_price(trade.instrument) or 0.0
            # Rebuild the trade's rolling own-timeframe regime from only the candles
            # visible at this replay minute. Option-chain intelligence is not invented;
            # it remains unavailable and therefore casts no vote.
            try:
                regime_frame = candle_view.get_candles(trade.instrument, "15min")
                regime_result = regime_detector.detect(
                    regime_frame,
                    trade.instrument,
                    "15min",
                )
                regime_value = getattr(
                    regime_result.get("regime"),
                    "value",
                    regime_result.get("regime", "UNKNOWN"),
                )
                probe._timeframe_regimes[
                    probe._timeframe_intel_key(trade.instrument, "15min")
                ] = str(regime_value or "UNKNOWN").upper()
            except Exception:
                pass
            trade.current_price = float(close_px)
            trade.peak_pnl = max(float(trade.peak_pnl or 0.0), float(peak_pnl_rs or 0.0))
            verdict = probe._fifteen_min_one_min_support_exit_decision(
                trade,
                float(current_spot),
                evaluation_time,
                current_mark=float(close_px),
                peak_pnl_rs=float(peak_pnl_rs or 0.0),
                observed_contract_ticks=[
                    (float(high_px), evaluation_time),
                    (float(close_px), evaluation_time),
                ],
            )
            trade.peak_pnl = max(
                float(trade.peak_pnl or 0.0),
                (float(high_px) - float(entry_price)) * max(1, int(qty)),
            )
            return verdict

        return evaluate
    def _select_historical_option_candidate(
        self,
        scanner,
        instrument: str,
        direction: str,
        price: float,
        trailing_stop: float,
        index_stop_distance: float,
        rr: float,
        cfg: Dict,
        start_ts: datetime,
        end_ts: datetime,
    ) -> Optional[Dict]:
        """
        Evaluate candidate option strikes (ATM and/or ITM) chronologically against intraday candles
        and return the selected option details.
        """
        from config.settings import get_settings
        interval = cfg.get("strike_interval", 50)
        atm_strike = round(float(price) / interval) * interval
        option_type = "CE" if direction == "LONG" else "PE"
        itm_strike = (atm_strike - interval) if option_type == "CE" else (atm_strike + interval)
        
        selection = getattr(get_settings(), "option_strike_selection", "ATM")
        target_strikes = []
        if selection == "ATM":
            target_strikes = [atm_strike]
        elif selection == "ITM":
            target_strikes = [itm_strike]
        else:  # BOTH
            # Evaluate both ATM and ITM. ITM is preferred/safer, so evaluate it first
            target_strikes = [itm_strike, atm_strike]
            
        for strike in target_strikes:
            option_df, trading_symbol, symbol_token = self._load_option_candles(
                scanner, instrument, strike, option_type, "1min", start_ts.date(), now_dt=start_ts
            )
            if option_df.empty:
                continue
                
            # Estimate entry/stop/target/exit using _simulate_actual_option_exit
            # Estimate entry premium (1.2% of spot)
            est_entry = _estimate_opt_premium(price, scanner, instrument, _sig_date(start_ts))
            spot_stop_dist = abs(float(price) - float(trailing_stop))
            sim_mult = 0.5
            natural_sl_premium_dist = abs(spot_stop_dist * sim_mult)
            options_sl_pct = getattr(scanner, "options_sl_pct", 15.0)
            max_sl_dist = est_entry * (options_sl_pct / 100.0)
            effective_sl_dist = max(0.05, min(natural_sl_premium_dist, max_sl_dist))
            opt_sl_price = max(0.05, est_entry - effective_sl_dist)
            
            actual_exit = self._simulate_actual_option_exit(
                option_df=option_df,
                start_ts=start_ts,
                end_ts=end_ts,
                sl_price=opt_sl_price,
                rr=rr,
                natural_sl_distance=natural_sl_premium_dist,
                max_sl_pct=options_sl_pct / 100.0,
            )
            if actual_exit:
                return {
                    "strike": strike,
                    "trading_symbol": trading_symbol,
                    "symbol_token": symbol_token,
                    "option_df": option_df,
                    "entry": actual_exit["entry"],
                    "stop": actual_exit["stop"],
                    "exit": actual_exit["exit"],
                    "target": actual_exit["target"],
                    "exit_time": actual_exit["exit_time"],
                    "reason": actual_exit["reason"] if actual_exit["reason"] != "HISTORICAL" else "SESSION_END",
                }
        return None

    @staticmethod
    def _simulate_futures_exit_path(
        window,
        entry: float,
        stop: float,
        target: float,
        direction: str,
    ):
        """Walk index candles intrabar for a replayed FUT trade — SL/target with
        the same open-proximity tie-break the OPTION replay gets (exit parity,
        2026-07-09 FUT-vs-OPT audit: shadows previously never simulated their
        target/stop path and just rode to the option trade's exit, inflating
        backtest FUT P&L). Returns (exit_price, reason, exit_ts) — reason is
        "" when neither level hit (caller keeps its default)."""
        is_long = str(direction or "").upper() == "LONG"
        for ts, row in window.iterrows():
            low = float(row.get("low", row.get("close", 0.0)) or 0.0)
            high = float(row.get("high", row.get("close", 0.0)) or 0.0)
            op = float(row.get("open", row.get("close", 0.0)) or 0.0)
            if is_long:
                sl_hit, tgt_hit = low <= stop, high >= target
            else:
                sl_hit, tgt_hit = high >= stop, low <= target
            if sl_hit:
                fill = min(op, stop) if is_long else max(op, stop)
                filled_at = ts if fill == op else ts + pd.Timedelta(minutes=1)
                return float(fill), "SL HIT", filled_at
            if tgt_hit:
                at_open = op >= target if is_long else op <= target
                return float(target), "TARGET_HIT", ts if at_open else ts + pd.Timedelta(minutes=1)
        return None, "", None

    @staticmethod
    def _simulate_s12_futures_exit_path(
        window,
        *,
        signal,
        levels: Dict,
        market: Dict,
        instrument: str,
        timeframe: str,
        direction: str,
        entry_price: float,
        settings,
    ) -> Optional[Dict]:
        """Walk closed 1m index bars using D1/D2 stops, engine exits and square-off."""
        if window is None or window.empty:
            return None
        trade = SimpleNamespace(
            instrument=instrument,
            timeframe=timeframe,
            direction=direction,
            source_lane=str(getattr(signal, "source_lane", "") or getattr(signal, "s12_lane", "")),
            s12_lane=str(getattr(signal, "s12_lane", "") or getattr(signal, "source_lane", "")),
            s12_trade_id=str(getattr(signal, "s12_trade_id", "") or ""),
            signal_time=getattr(signal, "timestamp", None),
            inst_type="FUT",
            entry_price=float(entry_price or 0.0),
            initial_stop=float(levels.get("s12_stop") or 0.0),
            **levels,
        )
        for name in ("s12_t1_hit", "s12_t2_hit", "s12_t3_hit"):
            setattr(trade, name, bool(getattr(signal, name, False)))
        for bar_ts, bar in window.iterrows():
            bar_ts = _market_naive(bar_ts)
            bar_end = bar_ts + pd.Timedelta(minutes=1)
            s12_lane_policy.apply_engine_view(trade, market, now=bar_ts.to_pydatetime())
            op = float(bar.get("open", bar.get("close", 0.0)) or 0.0)
            low = float(bar.get("low", bar.get("close", 0.0)) or 0.0)
            high = float(bar.get("high", bar.get("close", 0.0)) or 0.0)
            close = float(bar.get("close", op) or op)
            stop = s12_lane_policy.current_stop(trade)
            crossed = stop > 0 and (
                (direction == "LONG" and low <= stop)
                or (direction != "LONG" and high >= stop)
            )
            adverse = low if direction == "LONG" else high
            engine_exit = None
            exit_bar = getattr(signal, "s12_exit_time", None)
            exit_reason = str(getattr(signal, "s12_exit_reason", "") or "")
            if exit_bar is not None and exit_reason:
                minutes = {"1min": 1, "5min": 5, "15min": 15}.get(str(timeframe), 5)
                if bar_end >= _market_naive(exit_bar) + pd.Timedelta(minutes=minutes):
                    engine_exit = {
                        "reason": exit_reason,
                        "time": exit_bar,
                        "price": getattr(signal, "s12_exit_price", None),
                        "trade_id": trade.s12_trade_id,
                    }
            decision = s12_lane_policy.exit_decision(
                trade,
                adverse if crossed else close,
                0.0,
                now=bar_end.to_pydatetime(),
                engine_exit=engine_exit,
                settings=settings,
            )
            if decision.exit:
                if crossed:
                    fill = min(op, stop) if direction == "LONG" else max(op, stop)
                    return {
                        "trade": trade, "price": float(fill),
                        "time": bar_ts if fill == op else bar_end,
                        "reason": decision.reason, "stop_hit": True,
                    }
                return {"trade": trade, "price": close, "time": bar_end,
                        "reason": decision.reason, "stop_hit": False}
            s12_lane_policy.apply_engine_view(trade, market, now=bar_end.to_pydatetime())
        last_ts = _market_naive(window.index[-1])
        return {
            "trade": trade,
            "price": float(window.iloc[-1].get("close", entry_price) or entry_price),
            "time": last_ts + pd.Timedelta(minutes=1),
            "reason": "",
            "stop_hit": False,
        }

    @staticmethod
    def _hybrid_backtest_dual_futures_enabled(scanner) -> bool:
        pref = str(getattr(scanner, "inst_pref", "AUTO") or "AUTO").upper()
        enabled = _setting_enabled(
            getattr(get_settings(), "futures_shadow_enabled", False),
            False,
        )
        return (
            enabled
            and SignalProcessor._historical_trade_creation_allowed(scanner)
            and pref in {"AUTO", "HYBRID"}
        )

    @staticmethod
    def _historical_trade_creation_allowed(scanner) -> bool:
        guard = getattr(scanner, "allows_historical_trade_creation", None)
        if callable(guard):
            return bool(guard())
        return str(getattr(scanner, "mode", "") or "").upper() == "HISTORICAL"

    @staticmethod
    def _historical_signal_date_allowed(scanner, signal_ts) -> bool:
        if not SignalProcessor._historical_trade_creation_allowed(scanner):
            return False
        cutoff_getter = getattr(scanner, "historical_session_cutoff_date", None)
        if not callable(cutoff_getter) or signal_ts is None:
            return True
        try:
            return signal_ts.date() <= cutoff_getter()
        except Exception:
            return True

    def _build_historical_futures_shadow_trade(
        self,
        scanner,
        trade_id: str,
        instrument: str,
        timeframe: str,
        direction: str,
        signal,
        exit_spot: float,
        exit_time,
        grade: str,
        confidence: float,
        rr: float,
        lot_size: int,
        display_suffix: str,
        default_exit_reason: str,
        regime: str = "UNKNOWN",
    ) -> Optional[Trade]:
        entry = float(getattr(signal, "price", 0.0) or 0.0)
        if entry <= 0:
            return None
        trailing_stop = float(getattr(signal, "trailing_stop", entry) or entry)
        configured_cap = float(getattr(scanner, "futures_sl_pct", 0.30) or 0.30)
        effective_pct = futures_stop_tolerance_pct(configured_cap, regime)
        max_sl_dist = entry * (effective_pct / 100.0)
        if direction == "LONG":
            stop = max(trailing_stop, entry - max_sl_dist)
            target = entry + ((entry - stop) * rr)
        else:
            stop = min(trailing_stop, entry + max_sl_dist)
            target = entry - ((stop - entry) * rr)

        # Exit parity (2026-07-09 audit): simulate the shadow's OWN SL/target
        # path on the index candles — previously the shadow just rode to the
        # option trade's exit spot (no TARGET_HIT ever), inflating backtest FUT
        # P&L on trend days. Falls back to the legacy ride-to-exit when candles
        # are unavailable.
        resolved_exit = float(exit_spot or entry)
        exit_reason = default_exit_reason
        sim_exit_ts = None
        try:
            frame = scanner.candles.get_candles(instrument, timeframe)
            sig_ts = getattr(signal, "timestamp", None)
            if frame is not None and len(frame) and sig_ts is not None:
                start = sig_ts.replace(tzinfo=None) if getattr(sig_ts, "tzinfo", None) else sig_ts
                end = exit_time.replace(tzinfo=None) if (exit_time is not None and getattr(exit_time, "tzinfo", None)) else exit_time
                mask = frame.index > start
                if end is not None:
                    mask = mask & (frame.index <= end)
                window = frame.loc[mask]
                if not window.empty:
                    sim_price, sim_reason, sim_ts = self._simulate_futures_exit_path(
                        window, entry, stop, target, direction
                    )
                    if sim_reason:
                        resolved_exit = float(sim_price)
                        exit_reason = sim_reason
                        sim_exit_ts = sim_ts
        except Exception as exc:
            logger.debug(f"FUT shadow exit simulation skipped for {trade_id}: {exc}")

        if direction == "LONG":
            if resolved_exit <= stop:
                resolved_exit = stop
                exit_reason = "SL HIT"
            gross_pnl = (resolved_exit - entry) * (int(getattr(scanner, "user_lots_fut", {}).get(instrument, 1) or 1) * int(lot_size or 1))
        else:
            if resolved_exit >= stop:
                resolved_exit = stop
                exit_reason = "SL HIT"
            gross_pnl = (entry - resolved_exit) * (int(getattr(scanner, "user_lots_fut", {}).get(instrument, 1) or 1) * int(lot_size or 1))
        if sim_exit_ts is not None:
            exit_time = sim_exit_ts

        lots = int(getattr(scanner, "user_lots_fut", {}).get(instrument, 1) or 1)
        settings = get_settings()
        charges = estimate_trade_charges(
            entry,
            resolved_exit,
            lots * int(lot_size or 1),
            "FUT",
            settings,
        )
        entry_time = getattr(signal, "timestamp", None)
        e_time = IST.localize(entry_time) if (entry_time and getattr(entry_time, "tzinfo", None) is None) else entry_time
        ex_time = IST.localize(exit_time) if (exit_time and getattr(exit_time, "tzinfo", None) is None) else exit_time
        return Trade(
            id=trade_id,
            instrument=instrument,
            timeframe=timeframe,
            direction=direction,
            entry_price=entry,
            entry_time=e_time,
            trailing_stop=stop,
            current_stop=stop,
            lots=lots,
            lot_size=lot_size,
            grade=f"{grade} ({display_suffix}, Fut Shadow)",
            confidence=confidence,
            inst_type="FUT",
            option_type="",
            atm_strike=0.0,
            rr_ratio=rr,
            target=target,
            status="CLOSED",
            exit_price=resolved_exit,
            exit_time=ex_time,
            pnl=gross_pnl - charges,
            charges=charges,
            exit_reason=exit_reason,
            instrument_multiplier=1.0,
            entry_spot=entry,
            spot_stop=trailing_stop,
            spot_target=target,
            trading_symbol="FUT_SHADOW",
            symbol_token="",
            market_regime=str(regime or "UNKNOWN"),
            source_engine=str(getattr(signal, "source_engine", "") or ""),
            source_engines=list(getattr(signal, "source_engines", []) or []),
            source_lane=SignalProcessor._signal_source_lane(signal, get_settings().scanner_engine),
            data_provenance="BACKTEST",
            replay_exit_trace=self._historical_price_evidence("FUT", signal=signal, entry_time=entry_time, shadow=True),
        )

    @staticmethod
    def _signal_source_lane(signal, scanner_engine_mode: str = "") -> str:
        lane = str(getattr(signal, "source_lane", "") or "").upper()
        if lane in {"D1", "D2"}:
            return lane
        if lane in {"A", "B"}:
            return lane
        source = str(getattr(signal, "source_engine", "") or "").lower()
        if source in {"flow_burst", "apex_hybrid"}:
            return "B"
        if lane == "QP" or (configured_mode() in COMBINED_MODES and source in {
            "quad_prime", "utbot", "ema", "supertrend", "consensus", "multi",
        }):
            return "QP"
        if source == "s1":
            # Persisted on the trade (source_engine is not), so a restart keeps the S1 lane.
            return "S1"
        if source == "ut_smart":
            return "UTS"
        if str(scanner_engine_mode or "").lower() == "adaptive_dual":
            return "A"
        return ""

    @staticmethod
    def _historical_replay_source_keys(signal, scanner_engine_mode: str = "") -> List[str]:
        """Return independent replay streams for historical lifecycle building.

        Only true multi-entry engines split into multiple streams. Rank/router engines
        keep one managed lifecycle even when they carry source_engines for context.
        """
        mode = str(scanner_engine_mode or "").lower()
        if mode == "s1s2_dual":
            return [SignalProcessor._signal_source_lane(signal, mode) or "D1"]
        if mode == "adaptive_dual":
            return [SignalProcessor._signal_source_lane(signal, mode) or "A"]
        if mode == "quality_hybrid":
            return [SignalProcessor._quality_hybrid_replay_stream_key(signal)]

        split_modes = {"quad_union", "quad_prime", "utbot_ema", "utbot_consensus"}
        if mode not in split_modes:
            return ["main"]

        raw_sources = getattr(signal, "source_engines", None)
        sources: List[str] = []
        if isinstance(raw_sources, (list, tuple, set)):
            sources.extend(str(src).strip().lower() for src in raw_sources if str(src or "").strip())
        elif isinstance(raw_sources, str) and raw_sources.strip():
            sources.extend(
                part.strip().lower()
                for part in raw_sources.replace("+", ",").replace("/", ",").split(",")
                if part.strip()
            )

        src = str(getattr(signal, "source_engine", "") or "").strip().lower()
        if src == "both" and mode in {"utbot_ema", "utbot_consensus"}:
            sources.extend(["utbot", "ema" if mode == "utbot_ema" else "consensus"])
        elif src and src not in {"multi", "both", "partner", "adaptive_hybrid", "adaptive_dual", "quality_hybrid"}:
            sources.append(src)

        allowed = (
            {"utbot", "ema", "supertrend", "consensus"}
            if mode in {"quad_union", "quad_prime"}
            else {"utbot", "ema" if mode == "utbot_ema" else "consensus"}
        )
        clean: List[str] = []
        for item in sources:
            for part in str(item).replace("+", ",").replace("/", ",").split(","):
                part = part.strip().lower()
                if part in allowed and part not in clean:
                    clean.append(part)
        if clean:
            return clean
        if mode == "utbot_ema":
            return ["ema"]
        if mode == "utbot_consensus":
            return ["consensus"]
        return ["utbot"]

    @staticmethod
    def _quality_hybrid_replay_stream_key(signal) -> str:
        raw = (
            str(getattr(signal, "quality_hybrid_stream", "") or "")
            or str(getattr(signal, "quality_hybrid_route", "") or "")
            or "quality_core"
        ).strip().lower()
        aliases = {
            "consensus_core": "quality_core",
            "chop_strict_consensus": "quality_core",
            "vote_flip": "quality_core",
            "broad_non_atr": "quality_breakout",
            "volume_thrust": "quality_thrust",
        }
        raw = aliases.get(raw, raw)
        if "thrust" in raw:
            return "quality_thrust"
        if "broad" in raw:
            return "quality_broad"
        if "pullback" in raw or "resume" in raw:
            return "quality_pullback"
        if "breakout" in raw or "breakdown" in raw:
            return "quality_breakout"
        return "quality_core"

    @staticmethod
    def _historical_clone_for_replay_stream(signal, stream_key: str, scanner_engine_mode: str = ""):
        mode = str(scanner_engine_mode or "").lower()
        key = str(stream_key or "main")
        if mode not in {"quad_union", "quad_prime", "utbot_ema", "utbot_consensus", "quality_hybrid"}:
            try:
                setattr(signal, "_historical_replay_stream", key)
            except Exception:
                pass
            return signal

        cloned = copy.copy(signal)
        try:
            if mode != "quality_hybrid":
                setattr(cloned, "source_engine", key)
                setattr(cloned, "source_engines", [key])
            setattr(cloned, "_historical_replay_stream", key)
        except Exception:
            pass
        return cloned

    @staticmethod
    def _historical_replay_stream_key(signal, scanner_engine_mode: str = "") -> str:
        existing = str(getattr(signal, "_historical_replay_stream", "") or "").strip()
        if existing:
            return existing
        mode = str(scanner_engine_mode or "").lower()
        if mode == "adaptive_dual":
            return SignalProcessor._signal_source_lane(signal, mode) or "A"
        return SignalProcessor._historical_replay_source_keys(signal, mode)[0]

    @staticmethod
    def _historical_replay_streams(signals: List[UTBotSignal], scanner_engine_mode: str = "") -> List[List[UTBotSignal]]:
        def naive_ts(sig) -> datetime:
            value = getattr(sig, "timestamp", None)
            if value is None:
                return datetime.min
            return value.replace(tzinfo=None) if getattr(value, "tzinfo", None) else value

        mode = str(scanner_engine_mode or "").lower()
        stream_order = {
            "A": 0,
            "B": 1,
            "utbot": 0,
            "ema": 1,
            "supertrend": 2,
            "consensus": 3,
            "quality_core": 0,
            "quality_broad": 1,
            "quality_thrust": 2,
            "quality_breakout": 3,
            "quality_pullback": 4,
            "main": 0,
        }
        grouped: Dict[str, List[UTBotSignal]] = {}
        for sig in list(signals or []):
            for key in SignalProcessor._historical_replay_source_keys(sig, mode):
                cloned = SignalProcessor._historical_clone_for_replay_stream(sig, key, mode)
                if mode == "adaptive_dual":
                    try:
                        setattr(cloned, "source_lane", key)
                        setattr(cloned, "_historical_replay_stream", key)
                    except Exception:
                        pass
                grouped.setdefault(str(key or "main"), []).append(cloned)

        streams = []
        for key in sorted(grouped, key=lambda item: (stream_order.get(item, 99), item)):
            stream = sorted(grouped[key], key=naive_ts)
            if stream:
                streams.append(stream)
        return streams

    @staticmethod
    def _historical_replay_order(
        signals: List[UTBotSignal], scanner_engine_mode: str = "", *, source_attribution: bool = True,
    ) -> List[UTBotSignal]:
        if scanner_engine_mode in {"quad_union", "quad_prime"} and not source_attribution:
            # Same merged candidate as the live scanner, including all source tags.
            # This is still a bar-based historical simulation.
            ordered = []
            for signal in sorted(signals, key=lambda item: item.timestamp):
                cloned = copy.copy(signal)
                cloned._historical_replay_stream = "main"
                ordered.append(cloned)
            return ordered
        return [
            sig
            for stream in SignalProcessor._historical_replay_streams(signals, scanner_engine_mode)
            for sig in stream
        ]

    @staticmethod
    def _historical_same_replay_stream(left, right, scanner_engine_mode: str = "") -> bool:
        if str(scanner_engine_mode or "").lower() not in {
            "s1s2_dual",
            "adaptive_dual",
            "quad_union",
            "quad_prime",
            "utbot_ema",
            "utbot_consensus",
            "quality_hybrid",
        }:
            return True
        return SignalProcessor._historical_replay_stream_key(
            left,
            scanner_engine_mode,
        ) == SignalProcessor._historical_replay_stream_key(right, scanner_engine_mode)

    @staticmethod
    def _historical_trade_id(prefix: str, instrument: str, timeframe: str, signal, scanner_engine_mode: str = "", fmt: str = "%Y%m%d%H%M%S") -> str:
        ts = getattr(signal, "source_signal_timestamp", None) or getattr(signal, "timestamp", None)
        ts_part = ts.strftime(fmt) if hasattr(ts, "strftime") else str(ts or "")
        mode = str(scanner_engine_mode or "").lower()
        stream_key = SignalProcessor._historical_replay_stream_key(signal, mode)
        stream_part = ""
        if mode in {"adaptive_dual", "quad_union", "quad_prime", "utbot_ema", "utbot_consensus", "quality_hybrid", "s1s2_dual"} and stream_key:
            stream_part = "_" + re.sub(r"[^A-Za-z0-9]+", "_", str(stream_key)).strip("_").upper()
        return f"{prefix}_{instrument}_{timeframe}{stream_part}_{ts_part}" + scope_suffix(signal)

    @staticmethod
    def _latest_candle_timestamp(df) -> Optional[datetime]:
        try:
            ts = df.index[-1]
            return ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        except Exception:
            return None

    @staticmethod
    def _historical_compression_state(
        candle_5min: Optional[pd.DataFrame],
        signal_timestamp,
        signal_type: str,
    ) -> str:
        """Replay the one-lot box rule without using future session candles.

        Historical rows are completed bars.  The first three completed 5m bars
        qualify/freeze the box; a later completed close beyond the 3bp buffer is
        equivalent to live WATCH followed by confirmation.  A close back inside
        revokes that authorization for sizing at the current signal.
        """
        try:
            if (
                candle_5min is None
                or candle_5min.empty
                or not isinstance(candle_5min.index, pd.DatetimeIndex)
                or signal_timestamp is None
            ):
                return "FORMING"
            ts = pd.Timestamp(signal_timestamp)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
            frame = candle_5min.sort_index()
            idx = frame.index
            if idx.tz is not None:
                idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
                frame = frame.copy()
                frame.index = idx
            frame = frame[(frame.index.date == ts.date()) & (frame.index + pd.Timedelta(minutes=5) <= ts)]
            if len(frame) < 3:
                return "FORMING"
            first = frame.iloc[:3]
            ref = float(first["open"].iloc[0] or 0.0)
            box_low = float(first["low"].min())
            box_high = float(first["high"].max())
            if ref <= 0 or box_low <= 0 or box_high < box_low:
                return "FORMING"
            if (box_high - box_low) / ref * 100.0 > 0.20 + 1e-9:
                return "UNRESTRICTED"
            breach = max(ref * 0.0003, 0.01)
            latest_close = float(frame["close"].iloc[-1] or 0.0)
            want_up = str(signal_type or "").upper() == "BUY"
            if want_up and latest_close >= box_high + breach:
                return "BREAKOUT_CONFIRMED"
            if not want_up and latest_close <= box_low - breach:
                return "BREAKOUT_CONFIRMED"
            return "LOCKED"
        except Exception:
            return "FORMING"

    def _append_adaptive_dual_apex_candidates(
        self,
        *,
        scanner,
        instrument: str,
        mtf_result,
        intel_result: Dict,
        intel_score: float,
        regime: str,
        settings,
        candidates_raw: List[Dict],
    ) -> None:
        mode = scanner_mode(settings)
        if mode != "adaptive_dual":
            return
        if not _setting_enabled(getattr(settings, "dual_lane_enabled", True), True):
            return

        timeframes = ["5min", "15min"]
        if str(getattr(scanner, "mode", "") or "").upper() == "HISTORICAL":
            return

        for tf in timeframes:
            df = scanner.candles.get_candles(instrument, tf)
            if df is None or len(df) < 200:
                continue
            try:
                from engine.apex_hybrid import build_apex_signals

                signals = build_apex_signals(
                    df,
                    instrument=instrument,
                    timeframe=tf,
                    df_5m=scanner.candles.get_candles(instrument, "5min"),
                    df_15m=scanner.candles.get_candles(instrument, "15min"),
                    latest_only=True,
                )
            except Exception as exc:
                logger.debug(f"Adaptive Dual APEX Lane B skipped for {instrument} {tf}: {exc}")
                continue
            for signal in signals:
                signal_time = signal.timestamp
                if getattr(signal_time, "tzinfo", None) is not None:
                    signal_time = signal_time.astimezone(IST).replace(tzinfo=None)
                signal_type = str(getattr(signal, "signal_type", "") or "").upper()
                identity = (
                    f"{instrument}|{tf}|B|APEX|{signal_type}|"
                    f"{signal_time.isoformat(timespec='seconds')}|"
                    f"{round(float(getattr(signal, 'price', 0.0) or 0.0), 2)}"
                )
                seen = getattr(scanner, "_last_apex_lane_b_identity", None)
                if seen is None:
                    scanner._last_apex_lane_b_identity = {}
                    seen = scanner._last_apex_lane_b_identity
                seen_key = f"{instrument}_{tf}_{signal_type}"
                if seen.get(seen_key) == identity:
                    continue
                seen[seen_key] = identity

                no_entry_label, no_entry_after = _entry_cutoff(
                    settings, tf, s1=s1_lane_policy.lane_tag(signal, settings)
                )
                is_after_no_entry = tf in ["1min", "5min", "15min"] and signal_time.time() >= no_entry_after
                ts_str = signal_time.strftime("%Y-%m-%d %H:%M:00")
                vix_value = scanner._vix_data.get(ts_str)
                if vix_value is None:
                    # 0.0 is this codebase's absent-VIX sentinel (see _get_historical_vix, which
                    # returns 0 rather than 15 for exactly this reason). "or 15.0" defeated the
                    # working abstain branch in SignalManager._grade_signal: 15.0 sits between
                    # the 12.0 low and 20.0 high thresholds, so vix_penalty was always 0.0 and
                    # "VIX unavailable: no volatility adjustment" never printed once across
                    # 41,420 archived graded rows.
                    vix_value = scanner.data.get_ltp("NSE", "INDIAVIX", "99926017") or 0.0
                session_ctx = (
                    intel_result.get("session_context", {})
                    if isinstance(intel_result, dict)
                    else {}
                )
                grade, conf, reasons = scanner.signals._grade_signal(
                    signal,
                    mtf_result.confluence_score,
                    intel_score,
                    regime,
                    vix_value=vix_value,
                    day_type=str(
                        session_ctx.get("effective_day_type")
                        or session_ctx.get("day_type")
                        or ""
                    ),
                    day_type_confidence=float(
                        session_ctx.get("effective_confidence")
                        or session_ctx.get("phase_confidence")
                        or session_ctx.get("confidence")
                        or 0.0
                    ),
                )
                candidates_raw.append({
                    "sig": signal,
                    "tf": tf,
                    "grade": grade,
                    "confidence": conf,
                    "reasons": list(reasons or []) + list(getattr(signal, "apex_reasons", []) or []),
                    "after_no_entry": is_after_no_entry,
                    "after_no_entry_label": no_entry_label,
                    "dedup_key": f"{instrument}_{tf}_apex",
                    "signal_identity": identity,
                })
                scanner.log_event(
                    f"Adaptive Dual Lane B APEX {signal_type}: {instrument} {tf} @ {signal.price:.2f}",
                    "trade",
                )

    def _record_signal_pipeline_evidence(self, candidates, *, instrument: str, phase: str):
        """Evidence phase: stamp decision provenance before candidates leave processor."""
        from engine.decision_record import ensure_decision_record

        now = datetime.now(IST)
        items = list(candidates or [])
        for rank, candidate in enumerate(items, start=1):
            record = ensure_decision_record(
                candidate,
                rank=rank,
                decided_at=now,
                alternatives=[item for item in items if item is not candidate],
            )
            record["processor_phase"] = str(phase)
            record["instrument"] = str(instrument)
            candidate.decision_record = record
        return items

    def _finalize_signal_candidates(
        self,
        *,
        instrument,
        best_tf,
        results_list,
        scored,
        settings,
        scanner_engine_mode,
    ) -> List[TradeCandidate]:
        """Evidence/output phase: select strike winner and emit optional siblings."""
        if not results_list:
            return []
        scanner = self.scanner
        results_list.sort(key=lambda item: item.score, reverse=True)
        winner = self._prefer_liquid_strike(results_list, settings) or results_list[0]
        if winner is not results_list[0]:
            logger.info(
                f"Strike liquidity tie-break for {instrument}: "
                f"{getattr(winner, 'trading_symbol', '')} "
                f"(OI {getattr(winner, 'strike_oi', 0.0):,.0f} / "
                f"vol {getattr(winner, 'strike_volume', 0.0):,.0f}) preferred over "
                f"{getattr(results_list[0], 'trading_symbol', '')} "
                f"(OI {getattr(results_list[0], 'strike_oi', 0.0):,.0f} / "
                f"vol {getattr(results_list[0], 'strike_volume', 0.0):,.0f})"
            )
        getattr(scanner, "_live_signal_recheck_eligible", {}).pop(
            f"{instrument}_{best_tf}", None
        )
        scanner.log_event(
            f"🎯 Potential {winner.direction} Signal for {winner.instrument} ({winner.timeframe})",
            "trade",
        )
        output = [winner]
        try:
            siblings = self._emit_cascade_tf_siblings(
                winner, scored, settings, scanner_engine_mode
            )
            if siblings:
                output.extend(siblings)
                stack_desc = [
                    f"{item.timeframe}:{getattr(item, 'source_lane', '') or 'A'}"
                    for item in output
                ]
                logger.info(f"Cascade multi-TF stack for {instrument}: {stack_desc}")
                scanner.log_event(
                    f"Cascade multi-TF stack {instrument}: "
                    f"{len(output)} entries ({', '.join(item.timeframe for item in output)})",
                    "trade",
                )
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug(f"Cascade multi-TF siblings abstained for {instrument}: {exc}")
        if len(output) == 1:
            try:
                rare = self._emit_rare_dual_tf_siblings(
                    winner, scored, settings, scanner_engine_mode
                )
                if rare:
                    output.extend(rare)
                    scanner.log_event(
                        f"Rare dual-TF {instrument}: "
                        f"{', '.join(item.timeframe for item in output)}",
                        "trade",
                    )
            except (AttributeError, TypeError, ValueError) as exc:
                logger.debug(f"Rare dual-TF abstained for {instrument}: {exc}")
        return self._record_signal_pipeline_evidence(
            output, instrument=instrument, phase="candidate_output"
        )

    def _prepare_signal_pipeline_inputs(self, instrument, settings):
        """Data-preparation phase: resolve eligible frames and immutable mode inputs."""
        scanner = self.scanner
        if not scanner.risk_manager.can_trade("ANY"):
            return None
        timeframe_contexts = self._build_timeframe_contexts(scanner, instrument, settings)
        scanner_engine_mode = scanner_mode(settings)
        policy = normalize_timeframe_entry_policy(
            getattr(settings, "ut_timeframe_entry_policy", "INCLUDE_5MIN")
        )
        policy_timeframes = timeframe_entry_timeframes(policy)
        candidate_timeframes = [
            tf for tf in ("5min", "15min") if tf in policy_timeframes
        ]
        if scanner_engine_mode == "one_plus_one":
            # Two scanner cores execute once. Other intraday frames remain feature
            # inputs and must never become duplicate portfolio state machines.
            # This was previously an undeclared getattr and therefore could only
            # ever use its fallback. Keep that real behaviour explicit.
            execution_tf = "5min"
            candidate_timeframes = [
                execution_tf if execution_tf in {"1min", "5min", "15min"} else "5min"
            ]
        live = str(getattr(scanner, "mode", "") or "").upper() != "HISTORICAL"
        volume_spike = _setting_enabled(
            getattr(settings, "volume_spike_entry_enabled", True), True
        )
        one_min_scalp = _setting_enabled(
            getattr(settings, "one_min_scalping_enabled", True), True
        )
        reversion_enabled = _setting_enabled(
            getattr(settings, "post_exit_reversion_enabled", False), False
        )
        reversion_window = False
        if live and reversion_enabled:
            try:
                probe = getattr(scanner, "_reversion_window_active", None)
                reversion_window = bool(probe(instrument)) if callable(probe) else False
            except (AttributeError, TypeError, ValueError) as exc:
                logger.debug(f"Reversion-window input unavailable for {instrument}: {exc}")
        policy_one_min = "1min" in policy_timeframes and one_min_scalp
        legacy_one_min = live and (
            (volume_spike and one_min_scalp) or reversion_window
        )
        if policy_one_min or legacy_one_min or (
            live and "1min" in policy_timeframes and reversion_window
        ):
            candidate_timeframes.insert(0, "1min")
        return {
            "timeframe_policy": policy,
            "timeframe_contexts": timeframe_contexts,
            "scanner_engine_mode": scanner_engine_mode,
            "policy_timeframes": policy_timeframes,
            "candidate_timeframes": candidate_timeframes,
            "live": live,
            "volume_spike": volume_spike,
            "one_min_scalp": one_min_scalp,
            "reversion_enabled": reversion_enabled,
            "reversion_window": reversion_window,
        }

    def _collect_signal_candidates(
        self,
        instrument,
        mtf_result,
        intel_result,
        intel_score,
        regime,
        lots,
        lot_size,
        cfg,
        spot,
        atm_strike,
        settings,
        prepared,
        one_plus_one_source="",
    ):
        """Replay/collection phase: produce raw candidates without final admission."""
        scanner = self.scanner
        scanner_engine_mode = prepared["scanner_engine_mode"]
        candidate_timeframes = prepared["candidate_timeframes"]
        _timeframe_contexts = prepared["timeframe_contexts"]
        _live = prepared["live"]
        is_live = _live
        tf_policy = prepared["timeframe_policy"]
        _one_plus_one_source = one_plus_one_source
        inst_pref = str(getattr(scanner, "inst_pref", "AUTO") or "AUTO").upper()
        candidates_raw = []

        def naive_ts(value):
            if value is None:
                return datetime.min
            return value.replace(tzinfo=None) if getattr(value, "tzinfo", None) else value

        for tf in candidate_timeframes:
            sigs = [] # Master Fix: Clear sigs to prevent stale leak from previous iterations
            result = getattr(mtf_result, f"results_{tf}", None)
            if not result or result.get("error"):
                if tf == "15min": logger.debug(f"⚠️ No 15min results for {instrument} yet.")
                continue
            sigs = [s for s in result.get("signals", []) if s.instrument == instrument]
            if sigs and str(getattr(scanner, "mode", "") or "").upper() == "HISTORICAL":
                sigs = [signal for signal in sigs if not getattr(signal, "uts_forming", False)]
            logger.debug(f"🔍 [AUDIT-SIGS] {instrument} {tf}: found {len(sigs)} signals")
            if not sigs:
                continue
            source_key = str(_one_plus_one_source or "").lower()
            dedup_key = (
                f"{instrument}_{tf}_{source_key}"
                if scanner_engine_mode == "one_plus_one" and source_key
                else f"{instrument}_{tf}"
            )
            dedup_key += scope_suffix(sigs[-1], settings)
            is_live = scanner.mode.upper() != "HISTORICAL"
            if tf == "1min" and is_live and dedup_key not in scanner._last_signal_time:
                scanner._last_signal_time[dedup_key] = (
                    naive_ts(sigs[-2].timestamp)
                    if len(sigs) >= 2
                    else naive_ts(sigs[-1].timestamp) - timedelta(microseconds=1)
                )
            if dedup_key not in scanner._last_signal_time:
                # ═══ Mode-Aware Historical Backfill (Atomic Lock) ═══
                # REAL startup must not replay and price every earlier session signal:
                # those rows are already persisted for the dashboard, historical trade
                # creation is forbidden in REAL anyway, and the wasted option-history /
                # liquidity work held all four live scan threads for 60s+. Seed just
                # before the newest unpaired arrow so the current signal retains exactly
                # the established live eligibility semantics.
                if is_live:
                    scanner._last_signal_time[dedup_key] = (
                        naive_ts(sigs[-2].timestamp)
                        if len(sigs) >= 2
                        else naive_ts(sigs[-1].timestamp) - timedelta(microseconds=1)
                    )
                else:
                    # Immediately seed to prevent concurrent historical backfill.
                    scanner._last_signal_time[dedup_key] = naive_ts(datetime.now(IST))
                # Full reconstruction belongs exclusively to HISTORICAL mode.
                if scanner.mode == "HISTORICAL":
                    # Fetch 5min candles to determine trading dates for lookback
                    candles_df = scanner.candles.get_candles(instrument, "5min")
                    if candles_df is None or candles_df.empty:
                        return []
                    tf_candles_df = scanner.candles.get_candles(instrument, tf)
                    if tf_candles_df is None or getattr(tf_candles_df, "empty", True):
                        tf_candles_df = candles_df
                    
                    # Logic Fix: Data lookback vs Trade lookback
                    data_lookback = 1 if is_live else 45 # Live mode only replays the current session.
                    
                    # Trade lookback: The actual window the user wants to see/test
                    trade_lookback = 1 if is_live else scanner.backtest_days
                    
                    # ═══ Trading Day Awareness ═══
                    active_dates = sorted(list(set(candles_df.index.date)))
                    logger.debug(f"📅 Active dates in candles for {instrument}: {active_dates}")
                    if not active_dates: return []
                    
                    if is_live:
                        # Live/Paper mode replays the active trading session. After midnight but before
                        # 09:15 IST, that session is still the previous market day.
                        session_day_raw = (
                            scanner._get_current_session_day()
                            if hasattr(scanner, "_get_current_session_day")
                            else datetime.now(IST).date().isoformat()
                        )
                        session_day = datetime.fromisoformat(str(session_day_raw)).date()
                        trade_cutoff = session_day
                        data_cutoff = session_day
                    else:
                        # Trade Cutoff (for display/logging in closed_trades)
                        if trade_lookback <= 1 and hasattr(scanner, "_get_current_session_day"):
                            session_day = datetime.fromisoformat(str(scanner._get_current_session_day())).date()
                            session_dates = [d for d in active_dates if d <= session_day]
                            trade_cutoff = session_dates[-1] if session_dates else active_dates[-1]
                        else:
                            trade_cutoff = active_dates[-min(len(active_dates), trade_lookback)]
                        # Data Cutoff (for signal calculation)
                        data_cutoff = active_dates[-min(len(active_dates), data_lookback)]
                    
                    # Filter signals for this TF by data_cutoff and sort chronologically (oldest first)
                    def get_naive_ts(s):
                        return s.timestamp.replace(tzinfo=None) if s.timestamp.tzinfo else s.timestamp
                        
                    if (
                        not is_live
                        and scanner_engine_mode == "adaptive_dual"
                        and _setting_enabled(getattr(settings, "dual_lane_enabled", True), True)
                    ):
                        try:
                            from engine.engine_lab import EngineLab

                            sigs.extend(
                                EngineLab._apex_replay_signals(
                                    tf_candles_df,
                                    instrument,
                                    tf,
                                    df_5m=scanner.candles.get_candles(instrument, "5min"),
                                    df_15m=scanner.candles.get_candles(instrument, "15min"),
                                )
                            )
                        except Exception as exc:
                            logger.debug(f"Historical APEX Lane B replay signal build skipped for {instrument} {tf}: {exc}")

                    sigs = self._historical_replay_order(
                        [s for s in sigs if s.instrument == instrument and s.timestamp.date() >= data_cutoff],
                        scanner_engine_mode,
                        source_attribution=_setting_enabled(
                            getattr(settings, "historical_quad_source_attribution", False), False),
                    )
                    
                    # Process signals for this TF
                    for i in range(len(sigs)):
                        s1 = sigs[i]
                        s2 = sigs[i + 1] if i + 1 < len(sigs) else None
                        hist_s1 = s1_lane_policy.is_lane(s1, settings)
                        hist_lane_tag = s1_lane_policy.lane_tag(s1, settings)
                        hist_s12 = hist_lane_tag in {"D1", "D2"}
                        if hist_s12:
                            # D1/D2 trades carry their own engine EXIT and may be the
                            # final event in the replay window. They are not paired to
                            # an opposite arrow from another lane or another trade.
                            if s2 is None or not self._historical_same_replay_stream(s1, s2, scanner_engine_mode):
                                s2 = s1
                        elif s2 is None or not self._historical_same_replay_stream(s1, s2, scanner_engine_mode):
                            continue
                        hist_uts = hist_lane_tag == "UTS"
                        # Earlier rows are indicator warm-up only. They must not inflate
                        # diagnostics for the selected backtest window.
                        if s1.timestamp.date() < trade_cutoff:
                            continue
                        
                        hist_vix = scanner._get_historical_vix(s1.timestamp)
                        hist_intel_score, hist_intel_regime = scanner._compute_historical_intel_at(
                            instrument,
                            s1.timestamp,
                            tf,
                        )
                        hist_regime = hist_intel_regime if hist_intel_regime and hist_intel_regime != "UNKNOWN" else "UNKNOWN"
                        if hist_regime == "UNKNOWN":
                            try:
                                sub_df = _closed_candles_at(tf_candles_df, s1.timestamp, tf)
                            except Exception:
                                sub_df = _closed_candles_at(tf_candles_df, s1.timestamp, tf)
                            hist_regime_res = scanner.intel.regime.detect(sub_df, instrument, tf)
                            hist_regime = hist_regime_res.get("regime", "UNKNOWN")
                        hist_confluence = scanner._compute_historical_confluence_at(
                            mtf_result,
                            s1.timestamp,
                        )

                        _hist_day_type, _hist_day_conf = "", 0.0
                        try:
                            _day_at = getattr(scanner, "_historical_day_type_at", None)
                            if callable(_day_at):
                                _hist_day_type, _hist_day_conf = _day_at(instrument, s1.timestamp, tf)
                        except Exception:
                            pass
                        hist_grade, hist_conf, hist_reasons = scanner.signals._grade_signal(
                            s1,
                            hist_confluence,
                            hist_intel_score,
                            regime=hist_regime,
                            vix_value=hist_vix,
                            day_type=_hist_day_type,
                            day_type_confidence=_hist_day_conf,
                        )
                        direction = "LONG" if s1.signal_type == "BUY" else "SHORT"

                        def record_reject(reason: str, inst_type: str = "FUT", strike: float = 0.0, option_type: str = "") -> None:
                            recorder = getattr(scanner, "_record_historical_reject", None)
                            if recorder:
                                recorder(
                                    s1,
                                    instrument,
                                    tf,
                                    direction,
                                    hist_grade,
                                    hist_conf,
                                    reason,
                                    inst_type,
                                    lot_size,
                                    strike,
                                    option_type,
                                    regime=hist_regime,
                                )
                            emit_once = getattr(scanner, "log_signal_decision_once", None)
                            if callable(emit_once):
                                emit_once(
                                    f"hist|{instrument}|{tf}|{s1.signal_type}|{s1.timestamp.isoformat(timespec='seconds')}|{inst_type}|{reason}",
                                    (
                                        f"Signal rejected: {instrument} {s1.signal_type} {tf} "
                                        f"@ {float(getattr(s1, 'price', 0.0) or 0.0):.2f} - {reason}"
                                    ),
                                    "trade",
                                )
                        
                        logger.debug(f"🔍 [AUDIT] Signal {i} for {instrument} at {s1.timestamp}: Grade={hist_grade}, Conf={hist_conf:.2f}, Regime={hist_regime}, Type={s1.signal_type}")

                        direction = "LONG" if s1.signal_type == "BUY" else "SHORT"
                        trend_15m_agrees_hist = self._trend_15m_agrees_at(
                            mtf_result,
                            s1.timestamp,
                            s1.signal_type,
                            closed_only=True,
                        )
                        hist_structural_impulse = bool(
                            tf == "5min"
                            and self._is_5min_impulse_reversal(
                                s1,
                                mtf_result,
                                hist_conf,
                                settings,
                            )
                        )
                        hist_impulse_evidence, _ = self._explicit_signal_quality_evidence(
                            s1,
                            structural_impulse=hist_structural_impulse,
                        )
                        hist_reason_text = " ".join(str(r).lower() for r in (hist_reasons or []))
                        hist_extreme_exhaustion = "extreme" in hist_reason_text and "exhaustion" in hist_reason_text
                        hist_true_5min_impulse = tf == "5min" and five_min_true_impulse_allowed(
                            hist_grade,
                            hist_conf,
                            settings,
                            trend_15m_agrees=trend_15m_agrees_hist,
                            impulse_confirmed=hist_impulse_evidence,
                            explosive_bypass=bool(getattr(s1, "is_explosive_bypass", False)),
                            extreme_exhaustion=hist_extreme_exhaustion,
                        )

                        # â”€â”€ Determine Instrument Type via Hybrid Rules â”€â”€
                        sim_inst = scanner._resolve_instrument_type(
                            instrument=instrument,
                            grade=hist_grade,
                            regime=hist_regime,
                            intel_score=hist_intel_score,
                            signal_score=hist_conf,
                            confidence=hist_conf,
                            adx_value=getattr(s1, "adx_value", 0.0),
                            atr_value=getattr(s1, "atr_value", 0.0),
                            price=s1.price,
                            signal_time=s1.timestamp,
                            timeframe=tf,
                            five_min_true_impulse=hist_true_5min_impulse,
                        )

                        # Replay 1m with same burst-only admission as live. Resolve
                        # product first so a rejected signal is recorded against its
                        # actual OPT/FUT lane instead of reading an unbound sim_inst.
                        # Historical L1 depth is unavailable, so never leak current
                        # depth into a past decision.
                        if tf == "1min" and not hist_s1:
                            hist_burst_detector = getattr(scanner, "_detect_volume_spike_entry", None)
                            hist_volume_spike_metrics = (
                                hist_burst_detector(
                                    instrument,
                                    tf,
                                    s1.signal_type,
                                    adx_value=float(getattr(s1, "adx_value", 0.0) or 0.0),
                                    intel_score=float(hist_intel_score or 0.0),
                                    as_of=s1.timestamp,
                                    depth_context={},
                                )
                                if callable(hist_burst_detector)
                                else None
                            )
                            setattr(s1, "volume_spike_entry", bool(hist_volume_spike_metrics))
                            setattr(s1, "volume_spike_metrics", dict(hist_volume_spike_metrics or {}))
                            if not hist_volume_spike_metrics and tf_policy != "PRIMARY_1":
                                record_reject("1min requires confirmed directional burst", sim_inst)
                                continue
                        diag_inc_unique = getattr(scanner, "_diag_inc_unique", None)
                        if diag_inc_unique:
                            diag_inc_unique(
                                "instrument_selection",
                                sim_inst,
                                "|".join([instrument, tf, str(s1.timestamp), str(s1.signal_type)]),
                            )
                        
                        base_grade = hist_grade

                        # Data-confirmed power move (historical parity with live): a big
                        # volume/range/close spike with momentum+alignment bypasses the grade
                        # floor and the no-entry cutoff here too, so backtests reflect live.
                        power_move_hist = scanner._detect_power_move(
                            instrument,
                            tf,
                            s1.signal_type,
                            adx_value=float(getattr(s1, "adx_value", 0.0) or 0.0),
                            intel_score=hist_intel_score,
                            trend_15m_agrees=trend_15m_agrees_hist,
                            as_of=s1.timestamp,
                        )

                        # ── R2: regime veto + directional census (replay parity) ──
                        # Same gates as the live path so 22-Jun-style replays reflect them.
                        _hist_dir = "LONG" if s1.signal_type == "BUY" else "SHORT"
                        self._record_directional_census(
                            scanner, s1, instrument, tf, _hist_dir,
                            str(s1.timestamp), settings,
                        )
                        _hv = directional_fut_veto(
                            settings,
                            inst_type=sim_inst,
                            signal_type=s1.signal_type,
                            regime=hist_regime,
                            adx_value=float(getattr(s1, "adx_value", 0.0) or 0.0),
                            power_move=power_move_hist,
                        )
                        if _hv:
                            # If we got a regime veto on FUT and inst_pref is AUTO, fall back to OPT
                            if sim_inst == "FUT" and inst_pref == "AUTO":
                                self._historical_audit_log(
                                    f"[{instrument}] FUT regime vetoed ({_hv}), falling back to OPT at {s1.timestamp}"
                                )
                                sim_inst = "OPT"
                            else:
                                record_reject(_hv, sim_inst)
                                continue
                        _hist_reversal_relief = None
                        try:
                            _relief_fn = getattr(scanner, "_entry_reversal_relief_context", None)
                            if callable(_relief_fn):
                                _hist_reversal_relief = _relief_fn(
                                    instrument,
                                    tf,
                                    s1.signal_type,
                                    s1.timestamp,
                                    regime=hist_regime,
                                    confidence=hist_conf,
                                )
                        except Exception as exc:
                            logger.debug(f"Historical reversal relief check skipped for {instrument} {tf}: {exc}")
                        _hc = directional_census_block(
                            settings,
                            getattr(scanner, "_dir_census", {}) or {},
                            signal_direction=_hist_dir,
                            confidence=hist_conf,
                            grade_floor=self._grade_min_confidence("B+"),
                            power_move=power_move_hist,
                            reversal_context=_hist_reversal_relief,
                            day_type=_hist_day_type,
                            day_type_confidence=_hist_day_conf,
                        )
                        if _hc:
                            record_reject(_hc, sim_inst)
                            continue

                        # One lane-independent quality verdict. Power-move is evidence,
                        # not a bypass, and exact source multiplicity must match live.
                        hist_source_engines = list(getattr(s1, "source_engines", []) or [])
                        hist_source_count = max(1, len(hist_source_engines))
                        # ── No-Entry After Gate ──
                        try:
                            no_entry_after, gate_time = _entry_cutoff(settings, tf)
                            s1_time = s1.timestamp.replace(tzinfo=None).time() if s1.timestamp.tzinfo else s1.timestamp.time()
                            power_move_cutoff_ok = bool(power_move_hist and scanner._power_move_late_cutoff_ok(s1.timestamp))
                            if (
                                not self._historical_entry_cutoff_allows(
                                    s1, tf, hist_lane_tag, settings
                                )
                                and not power_move_cutoff_ok
                            ):
                                record_reject(f"After no-entry gate {no_entry_after}", sim_inst)
                                self._historical_audit_log(f"🚫 [AUDIT] Skipping backfill entry at {s1.timestamp} due to ut_no_entry_after ({no_entry_after})")
                                continue
                        except Exception as e:
                            logger.exception(f"Error checking entry cutoff: {e}")
                        # Historical workers run concurrently. Filtering against the
                        # partially built shared trade list makes replay results depend
                        # on task completion order. TradeManager applies the configured
                        # concurrency policy deterministically after generation.
                        replay_pair_ready = (
                            hist_s12
                            or (s2.timestamp.date() >= trade_cutoff and s1.signal_type != s2.signal_type)
                        )
                        if s1.timestamp.date() >= trade_cutoff and replay_pair_ready:
                            force_exit_time = datetime.strptime(
                                str(getattr(settings, "ut_force_exit_time", "15:25") or "15:25"),
                                "%H:%M",
                            ).time()
                            if hist_s12 and _setting_enabled(
                                getattr(settings, "replay_s12_native_exits", True), True
                            ):
                                force_exit_time = datetime.strptime(
                                    s12_lane_policy.square_off(settings), "%H:%M"
                                ).time()
                            entry_ts = s1.timestamp.replace(tzinfo=None) if s1.timestamp.tzinfo else s1.timestamp
                            next_signal_ts = s2.timestamp.replace(tzinfo=None) if s2.timestamp.tzinfo else s2.timestamp
                            session_exit_ts = datetime.combine(entry_ts.date(), force_exit_time)
                            crosses_session = (
                                next_signal_ts.date() != entry_ts.date()
                                or next_signal_ts > session_exit_ts
                            )
                            effective_exit_ts = session_exit_ts if crosses_session or hist_s12 else next_signal_ts
                            hist_chart_reason = ""
                            if hist_s12 and _setting_enabled(
                                getattr(settings, "replay_s12_native_exits", True), True
                            ):
                                s12_minutes = {"1min": 1, "5min": 5, "15min": 15}.get(str(tf), 5)
                                # D1 SETUP is filled at the next candle open; D2's
                                # close-based ENTRY is executable at that same close
                                # boundary. Both begin at signal-open + one bar.
                                entry_ts = entry_ts + timedelta(minutes=s12_minutes)
                                native_exit_time = getattr(s1, "s12_exit_time", None)
                                native_exit_reason = str(getattr(s1, "s12_exit_reason", "") or "")
                                if native_exit_time is not None and native_exit_reason:
                                    native_exit_close = _market_naive(native_exit_time).to_pydatetime() + timedelta(minutes=s12_minutes)
                                    if entry_ts < native_exit_close < effective_exit_ts:
                                        effective_exit_ts = native_exit_close
                                    hist_chart_reason = native_exit_reason
                            elif hist_uts:
                                entry_ts, effective_exit_ts, hist_chart_reason = self._uts_replay_window(
                                    s1, tf, session_exit_ts, next_signal=s2,
                                )
                            elif not hist_s12:
                                # Signals are stamped at their candle OPEN but only exist once that
                                # candle closes. Entering (and taking the next opposite signal) at the
                                # stamp booked a move the system could not have traded: the 2026-10-08
                                # payoff audit found every QP replay profit inside that one-bar window.
                                entry_ts, effective_exit_ts = self._closed_candle_replay_window(
                                    entry_ts, tf, effective_exit_ts, session_exit_ts, crosses_session,
                                )

                            if sim_inst == "FUT":
                                sim_option_type = ""
                                sim_multiplier = 1.0
                                hist_strike = 0.0
                            else:
                                sim_option_type = "CE" if direction == "LONG" else "PE"
                                hist_strike = self._historical_option_strike(instrument, direction, s1.price, cfg)
                                # Signed delta (PE negative) — parity with live path (audit #6).
                                try:
                                    dte_h = scanner.expiry.get_dte(instrument, s1.timestamp.date())
                                except Exception:
                                    dte_h = 1
                                sim_multiplier = self._option_signed_delta(
                                    s1.price,
                                    hist_strike,
                                    sim_option_type,
                                    {},
                                    dte_h,
                                    0.5 if sim_option_type == "CE" else -0.5,
                                )

                            hist_s12_levels = {}
                            hist_s12_entry_spot = float(
                                getattr(s1, "s12_engine_entry", 0.0) or getattr(s1, "price", 0.0) or 0.0
                            ) if hist_s12 else 0.0
                            if hist_s12:
                                try:
                                    entry_spot_frame = scanner.candles.get_candles(instrument, "1min")
                                    if entry_spot_frame is not None and not entry_spot_frame.empty:
                                        normalized_index = pd.DatetimeIndex([
                                            _market_naive(ts) for ts in entry_spot_frame.index
                                        ])
                                        positions = normalized_index.get_indexer([_market_naive(entry_ts)])
                                        if len(positions) and positions[0] >= 0:
                                            hist_s12_entry_spot = float(
                                                entry_spot_frame.iloc[int(positions[0])].get("open", hist_s12_entry_spot)
                                                or hist_s12_entry_spot
                                            )
                                except Exception:
                                    pass
                                hist_s12_levels = s12_lane_policy.admission_levels(
                                    s1, hist_s12_entry_spot, settings
                                )
                                index_stop_distance = abs(
                                    hist_s12_entry_spot - float(hist_s12_levels.get("s12_stop") or 0.0)
                                )
                            else:
                                index_stop_distance = s1.stop_distance
                            hist_s12_market = {"lane": hist_lane_tag, "trade_id": str(
                                getattr(s1, "s12_trade_id", "") or ""
                            ), "stop_updates": [], "milestones": {}}
                            if hist_s12:
                                for update in getattr(s1, "s12_stop_updates", []) or []:
                                    try:
                                        when, level = update
                                        hist_s12_market["stop_updates"].append({
                                            "effective_from": _market_naive(when), "stop": float(level),
                                        })
                                    except (TypeError, ValueError):
                                        continue
                                for name, when in (getattr(s1, "s12_milestones", {}) or {}).items():
                                    try:
                                        hist_s12_market["milestones"][str(name)] = {"time": _market_naive(when)}
                                    except Exception:
                                        continue
                            # Provisional entry/stop for risk sizing (FUT uses index; OPT uses abs delta).
                            if sim_inst == "FUT":
                                hist_entry_px = float(hist_s12_entry_spot or s1.price or 0.0)
                                hist_stop_px = float(
                                    hist_s12_levels.get("s12_stop") or s1.trailing_stop or 0.0
                                )
                                if direction == "LONG" and not hist_s12:
                                    hist_stop_px = max(
                                        hist_stop_px,
                                        hist_entry_px * (1.0 - float(scanner.futures_sl_pct or 0.0) / 100.0),
                                    )
                                elif direction != "LONG" and not hist_s12:
                                    hist_stop_px = min(
                                        hist_stop_px if hist_stop_px > 0 else hist_entry_px * 1.01,
                                        hist_entry_px * (1.0 + float(scanner.futures_sl_pct or 0.0) / 100.0),
                                    )
                                requested = int(getattr(scanner, "user_lots_fut", {}).get(instrument, 1) or 1)
                            else:
                                hist_entry_px = float(_estimate_opt_premium(s1.price, scanner, instrument, _sig_date(s1)) or 0.0)
                                natural = abs(float(index_stop_distance or 0.0) * abs(float(sim_multiplier or 0.5)))
                                max_sl = hist_entry_px * (float(scanner.options_sl_pct or 15.0) / 100.0)
                                hist_stop_px = max(0.05, hist_entry_px - max(0.05, min(natural, max_sl)))
                                requested = int(getattr(scanner, "user_lots", {}).get(instrument, 1) or 1)

                            hist_compression_state = self._historical_compression_state(
                                candles_df,
                                s1.timestamp,
                                s1.signal_type,
                            )
                            if hist_uts and sim_inst == "OPT":
                                # UTS admission requires the observed contract fill and
                                # premium cap. A provisional delta estimate cannot size it.
                                hist_lots = max(1, requested)
                            else:
                                size_result = resolve_trade_lots(
                                    settings=settings,
                                    inst_type=sim_inst,
                                    requested_lots=requested,
                                    lot_size=lot_size,
                                    entry_price=hist_entry_px,
                                    stop_price=hist_stop_px,
                                    compression_state=hist_compression_state,
                                    intel_score=float(hist_intel_score or 0.0),
                                )
                                hist_lots = int(size_result.lots or 0)
                                if hist_lots < 1:
                                    self._historical_audit_log(
                                        f"Skipping {instrument} {tf} {sim_inst}: {size_result.reason}"
                                    )
                                    continue

                            qty = hist_lots * lot_size
                            
                            actual_option = None
                            trading_symbol = ""
                            symbol_token = ""
                            if sim_inst == "OPT":
                                option_df, trading_symbol, symbol_token = self._load_option_candles(
                                    scanner, instrument, hist_strike, sim_option_type, "1min", s1.timestamp.date(), now_dt=s1.timestamp
                                )
                                if hist_uts and option_df.empty:
                                    record_reject("UTS replay requires observed option prices", "OPT")
                                    continue
                                if option_df.empty and getattr(settings, "ut_option_history_mode", "fetch_or_skip") == "fetch_or_skip":
                                    option_pref = (getattr(scanner, "inst_pref", "AUTO") or "AUTO").upper()
                                    if option_pref == "OPT":
                                        self._historical_audit_log(f"Skipping {instrument} {sim_option_type} {hist_strike} at {s1.timestamp}: no real option candles.")
                                        continue
                                    self._historical_audit_log(
                                        f"Falling back to FUT for {instrument} {tf} at {s1.timestamp}: "
                                        "no real option candles in AUTO/HYBRID."
                                    )
                                    diag = getattr(scanner, "_diag_inc", None)
                                    if diag:
                                        diag("option_history", "fallback_to_fut")
                                    sim_inst = "FUT"
                                    sim_option_type = ""
                                    sim_multiplier = 1.0
                                    hist_strike = 0.0
                                    entry_premium = s1.price
                                    fut_stop = float(
                                        hist_s12_levels.get("s12_stop") or s1.trailing_stop or 0.0
                                    )
                                    size_result = resolve_trade_lots(
                                        settings=settings,
                                        inst_type="FUT",
                                        requested_lots=int(
                                            getattr(scanner, "user_lots_fut", {}).get(instrument, 1) or 1
                                        ),
                                        lot_size=lot_size,
                                        entry_price=float(entry_premium or 0.0),
                                        stop_price=fut_stop,
                                        compression_state=hist_compression_state,
                                        intel_score=0.0,  # FUT: never enlarged by conviction
                                    )
                                    hist_lots = int(size_result.lots or 0)
                                    if hist_lots < 1:
                                        continue
                                    qty = hist_lots * lot_size
                                else:
                                    entry_premium = _estimate_opt_premium(s1.price, scanner, instrument, _sig_date(s1))
                            else:
                                entry_premium = s1.price
                            
                            if sim_inst == "FUT":
                                sl_price = float(
                                    hist_s12_levels.get("s12_stop") or s1.trailing_stop or 0.0
                                )
                                if not hist_s12:
                                    hist_fut_sl_pct = futures_stop_tolerance_pct(
                                        scanner.futures_sl_pct,
                                        hist_regime,
                                    )
                                    max_sl_dist = entry_premium * (hist_fut_sl_pct / 100.0)
                                    if direction == "LONG":
                                        sl_price = max(sl_price, entry_premium - max_sl_dist)
                                    else:
                                        sl_price = min(sl_price, entry_premium + max_sl_dist)
                                index_stop_distance = max(1.0, abs(entry_premium - sl_price))
                            else:
                                # Options are always long-premium instruments (CE/PE buy). Keep SL below entry.
                                # Use absolute spot stop distance mapped by multiplier, capped by option SL%.
                                spot_stop_dist = abs(float(s1.price) - float(s1.trailing_stop))
                                hist_option_cap = (
                                    s12_lane_policy.premium_cap_pct(s1, settings, timeframe=tf)
                                    if hist_s12 else option_stop_tolerance_pct(
                                        scanner.options_sl_pct, hist_regime,
                                        confidence=hist_conf, grade=hist_grade,
                                        setup=s1,
                                    )
                                )
                                if hist_uts:
                                    effective_sl_dist = self._historical_uts_premium_stop_distance(
                                        entry_premium,
                                        spot_stop_dist,
                                        sim_multiplier,
                                        settings,
                                        regime=hist_regime,
                                        confidence=hist_conf,
                                        grade=hist_grade,
                                        setup=s1, timeframe=tf,
                                    )
                                elif hist_s12:
                                    effective_sl_dist = entry_premium * hist_option_cap / 100.0
                                else:
                                    natural_sl_dist = abs(spot_stop_dist * sim_multiplier)
                                    max_sl_dist = entry_premium * (hist_option_cap / 100.0)
                                    effective_sl_dist = max(0.05, min(natural_sl_dist, max_sl_dist))
                                sl_price = max(0.05, entry_premium - effective_sl_dist)
                            
                            raw_exit_spot = (
                                float(getattr(s1, "s12_exit_price", 0.0) or s1.price or 0.0)
                                if hist_s12 else s2.price
                            )
                            if crosses_session:
                                raw_exit_spot = s1.price
                                day_candles = candles_df[
                                    (candles_df.index.date == entry_ts.date())
                                    & (candles_df.index.time <= force_exit_time)
                                ]
                                if not day_candles.empty:
                                    raw_exit_spot = float(day_candles.iloc[-1]["close"])
                            
                            if sim_inst == "OPT":
                                natural_sl_distance = abs(index_stop_distance * sim_multiplier)
                                hist_rr = self._dynamic_rr(hist_grade, hist_conf, timeframe=tf)
                                actual_option = self._simulate_actual_option_exit(
                                    option_df if "option_df" in locals() else pd.DataFrame(),
                                    entry_ts,
                                    effective_exit_ts,
                                    sl_price,
                                    uts_lane_policy.target_rr(settings) if hist_uts else hist_rr,
                                    natural_sl_distance=effective_sl_dist if hist_uts else natural_sl_distance,
                                    max_sl_pct=hist_option_cap / 100.0,
                                    qty=qty,
                                    tf15_qty=qty,
                                    tf15_exit_scanner=scanner,
                                    tf15_instrument=instrument,
                                    tf15_direction=direction,
                                    tf15_option_type=sim_option_type,
                                    tf15_timeframe=tf,
                                    replay_grade=hist_grade,
                                    replay_confidence=hist_conf,
                                    replay_regime=hist_regime,
                                    replay_source_lane=hist_lane_tag,
                                    replay_setup=s1,
                                    required_stop_distance=natural_sl_distance if hist_uts else 0.0,
                                    chart_exit_reason=hist_chart_reason,
                                )
                                if actual_option:
                                    entry_premium = actual_option["entry"]
                                    sl_price = actual_option["stop"]
                                    raw_exit_premium = actual_option["exit"]
                                    exit_premium = raw_exit_premium
                                    exit_time = actual_option["exit_time"]
                                    is_sl_hit = actual_option["reason"] == "SL HIT"
                                    if hist_uts:
                                        size_result = resolve_trade_lots(
                                            settings=settings, inst_type="OPT",
                                            requested_lots=requested, lot_size=lot_size,
                                            entry_price=entry_premium, stop_price=sl_price,
                                            compression_state=hist_compression_state,
                                            intel_score=float(hist_intel_score or 0.0),
                                        )
                                        hist_lots = int(size_result.lots or 0)
                                        if hist_lots < 1:
                                            record_reject(size_result.reason, "OPT")
                                            continue
                                        observed_qty = hist_lots * lot_size
                                        if observed_qty != qty:
                                            for sample in actual_option.get("price_path") or []:
                                                for value in ("pnl", "peak_pnl"):
                                                    if value in sample:
                                                        sample[value] *= observed_qty / qty
                                        qty = observed_qty
                                else:
                                    if hist_uts:
                                        record_reject("UTS replay lacks a complete observed option path or compatible stop", "OPT")
                                        continue
                                    spot_diff = raw_exit_spot - s1.price if direction == "LONG" else s1.price - raw_exit_spot
                                    raw_exit_premium = entry_premium + (spot_diff * sim_multiplier)
                                    
                                    time_held_hours = (effective_exit_ts - entry_ts).total_seconds() / 3600.0
                                    if time_held_hours > 1.0 and abs(spot_diff) < (s1.atr_value * 0.5):
                                        time_held_hours = 1.0
                                        spot_diff = 0
                                        raw_exit_spot = s1.price
                                        
                                    if time_held_hours > 0:
                                        dte = scanner.expiry.get_dte(instrument, s1.timestamp.date())
                                        # Allow 0 or 1 to exist to apply high expiry day decay if not rolled over
                                        actual_dte = max(0, dte)
                                        theta_rate = 0.05 if actual_dte == 0 else (0.02 if actual_dte == 1 else 0.005)
                                        theta_decay = entry_premium * theta_rate * time_held_hours
                                        if spot_diff > 0:
                                            theta_decay *= 0.5 
                                        raw_exit_premium -= theta_decay
                                        
                                    raw_exit_premium -= (entry_premium * 0.002)
                                    raw_exit_premium = max(0.05, raw_exit_premium)
                            else:
                                raw_exit_premium = raw_exit_spot
                            
                            is_sl_hit = bool(actual_option and actual_option["reason"] == "SL HIT")
                            fut_target_hit = False
                            s12_exit_reason = ""
                            hist_s12_trade = None
                            exit_premium = raw_exit_premium
                            exit_time = actual_option["exit_time"] if actual_option else effective_exit_ts
                            
                            candles_1min = scanner.candles.get_candles(instrument, "1min")
                            futures_replay_window = None
                            
                            if actual_option is None and candles_1min is not None and not candles_1min.empty:
                                try:
                                    # Closed-candle start for every lane (entry_ts is already
                                    # shifted past the signal candle; see _closed_candle_replay_window).
                                    s1_ts = entry_ts
                                    between_candles = candles_1min.loc[s1_ts:effective_exit_ts]
                                    if (hist_uts or (hist_s12 and sim_inst == "FUT")) and sim_inst == "FUT":
                                        # Only bars traded after confirmation and fully
                                        # known by the exit decision may hit a resting stop.
                                        between_candles = candles_1min.loc[
                                            (candles_1min.index >= entry_ts)
                                            & (candles_1min.index + pd.Timedelta(minutes=1) <= effective_exit_ts)
                                        ]
                                        futures_replay_window = between_candles
                                        if not between_candles.empty:
                                            raw_exit_spot = float(between_candles.iloc[-1]["close"])
                                            raw_exit_premium = exit_premium = raw_exit_spot
                                    
                                    if not between_candles.empty:
                                        if sim_inst == "FUT":
                                            if hist_s12 and _setting_enabled(
                                                getattr(settings, "replay_s12_native_exits", True), True
                                            ):
                                                replay = self._simulate_s12_futures_exit_path(
                                                    between_candles,
                                                    signal=s1,
                                                    levels=hist_s12_levels,
                                                    market=hist_s12_market,
                                                    instrument=instrument,
                                                    timeframe=tf,
                                                    direction=direction,
                                                    entry_price=entry_premium,
                                                    settings=settings,
                                                )
                                                if replay is not None:
                                                    hist_s12_trade = replay["trade"]
                                                    s12_exit_reason = str(replay.get("reason") or "")
                                                    exit_premium = float(replay["price"])
                                                    exit_time = replay["time"]
                                                    raw_exit_spot = exit_premium
                                                    is_sl_hit = bool(replay.get("stop_hit"))
                                            else:
                                            # Exit parity (2026-07-09 audit): FUT replay
                                            # previously scanned ONLY the stop and rode
                                            # winners to the next flip/day end (zero
                                            # TARGET_HIT among 334 FUT rows) — the same
                                            # intrabar SL/target walk the OPTION replay
                                            # gets applies here now.
                                                fut_rr = self._dynamic_rr(hist_grade, hist_conf, timeframe=tf)
                                                fut_target = (
                                                    entry_premium + (index_stop_distance * fut_rr)
                                                    if direction == "LONG"
                                                    else entry_premium - (index_stop_distance * fut_rr)
                                                )
                                                sim_price, sim_reason, sim_ts = self._simulate_futures_exit_path(
                                                    between_candles, entry_premium, sl_price, fut_target, direction
                                                )
                                                if sim_reason == "SL HIT":
                                                    is_sl_hit = True
                                                    exit_premium = float(sim_price) if hist_uts else sl_price
                                                    exit_time = sim_ts
                                                elif sim_reason == "TARGET_HIT":
                                                    fut_target_hit = True
                                                    exit_premium = float(sim_price)
                                                    exit_time = sim_ts
                                        else:
                                            if direction == "LONG":
                                                all_premiums = entry_premium + ((between_candles['low'] - s1.price) * sim_multiplier)
                                                hit_rows = between_candles[all_premiums < sl_price]
                                                if not hit_rows.empty:
                                                    is_sl_hit = True
                                                    exit_premium = sl_price
                                                    exit_time = hit_rows.index[0]
                                            else:
                                                all_premiums = entry_premium + ((s1.price - between_candles['high']) * sim_multiplier)
                                                hit_rows = between_candles[all_premiums < sl_price]
                                                if not hit_rows.empty:
                                                    is_sl_hit = True
                                                    exit_premium = sl_price
                                                    exit_time = hit_rows.index[0]
                                except Exception as e:
                                    logger.warning(f"⚠️ Granular SL check failed: {e}")
                                    
                            if not is_sl_hit and not fut_target_hit:
                                if sim_inst == "FUT":
                                    if direction == "LONG" and raw_exit_spot < sl_price:
                                        exit_premium = sl_price
                                        is_sl_hit = True
                                    elif direction == "SHORT" and raw_exit_spot > sl_price:
                                        exit_premium = sl_price
                                        is_sl_hit = True
                                else:
                                    if raw_exit_premium < sl_price:
                                        exit_premium = sl_price
                                        is_sl_hit = True

                            if sim_inst == "OPT":
                                gross_pnl = (exit_premium - entry_premium) * qty
                            else:
                                if direction == "LONG":
                                    gross_pnl = (exit_premium - entry_premium) * qty
                                else:
                                    gross_pnl = (entry_premium - exit_premium) * qty
                            
                            sim_charges = estimate_trade_charges(
                                entry_premium,
                                exit_premium,
                                qty,
                                sim_inst,
                                settings,
                            )
                            net_pnl = gross_pnl - sim_charges
                            
                            if actual_option:
                                hist_exit_reason = actual_option["reason"]
                                if crosses_session and hist_exit_reason == "HISTORICAL":
                                    hist_exit_reason = "SESSION_END"
                            else:
                                hist_exit_reason = s12_exit_reason if hist_s12 and s12_exit_reason else (
                                    "SL HIT" if is_sl_hit else (
                                        "TARGET_HIT" if fut_target_hit
                                        else ("SESSION_END" if crosses_session else "HISTORICAL")
                                    )
                                )
                                if hist_s12 and not s12_exit_reason and hist_chart_reason:
                                    hist_exit_reason = s12_lane_policy.exit_reason_for(hist_chart_reason)
                                if hist_uts and sim_inst == "FUT" and not is_sl_hit and not fut_target_hit and hist_chart_reason:
                                    hist_exit_reason = uts_lane_policy.EXIT_REASONS.get(hist_chart_reason, "UTS_EXIT")
                            hist_rr = (uts_lane_policy.target_rr(settings) if hist_uts and sim_inst == "OPT"
                                       else self._dynamic_rr(hist_grade, hist_conf, timeframe=tf))
                            hist_lane = self._signal_source_lane(s1, scanner_engine_mode)
                            hist_source_engine = str(getattr(s1, "source_engine", "") or "")
                            hist_source_engines = list(getattr(s1, "source_engines", []) or [])
                            hist_id = self._historical_trade_id("H", instrument, tf, s1, scanner_engine_mode)

                            if not self._historical_signal_date_allowed(scanner, s1.timestamp):
                                logger.info(
                                    f"Skipping in-progress session historical trade before 15:30 handoff: {hist_id}"
                                )
                                continue
                            
                            if any(ct.id == hist_id for ct in scanner.trades.closed_trades):
                                continue
                                
                            if entry_premium <= 0:
                                logger.warning(f"Skipping simulated candidate {hist_id} due to invalid entry premium: {entry_premium}")
                                continue
                                
                            if sim_inst == "OPT":
                                hist_target = (
                                    0.0 if hist_s12 else
                                    actual_option["target"] if actual_option else entry_premium + ((entry_premium - sl_price) * hist_rr)
                                )
                            elif direction == "LONG":
                                hist_target = 0.0 if hist_s12 else s1.price + (index_stop_distance * hist_rr)
                            else:
                                hist_target = 0.0 if hist_s12 else s1.price - (index_stop_distance * hist_rr)

                            # Evaluate once, after option-history/product fallback has
                            # selected the product that the simulated trade will store.
                            hist_quality_verdict = self._quality_verdict_for_signal(
                                s1,
                                tf=tf,
                                grade=base_grade,
                                confidence=hist_conf,
                                regime=hist_regime,
                                inst_type=sim_inst,
                                settings=settings,
                                trend_15m_agrees=trend_15m_agrees_hist,
                                source_count=hist_source_count,
                                structural_impulse=hist_structural_impulse,
                                structural_power=power_move_hist,
                            )
                            if not hist_quality_verdict.allowed:
                                record_reject("Quality gate blocked", sim_inst)
                                self._historical_audit_log(
                                    f"Skipping {instrument} {tf} signal due to quality gate at {s1.timestamp}"
                                )
                                continue
                            hist_context = self._replay_entry_context(
                                scanner, instrument, tf, s1, direction, hist_regime, hist_lane_tag)
                            if hist_context is not None and not hist_context.allowed:
                                record_reject(f"Context gate {hist_context.rule_id}", sim_inst)
                                continue

                            historical_entry_time = (
                                actual_option["entry_time"] if actual_option
                                else entry_ts
                            )
                            e_time = (IST.localize(historical_entry_time)
                                      if (historical_entry_time and getattr(historical_entry_time, "tzinfo", None) is None)
                                      else historical_entry_time)
                            ex_time = IST.localize(exit_time) if (exit_time and getattr(exit_time, "tzinfo", None) is None) else exit_time
                            t = Trade(
                                id=hist_id, instrument=instrument, timeframe=tf,
                                direction=direction, entry_price=entry_premium, entry_time=e_time,
                                trailing_stop=sl_price,
                                current_stop=sl_price,
                                lots=hist_lots, lot_size=lot_size, grade=f"{hist_grade} (Hist)", confidence=hist_conf,
                                inst_type=sim_inst, option_type=sim_option_type, atm_strike=hist_strike,
                                rr_ratio=hist_rr, target=hist_target,
                                status="CLOSED", exit_price=exit_premium, exit_time=ex_time,
                                pnl=net_pnl, charges=sim_charges, exit_reason=hist_exit_reason,
                                instrument_multiplier=sim_multiplier,
                                entry_spot=(hist_s12_entry_spot or s1.price),
                                spot_stop=(hist_s12_levels.get("s12_stop") or s1.trailing_stop),
                                spot_target=(0.0 if hist_s12 else
                                    (s1.price + (index_stop_distance * hist_rr)) if direction == "LONG"
                                    else (s1.price - (index_stop_distance * hist_rr))),
                                trading_symbol=trading_symbol, symbol_token=symbol_token,
                                source_engine=hist_source_engine,
                                source_engines=hist_source_engines,
                                source_lane=hist_lane,
                                signal_time=(getattr(s1, "source_signal_timestamp", None)
                                             or s1.timestamp) if (hist_uts or hist_s12) else None,
                                data_provenance="BACKTEST",
                                market_regime=str(hist_regime or "UNKNOWN"),
                                entry_intelligence_score=float(hist_intel_score or 0.0),
                                intelligence_score=float(hist_intel_score or 0.0),
                                market_wide_reversal_entry=bool(getattr(s1, "market_wide_reversal_entry", False)),
                                volume_spike_entry=bool(getattr(s1, "volume_spike_entry", False)),
                                power_move=bool(power_move_hist or getattr(s1, "power_move", False)),
                                quad_union_entry=bool(getattr(s1, "quad_union_entry", False)),
                                raw_union_entry=bool(getattr(s1, "raw_union_entry", False)),
                                initial_stop=float(sl_price or 0.0),
                                replay_price_path=list((actual_option or {}).get("price_path") or []),
                                replay_exit_trace=(self._historical_futures_fill_evidence(
                                    futures_replay_window, signal=s1, entry_time=historical_entry_time,
                                    exit_time=exit_time, entry_price=entry_premium,
                                    exit_price=exit_premium, exit_reason=hist_exit_reason,
                                ) if hist_uts and sim_inst == "FUT" else
                                    list((actual_option or {}).get("exit_trace") or [])),
                                quality_verdict=self._with_entry_context(hist_quality_verdict.to_dict(), hist_context),
                            )
                            if hist_s12:
                                for key in s12_lane_policy.SESSION_STATE_FIELDS:
                                    if hasattr(s1, key):
                                        setattr(t, key, copy.deepcopy(getattr(s1, key)))
                                for key, value in ((actual_option or {}).get("s12_state") or {}).items():
                                    setattr(t, key, copy.deepcopy(value))
                                for key, value in hist_s12_levels.items():
                                    if key not in {"s12_stop", "s12_stop_initial"} or not hasattr(t, key):
                                        setattr(t, key, value)
                                t.s12_lane = hist_lane_tag
                                t.s12_trade_id = hist_s12_market["trade_id"]
                                hist_s12_final_state = (actual_option or {}).get("s12_state") or {}
                                t.s12_stop = float(
                                    hist_s12_final_state.get("s12_stop", 0.0)
                                    or getattr(hist_s12_trade, "s12_stop", 0.0)
                                    or hist_s12_levels.get("s12_stop") or 0.0
                                ) if hist_s12_trade is not None else float(
                                    hist_s12_final_state.get("s12_stop", 0.0)
                                    or hist_s12_levels.get("s12_stop") or 0.0
                                )
                                t.s12_stop_initial = float(
                                    hist_s12_levels.get("s12_stop_initial") or hist_s12_levels.get("s12_stop") or 0.0
                                )
                                t.s12_premium_cap_pct = float(hist_option_cap) if sim_inst == "OPT" else 0.0
                            
                            live_session_day_raw = (
                                scanner._get_current_session_day()
                                if hasattr(scanner, "_get_current_session_day")
                                else datetime.now(IST).date().isoformat()
                            )
                            live_session_day = datetime.fromisoformat(str(live_session_day_raw)).date()
                            if is_live and s1.timestamp.date() == live_session_day:
                                logger.info(
                                    f"Skipping simulated backfill signal {hist_id} in live mode; "
                                    "live dashboard ledger only accepts real post-gate session rows."
                                )
                                continue
                            if is_live and s1.timestamp.date() == live_session_day:
                                exit_candidate = TradeCandidate(
                                    instrument=instrument,
                                    direction=direction,
                                    price=entry_premium,
                                    stop=sl_price,
                                    target=hist_target,
                                    lots=hist_lots,
                                    lot_size=lot_size,
                                    grade=hist_grade,
                                    confidence=hist_conf,
                                    timeframe=tf,
                                    inst_type=sim_inst,
                                    option_type=sim_option_type,
                                    atm_strike=hist_strike,
                                    multiplier=sim_multiplier,
                                    trading_symbol=trading_symbol,
                                    symbol_token=symbol_token,
                                    rr=hist_rr,
                                    score=hist_conf,
                                    reasons=list(hist_reasons or []) + [f"Signal closed by {hist_exit_reason}"],
                                    spot_stop=s1.trailing_stop,
                                    spot_target=(s1.price + (index_stop_distance * hist_rr)) if direction == "LONG" else (s1.price - (index_stop_distance * hist_rr)),
                                    signal_timestamp=s1.timestamp,
                                    current_price=exit_premium,
                                    pnl=net_pnl,
                                    status="EXIT SIGNAL",
                                    action="EXIT",
                                    exit_reason=hist_exit_reason,
                                    source_engine=hist_source_engine,
                                    source_engines=hist_source_engines,
                                    source_lane=hist_lane,
                                )
                                setattr(exit_candidate, "exit_timestamp", exit_time)
                                setattr(exit_candidate, "exit_price", exit_premium)
                                scanner._remember_trade_candidates([exit_candidate])
                                logger.info(f"⛔ Skipping adding historical trade {hist_id} to closed_trades in live mode.")
                            elif self._historical_trade_creation_allowed(scanner):
                                scanner.trades.closed_trades.append(t)
                                if sim_inst == "OPT" and self._hybrid_backtest_dual_futures_enabled(scanner):
                                    shadow_id = f"{hist_id}_FUT"
                                    if not any(ct.id == shadow_id for ct in scanner.trades.closed_trades):
                                        shadow = self._build_historical_futures_shadow_trade(
                                            scanner=scanner,
                                            trade_id=shadow_id,
                                            instrument=instrument,
                                            timeframe=tf,
                                            direction=direction,
                                            signal=s1,
                                            exit_spot=raw_exit_spot,
                                            exit_time=exit_time,
                                            grade=hist_grade,
                                            confidence=hist_conf,
                                            rr=hist_rr,
                                            lot_size=lot_size,
                                            display_suffix="Hist",
                                            default_exit_reason=("SL HIT" if is_sl_hit else "HISTORICAL"),
                                            regime=hist_regime,
                                        )
                                        if shadow:
                                            scanner.trades.closed_trades.append(shadow)
                            else:
                                logger.info(
                                    f"Blocked historical trade creation outside HISTORICAL mode: {hist_id}"
                                )

                # CRITICAL FIX: Only recover signals from TODAY's session.
                    # A signal from a previous day (e.g. 07 May 14:30) would have been 
                    # squared off at session end (15:30). It is NOT a live position.
                    # Master Fix: We DO want to recover trades if the bot crashes during a live/paper session!
                    allow_inferred_recovery = getattr(
                        scanner,
                        "_allow_inferred_signal_recovery",
                        lambda: False,
                    )
                    historical_final_uts = bool(
                        not is_live and sigs
                        and str(getattr(scanner, "mode", "")).upper() == "HISTORICAL"
                        and uts_lane_policy.is_uts(sigs[-1], settings)
                    )
                    if historical_final_uts or (is_live and sigs and bool(allow_inferred_recovery())):
                        logger.info(f"Running live recovery for {instrument} {tf}...")
                        last_s = sigs[-1]
                        session_day_raw = (
                            scanner._get_current_session_day()
                            if hasattr(scanner, "_get_current_session_day")
                            else datetime.now(IST).date().isoformat()
                        )
                        session_day = datetime.fromisoformat(str(session_day_raw)).date()
                        try:
                            now_ist = datetime.now(IST)
                            sess_end_str = get_settings().ut_session_end
                            sess_hr, sess_min = map(int, sess_end_str.split(':'))
                            kill_threshold = dtime(sess_hr, sess_min)
                            
                            market_session_active = (
                                now_ist.date() == session_day
                                and now_ist.time() < kill_threshold
                                and bool(scanner.data.is_market_open())
                            )
                        except Exception:
                            now_ist = datetime.now(IST)
                            sess_end_str = get_settings().ut_session_end
                            sess_hr, sess_min = map(int, sess_end_str.split(':'))
                            market_session_active = now_ist.date() == session_day and now_ist.time() < dtime(sess_hr, sess_min)
                        signal_is_from_today = (
                            (scanner.mode == "REAL")
                            and last_s.timestamp.date() == session_day
                            and market_session_active
                        )
                        
                        tf_minutes = {"1min": 1, "5min": 5, "15min": 15}.get(tf, 5)
                        candle_close_time = naive_ts(last_s.timestamp) + timedelta(minutes=tf_minutes)
                        now_naive = naive_ts(datetime.now(IST))
                        
                        recovery_window_ok = True
                        recovery_window_fn = getattr(scanner, "_signal_in_restart_recovery_window", None)
                        if callable(recovery_window_fn):
                            recovery_window_ok = bool(recovery_window_fn(last_s.timestamp, tf))

                        if signal_is_from_today and now_naive >= candle_close_time and recovery_window_ok:
                            direction = "LONG" if last_s.signal_type == "BUY" else "SHORT"
                            logger.info(f"🔄 Recovering Active {direction} state for {instrument} from {last_s.timestamp}")
                            if not scanner.is_warmup:
                                scanner.log_event(f"🔄 Recovered {direction} signal for {instrument} from {last_s.timestamp.strftime('%H:%M')}", "trade")
                            
                            # ── Recovery Cross-TF Guard ──
                            existing = [
                                t
                                for t in scanner.trades.open_trades.values()
                                if t.instrument == instrument and self._is_live_book_trade(t)
                            ]
                            
                            # Check if signal was already processed and closed
                            existing_closed = [
                                t for t in scanner.trades.closed_trades 
                                if t.instrument == instrument and 
                                getattr(t, 'entry_time', None) and
                                (t.entry_time.strftime('%Y%m%d%H%M') == last_s.timestamp.strftime('%Y%m%d%H%M') if isinstance(t.entry_time, datetime) else str(t.entry_time) == last_s.timestamp.strftime('%d %b %H:%M:%S'))
                                and "REPAINT" not in str(getattr(t, "exit_reason", "")).upper()
                            ]
                            if existing_closed:
                                logger.info(f"⛔ Signal for {instrument} at {last_s.timestamp} already processed and closed. Skipping recovery.")
                                continue
                            
                            logger.info(f"🔍 [AUDIT] Recovery logic running for {instrument} {tf}. last_s={last_s.timestamp} {last_s.signal_type}. scanner.inst_pref: {scanner.inst_pref}")
                            
                            # Data Sanity Check: Detect swapped prices!
                            if _is_bfo_instrument(instrument) and last_s.price < 50000:
                                logger.error(
                                    f"🚨 [AUDIT] {instrument} signal has NIFTY price: "
                                    f"{last_s.price}. Skipping recovery."
                                )
                                continue
                            if instrument == "NIFTY" and last_s.price > 50000:
                                logger.error(f"🚨 [AUDIT] NIFTY signal has SENSEX price: {last_s.price}. Skipping recovery.")
                                continue

                            # Strict instrument preference (same rule as missed re-entry / resolver):
                            # OPT → options only, FUT → futures only, AUTO → recovery defaults to OPT.
                            pref = str(getattr(scanner, "inst_pref", "AUTO") or "AUTO").upper()
                            if pref == "OPT":
                                rec_inst_type = "OPT"
                            elif pref == "FUT":
                                rec_inst_type = "FUT"
                            else:
                                rec_inst_type = "OPT"

                            if existing:
                                logger.info(f"🚷 Recovery Guard: Already have an open trade for {instrument}. Skipping recovery.")
                                continue

                            
                            try:
                                live_regime = regime
                            except Exception:
                                live_regime = "UNKNOWN"
                            try:
                                live_confluence = float(getattr(mtf_result, "confluence_score", 0.5) or 0.5)
                            except Exception:
                                live_confluence = 0.5
                            try:
                                live_intel_score = float(intel_score or 0.0)
                            except Exception:
                                live_intel_score = 0.0
                            # Restart recovery must not resurrect an ordinary 1m UT
                            # flip. Stateful snapback/cascade routes are evaluated in
                            # normal live processing; recovery permits confirmed bursts.
                            if tf == "1min":
                                rec_burst_detector = getattr(scanner, "_detect_volume_spike_entry", None)
                                rec_volume_spike_metrics = (
                                    rec_burst_detector(
                                        instrument,
                                        tf,
                                        last_s.signal_type,
                                        adx_value=float(getattr(last_s, "adx_value", 0.0) or 0.0),
                                        intel_score=live_intel_score,
                                        as_of=last_s.timestamp,
                                    )
                                    if callable(rec_burst_detector)
                                    else None
                                )
                                setattr(last_s, "volume_spike_entry", bool(rec_volume_spike_metrics))
                                setattr(last_s, "volume_spike_metrics", dict(rec_volume_spike_metrics or {}))
                                if not rec_volume_spike_metrics:
                                    logger.info(
                                        f"Recovery Guard: {instrument} 1min signal at {last_s.timestamp} "
                                        "requires confirmed directional burst."
                                    )
                                    continue
                            try:
                                rec_vix_key = last_s.timestamp.strftime("%Y-%m-%d %H:%M:00")
                                rec_vix = scanner._vix_data.get(rec_vix_key)
                                if rec_vix is None:
                                    rec_vix = scanner.data.get_ltp("NSE", "INDIAVIX", "99926017") or 15.0
                            except Exception:
                                rec_vix = 15.0
                            try:
                                rec_grade, rec_conf, _ = scanner.signals._grade_signal(
                                    last_s,
                                    live_confluence,
                                    live_intel_score,
                                    live_regime,
                                    vix_value=rec_vix,
                                )
                            except Exception as exc:
                                logger.warning(f"Recovery quality grading failed for {instrument} {tf}: {exc}")
                                rec_grade, rec_conf, rec_reasons = "C", 0.0, ["Recovery grading failed"]

                            rec_source_engines = list(getattr(last_s, "source_engines", []) or [])
                            rec_trend_15m_agrees = self._trend_15m_agrees_at(
                                mtf_result,
                                last_s.timestamp,
                                last_s.signal_type,
                            )
                            rec_structural_impulse = bool(
                                tf == "5min"
                                and self._is_5min_impulse_reversal(
                                    last_s,
                                    mtf_result,
                                    rec_conf,
                                    settings,
                                )
                            )
                            rec_quality_verdict = self._quality_verdict_for_signal(
                                last_s,
                                tf=tf,
                                grade=rec_grade,
                                confidence=rec_conf,
                                regime=live_regime,
                                inst_type=rec_inst_type,
                                settings=settings,
                                trend_15m_agrees=rec_trend_15m_agrees,
                                source_count=max(1, len(rec_source_engines)),
                                structural_impulse=rec_structural_impulse,
                            )
                            if not rec_quality_verdict.allowed:
                                logger.info(
                                    f"Recovery Guard: {instrument} {tf} {last_s.signal_type} at {last_s.timestamp} "
                                    f"rejected by live quality gate (grade={rec_grade}, conf={rec_conf:.3f}, regime={live_regime})."
                                )
                                continue

                            # Calculate atm_strike for recovery if it's an Option
                            rec_atm_strike = 0.0
                            rec_option_type = ""
                            rec_trading_symbol = str(getattr(last_s, "trading_symbol", "") or "")
                            rec_symbol_token = str(getattr(last_s, "symbol_token", "") or "")
                            rec_multiplier = 1.0
                            if rec_inst_type == "OPT":
                                strike_interval = cfg.get("strike_interval", 50)
                                rec_atm_strike = round(last_s.price / strike_interval) * strike_interval
                                rec_option_type = "CE" if direction == "LONG" else "PE"
                                # Audit #4: never invent synthetic OPT recovery marks.
                                # Require a verified live premium; REAL also requires a
                                # confirmed broker position before opening a local trade.
                                # Instrument-lock (2026-07-24): under OPT the retained seed
                                # symbol/token may be a FUTURES contract inherited from an
                                # AUTO/hybrid signal. Re-resolve to a genuine option unless
                                # the retained symbol is already an option (…CE/…PE), and
                                # never open OPT on a futures contract.
                                _rec_sym_is_option = str(rec_trading_symbol).upper().endswith(("CE", "PE"))
                                if not rec_symbol_token or not rec_trading_symbol or not _rec_sym_is_option:
                                    opt_info = None
                                    try:
                                        opt_info = scanner.data.get_option_token(
                                            instrument, rec_atm_strike, rec_option_type
                                        )
                                    except Exception as exc:
                                        logger.warning(
                                            f"Recovery Guard: option token lookup failed for "
                                            f"{instrument} {rec_atm_strike} {rec_option_type}: {exc}"
                                        )
                                    if opt_info:
                                        rec_trading_symbol = str(opt_info.get("symbol") or "")
                                        rec_symbol_token = str(opt_info.get("token") or "")
                                if not rec_symbol_token or not str(rec_trading_symbol).upper().endswith(("CE", "PE")):
                                    logger.warning(
                                        f"Recovery Guard: skip OPT recovery for {instrument} {tf} — "
                                        f"no genuine option contract (refusing to open OPT on "
                                        f"'{rec_trading_symbol or 'unknown'}')."
                                    )
                                    continue
                                exchange = "BFO" if _is_bfo_instrument(instrument) else "NFO"
                                rec_entry = float(
                                    self._contract_ltp(
                                        scanner, exchange, rec_trading_symbol, rec_symbol_token
                                    )
                                    or 0.0
                                )
                                if rec_entry <= 0:
                                    logger.warning(
                                        f"Recovery Guard: skip OPT recovery for {instrument} {tf} — "
                                        f"no verified LTP for {rec_trading_symbol}."
                                    )
                                    continue
                                try:
                                    dte_val = scanner.expiry.get_dte(instrument, last_s.timestamp.date())
                                except Exception:
                                    dte_val = 1
                                rec_multiplier = self._option_signed_delta(
                                    last_s.price,
                                    rec_atm_strike,
                                    rec_option_type,
                                    {},
                                    dte_val,
                                    0.5 if rec_option_type == "CE" else -0.5,
                                )
                                if str(getattr(scanner, "mode", "") or "").upper() == "REAL":
                                    broker = getattr(scanner.trades, "broker", None)
                                    positions_fn = getattr(broker, "get_positions", None) if broker else None
                                    broker_qty = 0
                                    broker_avg = 0.0
                                    if callable(positions_fn):
                                        try:
                                            pos_resp = positions_fn() or {}
                                            for row in (pos_resp.get("data") or []):
                                                sym = str(
                                                    row.get("tradingsymbol")
                                                    or row.get("symbol")
                                                    or ""
                                                )
                                                tok = str(row.get("symboltoken") or row.get("token") or "")
                                                if tok == rec_symbol_token or (
                                                    rec_trading_symbol and rec_trading_symbol in sym
                                                ):
                                                    net = int(
                                                        float(
                                                            row.get("netqty")
                                                            or row.get("quantity")
                                                            or 0
                                                        )
                                                    )
                                                    broker_qty = abs(net)
                                                    broker_avg = float(
                                                        row.get("avgnetprice")
                                                        or row.get("averageprice")
                                                        or row.get("avgprice")
                                                        or 0.0
                                                    )
                                                    break
                                        except Exception as exc:
                                            logger.warning(
                                                f"Recovery Guard: broker position check failed: {exc}"
                                            )
                                    if broker_qty <= 0:
                                        logger.critical(
                                            f"Recovery Guard: REAL mode refuses synthetic OPT open for "
                                            f"{instrument} {rec_trading_symbol} — no broker position. "
                                            "Reconcile manually; not inventing a local book."
                                        )
                                        continue
                                    if broker_avg > 0:
                                        rec_entry = broker_avg
                            else:
                                rec_entry = float(last_s.price or 0.0)
                                if rec_entry <= 0:
                                    continue

                            # Use the same quality-scaled RR as live so a recovered
                            # position carries the target it would have had originally.
                            rec_rr = self._dynamic_rr(rec_grade, rec_conf, timeframe=tf)
                            if rec_inst_type == "OPT":
                                rec_risk = max(0.5, rec_entry * (scanner.options_sl_pct / 100.0))
                                rec_stop = max(0.05, rec_entry - rec_risk)
                                rec_target = rec_entry + (rec_risk * rec_rr)
                                requested_lots = int(
                                    getattr(scanner, "user_lots", {}).get(instrument, 1) or 1
                                )
                            else:
                                rec_fut_sl_pct = futures_stop_tolerance_pct(
                                    scanner.futures_sl_pct,
                                    live_regime,
                                )
                                rec_risk = max(1.0, rec_entry * (rec_fut_sl_pct / 100.0))
                                if direction == "LONG":
                                    rec_stop = max(last_s.trailing_stop, rec_entry - rec_risk)
                                    rec_target = rec_entry + ((rec_entry - rec_stop) * rec_rr)
                                else:
                                    rec_stop = min(last_s.trailing_stop, rec_entry + rec_risk)
                                    rec_target = rec_entry - ((rec_stop - rec_entry) * rec_rr)
                                requested_lots = int(
                                    getattr(scanner, "user_lots_fut", {}).get(instrument, 1) or 1
                                )
                            size_result = resolve_trade_lots(
                                settings=settings,
                                inst_type=rec_inst_type,
                                requested_lots=requested_lots,
                                lot_size=lot_size,
                                entry_price=rec_entry,
                                stop_price=rec_stop,
                                # Recovery re-establishes an EXISTING position; there is
                                # no fresh intelligence reading behind it and conviction
                                # must not resize what is already on.
                                intel_score=0.0,
                            )
                            rec_lots = int(size_result.lots or 0)
                            if rec_lots < 1:
                                logger.info(
                                    f"Recovery Guard: skip {instrument} {tf} — {size_result.reason}"
                                )
                                continue
                            scanner.trades.open_trade(
                                instrument=instrument, timeframe=tf, direction=direction,
                                price=rec_entry,
                                trailing_stop=rec_stop,
                                target=rec_target,
                                lots=rec_lots, lot_size=lot_size, grade=rec_grade,
                                confidence=rec_conf,
                                entry_time=last_s.timestamp,
                                instrument_multiplier=rec_multiplier,
                                option_type=rec_option_type,
                                atm_strike=rec_atm_strike,
                                trading_symbol=rec_trading_symbol,
                                symbol_token=rec_symbol_token,
                                inst_type=rec_inst_type,
                                entry_spot=last_s.price, 
                                is_explosive_bypass=bool(
                                    getattr(last_s, "is_explosive_bypass", False)
                                ),
                                is_recovery=True,
                                market_regime=str(live_regime or ""),
                                data_provenance="RECOVERY",
                                intelligence_score=float(live_intel_score or 0.0),
                                strategy_revision=str(getattr(settings, "strategy_revision", "") or ""),
                                strategy_direction_mode=str(getattr(settings, "strategy_direction_mode", "") or ""),
                                source_engine=str(getattr(last_s, "source_engine", "") or ""),
                                source_engines=list(getattr(last_s, "source_engines", []) or []),
                                source_lane=self._signal_source_lane(last_s, scanner_engine_mode),
                                initial_stop=rec_stop,
                                scalp_lock_mode=False,
                                quality_verdict=rec_quality_verdict.to_dict(),
                            )
                            if scanner.trades.open_trades:
                                active_tid = list(scanner.trades.open_trades.keys())[-1]
                                active_trade = scanner.trades.open_trades[active_tid]
                                ts = pd.Timestamp(last_s.timestamp)
                                if ts.tzinfo is not None:
                                    active_trade.entry_time = ts.tz_convert(IST).tz_localize(None).to_pydatetime()
                                else:
                                    active_trade.entry_time = ts.to_pydatetime()
                                # Marks already set from verified LTP / broker avg — do not overwrite with estimates.
                                active_trade.entry_price = rec_entry
                                active_trade.current_price = rec_entry
                                active_trade.instrument_multiplier = rec_multiplier
                                active_trade.current_stop = rec_stop
                                active_trade.target = rec_target
                        elif last_s.timestamp.date() >= trade_cutoff:
                            # ═══ SYNTHETIC SESSION-END CLOSE ═══
                            # Signal from a previous day — create a closed trade at session end price
                            
                            # ── Grade Filter: Evaluate what we would trade ──
                            eod_regime = "UNKNOWN"
                            eod_intel_score = 0.0
                            eod_confluence = 0.0
                            eod_vix = scanner._get_historical_vix(last_s.timestamp)
                            eod_day_type, eod_day_conf = "", 0.0
                            try:
                                eod_intel_score, eod_intel_regime = scanner._compute_historical_intel_at(
                                    instrument,
                                    last_s.timestamp,
                                    tf,
                                )
                                if eod_intel_regime and eod_intel_regime != "UNKNOWN":
                                    eod_regime = eod_intel_regime
                                eod_confluence = scanner._compute_historical_confluence_at(
                                    mtf_result,
                                    last_s.timestamp,
                                )
                                day_at = getattr(scanner, "_historical_day_type_at", None)
                                if callable(day_at):
                                    eod_day_type, eod_day_conf = day_at(
                                        instrument,
                                        last_s.timestamp,
                                        tf,
                                    )
                                eod_df = scanner.candles.get_candles(instrument, tf)
                                if eod_df is not None and not eod_df.empty:
                                    if eod_regime == "UNKNOWN":
                                        eod_regime_res = scanner.intel.regime.detect(
                                            eod_df[eod_df.index <= last_s.timestamp],
                                            instrument,
                                            tf,
                                        )
                                        eod_regime = eod_regime_res.get("regime", "UNKNOWN")
                            except Exception as exc:
                                logger.debug(f"EOD option regime fallback for {instrument}: {exc}")

                            eod_grade, eod_conf, _ = scanner.signals._grade_signal(
                                last_s,
                                eod_confluence,
                                eod_intel_score,
                                regime=eod_regime,
                                vix_value=eod_vix,
                                day_type=eod_day_type,
                                day_type_confidence=eod_day_conf,
                            )
                            if eod_grade == "C" and not historical_final_uts:
                                continue

                            if historical_final_uts:
                                # This final arrow has no next arrow to pair with,
                                # but admission still belongs to the same policy as
                                # every earlier historical UTS entry. Grade is display
                                # metadata; direction, cutoff and native evidence bind.
                                direction_allowed, _ = self._direction_mode_allows(
                                    last_s.signal_type, settings
                                )
                                if not direction_allowed or not self._historical_entry_cutoff_allows(
                                    last_s, tf, "UTS", settings
                                ):
                                    continue
                                final_native_verdict = self._quality_verdict_for_signal(
                                    last_s, tf=tf, grade=eod_grade, confidence=eod_conf,
                                    regime=eod_regime, inst_type="OPT", settings=settings,
                                    trend_15m_agrees=self._trend_15m_agrees_at(
                                        mtf_result, last_s.timestamp, last_s.signal_type,
                                        closed_only=True,
                                    ),
                                )
                                if not final_native_verdict.allowed:
                                    continue

                            # Apply burst-only admission to the unpaired final 1m
                            # signal too. No historical L1 snapshot exists, so depth is
                            # deliberately unavailable rather than read from live book.
                            if tf == "1min" and not historical_final_uts:
                                eod_burst_detector = getattr(scanner, "_detect_volume_spike_entry", None)
                                eod_volume_spike_metrics = (
                                    eod_burst_detector(
                                        instrument,
                                        tf,
                                        last_s.signal_type,
                                        adx_value=float(getattr(last_s, "adx_value", 0.0) or 0.0),
                                        intel_score=float(eod_intel_score or 0.0),
                                        as_of=last_s.timestamp,
                                        depth_context={},
                                    )
                                    if callable(eod_burst_detector)
                                    else None
                                )
                                setattr(last_s, "volume_spike_entry", bool(eod_volume_spike_metrics))
                                setattr(last_s, "volume_spike_metrics", dict(eod_volume_spike_metrics or {}))
                                if not eod_volume_spike_metrics:
                                    self._historical_audit_log(
                                        f"Skipping {instrument} 1min final signal at {last_s.timestamp}: "
                                        "confirmed directional burst required"
                                    )
                                    continue
                                
                            # Historical concurrency is applied after all workers finish.
                            # Check for existing manual exits before applying EOD close
                            already_exited = False
                            book = getattr(scanner, "_session_trade_candidates", {}).get(instrument, {})
                            for c in book.values():
                                if getattr(c, "action", "ENTRY") == "EXIT" and c.timeframe == tf and c.direction == ("LONG" if last_s.signal_type == "BUY" else "SHORT"):
                                    if getattr(c, "signal_timestamp", None) == last_s.timestamp:
                                        already_exited = True
                                        break
                            if already_exited:
                                continue

                            direction = "LONG" if last_s.signal_type == "BUY" else "SHORT"
                            sig_date = last_s.timestamp.date()
                            # Quality-scaled RR (same as live) so EOD-simulated targets
                            # match the live system rather than a divergent flat 1.5.
                            eod_rr = self._dynamic_rr(eod_grade, eod_conf, timeframe=tf)

                            # Find the closing price on that date from candle data
                            candles_for_tf = scanner.candles.get_candles(instrument, tf)
                            eod_price = last_s.price  # fallback
                            if candles_for_tf is not None and not candles_for_tf.empty:
                                day_candles = candles_for_tf[candles_for_tf.index.date == sig_date]
                                force_exit_time = datetime.strptime(
                                    str(getattr(settings, "ut_force_exit_time", "15:25") or "15:25"),
                                    "%H:%M",
                                ).time()
                                day_candles = day_candles[day_candles.index.time <= force_exit_time]
                                if not day_candles.empty:
                                    eod_price = day_candles['close'].iloc[-1]
                            
                            eod_lane = self._signal_source_lane(last_s, scanner_engine_mode)
                            eod_uts = eod_lane == "UTS"
                            eod_source_engine = str(getattr(last_s, "source_engine", "") or "")
                            eod_source_engines = list(getattr(last_s, "source_engines", []) or [])
                            eod_id = self._historical_trade_id("EOD", instrument, tf, last_s, scanner_engine_mode, fmt="%m%d%H%M")
                            if not self._historical_signal_date_allowed(scanner, last_s.timestamp):
                                logger.info(
                                    f"Skipping in-progress session historical EOD trade before 15:30 handoff: {eod_id}"
                                )
                                continue
                            if not any(ct.id == eod_id for ct in scanner.trades.closed_trades):
                                eod_exit_time = datetime.combine(sig_date, force_exit_time)
                                eod_start = last_s.timestamp
                                eod_chart_reason = ""
                                if eod_uts:
                                    eod_start, eod_exit_time, eod_chart_reason = self._uts_replay_window(last_s, tf, eod_exit_time)
                                else:
                                    eod_naive = eod_start.replace(tzinfo=None) if getattr(eod_start, "tzinfo", None) else eod_start
                                    eod_start, _ = self._closed_candle_replay_window(
                                        eod_naive, tf, eod_exit_time, eod_exit_time, True)
                                e_time = IST.localize(eod_start) if eod_start.tzinfo is None else eod_start
                                ex_time = IST.localize(eod_exit_time) if eod_exit_time.tzinfo is None else eod_exit_time
                                
                                
                                # ═══ EOD Mode-Aware Intelligence ═══
                                if scanner.inst_pref == "FUT":
                                    sim_inst = "FUT"
                                    sim_option_type = ""
                                    sim_multiplier = 1.0
                                    entry_premium = last_s.price
                                    exit_premium = eod_price
                                elif scanner.inst_pref == "OPT":
                                    sim_inst = "OPT"
                                    sim_option_type = "CE" if direction == "LONG" else "PE"
                                    sim_multiplier = 0.5
                                    entry_premium = _estimate_opt_premium(last_s.price, scanner, instrument, _sig_date(last_s))
                                    spot_diff = eod_price - last_s.price if direction == "LONG" else last_s.price - eod_price
                                    exit_premium = entry_premium + (spot_diff * sim_multiplier)
                                else: # AUTO
                                    if historical_final_uts:
                                        sim_inst = scanner._resolve_instrument_type(
                                            instrument=instrument, grade=eod_grade,
                                            regime=eod_regime, intel_score=eod_intel_score,
                                            signal_score=eod_conf, confidence=eod_conf,
                                            adx_value=getattr(last_s, "adx_value", 0.0),
                                            atr_value=getattr(last_s, "atr_value", 0.0),
                                            price=last_s.price, signal_time=last_s.timestamp,
                                            timeframe=tf, five_min_true_impulse=False,
                                        )
                                    else:
                                        eod_option_allowed = self._option_grade_allowed(
                                            eod_grade,
                                            eod_conf,
                                            eod_regime,
                                            adx_value=getattr(last_s, "adx_value", 0.0),
                                            intel_score=eod_intel_score,
                                            signal_score=eod_conf,
                                            timeframe=tf,
                                        )
                                        sim_inst = "OPT" if eod_option_allowed else "FUT"
                                    sim_option_type = ("CE" if direction == "LONG" else "PE") if sim_inst == "OPT" else ""
                                    sim_multiplier = 0.5 if sim_inst == "OPT" else 1.0
                                    if sim_inst == "OPT":
                                        entry_premium = _estimate_opt_premium(last_s.price, scanner, instrument, _sig_date(last_s))
                                        spot_diff = eod_price - last_s.price if direction == "LONG" else last_s.price - eod_price
                                        exit_premium = entry_premium + (spot_diff * sim_multiplier)
                                    else:
                                        entry_premium = last_s.price
                                        exit_premium = eod_price

                                sim_atm_strike = 0.0
                                trading_symbol = ""
                                symbol_token = ""
                                eod_exit_reason = "SESSION_END"
                                eod_stop = last_s.trailing_stop
                                if sim_inst == "FUT":
                                    eod_fut_sl_pct = futures_stop_tolerance_pct(
                                        scanner.futures_sl_pct,
                                        eod_regime,
                                    )
                                    max_fut_sl_dist = entry_premium * (eod_fut_sl_pct / 100.0)
                                    if direction == "LONG":
                                        eod_stop = max(last_s.trailing_stop, entry_premium - max_fut_sl_dist)
                                        eod_target = entry_premium + ((entry_premium - eod_stop) * eod_rr)
                                    else:
                                        eod_stop = min(last_s.trailing_stop, entry_premium + max_fut_sl_dist)
                                        eod_target = entry_premium - ((eod_stop - entry_premium) * eod_rr)
                                else:
                                    eod_target = last_s.price + (abs(last_s.price - last_s.trailing_stop) * eod_rr) if direction == "LONG" else last_s.price - (abs(last_s.price - last_s.trailing_stop) * eod_rr)

                                actual_eod = None
                                if sim_inst == "OPT":
                                    sim_atm_strike = self._historical_option_strike(instrument, direction, last_s.price, cfg)
                                    option_df, trading_symbol, symbol_token = self._load_option_candles(
                                        scanner, instrument, sim_atm_strike, sim_option_type, "1min", sig_date,
                                        now_dt=getattr(last_s, "timestamp", None)
                                    )
                                    if eod_uts and option_df.empty:
                                        continue
                                    history_mode = getattr(settings, "ut_option_history_mode", "fetch_or_skip")
                                    if option_df.empty and history_mode == "fetch_or_skip":
                                        option_pref = (getattr(scanner, "inst_pref", "AUTO") or "AUTO").upper()
                                        if option_pref == "OPT":
                                            logger.info(f"Skipping EOD option {instrument} {sim_option_type} {sim_atm_strike} at {last_s.timestamp}: no real option candles.")
                                            continue
                                        logger.info(
                                            f"Falling back to FUT for EOD {instrument} {tf} at {last_s.timestamp}: "
                                            "no real option candles in AUTO/HYBRID."
                                        )
                                        sim_inst = "FUT"
                                        sim_option_type = ""
                                        sim_multiplier = 1.0
                                        entry_premium = last_s.price
                                        exit_premium = eod_price
                                        eod_fut_sl_pct = futures_stop_tolerance_pct(
                                            scanner.futures_sl_pct,
                                            eod_regime,
                                        )
                                        max_fut_sl_dist = entry_premium * (eod_fut_sl_pct / 100.0)
                                        if direction == "LONG":
                                            eod_stop = max(last_s.trailing_stop, entry_premium - max_fut_sl_dist)
                                            eod_target = entry_premium + ((entry_premium - eod_stop) * eod_rr)
                                        else:
                                            eod_stop = min(last_s.trailing_stop, entry_premium + max_fut_sl_dist)
                                            eod_target = entry_premium - ((eod_stop - entry_premium) * eod_rr)

                                    est_entry = float(entry_premium or 0.0)
                                    spot_stop_dist = abs(float(last_s.price) - float(last_s.trailing_stop))
                                    eod_option_cap = option_stop_tolerance_pct(
                                        scanner.options_sl_pct, eod_regime,
                                        confidence=eod_conf, grade=eod_grade,
                                        setup=last_s,
                                    )
                                    if eod_uts:
                                        natural_sl_premium_dist = self._historical_uts_premium_stop_distance(
                                            est_entry,
                                            spot_stop_dist,
                                            sim_multiplier,
                                            settings,
                                            regime=eod_regime,
                                            confidence=eod_conf,
                                            grade=eod_grade,
                                            setup=last_s, timeframe=tf,
                                        )
                                        effective_sl_dist = natural_sl_premium_dist
                                    else:
                                        natural_sl_premium_dist = abs(spot_stop_dist * sim_multiplier)
                                        max_sl_dist = est_entry * (eod_option_cap / 100.0)
                                        effective_sl_dist = max(
                                            0.05,
                                            min(natural_sl_premium_dist, max_sl_dist),
                                        )
                                    opt_sl_price = max(0.05, est_entry - effective_sl_dist)
                                    if sim_inst == "OPT" and not option_df.empty:
                                        eod_probe_qty = max(
                                            1,
                                            int(
                                                getattr(scanner, "user_lots", {}).get(instrument, 1)
                                                or 1
                                            ) * int(lot_size or 1),
                                        )
                                        actual_eod = self._simulate_actual_option_exit(
                                            option_df,
                                            eod_start,
                                            eod_exit_time,
                                            opt_sl_price,
                                            uts_lane_policy.target_rr(settings) if eod_uts else eod_rr,
                                            natural_sl_distance=natural_sl_premium_dist,
                                            max_sl_pct=eod_option_cap / 100.0,
                                            qty=eod_probe_qty,
                                            tf15_qty=eod_probe_qty,
                                            tf15_exit_scanner=scanner,
                                            tf15_instrument=instrument,
                                            tf15_direction=direction,
                                            tf15_option_type=sim_option_type,
                                            tf15_timeframe=tf,
                                            replay_grade=eod_grade,
                                            replay_confidence=eod_conf,
                                            replay_regime=eod_regime,
                                            replay_source_lane=eod_lane,
                                            replay_setup=last_s,
                                            required_stop_distance=abs(spot_stop_dist * sim_multiplier) if eod_uts else 0.0,
                                            chart_exit_reason=eod_chart_reason,
                                        )
                                        if actual_eod:
                                            entry_premium = actual_eod["entry"]
                                            if eod_uts:
                                                observed_entry_time = actual_eod["entry_time"]
                                                e_time = (IST.localize(observed_entry_time)
                                                          if observed_entry_time.tzinfo is None
                                                          else observed_entry_time)
                                            opt_sl_price = actual_eod["stop"]
                                            exit_premium = actual_eod["exit"]
                                            eod_target = actual_eod["target"]
                                            eod_exit_time = actual_eod["exit_time"]
                                            ex_time = IST.localize(eod_exit_time) if (eod_exit_time and getattr(eod_exit_time, "tzinfo", None) is None) else eod_exit_time
                                            eod_exit_reason = actual_eod["reason"] if actual_eod["reason"] != "HISTORICAL" else "SESSION_END"
                                        else:
                                            if eod_uts:
                                                continue
                                            fallback_risk = max(0.5, est_entry - opt_sl_price)
                                            eod_target = est_entry + (fallback_risk * eod_rr)
                                    else:
                                        if eod_uts:
                                            continue
                                        fallback_risk = max(0.5, entry_premium - opt_sl_price)
                                        eod_target = entry_premium + (fallback_risk * eod_rr)

                                # Check the same structured quality policy as live/recovery.
                                eod_trend_15m_agrees = self._trend_15m_agrees_at(
                                    mtf_result,
                                    last_s.timestamp,
                                    last_s.signal_type,
                                )
                                eod_structural_impulse = bool(
                                    tf == "5min"
                                    and self._is_5min_impulse_reversal(
                                        last_s,
                                        mtf_result,
                                        eod_conf,
                                        settings,
                                    )
                                )
                                eod_quality_verdict = self._quality_verdict_for_signal(
                                    last_s,
                                    tf=tf,
                                    grade=eod_grade,
                                    confidence=eod_conf,
                                    regime=eod_regime,
                                    inst_type=sim_inst,
                                    settings=settings,
                                    trend_15m_agrees=eod_trend_15m_agrees,
                                    source_count=max(1, len(eod_source_engines)),
                                    structural_impulse=eod_structural_impulse,
                                )
                                if not eod_quality_verdict.allowed:
                                    logger.info(f"⭐️ Skipping EOD {instrument} {tf} signal due to quality gate at {last_s.timestamp}")
                                    continue
                                eod_context = self._replay_entry_context(
                                    scanner, instrument, tf, last_s,
                                    "LONG" if str(getattr(last_s, "signal_type", "")).upper() == "BUY" else "SHORT",
                                    eod_regime, eod_lane)
                                if eod_context is not None and not eod_context.allowed:
                                    logger.info(f"Skipping EOD {instrument} {tf} signal: context gate {eod_context.rule_id}")
                                    continue

                                eod_lots = getattr(scanner, "user_lots_fut", {}).get(instrument, 1) if sim_inst == "FUT" else getattr(scanner, "user_lots", {}).get(instrument, 1)
                                if historical_final_uts:
                                    size_result = resolve_trade_lots(
                                        settings=settings, inst_type=sim_inst,
                                        requested_lots=int(eod_lots or 0), lot_size=lot_size,
                                        entry_price=float(entry_premium or 0.0),
                                        stop_price=float(opt_sl_price if sim_inst == "OPT" else eod_stop),
                                        compression_state=self._historical_compression_state(
                                            candles_df, last_s.timestamp, last_s.signal_type
                                        ),
                                        intel_score=float(eod_intel_score or 0.0) if sim_inst == "OPT" else 0.0,
                                    )
                                    eod_lots = int(size_result.lots or 0)
                                    if eod_lots < 1:
                                        continue
                                    if actual_eod and eod_probe_qty != eod_lots * lot_size:
                                        # UTS premium exits do not depend on quantity.
                                        # Keep the observed path's rupee fields aligned
                                        # with the same sizing rule as paired entries.
                                        quantity_ratio = eod_lots * lot_size / eod_probe_qty
                                        for sample in actual_eod.get("price_path") or []:
                                            for value in ("pnl", "peak_pnl"):
                                                if value in sample:
                                                    sample[value] *= quantity_ratio
                                if historical_final_uts and sim_inst == "FUT":
                                    minute_candles = scanner.candles.get_candles(instrument, "1min")
                                    if minute_candles is None or minute_candles.empty:
                                        continue
                                    exit_window = minute_candles.loc[
                                        (minute_candles.index >= eod_start)
                                        & (minute_candles.index + pd.Timedelta(minutes=1) <= eod_exit_time)
                                    ]
                                    if exit_window.empty:
                                        continue
                                    exit_premium = float(exit_window.iloc[-1]["close"])
                                    sim_price, sim_reason, sim_ts = self._simulate_futures_exit_path(
                                        exit_window, entry_premium, eod_stop, eod_target, direction,
                                    )
                                    if sim_reason:
                                        exit_premium, eod_exit_reason = float(sim_price), sim_reason
                                        eod_exit_time = sim_ts
                                        ex_time = IST.localize(sim_ts) if sim_ts.tzinfo is None else sim_ts
                                    elif eod_chart_reason:
                                        eod_exit_reason = uts_lane_policy.EXIT_REASONS.get(eod_chart_reason, "UTS_EXIT")
                                sim_charges = estimate_trade_charges(
                                    last_s.price if sim_inst == "FUT" else entry_premium,
                                    exit_premium,
                                    eod_lots * lot_size,
                                    sim_inst,
                                    settings,
                                )
                                if sim_inst == "FUT":
                                    if not historical_final_uts and direction == "LONG" and eod_price <= eod_stop:
                                        exit_premium = eod_stop
                                        eod_exit_reason = "SL HIT"
                                    elif not historical_final_uts and direction == "SHORT" and eod_price >= eod_stop:
                                        exit_premium = eod_stop
                                        eod_exit_reason = "SL HIT"
                                    if direction == "LONG":
                                        eod_pnl = (exit_premium - last_s.price) * (eod_lots * lot_size)
                                    else:
                                        eod_pnl = (last_s.price - exit_premium) * (eod_lots * lot_size)
                                    eod_pnl -= sim_charges
                                else:
                                    eod_pnl = (exit_premium - entry_premium) * (eod_lots * lot_size)
                                    eod_pnl -= sim_charges

                                t = Trade(
                                    id=eod_id, instrument=instrument, timeframe=tf,
                                    direction=direction, entry_price=entry_premium, entry_time=e_time,
                                    trailing_stop=(opt_sl_price if sim_inst == "OPT" else eod_stop),
                                    current_stop=(opt_sl_price if sim_inst == "OPT" else eod_stop),
                                    lots=eod_lots, lot_size=lot_size, grade=f"{eod_grade} (EOD, Hist)", confidence=eod_conf,
                                    inst_type=sim_inst, option_type=sim_option_type,
                                    rr_ratio=(uts_lane_policy.target_rr(settings) if eod_uts and sim_inst == "OPT"
                                              else eod_rr if eod_uts else 1.5),
                                    target=eod_target,
                                    status="CLOSED", exit_price=exit_premium, exit_time=ex_time,
                                    pnl=eod_pnl, charges=sim_charges, exit_reason=eod_exit_reason,
                                    instrument_multiplier=sim_multiplier, entry_spot=last_s.price,
                                    spot_stop=last_s.trailing_stop,
                                    spot_target=last_s.price + (abs(last_s.price - last_s.trailing_stop) * eod_rr) if direction == "LONG" else last_s.price - (abs(last_s.price - last_s.trailing_stop) * eod_rr),
                                    atm_strike=sim_atm_strike,
                                    trading_symbol=trading_symbol, symbol_token=symbol_token,
                                    source_engine=eod_source_engine,
                                    source_engines=eod_source_engines,
                                    source_lane=eod_lane,
                                    quality_verdict=self._with_entry_context(eod_quality_verdict.to_dict(), eod_context),
                                    data_provenance="BACKTEST",
                                    market_regime=str(eod_regime or "UNKNOWN"),
                                    entry_intelligence_score=float(eod_intel_score or 0.0),
                                    intelligence_score=float(eod_intel_score or 0.0),
                                    market_wide_reversal_entry=bool(getattr(last_s, "market_wide_reversal_entry", False)),
                                    volume_spike_entry=bool(getattr(last_s, "volume_spike_entry", False)),
                                    power_move=bool(getattr(last_s, "power_move", False)),
                                    quad_union_entry=bool(getattr(last_s, "quad_union_entry", False)),
                                    raw_union_entry=bool(getattr(last_s, "raw_union_entry", False)),
                                    initial_stop=float((opt_sl_price if sim_inst == "OPT" else eod_stop) or 0.0),
                                    signal_time=(getattr(last_s, "source_signal_timestamp", None)
                                                 or last_s.timestamp) if eod_uts else None,
                                    replay_price_path=list((actual_eod or {}).get("price_path") or []),
                                    replay_exit_trace=(self._historical_futures_fill_evidence(
                                        exit_window, signal=last_s, entry_time=eod_start,
                                        exit_time=eod_exit_time, entry_price=entry_premium,
                                        exit_price=exit_premium, exit_reason=eod_exit_reason,
                                    ) if historical_final_uts and sim_inst == "FUT" else
                                        list((actual_eod or {}).get("exit_trace") or [])),
                                )
                                live_session_day_raw = (
                                    scanner._get_current_session_day()
                                    if hasattr(scanner, "_get_current_session_day")
                                    else datetime.now(IST).date().isoformat()
                                )
                                live_session_day = datetime.fromisoformat(str(live_session_day_raw)).date()
                                if is_live and last_s.timestamp.date() == live_session_day:
                                    logger.info(
                                        f"Skipping simulated EOD backfill signal {eod_id} in live mode; "
                                        "live dashboard ledger only accepts real post-gate session rows."
                                    )
                                    continue
                                if is_live and last_s.timestamp.date() == live_session_day:
                                    exit_candidate = TradeCandidate(
                                        instrument=instrument,
                                        direction=direction,
                                        price=entry_premium,
                                        stop=(opt_sl_price if sim_inst == "OPT" else eod_stop),
                                        target=eod_target,
                                        lots=eod_lots,
                                        lot_size=lot_size,
                                        grade=eod_grade,
                                        confidence=eod_conf,
                                        timeframe=tf,
                                        inst_type=sim_inst,
                                        option_type=sim_option_type,
                                        atm_strike=sim_atm_strike,
                                        multiplier=sim_multiplier,
                                        trading_symbol=trading_symbol,
                                        symbol_token=symbol_token,
                                        rr=1.5,
                                        score=eod_conf,
                                        reasons=[f"Signal closed by {eod_exit_reason}"],
                                        spot_stop=last_s.trailing_stop,
                                        spot_target=(
                                            last_s.price + (abs(last_s.price - last_s.trailing_stop) * eod_rr)
                                            if direction == "LONG"
                                            else last_s.price - (abs(last_s.price - last_s.trailing_stop) * eod_rr)
                                        ),
                                        signal_timestamp=last_s.timestamp,
                                        current_price=exit_premium,
                                        pnl=eod_pnl,
                                        status="EXIT SIGNAL",
                                        action="EXIT",
                                        exit_reason=eod_exit_reason,
                                        source_engine=eod_source_engine,
                                        source_engines=eod_source_engines,
                                        source_lane=eod_lane,
                                    )
                                    setattr(exit_candidate, "exit_timestamp", eod_exit_time)
                                    setattr(exit_candidate, "exit_price", exit_premium)
                                    scanner._remember_trade_candidates([exit_candidate])
                                elif self._historical_trade_creation_allowed(scanner):
                                    scanner.trades.closed_trades.append(t)
                                    if sim_inst == "OPT" and self._hybrid_backtest_dual_futures_enabled(scanner):
                                        shadow_id = f"{eod_id}_FUT"
                                        if not any(ct.id == shadow_id for ct in scanner.trades.closed_trades):
                                            shadow = self._build_historical_futures_shadow_trade(
                                                scanner=scanner,
                                                trade_id=shadow_id,
                                                instrument=instrument,
                                                timeframe=tf,
                                                direction=direction,
                                                signal=last_s,
                                                exit_spot=eod_price,
                                                exit_time=eod_exit_time,
                                                grade=eod_grade,
                                                confidence=eod_conf,
                                                rr=1.5,
                                                lot_size=lot_size,
                                                display_suffix="EOD, Hist",
                                                default_exit_reason=eod_exit_reason,
                                                regime=eod_regime,
                                            )
                                            if shadow:
                                                scanner.trades.closed_trades.append(shadow)
                                else:
                                    logger.info(
                                        f"Blocked historical EOD trade creation outside HISTORICAL mode: {eod_id}"
                                    )

                                logger.debug(f"📊 Synthetic EOD close: {eod_id} | PnL: ₹{eod_pnl:,.0f}")

                # Update lock with precise last signal time
                # CRITICAL FIX: Use the SECOND-TO-LAST signal's timestamp
                # The LAST signal may be unpaired (no exit yet) — if we set lock
                # to its timestamp, the live detector won't see it as "new"
                # (because s_ts > l_ts fails when they're equal).
                if len(sigs) >= 2:
                    scanner._last_signal_time[dedup_key] = naive_ts(sigs[-2].timestamp)
                elif sigs:
                    scanner._last_signal_time[dedup_key] = naive_ts(sigs[-1].timestamp - pd.Timedelta(seconds=1))
                else:
                    scanner._last_signal_time[dedup_key] = naive_ts(datetime.now(IST)) if is_live else datetime.min
                
            last_time = naive_ts(scanner._last_signal_time[dedup_key])
            
            for sig in sigs:
                # strategy_direction_mode was only ever persisted and displayed, never
                # enforced, so LONG_ONLY / SHORT_ONLY did not actually restrict trading.
                # No-op under the default LONG_SHORT.
                _dir_ok, _dir_reason = (
                    (True, "1+1 native direction")
                    if scanner_engine_mode == "one_plus_one"
                    else self._direction_mode_allows(sig.signal_type, settings)
                )
                if not _dir_ok:
                    scanner.log_signal_decision_once(
                        f"{instrument}|{tf}|{sig.signal_type}|direction-mode",
                        f"Signal rejected: {instrument} {sig.signal_type} {tf} - {_dir_reason}",
                        "trade",
                    )
                    continue
                s_ts = naive_ts(sig.timestamp)
                l_ts = last_time
                identity_source = str(getattr(sig, "source_engine", "") or "").lower()
                signal_identity = f"{s_ts.isoformat()}|{sig.signal_type}|{identity_source}"
                last_identity = getattr(scanner, "_last_signal_identity", {}).get(dedup_key, "")
                now_naive = naive_ts(datetime.now(IST))
                tf_minutes = {"1min": 1, "5min": 5, "15min": 15}.get(tf, 5)
                pending_key = (
                    f"{instrument}_{tf}_"
                    f"{'LONG' if sig.signal_type == 'BUY' else 'SHORT'}"
                    f"{('_' + identity_source) if scanner_engine_mode == 'one_plus_one' else ''}"
                )
                pending_key += scope_suffix(sig, settings)
                recheck_eligible = (
                    getattr(scanner, "_live_signal_recheck_eligible", {}).get(dedup_key)
                    == signal_identity
                )
                last_recheck = getattr(scanner, "_live_signal_recheck_at", {}).get(
                    dedup_key,
                    datetime.min,
                )
                signal_mode_getter = getattr(settings, "get_signal_fire_mode", None)
                intrabar_signals_enabled = (
                    signal_mode_getter() == "INTRACANDLE"
                    if callable(signal_mode_getter)
                    else bool(getattr(settings, "live_intrabar_signal_entries", True))
                )
                recheck_due = is_live and (intrabar_signals_enabled or recheck_eligible) and self._live_signal_recheck_due(
                    s_ts,
                    tf,
                    now_naive,
                    recheck_eligible,
                    pending_key in getattr(scanner, "_pending_live_signals", {}),
                    naive_ts(last_recheck),
                    float(getattr(settings, "live_signal_recheck_seconds", 5.0) or 5.0),
                )
                post_repaint_recheck_due = False
                post_repaint_recheck = getattr(scanner, "_post_repaint_recheck_due", None)
                if is_live and intrabar_signals_enabled and callable(post_repaint_recheck):
                    post_repaint_recheck_due = bool(
                        post_repaint_recheck(
                            instrument,
                            tf,
                            sig.signal_type,
                            s_ts,
                            now_naive,
                            pending_key in getattr(scanner, "_pending_live_signals", {}),
                        )
                    )

                is_new_signal = (
                    self._live_signal_is_new(s_ts, sig.signal_type, l_ts, last_identity)
                    if is_live
                    else s_ts > l_ts
                )
                is_new_signal = is_new_signal or recheck_due or post_repaint_recheck_due
                if is_new_signal:
                    if (
                        scanner_engine_mode == "one_plus_one"
                        and not bool(getattr(sig, "native_exit_only", False))
                        and self._native_entry_has_exit(sig, sigs)
                    ):
                        # The native state machine opened and closed this lifecycle
                        # before platform admission (including same-bar stop-outs).
                        # Consuming it here prevents a stale entry from reaching the
                        # 20-30s buffer after its owning scanner is already flat.
                        self._remember_live_signal_identity(
                            scanner, dedup_key, s_ts, signal_identity
                        )
                        getattr(scanner, "_live_signal_recheck_eligible", {}).pop(
                            dedup_key, None
                        )
                        continue
                    if not hasattr(scanner, "_live_signal_recheck_at"):
                        scanner._live_signal_recheck_at = {}
                    scanner._live_signal_recheck_at[dedup_key] = now_naive
                    # ── Ghost Signal Age Guard (live mode only) ──
                    # Even though this signal is "new" per the dedup lock, in
                    # live/real mode we must verify its candle is actually
                    # current.  A signal from a candle that closed minutes ago
                    # is a historical-backfill echo, not a live signal.
                    if is_live:
                        deadline_fn = getattr(scanner, "_live_signal_processing_deadline", None)
                        deadline = (
                            deadline_fn(s_ts, tf)
                            if callable(deadline_fn)
                            else s_ts + timedelta(minutes=tf_minutes, seconds=90)
                        )
                        if deadline is not None and now_naive > deadline:
                            logger.warning(
                                f"👻 [GHOST GUARD] Skipping stale backfill signal "
                                f"{instrument} {tf} at {s_ts} "
                                f"(closed_age={(now_naive - (s_ts + timedelta(minutes=tf_minutes))).total_seconds():.0f}s). "
                                f"Not a live signal."
                            )
                            emit_once = getattr(scanner, "log_signal_decision_once", None)
                            if callable(emit_once):
                                emit_once(
                                    "stale-backfill-summary",
                                    "Live guard ignored stale historical signal echo(s); details retained in the server log.",
                                    "data",
                                )
                            # Still advance the dedup lock so we don't re-check. Keep
                            # every identity at this timestamp: quad-prime can retain an
                            # opposite source-family arrow on the same bar.
                            self._remember_live_signal_identity(
                                scanner, dedup_key, s_ts, signal_identity
                            )
                            getattr(scanner, "_live_signal_recheck_eligible", {}).pop(dedup_key, None)
                            continue

                        candle_close_time = s_ts + timedelta(minutes=tf_minutes)
                        intrabar_enabled = intrabar_signals_enabled
                        if (
                            not intrabar_enabled
                            and scanner_engine_mode != "one_plus_one"
                            and not self._live_signal_candle_is_closed(s_ts, tf, now_naive)
                        ):
                            logger.debug(
                                f"[PARTIAL CANDLE] Deferring live signal processing for {instrument} {tf} at {s_ts}. "
                                f"Candle closes at {candle_close_time}."
                            )
                            emit_once = getattr(scanner, "log_signal_decision_once", None)
                            if callable(emit_once):
                                emit_once(
                                    f"{instrument}|{tf}|{sig.signal_type}|{s_ts.isoformat(timespec='seconds')}|partial{scope_suffix(sig, settings)}",
                                    f"Signal waiting: {instrument} {sig.signal_type} {tf} @ {sig.price:.2f}"
                                    f"{(' [' + scope_suffix(sig, settings)[1:] + ']') if scope_suffix(sig, settings) else ''}"
                                    f" - partial candle deferred until {candle_close_time.strftime('%H:%M:%S')}",
                                    "trade",
                                )
                            continue

                    lane_tag = self._signal_source_lane(sig, scanner_engine_mode)
                    if lane_tag:
                        no_entry_label = entry_window.last_entry(tf, settings)
                        is_after_no_entry = not entry_window.entry_allowed(
                            sig.timestamp,
                            tf,
                            settings,
                            now=datetime.now(IST),
                        )
                    else:
                        no_entry_label, no_entry_after = _entry_cutoff(settings, tf)
                        is_after_no_entry = (
                            tf in ["1min", "5min", "15min"]
                            and sig.timestamp.time() >= no_entry_after
                        )
                    volume_spike_metrics = None
                    if is_live and tf in {"1min", "5min"} and scanner_engine_mode != "one_plus_one":
                        volume_spike_metrics = scanner._detect_volume_spike_entry(
                            instrument,
                            tf,
                            sig.signal_type,
                            adx_value=float(getattr(sig, "adx_value", 0.0) or 0.0),
                            intel_score=float(intel_score or 0.0),
                            as_of=sig.timestamp,
                        )
                    setattr(sig, "volume_spike_entry", bool(volume_spike_metrics))
                    setattr(
                        sig,
                        "volume_spike_metrics",
                        dict(volume_spike_metrics or {}),
                    )
                    ts_str = sig.timestamp.strftime("%Y-%m-%d %H:%M:00")
                    vix_value = scanner._vix_data.get(ts_str)
                    if vix_value is None:
                        vix_value = scanner.data.get_ltp("NSE", "INDIAVIX", "99926017") or 15.0

                    if is_live:
                        raw_source = str(getattr(sig, "source_engine", "") or "").lower()
                        raw_key = (
                            f"{instrument}|{tf}|{sig.signal_type}|"
                            f"{s_ts.isoformat(timespec='seconds')}|{raw_source}"
                        )
                        seen_raw = getattr(scanner, "_raw_utbot_activity_keys", None)
                        if seen_raw is None:
                            scanner._raw_utbot_activity_keys = set()
                            seen_raw = scanner._raw_utbot_activity_keys
                        if raw_key not in seen_raw:
                            seen_raw.add(raw_key)
                            if len(seen_raw) > 500:
                                scanner._raw_utbot_activity_keys = set(list(seen_raw)[-250:])
                            # Keep no-spike 1min raw detections off the live tape: they
                            # cannot enter without its configured surge gate, so logging them only
                            # floods the tape.
                            suppress_1min_raw = self._suppress_1min_no_spike_tape(
                                tf,
                                getattr(sig, "volume_spike_entry", False),
                                settings,
                                native_burst=self._is_s12_native_burst(sig),
                            )
                            if not suppress_1min_raw:
                                scanner.log_event(
                                    f"UT1 raw {sig.signal_type} detected: {instrument} {tf} @ {sig.price:.2f}. Watching for repaint.",
                                    "trade",
                                )
                        
                    # Grade each candidate with its OWN timeframe's intelligence when
                    # per-timeframe grading is enabled; otherwise these resolve to the
                    # caller's 5-minute values and grading is unchanged.
                    if (
                        scanner_engine_mode != "one_plus_one"
                        and
                        _timeframe_contexts is not None
                        and _setting_enabled(
                            getattr(settings, "own_timeframe_intel_fail_closed", True), True
                        )
                        and tf not in _timeframe_contexts
                    ):
                        # An own-timeframe rule may wait for its own evidence; it may not
                        # quietly rename the caller's 5m bundle as 1m/15m intelligence.
                        try:
                            scanner.log_signal_decision_once(
                                f"{instrument}|{tf}|{sig.signal_type}|own-tf-intel-missing",
                                f"Signal held: {instrument} {tf} {sig.signal_type} has no fresh "
                                "own-timeframe intelligence; fail-closed grading is active.",
                                "trade",
                            )
                        except Exception:
                            pass
                        # This is a temporary data hold, not a processed/rejected
                        # signal. Keep its identity eligible until its own candle
                        # closes so a refreshed intelligence bundle can grade it.
                        deadline_fn = getattr(scanner, "_live_signal_processing_deadline", None)
                        retry_deadline = (
                            deadline_fn(s_ts, tf)
                            if callable(deadline_fn)
                            else s_ts + timedelta(minutes=tf_minutes, seconds=90)
                        )
                        if is_live and (retry_deadline is None or now_naive <= retry_deadline):
                            if not hasattr(scanner, "_live_signal_recheck_eligible"):
                                scanner._live_signal_recheck_eligible = {}
                            scanner._live_signal_recheck_eligible[dedup_key] = signal_identity
                        self._remember_live_signal_identity(
                            scanner, dedup_key, s_ts, signal_identity
                        )
                        continue

                    _tf_intel_result, _tf_intel_score, _tf_regime = self._candidate_timeframe_context(
                        tf, intel_result, intel_score, regime, _timeframe_contexts
                    )
                    # Telemetry: record WHICH context actually graded this candidate.
                    # Without it the per-timeframe grading flag cannot be A/B'd offline —
                    # the tracker showed the candidate's regime but not the one grading used.
                    try:
                        setattr(sig, "graded_timeframe", str(tf or ""))
                        setattr(sig, "graded_intel_score", float(_tf_intel_score or 0.0))
                        setattr(sig, "graded_regime", str(_tf_regime or ""))
                        # Kept off GRADING_TELEMETRY_ATTRS deliberately: the decision
                        # tape needs the scalar/pillars, not a duplicate full bundle.
                        # Candidate construction below consumes it directly and copies
                        # only the established compact context fields.
                        setattr(sig, "graded_intel_result", _tf_intel_result)
                    except Exception:
                        pass
                    _live_session_ctx = (
                        _tf_intel_result.get("session_context", {})
                        if isinstance(_tf_intel_result, dict)
                        else {}
                    )
                    grade, conf, reasons = scanner.signals._grade_signal(
                        sig,
                        mtf_result.confluence_score,
                        _tf_intel_score,
                    _tf_regime,
                    vix_value=vix_value,
                    day_type=str(
                        _live_session_ctx.get("effective_day_type")
                        or _live_session_ctx.get("day_type")
                        or ""
                    ),
                    day_type_confidence=float(
                        _live_session_ctx.get("phase_confidence")
                        if _live_session_ctx.get("phase_confidence") is not None
                        else (_live_session_ctx.get("confidence") or 0.0)
                    ),
                    )
                    candidates_raw.append({
                        "sig": sig, 
                        "tf": tf, 
                        "grade": grade, 
                        "confidence": conf,
                        "reasons": reasons,
                        "after_no_entry": is_after_no_entry,
                        "after_no_entry_label": no_entry_label,
                        "dedup_key": dedup_key,
                        "signal_identity": signal_identity,
                    })
                    
                    # Final selection owns consumption. Consuming alternatives here made
                    # a rejected/reversing candidate disappear before its recheck pass.
        return candidates_raw

    def process_best_signal(
        self, instrument, mtf_result, intel_result, intel_score, regime,
        lots, lot_size, cfg, spot, atm_strike, _one_plus_one_source="",
        _retry_state=None,
    ) -> List[TradeCandidate]:

        # Access Scanner properties via self.scanner
        scanner = self.scanner
        settings = get_settings()
        if configured_mode(settings) == "ut_ema":
            from engine.ut_ema_lane import build_candidates
            return build_candidates(self, instrument, mtf_result, cfg, lots, lot_size, settings)
        original_context = (intel_result, intel_score, regime)

        scanner_engine_mode = scanner_mode(settings)
        combined_mode = configured_mode(settings)
        if combined_mode in COMBINED_MODES and not _one_plus_one_source:
            from engine.combined_scanner import native_mtf_view
            candidates = []
            for mode in (COMBINED_MODES[combined_mode], 's1'):
                partition = native_mtf_view(mtf_result, mode)
                with native_policy(mode):
                    candidates.extend(self.process_best_signal(
                        instrument, partition, intel_result, intel_score, regime,
                        lots, lot_size, cfg, spot, atm_strike, _one_plus_one_source=mode))
            return candidates
        if scanner_engine_mode == "one_plus_one" and not _one_plus_one_source:
            # Do not make two native cores compete in the platform's best-signal
            # tournament. Each source gets an independent pass across 1m/5m/15m.
            sources = []
            for tf in ("1min", "5min", "15min"):
                result = getattr(mtf_result, f"results_{tf}", None) or {}
                for signal in result.get("signals", []) or []:
                    source = str(getattr(signal, "source_engine", "") or "").lower()
                    if source in {"apex_v57", "stock_master_v732"} and source not in sources:
                        sources.append(source)
            independent = []
            for source in sources:
                partition = copy.copy(mtf_result)
                for tf in ("1min", "5min", "15min"):
                    original = getattr(mtf_result, f"results_{tf}", None)
                    if original is None:
                        continue
                    filtered = dict(original)
                    filtered["signals"] = [
                        signal for signal in (original.get("signals", []) or [])
                        if str(getattr(signal, "source_engine", "") or "").lower() == source
                    ]
                    setattr(partition, f"results_{tf}", filtered)
                independent.extend(self.process_best_signal(
                    instrument, partition, intel_result, intel_score, regime,
                    lots, lot_size, cfg, spot, atm_strike,
                    _one_plus_one_source=source,
                ))
            return independent

        def naive_ts(value):
            if value is None:
                return datetime.min
            return value.replace(tzinfo=None) if getattr(value, "tzinfo", None) else value

        prepared = (_retry_state["prepared"] if _retry_state is not None
                    else self._prepare_signal_pipeline_inputs(instrument, settings))
        if prepared is None:
            return []

        # None unless per_timeframe_grading_enabled is set, in which case each candidate
        # is graded with its own timeframe's cached intelligence instead of the 5-minute one.
        _timeframe_contexts = prepared["timeframe_contexts"]

        # Collect NEW signals from the normal entry TFs. Live mode also watches
        # 1min when: (a) 1M spike scalping is ON, or (b) post-exit reversion is
        # ON and this index has an open reversion window (decoupled toggles).
        scanner_engine_mode = prepared["scanner_engine_mode"]
        candidates_raw = []
        tf_policy = prepared["timeframe_policy"]
        policy_timeframes = prepared["policy_timeframes"]
        candidate_timeframes = prepared["candidate_timeframes"]
        _live = prepared["live"]
        _vol_spike_on = prepared["volume_spike"]
        _one_min_scalp_on = prepared["one_min_scalp"]
        _reversion_on = prepared["reversion_enabled"]
        _reversion_window = prepared["reversion_window"]
        is_live = _live
        # A 1m-inclusive policy collects 1m rows while 1MSCALP is ON. A plain 1m UT
        # flip enters only under 1M alone (PRIMARY_1, 2026-09-26 user); every other
        # policy uses 1m rows for the burst/snapback/cascade/reclaim paths only.
        # 1M SPIKE path: volume_spike_entry + one_min_scalping both ON.
        # POST-WIN SNAPBACK path: reversion window alone is enough (decoupled —
        # must not require volume_spike_entry_enabled, or pure snapback dies).
        policy_allows_one_min_burst = "1min" in policy_timeframes and _one_min_scalp_on
        # 1m bursts/snapbacks are collected under EVERY timeframe policy; unless the
        # toggle is 1M alone they are the only 1m entries (2026-09-26 user).
        legacy_fast_one_min = _live and (
            (_vol_spike_on and _one_min_scalp_on) or bool(_reversion_window)
        )
        candidates_raw = list(_retry_state["remaining"]) if _retry_state is not None else self._collect_signal_candidates(
            instrument,
            mtf_result,
            intel_result,
            intel_score,
            regime,
            lots,
            lot_size,
            cfg,
            spot,
            atm_strike,
            settings,
            prepared,
            one_plus_one_source=_one_plus_one_source,
        )

        # ── No-Entry After Gate ──
        # Replay implementation lives in _collect_signal_candidates and calls:
        # self._historical_entry_cutoff_allows(s1, tf, hist_lane_tag, settings)
        # before `continue` on an ineligible historical signal.

        if _retry_state is None:
            self._append_adaptive_dual_apex_candidates(
                scanner=scanner,
                instrument=instrument,
                mtf_result=mtf_result,
                intel_result=intel_result,
                intel_score=intel_score,
                regime=regime,
                settings=settings,
                candidates_raw=candidates_raw,
            )

        # In HISTORICAL mode, we only run the backfill once and do not return live candidates
        if scanner.mode == "HISTORICAL":
            return []

        if not candidates_raw:
            return []

        def _emit_signal_decision(raw: Dict, reason_msg: str, status_msg: str) -> None:
            sig = raw["sig"]
            tf = raw["tf"]
            diagnostic_scope = scope_suffix(sig, settings)
            decision_key = (
                f"{instrument}|{tf}|{sig.signal_type}|"
                f"{sig.timestamp.isoformat(timespec='seconds')}|{status_msg}|{reason_msg}{diagnostic_scope}"
            )
            # Tape hygiene: a 1min signal that never cleared its basis-aware surge
            # first filter can't enter, so its rejections only flood the live tape.
            # Suppress them (debug file log only); reject diagnostics still count.
            if self._suppress_1min_no_spike_tape(
                tf,
                getattr(sig, "volume_spike_entry", False),
                settings,
                native_burst=self._is_s12_native_burst(sig),
            ):
                logger.debug(
                    f"Signal decision (1min no-spike, tape-suppressed): {instrument} "
                    f"{tf} {sig.signal_type} -> {status_msg}: {reason_msg}"
                )
                diag_inc_unique = getattr(scanner, "_diag_inc_unique", None)
                if callable(diag_inc_unique):
                    diag_inc_unique("rejects", reason_msg, f"raw|{decision_key}")
                return
            logger.info(f"Signal decision: {instrument} {tf} {sig.signal_type} -> {status_msg}: {reason_msg}")
            emit_once = getattr(scanner, "log_signal_decision_once", None)
            message = (
                f"Signal rejected: {instrument} {sig.signal_type} {tf} @ {sig.price:.2f}"
                f"{(' [' + diagnostic_scope[1:] + ']') if diagnostic_scope else ''} - "
                f"{reason_msg}"
            )
            if callable(emit_once):
                emit_once(decision_key, message, "trade")
            else:
                scanner.log_event(message, "trade")
                
            # The activity persistence path owns the unique diagnostic count.
            # Counting again here inflated one refusal on every gate recheck.
            # Bare scanner adapters with no persistence still get unique counts.
            if not callable(getattr(scanner, "_persist_rejected_signal_event", None)):
                diag_inc_unique = getattr(scanner, "_diag_inc_unique", None)
                if callable(diag_inc_unique):
                    diag_inc_unique("rejects", reason_msg, f"raw|{decision_key}")

        def _build_rejected_from_raw(raw: Dict, reason_msg: str, status_msg: str) -> List[TradeCandidate]:
            _emit_signal_decision(raw, reason_msg, status_msg)
            sig = raw["sig"]
            tf = raw["tf"]
            grade = raw["grade"]
            confidence = raw["confidence"]
            reasons = raw["reasons"]
            recheckable_status = self._recheckable_rejection_status(status_msg)
            if is_live and recheckable_status:
                now_check = naive_ts(datetime.now(IST))
                tf_minutes = {"1min": 1, "5min": 5, "15min": 15}.get(tf, 5)
                if now_check < naive_ts(sig.timestamp) + timedelta(minutes=tf_minutes):
                    if not hasattr(scanner, "_live_signal_recheck_eligible"):
                        scanner._live_signal_recheck_eligible = {}
                    scanner._live_signal_recheck_eligible[
                        raw.get("dedup_key", f"{instrument}_{tf}")
                    ] = raw.get(
                        "signal_identity",
                        f"{naive_ts(sig.timestamp).isoformat()}|{sig.signal_type}",
                    )
            direction = "LONG" if sig.signal_type == "BUY" else "SHORT"
            rej = TradeCandidate(
                instrument=instrument,
                direction=direction,
                price=sig.price,
                stop=sig.trailing_stop,
                target=sig.price,
                lots=0,
                lot_size=0,
                grade=grade,
                confidence=confidence,
                timeframe=tf,
                inst_type="FUT",
                option_type="",
                atm_strike=0.0,
                multiplier=1.0,
                trading_symbol="",
                symbol_token="",
                rr=0.0,
                score=confidence,
                reasons=list(reasons or []) + [reason_msg],
                signal_timestamp=sig.timestamp,
                status=status_msg,
                action="NO_ENTRY"
            )
            carry_grading_telemetry(rej, sig)
            src_engine = str(getattr(sig, "source_engine", "") or "").lower()
            src_engines = [
                str(src).lower()
                for src in (getattr(sig, "source_engines", None) or [])
                if str(src or "").strip()
            ]
            setattr(rej, "source_engine", src_engine)
            setattr(rej, "source_engines", src_engines)
            setattr(rej, "source_lane", self._signal_source_lane(sig, scanner_engine_mode))
            setattr(rej, "quad_union_entry", scanner_engine_mode in {"quad_union", "quad_prime"})
            setattr(rej, "raw_union_entry", scanner_engine_mode in {"quad_union", "quad_prime"})
            setattr(rej, "source_signal_type", sig.signal_type)
            setattr(rej, "source_signal_price", float(getattr(sig, "price", 0.0) or 0.0))
            setattr(rej, "adx_value", float(getattr(sig, "adx_value", 0.0) or 0.0))
            rejected_regime, rejected_intel_norm = self._graded_candidate_context(
                sig, regime, normalize_intelligence_score(intel_score)
            )
            setattr(rej, "market_regime", rejected_regime)
            setattr(rej, "intelligence_score", float(rejected_intel_norm))
            rejected_intel_result = self._graded_candidate_intel_result(sig, intel_result)
            if isinstance(rejected_intel_result, dict):
                setattr(
                    rej,
                    "market_intelligence_context",
                    {
                        "aggregate": dict(rejected_intel_result.get("aggregate", {}) or {}),
                        "regime": dict(rejected_intel_result.get("regime", {}) or {}),
                        "volume": dict(rejected_intel_result.get("volume", {}) or {}),
                        "order_flow": dict(rejected_intel_result.get("order_flow", {}) or {}),
                        "session_context": dict(rejected_intel_result.get("session_context", {}) or {}),
                        "anchor_regime": dict(rejected_intel_result.get("anchor_regime", {}) or {}),
                        "breadth_context": dict(rejected_intel_result.get("breadth_context", {}) or {}),
                        "top_down_context": dict(rejected_intel_result.get("top_down_context", {}) or {}),
                        "market_structure": dict(rejected_intel_result.get("market_structure", {}) or {}),
                        "aroon": dict(rejected_intel_result.get("aroon_context", {}) or {}),
                        "wavetrend": dict(rejected_intel_result.get("wavetrend", {}) or {}),
                    },
                )
            # Every refusal funnels through here, and all 15 early returns above the
            # accepted-candidate stamp used to reach rtracker with an empty revision and
            # an empty data_provenance -- 1,057 of 2,295 archived verdict rows, and the
            # exact set whose ADX also arrived as 0.0. Stamping here rather than at each
            # return means a new refusal path cannot reintroduce the gap.
            for _field, _value in build_identity(settings, scanner).items():
                setattr(rej, _field, _value)
            return [rej]

        exit_handled = False
        opposite_signal_raw = None
        for c in sorted(candidates_raw if _retry_state is None else [], key=lambda item: item["sig"].timestamp):
            if scanner._handle_opposite_signal_exit(instrument, c["sig"], c["tf"]):
                exit_handled = True
                opposite_signal_raw = c  # the signal whose flip forced the exit
            if (
                is_live
                and scanner_engine_mode == "one_plus_one"
                and bool(getattr(c.get("sig"), "native_exit_only", False))
            ):
                self._remember_live_signal_identity(
                    scanner,
                    str(c.get("dedup_key") or f"{instrument}_{c.get('tf', '')}"),
                    naive_ts(c["sig"].timestamp),
                    str(c.get("signal_identity") or ""),
                )
                getattr(scanner, "_live_signal_recheck_eligible", {}).pop(
                    str(c.get("dedup_key") or ""), None
                )

        # Live/manual mode must not flip into a fresh opposite entry in the same
        # index/timeframe until the existing signal/trade has been exited. We block
        # the entry THIS cycle but reject it with a recheck-eligible status against
        # the *triggering* signal, so the NEXT cycle re-evaluates it (existing trade
        # now gone) and the reversal entry can fire — otherwise the dedup lock would
        # advance past the signal and orphan the reversal forever.
        #
        # Exception: multi-index market LTP shock / market-wide reverse — force-exit
        # the old side and allow the reverse entry this cycle so late dumps/rallies
        # are not missed while waiting for the next scan.
        if (
            exit_handled
            and scanner_engine_mode != "one_plus_one"
            and getattr(scanner, "mode", "") != "HISTORICAL"
        ):
            blocked_raw = opposite_signal_raw or candidates_raw[-1]
            _shock_relax = False
            try:
                settings_for_shock = get_settings()
                if _setting_enabled(
                    getattr(settings_for_shock, "market_ltp_shock_opposite_trade_relax", True),
                    True,
                ):
                    _flip_sig = (blocked_raw or {}).get("sig")
                    _flip_tf = str((blocked_raw or {}).get("tf") or "")
                    if _flip_sig is not None and (
                        bool(getattr(_flip_sig, "market_ltp_shock_entry", False))
                        or bool(getattr(_flip_sig, "market_wide_reversal_entry", False))
                    ):
                        _shock_relax = True
                    elif _flip_sig is not None and callable(
                        getattr(scanner, "_evaluate_market_ltp_shock", None)
                    ):
                        _shock_ctx = scanner._evaluate_market_ltp_shock(
                            instrument,
                            str(getattr(_flip_sig, "signal_type", "") or ""),
                            getattr(_flip_sig, "timestamp", None),
                            timeframe=_flip_tf,
                            setup_confidence=float(
                                getattr(_flip_sig, "graded_confidence", 0.0) or 0.0
                            ),
                            setup_adx=float(getattr(_flip_sig, "adx_value", 0.0) or 0.0),
                            setup_intel_score=float(
                                getattr(_flip_sig, "graded_intel_score", 0.0) or 0.0
                            ),
                            trend_15m_agrees=bool(
                                getattr(_flip_sig, "trend_15m_agrees", False)
                            ),
                        )
                        if _shock_ctx:
                            setattr(_flip_sig, "market_ltp_shock_entry", True)
                            setattr(_flip_sig, "market_ltp_shock_context", _shock_ctx)
                            _shock_relax = True
                    if not _shock_relax and _flip_sig is not None:
                        _mwr = getattr(_flip_sig, "market_wide_reversal_context", None)
                        if not _mwr and callable(
                            getattr(scanner, "_market_wide_reversal_context", None)
                        ):
                            _mwr = scanner._market_wide_reversal_context(
                                instrument,
                                _flip_tf,
                                str(getattr(_flip_sig, "signal_type", "") or ""),
                                getattr(_flip_sig, "timestamp", None),
                            )
                            if _mwr:
                                setattr(_flip_sig, "market_wide_reversal_entry", True)
                                setattr(_flip_sig, "market_wide_reversal_context", _mwr)
                                _shock_relax = True
            except Exception as _shock_exc:
                logger.debug(f"Opposite-trade shock relax skipped for {instrument}: {_shock_exc}")
                _shock_relax = False
            if _shock_relax:
                scanner.log_event(
                    f"Opposite-trade relax: {instrument} market-wide shock allows reverse entry this cycle",
                    "trade",
                )
            else:
                scanner.log_event(
                    f"Entry blocked: {instrument} opposite signal first requires the existing {blocked_raw['tf']} signal/trade to exit",
                    "trade",
                )
                return _build_rejected_from_raw(blocked_raw, "Opposite signal must exit existing signal/trade first", "REJECTED - Opposite Signal Wait")

        if getattr(scanner, "mode", "") != "HISTORICAL":
            # UT SMART EXIT reverses only on a CLOSED candle (2026-09-26 user). A flip on the
            # forming candle against an open trade is held as a recheck, not refused: the
            # refusal ("Already has active ... trade") consumed the arrow, so the close could
            # neither reverse the trade nor enter the new side.
            _has_opposite = getattr(scanner, "_has_opposite_exposure", None)
            for _raw in sorted(candidates_raw, key=lambda item: item["sig"].timestamp, reverse=True):
                _sig = _raw["sig"]
                if scanner_engine_mode == "one_plus_one":
                    index_stop_distance = abs(
                        float(best_sig.price)
                        - float(getattr(best_sig, "native_stop", best_sig.trailing_stop) or best_sig.trailing_stop)
                    )
                    effective_fut_sl_pct = (
                        index_stop_distance / max(float(best_sig.price), 1e-9) * 100.0
                    )
                elif (
                    getattr(_sig, "uts_forming", False)
                    and uts_lane_policy.is_uts(_sig, settings)
                    and callable(_has_opposite)
                    and _has_opposite(instrument, _sig, _raw["tf"])
                ):
                    return _build_rejected_from_raw(
                        _raw, "UT SMART EXIT reverses on the candle close", "REJECTED - Opposite Signal Wait"
                    )

        # 2026-09-26 user: a plain 1m signal enters only when the TF toggle is 1M alone;
        # every other policy (1M+5M, 1M+15M, 1M+5M+15M included) takes 1m bursts,
        # snapbacks, cascades and reclaims only.
        plain_one_min = tf_policy == "PRIMARY_1"
        if not plain_one_min:
            self._tag_one_min_fast_contexts(scanner, instrument, candidates_raw)
        entry_candidates_raw = self._entry_admissible_raw_candidates(
            candidates_raw,
            reversion_window_open=bool(_reversion_window),
            plain_one_min=plain_one_min,
        )
        if not entry_candidates_raw:
            return []
        # All explicit policies apply to every engine and historical replay. Legacy
        # 15M retains Quad Prime's established raw multi-source exception; old 15M /
        # 5M+15M modes also preserve established fast 1m spike/snapback routes.
        strict_timeframes = bool(getattr(settings, "strict_entry_timeframes", False))
        legacy_fast_one_min_policy = not strict_timeframes
        legacy_quad_prime_bypass = (
            not strict_timeframes and tf_policy == "PRIMARY_15" and scanner_engine_mode == "quad_prime"
        )
        if not legacy_quad_prime_bypass and (strict_timeframes or scanner_engine_mode != "one_plus_one"):
            entry_candidates_raw = [
                c
                for c in entry_candidates_raw
                if str(c.get("tf") or "") in policy_timeframes
                or (
                    legacy_fast_one_min_policy
                    and str(c.get("tf") or "") == "1min"
                    and (
                        self._has_1min_burst_or_spike(c.get("sig"))
                        or bool(getattr(c.get("sig"), "late_session_reclaim_entry", False))
                        or bool(getattr(c.get("sig"), "one_way_cascade_entry", False))
                        or bool(_reversion_window)
                    )
                )
            ]
        if not entry_candidates_raw:
            selected = "+".join(sorted(policy_timeframes))
            scanner.log_event(
                f"Entry blocked: {instrument} has no candidate inside {tf_policy} ({selected})",
                "trade",
            )
            return _build_rejected_from_raw(
                candidates_raw[-1],
                f"No signal inside selected timeframe policy {tf_policy}",
                "REJECTED - Timeframe Policy Blocked",
            )

        scored = rank_entry_candidates(
            entry_candidates_raw,
            lambda signal: self._signal_source_lane(signal, scanner_engine_mode),
        )
        best_score, best_sig, best_tf, best_grade, best_conf, best_reasons, best_after_no_entry = scored[0]
        # All decisions for the selected setup must consume the evidence that graded
        # it. Previously only the final candidate stamp used this context, after
        # routing, FUT veto, stop sizing and scalp classification had read the caller's
        # 5m bundle. Explicit 15m confirmation and session-wide census remain separate.
        regime, intel_score = self._graded_candidate_context(
            best_sig, regime, normalize_intelligence_score(intel_score)
        )
        intel_result = self._graded_candidate_intel_result(best_sig, intel_result)
        selected_raw = next(
            (
                raw for raw in entry_candidates_raw
                if raw.get("sig") is best_sig and str(raw.get("tf") or "") == str(best_tf or "")
            ),
            None,
        )
        if is_live and selected_raw is not None:
            selected_dedup_key = str(
                selected_raw.get("dedup_key") or f"{instrument}_{best_tf}"
            )
            self._remember_live_signal_identity(
                scanner,
                selected_dedup_key,
                naive_ts(best_sig.timestamp),
                str(selected_raw.get("signal_identity") or ""),
            )
            getattr(scanner, "_live_signal_recheck_eligible", {}).pop(
                selected_dedup_key, None
            )
        best_source_engine = str(getattr(best_sig, "source_engine", "") or "").lower()
        best_source_engines = [
            str(src).lower()
            for src in (getattr(best_sig, "source_engines", None) or [])
            if str(src or "").strip()
        ]
        if not best_source_engines and best_source_engine:
            if best_source_engine == "both":
                best_source_engines = ["utbot", "partner"]
            elif best_source_engine == "multi":
                best_source_engines = ["multi"]
            else:
                best_source_engines = [best_source_engine]
        best_source_lane = self._signal_source_lane(best_sig, scanner_engine_mode)
        quad_union_entry = scanner_engine_mode in {"quad_union", "quad_prime"}
        quad_prime_entry = scanner_engine_mode == "quad_prime"
        best_direction = normalized_direction(best_sig.signal_type)
        # S1 lane: the user's S1 script already selected this signal with its own five
        # gates, so the other engines' SELECTION filters below are skipped for it. Safety
        # gates (loss breakers, caps, cutoff, quotes, sizing) still run.
        # "S1" / "UTS" / "" (truthy exactly for a lane signal); UT SMART EXIT shares the
        # lane's admission and exits, with its own stop, target, guard and cutoffs.
        s1_lane = s1_lane_policy.lane_tag(best_sig, settings)
        uts_lane_on = s1_lane == "UTS"
        s12_lane_on = s1_lane in {"D1", "D2"}

        # ── DIAGNOSTIC: 5min-vs-15min selection (debug TF misdetection) ──
        # Fires live whenever more than one TF competed or 15min won, so the next
        # "5min real / 15min phantom" event is captured with full context.
        try:
            _now_dbg = naive_ts(datetime.now(IST))
            _admitted = []
            for _sc, _sg, _t, _gr, _cf, _r, _an in scored:
                _ts = naive_ts(_sg.timestamp)
                _closed = self._live_signal_candle_is_closed(_ts, _t, _now_dbg)
                _admitted.append(
                    f"{_t}@{_ts.strftime('%H:%M')}/{_sg.signal_type}/{_gr}/{_cf:.2f}/"
                    f"{'closed' if _closed else 'FORMING'}/score={_sc:.4f}"
                )
            _raw = []
            for _t, _res in (
                ("1min", getattr(mtf_result, "results_1min", None)),
                ("5min", mtf_result.results_5min),
                ("15min", mtf_result.results_15min),
            ):
                _rs = [s for s in (_res or {}).get("signals", []) if s.instrument == instrument]
                _latest = (
                    f"{naive_ts(_rs[-1].timestamp).strftime('%m-%d %H:%M')}/{_rs[-1].signal_type}"
                    if _rs else "none"
                )
                _raw.append(f"{_t}:raw={len(_rs)}(latest {_latest})")
            _open_trades = getattr(getattr(scanner, "trades", None), "open_trades", {}) or {}
            _trade_involved = any(
                getattr(_trade, "instrument", None) == instrument
                and getattr(_trade, "status", None) == "OPEN"
                and self._is_live_book_trade(_trade)
                for _trade in _open_trades.values()
            )
            if self._should_log_tf_select_diagnostic(
                best_tf,
                best_sig,
                scored,
                settings,
                trade_involved=_trade_involved,
            ):
                scanner.log_event(
                    f"🧪 [TF-SELECT] {instrument}: chose {best_tf}@"
                    f"{naive_ts(best_sig.timestamp).strftime('%H:%M')} | "
                    f"admitted=[{' ; '.join(_admitted)}] | {' '.join(_raw)}",
                    "trade",
                )
        except Exception as _dbg_exc:
            logger.debug(f"TF-select diagnostic failed for {instrument}: {_dbg_exc}")

        def _build_rejected_candidate(reason_msg: str, status_msg: str) -> List[TradeCandidate]:
            rejected = _build_rejected_from_raw(
                {
                    "sig": best_sig,
                    "tf": best_tf,
                    "grade": best_grade,
                    "confidence": best_conf,
                    "reasons": best_reasons,
                },
                reason_msg,
                status_msg,
            )
            # Only candidate-specific refusal permits trying another setup. Book
            # capacity, capital, circuit breakers and opposite exits stop the wave.
            retryable = {
                "REJECTED - Regime Veto", "REJECTED - Session Trend Guard",
                "REJECTED - Pre-Candle Flip Veto", "REJECTED - Counter Census",
                "REJECTED - Price Action", "REJECTED - FUT Stop Distance",
                "REJECTED - Weak 1min", "REJECTED - Monopoly",
            }
            remaining = [row for row in entry_candidates_raw if row["sig"] is not best_sig]
            if status_msg in retryable and remaining:
                alternatives = self.process_best_signal(
                    instrument, mtf_result, *original_context, lots, lot_size,
                    cfg, spot, atm_strike, _one_plus_one_source=_one_plus_one_source,
                    _retry_state={"prepared": prepared, "remaining": remaining},
                )
                if alternatives:
                    return rejected + alternatives
            return rejected

        for tid, trade in list(scanner.trades.open_trades.items()):
            if not same_owner(trade, best_sig, settings):
                continue
            # Ghost / hist / replay rows never block live opposite/same-side guards.
            if not self._is_live_book_trade(trade):
                continue
            if trade.instrument == instrument and trade.status == "OPEN":
                if scanner_engine_mode == "one_plus_one":
                    # Each native core owns a separate lifecycle.  An APEX position
                    # cannot absorb, upgrade, or reject a Stock Master entry (and vice
                    # versa); only a still-open position from the same native source
                    # suppresses another entry from that source.
                    trade_source = str(getattr(trade, "source_engine", "") or "").lower()
                    candidate_source = str(best_source_engine or "").lower()
                    if trade_source != candidate_source:
                        continue
                    _emit_signal_decision(
                        {
                            "sig": best_sig,
                            "tf": best_tf,
                            "grade": best_grade,
                            "confidence": best_conf,
                            "reasons": best_reasons,
                        },
                        f"Native source {candidate_source} already has active "
                        f"{trade.timeframe} {trade.direction} trade {tid}",
                        "REJECTED - Existing Native Source Trade",
                    )
                    return []
                if scanner_engine_mode == "adaptive_dual" and best_source_lane == "B":
                    probe = SimpleNamespace(
                        instrument=instrument,
                        timeframe=best_tf,
                        direction=best_direction,
                        grade=best_grade,
                        confidence=best_conf,
                        lots=1,
                        lot_size=1,
                        price=float(getattr(best_sig, "price", 0.0) or 0.0),
                        stop=float(getattr(best_sig, "trailing_stop", 0.0) or 0.0),
                        inst_type="FUT",
                        source_engine=best_source_engine,
                        source_engines=list(best_source_engines),
                        source_lane=best_source_lane,
                        signal_timestamp=getattr(best_sig, "timestamp", None),
                        source_signal_bar_index=int(getattr(best_sig, "bar_index", -1) or -1),
                        volume_spike_entry=bool(getattr(best_sig, "volume_spike_entry", False)),
                        power_move=bool(getattr(best_sig, "power_move", False)),
                        market_wide_reversal_entry=bool(getattr(best_sig, "market_wide_reversal_entry", False)),
                        vwap_break_context=getattr(best_sig, "vwap_break_context", None),
                        reasons=list(best_reasons or []) + list(getattr(best_sig, "apex_reasons", []) or []),
                    )
                    lane_decision = getattr(scanner, "_dual_lane_concurrency_decision", lambda *a, **k: (None, ""))(
                        probe,
                        [trade],
                        adjust_size=False,
                    )
                    if isinstance(lane_decision, tuple) and lane_decision[0] is not None:
                        allowed, lane_reason = lane_decision
                        if allowed:
                            best_reasons = list(best_reasons or []) + [f"Adaptive Dual Lane B active-trade pass: {lane_reason}"]
                            continue
                        _emit_signal_decision(
                            {
                                "sig": best_sig,
                                "tf": best_tf,
                                "grade": best_grade,
                                "confidence": best_conf,
                                "reasons": best_reasons,
                            },
                            f"Adaptive Dual Lane B blocked: {lane_reason}",
                            "REJECTED - Dual Lane Guard",
                        )
                        return []
                if quad_prime_entry:
                    trade_direction = str(getattr(trade, "direction", "") or "").upper()
                    if trade_direction != best_direction:
                        # _build_rejected_from_raw, not the bare emitter: this is a FLIP
                        # being refused only because the far-side trade has not closed
                        # yet, so it must stay recheck-eligible for the cycle after it
                        # does. The bare _emit_signal_decision only logs and counts, and
                        # the dedup lock (advanced when the raw signal was collected)
                        # then orphans the arrow permanently. The sibling refusal at
                        # "REJECTED - Opposite Signal Wait" has always done it this way.
                        # dedup_key/signal_identity are what the recheck is armed under,
                        # so they are carried through exactly as that path carries them.
                        _best_raw = next(
                            (
                                c for c in entry_candidates_raw
                                if c.get("sig") is best_sig
                            ),
                            {},
                        )
                        return _build_rejected_from_raw(
                            {
                                "sig": best_sig,
                                "tf": best_tf,
                                "grade": best_grade,
                                "confidence": best_conf,
                                "reasons": best_reasons,
                                "dedup_key": _best_raw.get(
                                    "dedup_key", f"{instrument}_{best_tf}"
                                ),
                                "signal_identity": _best_raw.get("signal_identity"),
                            },
                            f"Opposite {getattr(trade, 'timeframe', '')} {trade_direction} trade {tid} still active",
                            "REJECTED - Existing Opposite Trade",
                        )

                    candidate_sources = scanner._quad_union_sources(best_sig)
                    trade_sources = scanner._quad_union_sources(trade)
                    same_tf = str(getattr(trade, "timeframe", "") or "") == best_tf
                    source_overlaps = (
                        not candidate_sources
                        or not trade_sources
                        or bool(candidate_sources & trade_sources)
                    )
                    if same_tf and source_overlaps:
                        # THIS is the cascade confirming itself: the same engine firing
                        # again on the same chart, which is the commonest shape of the
                        # operator's add-a-lot rule. It was discarded here, upstream of
                        # the only place the add is offered -- _offer_cascade_add is
                        # reached from _coordinate_and_execute, and a candidate that
                        # never leaves process_best_signal never gets there. So the rule
                        # produced nothing at all in its most likely case.
                        #
                        # Gated on auto_mode for the same reason as the scanner-side call
                        # site: scale_in_trade reaches broker.place_order, and manual mode
                        # means the operator approves every entry.
                        add_note = ""
                        adder = getattr(scanner, "_offer_cascade_add", None)
                        if callable(adder) and bool(getattr(scanner, "auto_mode", False)):
                            try:
                                added, add_note = adder(best_sig)
                                if added:
                                    _emit_signal_decision(
                                        {
                                            "sig": best_sig,
                                            "tf": best_tf,
                                            "grade": best_grade,
                                            "confidence": best_conf,
                                            "reasons": best_reasons,
                                        },
                                        f"Cascade add: one more lot on {tid} instead of a "
                                        f"second {best_tf} trade",
                                        "CASCADE_ADD",
                                    )
                                    return []
                            except Exception as exc:
                                logger.debug(f"Cascade add offer failed for {instrument}: {exc}")
                        _emit_signal_decision(
                            {
                                "sig": best_sig,
                                "tf": best_tf,
                                "grade": best_grade,
                                "confidence": best_conf,
                                "reasons": best_reasons,
                            },
                            f"Already has active quad-prime {best_tf} {trade_direction} source {tid}"
                            + (f"; cascade add declined ({add_note})" if add_note else ""),
                            "REJECTED - Existing Active Trade",
                        )
                        return []
                    continue

                if best_tf == "15min" and trade.timeframe == "5min" and not s1_lane_policy.is_managed_lane(trade, settings):
                    trade.timeframe = "15min"
                    trade.grade = best_grade
                    trade.confidence = best_conf
                    logger.info(f"🚀 UPGRADED {instrument} trade to 15-min status")
                    return []
                if s1_lane and str(getattr(trade, "direction", "") or "").upper() != best_direction:
                    # A lane FLIP (S1 / UT SMART EXIT) meeting the trade it reverses: the exit
                    # order is out (EXIT_PENDING) but not filled when the next scan builds the
                    # new side. The bare refusal below consumed the arrow, so a fill slower
                    # than one scan (2-4s live) lost the reversal entry. Keep it
                    # recheck-eligible, exactly as the quad_prime flip path above does.
                    _best_raw = next((c for c in entry_candidates_raw if c.get("sig") is best_sig), {})
                    _flip_key = str(_best_raw.get("dedup_key") or f"{instrument}_{best_tf}")
                    _flip_identity = _best_raw.get("signal_identity")
                    _flip_rejection = _build_rejected_from_raw(
                        {
                            "sig": best_sig,
                            "tf": best_tf,
                            "grade": best_grade,
                            "confidence": best_conf,
                            "reasons": best_reasons,
                            "dedup_key": _flip_key,
                            "signal_identity": _flip_identity,
                        },
                        f"Opposite {getattr(trade, 'timeframe', '')} {trade.direction} trade {tid} still closing",
                        "REJECTED - Existing Opposite Trade",
                    )
                    if is_live and _flip_identity:
                        # The recheck above only arms while the candle forms; a CLOSED-candle
                        # flip (every UT SMART EXIT flip) needs its dedup lock released so the
                        # next scan sees the arrow again once the old trade has filled out.
                        self._forget_live_signal_identity(
                            scanner, _flip_key, naive_ts(best_sig.timestamp), str(_flip_identity)
                        )
                    return _flip_rejection
                _emit_signal_decision(
                    {
                        "sig": best_sig,
                        "tf": best_tf,
                        "grade": best_grade,
                        "confidence": best_conf,
                        "reasons": best_reasons,
                    },
                    f"Already has active {trade.timeframe} {trade.direction} trade {tid}",
                    "REJECTED - Existing Active Trade",
                )
                return []

        # Do not call process_signal here. In live mode this stage can run
        # before the 25s repaint guard has matured, and process_signal mutates
        # active_signals/dedup state. Keep grading stateless; the execution
        # lifecycle starts only after stabilization in scanner._coordinate_and_execute.
        graded = GradedSignal(
            signal=best_sig,
            grade=best_grade,
            confidence=best_conf,
            reasons=list(best_reasons or []),
            intelligence_score=intel_score,
            confluence_score=mtf_result.confluence_score,
            is_actionable=best_grade in ["A+", "A", "B+", "B"],
        )
        state_15m = (getattr(mtf_result, "results_15min", None) or {}).get("state", {}) or {}
        # "The 15m engine has no state" and "the 15m trend disagrees" both collapse to
        # position 0, and 0 never equals +/-1, so both produce trend_15m_agrees=False.
        # In REAL mode that flag alone closes the entire 5min lane -- swept exhaustively,
        # 0 of 5,775 (confidence x ADX x intel x IV x premium) combinations pass with it
        # False, in every one of the seven regimes. 5min carries 181 of 252 recorded
        # entries, 72% of the book.
        #
        # The failure mode is not the refusal, which is the design intent -- a 5min entry
        # SHOULD be 15m-aligned. It is that a cold start, a 15m engine that failed to
        # build, or a lookup that raised is INVISIBLE: no decision row is written, so the
        # dashboard reads "no 5min signal today" rather than "the 15m state is missing".
        #
        # Admission is deliberately unchanged here. This only separates the two cases and
        # says the silent one out loud. A position of 0 with real state is a legitimate
        # "15m is flat", which is a genuine disagreement, not an absence.
        fifteen_min_available = bool(state_15m) and state_15m.get("position") is not None
        position_15m = int(state_15m.get("position") or 0)
        direction_position = 1 if best_sig.signal_type == "BUY" else -1
        trend_15m_agrees = position_15m == direction_position
        try:
            setattr(
                best_sig,
                "trend_15m_state",
                "agrees" if trend_15m_agrees
                else ("unavailable" if not fifteen_min_available else "disagrees"),
            )
            if not fifteen_min_available and scanner is not None:
                day = getattr(getattr(best_sig, "timestamp", None), "date", lambda: "")()
                scanner.log_signal_decision_once(
                    f"15m-state-unavailable|{instrument}|{best_tf}|{day}",
                    f"{instrument} {best_tf}: 15-minute state unavailable, so every "
                    f"5min entry for it is refused as 15m-misaligned. This is a missing "
                    f"engine or a cold start, not a disagreeing trend.",
                    "warning",
                )
        except Exception as exc:  # telemetry may never cost a signal
            logger.debug(f"15m availability note skipped for {instrument}: {exc}")
        adx_value = float(getattr(best_sig, "adx_value", 0.0) or 0.0)
        volume_spike_entry = bool(getattr(best_sig, "volume_spike_entry", False))
        s12_burst_entry = bool(s12_lane_on and self._is_s12_native_burst(best_sig))
        one_min_burst_or_spike_entry = bool(volume_spike_entry or s12_burst_entry)
        volume_spike_metrics = dict(getattr(best_sig, "volume_spike_metrics", {}) or {})
        market_ltp_shock_entry = bool(getattr(best_sig, "market_ltp_shock_entry", False))
        market_ltp_shock_context = dict(getattr(best_sig, "market_ltp_shock_context", {}) or {})
        if scanner_engine_mode != "one_plus_one" and not market_ltp_shock_entry:
            try:
                _shock_fn = getattr(scanner, "_evaluate_market_ltp_shock", None)
                if callable(_shock_fn):
                    _shock_ctx = _shock_fn(
                        instrument,
                        str(best_sig.signal_type),
                        best_sig.timestamp,
                        timeframe=best_tf,
                        setup_confidence=float(graded.confidence or 0.0),
                        setup_adx=float(adx_value or 0.0),
                        setup_intel_score=float(normalize_intelligence_score(intel_score)),
                        trend_15m_agrees=bool(trend_15m_agrees),
                    )
                    if _shock_ctx:
                        market_ltp_shock_entry = True
                        market_ltp_shock_context = dict(_shock_ctx)
                        setattr(best_sig, "market_ltp_shock_entry", True)
                        setattr(best_sig, "market_ltp_shock_context", market_ltp_shock_context)
                        best_reasons = list(best_reasons or []) + [
                            str(_shock_ctx.get("reason") or "market LTP shock")
                        ]
            except Exception as _shock_exc:
                logger.debug(f"Market LTP shock detect skipped for {instrument}: {_shock_exc}")

        # ── Session directional census ──
        # Count each DISTINCT directional arrow once so we know the day's bias.
        # Used by the counter-trend guard below. Keyed identity dedups the many
        # re-evaluations of the same forming candle within a scan cycle.
        signal_direction = best_direction
        self._record_directional_census(
            scanner, best_sig, instrument, best_tf, signal_direction,
            naive_ts(best_sig.timestamp).isoformat(), settings,
        )
        momentum_threshold = float(getattr(settings, "momentum_override_threshold", 25.0) or 25.0)
        intel_norm = normalize_intelligence_score(intel_score)
        # Data-confirmed late-session power move (volume + range + close + momentum/align).
        # When confirmed it may bypass the grade floor and (after 15:00) the no-entry
        # cutoff, entering in tight scalp-lock mode.
        # ``as_of`` matters: without it _detect_power_move takes df.tail(...).iloc[-1],
        # which live is the FORMING bar, so the volume surge, range expansion and close
        # location that define a power move were read off an unfinished candle that could
        # still reverse. The sibling thrust lane already pins its own bar --
        # _detect_volume_spike_entry is called with as_of=sig.timestamp -- and the
        # HISTORICAL reconstruction of this very detector passes as_of too. Only the live
        # power-move call omitted it, so live and replay disagreed about the same bar.
        power_move = scanner._detect_power_move(
            instrument,
            best_tf,
            best_sig.signal_type,
            adx_value=adx_value,
            intel_score=intel_score,
            trend_15m_agrees=trend_15m_agrees,
            as_of=getattr(best_sig, "timestamp", None),
        ) if scanner_engine_mode != "one_plus_one" else False
        # Market LTP shock is a multi-index fast dump/rally — treat like volume spike
        # for gate bypasses (trend guard, quality floor) on 1min/5min only.
        fast_thrust_override = bool(
            power_move or volume_spike_entry or market_ltp_shock_entry or s12_burst_entry
        )
        _reversal_relief_context = None
        try:
            _relief_fn = getattr(scanner, "_entry_reversal_relief_context", None)
            if scanner_engine_mode != "one_plus_one" and callable(_relief_fn):
                _reversal_relief_context = _relief_fn(
                    instrument,
                    best_tf,
                    best_sig.signal_type,
                    best_sig.timestamp,
                    regime=regime,
                    confidence=graded.confidence,
                )
        except Exception as exc:
            logger.debug(f"Reversal relief check skipped for {instrument} {best_tf}: {exc}")
        if not isinstance(_reversal_relief_context, dict) and best_tf == "1min":
            # A 1m row admitted on reclaim / cascade evidence before selection
            # (_tag_one_min_fast_contexts) keeps that evidence.
            for _flag in ("late_session_reclaim", "one_way_cascade"):
                _pre_ctx = getattr(best_sig, f"{_flag}_context", None)
                if getattr(best_sig, f"{_flag}_entry", False) and isinstance(_pre_ctx, dict):
                    _reversal_relief_context = dict(_pre_ctx)
                    _reversal_relief_context.setdefault("type", _flag)
                    break
        late_session_reclaim_entry = bool(
            isinstance(_reversal_relief_context, dict)
            and _reversal_relief_context.get("type") == "late_session_reclaim"
        )
        one_way_cascade_entry = bool(
            isinstance(_reversal_relief_context, dict)
            and _reversal_relief_context.get("type") == "one_way_cascade"
        )
        if late_session_reclaim_entry:
            try:
                setattr(best_sig, "late_session_reclaim_entry", True)
                setattr(best_sig, "late_session_reclaim_context", dict(_reversal_relief_context))
            except Exception:
                pass
            best_reasons = list(best_reasons or []) + [
                str(_reversal_relief_context.get("reason") or "late-session synchronized reclaim")
            ]
        if one_way_cascade_entry:
            try:
                setattr(best_sig, "one_way_cascade_entry", True)
                setattr(best_sig, "one_way_cascade_context", dict(_reversal_relief_context))
            except Exception:
                pass
            best_reasons = list(best_reasons or []) + [
                str(_reversal_relief_context.get("reason") or "one-way cascade")
            ]
        if uts_lane_policy.is_uts(best_sig, settings):
            count_aligned = getattr(scanner, "_count_uts_aligned_indices", None)
            if callable(count_aligned):
                best_sig.uts_multi_index_alignment = count_aligned(
                    "LONG" if best_sig.signal_type == "BUY" else "SHORT",
                    best_tf, include_self=instrument) >= 2
        if s1_lane:
            # 2026-09-24: an S1 SELL tagged as an LTP shock was refused five times by the
            # shock lane's 1-minute cash-volume demand while the S1 move ran, so that tag is
            # dropped. Volume-spike, cascade, reclaim and snapback stay (2026-09-25 user):
            # they admit 1m S1 bursts and label the trade; S1 keeps its own levels, exits,
            # cutoffs and sizing.
            market_ltp_shock_entry = False
            if getattr(best_sig, "market_ltp_shock_entry", False):
                setattr(best_sig, "market_ltp_shock_entry", False)
            fast_thrust_override = bool(power_move or volume_spike_entry or s12_burst_entry)
        # Kept apart from _reversal_relief_context on purpose: that context also
        # re-admits repainting arrows and late-session scalps, which the stale-regime
        # evidence never covered (it was measured on non-repainted signals only).
        _stale_regime_context = None
        try:
            _stale_fn = getattr(scanner, "_stale_regime_reversal_context", None)
            if scanner_engine_mode != "one_plus_one" and callable(_stale_fn):
                _stale_regime_context = _stale_fn(
                    instrument,
                    best_tf,
                    best_sig.signal_type,
                    best_sig.timestamp,
                    regime=self._graded_candidate_context(best_sig, regime, intel_norm)[0],
                )
        except Exception as exc:
            logger.debug(f"Stale-regime check skipped for {instrument} {best_tf}: {exc}")
        stale_regime_reversal_entry = bool(
            isinstance(_stale_regime_context, dict) and _stale_regime_context.get("enabled") is True
        )
        if stale_regime_reversal_entry:
            try:
                setattr(best_sig, "stale_regime_reversal_entry", True)
                setattr(best_sig, "stale_regime_reversal_context", dict(_stale_regime_context))
            except Exception:
                pass
            best_reasons = list(best_reasons or []) + [str(_stale_regime_context.get("reason"))]
        extreme_exhaustion = any(
            "extreme" in str(reason).lower() and "exhaustion" in str(reason).lower()
            for reason in (best_reasons or [])
        )
        impulse_confirmed = (
            best_tf == "5min"
            and self._is_5min_impulse_reversal(
                best_sig,
                mtf_result,
                graded.confidence,
                settings,
            )
        )
        is_explosive_bypass = bool(
            impulse_confirmed
            and (
                adx_value >= momentum_threshold
                or abs(intel_norm) >= 0.70
            )
        )
        if best_tf == "5min" and self._is_choppy_regime(regime) and extreme_exhaustion:
            exhaustion_follow_through = (
                trend_15m_agrees
                and impulse_confirmed
                and (best_score >= 0.45 or abs(intel_norm) >= 0.70 or graded.confidence >= 0.90)
            )
            is_explosive_bypass = bool(is_explosive_bypass and exhaustion_follow_through)
        impulse_context_confirmed = bool(impulse_confirmed or is_explosive_bypass)
        iv_percentile = intel_result.get("greeks", {}).get("iv_percentile")
        chain_quality = intel_result.get("data_quality", {}) or {}
        option_premium_ok = option_premium_health_ok(
            settings,
            iv_percentile=iv_percentile,
            chain_quality=chain_quality,
        )
        true_5min_impulse = best_tf == "5min" and five_min_true_impulse_allowed(
            graded.grade,
            graded.confidence,
            settings,
            trend_15m_agrees=trend_15m_agrees,
            impulse_confirmed=impulse_confirmed,
            explosive_bypass=is_explosive_bypass,
            extreme_exhaustion=extreme_exhaustion,
            regime=regime,
            adx_value=adx_value,
            intel_score=intel_score,
            iv_percentile=iv_percentile,
            chain_quality=chain_quality,
            option_premium_ok=option_premium_ok,
        )
        quality_hybrid_entry = False
        try:
            quality_hybrid_entry = bool(
                getattr(scanner, "_is_quality_hybrid_route_entry", lambda *a, **k: False)(
                    settings,
                    best_sig,
                    confidence=float(getattr(graded, "confidence", 0.0) or 0.0),
                    grade_rank=self._grade_rank(getattr(graded, "grade", "")),
                    timeframe=best_tf,
                    trend_15m_agrees=trend_15m_agrees,
                    explosive_move=bool(power_move or volume_spike_entry or is_explosive_bypass),
                )
            )
        except Exception:
            quality_hybrid_entry = False
        # Grade, confidence, regime, ADX and contemporaneous confirmations are
        # intentionally not admitted/rejected here.  The scanner's canonical verdict
        # evaluates the fully built candidate once, after this processor has attached all
        # evidence.  Keeping a second 5m/chop/dynamic-confidence ladder here caused valid
        # B+ setups to disappear before the authoritative gate could inspect them.

        # --- REVERSION / SNAPBACK eligibility (before Weak-1min gate) ---
        # Must run early: pure SNAPBACK has no volume_spike_entry and would die
        # at Weak-1min if tagged only later. SPIKE and SNAPBACK share the 1m
        # chassis but only ONE trade fires (best winner); dual-tag when both true.
        if scanner_engine_mode == "one_plus_one":
            is_reversion_scalp, _rev_meta = False, {}
        else:
            is_reversion_scalp, _rev_meta = self._evaluate_reversion_scalp(
                scanner,
                settings,
                instrument=instrument,
                best_tf=best_tf,
                best_sig=best_sig,
            )
        if is_reversion_scalp:
            logger.info(
                f"⚡ [REVERSION SCALP] {instrument} 1min {best_sig.signal_type} "
                f"eligible ({_rev_meta.get('minutes_since', 0):.1f}m after "
                f"{_rev_meta.get('view') or 'prior'} {_rev_meta.get('reason') or 'event'}; "
                f"move={_rev_meta.get('move_pct', 0):.2f}%)."
            )

        if best_tf == "1min" and scanner_engine_mode != "one_plus_one":
            # Every engine, S1 included (2026-09-26 user): a plain 1m signal enters only
            # when the TF toggle is 1M alone; otherwise 1m enters only on a burst, spike,
            # snapback, cascade or late-session reclaim.
            if s12_burst_entry and not volume_spike_entry:
                logger.info(f"🚀 Confirmed 1min S1+S2 D2 native burst entry for {instrument}")
            elif not one_min_burst_or_spike_entry:
                if one_way_cascade_entry:
                    logger.info(f"🚀 Confirmed 1min one-way cascade entry for {instrument}")
                elif late_session_reclaim_entry:
                    logger.info(f"🚀 Confirmed 1min late-session reclaim entry for {instrument}")
                elif is_reversion_scalp:
                    logger.info(
                        f"🚀 Confirmed 1min POST-WIN SNAPBACK entry for {instrument} "
                        f"(window open; spike not required)"
                    )
                elif tf_policy == "PRIMARY_1" or _setting_enabled(
                    getattr(settings, "one_min_raw_entry_enabled", False), False
                ):
                    # The operator's fast lane. OFF by default: with the burst
                    # requirement in place only 4 raw 1min signals reached the buffer in
                    # 52 sessions, so there is no sample to judge the open lane on. When
                    # it is switched on, the 1min buffer is the short one
                    # (volume_spike_entry_stabilization_seconds, 3-15s) rather than the
                    # 30s 5m/15m window, and every other gate still applies.
                    # A plain 1m entry under 1M alone takes the same fast 1m chassis.
                    setattr(best_sig, "one_min_raw_entry", True)
                    logger.info(
                        f"⚡ 1min raw entry admitted for {instrument} {best_sig.signal_type} ("
                        + (f"TF policy {tf_policy}" if tf_policy == "PRIMARY_1" else "ONE_MIN_RAW_ENTRY_ENABLED")
                        + ")"
                    )
                else:
                    spike_requirement = self._volume_spike_requirement_text(
                        settings, instrument
                    )
                    return _build_rejected_candidate(
                        f"1min entry requires directional burst: {spike_requirement}; "
                        "POST-WIN SNAPBACK; confirmed one-way cascade or late-session reclaim "
                        "(or select 1M alone in the TF toggle)",
                        "REJECTED - Weak 1min",
                    )
            else:
                path = "volume-spike"
                if is_reversion_scalp:
                    path = "volume-spike + SNAPBACK"
                logger.info(
                    f"🚀 Confirmed 1min {path} entry for {instrument}: "
                    f"{self._volume_spike_metric_text(volume_spike_metrics, settings, instrument, best_tf)}"
                )

        can_open, _ = scanner.trades.can_open_trade()
        if scanner_engine_mode != "one_plus_one" and not can_open:
            return _build_rejected_candidate("Max overall trades reached or capital depleted", "REJECTED - Max Trades")

        iv_percentile_for_routing = float(iv_percentile if iv_percentile is not None else 50.0)
        is_high_iv = iv_percentile_for_routing > 80.0
        strong_momentum = abs(intel_norm) > 0.60 or best_score > 0.4
        
        trade_date = best_sig.timestamp.date()
        is_0dte = _scanner_is_expiry_day(scanner, instrument, trade_date)
        is_late_session = best_sig.timestamp.time() >= datetime.strptime("14:00", "%H:%M").time()
        
        inst_type = scanner._resolve_instrument_type(
            instrument=instrument,
            grade=graded.grade,
            regime=regime,
            intel_score=intel_score,
            signal_score=best_score,
            confidence=graded.confidence,
            adx_value=getattr(best_sig, "adx_value", 0.0),
            atr_value=getattr(best_sig, "atr_value", 0.0),
            price=best_sig.price,
            signal_time=best_sig.timestamp,
            iv_percentile=iv_percentile_for_routing,
            timeframe=best_tf,
            five_min_true_impulse=true_5min_impulse,
        )
        if scanner_engine_mode != "one_plus_one" and not scanner.risk_manager.can_trade(inst_type):
            return _build_rejected_candidate(
                f"{inst_type} circuit breaker is active",
                f"REJECTED - {inst_type} Risk Halt",
            )

        # ── Regime veto + directional census (shared with replay path) ──
        _veto_block = directional_fut_veto(
            settings,
            inst_type=inst_type,
            signal_type=best_sig.signal_type,
            regime=regime,
            adx_value=adx_value,
            power_move=fast_thrust_override or is_reversion_scalp or s1_lane,
        )
        if scanner_engine_mode != "one_plus_one" and _veto_block and not (
            is_reversion_scalp or
            (late_session_reclaim_entry
            and isinstance(_reversal_relief_context, dict)
            and bool(_reversal_relief_context.get("allow_regime_veto")))
        ):
            # If we got a regime veto on FUT and inst_pref is AUTO, fall back to OPT
            scanner_inst_pref = (getattr(scanner, "inst_pref", "AUTO") or "AUTO").upper()
            if inst_type == "FUT" and scanner_inst_pref == "AUTO":
                logger.warning(
                    f"[{instrument}] FUT regime vetoed ({_veto_block}), falling back to OPT"
                )
                inst_type = "OPT"
                # Check if OPT is allowed by risk manager
                if not scanner.risk_manager.can_trade(inst_type):
                    return _build_rejected_candidate(
                        f"{inst_type} circuit breaker is active",
                        f"REJECTED - {inst_type} Risk Halt",
                    )
            else:
                return _build_rejected_candidate(_veto_block, "REJECTED - Regime Veto")
        if _veto_block and late_session_reclaim_entry:
            best_reasons = list(best_reasons or []) + [
                f"Late-session reclaim overrode stale regime veto: {_veto_block}"
            ]

        _session_ctx = intel_result.get("session_context", {}) if isinstance(intel_result, dict) else {}
        if (
            scanner_engine_mode != "one_plus_one"
            and
            _setting_enabled(getattr(settings, "session_context_trend_day_guard_enabled", True), True)
            and isinstance(_session_ctx, dict)
            and best_tf in {"1min", "5min"}
            and not fast_thrust_override
            and not is_reversion_scalp
            and not late_session_reclaim_entry
            and not _reversal_relief_context
            and not s1_lane
        ):
            # Prefer rolling effective phase so the guard tracks mood flips
            # (minute/hour/session pockets) instead of a sticky full-day label.
            _use_dynamic_mood = _setting_enabled(
                getattr(settings, "session_mood_dynamic_enabled", True), True
            )
            _day_type = str(
                (
                    (
                        _session_ctx.get("effective_day_type")
                        or _session_ctx.get("market_phase")
                        or _session_ctx.get("day_type")
                    )
                    if _use_dynamic_mood
                    else _session_ctx.get("day_type")
                )
                or ""
            ).upper()
            if _day_type == "MOOD_DUMP":
                _day_type = "TREND_DAY_DOWN"
            elif _day_type == "MOOD_RALLY":
                _day_type = "TREND_DAY_UP"
            elif _day_type == "MOOD_TRANSITION":
                _day_type = ""
            _day_conf = float(
                (
                    _session_ctx.get("phase_confidence")
                    if _use_dynamic_mood and _session_ctx.get("phase_confidence") is not None
                    else _session_ctx.get("confidence")
                )
                or 0.0
            )
            _opening_day = str(_session_ctx.get("opening_day_type") or _session_ctx.get("day_type") or "").upper()
            _expected = "BUY" if _day_type == "TREND_DAY_UP" else ("SELL" if _day_type == "TREND_DAY_DOWN" else "")
            _min_day_conf = float(getattr(settings, "session_context_trend_day_guard_min_confidence", 0.70) or 0.70)
            # Late-session multi-index shock overrides sticky opening TREND_DAY_*.
            # Active only inside shock tenure window (default 14:45–15:11 IST).
            _shock_yield = False
            if market_ltp_shock_entry or bool(getattr(best_sig, "market_wide_reversal_entry", False)):
                try:
                    _sig_t = best_sig.timestamp
                    if getattr(_sig_t, "tzinfo", None) is not None:
                        _sig_t = _sig_t.astimezone(IST).replace(tzinfo=None)
                    _win_start = str(
                        getattr(settings, "market_ltp_shock_session_start", None)
                        or getattr(settings, "market_ltp_shock_session_override_start", "14:45")
                        or "14:45"
                    )
                    _win_end = str(getattr(settings, "market_ltp_shock_session_end", "15:11") or "15:11")
                    _start_t = datetime.strptime(_win_start, "%H:%M").time()
                    _end_t = datetime.strptime(_win_end, "%H:%M").time()
                    _sig_clock = _sig_t.time()
                    if _start_t <= _sig_clock <= _end_t:
                        _shock_side = str(best_sig.signal_type).upper()
                        _opening_expected = (
                            "BUY" if _opening_day == "TREND_DAY_UP"
                            else ("SELL" if _opening_day == "TREND_DAY_DOWN" else "")
                        )
                        if _opening_expected and _shock_side != _opening_expected:
                            _shock_yield = True
                            best_reasons = list(best_reasons or []) + [
                                f"Session trend guard yielded: late LTP shock {_shock_side} "
                                f"overrides sticky {_opening_day} in {_win_start}-{_win_end}"
                            ]
                except Exception:
                    _shock_yield = False
            if stale_regime_reversal_entry and _expected and str(best_sig.signal_type).upper() != _expected:
                # The day label is built on the same move whose extreme went stale.
                best_reasons = list(best_reasons or []) + [
                    f"Session trend guard yielded: {_day_type} is stale "
                    f"({_stale_regime_context.get('reason')})"
                ]
            if (
                _expected
                and _day_conf >= _min_day_conf
                and str(best_sig.signal_type).upper() != _expected
                and not _shock_yield
                and not stale_regime_reversal_entry
            ):
                # Census-contradiction yield: when the session's own directional
                # census strongly opposes the day-type label, the label is stale
                # (2026-07-08: TREND_DAY_UP kept vetoing midcap SELLs 11:19-12:36
                # while the census was already 5L/10S; midcap then fell 1.9%).
                _census_yield_reason = session_trend_guard_census_yield(
                    settings,
                    census=getattr(scanner, "_dir_census", {}) or {},
                    signal_type=str(best_sig.signal_type).upper(),
                    expected=_expected,
                    day_type=_day_type,
                )
                if _census_yield_reason:
                    best_reasons = list(best_reasons or []) + [_census_yield_reason]
                else:
                    _session_bias = str(
                        _session_ctx.get("effective_bias")
                        or _session_ctx.get("bias")
                        or "NEUTRAL"
                    ).upper()
                    _delta = _session_ctx.get("session_delta", {}) if isinstance(_session_ctx.get("session_delta"), dict) else {}
                    msg = (
                        f"Session trend-phase guard: {_day_type} {_session_bias} "
                        f"conf={_day_conf:.0%}, delta={_delta.get('bias', 'NEUTRAL')} blocks "
                        f"{best_tf} counter-trend scalp"
                    )
                    return _build_rejected_candidate(msg, "REJECTED - Session Trend Guard")

        # ── Pre-candle flip veto (default OFF; telemetry-first) ──
        # The predictor projected the 15m flip that stopped the 14:53 A+ short on
        # 2026-07-03 but was wired to nothing. Telemetry always logs a would-veto
        # so live sessions accumulate evidence; the gate itself is opt-in.
        try:
            _pre_map = (getattr(scanner, "_pre_candle_predictions", {}) or {}).get(
                str(instrument or "").upper(), {}
            )
            _pre = _pre_map.get(best_tf, {}) if isinstance(_pre_map, dict) else {}
            _pre_state = str(_pre.get("state") or "").upper()
            _pre_dir = str(_pre.get("direction") or "").upper()
            _pre_conf = float(_pre.get("confidence") or 0.0)
            _sig_dir = "LONG" if str(best_sig.signal_type).upper() == "BUY" else "SHORT"
            _min_flip_conf = float(
                getattr(settings, "pre_candle_flip_veto_min_confidence", 0.70) or 0.70
            )
            if (
                scanner_engine_mode != "one_plus_one"
                and
                _pre_state in {"LIKELY_5M_FLIP", "LIKELY_15M_FLIP"}
                and _pre_dir in {"LONG", "SHORT"}
                and _pre_dir != _sig_dir
                and _pre_conf >= _min_flip_conf
                and not fast_thrust_override
                and not is_reversion_scalp
                and not s1_lane
            ):
                _flip_msg = (
                    f"Pre-candle flip against entry: {instrument} {best_tf} {_sig_dir} vs "
                    f"{_pre_state} {_pre_dir} conf {_pre_conf:.0%}"
                )
                if _setting_enabled(
                    getattr(settings, "pre_candle_flip_veto_enabled", False), False
                ):
                    return _build_rejected_candidate(_flip_msg, "REJECTED - Pre-Candle Flip Veto")
                logger.warning(f"[pre-candle would-veto] {_flip_msg}")
        except Exception as _flip_exc:
            logger.debug(f"Pre-candle flip veto check skipped for {instrument}: {_flip_exc}")

        # Day-type label for the census yield. Derived here rather than reusing the
        # trend-guard's _day_type, which is computed inside a conditional above and is
        # not guaranteed to be bound on this path. Same MOOD_* normalisation, so both
        # read the label the same way.
        _cs_day_type, _cs_day_conf = "", 0.0
        try:
            _cs_dyn = _setting_enabled(
                getattr(settings, "session_mood_dynamic_enabled", True), True
            )
            _cs_day_type = str(
                (
                    _session_ctx.get("effective_day_type")
                    or _session_ctx.get("market_phase")
                    or _session_ctx.get("day_type")
                )
                if _cs_dyn
                else _session_ctx.get("day_type")
                or ""
            ).upper()
            if _cs_day_type == "MOOD_DUMP":
                _cs_day_type = "TREND_DAY_DOWN"
            elif _cs_day_type == "MOOD_RALLY":
                _cs_day_type = "TREND_DAY_UP"
            elif _cs_day_type == "MOOD_TRANSITION":
                _cs_day_type = ""
            _cs_raw = (
                _session_ctx.get("phase_confidence")
                if _cs_dyn and _session_ctx.get("phase_confidence") is not None
                else _session_ctx.get("confidence")
            )
            _cs_day_conf = float(_cs_raw or 0.0)
        except Exception as _cs_exc:
            logger.debug(f"Census day-type context unavailable for {instrument}: {_cs_exc}")

        _census_block = directional_census_block(
            settings,
            getattr(scanner, "_dir_census", {}) or {},
            signal_direction=signal_direction,
            confidence=graded.confidence,
            grade_floor=self._grade_min_confidence("B+"),
            power_move=fast_thrust_override or is_reversion_scalp,
            reversal_context=_reversal_relief_context,
            day_type=_cs_day_type,
            day_type_confidence=_cs_day_conf,
        )
        if _census_block and stale_regime_reversal_entry:
            # The census counts this scanner's own lagging arrows; 22 census-refused
            # stale-regime reversals averaged +1.29 ATR at 30m (2026-08-18..09-17).
            best_reasons = list(best_reasons or []) + [
                f"Counter census yielded to stale regime: {_census_block}"
            ]
            _census_block = ""
        if _census_block and not is_reversion_scalp and not s1_lane:
            return _build_rejected_candidate(_census_block, "REJECTED - Counter Census")

        # ── Structural price-action patch (SEPARATE, gated; default OFF) ──
        # Adds S/R + swing-structure + candle-pattern + volume-profile awareness.
        # Advisory by default (telemetry only); blocks ADVERSE setups (e.g. LONG
        # into resistance / against structure) when price_action_block_adverse is on.
        # Power-moves / volume-spike thrusts are exempt. Remove the call + helper +
        # intelligence/price_action.py to fully drop the feature.
        _pa_ctx = price_action_context(scanner, instrument, best_tf, best_sig, settings)
        if _pa_ctx is not None:
            try:
                scanner._last_price_action_ctx = _pa_ctx
            except Exception:
                pass
            if (
                scanner_engine_mode != "one_plus_one"
                and
                _setting_enabled(getattr(settings, "price_action_block_adverse", False), False)
                and not fast_thrust_override
                and not s1_lane
                and _pa_ctx.get("verdict") == "ADVERSE"
            ):
                # Shadow only: would a holding opening gap make prior-session structure moot?
                from engine import gap_open_shadow as _gap_open_shadow
                _gap_open_shadow.record(scanner, instrument, best_tf, best_sig, settings, _pa_ctx)
                _pa_relief = _reversal_relief_context or (
                    _stale_regime_context if stale_regime_reversal_entry else None
                )
                if _setting_enabled(
                    getattr(settings, "price_action_reversal_relief_enabled", True),
                    True,
                ) and _pa_relief:
                    # "SELL into support" is where a stale uptrend's reversal starts;
                    # BANKNIFTY 10:25/10:30 on 2026-09-17 were refused there and fell 400.
                    try:
                        _pa_ctx["relief_reason"] = _pa_relief.get(
                            "reason",
                            "confirmed reversal relief",
                        )
                    except Exception:
                        pass
                else:
                    msg = (
                        f"Price-action ADVERSE [{_pa_ctx.get('score', 0):+.2f}: "
                        f"{', '.join(_pa_ctx.get('reasons', [])[:3])}]"
                    )
                    return _build_rejected_candidate(msg, "REJECTED - Price Action")

        # Product eligibility below is limited to contract/quote/liquidity integrity.
        # The removed option grade/chop ladder duplicated the canonical quality verdict
        # and could silently flip an otherwise valid OPT setup into FUT.

        # ═══ Cross-Timeframe Monopoly Guard ═══
        existing_index_trades = [
            t
            for t in scanner.trades.open_trades.values()
            if t.instrument == instrument and self._is_live_book_trade(t)
        ]
        concurrency_guard_on = _setting_enabled(getattr(settings, "ut_concurrency_guard", True), True)
        cascade_multi_tf = (
            one_way_cascade_entry
            and _setting_enabled(getattr(settings, "cascade_multi_tf_stack_enabled", True), True)
        )
        if (
            scanner_engine_mode != "one_plus_one"
            and existing_index_trades
            and concurrency_guard_on
            and not quad_prime_entry
        ):
            # Cascade multi-TF: allow additional TFs up to cascade cap (no forced FUT/OPT flip).
            if cascade_multi_tf:
                primary_max = max(1, int(getattr(settings, "cascade_primary_max_tf", 3) or 3))
                dual_max = max(0, int(getattr(settings, "cascade_dual_addon_max", 2) or 2))
                cascade_cap = primary_max + dual_max
                same_tf_open = any(
                    str(getattr(t, "timeframe", "") or "") == best_tf
                    and (
                        self._signal_source_lane(t, scanner_engine_mode)
                        == best_source_lane
                        or not best_source_lane
                    )
                    for t in existing_index_trades
                )
                if same_tf_open and best_source_lane != "B":
                    msg = f"Cascade same-TF {best_tf} already active"
                    logger.info(f"🚫 Monopoly Guard: {msg}. Skipping.")
                    return _build_rejected_candidate(msg, "REJECTED - Monopoly")
                if len(existing_index_trades) >= cascade_cap:
                    msg = f"Cascade multi-TF cap {cascade_cap} already filled"
                    logger.info(f"🚫 Monopoly Guard: {msg}. Skipping {best_tf}.")
                    return _build_rejected_candidate(msg, "REJECTED - Monopoly")
                logger.info(
                    f"🌟 Cascade multi-TF stack ({best_tf}): allowing alongside "
                    f"{len(existing_index_trades)} open on {instrument}"
                )
            else:
                # Check if this is a DIFFERENT best_tf
                is_different_tf = all(getattr(t, "timeframe", "") != best_tf for t in existing_index_trades)
                candidate_rank = self._grade_rank(graded.grade)
                candidate_conf = float(getattr(graded, "confidence", 0.0) or 0.0)

                def _same_grade_conf_or_better(existing_trade) -> bool:
                    existing_rank = self._grade_rank(getattr(existing_trade, "grade", ""))
                    existing_conf = float(getattr(existing_trade, "confidence", 0.0) or 0.0)
                    if candidate_rank > existing_rank:
                        return True
                    return candidate_rank == existing_rank and candidate_conf + 1e-9 >= existing_conf

                can_add = False
                if (
                    is_different_tf
                    and len(existing_index_trades) == 1
                    and _same_grade_conf_or_better(existing_index_trades[0])
                ):
                    # Force switch type (FUT -> OPT or vice versa)
                    existing_trade = existing_index_trades[0]
                    existing_type = getattr(existing_trade, 'inst_type', 'FUT')
                    
                    # Respect instrument preference!
                    if scanner.inst_pref == "FUT" and existing_type == "FUT":
                        msg = "Already have FUT trade and preference is FUT"
                        logger.info(f"🚫 Monopoly Guard: {msg}. Skipping.")
                        return _build_rejected_candidate(msg, "REJECTED - Monopoly")
                    elif scanner.inst_pref == "OPT" and existing_type == "OPT":
                        msg = "Already have OPT trade and preference is OPT"
                        logger.info(f"🚫 Monopoly Guard: {msg}. Skipping.")
                        return _build_rejected_candidate(msg, "REJECTED - Monopoly")
                    
                    # If no preference conflict, force switch type
                    inst_type = "OPT" if existing_type == "FUT" else "FUT"
                    can_add = True
                    logger.info(f"🌟 Cross-TF Confluence ({best_tf}): Adding {inst_type} to complement {getattr(existing_trade, 'timeframe', '')} {existing_type}")

                # Rare dual-TF: same-direction, both excellent, different TF →
                # allow second leg as OPT (never 2x FUT) even when preference is OPT.
                if (
                    not can_add
                    and is_different_tf
                    and len(existing_index_trades) == 1
                    and _setting_enabled(getattr(settings, "rare_dual_tf_entry_enabled", True), True)
                ):
                    existing_trade = existing_index_trades[0]
                    min_grade = str(getattr(settings, "rare_dual_tf_min_grade", "A") or "A").upper()
                    min_conf = float(getattr(settings, "rare_dual_tf_min_confidence", 0.70) or 0.70)
                    min_rank = self._grade_rank(min_grade)
                    ex_dir = str(getattr(existing_trade, "direction", "") or "").upper()
                    ex_tf = str(getattr(existing_trade, "timeframe", "") or "")
                    ex_grade = str(getattr(existing_trade, "grade", "") or "")
                    ex_conf = float(getattr(existing_trade, "confidence", 0.0) or 0.0)
                    same_dir = ex_dir == best_direction
                    tfs_pair = {ex_tf, best_tf}
                    has_fast = "1min" in tfs_pair
                    has_struct = bool(tfs_pair & {"5min", "15min"})
                    both_strong = (
                        self._grade_rank(ex_grade) >= min_rank
                        and ex_conf + 1e-9 >= min_conf
                        and candidate_rank >= min_rank
                        and candidate_conf + 1e-9 >= min_conf
                    )
                    if same_dir and has_fast and has_struct and both_strong:
                        prefer_opt = _setting_enabled(
                            getattr(settings, "rare_dual_tf_prefer_opt_second", True), True
                        )
                        existing_type = str(getattr(existing_trade, "inst_type", "FUT") or "FUT").upper()
                        if prefer_opt or existing_type == "FUT":
                            # Second leg OPT so we never dual FUT.
                            if existing_type == "FUT" or scanner.inst_pref != "FUT":
                                inst_type = "OPT"
                                can_add = True
                                logger.info(
                                    f"🌟 Rare dual-TF monopoly allow ({best_tf}): "
                                    f"OPT alongside {ex_tf} {existing_type} "
                                    f"(both ≥{min_grade}/{min_conf:.0%})"
                                )

                if not can_add:
                    msg = f"Already has an active {getattr(existing_index_trades[0], 'timeframe', '')} trade"
                    logger.info(f"🚫 Monopoly Guard: {instrument} {msg}. Skipping {best_tf}.")
                    return _build_rejected_candidate(msg, "REJECTED - Monopoly")

        strike_interval = cfg.get("strike_interval", 50)
        strike_selection = getattr(settings, "option_strike_selection", "ATM")
        option_type, atm_strike_val, itm_strike_val, target_strikes = select_contract_strikes(
            inst_type=inst_type,
            signal_type=best_sig.signal_type,
            spot=spot,
            strike_interval=strike_interval,
            strike_selection=strike_selection,
        )

        # Read ONCE whenever an option needs either strike ranking or absolute
        # liquidity admission. allow_fetch=False means this uses the chain already
        # in cache or nothing at all -- no broker call and no latency added to entry.
        _liquidity_chain = None
        _needs_liquidity_chain = str(inst_type or "").upper() == "OPT" and (
            _setting_enabled(
                getattr(settings, "option_contract_liquidity_gate_enabled", True), True
            )
            or (
                len(target_strikes) > 1
                and _setting_enabled(
                    getattr(settings, "option_strike_liquidity_ranking_enabled", True), True
                )
            )
        )
        if _needs_liquidity_chain:
            try:
                _liquidity_chain, _ = scanner._get_option_chain_cached(
                    instrument, cfg, allow_fetch=False
                )
            except Exception as exc:
                logger.debug(f"Strike liquidity chain unavailable for {instrument}: {exc}")
                _liquidity_chain = None

        results_list = []
        for strike in target_strikes:
            if inst_type == "FUT":
                option_type_local = ""
                strike_local = 0.0
                instrument_multiplier = 1.0
            else:
                option_type_local = option_type
                strike_local = strike
                # Per-strike Black-Scholes delta (ITM/OTM aware), not the flat ATM delta.
                try:
                    dte_val = _scanner_days_to_expiry(scanner, instrument, best_sig.timestamp)
                except Exception:
                    dte_val = 1
                instrument_multiplier = self._option_signed_delta(
                    spot, strike, option_type_local, intel_result, dte_val,
                    0.5 if option_type_local == "CE" else -0.5,
                )

            final_rr = self._dynamic_rr(
                best_grade,
                best_conf,
                intel_score,
                timeframe=best_tf,
                direction=best_direction,
            )
            if volume_spike_entry or market_ltp_shock_entry:
                if best_tf == "1min" or market_ltp_shock_entry:
                    # Late shock dumps/rallies are short-horizon scalp/OPT paths.
                    final_rr = float(
                        getattr(settings, "volume_spike_entry_1min_scalp_rr", 1.5)
                        or 1.5
                    )
                else:
                    final_rr = max(
                        final_rr,
                        float(getattr(settings, "volume_spike_entry_min_rr", 2.5) or 2.5),
                    )
            if late_session_reclaim_entry:
                final_rr = float(getattr(settings, "late_session_reclaim_rr", 1.5) or 1.5)
            if one_way_cascade_entry:
                final_rr = float(getattr(settings, "one_way_cascade_rr", 1.5) or 1.5)
            if is_reversion_scalp:
                # RR 1.5 floor (aligned with 1m spike scalp RR).
                final_rr = max(
                    1.5,
                    float(getattr(settings, "post_exit_reversion_min_rr", 1.5) or 1.5),
                )
            s1_guard = 0.0
            s1_weak_target = False
            if uts_lane_on:
                # UT SMART EXIT: the chart has no target, so this R multiple is nominal (the
                # lane never closes on it); the guard is set from the premium below.
                s1_weak_target = False
                final_rr = uts_lane_policy.target_rr(settings)
            elif s12_lane_on:
                final_rr = s12_lane_policy.target_rr(settings)
                s12_floor = s12_lane_policy.min_entry_rr(settings)
                if final_rr + 1e-9 < s12_floor:
                    logger.info(
                        f"S1+S2 Dual {instrument} {best_tf}: target {final_rr:.2f}R is below "
                        f"the {s12_floor:.2f}R entry floor; signal not taken"
                    )
                    continue
                s1_guard = s12_lane_policy.premium_cap_pct(
                    best_sig, settings, regime=regime, timeframe=best_tf
                )
            elif s1_lane == "S1":
                # S1 targets are R multiples of its SL1; T3 is the hard target, T1/T2 and the
                # profit lock are managed by the S1 lane while the trade is open.
                # 2026-09-25 user: 1R on a normal trade, 0.7R when the move is weak; the
                # runner keeps a trade open past it while the S1 trend is with it.
                s1_weak_target = s1_lane_policy.entry_is_weak(
                    best_sig, regime, "LONG" if best_sig.signal_type == "BUY" else "SHORT", settings
                )
                final_rr = s1_lane_policy.target_rr(settings, weak=s1_weak_target)
                # 2026-10-08 user (item 7): the RR floor is the base target, so a weak-move
                # S1 entry (0.7R target, cannot pay after costs) is refused rather than having
                # its target silently rebuilt to the floor at execution.
                s1_floor = s1_lane_policy.lane_min_entry_rr(best_sig, settings)
                if final_rr + 1e-9 < s1_floor:
                    logger.info(
                        f"S1 {instrument} {best_tf}: weak-move target {final_rr:.2f}R below the "
                        f"{s1_floor:.2f}R entry floor; signal not taken"
                    )
                    try:
                        scanner._diag_inc("s1_lane", "weak_target_refused")
                    except Exception:
                        pass
                    continue
                s1_guard = s1_lane_policy.premium_guard_pct(
                    best_tf,
                    datetime.now(IST).replace(tzinfo=None),
                    float(getattr(best_sig, "s1_score", 0.0) or 0.0),
                    settings,
                    regime=regime,
                )
                s1_guard = min(
                    s1_guard,
                    option_stop_tolerance_pct(
                        scanner.options_sl_pct, regime,
                        confidence=best_conf, grade=best_grade,
                    ),
                )
            direction = normalized_direction(best_sig.signal_type)

            # RR-scaled sizing. The legacy `1.0 if final_rr >= 1.5 else final_rr/1.5`
            # was dead code — dynamic_rr already floors at 1.5R, so it was always 1.0 and never
            # shrank size. Gate a real implementation behind an explicit setting
            # (default OFF = full size, preserving live behavior); when enabled, size
            # scales down for low-RR setups toward rr_size_floor and reaches full size
            # once final_rr >= rr_size_reference_rr.
            quality_factor, rr_factor, total_multiplier = candidate_size_multipliers(
                confidence=best_conf,
                grade=best_grade,
                final_rr=final_rr,
                rr_scaling_enabled=_setting_enabled(
                    getattr(settings, "rr_size_scaling_enabled", False), False
                ),
                rr_reference=float(
                    getattr(settings, "rr_size_reference_rr", 2.5) or 2.5
                ),
                rr_floor=float(getattr(settings, "rr_size_floor", 0.5) or 0.5),
            )
            cap_based_lots = None
            effective_fut_sl_pct = None

            if inst_type == "FUT":
                if s12_lane_on:
                    s12_levels = s12_lane_policy.admission_levels(best_sig, spot, settings)
                    index_stop_distance = abs(float(spot or 0.0) - float(s12_levels.get("s12_stop") or 0.0))
                    effective_fut_sl_pct = (
                        index_stop_distance / max(float(spot or 0.0), 1e-9) * 100.0
                    )
                elif not s1_lane and (
                    volume_spike_entry
                    or market_ltp_shock_entry
                    or late_session_reclaim_entry
                    or one_way_cascade_entry
                    or is_reversion_scalp
                ):
                    if is_reversion_scalp:
                        effective_fut_sl_pct = self._tight_futures_stop_pct(
                            scanner.futures_sl_pct,
                            settings,
                            "reversion_scalp_futures_sl_pct",
                            0.05,
                        )
                    elif one_way_cascade_entry:
                        effective_fut_sl_pct = self._tight_futures_stop_pct(
                            scanner.futures_sl_pct,
                            settings,
                            "one_way_cascade_futures_sl_pct",
                            0.05,
                        )
                    elif late_session_reclaim_entry:
                        effective_fut_sl_pct = self._tight_futures_stop_pct(
                            scanner.futures_sl_pct,
                            settings,
                            "late_session_reclaim_futures_sl_pct",
                            0.05,
                        )
                    else:
                        # volume spike + market LTP shock share tight FUT stop
                        effective_fut_sl_pct = self._volume_spike_futures_stop_pct(
                            scanner.futures_sl_pct,
                            settings,
                        )
                    index_stop_distance = min(
                        max(0.0, float(best_sig.stop_distance or 0.0)),
                        max(0.0, float(spot or 0.0))
                        * (effective_fut_sl_pct / 100.0),
                    )
                else:
                    effective_fut_sl_pct = futures_stop_tolerance_pct(
                        scanner.futures_sl_pct,
                        regime,
                    )
                    index_stop_distance = self._futures_stop_distance(
                        best_sig.stop_distance,
                        spot,
                        scanner.futures_sl_pct,
                        regime,
                    )

                    stop_rejection = self._futures_stop_distance_rejection(
                        best_sig.stop_distance,
                        spot,
                        scanner.futures_sl_pct,
                        safety_multiplier=float(getattr(settings, "futures_sl_safety_multiplier", 2.0) or 2.0),
                        regime=regime,
                    )
                    if stop_rejection:
                        return _build_rejected_candidate(
                            stop_rejection,
                            "REJECTED - FUT Stop Distance",
                        )
                # Final lots come from resolve_trade_lots (fixed settings lots).
            else:
                est_premium = _estimate_opt_premium(
                    spot, getattr(self, "scanner", None), instrument, _sig_date(best_sig)
                )
                # No capital hardcap on lots when auto sizing is OFF (default).
                cap_based_lots = None
                if s12_lane_on:
                    s12_levels = s12_lane_policy.admission_levels(best_sig, spot, settings)
                    index_stop_distance = abs(float(spot or 0.0) - float(s12_levels.get("s12_stop") or 0.0))
                    option_stop_pct = s1_guard
                elif scanner_engine_mode == "one_plus_one":
                    # Native spot geometry is mapped later with the live signed delta;
                    # no platform premium-percent or regime cap may reshape it.
                    index_stop_distance = abs(float(spot) - native_stop_spot)
                else:
                    option_stop_pct = (
                    float(getattr(settings, "one_way_cascade_options_sl_pct", 5.0) or 5.0)
                    if one_way_cascade_entry
                    else
                    float(getattr(settings, "late_session_reclaim_options_sl_pct", 5.0) or 5.0)
                    if late_session_reclaim_entry
                    else
                    float(getattr(settings, "volume_spike_entry_options_sl_pct", 5.0) or 5.0)
                    if (volume_spike_entry or market_ltp_shock_entry or is_reversion_scalp)
                    else float(scanner.options_sl_pct or 0.0)
                )
                if uts_lane_on:
                    s1_guard = uts_lane_policy.premium_guard_pct(
                        est_premium, float(best_sig.stop_distance or 0.0), abs(instrument_multiplier), settings,
                        regime=regime, confidence=best_conf, grade=best_grade,
                        setup=best_sig, timeframe=best_tf,
                    )
                option_stop_pct = min(
                    option_stop_pct,
                    option_stop_tolerance_pct(
                        scanner.options_sl_pct, regime,
                        confidence=best_conf, grade=best_grade,
                        setup=best_sig,
                        fast_entry=bool(
                            one_way_cascade_entry or late_session_reclaim_entry
                            or volume_spike_entry or market_ltp_shock_entry or is_reversion_scalp
                        ),
                    ),
                )
                if s1_lane == "S1":
                    option_stop_pct = s1_guard
                max_premium_sl_pts = est_premium * (option_stop_pct / 100.0)
                hard_cap_index_sl = max_premium_sl_pts / max(0.1, abs(instrument_multiplier))
                # Option premium risk and underlying index risk are separate ceilings.
                # The 15% premium cap used to permit roughly 0.36% of NIFTY spot at
                # ATM delta, even though the approved underlying regime ceiling is
                # 0.10-0.15%. Clamp the underlying geometry independently before it is
                # converted into premium space.
                policy_caps = getattr(
                    settings, "entry_policy_underlying_stop_max_pct_by_index", {}
                )
                try:
                    configured_index_cap = effective_underlying_stop_cap_pct(
                        instrument,
                        (policy_caps or {}).get(instrument, scanner.futures_sl_pct),
                    )
                except (AttributeError, TypeError, ValueError):
                    configured_index_cap = float(scanner.futures_sl_pct or 0.0)
                regime_index_cap = futures_stop_tolerance_pct(
                    configured_index_cap, regime
                )
                policy_index_sl = max(0.0, float(spot or 0.0)) * (
                    regime_index_cap / 100.0
                )
                index_stop_distance = min(
                    best_sig.stop_distance,
                    hard_cap_index_sl,
                    policy_index_sl,
                )
                if s1_lane:
                    # The S1 SL1 (0.10-0.25% of the index) is the index stop; the premium
                    # guard is applied separately in premium space below.
                    index_stop_distance = max(0.0, float(best_sig.stop_distance or 0.0))

            if direction == "LONG":
                entry_stop = spot - index_stop_distance
                target = spot + (index_stop_distance * final_rr)
            else:
                entry_stop = spot + index_stop_distance
                target = spot - (index_stop_distance * final_rr)
            if s12_lane_on:
                s12_levels = s12_lane_policy.admission_levels(best_sig, spot, settings)
                entry_stop = float(s12_levels.get("s12_stop") or entry_stop)
                target = 0.0  # T1-T3 are milestones; native engine exits own the trade.

            trading_symbol = ""
            symbol_token = ""
            if inst_type == "FUT":
                info = scanner.market_info.get(instrument, {})
                trading_symbol = info.get("current_fut", "")
                symbol_token = info.get("current_fut_token", "")
            else:
                option_token_kwargs = {}
                native_expiry_preference = (
                    str(
                        getattr(best_sig, "native_option_expiry_preference", "") or ""
                    )
                    if scanner_engine_mode == "one_plus_one"
                    else ""
                )
                if native_expiry_preference:
                    option_token_kwargs["expiry_preference"] = native_expiry_preference
                opt_info = scanner.data.get_option_token(
                    instrument,
                    strike_local,
                    option_type_local,
                    **option_token_kwargs,
                )
                if opt_info:
                    trading_symbol = opt_info['symbol']
                    symbol_token = opt_info['token']

            # Never emit an OPT entry candidate without a tradeable contract token: the
            # premium-space stop/target are computed only inside the `if symbol_token:`
            # block below, so a token-less option would otherwise carry a spot-space stop
            # on a premium instrument. Skip it (an untradeable contract anyway).
            if inst_type == "OPT" and not symbol_token:
                logger.warning(
                    "Skipping live option candidate {} {}: no contract token for strike {} {}",
                    instrument,
                    best_tf,
                    strike_local,
                    option_type_local,
                )
                continue

            if symbol_token:
                strike_alpha = (instrument_multiplier * 0.4)
                oi_score = intel_result.get("oi", {}).get("score", 50) / 100.0
                vol_score = intel_result.get("volume", {}).get("score", 50) / 100.0
                itm_bonus = 0.15 if (strike_local == itm_strike_val and inst_type == "OPT") else 0.0
                final_strike_score = strike_alpha + (oi_score * 0.3) + (vol_score * 0.3) + itm_bonus

                live_trade_price = spot
                exchange = "BFO" if _is_bfo_instrument(instrument) else "NFO"
                candidate_stop = entry_stop
                candidate_target = target
                
                if symbol_token:
                    try:
                        if inst_type == "OPT":
                            quote_probe = SimpleNamespace(
                                instrument=instrument,
                                inst_type="OPT",
                                option_type=option_type_local,
                                atm_strike=strike_local,
                                trading_symbol=trading_symbol,
                                symbol_token=symbol_token,
                                price=spot,
                                target=target,
                                entry_spot=spot,
                            )
                            live_premium = (
                                scanner._get_live_premium_cached(quote_probe)
                                if hasattr(scanner, "_get_live_premium_cached")
                                else self._contract_ltp(scanner, exchange, trading_symbol, symbol_token)
                            )
                            live_premium = float(live_premium or 0.0)
                            if live_premium <= 0:
                                # Bounded retry before dropping: transient chain/quote
                                # hiccups cost 16 tradeable candidates on 2026-07-03.
                                import time as _time
                                retry_attempts = max(0, int(getattr(settings, "option_quote_retry_attempts", 2) or 0))
                                retry_delay = max(0.1, float(getattr(settings, "option_quote_retry_delay_seconds", 0.5) or 0.5))
                                for _ in range(retry_attempts):
                                    _time.sleep(retry_delay)
                                    live_premium = float(
                                        (
                                            scanner._get_live_premium_cached(quote_probe)
                                            if hasattr(scanner, "_get_live_premium_cached")
                                            else self._contract_ltp(scanner, exchange, trading_symbol, symbol_token)
                                        )
                                        or 0.0
                                    )
                                    if live_premium > 0:
                                        logger.info(
                                            "Option quote recovered on retry for {} {} ({})",
                                            instrument,
                                            best_tf,
                                            trading_symbol,
                                        )
                                        break
                            if live_premium <= 0:
                                logger.warning(
                                    "Skipping live option candidate {} {}: no verified LTP for {} ({})",
                                    instrument,
                                    best_tf,
                                    trading_symbol,
                                    symbol_token,
                                )
                                continue
                            live_trade_price = live_premium
                            if uts_lane_on:
                                # Catastrophe stop beyond the chart's close-based UT stop.
                                s1_guard = uts_lane_policy.premium_guard_pct(
                                    live_trade_price, index_stop_distance, abs(instrument_multiplier), settings,
                                    regime=regime, confidence=best_conf, grade=best_grade,
                                    setup=best_sig, timeframe=best_tf,
                                )
                                max_sl_dist = uts_lane_policy.premium_stop_distance(
                                    live_trade_price, index_stop_distance, abs(instrument_multiplier), settings,
                                    regime=regime, confidence=best_conf, grade=best_grade,
                                    setup=best_sig, timeframe=best_tf,
                                )
                            elif s12_lane_on:
                                max_sl_dist = s12_lane_policy.premium_stop_distance(
                                    live_trade_price,
                                    index_stop_distance,
                                    abs(instrument_multiplier),
                                    settings,
                                    setup=best_sig,
                                    timeframe=best_tf,
                                )
                            elif s1_lane:
                                max_sl_dist = s1_lane_policy.premium_stop_distance(
                                    live_trade_price,
                                    index_stop_distance,
                                    instrument_multiplier,
                                    s1_guard,
                                )
                            elif (
                                volume_spike_entry
                                or market_ltp_shock_entry
                                or late_session_reclaim_entry
                                or one_way_cascade_entry
                                or is_reversion_scalp
                            ):
                                tight_pct = (
                                    float(getattr(settings, "one_way_cascade_options_sl_pct", 5.0) or 5.0)
                                    if one_way_cascade_entry
                                    else
                                    float(getattr(settings, "late_session_reclaim_options_sl_pct", 5.0) or 5.0)
                                    if late_session_reclaim_entry
                                    else float(getattr(settings, "volume_spike_entry_options_sl_pct", 5.0) or 5.0)
                                )
                                max_sl_dist = max(0.5, live_trade_price * (tight_pct / 100.0))
                            else:
                                max_sl_dist = self._option_premium_stop_distance(
                                    live_trade_price,
                                    scanner,
                                    regime,
                                    timeframe=best_tf,
                                    atr_premium_stop=float(index_stop_distance) * abs(float(instrument_multiplier or 0.0)),
                                    settings=settings,
                                    intel_score=float(intel_score or 0.0),
                                    direction=direction,
                                    score_unit="canonical_signed",
                                )
                            if not s12_lane_on:
                                max_sl_dist = min(
                                    max_sl_dist,
                                    live_trade_price * option_stop_tolerance_pct(
                                        scanner.options_sl_pct, regime,
                                        confidence=best_conf, grade=best_grade,
                                        setup=best_sig,
                                        fast_entry=bool(
                                            volume_spike_entry or market_ltp_shock_entry
                                            or late_session_reclaim_entry or one_way_cascade_entry
                                            or is_reversion_scalp
                                        ),
                                    ) / 100.0,
                                )
                            if scanner_engine_mode != "one_plus_one":
                                candidate_stop = live_trade_price - max_sl_dist
                                candidate_target = 0.0 if s12_lane_on else live_trade_price + (max_sl_dist * final_rr)
                        else: # FUT
                            live_fut_ltp = self._contract_ltp(scanner, exchange, trading_symbol, symbol_token)
                            live_trade_price = live_fut_ltp if live_fut_ltp > 0 else spot
                            if scanner_engine_mode == "one_plus_one":
                                basis = live_trade_price - float(spot)
                                candidate_stop = native_stop_spot + basis
                                candidate_target = (
                                    native_target_spot + basis if native_has_hard_target else 0.0
                                )
                            elif s12_lane_on:
                                candidate_stop = live_trade_price + (entry_stop - float(spot or 0.0))
                                candidate_target = 0.0
                            else:
                                max_sl_dist = self._futures_stop_distance(
                                    index_stop_distance,
                                    live_trade_price,
                                    scanner.futures_sl_pct,
                                    regime,
                                )
                            if scanner_engine_mode != "one_plus_one" and not s12_lane_on and direction == "LONG":
                                candidate_stop = live_trade_price - max_sl_dist
                                candidate_target = live_trade_price + (max_sl_dist * final_rr)
                            elif scanner_engine_mode != "one_plus_one" and not s12_lane_on:
                                candidate_stop = live_trade_price + max_sl_dist
                                candidate_target = live_trade_price - (max_sl_dist * final_rr)
                    except Exception as e:
                        if inst_type == "OPT":
                            logger.warning(
                                "Skipping live option candidate {} {} after quote failure for {}: {}",
                                instrument,
                                best_tf,
                                trading_symbol,
                                e,
                            )
                            continue
                        else:
                            live_trade_price = spot
                            if scanner_engine_mode == "one_plus_one":
                                candidate_stop = native_stop_spot
                                candidate_target = native_target_spot if native_has_hard_target else 0.0
                            elif s12_lane_on:
                                candidate_stop = entry_stop
                                candidate_target = 0.0
                            else:
                                max_sl_dist = self._futures_stop_distance(
                                    index_stop_distance,
                                    live_trade_price,
                                    scanner.futures_sl_pct,
                                    regime,
                                )
                            if scanner_engine_mode != "one_plus_one" and not s12_lane_on and direction == "LONG":
                                candidate_stop = live_trade_price - max_sl_dist
                                candidate_target = live_trade_price + (max_sl_dist * final_rr)
                            elif scanner_engine_mode != "one_plus_one" and not s12_lane_on:
                                candidate_stop = live_trade_price + max_sl_dist
                                candidate_target = live_trade_price - (max_sl_dist * final_rr)
                
                no_entry_label, _ = _entry_cutoff(settings, best_tf, s1=s1_lane)
                # A confirmed power-move may enter past the no-entry cutoff, but only
                # up to the hard late cutoff so there is still time before square-off.
                power_move_cutoff_ok = bool(
                    best_after_no_entry
                    and fast_thrust_override
                    and not s1_lane
                    and scanner._power_move_late_cutoff_ok(best_sig.timestamp)
                )
                effective_no_entry = (
                    False
                    if scanner_engine_mode == "one_plus_one"
                    else best_after_no_entry and not power_move_cutoff_ok
                )
                s12_signal_only = bool(s12_lane_on and s12_lane_policy.signal_only(settings))
                candidate_status = (
                    f"NO ENTRY - AFTER {no_entry_label}" if effective_no_entry
                    else "SIGNAL ONLY" if s12_signal_only
                    else "TRADE SIGNAL"
                )
                candidate_action = "NO_ENTRY" if effective_no_entry else "ENTRY"
                candidate_reasons = list(best_reasons or [])
                if effective_no_entry:
                    candidate_reasons.append(f"After no-fresh-entry gate ({no_entry_label} IST)")
                elif power_move_cutoff_ok:
                    candidate_reasons.append(
                        f"⚡ Fast-thrust override: late entry allowed past {no_entry_label} cutoff"
                    )
                if volume_spike_entry:
                    volume_evidence = self._volume_spike_metric_text(
                        volume_spike_metrics, settings, instrument, best_tf
                    )
                    candidate_reasons.append(
                        f"⚡ Volume spike entry: {volume_evidence}, "
                        f"{volume_spike_metrics.get('range_atr_ratio', 0.0):.2f}x range, "
                        f"{volume_spike_metrics.get('close_location', 0.0):.0%} directional close"
                    )
                if market_ltp_shock_entry:
                    candidate_reasons.append(
                        f"⚡ Market LTP shock {market_ltp_shock_context.get('side', '')}: "
                        f"avg |move|={float(market_ltp_shock_context.get('avg_abs_move_pct', 0.0) or 0.0):.2f}%, "
                        f"indices={', '.join(market_ltp_shock_context.get('indices', []) or [])}, "
                        f"RR {final_rr:.1f}, short stab"
                    )
                if volume_spike_entry and best_tf == "1min":
                    candidate_reasons.append(
                        f"1min scalp mode: fast thrust entry with RR {final_rr:.1f}, tight stop, and scalp-lock"
                    )
                if late_session_reclaim_entry:
                    _late_move_label = str(
                        (_reversal_relief_context or {}).get("move_label")
                        or ("reclaim" if str(best_sig.signal_type).upper() == "BUY" else "rollover")
                    )
                    candidate_reasons.append(
                        f"Late-session synchronized {str(best_sig.signal_type).upper()} {_late_move_label}: tight {float(getattr(settings, 'late_session_reclaim_options_sl_pct', 5.0) or 5.0):.1f}% option stop, RR {final_rr:.1f}, scalp-lock"
                    )
                if one_way_cascade_entry:
                    candidate_reasons.append(
                        f"One-way cascade {str(best_sig.signal_type).upper()}: tight {float(getattr(settings, 'one_way_cascade_options_sl_pct', 5.0) or 5.0):.1f}% option stop, RR {final_rr:.1f}, runner trail {float(getattr(settings, 'one_way_cascade_runner_lock_pct', 0.85) or 0.85):.0%}"
                    )
                if quad_union_entry:
                    source_label = "+".join(best_source_engines or [best_source_engine or "quad_union"])
                    entry_label = (
                        "Adaptive dual"
                        if scanner_engine_mode == "adaptive_dual"
                        else "Adaptive hybrid"
                        if scanner_engine_mode == "adaptive_hybrid"
                        else ("Quality hybrid" if scanner_engine_mode == "quality_hybrid" else ("Quad prime" if scanner_engine_mode == "quad_prime" else "Quad union"))
                    )
                    candidate_reasons.append(
                        f"{entry_label} raw entry after repaint buffer: {source_label}"
                    )
                if inst_type == "FUT":
                    candidate_reasons.append(
                        f"FUT stop tolerance: {float(effective_fut_sl_pct or 0.0):.3f}% "
                        f"for {str(regime or 'UNKNOWN').upper()} "
                        f"(configured cap {float(scanner.futures_sl_pct):.3f}%)"
                    )

                if live_trade_price <= 0:
                    logger.warning(f"Skipping candidate for {instrument} due to invalid live price: {live_trade_price}")
                    continue

                if inst_type == "OPT":
                    entry_cap_pct = option_stop_tolerance_pct(
                        scanner.options_sl_pct, regime,
                        confidence=best_conf, grade=best_grade,
                        setup=best_sig,
                        fast_entry=bool(volume_spike_entry or market_ltp_shock_entry
                                        or late_session_reclaim_entry or one_way_cascade_entry
                                        or is_reversion_scalp),
                    )
                    candidate_stop = max(
                        float(candidate_stop or 0.0),
                        live_trade_price * (1.0 - entry_cap_pct / 100.0),
                    )
                    if s1_guard > 0:
                        s1_guard = min(s1_guard, entry_cap_pct)

                requested_lots = (
                    int(getattr(scanner, "user_lots", {}).get(instrument, 1) or 1)
                    if inst_type == "OPT"
                    else int(getattr(scanner, "user_lots_fut", {}).get(instrument, 1) or 1)
                )
                # Lot scale: cascade OPT 2–3×; heavy-trend (2/4 align) OPT 2×; else 1×.
                # FUT always 1×. A+/power_move alone never multiplies.
                # 2026-09-25 user, S1: the configured lots per index/timeframe, 1x even
                # on a cascade; only a heavy trend doubles (OPT).
                cascade_scale = bool(one_way_cascade_entry) and not s1_lane
                heavy_trend_scale = False
                if not cascade_scale and str(inst_type or "").upper() == "OPT":
                    try:
                        _ht_fn = getattr(scanner, "_heavy_trend_lot_scale_eligible", None)
                        if callable(_ht_fn):
                            heavy_trend_scale = bool(
                                _ht_fn(instrument, best_direction, best_tf)
                            )
                    except Exception:
                        heavy_trend_scale = False
                # Scaling is not a breakout detector.  It is permitted only
                # after the locked compression box has a confirmed breach and
                # the current canonical regime agrees with this trade.
                # Missing compression evidence is fail-closed during live sizing.
                _compression_state = str((_session_ctx or {}).get("compression_state") or "FORMING").upper()
                _compression_box = (_session_ctx or {}).get("compression_box", {}) or {}
                _breakout_direction = str(_compression_box.get("breakout_direction") or "").upper()
                _signal_breakout_direction = "UP" if str(best_sig.signal_type).upper() == "BUY" else "DOWN"
                _confirmed_breakout_valid = bool(
                    _compression_state == "BREAKOUT_CONFIRMED"
                    and _compression_box.get("live_valid") is True
                    and _breakout_direction == _signal_breakout_direction
                )
                _compression_allows_scale = bool(
                    _compression_state == "UNRESTRICTED"
                    or _confirmed_breakout_valid
                )
                _strict_trend = bool(
                    _compression_allows_scale
                    and is_trending_regime(str(regime or ""))
                    and regime_aligns(str(regime or ""), best_sig.signal_type)
                )
                cascade_scale = bool(cascade_scale and _strict_trend)
                heavy_trend_scale = bool(heavy_trend_scale and _strict_trend)
                _effective_compression_state = (
                    "REVERSION_LOCKED"
                    if _compression_state == "BREAKOUT_CONFIRMED"
                    and not _confirmed_breakout_valid
                    else _compression_state
                )
                if scanner_engine_mode == "one_plus_one":
                    # Native admission uses the operator's configured lot count exactly.
                    # No compression, regime, grade, intelligence, or risk-budget sizer
                    # may resize or veto a scanner-owned entry.
                    size_result = SimpleNamespace(
                        lots=max(0, int(requested_lots or 0)),
                        mode="native_fixed",
                        reason="1+1 configured fixed lots",
                    )
                else:
                    size_result = resolve_trade_lots(
                        settings=settings,
                        inst_type=inst_type,
                        requested_lots=requested_lots,
                        lot_size=lot_size,
                        entry_price=float(live_trade_price or 0.0),
                        stop_price=float(candidate_stop or 0.0),
                        cap_lots=cap_based_lots if inst_type == "OPT" else None,
                        cascade_scale=cascade_scale,
                        heavy_trend_scale=heavy_trend_scale,
                        high_conviction=False,
                        compression_state=_effective_compression_state,
                        intel_score=float(intel_score or 0.0),
                    )
                if size_result.lots < 1:
                    logger.info(
                        "Skipping %s %s %s: %s",
                        instrument,
                        best_tf,
                        inst_type,
                        size_result.reason,
                    )
                    continue
                actual_lots = size_result.lots
                setattr(best_sig, "lot_sizing_mode", str(size_result.mode or "fixed"))
                setattr(best_sig, "sizing_compression_state", _effective_compression_state)
                if size_result.mode == "cascade":
                    candidate_reasons.append(
                        f"Cascade lot scale: {actual_lots} lot(s) "
                        f"(base {requested_lots}; OPT 2–3× max, FUT 1×)"
                    )
                elif size_result.mode == "heavy_trend":
                    candidate_reasons.append(
                        f"Heavy-trend lot scale: {actual_lots} lot(s) "
                        f"(base {requested_lots}; OPT 2× only, 2/4 index align)"
                    )
                else:
                    candidate_reasons.append(
                        f"Fixed lot sizing: {actual_lots} lot(s) from settings "
                        f"(1×; auto risk sizing OFF; stop from settings SL%)"
                    )

                candidate = TradeCandidate(
                    instrument=instrument, direction=direction, price=live_trade_price,
                    stop=candidate_stop, target=candidate_target, lots=actual_lots, lot_size=lot_size, grade=best_grade,
                    confidence=best_conf, timeframe=best_tf, inst_type=inst_type, option_type=option_type_local,
                    atm_strike=strike_local, multiplier=instrument_multiplier, trading_symbol=trading_symbol,
                    symbol_token=symbol_token, rr=round(final_rr, 2), score=final_strike_score,
                    reasons=candidate_reasons, signal_timestamp=best_sig.timestamp,
                    decision_timestamp=datetime.now(IST),
                    entry_spot=float(spot or 0.0),
                    spot_stop=float(entry_stop or 0.0),
                    spot_target=float(target or 0.0),
                    underlying_entry=float(spot or 0.0),
                    underlying_stop=float(entry_stop or 0.0),
                    underlying_target=float(target or 0.0),
                    premium_entry=float(live_trade_price or 0.0) if inst_type == "OPT" else 0.0,
                    premium_stop=float(candidate_stop or 0.0) if inst_type == "OPT" else 0.0,
                    premium_target=float(candidate_target or 0.0) if inst_type == "OPT" else 0.0,
                    geometry_observed_at=(
                        best_sig.timestamp.isoformat()
                        if isinstance(best_sig.timestamp, datetime)
                        else str(best_sig.timestamp or "")
                    ),
                    geometry_source="signal_processor.dual_geometry",
                    geometry_verified=True,
                    status=candidate_status,
                    action=candidate_action,
                    watch_only=s12_signal_only,
                    volume_spike_entry=volume_spike_entry,
                    market_ltp_shock_entry=market_ltp_shock_entry,
                    late_session_reclaim_entry=late_session_reclaim_entry,
                    one_way_cascade_entry=one_way_cascade_entry,
                    volume_ratio=float(volume_spike_metrics.get("volume_ratio", 0.0) or 0.0),
                    volume_basis=str(volume_spike_metrics.get("volume_basis", "") or ""),
                    range_atr_ratio=float(volume_spike_metrics.get("range_atr_ratio", 0.0) or 0.0),
                    close_location=float(volume_spike_metrics.get("close_location", 0.0) or 0.0),
                    initial_stop=float(candidate_stop or 0.0),
                    source_engine=best_source_engine,
                    source_engines=list(best_source_engines),
                    source_lane=best_source_lane,
                    quad_union_entry=bool(quad_union_entry),
                    raw_union_entry=bool(quad_union_entry),
                    tight_stop_pct=(
                        float(
                            (
                                getattr(settings, "one_way_cascade_options_sl_pct", 5.0)
                                if one_way_cascade_entry
                                else getattr(settings, "late_session_reclaim_options_sl_pct", 5.0)
                                if late_session_reclaim_entry
                                else getattr(settings, "volume_spike_entry_options_sl_pct", 5.0)
                            )
                            if inst_type == "OPT"
                            else effective_fut_sl_pct
                        )
                        if (
                            volume_spike_entry
                            or market_ltp_shock_entry
                            or late_session_reclaim_entry
                            or one_way_cascade_entry
                            or is_reversion_scalp
                        )
                        else 0.0
                    ),
                )
                candidate.quote_verified = bool(inst_type != "OPT" or live_trade_price > 0)
                candidate.quote_source = (
                    str(getattr(quote_probe, "quote_source", "") or "broker_ltp")
                    if inst_type == "OPT"
                    else "contract_ltp"
                )
                carry_grading_telemetry(candidate, best_sig)
                setattr(candidate, "source_signal_type", best_sig.signal_type)
                setattr(candidate, "one_plus_one", scanner_engine_mode == "one_plus_one")
                setattr(candidate, "native_scanner", str(getattr(best_sig, "native_scanner", "") or ""))
                for _native_field in (
                    "native_stop", "native_stop2", "native_target1", "native_target2",
                    "native_target3", "native_trail_start_r", "native_trail_atr",
                ):
                    if hasattr(best_sig, _native_field):
                        setattr(candidate, _native_field, getattr(best_sig, _native_field))
                setattr(candidate, "source_signal_price", float(best_sig.price or 0.0))
                setattr(candidate, "source_signal_bar_index", int(getattr(best_sig, "bar_index", -1) or -1))
                setattr(candidate, "source_engine", best_source_engine)
                setattr(candidate, "source_engines", list(best_source_engines))
                setattr(candidate, "source_lane", best_source_lane)
                setattr(candidate, "s1_lane", bool(s1_lane))
                if uts_lane_on:
                    for _k in ("uts_trail", "uts_atr"):
                        setattr(candidate, _k, getattr(best_sig, _k, None))
                if s1_lane:
                    setattr(candidate, "s1_guard_pct", float(s1_guard))
                    setattr(candidate, "s1_weak_target", bool(s1_weak_target))
                    for _k in ("s1_score", "s1_setup", "s1_sl_pct", "s1_sl1", "s1_targets",
                               "s1_chart_entry", "s1_chart_signal_close", "s1_chart_sl1", "s1_chart_sl2", "s1_chart_targets"):
                        setattr(candidate, _k, getattr(best_sig, _k, None))
                if s12_lane_on:
                    for _k in (*s12_lane_policy.SESSION_STATE_FIELDS, "s12_stop_updates", "s12_engine_entry", "s12_fill_cancel"):
                        if hasattr(best_sig, _k):
                            setattr(candidate, _k, copy.deepcopy(getattr(best_sig, _k)))
                    for _k, _v in s12_lane_policy.admission_levels(best_sig, spot, settings).items():
                        setattr(candidate, _k, _v)
                    candidate.s12_lane = s12_lane
                    candidate.s12_premium_cap_pct = float(s1_guard)
                setattr(candidate, "quad_union_entry", bool(quad_union_entry))
                setattr(candidate, "raw_union_entry", bool(quad_union_entry))
                setattr(candidate, "late_session_reclaim_entry", bool(late_session_reclaim_entry))
                if late_session_reclaim_entry and isinstance(_reversal_relief_context, dict):
                    setattr(candidate, "late_session_reclaim_context", dict(_reversal_relief_context))
                setattr(candidate, "one_way_cascade_entry", bool(one_way_cascade_entry))
                setattr(candidate, "lot_sizing_mode", str(size_result.mode or "fixed"))
                setattr(candidate, "sizing_compression_state", _effective_compression_state)
                setattr(candidate, "market_ltp_shock_entry", bool(market_ltp_shock_entry))
                if market_ltp_shock_entry and market_ltp_shock_context:
                    setattr(candidate, "market_ltp_shock_context", dict(market_ltp_shock_context))
                setattr(candidate, "stale_regime_reversal_entry", bool(stale_regime_reversal_entry))
                if stale_regime_reversal_entry:
                    setattr(candidate, "stale_regime_reversal_context", dict(_stale_regime_context))
                setattr(
                    candidate,
                    "market_wide_reversal_entry",
                    bool(getattr(best_sig, "market_wide_reversal_entry", False)),
                )
                if getattr(best_sig, "market_wide_reversal_context", None):
                    setattr(
                        candidate,
                        "market_wide_reversal_context",
                        dict(getattr(best_sig, "market_wide_reversal_context") or {}),
                    )
                setattr(candidate, "is_reversion_scalp", bool(is_reversion_scalp))
                setattr(
                    candidate,
                    "one_min_raw_entry",
                    bool(getattr(best_sig, "one_min_raw_entry", False)),
                )
                if one_way_cascade_entry and isinstance(_reversal_relief_context, dict):
                    setattr(candidate, "one_way_cascade_context", dict(_reversal_relief_context))
                setattr(candidate, "trend_15m_agrees", bool(trend_15m_agrees))
                # Stamp the intelligence that GRADED this candidate, not the 5-minute
                # snapshot in scope. No-op while per-timeframe grading is off.
                _stamp_regime, _stamp_intel_norm = self._graded_candidate_context(
                    best_sig, regime, intel_norm
                )
                setattr(candidate, "intelligence_score", float(_stamp_intel_norm))
                candidate_intel_result = self._graded_candidate_intel_result(
                    best_sig, intel_result
                )
                if isinstance(candidate_intel_result, dict):
                    setattr(
                        candidate,
                        "market_intelligence_context",
                        {
                            "aggregate": dict(candidate_intel_result.get("aggregate", {}) or {}),
                            "regime": dict(candidate_intel_result.get("regime", {}) or {}),
                            "volume": dict(candidate_intel_result.get("volume", {}) or {}),
                            "order_flow": dict(candidate_intel_result.get("order_flow", {}) or {}),
                            "session_context": dict(candidate_intel_result.get("session_context", {}) or {}),
                            "anchor_regime": dict(candidate_intel_result.get("anchor_regime", {}) or {}),
                            "breadth_context": dict(candidate_intel_result.get("breadth_context", {}) or {}),
                            "top_down_context": dict(candidate_intel_result.get("top_down_context", {}) or {}),
                            "market_structure": dict(candidate_intel_result.get("market_structure", {}) or {}),
                            "aroon": dict(candidate_intel_result.get("aroon_context", {}) or {}),
                            "wavetrend": dict(candidate_intel_result.get("wavetrend", {}) or {}),
                        },
                    )
                # One stamp for both outcomes. Rejections are stamped in
                # _build_rejected_from_raw; without the same call here an ACCEPTED row
                # would carry revision and provenance but no build_fingerprint, which is
                # the same unattributable gap the rejections had, only inverted.
                for _field, _value in build_identity(settings, scanner).items():
                    setattr(candidate, _field, _value)
                if iv_percentile is not None:
                    setattr(candidate, "iv_percentile", float(iv_percentile))
                setattr(candidate, "option_premium_ok", bool(option_premium_ok))
                setattr(candidate, "chain_quality", {
                    **dict(chain_quality or {}), "captured_at_epoch": time.time(),
                })
                setattr(candidate, "market_regime", _stamp_regime)
                # Stamp this worker's decision context. Another instrument can
                # overwrite the scanner's diagnostic snapshot before construction.
                _pa = _pa_ctx
                setattr(candidate, "price_action_verdict", "")
                setattr(candidate, "price_action_score", 0.0)
                if isinstance(_pa, dict):
                    setattr(candidate, "price_action_verdict", str(_pa.get("verdict") or ""))
                    setattr(candidate, "price_action_score", float(_pa.get("score") or 0.0))
                    profile_note = str(_pa.get("target_context") or "")
                    trail_note = str(_pa.get("trailing_context") or "")
                    if profile_note or trail_note:
                        candidate.reasons = list(candidate.reasons or []) + [
                            f"Market-structure context: {profile_note}; {trail_note}"
                        ]
                candidate.adx_value = adx_value
                candidate.impulse_confirmed = bool(impulse_confirmed)
                candidate.is_explosive_bypass = bool(is_explosive_bypass)
                # Power-move: bypass grade floor / late cutoff downstream, enter tight (scalp-lock).
                candidate.power_move = bool(power_move)
                if fast_thrust_override and not s1_lane:
                    candidate.scalp_lock_mode = True
                    candidate.reasons = list(candidate.reasons or []) + [
                        "⚡ Fast thrust (data-confirmed): tight stop + scalp-lock risk"
                    ]
                if late_session_reclaim_entry and not s1_lane:
                    candidate.scalp_lock_mode = True
                if market_ltp_shock_entry:
                    candidate.scalp_lock_mode = True
                    candidate.reasons = list(candidate.reasons or []) + [
                        "⚡ Market LTP shock (multi-index): tight stop + short stab + scalp-lock"
                    ]
                if is_reversion_scalp and not s1_lane:
                    candidate.scalp_lock_mode = True
                    _rev_rr = max(
                        1.5,
                        float(getattr(settings, "post_exit_reversion_min_rr", 1.5) or 1.5),
                    )
                    candidate.reasons = list(candidate.reasons or []) + [
                        f"⚡ Post-exit snapback scalp: RR {_rev_rr:.2f}+, tight stop, scalp-lock"
                    ]
                if one_way_cascade_entry:
                    candidate.scalp_lock_mode = False
                # ── Weak-market scalp mode ──
                # On a confirmed range day a non-burst entry is a scalp, not a
                # runner: arm scalp-lock trailing from entry and tighten the RR
                # target so the book beats charges instead of round-tripping
                # toward a 2.5-4.5R target that chop never pays (2026-07-03).
                # Bursts (thrust / market-wide reversal / explosive) run free.
                try:
                    _wm_ctx = (
                        intel_result.get("session_context", {})
                        if isinstance(intel_result, dict)
                        else {}
                    )
                    _wm_day = str(
                        _wm_ctx.get("effective_day_type")
                        or _wm_ctx.get("market_phase")
                        or _wm_ctx.get("day_type")
                        or ""
                    ).upper()
                    _wm_conf = float(
                        _wm_ctx.get("effective_confidence")
                        or _wm_ctx.get("phase_confidence")
                        or _wm_ctx.get("confidence")
                        or 0.0
                    )
                    if (
                        scanner_engine_mode != "one_plus_one"
                        and
                        _setting_enabled(getattr(settings, "weak_market_scalp_mode_enabled", True), True)
                        and _wm_day == "RANGE_DAY"
                        and _wm_conf
                        >= float(getattr(settings, "weak_market_scalp_min_day_confidence", 0.60) or 0.60)
                        and not fast_thrust_override
                        and not bool(getattr(candidate, "s1_lane", False))
                        and not bool(getattr(candidate, "market_wide_reversal_entry", False))
                        and not bool(getattr(candidate, "one_way_cascade_entry", False))
                        and not candidate.is_explosive_bypass
                    ):
                        candidate.scalp_lock_mode = True
                        _wm_max_rr = float(getattr(settings, "weak_market_scalp_max_rr", 1.5) or 1.5)
                        _wm_risk = abs(float(candidate.price or 0.0) - float(candidate.stop or 0.0))
                        _wm_span = abs(float(candidate.target or 0.0) - float(candidate.price or 0.0))
                        if _wm_risk > 0 and _wm_span > _wm_risk * _wm_max_rr:
                            _wm_sign = 1.0 if float(candidate.target or 0.0) >= float(candidate.price or 0.0) else -1.0
                            candidate.target = float(candidate.price) + _wm_sign * _wm_risk * _wm_max_rr
                            if hasattr(candidate, "rr"):
                                candidate.rr = _wm_max_rr
                        candidate.reasons = list(candidate.reasons or []) + [
                            f"Weak-market scalp: range day conf {_wm_conf:.0%} -> scalp-lock + RR capped {_wm_max_rr:.1f}"
                        ]
                except Exception as _wm_exc:
                    logger.debug(f"Weak-market scalp mode skipped for {instrument}: {_wm_exc}")
                if candidate.is_explosive_bypass:
                    candidate.reasons = list(candidate.reasons or []) + [
                        f"Confirmed impulse bypass ADX={adx_value:.1f}, intel={normalize_intelligence_score(intel_score):.2f}"
                    ]
                # Tradeability of THIS strike, for the near-tie break below.
                _liquidity = self._strike_liquidity_snapshot(
                    _liquidity_chain, strike_local, option_type_local
                )
                candidate.strike_oi = float(_liquidity["oi"])
                candidate.strike_volume = float(_liquidity["volume"])
                candidate.strike_oi_available = bool(_liquidity["oi_available"])
                candidate.strike_volume_available = bool(
                    _liquidity["volume_available"]
                )
                candidate.strike_liquidity_available = bool(
                    _liquidity["available"]
                )
                # The ITM bonus is a standing PREFERENCE for ITM, not a measurement of
                # this contract. Carried separately so the tie-break can compare the two
                # candidates on measured quality alone: at 0.15 it is three times the
                # default 0.05 band, so leaving it inside the score meant the ATM and ITM
                # scores were never "near" and the liquidity comparison never ran.
                setattr(candidate, "strike_preference_bonus", float(itm_bonus))
                results_list.append(candidate)
                logger.info(
                    f"Resolved candidate: {instrument} {best_tf} {best_sig.signal_type} -> "
                    f"{direction} {inst_type} {option_type_local or 'FUT'} {trading_symbol}"
                    + (" | no-entry display only" if effective_no_entry else "")
                    + (" | ⚡power-move" if power_move else "")
                    + (" | ⚡volume-spike" if volume_spike_entry else "")
                    + (" | ⚡ltp-shock" if market_ltp_shock_entry else "")
                    + (" | late-directional-move" if late_session_reclaim_entry else "")
                    + (" | one-way-cascade" if one_way_cascade_entry else "")
                )

        return self._finalize_signal_candidates(
            instrument=instrument,
            best_tf=best_tf,
            results_list=results_list,
            scored=scored,
            settings=settings,
            scanner_engine_mode=scanner_engine_mode,
        )
