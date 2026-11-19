"""
Trade Manager - Position Lifecycle with Smart Exits
===============================================================

Handles:
1. Trade open/close with full audit trail
2. 1-min trailing stop updates
3. Automatic trailing stop exit detection
4. Session-end auto-exit
5. P&L tracking
"""

import uuid
import copy
import inspect
import math
import re
import asyncio
import threading
import time
import json
from pathlib import Path
from datetime import date, datetime, time as dtime, timedelta
import pytz
IST = pytz.timezone('Asia/Kolkata')

# Postback states after which an order is finished. A live-state payload arriving after
# one of these is a late redelivery and must not reopen the order's recorded state.
_POSTBACK_TERMINAL_STATUSES = frozenset({"COMPLETE", "CANCELLED", "REJECTED"})

# Direction each leg moves the holding, for the SIGNED fill ledger. Adding or removing a
# leg here without its sign silently corrupts _ledger_owned_quantity, which is summed.
_POSTBACK_LEG_SIGNS = {
    "ENTRY": 1,
    "ADD": 1,
    "EXIT": -1,
    "PROTECTIVE": -1,
    "PARTIAL_EXIT": -1,
}

from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, replace
from loguru import logger
from data.sqlite_db import db
from trading.performance_metrics import annualize_sharpe, format_sharpe_pair
from trading.trade_dashboard import build_dashboard_rows
from trading.broker import order_filled_quantity
from trading.order_retry import (
    execute_order_with_retry,
    BrokerOrderAmbiguous,
    BrokerOrderRejected,
    broker_rejection,
)


def sample_pnl_path(holder, pnl=None, entry_time=None) -> None:
    """Append a bounded (seconds_since_entry, pnl) sample to `holder.pnl_path`.

    Shared by the two objects that independently track a live position's P&L: the Trade
    in this module and the TradeCandidate the scanner monitors (which is what rtracker
    actually records). One implementation, because two would drift and the whole point
    of the path is that it can be trusted.

    Called from monitor loops that run every ~100ms, so it is written to do nothing in
    the common case. Three bounds keep it cheap and the record small:

      * TIME    — at most one sample per pnl_path_min_interval_seconds
      * MOTION  — unless P&L moved by pnl_path_min_move_rs, which keeps a fast reversal
                  legible instead of letting the timer average it away
      * SIZE    — a hard cap; first and last samples are always kept and the middle is
                  thinned by dropping every other one, so a long trade loses resolution
                  rather than losing its tail

    Median hold in the measured book is 1.2 minutes, so at the 2s default a typical
    trade stores ~36 samples and the cap is never approached.
    """
    from config.settings import get_settings

    settings = get_settings()
    if not _setting_enabled(getattr(settings, "pnl_path_logging_enabled", True), True):
        return
    try:
        if entry_time is None:
            entry_time = getattr(holder, "entry_time", None) or getattr(
                holder, "entry_timestamp", None
            )
        if entry_time is None:
            return
        if pnl is None:
            pnl = getattr(holder, "pnl", 0.0)
        pnl = float(pnl or 0.0)

        path = getattr(holder, "pnl_path", None)
        if not isinstance(path, list):
            path = []
            setattr(holder, "pnl_path", path)

        ref = entry_time if getattr(entry_time, "tzinfo", None) else IST.localize(entry_time)
        elapsed = max(0.0, (datetime.now(IST) - ref).total_seconds())

        if path:
            last_t, last_pnl = path[-1]
            min_gap = float(getattr(settings, "pnl_path_min_interval_seconds", 2.0) or 2.0)
            min_move = float(getattr(settings, "pnl_path_min_move_rs", 50.0) or 50.0)
            if (elapsed - last_t) < min_gap and abs(pnl - last_pnl) < min_move:
                return

        path.append((round(elapsed, 2), round(pnl, 2)))

        cap = int(getattr(settings, "pnl_path_max_samples", 400) or 400)
        if cap > 2 and len(path) > cap:
            # Keep both ends; halve the resolution of everything between them.
            setattr(holder, "pnl_path", [path[0]] + path[1:-1:2] + [path[-1]])
    except Exception as exc:
        # Diagnostics must never interfere with a live position.
        logger.debug(f"P&L path sample skipped: {exc}")


def sample_intel_path(holder, snapshot_fn, entry_time=None) -> None:
    """Append a bounded (seconds_since_entry, intel snapshot) row to `holder.intel_path`.

    The market-side twin of sample_pnl_path. pnl_path says what the POSITION did; this
    says what the MARKET was doing while it did it, so the two can be replayed together.

    `snapshot_fn` is a zero-argument callable, not a dict, and that is deliberate: the
    timer is checked BEFORE it is invoked, so building the snapshot costs nothing on the
    ~9 cycles out of 10 that are not due. The caller supplies it because the snapshot
    lives on the scanner (Scanner._entry_intel_snapshot) while the bounding logic belongs
    next to the path it mirrors -- one definition of each, neither importing the other.

    Two bounds, both cheap:

      * TIME   -- at most one snapshot per intel_path_min_interval_seconds
      * CHANGE -- and it is only KEPT if it differs from the last row kept. PCR changes
                  every ~99s against a ~103s median hold, so an unfiltered timer would
                  store the same numbers over and over. Storing only transitions makes
                  the series both small and exactly what a replay needs to read.

    A hard SIZE cap thins the middle and keeps both ends, matching sample_pnl_path, so a
    long position loses resolution rather than losing its tail.

    Observation only: every failure is swallowed. A diagnostic may never cost a position.
    """
    from config.settings import get_settings

    settings = get_settings()
    if not _setting_enabled(getattr(settings, "intel_path_logging_enabled", True), True):
        return
    try:
        if entry_time is None:
            entry_time = getattr(holder, "entry_time", None) or getattr(
                holder, "entry_timestamp", None
            )
        if entry_time is None:
            return

        path = getattr(holder, "intel_path", None)
        if not isinstance(path, list):
            path = []
            setattr(holder, "intel_path", path)

        ref = entry_time if getattr(entry_time, "tzinfo", None) else IST.localize(entry_time)
        elapsed = max(0.0, (datetime.now(IST) - ref).total_seconds())

        min_gap = float(getattr(settings, "intel_path_min_interval_seconds", 5.0) or 5.0)
        if path and (elapsed - path[-1][0]) < min_gap:
            return

        snapshot = snapshot_fn() if callable(snapshot_fn) else snapshot_fn
        if not isinstance(snapshot, dict) or not snapshot:
            return
        # Only transitions. An identical snapshot carries no information the previous row
        # does not already carry, and the timestamp of a NON-change is not interesting.
        if path and path[-1][1] == snapshot:
            return

        path.append((round(elapsed, 2), snapshot))

        cap = int(getattr(settings, "intel_path_max_samples", 60) or 60)
        if cap > 2 and len(path) > cap:
            setattr(holder, "intel_path", [path[0]] + path[1:-1:2] + [path[-1]])
    except Exception as exc:
        # Diagnostics must never interfere with a live position.
        logger.debug(f"Intel path sample skipped: {exc}")


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


@dataclass
class Trade:
    id: str
    instrument: str
    timeframe: str
    direction: str          # LONG or SHORT
    entry_price: float
    entry_time: datetime
    trailing_stop: float    # Initial trailing stop at entry
    current_stop: float     # Current trailing stop (1-min updated)
    lots: int
    lot_size: int
    grade: str
    # ═══ Execution Details ═══
    broker_order_id: str = ""
    pending_execution: bool = False
    signal_time: Optional[datetime] = None
    entry_submitted_at: Optional[datetime] = None
    entry_replace_count: int = 0
    # How many times the remainder of a partially filled exit has been re-submitted after
    # the original exit order died. Bounded so a contract the broker keeps rejecting
    # cannot spin a fresh market order on every reconcile pass.
    partial_exit_flatten_count: int = 0
    # Deliberate scale-out state. partial_derisk_done latches only on a BOOKED success, so
    # a reduce that failed can be retried on the next tick; the other two persist an
    # in-flight reduce so a restart reconciles it from broker positions instead of
    # submitting a second one against the same lots.
    partial_derisk_done: bool = False
    partial_reduce_order_id: str = ""
    partial_reduce_requested_quantity: int = 0
    # Deliberate scale-IN state, the mirror of the three above.
    #   scale_in_count   — how many add-ons this trade has taken, against the per-trade cap.
    #   scale_in_lots    — lots added since entry, so the original size stays recoverable.
    #   frozen_initial_risk_rs — the rupee risk committed at ENTRY, held across every
    #     add-on. Without it, averaging the entry price up moves the stop distance and
    #     therefore silently redefines 1R: a trade up 1.5R reads as 0.9R the moment lots
    #     are added, and every R-based exit rule changes behaviour because of a decision
    #     the operator made rather than anything the market did.
    #   scale_in_order_id / scale_in_requested_quantity — an in-flight add, persisted so a
    #     restart reconciles it from broker positions instead of sending a second one.
    scale_in_count: int = 0
    scale_in_lots: int = 0
    frozen_initial_risk_rs: float = 0.0
    scale_in_order_id: str = ""
    scale_in_requested_quantity: int = 0
    exit_order_id: str = ""
    protective_order_id: str = ""
    protective_trigger: float = 0.0
    protective_quantity: int = 0
    # Zerodha GTT trigger id. Armed ONLY as a fallback when no exchange-side SL-M could
    # be established, never alongside a working one: an SL-M and a GTT that both fire
    # exit the position twice, and the second fill opens a naked reversed leg.
    gtt_trigger_id: str = ""
    protective_updated_at: Optional[datetime] = None
    recovery_required_since: Optional[datetime] = None
    # Set when a flatten went out while a protective stop MAY still have been resting.
    # If that stop later fires it opens a position in the opposite direction, so this
    # contract needs watching until reconciliation confirms it is flat.
    reversal_watch_required: bool = False
    execution_status: str = "OPEN"
    execution_error: str = ""
    pending_exit: bool = False
    broker_quantity: int = 0
    trading_symbol: str = ""
    symbol_token: str = ""
    # ═══ NEW: Instrument & signal details ═══
    atm_strike: float = 0.0
    option_type: str = ""   # CE or PE
    target: float = 0.0     # Target price (1.5× risk)
    rr_ratio: float = 1.5   # Risk-reward ratio
    confidence: float = 0.0 # Signal confidence 0.0-1.0
    inst_type: str = "FUT"  # FUT or OPT
    instrument_multiplier: float = 1.0 # 1.0 for FUT, Delta (~0.5) for OPT
    exec_type: str = "A"    # A for Auto, M for Manual
    status: str = "OPEN"    # OPEN / CLOSED
    current_price: float = 0.0
    entry_spot: float = 0.0  # Spot price at entry (for option tracking)
    spot_stop: float = 0.0   # Original index/spot stop for premium-led option trades
    spot_target: float = 0.0 # Original index/spot target for premium-led option trades
    # Canonical dual-price geometry. Legacy entry_spot/spot_* mirror only the
    # underlying tuple; premium values must never enter those aliases.
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
    # Forecast and final-policy evidence are scalar so SQLite can retain every fact
    # without serialising opaque mutable dictionaries.
    expected_move_pct: float = 0.0
    expected_move_floor_pct: float = 0.0
    expected_move_uncertainty_pct: float = 0.0
    expected_move_probability: float = 0.0
    expected_move_sample_count: int = 0
    expected_move_horizon_minutes: int = 0
    expected_move_generated_at: str = ""
    expected_move_data_as_of: str = ""
    expected_move_model_version: str = ""
    expected_move_source: str = ""
    entry_policy_allowed: bool = False
    entry_policy_phase: str = ""
    entry_policy_checked_at: str = ""
    entry_policy_reason: str = ""
    entry_policy_actual_rr: float = 0.0
    entry_policy_underlying_stop_pct: float = 0.0
    entry_policy_version: str = ""
    # Full versioned signal-quality evidence, kept as JSON-compatible data so the
    # ACCEPTED decision and eventual EXIT can be attributed to the same gate route.
    quality_verdict: Dict[str, Any] = field(default_factory=dict)
    decision_record: Dict[str, Any] = field(default_factory=dict)
    # Why the position was closed (engine/exit_decision_record.py): owner, controlling
    # reason, profit-lock trace and proposals a HOLD suppressed.
    exit_decision_record: Dict[str, Any] = field(default_factory=dict)
    # Frozen entry evidence used for exit comparison and offline outcome calibration.
    # These used to be runtime-only setattrs and disappeared on restart.
    entry_intel: Dict[str, Any] = field(default_factory=dict)
    decision_intel: Dict[str, Any] = field(default_factory=dict)
    calibration_outcome_recorded: bool = False
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    pnl: float = 0.0
    charges: float = 0.0
    # Kite's position.realised after this close. It is cumulative per contract/day,
    # so persisting the consumed total on the same CLOSED row lets a same-day process
    # restart restore the delta baseline without booking earlier P&L a second time.
    broker_realised_cumulative: Optional[float] = None
    peak_pnl: float = 0.0
    max_drawdown: float = 0.0
    # Bounded (seconds_since_entry, pnl) samples. peak_pnl and max_drawdown are the only
    # shape a closed trade has ever recorded, and from those three numbers the question
    # "would a tighter trail have kept more?" is not answerable: a rule that exits at
    # f x peak fires only if the P&L fell to that level AFTER the peak, and the ORDER of
    # the peak and the trough is exactly what a max cannot carry. On the 2026-08-11 book
    # that gap put the answer between -Rs.18,114 and +Rs.10,988 — a range wide enough to
    # be useless. The path closes it. See _sample_pnl_path for the cost bound.
    pnl_path: List[Tuple[float, float]] = field(default_factory=list)
    # Units of pnl_path: gross before round-trip cost, the same basis as peak_pnl and the
    # TradeCandidate path (rtracker's PNL_BASIS_GROSS). The path is not persisted, so a
    # restored trade restarts it on this basis. Net before 2026-09-17.
    pnl_path_basis: str = "GROSS_BEFORE_ROUND_TRIP_COST"
    # Historical-only point-in-time audit evidence. Empty on live trades. Keeping it
    # on the trade row lets backtest reports prove which guards were evaluated and
    # which minute path produced the exit instead of exposing only the final P&L.
    replay_price_path: List[Dict[str, Any]] = field(default_factory=list)
    replay_exit_trace: List[Dict[str, Any]] = field(default_factory=list)
    is_ghost: bool = False
    # True only when this row reconstructs exposure that already existed at the
    # broker. Persist it so a restart cannot mistake recovery reconciliation for a
    # fresh fill and emergency-flatten it for lacking new-entry forecast evidence.
    is_recovery: bool = False
    is_reversion_scalp: bool = False
    signal_fire_mode: str = "INTRACANDLE"
    entered_intracandle: bool = True
    uts_confirmed: bool = False
    uts_repaint_exit_required: bool = False
    # Has the post-entry repaint check completed for this trade? PERSISTED, because it
    # was previously only a dynamic setattr: a restart lost it, and an intracandle trade
    # then rehydrated as "structurally unconfirmed" for the rest of the day — which
    # disables let-winners-run and arms the blunt sub-1R give-back guard on a position
    # whose source candle closed hours earlier.
    repaint_guard_checked: bool = False
    # A GTT was armed without being able to prove no prior trigger existed, or its
    # placement came back ambiguous. Either way a duplicate server-side trigger may be
    # resting against this contract with no local id, and two triggers firing exits
    # the position twice — the second fill opening a naked reversed leg.
    gtt_arm_uncertain: bool = False
    market_regime: str = ""   # Regime at entry (for post-hoc attribution)
    data_provenance: str = "" # LIVE / HISTORICAL / RECOVERY / EOD — data source of the entry
    price_action_verdict: str = "" # FAVORABLE/NEUTRAL/ADVERSE from the price-action patch (advisory)
    # Entry-time attribution only.  Exit decisions must use the fresh intelligence
    # context supplied by the scanner, not this frozen telemetry snapshot.
    entry_intelligence_score: float = 0.0
    # Backward-compatible serialized alias used by older archives/readers.
    intelligence_score: float = 0.0
    strategy_revision: str = ""
    strategy_direction_mode: str = ""
    market_wide_reversal_entry: bool = False
    volume_spike_entry: bool = False
    power_move: bool = False
    source_engine: str = ""
    source_engines: List[str] = field(default_factory=list)
    source_lane: str = ""
    # Exit-policy owner frozen at admission ("S1"/"UTS"/"GENERIC"; "" = legacy row).
    # Management reads this, never the lane admission switches. See s1_lane.
    exit_policy_owner: str = ""
    s1_best_spot: Optional[float] = None
    s1_t1_locked: bool = False
    s1_peak_premium: Optional[float] = None
    quad_union_entry: bool = False
    raw_union_entry: bool = False
    initial_stop: float = 0.0
    scalp_lock_mode: bool = False
    # Runner state changes exit semantics: while armed, the fixed target is no longer
    # allowed to close the trade (see _check_exit_conditions).  These therefore cannot
    # remain dynamic setattrs that disappear on a process restart.
    lifecycle_id: str = ""
    runner_mode: bool = False
    # Contract-price distance frozen when the hard target is removed. Retain it
    # across restart so runner protection does not revert to a planned RR.
    original_target_distance: float = 0.0
    runner_activation_timestamp: Optional[datetime] = None
    runner_activation_reason: str = ""
    runner_stop: float = 0.0
    # Unit/price-space discriminator for runner_stop.  Options use different
    # spaces in the session-candidate and live-Trade managers, so the number is
    # not auditable without this field.
    runner_stop_basis: str = ""
    stop_provenance: str = ""
    # ═══ Realized slippage (REAL broker fills only) ═══
    signal_entry_price: float = 0.0  # pre-fill signal/decision price (slippage reference)
    entry_slippage: float = 0.0      # adverse ₹/unit paid vs signal at entry fill
    exit_slippage: float = 0.0       # adverse ₹/unit vs exit decision price at exit fill
    # ── How the order filled, not merely that it did ──
    # The trades table has carried these six columns for some time and NOTHING wrote
    # them: no code anywhere referenced requested_entry_price, requested_exit_price,
    # fill_latency_ms, exit_fill_latency_ms, entry_filled_at or exit_filled_at. They
    # were schema with no producer, so every row read 0/NULL.
    #
    # They matter because entry_price is OVERWRITTEN by the broker fill on
    # confirmation, and signal_entry_price -- the reference slippage is measured
    # against -- is not persisted at all. Once a row is saved, the price that was asked
    # for is gone, so realised slippage cannot be recomputed from the stored history:
    # only a live in-memory EWMA could ever see it, and one restart erased the evidence.
    # Storing the requested price beside the filled one makes each row self-sufficient:
    # entry_price - requested_entry_price IS the entry slippage, re-derivable forever.
    requested_entry_price: float = 0.0   # price the entry order was sent at
    requested_exit_price: float = 0.0    # decision price the exit was sent at
    entry_filled_at: Optional[datetime] = None
    exit_filled_at: Optional[datetime] = None
    fill_latency_ms: float = 0.0         # entry submit -> fill confirmation
    exit_fill_latency_ms: float = 0.0    # exit submit -> fill confirmation
    execution_bid_price: float = 0.0
    execution_ask_price: float = 0.0
    execution_touch_source: str = ""
    execution_touch_age_seconds: float = 0.0
    execution_touch_usable: bool = False
    execution_touch_reason: str = ""

    @property
    def stop_loss(self) -> float:
        """Alias used by UI/risk helpers: prefer live stop, then trail, then initial."""
        # Contract-space property: never fall through to an underlying index level.
        for attr in ("current_stop", "trailing_stop", "initial_stop"):
            try:
                val = float(getattr(self, attr, 0.0) or 0.0)
            except Exception:
                val = 0.0
            if val > 0:
                return val
        return 0.0

    @stop_loss.setter
    def stop_loss(self, value: float) -> None:
        try:
            v = float(value or 0.0)
        except Exception:
            return
        if v <= 0:
            return
        self.current_stop = v
        if not float(getattr(self, "trailing_stop", 0.0) or 0.0):
            self.trailing_stop = v

    # (deduped 2026-07-10: a second `quantity` property later in the class
    # shadowed a clamped copy here — the ACTIVE definition lives next to
    # unrealized_pnl; behavior unchanged by removing the dead one.)

    def _never_reached_market(self) -> bool:
        """True when this trade demonstrably never became a broker position.

        Deliberately narrow: it requires BOTH a terminal not-filled status AND the
        absence of any broker order id. A trade whose acknowledgement was merely
        ambiguous is NOT covered — that one may well be live at the exchange, and
        zeroing its P&L would hide real money.
        """
        status = str(getattr(self, "execution_status", "") or "").upper()
        reason = str(getattr(self, "exit_reason", "") or "").upper()
        if status not in {"REJECTED"} and reason not in {"BROKER_ERROR", "CONTRACT_MISSING"}:
            return False
        if str(getattr(self, "broker_order_id", "") or "").strip():
            return False
        return int(getattr(self, "broker_quantity", 0) or 0) <= 0

    def to_dict(self) -> Dict:
        """Convert trade to dictionary for dashboard sync"""
        display_name = self.instrument
        if self.inst_type == "OPT" and self.atm_strike is not None and self.atm_strike > 0:
            display_name = f"{self.instrument} {int(self.atm_strike)} {self.option_type}"
            
        current_display_price = self.current_price
        if self.status == "CLOSED":
            current_display_price = (
                self.current_price
                if self.current_price is not None and self.current_price > 0 and abs(self.current_price - self.exit_price) > 1e-9
                else 0.0
            )
            
        return {
            "id": self.id,
            "status": self.status,
            "is_ghost": getattr(self, 'is_ghost', False),
            "is_recovery": bool(getattr(self, "is_recovery", False)),
            "execution_status": self.execution_status,
            "execution_error": self.execution_error,
            "pending_execution": self.pending_execution,
            "signal_time": self.signal_time.isoformat() if self.signal_time else "",
            "signal_timestamp": self.signal_time.isoformat() if self.signal_time else "",
            "entry_submitted_at": self.entry_submitted_at.isoformat() if self.entry_submitted_at else "",
            "entry_replace_count": self.entry_replace_count,
            "partial_exit_flatten_count": self.partial_exit_flatten_count,
            "partial_derisk_done": self.partial_derisk_done,
            "partial_reduce_order_id": self.partial_reduce_order_id,
            "partial_reduce_requested_quantity": self.partial_reduce_requested_quantity,
            "scale_in_count": self.scale_in_count,
            "scale_in_lots": self.scale_in_lots,
            "frozen_initial_risk_rs": self.frozen_initial_risk_rs,
            "scale_in_order_id": self.scale_in_order_id,
            "scale_in_requested_quantity": self.scale_in_requested_quantity,
            # Both of these were absent from the round trip, so a restart silently reset
            # them. reversal_watch_required is the one that matters to an operator: it
            # marks a contract whose stop MAY still be resting after a flatten, and the
            # log that sets it says "watch this contract" — without serialising it there
            # was no surface naming which contract that was.
            "reversal_watch_required": self.reversal_watch_required,
            "is_reversion_scalp": self.is_reversion_scalp,
            "pending_exit": self.pending_exit,
            "broker_order_id": self.broker_order_id,
            "exit_order_id": self.exit_order_id,
            "protective_order_id": self.protective_order_id,
            "protective_trigger": self.protective_trigger,
            "protective_quantity": self.protective_quantity,
            "gtt_trigger_id": self.gtt_trigger_id,
            "protective_updated_at": self.protective_updated_at.isoformat() if self.protective_updated_at else "",
            "recovery_required_since": self.recovery_required_since.isoformat() if self.recovery_required_since else "",
            "broker_quantity": self.broker_quantity,
            # The broker contract identity MUST survive the round-trip. to_dict() is
            # both the dashboard payload and the only serializer save_state() feeds to
            # db.save_trades(), which writes INSERT OR REPLACE — so any column missing
            # here is actively NULLed on every save, not merely left alone. Dropping
            # these two stranded every 2026-07-31 trade with trading_symbol/symbol_token
            # NULL, which is exactly what emergency_flatten_trade() and
            # _ensure_protective_stop() require: a reloaded position could then neither
            # be flattened nor given an exchange-side stop.
            "trading_symbol": self.trading_symbol,
            "symbol_token": self.symbol_token,
            "instrument": display_name,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "current_price": current_display_price,
            "exit_price": self.exit_price if self.status == "CLOSED" else 0.0,
            "trailing_stop": self.trailing_stop if self.trailing_stop is not None else 0.0,
            "current_stop": self.current_stop if self.current_stop is not None else 0.0,
            "target": self.target if self.target is not None else 0.0,
            # A trade the broker never accepted has no P&L — there was no position to
            # gain or lose on. The 2026-07-31 rows carried a phantom -80 because an
            # UNREALISED mark was written into the persisted `pnl` column while the
            # entry was still pending, and it survived onto the closed row after the
            # order turned out to have been rejected.
            "pnl": 0.0 if self._never_reached_market() else round(
                (self.unrealized_pnl if self.status == "OPEN" else self.pnl) or 0.0,
                2,
            ),
            "charges": self.charges,
            "broker_realised_cumulative": self.broker_realised_cumulative,
            "lots": self.lots,
            "lot_size": self.lot_size,
            "grade": self.grade,
            "confidence": round(self.confidence or 0.0, 2),
            "rr_ratio": self.rr_ratio,
            "atm_strike": self.atm_strike,
            "option_type": self.option_type,
            "inst_type": self.inst_type,
            "timeframe": self.timeframe,
            "entry_time": self.entry_time.strftime("%d %b %H:%M:%S") if self.entry_time else "--",
            "exit_time": self.exit_time.strftime("%d %b %H:%M:%S") if self.exit_time else "--",
            "entry_timestamp": self.entry_time.timestamp() if self.entry_time else 0,
            "exit_timestamp": self.exit_time.isoformat() if self.exit_time else "",
            "exit_reason": self.exit_reason,
            "exec_type": self.exec_type,
            "entry_spot": self.entry_spot,
            "spot_stop": self.spot_stop,
            "spot_target": self.spot_target,
            "underlying_entry": self.underlying_entry,
            "underlying_stop": self.underlying_stop,
            "underlying_target": self.underlying_target,
            "premium_entry": self.premium_entry,
            "premium_stop": self.premium_stop,
            "premium_target": self.premium_target,
            "geometry_observed_at": self.geometry_observed_at,
            "geometry_source": self.geometry_source,
            "geometry_verified": self.geometry_verified,
            "geometry_schema": self.geometry_schema,
            "expected_move_pct": self.expected_move_pct,
            "expected_move_floor_pct": self.expected_move_floor_pct,
            "expected_move_uncertainty_pct": self.expected_move_uncertainty_pct,
            "expected_move_probability": self.expected_move_probability,
            "expected_move_sample_count": self.expected_move_sample_count,
            "expected_move_horizon_minutes": self.expected_move_horizon_minutes,
            "expected_move_generated_at": self.expected_move_generated_at,
            "expected_move_data_as_of": self.expected_move_data_as_of,
            "expected_move_model_version": self.expected_move_model_version,
            "expected_move_source": self.expected_move_source,
            # Ephemeral monitor view. Rebuilt from live marks after every restart;
            # intentionally not persisted as immutable entry-policy evidence.
            "live_expected_move_forecast": dict(
                getattr(self, "live_expected_move_forecast", {}) or {}
            ),
            "entry_policy_allowed": self.entry_policy_allowed,
            "entry_policy_phase": self.entry_policy_phase,
            "entry_policy_checked_at": self.entry_policy_checked_at,
            "entry_policy_reason": self.entry_policy_reason,
            "entry_policy_actual_rr": self.entry_policy_actual_rr,
            "entry_policy_underlying_stop_pct": self.entry_policy_underlying_stop_pct,
            "entry_policy_version": self.entry_policy_version,
            "quality_verdict": dict(self.quality_verdict or {}),
            "decision_record": dict(self.decision_record or {}),
            "exit_decision_record": dict(self.exit_decision_record or {}),
            "entry_intel": dict(self.entry_intel or {}),
            "decision_intel": dict(self.decision_intel or {}),
            "calibration_outcome_recorded": bool(self.calibration_outcome_recorded),
            "instrument_multiplier": self.instrument_multiplier,
            "peak_pnl": self.peak_pnl,
            "max_drawdown": self.max_drawdown,
            # Persisted so a mid-session restart does not leave a trade with a path that
            # starts partway through — a truncated path reads as a trade that never
            # visited the levels it actually visited.
            "pnl_path": [[float(t), float(p)] for t, p in (self.pnl_path or [])],
            "replay_price_path": list(self.replay_price_path or []),
            "replay_exit_trace": list(self.replay_exit_trace or []),
            "signal_fire_mode": self.signal_fire_mode,
            "entered_intracandle": self.entered_intracandle,
            "uts_confirmed": self.uts_confirmed,
            "uts_repaint_exit_required": self.uts_repaint_exit_required,
            "repaint_guard_checked": self.repaint_guard_checked,
            "gtt_arm_uncertain": self.gtt_arm_uncertain,
            "market_regime": self.market_regime,
            "data_provenance": self.data_provenance,
            "price_action_verdict": self.price_action_verdict,
            "entry_intelligence_score": self.entry_intelligence_score,
            "intelligence_score": self.intelligence_score,
            "strategy_revision": self.strategy_revision,
            "strategy_direction_mode": self.strategy_direction_mode,
            "market_wide_reversal_entry": self.market_wide_reversal_entry,
            "volume_spike_entry": self.volume_spike_entry,
            "power_move": self.power_move,
            "source_engine": self.source_engine,
            "source_engines": list(self.source_engines or []),
            "source_lane": self.source_lane,
            "exit_policy_owner": self.exit_policy_owner,
            "s1_best_spot": self.s1_best_spot,
            "s1_t1_locked": self.s1_t1_locked,
            "s1_peak_premium": self.s1_peak_premium,
            "quad_union_entry": self.quad_union_entry,
            "raw_union_entry": self.raw_union_entry,
            "initial_stop": self.initial_stop if self.initial_stop is not None else 0.0,
            "scalp_lock_mode": bool(self.scalp_lock_mode),
            "lifecycle_id": str(self.lifecycle_id or self.id or ""),
            "runner_mode": bool(self.runner_mode),
            "original_target_distance": float(self.original_target_distance or 0.0),
            "runner_activation_timestamp": (
                self.runner_activation_timestamp.isoformat()
                if self.runner_activation_timestamp
                else ""
            ),
            "runner_activation_reason": str(self.runner_activation_reason or ""),
            "runner_stop": float(self.runner_stop or 0.0),
            "runner_stop_basis": str(self.runner_stop_basis or ""),
            "stop_provenance": str(self.stop_provenance or ""),
            "signal_entry_price": self.signal_entry_price,
            "entry_slippage": self.entry_slippage,
            "exit_slippage": self.exit_slippage,
            "requested_entry_price": self.requested_entry_price,
            "requested_exit_price": self.requested_exit_price,
            "entry_filled_at": self.entry_filled_at.isoformat() if self.entry_filled_at else "",
            "exit_filled_at": self.exit_filled_at.isoformat() if self.exit_filled_at else "",
            "fill_latency_ms": self.fill_latency_ms,
            "exit_fill_latency_ms": self.exit_fill_latency_ms,
            "execution_bid_price": self.execution_bid_price,
            "execution_ask_price": self.execution_ask_price,
            "execution_touch_source": self.execution_touch_source,
            "execution_touch_age_seconds": self.execution_touch_age_seconds,
            "execution_touch_usable": self.execution_touch_usable,
            "execution_touch_reason": self.execution_touch_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Trade':
        """Reconstruct a Trade from a serialized dictionary with IST awareness"""
        import pytz
        ist = pytz.timezone('Asia/Kolkata')
        
        ts = data.get("entry_timestamp", 0)
        entry_time = datetime.fromtimestamp(ts, ist) if ts else datetime.now(ist)
        
        exit_time = None
        if data.get("exit_timestamp"):
            try:
                exit_time = datetime.fromisoformat(str(data["exit_timestamp"]))
                if exit_time.tzinfo is None:
                    exit_time = ist.localize(exit_time)
                else:
                    exit_time = exit_time.astimezone(ist)
            except Exception as e:
                logger.debug(f"Failed to parse canonical exit timestamp: {e}")
                exit_time = None
        if exit_time is None and data.get("exit_time") and data["exit_time"] != "--":
            try:
                # Parse display string but force IST context
                exit_year = datetime.now(ist).year
                exit_time = datetime.strptime(f"{exit_year} {data['exit_time']}", "%Y %d %b %H:%M:%S")
                exit_time = ist.localize(exit_time)
                # Guard against year rollover: Dec trade loaded in Jan
                if exit_time > datetime.now(ist):
                    exit_time = exit_time.replace(year=exit_year - 1)
            except Exception as e:

                logger.debug(f"Caught bare exception: {e}")
                exit_time = None

        entry_submitted_at = None
        if data.get("entry_submitted_at"):
            try:
                entry_submitted_at = datetime.fromisoformat(
                    str(data["entry_submitted_at"])
                )
                if entry_submitted_at.tzinfo is None:
                    entry_submitted_at = ist.localize(entry_submitted_at)
                else:
                    entry_submitted_at = entry_submitted_at.astimezone(ist)
            except (TypeError, ValueError):
                entry_submitted_at = None

        def parse_ist_datetime(value):
            if not value:
                return None
            try:
                parsed = datetime.fromisoformat(str(value))
                return ist.localize(parsed) if parsed.tzinfo is None else parsed.astimezone(ist)
            except (TypeError, ValueError):
                return None

        entered_intracandle = data.get("entered_intracandle")
        if entered_intracandle is None:
            entered_intracandle = True  # Unknown legacy entry timing is not confirmation.
            if str(data.get("source_lane") or "").upper() == "UTS" or data.get("source_engine") == "ut_smart":
                source_time = parse_ist_datetime(data.get("signal_time") or data.get("signal_timestamp"))
                fill_time = parse_ist_datetime(data.get("entry_filled_at")) or (entry_time if ts else None)
                minutes = {"1min": 1, "5min": 5, "15min": 15}.get(str(data.get("timeframe") or ""))
                if source_time is not None and fill_time is not None and minutes is not None:
                    entered_intracandle = fill_time < source_time + timedelta(minutes=minutes)
        
        # Extract base instrument name (strip strike info like "NIFTY 24100 CE" -> "NIFTY")
        raw_instrument = data.get("instrument", "")
        base_instrument = raw_instrument.split()[0] if raw_instrument else ""
        # Older persisted rows predate `inst_type`.  A CE/PE contract is
        # unambiguously a long-premium option even when its signal direction is
        # SHORT (bought PE).  Defaulting that row to FUT makes recovery, P&L and
        # exit-side logic treat a live option as a short future.
        raw_inst_type = str(data.get("inst_type") or "").strip().upper()
        if raw_inst_type not in {"OPT", "FUT"}:
            option_type = str(data.get("option_type") or "").strip().upper()
            contract_text = " ".join(
                str(data.get(key) or "").upper()
                for key in ("trading_symbol", "instrument")
            )
            raw_inst_type = (
                "OPT"
                if option_type in {"CE", "PE"}
                or " CE" in contract_text
                or " PE" in contract_text
                or contract_text.endswith(("CE", "PE"))
                else "FUT"
            )
        entry_intelligence_score = data.get("entry_intelligence_score")
        if entry_intelligence_score is None:
            # Rows written before the explicit entry field used intelligence_score
            # for this same entry-time telemetry.
            entry_intelligence_score = data.get("intelligence_score", 0.0)
        
        return cls(
            id=data.get("id", ""),
            instrument=base_instrument,
            timeframe=data.get("timeframe", "5min"),
            direction=data.get("direction", "LONG"),
            entry_price=data.get("entry_price", 0.0),
            entry_time=entry_time,
            trailing_stop=data.get("trailing_stop", 0.0),
            current_stop=data.get("current_stop", data.get("trailing_stop", 0.0)),
            lots=data.get("lots", 1),
            lot_size=data.get("lot_size", 1),
            grade=data.get("grade", "B"),
            atm_strike=data.get("atm_strike", 0.0),
            option_type=data.get("option_type", ""),
            target=data.get("target", 0.0),
            rr_ratio=data.get("rr_ratio", 1.5),
            confidence=data.get("confidence", 0.0),
            inst_type=raw_inst_type,
            instrument_multiplier=data.get("instrument_multiplier", 1.0),
            exec_type=data.get("exec_type", "A"),
            status=data.get("status", "CLOSED"),
            broker_order_id=data.get("broker_order_id", ""),
            exit_order_id=data.get("exit_order_id", ""),
            protective_order_id=data.get("protective_order_id", ""),
            gtt_trigger_id=data.get("gtt_trigger_id", ""),
            protective_trigger=float(data.get("protective_trigger", 0.0) or 0.0),
            protective_quantity=int(data.get("protective_quantity", 0) or 0),
            protective_updated_at=parse_ist_datetime(data.get("protective_updated_at")),
            recovery_required_since=parse_ist_datetime(data.get("recovery_required_since")),
            execution_status=data.get("execution_status") or data.get("status", "CLOSED"),
            execution_error=data.get("execution_error", ""),
            pending_execution=bool(data.get("pending_execution", False)),
            signal_time=parse_ist_datetime(data.get("signal_time") or data.get("signal_timestamp")),
            entry_submitted_at=entry_submitted_at,
            entry_replace_count=int(data.get("entry_replace_count", 0) or 0),
            partial_exit_flatten_count=int(data.get("partial_exit_flatten_count", 0) or 0),
            partial_derisk_done=bool(data.get("partial_derisk_done", False)),
            partial_reduce_order_id=data.get("partial_reduce_order_id", "") or "",
            partial_reduce_requested_quantity=int(
                data.get("partial_reduce_requested_quantity", 0) or 0
            ),
            scale_in_count=int(data.get("scale_in_count", 0) or 0),
            scale_in_lots=int(data.get("scale_in_lots", 0) or 0),
            frozen_initial_risk_rs=float(data.get("frozen_initial_risk_rs", 0.0) or 0.0),
            scale_in_order_id=data.get("scale_in_order_id", "") or "",
            scale_in_requested_quantity=int(
                data.get("scale_in_requested_quantity", 0) or 0
            ),
            reversal_watch_required=bool(data.get("reversal_watch_required", False)),
            is_reversion_scalp=bool(data.get("is_reversion_scalp", False)),
            pending_exit=bool(data.get("pending_exit", False)),
            broker_quantity=int(data.get("broker_quantity", 0) or 0),
            trading_symbol=data.get("trading_symbol", ""),
            symbol_token=data.get("symbol_token", ""),
            current_price=data.get("current_price", 0.0),
            entry_spot=data.get("entry_spot", 0.0),
            spot_stop=data.get("spot_stop", 0.0),
            spot_target=data.get("spot_target", 0.0),
            underlying_entry=float(data.get("underlying_entry") or data.get("entry_spot") or 0.0),
            underlying_stop=float(data.get("underlying_stop") or data.get("spot_stop") or 0.0),
            underlying_target=float(data.get("underlying_target") or data.get("spot_target") or 0.0),
            premium_entry=float(data.get("premium_entry") or 0.0),
            premium_stop=float(data.get("premium_stop") or 0.0),
            premium_target=float(data.get("premium_target") or 0.0),
            geometry_observed_at=str(data.get("geometry_observed_at") or ""),
            geometry_source=str(data.get("geometry_source") or ""),
            geometry_verified=bool(data.get("geometry_verified", False)),
            geometry_schema=str(data.get("geometry_schema") or "dual_price_geometry.v1"),
            expected_move_pct=float(data.get("expected_move_pct") or 0.0),
            expected_move_floor_pct=float(data.get("expected_move_floor_pct") or 0.0),
            expected_move_uncertainty_pct=float(data.get("expected_move_uncertainty_pct") or 0.0),
            expected_move_probability=float(data.get("expected_move_probability") or 0.0),
            expected_move_sample_count=int(data.get("expected_move_sample_count") or 0),
            expected_move_horizon_minutes=int(data.get("expected_move_horizon_minutes") or 0),
            expected_move_generated_at=str(data.get("expected_move_generated_at") or ""),
            expected_move_data_as_of=str(data.get("expected_move_data_as_of") or ""),
            expected_move_model_version=str(data.get("expected_move_model_version") or ""),
            expected_move_source=str(data.get("expected_move_source") or ""),
            entry_policy_allowed=bool(data.get("entry_policy_allowed", False)),
            entry_policy_phase=str(data.get("entry_policy_phase") or ""),
            entry_policy_checked_at=str(data.get("entry_policy_checked_at") or ""),
            entry_policy_reason=str(data.get("entry_policy_reason") or ""),
            entry_policy_actual_rr=float(data.get("entry_policy_actual_rr") or 0.0),
            entry_policy_underlying_stop_pct=float(
                data.get("entry_policy_underlying_stop_pct") or 0.0
            ),
            entry_policy_version=str(data.get("entry_policy_version") or ""),
            quality_verdict=dict(data.get("quality_verdict") or {}),
            decision_record=dict(data.get("decision_record") or {}),
            exit_decision_record=(
                dict(data.get("exit_decision_record"))
                if isinstance(data.get("exit_decision_record"), dict) else {}
            ),
            entry_intel=dict(data.get("entry_intel") or {}),
            decision_intel=dict(data.get("decision_intel") or {}),
            calibration_outcome_recorded=bool(data.get("calibration_outcome_recorded", False)),
            exit_price=data.get("exit_price", data.get("current_price", 0.0)),
            exit_time=exit_time,
            exit_reason=data.get("exit_reason", ""),
            pnl=data.get("pnl", 0.0),
            charges=float(data.get("charges") or 0.0),
            broker_realised_cumulative=(
                float(data.get("broker_realised_cumulative"))
                if data.get("broker_realised_cumulative") is not None
                else None
            ),
            peak_pnl=data.get("peak_pnl", 0.0),
            max_drawdown=data.get("max_drawdown", 0.0),
            is_ghost=bool(data.get("is_ghost", False)),
            is_recovery=bool(data.get("is_recovery", False)),
            pnl_path=[
                (float(row[0]), float(row[1]))
                for row in (data.get("pnl_path") or [])
                if isinstance(row, (list, tuple)) and len(row) >= 2
            ],
            replay_price_path=[
                dict(row) for row in (data.get("replay_price_path") or [])
                if isinstance(row, dict)
            ],
            replay_exit_trace=[
                dict(row) for row in (data.get("replay_exit_trace") or [])
                if isinstance(row, dict)
            ],
            signal_fire_mode=str(data.get("signal_fire_mode") or "INTRACANDLE"),
            entered_intracandle=bool(entered_intracandle),
            uts_confirmed=bool(data.get("uts_confirmed", False)),
            uts_repaint_exit_required=bool(data.get("uts_repaint_exit_required", False)),
            repaint_guard_checked=bool(data.get("repaint_guard_checked", False)),
            gtt_arm_uncertain=bool(data.get("gtt_arm_uncertain", False)),
            market_regime=str(data.get("market_regime") or ""),
            data_provenance=str(data.get("data_provenance") or ""),
            price_action_verdict=str(data.get("price_action_verdict") or ""),
            entry_intelligence_score=float(entry_intelligence_score or 0.0),
            intelligence_score=float(
                data.get("intelligence_score")
                or data.get("entry_intelligence_score")
                or 0.0
            ),
            strategy_revision=str(data.get("strategy_revision") or ""),
            strategy_direction_mode=str(data.get("strategy_direction_mode") or ""),
            market_wide_reversal_entry=bool(data.get("market_wide_reversal_entry", False)),
            volume_spike_entry=bool(data.get("volume_spike_entry", False)),
            power_move=bool(data.get("power_move", False)),
            source_engine=str(data.get("source_engine") or ""),
            source_engines=list(data.get("source_engines") or []),
            source_lane=str(data.get("source_lane") or ""),
            exit_policy_owner=str(data.get("exit_policy_owner") or ""),
            s1_best_spot=data.get("s1_best_spot"),
            s1_t1_locked=bool(data.get("s1_t1_locked", False)),
            s1_peak_premium=data.get("s1_peak_premium"),
            quad_union_entry=bool(data.get("quad_union_entry", False)),
            raw_union_entry=bool(data.get("raw_union_entry", False)),
            initial_stop=float(data.get("initial_stop") or data.get("trailing_stop") or 0.0),
            scalp_lock_mode=bool(data.get("scalp_lock_mode", False)),
            lifecycle_id=str(data.get("lifecycle_id") or data.get("id") or ""),
            runner_mode=bool(data.get("runner_mode", False)),
            original_target_distance=float(data.get("original_target_distance") or 0.0),
            runner_activation_timestamp=parse_ist_datetime(
                data.get("runner_activation_timestamp")
            ),
            runner_activation_reason=str(data.get("runner_activation_reason") or ""),
            runner_stop=float(data.get("runner_stop") or 0.0),
            runner_stop_basis=str(data.get("runner_stop_basis") or ""),
            stop_provenance=str(data.get("stop_provenance") or ""),
            signal_entry_price=float(data.get("signal_entry_price") or 0.0),
            entry_slippage=float(data.get("entry_slippage") or 0.0),
            exit_slippage=float(data.get("exit_slippage") or 0.0),
            requested_entry_price=float(data.get("requested_entry_price") or 0.0),
            requested_exit_price=float(data.get("requested_exit_price") or 0.0),
            entry_filled_at=parse_ist_datetime(data.get("entry_filled_at")),
            exit_filled_at=parse_ist_datetime(data.get("exit_filled_at")),
            fill_latency_ms=float(data.get("fill_latency_ms") or 0.0),
            exit_fill_latency_ms=float(data.get("exit_fill_latency_ms") or 0.0),
            execution_bid_price=float(data.get("execution_bid_price") or 0.0),
            execution_ask_price=float(data.get("execution_ask_price") or 0.0),
            execution_touch_source=str(data.get("execution_touch_source") or ""),
            execution_touch_age_seconds=float(data.get("execution_touch_age_seconds") or 0.0),
            execution_touch_usable=bool(data.get("execution_touch_usable", False)),
            execution_touch_reason=str(data.get("execution_touch_reason") or ""),
        )

    @property
    def quantity(self) -> int:
        return self.lots * self.lot_size

    @property
    def pnl_quantity(self) -> int:
        """The size P&L should be measured on: what the broker actually filled.

        `quantity` is the INTENDED size (lots * lot_size). On a partial fill the broker
        confirms less, and reconcile records that in broker_quantity while flagging the
        trade PARTIAL. broker_quantity is already authoritative for order sizing — exit
        quantity, protective-stop quantity — but P&L was still computed on the intended
        size, overstating it by the fill ratio. A 65-intended / 30-filled option trade
        reported -860 against a true -440.

        That number feeds the daily loss cap and the loss guard, so an overstatement
        halts trading early on money that was never at risk. Falls back to the intended
        size when nothing is confirmed yet, which is the pre-fill estimate.
        """
        try:
            confirmed = int(getattr(self, "broker_quantity", 0) or 0)
        except (TypeError, ValueError):
            confirmed = 0
        return confirmed if confirmed > 0 else self.quantity

    @property
    def unrealized_pnl(self) -> float:
        """Net P&L including estimated charges and instrument multipliers"""
        if self.status != "OPEN":
            return 0.0
        
        from config.settings import get_settings
        settings = get_settings()
        from engine.trade_accounting import estimate_trade_charges
        # Size on what the broker actually filled, not on what was intended — see
        # pnl_quantity. Charges too: brokerage and taxes scale with the filled lot.
        sized = self.pnl_quantity
        est_charges = estimate_trade_charges(
            self.entry_price,
            self.current_price,
            sized,
            self.inst_type,
            settings,
            self.instrument_multiplier,
            instrument=self.instrument,
        )

        # NOTE: self.current_price is ALREADY adjusted for Delta in update_trade()
        # For Options, we are always LONG (buying CE or PE).
        if self.inst_type == "OPT":
            gross = (self.current_price - self.entry_price) * sized
        else:
            # For Futures, it depends on direction
            if self.direction == "LONG":
                gross = (self.current_price - self.entry_price) * sized
            else:
                gross = (self.entry_price - self.current_price) * sized
            
        return gross - est_charges

    @property
    def unrealized_gross_pnl(self) -> float:
        """The gross half of unrealized_pnl: mark-to-market before any charges.

        Mirrors the gross arithmetic above exactly (pinned by a test) without touching
        the net figure the loss cap reads. pnl_path and peak_pnl are gross; sampling the
        net ``pnl`` left broker rows on a different basis from every paper row.
        """
        if self.status != "OPEN":
            return 0.0
        sized = self.pnl_quantity
        if self.inst_type == "OPT":
            return (self.current_price - self.entry_price) * sized
        if self.direction == "LONG":
            return (self.current_price - self.entry_price) * sized
        return (self.entry_price - self.current_price) * sized


class TradeManager:
    """Full trade lifecycle management with persistence"""

    def _log_ui(self, message: str, level: str = "info"):
        if self.ui_logger:
            try:
                self.ui_logger(message, level)
            except Exception as e:
                logger.error(f"Failed to dispatch UI log: {e}")

    def _mark_dirty(self):
        """Signal the write-behind persistence thread that state has changed.

        Safe to call even on partially-initialised instances (e.g. tests that
        use ``TradeManager.__new__`` without calling ``__init__``).
        """
        dirty = getattr(self, "_dirty", None)
        if dirty is not None:
            dirty.set()

    def __init__(
        self,
        broker=None,
        max_positions: int = 3,
        session_end: str = "15:25",
        product_type: str = "CARRYFORWARD",
        on_notification=None,
        entry_submission_guard: Optional[Callable[[Trade], Any]] = None,
    ):
        self.broker = broker
        # Write-behind DB persistence event — initialised first so that any
        # code path that touches self._mark_dirty() during __init__ is safe.
        self._dirty = threading.Event()
        self.ui_logger = None
        # Injected desktop notifier (main.py passes send_desktop_notification). Falls
        # back to importing it directly, so behaviour is unchanged when not supplied.
        self.on_notification = on_notification
        # Optional last-moment AUTO-entry admission hook.  Unlike the scanner's signal
        # gates, this runs inside the queued worker immediately before each broker
        # submission (including retries), so a decision that became stale while waiting
        # in the queue cannot leak through.  Manual entries deliberately bypass it.
        self.entry_submission_guard = entry_submission_guard
        # Separate, always-on fresh-exposure policy. Manual approval bypasses Auto
        # authority, not execution RR/geometry/forecast safety. Recovery alone bypasses.
        self.fresh_entry_policy_guard: Optional[Callable[[Trade], Any]] = None
        self.max_positions = max_positions
        self.session_end = self._parse_time(session_end)
        self.product_type = str(product_type or "CARRYFORWARD").upper()
        self.open_trades: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        # A final fill is fully accounted before the Trade is removed from
        # open_trades, but terminal telemetry is emitted immediately afterwards.
        # This ownership bridge closes that tiny removal-to-callback interval so a
        # scanner/session advisory cannot publish the canonical EXIT first.
        self._terminal_callback_lifecycles: set[str] = set()
        self._trade_counter = 0
        self._lock = threading.RLock()
        # Serialises exchange-side protective-stop placement so two callers
        # (PositionSync reconcile + scanner tick) can never place a duplicate SL-M
        # for the same fresh fill. Dedicated lock (not self._lock) keeps the ~1s
        # broker round-trip from blocking unrelated trade-manager operations; its
        # callers never hold self._lock, so there is no lock-ordering inversion.
        self._protection_lock = threading.RLock()
        self._state_file = "data_store/trade_state.json"
        self._last_mark_to_market_save = 0.0
        self._last_protective_sync = {}
        self._last_protective_status_poll = {}
        self._last_recovery_alert = {}
        self.orphan_positions: Dict[str, Dict] = {}
        
        # Risk settings (updated via Scanner.configure)
        self.futures_sl_pct = 0.30
        self.options_sl_pct = 15.0
        
        # Load state on init
        self.load_state()
        
        # Start background broker position sync
        self._sync_active = True
        # Wakes the position-sync loop immediately instead of letting it finish a blind
        # sleep. Without it a new position waited out whatever interval the loop had
        # already committed to — up to position_sync_idle_interval_seconds (30s), which
        # is precisely the interval in force when the book is empty, i.e. before the
        # FIRST entry of the day. That is the 5-30s lag between a fill happening and
        # anything on screen reflecting it.
        self._sync_wake = threading.Event()
        self._sync_thread = threading.Thread(target=self._position_sync_loop, daemon=True, name="PositionSync")
        self._sync_thread.start()

        # Start write-behind persistence thread
        self._persistence_thread = threading.Thread(
            target=self._persistence_loop, daemon=True, name="Persistence"
        )
        self._persistence_thread.start()

    def _persistence_loop(self):
        """Background loop to persist state out-of-band to avoid blocking the hot path."""
        while getattr(self, "_sync_active", True):
            # Wait up to 1 second for a write request. This naturally coalesces
            # rapid bursts of changes (like 10 PnL ticks) into a single write.
            if self._dirty.wait(timeout=1.0):
                self._dirty.clear()
                try:
                    self.save_state()
                except Exception as exc:
                    # A failed checkpoint is still dirty. Retain the wake and retry
                    # instead of killing this daemon thread and reporting later writes
                    # as successful.
                    self._dirty.set()
                    logger.error(f"Trade persistence checkpoint failed; retrying: {exc}")
                    time.sleep(1.0)

    def _position_sync_loop(self):
        """Background loop to sync internal open_trades with broker positions.

        Adaptive cadence so a rate-limited/slow broker API isn't hammered:
          * base interval while positions are OPEN (need close-reconciliation),
          * slower idle interval when FLAT (nothing to reconcile),
          * exponential backoff on a FAILED reconcile (rate-limit/timeout),
            resetting to base on the next success.
        Previously a flat every-5s get_positions() that spammed the broker even
        when flat and even while being rate-limited.
        """
        import time
        from config.settings import get_settings
        backoff = 0.0
        while getattr(self, "_sync_active", True):
            interval = 5.0
            try:
                s = get_settings()
                base = max(1.0, float(getattr(s, "position_sync_interval_seconds", 5.0) or 5.0))
                idle = max(base, float(getattr(s, "position_sync_idle_interval_seconds", 30.0) or 30.0))
                max_backoff = max(base, float(getattr(s, "position_sync_max_backoff_seconds", 60.0) or 60.0))
                result = self.reconcile_broker_positions()
                ok = isinstance(result, dict) and bool(result.get("checked", False))
                has_open = bool(getattr(self, "open_trades", {}))
                if ok:
                    backoff = 0.0
                    interval = base if has_open else idle
                else:
                    # Reconcile could not complete (rate-limit / timeout / no broker):
                    # back off instead of retrying into the same wall every 5s.
                    backoff = min(base if backoff <= 0 else backoff * 2.0, max_backoff)
                    interval = max(backoff, base if has_open else idle)
            except Exception as exc:
                logger.error(f"Error in position sync loop: {exc}")
                backoff = min(5.0 if backoff <= 0 else backoff * 2.0, 60.0)
                interval = backoff
            # Interruptible: nudge_position_sync() cuts the wait short the moment a
            # position changes, so a fill is reconciled in milliseconds rather than
            # after the remainder of a 5s (or 30s idle) sleep.
            if self._sync_wake.wait(timeout=interval):
                self._sync_wake.clear()

    def nudge_position_sync(self) -> None:
        """Ask the sync loop to reconcile now rather than after its current wait.

        Cheap and idempotent: setting an already-set Event is a no-op, and the loop
        clears it on wake. Safe to call from any thread and from paths that may run
        before the loop exists.
        """
        wake = getattr(self, "_sync_wake", None)
        if wake is not None:
            wake.set()

    @staticmethod
    def _position_quantity(position: Dict) -> int:
        try:
            return int(float(position.get("netqty", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _reconciled_exit_price(self, trade: Trade, position: Dict) -> float:
        """Exit fill price, preferring the exit ORDER's own executions.

        _position_exit_price reads the position row's sellavgprice, which Kite blends
        across EVERY sell on that contract for the day. Two round trips on one strike
        therefore priced the second exit partly at the first one's fill. The entry leg
        has always been sourced correctly — from the order's own trades — and get_fill()
        returns exactly that for the exit too (a quantity-weighted average of its own
        executions); it was already being called on this path with only the quantity
        read and average_price discarded. Position aggregate stays the fallback for a
        trade with no exit order id, e.g. one closed by a broker square-off.
        """
        order_id = str(getattr(trade, "exit_order_id", "") or "").strip()
        broker = getattr(self, "broker", None)
        filler = getattr(broker, "get_fill", None) if broker is not None else None
        if order_id and callable(filler):
            try:
                detail = filler(order_id) or {}
                if str(detail.get("state") or "") == "FILLED":
                    price = float(detail.get("average_price") or 0.0)
                    if price > 0:
                        return price
            except Exception as exc:
                logger.debug(f"Exit-fill lookup failed for {order_id}: {exc}")
        return self._position_exit_price(trade, position)

    def _adopt_completed_protective_fill(self, trade: Trade) -> bool:
        """Bind a flat position to its own completed protective order, before cleanup."""
        order_id = str(getattr(trade, "protective_order_id", "") or "").strip()
        getter = getattr(getattr(self, "broker", None), "get_fill", None)
        if not order_id or not callable(getter):
            return False
        try:
            fill = getter(order_id) or {}
            quantity = int(fill.get("quantity") or 0)
            required = int(getattr(trade, "protective_quantity", 0) or trade.quantity)
            price = float(fill.get("average_price") or 0.)
            # get_fill's FILLED includes partial executions. A partial protective
            # fill must not replace a separate exit or erase a still-working stop.
            if (fill.get("state") != "FILLED" or quantity < required or required <= 0
                    or not math.isfinite(price) or price <= 0):
                return False
        except (TypeError, ValueError, OverflowError):
            return False
        except Exception as exc:
            logger.debug(f"Protective fill lookup deferred for {order_id}: {type(exc).__name__}")
            return False
        trade.exit_order_id = order_id
        trade.exit_price = price
        trade.exit_reason = "STOP_HIT"
        # The order has filled. Retain its identity in exit_order_id and never
        # send a cancellation or an unowned-stop alert for this completed order.
        trade.protective_order_id = ""
        self._record_fill(trade, order_id, -quantity, "PROTECTIVE")
        return True

    @staticmethod
    def _position_exit_price(trade: Trade, position: Dict) -> float:
        value = (
            position.get("sellavgprice", 0.0)
            if trade.direction in {"LONG", "BUY"} or trade.inst_type == "OPT"
            else position.get("buyavgprice", 0.0)
        )
        try:
            price = float(value or 0.0)
        except (TypeError, ValueError):
            price = 0.0
        return price if price > 0 else (trade.current_price or trade.entry_price)

    @staticmethod
    def _confirmed_broker_quantity(trade: Trade) -> int:
        try:
            return max(0, int(getattr(trade, "broker_quantity", 0) or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _is_utema_trade(trade) -> bool:
        """Persisted native ownership, independent of the selected scanner or switches."""
        return bool(
            str(getattr(trade, "exit_policy_owner", "") or "").upper() == "UTEMA"
            and str(getattr(trade, "source_lane", "") or "").upper() == "UTEMA"
            and str(getattr(trade, "source_engine", "") or "").lower() in {"", "ut_ema"}
        )

    @staticmethod
    def _utema_execution_geometry(inst_type, direction, entry, stop, target, *, protected_add=False):
        """Validate native execution prices without another strategy's RR or stop floor."""
        try:
            entry, stop, target = float(entry), float(stop), float(target)
        except (TypeError, ValueError, OverflowError):
            return False, "UTEMA execution geometry is malformed"
        if not all(math.isfinite(v) for v in (entry, stop, target)) or entry <= 0 or stop <= 0 or target < 0:
            return False, "UTEMA execution geometry must be finite and positive"
        kind, side = str(inst_type or "").upper(), str(direction or "").upper()
        if kind not in {"OPT", "FUT"} or side not in {"LONG", "SHORT"}:
            return False, "UTEMA execution identity is unsupported"
        long_price = kind == "OPT" or side == "LONG"
        if not protected_add and (stop >= entry if long_price else stop <= entry):
            return False, "UTEMA initial stop is not protective"
        if target > 0 and (target <= entry if long_price else target >= entry):
            return False, "UTEMA target is not on the profitable side"
        return True, ""

    @staticmethod
    def _utema_touch_admission(price, quote, *, inst_type="OPT", direction="LONG", required=False):
        """Require measured executable prices, without inherited spread/liquidity filters."""
        if not required:
            return True, ""
        from config.settings import get_settings
        try:
            quote = quote if isinstance(quote, dict) else {}
            bid, ask, age = float(quote.get("bid") or 0), float(quote.get("ask") or 0), float(quote["age_seconds"])
            limit = max(.05, float(getattr(get_settings(), "execution_touch_quote_max_age_seconds", 1.5) or 1.5))
            mark = float(price)
        except (KeyError, TypeError, ValueError, OverflowError):
            return False, "UTEMA fresh executable quote is unavailable"
        if (not quote.get("usable") or not all(math.isfinite(v) for v in (bid, ask, age, mark))
                or bid <= 0 or ask < bid or not 0 <= age <= limit):
            return False, "UTEMA executable quote is invalid or stale"
        touch = ask if str(inst_type).upper() == "OPT" or str(direction).upper() == "LONG" else bid
        if abs(mark - touch) > .050000001:
            return False, "UTEMA execution price does not match the current touch"
        return True, ""

    def _same_contract_entry_blocked(
        self, trading_symbol: str, *, is_recovery: bool = False, is_ghost: bool = False,
        native_owner: str = "",
    ) -> Tuple[bool, str]:
        """Refuse a second REAL position on a contract that already holds one.

        INTERIM MEASURE, and deliberately blunt. Kite reports POSITIONS per
        tradingsymbol, so when several trades share a contract the broker cannot tell us
        who owns what -- _allocate_broker_quantity splits the row by a rule (confirmed
        fills first, then oldest-first by intended size) and that rule is a GUESS the
        moment one sibling exits and the remainder has to be re-attributed.

        The real answer is a persisted per-order fill ledger, which is being built. Until
        it exists, the safe state is one REAL trade per contract: the guess never has to
        be made, so it can never be wrong about the quantity that sizes a stop.

        What this costs: cascade multi-timeframe stacking on a SINGLE strike. Stacking
        across different strikes or instruments is untouched, and so is every non-REAL
        mode -- backtests and replays keep exercising the multi-trade paths, including
        the allocator's own tests.

        Lifts by setting forbid_concurrent_same_contract_real=False, which is how the
        ledger work will be validated before this is retired.
        """
        if not self._live_orders_permitted():
            return False, ""
        if is_recovery or is_ghost:
            # Recovery re-establishes a position that ALREADY exists; refusing it would
            # strand a live position instead of preventing a second one.
            return False, ""
        symbol = str(trading_symbol or "").strip()
        if not symbol:
            return False, ""
        from config.settings import get_settings

        settings = get_settings()
        siblings = [
            t for t in list(getattr(self, "open_trades", {}).values())
            if not getattr(t, "is_ghost", False)
            and str(getattr(t, "trading_symbol", "") or "").strip() == symbol
        ]
        if not siblings:
            return False, ""

        native_stack = str(native_owner or "").upper() == "UTEMA" and all(
            self._is_utema_trade(existing) for existing in siblings
        )

        if _setting_enabled(
            getattr(settings, "forbid_concurrent_same_contract_real", True), True
        ) and not native_stack:
            first = siblings[0]
            return True, (
                f"{symbol} already held by trade {getattr(first, 'id', '?')} "
                f"({getattr(first, 'timeframe', '?')}); one REAL position per contract "
                "while same-contract stacking is disabled"
            )

        # The master switch is off, which does NOT mean unrestricted. Stacking is only
        # safe while the ledger can say who owns what on THIS contract, so it has to
        # prove that per contract rather than once, globally, by operator judgement.
        #
        # A trade the ledger cannot account for is the whole hazard: reconciliation
        # falls back to splitting the aggregate, and that split is a guess the moment a
        # sibling exits. Refusing to ADD to such a contract keeps the guess confined to
        # a single position, where it cannot mis-size anything.
        if not native_stack and not _setting_enabled(
            getattr(settings, "same_contract_requires_ledger_evidence", True), True
        ):
            return False, ""
        for existing in siblings:
            tid = str(getattr(existing, "id", "") or "")
            if self._ledger_owned_quantity(tid) is None:
                return True, (
                    f"{symbol} already held by trade {tid or '?'}, and the fill ledger "
                    "cannot account for it; refusing to stack onto a contract whose "
                    "ownership would have to be guessed"
                )
        return False, ""

    def _transactable_quantity(self, trade, requested: Optional[int] = None) -> int:
        """How many contracts this trade may transact. ZERO IS AN ANSWER, not a gap.

        The sizing sites all read `quantity or trade.broker_quantity or trade.quantity`,
        which treats an explicit 0 as absent and falls through to the intended size. So
        a trade that owns nothing asked the broker for a full lot: an SL-M for 75 with
        nothing behind it, or an exit for 75 that sells a SIBLING's position on the same
        contract. On options that writes a naked reverse leg.

        That shape was harmless while 0 only ever meant "not filled yet". The fill
        ledger changed it: 0 is now a real, evidenced answer -- this trade's own orders
        net to nothing -- and it arrives on exactly the contracts where over-sizing does
        the damage.

        Precedence, most authoritative first:
          1. an explicitly REQUESTED quantity, including 0
          2. the ledger, which distinguishes a known 0 from no information at all,
             capped by a confirmed broker holding -- see below
          3. a confirmed broker fill
          4. intent -- and only here, where nothing is known and the entry may simply
             not have been confirmed yet, which is the case the old fallback was for

        The ledger is authoritative about ATTRIBUTION -- which of several trades on one
        contract owns which lots -- and never about TOTAL exposure. It sums only orders
        THIS system placed, so anything that removes quantity without a recorded order
        leaves it stale HIGH: an operator squaring off part of the position at the Kite
        terminal (which this file tells them to do in four places), or, with no operator
        at all, a partial de-risk whose own reduce order cannot be read back --
        _record_leg_fill_from_order stays deliberately silent on an UNKNOWN fill, while
        the positions-derived broker_quantity still drops. The next full exit then sells
        the ORIGINAL size against what is left, which on options writes a naked short in
        a CARRYFORWARD/NRML product the broker will not square off.

        _allocate_broker_quantity already applies exactly this rule, capping every
        ledger answer by what remains of the broker's row so "nothing is ever armed for
        more than the account actually holds". Reading db.owned_quantity directly was
        the one path that skipped it.
        """
        if requested is not None:
            try:
                return max(0, int(requested))
            except (TypeError, ValueError):
                return 0
        owned = self._ledger_owned_quantity(str(getattr(trade, "id", "") or ""))
        if owned is not None:
            owned = max(0, int(owned))
            confirmed = self._confirmed_broker_quantity(trade)
            # Only a POSITIVE confirmation may cap. Zero there means "the entry is not
            # reconciled yet", not "the account is flat", and clamping to it would
            # refuse to exit a position that is genuinely live.
            if confirmed > 0 and confirmed < owned:
                # Say so. The clamp is the right answer -- selling what the broker does
                # not confirm is how a naked short is written -- but it deliberately
                # UNDER-sizes, so an exit taken on this number leaves a residual. That
                # residual is squared off by the 15:25 sweep (check_session_end routes
                # PARTIAL_EXIT to emergency_flatten_trade) and not before, so it can sit
                # unprotected for the rest of the session. Silent is the one thing it
                # must not be.
                logger.warning(
                    f"Ledger/broker divergence on {getattr(trade, 'trading_symbol', '?')} "
                    f"({getattr(trade, 'id', '?')}): the ledger attributes {owned} but the "
                    f"broker confirms {confirmed}. Sizing on {confirmed}; any remainder is "
                    "left for reconciliation and the session-end sweep."
                )
                return confirmed
            return owned
        confirmed = self._confirmed_broker_quantity(trade)
        if confirmed > 0:
            return confirmed
        try:
            return max(0, int(getattr(trade, "quantity", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _record_fill(
        self, trade, order_id: str, quantity_delta: int, leg: str
    ) -> None:
        """Record one ORDER's fill against one TRADE, durably.

        An order belongs to exactly one trade and its fill is a fact the broker states,
        so this is the only place ownership is ever KNOWN rather than inferred. Signed:
        an entry or add is positive, an exit negative, so a trade's holding is the sum.

        Never raises. The ledger improves attribution; it must not be able to take down
        an order path that is otherwise working, and a missing row degrades to the older
        splitting rule rather than to a wrong number.
        """
        try:
            oid = str(order_id or "").strip()
            if not oid or not int(quantity_delta or 0):
                return
            db.record_order_fill(
                oid,
                str(getattr(trade, "id", "") or ""),
                int(quantity_delta),
                tradingsymbol=str(getattr(trade, "trading_symbol", "") or ""),
                leg=str(leg or ""),
                tag=self._broker_order_tag(trade, leg) if leg else "",
            )
        except Exception as exc:
            logger.debug(f"Fill ledger write skipped for {getattr(trade, 'id', '?')}: {exc}")

    def _order_filled_quantity(self, order_id: str) -> Optional[int]:
        """What ONE order actually executed, from the order's own trades.

        get_fill asks Kite for that order's executions, so the answer is per-ORDER. The
        positions-derived helpers next to it (_confirmed_reduced_quantity,
        _confirmed_scaled_quantity) read the per-tradingsymbol AGGREGATE, which is the
        guess the ledger exists to remove -- on a shared contract they cannot tell this
        trade's reduction from a sibling's.

        Three answers, kept distinct: a number when the broker states the fill, 0 when
        it states there were no executions, and None when it cannot say. None must never
        become 0 here; an unknown fill that recorded as zero would leave the ledger
        confidently wrong, which is worse than leaving it silent.
        """
        oid = str(order_id or "").strip()
        if not oid:
            return None
        getter = getattr(getattr(self, "broker", None), "get_fill", None)
        if not callable(getter):
            return None
        try:
            detail = getter(oid) or {}
        except Exception as exc:
            logger.debug(f"Order fill lookup failed for {oid}: {exc}")
            return None
        state = str(detail.get("state") or "").upper()
        if state == "FILLED":
            try:
                return max(0, int(float(detail.get("quantity") or 0)))
            except (TypeError, ValueError):
                return None
        if state == "NO_FILL":
            return 0
        return None

    def _record_leg_fill_from_order(self, trade, order_id: str, leg: str, sign: int) -> bool:
        """Record a leg into the ledger from the ORDER, or record nothing at all.

        Silence is the safe failure: a missing row makes _allocate_broker_quantity fall
        back to the older splitting rule, which is worse than the ledger but honest. A
        row derived from the aggregate would instead put the guess INSIDE the ledger and
        make it look authoritative.
        """
        qty = self._order_filled_quantity(order_id)
        if not qty:
            if qty is None:
                logger.debug(
                    f"Ledger left silent for {getattr(trade, 'id', '?')} {leg}: "
                    f"order {order_id} fill unknown"
                )
            return False
        self._record_fill(trade, order_id, sign * int(qty), leg)
        return True

    @staticmethod
    def _ledger_owned_quantity(trade_id: str) -> Optional[int]:
        """This trade's holding per the ledger, or None when the ledger is silent.

        None and 0 mean different things and must never be collapsed: 0 is "its own
        orders net to nothing", a fact; None is "no rows", no information.
        """
        try:
            return db.owned_quantity(str(trade_id or ""))
        except Exception:
            return None

    @staticmethod
    def _allocate_broker_quantity(trades_on_symbol, net_qty, ledger=None) -> Dict[str, int]:
        """Split ONE broker position row across the local trades that share the contract.

        Kite reports POSITIONS per tradingsymbol, not per order, so three trades on the
        same strike read one row. reconcile_broker_positions assigned that row's net
        quantity to every one of them:

            trade.broker_quantity = abs(net_qty)

        Three 75-lot trades against a broker net of 225 each became broker_quantity=225.
        That field is authoritative for order SIZING -- ensure_broker_protection, the
        flatten path and _request_broker_exit all size from it -- so each trade then
        armed a 225 stop against the 75 it actually held. On options an oversized SL-M
        sells more than the account owns: a naked written option, which this file
        already documents as the outcome that must never happen.

        Allocation rules, in order:
          1. One trade on the contract -> the row is unambiguously its own. Return the
             net verbatim, so every existing single-trade path (including an excess that
             callers detect by comparing against trade.quantity) is unchanged.
          2. A trade whose own ENTRY FILL was already confirmed keeps that quantity.
             Per-order fills come from the order itself and are real evidence; a
             positions-derived split is a guess and must not overwrite them.
          3. Whatever is left is handed out oldest-trade-first, each capped at the
             quantity that trade actually intended.

        Under-allocation is the safe direction: a stop sized short of the true position
        leaves part of it unprotected, while an oversized one sells stock the account
        does not hold. Positions alone genuinely cannot say which trade owns which lot
        when fills are ambiguous -- only the per-order fill path can, which is why an
        already-confirmed fill wins over this split.
        """
        try:
            total = abs(int(net_qty or 0))
        except (TypeError, ValueError):
            total = 0

        if len(trades_on_symbol) <= 1:
            return {str(tid): total for tid, _ in trades_on_symbol}

        def _order_key(item):
            tid, trade = item
            return (str(getattr(trade, "entry_time", "") or ""), str(tid))

        ordered = sorted(trades_on_symbol, key=_order_key)
        allocation: Dict[str, int] = {}
        remaining = total

        # THE LEDGER FIRST. A trade whose own orders are recorded does not need to be
        # guessed at: its holding is the sum of its own fills, which stays correct after
        # a sibling exits — the case the splitting rule below cannot get right, because
        # once the aggregate shrinks it cannot say whose quantity left.
        #
        # Still capped by `remaining`: the broker is authoritative about TOTAL exposure
        # even when the ledger is authoritative about attribution. If the two disagree
        # the aggregate wins on size, so nothing is ever armed for more than the account
        # actually holds.
        for tid, _trade in ordered:
            known = ledger(str(tid)) if callable(ledger) else None
            if known is None:
                continue  # silent ledger is no information, not a zero
            take = min(remaining, max(0, int(known)))
            allocation[str(tid)] = take
            remaining -= take

        for tid, trade in ordered:
            if str(tid) in allocation:
                continue
            own = TradeManager._confirmed_broker_quantity(trade)
            if own > 0:
                take = min(remaining, own)
                allocation[str(tid)] = take
                remaining -= take

        for tid, trade in ordered:
            if str(tid) in allocation:
                continue
            try:
                want = max(0, int(getattr(trade, "quantity", 0) or 0))
            except (TypeError, ValueError):
                want = 0
            take = min(remaining, want)
            allocation[str(tid)] = take
            remaining -= take

        return allocation

    @staticmethod
    def _entry_still_confirming(trade: Trade) -> bool:
        status = str(getattr(trade, "execution_status", "") or "").upper()
        return bool(getattr(trade, "pending_execution", False)) or status in {
            "ENTRY_PENDING",
            "ENTRY_SUBMITTED",
            "PARTIAL",
        }

    @staticmethod
    def _reconciliation_symbol_prefixes() -> Tuple[str, ...]:
        fallback = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX"}
        prefixes = set(fallback)
        try:
            payload = json.loads(Path("config/instruments.json").read_text(encoding="utf-8"))
            indices = payload.get("indices", {}) if isinstance(payload, dict) else {}
            if isinstance(indices, dict):
                for name, cfg in indices.items():
                    if name:
                        prefixes.add(str(name).upper())
                    if isinstance(cfg, dict):
                        for key in ("symbol", "option_symbol_prefix"):
                            value = str(cfg.get(key) or "").strip().upper()
                            if value:
                                prefixes.add(value)
        except Exception as exc:
            logger.debug(f"Using fallback reconciliation prefixes: {exc}")
        return tuple(sorted(prefixes, key=len, reverse=True))

    def _maybe_auto_flatten_unprotected(self, trade: Trade, reason: str) -> bool:
        if getattr(self, "mode", "HISTORICAL") != "REAL" or getattr(trade, "is_ghost", False):
            return False
        if self._confirmed_broker_quantity(trade) <= 0:
            return False
        if getattr(trade, "protective_order_id", ""):
            return False
        if str(getattr(trade, "execution_status", "") or "").upper() != "RECOVERY_REQUIRED":
            return False

        from config.settings import get_settings

        recovery_since = self._aware_ist(trade.recovery_required_since)
        grace = max(
            0.0,
            float(getattr(get_settings(), "recovery_auto_flatten_seconds", 5.0) or 0.0),
        )
        if (datetime.now(IST) - recovery_since).total_seconds() < grace:
            return False
        return self.emergency_flatten_trade(
            trade,
            f"{reason} for at least {grace:.0f}s",
        )

    def _finalize_closed_trade(
        self,
        trade_id: str,
        trade: Trade,
        exit_price: float,
        reason: str,
        exit_time: Optional[datetime] = None,
        realised_pnl: Optional[float] = None,
        broker_realised_cumulative: Optional[float] = None,
    ) -> bool:
        # The trade is finished, so whatever it still held has left the book. Booked
        # here rather than at the exit ACK because an acknowledgement is not a fill, and
        # a ledger that decrements early would under-state a position that is still on.
        _held = self._confirmed_broker_quantity(trade)
        # Ask the LEDGER whether anything is still attributed to this trade, not just
        # broker_quantity. Gating the whole block on broker_quantity meant the exit leg
        # was never recorded on the one path that closes REAL trades: reconcile reads a
        # flat position row, sets trade.broker_quantity = allocated_qty (which is 0 for a
        # netqty==0 row) and calls straight through to here, so `_held` was ALWAYS 0 and
        # the ENTRY row stayed on the books at full size forever. Nothing in the live
        # path reads a closed trade's row, so no order was mis-sized -- but
        # scripts/ledger_certification.py sums every row per contract, so every contract
        # that had ever closed reported broker 0 vs ledger 75 and the certification gate
        # (the thing that decides whether same-contract stacking may come back) could
        # never pass. The number this block exists to correct cannot also be its guard.
        _sum = self._ledger_owned_quantity(getattr(trade, "id", ""))
        if _held or _sum:
            # Prefer the exit ORDER's own executions, like every other leg. When the
            # order cannot state them, close the ledger out EXACTLY: a finished trade
            # holds nothing, so the closing delta is the negative of whatever the ledger
            # currently believes -- not of trade.broker_quantity, which is the number the
            # ledger exists to stop trusting.
            if self._record_leg_fill_from_order(
                trade, str(getattr(trade, "exit_order_id", "") or ""), "FULL_EXIT", -1
            ):
                # The exit order stated its own fill, which is the best evidence there
                # is -- but it may be a PARTIAL one. A trade that is finished holds
                # nothing by definition, so whatever the ledger still attributes to it
                # after that write is a remainder that no later order will ever clear,
                # and it reports as a broker/ledger mismatch forever. Re-read and settle
                # the balance rather than assuming the order covered the whole holding.
                _residual = self._ledger_owned_quantity(getattr(trade, "id", ""))
                _close = _residual if _residual is not None else 0
            else:
                _close = _held if _sum is None else _sum
            if _close:
                self._record_fill(
                    trade, f"{getattr(trade, 'id', '')}|CLOSE", -_close, "FULL_EXIT"
                )
        from config.settings import get_settings
        from engine.trade_accounting import estimate_trade_charges

        on_trade_closed = getattr(self, "on_trade_closed", None)
        terminal_lifecycle = str(
            getattr(trade, "lifecycle_id", "") or getattr(trade, "id", "") or trade_id
        )
        terminal_owners = getattr(self, "_terminal_callback_lifecycles", None)
        if not isinstance(terminal_owners, set):
            terminal_owners = set()
            self._terminal_callback_lifecycles = terminal_owners

        with self._lock:
            if trade_id not in self.open_trades:
                return False
            # The decision price was stamped into exit_price by close_trade before the
            # broker exit was requested; the incoming exit_price is the real fill.
            exit_decision_price = float(trade.exit_price or 0.0)
            trade.status = "CLOSED"
            trade.execution_status = "CLOSED"
            trade.pending_execution = False
            trade.pending_exit = False
            trade.execution_error = ""
            trade.exit_price = float(exit_price or trade.current_price or trade.entry_price)
            trade.current_price = trade.exit_price
            trade.exit_time = exit_time or datetime.now(IST)
            trade.exit_reason = reason
            if broker_realised_cumulative is not None:
                trade.broker_realised_cumulative = float(broker_realised_cumulative)
            if getattr(trade, "exit_order_id", ""):
                self._record_exit_slippage(trade, exit_decision_price, trade.exit_price)
                self._stamp_fill_quality(trade, leg="EXIT", requested=exit_decision_price)
            # Realised P&L on the size the broker actually filled, not the intended
            # size. A partial fill otherwise books a loss (or gain) proportionally
            # larger than the position ever was, and that figure is what the daily
            # loss cap and the session summary are built from.
            sized = trade.pnl_quantity
            if trade.inst_type == "OPT":
                gross = (trade.exit_price - trade.entry_price) * sized
            elif trade.direction == "LONG":
                gross = (trade.exit_price - trade.entry_price) * sized
            else:
                gross = (trade.entry_price - trade.exit_price) * sized
            settings = get_settings()
            exact_broker_fill_pair = bool(
                getattr(self, "mode", "HISTORICAL") == "REAL"
                and str(getattr(trade, "broker_order_id", "") or "").strip()
                and str(
                    getattr(trade, "exit_order_id", "")
                    or getattr(trade, "protective_order_id", "")
                    or ""
                ).strip()
                and trade.entry_price > 0
                and trade.exit_price > 0
            )
            if exact_broker_fill_pair:
                # Entry/exit are actual broker fills, so their difference already
                # contains realised slippage.  Adding the simulation slippage model
                # again double-counts it.  Statutory/broker charges are not present in
                # Kite's price P&L, so retain the configured flat round-trip estimate.
                configured_cost = getattr(
                    settings, "opt_cost" if trade.inst_type == "OPT" else "fut_cost", 0.0
                )
                trade.charges = max(0.0, float(configured_cost or 0.0))
            else:
                trade.charges = estimate_trade_charges(
                    trade.entry_price,
                    trade.exit_price,
                    sized,
                    trade.inst_type,
                    settings,
                    trade.instrument_multiplier,
                    instrument=trade.instrument,
                )
            # Charges come off BOTH branches (2026-08-16). Kite's position `realised` is
            # a price-based figure — brokerage, STT, exchange and GST are reported
            # separately and are not in it — so taking it verbatim booked every
            # reconcile-closed trade gross while the computed branch right above booked
            # net. trade.charges was already being calculated on this path and then
            # thrown away. Both numbers feed the daily-loss breaker, so the two branches
            # disagreeing meant the breaker's threshold meant different things depending
            # on which code path happened to close the trade.
            # A freshly-flat position snapshot can report realised=0 briefly.  When
            # both tracked order fills exist, their own prices are stronger per-trade
            # evidence than that delayed/cumulative contract-level field.
            trade.pnl = (
                gross - trade.charges
                if exact_broker_fill_pair
                else (
                    float(realised_pnl) - trade.charges
                    if realised_pnl is not None
                    else gross - trade.charges
                )
            )
            if callable(on_trade_closed) and terminal_lifecycle:
                # Add before removing from open_trades. At every observable instant
                # either the live row or this bridge owns the lifecycle.
                terminal_owners.add(terminal_lifecycle)
            self.closed_trades.append(trade)
            del self.open_trades[trade_id]
        self._mark_dirty()
        
        profit_str = f"+₹{trade.pnl:.2f}" if trade.pnl >= 0 else f"-₹{abs(trade.pnl):.2f}"
        msg = f"✅ Trade Closed: {trade.instrument} @ {trade.exit_price:.2f} (Reason: {reason}) | PnL: {profit_str}"
        logger.info(msg)
        self._log_ui(msg, "trade")
        # Notify only after broker reconciliation/final accounting has made the
        # terminal state authoritative. An exit-order acknowledgement is not a
        # fill, so callers must not write terminal telemetry from the ACK path.
        if callable(on_trade_closed):
            try:
                on_trade_closed(trade)
            except Exception as exc:
                # Observability must never unwind a completed broker lifecycle.
                logger.error(f"Trade-closed callback failed for {trade.id}: {exc}")
            finally:
                with self._lock:
                    terminal_owners.discard(terminal_lifecycle)
        
        return True

    @staticmethod
    def _expected_position_sign(trade: Trade) -> int:
        if trade.inst_type == "OPT":
            return 1
        return 1 if trade.direction in {"LONG", "BUY"} else -1

    @staticmethod
    def _slippage_sample_is_a_real_fill(trade: Trade, leg: str) -> bool:
        """Whether this leg may contribute to the realized-slippage EWMA.

        SlippageTracker.record documents itself as taking "one real fill", and
        trade_accounting lets the measured EWMA REPLACE the static per-leg assumption
        once slippage_calibration_min_samples is reached. So anything that is not a
        broker fill does not merely add noise — it overrides the model with itself.

        2026-08-11: slippage_stats.json held 12,051 FUT samples at ewma_frac exactly
        0.0, every row stamped last_leg "ENTRY". The exit leg has always required an
        exit_order_id and skipped a fill equal to its reference; the entry leg required
        neither, so simulated entries whose "fill" was the signal price recorded a
        perfect zero, 12,051 times. That drove modelled futures slippage to the floor of
        _FUT_CALIBRATION_BOUNDS (0.2bp) — a backtest grading its own execution.

        Two necessary conditions for a real fill, and both are plain dataclass fields:
        the trade is not a ghost, and the leg has a broker order id.

        2026-08-15: those two conditions were not enough. The store had grown to 14,244
        FUT samples -- 2,193 of them added AFTER the guard above -- still at ewma_frac
        exactly 0.0, still refreshing (00:55, market closed, no order in the log), while
        every one of the 103 persisted trades was an OPTION. Nothing that trades options
        can measure futures execution, so the samples come from a simulated futures path
        that satisfies both conditions: not flagged a ghost, and carrying an order id.

        Refreshing is what makes it dangerous rather than merely wrong. The age bound
        added alongside the guard exists to stop stale rows outranking the static model,
        and rows that keep updating defeat it: calibrated_fraction("FUT") returns 0.0
        today, clamped to the 0.2bp floor of _FUT_CALIBRATION_BOUNDS, in place of the
        2.0bp static assumption. Futures are modelled ten times cheaper than configured.

        So the sample must also come from a LIVE entry -- HISTORICAL and RECOVERY
        reconstruct a fill rather than observe one -- and must not be a futures shadow
        companion, which is synthetic by construction.
        """
        if bool(getattr(trade, "is_ghost", False)):
            return False
        if str(getattr(trade, "trading_symbol", "") or "").upper() == "FUT_SHADOW":
            return False
        # Absent provenance is not evidence of a live fill.
        if str(getattr(trade, "data_provenance", "") or "").upper() != "LIVE":
            return False
        order_id = (
            getattr(trade, "exit_order_id", "")
            if str(leg or "").upper() == "EXIT"
            else getattr(trade, "broker_order_id", "")
        )
        return bool(str(order_id or "").strip())

    @staticmethod
    def _stamp_fill_quality(trade: Trade, *, leg: str, requested: float) -> None:
        """Record HOW the order filled, durably, at the moment of confirmation.

        Slippage is measured against a reference that lives only in memory:
        signal_entry_price for the entry, the decision price stamped into exit_price
        for the exit. entry_price is then OVERWRITTEN by the broker fill. So once the
        row is saved the price that was asked for is gone, and realised slippage can
        never be recomputed from stored history -- the in-memory EWMA was the only
        consumer that ever saw it, and a restart erased the evidence behind it.

        Writing the requested price beside the filled one makes the row self-sufficient:
        entry_price - requested_entry_price IS the slippage, re-derivable at any time,
        which is what a static assumption has to be tuned against.

        Latency is measured only when a submit time exists. It is genuinely absent on a
        recovered or reconstructed leg, and a fabricated zero there would read as an
        instant fill.
        """
        try:
            reference = float(requested or 0.0)
            if reference <= 0:
                return
            now = datetime.now(IST)
            if str(leg or "").upper() == "EXIT":
                trade.requested_exit_price = round(reference, 4)
                trade.exit_filled_at = now
                submitted = getattr(trade, "exit_submitted_at", None)
                if submitted is not None:
                    trade.exit_fill_latency_ms = round(
                        max(0.0, (now - submitted).total_seconds() * 1000.0), 1
                    )
                return
            trade.requested_entry_price = round(reference, 4)
            trade.entry_filled_at = now
            submitted = getattr(trade, "entry_submitted_at", None)
            if submitted is not None:
                trade.fill_latency_ms = round(
                    max(0.0, (now - submitted).total_seconds() * 1000.0), 1
                )
        except Exception as exc:
            logger.debug(f"Fill-quality stamp skipped for {getattr(trade, 'id', '')}: {exc}")

    @staticmethod
    def _record_entry_slippage(trade: Trade, fill_price: float) -> None:
        """Measure the real entry fill against the signal price that triggered it.

        Adverse is signed positive when the fill was worse than the signal:
        entries are BUYs for all options and LONG futures; SHORT futures enter
        by SELLING, so worse = lower fill.
        """
        try:
            reference = float(getattr(trade, "signal_entry_price", 0.0) or 0.0)
            fill = float(fill_price or 0.0)
            if reference <= 0 or fill <= 0:
                return
            # entry_slippage itself is still stamped below for any trade — it is
            # reporting. Only the shared EWMA, which overrides the cost model, is
            # restricted to real broker fills.
            record_to_ewma = TradeManager._slippage_sample_is_a_real_fill(trade, "ENTRY")
            if trade.inst_type == "OPT" or trade.direction in {"LONG", "BUY"}:
                adverse = fill - reference
            else:
                adverse = reference - fill
            trade.entry_slippage = round(adverse, 4)
            # A genuine zero from a real fill is deliberately KEPT: it is a measurement,
            # and discarding it would bias the EWMA toward adverse fills. The 14,244
            # poisoned samples were all exact zeros, but zero was their symptom, not the
            # defect — they were simulated. _slippage_sample_is_a_real_fill excludes them
            # at the source, which is the precise fix; filtering zeros here would be the
            # blunt one, and would throw away real execution data with it.
            if not record_to_ewma:
                return
            from engine.slippage_tracker import get_slippage_tracker

            get_slippage_tracker().record(
                inst_type=trade.inst_type,
                instrument=trade.instrument,
                reference_price=reference,
                adverse_points=adverse,
                leg="ENTRY",
            )
        except Exception as exc:
            logger.debug(f"Entry slippage recording skipped: {exc}")

    @staticmethod
    def _record_exit_slippage(trade: Trade, decision_price: float, fill_price: float) -> None:
        """Measure the real exit fill against the price at the exit decision.

        Exits reverse the entry side: options and LONG futures exit by SELLING
        (worse = lower fill); SHORT futures exit by BUYING (worse = higher fill).
        """
        try:
            reference = float(decision_price or 0.0)
            fill = float(fill_price or 0.0)
            if reference <= 0 or fill <= 0 or abs(fill - reference) < 1e-9:
                return
            if trade.inst_type == "OPT" or trade.direction in {"LONG", "BUY"}:
                adverse = reference - fill
            else:
                adverse = fill - reference
            trade.exit_slippage = round(adverse, 4)
            if not TradeManager._slippage_sample_is_a_real_fill(trade, "EXIT"):
                return
            from engine.slippage_tracker import get_slippage_tracker

            get_slippage_tracker().record(
                inst_type=trade.inst_type,
                instrument=trade.instrument,
                reference_price=reference,
                adverse_points=adverse,
                leg="EXIT",
            )
        except Exception as exc:
            logger.debug(f"Exit slippage recording skipped: {exc}")

    @staticmethod
    def _order_status_text(order: Optional[Dict]) -> str:
        if not isinstance(order, dict):
            return ""
        return str(
            order.get("orderstatus")
            or order.get("status")
            or order.get("orderStatus")
            or ""
        ).strip().lower().replace("_", " ")

    @staticmethod
    def _aware_ist(value: Optional[datetime]) -> datetime:
        value = value or datetime.now(IST)
        return IST.localize(value) if value.tzinfo is None else value.astimezone(IST)

    def _mark_recovery(self, trade: Trade, reason: str) -> None:
        trade.execution_status = "RECOVERY_REQUIRED"
        trade.execution_error = str(reason or "Broker reconciliation required")
        if trade.recovery_required_since is None:
            trade.recovery_required_since = datetime.now(IST)
        self._alert_recovery(trade)

    def _alert_recovery(self, trade: Trade) -> None:
        if not hasattr(self, "_last_recovery_alert"):
            self._last_recovery_alert = {}
        now = time.monotonic()
        last = float(self._last_recovery_alert.get(trade.id, 0.0) or 0.0)
        if now - last < 30.0:
            return
        self._last_recovery_alert[trade.id] = now
        message = (
            f"RECOVERY REQUIRED: {trade.trading_symbol or trade.instrument}. "
            f"{trade.execution_error or 'Broker state is ambiguous.'}"
        )
        logger.critical(message)
        self._log_ui(message, "error")
        try:
            notifier = getattr(self, "on_notification", None)
            if not callable(notifier):
                from notifications import send_desktop_notification as notifier
            notifier(message, "error")
        except Exception as exc:
            logger.debug(f"Recovery desktop alert unavailable: {exc}")

    # Broker conditions that refuse EVERY order until a human changes something
    # outside this process. Matched against the broker's own refusal text.
    _BLOCKING_BROKER_CONDITIONS = (
        # "registered ip" is the AngelOne-era wording and was the ONLY IP marker here
        # until 2026-08-16, so a Kite block never reached this alert at all. The refusal
        # this project actually captured on 2026-08-04 reads:
        #   PermissionException: IP (2405:201:...:1b61) is not allowed to place orders
        #   for this app. Update allowed IPs on the Kite developer console.
        # which contains none of the four markers that were listed. KiteBroker's own
        # _escalate_if_ip_blocked did fire, but it is logger.critical only — this tuple
        # is the single path that reaches _log_ui and the desktop notifier, so the one
        # condition that kills every order was also the one the operator never saw.
        # Both Kite shapes are covered: a refused address, and an empty allow-list.
        (
            "is not allowed to place orders",
            "The broker is refusing orders from this machine's public IP ({detail}). "
            "Add that address on the Kite developer console (developers.kite.trade -> "
            "your app) and restart. Check the address family: a whitelisted IPv4 does "
            "NOT cover an IPv6 egress. Until then NO order can reach the exchange.",
        ),
        (
            "allowed ips",
            "The broker is refusing orders on the app's allowed-IP list ({detail}). "
            "Add this machine's address on the Kite developer console. An EMPTY "
            "allowed-IP list denies every order rather than allowing all.",
        ),
        (
            "registered ip",
            "The broker is refusing orders from this machine's public IP ({detail}). "
            "Add that address on the Kite developer console (developers.kite.trade -> "
            "your app) and restart. Check the address family: a whitelisted IPv4 does "
            "NOT cover an IPv6 egress. An EMPTY allowed-IP list denies every order "
            "rather than allowing all. Until then NO order can reach the exchange.",
        ),
        (
            "invalid token",
            "The Kite session token is invalid. Re-login through the dashboard and "
            "restart; no order can reach the exchange until the session is renewed.",
        ),
        (
            "invalid session",
            "The Kite session has expired. Re-login through the dashboard and restart; "
            "no order can reach the exchange until the session is renewed.",
        ),
        (
            "session expired",
            "The Kite session has expired. Re-login through the dashboard and restart; "
            "no order can reach the exchange until the session is renewed.",
        ),
    )

    def _alert_blocking_broker_condition(self, error: str) -> None:
        """Escalate a refusal that no retry and no later signal can get past.

        A generic "entry failed" line scrolls past in a busy log. On 2026-07-31 the
        account refused every order from 14:51 onward with AG7002 and the session ran
        to the close still trying, because nothing separated "this order failed" from
        "nothing can succeed until you change something".
        """
        text = str(error or "").lower()
        for marker, template in self._BLOCKING_BROKER_CONDITIONS:
            if marker not in text:
                continue
            if not hasattr(self, "_last_blocking_alert"):
                self._last_blocking_alert = {}
            now = time.monotonic()
            if now - float(self._last_blocking_alert.get(marker, 0.0) or 0.0) < 600.0:
                return
            self._last_blocking_alert[marker] = now
            # The broker names the offending IP in its own message; quote it back
            # rather than guessing, because the SDK's X-ClientPublicIP header is a
            # hardcoded constant and never reflects this machine.
            # IPv4-only until 2026-08-16 — and the incident this alert exists for was an
            # IPv6 egress, so on the very case it was written for the address rendered
            # as "unknown". Parenthesised form first (what Kite actually sent), then a
            # bare IPv6, then a bare IPv4.
            text_for_ip = str(error or "")
            found = (
                re.search(r"\bIP\s+\(([^)]+)\)", text_for_ip)
                # At least FOUR colon groups. A looser {2,7} also matches a clock time
                # such as 12:37:25, and these messages routinely carry one.
                or re.search(r"\b([0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){4,7})\b", text_for_ip)
                or re.search(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b", text_for_ip)
            )
            # group(1), not group(0): the parenthesised pattern's group(0) carries the
            # literal "IP (" prefix along with the address.
            message = "TRADING BLOCKED: " + template.format(
                detail=found.group(1) if found else "unknown"
            )
            logger.critical(message)
            self._log_ui(message, "error")
            try:
                notifier = getattr(self, "on_notification", None)
                if not callable(notifier):
                    from notifications import send_desktop_notification as notifier
                notifier(message, "error")
            except Exception as exc:
                logger.debug(f"Blocking-condition desktop alert unavailable: {exc}")
            return

    def _protective_exit_side(self, trade: Trade) -> str:
        return "SELL" if trade.inst_type == "OPT" or trade.direction in {"LONG", "BUY"} else "BUY"

    def _protective_trigger_for(self, trade: Trade) -> float:
        stop = float(getattr(trade, "current_stop", 0.0) or getattr(trade, "trailing_stop", 0.0) or 0.0)
        if stop > 0:
            return stop
        if trade.inst_type == "OPT":
            pct = self.options_sl_pct
        else:
            from engine.market_regime import futures_stop_tolerance_pct

            pct = futures_stop_tolerance_pct(
                self.futures_sl_pct,
                getattr(trade, "market_regime", "UNKNOWN"),
            )
        if trade.inst_type == "OPT" or trade.direction in {"LONG", "BUY"}:
            return max(0.05, trade.entry_price * (1.0 - pct / 100.0))
        return trade.entry_price * (1.0 + pct / 100.0)

    def on_broker_postback(self, payload: Dict[str, Any]) -> bool:
        """Apply a verified Zerodha order postback to the matching trade.

        Push beats polling here: the postback is the broker telling us what happened,
        so a REJECTED entry is known immediately with its reason instead of being
        inferred from an order-book scan on the next cycle.

        Deliberately conservative — it records state and surfaces the reason, and does
        NOT close or open positions. Reconciliation against actual broker positions
        stays the single authority for that, so a duplicated or out-of-order postback
        can never invent a fill.
        """
        order_id = str((payload or {}).get("order_id") or "")
        status = str((payload or {}).get("status") or "").upper()
        if not order_id:
            return False
        # The order API reports CANCELLED; the postbacks prose uses CANCEL. Accept both.
        if status == "CANCEL":
            status = "CANCELLED"

        stamp = str(
            payload.get("exchange_update_timestamp")
            or payload.get("exchange_timestamp")
            or payload.get("order_timestamp")
            or ""
        )

        # Whole body under the lock as of 2026-08-16. It used to take the lock ONLY to
        # find the trade and then mutate outside it, so a postback could rewrite
        # broker_quantity while the exit path was reading it.
        with self._lock:
            trade, leg = self._trade_for_broker_order(order_id)
            if trade is None:
                return False

            # Idempotency + ordering. Zerodha redelivers, and delivery order is not
            # guaranteed. Neither was checked: the concrete hazard is a delayed
            # entry-COMPLETE arriving after a partial reduce and rewriting
            # broker_quantity back to the pre-reduce fill — moments before an SL-M is
            # re-armed from exactly that field, for a quantity no longer held.
            applied = getattr(self, "_postback_applied", None)
            if applied is None:
                applied = self._postback_applied = {}
            seen = applied.get(order_id)
            if seen is not None:
                prev_stamp, prev_status = seen
                if prev_status in _POSTBACK_TERMINAL_STATUSES and status not in _POSTBACK_TERMINAL_STATUSES:
                    return False          # a live-state update arriving after a final one
                if stamp and prev_stamp:
                    if stamp < prev_stamp:
                        return False      # strictly older than what we already applied
                    if stamp == prev_stamp and status == prev_status:
                        return False      # verbatim redelivery
            applied[order_id] = (stamp, status)

            message = str(payload.get("status_message") or "").strip()
            try:
                filled = int(float(payload.get("filled_quantity") or 0))
            except (TypeError, ValueError):
                filled = 0
            try:
                average = float(payload.get("average_price") or 0.0)
            except (TypeError, ValueError):
                average = 0.0

            if status == "REJECTED":
                trade.execution_error = message or "Broker rejected the order"
                logger.error(
                    f"Postback: {trade.trading_symbol} {leg} REJECTED — {trade.execution_error}"
                )
                self._alert_blocking_broker_condition(trade.execution_error)
            elif status == "COMPLETE":
                if filled > 0:
                    # broker_quantity describes the ENTRY holding, so only the entry leg
                    # may set it. Before this, an exit or stop fill was never matched at
                    # all; matching it and then writing its quantity here would be worse
                    # than the bug it replaces.
                    if leg == "ENTRY":
                        trade.broker_quantity = filled
                    # SIGNED. _record_fill's contract is "an entry or add is positive, an
                    # exit negative, so a trade's holding is the sum" — and the first cut
                    # of this rewrite passed `filled` positive for every leg. Matching the
                    # exit legs (A2) without their sign was worse than not matching them:
                    # a protective stop filling 65 wrote +65 into the ledger, so a FLAT
                    # position reported 130 owned, and that total feeds exit sizing, the
                    # frozen 1R and the scale-in gate. Order id is the ledger's primary
                    # key under INSERT OR REPLACE, and for an exchange-side SL-M there is
                    # no competing correctly-signed writer at all.
                    self._record_fill(
                        trade, order_id, _POSTBACK_LEG_SIGNS.get(leg, 1) * filled, leg
                    )
                    if leg == "PROTECTIVE":
                        # Reconciliation remains the sole authority that closes the
                        # position, but it must retain what the broker just proved:
                        # this flatten was caused by the exchange-side stop. Without
                        # this hint a subsequent net_qty=0 row was labelled
                        # MANUAL_SYNC, disabling stop-out re-entry and corrupting exit
                        # analytics. exit_reason is already restart-persistent.
                        trade.exit_reason = "STOP_HIT"
                if average > 0 and leg == "ENTRY":
                    trade.execution_error = ""
                logger.info(
                    f"Postback: {trade.trading_symbol} {leg} COMPLETE filled={filled} avg={average}"
                )
            elif status == "UPDATE":
                # Partial fills arrive as UPDATE. There was no branch for it and no
                # else, so the push channel was structurally blind to every partial.
                # Recorded only — reconciliation remains the sole authority that may
                # open or close a position, which is what keeps a duplicated or
                # out-of-order postback from inventing a fill.
                logger.info(
                    f"Postback: {trade.trading_symbol} {leg} partial fill "
                    f"filled={filled} avg={average}"
                )
            elif status == "CANCELLED":
                trade.execution_error = message or "Broker cancelled the order"
        self._mark_dirty()
        return True

    # Order-id fields in the order a postback should be attributed. Until 2026-08-16
    # ONLY broker_order_id was matched, so every postback for an exit, a protective
    # stop, a partial reduce or a scale-in returned False and was dropped — including
    # "your protective stop just filled", which is the commonest real exit there is.
    _POSTBACK_LEG_FIELDS = (
        ("broker_order_id", "ENTRY"),
        ("exit_order_id", "EXIT"),
        ("protective_order_id", "PROTECTIVE"),
        ("protective_stop_order_id", "PROTECTIVE"),
        ("partial_reduce_order_id", "PARTIAL_EXIT"),
        ("scale_in_order_id", "ADD"),
    )

    def _trade_for_broker_order(self, order_id: str):
        """(trade, leg) owning this broker order id, or (None, "")."""
        wanted = str(order_id or "").strip()
        if not wanted:
            return None, ""
        for trade in list(self.open_trades.values()):
            for field, leg in self._POSTBACK_LEG_FIELDS:
                if str(getattr(trade, field, "") or "").strip() == wanted:
                    return trade, leg
        return None, ""

    def _withdraw_stale_entry_remainder(self, trade: Trade, filled_qty: int, expected_qty: int) -> None:
        """Cancel the unfilled tail of a partially filled ENTRY once it has gone stale.

        The 12s cancel/replace timer only runs while `position is None` and
        execution_status == 'ENTRY_SUBMITTED'. The instant a partial fill lands, a
        position row exists and the status becomes 'PARTIAL', so BOTH preconditions are
        permanently false and nothing ages the remainder again. The PARTIAL branch
        re-reads the fill and re-arms protection every pass, but never touches the order
        still resting at the exchange.

        So a 3-lot entry that fills 1 leaves 2 working all session. Hours later a spike
        sweeps them: the position becomes 3 lots while the protective stop covers 1, and
        until the next reconcile tick two lots are naked. Meanwhile P&L, risk headroom
        and the concentration guards were all sized on 1 lot for the whole interval.

        The cancel is PROVEN, not acknowledged — same rule as the stale-entry replace.
        cancel_order_confirmed answers True only for a broker-confirmed terminal
        non-fill, False when the order actually filled, and None when it cannot tell.
        A False or None must NOT be read as "the remainder is gone": the next reconcile
        pass re-reads the position and arms protection for whatever really filled.
        """
        order_id = str(getattr(trade, "broker_order_id", "") or "")
        if not order_id or filled_qty >= expected_qty:
            return
        if getattr(trade, "entry_remainder_withdrawn", False):
            return
        from config.settings import get_settings

        settings = get_settings()
        if not _setting_enabled(
            getattr(settings, "partial_entry_remainder_cancel_enabled", True), True
        ):
            return
        timeout = max(
            5.0,
            float(getattr(settings, "partial_entry_remainder_timeout_seconds", 120.0) or 120.0),
        )
        submitted = getattr(trade, "entry_submitted_at", None)
        if submitted is None:
            return
        try:
            if isinstance(submitted, str):
                submitted = datetime.fromisoformat(submitted)
            if submitted.tzinfo is None:
                submitted = IST.localize(submitted)
            age = (datetime.now(IST) - submitted.astimezone(IST)).total_seconds()
        except Exception:
            return
        if age < timeout:
            return

        confirm = getattr(self.broker, "cancel_order_confirmed", None)
        if not callable(confirm):
            return
        try:
            cancelled = confirm(order_id, "NORMAL")
        except Exception as exc:
            logger.error(
                f"Could not withdraw the unfilled remainder of {trade.id} "
                f"({trade.trading_symbol}): {exc}"
            )
            return

        if cancelled is True:
            trade.entry_remainder_withdrawn = True
            trade.pending_execution = False
            trade.execution_error = (
                f"Entry filled {filled_qty} of {expected_qty}; unfilled remainder "
                "withdrawn after it went stale"
            )
            logger.warning(
                f"[PARTIAL ENTRY] {trade.id} {trade.trading_symbol}: filled "
                f"{filled_qty}/{expected_qty} and the remainder rested {age:.0f}s. "
                "Cancel proven — position stands at the filled quantity."
            )
            self._log_ui(
                f"Withdrew the unfilled {expected_qty - filled_qty} of "
                f"{trade.trading_symbol}; the trade stands at {filled_qty}.",
                "warning",
            )
        elif cancelled is False:
            # It filled while we were cancelling. Say so and let the next reconcile
            # pass size protection on the real quantity.
            logger.warning(
                f"[PARTIAL ENTRY] {trade.id} {trade.trading_symbol}: the remainder "
                "filled during the cancel; protection will be re-armed on the full size."
            )
        else:
            logger.error(
                f"[PARTIAL ENTRY] {trade.id} {trade.trading_symbol}: remainder state "
                "UNKNOWN after a cancel attempt; leaving it for the next pass rather "
                "than assuming it is gone."
            )

    def _arm_gtt_fallback(self, trade: Trade, qty: int, side: str, trigger: float) -> bool:
        """Arm a server-side GTT stop. Only valid when NO SL-M is working.

        A GTT and an SL-M that both fire exit the position twice, and the second fill
        opens a naked reversed leg — so this must never run alongside a live stop.
        """
        placer = getattr(self.broker, "place_gtt_stop", None)
        if not callable(placer):
            return False
        if str(getattr(trade, "gtt_trigger_id", "") or ""):
            return True                      # already armed; never stack triggers
        # Nor stack one we armed but never recorded — a lost acknowledgement on a
        # previous cycle leaves an active trigger with no local id.
        finder = getattr(self.broker, "find_gtt_stop", None)
        scan_state = ""
        if callable(finder):
            def _scan() -> dict:
                try:
                    return finder(trade.trading_symbol) or {}
                except Exception as exc:
                    # A raising finder is the same evidence as UNKNOWN: we cannot
                    # tell. Collapsing it to {} made it indistinguishable from NONE.
                    return {"state": "UNKNOWN", "trigger_id": "", "reason": str(exc)}

            prior = _scan()
            if str(prior.get("state") or "") == "UNKNOWN":
                # "Cannot tell" was falling straight through to placer() — the same
                # never-conclude-from-a-failure rule the SL-M path already follows,
                # broken here. One retry usually resolves a transient scan failure.
                prior = _scan()
            scan_state = str(prior.get("state") or "")
            prior_id = str(prior.get("trigger_id") or "")
            if scan_state == "FOUND" and prior_id:
                # find_gtt_stop matches the CONTRACT, not the trade — no side, quantity
                # or trigger level. Two trades on one strike is supported, so adopting
                # blindly points both at one trigger: this trade would sit behind the
                # OTHER trade's stop level, and whichever exits first cancels it and
                # silently strips the survivor's server-side protection.
                owner = self._trade_owning_gtt(prior_id, exclude_id=trade.id)
                if owner is not None:
                    logger.error(
                        f"GTT {prior_id} on {trade.trading_symbol} belongs to trade "
                        f"{owner.id}; not adopting it for {trade.id}."
                    )
                    return False
                if not self._gtt_matches_trade(prior, trigger):
                    logger.error(
                        f"Active GTT {prior_id} on {trade.trading_symbol} does not match "
                        f"this trade's trigger {trigger:.2f}; not adopting it."
                    )
                    return False
                trade.gtt_trigger_id = prior_id
                logger.warning(
                    f"Adopted existing GTT {prior_id} for {trade.trading_symbol} "
                    "instead of arming a second one."
                )
                return True
        # Still UNKNOWN after the retry. Arm anyway — this runs only because the SL-M
        # was refused, so declining would leave the position with NO server-side stop,
        # which is worse than a possible duplicate trigger. But flag it, so a duplicate
        # is reconciled by a human rather than sitting silent until both fire.
        if scan_state == "UNKNOWN":
            logger.error(
                f"GTT scan for {trade.trading_symbol} is UNKNOWN after a retry; arming "
                "anyway because the position is otherwise unprotected. A duplicate "
                "trigger may exist."
            )
            trade.gtt_arm_uncertain = True

        try:
            trigger_id = placer(
                trade.trading_symbol, qty, side, trigger,
                last_price=float(getattr(trade, "current_price", 0.0) or trade.entry_price),
                product_type=self.product_type,
            )
        except BrokerOrderAmbiguous as exc:
            # The trigger may or may not exist server-side and we have no id for it.
            # This is exactly what creates the orphan the scan above cannot see, so
            # record it rather than returning a bare False.
            trade.gtt_arm_uncertain = True
            self._mark_recovery(
                trade,
                f"GTT arm ambiguous for {trade.trading_symbol}: {exc}. A server-side "
                "trigger may exist with no local id.",
            )
            logger.error(f"GTT fallback ambiguous for {trade.trading_symbol}: {exc}")
            return False
        except Exception as exc:
            logger.error(f"GTT fallback failed for {trade.trading_symbol}: {exc}")
            return False
        if not trigger_id:
            return False
        trade.gtt_trigger_id = str(trigger_id)
        logger.critical(
            f"GTT fallback armed for {trade.trading_symbol} at {trigger:.2f} "
            f"(id {trigger_id}) — the position is protected server-side even if this "
            "process stops. Reconcile the missing SL-M."
        )
        return True

    def _trade_owning_gtt(self, trigger_id: str, exclude_id: str = ""):
        """The open trade already holding this GTT trigger id, if any."""
        if not trigger_id:
            return None
        with self._lock:
            for other in list(getattr(self, "open_trades", {}).values()):
                if other.id == exclude_id:
                    continue
                if str(getattr(other, "gtt_trigger_id", "") or "") == str(trigger_id):
                    return other
        return None

    @staticmethod
    def _gtt_matches_trade(row: Dict, trigger: float) -> bool:
        """Only adopt a trigger armed at this trade's own stop level.

        A GTT at someone else's level is not protection for this position — it fires
        early or late by exactly the difference between the two stops.
        """
        values = (row or {}).get("trigger_values") or []
        if not values:
            return False
        try:
            return any(abs(float(v) - float(trigger)) <= 0.051 for v in values)
        except (TypeError, ValueError):
            return False

    def _release_gtt_fallback(self, trade: Trade, note: str) -> bool:
        """Cancel the GTT once real stop management takes over, or before flattening.

        Returns False when the broker refused, so callers can withhold an exit rather
        than market out on top of a trigger that is still armed.
        """
        trigger_id = str(getattr(trade, "gtt_trigger_id", "") or "")
        if not trigger_id:
            return True
        canceller = getattr(self.broker, "cancel_gtt_stop", None)
        if not callable(canceller):
            return True
        try:
            cancelled = bool(canceller(trigger_id))
        except Exception as exc:
            logger.error(f"GTT cancellation raised for {trade.trading_symbol}: {exc}")
            return False
        if not cancelled:
            logger.error(
                f"GTT {trigger_id} for {trade.trading_symbol} could NOT be cancelled; "
                "withholding actions that could double-exit the position."
            )
            return False
        trade.gtt_trigger_id = ""
        logger.info(f"GTT {trigger_id} released for {trade.trading_symbol}: {note}")
        return True

    @staticmethod
    def _verified_uts_protective_stop(order, trade, quantity, expected_trigger) -> bool:
        """Match the actual exchange stop before a UTS extension or its restoration."""
        if not isinstance(order, dict):
            return False
        try:
            actual = float(order.get("triggerprice") or order.get("trigger_price") or 0.0)
            units = float(order.get("quantity", 0))
            filled = float(order.get("filledshares") or order.get("filled_quantity") or 0.0)
        except (TypeError, ValueError):
            return False
        return (
            TradeManager._order_status_text(order) in {"trigger pending", "triggerpending"}
            and str(order.get("orderid") or order.get("order_id") or "") == str(trade.protective_order_id)
            and str(order.get("tradingsymbol") or order.get("trading_symbol") or "") == trade.trading_symbol
            and str(order.get("transactiontype") or order.get("transaction_type") or "").upper() == "SELL"
            and all(math.isfinite(value) for value in (actual, units, filled))
            and actual >= expected_trigger - 1e-9 and abs(actual - expected_trigger) <= 0.05 + 1e-9
            and units == quantity and filled == 0
        )

    def request_uts_stop_extension(self, trade, stop, *, quote_verified: bool, now) -> bool:
        """Queue one UTS exception; the original stop stays active until broker acknowledgement."""
        from engine import uts_lane
        if not quote_verified or not uts_lane.is_uts(trade) or stop <= 0:
            return False
        guard = getattr(self, "uts_stop_extension_guard", None)
        entry = float(getattr(trade, "entry_price", 0.0) or 0.0)
        current = float(getattr(trade, "current_stop", 0.0) or 0.0)
        recorded = getattr(trade, "quality_verdict", {}) or {}
        checks = recorded.get("checks") or {}
        _, normal, wide = uts_lane.premium_cap_tiers(trade)
        normal_ratio, wide_ratio = 1.0 - normal / 100., 1.0 - wide / 100.
        approach_ratio = 1.0 - normal * .7 / 100.
        if (entry <= 0 or recorded.get("policy_version") != "uts_native.v1"
                or checks.get("uts_premium_cap_pct") != normal
                or checks.get("uts_stop_extension_attempted") or checks.get("uts_stop_extended")
                or str(getattr(trade, "timeframe", "")) != "5min"
                or abs(current / entry - normal_ratio) > 0.001 or stop < entry * wide_ratio - 1e-9
                or abs(stop / entry - wide_ratio) > 0.001
                or not current < float(getattr(trade, "current_price", 0.0)) <= entry * approach_ratio
                or getattr(self, "options_sl_pct", 15.0) < wide
                or getattr(trade, "is_recovery", False)
                or stop >= current or getattr(trade, "pending_exit", False)):
            return False
        if not callable(guard) or not guard(trade, stop):
            return False
        if self.mode != "REAL" or getattr(trade, "is_ghost", False):
            uts_lane.commit_extension(trade, stop, now)
            self._mark_dirty()
            return True
        if not getattr(trade, "protective_order_id", ""):
            return False
        if not getattr(self, "_protection_lock", None):
            self._protection_lock = threading.RLock()
        with self._protection_lock:
            if getattr(trade, "_uts_extension_request", None):
                return False
            trade._uts_extension_request = dict(stop=float(stop), at=time.monotonic(), clock=now)
        self.nudge_position_sync()
        return False

    def _apply_uts_stop_extension(self, trade, quantity) -> bool:
        """Called only under the broker protection lock; never relax a touched/profit stop."""
        request = getattr(trade, "_uts_extension_request", None)
        if not request:
            return False
        trade._uts_extension_request = None
        from engine import uts_lane
        recorded = dict(getattr(trade, "quality_verdict", {}) or {})
        checks = dict(recorded.get("checks") or {})
        entry, stop, price = float(trade.entry_price), float(trade.current_stop), float(trade.current_price)
        desired = math.ceil(float(request["stop"]) / 0.05 - 1e-9) * 0.05
        guard = getattr(self, "uts_stop_extension_guard", None)
        _, normal, wide = uts_lane.premium_cap_tiers(trade)
        normal_ratio, wide_ratio = 1.0 - normal / 100., 1.0 - wide / 100.
        approach_ratio = 1.0 - normal * .7 / 100.
        if (not uts_lane.is_uts(trade) or recorded.get("policy_version") != "uts_native.v1"
                or entry <= 0 or checks.get("uts_premium_cap_pct") != normal
                or trade.timeframe != "5min" or getattr(trade, "is_recovery", False)
                or getattr(self, "options_sl_pct", 15.0) < wide
                or checks.get("uts_stop_extension_attempted") or checks.get("uts_stop_extended")
                or time.monotonic() - request["at"] > 2.0 or trade.pending_exit
                or getattr(trade, "quote_verified", True) is False
                or getattr(trade, "_uts_stop_touch_pending", False)
                or abs(stop / entry - normal_ratio) > 0.001 or not stop < price <= entry * approach_ratio
                or desired < entry * wide_ratio - 1e-9 or desired > entry * wide_ratio + 0.05 or desired >= stop
                or not callable(guard) or not guard(trade, desired)):
            return False
        existing = str(getattr(trade, "protective_order_id", "") or "")
        qty = self._transactable_quantity(trade, quantity)
        if not existing or qty <= 0:
            return False
        if self._trade_owning_protective_order(existing, exclude_id=trade.id) is not None:
            return False
        getter = getattr(self.broker, "get_order_status", None)

        try:
            before = getter(existing) if callable(getter) else None
        except Exception:
            return False
        if not self._verified_uts_protective_stop(before, trade, qty, stop):
            return False  # Never relax a tighter, mismatched, triggered or unknown stop.
        # A slow status request can outlive the permission window, or another
        # cycle can observe a stop touch while that broker read is in flight.
        if (trade.pending_exit or getattr(trade, "_uts_stop_touch_pending", False)
                or getattr(trade, "quote_verified", True) is False
                or float(trade.current_stop) != stop
                or not stop < float(trade.current_price) <= entry * approach_ratio
                or not guard(trade, desired) or time.monotonic() - request["at"] > 2.0):
            return False
        checks["uts_stop_extension_attempted"] = True
        recorded["checks"] = checks
        trade.quality_verdict = recorded
        self._mark_dirty()
        try:
            accepted = self.broker.modify_protective_stop(existing, trade.trading_symbol,
                trade.symbol_token, qty, "SELL", round(desired, 2), self.product_type)
        except Exception as exc:
            logger.warning("UTS stop extension refused: {}", type(exc).__name__)
            accepted = False
        if not accepted:
            # A timeout/refusal can arrive after the exchange accepted the write.
            # Preserve the tighter local stop and force conservative reconciliation.
            trade.protective_trigger = min(float(getattr(trade, "protective_trigger", 0.0) or desired), desired)
            trade.protective_quantity = qty
            checks["uts_stop_extension_uncertain"] = True
            trade.quality_verdict = {**recorded, "checks": checks}
            self._mark_recovery(trade, "UTS stop extension not acknowledged; restore original protection")
            self._mark_dirty()
            self.nudge_position_sync()
            return False
        try:
            after = getter(existing)
        except Exception:
            after = None
        if (not self._verified_uts_protective_stop(after, trade, qty, desired)
                or trade.pending_exit or getattr(trade, "_uts_stop_touch_pending", False)
                or getattr(trade, "quote_verified", True) is False
                or float(trade.current_stop) != stop
                or not stop < float(trade.current_price) <= entry * approach_ratio
                or not guard(trade, desired)
                or self._transactable_quantity(trade, quantity) != qty
                or time.monotonic() - request["at"] > 2.0):
            # The broker may already hold the wider stop even though confirmation or
            # this cycle's permission disappeared. Keep the local stop, account for
            # that possible lower trigger and let normal protection tighten it back.
            trade.protective_trigger = min(float(getattr(trade, "protective_trigger", 0.0) or desired), desired)
            trade.protective_quantity = qty
            checks["uts_stop_extension_uncertain"] = True
            trade.quality_verdict = {**recorded, "checks": checks}
            self._mark_recovery(trade, "UTS stop extension state uncertain; restore original protection")
            self._mark_dirty()
            self.nudge_position_sync()
            return False
        uts_lane.commit_extension(trade, round(desired, 2), request["clock"])
        trade._entry_option_stop_cap_pct = wide
        trade.protective_trigger = round(desired, 2)
        trade.protective_quantity = qty
        trade.protective_updated_at = datetime.now(IST)
        self._mark_dirty()
        return True

    def ensure_broker_protection(self, trade: Trade, quantity: Optional[int] = None) -> bool:
        """Create or tighten the exchange-side SL-M for a confirmed REAL position."""
        if getattr(self, "mode", "HISTORICAL") != "REAL" or getattr(trade, "is_ghost", False):
            return True
        if not self.broker or not trade.symbol_token or not trade.trading_symbol:
            self._mark_recovery(trade, "Cannot establish broker protection without broker contract details")
            return False
        qty = self._transactable_quantity(trade, quantity)
        if qty <= 0:
            return False
        trigger = self._protective_trigger_for(trade)
        side = self._protective_exit_side(trade)
        if trigger <= 0:
            self._mark_recovery(trade, "Protective stop trigger is invalid")
            return False

        # The read-decide-place window below is serialised so the PositionSync reconcile
        # thread and the scanner tick thread cannot both observe protective_order_id==""
        # for the same fill and each place a stop (→ duplicate SL-M → naked reversed leg
        # on trigger). RLock: safe if this thread re-enters via save_state().
        if not hasattr(self, "_protection_lock"):
            self._protection_lock = threading.RLock()
        with self._protection_lock:
            if self._apply_uts_stop_extension(trade, quantity):
                return True
            now = time.monotonic()
            if not hasattr(self, "_last_protective_sync"):
                self._last_protective_sync = {}
            if not hasattr(self, "_last_protective_status_poll"):
                self._last_protective_status_poll = {}
            if not hasattr(self, "_last_recovery_alert"):
                self._last_recovery_alert = {}
            last_sync = float(self._last_protective_sync.get(trade.id, 0.0) or 0.0)
            existing = str(getattr(trade, "protective_order_id", "") or "")
            last_status_poll = float(self._last_protective_status_poll.get(trade.id, 0.0) or 0.0)
            if existing and now - last_status_poll >= 15.0:
                getter = getattr(self.broker, "get_order_status", None)
                stop_status = self._order_status_text(getter(existing)) if callable(getter) else ""
                self._last_protective_status_poll[trade.id] = now
                if stop_status in {"rejected", "cancelled", "canceled"}:
                    trade.protective_order_id = ""
                    existing = ""
                elif stop_status in {"complete", "filled"}:
                    # The SL-M sold this trade's quantity, so the ledger has to hear
                    # about it. An exchange-side stop firing is the commonest real exit
                    # on a shared contract and it arrives through no other path here:
                    # unrecorded, the ledger keeps reporting a holding the stop already
                    # closed, and the next arm sizes protection from that stale number.
                    # Read from the stop ORDER, like every other leg.
                    self._record_leg_fill_from_order(trade, existing, "PROTECTIVE", -1)
                    self._mark_recovery(
                        trade,
                        "Protective stop reports filled while broker position is still open",
                    )
                    return False
            if existing:
                tighter = (
                    trigger > float(trade.protective_trigger or 0.0)
                    if side == "SELL"
                    else trigger < float(trade.protective_trigger or float("inf"))
                )
                qty_changed = qty != int(trade.protective_quantity or 0)
                if not tighter and not qty_changed:
                    return True
                if now - last_sync < 2.0:
                    return True
                try:
                    ok = self.broker.modify_protective_stop(
                        existing, trade.trading_symbol, trade.symbol_token, qty, side,
                        trigger, self.product_type,
                    )
                except (BrokerOrderRejected, PermissionError) as exc:
                    # modify_protective_stop now requires the account lease, so it can
                    # refuse by raising. An escape here would skip _mark_recovery and
                    # leave the trade believing its stop was tightened.
                    logger.error(
                        f"Protective-stop tighten refused for {trade.trading_symbol}: {exc}"
                    )
                    ok = False
                self._last_protective_sync[trade.id] = now
                if ok:
                    verdict = getattr(trade, "quality_verdict", {}) or {}
                    checks = verdict.get("checks", {}) if isinstance(verdict, dict) else {}
                    uncertain_uts = (
                        str(getattr(trade, "source_lane", "") or "").upper() == "UTS"
                        and isinstance(verdict, dict) and verdict.get("policy_version") == "uts_native.v1"
                        and isinstance(checks, dict) and checks.get("uts_stop_extension_uncertain")
                    )
                    if uncertain_uts:
                        getter = getattr(self.broker, "get_order_status", None)
                        try:
                            restored = getter(existing) if callable(getter) else None
                        except Exception:
                            restored = None
                        if not self._verified_uts_protective_stop(restored, trade, qty, trigger):
                            self._mark_recovery(trade, "UTS original protection restoration not verified")
                            self._mark_dirty()
                            return False
                        clean_checks = dict(checks)
                        clean_checks.pop("uts_stop_extension_uncertain", None)
                        trade.quality_verdict = {**verdict, "checks": clean_checks}
                    trade.protective_trigger = trigger
                    trade.protective_quantity = qty
                    trade.protective_updated_at = datetime.now(IST)
                    self._mark_dirty()
                    return True
                self._mark_recovery(trade, "Broker protective stop could not be tightened")
                self._mark_dirty()
                return False

            # ── Idempotent placement (audit BLOCKER-1/2 fix) ──
            finder = getattr(self.broker, "find_working_protective_stop", None)

            def _adopt(order_id: str, note: str, actual_trigger=None) -> bool:
                # A working SL-M is now in charge, so the fallback trigger must go —
                # leaving both armed means two exits and a naked reversed leg. The
                # return value is the whole point of that helper and was discarded
                # here: a REFUSED cancellation left the GTT armed alongside the SL-M,
                # and the trade was then reported protected.
                # Record the id FIRST. This order is live at the exchange whether or not
                # the GTT can be cancelled, and losing its id is how the position ends up
                # with a SECOND SL-M next cycle — the duplicate this subsystem exists to
                # prevent. _adopt is also the success path for a freshly PLACED stop.
                trade.protective_order_id = str(order_id)
                if not self._release_gtt_fallback(trade, "exchange SL-M took over"):
                    trade.protective_quantity = qty
                    trade.protective_updated_at = datetime.now(IST)
                    self._last_protective_sync[trade.id] = now
                    self._mark_recovery(
                        trade,
                        f"SL-M {order_id} is live but the GTT fallback could not be "
                        "cancelled; both are armed and would exit the position twice.",
                    )
                    self._mark_dirty()
                    return False
                # The level the EXCHANGE holds, which is not always the level we want.
                # Writing the desired trigger for a stale-trigger adoption made every
                # later tightening check compare against a value no order carried, see
                # "already tighter", and never contact the broker again — the position
                # reads as protected at a price nothing is protecting it at.
                trade.protective_trigger = (
                    float(actual_trigger) if actual_trigger is not None else trigger
                )
                trade.protective_quantity = qty
                trade.protective_updated_at = datetime.now(IST)
                trade.recovery_required_since = None
                self._last_protective_sync[trade.id] = now
                self._mark_dirty()
                logger.info(
                    f"{note} for {trade.trading_symbol}: {order_id} qty={qty} trigger={trigger:.2f}"
                )
                return True

            # Adopt any SL-M already resting at the exchange for this contract before
            # placing a new one — covers a concurrent caller that just placed, and an
            # entry whose acknowledgement was lost but which reached the exchange.
            #
            # This reads the TRI-STATE, not the collapsed id. find_working_protective_stop
            # returns "" for AMBIGUOUS and CONFLICT just as it does for NONE, and ""
            # here means "place a new stop" — so two resting SL-Ms would get a third.
            prior_state, prior, prior_trigger = self._inspect_existing_stop(trade, side, qty, trigger)
            if prior_state == "FOUND_STALE_TRIGGER" and prior:
                # It IS this position's stop, just armed at a different level. Adopt it
                # and let the ordinary tightening path move the trigger — placing a
                # second stop here would be the duplicate-SL-M bug.
                owner = self._trade_owning_protective_order(prior, exclude_id=trade.id)
                if owner is None:
                    adopted = _adopt(
                        prior,
                        "Adopted existing broker SL-M (stale trigger)",
                        actual_trigger=prior_trigger,
                    )
                    if adopted:
                        # Clear the sync clock AND leave protective_trigger at the level
                        # the exchange actually holds, so the next pass sees a genuine
                        # divergence and issues the modify. Recording the desired level
                        # here is what let a stale stop sit un-rearmed forever.
                        self._last_protective_sync[trade.id] = 0.0
                    return adopted
                prior_state, prior = "CONFLICT", ""
            if prior_state in {"AMBIGUOUS", "CONFLICT", "UNKNOWN"}:
                # Something is resting that we cannot match to this position. Placing
                # another is how a contract ends up with several stops, and each extra
                # one is a naked reversed leg waiting for the others to fill.
                self._last_protective_sync[trade.id] = now
                self._mark_recovery(
                    trade,
                    f"Existing protective stop(s) on {trade.trading_symbol} are "
                    f"{prior_state.lower()}; refusing to place another. Reconcile the "
                    "order book manually.",
                )
                self._mark_dirty()
                return False
            if prior:
                owner = self._trade_owning_protective_order(prior, exclude_id=trade.id)
                if owner is not None:
                    # Adopting it would point two trades at one SL-M; whichever exits
                    # first cancels it and silently strips the other's protection.
                    self._last_protective_sync[trade.id] = now
                    self._mark_recovery(
                        trade,
                        f"The only working stop on {trade.trading_symbol} belongs to trade "
                        f"{owner.id}; refusing to adopt or duplicate it.",
                    )
                    self._mark_dirty()
                    return False
                return _adopt(prior, "Adopted existing broker SL-M")

            try:
                order_id = self.broker.place_protective_stop(
                    trade.trading_symbol, trade.symbol_token, qty, side, trigger, self.product_type,
                    **self._tag_kwargs(
                        self.broker.place_protective_stop, trade, "PROTECTIVE"
                    ),
                )
            except BrokerOrderAmbiguous as exc:
                # The SL-M may already rest at the exchange despite the lost ack. NEVER
                # blindly re-place. Re-scan and adopt if found, else fail closed to
                # RECOVERY_REQUIRED for supervised reconciliation.
                self._last_protective_sync[trade.id] = now
                # Ask the TAG first. The symbol/side scan below can only answer "some
                # SL-M for this contract exists", which is already wrong when two trades
                # sit on one strike; the tag identifies this trade's own order exactly.
                recovered = self._recover_order_id_from_tag(trade, "PROTECTIVE")
                if not recovered and callable(finder):
                    try:
                        recovered = str(finder(trade.trading_symbol, trade.symbol_token, side) or "")
                    except Exception as scan_exc:
                        logger.warning(
                            f"Post-ambiguous protective-stop scan failed for {trade.trading_symbol}: {scan_exc}"
                        )
                if recovered:
                    return _adopt(recovered, "Recovered ambiguous broker SL-M")
                self._mark_recovery(trade, f"Protective stop acknowledgement ambiguous: {exc}")
                self._mark_dirty()
                return False
            except (BrokerOrderRejected, PermissionError) as exc:
                # The OTHER two ways place_protective_stop fails: a dead broker session
                # and an account lease held by another process. Both refuse before any
                # order is sent, so nothing is resting — but neither was caught here, so
                # both unwound out of this method AND out of reconcile_broker_positions.
                # A filled position was left booked OPEN with no stop and no recovery
                # flag (a healthy-looking row on the dashboard), and every trade after it
                # in the reconcile loop went unreconciled.
                self._last_protective_sync[trade.id] = now
                # A GTT is safe when the refusal was ours to fix — nothing is resting.
                # It is NOT safe when another process owns the account: that holder is
                # managing this position's stop, and our trigger would stack on theirs.
                lease_denied = isinstance(exc, PermissionError)
                armed = False if lease_denied else self._arm_gtt_fallback(trade, qty, side, trigger)
                self._mark_recovery(
                    trade,
                    f"Protective stop refused: {exc}"
                    + ("; GTT fallback armed on the server" if armed else ""),
                )
                self._mark_dirty()
                return False

            self._last_protective_sync[trade.id] = now
            if order_id:
                return _adopt(order_id, "Broker SL-M armed")
            # The broker REFUSED the stop outright — so unlike the ambiguous branch
            # above, nothing is resting at the exchange and a GTT cannot double up with
            # an invisible SL-M. Arm one: it lives on Zerodha's servers and fires even
            # if this process dies, which is precisely the exposure that left positions
            # unprotected on 2026-07-31. Still marked RECOVERY_REQUIRED — a GTT is a
            # safety net, not working stop management.
            armed = self._arm_gtt_fallback(trade, qty, side, trigger)
            self._mark_recovery(
                trade,
                "Confirmed broker position has no exchange-side protective stop"
                + ("; GTT fallback armed on the server" if armed else ""),
            )
            self._mark_dirty()
            return False

    def _clear_flatten_block(self, trade: Trade) -> None:
        for attr in ("_flatten_block_counts", "_flatten_block_since", "_last_flatten_block_alert"):
            blocks = getattr(self, attr, None)
            if blocks:
                blocks.pop(trade.id, None)

    def _flatten_block_deadline_passed(self, trade: Trade, blocked_since: float) -> bool:
        """Has the operator opted into flattening anyway after being blocked this long?

        Off by default (0). Withholding is the safe default — a stop still resting plus
        our market exit is a naked reversed leg — but it can carry an NRML position past
        the close on nothing but an alert, so the trade-off is exposed as a setting
        rather than decided here.
        """
        from config.settings import get_settings

        try:
            deadline = float(
                getattr(get_settings(), "recovery_flatten_block_deadline_seconds", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            # Deliberately narrow: a broad except here swallowed a NameError from a
            # missing import and turned an opted-in deadline into a silent no-op.
            return False
        if deadline <= 0 or (time.time() - float(blocked_since or 0.0)) < deadline:
            return False
        trade.reversal_watch_required = True
        logger.critical(
            f"FLATTEN DEADLINE REACHED for {trade.trading_symbol} after "
            f"{deadline:.0f}s blocked: exiting anyway per "
            "recovery_flatten_block_deadline_seconds. A protective stop MAY still be "
            "resting — if it fires it will open a REVERSED position. Watch this contract."
        )
        return True

    def _withhold_flatten(self, trade: Trade, note: str) -> bool:
        """Refuse a flatten, count it, and escalate once it stops being transient.

        Every withholding path routes through here. Withholding is correct — a double
        exit is worse than a late one — but check_session_end retries on every pass, so
        an unbounded refusal silently carries a live NRML position past the close.
        """
        if not hasattr(self, "_flatten_block_counts"):
            self._flatten_block_counts = {}
        if not hasattr(self, "_flatten_block_since"):
            self._flatten_block_since = {}
        seen = self._flatten_block_counts.get(trade.id, 0) + 1
        self._flatten_block_counts[trade.id] = seen
        blocked_since = self._flatten_block_since.setdefault(trade.id, time.time())
        if self._flatten_block_deadline_passed(trade, blocked_since):
            return False
        if seen >= 3:
            note += (
                f" ({seen} consecutive blocked attempts). ACT MANUALLY: inspect the order "
                f"book for {trade.trading_symbol} at your Zerodha terminal, cancel any "
                "resting stop, and flatten the position by hand."
            )
            # Throttled: check_session_end runs ~1 Hz, and an unthrottled CRITICAL here
            # reproduces the log flood the recovery latch was added to stop.
            last = getattr(self, "_last_flatten_block_alert", {}).get(trade.id, 0.0)
            if time.time() - last >= 60.0:
                if not hasattr(self, "_last_flatten_block_alert"):
                    self._last_flatten_block_alert = {}
                self._last_flatten_block_alert[trade.id] = time.time()
                logger.critical(
                    f"FLATTEN BLOCKED {seen}x for {trade.trading_symbol}: neither exiting "
                    "nor leaving the position is provably safe. Manual reconciliation required."
                )
        self._mark_recovery(trade, note)
        return True

    def _inspect_existing_stop(self, trade: Trade, side: str, qty: int, trigger: float = 0.0):
        """(state, order_id, resting_trigger) for a stop already on this contract.

        Prefers the tri-state inspector and passes the position's own size/product, so
        the broker's quantity/product discrimination is actually exercised — the
        collapsed find_working_protective_stop() reports "" for AMBIGUOUS and CONFLICT
        alike, which reads as "nothing there, go ahead and place one".
        """
        inspector = getattr(self.broker, "inspect_working_protective_stop", None)
        if callable(inspector):
            try:
                found = inspector(
                    trade.trading_symbol, trade.symbol_token, side,
                    quantity=int(qty or 0), product_type=self.product_type,
                    trigger_price=float(trigger or 0.0),
                ) or {}
            except Exception as exc:
                logger.warning(
                    f"Protective-stop idempotency scan failed for {trade.trading_symbol}: {exc}"
                )
                return "UNKNOWN", "", None
            level = found.get("trigger_price")
            try:
                level = float(level) if level is not None else None
            except (TypeError, ValueError):
                level = None
            return (
                str(found.get("state") or "UNKNOWN").upper(),
                str(found.get("order_id") or ""),
                level,
            )
        finder = getattr(self.broker, "find_working_protective_stop", None)
        if not callable(finder):
            return "NONE", "", None
        try:
            oid = str(finder(trade.trading_symbol, trade.symbol_token, side) or "")
        except Exception as exc:
            logger.warning(
                f"Protective-stop idempotency scan failed for {trade.trading_symbol}: {exc}"
            )
            return "UNKNOWN", "", None
        return ("FOUND", oid, None) if oid else ("NONE", "", None)

    def _entry_margin_verdict(
        self,
        symbol: str,
        qty: int,
        side: str,
        price: float,
        *,
        reserve: bool = True,
    ):
        """Can this entry be funded? None when the broker cannot answer.

        Deliberately fail-OPEN: a missing or erroring margin endpoint must never stop
        trading, only a definitive "not enough money" may. The check is one broker
        round-trip per entry attempt, which is cheap next to sending an order that is
        certain to be refused.
        """
        preflight = getattr(self.broker, "preflight_entry_margin", None)
        if not callable(preflight):
            return None
        try:
            verdict = preflight(
                symbol=symbol, qty=int(qty or 0), side=str(side or "BUY"),
                product_type=self.product_type, price=float(price or 0.0),
                reserve=bool(reserve),
            ) or {}
        except Exception as exc:
            logger.debug(f"Entry margin preflight unavailable for {symbol}: {exc}")
            return None
        if not verdict.get("ok", True):
            logger.critical(
                f"ENTRY BLOCKED (funds) {symbol} x{qty}: "
                f"{verdict.get('limitation') or verdict.get('reason') or 'insufficient margin'}"
            )
        return verdict

    def _trim_lots_to_funds_enabled(self) -> bool:
        try:
            from config.settings import get_settings

            value = getattr(get_settings(), "entry_trim_lots_to_funds_enabled", True)
        except Exception:
            return True
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off", ""}
        return bool(value)

    def _trim_entry_lots_to_funds(self, trade: Trade, symbol: str, side: str, price: float, verdict):
        """Shrink an unaffordable multi-lot entry to the most lots the funds cover.

        2026-09-25 user: "if lack of capital then use 1 lot or whatever possible with the
        available capital and avoid the rest ... without cancelling the full trade". Only a
        definitive shortfall with both numbers known is trimmed; the trimmed size is priced
        by the broker again (and reserved) before it is used. Returns the passing verdict,
        or None when not even one lot can be funded.
        """
        lots = int(getattr(trade, "lots", 1) or 1)
        if lots <= 1 or not self._trim_lots_to_funds_enabled() or not isinstance(verdict, dict):
            return None
        try:
            available = float(verdict.get("available"))
            required = float(verdict.get("required"))
            reserved = float(verdict.get("reserved") or 0.0)
        except (TypeError, ValueError):
            return None
        if required <= 0:
            return None
        per_lot = required / lots
        fit = int((available - reserved) // per_lot) if per_lot > 0 else 0
        fit = max(0, min(lots - 1, fit))
        lot_size = int(getattr(trade, "lot_size", 1) or 1)
        for size in range(fit, max(0, fit - 3), -1):
            if size < 1:
                break
            check = self._entry_margin_verdict(symbol, size * lot_size, side, price)
            if check is None or check.get("ok", True):
                with self._lock:
                    setattr(trade, "requested_lots", lots)
                    trade.lots = size
                self._mark_dirty()
                msg = (
                    f"Entry trimmed to funds: {symbol} {size} of {lots} lots "
                    f"(available {available:,.0f}, {lots} lots need {required:,.0f})"
                )
                logger.warning(msg)
                self._log_ui(msg, "warning")
                return check if check is not None else {"ok": True, "reservation_id": ""}
        return None

    def _live_position_for_trade(self, trade: Trade):
        """This trade's broker position row, or None when the book says flat.

        Returns (row, readable). `readable` is False when the positions call itself
        failed — the caller must not read that as flat, which is the same rule every
        other scan in this file follows.
        """
        getter = getattr(self.broker, "get_positions", None)
        if not callable(getter):
            return None, False
        try:
            response = getter()
        except Exception as exc:
            logger.debug(f"Positions read failed during partial reduce: {exc}")
            return None, False
        if not isinstance(response, dict) or not isinstance(response.get("data"), list):
            return None, False
        symbol = str(getattr(trade, "trading_symbol", "") or "").upper()
        key = str(getattr(trade, "symbol_token", "") or "")
        for row in response["data"]:
            row_key = str(row.get("symboltoken") or row.get("instrument_token") or "")
            if not self._matches_position_product(row):
                continue
            exact_symbol = str(row.get("tradingsymbol") or "").upper() == symbol
            legacy_symbol = (
                str(row.get("execution_symbol") or "").upper() == symbol
                and key and str(row.get("execution_token") or "") == key
            )
            if (exact_symbol and (not key or not row_key or row_key == key)) or legacy_symbol:
                return row, True
        return None, True

    def _matches_position_product(self, row: Dict) -> bool:
        mapper = getattr(getattr(self, "broker", None), "_product", None)
        product = row.get("product") or row.get("producttype")
        wanted = getattr(self, "product_type", "")
        return not (callable(mapper) and product and wanted) or mapper(product) == mapper(wanted)

    def _sole_trade_on_contract(self, trade) -> bool:
        """True when no OTHER open trade shares this contract.

        The per-tradingsymbol position is only evidence about one trade when one trade
        is all there is. With a sibling on the same strike the aggregate answers a
        different question than the caller is asking, and answering it anyway is how a
        sibling's quantity confirms an order that never filled.
        """
        symbol = str(getattr(trade, "trading_symbol", "") or "").strip()
        if not symbol:
            return True
        mine = getattr(trade, "id", None)
        for other in list(getattr(self, "open_trades", {}).values()):
            if other is trade or getattr(other, "id", None) == mine:
                continue
            if getattr(other, "is_ghost", False):
                continue
            if str(getattr(other, "trading_symbol", "") or "").strip() == symbol:
                return False
        return True

    def _confirmed_reduced_quantity(
        self, trade: Trade, expected_quantity: int, timeout: float = 2.0
    ) -> Optional[int]:
        """Prove the reduce landed, and prove it about THIS trade.

        An acknowledgement says the order was accepted, not that it filled. Booking a
        realised child off an ack is how a phantom P&L row gets written for lots that
        never left. None means unproven, never "it did not fill".

        The proof used to be the per-tradingsymbol POSITION, which describes the whole
        contract. With a sibling on the same strike that never falls to this trade's
        expected remainder, so a genuine reduce could never be confirmed and the
        protective stop was never re-armed on the remainder.

        The reduce ORDER answers the question directly, so it is asked first. The
        aggregate is still consulted, but only when this trade is alone on the contract
        -- where it is unambiguous and remains the better evidence, because it reflects
        the exchange rather than one order's view of itself.
        """
        prior = self._confirmed_broker_quantity(trade)
        order_id = str(getattr(trade, "partial_reduce_order_id", "") or "")
        sole = self._sole_trade_on_contract(trade)
        deadline = time.monotonic() + max(0.5, float(timeout or 0.5))
        while time.monotonic() < deadline:
            filled = self._order_filled_quantity(order_id) if order_id else None
            if filled:
                return max(0, prior - int(filled))
            if sole:
                position, readable = self._live_position_for_trade(trade)
                if readable:
                    if position is None:
                        return 0
                    signed = self._position_quantity(position)
                    if signed and (1 if signed > 0 else -1) != self._expected_position_sign(trade):
                        return None          # direction flipped: not our reduce
                    if abs(signed) <= expected_quantity:
                        return abs(signed)
            time.sleep(0.2)
        return None

    def _book_partial_child(
        self, trade: Trade, close_lots: int, exit_price: float, reason: str,
        exit_time: Optional[datetime] = None, closed_quantity: Optional[int] = None,
    ) -> None:
        """Realise `close_lots` as a closed child row and shrink the parent.

        `closed_quantity` is what the reduce order ACTUALLY executed. A MARKET exit in a
        thin option book can come back part-filled, and sizing the realised row off the
        requested lots then books P&L on quantity that never left: request 75, fill 30,
        and the child carries 75 units of profit against 30 units of reality while the
        parent still holds the rest. That figure feeds the daily loss cap and every R
        computed downstream. The protective stop is NOT affected -- it is armed from the
        corrected broker_quantity -- so this is an accounting error, not a naked-position
        one, which is why it survived this long.
        """
        from dataclasses import replace as _replace

        realised_qty = (
            max(0, int(closed_quantity))
            if closed_quantity is not None
            else int(close_lots) * int(trade.lot_size or 1)
        )
        child = _replace(trade)
        child.id = f"{trade.id}#P{int(close_lots)}"
        child.lots = int(close_lots)
        child.status = "CLOSED"
        child.execution_status = "CLOSED"
        child.pending_exit = False
        child.pending_execution = False
        child.exit_reason = reason
        child.exit_time = exit_time or datetime.now(IST)
        child.exit_price = float(exit_price)
        child.broker_quantity = realised_qty
        child.protective_order_id = ""
        child.gtt_trigger_id = ""
        gross = (float(exit_price) - float(trade.entry_price)) * child.broker_quantity
        if trade.inst_type != "OPT" and str(trade.direction or "").upper() == "SHORT":
            gross = -gross
        try:
            from engine.trade_accounting import estimate_trade_charges
            from config.settings import get_settings

            child.charges = estimate_trade_charges(
                trade.entry_price, exit_price, child.broker_quantity, trade.inst_type,
                get_settings(), trade.instrument_multiplier, instrument=trade.instrument,
            )
        except Exception:
            child.charges = 0.0
        child.pnl = gross - float(child.charges or 0.0)
        self.closed_trades.append(child)

        remaining = max(1, int(trade.lots or 1) - int(close_lots))
        trade.lots = remaining
        trade.broker_quantity = remaining * int(trade.lot_size or 1)

    def partial_close_trade(
        self, trade_id: str, close_lots: int, exit_price: float,
        reason: str = "PARTIAL_DERISK", exit_time: Optional[datetime] = None,
    ) -> bool:
        """Bank part of a multi-lot winner and let the remainder run.

        Refuses unless the trade is cleanly OPEN with no exit already in flight, and only
        once per trade — partial_derisk_done latches on BOOKED success, so a reduce that
        failed can be retried on the next tick while one that worked cannot be repeated.

        On a REAL position the broker work is ordered so the position can never be
        over-exited or left naked; see _request_broker_partial_reduce. Simulated and ghost
        books are booked synchronously at the decision price.
        """
        with self._lock:
            trade = self.open_trades.get(trade_id)
            if not trade or trade.status != "OPEN":
                return False
            if trade.pending_exit or str(trade.execution_status or "").upper() in {
                "EXIT_PENDING", "PARTIAL_EXIT", "RECOVERY_REQUIRED",
            }:
                return False
            if bool(getattr(trade, "partial_derisk_done", False)):
                return False
            if (
                str(getattr(trade, "partial_reduce_order_id", "") or "")
                or int(getattr(trade, "partial_reduce_requested_quantity", 0) or 0) > 0
            ):
                # A persisted in-flight reduce must be reconciled against broker positions
                # before a second one may be submitted against the same lots.
                return False
            total = max(1, int(trade.lots or 1))
            close_lots = int(close_lots or 0)
            if close_lots < 1 or close_lots >= total:
                # Closing everything is an exit, not a de-risk; that path has its own
                # protective-stop handling and must not be reached through here.
                return False
            price = float(exit_price or trade.current_price or trade.entry_price or 0.0)
            if price <= 0:
                return False
            is_real = bool(
                getattr(self, "mode", "HISTORICAL") == "REAL"
                and self.broker
                and getattr(trade, "symbol_token", "")
                and not getattr(trade, "is_ghost", False)
            )
            if is_real:
                if bool(getattr(trade, "_partial_reduce_inflight", False)):
                    return False
                setattr(trade, "_partial_reduce_inflight", True)
            else:
                self._book_partial_child(trade, close_lots, price, reason, exit_time)
                trade.partial_derisk_done = True
                self._mark_dirty()
                return True

        self._request_broker_partial_reduce(trade, close_lots, price, reason, exit_time)
        return True

    def _request_broker_partial_reduce(
        self, trade: Trade, close_lots: int, exit_price: float, reason: str,
        exit_time: Optional[datetime],
    ) -> None:
        """Reduce a REAL position by `close_lots`, in the only safe order.

          1. Cancel the full-size protective stop and PROVE it. An unproven cancel means
             the SL-M may still rest for the full size, and the partial exit on top of it
             double-exits — so an unproven cancel aborts and changes nothing.
          2. Place the partial exit. If it fails, re-arm the full-size stop before
             returning: the position is unchanged, so it must not be left unprotected.
          3. Only once POSITIONS prove the reduce, book the child and re-arm a stop for
             the remainder. If that re-arm fails the trade is marked for recovery rather
             than left running naked.
        """
        quantity = int(close_lots) * int(trade.lot_size or 1)

        def reduce_bg():
            try:
                self._run_partial_reduce(trade, close_lots, quantity, exit_price, reason, exit_time)
            except Exception as exc:
                logger.error(f"Partial reduce crashed for {trade.trading_symbol}: {exc}")
                self._mark_recovery(trade, f"Partial de-risk failed unexpectedly: {exc}")
            finally:
                setattr(trade, "_partial_reduce_inflight", False)
                trade.partial_reduce_order_id = ""
                trade.partial_reduce_requested_quantity = 0
                self._mark_dirty()

        from engine.async_queue import execution_queue

        if execution_queue.submit(f"PartialReduce_{trade.trading_symbol}", reduce_bg) is False:
            setattr(trade, "_partial_reduce_inflight", False)
            logger.warning(
                f"Partial de-risk for {trade.trading_symbol} was refused by the execution "
                "queue; the position is untouched and fully protected."
            )

    def _run_partial_reduce(
        self, trade: Trade, close_lots: int, quantity: int, exit_price: float,
        reason: str, exit_time: Optional[datetime],
    ) -> None:
        existing_stop = str(getattr(trade, "protective_order_id", "") or "")
        if existing_stop:
            confirm = getattr(self.broker, "cancel_order_confirmed", None)
            cancelled = None
            if callable(confirm):
                try:
                    cancelled = confirm(existing_stop, "STOPLOSS")
                except (BrokerOrderRejected, PermissionError) as exc:
                    logger.error(f"Partial-reduce stop cancel refused: {exc}")
                    cancelled = None
            if cancelled is not True:
                # Unproven cancel: the full-size SL-M may still be resting, and a partial
                # exit on top of it exits the position twice.
                logger.warning(
                    f"Partial de-risk aborted for {trade.trading_symbol}: the full-size "
                    "protective stop could not be confirmed cancelled."
                )
                return
            trade.protective_order_id = ""

        side = self._protective_exit_side(trade)
        try:
            order_id = self.broker.close_order(
                trade.trading_symbol, trade.symbol_token, quantity, side,
                self.product_type,
                **self._tag_kwargs(self.broker.close_order, trade, "PARTIAL_EXIT"),
            )
        except (BrokerOrderRejected, PermissionError) as exc:
            logger.error(f"Partial exit refused for {trade.trading_symbol}: {exc}")
            self.ensure_broker_protection(trade, trade.broker_quantity)
            return
        except BrokerOrderAmbiguous as exc:
            # The reduce MAY be live. Do not re-arm a full-size stop on top of it and do
            # not book anything; let reconciliation read the real quantity.
            self._mark_recovery(
                trade,
                f"Partial de-risk acknowledgement ambiguous ({exc}); broker quantity "
                "must be reconciled before this position is managed further.",
            )
            return
        if not order_id:
            self.ensure_broker_protection(trade, trade.broker_quantity)
            return

        trade.partial_reduce_order_id = str(order_id)
        trade.partial_reduce_requested_quantity = quantity
        self._mark_dirty()

        expected_remaining = max(0, int(trade.broker_quantity or 0) - quantity)
        confirmed = self._confirmed_reduced_quantity(trade, expected_remaining)
        if confirmed is None:
            self._mark_recovery(
                trade,
                "Partial de-risk could not be confirmed from broker positions; the "
                "protective stop is NOT re-armed until the real quantity is known.",
            )
            return

        # What the reduce order actually executed. Ask the ORDER, which states its own
        # executions, rather than subtracting two independently-read snapshots of
        # trade.broker_quantity: reconcile rewrites that field on a ~5s timer while the
        # confirm helper polls for up to 2s, so a pass landing in between makes the
        # difference meaningless -- and a difference of 0 would fall back to booking the
        # REQUESTED size, which is the arithmetic this replaced.
        # _record_leg_fill_from_order fetches this same number a few lines below.
        filled_now = self._order_filled_quantity(
            str(getattr(trade, "partial_reduce_order_id", "") or "")
        )
        if filled_now is None:
            # The order cannot state it; fall back to the positions-derived difference.
            prior_quantity = int(trade.broker_quantity or 0)
            actually_closed = max(0, min(int(quantity), prior_quantity - int(confirmed)))
        else:
            actually_closed = max(0, min(int(quantity), int(filled_now)))
        if actually_closed and actually_closed != int(quantity):
            logger.warning(
                f"Partial de-risk on {trade.trading_symbol} filled {actually_closed} of "
                f"{int(quantity)}; realising the filled quantity, not the requested one."
            )
        with self._lock:
            # `actually_closed or None` until 2026-08-16, which re-introduced the exact
            # fallback the comment above warns about: _book_partial_child reads None as
            # "size off the REQUESTED lots" (line 2556), so a stated zero fill booked a
            # full-size closed child carrying P&L on lots that never left the account,
            # and then set broker_quantity to 0. Reachable: _order_filled_quantity
            # returns 0 for a NO_FILL order. Nothing closed means no child at all.
            if actually_closed > 0:
                self._book_partial_child(
                    trade, close_lots, exit_price, reason, exit_time,
                    closed_quantity=actually_closed,
                )
            else:
                logger.warning(
                    f"Partial de-risk on {trade.trading_symbol} closed NOTHING — the "
                    "reduce order stated a zero fill. Booking no realised child; the "
                    "holding is taken from the positions book below."
                )
            # int(), not `or`. UNKNOWN already returned above via _mark_recovery, so a 0
            # arriving here is a FACT the positions book stated: the reduce closed the
            # whole holding, or the row is simply gone. `or` read that fact as absent and
            # kept the pre-reduce number -- and the very next line re-arms protection from
            # it, which on a bought option is an oversized SELL SL-M against an account
            # that now holds nothing: a naked written leg the moment it triggers.
            trade.broker_quantity = int(confirmed)
            trade.partial_derisk_done = True
            # From the REDUCE ORDER's own executions. `confirmed` is positions-derived
            # (_confirmed_reduced_quantity reads the per-tradingsymbol aggregate), so on
            # a shared contract broker_quantity - confirmed cannot tell this trade's
            # reduction from a sibling's -- with two 75s on one strike and this trade
            # reducing 75, the aggregate falls 150 -> 75 and the difference computes as
            # zero. That is precisely the guess the ledger exists to remove, so it must
            # not be the thing written INTO it.
            self._record_leg_fill_from_order(
                trade, getattr(trade, "partial_reduce_order_id", ""), "PARTIAL_EXIT", -1
            )
        self._mark_dirty()
        logger.success(
            f"Partial de-risk booked for {trade.trading_symbol}: {close_lots} lot(s) "
            f"realised, {trade.lots} running."
        )
        if not self.ensure_broker_protection(trade, trade.broker_quantity):
            self._mark_recovery(
                trade, "Remainder of a partial de-risk could not be re-protected"
            )

    # ── scale-in ─────────────────────────────────────────────────────────────
    def scale_in_trade(
        self, trade_id: str, add_lots: Optional[int] = None, price: float = 0.0,
        reason: str = "CASCADE_ADD", *, native_alignment: bool = False,
    ) -> Tuple[bool, str]:
        """Add lots to a position that is already working.

        The mirror of partial_close_trade, and deliberately ordered the other way round.
        The reduce must cancel the full-size stop FIRST, because exiting on top of a live
        SL-M double-exits. An add has the opposite failure direction: the existing stop
        already covers what is held, so the order is ADD -> confirm from broker positions
        -> re-arm the stop for the new total. If the add never fills, nothing changed and
        the position stayed protected the whole time; if it fills and the re-arm fails,
        the trade is marked for recovery rather than left running under-protected.

        Returns (accepted, reason) so a caller can log WHY an add was refused instead of
        guessing from a bare False.
        """
        from config.settings import get_settings

        settings = get_settings()
        if not native_alignment and not _setting_enabled(getattr(settings, "scale_in_enabled", False), False):
            return False, "scale-in disabled"

        with self._lock:
            trade = self.open_trades.get(trade_id)
            if not trade or trade.status != "OPEN":
                return False, "trade is not cleanly open"
            if native_alignment:
                if not self._is_utema_trade(trade):
                    return False, "native alignment requires persisted UTEMA ownership"
                if add_lots != 1 or int(getattr(trade, "scale_in_count", 0) or 0) != 0:
                    return False, "UTEMA alignment permits exactly one additional lot once"
                if getattr(trade, "pending_execution", False):
                    return False, "UTEMA entry fill is not confirmed"
                if self._live_orders_permitted() and not getattr(trade, "is_ghost", False):
                    if not bool(getattr(getattr(self, "scanner", None), "auto_mode", False)):
                        return False, "UTEMA alignment is virtual while Auto is off"
                    if getattr(self, "broker", None) is None:
                        return False, "UTEMA broker is unavailable"
                    owned = self._ledger_owned_quantity(str(trade.id))
                    if owned is None or owned <= 0 or self._confirmed_broker_quantity(trade) <= 0:
                        return False, "UTEMA alignment requires its own confirmed entry fill"
            if trade.pending_exit or str(trade.execution_status or "").upper() in {
                "EXIT_PENDING", "PARTIAL_EXIT", "RECOVERY_REQUIRED",
            }:
                return False, f"exit or recovery in flight ({trade.execution_status})"
            if bool(getattr(trade, "partial_derisk_done", False)):
                # De-risking and scaling in are opposite decisions about the same trade.
                return False, "already de-risked; not adding back"
            if (
                str(getattr(trade, "scale_in_order_id", "") or "")
                or int(getattr(trade, "scale_in_requested_quantity", 0) or 0) > 0
            ):
                return False, "an add is already in flight"
            if bool(getattr(trade, "_scale_in_inflight", False)):
                return False, "an add is already in flight"

            if add_lots is None:
                # scale_in_lots_per_add is the size of one add. A caller that knows
                # better may override it; most just say "add".
                add_lots = int(getattr(settings, "scale_in_lots_per_add", 1) or 1)
            add_lots = max(0, int(add_lots or 0))
            if add_lots < 1:
                return False, "no lots to add"
            max_adds = max(0, int(getattr(settings, "scale_in_max_adds_per_trade", 2) or 0))
            if not native_alignment and int(getattr(trade, "scale_in_count", 0) or 0) >= max_adds:
                return False, f"add cap {max_adds} reached"
            max_total = max(1, int(getattr(settings, "scale_in_max_total_lots", 6) or 6))
            if not native_alignment and int(trade.lots or 0) + add_lots > max_total:
                return False, f"would exceed {max_total} total lots"

            price = float(price or trade.current_price or trade.entry_price or 0.0)
            if price <= 0:
                return False, "no usable price"

            if self._live_contract_risk_required():
                problem = self._upstox_contract_entry_error(
                    trade.trading_symbol, trade.symbol_token, trade.lot_size, trade.inst_type)
                if problem:
                    return False, problem

            ok, why = (True, "") if native_alignment else self._scale_in_is_earned(trade, settings)
            if not ok:
                return False, why

            # An add-on is new money at risk even though it is not a new position, so it
            # faces the rupee budget. It does NOT face the per-index/bucket count caps —
            # those bound how many ideas are live, and this is the same idea.
            admitted, risk_why = (True, "") if native_alignment and not self._live_contract_risk_required() else self._entry_risk_admission(
                inst_type=trade.inst_type,
                direction=trade.direction,
                price=price,
                trailing_stop=float(trade.current_stop or trade.trailing_stop or 0.0),
                initial_stop=float(getattr(trade, "initial_stop", 0.0) or trade.trailing_stop or 0.0),
                lots=add_lots,
                lot_size=int(trade.lot_size or 1),
                instrument_multiplier=float(getattr(trade, "instrument_multiplier", 1.0) or 1.0),
                is_recovery=False,
                instrument=trade.instrument,
            )
            if not admitted:
                return False, f"entry risk budget: {risk_why}"

            is_real = bool(
                getattr(self, "mode", "HISTORICAL") == "REAL"
                and self.broker
                and getattr(trade, "symbol_token", "")
                and not getattr(trade, "is_ghost", False)
            )
            if not is_real:
                # A paper/replay add is still fresh exposure. Previously it was booked
                # directly here, bypassing the one policy seam exercised by live
                # scale-ins. Validate the exact simulated fill and combined-position
                # geometry before mutating the parent ledger.
                if getattr(self, "fresh_entry_policy_guard", None) is not None:
                    existing_qty = max(1, int(self._transactable_quantity(trade) or 0))
                    add_qty = int(add_lots) * int(trade.lot_size or 1)
                    projected_entry = (
                        float(trade.entry_price or 0.0) * existing_qty
                        + float(price) * add_qty
                    ) / max(1, existing_qty + add_qty)
                    setattr(
                        trade,
                        "_entry_policy_execution_levels",
                        (
                            projected_entry,
                            float(trade.current_stop or trade.trailing_stop or 0.0),
                            float(trade.target or 0.0),
                        ),
                    )
                    setattr(trade, "_entry_policy_phase", "scale_in_actual_fill")
                    setattr(trade, "_entry_policy_prior_quantity", existing_qty)
                    setattr(trade, "_entry_policy_add_quantity", add_qty)
                    try:
                        allowed, policy_reason = self._entry_submission_guard_verdict(
                            trade,
                            getattr(trade, "_entry_submission_guard", None),
                            force_auto_exposure=True,
                        )
                    finally:
                        for attr_name in (
                            "_entry_policy_execution_levels",
                            "_entry_policy_phase",
                            "_entry_policy_prior_quantity",
                            "_entry_policy_add_quantity",
                        ):
                            try:
                                delattr(trade, attr_name)
                            except AttributeError:
                                pass
                    if not allowed:
                        return False, (
                            "final entry policy: "
                            f"{policy_reason or 'simulated scale-in rejected'}"
                        )
                self._book_scale_in(trade, add_lots, price, reason)
                self._mark_dirty()
                return True, "booked"
            setattr(trade, "_scale_in_inflight", True)

        submitted = self._request_broker_scale_in(trade, add_lots, price, reason)
        if native_alignment and submitted is False:
            return False, "UTEMA alignment execution queue is unavailable"
        return True, "submitted"

    def _scale_in_is_earned(self, trade: Trade, settings) -> Tuple[bool, str]:
        """Only add to a trade that is already working, and still with the trend.

        Adding to a loser is averaging down, which turns one bad idea into a bigger one.
        Profit is measured in R against the ORIGINAL risk, not the current unrealised
        rupees, so the bar does not get easier as the position grows.
        """
        risk = self._frozen_initial_risk_rs(trade)
        min_r = max(0.0, float(getattr(settings, "scale_in_min_profit_r", 0.75) or 0.0))
        if min_r > 0:
            if risk <= 0:
                return False, "original risk unknown, cannot measure R"
            entry = float(trade.entry_price or 0.0)
            mark = float(trade.current_price or 0.0)
            if entry <= 0 or mark <= 0:
                return False, "no mark to measure profit against"
            # OPTIONS ARE ALWAYS BOUGHT. A bearish view is a bought PE, stored with
            # direction="SHORT", and it profits when its PREMIUM RISES exactly as a CE
            # does. Reading `direction` alone inverted the sign for every PE: premium
            # 120 -> 100 is a Rs.1,300 loss on 65 qty, about -1.1R, but computed as
            # (120-100) = +20 it reported +1.11R and cleared the 0.75R bar.
            #
            # So this function did the one thing its own docstring forbids -- it added
            # size to a loser. The `mult` line just below already pins 1.0 for OPT, so
            # the option case was known here; only the sign was missed.
            signed = (
                (mark - entry)
                if (str(trade.inst_type or "").upper() == "OPT"
                    or str(trade.direction).upper() == "LONG")
                else (entry - mark)
            )
            qty = self._transactable_quantity(trade)
            mult = 1.0 if str(trade.inst_type or "").upper() == "OPT" else float(
                getattr(trade, "instrument_multiplier", 1.0) or 1.0
            )
            open_r = (signed * max(1, qty) * mult) / risk
            if open_r < min_r:
                return False, f"open profit {open_r:.2f}R below {min_r:.2f}R"

        if _setting_enabled(getattr(settings, "scale_in_requires_trend", True), True):
            regime = str(getattr(trade, "market_regime", "") or "").upper()
            if regime:
                bullish = "UP" in regime or "BULL" in regime
                bearish = "DOWN" in regime or "BEAR" in regime
                wants_up = str(trade.direction).upper() == "LONG"
                if (wants_up and bearish) or (not wants_up and bullish):
                    return False, f"regime {regime} opposes the trade"
                if not (bullish or bearish):
                    return False, f"regime {regime} is not trending"
        return True, ""

    def _frozen_initial_risk_rs(self, trade: Trade) -> float:
        """Rupee risk committed at ENTRY, held constant across every add-on.

        Computed once, on the first add, from the pre-scale entry and stop. After that the
        weighted-average entry has moved and the same arithmetic would give a different —
        and meaningless — answer.
        """
        stored = float(getattr(trade, "frozen_initial_risk_rs", 0.0) or 0.0)
        if stored > 0:
            return stored
        entry = float(trade.entry_price or 0.0)
        stop = float(getattr(trade, "initial_stop", 0.0) or trade.trailing_stop or 0.0)
        if entry <= 0 or stop <= 0:
            return 0.0
        qty = self._transactable_quantity(trade)
        mult = 1.0 if str(trade.inst_type or "").upper() == "OPT" else float(
            getattr(trade, "instrument_multiplier", 1.0) or 1.0
        )
        return abs(entry - stop) * max(1, qty) * max(0.0, mult)

    def _book_scale_in(
        self, trade: Trade, add_lots: int, fill_price: float, reason: str,
        filled_quantity: Optional[int] = None,
    ) -> None:
        """Fold a confirmed add into the parent. Caller holds the lock.

        `filled_quantity` is what the add order ACTUALLY executed. The weighted-average
        entry price is the whole point of this method, and weighting it by the REQUESTED
        size on a part-filled add moves it to a price the position never paid -- which
        then defines every R multiple, the frozen initial risk, and the open P&L on the
        screen. Requesting 75 and filling 30 booked an average of 220 where the truth was
        210.
        """
        # Freeze R BEFORE the entry price moves — afterwards the original is unrecoverable.
        if float(getattr(trade, "frozen_initial_risk_rs", 0.0) or 0.0) <= 0:
            trade.frozen_initial_risk_rs = self._frozen_initial_risk_rs(trade)

        old_lots = max(0, int(trade.lots or 0))
        # Weight the EXISTING side by what is actually held, not by lots x lot_size.
        # trade.lots is deliberately left at the requested count on a partial fill (order
        # sizing never reads it), so after one part-filled add the lots-derived figure
        # overstates the position -- and a SECOND add would then compound the first add's
        # error into the average entry. broker_quantity is the reconciled truth; the `or`
        # tail keeps the unreconciled and non-REAL paths exactly as they were.
        old_qty = int(getattr(trade, "broker_quantity", 0) or 0) or old_lots * int(trade.lot_size or 1)
        add_qty = (
            max(0, int(filled_quantity))
            if filled_quantity is not None
            else int(add_lots) * int(trade.lot_size or 1)
        )
        total_qty = old_qty + add_qty
        if total_qty > 0:
            trade.entry_price = (
                float(trade.entry_price or 0.0) * old_qty + float(fill_price) * add_qty
            ) / total_qty
        trade.lots = old_lots + int(add_lots)
        trade.scale_in_lots = int(getattr(trade, "scale_in_lots", 0) or 0) + int(add_lots)
        trade.scale_in_count = int(getattr(trade, "scale_in_count", 0) or 0) + 1
        trade.broker_quantity = total_qty
        if self._is_utema_trade(trade):
            self._cap_option_entry_stop(trade)
            if str(getattr(trade, "inst_type", "")).upper() == "OPT":
                trade.premium_entry = float(trade.entry_price)
            state = getattr(trade, "utema_state", None)
            if not isinstance(state, dict):
                state = ((getattr(trade, "quality_verdict", {}) or {}).get("checks", {}) or {}).get("utema_state")
            if isinstance(state, dict):
                state["alignment_added"] = True
                trade.utema_state = state
                recorded = dict(getattr(trade, "quality_verdict", {}) or {})
                checks = dict(recorded.get("checks", {}) or {})
                checks["utema_state"] = dict(state)
                recorded["checks"] = checks
                trade.quality_verdict = recorded
        # add_qty is what was INTENDED; the ledger records what the add order executed.
        self._record_leg_fill_from_order(
            trade, getattr(trade, "scale_in_order_id", ""), "ADD", 1
        )
        logger.success(
            f"Scale-in booked for {trade.trading_symbol}: +{add_lots} lot(s) @ "
            f"{fill_price:.2f} ({reason}); {trade.lots} lots now, average entry "
            f"{trade.entry_price:.2f}, 1R still Rs.{trade.frozen_initial_risk_rs:.0f}"
        )

    def _request_broker_scale_in(
        self, trade: Trade, add_lots: int, price: float, reason: str,
    ) -> bool:
        quantity = int(add_lots) * int(trade.lot_size or 1)

        def add_bg():
            try:
                self._run_scale_in(trade, add_lots, quantity, price, reason)
            except Exception as exc:
                logger.error(f"Scale-in crashed for {trade.trading_symbol}: {exc}")
                self._mark_recovery(trade, f"Scale-in failed unexpectedly: {exc}")
            finally:
                setattr(trade, "_scale_in_inflight", False)
                trade.scale_in_order_id = ""
                trade.scale_in_requested_quantity = 0
                self._mark_dirty()

        from engine.async_queue import execution_queue

        if execution_queue.submit(f"ScaleIn_{trade.trading_symbol}", add_bg) is False:
            setattr(trade, "_scale_in_inflight", False)
            logger.warning(
                f"Scale-in for {trade.trading_symbol} was refused by the execution queue; "
                "the position is untouched and fully protected."
            )
            return False
        return True

    def _option_contract_liquidity_admission(
        self,
        *,
        inst_type: str,
        lot_size: int,
        touch_quote: Optional[Dict[str, Any]],
        trading_symbol: str = "",
        is_recovery: bool = False,
    ) -> Tuple[bool, str]:
        """Apply absolute OI/volume/depth floors when those facts exist.

        Missing evidence is not silently converted to zero.  Initial REAL AUTO
        entries already require an executable touch quote; manual/recovery flows
        retain their existing semantics.  A broker- or chain-measured zero,
        however, is a real observation and must face the configured floor.
        """
        if is_recovery or str(inst_type or "").upper() != "OPT":
            return True, ""
        from config.settings import get_settings

        settings = get_settings()
        if not _setting_enabled(
            getattr(settings, "option_contract_liquidity_gate_enabled", True), True
        ):
            return True, ""
        quote = touch_quote if isinstance(touch_quote, dict) else {}
        size = max(1, int(lot_size or 1))

        try:
            min_oi = max(
                0.0, float(getattr(settings, "option_contract_min_oi_lots", 5.0) or 0.0)
            ) * size
            min_volume = max(
                0.0,
                float(getattr(settings, "option_contract_min_volume_lots", 1.0) or 0.0),
            ) * size
            min_depth = max(
                0.0,
                float(
                    getattr(settings, "option_contract_min_touch_depth_lots", 1.0)
                    or 0.0
                ),
            ) * size
        except (TypeError, ValueError):
            return False, "invalid option-contract liquidity configuration"

        def _available(flag: str, value_key: str) -> bool:
            # New provider snapshots carry explicit flags.  Hand-built/manual
            # integrations that predate them may still communicate evidence by
            # supplying the value key itself.
            if flag in quote:
                return bool(quote.get(flag))
            return value_key in quote and quote.get(value_key) is not None

        def _value(key: str) -> Optional[float]:
            try:
                value = float(quote.get(key))
            except (TypeError, ValueError):
                return None
            if value != value or value < 0:
                return None
            return value

        # Prefer the current broker quote; fall back to the selected strike's
        # cached-chain evidence when the quote endpoint does not publish OI/volume.
        evidence = (
            ("open interest", "oi", "oi_available", min_oi),
            ("traded volume", "volume", "volume_available", min_volume),
        )
        for label, direct_key, direct_flag, threshold in evidence:
            key, available = direct_key, _available(direct_flag, direct_key)
            if not available:
                key = "strike_oi" if direct_key == "oi" else "strike_volume"
                flag = f"{key}_available"
                available = _available(flag, key)
            if not available or threshold <= 0:
                continue
            value = _value(key)
            if value is None:
                return False, f"invalid measured option {label} on {trading_symbol or 'contract'}"
            if value + 1e-9 < threshold:
                return False, (
                    f"option {label} {value:,.0f} below {threshold:,.0f} "
                    f"({threshold / size:.1f} lots) on {trading_symbol or 'contract'}"
                )

        if min_depth > 0:
            for side in ("bid", "ask"):
                key = f"{side}_qty"
                if not _available(f"{key}_available", key):
                    continue
                value = _value(key)
                if value is None:
                    return False, f"invalid measured option {side} depth on {trading_symbol or 'contract'}"
                if value + 1e-9 < min_depth:
                    return False, (
                        f"option {side} depth {value:,.0f} below {min_depth:,.0f} "
                        f"({min_depth / size:.1f} lots) on {trading_symbol or 'contract'}"
                    )
        return True, ""

    def _scale_in_option_touch_quote(
        self, trade: Trade
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Fresh executable option book for an add, or a fail-closed reason."""
        from config.settings import get_settings

        place_order = getattr(getattr(self, "broker", None), "place_order", None)
        if not callable(place_order) or not self._accepts_kwarg(place_order, "touch_quote"):
            return None, "broker cannot accept a bounded touch quote for option adds"
        scanner = getattr(self, "scanner", None)
        data = getattr(scanner, "data", None) or getattr(self, "data", None)
        getter = getattr(data, "get_contract_touch_quote", None)
        if not callable(getter):
            return None, "no executable option-book provider is connected"

        settings = get_settings()
        try:
            max_age = max(
                0.05,
                float(
                    getattr(settings, "execution_touch_quote_max_age_seconds", 1.5)
                    or 1.5
                ),
            )
        except (TypeError, ValueError):
            max_age = 1.5
        symbol = str(getattr(trade, "trading_symbol", "") or "")
        token = str(getattr(trade, "symbol_token", "") or "")
        if not symbol or not token:
            return None, "option contract identity is incomplete"
        exchange = "BFO" if "SENSEX" in symbol.upper() else "NFO"
        try:
            quote = getter(exchange, symbol, token, max_age_seconds=max_age)
        except Exception as exc:
            return None, f"option-book lookup failed: {exc}"
        if not isinstance(quote, dict) or not bool(quote.get("usable", False)):
            detail = str((quote or {}).get("reason") or "unavailable")
            return None, f"executable option book is {detail}"
        try:
            bid = float(quote.get("bid") or 0.0)
            ask = float(quote.get("ask") or 0.0)
            age = float(quote["age_seconds"])
        except (TypeError, ValueError, KeyError):
            return None, "executable option book has no valid price/freshness evidence"
        if bid <= 0 or ask <= 0 or ask < bid:
            return None, "executable option book is empty or crossed"
        if age != age or age < 0 or age > max_age:
            return None, f"executable option book is stale ({age:.2f}s)"
        if self._is_utema_trade(trade):
            if not math.isfinite(bid) or not math.isfinite(ask):
                return None, "executable option book has non-finite prices"
            return dict(quote), ""

        spread = ask - bid
        mid = (ask + bid) / 2.0
        spread_pct = spread / mid * 100.0 if mid > 0 else float("inf")
        try:
            max_pct = max(
                0.0,
                float(getattr(settings, "option_touch_quote_max_spread_pct", 1.0) or 0.0),
            )
            max_ticks = max(
                0,
                int(getattr(settings, "option_touch_quote_max_spread_ticks", 40) or 0),
            )
        except (TypeError, ValueError):
            max_pct, max_ticks = 1.0, 40
        if (max_pct > 0 and spread_pct > max_pct) or (
            max_ticks > 0 and spread > max_ticks * 0.05 + 1e-9
        ):
            return None, f"executable option spread is too wide ({spread_pct:.2f}%)"
        liquid, reason = self._option_contract_liquidity_admission(
            inst_type="OPT",
            lot_size=int(getattr(trade, "lot_size", 1) or 1),
            touch_quote=quote,
            trading_symbol=symbol,
        )
        if not liquid:
            return None, reason
        return dict(quote), ""

    def _run_scale_in(
        self, trade: Trade, add_lots: int, quantity: int, price: float, reason: str,
    ) -> None:
        if self._is_utema_trade(trade) and (
            not self._live_orders_permitted()
            or not bool(getattr(getattr(self, "scanner", None), "auto_mode", False))
        ):
            logger.warning(f"UTEMA alignment cancelled for {trade.id}: live Auto authority changed")
            return
        entry_side = "BUY" if trade.inst_type == "OPT" else (
            "BUY" if str(trade.direction).upper() in {"LONG", "BUY"} else "SELL"
        )
        is_option = str(getattr(trade, "inst_type", "") or "").upper() == "OPT"
        order_kwargs = self._tag_kwargs(self.broker.place_order, trade, "ADD")
        order_price = float(price or 0.0)
        policy_guard_installed = getattr(self, "fresh_entry_policy_guard", None) is not None
        if is_option:
            touch_quote, touch_reason = self._scale_in_option_touch_quote(trade)
            if not touch_quote:
                logger.warning(
                    f"Option scale-in refused for {trade.trading_symbol}: {touch_reason}; "
                    "the existing position and protective stop are unchanged."
                )
                return
            order_price = float(touch_quote.get("ask") or 0.0)
            order_kwargs["touch_quote"] = touch_quote
            # A non-zero ask plus maximum confidence is a second guard against an
            # integration disabling touch routing: Kite still chooses LIMIT, never
            # the old unbounded MARKET add.
            if self._accepts_kwarg(self.broker.place_order, "trade_id"):
                order_kwargs["trade_id"] = str(getattr(trade, "id", "") or "")
            if self._accepts_kwarg(self.broker.place_order, "confidence"):
                order_kwargs["confidence"] = 1.0
        elif policy_guard_installed:
            # Futures adds are submitted as MARKET orders below, but the final policy
            # still needs a measured marketable touch rather than the price captured
            # when the scanner first offered the add.  Refuse when that evidence cannot
            # be refreshed; the existing parent exposure remains untouched.
            scanner = getattr(self, "scanner", None)
            quote_reader = getattr(scanner, "_candidate_execution_touch_quote", None)
            if not callable(quote_reader):
                logger.warning(
                    f"Futures scale-in refused for {trade.trading_symbol}: "
                    "fresh executable touch reader unavailable."
                )
                return
            try:
                touch_quote = quote_reader(trade) or {}
                bid = float(touch_quote.get("bid") or 0.0)
                ask = float(touch_quote.get("ask") or 0.0)
            except Exception as exc:
                logger.warning(
                    f"Futures scale-in refused for {trade.trading_symbol}: "
                    f"fresh executable touch failed ({exc})."
                )
                return
            if not bool(touch_quote.get("usable")) or bid <= 0 or ask < bid:
                logger.warning(
                    f"Futures scale-in refused for {trade.trading_symbol}: "
                    "fresh executable bid/ask unavailable."
                )
                return
            order_price = (
                ask
                if str(getattr(trade, "direction", "") or "").upper() in {"LONG", "BUY"}
                else bid
            )
            if self._accepts_kwarg(self.broker.place_order, "touch_quote"):
                order_kwargs["touch_quote"] = touch_quote
        # Every new entry funnels through _entry_margin_verdict, which both checks the
        # exact Kite order margin against free cash AND reserves it so a concurrent entry
        # cannot spend the same rupees. The add path did neither. The missing reservation
        # was the sharper half: with no record that this money is committed, an ordinary
        # entry arriving seconds later reads `available - outstanding` as if the add had
        # never happened, passes its own preflight, and the account over-commits — exactly
        # the race the reservation ledger exists to prevent. Nothing else gated it either:
        # _entry_risk_admission is entry_risk_budget_enabled-gated and that ships False,
        # so an enabled scale-in had no rupee gate at all.
        #
        # Same fail-open contract as an entry: a missing or erroring margin endpoint must
        # never stop trading, only a definitive "not enough money" may. Refusing here
        # touches nothing on the parent position — it is still running, still protected by
        # the stop that covers its current quantity.
        if self._live_contract_risk_required():
            with self._lock:
                admitted, risk_reason = self._entry_risk_admission(
                    inst_type=trade.inst_type, direction=trade.direction, price=order_price,
                    trailing_stop=trade.current_stop or trade.trailing_stop,
                    initial_stop=trade.initial_stop, lots=add_lots, lot_size=trade.lot_size,
                    instrument_multiplier=trade.instrument_multiplier, is_recovery=False,
                    instrument=trade.instrument, scale_in_trade_id=trade.id,
                )
            if not admitted:
                logger.warning(f"Scale-in refused at broker submission: {risk_reason}")
                return
        preflight = self._entry_margin_verdict(
            trade.trading_symbol, quantity, entry_side, order_price
        )
        if preflight is not None and not preflight.get("ok", True):
            logger.error(
                f"Scale-in refused for {trade.trading_symbol} x{quantity}: "
                f"{preflight.get('limitation') or preflight.get('reason') or 'insufficient margin'}"
            )
            return
        reservations = [str((preflight or {}).get("reservation_id") or "")]
        reservations = [r for r in reservations if r]

        # This worker may have waited behind another broker task after the Scanner
        # decided to add exposure. Re-ask at the same last-moment boundary used by a
        # first entry. A refusal leaves the parent position and its existing protective
        # stop untouched, and the just-created margin reservation is returned.
        if policy_guard_installed:
            existing_qty = max(1, int(self._transactable_quantity(trade) or 0))
            prior_entry = float(getattr(trade, "entry_price", order_price) or order_price)
            projected_entry = (
                prior_entry * existing_qty + order_price * int(quantity)
            ) / max(1, existing_qty + int(quantity))
            setattr(
                trade,
                "_entry_policy_execution_levels",
                (
                    projected_entry,
                    float(
                        getattr(trade, "current_stop", 0.0)
                        or getattr(trade, "trailing_stop", 0.0)
                        or 0.0
                    ),
                    float(getattr(trade, "target", 0.0) or 0.0),
                ),
            )
            setattr(trade, "_entry_policy_phase", "scale_in_pre_submit")
            setattr(trade, "_entry_policy_prior_quantity", existing_qty)
            setattr(trade, "_entry_policy_add_quantity", int(quantity))
        try:
            allowed, guard_reason = self._entry_submission_guard_verdict(
                trade,
                getattr(trade, "_entry_submission_guard", None),
                # A cascade add is automated new exposure even when the parent trade
                # was originally opened manually, so the parent's exec_type must not
                # bypass the final Auto/resource authority.
                force_auto_exposure=True,
            )
        finally:
            if policy_guard_installed:
                try:
                    delattr(trade, "_entry_policy_execution_levels")
                    delattr(trade, "_entry_policy_phase")
                    delattr(trade, "_entry_policy_prior_quantity")
                    delattr(trade, "_entry_policy_add_quantity")
                except AttributeError:
                    pass
        if self._is_utema_trade(trade) and (
            not self._live_orders_permitted()
            or not bool(getattr(getattr(self, "scanner", None), "auto_mode", False))
        ):
            allowed, guard_reason = False, "UTEMA live Auto authority changed during preflight"
        if not allowed:
            self._release_margin_reservations(reservations)
            logger.warning(
                f"Scale-in submission guard rejected {trade.trading_symbol}: "
                f"{guard_reason}. Existing position is unchanged and remains protected."
            )
            return

        # The existing stop is NOT cancelled first. It covers the quantity currently held,
        # so leaving it armed means a failed add changes nothing and a filled add is
        # merely under-protected on the new lots until the re-arm below — never naked, and
        # never over-exited.
        try:
            order_id = self.broker.place_order(
                trade.trading_symbol, trade.symbol_token, quantity, entry_side,
                self.product_type, price=order_price if is_option else 0.0,
                **order_kwargs,
            )
        except (BrokerOrderRejected, PermissionError) as exc:
            # Definite refusal: this order consumed nothing, so give the margin back.
            self._release_margin_reservations(reservations)
            logger.error(f"Scale-in refused for {trade.trading_symbol}: {exc}")
            return
        except BrokerOrderAmbiguous as exc:
            # The add MAY be live. Book nothing and re-arm nothing: a stop sized for the
            # old quantity on a position that may now be larger is the safe direction, and
            # reconciliation will read the real quantity.
            self._mark_recovery(
                trade,
                f"Scale-in acknowledgement ambiguous ({exc}); broker quantity must be "
                "reconciled before this position is managed further.",
            )
            return
        if not order_id:
            # No order id and no exception: nothing was accepted, so the margin is free.
            # (An AMBIGUOUS acknowledgement above deliberately keeps its reservation —
            # that order may be live, and letting the next entry spend money it might
            # already be consuming is how an account over-commits on orders it cannot see.)
            self._release_margin_reservations(reservations)
            logger.warning(f"Scale-in for {trade.trading_symbol} returned no order id.")
            return

        trade.scale_in_order_id = str(order_id)
        trade.scale_in_requested_quantity = quantity
        self._mark_dirty()

        expected_total = int(trade.broker_quantity or trade.quantity or 0) + quantity
        confirmed = self._confirmed_scaled_quantity(trade, expected_total)
        if confirmed is None:
            self._mark_recovery(
                trade,
                "Scale-in could not be confirmed from broker positions; the protective "
                "stop is NOT re-sized until the real quantity is known.",
            )
            return

        # What the add order actually executed, so the weighted average entry moves by
        # the quantity that truly arrived. Read through _confirmed_broker_quantity, the
        # SAME expression _confirmed_scaled_quantity used for its own `prior`: taking
        # `broker_quantity or trade.quantity` here instead made the two ends disagree on
        # an unreconciled trade, the subtraction came out 0, and the booking silently
        # fell back to the requested size -- the very arithmetic this is replacing.
        filled_now = self._order_filled_quantity(
            str(getattr(trade, "scale_in_order_id", "") or "")
        )
        if filled_now is None:
            prior_quantity = self._confirmed_broker_quantity(trade)
            actually_added = max(0, min(int(quantity), int(confirmed) - prior_quantity))
        else:
            actually_added = max(0, min(int(quantity), int(filled_now)))
        if actually_added and actually_added != int(quantity):
            logger.warning(
                f"Scale-in on {trade.trading_symbol} filled {actually_added} of "
                f"{int(quantity)}; weighting the average entry on the filled quantity."
            )
        actual_add_price = float(order_price or 0.0)
        actual_add_price_verified = False
        filler = getattr(self.broker, "get_fill", None)
        if callable(filler) and getattr(trade, "scale_in_order_id", ""):
            try:
                detail = filler(str(trade.scale_in_order_id)) or {}
                if str(detail.get("state") or "").upper() == "FILLED":
                    measured = float(detail.get("average_price") or 0.0)
                    if measured > 0:
                        actual_add_price = measured
                        actual_add_price_verified = True
            except Exception as exc:
                logger.debug(f"Scale-in own-order fill price unavailable: {exc}")
        with self._lock:
            # Same rule as the reduce, and for the same reason: `or None` would make
            # _book_scale_in size off the REQUESTED lots. An add cannot confirm to 0
            # today, so this is defensive rather than a live bug — but the two paths
            # must not disagree about what a 0 means, which is what made the reduce
            # dangerous in the first place.
            if actually_added > 0:
                self._book_scale_in(
                    trade, add_lots, actual_add_price, reason,
                    filled_quantity=actually_added,
                )
            else:
                logger.warning(
                    f"Scale-in on {trade.trading_symbol} added NOTHING — the add order "
                    "stated a zero fill. Booking no add."
                )
            # Same rule as the reduce: UNKNOWN returned above, so this is a stated fact.
            # An add cannot confirm to 0 today (prior + filled, both non-negative, with
            # filled > 0 on the order branch), but the two paths must not disagree about
            # what a 0 MEANS -- that divergence is what made the reduce dangerous.
            trade.broker_quantity = int(confirmed)
        self._mark_dirty()

        setattr(trade, "_entry_policy_prior_quantity", max(0, int(confirmed) - actually_added))
        setattr(trade, "_entry_policy_add_quantity", actually_added)
        setattr(
            trade,
            "_entry_policy_actual_fill_verified",
            actual_add_price_verified,
        )
        try:
            policy_ok, policy_reason = self._actual_fill_entry_policy_verdict(
                trade, phase="scale_in_actual_fill"
            )
        finally:
            try:
                delattr(trade, "_entry_policy_prior_quantity")
                delattr(trade, "_entry_policy_add_quantity")
                delattr(trade, "_entry_policy_actual_fill_verified")
            except AttributeError:
                pass
        if not policy_ok:
            reason_text = f"Scale-in fill violates final entry policy: {policy_reason}"
            logger.error(reason_text)
            if not self.emergency_flatten_trade(trade, reason_text):
                self._mark_recovery(trade, reason_text)
            return

        if not self.ensure_broker_protection(trade, trade.broker_quantity):
            self._mark_recovery(
                trade, "Scale-in filled but the enlarged position could not be re-protected"
            )

    def _confirmed_scaled_quantity(
        self, trade: Trade, expected_total: int, timeout: float = 4.0
    ) -> Optional[int]:
        """Wait for POSITIONS — not an order ack — to prove the add landed.

        The exact mirror of _confirmed_reduced_quantity, including its rule that None
        means UNKNOWN and never "it did not fill". Booking an add off an acknowledgement
        would move the average entry price for lots that may never have arrived, and every
        R and P&L figure downstream is computed from that price.
        """
        prior = self._confirmed_broker_quantity(trade)
        order_id = str(getattr(trade, "scale_in_order_id", "") or "")
        sole = self._sole_trade_on_contract(trade)
        deadline = time.monotonic() + max(0.5, float(timeout or 0.5))
        while time.monotonic() < deadline:
            filled = self._order_filled_quantity(order_id) if order_id else None
            if filled:
                return prior + int(filled)
            if sole:
                position, readable = self._live_position_for_trade(trade)
                if readable and position is not None:
                    signed = self._position_quantity(position)
                    if signed and (1 if signed > 0 else -1) != self._expected_position_sign(trade):
                        return None          # direction flipped: not our add
                    if abs(signed) >= int(expected_total):
                        return abs(signed)
            time.sleep(0.2)
        return None

    def _sample_pnl_path(self, trade) -> None:
        # Gross, like peak_pnl and the TradeCandidate path, so every rtracker EXIT path
        # is on one basis. Diagnostics only: no exit guard reads pnl_path.
        # Trade.pnl_path_basis declares it; a sample that cannot be gross is skipped.
        try:
            gross = float(trade.unrealized_gross_pnl)
        except Exception:
            return None
        return sample_pnl_path(trade, pnl=gross)

    @staticmethod
    def _session_live_start() -> dtime:
        """Configured regular-session open (default 09:15:00)."""
        from config.settings import get_settings

        label = str(
            getattr(get_settings(), "real_mode_live_start", "09:15:00") or "09:15:00"
        ).strip()
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(label, fmt).time()
            except ValueError:
                continue
        return dtime(9, 15, 0)

    def _entry_mark_admission(
        self, *, inst_type: str, price: float, signal_time: Optional[datetime],
        entry_time: Optional[datetime], touch_quote: Optional[Dict[str, Any]],
        trading_symbol: str, is_recovery: bool, require_usable_touch: bool = False,
    ) -> Tuple[bool, str]:
        """Refuse an entry built on a pre-open signal or a stale entry mark.

        The 2026-07-28 loss (see config/settings.py, entry_preopen_signal_guard_enabled)
        was NOT a trading loss: a 09:10 pre-open signal executed at 09:15:34 on a
        previous-close mark of 131.70 against a live contract at 52.30, round-tripped in
        40ms for -Rs.10,402. Two independent checks, because neither covers the other:

          (a) SIGNAL FRESHNESS. A signal stamped before the session open was computed on
              pre-open data that never traded. Nothing downstream re-derives it, so the
              staleness survives all the way to the order.
          (b) MARK CONSISTENCY. An option's LTP reads yesterday's close until its first
              trade of the day prints, while the bid/ask are live from the first moment.
              A mark far outside the touch quote is therefore stale by construction.

        (b) needs a usable touch quote and (a) does not, so (a) still covers the open
        when the feed is not yet quoting. Recovery entries bypass both, matching
        _entry_risk_admission: a recovery re-establishes protection on exposure that
        already exists, and refusing it strands the very position this protects.
        """
        from config.settings import get_settings

        if is_recovery:
            return True, ""
        settings = get_settings()

        if _setting_enabled(
            getattr(settings, "entry_preopen_signal_guard_enabled", True), True
        ) and signal_time is not None and entry_time is not None:
            try:
                live_start = self._session_live_start()
                # Compare wall-clock in the session's own terms. Both stamps are IST in
                # this system; normalising defensively keeps a naive/aware mix from
                # raising inside an entry path.
                sig_local = signal_time.astimezone(IST) if signal_time.tzinfo else signal_time
                ent_local = entry_time.astimezone(IST) if entry_time.tzinfo else entry_time
                if (
                    sig_local.date() == ent_local.date()
                    and sig_local.time() < live_start
                    and ent_local.time() >= live_start
                ):
                    return False, (
                        f"signal {sig_local.strftime('%H:%M:%S')} predates the "
                        f"{live_start.strftime('%H:%M')} session open "
                        f"(entry {ent_local.strftime('%H:%M:%S')})"
                    )
            except Exception as exc:
                # A clock comparison must never stop trading.
                logger.debug(f"Pre-open signal guard skipped: {exc}")

        if not _setting_enabled(
            getattr(settings, "entry_mark_touch_validation_enabled", True), True
        ):
            return True, ""
        if str(inst_type or "").upper() != "OPT":
            return True, ""
        quote = touch_quote if isinstance(touch_quote, dict) else {}
        if not quote or not bool(quote.get("usable", False)):
            if require_usable_touch:
                detail = str(quote.get("reason") or "unavailable")
                return False, (
                    f"fresh executable option book required on {trading_symbol or inst_type} "
                    f"(touch quote {detail})"
                )
            return True, ""
        try:
            bid = float(quote.get("bid") or 0.0)
            ask = float(quote.get("ask") or 0.0)
            mark = float(price or 0.0)
        except (TypeError, ValueError):
            if require_usable_touch:
                return False, f"invalid executable option book on {trading_symbol or inst_type}"
            return True, ""
        if mark <= 0 or bid <= 0 or ask <= 0 or ask < bid:
            if require_usable_touch:
                return False, f"invalid executable option book on {trading_symbol or inst_type}"
            return True, ""
        if require_usable_touch:
            spread = ask - bid
            mid = (ask + bid) / 2.0
            spread_pct = (spread / mid * 100.0) if mid > 0 else float("inf")
            try:
                max_spread_pct = float(
                    getattr(settings, "option_touch_quote_max_spread_pct", 1.0) or 1.0
                )
            except (TypeError, ValueError):
                max_spread_pct = 1.0
            try:
                max_spread_ticks = int(
                    getattr(settings, "option_touch_quote_max_spread_ticks", 40) or 40
                )
            except (TypeError, ValueError):
                max_spread_ticks = 40
            tick_size = 0.05
            if max_spread_pct > 0 and spread_pct > max_spread_pct:
                return False, (
                    f"option book spread {spread_pct:.2f}% exceeds {max_spread_pct:.2f}% "
                    f"on {trading_symbol or inst_type}"
                )
            if max_spread_ticks > 0 and spread > max_spread_ticks * tick_size + 1e-9:
                return False, (
                    f"option book spread {spread:.2f} exceeds {max_spread_ticks} ticks "
                    f"on {trading_symbol or inst_type}"
                )
        max_mult = float(getattr(settings, "entry_mark_max_ask_multiple", 1.50) or 1.50)
        min_mult = float(getattr(settings, "entry_mark_min_bid_multiple", 0.50) or 0.50)
        if max_mult > 0 and mark > ask * max_mult:
            return False, (
                f"entry mark {mark:.2f} is {mark / ask:.2f}x the live ask {ask:.2f} "
                f"on {trading_symbol or inst_type} (stale mark)"
            )
        if min_mult > 0 and mark < bid * min_mult:
            return False, (
                f"entry mark {mark:.2f} is {mark / bid:.2f}x the live bid {bid:.2f} "
                f"on {trading_symbol or inst_type} (stale mark)"
            )
        return True, ""

    def _live_contract_risk_required(self) -> bool:
        from trading.angelone_broker import AngelOneBroker
        from trading.upstox_broker import UpstoxBroker

        return self._live_orders_permitted() and isinstance(
            getattr(self, "broker", None), (AngelOneBroker, UpstoxBroker)
        )

    @staticmethod
    def _upstox_contract_entry_error(symbol, key, lot_size, inst_type):
        from engine.expiry_manager import expiry_manager

        contract = expiry_manager.upstox_contract(str(key or ""), require_current=True)
        expected_type = "OPTIDX" if inst_type == "OPT" else "FUTIDX"
        if not contract or contract.get("symbol") != symbol or contract.get("instrumenttype") != expected_type:
            return "Exact execution contract missing from today's Upstox instrument master"
        if contract.get("lotsize") != lot_size:
            return "Entry lot size differs from today's Upstox instrument master"
        return ""

    def _entry_risk_admission(
        self, *, inst_type: str, direction: str, price: float, trailing_stop: float,
        initial_stop: float, lots: int, lot_size: int, instrument_multiplier: float,
        is_recovery: bool, instrument: str = "", scale_in_trade_id: str = "",
    ) -> Tuple[bool, str]:
        """Refuse an entry that would push the book past its stop-risk budget.

        Required for live exposure using an authoritative derivative master.

        Recovery entries bypass it, matching can_open_trade: a recovery is re-establishing
        protection on a position that already exists, and refusing it strands the very
        exposure this is meant to bound.
        """
        from config.settings import get_settings

        settings = get_settings()
        live_contract = self._live_contract_risk_required()
        if not live_contract and not _setting_enabled(
            getattr(settings, "entry_risk_budget_enabled", False), False
        ):
            return True, ""
        if is_recovery:
            return True, ""
        positions = list(self.open_trades.values())
        if live_contract:
            for trade in positions:
                if str(getattr(trade, "execution_status", "") or "").upper() == "RECOVERY_REQUIRED":
                    return False, "Broker exposure requires reconciliation"
                if getattr(trade, "id", "") != scale_in_trade_id and (
                    getattr(trade, "_scale_in_inflight", False)
                    or getattr(trade, "scale_in_requested_quantity", 0)
                ):
                    return False, "An unsettled scale-in prevents proof of available risk"
        # Enabled means the operator asked for a PROOF that this entry fits the budget.
        # When that proof cannot be computed the entry is refused, the same answer the
        # pure evaluator gives for unknown risk. This used to admit the entry on import
        # or evaluation failure (guard audit 2026-10-06, defect 5). Disabled, recovery
        # and exits are unaffected; the broker-margin fallback is a separate check.
        try:
            from engine.entry_risk_policy import evaluate_entry_risk
        except Exception as exc:
            logger.error(f"Entry-risk policy unavailable, refusing entry: {exc}")
            return False, (
                f"entry-risk budget enabled but its policy is unavailable "
                f"({type(exc).__name__}); new exposure refused"
            )

        candidate = {
            "inst_type": inst_type, "direction": direction, "price": price,
            "current_stop": trailing_stop, "initial_stop": initial_stop,
            "lots": lots, "lot_size": lot_size, "instrument_multiplier": instrument_multiplier,
            "instrument": instrument,
        }
        # Same budgets the circuit breaker enforces, so there is one source of truth for
        # "how much may be lost" rather than two ladders that can drift apart.
        capital_fut = float(getattr(settings, "capital_fut", 0.0) or 0.0)
        capital_opt = float(getattr(settings, "capital_opt", 0.0) or 0.0)
        capital_total = float(getattr(settings, "capital_total", 0.0) or 0.0)
        if live_contract:
            stop = float(trailing_stop or initial_stop or 0)
            long_premium = inst_type == "OPT" or str(direction).upper() in {"LONG", "BUY"}
            wrong_side = stop >= price if long_premium else stop <= price
            if wrong_side:
                return False, "Protective trigger is on the wrong side of the planned entry"

        def estimate_cost(item, entry, stop, qty, segment, multiplier):
            from engine.trade_accounting import estimate_trade_charges, estimate_upstox_derivative_fees
            from trading.kite_broker import KiteBroker
            details = item if isinstance(item, dict) else vars(item)
            index = str(details.get("instrument") or "").split()[0] if details.get("instrument") else ""
            from trading.upstox_broker import UpstoxBroker
            fees = estimate_upstox_derivative_fees(
                entry, stop, qty, segment, instrument=index, direction=str(details.get("direction") or ""),
                trade_date=datetime.now(IST).date(), multiplier=multiplier,
            ) if isinstance(self.broker, UpstoxBroker) else 0.0
            stop_buffer = 0.0
            if segment == "OPT":
                limit = KiteBroker._protective_stop_params("OPTCE", "SELL", stop)["price"]
                stop_buffer = max(0.0, stop - limit) * qty
            return stop_buffer + estimate_trade_charges(
                entry, stop, qty, segment, settings, multiplier,
                instrument=index, minimum_base_charge=fees,
            )

        try:
            decision = evaluate_entry_risk(
                candidate,
                positions,
                fut_limit_rs=capital_fut * float(getattr(settings, "risk_fut_pct", 0.0) or 0.0) / 100.0,
                opt_limit_rs=capital_opt * float(getattr(settings, "risk_opt_pct", 0.0) or 0.0) / 100.0,
                global_limit_rs=capital_total * float(getattr(settings, "max_daily_loss_pct", 0.0) or 0.0) / 100.0,
                max_trade_loss_abs=float(getattr(settings, "max_trade_loss_abs", 0.0) or 0.0),
                slippage_bps=float(getattr(settings, "slippage_fut_bps", 0.0) or 0.0),
                realized_loss_by_segment=self._realized_loss_by_segment(),
                estimate_cost=estimate_cost if live_contract else None,
            )
        except Exception as exc:
            logger.error(f"Entry-risk evaluation failed, refusing entry: {exc}")
            return False, (
                f"entry-risk budget could not be evaluated ({type(exc).__name__}: {exc}); "
                "new exposure refused"
            )
        return bool(decision.allowed), str(decision.reason)

    def _realized_loss_by_segment(self) -> Dict[str, float]:
        """Today's realised LOSSES per segment, as positive rupees.

        Folded into the budget so it shrinks as the session goes against you; without it
        the last entry of a losing day is admitted against the same headroom as the first.
        """
        today = datetime.now(IST).date()
        losses = {"FUT": 0.0, "OPT": 0.0}
        for trade in list(getattr(self, "closed_trades", []) or []):
            try:
                exit_time = getattr(trade, "exit_time", None) or getattr(trade, "entry_time", None)
                if exit_time is None:
                    continue
                stamp = exit_time if exit_time.tzinfo else IST.localize(exit_time)
                if stamp.astimezone(IST).date() != today:
                    continue
                pnl = float(getattr(trade, "pnl", 0.0) or 0.0)
            except (TypeError, ValueError, AttributeError):
                continue
            if pnl >= 0:
                continue
            segment = "FUT" if str(getattr(trade, "inst_type", "") or "").upper() == "FUT" else "OPT"
            losses[segment] += abs(pnl)
        return losses

    def _stale_entry_still_worth_filling(self, trade: Trade) -> bool:
        """Would this entry still be taken at the current mark?

        A stale LIMIT is cancelled and replaced with a MARKET order, which GUARANTEES a
        fill. That is the right move when the setup still stands and the limit merely
        missed by a tick. It is the wrong move when price has already run past the level
        the stop sits at, because the fill then enters a position that is a loser the
        instant it exists — the entry was never valid at that price, and the 12s window
        is long enough on a 1min option for exactly that to happen.

        Fails OPEN: with no usable mark or stop this returns True and the existing
        replace behaviour is unchanged. A missing price must not silently stop entries.
        """
        # An operator-supplied revalidator wins when one is wired; otherwise the
        # self-contained stop check below is what decides.
        revalidate = getattr(self, "entry_revalidator", None)
        if callable(revalidate):
            try:
                if revalidate(trade) is False:
                    return False
            except Exception as exc:
                logger.debug(f"Entry revalidation unavailable for {trade.id}: {exc}")

        try:
            mark = float(getattr(trade, "current_price", 0.0) or 0.0)
            stop = float(
                getattr(trade, "current_stop", 0.0)
                or getattr(trade, "trailing_stop", 0.0)
                or 0.0
            )
        except (TypeError, ValueError):
            return True
        if mark <= 0 or stop <= 0:
            return True
        # Options here are always long premium, so a breach is always downward.
        if str(getattr(trade, "inst_type", "") or "").upper() == "OPT":
            return mark > stop
        if str(getattr(trade, "direction", "LONG") or "LONG").upper() in {"LONG", "BUY"}:
            return mark > stop
        return mark < stop

    def _abandon_invalidated_entry(self, trade: Trade) -> None:
        """Close a cancelled entry that is no longer worth filling."""
        trade.status = "CLOSED"
        trade.pending_execution = False
        trade.execution_status = "REJECTED"
        trade.exit_reason = "SIGNAL_INVALIDATED"
        trade.exit_time = datetime.now(IST)
        trade.exit_price = trade.entry_price
        trade.execution_error = (
            "Stale entry cancelled and NOT replaced: the mark had already passed the "
            "stop, so a market fill would have opened a losing position"
        )
        with self._lock:
            if self.open_trades.pop(trade.id, None) is trade:
                self.closed_trades.append(trade)
        logger.warning(
            f"Abandoned stale entry for {trade.trading_symbol}: price moved past the "
            "stop while the limit rested, so it was not force-filled at market."
        )
        self._notify_delayed_entry_result(trade, False)

    def _notify_delayed_entry_result(self, trade: Trade, accepted: bool) -> None:
        """Deliver an entry result discovered after the initial broker ACK.

        The per-entry callback preserves the exact signal object during the normal
        process lifetime.  The manager-level callback is a restart fallback that lets
        Scanner resolve the signal by immutable lifecycle id.
        """
        callback = getattr(trade, "_entry_execution_result_callback", None)
        if callable(callback):
            try:
                callback(trade, bool(accepted))
                return
            except Exception as exc:
                logger.error(f"Delayed entry-result callback failed for {trade.id}: {exc}")
        fallback = getattr(self, "on_delayed_entry_result", None)
        if callable(fallback):
            try:
                fallback(trade, bool(accepted))
            except Exception as exc:
                logger.error(f"Global delayed entry-result callback failed for {trade.id}: {exc}")

    def _release_margin_reservations(self, reservation_ids) -> None:
        """Hand back margin an entry definitively did not consume.

        Never called on an ambiguous acknowledgement: an order that may be live has to
        keep holding its cash, or the next entry spends money this one is about to.
        """
        release = getattr(self.broker, "release_entry_margin_reservation", None)
        if not callable(release):
            return
        for reservation_id in list(reservation_ids or []):
            try:
                release(reservation_id)
            except Exception as exc:
                logger.debug(f"Margin reservation {reservation_id} release failed: {exc}")

    def _trade_owning_protective_order(self, order_id: str, exclude_id: str = ""):
        """The open trade that holds this protective order id, if any."""
        if not order_id:
            return None
        # Partially-constructed managers are a real pattern here (recovery paths and
        # tests), and an AttributeError would abort a flatten mid-way.
        with self._lock:
            for other in list(getattr(self, "open_trades", {}).values()):
                if other.id == exclude_id:
                    continue
                if str(getattr(other, "protective_order_id", "") or "") == str(order_id):
                    return other
        return None

    def _unowned_protective_stop_blocks_flatten(self, trade: Trade, reason: str) -> bool:
        """Deal with a working stop this trade has no order id for.

        Returns True when the flatten must be WITHHELD. Only a scan that definitively
        reports no stop, or one whose stop we successfully cancelled, may proceed —
        UNKNOWN/AMBIGUOUS/CONFLICT all mean we cannot prove the position is unprotected,
        and exiting on that basis is how a position gets closed twice.
        """
        inspector = getattr(self.broker, "inspect_working_protective_stop", None)
        if not callable(inspector):
            return False
        side = self._protective_exit_side(trade)
        try:
            found = inspector(
                trade.trading_symbol,
                trade.symbol_token,
                side,
                quantity=int(trade.broker_quantity or trade.quantity or 0),
                product_type=self.product_type,
            )
        except Exception as exc:
            return self._withhold_flatten(
                trade,
                f"{reason}; protective-stop scan failed ({exc}), market flatten withheld "
                "to avoid exiting over a stop that may be resting",
            )
        state = str((found or {}).get("state") or "UNKNOWN").upper()
        if state == "NONE":
            # Only a scan that actually cleared the contract resets the counter. Doing it
            # for FOUND too let an alternating UNKNOWN / FOUND-but-uncancellable sequence
            # withhold the flatten forever without ever reaching the escalation.
            self._clear_flatten_block(trade)
            return False
        if state == "FOUND":
            order_id = str((found or {}).get("order_id") or "")
            # Two trades on one strike is a supported configuration (same index, different
            # timeframes produce the same option symbol and the same exit side), and the
            # order-book scan matches on contract, not on trade. Cancelling a stop that
            # another OPEN trade is actively managing would strip its protection.
            owner = self._trade_owning_protective_order(order_id, exclude_id=trade.id)
            if owner is not None:
                return self._withhold_flatten(
                    trade,
                    f"{reason}; the only working stop on {trade.trading_symbol} belongs to "
                    f"trade {owner.id}, market flatten withheld to avoid stripping its protection",
                )
            try:
                cancelled = bool(order_id) and self.broker.cancel_order(order_id, "STOPLOSS")
            except (BrokerOrderRejected, PermissionError) as exc:
                # cancel_order takes the account lease and raises when another instance
                # holds it. Bare, this unwound out of check_session_end's per-trade loop
                # and aborted the EOD flatten for every remaining trade.
                return self._withhold_flatten(
                    trade, f"{reason}; orphan stop cancellation refused ({exc})"
                )
            if cancelled:
                logger.warning(
                    f"Cancelled orphan SL-M {order_id} for {trade.trading_symbol} before "
                    "flattening; the trade never held its order id."
                )
                return False
            return self._withhold_flatten(
                trade,
                f"{reason}; orphan protective stop {order_id or '(no id)'} could not be "
                "cancelled, market flatten withheld to avoid over-exit",
            )
        return self._withhold_flatten(
            trade,
            f"{reason}; protective-stop state is {state}, market flatten withheld to "
            "avoid double-exiting the position",
        )

    @staticmethod
    def _accepts_kwarg(func, name: str) -> bool:
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return False
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            return True
        return name in signature.parameters

    def _tag_kwargs(self, func, trade: Trade, leg: str, attempt: int = 0) -> Dict[str, str]:
        """`{"tag": ...}` only when the broker actually accepts it.

        SignalSimBroker and every test double predate the argument, and an unexpected
        keyword is a TypeError that aborts the whole order path — which is worse than
        going untagged. Tagging is an enhancement to recovery, so it must degrade to
        nothing rather than take the order down with it.
        """
        if not callable(func) or not self._accepts_kwarg(func, "tag"):
            return {}
        tag = self._broker_order_tag(trade, leg, attempt)
        return {"tag": tag} if tag else {}

    def _broker_order_tag(self, trade: Trade, leg: str, attempt: int = 0) -> str:
        """The exact tag this trade's `leg` order carries, or "" when unsupported."""
        tagger = getattr(self.broker, "order_tag", None)
        if not callable(tagger):
            return ""
        try:
            return str(tagger(trade.id, leg, attempt) or "")
        except TypeError:
            # Simulation brokers still on the single-argument form.
            try:
                return str(tagger(trade.id) or "")
            except Exception:
                return ""
        except Exception:
            return ""

    # Exits used to share one leg identity, "EXIT". Splitting them into
    # PARTIAL_EXIT / FULL_EXIT / RECOVERY_EXIT is what lets two orders on one trade be
    # told apart -- but an order placed by the PREVIOUS build is resting at the exchange
    # under the old tag, and a lookup for the new leg would not find it. That trade
    # would fall through to shape matching, which cannot tell two trades on one strike
    # apart: the exact ambiguity the tag exists to remove, reintroduced by the upgrade
    # that was meant to remove it. So each new exit leg names the legacy tag it replaces
    # and the lookup tries both.
    _LEGACY_TAG_LEGS = {
        "PARTIAL_EXIT": ("EXIT",),
        "FULL_EXIT": ("EXIT",),
        "RECOVERY_EXIT": ("EXIT",),
    }

    def _recover_order_id_from_tag(self, trade: Trade, leg: str) -> str:
        """Ask the broker whether this leg's tagged order actually reached the exchange.

        This is what a tag is FOR. Tags were already being written on entry orders and
        read by nothing, so a lost acknowledgement stayed unknowable and every caller had
        to latch and escalate to a human. The order book echoes the tag, so the question
        has a real answer: either an order carrying this exact tag exists or it does not.

        Returns the order id when exactly one match exists. Everything else returns "" --
        and the distinction matters to the caller, so AMBIGUOUS (two orders sharing a
        tag) escalates rather than being quietly treated as absent.
        """
        leg_name = str(leg).upper()
        attempt = (
            int(getattr(trade, "entry_replace_count", 0) or 0)
            if leg_name == "ENTRY"
            else 0
        )
        # ENTRY has one format migration that no other leg has: pre-2026-08-16
        # attempt-zero orders used an underscore.  A restart with a persisted
        # ENTRY_PENDING/RECOVERY_REQUIRED row has no broker id, so it must query every
        # owned alias before it can safely reconcile or cancel the resting order.  The
        # Kite helper reads one order-book snapshot and reports AMBIGUOUS if aliases
        # resolve to more than one row.
        entry_inspector = (
            getattr(self.broker, "inspect_entry_order_for_trade", None)
            if leg_name == "ENTRY"
            else None
        )
        if callable(entry_inspector):
            try:
                result = entry_inspector(trade.id, attempts=max(8, attempt + 1)) or {}
            except Exception as exc:
                logger.warning(f"Tagged entry recovery failed for {trade.id}: {exc}")
                return ""
            state = str(result.get("state") or "UNKNOWN").upper()
            matched_tag = str(result.get("tag") or "")
            if state == "AMBIGUOUS":
                self._mark_recovery(
                    trade,
                    "Multiple broker orders match this trade's entry-tag aliases",
                )
                return ""
            if state != "FOUND":
                return ""
            order_id = str(result.get("order_id") or "")
            if not order_id:
                self._mark_recovery(
                    trade,
                    "Owned entry tag"
                    + (f" {matched_tag}" if matched_tag else "")
                    + " was found without an order id",
                )
                return ""
            if matched_tag and matched_tag != self._broker_order_tag(trade, "ENTRY", attempt):
                logger.info(
                    f"Recovered {trade.id} entry under a legacy entry tag {matched_tag}."
                )
            return order_id

        inspector = getattr(self.broker, "inspect_order_by_tag", None)
        if not callable(inspector):
            return ""
        tag = self._broker_order_tag(trade, leg, attempt)
        if not tag:
            return ""
        legacy = [
            self._broker_order_tag(trade, alt, attempt)
            for alt in self._LEGACY_TAG_LEGS.get(leg_name, ())
        ]
        result, matched_tag = {}, tag
        for candidate_tag in [tag] + [t for t in legacy if t]:
            try:
                result = inspector(candidate_tag) or {}
            except Exception as exc:
                logger.warning(f"Tagged {leg.lower()} recovery failed for {trade.id}: {exc}")
                return ""
            state = str(result.get("state") or "UNKNOWN").upper()
            if state in {"FOUND", "AMBIGUOUS"}:
                matched_tag = candidate_tag
                if candidate_tag != tag:
                    logger.info(
                        f"Recovered {trade.id} {leg.lower()} under the legacy tag "
                        f"{candidate_tag}; placed before exit legs were split."
                    )
                break
        state = str(result.get("state") or "UNKNOWN").upper()
        if state == "AMBIGUOUS":
            self._mark_recovery(
                trade,
                f"Multiple broker orders share the exact {leg.lower()} tag {matched_tag}",
            )
            return ""
        if state != "FOUND":
            return ""
        order_id = str(result.get("order_id") or "")
        if not order_id:
            self._mark_recovery(
                trade, f"Exact {leg.lower()} tag {matched_tag} was found without an order id"
            )
        return order_id

    def _tagged_exit_state(self, trade: Trade, leg: str = "EXIT") -> str:
        """FOUND / AMBIGUOUS / ABSENT / UNKNOWN for this trade's exit order, by tag.

        Takes the LEG, because the answer is only meaningful about the tag that was
        actually written. Hardcoding "EXIT" while emergency_flatten_trade sends its order
        under "RECOVERY_EXIT" did not fail closed: both are valid legs, so it built a
        real tag string (U1X0<compact>) that nothing ever writes and asked the order book
        about it. A successful read that does not find a never-written tag answers
        ABSENT -- so a book positively holding the RECOVERY_EXIT order reported "nothing
        reached the exchange, a retry is safe", the caller returned before latching, and
        check_session_end re-dispatched a fresh MARKET exit on every pass.

        ABSENT now requires EVERY candidate tag to be absent. Anything else -- FOUND,
        AMBIGUOUS, or an unreadable book -- is returned as it stands, because only
        "definitely not at the exchange" may license another exit.
        """
        inspector = getattr(self.broker, "inspect_order_by_tag", None)
        if not callable(inspector):
            return "UNKNOWN"
        candidates = [str(leg or "").upper(), *self._LEGACY_TAG_LEGS.get(str(leg or "").upper(), ())]
        verdict = "UNKNOWN"
        for candidate in candidates:
            tag = self._broker_order_tag(trade, candidate, 0)
            if not tag:
                continue
            try:
                state = str((inspector(tag) or {}).get("state") or "UNKNOWN").upper()
            except Exception as exc:
                logger.warning(f"Tagged exit lookup failed for {trade.id}: {exc}")
                return "UNKNOWN"
            if state != "ABSENT":
                return state
            verdict = "ABSENT"
        return verdict

    # The states a Kite order can finish in having NOT filled the rest of its quantity.
    # COMPLETE is deliberately absent: it means the full quantity did fill, so a residual
    # sitting alongside it is stale position data, and exiting on that would sell a
    # quantity that is no longer there and open a naked short.
    _DEAD_EXIT_STATES = {"CANCELLED", "REJECTED"}

    def _abandoned_exit_verdict(self, trade: Trade) -> str:
        """Can the exit order that partially filled still fill the remainder?

        "DEAD" only when the broker states a terminal non-fill, "WORKING" while it is
        still live, "UNKNOWN" when the question cannot be answered. UNKNOWN must never
        read as DEAD — sending a second exit on top of one that is still resting is how
        an intended exit becomes a reversed naked position.
        """
        order_id = str(getattr(trade, "exit_order_id", "") or "")
        if not order_id:
            # An exit was submitted (that is what set PARTIAL_EXIT) but we hold no id to
            # interrogate, so its fate is unknowable here.
            return "UNKNOWN"
        lifecycle = getattr(self.broker, "get_order_lifecycle", None)
        if not callable(lifecycle):
            return "UNKNOWN"
        try:
            detail = lifecycle(order_id) or {}
        except Exception as exc:
            logger.debug(f"Exit lifecycle lookup failed for {trade.id}: {exc}")
            return "UNKNOWN"
        state = str(detail.get("state") or "").upper()
        if not state or state == "UNKNOWN":
            return "UNKNOWN"
        return "DEAD" if state in self._DEAD_EXIT_STATES else "WORKING"

    def _resume_abandoned_partial_exit(self, trade: Trade, residual: int) -> bool:
        """Square off the remainder once the broker is finished with the exit order."""
        attempts = int(getattr(trade, "partial_exit_flatten_count", 0) or 0)
        if attempts >= 3:
            self._mark_recovery(
                trade,
                f"Partially filled exit left {residual} quantity and {attempts} re-exit "
                "attempts were all refused by the broker. This position is unprotected "
                "— reconcile it manually.",
            )
            return False
        trade.partial_exit_flatten_count = attempts + 1
        # The dead order cannot fill anything further, so its id must not be carried into
        # the new submission; emergency_flatten_trade stamps the replacement's own id.
        trade.exit_order_id = ""
        trade.pending_exit = False
        return self.emergency_flatten_trade(trade, "PARTIAL_EXIT_ABANDONED")

    def emergency_flatten_trade(self, trade: Trade, reason: str) -> bool:
        """Flatten a confirmed recovery position after safely cancelling its SL-M."""
        if self._live_account_send_blocked():
            # Reached mode-blind from check_session_end, risk_manager.halt_trading and
            # close_segment_trades; only _maybe_auto_flatten_unprotected checked mode.
            logger.error(
                f"Emergency flatten refused for {trade.trading_symbol}: mode is "
                f"{getattr(self, 'mode', '')!r} but the live account broker is still attached."
            )
            return False
        if not self.broker or not trade.symbol_token or not trade.trading_symbol:
            # There is no contract to submit against, so re-entering here every cycle
            # cannot ever succeed. check_session_end() calls this on every pass for as
            # long as the row stays RECOVERY_REQUIRED, which on 2026-07-31 turned into
            # CRITICAL "emergency flatten unavailable" for NIFTY and SENSEX every 30s
            # until the session ended. Latch it once and escalate to the operator: the
            # position may or may not exist at the broker, and only a human can settle
            # that — we must not invent a terminal state for an unknown position.
            if not hasattr(self, "_unflattenable_latched"):
                self._unflattenable_latched = set()
            if trade.id in self._unflattenable_latched:
                return False
            self._unflattenable_latched.add(trade.id)
            self._mark_recovery(
                trade,
                f"{reason}; emergency flatten unavailable (no broker contract on the "
                "trade). Reconcile this position manually at your Kite terminal — "
                "the system cannot act on it and will stop retrying.",
            )
            return False
        if trade.id in getattr(self, "_flatten_ambiguous_latched", set()):
            # A previous exit for this trade was never acknowledged. Sending another is
            # how one ambiguous exit becomes two real ones.
            return False
        if trade.protective_order_id:
            # Initialised BEFORE the try. `cancelled` used to be bound only by the call
            # itself, so the refused-cancel path below fell through to `if not cancelled`
            # with the name unbound and raised UnboundLocalError. check_session_end has no
            # per-trade try, so that escaped the 15:25 force exit and took every remaining
            # trade in the snapshot with it — and the orphan sweep after it. Reproduced.
            cancelled = False
            refused = False
            try:
                cancelled = bool(
                    self.broker.cancel_order(trade.protective_order_id, "STOPLOSS")
                )
            except (BrokerOrderRejected, PermissionError) as exc:
                # cancel_order takes the account lease and raises when it is held
                # elsewhere. ensure_broker_protection was hardened against exactly this;
                # this path was not.
                # _withhold_flatten returns False once the operator's blocked-flatten
                # deadline elapses ("stop withholding"). Discarding that logged
                # "exiting anyway", set reversal_watch_required — and then did not exit.
                refused = True
                if self._withhold_flatten(
                    trade, f"{reason}; protective stop cancellation refused ({exc})"
                ):
                    return False
            if not cancelled and not refused:
                # A refusal has already been through _withhold_flatten once; running it
                # again here would double-count the block deadline for one failure.
                if self._withhold_flatten(
                    trade,
                    f"{reason}; protective stop cancellation unconfirmed, market flatten "
                    "withheld to avoid over-exit",
                ):
                    return False
            if cancelled:
                trade.protective_order_id = ""
            else:
                # Past the deadline we flatten anyway, but the stop was never PROVEN gone.
                # Clearing the id here is how an SL-M ends up owned by nobody: nothing
                # scans closed trades for it, so when it later triggers it sells against a
                # flat book — a naked written option in a product the broker will not
                # square off. Keep the id and say so.
                self._mark_recovery(
                    trade,
                    f"{reason}; flattened past the block deadline with protective stop "
                    f"{trade.protective_order_id} UNCONFIRMED. It may still be resting — "
                    "cancel it at the Zerodha terminal before it triggers against a flat "
                    "book.",
                )
        elif self._unowned_protective_stop_blocks_flatten(trade, reason):
            # Holding no stop id is NOT proof that no stop exists. A lost acknowledgement
            # leaves an SL-M resting that we never learned the id of; market-exiting on
            # top of it flattens the position while the stop is still live, and when the
            # stop later triggers it opens a naked reversed leg.
            return False
        # Same rule for the server-side trigger: a GTT that survives this flatten will
        # fire later against a flat book and open a naked reversed leg.
        if not self._release_gtt_fallback(trade, "flattening the position"):
            if self._withhold_flatten(
                trade,
                f"{reason}; GTT cancellation unconfirmed, market flatten withheld to avoid over-exit",
            ):
                return False
        # NEVER size a flatten from the INTENDED quantity. broker_quantity is 0 on every
        # row whose entry the broker did not confirm: an ambiguous acknowledgement, an
        # entry not found in the order book within the timeout, a stale entry that could
        # not be safely cancelled, and every prior-session row on restart. check_session_end
        # routes all of those straight here.
        #
        # Options are always BOUGHT, so the flatten side is SELL. Selling a fill that never
        # happened does not close anything — it OPENS a naked written option in
        # CARRYFORWARD/NRML, the product chosen precisely so the broker will not square it
        # off, at 15:25 with no session left to correct it. The sibling
        # _maybe_auto_flatten_unprotected already refuses on exactly this test.
        qty = self._confirmed_broker_quantity(trade)
        if qty <= 0:
            # Ask the book once before concluding anything. A real position we simply
            # failed to record is still worth flattening; a flat book is nothing to do;
            # an unreadable book is NOT flat and must not be guessed at.
            row, readable = self._live_position_for_trade(trade)
            if not readable:
                self._mark_recovery(
                    trade,
                    f"{reason}; the broker never confirmed a fill and the positions book "
                    "is unreadable, so the size to flatten is unknown. NOT sending an "
                    "order — a market exit here could open a naked short. Reconcile this "
                    "position manually at your Kite terminal.",
                )
                return False
            try:
                qty = abs(int(float((row or {}).get("netqty") or 0)))
            except (TypeError, ValueError):
                qty = 0
            if qty <= 0:
                logger.info(
                    f"Flatten skipped for {trade.trading_symbol}: the broker never "
                    "confirmed a fill and the book shows flat, so there is nothing to "
                    "square off."
                )
                return False
        side = self._protective_exit_side(trade)
        try:
            order_id = self.broker.close_order(
                trade.trading_symbol, trade.symbol_token, qty, side, self.product_type,
                **self._tag_kwargs(self.broker.close_order, trade, "RECOVERY_EXIT"),
            )
        except (BrokerOrderRejected, PermissionError) as exc:
            # close_order takes the account lease and needs a session, so it raises just
            # like the cancels above. Bare, it unwound out of check_session_end's
            # per-trade loop and aborted the EOD flatten for every remaining trade.
            self._withhold_flatten(trade, f"{reason}; emergency flatten refused ({exc})")
            return False
        except BrokerOrderAmbiguous as exc:
            # The exit MAY have reached the exchange, so it must never be re-sent blind.
            # Marking recovery alone did NOT achieve that: check_session_end re-dispatches
            # every RECOVERY_REQUIRED row to this method on each pass (~1 Hz), so a fresh
            # MARKET exit went out every second until the close. If the first one had in
            # fact filled, the account ends up SHORT and naked in an NRML product the
            # broker will not square off. Latch it: only a reconciliation that reads the
            # real net quantity may release it.
            #
            # But "may be live" is now a question with a real answer. The exit carries an
            # exact tag, so the order book can be asked directly instead of freezing and
            # escalating to a human:
            #   FOUND  -> it DID reach the exchange. Adopt the id and treat it as the
            #             submission it actually is; nothing to latch and nobody to call.
            #   ABSENT -> a SUCCESSFUL read that found no such tag proves it never
            #             arrived, so the next pass may retry safely. Still recovery, but
            #             not latched — latching here is what strands a live position.
            # Only UNKNOWN (unreadable book) and AMBIGUOUS fall through to the latch,
            # which is the honest outcome when the question genuinely cannot be answered.
            # Must match the leg this method SENDS. emergency_flatten_trade now tags
            # its order RECOVERY_EXIT, so looking up "EXIT" would search for a tag that
            # is never written and silently fall through to shape matching -- which
            # cannot tell two trades on one strike apart, the exact ambiguity the tag
            # exists to remove.
            recovered_exit = self._recover_order_id_from_tag(trade, "RECOVERY_EXIT")
            if recovered_exit:
                trade.exit_order_id = recovered_exit
                trade.pending_exit = True
                trade.pending_execution = False
                trade.execution_status = "EXIT_PENDING"
                trade.execution_error = reason
                trade.exit_reason = "RECOVERY_AUTO_FLATTEN"
                self._clear_flatten_block(trade)
                self._mark_dirty()
                logger.warning(
                    f"Flatten acknowledgement was lost for {trade.trading_symbol}, but "
                    f"its exit tag resolved to order {recovered_exit} — adopted instead "
                    "of latching."
                )
                return True
            # The leg this method SENDS (:RECOVERY_EXIT above), not the legacy "EXIT".
            # _recover_order_id_from_tag has already returned "" by this point, and it
            # does that for AMBIGUOUS and for FOUND-without-an-order-id as well as for a
            # genuine absence -- so asking about the wrong tag here turned a POSITIVE
            # sighting of the flatten order into "a retry is safe".
            if self._tagged_exit_state(trade, "RECOVERY_EXIT") == "ABSENT":
                self._mark_recovery(
                    trade,
                    f"{reason}; flatten acknowledgement was lost ({exc}), but the exit "
                    "tag is absent from the order book, so nothing reached the exchange. "
                    "A retry is safe.",
                )
                self._mark_dirty()
                return False
            if not hasattr(self, "_flatten_ambiguous_latched"):
                self._flatten_ambiguous_latched = set()
            self._flatten_ambiguous_latched.add(trade.id)
            self._mark_recovery(
                trade,
                f"{reason}; emergency flatten acknowledgement ambiguous ({exc}). The exit "
                "may be live — no further exit will be sent until broker positions "
                "confirm the real quantity. Reconcile this position manually.",
            )
            logger.critical(
                f"FLATTEN AMBIGUOUS for {trade.trading_symbol}: an exit may already be "
                "resting at the exchange. Further automatic exits are latched off for "
                "this trade until reconciliation proves its net quantity."
            )
            self._mark_dirty()
            return False
        if not order_id:
            self._withhold_flatten(trade, f"{reason}; emergency flatten order failed")
            return False
        trade.exit_order_id = str(order_id)
        trade.pending_exit = True
        trade.pending_execution = False
        trade.execution_status = "EXIT_PENDING"
        trade.execution_error = reason
        trade.exit_reason = "RECOVERY_AUTO_FLATTEN"
        self._clear_flatten_block(trade)
        self._mark_dirty()
        logger.critical(f"Recovery fallback flatten submitted for {trade.trading_symbol}: {order_id}")
        return True

    def _contract_is_no_longer_listed(self, trade: Trade) -> Optional[bool]:
        """Is this trade's contract absent from the current instrument master?

        None when the question cannot be answered — the master is not loaded, or the
        lookup itself failed. The caller must treat None as "unknown", never as "gone":
        concluding a contract has expired from a master we could not read is the same
        mistake as concluding a position is flat from an order-book scan that errored.
        """
        try:
            from engine.expiry_manager import expiry_manager
        except Exception:
            return None
        df = getattr(expiry_manager, "df", None)
        if df is None or getattr(df, "empty", True):
            return None
        symbol = str(getattr(trade, "trading_symbol", "") or "").upper()
        token = str(getattr(trade, "symbol_token", "") or "")
        if not symbol and not token:
            return None
        try:
            if token and (df["token"].astype(str) == token).any():
                return False
            if symbol and (df["symbol"].astype(str).str.upper() == symbol).any():
                return False
        except Exception:
            return None
        return True

    def _settle_expired_contract(
        self, trade_id: str, trade: Trade, now: Optional[datetime] = None
    ) -> bool:
        """Close a row whose option contract has expired and left the broker's book.

        An expired contract never returns to the positions book, so reconciliation can
        never close this row on its own: it sat OPEN at a stale premium indefinitely,
        counted against exposure and position slots, and emergency_flatten_trade kept
        submitting exits against a contract that no longer trades.

        The settlement VALUE is genuinely unknowable here — the contract cash-settles at
        intrinsic against the closing spot, and by the time we notice, that spot is not
        something this object can reconstruct. So it books the conservative case (expired
        worthless, the worst outcome for a bought option, which can never overstate the
        account), marks the P&L as estimated, and escalates. Options are always BOUGHT
        here, so this cannot hide a liability — only understate a gain.
        """
        if str(getattr(trade, "inst_type", "") or "").upper() != "OPT":
            return False
        entry = getattr(trade, "entry_time", None)
        if entry is None:
            return False
        entry_ist = IST.localize(entry) if entry.tzinfo is None else entry.astimezone(IST)
        # Injectable so the session-boundary rule below can be tested at a fixed point.
        # Reading the wall clock unconditionally made this method behave differently
        # depending on when the process ran, and its test passed before 15:35 and failed
        # after — which is exactly how it broke, at 21:23.
        now = now or datetime.now(IST)
        if now.tzinfo is None:
            now = IST.localize(now)
        # Never intraday on the entry's own session: a contract missing from one
        # positions read is far more likely to be a slow book than an expired contract.
        if entry_ist.date() >= now.date() and now.time() < self._parse_time("15:35"):
            return False
        if self._contract_is_no_longer_listed(trade) is not True:
            return False

        trade.status = "CLOSED"
        trade.pending_exit = False
        trade.pending_execution = False
        trade.execution_status = "CLOSED"
        trade.exit_reason = "EXPIRY_SETTLED"
        trade.exit_time = now
        trade.exit_price = 0.0
        trade.execution_error = (
            "Contract expired and was cash-settled at the exchange; it is gone from the "
            "instrument master and will never return to the positions book. Booked as "
            "expired worthless — the P&L here is an ESTIMATE and must be verified against "
            "the broker's own statement."
        )
        trade.protective_order_id = ""
        trade.gtt_trigger_id = ""
        with self._lock:
            if self.open_trades.pop(trade_id, None) is trade:
                self.closed_trades.append(trade)
        logger.critical(
            f"EXPIRY SETTLED: {trade.trading_symbol} left open through expiry and has "
            "been closed locally as worthless. Verify the realised amount against your "
            "Zerodha P&L — if it settled in the money, this row understates it."
        )
        self._mark_dirty()
        return True

    def reconcile_broker_positions(self, positions_resp: Optional[Dict] = None) -> Dict:
        """Reconcile local trades and broker-only positions in both directions."""
        snapshot_broker = self.broker
        if not snapshot_broker:
            return {"checked": False, "orphans": 0, "closed": 0, "confirmed": 0}
        get_positions = getattr(snapshot_broker, "get_positions", None)
        if positions_resp is None:
            if not callable(get_positions):
                return {"checked": False, "orphans": 0, "closed": 0, "confirmed": 0}
            positions_resp = get_positions()
        if not isinstance(positions_resp, dict):
            return {"checked": False, "orphans": 0, "closed": 0, "confirmed": 0}
        rows = positions_resp.get("data")
        if not isinstance(rows, list):
            return {"checked": False, "orphans": 0, "closed": 0, "confirmed": 0}

        allowed_prefixes = self._reconciliation_symbol_prefixes()
        # Keyed on tradingsymbol ALONE until 2026-08-16, with no product term, so when a
        # contract carried more than one product row the comprehension silently kept
        # whichever happened to come last. Not reachable from this system's own orders —
        # they all go out under one product — but a manual MIS trade taken on the same
        # strike from the Kite app puts a second row on that symbol, and reconciliation
        # is the single authority allowed to close a position. Rank deterministically:
        # our own product first, then a row that actually holds quantity.
        broker_pos: Dict[str, Dict] = {}
        foreign = {}
        for row in rows:
            symbol = str(row.get("tradingsymbol") or "")
            if not symbol or not symbol.upper().startswith(allowed_prefixes):
                continue
            if not self._matches_position_product(row):
                if self._position_quantity(row):
                    product = str(row.get("product") or row.get("producttype") or "")
                    foreign[f"{symbol} [{product}]"] = dict(
                        row, foreign_product=True, status="ORPHANED", execution_status="ORPHANED"
                    )
                continue
            held_row = broker_pos.get(symbol)
            if held_row is None or (not self._position_quantity(held_row) and self._position_quantity(row)):
                broker_pos[symbol] = row
        closed = 0
        confirmed = 0
        changed = False
        with self._lock:
            if self.broker is not snapshot_broker:
                return {"checked": False, "orphans": 0, "closed": 0, "confirmed": 0}
            local_snapshot = list(self.open_trades.items())
            native = {(row.get("execution_symbol"), row.get("execution_token")): row
                      for row in broker_pos.values() if row.get("execution_symbol") and row.get("execution_token")}
            for _, trade in local_snapshot:
                row = native.get((trade.trading_symbol, str(trade.symbol_token or "")))
                if row and (trade.trading_symbol, trade.symbol_token) != (row["tradingsymbol"], row["symboltoken"]):
                    trade.trading_symbol, trade.symbol_token = row["tradingsymbol"], row["symboltoken"]
                    changed = True

        # One position row can back several local trades on the same contract, so the
        # row's net quantity is split ONCE here rather than handed to each of them.
        # See _allocate_broker_quantity: with a single trade on the symbol this is the
        # net verbatim and nothing downstream changes.
        symbol_groups: Dict[str, list] = {}
        for _tid, _trade in local_snapshot:
            _symbol = str(_trade.trading_symbol or _trade.instrument)
            symbol_groups.setdefault(_symbol, []).append((_tid, _trade))
        broker_allocation: Dict[str, int] = {}
        for _symbol, _group in symbol_groups.items():
            _row = broker_pos.get(_symbol)
            if _row is None:
                continue
            broker_allocation.update(
                self._allocate_broker_quantity(
                    _group, self._position_quantity(_row), self._ledger_owned_quantity
                )
            )

        local_symbols = set()
        for trade_id, trade in local_snapshot:
            symbol = str(trade.trading_symbol or trade.instrument)
            local_symbols.add(symbol)
            position = broker_pos.get(symbol)
            # Kite keeps a squared-off day position in `net` with quantity 0 for the rest
            # of the session. For a trade whose entry has NOT been confirmed filled, that
            # row is the residue of an EARLIER trade on this contract and says nothing
            # about this order. Reading it as "the position went flat" closed a trade
            # whose entry was still working at the exchange: it booked the earlier
            # trade's realised P&L onto the new trade (feeding the daily loss breaker),
            # threw away the live order id, and left a real position with no stop that
            # even the EOD force-exit could not see.
            #
            # Treat it as "no position" so the stale-entry logic below handles it — that
            # branch already consults the order book, cancel/replaces a stale working
            # order, and marks RECOVERY_REQUIRED past the timeout.
            if (
                position is not None
                and self._position_quantity(position) == 0
                and self._entry_still_confirming(trade)
                and self._confirmed_broker_quantity(trade) <= 0
            ):
                position = None
            # Restart-safe entry recovery: a pre-upgrade pending row can have no order
            # id even while its legacy-tagged order is still resting at Kite.  Recover
            # the exact id before any status/cancel-replace logic.  This is read-only;
            # an absent or unreadable book never manufactures a new entry.
            entry_status = str(getattr(trade, "execution_status", "") or "").upper()
            if (
                not str(getattr(trade, "broker_order_id", "") or "").strip()
                and not bool(getattr(trade, "pending_exit", False))
                and not bool(getattr(trade, "is_ghost", False))
                and entry_status in {
                    "ENTRY_PENDING", "ENTRY_SUBMITTED", "PARTIAL", "RECOVERY_REQUIRED",
                }
            ):
                recovered_entry_id = self._recover_order_id_from_tag(trade, "ENTRY")
                if recovered_entry_id:
                    trade.broker_order_id = recovered_entry_id
                    trade.pending_execution = True
                    trade.execution_status = "ENTRY_SUBMITTED"
                    if getattr(trade, "entry_submitted_at", None) is None:
                        trade.entry_submitted_at = datetime.now(IST)
                    trade.execution_error = ""
                    changed = True

            order_status = None
            if trade.execution_status == "ENTRY_SUBMITTED" and trade.broker_order_id:
                getter = getattr(self.broker, "get_order_status", None)
                if callable(getter):
                    order_status = getter(trade.broker_order_id)
            if position is None:
                # Before anything else: if the contract itself has expired, no future
                # reconcile can ever resolve this row. Only the ENTRY_SUBMITTED case was
                # handled below, so every other status fell through to the bare `continue`
                # at the end of this branch and the trade stayed OPEN forever.
                if self._settle_expired_contract(trade_id, trade):
                    closed += 1
                    changed = True
                    continue
                if (
                    trade.execution_status == "ENTRY_SUBMITTED"
                    and trade.entry_submitted_at is not None
                ):
                    submitted_at = trade.entry_submitted_at
                    if submitted_at.tzinfo is None:
                        submitted_at = IST.localize(submitted_at)
                    age_seconds = (
                        datetime.now(IST) - submitted_at.astimezone(IST)
                    ).total_seconds()
                    status_text = self._order_status_text(order_status)
                    if status_text in {"rejected", "cancelled", "canceled"}:
                        # A TERMINAL status is not the same as UNFILLED. Kite reports an
                        # order that partially executed and was then cancelled as
                        # CANCELLED with a non-zero filled quantity, and this branch used
                        # to read the status STRING alone — booking the trade CLOSED at
                        # pnl 0 and popping it out of open_trades while that quantity sat
                        # live at the broker with no protective stop and no local row.
                        # The positions snapshot is taken once at the top of the pass
                        # while get_order_status is called live per trade, so `position
                        # is None` here does NOT mean flat.
                        filled_qty = order_filled_quantity(order_status)
                        if filled_qty > 0:
                            trade.broker_quantity = int(filled_qty)
                            # The second way an entry fill becomes known: the postback
                            # was missed and the order book states it instead. Same
                            # order id, so recording here is the same idempotent row.
                            self._record_fill(
                                trade, getattr(trade, "broker_order_id", ""),
                                int(filled_qty), "ENTRY",
                            )
                            trade.execution_status = "PARTIAL"
                            trade.pending_execution = True
                            trade.execution_error = (
                                f"{status_text} after filling {int(filled_qty)}; "
                                "position is live at the broker"
                            )
                            logger.critical(
                                f"[PARTIAL FILL ON {status_text.upper()}] {trade.id} "
                                f"{trade.trading_symbol}: {int(filled_qty)} filled and "
                                "live. Arming protection instead of closing the row."
                            )
                            try:
                                self.ensure_broker_protection(trade, int(filled_qty))
                            except Exception as exc:
                                logger.error(
                                    f"Protection failed for partial fill on {trade.id}: {exc}"
                                )
                                self._mark_recovery(
                                    trade,
                                    f"{int(filled_qty)} filled on a {status_text} order "
                                    "and the protective stop could not be armed.",
                                )
                            changed = True
                            continue
                        trade.status = "CLOSED"
                        trade.pending_execution = False
                        trade.execution_status = "REJECTED"
                        trade.execution_error = (
                            str((order_status or {}).get("text") or status_text)
                        )
                        trade.exit_reason = "BROKER_REJECTED"
                        trade.exit_time = datetime.now(IST)
                        trade.exit_price = trade.entry_price
                        with self._lock:
                            if self.open_trades.pop(trade_id, None) is trade:
                                self.closed_trades.append(trade)
                        self._notify_delayed_entry_result(trade, False)
                        changed = True
                        continue

                    from config.settings import get_settings
                    settings = get_settings()
                    timeout = max(
                        2.0,
                        float(getattr(settings, "broker_fill_confirmation_timeout_seconds", 12.0) or 12.0),
                    )
                    working = status_text in {
                        "open", "pending", "put order req received", "validation pending",
                        "after market order req received", "modify pending",
                    }
                    if (
                        working
                        and age_seconds >= timeout
                        and trade.entry_replace_count < 1
                        and _setting_enabled(getattr(settings, "broker_cancel_replace_enabled", True), True)
                        # Reconcile runs from the PositionSync daemon every ~5s in EVERY
                        # mode; this branch cancels a resting entry and MARKET-replaces
                        # it. The read half above must keep running, so the gate is here
                        # rather than at the top of the pass.
                        and not self._live_account_send_blocked()
                    ):
                        # PROVE the cancel, do not just acknowledge it. cancel_order
                        # returns True on the API acknowledgement alone, and this order is
                        # being replaced precisely BECAUSE price is moving toward its
                        # limit — so the fill routinely wins the race against the cancel.
                        # Acting on the ack then leaves the original filled AND a MARKET
                        # replacement live: double the intended size, with the stop later
                        # armed for the doubled quantity, so one stop-out costs twice the
                        # per-trade risk the sizer approved.
                        #
                        # cancel_order_confirmed returns True only for a broker-confirmed
                        # terminal non-fill, False when the order actually filled, and
                        # None when it cannot tell. It has existed unused since the
                        # AngelOne era with exactly this docstring.
                        confirm = getattr(self.broker, "cancel_order_confirmed", None)
                        if callable(confirm):
                            try:
                                cancelled = confirm(trade.broker_order_id, "NORMAL")
                            except (BrokerOrderRejected, PermissionError) as exc:
                                logger.error(
                                    f"Stale-entry cancel refused for {trade.trading_symbol}: {exc}"
                                )
                                cancelled = None
                        else:
                            # Guarded like its sibling above. Only a double without
                            # cancel_order_confirmed reaches this, but an escape here
                            # unwinds out of the reconcile loop and abandons every
                            # remaining trade in the snapshot — the same shape that took
                            # out the force exit.
                            try:
                                cancelled = self.broker.cancel_order(
                                    trade.broker_order_id, "NORMAL"
                                )
                            except (BrokerOrderRejected, PermissionError) as exc:
                                logger.error(
                                    f"Stale-entry cancel refused for {trade.trading_symbol}: {exc}"
                                )
                                cancelled = None
                        if cancelled is True and not self._stale_entry_still_worth_filling(trade):
                            # The cancel is PROVEN, so nothing is resting and nothing
                            # will fill. Replacing it with a MARKET order here would
                            # guarantee entry into a setup that has already failed.
                            self._abandon_invalidated_entry(trade)
                            changed = True
                            continue
                        if cancelled is True:
                            entry_side = (
                                "BUY"
                                if trade.inst_type == "OPT" or trade.direction in {"LONG", "BUY"}
                                else "SELL"
                            )
                            # The original limit is now proven cancelled, so this is a
                            # fresh broker mutation rather than a harmless bookkeeping
                            # retry. Re-ask the queued-entry authority immediately before
                            # sending the MARKET replacement. A refusal can be closed
                            # cleanly because the cancel proved that no original order is
                            # still resting or filled.
                            setattr(
                                trade,
                                "_entry_policy_phase",
                                "stale_entry_replacement_pre_submit",
                            )
                            try:
                                allowed, guard_reason = self._entry_submission_guard_verdict(
                                    trade, getattr(trade, "_entry_submission_guard", None)
                                )
                            finally:
                                try:
                                    delattr(trade, "_entry_policy_phase")
                                except AttributeError:
                                    pass
                            if not allowed:
                                trade.status = "CLOSED"
                                trade.pending_execution = False
                                trade.execution_status = "REJECTED"
                                trade.exit_reason = "ENTRY_SUBMISSION_GUARD"
                                trade.exit_time = datetime.now(IST)
                                trade.exit_price = trade.entry_price
                                trade.execution_error = str(guard_reason)
                                with self._lock:
                                    if self.open_trades.pop(trade.id, None) is trade:
                                        self.closed_trades.append(trade)
                                logger.warning(
                                    f"Stale-entry replacement rejected by submission guard "
                                    f"for {trade.trading_symbol}: {guard_reason}"
                                )
                                self._notify_delayed_entry_result(trade, False)
                                changed = True
                                continue
                            try:
                                replacement_touch_kwargs: Dict[str, Any] = {}
                                replacement_touch = getattr(
                                    trade,
                                    "_entry_policy_submission_touch_quote",
                                    None,
                                )
                                if (
                                    isinstance(replacement_touch, dict)
                                    and self._accepts_kwarg(
                                        self.broker.place_order, "touch_quote"
                                    )
                                ):
                                    replacement_touch_kwargs["touch_quote"] = replacement_touch
                                replacement = self.broker.place_order(
                                    trade.trading_symbol,
                                    trade.symbol_token,
                                    trade.quantity,
                                    entry_side,
                                    self.product_type,
                                    price=0.0,
                                    # The one order path that went out untagged. Every
                                    # other leg carries one -- protective, ADD, EXIT --
                                    # and the entry tag is written inside the broker.
                                    # A replacement is the leg MOST likely to need it:
                                    # it exists because the first acknowledgement was
                                    # already unreliable, and without a tag a lost ack
                                    # on the replacement can only be resolved by
                                    # contract shape, which cannot tell two trades on
                                    # one strike apart. attempt distinguishes it from
                                    # the original entry's tag.
                                    **replacement_touch_kwargs,
                                    **self._tag_kwargs(
                                        self.broker.place_order,
                                        trade,
                                        "ENTRY",
                                        attempt=int(
                                            getattr(trade, "entry_replace_count", 0) or 0
                                        )
                                        + 1,
                                    ),
                                )
                            except (BrokerOrderAmbiguous, BrokerOrderRejected, PermissionError) as exc:
                                # Unguarded, this unwound out of reconcile and abandoned
                                # every remaining trade in the snapshot.
                                logger.error(
                                    f"Stale-entry replacement failed for {trade.trading_symbol}: {exc}"
                                )
                                replacement = None
                            if replacement:
                                trade.broker_order_id = str(replacement)
                                trade.entry_submitted_at = datetime.now(IST)
                                trade.entry_replace_count += 1
                                trade.execution_error = "Stale limit cancelled and replaced with MARKET"
                                changed = True
                                continue
                        if cancelled is False:
                            # The order FILLED while we were cancelling. Nothing is
                            # stale and nothing needs replacing — let the next reconcile
                            # confirm the fill normally.
                            trade.pending_execution = True
                            changed = True
                            continue
                        self._mark_recovery(trade, "Stale entry could not be safely cancelled/replaced")
                        trade.pending_execution = True
                        changed = True
                    elif age_seconds >= timeout * 2.0:
                        self._mark_recovery(
                            trade,
                            f"Submitted entry was not confirmed by order book or positions within {timeout * 2.0:.0f}s",
                        )
                        trade.pending_execution = True
                        changed = True
                continue
            net_qty = self._position_quantity(position)
            # This trade's SHARE of the contract's position, not the whole row.
            allocated_qty = broker_allocation.get(str(trade_id))
            if allocated_qty is None:
                # Unreachable as written -- reaching here means broker_pos held this
                # symbol, so its group was allocated and this id is in the map. But the
                # fallback must not be abs(net_qty): that is precisely the defect this
                # split exists to remove, and a silent default that restores it is how
                # it would come back. This trade's own intended size, capped by what the
                # row actually holds, is the safe answer; 0 is safer still, because
                # every unfunded path in this class defers rather than acting.
                allocated_qty = min(
                    abs(net_qty), max(0, int(getattr(trade, "quantity", 0) or 0))
                )
            expected_sign = self._expected_position_sign(trade)
            if net_qty and (1 if net_qty > 0 else -1) != expected_sign:
                trade.broker_quantity = allocated_qty
                trade.pending_execution = True
                self._mark_recovery(
                    trade,
                    f"Broker position direction mismatch: netqty={net_qty}, "
                    f"expected {'LONG' if expected_sign > 0 else 'SHORT'}",
                )
                changed = True
                continue
            # A position row IS the reconciliation an ambiguous flatten was waiting for:
            # the broker has told us the real net quantity, so automatic exits may resume.
            latched = getattr(self, "_flatten_ambiguous_latched", None)
            if latched:
                latched.discard(trade_id)
            trade.broker_quantity = allocated_qty
            expected_qty = max(1, trade.quantity)
            if net_qty == 0:
                protective_fill = self._adopt_completed_protective_fill(trade)
                if trade.protective_order_id:
                    # This call used to be bare, and both halves of that were wrong.
                    #
                    # Unguarded: cancel_order takes the account lease and RAISES when a
                    # second instance holds it, and there is no per-trade try in this
                    # loop — so one throttled contract aborted the whole reconcile pass,
                    # skipping every later trade AND the orphan-detection block below,
                    # then did it again on every 5s tick for as long as the lease was held.
                    #
                    # Result discarded: cancel_order returns False on refusal, and the
                    # next line cleared the id and closed the row anyway. The SL-M is then
                    # owned by nobody — _trade_owning_protective_order and
                    # _unowned_protective_stop_blocks_flatten only scan open_trades — so
                    # when the premium later ticks through the trigger it SELLS against a
                    # flat book: a naked written option in a product the broker will not
                    # square off.
                    stop_id = str(trade.protective_order_id)
                    cleared = False
                    try:
                        cleared = bool(
                            not self._live_account_send_blocked()
                            and self.broker.cancel_order(stop_id, "STOPLOSS")
                        )
                    except (BrokerOrderRejected, PermissionError) as exc:
                        logger.error(
                            f"Protective stop {stop_id} cancel refused while closing "
                            f"{trade.trading_symbol}: {exc}"
                        )
                    if cleared:
                        trade.protective_order_id = ""
                    else:
                        self._alert_unowned_protective_stop(trade, stop_id)
                # A GTT that survives this close fires later against a flat book with the
                # same outcome. The flatten path releases it; this one never did.
                if str(getattr(trade, "gtt_trigger_id", "") or ""):
                    try:
                        self._release_gtt_fallback(trade, "closing a flat position")
                    except Exception as exc:
                        logger.error(
                            f"GTT release failed while closing {trade.trading_symbol}: {exc}"
                        )
                reason = (
                    trade.exit_reason
                    if trade.exit_reason
                    and (
                        trade.execution_status == "EXIT_PENDING"
                        or str(trade.exit_reason).upper() == "STOP_HIT"
                    )
                    else "MANUAL_SYNC"
                )
                realised = position.get("realised")
                try:
                    realised_total = float(realised) if realised not in (None, "") else None
                except (TypeError, ValueError):
                    realised_total = None
                # Kite reports `realised` CUMULATIVELY per contract for the day, not per
                # trade (2026-08-16). Booking it verbatim meant a second round trip on
                # the same strike inherited the first trade's P&L on top of its own, and
                # this figure feeds the daily-loss breaker. Book the DELTA against what
                # this contract has already had attributed today.
                realised_pnl = None
                booked = None
                key = ""
                if realised_total is not None:
                    booked = getattr(self, "_realised_booked_today", None)
                    if booked is None:
                        booked = self._realised_booked_today = {}
                    key = str(getattr(trade, "trading_symbol", "") or "")
                    realised_pnl = realised_total - float(booked.get(key, 0.0) or 0.0)
                if self._finalize_closed_trade(
                    trade_id,
                    trade,
                    trade.exit_price if protective_fill else self._reconciled_exit_price(trade, position),
                    reason,
                    realised_pnl=realised_pnl,
                    broker_realised_cumulative=realised_total,
                ):
                    # Stamp the baseline ONLY after a close that actually happened. The
                    # first cut of this advanced it BEFORE the call, and
                    # _finalize_closed_trade returns False when the trade has already
                    # left open_trades (line 1281) — so on that path the baseline moved
                    # for a trade that never closed and the NEXT close on the same
                    # contract booked a zero delta, losing its P&L entirely. That is
                    # worse than the cumulative-double-count this delta exists to fix.
                    if booked is not None and key:
                        booked[key] = realised_total
                    closed += 1
                continue
            if (
                bool(getattr(trade, "pending_exit", False))
                and str(getattr(trade, "execution_status", "") or "").upper()
                == "RECOVERY_REQUIRED"
            ):
                # A lost exit acknowledgement is neither permission to submit another
                # MARKET order nor a reason to abandon a known live position forever.
                # First reconcile the exact tags. Only a broker-confirmed
                # absence proves that the original exit never reached the exchange.
                recovered_exit_id = self._recover_order_id_from_tag(trade, "FULL_EXIT") or self._recover_order_id_from_tag(trade, "RECOVERY_EXIT")
                if recovered_exit_id:
                    trade.exit_order_id = recovered_exit_id
                    trade.execution_status = "EXIT_PENDING"
                    trade.execution_error = "Recovered ambiguous exit from broker order tag"
                    changed = True
                else:
                    tagged_state = self._tagged_exit_state(trade, "FULL_EXIT")
                    recovery_state = self._tagged_exit_state(trade, "RECOVERY_EXIT")
                    if tagged_state == "ABSENT" and recovery_state == "ABSENT":
                        trade.pending_exit = False
                        trade.execution_status = "OPEN"
                        trade.execution_error = (
                            "Ambiguous exit was absent from authoritative broker order book; "
                            "submitting one recovery exit"
                        )
                        changed = True
                        # The position snapshot above proves this trade still owns a
                        # quantity, and the tag lookup proves no original full exit is
                        # resting. emergency_flatten_trade uses a distinct recovery tag
                        # and retains its normal protection/cancel safeguards.
                        if not self.emergency_flatten_trade(
                            trade, "AMBIGUOUS_EXIT_ORDER_ABSENT"
                        ):
                            self.ensure_broker_protection(trade, allocated_qty)
                        continue
                    # FOUND / AMBIGUOUS / UNKNOWN are intentionally held. Any of them
                    # could still represent a live broker exit; retrying would over-sell.
                    continue
            if (
                trade.execution_status in {
                    "ENTRY_PENDING",
                    "ENTRY_SUBMITTED",
                    "PARTIAL",
                    "RECOVERY_REQUIRED",
                }
                and not trade.pending_exit
            ):
                # THIS trade's share of the row, not the contract's total. The aggregate
                # here did more than mis-size the stop below: filled_qty also decides
                # pending_execution and whether the row is OPEN or PARTIAL, so three
                # fully-filled trades on one strike each saw 195 against an expected 65
                # and were all marked PARTIAL.
                filled_qty = allocated_qty
                trade.pending_execution = filled_qty != expected_qty
                trade.execution_status = "OPEN" if filled_qty == expected_qty else "PARTIAL"
                
                # ACTUAL BROKER FILL PRICE SYNC
                # Prefer this order's own executions over the position average: the
                # position blends every trade on the contract, so a second entry on the
                # same strike silently rewrites the first one's entry price. Only a
                # FILLED answer is used — UNKNOWN falls through to the position figure
                # rather than being treated as "no fill".
                fill_price = 0.0
                fill_geometry_verified = False
                shadow_fill_quantity = filled_qty
                filler = getattr(self.broker, "get_fill", None)
                if callable(filler) and getattr(trade, "broker_order_id", ""):
                    try:
                        detail = filler(str(trade.broker_order_id))
                        if str(detail.get("state") or "").upper() == "FILLED":
                            fill_price = float(detail.get("average_price") or 0.0)
                            # The same lookup states this ORDER's own filled quantity,
                            # which was being fetched and thrown away. It is the
                            # authoritative entry figure -- better than the position
                            # aggregate and better than the allocation derived from it --
                            # and it costs no extra call.
                            try:
                                _order_qty = max(0, int(float(detail.get("quantity") or 0)))
                            except (TypeError, ValueError):
                                _order_qty = 0
                            if _order_qty:
                                shadow_fill_quantity = _order_qty
                                self._record_fill(
                                    trade, str(trade.broker_order_id), _order_qty, "ENTRY"
                                )
                    except Exception as exc:
                        logger.debug(f"Order-level fill lookup failed for {trade.id}: {exc}")
                if fill_price <= 0.1:
                    fill_price = float(position.get("buyavgprice", 0.0) if trade.direction in {"LONG", "BUY"} or trade.inst_type == "OPT" else position.get("sellavgprice", 0.0))
                if fill_price > 0.1:
                    if trade.inst_type != "OPT" or self._option_premium_quote_plausible(
                        trade,
                        fill_price,
                        getattr(trade, "entry_spot", 0.0),
                    ):
                        self._record_entry_slippage(trade, fill_price)
                        # Before the overwrite below: entry_price is about to become the
                        # fill, and signal_entry_price is not persisted, so this is the
                        # last moment the requested price exists anywhere.
                        self._stamp_fill_quality(
                            trade,
                            leg="ENTRY",
                            requested=float(getattr(trade, "signal_entry_price", 0.0) or 0.0)
                            or float(trade.entry_price or 0.0),
                        )
                        trade.entry_price = fill_price
                        fill_geometry_verified = True
                        shadow_fill_observer = getattr(
                            getattr(self, "scanner", None),
                            "_update_accepted_entry_shadow_fill",
                            None,
                        )
                        if callable(shadow_fill_observer):
                            try:
                                shadow_fill_observer(
                                    trade,
                                    fill_price,
                                    shadow_fill_quantity,
                                )
                            except Exception as exc:
                                logger.debug(f"Accepted entry shadow fill update skipped: {exc}")
                    else:
                        logger.warning(
                            f"Ignored implausible broker fill {fill_price:.2f} "
                            f"for {trade.trading_symbol or trade.instrument}; "
                            f"kept local entry {trade.entry_price:.2f}"
                        )
                
                trade.execution_error = (
                    ""
                    if filled_qty == expected_qty
                    else f"Broker entry partially filled; {filled_qty} of {expected_qty} quantity confirmed"
                )
                confirmed += 1
                changed = True
                setattr(
                    trade,
                    "_entry_policy_actual_fill_verified",
                    fill_geometry_verified,
                )
                try:
                    policy_ok, policy_reason = self._actual_fill_entry_policy_verdict(trade)
                finally:
                    try:
                        delattr(trade, "_entry_policy_actual_fill_verified")
                    except AttributeError:
                        pass
                if not policy_ok:
                    reason = (
                        "Actual fill violates immutable final entry policy: "
                        f"{policy_reason}"
                    )
                    logger.error(f"{reason} ({trade.trading_symbol or trade.instrument})")
                    if not self.emergency_flatten_trade(trade, reason):
                        # If a flatten cannot be submitted, protection is still safer
                        # than leaving the now-known position naked while recovery runs.
                        if not self.ensure_broker_protection(trade, filled_qty):
                            self._mark_recovery(trade, reason)
                    continue
                protected = self.ensure_broker_protection(trade, filled_qty)
                if not protected:
                    self._maybe_auto_flatten_unprotected(
                        trade,
                        "Broker protection unavailable",
                    )
                if filled_qty < expected_qty:
                    self._withdraw_stale_entry_remainder(trade, filled_qty, expected_qty)
            elif (
                trade.execution_status in {"EXIT_PENDING", "PARTIAL_EXIT"}
                and allocated_qty >= expected_qty
            ):
                # The position has not changed, but its owned exit can still have a
                # definitive terminal non-fill.  Previously this full-size case was
                # examined only when quantity had already fallen (the partial-exit
                # branch below), so a CANCELLED/REJECTED recovered exit remained
                # pending forever. COMPLETE deliberately remains non-retryable: a
                # delayed positions snapshot can still be stale after a real fill.
                verdict = self._abandoned_exit_verdict(trade)
                if verdict == "DEAD":
                    trade.pending_exit = False
                    trade.exit_order_id = ""
                    trade.execution_status = "OPEN"
                    trade.execution_error = (
                        "Broker confirmed the pending exit was cancelled/rejected; "
                        "submitting recovery exit for the reconciled remaining quantity"
                    )
                    changed = True
                    if not self.emergency_flatten_trade(
                        trade, "EXIT_ORDER_CONFIRMED_DEAD"
                    ):
                        self.ensure_broker_protection(trade, allocated_qty)
                    continue
                if verdict == "UNKNOWN":
                    if not hasattr(self, "_full_exit_unknown_logged"):
                        self._full_exit_unknown_logged = set()
                    if trade.id not in self._full_exit_unknown_logged:
                        self._full_exit_unknown_logged.add(trade.id)
                        logger.warning(
                            f"Cannot determine whether full exit {trade.exit_order_id} for "
                            f"{trade.trading_symbol} is still working; holding the position "
                            "to avoid a duplicate exit."
                        )
            elif (
                trade.execution_status in {"EXIT_PENDING", "PARTIAL_EXIT"}
                and allocated_qty < expected_qty
            ):
                trade.pending_exit = True
                trade.execution_status = "PARTIAL_EXIT"
                trade.execution_error = (
                    f"Broker exit partially filled; {allocated_qty} of "
                    f"{expected_qty} quantity remains; awaiting broker completion"
                )
                changed = True
                # PARTIAL_EXIT used to be re-examined by NOTHING until the 15:25 force
                # exit: EXIT_PENDING was the only status this branch accepted, and the
                # branch below excludes PARTIAL_EXIT explicitly. So the residual sat for
                # the rest of the session with its protective stop already cancelled
                # (cancelling it is what precedes sending an exit at all).
                #
                # "Awaiting broker completion" is the right posture only while the exit
                # order can still fill the rest. Once the broker says that order is
                # CANCELLED or REJECTED, nothing is coming: the remainder is a live
                # position with no stop, no working exit, and nothing scheduled to touch
                # it before the close. That is the moment to square it off, and it is a
                # fact the broker can state rather than something to guess at.
                verdict = self._abandoned_exit_verdict(trade)
                if verdict == "DEAD":
                    self._resume_abandoned_partial_exit(trade, allocated_qty)
                elif verdict == "UNKNOWN":
                    if not hasattr(self, "_partial_exit_unknown_logged"):
                        self._partial_exit_unknown_logged = set()
                    if trade.id not in self._partial_exit_unknown_logged:
                        self._partial_exit_unknown_logged.add(trade.id)
                        logger.warning(
                            f"Cannot determine whether the exit order for "
                            f"{trade.trading_symbol} is still working; leaving "
                            f"{allocated_qty} quantity in place rather than risking a "
                            "double exit. The force-exit remains the backstop."
                        )
            elif (
                allocated_qty != expected_qty
                and trade.execution_status not in {"EXIT_PENDING", "PARTIAL_EXIT"}
            ):
                trade.execution_status = "PARTIAL"
                self.ensure_broker_protection(trade, allocated_qty)
                changed = True

        orphans = dict(foreign)
        for symbol, position in broker_pos.items():
            if self._position_quantity(position) != 0 and symbol not in local_symbols:
                orphan = dict(position)
                orphan["execution_status"] = "ORPHANED"
                orphan["status"] = "ORPHANED"
                orphans[symbol] = orphan
        def exposure(positions):
            return {symbol: (self._position_quantity(row), row.get("symboltoken"),
                             row.get("producttype") or row.get("product"))
                    for symbol, row in positions.items()}

        with self._lock:
            if self.broker is not snapshot_broker:
                return {"checked": False, "orphans": 0, "closed": closed, "confirmed": confirmed}
            changed_exposure = exposure(orphans) != exposure(self.orphan_positions)
            self.orphan_positions = orphans
            if self.broker is snapshot_broker and self._live_contract_risk_required():
                self._live_position_snapshot = (snapshot_broker, time.monotonic())
        if changed_exposure and orphans:
            logger.critical(
                "Broker-only positions require reconciliation. New entries blocked: "
                + ", ".join(sorted(orphans))
            )
        elif changed_exposure:
            logger.info("Broker-only position reconciliation cleared")
        if changed:
            self._mark_dirty()
        return {
            "checked": True,
            "orphans": len(orphans),
            "closed": closed,
            "confirmed": confirmed,
        }

    def update_risk_settings(self, futures_sl_pct: float = None, options_sl_pct: float = None):
        """Update settings and re-calibrate open trades if needed"""
        if futures_sl_pct is not None:
            self.futures_sl_pct = futures_sl_pct
        if options_sl_pct is not None:
            self.options_sl_pct = options_sl_pct
            
        # Optional: Sync open trades to new settings (Aggressive approach)
        # For now, we just update the manager state so NEW trades get the correct SL.
        # If we want to move EXISTING stops, we would loop through self.open_trades here.
        logger.info(f"🛡️ Risk Settings Synced: FUT SL={self.futures_sl_pct}%, OPT SL={self.options_sl_pct}%")

    def save_state(self) -> bool:
        """Save active and recent closed trades to SQLite for crash recovery"""
        try:
            # Serialize checkpoints in snapshot order, but let price/exit updates
            # acquire the trade book while SQLite performs its transaction.
            save_lock = self.__dict__.setdefault("_save_state_lock", threading.RLock())
            with save_lock:
                with self._lock:
                    open_tr = list(self.open_trades.values())
                    self.closed_trades = self.closed_trades[-500:]
                    closed = list(self.closed_trades)
                    snapshot = copy.deepcopy([t.to_dict() for t in open_tr + closed])
                db.save_trades(snapshot)
        except Exception as e:
            self._mark_dirty()
            logger.error(f"Failed to save trade state to DB: {e}")
            raise
        return True

    @staticmethod
    def _realised_baseline_from_closed_trades(
        closed_trades: List[Trade], session_date: Optional[date] = None
    ) -> Dict[str, float]:
        """Rebuild Kite's consumed cumulative-realised totals for one session.

        New rows carry the exact cumulative value observed after the close. For rows
        written before that field existed, net P&L plus stored charges reconstructs the
        gross broker delta. An exact observation supersedes earlier inference because its
        cumulative total already includes those closes; later legacy rows advance it.
        """
        target_date = session_date or datetime.now(IST).date()
        events: Dict[str, List[Tuple[datetime, Optional[float], float]]] = {}

        for trade in closed_trades or []:
            symbol = str(getattr(trade, "trading_symbol", "") or "").strip()
            exited = getattr(trade, "exit_time", None)
            if not symbol or exited is None or bool(getattr(trade, "is_ghost", False)):
                continue
            if isinstance(exited, str):
                try:
                    exited = datetime.fromisoformat(exited)
                except (TypeError, ValueError):
                    continue
            if not isinstance(exited, datetime):
                continue
            exited_local = IST.localize(exited) if exited.tzinfo is None else exited.astimezone(IST)
            if exited_local.date() != target_date:
                continue

            cumulative = getattr(trade, "broker_realised_cumulative", None)
            if cumulative is not None:
                try:
                    measured = float(cumulative)
                except (TypeError, ValueError):
                    continue
                events.setdefault(symbol, []).append((exited_local, measured, 0.0))
                continue

            # Backward compatibility for same-day rows created before the explicit
            # cumulative field was deployed. Closed trade.pnl is net of charges.
            try:
                gross_delta = float(getattr(trade, "pnl", 0.0) or 0.0) + float(
                    getattr(trade, "charges", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                continue
            events.setdefault(symbol, []).append((exited_local, None, gross_delta))

        restored: Dict[str, float] = {}
        for symbol, rows in events.items():
            consumed = 0.0
            for _, measured, inferred_delta in sorted(rows, key=lambda row: row[0]):
                # An exact cumulative observation supersedes every earlier inference;
                # a later legacy/missing-realised row must still advance that baseline.
                consumed = measured if measured is not None else consumed + inferred_delta
            restored[symbol] = consumed
        return restored

    @classmethod
    def _drop_delisted_open_records(cls, records):
        """Refuse only the known persisted test-fixture signature.

        2026-08-19: four rows sat OPEN across six restarts and produced a dashboard
        reading of DAILY P&L Rs.199,081 on 4 trades with W/L 0/0 —

            T-MIDCPNIFTY  MIDCPNIFTY25AUG12800CE   88.15 -> mark 1118.94  +Rs.123,180
            T-BANKNIFTY   BANKNIFTY25AUG52000CE   213.30 -> mark 2795.64  + Rs.89,986
            T-SENSEX      SENSEX25AUG81000CE      301.60 -> mark    0.05  -  Rs.6,129
            T-NIFTY       NIFTY25AUG24500CE       104.75 -> mark    0.05  -  Rs.7,956

        Fixtures, not trades: ids T-<INDEX>, broker ids O-<INDEX>, symbol_token '1' on
        all four, all written inside 17ms at 01:57:35, no provenance. The live series
        was 26AUG; these are 25AUG. Marked to today's spot, an expired deep-ITM call
        invents six figures.

        Deleting the rows does not hold — this process re-persists them from memory
        within seconds, with byte-identical entry_filled_at stamps. They have to be
        refused at LOAD.

        Current-listing absence is necessary but not sufficient. A carried position can
        temporarily be absent from a stale/incomplete instrument dump, and the broker's
        position book is authoritative for exposure. Therefore a normal or legacy OPEN
        row must survive for reconciliation even when its symbol is absent. The only rows
        removed here are the four recorded fixtures whose synthetic id/order-id/token and
        empty provenance jointly prove that they were never broker positions.

        This lives here rather than in db.load_open_trades because that is a plain
        accessor: "save a row then read it back" has to keep meaning exactly that.

        Fails OPEN throughout — no listing, an unreadable or truncated one, a row with no
        symbol, or any row without the complete fixture fingerprint is kept.
        """
        try:
            tradable = db.__class__._tradable_symbols()
        except Exception as exc:
            logger.warning(f"Delisted-contract check skipped: {exc}")
            return records
        if not tradable:
            return records

        live = []
        for record in records:
            symbol = str((record or {}).get("trading_symbol") or "").strip()
            trade_id = str((record or {}).get("id") or "").strip().upper()
            fixture_index = trade_id[2:] if trade_id.startswith("T-") else ""
            known_fixture = (
                fixture_index in {"NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"}
                and str((record or {}).get("broker_order_id") or "").strip().upper()
                == f"O-{fixture_index}"
                and str((record or {}).get("symbol_token") or "").strip() == "1"
                and not str((record or {}).get("data_provenance") or "").strip()
            )
            if symbol and symbol not in tradable and known_fixture:
                logger.warning(
                    f"Refusing to restore known test fixture {(record or {}).get('id')} "
                    f"{symbol}: its synthetic order/token fingerprint is not broker "
                    f"exposure and its Rs.{float((record or {}).get('pnl') or 0.0):,.0f} "
                    "would otherwise be marked to live spot and counted in the total."
                )
                continue
            if symbol and symbol not in tradable and not cls._unfilled_replay_record(record):
                logger.critical(
                    f"Retaining unlisted OPEN recovery row {(record or {}).get('id')} "
                    f"{symbol} until authoritative broker reconciliation; listing "
                    "absence alone does not prove the position is flat."
                )
            live.append(record)
        return live

    @staticmethod
    def _unfilled_replay_record(row):
        if not str(row.get("id") or "").startswith(("H_", "EOD_")):
            return False
        if not str(row.get("data_provenance") or "").upper().startswith(("BACKTEST", "ENGINE_REPLAY")):
            return False
        evidence = ("broker_order_id", "exit_order_id", "protective_stop_order_id",
                    "protective_order_id", "entry_filled_at", "exit_filled_at")
        if any(str(row.get(key) or "").strip() for key in evidence):
            return False
        try:
            return float(row.get("broker_quantity") or 0) == 0
        except (TypeError, ValueError):
            return False

    def load_state(self):
        """Load state from SQLite on startup"""
        self._trade_state_loaded = False
        try:
            self._realised_booked_today = {}
            stored_open_records = db.load_open_trades()
            open_records = self._drop_delisted_open_records(stored_open_records)
            kept_ids = {str((row or {}).get("id") or "") for row in open_records}
            quarantined_ids = [
                str((row or {}).get("id") or "")
                for row in stored_open_records
                if str((row or {}).get("id") or "") not in kept_ids
            ]
            if quarantined_ids:
                try:
                    changed = db.quarantine_open_trades(
                        quarantined_ids,
                        "Known synthetic fixture fingerprint; contract absent from current instrument listing",
                    )
                    logger.warning(
                        f"Quarantined {changed} invalid OPEN recovery row(s); they cannot resurrect on restart."
                    )
                except Exception as exc:
                    # Keep the proven-invalid rows out of this process even if the DB
                    # update fails, but make the next-restart hazard impossible to miss.
                    logger.critical(
                        f"Failed to quarantine invalid OPEN recovery rows {quarantined_ids}: {exc}"
                    )
            closed_records = db.load_closed_trades()
            
            stale_ghosts = 0
            auto_closed_missing_contracts = 0
            self._restored_replay_open_rows = {}
            for data in open_records:
                tid = data.get("id", "")
                if not tid: continue
                # A persisted engine replay is not a broker position. Keep its
                # original record for audit without inventing an exit or submitting
                # reconciliation/flatten requests. Any broker evidence wins.
                if self._unfilled_replay_record(data):
                    self._restored_replay_open_rows[tid] = dict(data)
                    continue
                t = Trade.from_dict(data)
                t.status = "OPEN"
                # A source candle that has already closed cannot repaint, so the guard is
                # satisfied by definition. Rows written before repaint_guard_checked was
                # persisted carry False, and the async repaint tracker is started from
                # exactly one place — the entry acknowledgement — with no rehydration
                # path, so without this a restarted trade would sit "structurally
                # unconfirmed" for the rest of the session: let-winners-run off and the
                # blunt sub-1R give-back guard armed on a position whose candle closed
                # hours ago.
                if not t.repaint_guard_checked and t.entered_intracandle:
                    try:
                        minutes = {"1min": 1, "5min": 5, "15min": 15}.get(
                            str(getattr(t, "timeframe", "") or "").lower(), 5
                        )
                        entry_time = t.entry_time
                        if entry_time.tzinfo is None:
                            entry_time = IST.localize(entry_time)
                        if datetime.now(IST) >= entry_time + timedelta(minutes=minutes):
                            t.repaint_guard_checked = True
                    except (AttributeError, TypeError, ValueError):
                        pass
                symbol = str(getattr(t, "trading_symbol", "") or "").strip()
                token = str(getattr(t, "symbol_token", "") or "").strip()
                broker_order_id = str(getattr(t, "broker_order_id", "") or "").strip()
                unusable_contract = (
                    not bool(getattr(t, "is_ghost", False))
                    and not broker_order_id
                    and (not symbol or not token)
                )
                if t.entry_time.date() < datetime.now(IST).date():
                    # Ghost/shadow rows (engine-lab, tests) never had broker
                    # orders — there is nothing to reconcile. Retaining them as
                    # RECOVERY_REQUIRED spammed CRITICAL alerts + strategy-exit
                    # loops on every restart (2026-07-10: 111 stale ghosts).
                    if bool(getattr(t, "is_ghost", False)):
                        t.status = "CLOSED"
                        t.pending_exit = False
                        t.execution_status = "CLOSED"
                        t.exit_reason = getattr(t, "exit_reason", "") or "STALE_SESSION_END"
                        t.exit_time = getattr(t, "exit_time", None) or t.entry_time
                        t.exit_price = float(
                            getattr(t, "exit_price", 0.0)
                            or getattr(t, "current_price", 0.0)
                            or getattr(t, "entry_price", 0.0)
                            or 0.0
                        )
                        self.closed_trades.append(t)
                        stale_ghosts += 1
                        continue
                    t.execution_status = "RECOVERY_REQUIRED"
                    t.execution_error = "Previous-session trade requires broker reconciliation"
                    logger.critical(f"Previous-session active trade retained for reconciliation: {tid}")
                elif unusable_contract:
                    # Same-session row the broker never acknowledged: no order id AND no
                    # contract to act on, so there is no position to reconcile and nothing
                    # emergency_flatten_trade() could ever submit. Left open it becomes the
                    # 2026-07-31 EOD loop — FORCE_EOD_KILL retrying "emergency flatten
                    # unavailable" every cycle until the session ends.
                    #
                    # Deliberately restricted to same-session rows. A trade that survived a
                    # day boundary keeps the reconciliation path above even when its
                    # metadata looks unusable, because discarding a position that may be
                    # live at the broker is far worse than one noisy alert.
                    t.status = "CLOSED"
                    t.pending_exit = False
                    t.pending_execution = False
                    t.execution_status = "REJECTED"
                    t.execution_error = (
                        "Trade discarded: broker contract metadata missing (trading_symbol/symbol_token)"
                    )
                    t.exit_reason = "CONTRACT_MISSING"
                    t.exit_time = datetime.now(IST)
                    t.exit_price = float(getattr(t, "entry_price", 0.0) or 0.0)
                    self.closed_trades.append(t)
                    auto_closed_missing_contracts += 1
                    continue
                elif not t.execution_status:
                    t.execution_status = "OPEN"
                self.open_trades[tid] = t
            if self._restored_replay_open_rows:
                logger.info(f"Retained {len(self._restored_replay_open_rows)} unfilled replay row(s) for audit; excluded from broker recovery.")
            if stale_ghosts:
                logger.info(
                    f"Closed {stale_ghosts} stale previous-session ghost trade(s) as "
                    "STALE_SESSION_END on load (no broker reconciliation applies)."
                )
                try:
                    self._mark_dirty()
                except Exception:
                    pass
            if auto_closed_missing_contracts:
                logger.warning(
                    f"Closed {auto_closed_missing_contracts} trade(s) with missing broker contract metadata on load."
                )
                try:
                    self._mark_dirty()
                except Exception:
                    pass

            for data in closed_records:
                try:
                    t = Trade.from_dict(data)
                    t.status = "CLOSED"
                    self.closed_trades.append(t)
                except Exception as e:
                    logger.warning(f"Failed to parse historical trade: {e}")

            self._realised_booked_today = self._realised_baseline_from_closed_trades(
                self.closed_trades,
                datetime.now(IST).date(),
            )
            
            self._recover_trade_counter()
            self._trade_state_loaded = True
            if self.open_trades or self.closed_trades:
                logger.success(f"♻️ Recovered {len(self.open_trades)} active and {len(self.closed_trades)} historical trades from DB.")
        except Exception:
            logger.exception("Failed to load trade state from DB. New entries blocked.")

    def _recover_trade_counter(self) -> None:
        """Continue generated trade IDs above every persisted T-number."""
        persisted = list(self.open_trades.values()) + list(self.closed_trades)
        for trade in persisted:
            match = re.match(r"^T(\d+)_", str(getattr(trade, "id", "") or ""))
            if match:
                self._trade_counter = max(self._trade_counter, int(match.group(1)))

    def reset_pnl(self):
        """Reset all trades and P&L for a fresh simulation"""
        if getattr(self, "mode", "HISTORICAL") == "REAL" and self.has_live_exposure():
            logger.critical("Blocked P&L reset while REAL broker exposure is still active.")
            return False
        try:
            backup_path = ""
            with self._lock:
                existing_rows = db.load_all_trades()
                if existing_rows:
                    backup_dir = Path("data_store/reset_backups")
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    backup_path = str(
                        backup_dir
                        / f"trades_reset_backup_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    Path(backup_path).write_text(
                        json.dumps(existing_rows, indent=2, default=str),
                        encoding="utf-8",
                    )
                with db.write_lock:
                    db.conn.execute("DELETE FROM trades")
                    db.conn.commit()
                self.open_trades.clear()
                self.closed_trades.clear()
        except Exception as exc:
            logger.error(f"Failed to clear persisted trades during P&L reset: {exc}")
            return False
        backup_note = f" Backup: {backup_path}" if backup_path else ""
        logger.info(f"🧹 Trade Manager: Cleared all trades and P&L from DB.{backup_note}")
        return True

    def has_live_exposure(self) -> bool:
        """Return True while broker-backed or broker-only positions need protection."""
        with self._lock:
            active = any(
                trade.status == "OPEN" and not getattr(trade, "is_ghost", False)
                for trade in self.open_trades.values()
            )
            return active or bool(self.orphan_positions)

    def reset_session(self, session_date: date) -> Dict[str, int]:
        """Start a new accounting session without deleting active broker state.

        Ghost/shadow trades (engine-lab, premarket HISTORICAL replays) never had
        broker orders — at rollover they are closed as STALE_SESSION_END instead
        of being retained for a "reconciliation" that can never happen
        (2026-07-10: 24 premarket ghosts raised a false CRITICAL at 09:00).
        """
        stale_ghosts: List[Trade] = []
        with self._lock:
            # Kite reports position `realised` cumulatively PER DAY, and the delta
            # baseline that turns it into a per-trade figure is only valid within one
            # session. Left standing across a rollover, day 2's first exit on a strike
            # also traded on day 1 books (today - yesterday) and is wrong by the whole
            # of yesterday's P&L on that contract.
            self._realised_booked_today = {}
            closed = len(self.closed_trades)
            for tid, trade in list(self.open_trades.items()):
                if bool(getattr(trade, "is_ghost", False)):
                    trade.status = "CLOSED"
                    trade.pending_exit = False
                    trade.execution_status = "CLOSED"
                    trade.exit_reason = getattr(trade, "exit_reason", "") or "STALE_SESSION_END"
                    trade.exit_time = getattr(trade, "exit_time", None) or datetime.now(IST)
                    trade.exit_price = float(
                        getattr(trade, "exit_price", 0.0)
                        or getattr(trade, "current_price", 0.0)
                        or getattr(trade, "entry_price", 0.0)
                        or 0.0
                    )
                    self.closed_trades.append(trade)
                    self.open_trades.pop(tid, None)
                    stale_ghosts.append(trade)
                    continue
                if trade.entry_time.date() < session_date:
                    trade.execution_status = "RECOVERY_REQUIRED"
                    trade.execution_error = "Previous-session trade requires broker reconciliation"
            active = len(self.open_trades)
        self._mark_dirty()
        if stale_ghosts:
            logger.info(
                f"Session rollover closed {len(stale_ghosts)} ghost/shadow trade(s) as "
                "STALE_SESSION_END (no broker reconciliation applies)."
            )
        if active:
            logger.critical(
                f"Session rollover retained {active} active trade(s); broker reconciliation required."
            )
        return {"active_retained": active, "closed_history_retained": closed}

    def _parse_time(self, t: str) -> dtime:
        parts = t.split(":")
        return dtime(int(parts[0]), int(parts[1]))

    def _effective_max_positions(self) -> int:
        """Base max_positions, raised during cascade multi-TF waves (R4)."""
        base = max(1, int(getattr(self, "max_positions", 12) or 12))
        try:
            from config.settings import get_settings

            s = get_settings()
            casc_cap = max(
                0, int(getattr(s, "cascade_max_concurrent_positions", 0) or 0)
            )
            if casc_cap <= base:
                return base
            # Any live cascade entry or sticky wave flag unlocks cascade ceiling
            has_cascade = any(
                bool(getattr(t, "one_way_cascade_entry", False))
                for t in (getattr(self, "open_trades", {}) or {}).values()
                if not getattr(t, "is_ghost", False)
            )
            if has_cascade or bool(getattr(self, "_cascade_wave_capacity", False)):
                return casc_cap
        except Exception:
            pass
        return base

    def arm_cascade_wave_capacity(self, enabled: bool = True) -> None:
        """Scanner arms this before opening a multi-signal cascade batch."""
        setattr(self, "_cascade_wave_capacity", bool(enabled))

    def can_open_trade(self, is_recovery: bool = False, *, native_utema: bool = False) -> Tuple[bool, str]:
        """Check if we can open a new trade"""
        with self._lock:
            if not is_recovery and not getattr(self, "_trade_state_loaded", True):
                return False, "Trade ledger restore failed"
            if not is_recovery and self.orphan_positions:
                return False, "Broker-only orphan position requires reconciliation"
            if not is_recovery and self._live_contract_risk_required():
                from config.settings import get_settings

                broker, observed = getattr(self, "_live_position_snapshot", (None, 0.0))
                settings = get_settings()
                delay = max(float(settings.position_sync_interval_seconds),
                            float(settings.position_sync_idle_interval_seconds))
                if broker is not self.broker or not 0 <= time.monotonic() - observed <= delay + 10:
                    self.nudge_position_sync()
                    return False, "A fresh execution broker position reconciliation is required"
            active_positions = sum(
                1
                for trade in self.open_trades.values()
                if not getattr(trade, "is_ghost", False)
            )
            cap = self._effective_max_positions()
            if not is_recovery and not native_utema and active_positions >= cap:
                return False, f"Max positions ({cap}) reached"
        return True, ""

    @staticmethod
    def _extract_order_id(result) -> str:
        if not result:
            return ""
        if isinstance(result, dict):
            if broker_rejection(result):
                return ""
            data = result.get("data", result)
            if isinstance(data, dict):
                return str(data.get("orderid") or data.get("uniqueorderid") or "")
            return str(data or "")
        return str(result)

    def _entry_submission_guard_verdict(
        self,
        trade: Trade,
        per_entry_guard: Optional[Callable[[Trade], Any]] = None,
        *,
        force_auto_exposure: bool = False,
    ) -> Tuple[bool, str]:
        """Evaluate the optional queued AUTO-entry guard on a strict, fail-closed API.

        A guard may return ``bool``, ``(allowed, reason)``, or a mapping with
        ``allowed``/``ok`` and ``reason``.  It is intentionally skipped for manual
        entries: an operator-approved order must not acquire a new automated veto merely
        because both paths share ``open_trade``.  Exceptions and malformed verdicts are
        clean rejections because allowing an entry after its final safety check crashed
        would defeat the purpose of placing that check at the broker boundary.
        """
        # Recovery reconstructs exposure that already exists, so its one-time
        # bookkeeping path may bypass a *new exposure* guard.  That exemption must
        # not follow the recovered Trade into a later automated scale-in: the add is
        # fresh broker exposure and callers mark it with force_auto_exposure=True.
        if (
            bool(getattr(trade, "_entry_submission_guard_bypass", False))
            and not force_auto_exposure
        ):
            return True, ""

        policy_guard = getattr(self, "fresh_entry_policy_guard", None)
        if policy_guard is not None:
            if not callable(policy_guard):
                return False, "fresh-entry policy guard is not callable"
            try:
                policy_verdict = policy_guard(trade)
            except Exception as exc:
                return False, f"fresh-entry policy guard raised {type(exc).__name__}: {exc}"
            if isinstance(policy_verdict, tuple):
                policy_allowed = bool(policy_verdict[0]) if policy_verdict else False
                policy_reason = str(policy_verdict[1] if len(policy_verdict) > 1 else "")
            elif isinstance(policy_verdict, dict):
                if "allowed" in policy_verdict:
                    policy_allowed = bool(policy_verdict.get("allowed"))
                elif "ok" in policy_verdict:
                    policy_allowed = bool(policy_verdict.get("ok"))
                else:
                    return False, "fresh-entry policy returned no allowed/ok field"
                policy_reason = str(policy_verdict.get("reason") or "")
            elif isinstance(policy_verdict, bool):
                policy_allowed, policy_reason = policy_verdict, ""
            else:
                return False, "fresh-entry policy returned an unsupported verdict"
            if not policy_allowed:
                return False, policy_reason or "fresh-entry policy rejected the submission"
        if (
            not force_auto_exposure
            and str(getattr(trade, "exec_type", "A") or "A").upper()
            not in {"A", "AUTO"}
        ):
            return True, ""
        guard = per_entry_guard
        if guard is None:
            guard = getattr(self, "entry_submission_guard", None)
        if guard is None:
            return True, ""
        if not callable(guard):
            return False, "entry submission guard is not callable"
        try:
            verdict = guard(trade)
        except Exception as exc:
            return False, f"entry submission guard raised {type(exc).__name__}: {exc}"
        reason = ""
        if isinstance(verdict, dict):
            if "allowed" in verdict:
                allowed = bool(verdict.get("allowed"))
            elif "ok" in verdict:
                allowed = bool(verdict.get("ok"))
            else:
                return False, "entry submission guard returned a mapping without allowed/ok"
            reason = str(verdict.get("reason") or verdict.get("limitation") or "")
        elif isinstance(verdict, tuple):
            if not verdict:
                return False, "entry submission guard returned an empty tuple"
            allowed = bool(verdict[0])
            reason = str(verdict[1] if len(verdict) > 1 else "")
        elif isinstance(verdict, bool):
            allowed = verdict
        else:
            return False, (
                "entry submission guard returned an unsupported verdict "
                f"({type(verdict).__name__})"
            )
        if allowed:
            return True, reason
        return False, reason or "entry submission guard rejected the queued entry"

    def _cap_option_entry_stop(self, trade: Trade) -> None:
        """Enforce the option premium ceiling at new-entry and actual-fill boundaries.

        Existing recovered exposure is not repriced or given a retroactive stop.
        This only tightens a stop; it never changes broker quantity or sends an order.
        """
        if trade.inst_type != "OPT" or bool(getattr(trade, "is_recovery", False)):
            return
        from engine.market_regime import option_stop_tolerance_pct

        entry = float(getattr(trade, "entry_price", 0.0) or 0.0)
        if not math.isfinite(entry) or entry <= 0:
            return
        cap = 15.0 if self._is_utema_trade(trade) else option_stop_tolerance_pct(
            getattr(self, "options_sl_pct", 15.0), getattr(trade, "market_regime", "UNKNOWN"),
            confidence=getattr(trade, "confidence", None), grade=getattr(trade, "grade", ""),
            setup=trade,
            fast_entry=bool(getattr(trade, "volume_spike_entry", False)
                            or getattr(trade, "is_reversion_scalp", False)),
        )
        native_state = None
        if self._is_utema_trade(trade):
            state = getattr(trade, "utema_state", None)
            if not isinstance(state, dict):
                state = ((getattr(trade, "quality_verdict", {}) or {}).get("checks", {}) or {}).get("utema_state", {})
            try:
                frozen_cap = float(state.get("premium_risk_pct", 15.))
                if math.isfinite(frozen_cap) and 0 < frozen_cap <= 15.:
                    cap = min(cap, frozen_cap)
                    native_state = state
            except (TypeError, ValueError, AttributeError):
                pass
        selected_cap = getattr(trade, "_entry_option_stop_cap_pct", None)
        if native_state is None:
            current_stop = float(getattr(trade, "current_stop", 0.0) or 0.0)
            if selected_cap is None and 0 < current_stop < entry:
                selected_cap = (entry - current_stop) / entry * 100.0
            if selected_cap is not None and 0 < float(selected_cap) <= 15.0:
                cap = min(cap, float(selected_cap))
        # Carry the selected risk percentage across the order queue and actual fill,
        # including tight ATR/burst geometry whose candidate-only flags were dropped.
        trade._entry_option_stop_cap_pct = cap
        # Round the minimum SELL trigger upward to the broker's 0.05 option grid.
        # Nearest-tick rounding can otherwise put a 15% stop slightly below its cap.
        minimum_stop = math.ceil(entry * (1.0 - cap / 100.0) / 0.05 - 1e-9) * 0.05
        stop = max(float(getattr(trade, "current_stop", 0.0) or 0.0), round(minimum_stop, 2))
        if self._is_utema_trade(trade):
            # Native alignment retains any protection already earned by the filled
            # position. Reprice its option target from the effective stop, rather
            # than retaining a target calculated from the old quote or old average.
            stop = round(math.ceil(stop / 0.05 - 1e-9) * 0.05, 2)
            native_state = native_state if isinstance(native_state, dict) else {}
            try:
                rr = float(native_state.get("rr", getattr(trade, "rr_ratio", 0.)))
            except (TypeError, ValueError, OverflowError):
                rr = 0.
            if math.isfinite(rr) and rr > 0 and stop < entry:
                trade.target = entry + (entry - stop) * rr
                trade.rr_ratio = rr
                trade.entry_policy_actual_rr = rr
            # A protected add may have stop >= its new average. Preserve that
            # protection and its existing profitable target; negative risk cannot
            # define a new positive RR target.
            trade.premium_entry = entry
            trade.premium_target = float(getattr(trade, "target", 0.) or 0.)
        trade.current_stop = stop
        trade.trailing_stop = stop
        trade.initial_stop = stop
        trade.premium_stop = stop

    def _actual_fill_entry_policy_verdict(
        self,
        trade: Trade,
        *,
        phase: str = "actual_broker_fill",
    ) -> Tuple[bool, str]:
        """Revalidate 1.5R+ on the broker's own fill before arming protection.

        An adverse fill is never repaired by stretching the profit target beyond the
        approved expected move. If possible, the stop is tightened toward the fill;
        otherwise the filled exposure is risked off by the caller.
        """
        # TradeManager is also used on its own by historical/reconciliation tools that
        # have no candle cache and therefore cannot produce the immutable final-policy
        # evidence. Scanner installs this guard for every live fresh-entry path. Do not
        # fabricate a partial RR-only policy in the unintegrated case; the queued check
        # above already follows the same optional-service boundary.
        self._cap_option_entry_stop(trade)
        if self._is_utema_trade(trade):
            if not bool(getattr(trade, "_entry_policy_actual_fill_verified", True)):
                return False, "actual broker fill price is unavailable or unverified"
            state = getattr(trade, "utema_state", None)
            if not isinstance(state, dict):
                state = ((getattr(trade, "quality_verdict", {}) or {}).get("checks", {}) or {}).get("utema_state", {})
            try:
                rr = float(state.get("rr", getattr(trade, "rr_ratio", 0.)))
            except (TypeError, ValueError, OverflowError, AttributeError):
                rr = 0.
            if not math.isfinite(rr) or rr <= 0:
                return False, "UTEMA selected option RR is invalid"
            allowed, reason = self._utema_execution_geometry(
                trade.inst_type, trade.direction, trade.entry_price, trade.current_stop,
                getattr(trade, "target", 0.), protected_add=str(phase).startswith("scale_in_"),
            )
            if allowed and str(trade.inst_type).upper() == "OPT":
                trade.premium_entry = float(trade.entry_price)
                trade.premium_stop = float(trade.current_stop)
                trade.premium_target = float(getattr(trade, "target", 0.) or 0.)
            return allowed, reason
        guard = getattr(self, "fresh_entry_policy_guard", None)
        if guard is None:
            return True, ""
        if not bool(getattr(trade, "_entry_policy_actual_fill_verified", True)):
            return False, "actual broker fill price is unavailable or unverified"

        from config.settings import get_settings

        settings = get_settings()
        try:
            configured_min_rr = float(getattr(settings, "entry_policy_min_rr", 1.5))
            entry = float(getattr(trade, "entry_price", 0.0) or 0.0)
            stop = float(getattr(trade, "current_stop", 0.0) or 0.0)
            target = float(getattr(trade, "target", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            return False, "actual fill has malformed policy or execution geometry"
        if not all(math.isfinite(value) for value in (configured_min_rr, entry, stop, target)):
            return False, "actual fill has non-finite policy or execution geometry"
        min_rr = max(1.5, configured_min_rr)
        try:
            from engine import s1_lane

            if s1_lane.is_managed_lane(trade, settings):
                # S1 targets are 1R (0.7R on a weak move); holding them to 1.5R here
                # tightened every S1 stop by a third on the fill, or refused the fill.
                # UT SMART EXIT has its own floor (nominal target, the chart exits).
                min_rr = s1_lane.lane_min_entry_rr(trade, settings)
        except Exception:
            pass
        inst_type = str(getattr(trade, "inst_type", "FUT") or "FUT").upper()
        direction = str(getattr(trade, "direction", "LONG") or "LONG").upper()
        if inst_type not in {"OPT", "FUT"} or direction not in {"LONG", "SHORT"}:
            return False, "actual fill has unsupported execution identity"
        if entry <= 0 or stop <= 0 or target <= 0:
            return False, "actual fill has incomplete entry/stop/target geometry"

        if inst_type == "OPT" or direction != "SHORT":
            risk, reward = entry - stop, target - entry
            if reward > 0 and risk > 0 and reward / risk + 1e-9 < min_rr:
                tightened = entry - (reward / min_rr)
                if 0 < tightened < entry:
                    stop = max(stop, tightened)
        else:
            risk, reward = stop - entry, entry - target
            if reward > 0 and risk > 0 and reward / risk + 1e-9 < min_rr:
                tightened = entry + (reward / min_rr)
                if tightened > entry:
                    stop = min(stop, tightened)

        trade.current_stop = stop
        trade.trailing_stop = stop
        trade.initial_stop = stop
        if inst_type == "OPT":
            trade.premium_entry = entry
            trade.premium_stop = stop
            trade.premium_target = target

        if inst_type == "OPT" or direction != "SHORT":
            risk, reward = entry - stop, target - entry
        else:
            risk, reward = stop - entry, entry - target
        actual_rr = reward / risk if reward > 0 and risk > 0 else 0.0
        trade.entry_policy_actual_rr = actual_rr
        if actual_rr + 1e-9 < min_rr:
            return False, f"actual broker fill RR {actual_rr:.4f} is below {min_rr:.2f}R"

        setattr(trade, "_entry_policy_execution_levels", (entry, stop, target))
        setattr(trade, "_entry_policy_phase", str(phase or "actual_broker_fill"))
        try:
            verdict = guard(trade)
        except Exception as exc:
            return False, f"actual-fill policy crashed: {type(exc).__name__}: {exc}"
        finally:
            try:
                delattr(trade, "_entry_policy_execution_levels")
                delattr(trade, "_entry_policy_phase")
            except AttributeError:
                pass
        if isinstance(verdict, tuple):
            allowed = bool(verdict[0]) if verdict else False
            reason = str(verdict[1] if len(verdict) > 1 else "")
        elif isinstance(verdict, dict):
            allowed = bool(verdict.get("allowed", verdict.get("ok", False)))
            reason = str(verdict.get("reason") or "")
        else:
            allowed, reason = bool(verdict), ""
        return allowed, reason or ("" if allowed else "actual-fill policy rejected exposure")

    def open_trade(
        self,
        instrument: str,
        timeframe: str,
        direction: str,
        price: float,
        trailing_stop: float,
        lots: int,
        lot_size: int,
        grade: str,
        atm_strike: float = 0.0,
        option_type: str = "",
        target: float = 0.0,
        rr_ratio: float = 1.5,
        confidence: float = 0.0,
        instrument_multiplier: float = 1.0,
        trading_symbol: str = "",
        symbol_token: str = "",
        inst_type: str = "FUT",
        exec_type: str = "A",
        entry_spot: float = 0.0,
        spot_stop: float = 0.0,
        spot_target: float = 0.0,
        is_recovery: bool = False,
        is_ghost: bool = False,
        is_reversion_scalp: bool = False,
        is_explosive_bypass: bool = False,
        entry_time: Optional[datetime] = None,
        signal_time: Optional[datetime] = None,
        signal_fire_mode: str = "INTRACANDLE",
        entered_intracandle: bool = True,
        market_regime: str = "",
        data_provenance: str = "",
        price_action_verdict: str = "",
        intelligence_score: float = 0.0,
        entry_intelligence_score: Optional[float] = None,
        entry_momentum_urgency: float = 0.0,
        strategy_revision: str = "",
        strategy_direction_mode: str = "",
        market_wide_reversal_entry: bool = False,
        volume_spike_entry: bool = False,
        power_move: bool = False,
        source_engine: str = "",
        source_engines: Optional[List[str]] = None,
        source_lane: str = "",
        exit_policy_owner: str = "",
        quad_union_entry: bool = False,
        raw_union_entry: bool = False,
        initial_stop: float = 0.0,
        scalp_lock_mode: bool = False,
        underlying_entry: float = 0.0,
        underlying_stop: float = 0.0,
        underlying_target: float = 0.0,
        premium_entry: float = 0.0,
        premium_stop: float = 0.0,
        premium_target: float = 0.0,
        geometry_observed_at: str = "",
        geometry_source: str = "",
        geometry_verified: bool = False,
        expected_move_forecast: Optional[Dict[str, Any]] = None,
        entry_policy_verdict: Optional[Dict[str, Any]] = None,
        quality_verdict: Optional[Dict[str, Any]] = None,
        decision_record: Optional[Dict[str, Any]] = None,
        execution_touch_quote: Optional[Dict[str, Any]] = None,
        on_execution_result: Optional[Callable[[Trade, bool], None]] = None,
        lifecycle_id: str = "",
        entry_submission_guard: Optional[Callable[[Trade], Any]] = None,
        entry_refusal_outcome: Optional[Dict[str, Any]] = None,
    ) -> Optional[Trade]:
        """Open a new trade and execute via broker"""
        native_utema = (str(exit_policy_owner or "").upper() == "UTEMA"
                        and str(source_lane or "").upper() == "UTEMA"
                        and str(source_engine or "").lower() == "ut_ema")
        if native_utema:
            valid, reason = self._utema_execution_geometry(inst_type, direction, price, trailing_stop, target)
            if not valid:
                logger.warning(f"Cannot open UTEMA trade: {reason}")
                return None
        if entry_time is not None:
            now = entry_time
        elif is_recovery and signal_time is not None:
            now = signal_time
        else:
            now = datetime.now(IST)
        with self._lock:
            can, reason = (self.can_open_trade(is_recovery, native_utema=True) if native_utema
                           else self.can_open_trade(is_recovery))
            if not can:
                logger.warning(f"Cannot open trade: {reason}")
                return None
            if self._live_contract_risk_required() and not (is_recovery or is_ghost or getattr(self, "is_warmup", False)):
                contract_problem = self._upstox_contract_entry_error(trading_symbol, symbol_token, lot_size, inst_type)
                if contract_problem:
                    self._log_ui(f"Entry refused: {contract_problem}", "warning")
                    return None
            same, same_reason = self._same_contract_entry_blocked(
                trading_symbol, is_recovery=is_recovery, is_ghost=is_ghost,
                **({"native_owner": "UTEMA"} if native_utema else {}),
            )
            if same:
                logger.warning(f"Cannot open trade: {same_reason}")
                self._log_ui(f"Entry refused: {same_reason}", "warning")
                return None
            # Is the price we are about to trade on real? Checked before anything sizes,
            # logs or reaches the broker, because a stale mark poisons the stop, the
            # target and every P&L reading derived from them.
            require_usable_touch = (
                self._live_orders_permitted()
                and (native_utema or str(exec_type or "").upper() in {"A", "AUTO"})
                and not is_ghost
                and not bool(getattr(self, "is_warmup", False))
            )
            mark_ok, mark_reason = (self._utema_touch_admission(
                price, execution_touch_quote, inst_type=inst_type, direction=direction,
                required=require_usable_touch,
            ) if native_utema else self._entry_mark_admission(
                inst_type=inst_type, price=price, signal_time=signal_time,
                entry_time=now, touch_quote=execution_touch_quote,
                trading_symbol=trading_symbol, is_recovery=is_recovery,
                require_usable_touch=require_usable_touch,
            ))
            if not mark_ok:
                logger.warning(f"Entry refused on mark validation: {mark_reason}")
                self._log_ui(f"Entry refused: {mark_reason}", "warning")
                return None
            liquid_ok, liquid_reason = (True, "") if native_utema else self._option_contract_liquidity_admission(
                inst_type=inst_type,
                lot_size=lot_size,
                touch_quote=execution_touch_quote,
                trading_symbol=trading_symbol,
                is_recovery=is_recovery,
            )
            if not liquid_ok:
                logger.warning(f"Entry refused on option liquidity: {liquid_reason}")
                self._log_ui(f"Entry refused: {liquid_reason}", "warning")
                return None
            # Would accepting this candidate make the BOOK unsafe at its configured
            # stops? can_open_trade counts positions; the circuit breaker measures a book
            # that already exists. Neither refuses the entry that creates the breach.
            admitted, risk_reason = (True, "") if native_utema and not self._live_contract_risk_required() else self._entry_risk_admission(
                inst_type=inst_type, direction=direction, price=price,
                trailing_stop=trailing_stop, initial_stop=initial_stop,
                lots=lots, lot_size=lot_size,
                instrument_multiplier=instrument_multiplier,
                is_recovery=is_recovery,
                instrument=instrument,
            )
            if not admitted:
                logger.warning(f"Entry refused by projected stop-risk budget: {risk_reason}")
                self._log_ui(f"Entry refused: {risk_reason}", "warning")
                return None

            # ── Non-blocking early funds refusal ────────────────────────────
            # Margin/order endpoints are network calls. This method runs on the shared
            # scan loop, so it may consult only a broker-provided *cached refusal* here.
            # The definitive preflight (and its concurrency reservation) runs once in
            # _place_bg_body. A cached approval is intentionally never trusted.
            _is_warmup = getattr(self, "is_warmup", False)
            if (self._live_orders_permitted() and self.broker and trading_symbol
                    and not is_ghost and not _is_warmup and not is_recovery):
                _order_side = "BUY" if inst_type == "OPT" else ("BUY" if direction == "LONG" else "SELL")
                _qty = int(lots or 1) * int(lot_size or 1)
                _cached_refusal = getattr(self.broker, "cached_entry_margin_refusal", None)
                _early_verdict = None
                # A multi-lot entry is not refused here: the background preflight trims it
                # to the lots the funds cover (entry_trim_lots_to_funds_enabled).
                if int(lots or 1) > 1 and self._trim_lots_to_funds_enabled():
                    _cached_refusal = None
                if callable(_cached_refusal):
                    try:
                        _early_verdict = _cached_refusal(
                            symbol=trading_symbol, qty=_qty, side=_order_side,
                            product_type=self.product_type, price=float(price or 0.0),
                        )
                    except Exception as exc:
                        logger.debug(f"Cached funds refusal unavailable for {trading_symbol}: {exc}")
                if _early_verdict is not None and not _early_verdict.get("ok", True):
                    _limitation = (
                        _early_verdict.get("limitation")
                        or _early_verdict.get("reason")
                        or "insufficient margin"
                    )
                    if isinstance(entry_refusal_outcome, dict):
                        entry_refusal_outcome.update({
                            "code": "ENTRY_MARGIN_PREFLIGHT",
                            "reason": str(_limitation),
                            "basis": str(_early_verdict.get("basis") or ""),
                            "available": _early_verdict.get("available"),
                            "required": _early_verdict.get("required"),
                        })
                    logger.warning(
                        f"Entry refused on margin preflight: {trading_symbol} x{_qty}: {_limitation}"
                    )
                    self._log_ui(f"Entry refused (funds): {_limitation}", "warning")
                    return None

            known_ids = set(self.open_trades)
            known_ids.update(str(trade.id) for trade in self.closed_trades)
            while True:
                self._trade_counter += 1
                trade_id = f"T{self._trade_counter:04d}_{instrument}_{now.strftime('%m%d%H%M')}"
                if trade_id not in known_ids:
                    break

        # intelligence_score was the historical entry-telemetry parameter.  Keep it
        # as an alias for older callers and archives, but give the frozen entry value
        # an explicit name so decision code cannot confuse it with fresh exit context.
        resolved_entry_intelligence_score = (
            intelligence_score
            if entry_intelligence_score is None
            else entry_intelligence_score
        )
        forecast = dict(expected_move_forecast or {})
        policy = dict(entry_policy_verdict or {})
        resolved_underlying_entry = float(underlying_entry or entry_spot or 0.0)
        resolved_underlying_stop = float(underlying_stop or spot_stop or 0.0)
        resolved_underlying_target = float(underlying_target or spot_target or 0.0)

        trade = Trade(
            id=trade_id,
            instrument=instrument,
            timeframe=timeframe,
            direction=direction,
            entry_price=price,
            entry_time=now,
            signal_time=signal_time,
            trailing_stop=trailing_stop,
            current_stop=trailing_stop,
            lots=lots,
            lot_size=lot_size,
            grade=grade,
            atm_strike=atm_strike,
            option_type=option_type,
            target=target,
            rr_ratio=rr_ratio,
            confidence=confidence,
            inst_type=inst_type,
            instrument_multiplier=instrument_multiplier,
            current_price=price,
            trading_symbol=trading_symbol,
            symbol_token=symbol_token,
            exec_type=exec_type,
            # NEVER default the index-spot field to the CONTRACT price. On an OPT,
            # `price` is the option PREMIUM (~120) against a spot of ~24,850, and every
            # spot<->premium converter keys off this field -- one missing spot tick at
            # entry fabricated lakhs of P&L and fired TARGET_HIT on it. On a FUT it is
            # the futures LTP, which differs from spot by the basis.
            #
            # 0.0 is a supported value, not a hole: every reader guards `entry_spot > 0`
            # and falls back to holding entry_price (see update_trade below, "do NOT use
            # spot price as option price"). That guard was unreachable while this line
            # guaranteed entry_spot > 0 whenever price > 0.
            # Keep the legacy alias sourced only from its own spot-space argument.
            # ``underlying_entry`` below is canonical and may be populated by newer
            # callers independently; neither field may ever fall back to contract
            # ``price`` (option premium or futures LTP).
            entry_spot=float(entry_spot or 0.0),
            spot_stop=resolved_underlying_stop,
            spot_target=resolved_underlying_target,
            underlying_entry=resolved_underlying_entry,
            underlying_stop=resolved_underlying_stop,
            underlying_target=resolved_underlying_target,
            premium_entry=float(premium_entry or 0.0),
            premium_stop=float(premium_stop or 0.0),
            premium_target=float(premium_target or 0.0),
            geometry_observed_at=str(geometry_observed_at or ""),
            geometry_source=str(geometry_source or ""),
            geometry_verified=bool(geometry_verified),
            expected_move_pct=float(forecast.get("expected_favorable_move_pct") or 0.0),
            expected_move_floor_pct=float(forecast.get("floor_pct") or 0.0),
            expected_move_uncertainty_pct=float(forecast.get("uncertainty_pct") or 0.0),
            expected_move_probability=float(forecast.get("probability_meeting_floor") or 0.0),
            expected_move_sample_count=int(forecast.get("sample_count") or 0),
            expected_move_horizon_minutes=int(forecast.get("horizon_minutes") or 0),
            expected_move_generated_at=str(forecast.get("generated_at") or ""),
            expected_move_data_as_of=str(forecast.get("data_as_of") or ""),
            expected_move_model_version=str(forecast.get("model_version") or ""),
            expected_move_source=str(forecast.get("source") or ""),
            entry_policy_allowed=bool(policy.get("allowed", False)),
            entry_policy_phase=str(policy.get("phase") or ""),
            entry_policy_checked_at=str(policy.get("checked_at") or ""),
            entry_policy_reason=str("; ".join(policy.get("reasons") or [])),
            entry_policy_actual_rr=float(policy.get("actual_rr") or 0.0),
            entry_policy_underlying_stop_pct=float(policy.get("underlying_stop_pct") or 0.0),
            entry_policy_version=str(policy.get("policy_version") or ""),
            quality_verdict=dict(quality_verdict or {}),
            decision_record=dict(decision_record or {}),
            is_ghost=is_ghost,
            is_recovery=bool(is_recovery),
            is_reversion_scalp=is_reversion_scalp,
            signal_fire_mode=str(signal_fire_mode or "INTRACANDLE"),
            entered_intracandle=bool(entered_intracandle),
            market_regime=str(market_regime or ""),
            data_provenance=str(data_provenance or ""),
            price_action_verdict=str(price_action_verdict or ""),
            entry_intelligence_score=float(resolved_entry_intelligence_score or 0.0),
            intelligence_score=float(resolved_entry_intelligence_score or 0.0),
            strategy_revision=str(strategy_revision or ""),
            strategy_direction_mode=str(strategy_direction_mode or ""),
            market_wide_reversal_entry=market_wide_reversal_entry,
            volume_spike_entry=volume_spike_entry,
            power_move=power_move,
            source_engine=str(source_engine or ""),
            source_engines=list(source_engines or []),
            source_lane=str(source_lane or ""),
            quad_union_entry=bool(quad_union_entry),
            raw_union_entry=bool(raw_union_entry),
            initial_stop=float(initial_stop or trailing_stop or 0.0),
            scalp_lock_mode=bool(scalp_lock_mode),
            # Generated trade ids contain only MMDDHHMM and can repeat in a later
            # calendar year. Prefix the full session date so terminal telemetry stays
            # globally unambiguous while remaining deterministic across a restart.
            lifecycle_id=str(lifecycle_id or f"{now.date().isoformat()}|{trade_id}"),
            stop_provenance="signal",
            signal_entry_price=float(price or 0.0),
            execution_bid_price=float((execution_touch_quote or {}).get("bid") or 0.0),
            execution_ask_price=float((execution_touch_quote or {}).get("ask") or 0.0),
            execution_touch_source=str((execution_touch_quote or {}).get("source") or ""),
            execution_touch_age_seconds=float((execution_touch_quote or {}).get("age_seconds") or 0.0),
            execution_touch_usable=bool((execution_touch_quote or {}).get("usable", False)),
            execution_touch_reason=str((execution_touch_quote or {}).get("reason") or ""),
        )
        # Not a Trade field: it matters only at the instant the entry order is placed,
        # so it never needs to persist or survive a restart.
        if native_utema:
            trade.exit_policy_owner = "UTEMA"
            state = ((trade.quality_verdict or {}).get("checks", {}) or {}).get("utema_state")
            if isinstance(state, dict):
                trade.utema_state = dict(state)
        self._cap_option_entry_stop(trade)
        setattr(trade, "entry_momentum_urgency", float(entry_momentum_urgency or 0.0))
        # Freeze the exit-policy owner at this admission boundary. A caller that
        # already froze it (the candidate it executes) passes it through; otherwise it
        # is taken from the lane switches as they stand now.
        try:
            from engine import s1_lane

            if native_utema or str(exit_policy_owner or "").strip().upper() in s1_lane.EXIT_OWNERS:
                trade.exit_policy_owner = str(exit_policy_owner).strip().upper()
            if not native_utema:
                s1_lane.stamp_exit_owner(trade)
        except Exception as exc:
            logger.debug(f"Exit-policy owner stamp skipped: {exc}")
        # Runtime-only admission state. Recovery rows re-establish bookkeeping for
        # exposure that already exists and must never be blocked by a *new exposure*
        # guard. The callable is retained on fresh AUTO trades so later scale-ins and a
        # stale-limit replacement use the same last-moment authority as the first order.
        setattr(trade, "_entry_submission_guard_bypass", bool(is_recovery))
        setattr(trade, "_entry_submission_guard", entry_submission_guard)
        setattr(trade, "_entry_execution_result_callback", on_execution_result)

        trade.pending_execution = not is_recovery
        trade.execution_status = "OPEN" if (is_recovery or is_ghost) else "ENTRY_PENDING"
        with self._lock:
            can, reason = (self.can_open_trade(is_recovery, native_utema=True) if native_utema
                           else self.can_open_trade(is_recovery))
            if not can:
                logger.warning(f"Cannot reserve trade after concurrent update: {reason}")
                return None
            self.open_trades[trade_id] = trade
            # A new position exists; reconcile now instead of after the sync loop's
            # remaining wait, which is the 30s IDLE interval before the day's first entry.
            self.nudge_position_sync()
        self._mark_dirty()

        qty = lots * lot_size
        logger.info(
            f"🟢 TRADE LOGGED: {trade_id} | {direction} {trading_symbol or instrument} @ {price:.2f} | "
            f"Stop: {trade.current_stop:.2f} | {lots}×{lot_size}={qty}{' [GHOST]' if is_ghost else ''}"
        )

        # ⚡ REAL BROKER ENTRY (Background Thread) ⚡ 
        is_warmup = getattr(self, "is_warmup", False)
        if (self._live_orders_permitted() and self.broker and trading_symbol
                and symbol_token and not is_warmup and not is_ghost and not is_recovery):
            order_side = "BUY" if inst_type == "OPT" else ("BUY" if direction == "LONG" else "SELL")
            
            def _place_bg_body():
                # Every retry runs the preflight again and takes its own reservation.
                # Collected so a definitive rejection can hand all of them back at once.
                reservations: List[str] = []
                guard_rejection_reason = ""
                margin_rejection_reason = ""

                def place_order():
                    nonlocal guard_rejection_reason, margin_rejection_reason
                    # Ask the broker whether this order can be funded BEFORE sending it.
                    # preflight_entry_margin was written for exactly this and had no
                    # caller, so an underfunded account fired a doomed order per signal
                    # and got back a generic RMS rejection with no stated cause. It fails
                    # OPEN when it cannot verify, so an unavailable margin endpoint never
                    # blocks trading.
                    preflight = self._entry_margin_verdict(
                        trading_symbol, trade.quantity, order_side, price
                    )
                    if preflight is not None and not preflight.get("ok", True):
                        trimmed = self._trim_entry_lots_to_funds(
                            trade, trading_symbol, order_side, price, preflight
                        )
                        if trimmed is not None:
                            preflight = trimmed
                    if preflight is not None and not preflight.get("ok", True):
                        margin_rejection_reason = str(
                            preflight.get("limitation")
                            or preflight.get("reason")
                            or "entry margin preflight failed"
                        )
                        raise BrokerOrderRejected(
                            margin_rejection_reason,
                            code="ENTRY_MARGIN_PREFLIGHT",
                            terminal=True,
                        )
                    # An approved preflight has RESERVED this order's margin so a
                    # concurrent entry cannot spend the same cash. The reservation must
                    # be given back the moment we know this order will not consume it —
                    # but only on a definite refusal. An AMBIGUOUS acknowledgement means
                    # the order may be live, so it keeps holding its margin; letting the
                    # next entry spend that money is how an account over-commits on
                    # exactly the orders it cannot see.
                    reservation_id = str((preflight or {}).get("reservation_id") or "")
                    if reservation_id:
                        reservations.append(reservation_id)

                    def require_submission_guard() -> None:
                        nonlocal guard_rejection_reason
                        allowed, reason = self._entry_submission_guard_verdict(
                            trade, entry_submission_guard
                        )
                        if allowed:
                            return
                        guard_rejection_reason = str(reason or "queued entry no longer allowed")
                        raise BrokerOrderRejected(
                            guard_rejection_reason,
                            code="ENTRY_SUBMISSION_GUARD",
                            terminal=True,
                        )

                    # This is deliberately after margin reservation and immediately
                    # before broker mutation.  A retry can spend seconds in the queue;
                    # re-evaluate every attempt and let the definitive-rejection cleanup
                    # below release every reservation accumulated by earlier attempts.
                    require_submission_guard()
                    submission_levels = getattr(
                        trade, "_entry_policy_submission_levels", None
                    )
                    if isinstance(submission_levels, (tuple, list)) and len(submission_levels) >= 3:
                        submission_price = float(submission_levels[0] or 0.0)
                    else:
                        submission_price = float(price or 0.0)
                    submission_touch = getattr(
                        trade,
                        "_entry_policy_submission_touch_quote",
                        execution_touch_quote,
                    )
                    # Start the acknowledgement/fill clock BEFORE the first broker
                    # submission. Stamping after place_order returned excluded the whole
                    # API round trip and left no timestamp at all while a slow/ambiguous
                    # request was in flight. Retries preserve the first attempt's time.
                    stamped_submission = False
                    with self._lock:
                        if trade.entry_submitted_at is None:
                            trade.entry_submitted_at = datetime.now(IST)
                            stamped_submission = True
                    if stamped_submission:
                        self._mark_dirty()
                    try:
                        return self.broker.place_order(
                            symbol=trading_symbol,
                            token=symbol_token,
                            qty=trade.quantity,
                            side=order_side,
                            product_type=self.product_type,
                            price=submission_price,
                            touch_quote=submission_touch,
                            # Tag the order with its trade so the order book, trades
                            # and postbacks map back exactly instead of by heuristic.
                            trade_id=trade.id,
                            # Decides MARKET vs LIMIT: only a high-conviction signal
                            # is worth risking a missed fill to get a better price.
                            confidence=float(getattr(trade, "confidence", 0.0) or 0.0),
                            # How fast the premium is leaving the bid, from MACD
                            # acceleration. Not persisted: it is only meaningful at the
                            # instant the order is placed, so it never needs to survive a
                            # restart. Absent on a recovery or a replayed row, which
                            # correctly reads as "no urgency".
                            momentum_urgency=float(
                                getattr(trade, "entry_momentum_urgency", 0.0) or 0.0
                            ),
                        )
                    except TypeError as exc:
                        if not any(
                            arg in str(exc)
                            for arg in ("touch_quote", "trade_id", "confidence", "momentum_urgency")
                        ):
                            raise
                        # The compatibility fallback is another actual broker call.  Run
                        # the guard again in case its state changed after the first call
                        # failed on unsupported keyword arguments.
                        require_submission_guard()
                        fallback_levels = getattr(
                            trade, "_entry_policy_submission_levels", None
                        )
                        fallback_price = (
                            float(fallback_levels[0] or 0.0)
                            if isinstance(fallback_levels, (tuple, list))
                            and len(fallback_levels) >= 3
                            else float(price or 0.0)
                        )
                        return self.broker.place_order(
                            symbol=trading_symbol,
                            token=symbol_token,
                            qty=trade.quantity,
                            side=order_side,
                            product_type=self.product_type,
                            price=fallback_price,
                        )

                order_id, error, attempts = execute_order_with_retry(
                    place_order,
                    self._extract_order_id,
                    f"Entry order {trading_symbol}",
                )
                if order_id:
                    with self._lock:
                        if trade.id in self.open_trades:
                            trade.broker_order_id = order_id
                            trade.pending_execution = True
                            if trade.entry_submitted_at is None:
                                trade.entry_submitted_at = datetime.now(IST)
                            trade.execution_status = "ENTRY_SUBMITTED"
                            trade.execution_error = ""
                    self._mark_dirty()
                    if on_execution_result:
                        on_execution_result(trade, True)
                    msg = (
                        f"✅ Broker order placed successfully for {trading_symbol}: "
                        f"{order_id} (attempt {attempts})"
                    )
                    logger.info(msg)
                    self._log_ui(msg, "trade")
                    return

                ambiguous_ack = "ambiguous" in str(error or "").lower()
                if ambiguous_ack:
                    msg = (
                        f"⚠️ Broker entry acknowledgement ambiguous for {trading_symbol}; "
                        "trade requires broker position reconciliation before any new action."
                    )
                    logger.critical(msg)
                    self._log_ui(msg, "error")
                    with self._lock:
                        if trade.id in self.open_trades:
                            trade.pending_execution = True
                            if trade.entry_submitted_at is None:
                                trade.entry_submitted_at = datetime.now(IST)
                            trade.execution_status = "RECOVERY_REQUIRED"
                            trade.execution_error = error or "Broker entry acknowledgement ambiguous"
                    self._mark_dirty()
                    if on_execution_result:
                        on_execution_result(trade, False)
                    return

                # A definitive refusal consumes no margin, so give every reservation
                # this attempt took back immediately rather than waiting out the TTL and
                # blocking the next signal on money nobody is spending. The ambiguous
                # branch above deliberately does NOT reach here.
                self._release_margin_reservations(reservations)
                guard_blocked = bool(guard_rejection_reason)
                margin_blocked = bool(margin_rejection_reason)
                if guard_blocked:
                    msg = (
                        f"❌ Entry submission guard rejected {trading_symbol}: "
                        f"{guard_rejection_reason}"
                    )
                elif margin_blocked:
                    msg = (
                        f"❌ Entry margin preflight rejected {trading_symbol}: "
                        f"{margin_rejection_reason}"
                    )
                else:
                    msg = (
                        f"❌ Broker entry failed for {trading_symbol} after "
                        f"{attempts} attempts: {error}"
                    )
                logger.error(msg)
                self._log_ui(msg, "error")
                if not guard_blocked and not margin_blocked:
                    self._alert_blocking_broker_condition(error)
                with self._lock:
                    if trade.id in self.open_trades:
                        trade.status = "CLOSED"
                        trade.exit_reason = (
                            "ENTRY_SUBMISSION_GUARD"
                            if guard_blocked
                            else (
                                "ENTRY_MARGIN_PREFLIGHT"
                                if margin_blocked
                                else "BROKER_ERROR"
                            )
                        )
                        trade.exit_time = datetime.now(IST)
                        trade.exit_price = trade.entry_price
                        trade.pending_execution = False
                        trade.execution_status = "REJECTED"
                        trade.execution_error = error
                        self.closed_trades.append(trade)
                        del self.open_trades[trade.id]
                self._mark_dirty()
                if on_execution_result:
                    on_execution_result(trade, False)
            

            def place_bg():
                """Run the entry task, and never let an exception vanish into the queue.

                AsyncExecutionQueue catches and logs whatever escapes a task, then moves
                on — the worker survives, but the TRADE does not get cleaned up. Every
                statement after execute_order_with_retry returns is unprotected
                (save_state, _log_ui, _alert_blocking_broker_condition, the
                on_execution_result callback), so a throw there left the row sitting in
                open_trades with pending_execution True, holding a position slot and
                counting against exposure while nothing rested at the exchange.

                RECOVERY_REQUIRED, not "closed": an arbitrary exception says nothing
                about whether the order reached the broker, and concluding "not placed"
                from a failure is the mistake this codebase keeps making. The pending
                flags are deliberately left alone for the same reason.
                """
                try:
                    return _place_bg_body()
                except Exception as exc:
                    logger.critical(
                        f"[ENTRY TASK ESCAPED] {trade.id} {trade.trading_symbol}: {exc}. "
                        "Broker state unknown; marking for reconciliation."
                    )
                    try:
                        self._mark_recovery(
                            trade,
                            f"entry task raised after submission ({exc}); the order may or "
                            "may not be live at the broker.",
                        )
                    except Exception:
                        pass
                    return None
            from engine.async_queue import execution_queue
            # `is False` and not `not ...`: submit() is documented -> bool, and a test
            # double or simulation stub that returns None means "I do not signal", not
            # "I refused". Treating None as refusal closed 20 healthy trades outright.
            if execution_queue.submit(f"PlaceOrder_{trading_symbol}", place_bg) is False:
                # place_bg never runs, so NO order is sent. Ignoring this left the row
                # pending_execution forever: it held a position slot, counted against
                # exposure and looked live on the dashboard while the broker had never
                # heard of it. Nothing is resting at the exchange, so unlike the exit
                # path this can be closed outright rather than escalated.
                error = "Execution queue refused the entry task; no order was sent"
                logger.error(f"❌ {error} for {trading_symbol}")
                self._log_ui(f"❌ {error} for {trading_symbol}", "error")
                with self._lock:
                    if trade.id in self.open_trades:
                        trade.status = "CLOSED"
                        trade.exit_reason = "BROKER_ERROR"
                        trade.exit_time = datetime.now(IST)
                        trade.exit_price = trade.entry_price
                        trade.pending_execution = False
                        trade.execution_status = "REJECTED"
                        trade.execution_error = error
                        self.closed_trades.append(trade)
                        del self.open_trades[trade.id]
                self._mark_dirty()
                if on_execution_result:
                    on_execution_result(trade, False)
        else:
            if (
                getattr(self, "mode", "HISTORICAL") == "REAL"
                and not is_recovery
                and not is_ghost
            ):
                with self._lock:
                    trade.status = "CLOSED"
                    trade.pending_execution = False
                    trade.execution_status = "REJECTED"
                    trade.execution_error = "REAL broker or executable contract details unavailable"
                    trade.exit_reason = "BROKER_UNAVAILABLE"
                    trade.exit_time = datetime.now(IST)
                    trade.exit_price = trade.entry_price
                    self.closed_trades.append(trade)
                    self.open_trades.pop(trade.id, None)
                self._mark_dirty()
                if on_execution_result:
                    on_execution_result(trade, False)
            else:
                trade.pending_execution = False
                trade.execution_status = "OPEN"
                if on_execution_result:
                    on_execution_result(trade, True)

        return trade

    def update_trade(
        self, 
        trade_id: str, 
        current_spot: float, 
        new_trailing_stop: float = None,
        current_time: Optional[datetime] = None,
        real_premium: Optional[float] = None,
        real_fut_price: Optional[float] = None,
        quote_verified: Optional[bool] = None,
        allow_broker_protection: bool = True,
        observed_contract_ticks: Optional[List[Tuple[float, Optional[datetime]]]] = None,
        early_loss_cut_decision: Optional[bool] = None,
    ):
        """
        Update trade with latest price and trailing stop from 1-min data.
        Automatically closes trade if trailing stop is hit. ``observed_contract_ticks``
        carries exact, ordered websocket prices hidden between sampled monitor passes.
        ``early_loss_cut_decision`` is the scanner's verdict for this call only;
        None retains the standalone backstop. Hard stops/targets remain independent.
        """
        trade = self.open_trades.get(trade_id)
        if not trade or trade.status != "OPEN":
            return
        recovery_required = (
            str(getattr(trade, "execution_status", "") or "").upper()
            == "RECOVERY_REQUIRED"
        )
        if (
            getattr(trade, "pending_execution", False)
            and not getattr(trade, "pending_exit", False)
            and not recovery_required
            and self._confirmed_broker_quantity(trade) <= 0
        ):
            return  # Wait for broker confirmation before triggering stops

        # If it's an option, current_price is the synthetic premium. 
        # Premium change = (Spot Change * Signed Delta)
        is_option = trade.inst_type == "OPT" or "CE" in getattr(trade, 'trading_symbol', '') or "PE" in getattr(trade, 'trading_symbol', '')
        # Snapshot levels before this pass tightens the trail. Historical ticks must not
        # be tested against a stop that did not exist when they arrived.
        active_stop_before_update = float(getattr(trade, "current_stop", 0.0) or 0.0)
        active_target_before_update = float(getattr(trade, "target", 0.0) or 0.0)
        runner_before_update = bool(getattr(trade, "runner_mode", False))
        observed_exit_touch = None
        if observed_contract_ticks and quote_verified is not False:
            for raw_tick in observed_contract_ticks:
                try:
                    observed_price = float(raw_tick[0] if isinstance(raw_tick, (tuple, list)) else raw_tick)
                    observed_at = (
                        raw_tick[1]
                        if isinstance(raw_tick, (tuple, list)) and len(raw_tick) > 1
                        else current_time
                    )
                except (TypeError, ValueError, IndexError):
                    continue
                if observed_price <= 0:
                    continue
                if is_option and not self._option_premium_quote_plausible(
                    trade, observed_price, current_spot
                ):
                    continue
                if is_option:
                    stop_hit = active_stop_before_update > 0 and observed_price <= active_stop_before_update
                    target_hit = (
                        not runner_before_update
                        and active_target_before_update > 0
                        and observed_price >= active_target_before_update
                    )
                elif trade.direction == "LONG":
                    stop_hit = active_stop_before_update > 0 and observed_price <= active_stop_before_update
                    target_hit = (
                        not runner_before_update
                        and active_target_before_update > 0
                        and observed_price >= active_target_before_update
                    )
                else:
                    stop_hit = active_stop_before_update > 0 and observed_price >= active_stop_before_update
                    target_hit = (
                        not runner_before_update
                        and active_target_before_update > 0
                        and observed_price <= active_target_before_update
                    )
                if stop_hit or target_hit:
                    observed_exit_touch = (
                        "TRAILING_STOP" if stop_hit else "TARGET_HIT",
                        observed_price,
                        observed_at,
                    )
                    break
        
        if is_option:
            if (
                real_premium is not None
                and real_premium > 0
                and self._option_premium_quote_plausible(trade, real_premium, current_spot)
            ):
                trade.current_price = float(real_premium)
            elif real_premium is not None and real_premium > 0:
                logger.warning(
                    f"Ignored implausible option premium {float(real_premium):.2f} "
                    f"for {trade.instrument} {trade.trading_symbol}; using synthetic premium fallback."
                )
                if trade.entry_spot > 0:
                    spot_change = current_spot - trade.entry_spot
                    premium_move = spot_change * trade.instrument_multiplier
                    trade.current_price = trade.entry_price + premium_move
                else:
                    trade.current_price = trade.entry_price
            elif trade.entry_spot > 0:
                spot_change = current_spot - trade.entry_spot
                
                # Premium Move = Spot Move * Signed Delta (instrument_multiplier)
                # PE has negative delta, so spot down (-100) * delta (-0.5) = +50 premium.
                premium_move = spot_change * trade.instrument_multiplier
                trade.current_price = trade.entry_price + premium_move
            else:
                # Fallback: If entry_spot is missing, do NOT use spot price as option price!
                # Keep it at entry_price to avoid massive fake P&L.
                trade.current_price = trade.entry_price
                
            # Ensure premium doesn't go below 1.0 (option value floor)
            trade.current_price = max(0.05, trade.current_price)
        else:
            if real_fut_price is not None and real_fut_price > 0:
                trade.current_price = float(real_fut_price)
            elif trade.entry_spot > 0:
                trade.current_price = trade.entry_price + (current_spot - trade.entry_spot)
            else:
                trade.current_price = current_spot

        logger.debug(f"💓 Heartbeat {trade.id}: Price {trade.current_price:.2f} | Stop {trade.current_stop:.2f} | PnL ₹{trade.pnl:.0f}")

        # Update trailing stop (only ratchet — never move against trade).
        # Audit #9: single source of truth for units —
        #   * OPT: new_trailing_stop is PREMIUM space (scanner converts spot→prem)
        #   * FUT: new_trailing_stop is CONTRACT price space
        # Dual spot*delta math was removed; scanner owns spot→premium mapping.
        if new_trailing_stop is not None and (not is_option or quote_verified is not False):
            try:
                incoming_stop = float(new_trailing_stop)
            except (TypeError, ValueError):
                incoming_stop = 0.0
            if incoming_stop > 0:
                if is_option:
                    # Options are always long premium: only tighten (raise) the stop.
                    if incoming_stop > float(trade.current_stop or 0.0):
                        trade.current_stop = incoming_stop
                elif trade.direction == "LONG":
                    if incoming_stop > float(trade.current_stop or 0.0):
                        trade.current_stop = incoming_stop
                elif trade.direction == "SHORT":
                    current = float(trade.current_stop or 0.0)
                    if current <= 0 or incoming_stop < current:
                        trade.current_stop = incoming_stop

        trade.quote_verified = quote_verified is not False
        trade._uts_stop_touch_pending = bool(observed_exit_touch and observed_exit_touch[0] == "TRAILING_STOP")
        if int(getattr(trade, "broker_quantity", 0) or 0) > 0:
            if allow_broker_protection:
                self.ensure_broker_protection(trade, trade.broker_quantity)
            else:
                # REST/order-book work belongs to PositionSync, never to the shared
                # scanner/HFT loop. Wake it immediately rather than waiting for its
                # normal interval, while preserving the same protection authority.
                self.nudge_position_sync()
            if recovery_required:
                self._maybe_auto_flatten_unprotected(
                    trade,
                    "Broker protection unavailable",
                )

        if is_option and quote_verified is False:
            # Synthetic delta/theta marks are useful for display continuity, but they
            # are not executable prices. They must not ratchet persisted peaks/stops or
            # trigger a real strategy close while the contract book is stale/missing.
            setattr(trade, "quote_verified", False)
            now = time.monotonic()
            if now - float(getattr(self, "_last_mark_to_market_save", 0.0) or 0.0) >= 1.0:
                self._last_mark_to_market_save = now
                self._mark_dirty()
            return

        # Keep open-trade mark-to-market consistent with risk-manager net P&L.
        trade.pnl = trade.unrealized_pnl

        # Track peak and drawdown
        trade.peak_pnl = max(trade.peak_pnl, trade.pnl)
        trade.max_drawdown = max(trade.max_drawdown, trade.peak_pnl - trade.pnl)
        self._sample_pnl_path(trade)
        # The flight recorder, which until now had no caller at all. Both writers -- this
        # one and Scanner._emit_forensic_event -- were fully built, REAL-mode gated and
        # deduped, and neither was ever invoked: signal_lifecycle_events held exactly one
        # row across 24 sessions, and it was a fixture. The dashboard's Signal Forensics
        # panel has therefore never shown a real lifecycle.
        #
        # Here rather than in a per-cycle sweep because this is where the trade's state
        # actually changes, and it sits beside the sibling telemetry for the same reason.
        # It takes an iterable, so one trade is a valid batch. Cost is bounded the same
        # way pnl_path is: _record_trade_forensics keeps a state cache and throttles
        # marks to a 5s interval / Rs.25 move, and only touches sqlite when that produces
        # an event.
        self._record_trade_forensics((trade,))

        if recovery_required or getattr(trade, "pending_exit", False):
            now = time.monotonic()
            if now - self._last_mark_to_market_save >= 1.0:
                self._last_mark_to_market_save = now
                self._mark_dirty()
            return

        # ── EXIT CHECKS (Stop Loss, Target, Early Loss-Guard) ──
        # Symmetric for FUT and OPT:
        #   * trailing stop / target trail on winners
        #   * early loss-cut on losers (mirror of profit lock on the loss side)
        # Option trades are premium-led; futures are contract-price-led.
        runner_mode = bool(getattr(trade, "runner_mode", False))
        target = float(getattr(trade, "target", 0.0) or 0.0)

        # Honor this cycle's complete scanner decision, including cost/peak/age
        # deferrals. A persisted threshold cannot carry those guards and may be
        # stale. Independent callers (or scanner evaluation failures) keep the base
        # backstop; no verdict is stored on the position for a later cycle.
        if observed_exit_touch is not None:
            touch_reason, touch_price, touch_time = observed_exit_touch
            self.close_trade(
                trade_id,
                touch_price,
                touch_reason,
                exit_time=touch_time if isinstance(touch_time, datetime) else current_time,
            )
        elif early_loss_cut_decision is True or (
            early_loss_cut_decision is None
            and self._early_loss_cut_min_hold_met(trade)
            and self._early_loss_cut_hit(trade)
        ):
            self.close_trade(
                trade_id, trade.current_price, "EARLY_LOSS_CUT", exit_time=current_time
            )
        elif trade.inst_type == "OPT":
            if trade.current_price <= trade.current_stop:
                self.close_trade(trade_id, trade.current_price, "TRAILING_STOP", exit_time=current_time)
            elif (not runner_mode) and target > 0 and trade.current_price >= target:
                self.close_trade(trade_id, trade.current_price, "TARGET_HIT", exit_time=current_time)
        elif trade.direction == "LONG":
            if trade.current_price <= trade.current_stop:
                self.close_trade(trade_id, trade.current_price, "TRAILING_STOP", exit_time=current_time)
            elif (not runner_mode) and target > 0 and trade.current_price >= target:
                self.close_trade(trade_id, trade.current_price, "TARGET_HIT", exit_time=current_time)
        else:  # SHORT FUT
            if trade.current_price >= trade.current_stop:
                self.close_trade(trade_id, trade.current_price, "TRAILING_STOP", exit_time=current_time)
            elif (not runner_mode) and target > 0 and trade.current_price <= target:
                self.close_trade(trade_id, trade.current_price, "TARGET_HIT", exit_time=current_time)
        
        if trade_id in self.open_trades:
            from config.settings import get_settings
            checkpoint = max(
                0.25,
                float(getattr(get_settings(), "trade_state_checkpoint_seconds", 1.0) or 1.0),
            )
            now = time.monotonic()
            if now - self._last_mark_to_market_save >= checkpoint:
                self._last_mark_to_market_save = now
                self._mark_dirty()

    @staticmethod
    def _option_premium_quote_plausible(trade: Trade, premium: float, current_spot: float = 0.0) -> bool:
        """Reject option LTPs that are obviously index/futures prices."""
        try:
            premium = float(premium or 0.0)
        except (TypeError, ValueError):
            return False
        if premium <= 0:
            return False

        entry = float(getattr(trade, "entry_price", 0.0) or 0.0)
        target = float(getattr(trade, "target", 0.0) or 0.0)
        spot = float(current_spot or getattr(trade, "entry_spot", 0.0) or 0.0)
        strike = float(getattr(trade, "atm_strike", 0.0) or 0.0)
        opt_type = str(getattr(trade, "option_type", "") or "").upper()
        intrinsic = 0.0
        if spot > 0 and strike > 0:
            intrinsic = max(0.0, spot - strike) if opt_type == "CE" else max(0.0, strike - spot)

        bounds = [1000.0]
        if entry > 0:
            bounds.append(entry * 4.0)
        if target > 0:
            bounds.append(target * 3.0)
        if spot > 0:
            bounds.append(intrinsic + spot * 0.05)
            bounds.append(spot * 0.12)
        return premium <= max(bounds)

    def _early_loss_cut_min_hold_met(self, trade: "Trade") -> bool:
        """False while a live position is younger than the early-loss-cut hold.

        `_early_loss_cut_hit` is pure arithmetic and has two callers. The scanner's
        _loss_guard_decision (scanner.py:16658) has carried a 15s hold, a give-back
        deferral and a cost floor since 2026-08-16 (scanner.py:16589-16601). This
        backstop, reached from update_trade on every monitor tick, closes the trade
        before that path is ever consulted — so without the same hold none of those
        guards bind in the live lane.

        Deliberately the short 15s hold and not LOSS_THREAT's 60s: this exit exists
        to be fast, so the longer hold would disable it rather than guard it. The R
        thresholds and the hard stop are untouched.

        Absence never blocks — no entry stamp, an unreadable one, or a clock behind
        the fill all return True, the same never-conclude-from-absence rule the
        scanner-side holds follow. A replay keeps the old behaviour because "now"
        is meaningless there.
        """
        try:
            from config.settings import get_settings

            min_hold = float(
                getattr(get_settings(), "early_loss_cut_min_hold_seconds", 15.0) or 0.0
            )
        except Exception:
            return True
        if min_hold <= 0:
            return True
        if str(getattr(self, "mode", "HISTORICAL") or "").upper() != "REAL":
            return True
        entered = getattr(trade, "entry_filled_at", None) or getattr(
            trade, "entry_time", None
        )
        if entered is None:
            return True
        try:
            ref = entered if getattr(entered, "tzinfo", None) else IST.localize(entered)
            held = (datetime.now(IST) - ref).total_seconds()
        except Exception:
            return True
        # A negative age means the clock disagrees with the fill; direction-safety
        # says an inconsistent clock may not manufacture a young position.
        if held != held or held < 0:
            return True
        return held >= min_hold

    @staticmethod
    def _early_loss_cut_hit(trade: "Trade", cut_r_override: Optional[float] = None) -> bool:
        """True when unrealized loss reaches the early loss-cut R.

        Base threshold is settings fut/opt_loss_cut_r (default 0.8R). Scanner may
        pass a lower cut_r_override when intel confirms a big reversal (0.55 / 0.35).
        """
        try:
            from config.settings import get_settings

            settings = get_settings()
        except Exception:
            return False

        is_opt = str(getattr(trade, "inst_type", "FUT") or "FUT").upper() == "OPT"
        if is_opt:
            if not bool(getattr(settings, "opt_early_loss_cut_enabled", True)):
                return False
            cut_r = float(getattr(settings, "opt_loss_cut_r", 0.8) or 0.8)
        else:
            if not bool(getattr(settings, "fut_early_loss_cut_enabled", True)):
                return False
            cut_r = float(getattr(settings, "fut_loss_cut_r", 0.8) or 0.8)
        if cut_r_override is not None:
            try:
                cut_r = float(cut_r_override)
            except (TypeError, ValueError):
                pass
        cut_r = max(0.05, min(1.0, cut_r))

        entry = float(getattr(trade, "entry_price", 0.0) or 0.0)
        current = float(getattr(trade, "current_price", 0.0) or 0.0)
        if entry <= 0 or current <= 0:
            return False

        initial_stop = float(
            getattr(trade, "initial_stop", 0.0)
            or getattr(trade, "trailing_stop", 0.0)
            or 0.0
        )
        if initial_stop <= 0:
            return False

        qty = max(1, int(getattr(trade, "lots", 1) or 1) * int(getattr(trade, "lot_size", 1) or 1))
        # OPT is always long-premium; FUT uses direction.
        if is_opt:
            risk_pts = max(0.0, entry - initial_stop)
            pnl_pts = current - entry
        elif str(getattr(trade, "direction", "LONG") or "LONG").upper() == "LONG":
            risk_pts = max(0.0, entry - initial_stop)
            pnl_pts = current - entry
        else:
            risk_pts = max(0.0, initial_stop - entry)
            pnl_pts = entry - current

        if risk_pts <= 0:
            return False
        risk_rs = risk_pts * qty
        pnl_rs = pnl_pts * qty
        # Only cut losers; winners are handled by profit-lock / trail.
        if pnl_rs >= 0:
            return False
        return abs(pnl_rs) >= (cut_r * risk_rs) - 1e-9

    def _request_broker_exit(
        self,
        trade: Trade,
        on_exit_result: Optional[Callable[[Trade, bool], None]] = None,
    ) -> None:
        broker = self.broker
        product_type = self.product_type

        def _close_bg_body():
            with self._lock:
                current = self.open_trades.get(trade.id)
                if (
                    current is not trade
                    or trade.status != "OPEN"
                    or not trade.pending_exit
                ):
                    logger.warning(
                        f"Skipped stale queued exit for {trade.id}; "
                        "the position is no longer pending locally."
                    )
                    if on_exit_result:
                        on_exit_result(trade, False)
                    return

            # The mode gate ran at ENQUEUE. This body runs later on the execution queue,
            # and the staleness check above only asks about the TRADE. If REAL was left
            # in between, the captured `broker` is still the live account — so authority
            # is re-asked here, against the broker this closure will actually use.
            if self._live_account_send_blocked(broker):
                logger.error(
                    f"Queued exit for {trade.id} refused at dispatch: mode is "
                    f"{getattr(self, 'mode', '')!r} and this task still holds the live "
                    "account broker."
                )
                if on_exit_result:
                    on_exit_result(trade, False)
                return

            exit_side = "SELL" if trade.direction == "LONG" or trade.inst_type == "OPT" else "BUY"
            exit_quantity = self._transactable_quantity(trade)
            if exit_quantity <= 0:
                # Nothing to exit. Sending the intended size here would sell a SIBLING's
                # quantity on the same contract, which on options opens a naked reverse
                # leg -- the opposite of what an exit is for.
                logger.error(
                    f"Exit for {trade.id} refused: this trade owns no quantity "
                    f"({trade.trading_symbol}). Nothing was sent."
                )
                if on_exit_result:
                    on_exit_result(trade, False)
                return

            if trade.protective_order_id:
                # PROVE the cancel; do not just acknowledge it. cancel_order returns True
                # whenever the HTTP call did not raise. The SL-M trigger sits at
                # protective_trigger — the same level the local trailing stop fires at —
                # so the cancel races the trigger at exactly the moment an exit is
                # decided. Kite can accept the cancel request while the triggered order
                # still fills; the full-size MARKET exit below then sells on top of an
                # SL-M that already sold, and the second fill is a naked written option
                # in CARRYFORWARD/NRML.
                #
                # _run_partial_reduce already uses this `is not True` test verbatim, with
                # the rule spelled out: "the fill routinely wins the race against the
                # cancel". This path was the one that never adopted it.
                confirm = getattr(broker, "cancel_order_confirmed", None)
                if callable(confirm):
                    try:
                        cancelled = confirm(trade.protective_order_id, "STOPLOSS")
                    except (BrokerOrderRejected, PermissionError) as exc:
                        logger.error(f"Protective stop cancel refused: {exc}")
                        cancelled = None
                else:
                    # A broker without the proven variant keeps the old behaviour rather
                    # than deadlocking; partial broker stubs are a real pattern here.
                    cancelled = True if broker.cancel_order(
                        trade.protective_order_id, "STOPLOSS"
                    ) else None
                if cancelled is not True:
                    self._mark_recovery(
                        trade,
                        "Protective stop cancellation unconfirmed; market exit withheld to avoid over-exit",
                    )
                    trade.pending_exit = False
                    self._mark_dirty()
                    if on_exit_result:
                        on_exit_result(trade, False)
                    return
                trade.protective_order_id = ""
                self._mark_dirty()

            exit_tag_kwargs = self._tag_kwargs(broker.close_order, trade, "FULL_EXIT")

            def close_order():
                return broker.close_order(
                        symbol=trade.trading_symbol,
                        token=trade.symbol_token,
                        qty=exit_quantity,
                        side=exit_side,
                        product_type=product_type,
                        **exit_tag_kwargs,
                    )

            order_id, last_error, attempts = execute_order_with_retry(
                close_order,
                self._extract_order_id,
                f"Exit order {trade.trading_symbol}",
            )

            if order_id:
                with self._lock:
                    if self.open_trades.get(trade.id) is not trade:
                        logger.critical(
                            f"Exit order {order_id} acknowledged after {trade.id} "
                            "left local open state; broker reconciliation is required."
                        )
                        return
                    trade.exit_order_id = order_id
                    trade.execution_status = "EXIT_PENDING"
                    trade.execution_error = ""
                self._mark_dirty()
                if getattr(self, "mode", "HISTORICAL") != "REAL":
                    self._finalize_closed_trade(
                        trade.id,
                        trade,
                        trade.exit_price,
                        trade.exit_reason,
                        exit_time=trade.exit_time,
                    )
                if on_exit_result:
                    on_exit_result(trade, True)
                return

            ambiguous_ack = "ambiguous" in str(last_error or "").lower()
            if ambiguous_ack:
                trade.pending_exit = True
                trade.execution_status = "RECOVERY_REQUIRED"
                trade.execution_error = last_error or "Broker exit acknowledgement ambiguous"
            else:
                trade.pending_exit = False
                trade.execution_status = "EXIT_FAILED"
                trade.execution_error = last_error or "Unknown broker exit failure"
            self._mark_dirty()
            logger.critical(
                f"[BROKER CLOSE FAILED] Position remains active for "
                f"{trade.trading_symbol} after {attempts} attempts: "
                f"{trade.execution_error}"
            )
            if on_exit_result:
                on_exit_result(trade, False)


        def close_bg():
            """Run the exit task, and never let an exception vanish into the queue.

            Same hazard as the entry task: the queue logs and continues, leaving the row
            pending_exit against what may be a live broker position.

            pending_exit is deliberately NOT cleared here. An exception raised AFTER the
            exit order was accepted would then permit a second exit on top of a live one,
            which is worse than the stranded row. RECOVERY_REQUIRED is the signal a human
            reconciles from, and the session-end sweep already routes on it.
            """
            try:
                return _close_bg_body()
            except Exception as exc:
                logger.critical(
                    f"[EXIT TASK ESCAPED] {trade.id} {trade.trading_symbol}: {exc}. "
                    "Broker state unknown; marking for reconciliation."
                )
                try:
                    self._mark_recovery(
                        trade,
                        f"exit task raised after submission ({exc}); the exit may or may "
                        "not have reached the broker.",
                    )
                except Exception:
                    pass
                return None
        from engine.async_queue import execution_queue
        if execution_queue.submit(f"CloseOrder_{trade.trading_symbol}", close_bg) is False:
            # submit() returns False on shutdown and when the queue is still full after a
            # bounded 2s wait. Both were being ignored, so close_bg simply never ran: no
            # order was sent, nothing raised, and the row sat pending_exit forever with a
            # LIVE position behind it. The queue drains at shutdown and at EOD — exactly
            # when an unexited position matters most.
            trade.pending_exit = False
            self._mark_recovery(
                trade,
                "Execution queue refused the exit task, so no exit order was ever sent. "
                "The position is still open at the broker.",
            )
            self._mark_dirty()
            logger.critical(
                f"[EXIT NEVER QUEUED] {trade.trading_symbol} still holds a live position: "
                "the execution queue rejected the close task, so no order reached the "
                "broker. Flatten it manually if the automatic recovery does not."
            )
            if on_exit_result:
                on_exit_result(trade, False)

    def _live_orders_permitted(self) -> bool:
        """Whether this manager may send an order to a broker at all.

        Defense in depth, not the primary control. The factory does not attach a
        broker outside REAL mode, so this is a second lock on the same door —
        sited where the order is actually sent rather than where it is wired.

        Both broker-order paths previously guarded only on wiring (`self.broker`
        present, symbol/token resolved, not warmup, not ghost) and never asked
        what mode they were in, while the surrounding lifecycle checks at lines
        802, 1412, 3778 and 4692 all do. A broker injected by a test, a hot-swap,
        or a future wiring change would have reached the order path with nothing
        mode-aware in the way.

        trading_mode is Literal["HISTORICAL", "REAL"], so no legitimate mode is
        denied a fill by this: there is no third mode that trades a broker.
        """
        return str(getattr(self, "mode", "") or "").upper() == "REAL"

    def _live_account_send_blocked(self, broker=None) -> bool:
        """True when a send would reach the REAL ACCOUNT broker outside REAL mode.

        `broker` names the object the caller is actually about to send through.
        Queued work captures `broker = self.broker` at ENQUEUE time and dispatches
        later, so by then self.broker may already be the mock while the closure still
        holds the live one — checking self.broker would clear a send that is about to
        go to the real account anyway. Callers sending through self.broker pass
        nothing and are unaffected.

        _live_orders_permitted covers the two dispatch paths that ask it. Several
        broker-mutating paths never did: emergency_flatten_trade (and the flatten
        helpers below it), the three mutating branches of reconcile_broker_positions
        (cancel a resting entry, MARKET-replace it, cancel a protective stop) and the
        session-end orphan close. Their only gate is wiring — `if not self.broker`.

        In ordinary HISTORICAL running that costs nothing, because the factory attaches
        a SignalSimBroker whose order methods are stubs. The hole is the REAL ->
        HISTORICAL hot swap: scanner.configure sets self.trades.mode = "HISTORICAL"
        (scanner.py:4194) about a hundred lines before it swaps the broker back to the
        mock (scanner.py:4305), and the PositionSync daemon calls reconcile every ~5s
        throughout. In that window mode says HISTORICAL while self.broker is still the
        live KiteBroker.

        So the invariant this enforces is narrower and truer than "not REAL mode":
        never transact against the REAL ACCOUNT broker outside REAL mode. Gating those
        paths on mode alone would refuse the fake brokers that 53 test files inject in
        HISTORICAL on purpose; real_broker is wired only in main.py production startup
        (main.py:898), so it is absent under test and this predicate stays False there.
        """
        if self._live_orders_permitted():
            return False
        real = getattr(self, "real_broker", None)
        target = broker if broker is not None else getattr(self, "broker", None)
        return real is not None and target is real

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        reason: str = "MANUAL",
        exit_time: Optional[datetime] = None,
        on_exit_result: Optional[Callable[[Trade, bool], None]] = None,
    ):
        """Close a trade with IST aware timestamping"""
        needs_entry_sync = False
        with self._lock:
            trade = self.open_trades.get(trade_id)
            if not trade or trade.status != "OPEN":
                return False
            if trade.pending_exit or trade.execution_status in {"EXIT_PENDING", "PARTIAL_EXIT"}:
                return False
            requires_broker_exit = (
                getattr(self, "mode", "HISTORICAL") == "REAL"
                and not getattr(trade, "is_ghost", False)
            )
            if (
                self._entry_still_confirming(trade)
                and requires_broker_exit
                and self._confirmed_broker_quantity(trade) <= 0
                and self.broker
            ):
                needs_entry_sync = True
            elif self._entry_still_confirming(trade) and (
                not requires_broker_exit or self._confirmed_broker_quantity(trade) <= 0
            ):
                return False
            if requires_broker_exit and (
                not self._live_orders_permitted() or not self.broker or not trade.symbol_token
            ):
                trade.execution_status = "RECOVERY_REQUIRED"
                trade.execution_error = (
                    "REAL position cannot be closed without broker access and symbol token"
                )
                trade.pending_exit = False
                self._mark_dirty()
                if on_exit_result:
                    on_exit_result(trade, False)
                return False

        if needs_entry_sync:
            try:
                self.reconcile_broker_positions()
            except Exception as exc:
                logger.error(f"Broker reconciliation before close failed: {exc}")

        with self._lock:
            trade = self.open_trades.get(trade_id)
            if not trade or trade.status != "OPEN":
                return False
            if trade.pending_exit or trade.execution_status in {"EXIT_PENDING", "PARTIAL_EXIT"}:
                return False
            requires_broker_exit = (
                getattr(self, "mode", "HISTORICAL") == "REAL"
                and not getattr(trade, "is_ghost", False)
            )
            if self._entry_still_confirming(trade):
                if not requires_broker_exit or self._confirmed_broker_quantity(trade) <= 0:
                    return False
            if requires_broker_exit and (
                not self._live_orders_permitted() or not self.broker or not trade.symbol_token
            ):
                trade.execution_status = "RECOVERY_REQUIRED"
                trade.execution_error = (
                    "REAL position cannot be closed without broker access and symbol token"
                )
                trade.pending_exit = False
                self._mark_dirty()
                if on_exit_result:
                    on_exit_result(trade, False)
                return False
            measured_exit = float(exit_price or trade.current_price or 0.0)
            if measured_exit <= 0.0 and not requires_broker_exit:
                # A local/historical/ghost close has no later broker reconciliation.
                # Refuse to manufacture a zero-P&L fill from the entry price.
                return False
            trade.exit_price = measured_exit or float(trade.entry_price or 0.0)
            trade.current_price = trade.exit_price
            trade.exit_time = exit_time or datetime.now(IST)
            trade.exit_reason = reason
            trade.pending_exit = True
            trade.execution_status = "EXIT_PENDING"
            trade.execution_error = ""
        self._mark_dirty()

        if (self._live_orders_permitted() and self.broker and trade.symbol_token
                and not getattr(trade, 'is_ghost', False)):
            self._request_broker_exit(trade, on_exit_result=on_exit_result)
        else:
            self._finalize_closed_trade(
                trade_id,
                trade,
                trade.exit_price,
                reason,
                exit_time=trade.exit_time,
            )
            if on_exit_result:
                on_exit_result(trade, True)
        return True

    def check_session_end(self, current_time: Optional[datetime] = None):
        """
        Simple & Robust EOD Exit (Strict Session Bound)
        """
        if current_time:
            now = IST.localize(current_time) if current_time.tzinfo is None else current_time
        else:
            now = datetime.now(IST)
        now_t = now.time()
        
        from config.settings import get_settings
        force_exit_label = str(getattr(get_settings(), "ut_force_exit_time", "15:25") or "15:25")
        kill_threshold = self._parse_time(force_exit_label)
        
        if now_t < kill_threshold:
            return

        # ...and only on a day the exchange actually traded. This gated on the CLOCK
        # alone, so a position that survived Friday's square-off — a broker reject, an RMS
        # refusal, an IP block — kept the scanner running into the weekend, and at 15:23
        # on Saturday `now_t >= kill_threshold` was true again. It then sent exit orders
        # into a shut exchange, once a second, for two days.
        #
        # The position is real and does need closing, so this does NOT go quiet: it
        # escalates with the symbol and tells the operator to square off manually. What it
        # will not do is fire orders that can only be rejected.
        if not self._session_is_a_trading_day(now):
            self._alert_carrying_through_a_closed_day(now)
            return

        # ...and only while the F&O segment still accepts orders. The clock gate above
        # has no upper bound and the scanner does not idle while trades are open, so a
        # position that survived the square-off was re-sent every scan cycle for the
        # rest of the evening, and a rejected exit clears pending_exit so each cycle sent
        # a fresh one. Same rule as the closed-day gate: escalate, do not fire orders
        # that can only be rejected. Outside REAL mode no order is sent, so local
        # closes carry on.
        fno_close = self._fno_segment_closed(now)
        if fno_close is not None:
            self._alert_carrying_through_a_closed_day(
                now,
                headline=f"F&O CLOSED AT {fno_close.strftime('%H:%M')} ON",
                instruction=(
                    "The exchange no longer accepts F&O orders today, so they carry "
                    "overnight. Close them at the next open at the Zerodha terminal."
                ),
            )
            return

        with self._lock:
            snapshot = list(self.open_trades.items())
            
        for tid, trade in snapshot:
            if self._is_utema_trade(trade):
                continue  # Native Pine management spans sessions and has no time exit.

            # Strategy-owned square-off. CARRYFORWARD avoids broker intraday auto-squareoff.
            status = str(getattr(trade, "execution_status", "") or "").upper()
            # PARTIAL_EXIT joins RECOVERY_REQUIRED here. close_trade REFUSES that status
            # (and refuses pending_exit), so routing it there meant the unfilled remainder
            # of a partially filled exit was never squared off: its protective stop had
            # already been cancelled before the exit was sent, so the residual carried
            # overnight naked in a product the broker will not auto-square-off.
            # emergency_flatten_trade is the right path — it scans for a stop it does not
            # own, withholds when it cannot prove the position is unprotected, and latches
            # after an ambiguous acknowledgement.
            # Per-trade isolation. This loop had none, so ONE raise anywhere below took
            # every remaining position in the snapshot down with it and skipped the orphan
            # sweep after the loop — which is exactly how an UnboundLocalError in
            # emergency_flatten_trade turned one refused stop cancel into an account that
            # carried its whole book overnight in a product the broker will not square off.
            # The specific raiser is fixed; this makes the next one cost one trade, loudly,
            # instead of all of them, silently.
            try:
                if status in {"RECOVERY_REQUIRED", "PARTIAL_EXIT"}:
                    self.emergency_flatten_trade(trade, "FORCE_EOD_KILL")
                else:
                    self.close_trade(tid, trade.current_price, "FORCE_EOD_KILL", exit_time=now)
            except Exception as exc:
                logger.critical(
                    f"FORCE EXIT FAILED for {getattr(trade, 'trading_symbol', tid)}: "
                    f"{type(exc).__name__}: {exc}. This position may carry overnight — "
                    "square it off at the Zerodha terminal.",
                    exc_info=True,
                )
                self._log_ui(
                    f"FORCE EXIT FAILED for {getattr(trade, 'trading_symbol', tid)} "
                    f"({type(exc).__name__}). Square off manually.",
                    "error",
                )

        # Outside the loop and outside its try: an orphan must be swept even when every
        # local exit above failed.
        try:
            self._handle_orphans_at_session_end(now)
        except Exception as exc:
            logger.critical(f"Orphan sweep failed at session end: {exc}", exc_info=True)

    def eod_exit_recheck(self, current_time: Optional[datetime] = None) -> bool:
        """The 15:15:30 exit recheck (2026-09-26 user). Returns True when it ran.

        check_session_end re-sends a still-open trade's exit on every pass and PositionSync
        reconciles every 5s, so exits already retry on their own. This is the one explicit
        checkpoint: once a day, eod_exit_recheck_seconds after ut_force_exit_time, reconcile
        with the broker now and raise a critical alert naming anything still open.
        """
        from config.settings import get_settings

        now = current_time or datetime.now(IST)
        if now.tzinfo is None:
            now = IST.localize(now)
        settings = get_settings()
        force_exit = self._parse_time(str(getattr(settings, "ut_force_exit_time", "15:25") or "15:25"))
        delay = max(0.0, float(getattr(settings, "eod_exit_recheck_seconds", 30.0) or 0.0))
        checkpoint = IST.localize(datetime.combine(now.date(), force_exit)) + timedelta(seconds=delay)
        if now < checkpoint or getattr(self, "_eod_exit_recheck_date", None) == now.date():
            return False
        if not self._session_is_a_trading_day(now):
            return False
        self._eod_exit_recheck_date = now.date()
        if self._live_orders_permitted():
            try:
                self.reconcile_broker_positions()
            except Exception as exc:
                logger.error(f"EOD exit recheck: broker reconcile failed: {exc}")
        with self._lock:
            still_open = [
                str(getattr(t, "trading_symbol", "") or tid) for tid, t in self.open_trades.items()
                if not self._is_utema_trade(t)
            ]
        orphans = list(getattr(self, "orphan_positions", None) or [])
        label = checkpoint.strftime("%H:%M:%S")
        if still_open or orphans:
            message = (
                f"EOD EXIT RECHECK {label}: {len(still_open)} trade(s) still open "
                f"{still_open[:6]}" + (f" and {len(orphans)} broker-only position(s)" if orphans else "")
                + ". Exits keep retrying; square off at the Zerodha terminal if they do not clear."
            )
            logger.critical(message)
            self._log_ui(message, "error")
        else:
            logger.info(f"EOD exit recheck {label}: every position is flat.")
        return True

    def _fno_segment_closed(self, now: datetime) -> Optional[dtime]:
        """The F&O close time when `now` is at/after it and this manager sends real
        orders; None otherwise (including a blank fno_market_close_time)."""
        if not self._live_orders_permitted():
            return None
        from config.settings import get_settings

        label = str(getattr(get_settings(), "fno_market_close_time", "15:40") or "").strip()
        if not label:
            return None
        close = self._parse_time(label)
        return close if now.time() >= close else None

    def _session_is_a_trading_day(self, now: datetime) -> bool:
        """Did the exchange trade today? Weekends and holidays are not square-off days.

        Fails OPEN — an unreadable calendar returns True — because refusing to square off
        on a real trading day is far worse than sending an order that gets rejected on a
        holiday. The special-session case (a Saturday the exchange DOES open) is handled
        by is_market_holiday's own calendar, which knows about them.
        """
        provider = getattr(self, "data", None) or getattr(
            getattr(self, "scanner", None), "data", None
        )
        try:
            if now.weekday() >= 5:
                checker = getattr(provider, "is_market_special_session", None)
                if callable(checker) and checker(now):
                    return True
                # No provider to ask: a weekend is a weekend.
                if provider is None or not callable(checker):
                    return False
                return False
            holiday = getattr(provider, "is_market_holiday", None)
            if callable(holiday):
                return not bool(holiday(now))
        except Exception as exc:
            logger.debug(f"Trading-day check unavailable, assuming open: {exc}")
        return True

    def _alert_carrying_through_a_closed_day(
        self,
        now: datetime,
        headline: str = "MARKET CLOSED",
        instruction: str = (
            "Orders would only be rejected today. Square off at the Zerodha terminal, "
            "or they carry to the next session."
        ),
    ) -> None:
        """Say once per closed day (or closed F&O segment) that positions sit through it."""
        holding = [t for t in self.open_trades.values() if str(getattr(t, "status", "")) == "OPEN"]
        orphans = dict(getattr(self, "orphan_positions", {}) or {})
        if not holding and not orphans:
            return
        stamp = now.date().isoformat()
        if not hasattr(self, "_closed_day_alerted"):
            self._closed_day_alerted = set()
        if stamp in self._closed_day_alerted:
            return
        self._closed_day_alerted.add(stamp)
        names = [str(getattr(t, "trading_symbol", t.id)) for t in holding] + list(orphans)
        message = (
            f"{headline} {stamp}: {len(names)} position(s) still open and the force exit "
            f"cannot run — {', '.join(names[:4])}"
            + (f" +{len(names) - 4} more" if len(names) > 4 else "")
            + f". {instruction}"
        )
        logger.critical(message)
        self._log_ui(message, "error")

    def _handle_orphans_at_session_end(self, now: datetime) -> None:
        """Broker positions this system did not open are NOT covered by the sweep above.

        check_session_end iterates self.open_trades only, so a broker-only position — the
        first-start-mid-session case, or a local row lost to a corrupted state file — is
        detected by reconcile, blocks new entries, and is then carried OVERNIGHT. The
        product is CARRYFORWARD/NRML precisely so the broker does not auto-square-off, so
        nothing else is going to close it either.

        Escalating is the default rather than flattening, because an orphan is not
        necessarily ours: a position taken manually at the Zerodha terminal looks
        identical from here, and squaring one off uninvited is worse than the exposure it
        removes. force_exit_flattens_orphans lets an operator who never trades manually
        opt into the sweep.
        """
        orphans = dict(getattr(self, "orphan_positions", {}) or {})
        if not orphans:
            return
        from config.settings import get_settings

        settings = get_settings()
        flatten = _setting_enabled(
            getattr(settings, "force_exit_flattens_orphans", False), False
        ) and not self._live_account_send_blocked()
        for symbol, position in orphans.items():
            quantity = self._position_quantity(position)
            if not quantity:
                continue
            if not flatten or position.get("foreign_product"):
                self._alert_orphan_left_open(symbol, quantity)
                continue
            # Latch BEFORE sending, once per symbol per session. check_session_end runs
            # about once a second past the force-exit time, and orphan_positions is only
            # refreshed by reconcile on its 5s loop — so an un-latched success path
            # re-sends the same square-off four or five times before the book catches up.
            # On a BOUGHT option that is a naked short in a product the broker will not
            # square off, and once reconcile does update, the short is itself a
            # broker-only position, so the next pass computes the opposite side and the
            # loop oscillates, paying the spread each way.
            # Latching before the call, not after, is deliberate: an order that raises may
            # still have reached the exchange, and this file's rule everywhere else is
            # that an unacknowledged exit is never re-sent blind.
            if not hasattr(self, "_orphan_flatten_sent"):
                self._orphan_flatten_sent = set()
            if symbol in self._orphan_flatten_sent:
                continue
            self._orphan_flatten_sent.add(symbol)
            side = "SELL" if quantity > 0 else "BUY"
            try:
                order_id = self.broker.close_order(
                    str(symbol), str(position.get("symboltoken") or position.get("instrument_token") or ""),
                    abs(int(quantity)), side, self.product_type,
                )
            except (BrokerOrderRejected, PermissionError) as exc:
                logger.critical(f"Orphan square-off refused for {symbol}: {exc}")
                self._alert_orphan_left_open(symbol, quantity)
                continue
            except BrokerOrderAmbiguous as exc:
                # May be live. Never re-send blind; the operator settles it.
                logger.critical(
                    f"Orphan square-off for {symbol} was not acknowledged ({exc}). It may "
                    "be live — reconcile manually before sending another."
                )
                continue
            if order_id:
                logger.critical(
                    f"Orphan {symbol} x{abs(int(quantity))} squared off at the session "
                    f"force-exit: {order_id}"
                )
            else:
                self._alert_orphan_left_open(symbol, quantity)

    def _alert_unowned_protective_stop(self, trade, stop_id: str) -> None:
        """A stop we could not prove cancelled, on a row about to leave open_trades.

        Nothing scans closed trades for a protective order, so once the row is gone this
        SL-M has no owner in the system. Only the operator can settle it, and they need
        the id to do that. Throttled per order so the 5s reconcile loop cannot flood.
        """
        if not hasattr(self, "_unowned_stop_alerted"):
            self._unowned_stop_alerted = set()
        if stop_id in self._unowned_stop_alerted:
            return
        self._unowned_stop_alerted.add(stop_id)
        message = (
            f"UNOWNED STOP: {trade.trading_symbol} is flat at the broker and its row is "
            f"closing, but protective order {stop_id} was NOT confirmed cancelled. If it "
            "is still resting it will fire against a flat book and open a naked leg that "
            "the broker will not square off. Cancel it at the Zerodha terminal."
        )
        logger.critical(message)
        self._log_ui(message, "error")

    def _alert_orphan_left_open(self, symbol: str, quantity: int) -> None:
        """Say plainly that an unmanaged position is about to carry overnight."""
        if not hasattr(self, "_orphan_eod_alerted"):
            self._orphan_eod_alerted = set()
        if symbol in self._orphan_eod_alerted:
            return
        self._orphan_eod_alerted.add(symbol)
        message = (
            f"ORPHAN POSITION CARRYING OVERNIGHT: {symbol} x{abs(int(quantity))} is open "
            "at the broker, was not opened by this system, and is NOT covered by the "
            "force exit. CARRYFORWARD means the broker will not square it off either. "
            "Close it manually, or set force_exit_flattens_orphans to have the session "
            "end sweep it."
        )
        logger.critical(message)
        self._log_ui(message, "error")

    def check_expiry_gamma_time_exit(self, current_time: Optional[datetime] = None):
        """Square off expiry-day current-expiry (gamma) OPTION trades at the configured
        gamma time-exit, so an expiring ATM never bleeds into the afternoon theta crush.
        Gated behind expiry_day_trade_current_week and covers BOTH cohorts of expiry-day
        current-expiry trades:

          * the morning gamma cohort, entered before expiry_day_gamma_cutoff, closed at
            expiry_day_gamma_time_exit;
          * the LATE-WINDOW cohort, entered inside expiry_day_late_window (14:00-15:15),
            closed at expiry_day_late_window_time_exit.

        The late cohort previously had no time exit at all. It was unreachable twice over:
        the gamma time exit is 13:30, already past when the window opens, and the
        entry-time filter below dropped every trade entered at/after the 13:00 cutoff --
        i.e. structurally every trade the late window can create. That left a position on
        a contract expiring at 15:30 with nothing but the 15:25 force exit between it and
        settlement, in the least liquid minutes of an expiring option's life.
        """
        from config.settings import get_settings
        s = get_settings()
        if not bool(getattr(s, "expiry_day_trade_current_week", False)):
            return
        now = current_time or datetime.now(IST)
        now = IST.localize(now) if now.tzinfo is None else now.astimezone(IST)
        time_exit = self._parse_time(str(getattr(s, "expiry_day_gamma_time_exit", "13:30") or "13:30"))
        late_exit = self._parse_time(
            str(getattr(s, "expiry_day_late_window_time_exit", "15:20") or "15:20")
        )
        # Whichever cohort comes due first decides when this method starts doing work.
        if now.time() < min(time_exit, late_exit):
            return
        # Same upper bound as check_session_end, which runs right after this each cycle
        # and escalates the stuck position; sending here would only be rejected.
        if self._fno_segment_closed(now) is not None:
            return
        cutoff = self._parse_time(str(getattr(s, "expiry_day_gamma_cutoff", "13:00") or "13:00"))
        try:
            from engine.expiry_manager import expiry_manager
        except Exception:
            return
        with self._lock:
            snapshot = list(self.open_trades.items())
        for tid, trade in snapshot:
            if self._is_utema_trade(trade):
                continue  # Native Pine management spans sessions and has no time exit.

            if str(getattr(trade, "inst_type", "") or "").upper() != "OPT":
                continue
            et = getattr(trade, "entry_time", None)
            if et is None:
                continue
            et_ist = IST.localize(et) if et.tzinfo is None else et.astimezone(IST)
            try:
                is_exp = expiry_manager.is_expiry_day(trade.instrument, et_ist.date())
            except Exception:
                is_exp = False
            if not is_exp:
                continue
            # genuine T-0 gamma trade: entered on expiry day, before the morning cutoff
            if et_ist.time() < cutoff:
                if now.time() >= time_exit:
                    self.close_trade(
                        tid, trade.current_price, "EXPIRY_GAMMA_TIME_EXIT", exit_time=now
                    )
                continue
            # Entered at/after the cutoff. Only some of those sit on a contract expiring
            # TODAY -- after the cutoff the default mode rolls to the next expiry, and it
            # is the late window that re-engages the current one. in_gamma_window is the
            # exact predicate the ENTRY side used to make that choice, so asking it about
            # the entry timestamp identifies the expiring-contract cohort without
            # re-deriving the rule or touching rolled next-expiry trades.
            try:
                on_expiring_contract = bool(
                    expiry_manager.in_gamma_window(trade.instrument, et_ist)
                )
            except Exception:
                on_expiring_contract = False
            if on_expiring_contract and now.time() >= late_exit:
                self.close_trade(
                    tid, trade.current_price, "EXPIRY_LATE_WINDOW_TIME_EXIT", exit_time=now
                )

    def expire_stale_open_trades(self, now: Optional[datetime] = None, reason: str = "STALE_SESSION_END") -> int:
        """Retain previous-session trades until broker reconciliation proves closure."""
        now = now or datetime.now(IST)
        if now.tzinfo is None:
            now = IST.localize(now)
        marked = 0
        with self._lock:
            for trade in self.open_trades.values():
                if self._is_utema_trade(trade):
                    continue  # Native Pine management spans sessions and has no time exit.

                entry_time = trade.entry_time
                if entry_time.tzinfo is None:
                    entry_time = IST.localize(entry_time)
                else:
                    entry_time = entry_time.astimezone(IST)
                if entry_time.date() >= now.date():
                    continue
                # Count the TRANSITION, not the state. `marked` drives the save_state()
                # plus reconcile_broker_positions() block below, and reconcile makes a
                # synchronous Kite get_positions() call. This function runs from
                # _scan_cycle on the 1s event loop, so re-marking an already-marked row
                # fired a blocking broker REST call and a full trades-table snapshot
                # EVERY SECOND for as long as a prior-session position stayed open —
                # around the clock, on the same loop that hosts the 100ms exit monitor.
                #
                # A prior-session open position is exactly the state this system reaches
                # when the 15:25 force exit fails (NRML does not auto-square-off), i.e.
                # the one where exits matter most. A dedicated PositionSync thread
                # already reconciles on its own adaptive cadence.
                already_marked = (
                    str(getattr(trade, "execution_status", "") or "").upper()
                    == "RECOVERY_REQUIRED"
                )
                trade.execution_status = "RECOVERY_REQUIRED"
                trade.execution_error = "Previous-session trade requires broker reconciliation"
                if not already_marked:
                    marked += 1
        if marked:
            self._mark_dirty()
            try:
                self.reconcile_broker_positions()
            except Exception as exc:
                logger.error(f"Stale-position reconciliation failed: {exc}")
        return marked

    def update_stops(self, prices: Dict[str, float]):
        """Update trailing stops for all open trades based on favor-movements"""
        with self._lock:
            trades_snapshot = list(self.open_trades.items())
            
        for tid, trade in trades_snapshot:
            spot = prices.get(trade.instrument.split(' ')[0])
            if not spot: continue
            
            # Update current price for real-time PnL tracking
            self.update_trade(tid, spot)
            
            # Simple Trailing logic: if price moves in favor, trail the stop
            # Note: This is a backup; MTF usually manages this, but we keep it for safety.
            pass

    def get_dashboard_payload(self, is_historical: bool = False, backtest_days: int = 1, inst_pref: str = "AUTO") -> Dict:
        """Get filtered trades and summary based on active mode"""
        now_ist = datetime.now(IST)
        # If it's before 9:00 AM IST, we are likely reviewing the PREVIOUS day's session
        if now_ist.hour < 9:
            today = (now_ist - timedelta(days=1)).date()
        else:
            today = now_ist.date()
        
        # 1. Unified Session Identification
        today_str = today.strftime('%Y-%m-%d')
        
        def trade_session_date(trade: Trade) -> Optional[date]:
            value = getattr(trade, "entry_time", None)
            if value is None:
                return None
            try:
                import pandas as pd
                ts = pd.Timestamp(value)
                if ts.tzinfo is not None:
                    ts = ts.tz_convert(IST).tz_localize(None)
                return ts.date()
            except Exception:
                return None

        with self._lock:
            # Watch-only ghost records are represented by the scanner's session
            # signal ledger. They are not broker trades and must not affect the
            # trade-manager table, P&L, W/L, or trade count.
            snapshot_open = [
                trade for trade in self.open_trades.values()
                if not getattr(trade, "is_ghost", False)
            ]
            snapshot_closed = [
                trade for trade in self.closed_trades
                if not getattr(trade, "is_ghost", False)
            ]

        if not is_historical:
            # Live/manual mode must not project older sessions as current trades.
            session_closed = [t for t in snapshot_closed if trade_session_date(t) == today]
        else:
            all_trades = snapshot_closed + snapshot_open
            unique_dates = sorted({d for d in (trade_session_date(t) for t in all_trades) if d is not None})
            if unique_dates:
                cutoff_date = unique_dates[-min(len(unique_dates), backtest_days)]
                session_closed = [t for t in snapshot_closed if (trade_session_date(t) or date.min) >= cutoff_date]
            else:
                session_closed = []
            
        # 2. Get Open Trades
        if not is_historical:
            open_trades = [t for t in snapshot_open if trade_session_date(t) == today]
        else:
            def close_intraday_projection(trade: Trade) -> Trade:
                entry_time = trade.entry_time
                if entry_time.tzinfo is None:
                    entry_time = IST.localize(entry_time)
                else:
                    entry_time = entry_time.astimezone(IST)

                exit_price = trade.current_price if trade.current_price > 0 else trade.entry_price
                from config.settings import get_settings
                settings = get_settings()
                from engine.trade_accounting import estimate_trade_charges
                charges = trade.charges or estimate_trade_charges(
                    trade.entry_price,
                    exit_price,
                    trade.quantity,
                    trade.inst_type,
                    settings,
                    trade.instrument_multiplier,
                    instrument=trade.instrument,
                )
                if trade.inst_type == "OPT" or trade.direction == "LONG":
                    gross = (exit_price - trade.entry_price) * trade.quantity
                else:
                    gross = (trade.entry_price - exit_price) * trade.quantity

                projected = replace(trade)
                projected.status = "CLOSED"
                from config.settings import get_settings
                force_exit = self._parse_time(
                    str(getattr(get_settings(), "ut_force_exit_time", "15:25") or "15:25")
                )
                projected.exit_time = IST.localize(datetime.combine(entry_time.date(), force_exit))
                projected.exit_reason = "SESSION_END"
                projected.exit_price = exit_price
                projected.current_price = exit_price
                projected.charges = charges
                projected.pnl = gross - charges
                return projected

            filtered_open = []
            try:
                if 'unique_dates' in locals() and unique_dates:
                    filtered_open = [t for t in snapshot_open if (trade_session_date(t) or date.min) >= cutoff_date]
                else:
                    filtered_open = snapshot_open
            except Exception:
                filtered_open = snapshot_open
                
            session_closed = session_closed + [close_intraday_projection(t) for t in filtered_open]
            open_trades = []

        pref = (inst_pref or "AUTO").upper()
        if pref in {"FUT", "OPT"}:
            session_closed = [t for t in session_closed if (getattr(t, "inst_type", "FUT") or "FUT").upper() == pref]
            open_trades = [t for t in open_trades if (getattr(t, "inst_type", "FUT") or "FUT").upper() == pref]

        display_closed = self._filter_concurrency_compliant_closed(session_closed)
        display_closed = [t for t in display_closed if self._valid_closed_chronology(t)]
        display_closed = [t for t in display_closed if t.status == "CLOSED"]

        # Headline table = shadow-free + deduped (same set as get_summary).
        # Shadows remain only in summary.futures_shadow telemetry.
        from engine.trade_row_adapter import hygiene_closed_rows

        headline_closed, _ = hygiene_closed_rows(display_closed)

        # 3. Calculate Summary based on the EXACT same session pool
        summary = self.get_summary(filtered_trades=display_closed, open_trades=open_trades)

        # Stamp the LANE onto the summary the dashboard headlines.
        #
        # get_summary() totals whatever trades this manager holds, which is correct for
        # whichever mode it is running in -- but the header rendered that total as
        # "DAILY P&L" with nothing saying which mode produced it. On 2026-08-31 the
        # process was restarted at 16:46, after the close, so create_system refused REAL
        # ("outside REAL arming window") and came up HISTORICAL. The replay then showed
        # +Rs 6,065 under "DAILY P&L" for a session whose live result was -Rs 1,106.50.
        #
        # A losing day reading as a winning one is worse than any single bad trade: every
        # judgement made from that screen is made on a number that is not the money. The
        # lane is decided HERE, where is_historical is known, rather than inferred in the
        # browser, so no consumer of this payload can render the total unlabelled.
        summary["lane"] = "REPLAY" if is_historical else "LIVE"
        summary["is_replay"] = bool(is_historical)
        summary["pnl_label"] = "BACKTEST P&L" if is_historical else "DAILY P&L"

        # 4. Format already-filtered trades for UI.
        all_formatted = build_dashboard_rows(
            self,
            open_trades,
            headline_closed,
            include_orphans=not is_historical,
        )
        
        return {
            "open": [t for t in all_formatted if t["status"] in {"OPEN", "ORPHANED"}],
            "closed": [t for t in all_formatted if t["status"] == "CLOSED"],
            "summary": summary,
            "analytics": self._build_session_analytics(display_closed, open_trades),
        }

    def get_open_trades_list(self) -> List[Dict]:
        """Backward compatibility for existing API calls"""
        payload = self.get_dashboard_payload(is_historical=False)
        return payload["open"] + payload["closed"]



    def _grade_rank(self, grade: str) -> int:
        base = str(grade or "C").split()[0]
        return {"C": 0, "B": 1, "B+": 2, "A": 3, "A+": 4, "Recovered": 4}.get(base, 0)

    def _is_aplus_setup(self, trade: Trade) -> bool:
        # Delegates to the single definition in engine.signal_quality_policy so this and
        # scanner.py cannot drift apart (the expression was duplicated verbatim).
        from engine.signal_quality_policy import is_top_tier_setup

        return is_top_tier_setup(getattr(trade, "grade", ""), getattr(trade, "confidence", 0.0))

    def _valid_closed_chronology(self, trade: Trade) -> bool:
        if trade.status != "CLOSED" or not trade.entry_time or not trade.exit_time:
            return True
        entry_ts = trade.entry_time.replace(tzinfo=None) if trade.entry_time.tzinfo else trade.entry_time
        exit_ts = trade.exit_time.replace(tzinfo=None) if trade.exit_time.tzinfo else trade.exit_time
        return exit_ts > entry_ts

    def _filter_concurrency_compliant_closed(self, trades: List[Trade]) -> List[Trade]:
        """Apply live-style overlap rules to historical/session display metrics."""
        try:
            from config.settings import get_settings
            if not _setting_enabled(getattr(get_settings(), "ut_concurrency_guard", True), True):
                return sorted(
                    trades,
                    key=lambda t: (
                        t.entry_time.replace(tzinfo=None) if getattr(t.entry_time, "tzinfo", None) else t.entry_time,
                        str(getattr(t, "id", "") or ""),
                    ),
                    reverse=True,
                )
        except Exception:
            pass

        accepted: List[Trade] = []
        removed = {"same_index_cap": 0, "same_index_overlap": 0, "correlated_cap": 0}
        try:
            settings = get_settings()
            adaptive_dual_lanes = (
                str(getattr(settings, "scanner_engine", "") or "").lower() == "adaptive_dual"
                and _setting_enabled(getattr(settings, "dual_lane_enabled", True), True)
            )
        except Exception:
            adaptive_dual_lanes = False

        def is_futures_shadow(trade: Trade) -> bool:
            return (
                str(getattr(trade, "trading_symbol", "") or "").upper() == "FUT_SHADOW"
                or "FUT SHADOW" in str(getattr(trade, "grade", "") or "").upper()
            )

        def is_dual_lane_pair(left: Trade, right: Trade) -> bool:
            if not adaptive_dual_lanes:
                return False
            lanes = {
                str(getattr(left, "source_lane", "") or "").upper(),
                str(getattr(right, "source_lane", "") or "").upper(),
            }
            return lanes == {"A", "B"}

        ordered = sorted(
            trades,
            key=lambda t: (
                t.entry_time.replace(tzinfo=None) if getattr(t.entry_time, "tzinfo", None) else t.entry_time,
                -float(getattr(t, "confidence", 0.0) or 0.0),
                1 if is_futures_shadow(t) else 0,
                str(getattr(t, "instrument", "") or ""),
                str(getattr(t, "timeframe", "") or ""),
                str(getattr(t, "inst_type", "") or ""),
                str(getattr(t, "id", "") or ""),
            ),
        )

        for trade in ordered:
            entry = trade.entry_time.replace(tzinfo=None) if getattr(trade.entry_time, "tzinfo", None) else trade.entry_time
            active = []
            for existing in accepted:
                exit_time = existing.exit_time or existing.entry_time
                ex_entry = existing.entry_time.replace(tzinfo=None) if getattr(existing.entry_time, "tzinfo", None) else existing.entry_time
                ex_exit = exit_time.replace(tzinfo=None) if getattr(exit_time, "tzinfo", None) else exit_time
                if ex_entry <= entry < ex_exit:
                    active.append(existing)

            same_index = [t for t in active if t.instrument.split()[0] == trade.instrument.split()[0]]
            if len(same_index) >= 2:
                removed["same_index_cap"] += 1
                continue
            if same_index:
                existing = same_index[0]
                same_tf = getattr(existing, "timeframe", "") == getattr(trade, "timeframe", "")
                same_type = getattr(existing, "inst_type", "FUT") == getattr(trade, "inst_type", "FUT")
                higher_grade = self._grade_rank(trade.grade) > self._grade_rank(existing.grade) or self._is_aplus_setup(trade)
                if same_tf or same_type or not higher_grade:
                    if not is_dual_lane_pair(existing, trade):
                        removed["same_index_overlap"] += 1
                        continue

            same_dir = [t for t in active if t.direction == trade.direction]
            if len(same_dir) >= 2:
                all_aplus = all(self._is_aplus_setup(t) for t in same_dir) and self._is_aplus_setup(trade)
                if not all_aplus:
                    removed["correlated_cap"] += 1
                    continue

            accepted.append(trade)

        summary_key = (len(trades), len(accepted), tuple(removed.values()))
        now = time.time()
        if (
            summary_key != getattr(self, "_last_concurrency_filter_log_key", None)
            or now - getattr(self, "_last_concurrency_filter_log_time", 0.0) >= 60.0
        ):
            logger.info(
                "Historical Concurrency Guard: "
                f"accepted={len(accepted)}/{len(trades)}, removed={sum(removed.values())} "
                f"(index_cap={removed['same_index_cap']}, "
                f"overlap={removed['same_index_overlap']}, "
                f"correlated_cap={removed['correlated_cap']})."
            )
            self._last_concurrency_filter_log_key = summary_key
            self._last_concurrency_filter_log_time = now

        return sorted(
            accepted,
            key=lambda t: (
                t.entry_time.replace(tzinfo=None) if getattr(t.entry_time, "tzinfo", None) else t.entry_time,
                str(getattr(t, "id", "") or ""),
            ),
            reverse=True,
        )

    def _build_session_analytics(self, closed_trades: List[Trade], open_trades: List[Trade]) -> Dict:
        def bucketize(trades, key_fn):
            buckets = {}
            for t in trades:
                key = key_fn(t) or "--"
                bucket = buckets.setdefault(key, {"count": 0, "pnl": 0.0, "wins": 0, "losses": 0})
                pnl = float(getattr(t, "pnl", 0.0) or 0.0)
                bucket["count"] += 1
                bucket["pnl"] += pnl
                if pnl > 0:
                    bucket["wins"] += 1
                elif pnl < 0:
                    bucket["losses"] += 1
            for bucket in buckets.values():
                bucket["pnl"] = round(bucket["pnl"], 0)
                closed_count = bucket["wins"] + bucket["losses"]
                bucket["win_rate"] = round((bucket["wins"] / closed_count) * 100, 1) if closed_count else 0.0
            return dict(sorted(buckets.items(), key=lambda item: abs(item[1]["pnl"]), reverse=True))

        closed = list(closed_trades or [])
        return {
            "by_timeframe": bucketize(closed, lambda t: getattr(t, "timeframe", "--")),
            "by_grade": bucketize(closed, lambda t: str(getattr(t, "grade", "--")).split()[0]),
            "by_instrument": bucketize(closed, lambda t: str(getattr(t, "instrument", "--")).split()[0]),
            "by_type": bucketize(closed, lambda t: getattr(t, "inst_type", "--")),
            "by_exit_reason": bucketize(closed, lambda t: getattr(t, "exit_reason", "--")),
            # Price-action advisory: ADVERSE bucket's pnl = what blocking them would change.
            "by_price_action": bucketize(closed, lambda t: getattr(t, "price_action_verdict", "") or "--"),
            "open_by_type": bucketize(list(open_trades or []), lambda t: getattr(t, "inst_type", "--")),
        }

    def get_summary(self, filtered_trades: Optional[List] = None, open_trades: Optional[List] = None) -> Dict:
        """Get comprehensive performance analytics for dashboard"""
        closed = list(filtered_trades) if filtered_trades is not None else list(self.closed_trades)
        closed = [
            trade for trade in closed
            if getattr(trade, "execution_status", "") != "REJECTED"
        ]

        # ── Backtest-sweep hygiene (2026-07-09 FUT-vs-OPT audit) ──
        # Shared with dashboard rebuild so headline P&L never re-includes
        # FUT_SHADOW companions or multi-engine exact replay dups.
        from engine.trade_row_adapter import hygiene_closed_rows

        closed, _hygiene = hygiene_closed_rows(closed)
        shadow_pnl = float((_hygiene.get("futures_shadow") or {}).get("pnl") or 0.0)
        replay_excess_rows = int((_hygiene.get("engine_replay") or {}).get("excess_rows") or 0)
        replay_excess_pnl = float((_hygiene.get("engine_replay") or {}).get("excess_pnl") or 0.0)
        shadow_rows_n = int((_hygiene.get("futures_shadow") or {}).get("rows") or 0)
        raw_rows_n = int((_hygiene.get("engine_replay") or {}).get("raw_rows") or 0)

        open_for_summary = list(open_trades) if open_trades is not None else list(self.open_trades.values())
        closed_total = len(closed)
        total = closed_total + len(open_for_summary)
        
        wins = sum(1 for t in closed if t.pnl > 0)
        losses = sum(1 for t in closed if t.pnl < 0)
        
        win_rate = (wins / closed_total * 100) if closed_total > 0 else 0.0
        
        # Profit Factor
        total_profit = sum(t.pnl for t in closed if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in closed if t.pnl < 0))
        profit_factor = (total_profit / total_loss) if total_loss > 0 else (total_profit if total_profit > 0 else 1.0)
        
        # Daily/base Sharpe uses sample std; yearly is annualized for display.
        sharpe = 0.0
        if closed_total > 2:
            import numpy as np
            pnl_list = [t.pnl for t in closed]
            avg = np.mean(pnl_list)
            std = np.std(pnl_list, ddof=1)
            sharpe = (avg / std) if std > 0 else 0.0
        sharpe_yearly = annualize_sharpe(sharpe)
            
        # Max Drawdown
        max_dd = 0.0
        peak = 0.0
        running_equity = 0.0
        for t in closed:
            running_equity += t.pnl
            peak = max(peak, running_equity)
            max_dd = max(max_dd, peak - running_equity)

        # Calculate P&L dynamically (Closed + Open Unrealized)
        now_ist = datetime.now(IST)
        if now_ist.hour < 9:
            today_str = (now_ist - timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            today_str = now_ist.strftime('%Y-%m-%d')
        
        def is_option_trade(trade) -> bool:
            trading_symbol = str(getattr(trade, "trading_symbol", "") or "").upper()
            instrument = str(getattr(trade, "instrument", "") or "").upper()
            return (
                str(getattr(trade, "inst_type", "") or "").upper() == "OPT"
                or "CE" in trading_symbol
                or "PE" in trading_symbol
                or "CE" in instrument
                or "PE" in instrument
            )

        total_pnl = sum(t.pnl for t in closed)
        fut_pnl = sum(t.pnl for t in closed if not is_option_trade(t))
        opt_pnl = sum(t.pnl for t in closed if is_option_trade(t))
        
        fut_trades = sum(1 for t in closed if not is_option_trade(t))
        opt_trades = sum(1 for t in closed if is_option_trade(t))

        for ot in open_for_summary:
            try:
                is_opt = is_option_trade(ot)
                if is_opt:
                    opt_trades += 1
                else:
                    fut_trades += 1

                if ot.entry_time.strftime('%Y-%m-%d') == today_str:
                    unrealized = getattr(ot, "unrealized_pnl", 0.0) or 0.0
                    total_pnl += unrealized
                    if is_opt:
                        opt_pnl += unrealized
                    else:
                        fut_pnl += unrealized
            except Exception:
                pass
        
        return {
            "daily_pnl": round(total_pnl, 0),
            "fut_pnl": round(fut_pnl, 0),
            "opt_pnl": round(opt_pnl, 0),
            "total_trades": total,
            "fut_trades": fut_trades,
            "opt_trades": opt_trades,
            "open_count": len(open_for_summary),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "sharpe_ratio_daily": sharpe,
            "sharpe_ratio_yearly": sharpe_yearly,
            "sharpe_ratio_display": format_sharpe_pair(sharpe, sharpe_yearly),
            "max_drawdown": max_dd,
            # Sweep hygiene telemetry (headline above is shadow-free + deduped):
            "futures_shadow": {
                "rows": shadow_rows_n,
                "pnl": round(shadow_pnl, 0),
            },
            "engine_replay": {
                "raw_rows": raw_rows_n,
                "unique_rows": len(closed),
                "excess_rows": replay_excess_rows,
                "excess_pnl": round(replay_excess_pnl, 0),
            },
        }

    # ── signal forensics / flight recorder ────────────────────────────────────

    @staticmethod
    def _forensic_structural_state(trade, prior=None):
        """Derive (fingerprint_tuple, phase, decision) from trade state."""
        status = str(getattr(trade, "status", "") or "").upper()
        execution = str(getattr(trade, "execution_status", "") or "").upper()
        stop = float(getattr(trade, "current_stop", 0.0) or 0.0)
        target = float(getattr(trade, "target", 0.0) or 0.0)
        reason = str(
            getattr(trade, "exit_reason", "")
            or getattr(trade, "execution_error", "")
            or ""
        )
        if status == "CLOSED":
            phase, decision = "EXITED", "CLOSED"
        elif execution in {"ENTRY_PENDING", "PENDING"}:
            phase, decision = "ORDER_PENDING", "PENDING"
        elif execution == "ENTRY_SUBMITTED":
            phase, decision = "ORDER_SUBMITTED", "PENDING"
        elif execution == "REJECTED":
            phase, decision = "ORDER_REJECTED", "REJECTED"
        elif execution in {"EXIT_PENDING", "PARTIAL_EXIT"}:
            phase, decision = (
                "PARTIAL" if execution == "PARTIAL_EXIT" else "MANAGED"
            ), "OPEN"
        elif prior and abs(stop - float(prior[1] or 0.0)) > 1e-9:
            phase, decision = "TRAILING", "OPEN"
        elif prior and abs(target - float(prior[2] or 0.0)) > 1e-9:
            phase, decision = "TARGET_MOVED", "OPEN"
        else:
            phase, decision = "ENTERED", "OPEN"
        return (execution or status, stop, target, reason), phase, decision

    def _seed_trade_forensic_state(self, trade):
        """Seed recovered state so restart never replays historical trades."""
        cache = getattr(self, "_forensic_trade_states", None)
        if not isinstance(cache, dict):
            self._forensic_trade_states = {}
            cache = self._forensic_trade_states
        state, _, _ = self._forensic_structural_state(trade)
        cache[str(trade.id)] = state
        marks = getattr(self, "_forensic_mark_states", None)
        if not isinstance(marks, dict):
            self._forensic_mark_states = {}
            marks = self._forensic_mark_states
        marks[str(trade.id)] = (
            time.monotonic(),
            float(getattr(trade, "current_price", 0.0) or 0.0),
            float(getattr(trade, "pnl", 0.0) or 0.0),
            float(getattr(trade, "peak_pnl", 0.0) or 0.0),
            float(getattr(trade, "max_drawdown", 0.0) or 0.0),
        )

    def _record_trade_forensics(self, trades):
        """Persist meaningful trade-state transitions; never record replay rows."""
        if str(getattr(self, "mode", "") or "").upper() != "REAL":
            return
        cache = getattr(self, "_forensic_trade_states", None)
        if not isinstance(cache, dict):
            self._forensic_trade_states = {}
            cache = self._forensic_trade_states
        mark_cache = getattr(self, "_forensic_mark_states", None)
        if not isinstance(mark_cache, dict):
            self._forensic_mark_states = {}
            mark_cache = self._forensic_mark_states

        mark_interval = 5.0
        mark_price_delta = 0.05
        mark_pnl_delta = 25.0
        now_mono = time.monotonic()
        events = []

        for trade in trades:
            if bool(getattr(trade, "is_ghost", False)):
                continue
            trade_id = str(trade.id)
            prior = cache.get(trade_id)
            state, phase, decision = self._forensic_structural_state(trade, prior)
            structural_change = prior != state

            ltp = float(getattr(trade, "current_price", 0.0) or 0.0)
            pnl = float(
                getattr(trade, "pnl", 0.0)
                or getattr(trade, "unrealized_pnl", 0.0)
                or 0.0
            )
            peak = float(getattr(trade, "peak_pnl", 0.0) or 0.0)
            drawdown = float(getattr(trade, "max_drawdown", 0.0) or 0.0)

            previous_mark = mark_cache.get(trade_id)
            mark_due = False
            if (
                not structural_change
                and str(getattr(trade, "status", "") or "").upper() == "OPEN"
                and previous_mark
            ):
                elapsed = now_mono - float(previous_mark[0] or 0.0)
                moved = (
                    abs(ltp - float(previous_mark[1] or 0.0)) >= mark_price_delta
                    or abs(pnl - float(previous_mark[2] or 0.0)) >= mark_pnl_delta
                    or abs(peak - float(previous_mark[3] or 0.0)) >= mark_pnl_delta
                    or abs(drawdown - float(previous_mark[4] or 0.0)) >= mark_pnl_delta
                )
                mark_due = elapsed >= mark_interval and moved

            if not structural_change and not mark_due:
                continue

            cache[trade_id] = state
            mark_cache[trade_id] = (now_mono, ltp, pnl, peak, drawdown)

            if mark_due:
                phase, decision = "MARK_TO_MARKET", "OPEN"
                reason_text = "Periodic live mark: LTP, P&L, peak or drawdown changed"
                reason_code = "LIVE_MARK"
            else:
                reason_text = str(
                    getattr(trade, "exit_reason", "")
                    or getattr(trade, "execution_error", "")
                    or ""
                )
                execution = str(
                    getattr(trade, "execution_status", "") or ""
                ).upper()
                reason_code = (
                    reason_text.split(":", 1)[0][:96] if reason_text else execution
                )

            try:
                from datetime import datetime, timedelta, timezone
                _IST = timezone(timedelta(hours=5, minutes=30))
                lifecycle_id = str(
                    getattr(trade, "lifecycle_id", "") or trade.id
                )
                exec_type = str(
                    getattr(trade, "exec_type", "A") or "A"
                ).upper()
                quality_verdict = getattr(trade, "quality_verdict", {}) or {}
                if not isinstance(quality_verdict, dict):
                    quality_verdict = {}
                else:
                    # The forensic writer JSON-encodes payload later. Normalising here
                    # keeps observability best-effort even if a caller attached a
                    # datetime or another otherwise non-JSON-native diagnostic value.
                    quality_verdict = json.loads(
                        json.dumps(quality_verdict, default=str)
                    )
                events.append({
                    "session_date": (
                        getattr(trade, "entry_time", datetime.now(_IST))
                        .date().isoformat()
                        if hasattr(getattr(trade, "entry_time", None), "date")
                        else datetime.now(_IST).date().isoformat()
                    ),
                    "event_ts": datetime.now(_IST).isoformat(
                        timespec="milliseconds"
                    ),
                    "lifecycle_id": lifecycle_id,
                    "signal_id": str(trade.id),
                    "instrument": str(
                        getattr(trade, "instrument", "") or ""
                    ).split()[0],
                    "contract": str(
                        getattr(trade, "trading_symbol", "") or ""
                    ),
                    "symbol_token": str(
                        getattr(trade, "symbol_token", "") or ""
                    ),
                    "timeframe": str(
                        getattr(trade, "timeframe", "") or ""
                    ),
                    "direction": str(
                        getattr(trade, "direction", "") or ""
                    ),
                    "phase": phase,
                    "decision": decision,
                    "reason_code": reason_code,
                    "reason_text": reason_text,
                    "source": "trade_manager",
                    "grade": str(getattr(trade, "grade", "") or ""),
                    "confidence": float(
                        getattr(trade, "confidence", 0.0) or 0.0
                    ),
                    "engine": str(
                        getattr(trade, "source_engine", "") or ""
                    ),
                    "lane": str(
                        getattr(trade, "source_lane", "") or ""
                    ),
                    # This is frozen entry provenance, not current UI state. Auto Mode
                    # can be toggled while a position remains open; consulting the
                    # manager would relabel later lifecycle events for the same trade.
                    # `lane` already means the signal's source lane and stays separate.
                    "execution_mode": (
                        "MANUAL" if exec_type in {"M", "MANUAL"} else "AUTO"
                    ),
                    "ltp": ltp,
                    "entry_price": float(
                        getattr(trade, "entry_price", 0.0) or 0.0
                    ),
                    "initial_stop": float(
                        getattr(trade, "initial_stop", 0.0)
                        or getattr(trade, "trailing_stop", 0.0)
                        or 0.0
                    ),
                    "current_stop": float(
                        getattr(trade, "current_stop", 0.0) or 0.0
                    ),
                    "initial_target": float(
                        getattr(trade, "initial_target", 0.0)
                        or getattr(trade, "target", 0.0)
                        or 0.0
                    ),
                    "current_target": float(
                        getattr(trade, "target", 0.0) or 0.0
                    ),
                    "pnl": pnl,
                    "peak_pnl": peak,
                    "drawdown": drawdown,
                    "quote_source": str(
                        getattr(trade, "quote_source", "") or ""
                    ),
                    "quote_verified": bool(
                        getattr(trade, "quote_verified", False)
                    ),
                    "payload": {
                        "trade_id": trade.id,
                        "lifecycle_id": lifecycle_id,
                        "broker_order_id": getattr(
                            trade, "broker_order_id", ""
                        ),
                        "exit_order_id": getattr(
                            trade, "exit_order_id", ""
                        ),
                        "quality_verdict": quality_verdict,
                    },
                })
            except Exception:
                pass

        if events:
            # OFF the caller's thread, matching Scanner._emit_forensic_event, whose
            # docstring states the rule: "DB writes are pushed off the event loop so a
            # slow commit cannot stall the scan cycle."
            #
            # This matters because of where the caller sits. update_trade() runs on the
            # scanner's asyncio loop -- the same loop that runs the 1s scan cycle, the
            # 100ms trade monitor, and every dashboard _call_on_scanner_loop. And
            # append_signal_lifecycle_events takes db.write_lock, so an inline call is a
            # lock-contended sqlite commit on the hot path. That is precisely the shape
            # of the 2026-08-05 regression recorded in
            # tests/test_hft_worker_never_blocks_the_loop_2026_08_05.py, whose visible
            # symptom was a stream of "Error in periodic trades updater: TimeoutError"
            # once _call_on_scanner_loop could no longer be serviced inside its 5s budget.
            #
            # Forensics is observability only; it may never cost a scan cycle, and the
            # dispatch failing is not worth a log line on a hot path.
            batch = list(events)

            def _write() -> None:
                try:
                    from data.sqlite_db import db
                    db.append_signal_lifecycle_events(batch)
                except Exception:
                    return

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(asyncio.to_thread(_write))
            except RuntimeError:
                threading.Thread(
                    target=_write, name="trade-forensics", daemon=True
                ).start()
            except Exception:
                pass
