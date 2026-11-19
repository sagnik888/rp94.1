import asyncio
import copy
import math
import threading
from functools import wraps
from loguru import logger
from datetime import datetime
from typing import Dict, List, Optional
from trading.trade_manager import IST, TradeManager


def _risk_serialized(method):
    """Serialize breaker reset/latch/close decisions across scanner threads."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        lock = getattr(self, "_risk_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._risk_lock = lock
        with lock:
            return method(self, *args, **kwargs)
    return wrapped

class RiskManager:
    """
    Handles all risk-related logic:
    - Daily loss limits (Circuit Breakers)
    - Stop loss & Target management
    - Position sizing
    - Trailing SL logic
    """
    
    def __init__(self, trade_manager: TradeManager, capital_total: float):
        self.trades = trade_manager
        self.capital_total = capital_total
        
        # State
        self.daily_loss_breached = False
        self.fut_loss_breached = False
        self.opt_loss_breached = False
        self._risk_snapshot_unavailable = False
        self._session_date = datetime.now(IST).date()
        self._risk_lock = threading.RLock()
        
        # Risk settings
        self.max_daily_loss_pct = 3.0
        self.risk_fut_pct = 2.0
        self.risk_opt_pct = 8.0
        self.futures_sl_pct = 0.30
        self.options_sl_pct = 15.0
        # 0 = disabled (default): no absolute per-trade rupee backstop — the
        # user's stop caps + segment daily budgets govern. Scanner overrides
        # with the live setting either way.
        self.max_trade_loss_abs = 0.0
        self.circuit_breaker_slippage_bps = 10.0
        
        # Capital allocation
        self.capital_fut = capital_total * 0.5
        self.capital_opt = capital_total * 0.5

    @_risk_serialized
    def update_settings(self, **kwargs):
        """Update risk parameters dynamically"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        logger.info(f"🛡️ RiskManager updated: {kwargs}")

    @staticmethod
    def _is_non_live_trade(trade) -> bool:
        """Return true for virtual/replay rows that must not affect live risk.

        The closed-trade loop below summed EVERY closed row into the daily budget while
        the open-trade loop excluded ghosts, so virtual P&L drove the real circuit
        breaker: ghost profits could mask real losses (breaker fails to fire) and ghost
        losses could halt real trading. The scanner and dashboard already exclude these
        rows from live counts/P&L; this keeps the same boundary in the breaker.
        """
        if trade is None or bool(getattr(trade, "is_ghost", False)):
            return True
        trade_id = str(getattr(trade, "id", "") or "")
        if trade_id.startswith(("H_", "EOD_")):
            return True
        symbol = str(getattr(trade, "trading_symbol", "") or "").upper()
        grade = str(getattr(trade, "grade", "") or "").upper()
        if symbol == "FUT_SHADOW" or "FUT SHADOW" in grade:
            return True
        provenance = str(getattr(trade, "data_provenance", "") or "").upper()
        return provenance in {
            "HISTORICAL", "HISTORY", "HIST", "BACKTEST", "REPLAY",
            "SIM", "SIMULATION", "PAPER_HIST",
        }

    @staticmethod
    def _closed_trade_cache_fingerprint(closed_trades) -> Optional[tuple]:
        """Snapshot every field that can change a closed row's risk contribution.

        Closed history is capped by replacing the oldest row with the newest one, so
        its length is not a mutation counter.  Storing scalar representations also
        catches an in-place P&L correction without retaining a reference to the mutable
        Trade object.  If an unusual row cannot be fingerprinted, bypass the cache and
        let the fail-closed accounting path evaluate it on every pass.
        """
        try:
            return tuple(
                (
                    repr(getattr(trade, "id", "")),
                    bool(getattr(trade, "is_ghost", False)),
                    repr(getattr(trade, "trading_symbol", "")),
                    repr(getattr(trade, "grade", "")),
                    repr(getattr(trade, "data_provenance", "")),
                    repr(getattr(trade, "entry_time", None)),
                    repr(getattr(trade, "inst_type", "OPT")),
                    repr(getattr(trade, "pnl", None)),
                )
                for trade in closed_trades
            )
        except Exception as exc:
            logger.warning(f"Closed-trade risk cache bypassed: {exc}")
            return None

    def _projected_stop_risk(self, trade) -> Optional[float]:
        """Return the worst remaining loss at the configured/protective stop.

        A circuit breaker must not wait for mark-to-market P&L to turn red:
        an open trade can still lose its entire permitted stop distance.  The
        estimate uses the tighter of the live protective stop and the
        configured per-trade stop, then honours the absolute loss cap.
        """
        try:
            entry = abs(float(getattr(trade, "entry_price", 0.0) or 0.0))
            lots = int(getattr(trade, "lots", 0) or 0)
            lot_size = int(getattr(trade, "lot_size", 0) or 0)
            quantity = lots * lot_size
            inst_type = str(getattr(trade, "inst_type", "FUT") or "FUT").upper()
            # An OPTION's instrument_multiplier is the signed Black-Scholes DELTA, a
            # spot<->premium conversion factor -- NOT a contract-size multiplier. The
            # rupee value of a bought option is premium x quantity, full stop, which is
            # why every other rupee path pins it to 1.0 for OPT: entry_risk_policy:90,
            # trade_manager:552 (gross = (current - entry) * sized, no multiplier),
            # scanner's profit-lock and exit paths, and trade_accounting, which applies a
            # multiplier only in its FUT branch.
            #
            # Using |delta| here scaled projected risk by about 0.5. Measured against a
            # real row: BANKNIFTY CE entry 840.60, stop 806.98, 1 lot of 30, delta 0.5729
            # -> true stop risk Rs.1,008.60, computed Rs.577.80. A 43% understatement, so
            # the daily OPT loss budget admitted roughly twice the exposure configured.
            if inst_type == "OPT":
                multiplier = 1.0
            else:
                raw_multiplier = getattr(trade, "instrument_multiplier", 1.0)
                multiplier = abs(
                    float(1.0 if raw_multiplier is None else raw_multiplier)
                )
            if (
                entry <= 0
                or quantity <= 0
                or multiplier <= 0
                or not all(math.isfinite(value) for value in (entry, multiplier))
            ):
                return None

            sl_pct = self.futures_sl_pct if inst_type == "FUT" else self.options_sl_pct
            configured_risk = entry * quantity * multiplier * max(0.0, float(sl_pct or 0.0)) / 100.0

            # current_stop is the TRAILED stop; trailing_stop is documented as "Initial
            # trailing stop at entry" and is never rewritten after open_trade (the
            # ratchet writes current_stop). Reading the entry stop here projected every
            # open trade at its full original risk even after the stop had trailed into
            # profit, so projected_open_pnl looked far worse than reality and the daily
            # loss budget blocked new entries against risk that no longer existed.
            # trade_manager.py already resolves the stop this way.
            stop = getattr(trade, "current_stop", None)
            if not stop:
                stop = getattr(trade, "trailing_stop", None)
            if not stop:
                stop = getattr(trade, "stop", None)
            try:
                stop = float(stop)
            except (TypeError, ValueError):
                stop = 0.0
            if not math.isfinite(stop):
                return None

            # A UTS widening request can reach the exchange before its acknowledgement
            # is lost. During that specific recovery, current_stop remains the tighter
            # software stop and protective_trigger holds the conservative broker bound.
            # Reserve the actual possible exposure until protection is reconciled.
            verdict = getattr(trade, "quality_verdict", {}) or {}
            checks = verdict.get("checks", {}) if isinstance(verdict, dict) else {}
            uncertain_uts_extension = (
                inst_type == "OPT"
                and str(getattr(trade, "source_lane", "") or "").upper() == "UTS"
                and isinstance(verdict, dict) and verdict.get("policy_version") == "uts_native.v1"
                and isinstance(checks, dict) and checks.get("uts_stop_extension_attempted")
                and checks.get("uts_stop_extension_uncertain")
                and str(getattr(trade, "execution_status", "") or "").upper() == "RECOVERY_REQUIRED"
            )
            if uncertain_uts_extension:
                protective = float(getattr(trade, "protective_trigger", 0.0) or 0.0)
                if not math.isfinite(protective) or protective <= 0:
                    return None
                stop = min(stop, protective) if stop > 0 else protective

            # OPTIONS ARE ALWAYS BOUGHT, so the adverse move is always a FALL in premium
            # regardless of what `direction` says. A bearish view is a bought PE, and this
            # system stores it as direction="SHORT" while its stop sits BELOW entry: the
            # live book holds exactly two OPT shapes, 54 LONG/CE and 26 SHORT/PE, and
            # every PE row has stop < entry.
            #
            # Reading `direction` alone therefore computed max(0, stop - entry) = 0 for
            # EVERY bought PE. Measured on real rows: SENSEX PE projected 0.00 vs a true
            # Rs.1,134 of stop risk; BANKNIFTY PE 0.00 vs Rs.3,113; MIDCPNIFTY PE 0.00 vs
            # Rs.3,600. Half the option book was invisible to the forward-looking half of
            # the circuit breaker, which exists precisely so it does "not wait for
            # mark-to-market P&L to turn red".
            #
            # Every other risk surface branches on inst_type FIRST -- entry_risk_policy:88,
            # trade_manager:552, scanner's _trade_like_price_direction. This was the only
            # one that did not.
            direction = str(getattr(trade, "direction", "LONG") or "LONG").upper()
            if stop > 0:
                if inst_type == "OPT":
                    stop_distance = max(0.0, entry - stop)
                else:
                    stop_distance = (
                        max(0.0, entry - stop) if direction == "LONG" else max(0.0, stop - entry)
                    )
                actual_risk = stop_distance * quantity * multiplier
                stop_risk = actual_risk if uncertain_uts_extension else min(configured_risk, actual_risk)
            else:
                stop_risk = configured_risk

            cap = max(0.0, float(self.max_trade_loss_abs or 0.0))
            return min(stop_risk, cap) if cap > 0 and not uncertain_uts_extension else stop_risk
        except Exception as exc:
            logger.warning(f"Unable to estimate projected stop risk: {exc}")
            return None

    @_risk_serialized
    def check_circuit_breaker(self) -> bool:
        """
        Hard stop if daily loss exceeds limits.
        Returns True if system should HALT.
        """
        self.reset_for_session(datetime.now(IST).date())
        if self.capital_total <= 0: return False
        
        today = datetime.now(IST).date()
        fut_pnl = 0.0
        opt_pnl = 0.0
        fut_projected_pnl = 0.0
        opt_projected_pnl = 0.0
        
        # Calculate Realized + Unrealized per segment
        # Audit #8: never silently drop a trade from the daily sum — fail closed
        # by charging against its segment so the breaker still fires. When the
        # absolute per-trade cap is DISABLED (0), charge the segment's FULL
        # daily budget instead: one unparseable trade halts that segment, so
        # disabling the rupee backstop can never turn this into fail-open.
        abs_cap = max(0.0, float(self.max_trade_loss_abs or 0.0))

        def _fail_closed_charge_for(inst_type: str) -> float:
            if abs_cap > 0:
                return abs_cap
            if str(inst_type or "").upper() == "FUT":
                return max(0.0, self.capital_fut * float(self.risk_fut_pct or 0.0) / 100.0)
            return max(0.0, self.capital_opt * float(self.risk_opt_pct or 0.0) / 100.0)

        def _charge_fail_closed(inst_type: str, reason: str) -> None:
            nonlocal fut_pnl, opt_pnl, fut_projected_pnl, opt_projected_pnl
            fail_closed_charge = _fail_closed_charge_for(inst_type)
            charge = -fail_closed_charge if fail_closed_charge > 0 else 0.0
            logger.error(
                f"RiskManager PnL parse failure ({reason}); "
                f"counting {inst_type} as {charge:,.0f} (fail-closed)."
            )
            if str(inst_type or "").upper() == "FUT":
                fut_pnl += charge
                fut_projected_pnl += charge
            else:
                opt_pnl += charge
                opt_projected_pnl += charge

        # Take one immutable view for this complete risk pass.  Reconciliation
        # can move a Trade from open -> closed at the
        # same time; snapshots taken in separate critical sections can count it
        # twice or miss it entirely.
        manager_lock = getattr(self.trades, "_lock", None)
        try:
            if manager_lock:
                with manager_lock:
                    closed_snapshot = [copy.copy(t) for t in self.trades.closed_trades]
                    open_snapshot = [copy.copy(t) for t in self.trades.open_trades.values()]
            else:
                closed_snapshot = [copy.copy(t) for t in self.trades.closed_trades]
                open_snapshot = [copy.copy(t) for t in self.trades.open_trades.values()]
        except Exception as exc:
            if not getattr(self, "_risk_snapshot_unavailable", False):
                logger.critical(f"Risk snapshot failed. New entries blocked: {exc}")
            self._risk_snapshot_unavailable = True
            return True
        self._risk_snapshot_unavailable = False

        # 1. Closed Trades for today (Cached)
        current_count = len(closed_snapshot)
        closed_fingerprint = self._closed_trade_cache_fingerprint(closed_snapshot)
        cache_valid = (
            closed_fingerprint is not None
            and getattr(self, "_cached_cb_date", None) == today
            and getattr(self, "_cached_cb_count", -1) == current_count
            and getattr(self, "_cached_cb_fingerprint", None) == closed_fingerprint
        )
        if cache_valid:
            fut_pnl += self._cached_cb_fut
            fut_projected_pnl += self._cached_cb_fut
            opt_pnl += self._cached_cb_opt
            opt_projected_pnl += self._cached_cb_opt
        else:
            cached_fut = 0.0
            cached_opt = 0.0
            cacheable = closed_fingerprint is not None
            for t in closed_snapshot:
                if self._is_non_live_trade(t):
                    continue
                try:
                    et = t.entry_time.astimezone(IST) if t.entry_time.tzinfo else IST.localize(t.entry_time)
                    if et.date() == today:
                        if t.inst_type == "FUT":
                            cached_fut += t.pnl
                        else:
                            cached_opt += t.pnl
                except Exception as exc:
                    cacheable = False
                    _charge_fail_closed(getattr(t, "inst_type", "OPT"), f"closed trade: {exc}")

            if cacheable:
                self._cached_cb_date = today
                self._cached_cb_count = current_count
                self._cached_cb_fingerprint = closed_fingerprint
                self._cached_cb_fut = cached_fut
                self._cached_cb_opt = cached_opt
            else:
                # Never reuse a partial sum after fail-closed parsing was required.
                self._cached_cb_date = None
                self._cached_cb_count = -1
                self._cached_cb_fingerprint = None

            fut_pnl += cached_fut
            fut_projected_pnl += cached_fut
            opt_pnl += cached_opt
            opt_projected_pnl += cached_opt
        # 2. Open Trades (Unrealized)
        for t in open_snapshot:
            if getattr(t, "is_ghost", False):
                continue
            try:
                et = t.entry_time.astimezone(IST) if t.entry_time.tzinfo else IST.localize(t.entry_time)
                if et.date() == today:
                    open_pnl = float(getattr(t, 'unrealized_pnl', 0) or 0.0)
                    exit_price = float(getattr(t, "current_price", 0.0) or getattr(t, "entry_price", 0.0) or 0.0)
                    quantity = max(1, int(getattr(t, "lots", 1) or 1) * int(getattr(t, "lot_size", 1) or 1))
                    # Same delta-is-not-a-contract-multiplier rule as
                    # _projected_stop_risk above: an OPT row's instrument_multiplier is
                    # the signed Black-Scholes delta, so |delta| was shrinking the
                    # option slippage buffer by roughly half.
                    multiplier = (
                        1.0
                        if str(getattr(t, "inst_type", "FUT") or "FUT").upper() == "OPT"
                        else abs(float(getattr(t, "instrument_multiplier", 1.0) or 1.0))
                    )
                    slippage_buffer = (
                        exit_price
                        * quantity
                        * multiplier
                        * max(0.0, float(self.circuit_breaker_slippage_bps or 0.0))
                        / 10000.0
                    )
                    open_pnl -= slippage_buffer
                    projected_stop_risk = self._projected_stop_risk(t)
                    if projected_stop_risk is None:
                        raise ValueError("projected stop risk is invalid")
                    projected_open_pnl = min(
                        open_pnl,
                        -(projected_stop_risk + slippage_buffer),
                    )
                    if t.inst_type == "FUT":
                        fut_pnl += open_pnl
                        fut_projected_pnl += projected_open_pnl
                    else:
                        opt_pnl += open_pnl
                        opt_projected_pnl += projected_open_pnl
            except Exception as exc:
                _charge_fail_closed(getattr(t, "inst_type", "OPT"), f"open trade: {exc}")

        # Unreconciled exposure blocks entry through entry_block_reason().
        # Circuit breakers use the managed book's P&L and projected stop risk.
        # Calculate limits in absolute terms
        fut_limit = -(self.capital_fut * (self.risk_fut_pct / 100.0))
        opt_limit = -(self.capital_opt * (self.risk_opt_pct / 100.0))
        global_limit = -(self.capital_total * (self.max_daily_loss_pct / 100.0))
        
        total_pnl = fut_pnl + opt_pnl
        projected_total_pnl = fut_projected_pnl + opt_projected_pnl
        
        # ── 🚨 GLOBAL BREACH 🚨 ──
        if global_limit < 0 and projected_total_pnl <= global_limit and not self.daily_loss_breached:
            self.daily_loss_breached = True
            msg = (
                "💀 GLOBAL DAILY LOSS LIMIT BREACHED: "
                f"actual={total_pnl:,.0f}, projected-at-stop={projected_total_pnl:,.0f}"
            )
            logger.error(msg)
            if hasattr(self.trades, '_log_ui'): self.trades._log_ui(msg, "error")
            self.halt_trading("GLOBAL_DAILY_LOSS_LIMIT")
            return True
            
        # ── 🚨 FUT BREACH 🚨 ──
        if fut_limit < 0 and fut_projected_pnl <= fut_limit and not self.fut_loss_breached:
            self.fut_loss_breached = True
            msg = (
                "🚨 FUT DAILY LOSS LIMIT BREACHED: "
                f"actual={fut_pnl:,.0f}, projected-at-stop={fut_projected_pnl:,.0f}"
            )
            logger.warning(msg)
            if hasattr(self.trades, '_log_ui'): self.trades._log_ui(msg, "warning")
            self.close_segment_trades("FUT", "FUT_DAILY_LOSS_LIMIT")
            
        # ── 🚨 OPT BREACH 🚨 ──
        if opt_limit < 0 and opt_projected_pnl <= opt_limit and not self.opt_loss_breached:
            self.opt_loss_breached = True
            msg = (
                "🚨 OPT DAILY LOSS LIMIT BREACHED: "
                f"actual={opt_pnl:,.0f}, projected-at-stop={opt_projected_pnl:,.0f}"
            )
            logger.warning(msg)
            if hasattr(self.trades, '_log_ui'): self.trades._log_ui(msg, "warning")
            self.close_segment_trades("OPT", "OPT_DAILY_LOSS_LIMIT")
            
        return self.daily_loss_breached

    def halt_trading(self, reason: str):
        """Emergency stop: Close ALL positions and disable entry"""
        logger.error(f"🛑 HALTING ALL TRADING: {reason}")
        manager_lock = getattr(self.trades, "_lock", None)
        if manager_lock:
            with manager_lock:
                snapshot = list(self.trades.open_trades.items())
        else:
            snapshot = list(self.trades.open_trades.items())
        for tid, t in snapshot:
            if str(getattr(t, "execution_status", "") or "").upper() == "RECOVERY_REQUIRED":
                self.trades.emergency_flatten_trade(t, reason)
                continue
            self.trades.close_trade(tid, t.current_price, reason)

    def close_segment_trades(self, segment: str, reason: str):
        """Close all positions in a specific segment (FUT or OPT)"""
        logger.warning(f"🚫 Closing all {segment} positions: {reason}")
        manager_lock = getattr(self.trades, "_lock", None)
        if manager_lock:
            with manager_lock:
                snapshot = list(self.trades.open_trades.items())
        else:
            snapshot = list(self.trades.open_trades.items())
        for tid, t in snapshot:
            if str(getattr(t, "execution_status", "") or "").upper() == "RECOVERY_REQUIRED":
                if t.inst_type == segment:
                    self.trades.emergency_flatten_trade(t, reason)
                continue
            if t.inst_type == segment:
                self.trades.close_trade(tid, t.current_price, reason)

    def can_trade(self, inst_type: str) -> bool:
        """Check if trading is allowed for a segment"""
        return not self.entry_block_reason(inst_type)

    @_risk_serialized
    def entry_block_reason(self, inst_type: str) -> str:
        self.reset_for_session(datetime.now(IST).date())
        if not getattr(self.trades, "_trade_state_loaded", True):
            return "Trade ledger restore failed"
        if getattr(self, "_risk_snapshot_unavailable", False):
            return "Risk position snapshot unavailable"
        if getattr(self.trades, "orphan_positions", None):
            return "Broker-only orphan position requires reconciliation"
        if (self.daily_loss_breached or
                inst_type == "FUT" and self.fut_loss_breached or
                inst_type == "OPT" and self.opt_loss_breached):
            return f"Risk circuit breaker latched for {inst_type}"
        return ""

    @_risk_serialized
    def reset_for_session(self, session_date) -> bool:
        """Reset latched daily breakers exactly once when the session date changes."""
        if self._session_date == session_date:
            return False
        self._session_date = session_date
        self.daily_loss_breached = False
        self.fut_loss_breached = False
        self.opt_loss_breached = False
        logger.info(f"Risk circuit breakers reset for session {session_date}.")
        return True
