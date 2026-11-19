"""Prebuilt dashboard/API snapshots for the scanner."""

from __future__ import annotations

import copy
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from loguru import logger

from config.settings import get_settings
from engine.scanner_historical_results import build_historical_trade_result
from engine.trade_row_adapter import (
    accounting_pnl,
    hygiene_closed_rows,
    is_engine_replay_row,
    is_option_dashboard_row,
    is_simulated_history_row,
    normalize_dashboard_accounting_row,
    reconcile_legacy_trade_row,
    trade_row_event_seconds,
    trade_row_session_date,
)
from trading.performance_metrics import annualize_sharpe, format_sharpe_pair
from trading.trade_manager import IST


class ScannerDashboardCache:
    """Store lightweight snapshots so REST APIs do not rebuild heavy payloads."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {}
        self._trades: Dict[str, Any] = {"open": [], "closed": [], "signals": [], "summary": {}}
        self._diagnostics: Dict[str, Any] = {}
        self._updated_at = 0.0

    def update(self, results: Dict[str, Any] | None, *, detached: bool = False) -> None:
        if not isinstance(results, dict):
            return
        # Freeze scanner-owned mutable dictionaries/deques before publishing
        # across the ScannerEventLoop ↔ Uvicorn thread boundary.  The dashboard
        # only ever sees this detached generation, never a partially mutated
        # live object graph.
        #
        # ``detached`` is for a caller that already rebuilt every container itself
        # (dashboard.server's trades refresh runs make_serializable over the whole
        # state). Deep-copying that again on the scanner loop was 17% of all GIL time
        # on 2026-09-17 and timed out the trades updater every few seconds.
        if detached:
            snapshot = results
        else:
            try:
                snapshot = copy.deepcopy(results)
            except Exception as exc:
                logger.warning(f"Dashboard snapshot freeze skipped; retaining prior generation: {exc}")
                return
        with self._lock:
            self._state = snapshot
            if isinstance(snapshot.get("trades"), dict):
                self._trades = snapshot["trades"]
            if isinstance(snapshot.get("diagnostics"), dict):
                self._diagnostics = snapshot["diagnostics"]
            self._updated_at = time.time()

    def update_trades(
        self,
        trades: Dict[str, Any],
        prepare: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> bool:
        """Publish fresh, already-detached trades into the current state generation.

        The dashboard trades heartbeat used to rebuild and re-serialize the whole state
        (diagnostics, config, the trades twice) on the scanner loop every second just to
        refresh this one key; on 2026-10-06 that held the loop 96% of the time. The new
        generation is a shallow copy that shares every other part with the previous one,
        so ``prepare`` must replace, never edit in place, anything it changes. Returns
        False and publishes nothing when no state exists yet or ``prepare`` declines.
        """
        if not isinstance(trades, dict):
            return False
        with self._lock:
            if not self._state:
                return False
            state = dict(self._state)
            state["trades"] = trades
            if prepare is not None and not prepare(state):
                return False
            self._state = state
            self._trades = trades
            self._updated_at = time.time()
        return True

    def state(self) -> Dict[str, Any]:
        with self._lock:
            return self._state or {}

    def trades(self) -> Dict[str, Any]:
        with self._lock:
            return self._trades or {"open": [], "closed": [], "signals": [], "summary": {}}

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            return self._diagnostics or {}

    def cache_age_ms(self) -> int | None:
        with self._lock:
            if not self._updated_at:
                return None
            return round(max(0.0, time.time() - self._updated_at) * 1000.0)

    def status(self, scanner: Any, connected_clients: int) -> Dict[str, Any]:
        with self._lock:
            state = self._state or {}
            gateway_status = state.get("gateway_status") or {}
            config = state.get("config") or {}
            latency = state.get("latency")
        fyers_auth = {}
        data = getattr(scanner, "data", None)
        if data is not None and hasattr(data, "get_fyers_auth_status"):
            try:
                fyers_auth = data.get_fyers_auth_status()
            except Exception:
                fyers_auth = {}
        last_scan = getattr(scanner, "last_scan_time", None)
        freshness = {}
        if scanner is not None and hasattr(scanner, "_scanner_freshness_payload"):
            try:
                freshness = scanner._scanner_freshness_payload()
            except Exception:
                freshness = {}
        return {
            "status": "running",
            "connected_clients": int(connected_clients),
            "scanner_running": bool(getattr(scanner, "is_running", False)),
            "mode": getattr(scanner, "mode", None),
            "config": config or (
                scanner.get_broadcast_config()
                if scanner is not None and hasattr(scanner, "get_broadcast_config")
                else {}
            ),
            "last_scan": last_scan.isoformat() if last_scan else None,
            "scan_count": int(getattr(scanner, "scan_count", 0) or 0),
            "latency": latency,
            "pid": __import__("os").getpid(),
            "gateway_status": gateway_status,
            "fyers_auth": fyers_auth,
            "cache_age_ms": self.cache_age_ms(),
            "scanner_freshness": freshness,
            "calibration": (
                scanner.outcome_calibration.validation_status()
                if scanner is not None and hasattr(scanner, "outcome_calibration")
                else {}
            ),
            "timestamp": datetime.now().isoformat(),
        }


class ScannerTradePayloadBuilder:
    """Build dashboard trade payloads from scanner-owned state."""

    def __init__(self, scanner: Any) -> None:
        self.scanner = scanner

    def build(self) -> Dict[str, Any]:
        """Return trade-manager rows plus the persisted manual/live signal ledger."""
        scanner = self.scanner
        payload = scanner.trades.get_dashboard_payload(
            is_historical=(scanner.mode == "HISTORICAL"),
            backtest_days=scanner.backtest_days,
            inst_pref=getattr(scanner, "inst_pref", "AUTO"),
        )
        recalc_status = str(
            (getattr(scanner, "_recalculation_status", {}) or {}).get("status", "idle")
            or "idle"
        )
        historical_complete = (
            scanner.mode != "HISTORICAL"
            or (
                int(getattr(scanner, "scan_count", 0) or 0) > 0
                and not bool(getattr(scanner, "_calculation_lock", False))
                and not bool(getattr(scanner, "is_calculating", False))
                and not bool(getattr(scanner, "_historical_rebuild_pending", True))
                and int(getattr(scanner, "_historical_completed_simulation_id", 0) or 0)
                == int(getattr(scanner, "simulation_id", 0) or 0)
                and recalc_status not in {
                    "queued",
                    "waiting_initial_setup",
                    "worker_running",
                    "applying",
                    "running",
                    "awaiting_scan",
                }
            )
        )
        payload["meta"] = {
            "mode": scanner.mode,
            "session_day": str(scanner._get_current_session_day()) if scanner.mode == "REAL" else "",
            "backtest_days": int(getattr(scanner, "backtest_days", 1) or 1),
            "simulation_id": int(getattr(scanner, "simulation_id", 0) or 0),
            "complete": bool(historical_complete),
            "recalculation_status": recalc_status,
            "inst_pref": str(getattr(scanner, "inst_pref", "AUTO") or "AUTO").upper(),
            "scanner_engine": str(getattr(get_settings(), "scanner_engine", "utbot") or "utbot").lower(),
            "ut_concurrency_guard": scanner._setting_enabled(getattr(get_settings(), "ut_concurrency_guard", True), True),
            "futures_shadow_enabled": scanner._setting_enabled(
                getattr(get_settings(), "futures_shadow_enabled", False),
                False,
            ),
            "grade_preference": str(getattr(get_settings(), "signal_grade_preference", "auto") or "auto"),
            "timeframe_entry_policy": str(getattr(get_settings(), "ut_timeframe_entry_policy", "PRIMARY_15") or "PRIMARY_15").upper(),
            "strict_entry_timeframes": bool(getattr(get_settings(), "strict_entry_timeframes", False)),
            "open_rows": len(payload.get("open", []) or []),
            "closed_rows": len(payload.get("closed", []) or []),
            "restored_signal_rows": 0,
            "source": "trade_manager",
        }

        def inst_pref_match(c) -> bool:
            pref = (getattr(scanner, "inst_pref", "AUTO") or "AUTO").upper()
            if pref in ("AUTO", "HYBRID"): return True
            inst_type = getattr(c, "inst_type", "FUT") if not isinstance(c, dict) else c.get("inst_type", "FUT")
            if pref == "FUT" and inst_type == "FUT": return True
            if pref == "OPT" and inst_type in ("OPT", "CE", "PE"): return True
            return False

        def real_payload_row_allowed(row: Dict[str, Any]) -> bool:
            if is_simulated_history_row(row):
                return False
            day = trade_row_session_date(row)
            if day is None:
                return True
            session_day = datetime.fromisoformat(str(scanner._get_current_session_day())).date()
            return day == session_day

        def historical_payload_row_allowed(row: Dict[str, Any]) -> bool:
            day = trade_row_session_date(row)
            if day is None:
                return True
            return day <= scanner.historical_session_cutoff_date()

        payload["closed"] = [reconcile_legacy_trade_row(r) for r in payload.get("closed", []) or []]
        if scanner.mode == "REAL":
            payload["open"] = [r for r in payload.get("open", []) or [] if real_payload_row_allowed(r)]
            payload["closed"] = [r for r in payload.get("closed", []) or [] if real_payload_row_allowed(r)]
        elif scanner.mode == "HISTORICAL":
            payload["open"] = [r for r in payload.get("open", []) or [] if historical_payload_row_allowed(r)]
            payload["closed"] = [r for r in payload.get("closed", []) or [] if historical_payload_row_allowed(r)]
            # BACKTEST P&L must describe this engine replay only. TradeManager also
            # restores broker/session-ledger rows from SQLite; historical recalculation
            # preserves those records for audit, but mixing them into a fresh replay
            # makes build comparisons depend on each directory's database contents.
            # Keep ledger facts persisted; exclude them only from replay accounting.
            historical_open = list(payload.get("open", []) or [])
            historical_closed = list(payload.get("closed", []) or [])
            excluded_open = [row for row in historical_open if not is_engine_replay_row(row)]
            excluded_closed = [row for row in historical_closed if not is_engine_replay_row(row)]
            payload["open"] = [row for row in historical_open if is_engine_replay_row(row)]
            payload["closed"] = [row for row in historical_closed if is_engine_replay_row(row)]
            payload["meta"] = {
                **payload.get("meta", {}),
                "accounting_source": "current_engine_replay",
                "excluded_session_ledger_rows": len(excluded_open) + len(excluded_closed),
                "excluded_session_ledger_pnl": round(
                    sum(accounting_pnl(row) for row in excluded_open + excluded_closed),
                    2,
                ),
            }

        def rebuild_display_accounting(accounting_lane: Optional[str] = None) -> None:
            """Keep headline metrics identical to the final reconciled dashboard rows.

            Always apply shared shadow-free + exact-dedup hygiene so a prior
            TradeManager.get_summary() cannot be overwritten with inflated
            historical FUT_SHADOW / multi-engine replay totals.
            """
            lane = str(
                accounting_lane
                or ("REPLAY" if scanner.mode == "HISTORICAL" else "LIVE")
            ).upper()
            is_replay = lane == "REPLAY"
            is_signal_tracker = lane == "SIGNAL_TRACKER"
            pnl_label = (
                "BACKTEST P&L"
                if is_replay
                else "SIGNAL P&L"
                if is_signal_tracker
                else "DAILY P&L"
            )
            raw_closed = [
                normalize_dashboard_accounting_row(row)
                for row in payload.get("closed", []) or []
            ]
            open_rows = [
                normalize_dashboard_accounting_row(row)
                for row in payload.get("open", []) or []
            ]
            closed_rows, hygiene = hygiene_closed_rows(raw_closed)
            payload["closed"] = closed_rows
            payload["open"] = open_rows
            # Refused entries stay listed for audit but are not trades: no order filled
            # and no charge was paid. Every metric below reads the filtered rows.
            excluded_closed = [row for row in closed_rows if row.get("accounting_excluded")]
            closed_rows = [row for row in closed_rows if not row.get("accounting_excluded")]
            closed_pnls = [accounting_pnl(row) for row in closed_rows]
            open_pnls = [accounting_pnl(row) for row in open_rows]
            total_pnl = sum(closed_pnls) + sum(open_pnls)
            vector_backend = "cpu"
            vector_stats = None
            try:
                from engine.gpu_accelerator import get_gpu_accelerator
                vector_stats = get_gpu_accelerator().vector_stats(closed_pnls)
                vector_backend = vector_stats.backend
            except Exception:
                vector_stats = None
            wins = sum(1 for value in closed_pnls if value > 0)
            losses = sum(1 for value in closed_pnls if value < 0)
            profits = sum(value for value in closed_pnls if value > 0)
            losses_abs = abs(sum(value for value in closed_pnls if value < 0))
            profit_factor = profits / losses_abs if losses_abs > 0 else (profits if profits > 0 else 1.0)

            sharpe = 0.0
            if len(closed_pnls) > 2:
                if vector_stats is not None:
                    std = float(vector_stats.std)
                    mean = float(vector_stats.mean)
                else:
                    pnl_array = np.asarray(closed_pnls, dtype=float)
                    std = float(np.std(pnl_array, ddof=1)) if len(pnl_array) > 1 else 0.0
                    mean = float(np.mean(pnl_array))
                sharpe = float(mean / std) if std > 0 else 0.0
            sharpe_yearly = annualize_sharpe(sharpe)
            sharpe_display = format_sharpe_pair(sharpe, sharpe_yearly)

            chronological = sorted(
                closed_rows,
                key=trade_row_event_seconds,
            )
            running = 0.0
            peak = 0.0
            max_drawdown = 0.0
            current_drawdown = 0.0
            for row in chronological:
                running += accounting_pnl(row)
                peak = max(peak, running)
                max_drawdown = max(max_drawdown, peak - running)
            current_drawdown = max(0.0, peak - running)

            def parse_event_seconds(row: Dict[str, Any], prefer_exit: bool = False) -> float:
                keys = ("exit_timestamp", "entry_timestamp") if prefer_exit else ("entry_timestamp", "exit_timestamp")
                for key in keys:
                    raw = row.get(key)
                    if raw in (None, ""):
                        continue
                    try:
                        value = float(raw)
                        return value / 1000.0 if value > 1e12 else value
                    except Exception:
                        pass
                    try:
                        parsed = datetime.fromisoformat(str(raw))
                        if parsed.tzinfo is None:
                            parsed = IST.localize(parsed)
                        else:
                            parsed = parsed.astimezone(IST)
                        return parsed.timestamp()
                    except Exception:
                        continue
                return 0.0

            capital_total = float(getattr(scanner, "capital_total", 0.0) or getattr(get_settings(), "capital_total", 0.0) or 0.0)
            capital_fut = float(getattr(scanner, "capital_fut", 0.0) or getattr(get_settings(), "capital_fut", 0.0) or 0.0)
            capital_opt = float(getattr(scanner, "capital_opt", 0.0) or getattr(get_settings(), "capital_opt", 0.0) or 0.0)

            def trade_allocation(row: Dict[str, Any]) -> float:
                qty = int(row.get("lots") or 0) * int(row.get("lot_size") or 0)
                entry = float(row.get("entry_price") or 0.0)
                return max(0.0, entry * qty)

            active_allocation = sum(trade_allocation(row) for row in open_rows)

            def compact_trade(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
                if not row:
                    return {}
                return {
                    "id": row.get("id", ""),
                    "instrument": row.get("instrument", "--"),
                    "inst_type": row.get("inst_type", "--"),
                    "timeframe": row.get("timeframe", "--"),
                    "direction": row.get("direction", "--"),
                    "pnl": round(accounting_pnl(row), 2),
                    "entry_time": row.get("entry_time", "--"),
                    "exit_time": row.get("exit_time", "--"),
                    "exit_reason": row.get("exit_reason", "--"),
                }

            def bucketize_detail(rows: List[Dict[str, Any]], key_fn: Callable[[Dict[str, Any]], str]) -> Dict[str, Any]:
                buckets: Dict[str, Dict[str, Any]] = {}
                for row in rows:
                    name = key_fn(row) or "--"
                    value = accounting_pnl(row)
                    bucket = buckets.setdefault(name, {"count": 0, "pnl": 0.0, "wins": 0, "losses": 0})
                    bucket["count"] += 1
                    bucket["pnl"] += value
                    if value > 0:
                        bucket["wins"] += 1
                    elif value < 0:
                        bucket["losses"] += 1
                for bucket in buckets.values():
                    count = int(bucket["count"] or 0)
                    bucket["pnl"] = round(float(bucket["pnl"] or 0.0), 2)
                    bucket["win_rate"] = round((bucket["wins"] / count) * 100.0, 1) if count else 0.0
                    bucket["avg_pnl"] = round(bucket["pnl"] / count, 2) if count else 0.0
                return dict(sorted(buckets.items(), key=lambda item: abs(item[1]["pnl"]), reverse=True))

            def period_totals(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
                today = datetime.now(IST).date()
                week_start = today - timedelta(days=today.weekday())
                month_start = today.replace(day=1)
                totals = {"daily": 0.0, "weekly": 0.0, "monthly": 0.0}
                for row in rows:
                    ts = parse_event_seconds(row, prefer_exit=True) or parse_event_seconds(row)
                    if not ts:
                        continue
                    day = datetime.fromtimestamp(ts, tz=IST).date()
                    value = accounting_pnl(row)
                    if day == today:
                        totals["daily"] += value
                    if day >= week_start:
                        totals["weekly"] += value
                    if day >= month_start:
                        totals["monthly"] += value
                return {key: round(value, 2) for key, value in totals.items()}

            hold_minutes = []
            for row in closed_rows:
                entry_ts = parse_event_seconds(row)
                exit_ts = parse_event_seconds(row, prefer_exit=True)
                if entry_ts and exit_ts and exit_ts > entry_ts:
                    hold_minutes.append((exit_ts - entry_ts) / 60.0)
            avg_hold = sum(hold_minutes) / len(hold_minutes) if hold_minutes else 0.0

            largest_winner = max(closed_rows, key=accounting_pnl, default=None)
            largest_loser = min(closed_rows, key=accounting_pnl, default=None)
            max_dd_pct = round((max_drawdown / capital_total) * 100.0, 2) if capital_total > 0 else 0.0
            current_dd_pct = round((current_drawdown / capital_total) * 100.0, 2) if capital_total > 0 else 0.0

            monetary = {
                "accounting_basis": {
                    "lane": lane,
                    "pnl_label": pnl_label,
                    "is_live_money": lane == "LIVE",
                    "description": (
                        "Accepted signal marks only; no broker fill is implied"
                        if is_signal_tracker
                        else "Historical replay; no broker fill is implied"
                        if is_replay
                        else "Trade-manager broker/execution ledger"
                    ),
                },
                "capital": {
                    "configured_demat_capital": round(capital_total, 2),
                    "futures_allocation": round(capital_fut, 2),
                    "options_allocation": round(capital_opt, 2),
                    "active_trade_allocation": round(active_allocation, 2),
                    "free_configured_capital": round(max(0.0, capital_total - active_allocation), 2),
                    "source": "configured_settings",
                },
                "headline": {
                    "win_rate": round((wins / len(closed_rows) * 100.0) if closed_rows else 0.0, 1),
                    "profit_factor": round(profit_factor, 2),
                    "sharpe_ratio": round(sharpe, 2),
                    "sharpe_ratio_daily": round(sharpe, 2),
                    "sharpe_ratio_yearly": round(sharpe_yearly, 2),
                    "sharpe_ratio_display": sharpe_display,
                    "max_drawdown": round(max_drawdown, 2),
                    "max_drawdown_pct": max_dd_pct,
                    "current_drawdown": round(current_drawdown, 2),
                    "current_drawdown_pct": current_dd_pct,
                    "avg_hold_minutes": round(avg_hold, 1),
                },
                "period_pnl": period_totals(closed_rows),
                "largest_winner": compact_trade(largest_winner if largest_winner and accounting_pnl(largest_winner) > 0 else None),
                "largest_loser": compact_trade(largest_loser if largest_loser and accounting_pnl(largest_loser) < 0 else None),
                "breakdowns": {
                    "index": bucketize_detail(closed_rows, lambda row: str(row.get("instrument") or "--").split()[0]),
                    "instrument_type": bucketize_detail(closed_rows, lambda row: str(row.get("inst_type") or "--").upper()),
                    "timeframe": bucketize_detail(closed_rows, lambda row: str(row.get("timeframe") or "--")),
                },
                "live_positions": [
                    {
                        **compact_trade(row),
                        "allocation": round(trade_allocation(row), 2),
                        "target": round(float(row.get("target") or 0.0), 2),
                        "stop": round(float(row.get("trailing_stop") or 0.0), 2),
                        "rr_ratio": round(float(row.get("rr_ratio") or 0.0), 2),
                    }
                    for row in open_rows
                ],
            }

            payload["summary"] = {
                "lane": lane,
                "pnl_label": pnl_label,
                "is_replay": is_replay,
                "is_signal_tracker": is_signal_tracker,
                "pnl_is_money": lane == "LIVE",
                "daily_pnl": round(total_pnl, 2),
                "fut_pnl": round(sum(accounting_pnl(row) for row in closed_rows + open_rows if not is_option_dashboard_row(row)), 2),
                "opt_pnl": round(sum(accounting_pnl(row) for row in closed_rows + open_rows if is_option_dashboard_row(row)), 2),
                "fut_trades": len([row for row in closed_rows + open_rows if not is_option_dashboard_row(row)]),
                "opt_trades": len([row for row in closed_rows + open_rows if is_option_dashboard_row(row)]),
                "total_trades": len(closed_rows) + len(open_rows),
                "open_count": len(open_rows),
                "wins": wins,
                "losses": losses,
                "win_rate": (wins / len(closed_rows) * 100.0) if closed_rows else 0.0,
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe,
                "sharpe_ratio_daily": sharpe,
                "sharpe_ratio_yearly": sharpe_yearly,
                "sharpe_ratio_display": sharpe_display,
                "max_drawdown": max_drawdown,
                "vector_backend": vector_backend,
                "futures_shadow": dict(hygiene.get("futures_shadow") or {}),
                "engine_replay": dict(hygiene.get("engine_replay") or {}),
                "rejected_entries": len(excluded_closed),
            }
            payload["monetary"] = monetary

            def bucketize(key: str) -> Dict[str, Any]:
                buckets: Dict[str, Dict[str, Any]] = {}
                for row in closed_rows:
                    name = str(row.get(key) or "--")
                    if key == "grade":
                        name = name.split()[0]
                    if key == "instrument":
                        name = name.split()[0]
                    bucket = buckets.setdefault(name, {"count": 0, "pnl": 0.0, "wins": 0, "losses": 0})
                    value = accounting_pnl(row)
                    bucket["count"] += 1
                    bucket["pnl"] += value
                    if value > 0:
                        bucket["wins"] += 1
                    elif value < 0:
                        bucket["losses"] += 1
                for bucket in buckets.values():
                    bucket["pnl"] = round(bucket["pnl"], 0)
                    bucket["win_rate"] = round(bucket["wins"] / bucket["count"] * 100.0, 1) if bucket["count"] else 0.0
                return dict(sorted(buckets.items(), key=lambda item: abs(item[1]["pnl"]), reverse=True))

            payload["analytics"] = {
                "by_timeframe": bucketize("timeframe"),
                "by_grade": bucketize("grade"),
                "by_instrument": bucketize("instrument"),
                "by_type": bucketize("inst_type"),
                "by_exit_reason": bucketize("exit_reason"),
                "open_by_type": payload.get("analytics", {}).get("open_by_type", {}),
            }

        rebuild_display_accounting()

        if scanner.mode != "HISTORICAL":
            signal_rows = []
            if scanner._loaded_session_is_current_day():
                instruments = list(getattr(scanner, "active_indices", []) or [])
                for instrument in (getattr(scanner, "_session_trade_candidates", {}) or {}).keys():
                    if instrument not in instruments:
                        instruments.append(instrument)
                session_day = getattr(scanner, "_session_candidate_day", scanner._get_current_session_day())
                for instrument in instruments:
                    candidates = scanner._get_session_trade_candidates(instrument)
                    if hasattr(scanner, "_refresh_trade_candidates"):
                        cfg = (
                            getattr(scanner, "config", {}).get("indices", {}).get(instrument, {})
                            if isinstance(getattr(scanner, "config", {}), dict)
                            else {}
                        )
                        # This build runs on the scanner event loop. A stale
                        # dashboard mark must not turn its heartbeat into broker I/O.
                        scanner._refresh_trade_candidates(candidates, cfg, allow_blocking=False)
                    for candidate in candidates:
                        candidate_ts = getattr(candidate, "signal_timestamp", None)
                        candidate_day = candidate_ts.date().isoformat() if candidate_ts else session_day
                        # Monitoring may retain overnight positions. This dashboard
                        # is explicitly scoped to entries accepted today.
                        matches_session = getattr(scanner, "_candidate_matches_live_session_day", None)
                        if (not matches_session(candidate) if callable(matches_session)
                                else candidate_day != session_day):
                            continue
                        if inst_pref_match(candidate):
                            scanner._normalize_loaded_exit_candidate(candidate)
                            row = normalize_dashboard_accounting_row(
                                scanner._serialize_trade_candidate(candidate)
                            )
                            entry_seconds = trade_row_event_seconds(row)
                            if entry_seconds:
                                entry_day = datetime.fromtimestamp(entry_seconds, tz=IST).date().isoformat()
                                if entry_day != session_day:
                                    continue
                            elif candidate_day != session_day:
                                continue
                            # The session ledger also stores candidates that never
                            # became positions (including TradeManager pre-submit
                            # refusals). Only admitted manual/watch lifecycles belong
                            # in the signal-tracker lane; broker positions come from
                            # TradeManager's open/closed rows below.
                            if not (
                                bool(row.get("accepted_by_gate"))
                                or bool(row.get("watch_only"))
                            ):
                                continue
                            signal_rows.append(row)
            # A broker-backed TradeManager row is authoritative once a signal has
            # executed.  The session tracker may continue following the hypothetical
            # signal after the real position closes; exposing both creates a phantom
            # second trade and a fictional P&L.  Keep unexecuted/watch-only signals.
            manager_lifecycles = {
                str(row.get("lifecycle_id") or "").strip()
                for row in [*(payload.get("open", []) or []), *(payload.get("closed", []) or [])]
                if str(row.get("lifecycle_id") or "").strip()
            }
            if manager_lifecycles:
                signal_rows = [
                    row for row in signal_rows
                    if str(row.get("lifecycle_id") or "").strip() not in manager_lifecycles
                ]
            # One row per signal lifecycle: its exit once completed, its entry while it
            # is still open.  02729d7 hid completed lifecycles so a virtual exit could
            # not read as a second broker trade, which left the panel empty all session
            # on 2026-09-15 while Diagnostics counted that day's signals and exits.  They
            # stay visible as signal-tracker rows instead: an executed lifecycle is
            # dropped above in favour of its broker row, and exit P&L is booked only in
            # the non-money SIGNAL_TRACKER lane below.
            def lifecycle_clock(raw: Any) -> Optional[float]:
                if raw in (None, ""):
                    return None
                if isinstance(raw, datetime):
                    parsed = raw
                else:
                    try:
                        parsed = datetime.fromisoformat(str(raw))
                    except ValueError:
                        return None
                if parsed.tzinfo is None:
                    parsed = IST.localize(parsed)
                return parsed.timestamp()

            exit_rows = [r for r in signal_rows if r.get("action") == "EXIT"]
            exit_clocks: Dict[str, List[Optional[float]]] = {}
            for r in exit_rows:
                exit_clocks.setdefault(str(r.get("lifecycle_id") or ""), []).append(
                    lifecycle_clock(r.get("exit_timestamp"))
                )

            def closed_by_exit(entry: Dict[str, Any]) -> bool:
                # A re-signal on the same candle and contract shares the lifecycle id of
                # the signal that exited before it, so an exit only closes an entry
                # accepted at or before it.  A missing clock counts as closed.
                entered_at = lifecycle_clock(entry.get("entry_timestamp") or entry.get("timestamp"))
                return any(
                    exited_at is None or entered_at is None or exited_at >= entered_at
                    for exited_at in exit_clocks.get(str(entry.get("lifecycle_id") or ""), [])
                )

            active_signals = [
                r for r in signal_rows
                if r.get("action") == "ENTRY" and not closed_by_exit(r)
            ]
            for row in active_signals:
                # A manual signal is an execution opportunity, not an open position.
                # Do not let a mark-to-market estimate become apparent broker P&L.
                row["pnl"] = 0.0
                row["accounting_pnl"] = 0.0
            display_signal_rows = exit_rows + active_signals
            for row in display_signal_rows:
                row["execution_status"] = "SIGNAL_ONLY"
                row["execution_state"] = "NOT_PLACED"
            payload["signals"] = display_signal_rows
            # Instrument snapshots can be older than this ledger. The browser must
            # not reinsert their candidates, even when the current ledger is empty.
            payload["meta"]["signals_authoritative"] = True
            # Explicit session-row contract: consumers that read only open/closed would
            # otherwise see zero trades when the session is carried by signal rows.
            payload["session_rows_total"] = len(display_signal_rows)
            payload["session_closed_total"] = len(exit_rows)
            payload["session_open_total"] = len(active_signals)
            payload["meta"] = {
                **payload.get("meta", {}),
                "restored_signal_rows": len(display_signal_rows),
                "session_rows_total": len(display_signal_rows),
                "session_closed_total": len(exit_rows),
                "session_open_total": len(active_signals),
                "unexecuted_terminal_signal_rows": len(exit_rows),
                "source": "restored_session_signals" if display_signal_rows and not (payload.get("open") or payload.get("closed")) else "mixed",
            }
            # A refused entry is listed but moved no money, so on its own it must not
            # switch the headline to the broker lane.
            has_trade_rows = bool(
                payload.get("open")
                or [row for row in payload.get("closed") or [] if not row.get("accounting_excluded")]
            )
            if display_signal_rows and has_trade_rows:
                # Broker trades own the money headline, but signals taken while Auto was
                # OFF (restart, manual toggle) were silently dropped from it: on
                # 2026-09-17 seven such signals (+Rs.816 in marks) vanished once one
                # broker trade existed, and P&L / TRADES looked frozen. Publish them as a
                # separate, non-money lane instead of either hiding or mixing them.
                signal_accounting = exit_rows + active_signals
                signal_pnls = [accounting_pnl(r) for r in signal_accounting]
                payload["summary"] = {
                    **(payload.get("summary") or {}),
                    "signal_lane": {
                        "pnl_label": "SIGNAL P&L",
                        "pnl_is_money": False,
                        "pnl": round(sum(signal_pnls), 2),
                        "trades": len(signal_accounting),
                        "open_count": len(active_signals),
                        "wins": len([p for p in signal_pnls if p > 0]),
                        "losses": len([p for p in signal_pnls if p < 0]),
                        "description": "Accepted signals not placed with the broker (Auto Mode OFF); marks only",
                    },
                }
            if display_signal_rows and not has_trade_rows:
                active_rows = active_signals
                accounting_rows = exit_rows + active_rows
                pnls = [accounting_pnl(r) for r in accounting_rows]
                wins = len([p for p in pnls if p > 0])
                losses = len([p for p in pnls if p < 0])
                gross_win = sum(p for p in pnls if p > 0)
                gross_loss = abs(sum(p for p in pnls if p < 0))
                payload["summary"] = {
                    **(payload.get("summary") or {}),
                    "daily_pnl": round(sum(pnls), 2),
                    "fut_pnl": round(sum(accounting_pnl(r) for r in accounting_rows if not is_option_dashboard_row(r)), 2),
                    "opt_pnl": round(sum(accounting_pnl(r) for r in accounting_rows if is_option_dashboard_row(r)), 2),
                    "fut_trades": len([r for r in accounting_rows if not is_option_dashboard_row(r)]),
                    "opt_trades": len([r for r in accounting_rows if is_option_dashboard_row(r)]),
                    "total_trades": len(accounting_rows),
                    "open_count": len(active_rows),
                    "wins": wins,
                    "losses": losses,
                    "win_rate": round((wins / len(accounting_rows)) * 100, 1) if accounting_rows else 0.0,
                    "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else (99.99 if gross_win > 0 else 1.0),
                }
                original_open = payload.get("open", [])
                original_closed = payload.get("closed", [])
                try:
                    payload["open"] = active_rows
                    payload["closed"] = exit_rows
                    rebuild_display_accounting(accounting_lane="SIGNAL_TRACKER")
                finally:
                    payload["open"] = original_open
                    payload["closed"] = original_closed
        else:
            build_historical_trade_result(
                scanner=scanner,
                payload=payload,
                inst_pref_match=inst_pref_match,
                reconcile_display_row=reconcile_legacy_trade_row,
                row_event_seconds=trade_row_event_seconds,
            )
        return payload


def build_dashboard_trade_payload(scanner: Any) -> Dict[str, Any]:
    """Return the scanner dashboard trade payload without exposing builder internals."""
    return ScannerTradePayloadBuilder(scanner).build()
