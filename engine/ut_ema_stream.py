"""Ordered native UT/EMA observations outside the throttled candle publication path.

The feed callback only queues accepted observations. One daemon worker advances
native indicator state; event delivery remains on the scanner event loop. This
service never calls a broker and never turns historical replay into an entry.
"""
from __future__ import annotations

import asyncio
import copy
import json
import math
import os
from pathlib import Path
import threading
import time
from collections import deque
from types import SimpleNamespace

import pandas as pd
from loguru import logger

from engine.ut_ema_engine import ENGINE_KEY, market_stamp


class UTEmaTickService:
    def __init__(self, scanner, *, checkpoint_dir=None, capacity=16384):
        self.scanner = scanner
        self.capacity = max(1, int(capacity))
        self.directory = Path(checkpoint_dir) if checkpoint_dir is not None else Path(__file__).resolve().parents[1]/"data_store"/"utema_stream"
        self._condition = threading.Condition()
        self._queue = deque()
        self._closed = False
        self._thread = None
        self._last_observed = {}
        self._loaded = set()
        self._saved_at = {}
        self._delivery_lock = None
        self._pending_history = {}
        self._sequence = 0
        self._journal_position = {}
        self._overflowed = set()
        self._missing_path = {}
        self._watched_owners = set()
        self._next_owner_check = 0.0
        self._next_history_check = {}
        self.health = {}
        scanner._utema_stream_health = self.health
        scanner.mtf._utema_stream_service_installed = True
        for adapter in (getattr(scanner.mtf, "engines", {}) or {}).values():
            if getattr(adapter, "signal_source", "") == ENGINE_KEY:
                adapter._utema_stream_owned = True

    def _adapter(self, instrument, timeframe, *, include_unwatched=False):
        mtf = self.scanner.mtf
        key = f"{instrument}_{timeframe}"
        adapter = (getattr(mtf, "engines", {}) or {}).get(key)
        if (getattr(adapter, "signal_source", "") != ENGINE_KEY and
                (include_unwatched or key in self._watched_owners)):
            adapter = (getattr(mtf, "_retired_utema_engines", {}) or {}).get(key)
        return adapter if getattr(adapter, "signal_source", "") == ENGINE_KEY else None

    def monitor_owner(self, trade, lifecycle_id, entry_time):
        """Register a held owner and read its latest protection without blocking."""
        instrument, timeframe = trade.instrument, trade.timeframe
        if timeframe not in ("5min", "15min"):
            return None
        key = f"{instrument}_{timeframe}"
        adapter = self._adapter(instrument, timeframe, include_unwatched=True)
        if adapter is None:
            from engine.alt_scanners import UTEmaSignalEngine

            adapter = UTEmaSignalEngine()
            adapter._utema_stream_owned = True
            adapter = self.scanner.mtf._retired_utema_engines.setdefault(key, adapter)
        self._watched_owners.add(key)
        return adapter.monitor_snapshot(lifecycle_id, entry_time, wait=False)

    def _prune_closed_owners(self):
        from engine.ut_ema_lane import active_rows

        now = time.monotonic()
        if now < self._next_owner_check:
            return
        self._next_owner_check = now + 1.0
        for key in self._watched_owners.copy():
            instrument, timeframe = key.rsplit("_", 1)
            if not active_rows(self.scanner, instrument, timeframe):
                self._watched_owners.discard(key)

    def _status(self, key, status, detail="", **extra):
        if key in self._overflowed:
            status, detail = "OVERFLOW", "Observation queue exceeded capacity; native entries withheld"
        previous = self.health.get(key, {})
        self.health[key] = {**previous, "status": status, "detail": detail, **extra}
        if status in {"ERROR", "OVERFLOW", "WAITING_HISTORY"}:
            instrument, timeframe = key.rsplit("_", 1)
            adapter = self._adapter(instrument, timeframe)
            if adapter is not None:
                adapter._utema_executable_event_ids = set()
                for name in ("_stream_result", "_last_result"):
                    result = getattr(adapter, name, None)
                    if isinstance(result, dict):
                        setattr(adapter, name, {**result, "utema_executable_event_ids": [],
                                               "utema_entry_eligible": False})
        if detail and (previous.get("detail"), previous.get("status")) != (detail, status):
            logger.warning(f"UT/EMA stream {key}: {status}: {detail}")

    def _mark_missing_path(self, key, reason, stamp=None, detail=""):
        """Persist lost-observation provenance without taking an adapter lock."""
        previous = self._missing_path.get(key, {})
        metadata = {**previous, "reason": reason, "detail": detail, "requires_rebuild": True,
                    "live_tick_path_restored": False}
        if stamp is not None:
            metadata["missing_at"] = market_stamp(stamp).isoformat()
        if key in self._last_observed:
            metadata.setdefault("last_observed_at", self._last_observed[key].isoformat())
        self._missing_path[key] = metadata
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            target = self.directory/f"{key}.gap.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(metadata, allow_nan=False, separators=(",", ":")), encoding="utf-8")
            os.replace(temporary, target)
            self.health.setdefault(key, {}).update(durable_gap_saved=True, persistence_error="")
        except (OSError, ValueError, TypeError) as exc:
            # The process still refuses entries. An unavailable filesystem cannot
            # attest durable recovery, and that failure remains visible in health.
            self.health.setdefault(key, {}).update(durable_gap_saved=False,
                persistence_error=f"Missing-path marker could not be saved: {exc}")

    def _rebuilt_path(self, key, core, stamp):
        if key in self._overflowed:
            return
        metadata = dict(self._missing_path.get(key, {}))
        metadata.update(requires_rebuild=False, historical_rebuild=True,
                        live_tick_path_restored=False, rebuilt_at=stamp.isoformat())
        self._missing_path[key] = metadata
        core._stream_gap = "historical_rebuild_missing_realtime_path"
        core._stream_dirty = True
        self._status(key, "RECOVERED_OHLC",
            "Missing realtime path rebuilt from complete closed candles; replay entries are ineligible",
            historical_rebuild=True, live_tick_path_restored=False,
            missing_path_reason=metadata.get("reason", "missing_index_bars"))

    def submit(self, instrument, price, observed_at):
        if self._closed or str(getattr(self.scanner, "mode", "")).upper() != "REAL":
            return False
        if not getattr(self.scanner, "is_running", False):
            return False
        if not any(self._adapter(instrument, tf) is not None for tf in ("5min", "15min")):
            return False
        try:
            price = float(price)
            if observed_at is None or not math.isfinite(price) or price <= 0:
                return False
            stamp = market_stamp(observed_at)
            if pd.isna(stamp):
                return False
        except (TypeError, ValueError, OverflowError):
            return False
        session = getattr(self.scanner.data, "is_market_open", None)
        if callable(session) and not session(stamp.to_pydatetime()):
            return False
        with self._condition:
            if len(self._queue) >= self.capacity:
                for tf in ("5min", "15min"):
                    key = f"{instrument}_{tf}"
                    self._overflowed.add(key)
                    self._mark_missing_path(key, "queue_overflow", stamp)
                    self._status(key, "OVERFLOW", "Observation queue exceeded capacity; native entries withheld")
                return False
            self._sequence += 1
            self._queue.append((instrument, price, stamp, self._sequence))
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="UTEMAObservedTicks", daemon=True)
                self._thread.start()
            self._condition.notify()
        return True

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._queue)
                drained = self._closed and not self._queue
                observation = None if drained else self._queue.popleft()
            if drained:
                self._flush_checkpoints()
                return
            try:
                self._prune_closed_owners()
                self.process_observation(*observation, drain_on_shutdown=True)
            except Exception as exc:
                logger.error(f"UT/EMA tick worker: {type(exc).__name__}: {exc}")

    def _load(self, key, core):
        if key in self._loaded:
            return False
        self._loaded.add(key)
        path = self.directory/f"{key}.json"
        marker = self.directory/f"{key}.gap.json"
        if marker.exists():
            try:
                metadata = json.loads(marker.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict):
                    raise ValueError("Invalid missing-path marker")
                self._missing_path[key] = {**metadata, "requires_rebuild": True,
                                           "live_tick_path_restored": False}
            except (OSError, ValueError, TypeError) as exc:
                self._mark_missing_path(key, "invalid_gap_marker", detail=str(exc))
            self._status(key, "WAITING_HISTORY", "Persisted observation gap requires an OHLC baseline rebuild",
                         historical_rebuild=False, live_tick_path_restored=False)
            return False
        if not path.exists() or not callable(getattr(core, "restore_stream_checkpoint", None)):
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            recovery = payload.get("missing_path", {})
            if not isinstance(recovery, dict):
                raise ValueError("Invalid missing-path checkpoint metadata")
            if recovery:
                self._missing_path[key] = dict(recovery)
                if recovery.get("requires_rebuild"):
                    self._status(key, "WAITING_HISTORY", "Checkpoint records an unrecovered observation gap",
                                 historical_rebuild=False, live_tick_path_restored=False)
                    return False
            saved = payload["core"]
            if isinstance(saved, dict) and saved.get("instrument") is not None:
                if f"{saved.get('instrument')}_{saved.get('timeframe')}" != key:
                    raise ValueError("Checkpoint belongs to another index/timeframe")
            # Validate and replay into a private owner. A damaged journal must
            # never leave the running owner or recovery cursors half restored.
            recovered = copy.deepcopy(core)
            recovered.restore_stream_checkpoint(saved)
            if recovery:
                recovered._stream_gap = "historical_rebuild_missing_realtime_path"
                recovered._stream_dirty = True
            last_observed = market_stamp(payload["last_observed_at"])
            if pd.isna(last_observed):
                raise ValueError("Invalid checkpoint observation time")
            saved_journal = payload.get("journal", {})
            journal_name = str(saved_journal.get("file") or "")
            position = dict(saved_journal)
            if journal_name and (Path(journal_name).name != journal_name or
                    not journal_name.startswith(f"{key}_") or
                    not (self.directory/journal_name).is_file()):
                raise ValueError("Checkpoint observation journal is missing or invalid")
            for journal in sorted(self.directory.glob(f"{key}_*.jsonl")):
                if journal.name < journal_name:
                    continue
                with journal.open("rb") as stream:
                    if journal.name == journal_name:
                        offset = int(saved_journal.get("offset", 0))
                        if offset < 0 or offset > journal.stat().st_size:
                            raise ValueError("Invalid observation journal offset")
                        if offset:
                            stream.seek(offset-1)
                            if stream.read(1) != b"\n":
                                raise ValueError("Observation journal offset splits a record")
                        stream.seek(offset)
                    for raw in stream:
                        if not raw.endswith(b"\n"):
                            raise ValueError("Incomplete final observation journal record")
                        tick = json.loads(raw)
                        if not isinstance(tick, list) or len(tick) != 2:
                            raise ValueError("Invalid observation journal record")
                        at, price = market_stamp(tick[0]), float(tick[1])
                        if pd.isna(at) or not math.isfinite(price) or price <= 0 or at < last_observed:
                            raise ValueError("Invalid or unordered observation journal record")
                        self._observe(recovered, price, at)
                        last_observed = at
                    position = {"file": journal.name, "offset": stream.tell()}
            core.__dict__.clear()
            core.__dict__.update(recovered.__dict__)
            self._last_observed[key] = last_observed
            self._journal_position[key] = position
            self._status(key, "RESTORED", "Processed observations restored; offline market ticks are unavailable",
                         last_observed_at=payload["last_observed_at"], restored=True,
                         **({"historical_rebuild": True, "live_tick_path_restored": False,
                             "missing_path_reason": recovery.get("reason", "")} if recovery else {}))
            return True
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._mark_missing_path(key, "checkpoint_rejected", detail=str(exc))
            self._status(key, "WAITING_HISTORY", f"Checkpoint rejected: {exc}",
                         historical_rebuild=False, live_tick_path_restored=False)
            return False

    def _save(self, key, core, stamp, *, force=False):
        if not callable(getattr(core, "stream_checkpoint", None)):
            return
        if not force and time.monotonic()-self._saved_at.get(key, 0.) < 5.:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            payload = {"core": core.stream_checkpoint(), "last_observed_at": stamp.isoformat(),
                       "journal": self._journal_position.get(key, {}),
                       "missing_path": copy.deepcopy(self._missing_path.get(key, {}))}
            target = self.directory/f"{key}.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")), encoding="utf-8")
            os.replace(temporary, target)
            self._saved_at[key] = time.monotonic()
            # A new overflow can arrive while this worker writes its checkpoint.
            # The marker is removed only for the exact completed rebuild saved
            # above, under the same producer condition used to mark overflow.
            with self._condition:
                recovery = payload["missing_path"]
                if (recovery and not recovery.get("requires_rebuild") and key not in self._overflowed
                        and self._missing_path.get(key) == recovery):
                    (self.directory/f"{key}.gap.json").unlink(missing_ok=True)
        except (OSError, ValueError, TypeError) as exc:
            self._status(key, "CHECKPOINT_ERROR", str(exc), last_observed_at=stamp.isoformat())

    def _observe(self, core, price, stamp):
        opening, closing = self._session_bounds(stamp)
        if callable(getattr(core, "seed_stream", None)):
            options = {"session_open": opening, "session_close": closing}
            previous = getattr(core, "_stream_times", [])
            if previous and previous[-1].date() != stamp.date():
                options["previous_session_close"] = self._session_bounds(previous[-1])[1]
            return core.observe_tick(price, stamp, **options)
        return core.observe_tick(price, stamp)

    def _session_bounds(self, stamp):
        from data.market_calendar import index_session_bounds

        bounds = getattr(self.scanner.data, "get_index_session_bounds", None)
        opening, closing = (bounds if callable(bounds) else index_session_bounds)(stamp.to_pydatetime())
        return market_stamp(opening), market_stamp(closing)

    def _history_gap(self, frame, timeframe, clock, *, start=None):
        """Require closed bars on the host's exchange grid, including session tails."""
        from engine.ut_ema_engine import UTEmaEngine, minutes
        from data.market_calendar import load_nse_holidays, load_nse_special_trading_days
        history = UTEmaEngine.prepare(frame)
        if history.empty:
            return "Closed index candles unavailable"
        lower = min(market_stamp(start), history.index[0]) if start is not None else history.index[0]
        holidays, specials = set(load_nse_holidays()), set(load_nse_special_trading_days())
        period = pd.Timedelta(minutes=minutes(timeframe))
        stamps = history.index[history.index <= clock]
        for day in pd.date_range(lower.normalize(), clock.normalize(), freq="D"):
            label = day.date().isoformat()
            day_stamps = stamps[stamps.slice_indexer(day, day + pd.Timedelta(days=1, nanoseconds=-1))]
            if label not in specials and (day.weekday() >= 5 or label in holidays):
                if len(day_stamps):
                    return f"Index candle outside an exchange session: {day_stamps[0].isoformat()}"
                continue
            opening, closing = self._session_bounds(day)
            grid = pd.date_range(opening, closing, freq=period, inclusive="left")
            off_grid = day_stamps.difference(grid)
            if len(off_grid):
                return f"Index candle outside its timeframe grid: {off_grid[0].isoformat()}"
            closed = grid[(grid >= lower) & ((grid + period <= clock) | (closing <= clock))]
            missing = closed.difference(day_stamps)
            if len(missing):
                return f"Closed index candle missing: {missing[0].isoformat()}"
        return ""

    def _session_history(self, frame):
        """Exclude reference bars outside the exchange session before native seeding.

        Legacy history can retain pre-open index references. Those are not trading
        candles. In-session misalignment and missing bars still fail coverage proof.
        The official store is not mutated, and special session bounds remain valid.
        """
        if frame is None or not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
            return frame
        stamps = frame.index
        if stamps.tz is not None:
            stamps = stamps.tz_convert("Asia/Kolkata").tz_localize(None)
        keep = [False]*len(stamps)
        bounds = {}
        from data.market_calendar import load_nse_special_trading_days
        special_days = set(load_nse_special_trading_days())
        host_bounds = callable(getattr(self.scanner.data, "get_index_session_bounds", None))
        for i, stamp in enumerate(stamps):
            if pd.isna(stamp):
                # Preserve malformed timestamps for the existing validator.
                keep[i] = True
                continue
            day = stamp.normalize()
            if not host_bounds and day.date().isoformat() in special_days:
                # Without an explicit special-session schedule, preserve the
                # original input for coverage validation instead of guessing.
                keep[i] = True
                continue
            if day not in bounds:
                bounds[day] = self._session_bounds(stamp)
            opening, closing = bounds[day]
            keep[i] = opening <= stamp < closing
        return frame.loc[keep]

    def _bootstrap_history(self, instrument, timeframe, frame, clock):
        """Fill absent closed bars only from complete official one-minute candles."""
        from engine.ut_ema_engine import UTEmaEngine, minutes
        from data.market_calendar import INDEX_SPECIAL_SESSION_HOURS

        history = self._session_history(frame)
        if history is None or history.empty:
            return history
        upstox = getattr(self.scanner.data, "upstox_only", False) is True
        if not upstox and not self._history_gap(history, timeframe, clock).startswith("Closed index candle missing:"):
            return history
        base = self.scanner.candles.get_candles(instrument, "1min")
        start = market_stamp(frame.index[0])
        if upstox:
            # Tick-built higher candles can hide missing minutes. Recover the current
            # session and the special-session grid from closed Upstox candles only.
            today = clock.date().isoformat()
            rebuild_days = {today, *INDEX_SPECIAL_SESSION_HOURS}
            history = history.loc[history.index.strftime("%Y-%m-%d") != today]
            for special_day in INDEX_SPECIAL_SESSION_HOURS:
                opening, closing = self._session_bounds(market_stamp(special_day))
                grid = pd.date_range(opening, closing, freq=f"{minutes(timeframe)}min", inclusive="left")
                labels = history.index.strftime("%Y-%m-%d")
                history = history.loc[(labels != special_day) | history.index.isin(grid)]
            if base is not None and "price_source" in base:
                base = base.loc[base["price_source"].eq("upstox_history") &
                                base.index.strftime("%Y-%m-%d").isin(rebuild_days)]
            else:
                base = None
        if base is None or base.empty:
            return history
        base = UTEmaEngine.prepare(self._session_history(base))
        base = base.loc[base.index == base.index.floor("min")]
        if upstox and start.date().isoformat() in INDEX_SPECIAL_SESSION_HOURS:
            start = self._session_bounds(start)[0]
        period = pd.Timedelta(minutes=minutes(timeframe))
        recovered = []
        for day in base.index.normalize().unique():
            opening, closing = self._session_bounds(day)
            rows = base.loc[(base.index >= opening) & (base.index < closing)]
            batches = rows.resample(period, origin=opening)
            aggregate = batches.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            end = (aggregate.index + period).where(aggregate.index + period <= closing, closing)
            expected_count = (end - aggregate.index) // pd.Timedelta(minutes=1)
            complete = batches.size().eq(expected_count) & (end <= clock) & (aggregate.index >= start)
            missing = ~aggregate.index.isin(history.index)
            recovered.append(aggregate.loc[complete & missing])
        return pd.concat([history, *recovered]).sort_index() if recovered else history

    def _journal(self, key, price, stamp):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory/f"{key}_{stamp.date().isoformat()}.jsonl"
        line = (json.dumps([stamp.isoformat(), float(price)], allow_nan=False, separators=(",", ":"))+"\n").encode()
        with path.open("ab") as stream:
            stream.write(line)
            stream.flush()
            self._journal_position[key] = {"file": path.name, "offset": stream.tell()}

    def recover_verified_history(self, instrument, timeframe, frame, observed_at):
        """Rebuild missing closed bars after explicit chronological coverage proof.

        The caller supplies the existing official candle store. Missing realtime
        paths remain disclosed; only subsequent observed events can be executable.
        """
        import pandas as pd
        from engine.ut_ema_engine import UTEmaEngine, minutes
        key = f"{instrument}_{timeframe}"
        adapter = self._adapter(instrument, timeframe)
        frame = self._bootstrap_history(instrument, timeframe, frame, market_stamp(observed_at))
        if adapter is None or frame is None or len(frame) < 2:
            return False
        core = adapter._native
        clock = market_stamp(observed_at)
        previous = getattr(core, "_stream_times", [])
        if self._history_gap(frame, timeframe, clock, start=previous[-1] if previous else None):
            return False
        prior = core.stream_checkpoint()
        fresh = UTEmaEngine(core.params)
        opening, closing = self._session_bounds(clock)
        fresh.seed_stream(frame, instrument, timeframe, as_of=clock, session_open=opening, session_close=closing)
        self.directory.mkdir(parents=True, exist_ok=True)
        archive = self.directory/f"{key}_before_rebuild_{time.time_ns()}.json"
        archive.write_text(json.dumps(prior, allow_nan=False, separators=(",", ":")), encoding="utf-8")
        adapter._native = fresh
        adapter._stream_result = None
        adapter._utema_executable_event_ids = set()
        adapter._utema_stream_dirty = False
        self._pending_history.pop(key, None)
        self._rebuilt_path(key, fresh, clock)
        return True

    def _history_check_due(self, key):
        now = time.monotonic()
        if now < self._next_history_check.get(key, 0.0):
            return False
        self._next_history_check[key] = now + 1.0
        return True

    def process_observation(self, instrument, price, stamp, sequence=0, *, drain_on_shutdown=False):
        """Synchronous seam for ordered-worker tests; never execute broker mutations."""
        if (self._closed and not drain_on_shutdown) or str(getattr(self.scanner, "mode", "")).upper() != "REAL":
            return
        for timeframe in ("5min", "15min"):
            adapter = self._adapter(instrument, timeframe)
            if adapter is None:
                continue
            key = f"{instrument}_{timeframe}"
            if key in self._overflowed or self.health.get(key, {}).get("status") == "OVERFLOW":
                continue
            lock = getattr(adapter, "_utema_lock", None)
            if lock is None:
                adapter._utema_lock = lock = threading.RLock()
            try:
                with lock:
                    sync_timing = getattr(adapter, "sync_entry_timing", None)
                    if callable(sync_timing):
                        sync_timing()
                    core = adapter._native
                    restored = self._load(key, core)
                    if restored:
                        adapter._utema_executable_event_ids = set()
                        adapter._stream_result = None
                    if key in self._last_observed and stamp < self._last_observed[key]:
                        previous_health = self.health.get(key, {})
                        if previous_health.get("status") in {"ERROR", "OVERFLOW", "WAITING_HISTORY"}:
                            self._status(key, previous_health["status"], previous_health.get("detail", ""),
                                         ignored_stale_at=stamp.isoformat())
                        else:
                            self._status(key, "STALE_TICK", "Out-of-order observation ignored",
                                last_observed_at=self._last_observed[key].isoformat(),
                                ignored_stale_at=stamp.isoformat(),
                                last_stale_delay_seconds=(self._last_observed[key]-stamp).total_seconds(),
                                stale_ticks_ignored=int(previous_health.get("stale_ticks_ignored", 0))+1)
                        continue
                    adapter._utema_stream_owned = True
                    if (self.health.get(key, {}).get("status") == "ERROR" or
                            (self._missing_path.get(key, {}).get("requires_rebuild") and
                             getattr(core, "_timeline", None) is not None)):
                        if drain_on_shutdown and not self._history_check_due(key):
                            continue
                        frame = self.scanner.candles.get_candles(instrument, timeframe)
                        if not self.recover_verified_history(instrument, timeframe, frame, stamp):
                            continue
                        core = adapter._native
                    previous = getattr(adapter, "_stream_result", {}) or {}
                    old_count = len(previous.get("utema_events", []))
                    if getattr(core, "_timeline", None) is None:
                        backlog = self._pending_history.setdefault(key, [])
                        if len(backlog) >= self.capacity:
                            self._overflowed.add(key)
                            self._mark_missing_path(key, "bootstrap_overflow", stamp)
                            self._status(key, "OVERFLOW", "Native history bootstrap queue exceeded capacity")
                            continue
                        backlog.append((price, stamp))
                        if drain_on_shutdown and not self._history_check_due(key):
                            continue
                        frame = self.scanner.candles.get_candles(instrument, timeframe)
                        first_stamp = backlog[0][1]
                        frame = self._bootstrap_history(instrument, timeframe, frame, first_stamp)
                        if frame is None or len(frame) < 2:
                            self._status(key, "WAITING_HISTORY", "Closed candles required for native indicator bootstrap")
                            continue
                        seed = getattr(core, "seed_stream", None)
                        if callable(seed):
                            gap = self._history_gap(frame, timeframe, first_stamp)
                            if gap:
                                self._status(key, "WAITING_HISTORY", gap)
                                continue
                            opening, closing = self._session_bounds(first_stamp)
                            seed(frame, instrument, timeframe, as_of=first_stamp,
                                 session_open=opening, session_close=closing)
                            if self._missing_path.get(key, {}).get("requires_rebuild"):
                                self._rebuilt_path(key, core, first_stamp)
                        else:
                            core.process(frame, instrument, timeframe, as_of=first_stamp, realtime=True)
                        # Seed indicator history; historical pending instructions do
                        # not become executable observations by loading the frame.
                        self._pending_history.pop(key, None)
                        result = None
                        for observed_price, observed_stamp in backlog:
                            self._journal(key, observed_price, observed_stamp)
                            result = self._observe(core, observed_price, observed_stamp)
                    else:
                        self._journal(key, price, stamp)
                        result = self._observe(core, price, stamp)
                    adapter._utema_last_observed_at = stamp.isoformat()
                    adapter._utema_stream_dirty = result is None
                    if result is not None:
                        if not isinstance(result, dict):
                            raise ValueError("Native stream did not return a result")
                        result = copy.deepcopy(result)
                        # Only fresh observations authorize execution. Historical
                        # bootstrap, journal recovery and older buffered ticks do
                        # not authorize an entry, even if their lifecycle is open.
                        permitted = set(getattr(adapter, "_utema_executable_event_ids", set()))
                        records = result.get("utema_records", {})
                        permitted = {identity for identity in permitted if any(
                            identity.startswith(f"{lifecycle_id}|") and not record.get("exit_time")
                            for lifecycle_id, record in records.items())}
                        for signal in result.get("signals", []):
                            event = getattr(signal, "utema_event", {})
                            native = getattr(signal, "utema_state", {})
                            if (getattr(signal, "utema_action", "") in {"ENTRY", "ADD"}
                                    and (event.get("realtime") or event.get("confirmed")) and event.get("observed_at")
                                    and market_stamp(event["observed_at"]) == stamp
                                    and not records.get(native.get("lifecycle_id"), {}).get("exit_time")):
                                permitted.add(signal.utema_event_id)
                        adapter._utema_executable_event_ids = permitted
                        result.update(utema_stream_owned=True, utema_entry_eligible=True,
                                      utema_last_observed_at=stamp.isoformat(),
                                      utema_executable_event_ids=sorted(permitted))
                        adapter._stream_result = adapter._last_result = result
                    event_count = len(result.get("utema_events", [])) if result is not None else old_count
                    self._last_observed[key] = stamp
                    if key in self._overflowed:
                        self._status(key, "OVERFLOW", "Observation queue exceeded capacity; native entries withheld")
                        self._save(key, core, stamp, force=True)
                        continue
                    self._status(key, "READY", "", last_observed_at=stamp.isoformat(), sequence=sequence,
                                 restored=restored or self.health.get(key, {}).get("restored", False))
                    self._save(key, core, stamp, force=event_count != old_count)
                    if event_count != old_count:
                        self._schedule(instrument, timeframe, result, stamp)
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                self._mark_missing_path(key, "observation_error", stamp, detail=f"{type(exc).__name__}: {exc}")
                self._status(key, "ERROR", f"{type(exc).__name__}: {exc}")

    def _schedule(self, instrument, timeframe, result, observed_at):
        if self._closed:
            return
        loop = getattr(self.scanner, "_event_loop", None)
        if loop is None or loop.is_closed() or not loop.is_running():
            return
        future = asyncio.run_coroutine_threadsafe(self.deliver(instrument, timeframe, result, observed_at), loop)
        def completed(done):
            try:
                done.result()
            except Exception as exc:
                logger.error(f"UT/EMA event delivery {instrument} {timeframe}: {exc}")
        future.add_done_callback(completed)

    async def deliver(self, instrument, timeframe, event_result, observed_at):
        """Exit-before-entry delivery uses current quotes and current native ownership."""
        from config.settings import get_settings
        from engine.scanner_modes import configured_mode
        from engine import ut_ema_lane as lane
        scanner = self.scanner
        if self._closed or str(getattr(scanner, "mode", "")).upper() != "REAL" or not getattr(scanner, "is_running", False):
            return
        if str(getattr(scanner, "system_power", "ON")).upper() != "ON":
            return
        if self._delivery_lock is None:
            self._delivery_lock = asyncio.Lock()
        async with self._delivery_lock:
            # Authorization can change while this task waits for another event.
            if (self._closed or str(getattr(scanner, "mode", "")).upper() != "REAL"
                    or not getattr(scanner, "is_running", False)
                    or str(getattr(scanner, "system_power", "ON")).upper() != "ON"):
                return
            configuration = (getattr(scanner, "config", {}).get("indices", {}) or {}).get(instrument, {})
            rows = scanner._get_session_trade_candidates(instrument)
            scanner._refresh_trade_candidates(rows, configuration, allow_blocking=False)
            scanner._check_session_candidates_breaches(instrument)
            # A held owner's exits continue after changing the entry scanner.
            scanner._run_exit_tick()
            if (configured_mode(get_settings()) != ENGINE_KEY or
                    instrument not in getattr(scanner, "active_indices", [instrument])):
                return
            adapter = self._adapter(instrument, timeframe)
            snapshot = getattr(adapter, "stream_snapshot", None)
            if callable(snapshot):
                # Delivery replaces ``signals`` on its own copy, so a shallow
                # snapshot is enough. Asking for the deep copy here traversed the
                # whole result graph twice per delivery (191 + 241 of 2,995 GIL
                # samples per window, on this loop, on 2026-10-30).
                current = snapshot(deep=False) or {}
                filtered = dict(current)
            else:
                current = getattr(adapter, "_stream_result", {}) or {}
                filtered = copy.deepcopy(current)
            # Loading/replaying old pending orders is never an executable handoff.
            filtered["signals"] = [signal for signal in event_result.get("signals", [])
                if (getattr(signal, "utema_event", {}).get("realtime")
                    or getattr(signal, "utema_event", {}).get("confirmed"))
                and market_stamp(signal.utema_event["observed_at"]) == observed_at
                and not (current.get("utema_records", {}).get(lane.state(signal).get("lifecycle_id"), {}).get("exit_time"))]
            view = SimpleNamespace(**{f"results_{tf}": filtered if tf == timeframe else {} for tf in ("1min", "5min", "15min")})
            candidates = lane.build_candidates(scanner.signal_processor, instrument, view, configuration,
                int(getattr(scanner, "user_lots", {}).get(instrument, 1)),
                int(configuration.get("lot_size", 1)), get_settings())
            await lane.coordinate(scanner, candidates, is_warmup=bool(getattr(scanner, "is_warmup", False)))

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        # The worker drains observations already accepted before shutdown. It
        # cannot schedule execution after close and flushes its final checkpoint.
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.)
            if self._thread.is_alive():
                return
        self._flush_checkpoints()

    def _flush_checkpoints(self):
        for key, stamp in list(self._last_observed.items()):
            instrument, timeframe = key.rsplit("_", 1)
            adapter = self._adapter(instrument, timeframe, include_unwatched=True)
            if adapter is not None:
                with adapter._utema_lock:
                    self._save(key, adapter._native, stamp, force=True)


def install_tick_service(scanner, *, checkpoint_dir=None):
    """Keep the candle callback contract while adding ordered native observations."""
    existing = getattr(scanner, "_utema_tick_service", None)
    if existing is not None:
        return existing
    service = UTEmaTickService(scanner, checkpoint_dir=checkpoint_dir)
    scanner._utema_tick_service = service
    base_callback = scanner.data.on_tick
    def observed_tick(instrument, price, volume, timestamp, volume_source="", *, native_price=True):
        # The same input validity boundary protects the existing candle store and
        # the native stream; no rejected cross-index tick enters the new engine.
        plausible = scanner.candles._price_plausible(instrument, price)
        if plausible and native_price:
            service.submit(instrument, price, timestamp)
        # Candle mutation can wait behind chart/history work. Queue the accepted
        # observation first so another broker callback cannot overtake it while
        # the older callback is waiting to update the candle store.
        base_callback(instrument, price, volume, timestamp, volume_source)
    observed_tick._utema_accepts_native_price = True
    scanner.data.on_tick = observed_tick
    return service
