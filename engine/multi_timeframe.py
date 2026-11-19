"""
Multi-Timeframe Analysis Engine
═══════════════════════════════════════════════════════════════

Runs UT Bot across 1min, 5min, 15min simultaneously for each index.
- Signal timeframes (5min, 15min): Generate trade entries
- Tracking timeframe (1min): Tracks active trades for trailing

Confluence scoring: signal on 5min + aligned regime on 15min = higher confidence
"""

import copy
import hashlib
import time

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field
from loguru import logger

from engine.ut_bot_core import UTBotEngine, UTBotSignal


# Columns that participate in change detection. A late broker correction to any
# of these on a prior bar must invalidate the cached analysis result.
REVISION_COLUMNS = frozenset({
    "open", "high", "low", "close", "volume", "volume_basis",
    "tick_count", "volume_complete", "volume_unallocated",
})


ENGINE_PARAMETER_KEYS = frozenset({
    "key_value",
    "atr_period",
    "use_heikin_ashi",
    "signal_mode",
    "adx_filter",
    "adx_period",
    "adx_threshold",
    "strict_adx",
    "session_filter",
    "session_start",
    "session_end",
})

@dataclass
class MultiTFResult:
    """Result of multi-timeframe analysis for one instrument"""
    instrument: str
    timestamp: datetime
    results_1min: Optional[Dict] = None
    results_5min: Optional[Dict] = None
    results_15min: Optional[Dict] = None
    confluence_score: float = 0.0
    confluence_signal: str = "HOLD"  # BUY / SELL / HOLD
    active_signals: List[UTBotSignal] = field(default_factory=list)
    # Per-timeframe provenance for the result returned by this pass.  Consumers that
    # make execution decisions must be able to distinguish a freshly recomputed arrow
    # from a cached arrow whose source input has since disappeared or failed.
    timeframe_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class MultiTimeframeEngine:
    """
    Runs the configured scanner engine on multiple timeframes and computes confluence.

    Each instrument gets 3 independent engine instances (1min, 5min, 15min).
    Signals from 5min and 15min are considered for entries.
    1min is used for trade tracking and early exit detection.
    """

    def __init__(self, engine_params: Dict = None):
        """
        Args:
            engine_params: Dict of UT Bot parameters to override defaults
        """
        params = engine_params or {}
        self.engine_params = params
        self.engines: Dict[str, UTBotEngine] = {}
        engine_init_params = {
            key: value for key, value in params.items()
            if key in ENGINE_PARAMETER_KEYS
        }

        # Create engines for each instrument-timeframe combo. The concrete engine
        # class is chosen by the `scanner_engine` setting (default "utbot_consensus");
        # params are stored so the engines can be rebuilt live when switched.
        self._engine_init_params = dict(engine_init_params)
        self._instruments = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
        self._timeframes = ["1min", "5min", "15min"]
        # A selector change controls new entries, not held UT/EMA exit ownership.
        # Keep the original adapter because OHLC cannot recreate its live tick path.
        self._retired_utema_engines = {}
        self._retired_utema_results = {}
        self._build_all_engines()

        # Store latest OHLCV data per instrument-timeframe
        self._data_cache: Dict[str, pd.DataFrame] = {}

        # Confluence weights
        self.tf_weights = {
            "1min": 0.15,
            "5min": 0.40,
            "15min": 0.45,
        }

        # ═══ PROCESS RESULT CACHE — skip reprocessing unchanged data ═══
        self._result_cache: Dict[str, Dict] = {}
        self._data_hash: Dict[str, int] = {}

        # Input and result generations are deliberately independent.  update_data()
        # records every attempted timeframe input, including None/underfilled frames;
        # result metadata records only the exact input generation that was processed.
        # This closes a dangerous ambiguity where a prior arrow used to survive a
        # missing frame or an engine exception and looked current to stabilization.
        self._input_metadata: Dict[str, Dict[str, Any]] = {}
        self._result_metadata: Dict[str, Dict[str, Any]] = {}
        self._fresh_result_required_after: Dict[str, float] = {}

    def _build_all_engines(self):
        """(Re)build the per-(instrument,timeframe) engines from the active
        scanner_engine setting. The 1-minute engine stays realtime so active-trade
        trailing/early-exit tracking is not candle-close delayed."""
        from engine.alt_scanners import build_scanner_engine
        for inst in self._instruments:
            for tf in self._timeframes:
                key = f"{inst}_{tf}"
                tf_params = dict(self._engine_init_params)
                if tf == "1min":
                    tf_params["signal_mode"] = "realtime"
                engine = build_scanner_engine(tf_params)
                retained = getattr(self, "_retired_utema_engines", {}).get(key)
                if getattr(engine, "signal_source", "") == "ut_ema" and retained is not None:
                    # Its core validates the closed-history fingerprint on its next
                    # process call; corrections reset state rather than trusting ticks
                    # derived from a different prefix.
                    engine = retained
                if getattr(engine, "signal_source", "") == "ut_ema" and getattr(self, "_utema_stream_service_installed", False):
                    engine._utema_stream_owned = True
                self.engines[key] = engine

    def set_scanner_engine(self, mode: str) -> str:
        """Switch the live signal engine and rebuild.

        Persists onto settings, rebuilds every engine, and clears the result/hash
        caches so the next scan reprocesses with the new engine. Returns the mode
        actually applied."""
        mode = str(mode or "utbot").lower()
        if mode not in {"utbot", "ema", "consensus", "utbot_consensus", "quad_union", "quad_prime", "adaptive_dual", "s1", "ut_smart", "quad_prime_s1", "ut_smart_s1", "s1s2_dual", "ut_ema"}:
            mode = "utbot"
        if not hasattr(self, "_retired_utema_engines"):
            self._retired_utema_engines = {}
            self._retired_utema_results = {}
        for key, engine in self.engines.items():
            if getattr(engine, "signal_source", "") == "ut_ema":
                self._retired_utema_engines[key] = engine
                result = getattr(self, "_result_cache", {}).get(key) or getattr(engine, "_last_result", {})
                if isinstance(result, dict) and result.get("source_engine") == "ut_ema":
                    self._retired_utema_results[key] = copy.deepcopy(result)
        try:
            from config.settings import get_settings
            get_settings().scanner_engine = mode
        except Exception as exc:
            logger.error(f"set_scanner_engine: could not persist setting: {exc}")
        self._build_all_engines()
        self._ensure_freshness_state()
        self._result_cache.clear()
        self._data_hash.clear()
        self._result_metadata.clear()
        self._fresh_result_required_after.clear()
        getattr(self, "_combined_s1_observations", {}).clear()
        getattr(self, "_combined_base_observations", {}).clear()
        logger.info(f"Signal engine switched to '{mode}' and all engines rebuilt.")
        return mode

    def get_utema_owner_result(self, instrument, timeframe, df, as_of=None):
        """Read the held owner's result without changing selection or generic caches.

        While another scanner runs, the retained adapter continues observing prices
        for UT/EMA exits. A restart without that adapter can only replay historical
        OHLC; the returned provenance explicitly says its live tick path is missing.
        """
        key = f"{instrument}_{timeframe}"
        active = getattr(self, "engines", {}).get(key)
        if getattr(active, "signal_source", "") == "ut_ema":
            snapshot = getattr(active, "stream_snapshot", None)
            streamed = snapshot() if callable(snapshot) else getattr(active, "_stream_result", None)
            if streamed is not None:
                return copy.deepcopy(streamed)
            current = (getattr(self, "_result_cache", {}) or {}).get(key)
            if isinstance(current, dict) and current.get("source_engine") == "ut_ema":
                return copy.deepcopy(current)
            adapter = active
        else:
            adapter = (getattr(self, "_retired_utema_engines", {}) or {}).get(key)
        if df is None or getattr(df, "empty", True):
            return None
        from config.settings import get_settings

        clock = as_of if as_of is not None else datetime.now(timezone.utc)
        snapshot = getattr(adapter, "stream_snapshot", None)
        streamed = snapshot() if callable(snapshot) else getattr(adapter, "_stream_result", None)
        if streamed is not None:
            return copy.deepcopy(streamed)
        replay_only = adapter is None or getattr(getattr(adapter, "_native", None), "_timeline", None) is None
        if replay_only:
            from engine.alt_scanners import UTEmaSignalEngine

            if adapter is None:
                adapter = UTEmaSignalEngine()
                adapter._utema_stream_owned = bool(getattr(self, "_utema_stream_service_installed", False))
                self._retired_utema_engines[key] = adapter
            result = UTEmaSignalEngine(synchronous=True).process(df=df, instrument=instrument,
                timeframe=timeframe, as_of=clock, realtime=False)
            result["utema_replay_fallback"] = True
            result["utema_live_path_available"] = False
            return result
        realtime = not replay_only and str(getattr(get_settings(), "trading_mode", "HISTORICAL")).upper() == "REAL"
        result = adapter.process(df=df, instrument=instrument, timeframe=timeframe,
                                 as_of=clock, realtime=realtime)
        if not isinstance(result, dict) or result.get("source_engine") != "ut_ema":
            return None
        result = copy.deepcopy(result)
        if replay_only:
            result["utema_replay_fallback"] = True
            result["utema_live_path_available"] = False
        else:
            if not hasattr(self, "_retired_utema_results"):
                self._retired_utema_results = {}
            self._retired_utema_results[key] = copy.deepcopy(result)
        return result

    @staticmethod
    def _revision_digest(df, end_idx: int) -> bytes:
        """Deterministic digest of the applicable columns up to ``end_idx``.

        Hashes each column without materializing the ``df[columns]`` block copy:
        that subset selection alone cost ~1.6ms per call on the live 1s loop
        (2026-10-30 profile), more than the hash itself. Numeric columns feed
        their raw float64 buffer; string/object columns are factorized over the
        revision window so a late correction is still seen.
        """
        digest = hashlib.blake2b(digest_size=16)
        stamps = df.index[:end_idx]
        if isinstance(stamps, pd.DatetimeIndex):
            digest.update(np.asarray(stamps.view("int64"), dtype="int64").tobytes())
        else:
            digest.update("\x1f".join(map(str, stamps)).encode("utf-8"))
        for column in df.columns:
            if str(column).lower() not in REVISION_COLUMNS:
                continue
            series = df[column]
            if is_numeric_dtype(series):
                digest.update(series.to_numpy(dtype="float64", copy=False)[:end_idx].tobytes())
            else:
                window = series.iloc[:end_idx]
                codes, uniques = pd.factorize(window, use_na_sentinel=False)
                digest.update(np.asarray(codes, dtype="int64").tobytes())
                digest.update("\x1f".join(map(str, uniques)).encode("utf-8"))
        return digest.digest()

    @staticmethod
    def _hash_df(df, signal_mode: str = "realtime") -> int:
        """Quick hash of DataFrame to detect changes.
        Uses the second-to-last (confirmed) candle to avoid cache misses
        from live tick updates on the forming bar."""
        if df is None or len(df) == 0:
            return 0

        cols = [c.lower() for c in df.columns]
        if 'close' not in cols:
            return hash(len(df))

        try:
            c_col = 'close' if 'close' in df.columns else ('Close' if 'Close' in df.columns else None)
            h_col = 'high' if 'high' in df.columns else ('High' if 'High' in df.columns else None)
            l_col = 'low' if 'low' in df.columns else ('Low' if 'Low' in df.columns else None)
            v_col = 'volume' if 'volume' in df.columns else ('Volume' if 'Volume' in df.columns else None)

            if not c_col: return hash(len(df))

            # Confirmed entry engines ignore forming-bar changes; realtime
            # engines (including 1min tracking) invalidate on every live tick.
            ref_idx = -2 if signal_mode == "confirmed" and len(df) > 1 else -1
            last_timestamp = df.index[ref_idx].timestamp() if hasattr(df.index[ref_idx], "timestamp") else hash(str(df.index[ref_idx]))

            # The latest row alone cannot see a late broker correction to a prior
            # bar. Include the full applicable input: longer indicators and session
            # features also depend on corrections outside the most recent bars.
            end_idx = len(df) + ref_idx + 1
            revision_signature = MultiTimeframeEngine._revision_digest(df, end_idx)
            return hash((
                len(df),
                last_timestamp,
                float(df[c_col].iloc[ref_idx]),
                float(df[h_col].iloc[ref_idx]) if h_col else 0.0,
                float(df[l_col].iloc[ref_idx]) if l_col else 0.0,
                float(df[v_col].iloc[ref_idx]) if v_col else 0.0,
                revision_signature,
            ))
        except Exception as e:

            logger.debug(f"Caught bare exception: {e}")
            return hash(len(df))

    @staticmethod
    def _utc_iso(epoch: float) -> str:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()

    def _ensure_freshness_state(self) -> None:
        """Backfill metadata stores for lightweight ``__new__`` test doubles."""
        if not hasattr(self, "_input_metadata"):
            self._input_metadata = {}
        if not hasattr(self, "_result_metadata"):
            self._result_metadata = {}
        if not hasattr(self, "_fresh_result_required_after"):
            self._fresh_result_required_after = {}

    def update_data(self, instrument: str, timeframe: str, df: Optional[pd.DataFrame]):
        """Update candle data and record whether this timeframe input is usable.

        ``None`` and short frames are inputs too: recording them prevents a previous
        successful result from being silently presented as the result of this scan.
        """
        key = f"{instrument}_{timeframe}"
        self._ensure_freshness_state()
        previous = self._input_metadata.get(key) or {}
        version = int(previous.get("version") or 0) + 1
        observed_at = time.time()
        try:
            row_count = int(len(df)) if df is not None else 0
        except Exception:
            row_count = 0
        engine = getattr(self, "engines", {}).get(key)
        minimum = 2 if getattr(engine, "signal_source", "") == "ut_ema" else 15
        status = "ready" if row_count >= minimum else ("missing" if df is None else "underfilled")
        self._data_cache[key] = df
        self._input_metadata[key] = {
            "version": version,
            "observed_at_epoch": observed_at,
            "observed_at": self._utc_iso(observed_at),
            "row_count": row_count,
            "status": status,
        }

    def require_fresh_result(
        self,
        instrument: str,
        timeframe: str,
        *,
        newer_than_epoch: Optional[float] = None,
    ) -> float:
        """Force the next valid pass to recompute this timeframe.

        Stabilization calls this when it buffers a forming-bar arrow.  A same-hash
        cache hit from before that instant is analytically useful for charts, but is
        not a new observation and therefore cannot count as stabilization evidence.
        """
        self._ensure_freshness_state()
        key = f"{instrument}_{timeframe}"
        cutoff = float(newer_than_epoch if newer_than_epoch is not None else time.time())
        self._fresh_result_required_after[key] = max(
            cutoff,
            float(self._fresh_result_required_after.get(key) or 0.0),
        )
        return cutoff

    def get_timeframe_metadata(self, instrument: str, timeframe: str) -> Dict[str, Any]:
        """Return a copy of current input/result provenance for one timeframe."""
        self._ensure_freshness_state()
        key = f"{instrument}_{timeframe}"
        return {
            "input": dict(self._input_metadata.get(key) or {}),
            "result": dict(self._result_metadata.get(key) or {}),
            "required_after_epoch": float(
                self._fresh_result_required_after.get(key) or 0.0
            ),
        }

    def _invalidate_result(self, key: str, status: str, error: str = "") -> Dict[str, Any]:
        """Invalidate, rather than reuse, a result after an unusable processing pass."""
        self._ensure_freshness_state()
        previous = self._result_metadata.get(key) or {}
        invalidated_at = time.time()
        meta = {
            "version": int(previous.get("version") or 0) + 1,
            "status": str(status or "invalid"),
            "computed_at_epoch": 0.0,
            "computed_at": "",
            "invalidated_at_epoch": invalidated_at,
            "invalidated_at": self._utc_iso(invalidated_at),
            "input_version": int((self._input_metadata.get(key) or {}).get("version") or 0),
        }
        if error:
            meta["error"] = str(error)
        self._result_cache.pop(key, None)
        self._data_hash.pop(key, None)
        getattr(self, "_combined_s1_observations", {}).pop(key, None)
        getattr(self, "_combined_base_observations", {}).pop(key, None)
        self._result_metadata[key] = meta
        return meta

    def _s1_light_signature(self, instrument: str, tf: str, data) -> tuple:
        """What an S1 result depends on apart from the forming bar: every closed bar's
        numeric OHLCV and timestamp (so a broker correction to an older bar still
        recomputes at once), the dependency frames, and the S1 settings. Numeric bytes
        only - hashing the whole frame with its string columns is what cost the time."""
        def closed(df):
            if df is None or len(df) < 2:
                return (0,)
            # Column by column: df[list] on a live frame (arrow strings, many blocks)
            # went through take/reindex and cost 3.3ms a call, 36 calls a scan; a
            # single column's numpy view is 0.18ms.
            columns = tuple(
                hash(df[c].to_numpy(dtype=float, na_value=np.nan)[:-1].tobytes())
                for c in ("open", "high", "low", "close", "volume") if c in df.columns
            )
            stamps = df.index.asi8[:-1] if hasattr(df.index, "asi8") else np.asarray([], dtype=np.int64)
            return (len(df), str(df.index[-1]), columns, hash(np.asarray(stamps).tobytes()))
        try:
            from engine.s1_engine import params_from_settings

            params = params_from_settings(instrument=instrument)
        except Exception:
            params = None
        deps = tuple(
            closed(self._data_cache.get(f"{instrument}_{dep}")) for dep in ("5min", "15min") if dep != tf
        )
        return (closed(data), deps, params)

    @staticmethod
    def _forming_row(df) -> tuple:
        if df is None or len(df) == 0:
            return ()
        row = df.iloc[-1]
        return (str(df.index[-1]),) + tuple(
            float(row.get(c, 0.0) or 0.0) for c in ("open", "high", "low", "close", "volume")
        )

    def _remember_s1_signature(self, key: str, instrument: str, tf: str, data) -> None:
        if not isinstance(getattr(self, "_s1_light_signatures", None), dict):
            self._s1_light_signatures = {}
        try:
            self._s1_light_signatures[key] = self._s1_light_signature(instrument, tf, data)
        except Exception:
            self._s1_light_signatures.pop(key, None)

    def _reuse_s1_forming_result(self, key: str, instrument: str, tf: str, data):
        """The cached S1 result when only the forming bar moved within the throttle."""
        try:
            from config.settings import get_settings

            throttle = float(getattr(get_settings(), "s1_lane_forming_recompute_seconds", 15.0) or 0.0)
        except Exception:
            throttle = 0.0
        if throttle <= 0:
            return None
        cached = self._result_cache.get(key)
        meta = self._result_metadata.get(key) or {}
        if cached is None or meta.get("status") != "ready":
            return None
        computed_at = float(meta.get("computed_at_epoch") or 0.0)
        required_after = float(self._fresh_result_required_after.get(key) or 0.0)
        if required_after and computed_at <= required_after:
            return None          # stabilization needs a newer observation
        if time.time() - computed_at >= throttle:
            return None
        remembered = (getattr(self, "_s1_light_signatures", None) or {}).get(key)
        if remembered is None:
            return None
        try:
            current = self._s1_light_signature(instrument, tf, data)
        except Exception:
            return None
        return cached if current == remembered else None

    def _reuse_combined_s1_result(self, key: str, signature):
        """Reuse a native S1 observation without extending its throttle clock.

        The composite's base computation can run every tick. Its fresh envelope
        timestamp is not a new S1 observation and must never renew S1's interval.
        """
        from config.settings import get_settings

        try:
            throttle = float(getattr(get_settings(), "s1_lane_forming_recompute_seconds", 15.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if throttle <= 0:
            return None
        observation = (getattr(self, "_combined_s1_observations", {}) or {}).get(key)
        if not observation or observation.get("signature") != signature:
            return None
        native = observation.get("result")
        if not native or native.get("error"):
            return None
        # Session/recalculation owners also clear the public result store
        # directly. A retained companion observation cannot survive that reset
        # or replacement of the authoritative envelope it came from.
        envelope = self._result_cache.get(key) or {}
        cached_native = (envelope.get("native_results") or {}).get("s1")
        if cached_native is not native or (self._result_metadata.get(key) or {}).get("status") != "ready":
            return None
        computed_at = float(observation.get("computed_at_epoch") or 0.0)
        age = time.time() - computed_at
        required_after = float(self._fresh_result_required_after.get(key) or 0.0)
        if age < 0 or age >= throttle or (required_after and computed_at <= required_after):
            return None
        return native

    def _record_result(self, key: str, input_hash: int) -> Dict[str, Any]:
        self._ensure_freshness_state()
        previous = self._result_metadata.get(key) or {}
        computed_at = time.time()
        input_meta = self._input_metadata.get(key) or {}
        meta = {
            "version": int(previous.get("version") or 0) + 1,
            "status": "ready",
            "computed_at_epoch": computed_at,
            "computed_at": self._utc_iso(computed_at),
            "input_version": int(input_meta.get("version") or 0),
            "input_observed_at_epoch": float(input_meta.get("observed_at_epoch") or 0.0),
            "input_hash": int(input_hash),
        }
        self._result_metadata[key] = meta
        required_after = float(self._fresh_result_required_after.get(key) or 0.0)
        if computed_at > required_after:
            self._fresh_result_required_after.pop(key, None)
        return meta

    def _reuse_combined_base_result(self, key: str, signature, engine):
        """Reuse only evidence that the standalone base cache would retain.

        S1 dependencies do not affect the base calculation. A native UTS
        observation still ages under its existing execution age bound; reading
        the observation must never reset either that clock or its provenance.
        """
        observation = (getattr(self, '_combined_base_observations', {}) or {}).get(key)
        if not observation or observation.get('signature') != signature:
            return None
        if observation.get('engine') is not engine.native_engines[engine.base_mode]:
            return None
        native = observation.get('result')
        envelope = self._result_cache.get(key) or {}
        if (not native or native.get('error')
                or (envelope.get('native_results') or {}).get(engine.base_mode) is not native
                or (self._result_metadata.get(key) or {}).get('status') != 'ready'):
            return None
        computed_at = float(observation.get('computed_at_epoch') or 0.)
        age = time.time() - computed_at
        required_after = float(self._fresh_result_required_after.get(key) or 0.)
        if age < 0 or (required_after and computed_at <= required_after):
            return None
        if engine.base_mode == 'ut_smart':
            from config.settings import get_settings
            try:
                maximum_age = float(getattr(get_settings(), 'uts_lane_result_max_age_seconds', 30.))
            except (TypeError, ValueError):
                return None
            if not np.isfinite(maximum_age) or maximum_age <= 0 or age >= maximum_age:
                return None
        return native

    def process_instrument(
        self,
        instrument: str,
        lot_size: int = 25,
        lots: int = 1,
        capital: float = 100000.0,
        risk_pct: float = 1.0,
    ) -> MultiTFResult:
        """
        Run the configured scanner engine on all 3 timeframes for one instrument.

        Returns MultiTFResult with confluence scoring.
        """
        result = MultiTFResult(
            instrument=instrument,
            timestamp=datetime.now(),
        )

        tf_results = {}
        # Pass-local only: several charts read the same immutable dependency
        # frame. Reuse the exact hash bytes within this pass, never across ticks.
        frame_hashes = {}

        def frame_hash(frame, mode='realtime'):
            if frame is None:
                return None
            identity = (id(frame), mode)
            if identity not in frame_hashes:
                frame_hashes[identity] = self._hash_df(frame, mode)
            return frame_hashes[identity]

        sizing_signature = (capital, risk_pct, lots, lot_size)
        for tf in ["1min", "5min", "15min"]:
            timeframe_started = time.perf_counter()
            key = f"{instrument}_{tf}"
            engine = self.engines.get(key)
            data = self._data_cache.get(key)
            self._ensure_freshness_state()
            input_meta = dict(self._input_metadata.get(key) or {})

            try:
                row_count = int(len(data)) if data is not None else 0
            except Exception:
                row_count = 0

            if engine is None:
                result_meta = self._invalidate_result(key, "engine_missing")
                result.timeframe_metadata[tf] = {
                    "status": "engine_missing",
                    "input": input_meta,
                    "result": dict(result_meta),
                    "used_cache": False,
                    "duration_ms": round((time.perf_counter() - timeframe_started) * 1000.0, 3),
                    "signal_source": str(getattr(engine, "signal_source", "") or ""),
                }
                continue

            minimum = 2 if getattr(engine, "signal_source", "") == "ut_ema" else 15
            if data is None or row_count < minimum:
                input_status = "missing" if data is None else "underfilled"
                # Do not carry an old arrow across a failed/short fetch.  The old
                # behavior explicitly reused _result_cache here, allowing a pending
                # signal to accrue stable scans without any current timeframe input.
                result_meta = self._invalidate_result(key, f"input_{input_status}")
                result.timeframe_metadata[tf] = {
                    "status": f"input_{input_status}",
                    "input": input_meta,
                    "result": dict(result_meta),
                    "used_cache": False,
                    "duration_ms": round((time.perf_counter() - timeframe_started) * 1000.0, 3),
                    "signal_source": str(getattr(engine, "signal_source", "") or ""),
                }
                continue

            # S1 on a forming bar: reuse the last result for a few seconds instead of
            # re-hashing three full frames and recomputing the chart on every tick
            # (2026-09-25 open: 12-24s scans, S1 alone 37% of main.py CPU).
            from engine.scanner_modes import COMBINED_MODES
            combined = str(getattr(engine, 'signal_source', '')) in COMBINED_MODES
            if str(getattr(engine, "signal_source", "") or "") == "s1":
                reused = self._reuse_s1_forming_result(key, instrument, tf, data)
                if reused is not None:
                    tf_results[tf] = reused
                    result.timeframe_metadata[tf] = {
                        "status": "cached",
                        "input": input_meta,
                        "result": dict(self._result_metadata.get(key) or {}),
                        "used_cache": True,
                        "forming_bar_throttle": True,
                        "duration_ms": round((time.perf_counter() - timeframe_started) * 1000.0, 3),
                        "signal_source": "s1",
                    }
                    continue

            # ═══ SKIP if data hasn't changed ═══
            if combined:
                # A companion depends on both full histories and native S1 frames.
                # Include every native parameter and UTS close boundary in one hash;
                # lane errors retry immediately instead of being cached as success.
                from engine.s1_engine import params_from_settings as s1_params
                from engine.ut_smart_engine import params_from_settings as uts_params, evaluation_clock, closed_bar_mask
                from engine.uts_lane import intrabar_enabled
                native_clock = evaluation_clock()
                dependencies = [data, self._data_cache.get(f'{instrument}_5min'),
                                self._data_cache.get(f'{instrument}_15min')]
                closed_dependencies = tuple(
                    bool(closed_bar_mask(frame, dependency_tf, native_clock)[-1])
                    if frame is not None and len(frame) else None
                    for dependency_tf, frame in zip((tf, '5min', '15min'), dependencies)
                )
                # S1 keeps its native forming-bar throttle. Closed/corrected
                # history, native settings and a terminal bar closing (including
                # either dependency) require an immediate new observation.
                s1_signature = (
                    self._s1_light_signature(instrument, tf, data),
                    getattr(engine, 'signal_mode', 'realtime'),
                    sizing_signature,
                    closed_dependencies,
                    # The light signature excludes the terminal row to permit
                    # forming ticks. Once closed, that row is immutable evidence:
                    # broker corrections must invalidate its S1 observation too.
                    tuple(self._forming_row(frame) if closed else None
                          for frame, closed in zip(dependencies, closed_dependencies)),
                )
                base_engine = engine.native_engines[engine.base_mode]
                base_parameters = tuple((name, getattr(base_engine, name, None))
                                        for name in sorted(ENGINE_PARAMETER_KEYS | {
                                            'ema_fast', 'ema_slow', 'st_period', 'st_mult', 'auto_trend_adx'}))
                if engine.base_mode == 'ut_smart':
                    base_intrabar = intrabar_enabled()
                    base_mode = 'realtime' if base_intrabar or closed_dependencies[0] else 'confirmed'
                    base_signature = (frame_hash(data, base_mode), uts_params(timeframe=tf),
                                      base_intrabar, closed_dependencies[0], base_parameters, sizing_signature)
                else:
                    base_signature = (frame_hash(data, getattr(base_engine, 'signal_mode', 'realtime')),
                                      base_parameters, sizing_signature)
                cached_base = self._reuse_combined_base_result(key, base_signature, engine)
                new_hash = hash((tuple(frame_hash(frame, 'realtime')
                                      for frame in dependencies),
                                 s1_params(instrument=instrument), uts_params(timeframe=tf),
                                 intrabar_enabled(), closed_dependencies, base_parameters, sizing_signature))
            elif str(getattr(engine, "signal_source", "") or "") == "s1":
                # Closed bars of this frame and of the native 5m/15m frames (their OHLC
                # is needed before Heikin Ashi conversion), the S1 settings, and this
                # frame's forming bar. Numeric columns only: hash_pandas_object over the
                # whole frame with its string columns was 4.4ms x 3 frames per chart.
                new_hash = hash((self._s1_light_signature(instrument, tf, data), self._forming_row(data)))
            elif str(getattr(engine, "signal_source", "") or "") == "ut_smart":
                # UT SMART EXIT signals on closed candles only, so forming-bar ticks do
                # not change its answer; recompute on a close or a corrected bar.
                from engine.ut_smart_engine import params_from_settings as _uts_params, evaluation_clock, closed_bar_mask
                from engine.uts_lane import intrabar_enabled as _uts_intrabar

                # Intra-candle mode reads the forming candle too.
                _intrabar = _uts_intrabar()
                uts_clock = evaluation_clock()
                uts_last_closed = bool(closed_bar_mask(data, tf, uts_clock)[-1])
                new_hash = hash((self._hash_df(data, "realtime" if _intrabar or uts_last_closed else "confirmed"),
                                 _uts_params(timeframe=tf), _intrabar, uts_last_closed))
            elif str(getattr(engine, "signal_source", "") or "") == "s1s2_dual":
                # The adapter hashes only closed OHLCV plus its full config and required
                # auxiliary feed. Its worker may still be catching up, so cache policy
                # below must let process() poll the last published snapshot each scan.
                from engine.s12_engine import (
                    aux_feeds as s12_aux_feeds,
                    config_from_settings as s12_config_from_settings,
                    evaluation_clock as s12_evaluation_clock,
                    input_signature as s12_input_signature,
                    prepare_frame as s12_prepare_frame,
                )
                from engine.s12_lane import timeframe_enabled as s12_timeframe_enabled
                from config.settings import get_settings as get_s12_settings

                s12_settings = get_s12_settings()
                s12_clock = s12_evaluation_clock()
                if s12_timeframe_enabled(tf, s12_settings):
                    s12_frame = s12_prepare_frame(data, tf, as_of=s12_clock)
                    s12_df_5min = self._data_cache.get(f"{instrument}_5min") if tf == "15min" else None
                    s12_aux = s12_aux_feeds(tf, s12_df_5min, as_of=s12_clock)
                    new_hash = hash((s12_input_signature(
                        s12_frame, s12_aux, s12_config_from_settings(s12_settings)), tf))
                else:
                    new_hash = hash(("s12-disabled", tf, str(getattr(s12_settings, "s12_timeframes", ""))))
            elif str(getattr(engine, "signal_source", "") or "") == "ut_ema":
                from config.settings import get_settings as get_utema_settings

                # Historical corrections and the forming candle both belong to the
                # native input. Closing a candle also changes Pine's timing branch
                # even when its final OHLC did not change since the preceding tick.
                utema_clock = datetime.now(timezone.utc)
                utema_settings = get_utema_settings()
                from engine.ut_ema_engine import signal_fire_mode as utema_fire_mode
                utema_mode = (str(getattr(utema_settings, "trading_mode", "HISTORICAL")),
                              utema_fire_mode(utema_settings))
                last_bar = pd.Timestamp(data.index[-1])
                if last_bar.tzinfo is None:
                    last_bar = last_bar.tz_localize("Asia/Kolkata")
                minutes = {"1min": 1, "5min": 5, "15min": 15}.get(tf, 5)
                utema_closed = last_bar + pd.Timedelta(minutes=minutes) <= pd.Timestamp(utema_clock)
                streamed = getattr(engine, "_stream_result", None) or {}
                hash_data = data[["open", "high", "low", "close"]]
                observation = getattr(engine, "_utema_last_observed_at", streamed.get("utema_last_observed_at"))
                stream_owned = bool(getattr(engine, "_utema_stream_owned", False))
                replay_input_valid = True
                if (utema_mode[1] == "CANDLE_CLOSE"
                        and not stream_owned):
                    # OHLC replay excludes every forming candle. Its output changes
                    # only when a closed candle, its identity, or entry timing changes.
                    # Observed streams retain their tick identity because native
                    # exits and protection can advance between confirmed entries.
                    closed_clock = pd.Timestamp(utema_clock)
                    if hash_data.index.tz is None:
                        closed_clock = closed_clock.tz_convert("Asia/Kolkata").tz_localize(None)
                    closed_mask = hash_data.index + pd.Timedelta(minutes=minutes) <= closed_clock
                    try:
                        tail_values = hash_data.loc[~closed_mask].to_numpy(dtype=float)
                        replay_input_valid = bool(
                            hash_data.index.is_monotonic_increasing
                            and not hash_data.index.has_duplicates
                            and np.isfinite(tail_values).all()
                            and (tail_values > 0).all()
                            and (tail_values[:, 1] >= tail_values.max(axis=1)).all()
                            and (tail_values[:, 2] <= tail_values.min(axis=1)).all())
                    except (TypeError, ValueError, OverflowError):
                        replay_input_valid = False
                    # prepare() validates even rows excluded from replay. Preserve
                    # that failure path rather than hiding bad input behind a hit.
                    hash_data = hash_data.loc[closed_mask]
                    observation = None
                # Native UT/EMA consumes OHLC only. Hashing volume provenance and
                # Arrow string columns made unrelated feed updates rerun it.
                new_hash = hash((self._hash_df(hash_data, "realtime"), utema_mode,
                                 bool(utema_closed), sizing_signature, stream_owned,
                                 observation, replay_input_valid))
            else:
                new_hash = self._hash_df(data, getattr(engine, "signal_mode", "realtime"))
            cached_meta = self._result_metadata.get(key) or {}
            required_after = float(self._fresh_result_required_after.get(key) or 0.0)
            cached_computed_at = float(cached_meta.get("computed_at_epoch") or 0.0)
            cache_is_allowed = not required_after or cached_computed_at > required_after
            if str(getattr(engine, "signal_source", "") or "") == "s1s2_dual":
                cached_s12 = self._result_cache.get(key) or {}
                cached_s12_eval = cached_s12.get("s12_evaluation") or {}
                if cached_s12.get("s12_pending") or (cached_s12_eval and not cached_s12_eval.get("current", False)):
                    cache_is_allowed = False
            if (
                cache_is_allowed
                and new_hash == self._data_hash.get(key, 0)
                and key in self._result_cache
                and cached_meta.get("status") == "ready"
                and (not combined or cached_base is not None)
                and not (combined and any(lane.get('error') for lane in
                         self._result_cache[key].get('native_results', {}).values()))
            ):
                tf_results[tf] = self._result_cache[key]
                result.timeframe_metadata[tf] = {
                    "status": "cached",
                    "input": input_meta,
                    "result": dict(cached_meta),
                    "used_cache": True,
                    "duration_ms": round((time.perf_counter() - timeframe_started) * 1000.0, 3),
                    "signal_source": str(getattr(engine, "signal_source", "") or ""),
                }
            else:
                try:
                    extra = {}
                    if str(getattr(engine, 'signal_source', '') or '') == 'ut_smart':
                        extra['as_of'] = uts_clock
                    if str(getattr(engine, "signal_source", "") or "") == "ut_ema":
                        extra["as_of"] = utema_clock
                    if str(getattr(engine, 'signal_source', '') or '') == 's1s2_dual':
                        from engine.s12_engine import evaluation_clock as s12_evaluation_clock
                        extra['as_of'] = s12_evaluation_clock()
                        if tf == "15min":
                            extra["df_5min"] = self._data_cache.get(f"{instrument}_5min")
                    if combined:
                        extra['as_of'] = native_clock
                        if cached_base is not None:
                            extra['_base_cached_result'] = cached_base
                        cached_s1 = self._reuse_combined_s1_result(key, s1_signature)
                        if cached_s1 is not None:
                            extra['_s1_cached_result'] = cached_s1
                    if str(getattr(engine, "signal_source", "") or "") in {"s1", "s1s2_dual"} or combined:
                        for dep in ("5min", "15min"):
                            if dep != tf and not (getattr(engine, "signal_source", "") == "s1s2_dual" and dep == "15min"):
                                extra[f"df_{dep}"] = self._data_cache.get(f"{instrument}_{dep}")
                    res = engine.process(
                        df=data,
                        instrument=instrument,
                        timeframe=tf,
                        capital=capital,
                        risk_pct=risk_pct,
                        lots=lots,
                        lot_size=lot_size,
                        **extra,
                    )
                    tf_results[tf] = res
                    if res.get("error"):
                        result_meta = self._invalidate_result(key, "processing_error", str(res["error"]))
                        result.timeframe_metadata[tf] = {
                            "status": "processing_error", "input": input_meta, "result": dict(result_meta),
                            "used_cache": False, "signal_source": str(getattr(engine, "signal_source", "") or ""),
                        }
                        continue
                    self._result_cache[key] = res
                    self._data_hash[key] = new_hash
                    result_meta = self._record_result(key, new_hash)
                    if combined:
                        native_base = (res.get('native_results') or {}).get(engine.base_mode)
                        if not isinstance(getattr(self, '_combined_base_observations', None), dict):
                            self._combined_base_observations = {}
                        if not native_base or native_base.get('error'):
                            self._combined_base_observations.pop(key, None)
                        elif cached_base is None:
                            self._combined_base_observations[key] = {
                                'result': native_base,
                                'engine': base_engine,
                                'signature': base_signature,
                                'computed_at_epoch': result_meta['computed_at_epoch'],
                                'metadata': dict(result_meta),
                            }
                        native_s1 = (res.get('native_results') or {}).get('s1')
                        if not isinstance(getattr(self, '_combined_s1_observations', None), dict):
                            self._combined_s1_observations = {}
                        if not native_s1 or native_s1.get('error'):
                            self._combined_s1_observations.pop(key, None)
                        elif cached_s1 is None:
                            self._combined_s1_observations[key] = {
                                'result': native_s1,
                                'signature': s1_signature,
                                'computed_at_epoch': result_meta['computed_at_epoch'],
                                'metadata': dict(result_meta),
                            }
                        res['native_result_metadata'] = {
                            mode: dict(observation['metadata'])
                            for mode, observation in (
                                (engine.base_mode, self._combined_base_observations.get(key)),
                                ('s1', self._combined_s1_observations.get(key)),
                            ) if observation and observation.get('metadata')
                        }
                    if str(getattr(engine, "signal_source", "") or "") == "s1":
                        self._remember_s1_signature(key, instrument, tf, data)
                    result.timeframe_metadata[tf] = {
                        "status": "recomputed",
                        "input": input_meta,
                        "result": dict(result_meta),
                        "used_cache": False,
                        "duration_ms": round((time.perf_counter() - timeframe_started) * 1000.0, 3),
                        "signal_source": str(getattr(engine, "signal_source", "") or ""),
                    }
                except Exception as e:
                    logger.error(f"Error processing {key}: {e}")
                    result_meta = self._invalidate_result(key, "processing_error", str(e))
                    result.timeframe_metadata[tf] = {
                        "status": "processing_error",
                        "input": input_meta,
                        "result": dict(result_meta),
                        "used_cache": False,
                        "duration_ms": round((time.perf_counter() - timeframe_started) * 1000.0, 3),
                        "signal_source": str(getattr(engine, "signal_source", "") or ""),
                    }

        result.results_1min = tf_results.get("1min")
        result.results_5min = tf_results.get("5min")
        result.results_15min = tf_results.get("15min")

        # ── Confluence Scoring ──
        result.confluence_score, result.confluence_signal = self._compute_confluence(
            tf_results, instrument
        )

        # Collect active signals from signal timeframes
        for tf in ["5min", "15min"]:
            res = tf_results.get(tf)
            if res and res.get("signals"):
                # Realtime mode may expose the forming-bar arrow. Confirmed mode
                # only exposes signals whose source candle has closed.
                data_len = len(self._data_cache.get(f"{instrument}_{tf}", []))
                for sig in res["signals"]:
                    latest_allowed = data_len - 1 if getattr(self.engines.get(f"{instrument}_{tf}"), "signal_mode", "realtime") == "realtime" else data_len - 2
                    if sig.bar_index >= data_len - 3 and sig.bar_index <= latest_allowed:
                        result.active_signals.append(sig)

        return result

    def _compute_confluence(
        self,
        tf_results: Dict[str, Dict],
        instrument: str
    ) -> tuple:
        """
        Compute confluence score from multi-timeframe results.

        Scoring:
        - Each timeframe contributes its weight × direction
        - Direction: +1 (LONG), -1 (SHORT), 0 (FLAT)
        - Score > 0.5 → BUY, Score < -0.5 → SELL, else HOLD

        Combo/union engines still score the timeframe state, not every raw union
        arrow. Entries come from active_signals; confluence is context only.
        """
        score = 0.0
        total_weight = 0.0

        for tf, weight in self.tf_weights.items():
            res = tf_results.get(tf)
            if res and not res.get("error") and res.get("state"):
                pos = res["state"]["position"]
                trending = res["state"].get("is_trending", False)

                # Weight trending confirmations more
                effective_weight = weight * (1.3 if trending else 0.7)
                score += pos * effective_weight
                total_weight += effective_weight

        # Normalize to [-1, 1]
        if total_weight > 0:
            score = score / total_weight

        # Determine signal
        if score > 0.35:
            signal = "BUY"
        elif score < -0.35:
            signal = "SELL"
        else:
            signal = "HOLD"

        return round(score, 3), signal

    def apply_engine_params(self, params: dict):
        """Dynamically update UT Engine parameters across all internal engines"""
        self._ensure_freshness_state()
        self.engine_params = params
        self._engine_init_params.update({key: value for key, value in params.items() if key in ENGINE_PARAMETER_KEYS})
        for key, wrapper in self.engines.items():
            targets = [wrapper, *getattr(wrapper, "native_engines", {}).values()]
            for engine in targets:
                if getattr(engine, "signal_source", "") == "ut_ema":
                    continue
                for name, value in params.items():
                    if name in ENGINE_PARAMETER_KEYS and hasattr(engine, name):
                        setattr(engine, name, 'realtime' if name == 'signal_mode' and key.endswith('_1min') else value)
        self._result_cache.clear()
        self._data_hash.clear()
        self._result_metadata.clear()
        self._fresh_result_required_after.clear()
        getattr(self, "_combined_s1_observations", {}).clear()
        getattr(self, "_combined_base_observations", {}).clear()

    def get_all_states(self) -> Dict[str, Dict]:
        """Get current state for all instrument-timeframe combos"""
        states = {}
        for key, engine in self.engines.items():
            state = engine.get_state(key)
            states[key] = {
                "position": state.position,
                "trailing_stop": state.trailing_stop,
                "last_entry_price": state.last_entry_price,
                "last_exit_price": state.last_exit_price,
                "bar_count": state.bar_count,
            }
        return states

    def process_all(
        self,
        instruments_config: Dict,
        capital: float = 100000.0,
        risk_pct: float = 1.0,
        user_lots: Dict[str, int] = None,
    ) -> Dict[str, MultiTFResult]:
        """
        Process all instruments at once.

        Args:
            instruments_config: From instruments.json
            capital: Trading capital
            risk_pct: Risk per trade %
            user_lots: Optional dict of {instrument: lots}

        Returns:
            Dict of instrument -> MultiTFResult
        """
        results = {}
        lots_config = user_lots or {}

        for name, config in instruments_config.get("indices", {}).items():
            lot_size = config.get("lot_size", 25)
            lots = lots_config.get(name, 1)

            result = self.process_instrument(
                instrument=name,
                lot_size=lot_size,
                lots=lots,
                capital=capital,
                risk_pct=risk_pct,
            )
            results[name] = result

        return results
