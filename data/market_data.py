"""
Market Data Provider — Institutional Broker-Master Integration
═══════════════════════════════════════════════════════════════
Primary: Kite execution-broker quotes; supplementary: Angel One
Backup: FYERS, including capability-specific cash-volume and reference data
Yahoo is delayed opt-in only; legacy and two-provider routes remain selectable
"""

import os
import time
import json
import threading
import tempfile
import asyncio
import logging
from collections import deque
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, time as dtime, date
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from zoneinfo import ZoneInfo
from loguru import logger
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
from data.market_calendar import (
    index_session_bounds,
    load_nse_calendar_coverage,
    load_nse_holidays,
    load_nse_special_trading_days,
)
from data.angelone_market_data import AngelOneIndexQuoteSource
from data.market_source_health import build_market_source_snapshot, build_source_health
from data.kite_market_data import KiteDataSource
from engine.option_history_store import get_option_history_store
from trading.api_rate_limiter import get_shared_api_limiter, get_shared_history_limiter
from trading.secure_logging import redact_sensitive_text

# SmartAPI is isolated in data.angelone_market_data. The SDK can log complete request
# headers on errors, so that adapter silences its logger before constructing a client.
# The broad legacy Angel paths in this module remain inert; only index-price fallback is
# restored, never order execution or volume.

YAHOO_TICKERS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "MIDCPNIFTY": "NIFTY_MID_SELECT.NS",
}
INDEX_SPOT_TOKENS = {
    "NIFTY": ("99926000", "NSE"),
    "BANKNIFTY": ("99926009", "NSE"),
    "SENSEX": ("99919000", "BSE"),
    "MIDCPNIFTY": ("99926074", "NSE"),
}
INDEX_SHORT_TOKENS = {
    "26000": "NIFTY",
    "26009": "BANKNIFTY",
    "19000": "SENSEX",
    "26074": "MIDCPNIFTY",
}
FYERS_INDEX_SYMBOLS = {
    "NIFTY": "NSE:NIFTY50-INDEX",
    "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
    "SENSEX": "BSE:SENSEX-INDEX",
    "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX",
}
FYERS_SYMBOL_TO_INDEX = {v: k for k, v in FYERS_INDEX_SYMBOLS.items()}
# Quotable instruments that are NOT tradable indices, so they must stay out of
# FYERS_INDEX_SYMBOLS: that map feeds the tradable-index websocket list, reverse
# lookup and several "which index is this" helpers. Auxiliary quotes can still use
# their own websocket subscription and remain ordinary cached quotes.
#
# India VIX reached the system as AngelOne token 99926017 through the AngelOne socket.
# The Kite migration retired that socket and nothing re-subscribed the token, so
# get_ltp("NSE", "INDIAVIX", "99926017") has had no live tier since: Kite has no
# INDIAVIX instrument, FYERS_INDEX_SYMBOLS has no entry, INDEX_SPOT_TOKENS has no entry
# and YAHOO_TICKERS has no entry. data_store/vix_history.json stopped at
# 2026-08-01 00:06 for that reason. Fyers does quote it, verified live at 13.39.
FYERS_AUX_SYMBOLS = {
    "INDIAVIX": "NSE:INDIAVIX-INDEX",
}


def fyers_quote_symbol(symbol: str) -> str:
    """The Fyers symbol for a tradable index or a quotable auxiliary instrument."""
    key = str(symbol or "").upper()
    return FYERS_INDEX_SYMBOLS.get(key) or FYERS_AUX_SYMBOLS.get(key, "")
INDEX_STRIKE_INTERVALS = {"NIFTY": 50, "BANKNIFTY": 100, "SENSEX": 100, "MIDCPNIFTY": 25}
YAHOO_INTERVALS = {"1min": "1m", "5min": "5m", "15min": "15m"}
YAHOO_MAX_DAYS = {"1m": 7, "5m": 60, "15m": 60}
PREVIOUS_CLOSE_KEYS = ("prev_close_price", "previous_close", "prev_close", "close_previous", "close")
FYERS_LOG_FILENAMES = ("fyersApi.log", "fyersRequests.log", "fyersDataSocket.log")
FYERS_LOGS_READY = False


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _fyers_log_dir(log_dir: Optional[Path] = None) -> Path:
    configured = log_dir or Path(os.getenv("UT1_FYERS_LOG_DIR", "logs/fyers"))
    configured.mkdir(parents=True, exist_ok=True)
    return configured


def _archive_fyers_log(src: Path, archive_dir: Path, *, now: Optional[datetime] = None) -> bool:
    if not src.exists() or not src.is_file():
        return False
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    dest = archive_dir / f"{src.stem}.{stamp}{src.suffix}"
    counter = 1
    while dest.exists():
        dest = archive_dir / f"{src.stem}.{stamp}.{counter}{src.suffix}"
        counter += 1
    src.replace(dest)
    return True


def _prune_fyers_archives(archive_dir: Path, *, retention: int) -> None:
    if retention <= 0 or not archive_dir.exists():
        return
    for name in FYERS_LOG_FILENAMES:
        archives = sorted(
            archive_dir.glob(f"{Path(name).stem}.*{Path(name).suffix}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for old in archives[retention:]:
            try:
                old.unlink()
            except OSError as exc:
                logger.debug(f"Unable to prune old Fyers log archive {old}: {exc}")


def _rotate_fyers_logs(
    *,
    root_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    max_bytes: Optional[int] = None,
    retention: Optional[int] = None,
) -> Path:
    """Keep SDK-created Fyers logs out of the project root before clients open them."""
    root = root_dir or Path(".")
    managed_dir = _fyers_log_dir(log_dir)
    archive_dir = managed_dir / "archive"
    root_archive_dir = managed_dir / "root_archive"
    size_limit = max_bytes or _positive_int_env("UT1_FYERS_LOG_ROTATE_BYTES", 25 * 1024 * 1024)
    keep = retention if retention is not None else _positive_int_env("UT1_FYERS_LOG_RETENTION", 10)

    for name in FYERS_LOG_FILENAMES:
        root_log = root / name
        try:
            if root_log.exists() and root_log.is_file():
                _archive_fyers_log(root_log, root_archive_dir)
        except OSError as exc:
            logger.warning(f"Unable to archive root Fyers log {root_log}: {exc}")

        managed_log = managed_dir / name
        try:
            if managed_log.exists() and managed_log.is_file() and managed_log.stat().st_size >= size_limit:
                _archive_fyers_log(managed_log, archive_dir)
        except OSError as exc:
            logger.warning(f"Unable to rotate Fyers log {managed_log}: {exc}")

    _prune_fyers_archives(archive_dir, retention=keep)
    _prune_fyers_archives(root_archive_dir, retention=keep)
    return managed_dir


def _fyers_sdk_log_path() -> str:
    global FYERS_LOGS_READY
    if not FYERS_LOGS_READY:
        _rotate_fyers_logs()
        FYERS_LOGS_READY = True
    # FyersDataSocket concatenates log_path + filename, so keep the trailing slash.
    return _fyers_log_dir().as_posix().rstrip("/") + "/"

def drop_forming_tail_candle(
    df: "pd.DataFrame",
    interval_minutes: int,
    label: str = "",
    *,
    now: Optional[datetime] = None,
) -> "pd.DataFrame":
    """Drop trailing candles whose interval has not finished yet.

    An intraday history REST call returns the CURRENT bar too, holding only the volume
    traded so far. Merged into the candle store it becomes indistinguishable from a
    completed bar the moment the clock rolls past its minute: the profiler's forming-bar
    check only excuses a bar whose interval has yet to elapse, and its stub walk-back only
    rejects bars under 5% of their neighbours' median. A bar captured eight seconds into a
    minute holds ~13% — comfortably above that floor — so it is measured as a complete
    observation of a quiet market.

    Captured live on 2026-08-14, one fetch per index, all four caught mid-bar:
      BANKNIFTY  36,075 against a 349,417 average -> 0.103x
      MIDCPNIFTY 83,942 against a 539,076 average -> 0.156x
      SENSEX      5,774 against a  17,068 median  -> 0.34x
    and the reading persists until the next fetch, which the same session's log shows can
    be 4m47s later. 665 VOLUME RATIO LOW diagnostics were emitted in that one session.
    Every volume gate (2.5x power move, 1.6x thrust, 1.40x cascade, 1.05x scalp) refuses
    on those numbers, and the volume pillar scores a phantom "no participation" into the
    confidence of every candidate instead of abstaining.

    The forming bar is not ours to store: the live tick builder owns the right edge, and
    the next fetch returns this bar complete. Trimming only the tail, and never the whole
    frame, keeps an empty result from cascading into the volume-less Yahoo fallback.
    """
    if df is None or len(df) == 0 or not isinstance(df.index, pd.DatetimeIndex):
        return df
    try:
        minutes = int(interval_minutes)
    except (TypeError, ValueError):
        return df
    if minutes <= 0:
        return df
    try:
        reference = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=df.index.tz)
        if df.index.tz is None and reference.tzinfo is not None:
            reference = reference.tz_localize(None)
        elif df.index.tz is not None and reference.tzinfo is None:
            reference = reference.tz_localize(df.index.tz)
        cutoff = reference - pd.Timedelta(minutes=minutes)
    except (TypeError, ValueError, AttributeError) as exc:
        logger.debug(f"Forming-bar trim skipped for {label or 'frame'}: {exc}")
        return df

    # Count back from the end only. A hole punched mid-frame by a clock skew would be far
    # worse than the partial bar this exists to remove.
    keep = len(df)
    while keep > 0 and df.index[keep - 1] > cutoff:
        keep -= 1
    if keep == len(df):
        return df
    if keep == 0:
        # Everything looks unfinished — a stopped clock or a one-bar frame. Handing back
        # an empty frame sends the caller down the AngelOne/Yahoo path, and Yahoo reports
        # zero volume for every Indian index ticker, so that trade is never worth making.
        logger.debug(
            f"Forming-bar trim would empty {label or 'frame'} ({len(df)} bars); keeping it."
        )
        return df
    logger.debug(
        f"Dropped {len(df) - keep} still-forming {minutes}m bar(s) from {label or 'history'} "
        f"(newest complete {df.index[keep - 1]})."
    )
    return df.iloc[:keep]


import functools

def retry_on_exception(retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Exponential backoff retry decorator"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            m_delay = delay
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries - 1: raise e
                    logger.warning(f"Retry {i+1}/{retries} for {func.__name__} after {m_delay}s: {e}")
                    time.sleep(m_delay)
                    m_delay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator

from data.kite_angel_migration import KiteAngelMigration


class MarketDataProvider(KiteAngelMigration):
    _option_oi_snapshot_file_lock = threading.RLock()
    _bounded_call_init_lock = threading.Lock()

    def __init__(
        self,
        api_key: str = "",
        client_id: str = "",
        password: str = "",
        totp_secret: str = "",
        start_streams: bool = True,
        allow_auth_refresh: bool = True,
        upstox_only: Optional[bool] = None,
    ):
        # Live startup uses Upstox-only routing. Keep legacy adapters available to
        # callers that explicitly request the compatibility route.
        self.api_key = api_key
        self.client_id = client_id
        self.password = password
        self.totp_secret = totp_secret
        self.start_streams = bool(start_streams)
        # Read-only diagnostics may validate an existing Fyers session, but must not
        # exchange a refresh token or rewrite the credential cache.  Normal runtime
        # behaviour remains unchanged unless a caller opts out explicitly.
        self.allow_auth_refresh = bool(allow_auth_refresh)
        # Keep old general-purpose Angel branches inert. The narrow adapter below has
        # sanitized logging and accepts only index quotes plus independently resolved
        # option contracts; it never receives execution authority.
        self.smart_api = None
        _settings = None
        try:
            from config.settings import get_settings

            _settings = get_settings()
            _angel_api_key = api_key or getattr(_settings, "angelone_api_key", "")
            _angel_client_id = client_id or getattr(_settings, "angelone_client_id", "")
            _angel_password = password or getattr(_settings, "angelone_password", "")
            _angel_totp = totp_secret or getattr(_settings, "angelone_totp_secret", "")
        except Exception:
            _angel_api_key = api_key
            _angel_client_id = client_id
            _angel_password = password
            _angel_totp = totp_secret
        if upstox_only is None:
            route = str(getattr(_settings, "supplementary_market_data", "upstox") or "upstox")
            upstox_only = route in {"upstox", "upstox_angel"}
        self.upstox_only = upstox_only
        self.angel_index = None if upstox_only else AngelOneIndexQuoteSource(
            api_key=_angel_api_key,
            client_id=_angel_client_id,
            password=_angel_password,
            totp_secret=_angel_totp,
        )
        self._angel_prewarm_lock = threading.Lock()
        from data.upstox_market_data import UpstoxMarketDataClient
        self.upstox = UpstoxMarketDataClient(
            timeout=float(getattr(_settings, "upstox_http_timeout_seconds", 7.0)))
        self._upstox_stream = None
        if upstox_only:
            from data.upstox_stream import UpstoxStream
            self._upstox_stream = UpstoxStream(self._receive_upstox_quote, self.upstox.timeout)
        self._angel_prewarm_started = False
        self.kite = None if upstox_only else KiteDataSource(auth=None)
        self.on_kite_reauth = None
        self._is_connected_override = None
        self.kws = None                       # KiteTicker
        # Callable taking one Kite order-update payload. Wired by whoever owns the
        # TradeManager; when unset the feed only logs, so the socket is never coupled
        # to a consumer that may not exist yet.
        self.order_update_sink = None
        self._kite_ws_connected = False
        self._last_kite_ws_tick = 0.0
        self._active_kite_tokens: set = set()
        self._kite_index_names: Dict[str, str] = {}
        self._quote_divergence: Dict[str, Dict[str, Any]] = {}
        # Last cumulative index volume seen from Fyers. Kite reports none for indices,
        # so its price ticks carry this forward rather than zeroing the candle.
        self._index_cum_volume: Dict[str, float] = {}
        # Which feed produced the cumulative total above, and when. A change of
        # source means a change of UNIT (cash shares vs futures contracts), which
        # the candle builder must rebase across rather than subtract.
        self._index_cum_volume_source: Dict[str, str] = {}
        self._index_cum_volume_at: Dict[str, float] = {}
        self._index_intrabar_cash_volume: Dict[str, Dict[str, Any]] = {}
        self._index_intrabar_cash_volume_lock = threading.RLock()
        # Near-month futures volume/OI while the cash feed is quiet. Deliberately kept
        # OUT of the cumulative total above: that total is ticked into the index candle
        # series, whose volume column is Fyers cash SHARES backfilled from history, and
        # contracts are a different quantity by ~1000x rather than a scale of it. This
        # is telemetry and an OI read — reported, never ticked.
        self._index_futures_fallback: Dict[str, Dict[str, float]] = {}
        self._index_futures_volume_at: Dict[str, float] = {}
        self._index_futures_minute_lock = threading.RLock()
        self._index_futures_minute_state: Dict[str, Dict[str, Any]] = {}
        # One in-flight volume refresh at a time; the tick thread never waits on it.
        # Per-index in-flight set, guarded by a short-lived mutex. A single global lock
        # here starved three of four indices; see _request_index_volume_refresh.
        self._index_volume_refresh_guard = threading.Lock()
        self._index_volume_refresh_inflight: set = set()
        # One global Fyers SDK flight per provider. Initialize eagerly for production;
        # _call_bounded also has a locked lazy path for small __new__-based probes.
        self._bounded_call_lock = threading.Lock()
        self._bounded_call_active = None
        self._bounded_call_completed: Dict[Any, Any] = {}
        # Cached tick-callback arity; see _emit_index_tick.
        self._tick_callback_ref = None
        self._tick_callback_takes_source: Optional[bool] = None
        self._rest_lock = threading.Lock()
        self._angel_quote_lease_state = threading.local()
        self._angel_rest_backoff_until: float = 0.0
        self._angel_rest_backoff_failures: int = 0
        self._last_angel_rest_backoff_reason: str = ""
        self._fyers_chain_failure_streak: int = 0
        self._last_fyers_reinit_ts: float = 0.0
        self._ltp_cache: Dict[str, Dict] = {}
        
        # Versioned exchange calendar with optional data_store overrides.
        self.nse_holidays = load_nse_holidays()
        self.nse_special_trading_days = load_nse_special_trading_days()
        self.nse_calendar_coverage_years = load_nse_calendar_coverage()

        self._yf_cache: Dict[str, Dict] = {}
        self._cache_ttl = 5
        self._ltp_cache: Dict[str, Dict[str, Any]] = {}
        self._ws_cache: Dict[str, Dict[str, Any]] = {}
        # Ordered derivative ticks bridge Kite's socket thread to Scanner's 100ms
        # manager without doing trade logic or broker I/O on the socket thread.  A
        # one-value LTP cache loses 100 -> 112 -> 101 between passes; this bounded path
        # retains the 112 target touch and its ordering.  Index ticks already have the
        # candle builder, so only option/future contract tokens enter this history.
        self._contract_tick_history_lock = threading.Lock()
        self._contract_tick_history: Dict[str, deque] = {}
        self._contract_tick_sequences: Dict[str, int] = {}
        self._contract_tick_history_size = 2048
        # Open positions need a longer ordered tail when synchronous scan work
        # delays the 100 ms monitor. Keep candidate subscriptions at 2048.
        self._active_contract_tick_history_size = 32768
        self._active_contract_tick_tokens: set[str] = set()
        self._contract_tick_history_token_limit = 128
        # Kite and Fyers websocket callbacks run on different SDK threads.  Index
        # quotes therefore need their own short-lived guard: without one, a reader can
        # observe a half-transition between the two feeds and, more importantly, both
        # callbacks can independently decide that their price is the one CandleBuilder
        # should consume.  The route state below makes the feeds active/standby for
        # PRICE while still allowing Fyers' cumulative index VOLUME to travel alongside
        # the selected Kite price.
        self._index_quote_lock = threading.RLock()
        self._index_price_route: Dict[str, Dict[str, Any]] = {}
        self._touch_quote_cache: Dict[str, Dict[str, Any]] = {}
        self._index_tick_volume_diag: Dict[str, Dict[str, Any]] = {}
        self._active_angel_tokens = {1: set(["26000", "26009", "26017", "26074"]), 3: set(["19000"])}
        self._last_ltp_time: float = 0
        # When this provider came up. Source health needs it to tell "we waited and no
        # tick came" apart from "we only just started" — without it, a cold start inside
        # market hours looks identical to a dead feed.
        self._provider_started_at: float = time.time()
        self._last_ws_tick: float = 0.0  # Compatibility aggregate; provider health uses separate clocks.
        self._last_angel_ws_tick: float = 0.0
        self._last_fyers_ws_tick: float = 0.0
        self._fyers_ws_connected: bool = False
        self._fyers_ws_connected_since: float = 0.0
        self._fyers_ws_retry_count: int = 0
        self._fyers_ws_gateway_failures: int = 0
        self._fyers_ws_last_error: str = ""
        self._yahoo_request_count: int = 0
        self._yahoo_last_used_at: Optional[float] = None
        self._instruments_df: Optional[pd.DataFrame] = None
        self._inst_cache_path = Path("data_store/instruments.csv")
        self._option_chain_meta: Dict[str, Dict[str, Any]] = {}
        self._option_oi_snapshot_lock = self._option_oi_snapshot_file_lock
        self._fyers_refresh_disabled = False
        self._fyers_auth_required = False
        self._fyers_auth_reason = ""
        self._fyers_token_updated_at = None
        self.fyers: Optional[fyersModel.FyersModel] = None
        self._initialize_replacement(_settings)
        if upstox_only:
            self._market_data_route = "upstox"
        if not upstox_only and self._fyers_enabled():
            self._load_fyers_token()

    def _angel_rest_backoff_config(self) -> Tuple[float, float]:
        try:
            from config.settings import get_settings
            settings = get_settings()
            base = float(getattr(settings, "angel_rest_backoff_seconds", 12.0) or 12.0)
            max_delay = float(getattr(settings, "angel_rest_backoff_max_seconds", 60.0) or 60.0)
        except Exception:
            base, max_delay = 12.0, 60.0
        return max(1.0, base), max(1.0, max_delay)

    def _maybe_self_heal_fyers(self, context: str) -> None:
        """Re-create the Fyers client after a persistent chain-failure streak.

        The 2026-07-03 session needed a manual midday restart to clear ~1600
        chain errors/stale fallbacks; re-initialising the client from the
        stored token is what that restart effectively did, so do it
        automatically instead of waiting for an operator."""
        # Endpoint failures in an optional backup must not recycle its shared
        # session/socket or trigger token refresh while preferred brokers work.
        if self._fyers_backup_enabled():
            return
        try:
            from config.settings import get_settings
            settings = get_settings()
            if not bool(getattr(settings, "fyers_chain_self_heal_enabled", True)):
                return
            threshold = max(3, int(getattr(settings, "fyers_chain_reinit_failure_threshold", 8) or 8))
            cooldown = max(60.0, float(getattr(settings, "fyers_chain_reinit_cooldown_seconds", 300.0) or 300.0))
        except Exception:
            threshold, cooldown = 8, 300.0
        self._fyers_chain_failure_streak = int(getattr(self, "_fyers_chain_failure_streak", 0) or 0) + 1
        if self._fyers_chain_failure_streak < threshold:
            return
        now = time.time()
        if now - float(getattr(self, "_last_fyers_reinit_ts", 0.0) or 0.0) < cooldown:
            return
        self._last_fyers_reinit_ts = now
        self._fyers_chain_failure_streak = 0
        logger.warning(
            f"Fyers self-heal: re-initialising client after chain-failure streak ({context})"
        )
        try:
            self._load_fyers_token()
        except Exception as exc:
            logger.error(f"Fyers self-heal reinit failed: {exc}")

    def _angel_rest_error_text(self, payload: Any) -> str:
        if isinstance(payload, BaseException):
            return str(payload)
        if isinstance(payload, dict):
            try:
                return json.dumps(payload, default=str)
            except Exception:
                return str(payload)
        return str(payload or "")

    def _angel_rest_should_backoff(self, payload: Any) -> bool:
        text = self._angel_rest_error_text(payload).lower()
        markers = (
            "exceeding access rate",
            "access rate",
            "rate limit",
            "too many requests",
            "read timed out",
            "connect timeout",
            "connection timed out",
            "timed out",
            "timeout",
        )
        return any(marker in text for marker in markers)

    def _mark_angel_rest_backoff(self, context: str, payload: Any) -> None:
        base, max_delay = self._angel_rest_backoff_config()
        failures = int(getattr(self, "_angel_rest_backoff_failures", 0) or 0) + 1
        delay = min(max_delay, base * (2 ** max(0, failures - 1)))
        self._angel_rest_backoff_failures = failures
        self._angel_rest_backoff_until = time.time() + delay
        reason = redact_sensitive_text(self._angel_rest_error_text(payload))
        self._last_angel_rest_backoff_reason = f"{context}: {reason}"[:300]
        logger.warning(
            f"Angel REST backoff {delay:.1f}s after {context} failure "
            f"({self._last_angel_rest_backoff_reason})"
        )

    def _mark_angel_rest_success(self) -> None:
        self._angel_rest_backoff_failures = 0
        self._angel_rest_backoff_until = 0.0
        self._last_angel_rest_backoff_reason = ""

    def _angel_rest_ready(self, context: str) -> bool:
        until = float(getattr(self, "_angel_rest_backoff_until", 0.0) or 0.0)
        remaining = until - time.time()
        if remaining > 0:
            logger.debug(f"Skipping Angel REST {context}; backoff active for {remaining:.1f}s")
            return False
        return True

    def _throttle_angel_rest(self, context: str) -> bool:
        if not self._angel_rest_ready(context):
            return False
        if self.smart_api is not None:
            # Best-effort quotes yield immediately so they never queue ahead of
            # order status, protective stops, exits, or position reconciliation.
            limiter = get_shared_api_limiter(self.smart_api)
            acquired = limiter.try_begin_best_effort_quote()
            if acquired:
                lease_state = getattr(self, "_angel_quote_lease_state", None)
                if lease_state is None:
                    lease_state = threading.local()
                    self._angel_quote_lease_state = lease_state
                lease_state.active = True
            return acquired
        lock = getattr(self, "_rest_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._rest_lock = lock
        with lock:
            now = time.time()
            last_rest_time = float(getattr(self, "_last_rest_time", 0.0) or 0.0)
            wait = 0.5 - (now - last_rest_time)
            if wait > 0:
                time.sleep(wait)
                self._last_rest_time = time.time()
            else:
                self._last_rest_time = now
        return True

    def _release_angel_rest_quote(self) -> None:
        lease_state = getattr(self, "_angel_quote_lease_state", None)
        if lease_state is None or not bool(getattr(lease_state, "active", False)):
            return
        lease_state.active = False
        if self.smart_api is not None:
            get_shared_api_limiter(self.smart_api).end_best_effort_quote()

    def _best_effort_angel_rest_call(self, context: str, fn):
        """Run one non-critical REST request without competing with broker calls."""
        if not self._throttle_angel_rest(context):
            return None
        try:
            response = fn()
            if isinstance(response, dict) and response.get("status"):
                self._mark_angel_rest_success()
            elif isinstance(response, dict) and self._angel_rest_should_backoff(response):
                self._mark_angel_rest_backoff(context, response)
            return response
        except Exception as exc:
            if self._angel_rest_should_backoff(exc):
                self._mark_angel_rest_backoff(context, exc)
            logger.debug(f"Angel REST {context} failed: {redact_sensitive_text(exc)}")
            return None
        finally:
            self._release_angel_rest_quote()

    @staticmethod
    def _read_fyers_token_file() -> Dict[str, Any]:
        token_path = Path("fyers_token.json")
        if not token_path.exists():
            return {}
        try:
            return json.loads(token_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"⚠️ Failed to read fyers_token.json: {exc}")
            return {}

    @staticmethod
    def _fyers_profile_token_error(profile: Any) -> bool:
        if not isinstance(profile, dict):
            return False
        try:
            code = int(profile.get("code"))
        except (TypeError, ValueError):
            code = profile.get("code")
        return profile.get("s") == "error" and code in {-8, -15, -16, -17, -300}

    @staticmethod
    def _fyers_ws_gateway_timeout(error: Any) -> bool:
        text = str(error or "").lower()
        return (
            "504" in text
            or "gateway time-out" in text
            or "gateway timeout" in text
            or "cloudflare" in text
        )

    @staticmethod
    def _fyers_ws_backoff_seconds(retry_count: int) -> int:
        return min(60, 2 ** max(1, int(retry_count or 1)))

    @staticmethod
    def _quote_float(values: Dict[str, Any], *keys: str) -> float:
        for key in keys:
            try:
                value = float((values or {}).get(key) or 0.0)
            except (TypeError, ValueError, AttributeError):
                value = 0.0
            if value > 0:
                return value
        return 0.0

    def _remember_previous_close(self, symbol: str, previous_close: float, source: str) -> float:
        previous_close = float(previous_close or 0.0)
        if previous_close <= 0:
            return 0.0
        if not hasattr(self, "_prev_close_global_cache"):
            self._prev_close_global_cache = {}
        if not hasattr(self, "_prev_close_cache_time"):
            self._prev_close_cache_time = {}
        if not hasattr(self, "_prev_close_cache_source"):
            self._prev_close_cache_source = {}
        self._prev_close_global_cache[symbol] = previous_close
        self._prev_close_cache_time[symbol] = time.time()
        self._prev_close_cache_source[symbol] = str(source or "")
        return previous_close

    def _fyers_index_quote_values(self, symbol: str) -> Dict[str, Any]:
        fyers_symbol = FYERS_INDEX_SYMBOLS.get(symbol)
        if not getattr(self, "fyers", None) or not fyers_symbol:
            return {}
        ok, resp = self._call_fyers_bounded(
            "quotes",
            {"symbols": fyers_symbol},
            label=f"Fyers index quote {fyers_symbol}",
        )
        if not ok:
            return {}
        if not isinstance(resp, dict) or resp.get("s") != "ok":
            return {}
        values = (resp.get("d") or [{}])[0].get("v", {})
        return dict(values or {}) if isinstance(values, dict) else {}

    @staticmethod
    def _configure_fyers_http_timeout(client: Any, timeout_seconds: float) -> bool:
        """Install a real requests timeout under the Fyers sync SDK.

        Fyers' synchronous service creates a ``requests.Session`` but does not pass a
        timeout to ``Session.get/post``.  The single-flight wrapper protects our caller,
        yet an actually hung transport would otherwise own its sole worker forever.  A
        default at the SDK session boundary makes normal network stalls terminate and
        release that slot without spawning replacement threads.
        """
        try:
            session = getattr(getattr(client, "service", None), "session", None)
            if session is None or not callable(getattr(session, "request", None)):
                return False
            timeout = max(1.0, min(20.0, float(timeout_seconds or 8.0)))
            original = getattr(session, "_utbot_original_request", None)
            if not callable(original):
                original = session.request
                session._utbot_original_request = original

            def _request_with_timeout(method, url, *args, **kwargs):
                if kwargs.get("timeout") is None:
                    kwargs["timeout"] = timeout
                return original(method, url, *args, **kwargs)

            _request_with_timeout._utbot_default_timeout = timeout
            session.request = _request_with_timeout
            return True
        except Exception as exc:
            logger.debug(f"Could not configure Fyers HTTP timeout: {exc}")
            return False

    def _call_fyers_bounded(
        self,
        method_name: str,
        *args,
        label: str,
        timeout_seconds: Optional[float] = None,
        wait_if_busy: bool = True,
        **kwargs,
    ):
        """Serialize every recurring Fyers REST call behind the provider flight.

        The session-level HTTP timeout guarantees eventual transport release; this
        wrapper additionally keeps the scanner loop non-blocking and prevents quote,
        history and option-chain calls from concurrently reusing the same SDK session.
        Authentication/profile checks remain startup operations and already use the
        configured transport timeout.
        """
        client = getattr(self, "fyers", None)
        if bool(getattr(self, "_fyers_auth_required", False)):
            return False, None
        retry_at = self.__dict__.setdefault("_fyers_endpoint_retry_at", {})
        request_key = (method_name, json.dumps(kwargs.get("data"), sort_keys=True, default=str))
        if time.time() < max(float(retry_at.get(method_name, 0.0)), float(retry_at.get(request_key, 0.0))):
            return False, None
        method = getattr(client, str(method_name or ""), None)
        if not callable(method):
            return False, None
        try:
            from config.settings import get_settings

            timeout = float(
                timeout_seconds
                if timeout_seconds is not None
                else getattr(get_settings(), "fyers_history_timeout_seconds", 8.0) or 8.0
            )
        except Exception:
            timeout = float(timeout_seconds if timeout_seconds is not None else 8.0)
        ok, response = self._call_bounded(
            method,
            *args,
            timeout=timeout,
            label=str(label or f"Fyers {method_name}"),
            wait_if_busy=wait_if_busy,
            **kwargs,
        )
        if client is not getattr(self, "fyers", None):
            # An old request can finish after a manual login installs a new
            # session. Its rejection must not quarantine the replacement client.
            return ok, response
        if self._fyers_profile_token_error(response):
            self._mark_fyers_auth_required("FYERS backup session rejected; preferred broker sessions are unchanged")
        elif ok and isinstance(response, dict) and response.get("s") == "error":
            detail = str(response.get("message") or "").lower()
            limited = response.get("code") in {-360, 429} or any(
                term in detail for term in ("requests/minute", "requests/day", "too many requests"))
            retry_at[method_name if limited else request_key] = time.time() + (60.0 if limited else 180.0)
            # A bad symbol/expiry must not disable this endpoint for other indices.
            if len(retry_at) > 64:
                expired = [key for key, until in retry_at.items() if until <= time.time()]
                for key in expired:
                    retry_at.pop(key, None)
        elif ok:
            retry_at.pop(request_key, None)
        return ok, response

    def _fyers_ws_token_ready_for_reconnect(self, access_token: str, ws_generation: int) -> bool:
        token_data = self._read_fyers_token_file()
        file_token = str(token_data.get("access_token") or "")
        if file_token and file_token != access_token:
            logger.info("Fyers token file changed; restarting WebSocket with refreshed token.")
            self.start_fyers_websocket(file_token)
            return False

        token_to_check = file_token or access_token
        if not token_to_check:
            self._mark_fyers_auth_required("Fyers WebSocket reconnect has no access token")
            self._fyers_ws_auth_failed = True
            return False

        try:
            from config.settings import get_settings
            settings = get_settings()
            model = fyersModel.FyersModel(
                client_id=settings.fyers_app_id,
                token=token_to_check,
                log_path=_fyers_sdk_log_path(),
            )
            self._configure_fyers_http_timeout(
                model,
                getattr(settings, "fyers_history_timeout_seconds", 8.0),
            )
            profile = model.get_profile()
            if self._fyers_profile_token_error(profile):
                logger.warning(
                    f"Fyers token failed pre-reconnect validation ({profile}); attempting refresh."
                )
                if token_data and self._refresh_fyers_token(token_data):
                    return False
                self._fyers_ws_auth_failed = True
                self._mark_fyers_auth_required("Fyers WebSocket token invalid and refresh failed")
                return False
            if getattr(self, "_fyers_ws_generation", 0) == ws_generation:
                self.fyers = model
                self._clear_fyers_auth_required()
                return True
            return False
        except Exception as exc:
            logger.warning(f"Fyers token pre-reconnect validation failed; retrying existing WebSocket token: {exc}")
            return True

    def _load_fyers_token(self):
        if not self._fyers_enabled():
            return
        try:
            token_data = self._read_fyers_token_file()
            if not token_data:
                self.fyers = None
                if Path("fyers_token.json").exists():
                    reason = "Fyers token cache is empty or unreadable"
                else:
                    reason = "no cached Fyers token"
                self._mark_fyers_auth_required(reason)
                logger.warning(f"⚠️ {reason}; complete login at /api/fyers_login")
                return

            access_token = token_data.get("access_token")
                
            if access_token:
                from config.settings import get_settings
                settings = get_settings()
                self.fyers = fyersModel.FyersModel(
                    client_id=settings.fyers_app_id,
                    token=access_token,
                    log_path=_fyers_sdk_log_path(),
                )
                self._configure_fyers_http_timeout(
                    self.fyers,
                    getattr(settings, "fyers_history_timeout_seconds", 8.0),
                )
                
                # Verify token status
                profile = self.fyers.get_profile()
                if self._fyers_profile_token_error(profile):
                    if not bool(getattr(self, "allow_auth_refresh", True)):
                        self.fyers = None
                        self._mark_fyers_auth_required(
                            "Fyers token is invalid; read-only validation did not refresh it. "
                            "Manual login is required."
                        )
                        logger.warning(
                            "Fyers token is invalid; refresh suppressed for this read-only "
                            "provider. Complete login at /api/fyers_login."
                        )
                    else:
                        logger.warning(f"🔄 Fyers Token Issue (Code {profile.get('code')}). Attempting refresh...")
                        if not self._refresh_fyers_token(token_data):
                            self.fyers = None
                            if not self._fyers_auth_reason:
                                self._mark_fyers_auth_required("Fyers token expired and manual login is required")
                elif isinstance(profile, dict) and profile.get("s") == "ok":
                    self._clear_fyers_auth_required()
                    logger.info("✅ Fyers API Connected for Data!")
                    if self.start_streams:
                        self.start_fyers_websocket(access_token)
                    else:
                        logger.info("Fyers WebSocket skipped for history-only provider.")
                else:
                    # A missing/malformed/error profile is not proof of authentication.
                    # Keeping the SDK object here made `fyers is not None` falsely read
                    # as connected even though no authenticated call had succeeded.
                    self.fyers = None
                    detail = redact_sensitive_text(profile)
                    self._mark_fyers_unavailable(
                        f"Fyers session could not be validated: {detail or 'empty profile response'}"
                    )
                    logger.warning(
                        "⚠️ Fyers session validation did not return an authenticated "
                        "profile; the FYERS feed stays unavailable until validation succeeds. "
                        "Preferred broker sessions are evaluated independently."
                    )
            else:
                logger.warning("⚠️ Fyers access token missing in fyers_token.json")
                self._mark_fyers_auth_required("Fyers access token missing")
        except Exception as e:
            self.fyers = None
            self._mark_fyers_unavailable(
                f"Fyers session validation failed: {redact_sensitive_text(e)}"
            )
            logger.warning(f"⚠️ Failed to load Fyers token: {e}")

    @staticmethod
    def _close_ws_socket(sock, context: str) -> bool:
        """Tear down a broker websocket via the SDK's REAL close method.

        Neither FyersDataSocket nor SmartWebSocketV2 has `.close()` — both
        expose `close_connection()`. Calling `.close()` raised AttributeError
        on every silent-disconnect/auth/restart teardown (20x on 2026-07-08),
        so stale sockets were never closed before reconnecting. Tries the real
        method first, tolerates SDK variants, and never raises."""
        if not sock:
            return False
        for method_name in ("close_connection", "close", "stop"):
            method = getattr(sock, method_name, None)
            if callable(method):
                try:
                    method()
                    return True
                except Exception as exc:
                    logger.debug(f"WS {method_name}() during {context} failed: {exc}")
                    return False
        logger.debug(f"WS teardown during {context}: socket has no close method")
        return False

    def _call_bounded(
        self,
        func,
        *args,
        timeout: float,
        label: str,
        wait_if_busy: bool = True,
        **kwargs,
    ):
        """Run one broker SDK call at a time without ever blocking an asyncio loop.

        ``fyers.history`` has no transport timeout.  Killing a stuck Python thread is
        unsafe, so the correct containment is *single flight*: while one call is still
        outstanding, do not start another one against the same SDK object.  A completed
        late result is retained briefly and consumed by the next identical request.

        Synchronous/background callers may wait up to ``timeout``.  Calls made from an
        active asyncio loop return immediately with a cache miss, allowing the existing
        source fallback to run.  This is deliberately global per provider rather than
        per symbol: Fyers' SDK object is not documented as thread-safe, and a hung call
        must not cause concurrent re-use or an unbounded queue of daemon workers.
        """
        # A provider is sometimes constructed with __new__ in small tests and startup
        # probes, so make the state lazy instead of relying on __init__.
        if not hasattr(self, "_bounded_call_lock"):
            # Two concurrent first calls used to both observe the attribute missing and
            # install different instance locks, launching two SDK workers. Serialize the
            # one-time fallback initialization just as __init__ does eagerly.
            with type(self)._bounded_call_init_lock:
                if not hasattr(self, "_bounded_call_lock"):
                    self._bounded_call_lock = threading.Lock()
                    self._bounded_call_active = None
                    self._bounded_call_completed = {}
        # A late history response is useful only briefly.  Keeping it indefinitely can
        # replay a pre-outage candle payload after the provider has recovered, even
        # though the dictionary's entry-count cap prevents a memory leak.
        try:
            completed_ttl = min(
                30.0,
                max(
                    0.0,
                    float(getattr(self, "_bounded_call_completed_ttl_seconds", 30.0)),
                ),
            )
        except (TypeError, ValueError):
            completed_ttl = 30.0

        # ``repr`` is stable enough for the small JSON-like SDK payloads used here and
        # prevents a response for one date/resolution being handed to another request.
        request_key = (str(label), repr(args), repr(sorted(kwargs.items(), key=lambda item: item[0])))
        started = time.monotonic()
        try:
            timeout_seconds = max(0.0, float(timeout or 0.0))
        except (TypeError, ValueError):
            timeout_seconds = 0.0
        deadline = started + timeout_seconds
        try:
            asyncio.get_running_loop()
            on_asyncio_loop = True
        except RuntimeError:
            on_asyncio_loop = False

        # A synchronous scanner worker may arrive while another healthy request owns
        # the provider.  It waits only inside its ORIGINAL deadline, then acquires the
        # slot and performs its own request.  Returning an immediate miss here made the
        # concurrent index workers record a healthy serializer collision as a Fyers
        # outage/cooldown.  Async-loop callers still return immediately.
        while True:
            wait_state = None
            with self._bounded_call_lock:
                now = time.monotonic()
                for cached_key, cached_outcome in list(self._bounded_call_completed.items()):
                    try:
                        cached_at = float(cached_outcome[3])
                    except (IndexError, TypeError, ValueError):
                        cached_at = 0.0
                    if cached_at <= 0.0 or now - cached_at > completed_ttl:
                        self._bounded_call_completed.pop(cached_key, None)
                completed = self._bounded_call_completed.pop(request_key, None)
                if completed is not None:
                    ok, value, error, completed_at = completed
                    if ok:
                        logger.debug(
                            f"{label} consumed a late bounded result after "
                            f"{time.monotonic() - completed_at:.2f}s in cache"
                        )
                        return True, value
                    if error is not None:
                        logger.debug(f"{label} completed with an error: {error}")
                    return False, None

                active = self._bounded_call_active
                if active is not None:
                    active_key = active["key"]
                    if active_key == request_key:
                        logger.debug(f"{label} is already in flight")
                    else:
                        logger.debug(
                            f"{label} queued behind {active['label']} within its "
                            "bounded synchronous deadline"
                        )
                    if on_asyncio_loop or not wait_if_busy:
                        return False, None
                    wait_state = active
                else:
                    outcome: Dict[str, Any] = {}
                    done = threading.Event()
                    state = {
                        "key": request_key,
                        "label": str(label),
                        "outcome": outcome,
                        "done": done,
                        "started": started,
                    }
                    self._bounded_call_active = state

            if wait_state is None:
                break

            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0.0 or not wait_state["done"].wait(remaining):
                logger.warning(
                    f"⏱️ {label} exceeded {timeout_seconds:.1f}s while "
                    f"waiting for {wait_state['label']}; falling through to the next source"
                )
                return False, None
            if wait_state["key"] == request_key:
                # Every waiter for an identical request may share the completed result;
                # the launching caller also owns the same in-memory outcome object.
                shared = wait_state["outcome"]
                with self._bounded_call_lock:
                    self._bounded_call_completed.pop(request_key, None)
                if not shared.get("ok"):
                    error = shared.get("error")
                    if error is not None:
                        logger.debug(f"{label} completed with an error: {error}")
                    return False, None
                return True, shared.get("value")

        def _run() -> None:
            try:
                outcome["value"] = func(*args, **kwargs)
                outcome["ok"] = True
            except Exception as exc:  # noqa: BLE001 - converted to normal source miss
                outcome["error"] = exc
                outcome["ok"] = False
            finally:
                finished = time.monotonic()
                with self._bounded_call_lock:
                    # Keep only a small number of late completions.  This avoids both a
                    # memory leak and replaying an old payload after a long outage.
                    self._bounded_call_completed[request_key] = (
                        bool(outcome.get("ok")),
                        outcome.get("value"),
                        outcome.get("error"),
                        finished,
                    )
                    while len(self._bounded_call_completed) > 16:
                        oldest = next(iter(self._bounded_call_completed))
                        self._bounded_call_completed.pop(oldest, None)
                    if self._bounded_call_active is state:
                        self._bounded_call_active = None
                # Publish the cache entry and clear the active slot before waking a
                # synchronous waiter.  Signalling first allowed the waiter to consume
                # ``outcome`` and pop an as-yet absent cache entry; the worker then wrote
                # that already-consumed result back as a stale late completion.
                done.set()

        worker = threading.Thread(target=_run, name=f"FyersSingleFlight-{label}", daemon=True)
        worker.start()

        if on_asyncio_loop:
            # Never call Event.wait()/Thread.join() here: both would starve the scanner
            # loop and dashboard RPC budget.  The next poll consumes a healthy result.
            return False, None

        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0.0 or not done.wait(remaining):
            logger.warning(
                f"⏱️ {label} exceeded {timeout_seconds:.1f}s; retaining its "
                "single-flight slot and falling through to the next source"
            )
            return False, None

        # The worker also placed this response in the late-result cache. This caller
        # consumed it synchronously whether it succeeded or failed, so remove it before
        # returning. Leaving a synchronous failure cached skipped the next real retry.
        with self._bounded_call_lock:
            self._bounded_call_completed.pop(request_key, None)
        if not outcome.get("ok"):
            error = outcome.get("error")
            if error is not None:
                logger.debug(f"{label} failed after {time.monotonic() - started:.2f}s: {error}")
            return False, None
        return True, outcome.get("value")

    def shutdown_streams(self, timeout: float = 3.0) -> bool:
        """Bounded shutdown for the Fyers and Kite market-data sockets.

        SDK close calls are not uniformly bounded (and KiteTicker owns its own worker
        thread), so invoke each close on a daemon helper and wait only up to ``timeout``.
        Advancing the Fyers generation first prevents its reconnect loop from creating a
        replacement socket while shutdown is in progress. This is used by audits and the
        application shutdown path; it never submits or changes an order.
        """
        if getattr(self, "upstox_only", False):
            self.start_streams = False
            return self._upstox_stream.stop(timeout)
        self._fyers_ws_generation = getattr(self, "_fyers_ws_generation", 0) + 1
        self._fyers_ws_auth_failed = True
        self._fyers_ws_connected = False
        self._fyers_ws_connected_since = 0.0
        self._kite_ws_connected = False
        replacement_stop = getattr(self, "_replacement_stop", None)
        if replacement_stop is not None:
            replacement_stop.set()
            self._replacement_closed = True
        sockets = [
            ("angel shutdown", getattr(self, "angel_index", None)),
            ("fyers shutdown", getattr(self, "_fyers_ws", None)),
            ("kite shutdown", getattr(self, "kws", None)),
        ]
        workers = []
        results: Dict[str, bool] = {}

        def _close(label: str, socket: Any) -> None:
            results[label] = self._close_ws_socket(socket, label)

        for label, socket in sockets:
            if socket is None:
                results[label] = True
                continue
            worker = threading.Thread(
                target=_close,
                args=(label, socket),
                name=f"MarketDataShutdown-{label.split()[0]}",
                daemon=True,
            )
            worker.start()
            workers.append((label, worker))

        deadline = time.monotonic() + max(0.1, float(timeout or 0.1))
        for label, worker in workers:
            worker.join(max(0.0, deadline - time.monotonic()))
            if worker.is_alive():
                results[label] = False
                logger.warning(f"Market-data {label} exceeded shutdown timeout")

        self._fyers_ws = None
        self.kws = None
        return all(results.values())

    def start_upstox_stream(self):
        if getattr(self, "_upstox_stream", None) and self.start_streams:
            self._upstox_stream.start()

    def _receive_upstox_quote(self, key, quote):
        from data.upstox_market_data import UPSTOX_INDEX_KEYS

        received_at = quote["received_at"]
        observed_at = quote["time"]
        if not -1 <= received_at - observed_at <= 15:
            return
        self._ws_cache[key] = quote
        index = next((symbol for symbol, instrument_key in UPSTOX_INDEX_KEYS.items()
                      if key == instrument_key), "")
        if index:
            self._ws_cache[index] = quote
            callback = getattr(self, "on_tick", None)
            if index in INDEX_SPOT_TOKENS and callback and self.is_market_open():
                timestamp = datetime.fromtimestamp(observed_at, ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
                callback(index, quote["ltp"], 0, timestamp, volume_source="unavailable")
        else:
            last_trade_at = quote["last_trade_at"]
            previous = getattr(self, "_upstox_last_derivative_trade", None)
            if previous is None:
                previous = self._upstox_last_derivative_trade = {}
            if -1 <= received_at - last_trade_at <= 15 and last_trade_at > previous.get(key, 0):
                previous[key] = last_trade_at
                if len(previous) > 256:
                    previous.pop(min(previous, key=previous.get))
                self._record_contract_tick(key, quote["ltp"], received_at, exchange_at=last_trade_at)
            future = getattr(self, "_upstox_front_futures", {}).get(key)
            if future is not None:
                symbol, contract = future
                self._record_index_futures_volume_snapshot(
                    symbol, {"tradingsymbol": contract, "volume": quote.get("volume")},
                    now_ts=observed_at)
            else:
                self._record_option_participation_tick(key, quote, received_at)

    def _upstox_quote(self, key, max_age):
        if not key:
            return {}
        if self._upstox_stream:
            self._upstox_stream.watch(key)
        now = time.time()
        for row in (self._ws_cache.get(key), self._ltp_cache.get(key)):
            if row and -1 <= now - row["time"] <= max_age:
                return row
        quote = self.upstox.get_quote(key)
        if quote:
            self._ltp_cache[key] = quote
        return quote

    def enable_live_streams(self) -> bool:
        """Enable broker sockets after a runtime REPLAY -> live-mode switch.

        The provider can be constructed while the scanner is in REPLAY, which keeps
        both sockets off.  A later dashboard/premarket switch to PAPER or REAL used to
        change only scanner state; ``start_streams`` stayed false for the lifetime of
        the provider.  Manual broker logins then refreshed REST clients successfully
        but deliberately skipped their sockets, leaving the live dashboard degraded.

        Return True only for the first transition.  Later auth callbacks use the now
        enabled flag to restart their own socket with the new token.
        """
        if bool(getattr(self, "start_streams", False)):
            return False

        self.start_streams = True
        if getattr(self, "upstox_only", False):
            self.start_upstox_stream()
            return True
        self._prewarm_angelone_fallback()
        started = []

        # Check Upstox primary streaming / market data
        try:
            upstox = getattr(self, "upstox", None)
            if upstox is not None and getattr(upstox, "is_healthy", lambda: False)():
                started.append("Upstox")
        except Exception as exc:
            logger.error(f"Upstox activation after live-mode switch failed: {exc}")

        try:
            angel = getattr(self, "angel", None)
            if angel is not None and getattr(angel, "is_healthy", lambda: False)():
                started.append("AngelOne")
        except Exception:
            pass

        # Primary streaming cannot depend on FYERS' aggregate connection flag
        # or wait behind backup socket initialization.
        try:
            kite = getattr(self, "kite", None)
            if bool(getattr(kite, "is_connected", False)):
                if self.start_kite_ticker():
                    started.append("Kite")
        except Exception as exc:
            logger.error(f"Kite WebSocket activation after live-mode switch failed: {exc}")

        try:
            token_data = self._read_fyers_token_file()
            access_token = str(token_data.get("access_token") or "")
            if (
                access_token
                and getattr(self, "fyers", None) is not None
                and not bool(getattr(self, "_fyers_auth_required", False))
            ):
                self.start_fyers_websocket(access_token)
                started.append("Fyers")
        except Exception as exc:
            logger.error(f"Fyers WebSocket activation after live-mode switch failed: {exc}")

        logger.info(
            "Live market-data streams enabled after runtime mode switch"
            + (f": {', '.join(started)}" if started else "; awaiting broker login")
        )
        return True

    @staticmethod
    def _fyers_ws_tick_stalled(last_tick: float, connected_since: float,
                               now_ts: float, stall_seconds: float) -> bool:
        """True when the WS is connected but ticks have stalled past stall_seconds.

        Covers both 'connected but never delivered a tick' (last_tick<=0 after
        open) and 'ticks went silent' (last_tick aged out). Pure/ testable so the
        detection threshold can be tuned and regression-guarded independently of
        the reconnect loop and the market-hours guard around it."""
        try:
            stall = max(5.0, float(stall_seconds))
        except (TypeError, ValueError):
            stall = 60.0
        no_tick_after_open = (
            last_tick <= 0 and connected_since > 0 and (now_ts - connected_since) > stall
        )
        stale_tick_after_open = last_tick > 0 and (now_ts - last_tick) > stall
        return bool(no_tick_after_open or stale_tick_after_open)

    def start_fyers_websocket(self, access_token: str):
        """Start Fyers WebSocket in a separate thread."""
        if not self._fyers_enabled():
            return
        self._fyers_ws_generation = getattr(self, "_fyers_ws_generation", 0) + 1
        ws_generation = self._fyers_ws_generation
        self._fyers_ws_connected = False
        self._fyers_ws_connected_since = 0.0
        # A socket-open callback is not a market-data tick. Keep feed freshness false
        # until an actual quote arrives, otherwise startup DNS failures can look fresh.
        self._last_fyers_ws_tick = 0.0
        if hasattr(self, "_fyers_ws") and self._fyers_ws:
            self._close_ws_socket(self._fyers_ws, "pre-restart teardown")
            
        try:
            def on_message(msg):
                if getattr(self, "_fyers_ws_generation", 0) != ws_generation:
                    return
                if isinstance(msg, list):
                    messages = msg
                else:
                    messages = [msg]
                    
                for m in messages:
                    symbol = m.get('symbol')
                    ltp = m.get('ltp')
                    oi = m.get('oi')
                    vol, vol_key, vol_fields = self._extract_index_tick_volume(
                        m,
                        (
                            "vol_traded_today",
                            "volume_traded_today",
                            "volume",
                            "vol",
                        ),
                    )
                    
                    if symbol and ltp:
                        last_ws_debug = getattr(self, "_last_ws_debug_log", {})
                        now_ts = time.time()
                        exchange_at = self._exchange_timestamp_from_payload(m)
                        with self._index_quote_guard():
                            prior_cache = dict(self._ws_cache.get(symbol) or {})
                        prior_volume = float(prior_cache.get("vol", 0.0) or 0.0)
                        prior_volume_at = float(prior_cache.get("vol_time", 0.0) or 0.0)
                        # Price ticks often omit cumulative volume. Preserve the last
                        # valid total and its ORIGINAL observation clock. Re-stamping a
                        # frozen total on every price tick makes volume look fresh forever.
                        cached_volume = float(vol or 0.0) if vol else prior_volume
                        if vol and (float(vol) != prior_volume or prior_volume_at <= 0.0):
                            volume_observed_at = now_ts
                        else:
                            volume_observed_at = prior_volume_at
                        cache_entry = {
                            'ltp': float(ltp),
                            'oi': int(oi) if oi else 0,
                            'vol': int(cached_volume) if cached_volume else 0,
                            'vol_time': volume_observed_at,
                            'volume_key': vol_key,
                            'volume_fields': dict(vol_fields or {}),
                            'change': float(m.get('ch') or 0.0),
                            'change_pct': float(m.get('chp') or 0.0),
                            'previous_close': self._quote_float(m, *PREVIOUS_CLOSE_KEYS),
                            'time': now_ts,
                            'received_at': now_ts,
                            'exchange_at': exchange_at,
                            'source': 'fyers_ws',
                        }
                        with self._index_quote_guard():
                            self._ws_cache[symbol] = cache_entry
                        if now_ts - last_ws_debug.get(symbol, 0) >= 10:
                            logger.debug(f"Fyers WS Tick: {symbol} -> {ltp}")
                            last_ws_debug[symbol] = now_ts
                            self._last_ws_debug_log = last_ws_debug
                        
                        self._last_fyers_ws_tick = time.time()
                        self._last_ws_tick = self._last_fyers_ws_tick
                        self._fyers_ws_retry_count = 0
                        self._fyers_ws_gateway_failures = 0
                        self._fyers_ws_last_error = ""
                        
                        if hasattr(self, 'on_tick') and self.on_tick:
                            if symbol in FYERS_SYMBOL_TO_INDEX:
                                index_name = FYERS_SYMBOL_TO_INDEX[symbol]
                                # Fyers is the only feed carrying index volume (Kite
                                # returns None — an index is not a traded instrument).
                                # Remember the cumulative total so a Kite price tick can
                                # carry it forward instead of resetting the candle's
                                # volume delta to zero.
                                if vol:
                                    # Stamp freshness only when the cumulative total
                                    # actually ADVANCED. A vendor that keeps republishing
                                    # a frozen non-zero figure — the 2026-07-27 shape,
                                    # where the quote cache looked healthy while candles
                                    # were empty — would otherwise read as fresh forever
                                    # and the futures fallback would never engage.
                                    previous = float(self._index_cum_volume.get(index_name, 0.0) or 0.0)
                                    self._index_cum_volume[index_name] = float(vol)
                                    self._index_cum_volume_source[index_name] = "fyers_index"
                                    if float(vol) != previous:
                                        self._index_cum_volume_at[index_name] = now_ts
                                    elif index_name not in self._index_cum_volume_at:
                                        self._index_cum_volume_at[index_name] = now_ts
                                self._record_index_tick_volume_diag(
                                    index_name,
                                    "fyers_ws",
                                    float(vol or 0.0),
                                    vol_key,
                                    vol_fields,
                                    now_ts,
                                    volume_source="fyers_index" if vol else "",
                                    volume_basis="cumulative_cash_index_volume" if vol else "",
                                    volume_unit="shares" if vol else "",
                                    volume_comparable=bool(vol),
                                )
                                self._emit_arbitrated_index_tick(
                                    index_name,
                                    float(ltp),
                                    int(vol) if vol else 0,
                                    "fyers_index" if vol else "",
                                    "fyers_ws",
                                    received_at=now_ts,
                                    exchange_at=exchange_at,
                                )

            def on_open():
                if getattr(self, "_fyers_ws_generation", 0) != ws_generation:
                    return
                logger.info("🔌 Fyers WebSocket Connected")
                self._fyers_ws_connected = True
                self._fyers_ws_connected_since = time.time()
                # CRITICAL (2026-07-09): reset the stall clock on every (re)open.
                # The silent-disconnect detector fires when now-last_tick > stall.
                # On a RECONNECT, _last_fyers_ws_tick still held the pre-disconnect
                # value (already >stall old), so the first monitor iteration
                # instantly re-declared "silent disconnect" (~0.8s after subscribe)
                # and tore the fresh socket down again — a self-perpetuating
                # reconnect storm (43 drops / 64 reconnects on 2026-07-09, each
                # spawning WinError 10054 duplicate-connection fights + re-auth
                # bursts). Zeroing it hands the reconnected socket a full `stall`
                # grace window to deliver its first tick (no_tick_after_open branch,
                # measured from the fresh connected_since).
                self._last_fyers_ws_tick = 0.0
                symbols = [*FYERS_INDEX_SYMBOLS.values(), *FYERS_AUX_SYMBOLS.values()]
                self._fyers_ws.subscribe(symbols=symbols)
                logger.info(f"Subscribed to Fyers symbols: {symbols}")

            def on_error(e):
                if getattr(self, "_fyers_ws_generation", 0) != ws_generation:
                    return
                self._fyers_ws_connected = False
                self._fyers_ws_connected_since = 0.0
                self._fyers_ws_last_error = str(e)
                err_text = str(e).lower()
                if (
                    "valid token" in err_text
                    or "invalid token" in err_text
                    or "'code': -300" in err_text
                    or '"code": -300' in err_text
                ):
                    self._fyers_ws_auth_failed = True
                    self._mark_fyers_auth_required("Fyers WebSocket token invalid; refresh Fyers token before reconnecting")
                    logger.error(f"Fyers WS auth failed; reconnect loop stopped: {e}")
                    self._close_ws_socket(self._fyers_ws, "auth failure")
                    return
                if self._fyers_ws_gateway_timeout(e):
                    self._fyers_ws_gateway_failures = getattr(self, "_fyers_ws_gateway_failures", 0) + 1
                    logger.warning(
                        "Fyers WS transient gateway timeout "
                        f"(count={self._fyers_ws_gateway_failures}); reconnect will back off: {e}"
                    )
                else:
                    logger.error(f"Fyers WS Error: {e}")

            def on_close(message=None):
                # Fyers SDK invokes on_close with a close-message argument. The
                # old zero-arg signature raised "takes 0 positional arguments but
                # 1 was given", which crashed BEFORE the cleanup below ran — so
                # _fyers_ws_connected was never reset and the reconnect/state
                # machine desynced (app reported the broker offline). Accept the
                # arg so the cleanup always runs.
                if getattr(self, "_fyers_ws_generation", 0) != ws_generation:
                    return
                self._fyers_ws_connected = False
                self._fyers_ws_connected_since = 0.0
                logger.warning(f"🔌 Fyers WebSocket Closed{f': {message}' if message else ''}")

            def _build_fyers_socket():
                socket = data_ws.FyersDataSocket(
                    access_token=access_token,
                    log_path=_fyers_sdk_log_path(),
                    reconnect=False,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close
                )
                socket.on_open = on_open
                return socket

            self._fyers_ws_auth_failed = False
            self._fyers_ws = _build_fyers_socket()

            def _ws_runner():
                self._fyers_ws_retry_count = 0
                first_connect = True
                use_prebuilt_socket = True
                try:
                    from config.settings import get_settings
                    stall_seconds = float(
                        getattr(get_settings(), "fyers_ws_silent_disconnect_seconds", 90.0) or 90.0
                    )
                except Exception:
                    stall_seconds = 90.0
                while (
                    getattr(self, "_fyers_ws_generation", 0) == ws_generation
                    and not getattr(self, "_fyers_ws_auth_failed", False)
                ):
                    try:
                        if not first_connect and not self._fyers_ws_token_ready_for_reconnect(
                            access_token,
                            ws_generation,
                        ):
                            break
                        first_connect = False
                        
                        if use_prebuilt_socket and getattr(self, "_fyers_ws", None) is not None:
                            use_prebuilt_socket = False
                        else:
                            if hasattr(self, "_fyers_ws") and self._fyers_ws:
                                self._close_ws_socket(self._fyers_ws, "reconnect teardown")
                                time.sleep(2.0)  # Let Fyers server register the disconnection before reconnecting
                            self._fyers_ws = _build_fyers_socket()

                        if getattr(self, "_fyers_ws", None) is None:
                            raise RuntimeError("Fyers socket was not constructed")

                        self._fyers_ws.connect()
                    except Exception as e:
                        if getattr(self, "_fyers_ws_generation", 0) != ws_generation:
                            break
                        logger.error(f"Fyers WS Runner error: {e}")

                    if (
                        getattr(self, "_fyers_ws_generation", 0) != ws_generation
                        or getattr(self, "_fyers_ws_auth_failed", False)
                    ):
                        break
                    
                    while (
                        getattr(self, "_fyers_ws_generation", 0) == ws_generation
                        and getattr(self, "_fyers_ws_connected", False)
                        and not getattr(self, "_fyers_ws_auth_failed", False)
                    ):
                        last_tick = float(getattr(self, "_last_fyers_ws_tick", 0.0) or 0.0)
                        connected_since = float(getattr(self, "_fyers_ws_connected_since", 0.0) or 0.0)
                        now_ts = time.time()
                        if self._fyers_ws_tick_stalled(
                            last_tick, connected_since, now_ts, stall_seconds
                        ):
                            if not self.is_market_open():
                                time.sleep(10)
                                continue
                            logger.warning(
                                f"Fyers WS Silent Disconnect Detected (No ticks > {int(stall_seconds)}s)."
                            )
                            self._close_ws_socket(self._fyers_ws, "silent disconnect")
                            break
                        if connected_since and now_ts - connected_since >= 30:
                            if getattr(self, "_fyers_ws_retry_count", 0):
                                logger.info("Fyers WS stable for 30s; resetting reconnect backoff.")
                            self._fyers_ws_retry_count = 0
                            self._fyers_ws_gateway_failures = 0
                            self._fyers_ws_last_error = ""
                        time.sleep(1)

                    self._fyers_ws_retry_count = min(6, getattr(self, "_fyers_ws_retry_count", 0) + 1)
                    backoff = self._fyers_ws_backoff_seconds(self._fyers_ws_retry_count)
                    if getattr(self, "_fyers_ws_auth_failed", False):
                        break
                    logger.warning(f"🔄 Fyers WS Reconnecting in {backoff}s...")
                    time.sleep(backoff)

                if getattr(self, "_fyers_ws_auth_failed", False):
                    logger.warning("Fyers WS reconnect paused until token refresh.")

            import threading
            t = threading.Thread(target=_ws_runner, daemon=True)
            t.start()
            logger.info("🚀 Fyers WebSocket Thread Started")
            
        except Exception as e:
            logger.error(f"Failed to start Fyers WebSocket: {e}")

    def _mark_fyers_auth_required(self, reason: str):
        self._fyers_auth_required = True
        self._fyers_auth_reason = reason

    def _mark_fyers_unavailable(self, reason: str):
        """Record provider/transport failure without prescribing a new login."""
        self._fyers_auth_required = False
        self._fyers_auth_reason = str(reason or "Fyers service unavailable")

    def _clear_fyers_auth_required(self):
        self._fyers_auth_required = False
        self._fyers_auth_reason = ""

    @staticmethod
    def _fyers_missing_credentials(settings, require_pin: bool = False) -> List[str]:
        missing = []
        if not str(getattr(settings, "fyers_app_id", "") or "").strip():
            missing.append("FYERS_APP_ID")
        if not str(getattr(settings, "fyers_secret_key", "") or "").strip():
            missing.append("FYERS_SECRET_KEY")
        if require_pin and not str(getattr(settings, "fyers_password", "") or "").strip():
            missing.append("FYERS_PASSWORD")
        return missing

    def _refresh_fyers_token(self, token_data: Dict) -> bool:
        """Automatically refresh Fyers access token using refresh_token"""
        if self._fyers_refresh_disabled:
            return False
        try:
            from config.settings import get_settings
            settings = get_settings()
            refresh_token = token_data.get("refresh_token")
            
            if not refresh_token:
                logger.error("❌ Cannot refresh Fyers token: refresh_token missing")
                self._mark_fyers_auth_required("Fyers refresh token missing")
                return False

            missing = self._fyers_missing_credentials(settings, require_pin=True)
            if missing:
                message = f"Missing Fyers credentials in .env: {', '.join(missing)}"
                logger.error(f"❌ Cannot refresh Fyers token: {message}")
                self._mark_fyers_auth_required(message)
                return False

            # Fyers v3 validate-refresh-token expects ONLY appIdHash to be hashed:
            #   appIdHash = SHA256("<app_id>:<secret_key>")
            #   pin       = the 4-digit trading PIN in PLAINTEXT
            # Hashing the PIN too made every refresh fail (the broker rejected the
            # request and the failure was being reported as a policy block).
            import requests
            import hashlib
            app_id = settings.fyers_app_id
            secret = settings.fyers_secret_key
            app_id_hash = hashlib.sha256((app_id + ":" + secret).encode()).hexdigest()

            pin = str(settings.fyers_password or "").strip()

            payload = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "appIdHash": app_id_hash,
                "pin": pin,
            }
            
            url = "https://api-t1.fyers.in/api/v3/validate-refresh-token"
            response_raw = requests.post(url, json=payload, timeout=20)
            response = response_raw.json()
            if response and response.get('s') == 'ok' and response.get('access_token'):
                new_access_token = response.get('access_token')
                token_data['access_token'] = new_access_token
                
                # Update file
                with open("fyers_token.json", "w") as f:
                    json.dump(token_data, f)
                
                # Update current instance
                self.fyers = fyersModel.FyersModel(
                    client_id=settings.fyers_app_id,
                    token=new_access_token,
                    log_path=_fyers_sdk_log_path(),
                )
                self._configure_fyers_http_timeout(
                    self.fyers,
                    getattr(settings, "fyers_history_timeout_seconds", 8.0),
                )
                self._fyers_token_updated_at = datetime.now().isoformat()
                self._fyers_refresh_disabled = False
                self._clear_fyers_auth_required()
                logger.success("✅ Fyers Token Refreshed Successfully!")
                self.start_fyers_websocket(new_access_token)
                return True
            else:
                if response and response.get("code") == -16:
                    # Latch so we stop hammering the endpoint; a successful manual login
                    # clears it. Report the broker's ACTUAL response — this code was
                    # previously always described as a policy block, which masked a
                    # request-side defect (the PIN was being sent hashed).
                    self._fyers_refresh_disabled = True
                    detail = str(response.get("message") or response)
                    self._mark_fyers_auth_required(
                        f"Fyers refresh rejected (code -16): {detail}. Manual login required."
                    )
                    logger.warning(
                        f"Fyers token refresh rejected (code -16): {detail}; "
                        "using available cache/fallback data until manual login."
                    )
                else:
                    self._mark_fyers_auth_required(f"Fyers token refresh failed: {response}")
                    logger.error(f"❌ Fyers Token Refresh Failed: {response}")
                return False
        except Exception as e:
            self._mark_fyers_auth_required(f"Error during Fyers token refresh: {e}")
            logger.error(f"❌ Error during Fyers token refresh: {e}")
            return False

    def get_fyers_auth_status(self) -> Dict[str, Any]:
        if not self._fyers_enabled():
            return {"enabled": False, "connected": False, "auth_required": False,
                    "reason": "Retained for rollback; Kite/Angel route selected", "recovery": "none"}
        token_path = Path("fyers_token.json")
        token_mtime = None
        token_data: Dict[str, Any] = {}
        if token_path.exists():
            token_mtime = datetime.fromtimestamp(token_path.stat().st_mtime).isoformat()
            token_data = self._read_fyers_token_file()
        auth_required = bool(self._fyers_auth_required)
        connected = self.fyers is not None and not auth_required
        reason = str(self._fyers_auth_reason or "")
        if not connected and not reason:
            reason = "Fyers authenticated session unavailable"
        return {
            "enabled": True,
            "role": "backup" if self._fyers_backup_enabled() else "primary",
            "required_for_preferred_data": not self._fyers_backup_enabled(),
            "connected": connected,
            "auth_required": auth_required,
            "reason": reason,
            "refresh_disabled": self._fyers_refresh_disabled,
            "token_updated_at": self._fyers_token_updated_at or token_mtime,
            # Presence-only diagnostics: never return either token itself.
            "token_cache_present": token_path.exists(),
            "access_token_present": bool(token_data.get("access_token")),
            "refresh_token_present": bool(token_data.get("refresh_token")),
            "recovery": (
                "manual_login"
                if auth_required
                else ("retry_validation" if not connected else "none")
            ),
            "login_path": "/api/fyers_login" if auth_required else "",
        }

    def get_fyers_auth_url(self) -> str:
        from config.settings import get_settings
        settings = get_settings()
        missing = self._fyers_missing_credentials(settings)
        if missing:
            message = f"Missing Fyers credentials in .env: {', '.join(missing)}"
            self._mark_fyers_auth_required(message)
            raise ValueError(message)
        session = fyersModel.SessionModel(
            client_id=settings.fyers_app_id,
            secret_key=settings.fyers_secret_key,
            redirect_uri=settings.fyers_redirect_uri,
            response_type="code",
            grant_type="authorization_code",
        )
        return session.generate_authcode()

    def update_fyers_token_from_auth_code(self, auth_code: str) -> Dict[str, Any]:
        """Complete manual Fyers login after the user pastes auth_code or the redirect URL."""
        from config.settings import get_settings
        settings = get_settings()
        missing = self._fyers_missing_credentials(settings)
        if missing:
            message = f"Missing Fyers credentials in .env: {', '.join(missing)}"
            self._mark_fyers_auth_required(message)
            return {"status": "error", "message": message, "missing": missing}
        code = (auth_code or "").strip()
        
        # Robust parsing for both full URL and raw code
        if "auth_code=" in code or "code=" in code:
            from urllib.parse import parse_qs, urlparse
            parsed = urlparse(code)
            query = parse_qs(parsed.query)
            if not query and parsed.fragment:
                query = parse_qs(parsed.fragment)
            code = query.get("auth_code", query.get("code", [code]))[0]
            
        if not code:
            return {"status": "error", "message": "Missing Fyers auth_code"}

        session = fyersModel.SessionModel(
            client_id=settings.fyers_app_id,
            secret_key=settings.fyers_secret_key,
            redirect_uri=settings.fyers_redirect_uri,
            response_type="code",
            grant_type="authorization_code",
        )
        session.set_token(code)
        response = session.generate_token()
        if not response or response.get("s") != "ok" or not response.get("access_token"):
            self._mark_fyers_auth_required(f"Manual Fyers token generation failed: {response}")
            return {"status": "error", "message": "Fyers token generation failed", "response": response}

        token_data = {
            "access_token": response.get("access_token"),
            "refresh_token": response.get("refresh_token"),
            "timestamp": datetime.now().isoformat(),
        }
        with open("fyers_token.json", "w") as f:
            json.dump(token_data, f, indent=2)

        self.fyers = fyersModel.FyersModel(
            client_id=settings.fyers_app_id,
            token=token_data["access_token"],
            log_path=_fyers_sdk_log_path(),
        )
        self._configure_fyers_http_timeout(
            self.fyers,
            getattr(settings, "fyers_history_timeout_seconds", 8.0),
        )
        profile = self.fyers.get_profile()
        if profile and profile.get("s") == "error":
            self.fyers = None
            self._mark_fyers_auth_required(f"New Fyers token validation failed: {profile}")
            return {"status": "error", "message": "New Fyers token did not validate", "response": profile}

        self._fyers_token_updated_at = token_data["timestamp"]
        self._fyers_refresh_disabled = False
        self._fyers_endpoint_retry_at = {}
        self._fyers_history_retry_at = 0.0
        self._clear_fyers_auth_required()
        logger.success("✅ Fyers manual authorization completed successfully.")
        if getattr(self, "start_streams", True):
            self.start_fyers_websocket(token_data["access_token"])
        return {"status": "ok", "message": "Fyers token updated", "profile": profile}

    def is_market_holiday(self, dt: datetime = None) -> bool:
        """Check if current or given date is an NSE market holiday"""
        dt = dt or datetime.now()
        date_str = dt.strftime("%Y-%m-%d")
        coverage_years = getattr(self, "nse_calendar_coverage_years", None)
        if coverage_years is None:
            coverage_years = {
                int(str(day)[:4])
                for day in getattr(self, "nse_holidays", set())
                if str(day)[:4].isdigit()
            }
        if coverage_years and dt.year not in coverage_years:
            # Fail CLOSED: an uncovered year is treated as non-trading until its
            # calendar is provided (deliberate safety — do not trade a year whose
            # holidays we cannot verify; see test_unknown_future_calendar_year_*).
            # Log CRITICAL only ONCE per uncovered year rather than on every call, so
            # a look-ahead probe (e.g. a 2026 session touching 2027) does not spam the
            # log ×N — today that produced 16 identical CRITICALs.
            warned = getattr(self, "_holiday_coverage_warned_years", None)
            if warned is None:
                warned = set()
                self._holiday_coverage_warned_years = warned
            if dt.year not in warned:
                warned.add(dt.year)
                logger.critical(
                    f"NSE holiday calendar has no explicit coverage for {dt.year}; "
                    "treating that year's sessions as CLOSED until data_store/market_holidays.json "
                    "(or the NSE holiday list) is updated."
                )
            return True
        return date_str in self.nse_holidays

    def is_special_trading_session(self, dt: datetime = None) -> bool:
        dt = dt or datetime.now()
        return dt.strftime("%Y-%m-%d") in self.nse_special_trading_days

    @staticmethod
    def get_index_session_bounds(dt: datetime) -> tuple[datetime, datetime]:
        return index_session_bounds(dt)

    def is_market_open(self, dt: datetime = None) -> bool:
        """Robust check for NSE Market Status (09:15 - 15:30, Mon-Fri, No Holidays)"""
        dt = dt or datetime.now()
        if dt.tzinfo is not None:
            dt = dt.astimezone(ZoneInfo("Asia/Kolkata"))
        special_session = self.is_special_trading_session(dt)
        # 1. Weekend check
        if dt.weekday() >= 5 and not special_session: return False
        # 2. Holiday check
        if self.is_market_holiday(dt) and not special_session: return False
        opening, closing = self.get_index_session_bounds(dt)
        return opening <= dt <= closing

    def get_market_session_label(self, dt: datetime = None) -> str:
        """Return the display session for index quotes."""
        dt = dt or datetime.now()
        special_session = self.is_special_trading_session(dt)
        if (dt.weekday() >= 5 or self.is_market_holiday(dt)) and not special_session:
            return "CLOSED"
        curr_time = dt.time()
        preopen_start = self._clock_setting_time("real_mode_premarket_arm_start", "09:00:00")
        live_start = self._clock_setting_time("real_mode_live_start", "09:15:00")
        live_end = self._clock_setting_time("real_mode_live_end", "15:30:00")
        if special_session:
            opening, closing = self.get_index_session_bounds(dt)
            if opening.time() != dtime(9, 15):
                return "LIVE" if opening <= dt <= closing else "CLOSED"
        if preopen_start <= curr_time < live_start:
            return "PRE-OPEN"
        if live_start <= curr_time <= live_end:
            return "LIVE"
        return "CLOSED"

    @staticmethod
    def _clock_setting_time(setting_name: str, default: str) -> dtime:
        try:
            from config.settings import get_settings

            label = str(getattr(get_settings(), setting_name, default) or default).strip()
        except Exception:
            label = default
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(label, fmt).time()
            except ValueError:
                continue
        return datetime.strptime(default, "%H:%M:%S" if len(default.split(":")) == 3 else "%H:%M").time()

    def _preopen_rest_quote_allowed(self, dt: datetime = None, session_label: str = "") -> bool:
        """Use pre-open REST only during the configured settlement window."""
        if str(session_label or "").upper() != "PRE-OPEN":
            return False
        dt = dt or datetime.now()
        settle_end = self._clock_setting_time("real_mode_premarket_settle_end", "09:08:00")
        return dt.time() <= settle_end

    def _quote_label_for_session(self, session: str) -> str:
        session = str(session or "").upper()
        if session in {"LIVE", "PRE-OPEN"}:
            return session
        return "REF"

    @property
    def is_connected(self) -> bool:
        if getattr(self, "_is_connected_override", None) is not None:
            return bool(self._is_connected_override)
        if getattr(self, "upstox_only", False):
            from data.upstox_auth import get_upstox_token
            return bool(get_upstox_token())
        upstox = getattr(self, "upstox", None)
        if upstox is not None and upstox.is_healthy():
            return True
        try:
            from data.upstox_auth import get_upstox_auth
            return bool(get_upstox_auth().has_valid_token())
        except Exception:
            return False

    @is_connected.setter
    def is_connected(self, value):
        self._is_connected_override = bool(value) if value is not None else None

    @retry_on_exception(retries=3, delay=1.0)
    def connect(self) -> bool:
        """Bind market data to the current Upstox session."""
        if getattr(self, "upstox_only", False):
            self.start_upstox_stream()
            return self.is_connected
        try:
            from data.upstox_auth import get_upstox_auth
            auth = get_upstox_auth()
            if auth.has_valid_token():
                self.is_connected = True
                status = auth.status()
                logger.success(
                    f"Upstox market data connected ({status.get('user_name') or status.get('user_id') or 'user'})"
                )
                if hasattr(self, "upstox") and self.upstox:
                    self.upstox.reset_health()
            else:
                self.is_connected = False
                logger.warning(
                    "Upstox market data unavailable: daily login required. "
                    "Complete login at /api/upstox_login"
                )
            self._prewarm_angelone_fallback()
            return bool(self.is_connected)
        except Exception as e:
            logger.error(f"Upstox connection failed: {e}")
            self.is_connected = False
            self._prewarm_angelone_fallback()
            return False

    def _prewarm_angelone_fallback(self) -> bool:
        """Authenticate AngelOne off-loop once; Kite still owns all execution."""
        if getattr(self, "upstox_only", False) or getattr(self, "_replacement_closed", False):
            return False
        source = getattr(self, "angel_index", None)
        if source is None or not bool(getattr(source, "configured", False)):
            return False
        lock = getattr(self, "_angel_prewarm_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._angel_prewarm_lock = lock
        with lock:
            if bool(getattr(self, "_angel_prewarm_started", False)):
                return False
            self._angel_prewarm_started = True

        def _connect() -> None:
            try:
                prewarm = getattr(source, "prewarm", None)
                if callable(prewarm):
                    prewarm()
                else:
                    source.connect()
                self._start_replacement_stream()
            except Exception:
                logger.warning("AngelOne fallback prewarm failed.")
            finally:
                if self._replacement_enabled():
                    self._angel_prewarm_started = False

        threading.Thread(
            target=_connect, daemon=True, name="AngelOneFallbackPrewarm"
        ).start()
        return True

    def _record_index_intrabar_cash_volume(
        self,
        index_symbol: str,
        frame: pd.DataFrame,
        *,
        received_at: Optional[datetime] = None,
    ) -> None:
        """Cache only a tagged Fyers forming-minute volume observation."""
        if frame is None or frame.empty or "volume" not in frame.columns:
            return
        try:
            stamp = pd.Timestamp(frame.index[-1])
            if stamp.tzinfo is not None:
                stamp = stamp.tz_convert("Asia/Kolkata").tz_localize(None)
            now_dt = received_at or datetime.now()
            current_minute = pd.Timestamp(now_dt).replace(second=0, microsecond=0)
            volume = float(frame["volume"].iloc[-1] or 0.0)
            if stamp != current_minute or volume <= 0:
                return
            payload = {
                "available": True,
                "volume": volume,
                "bar_at": stamp.to_pydatetime(),
                "received_at": now_dt,
                "source": "fyers_history_intrabar",
                "basis": "fyers_index",
                "unit": "shares",
            }
        except (TypeError, ValueError, OverflowError):
            return
        lock = getattr(self, "_index_intrabar_cash_volume_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._index_intrabar_cash_volume_lock = lock
        with lock:
            if not hasattr(self, "_index_intrabar_cash_volume"):
                self._index_intrabar_cash_volume = {}
            self._index_intrabar_cash_volume[str(index_symbol or "").upper()] = payload

    def get_index_intrabar_cash_volume(
        self,
        index_symbol: str,
        *,
        now_dt: Optional[datetime] = None,
        max_age_seconds: float = 75.0,
    ) -> Dict[str, Any]:
        """Return the current-minute observation only; old partials never roll over."""
        now_dt = now_dt or datetime.now()
        lock = getattr(self, "_index_intrabar_cash_volume_lock", None)
        if lock is None:
            return {"available": False}
        with lock:
            payload = dict(
                (getattr(self, "_index_intrabar_cash_volume", {}) or {}).get(
                    str(index_symbol or "").upper()
                ) or {}
            )
        bar_at = payload.get("bar_at")
        received_at = payload.get("received_at")
        if not isinstance(bar_at, datetime) or not isinstance(received_at, datetime):
            return {"available": False}
        current_minute = now_dt.replace(second=0, microsecond=0)
        age = (now_dt - received_at).total_seconds()
        if bar_at != current_minute or age < -1.0 or age > max_age_seconds:
            return {"available": False}
        payload["age_seconds"] = round(age, 3)
        payload["progress"] = max(
            0.2,
            min(1.0, (now_dt.second + now_dt.microsecond / 1e6) / 60.0),
        )
        return payload

    # Index spot lives on the cash segment under a different name than the derivative
    # root: Kite quotes NIFTY 50, not NIFTY. Options are already Kite tradingsymbols
    # because expiry_manager now builds them from the Kite contract master.
    KITE_INDEX_SPOT = {
        "NIFTY": ("NSE", "NIFTY 50"),
        "BANKNIFTY": ("NSE", "NIFTY BANK"),
        "FINNIFTY": ("NSE", "NIFTY FIN SERVICE"),
        "MIDCPNIFTY": ("NSE", "NIFTY MID SELECT"),
        "SENSEX": ("BSE", "SENSEX"),
        "BANKEX": ("BSE", "BANKEX"),
    }

    # Two feeds quoting the SAME index tick should agree almost exactly — they are
    # reporting one number from one exchange, not an opinion. 0.05% of NIFTY at 24,400
    # is about 12 points: wide enough to absorb sub-second arrival skew between two
    # sockets, tight enough that a real fault cannot hide inside it.
    #
    # This started at 0.75%, which is far too loose for an index system: a 0.5% error
    # is ~120 NIFTY points and would sail through silently while every stop, target and
    # position size derived from it was wrong.
    # Two feeds disagreeing about one index by this much (percent of LTP) is a fault in
    # one of them, not market microstructure. Both quote the SAME computed index, so any
    # real gap is staleness or a bad tick — there is no spread to explain it away.
    #
    # Tightened from 0.05% on 2026-08-15. At 0.05% a NIFTY at 24,000 could carry a 12
    # point disagreement before anything was said, which is a large move on an index that
    # gates entries; 0.015% puts that at ~3.6 points. The default is the fallback here —
    # quote_divergence_alert_pct is the live knob.
    QUOTE_DIVERGENCE_ALERT_PCT = 0.015
    QUOTE_DIVERGENCE_CRITICAL_PCT = 0.05
    # Price-source routing is intentionally stricter than quote DISPLAY fallback.
    # A Kite tick is the price the execution venue will fill against, so it owns the
    # candle while it is fresh.  Fyers must see two ticks over a small dwell after
    # Kite goes stale before taking over; this prevents a one-packet scheduling gap
    # from flipping the active feed twice inside one candle.
    INDEX_KITE_PRICE_FRESH_SECONDS = 7.0
    INDEX_FYERS_PRICE_FRESH_SECONDS = 15.0
    INDEX_PRICE_FAILOVER_MIN_TICKS = 2
    INDEX_PRICE_FAILOVER_HYSTERESIS_SECONDS = 0.25

    def _quote_divergence_alert_pct(self) -> float:
        """The configured divergence tolerance, with the class default as fallback."""
        try:
            from config.settings import get_settings

            value = float(
                getattr(get_settings(), "quote_divergence_alert_pct", 0.0) or 0.0
            )
        except Exception:
            return float(MarketDataProvider.QUOTE_DIVERGENCE_ALERT_PCT)
        if value <= 0:
            return float(MarketDataProvider.QUOTE_DIVERGENCE_ALERT_PCT)
        return value

    @staticmethod
    def _naive_kolkata_index(df):
        """Force a candle frame's index to NAIVE Asia/Kolkata, the house convention.

        Every live producer already ends in `.tz_localize(None)` (see the Fyers and
        Kite history paths), so candles in memory are naive. Candles read back from a
        CSV cache are not necessarily: `parse_dates=True` on a file whose timestamps
        carry a `+05:30` suffix yields `datetime64[us, tzoffset(None, 19800)]`, and a
        different writer produces `datetime64[us, UTC+05:30]`. Both are "+05:30" to a
        human and neither is comparable with a naive Timestamp.

        That is the whole of the 2026-08-19 freeze. `scanner:process_task` slices these
        frames with `df.index <= as_of`, raises "Invalid comparison between
        dtype=datetime64[us, tzoffset(None, 19800)] and Timestamp", and its handler
        returns `{"error": ...}` for the index — which the dashboard renders as unknown
        data. It only ever bit within ~16-500s of a restart, because that is when the
        cache is read and merged; all 18 occurrences in the log sit inside a restart
        window and none outside one.

        Normalising on the way in means the mixed dtypes never meet.
        """
        try:
            idx = getattr(df, "index", None)
            if idx is None or getattr(idx, "tz", None) is None:
                return df
            df.index = idx.tz_convert("Asia/Kolkata").tz_localize(None)
        except (AttributeError, TypeError, ValueError) as exc:
            # A frame we cannot normalise is left exactly as it was: this is a
            # defensive tidy-up on a read path, never a reason to lose the candles.
            # Said out loud rather than swallowed — an unnormalised frame is the
            # precondition for the mixed-dtype comparison this method exists to stop,
            # so silence here would hide the next occurrence of the original bug.
            logger.debug(f"Candle index tz normalisation skipped: {exc}")
        return df

    def _quote_divergence_critical_pct(self) -> float:
        """The bar for a CRITICAL line and a desktop notification.

        Separate from the tracking tolerance above: tracking stays sensitive so the
        dashboard sees drift, while only a real disagreement is worth an alarm. Never
        returns below the tracking bar — a critical threshold under the threshold that
        gates it would be silently meaningless.
        """
        floor = self._quote_divergence_alert_pct()
        try:
            from config.settings import get_settings

            value = float(
                getattr(get_settings(), "quote_divergence_critical_pct", 0.0) or 0.0
            )
        except Exception:
            return max(floor, float(MarketDataProvider.QUOTE_DIVERGENCE_CRITICAL_PCT))
        if value <= 0:
            return max(floor, float(MarketDataProvider.QUOTE_DIVERGENCE_CRITICAL_PCT))
        return max(floor, value)

    def _flag_quote_divergence(
        self, symbol: str, primary: str, primary_ltp: float,
        secondary: str, secondary_ltp: float,
    ) -> float:
        """Report two feeds disagreeing about one instrument. Returns the deviation %."""
        if primary_ltp <= 0 or secondary_ltp <= 0:
            return 0.0
        # Lazily materialise state: this runs inside price resolution, and a monitor
        # that can raise is a monitor that takes down the thing it was watching.
        if not isinstance(getattr(self, "_quote_divergence", None), dict):
            self._quote_divergence = {}
        # The alert cooldown must OUTLIVE the divergence record. It used to be stored
        # inside that record, which is popped the moment the feeds agree again — so a
        # quote flickering across the threshold cleared its own cooldown and re-alerted
        # on the next crossing. 2026-08-18: 1,815 CRITICAL lines, 98% of them inside the
        # 300s window they were supposed to be suppressed by, at a median gap of 13-21s.
        if not isinstance(getattr(self, "_quote_divergence_alerted_at", None), dict):
            self._quote_divergence_alerted_at = {}

        deviation = abs(primary_ltp - secondary_ltp) / primary_ltp * 100.0
        if deviation < self._quote_divergence_alert_pct():
            self._quote_divergence.pop(symbol, None)
            return deviation

        now = time.time()
        last = float(self._quote_divergence_alerted_at.get(symbol, 0.0) or 0.0)
        self._quote_divergence[symbol] = {
            "symbol": symbol, "deviation_pct": round(deviation, 3),
            "primary": primary, "primary_ltp": primary_ltp,
            "secondary": secondary, "secondary_ltp": secondary_ltp,
            "at": now, "alerted_at": last or now,
        }
        # Tracking (above) is sensitive so the dashboard sees every disagreement.
        # Alerting (below) is not: only a divergence past the critical bar is worth a
        # CRITICAL line and a desktop notification.
        if deviation < self._quote_divergence_critical_pct():
            return deviation

        if now - last >= 300.0:
            self._quote_divergence_alerted_at[symbol] = now
            self._quote_divergence[symbol]["alerted_at"] = now
            message = (
                f"BAD MARKET DATA: {symbol} {secondary}={secondary_ltp:,.2f} disagrees with "
                f"{primary}={primary_ltp:,.2f} by {deviation:.2f}%. Using {primary} (the "
                f"execution broker). Treat {secondary}-sourced numbers as suspect."
            )
            logger.critical(message)
            try:
                from notifications import send_desktop_notification

                send_desktop_notification(message, "error")
            except Exception as exc:
                logger.debug(f"Divergence alert notification unavailable: {exc}")
        return deviation

    def get_quote_divergences(self) -> Dict[str, Any]:
        """Live feed-disagreement report, for the dashboard and the watchdog."""
        return {k: dict(v) for k, v in (self._quote_divergence or {}).items()}

    def _kite_quote_target(self, symbol: str, token: str, exchange: str):
        """Resolve (tradingsymbol, exchange) for a Kite quote, or ("", "")."""
        spot = self.KITE_INDEX_SPOT.get(str(symbol or "").upper())
        if spot:
            return spot[1], spot[0]
        name = str(symbol or "").upper()
        if not name:
            return "", ""
        segment = "BFO" if ("SENSEX" in name or "BANKEX" in name) else "NFO"
        return symbol, segment

    # ── Kite tick stream ─────────────────────────────────────────────────────
    def start_kite_ticker(self) -> bool:
        """Stream ticks over KiteTicker, replacing the retired AngelOne socket.

        Subscribed in MODE_FULL, which carries open interest and 5-level depth in the
        tick itself — AngelOne needed a separate SNAP_QUOTE mode for OI and never
        supplied a book at all. Writes the same token-keyed `_ws_cache` records the
        rest of the system already reads, so nothing downstream changes.
        """
        if not self.start_streams:
            logger.info("Kite ticker skipped for history-only provider.")
            return False
        auth = getattr(self.kite, "_auth", None)
        api_key = str(getattr(auth, "api_key", "") or "")
        access_token = str(getattr(getattr(auth, "kite", None), "access_token", "") or "")
        if not api_key or not access_token:
            logger.error("Kite ticker not started: no authenticated session.")
            return False

        try:
            from kiteconnect import KiteTicker
        except ImportError as exc:
            logger.error(f"KiteTicker unavailable: {exc}")
            return False

        # KiteTicker exposes close()/stop(), not close_connection() — the shared helper
        # tries each in turn so a stale socket is always torn down before reconnecting.
        self._close_ws_socket(getattr(self, "kws", None), "kite ticker")
        ticker = KiteTicker(api_key, access_token)

        def on_ticks(ws, ticks):
            now_ts = time.time()
            for tick in ticks or []:
                try:
                    token = str(tick.get("instrument_token") or "")
                    price = float(tick.get("last_price") or 0.0)
                    if not token or price <= 0:
                        continue
                    volume = int(tick.get("volume_traded") or tick.get("volume") or 0)
                    depth = tick.get("depth") or {}
                    buy = (depth.get("buy") or [{}])[0] if depth.get("buy") else {}
                    sell = (depth.get("sell") or [{}])[0] if depth.get("sell") else {}
                    exchange_at = self._exchange_timestamp_from_payload(tick)
                    cache_entry = {
                        "ltp": price,
                        "oi": int(tick.get("oi") or 0),
                        "oi_change_pct": 0.0,
                        "vol": volume,
                        "volume_key": "volume_traded",
                        "volume_fields": {"volume_traded": volume},
                        "previous_close": float((tick.get("ohlc") or {}).get("close") or 0.0),
                        # Depth arrives free in MODE_FULL. Carrying it here is what lets
                        # touch-quote execution anchor to a real book instead of a blind
                        # LTP mark-up, which is how every 2026-07-31 option order priced.
                        "bid": float(buy.get("price") or 0.0),
                        "ask": float(sell.get("price") or 0.0),
                        "bid_qty": int(buy.get("quantity") or 0),
                        "ask_qty": int(sell.get("quantity") or 0),
                        "time": now_ts,
                        "received_at": now_ts,
                        "exchange_at": exchange_at,
                        "source": "kite_ws",
                    }
                    with self._index_quote_guard():
                        self._ws_cache[token] = cache_entry
                    self._last_kite_ws_tick = now_ts
                    self._last_ws_tick = now_ts
                    self._last_ltp_time = now_ts
                    index_name = self._kite_index_names.get(token)
                    if not index_name:
                        self._record_option_participation_tick(token, tick, now_ts)
                        self._record_contract_tick(
                            token,
                            price,
                            now_ts,
                            exchange_at=exchange_at,
                        )
                    if index_name and callable(getattr(self, "on_tick", None)):
                        # Kite reports NO volume for indices — they are computed values,
                        # not traded instruments, so every field comes back None. The
                        # candle builder treats `volume` as a CUMULATIVE day total and
                        # stores volume - start_volume, so feeding it 0 alongside Fyers'
                        # real cumulative figure either zeroes the bar or fabricates a
                        # multi-million spike, depending on which feed ticks first.
                        # Carry the last known cumulative total so the delta stays
                        # coherent and Fyers remains the sole source of index volume.
                        carried, carried_source = self._index_volume_for_tick(index_name, now_ts)
                        volume = int(volume or carried)
                        self._record_index_tick_volume_diag(
                            index_name, "kite_ws", float(volume), "volume_traded",
                            {"volume_traded": volume}, now_ts,
                            volume_source=carried_source,
                            volume_basis=(
                                "cumulative_cash_index_volume"
                                if carried_source == "fyers_index"
                                else ""
                            ),
                            volume_unit="shares" if carried_source == "fyers_index" else "",
                            volume_comparable=bool(
                                volume and carried_source == "fyers_index"
                            ),
                        )
                        self._emit_arbitrated_index_tick(
                            index_name,
                            price,
                            volume,
                            carried_source,
                            "kite_ws",
                            received_at=now_ts,
                            exchange_at=exchange_at,
                        )
                except Exception as exc:
                    logger.debug(f"Kite tick parse failed: {exc}")

        def on_connect(ws, response):
            self._kite_ws_connected = True
            # Deliberately does NOT stamp _last_kite_ws_tick. Connecting is not a tick.
            # That field is the only input to kite_stream_stalled (market_source_health.py),
            # so stamping it here meant a socket reconnecting faster than the 45s freshness
            # bound while delivering NOTHING read as a healthy feed: broker_degraded,
            # all_brokers_unavailable, the "Kite feed silent" critical and the Yahoo arm
            # all stayed quiet on a dead stream.
            #
            # Not zeroed either, which is the tempting mirror of the Fyers on_open at ~744.
            # Fyers pairs its 0.0 with _fyers_ws_connected_since and a stall predicate that
            # measures the no-tick case from connect time; build_source_health has no such
            # pairing — when kite_age is None it falls back to provider UPTIME, which
            # mid-session is hours, so zeroing would report a stalled feed on every healthy
            # reconnect. Leaving the last real tick alone makes every case fall out right:
            # never ticked -> still 0.0 -> uptime branch keeps the cold-start grace; ticked
            # before -> the age keeps growing until a real tick lands.
            logger.info("🔌 Kite WebSocket Connected")
            tokens = sorted({int(t) for t in self._active_kite_tokens if str(t).isdigit()})
            if tokens:
                ws.subscribe(tokens)
                ws.set_mode(ws.MODE_FULL, tokens)
                logger.info(f"Subscribed to {len(tokens)} Kite token(s) in FULL mode")

        def on_close(ws, code, reason):
            self._kite_ws_connected = False
            logger.warning(f"🔌 Kite WebSocket closed ({code}): {reason}")

        def on_error(ws, code, reason):
            self._kite_ws_connected = False
            logger.error(f"Kite WebSocket error ({code}): {reason}")
            if "403" in str(reason):
                # Kite refused the socket upgrade: the session was reset (2026-09-25:
                # a dead token reconnected here every few seconds for hours). Settle
                # it off the socket thread; a proven-dead session stops the retries.
                threading.Thread(
                    target=self._on_kite_ticker_forbidden,
                    args=(ws, str(reason)),
                    name="kite-ticker-403",
                    daemon=True,
                ).start()

        def on_reconnect(ws, attempts):
            logger.warning(f"Kite WebSocket reconnecting (attempt {attempts})")

        def on_noreconnect(ws):
            self._kite_ws_connected = False
            logger.critical(
                "Kite WebSocket gave up reconnecting — live ticks are STOPPED. "
                "Prices fall back to slower REST polling until restart."
            )

        def on_order_update(ws, data):
            """Kite's order-state push, riding the socket already open for ticks.

            Added 2026-08-16. The payload is the same order object an HTTP postback
            carries, minus the checksum — the socket is already authenticated, so there
            is nothing to verify. Crucially it needs NO public URL, no tunnel and no
            inbound firewall rule, which is what makes it usable here: kite_postback_url
            defaults to empty, so before this the system had no push order-state channel
            at all and learned every fill, rejection and stop trigger from a 5s poll.

            Dispatched off-thread on purpose. kiteconnect calls this synchronously on
            the socket thread (see the note above on_ticks), and the sink takes the
            TradeManager lock and writes state — blocking here stalls the tick feed for
            all four indices.
            """
            sink = getattr(self, "order_update_sink", None)
            if not callable(sink):
                logger.debug(f"Kite order update, no sink attached: {str(data)[:200]}")
                return
            try:
                payload = dict(data or {})
            except (TypeError, ValueError):
                logger.warning(f"Kite order update had an unreadable payload: {data!r}")
                return
            threading.Thread(
                target=self._dispatch_order_update,
                args=(sink, payload),
                name="kite-order-update",
                daemon=True,
            ).start()

        ticker.on_ticks = on_ticks
        ticker.on_connect = on_connect
        ticker.on_close = on_close
        ticker.on_error = on_error
        ticker.on_reconnect = on_reconnect
        ticker.on_noreconnect = on_noreconnect
        ticker.on_order_update = on_order_update
        self.kws = ticker
        try:
            ticker.connect(threaded=True)
        except Exception as exc:
            logger.error(f"Kite ticker failed to start: {exc}")
            return False
        logger.info("🚀 Kite WebSocket thread started")
        return True

    def _on_kite_ticker_forbidden(self, ws, reason: str) -> bool:
        """A 403 on the ticker upgrade: confirm the session is dead, then stop retrying.

        A new browser login rebinds Kite and starts a fresh ticker (rebind_kite), so the
        stopped socket is never needed again.
        """
        auth = getattr(getattr(self, "kite", None), "_auth", None)
        if auth is None:
            return False
        dead = False
        probe = getattr(auth, "confirm_dead_session", None)
        if callable(probe):
            try:
                dead = bool(probe(f"Kite WebSocket upgrade refused: {reason}"))
            except Exception as exc:
                logger.debug(f"Kite ticker 403 probe failed: {exc}")
        try:
            dead = dead or bool((auth.status() or {}).get("auth_required"))
        except Exception as exc:
            logger.debug(f"Kite auth status unreadable after ticker 403: {exc}")
        if not dead:
            return False
        try:
            stop_retry = getattr(ws, "stop_retry", None)
            if callable(stop_retry):
                stop_retry()
        except Exception as exc:
            logger.debug(f"Kite ticker stop_retry failed: {exc}")
        self._close_ws_socket(ws, "kite ticker (session refused)")
        self._kite_ws_connected = False
        logger.critical(
            "Kite WebSocket stopped: Kite refused the session (403). Log in again at "
            "/api/kite_login; the ticker restarts on the new session."
        )
        return True

    @staticmethod
    def _dispatch_order_update(sink, payload: Dict) -> None:
        """Hand one order update to the sink, off the socket thread."""
        try:
            sink(payload)
        except Exception as exc:
            # Never let a consumer fault kill the dispatch thread silently — an order
            # update that vanishes is a fill the system does not know it has.
            logger.error(
                f"Kite order-update sink failed for order "
                f"{payload.get('order_id', '?')}: {exc}"
            )

    def _load_kite_index_tokens(self) -> None:
        """Map index instrument_tokens to names so on_tick still fires for indices.

        Resolved from the contract master rather than hardcoded: the AngelOne token
        constants (26000, 26009…) mean nothing to Kite, and a wrong constant would
        silently stop index candles rather than raise.
        """
        self._kite_index_names = {}
        source = getattr(self, "kite", None)
        if source is None or not source.is_connected:
            return
        wanted = {
            "NIFTY 50": "NIFTY", "NIFTY BANK": "BANKNIFTY",
            "NIFTY MID SELECT": "MIDCPNIFTY", "SENSEX": "SENSEX",
        }
        try:
            with source._lock:
                dumps = {ex: list(rows) for ex, rows in source._instruments.items()}
        except Exception:
            return
        for exchange in ("NSE", "BSE"):
            for row in dumps.get(exchange, []):
                name = wanted.get(str(row.get("tradingsymbol") or "").upper())
                if name and row.get("instrument_token"):
                    self._kite_index_names[str(row["instrument_token"])] = name
        if self._kite_index_names:
            logger.info(f"Kite index tick tokens resolved: {sorted(set(self._kite_index_names.values()))}")

    def rebind_kite(self, kite) -> None:
        """Adopt a session issued by a fresh browser login, mid-run.

        Everything holding the dead session has to be rebuilt here, not just the
        broker: the data source, the contract master and the tick socket all
        authenticate with the access token. Rebinding only the broker leaves orders
        working while quotes, ticks and contract lookups stay broken until a full
        restart — and the dashboard looks healthy throughout.
        """
        if getattr(self, "upstox_only", False):
            return
        from data.kite_auth import get_kite_auth

        self.kite = KiteDataSource(auth=get_kite_auth(), kite=kite)
        self.is_connected = kite is not None
        if kite is not None:
            try:
                self.kite.load_instruments(("NFO", "BFO", "NSE", "BSE"), force=True)
                self._load_kite_index_tokens()
                self._active_kite_tokens.update(self._kite_index_names)
            except Exception as exc:
                logger.warning(f"Contract master reload after re-auth failed: {exc}")
            # The socket authenticates with the access token, so a dead session means
            # a dead stream: it must be torn down and reopened with the new one.
            try:
                self.start_kite_ticker()
            except Exception as exc:
                logger.error(f"Kite ticker restart after re-auth failed: {exc}")
        callback = getattr(self, "on_kite_reauth", None)
        if callable(callback):
            try:
                callback(kite)
            except Exception as exc:
                logger.warning(f"Kite re-auth callback failed: {exc}")
        logger.success("Kite session rebound: data, contract master, ticker and broker.")

    def subscribe_tokens(self, tokens: List[str], exchange_type: int = 1):
        """Subscribe additional instruments to the Kite tick stream.

        `exchange_type` is retained only for call-site compatibility — it was an
        AngelOne segment id, and Kite derives the segment from the instrument_token
        itself. Tokens are remembered even when the socket is down so on_connect can
        replay them; before this the whole call was a silent no-op and option
        contracts were never streamed at all.
        """
        if getattr(self, "upstox_only", False):
            for token in tokens:
                self._upstox_stream.watch(str(token))
            return
        new_tokens = {
            str(token).strip() for token in tokens
            if str(token).strip().isdigit() and str(token).strip() not in self._active_kite_tokens
        }
        if not new_tokens:
            return
        self._active_kite_tokens.update(new_tokens)
        ticker = getattr(self, "kws", None)
        if ticker is None or not getattr(self, "_kite_ws_connected", False):
            logger.debug(
                f"Kite socket down; {len(new_tokens)} token(s) queued for the next connect."
            )
            return
        try:
            numeric = sorted(int(t) for t in new_tokens)
            ticker.subscribe(numeric)
            ticker.set_mode(ticker.MODE_FULL, numeric)
            logger.debug(f"Subscribed {len(numeric)} Kite token(s) in FULL mode")
        except Exception as exc:
            logger.warning(f"Kite token subscription failed: {exc}")

    def resubscribe_kite_tokens(self) -> bool:
        """Replay the subscription for every active token on a live socket.

        Recovery for the connected-but-silent stall. on_close/on_error never fire on a
        socket that is open but delivering nothing, so kiteconnect's own reconnect never
        runs and on_connect never gets to replay the tokens — the stream simply stops.
        Observed 2026-08-21: connected 10:23:05, no error until 15:30:19 (after the
        close), kite_ws age 38s -> 1,672s in between.

        subscribe_tokens() cannot do this: it filters to tokens NOT already active, so
        for an existing subscription it is a no-op by design.

        Re-issuing subscribe + set_mode is idempotent and does not disturb a healthy
        stream, which is what makes it safe for a watchdog to call speculatively.
        Returns True only when a request actually went out.
        """
        tokens = sorted(
            int(token) for token in (self._active_kite_tokens or set())
            if str(token).strip().isdigit()
        )
        if not tokens:
            return False
        ticker = getattr(self, "kws", None)
        if ticker is None or not getattr(self, "_kite_ws_connected", False):
            # A dead socket is on_connect's problem; it replays the tokens itself.
            return False
        try:
            ticker.subscribe(tokens)
            ticker.set_mode(ticker.MODE_FULL, tokens)
        except Exception as exc:
            # Never raise: the watchdog thread calls this and must survive.
            logger.warning(f"Kite resubscribe failed for {len(tokens)} token(s): {exc}")
            return False
        logger.info(f"Kite stream stalled; resubscribed {len(tokens)} token(s) in FULL mode.")
        return True

    def get_source_health(self) -> Dict[str, Any]:
        """Check each provider independently; Yahoo is an idle emergency source."""
        health = build_source_health(self)
        health["market_data_replacement"] = self._replacement_health()

        from data.upstox_auth import get_upstox_auth_status
        upstox_auth = get_upstox_auth_status()
        upstox_client = getattr(self, "upstox", None)
        upstox_healthy = bool(upstox_auth.get("authenticated")) and (
            upstox_client.is_healthy() if upstox_client is not None else True
        )
        health["upstox"] = {
            "healthy": upstox_healthy,
            "authenticated": bool(upstox_auth.get("authenticated")),
            "status": upstox_auth.get("status", "DISCONNECTED"),
            "user_id": upstox_auth.get("user_id", ""),
            "failure_seconds": round(upstox_client.failure_duration(), 1) if upstox_client is not None else 0.0,
        }
        health["upstox_healthy"] = upstox_healthy
        health["upstox_session_live"] = upstox_healthy
        health["upstox_auth"] = upstox_auth
        if getattr(self, "upstox_only", False):
            health["all_brokers_unavailable"] = not upstox_healthy
            health["broker_degraded"] = not upstox_healthy
            health["market_data_replacement"] = {
                "route": "upstox", "source_priority": ["upstox"], "independent_broker_sources": 1,
            }
            broker = getattr(self, "execution_broker", None)
            if broker is not None:
                health["broker_status"] = broker.execution_status()
            return health
        elif upstox_healthy:
            health["all_brokers_unavailable"] = False
            health["broker_degraded"] = False
        remaining = max(
            0.0,
            float(getattr(self, "_angel_rest_backoff_until", 0.0) or 0.0) - time.time(),
        )
        health.update({
            "angel_rest_backoff_active": remaining > 0.0,
            "angel_rest_backoff_seconds_remaining": round(remaining, 2),
            "angel_rest_backoff_failures": int(
                getattr(self, "_angel_rest_backoff_failures", 0) or 0
            ),
        })
        angel = getattr(self, "angel_index", None)
        health["angelone_index_quotes"] = (
            angel.health() if angel is not None else {
                "configured": False,
                "connected": False,
                "price_only": True,
                "index_volume_available": False,
            }
        )
        if self.smart_api is not None:
            try:
                limiter = get_shared_api_limiter(self.smart_api)
                health["angel_rest_quote_gate"] = {
                    **limiter.snapshot(),
                    "paced_calls": int(limiter.calls),
                    "pacing_timeouts": int(limiter.timeouts),
                }
            except (AttributeError, TypeError):
                # Lightweight test/dry-run stubs may be immutable bare objects.
                health["angel_rest_quote_gate"] = {"available": False}
        return health

    @staticmethod
    def _numeric_tick_fields(payload: Dict[str, Any]) -> Dict[str, float]:
        fields: Dict[str, float] = {}
        for key, value in (payload or {}).items():
            key_text = str(key or "").lower()
            if not (
                any(part in key_text for part in ("vol", "qty", "quantity"))
                or key_text in {"ltq", "last_traded_quantity"}
            ):
                continue
            try:
                numeric = float(value or 0.0)
            except Exception:
                continue
            fields[str(key)] = numeric
        return fields

    def _extract_index_tick_volume(
        self,
        payload: Dict[str, Any],
        preferred_keys: Tuple[str, ...],
    ) -> Tuple[float, str, Dict[str, float]]:
        fields = self._numeric_tick_fields(payload)
        for key in preferred_keys:
            if key in fields and fields[key] > 0:
                return fields[key], key, fields
        # Do not infer a cumulative day total from arbitrary `qty`/`ltq` fields.
        # Those are commonly per-trade quantities; passing one to CandleBuilder,
        # whose volume contract is cumulative, fabricates a new delta on every tick.
        first_key = next((key for key in preferred_keys if key in fields), "")
        return (fields.get(first_key, 0.0) if first_key else 0.0), first_key, fields

    def _record_index_tick_volume_diag(
        self,
        instrument: str,
        source: str,
        volume: float,
        volume_key: str,
        volume_fields: Dict[str, float],
        now_ts: Optional[float] = None,
        *,
        volume_source: str = "",
        volume_basis: str = "",
        volume_unit: str = "",
        volume_comparable: Optional[bool] = None,
    ) -> None:
        name = str(instrument or "").upper()
        if not name:
            return
        now = time.time() if now_ts is None else float(now_ts)
        minute_bucket = int(now // 60)
        previous = self._index_tick_volume_diag.get(name, {})
        tick_count = int(previous.get("tick_count_current_minute", 0) or 0)
        if int(previous.get("minute_bucket", -1) or -1) != minute_bucket:
            tick_count = 0
        tick_count += 1
        incoming_available = bool(float(volume or 0.0) > 0.0)
        cumulative_field_present = bool(str(volume_key or "").strip())
        previous_positive_at = float(previous.get("volume_observed_at", 0.0) or 0.0)
        previous_volume = float(previous.get("volume", 0.0) or 0.0)
        observed_at = previous_positive_at
        if incoming_available and (
            previous_positive_at <= 0.0 or float(volume) != previous_volume
        ):
            observed_at = now
        elif cumulative_field_present and not incoming_available:
            # An explicit cumulative zero is a reset/unavailable observation, not a
            # price-only tick. Do not carry yesterday's positive total through it.
            previous_volume = 0.0
            observed_at = 0.0
        retained_available = bool(
            not cumulative_field_present
            and previous_volume > 0.0
            and previous_positive_at > 0.0
        )
        available = bool(incoming_available or retained_available)
        effective_volume = float(volume or 0.0) if incoming_available else previous_volume
        quantity_only = any(
            str(key or "").lower()
            in {"ltq", "qty", "quantity", "last_traded_quantity"}
            for key in (volume_fields or {})
        )
        unavailable_reason = ""
        if not available:
            unavailable_reason = (
                "cumulative_volume_missing; per_trade_quantity_ignored"
                if quantity_only
                else "no_positive_cumulative_volume"
            )
        resolved_source = str(volume_source or previous.get("volume_source") or "")
        resolved_basis = str(volume_basis or previous.get("volume_basis") or "")
        resolved_unit = str(volume_unit or previous.get("volume_unit") or "")
        comparable = (
            bool(volume_comparable)
            if volume_comparable is not None
            else bool(previous.get("volume_comparable", False))
        )
        self._index_tick_volume_diag[name] = {
            "instrument": name,
            "source": str(source or ""),
            "volume_available": available,
            "volume_source": resolved_source,
            "volume_basis": resolved_basis,
            "volume_unit": resolved_unit,
            "volume_comparable": bool(available and comparable),
            "volume_unavailable_reason": unavailable_reason,
            "volume_reliable": bool(available and comparable),
            "volume": effective_volume,
            "volume_key": str(volume_key or ""),
            "volume_fields": dict(volume_fields or {}),
            "tick_count_current_minute": tick_count,
            "minute_bucket": minute_bucket,
            "updated_at": now,
            "volume_observed_at": observed_at if observed_at > 0.0 else None,
        }

    def get_index_tick_volume_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        now = time.time()
        result: Dict[str, Dict[str, Any]] = {}
        for inst, payload in (getattr(self, "_index_tick_volume_diag", {}) or {}).items():
            if not isinstance(payload, dict):
                continue
            row = dict(payload)
            positive_at = float(row.get("volume_observed_at", 0.0) or 0.0)
            # A volume age exists only after a positive cumulative-volume
            # observation; quote/tick receipt time is never a substitute.
            row["volume_age_seconds"] = (
                round(max(0.0, now - positive_at), 3) if positive_at > 0.0 else None
            )
            age = row["volume_age_seconds"]
            if (
                row.get("volume_available")
                and age is not None
                and float(age) > float(self.FYERS_VOLUME_STALE_SECONDS)
            ):
                row["volume_available"] = False
                row["volume_comparable"] = False
                row["volume_reliable"] = False
                row["volume_unavailable_reason"] = "cumulative_volume_stale"
            result[str(inst)] = row
        return result

    def get_market_source_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Return source latency metadata for live-vs-delayed signal weighting."""
        return build_market_source_snapshot(
            self,
            symbol,
            INDEX_SPOT_TOKENS,
            FYERS_INDEX_SYMBOLS,
        )

    # ═══════════════════════════════════════════════════════════
    # UNIFIED PRICE BRIDGE (The "Master" Lock)
    # ═══════════════════════════════════════════════════════════

    def _index_quote_guard(self):
        """The short-lived guard shared by both websocket callback threads.

        Focused tests construct the provider with ``__new__``. Lazily materialising
        the lock keeps those probes valid while normal production instances create it
        before either socket starts.
        """
        guard = getattr(self, "_index_quote_lock", None)
        if guard is None:
            guard = threading.RLock()
            self._index_quote_lock = guard
        return guard

    @staticmethod
    def _market_timestamp_epoch(value: Any) -> Optional[float]:
        """Normalise a broker market timestamp to epoch seconds when possible."""
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            try:
                return float(value.timestamp())
            except (OSError, OverflowError, ValueError):
                return None
        try:
            epoch = float(value)
        except (TypeError, ValueError):
            try:
                text = str(value).strip().replace("Z", "+00:00")
                epoch = float(datetime.fromisoformat(text).timestamp())
            except (TypeError, ValueError, OSError, OverflowError):
                return None
        # SDKs variously expose seconds, milliseconds, microseconds or nanoseconds.
        while epoch > 100_000_000_000.0:
            epoch /= 1000.0
        # Smaller values are generally a time-of-day or an expiry identifier, not an
        # exchange timestamp that can participate in a freshness decision.
        return epoch if epoch >= 946_684_800.0 else None

    @classmethod
    def _exchange_timestamp_from_payload(cls, payload: Dict[str, Any]) -> Optional[float]:
        if not isinstance(payload, dict):
            return None
        for key in (
            "exchange_timestamp",
            "exch_feed_time",
            "last_trade_time",
            "last_traded_time",
            "tt",
        ):
            epoch = cls._market_timestamp_epoch(payload.get(key))
            if epoch is not None:
                return epoch
        return None

    def _quote_age_components(
        self,
        payload: Dict[str, Any],
        now_ts: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        """Return receipt, exchange and conservative effective quote ages."""
        now = time.time() if now_ts is None else float(now_ts)
        receipt_raw = payload.get("received_at", payload.get("time"))
        if isinstance(receipt_raw, datetime):
            receipt_epoch = self._market_timestamp_epoch(receipt_raw) or 0.0
        else:
            try:
                receipt_epoch = float(receipt_raw or 0.0)
            except (TypeError, ValueError):
                receipt_epoch = 0.0
        exchange_epoch = self._market_timestamp_epoch(
            payload.get("exchange_at", payload.get("exchange_timestamp"))
        )
        receipt_age = max(0.0, now - receipt_epoch) if receipt_epoch > 0 else None
        exchange_age = max(0.0, now - exchange_epoch) if exchange_epoch else None
        comparable = [age for age in (receipt_age, exchange_age) if age is not None]
        return {
            "received_at": receipt_epoch or None,
            "exchange_at": exchange_epoch,
            "receipt_age": receipt_age,
            "exchange_age": exchange_age,
            # A freshly received replay of an old exchange packet is still old. max()
            # catches both that case and the ordinary stopped-socket case.
            "age": max(comparable) if comparable else None,
        }

    @staticmethod
    def _quote_epoch_iso(epoch: Optional[float]) -> Optional[str]:
        """Render a cached epoch without changing its timezone or freshness meaning."""
        if epoch is None:
            return None
        try:
            return datetime.fromtimestamp(float(epoch)).astimezone().isoformat()
        except (OSError, OverflowError, TypeError, ValueError):
            return None

    def _quote_timing_metadata(
        self,
        payload: Dict[str, Any],
        now_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Expose exchange time separately from local receipt time.

        Exchange time is preferable for diagnosing a replayed/delayed packet, while
        receipt time diagnoses a locally stalled consumer.  ``quote_age_seconds`` is
        deliberately the worse of the two so a just-received old packet never looks
        fresh merely because it crossed the process boundary a moment ago.
        """
        ages = self._quote_age_components(payload, now_ts)
        received_at = ages.get("received_at")
        exchange_at = ages.get("exchange_at")
        quote_at = exchange_at if exchange_at is not None else received_at
        return {
            "received_at": received_at,
            "received_at_epoch": received_at,
            "exchange_at": exchange_at,
            "exchange_timestamp_epoch": exchange_at,
            "receipt_timestamp": self._quote_epoch_iso(received_at),
            "exchange_timestamp": self._quote_epoch_iso(exchange_at),
            "quote_timestamp": self._quote_epoch_iso(quote_at),
            "receipt_age_seconds": ages.get("receipt_age"),
            "exchange_age_seconds": ages.get("exchange_age"),
            "quote_age_seconds": ages.get("age"),
            "freshness_basis": (
                "exchange_and_receipt" if exchange_at is not None else "receipt"
            ),
        }

    def _freshest_cached_index_quote(
        self,
        keys: Tuple[str, ...],
        max_age: float,
        now_ts: float,
    ) -> Optional[Tuple[float, Dict[str, Any]]]:
        candidates = []
        with self._index_quote_guard():
            ws_cache = getattr(self, "_ws_cache", {}) or {}
            cached_rows = [
                dict(ws_cache.get(key) or {})
                for key in dict.fromkeys(key for key in keys if key)
            ]
        for cached in cached_rows:
            if float(cached.get("ltp", 0.0) or 0.0) <= 0:
                continue
            ages = self._quote_age_components(cached, now_ts)
            age = ages.get("age")
            if age is not None and age <= max_age:
                candidates.append((float(age), cached))
        return min(candidates, key=lambda item: item[0]) if candidates else None

    def _index_broker_cache_keys(self, symbol: str, token: str = "") -> Tuple[str, ...]:
        kite_token = self._kite_token_for_index(symbol)
        configured_token = INDEX_SPOT_TOKENS.get(symbol, ("", ""))[0]
        legacy_token = token or configured_token
        short_token = legacy_token[3:] if legacy_token.startswith("999") else legacy_token
        return tuple(dict.fromkeys(k for k in (kite_token, legacy_token, short_token) if k))

    def _select_index_ws_quote(
        self,
        symbol: str,
        token: str = "",
        now_ts: Optional[float] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """Select one stable live source instead of oscillating between brokers.

        The EXECUTION BROKER's tick is authoritative — it is the price orders fill
        against. Fyers is a display/analysis failover, not a competing tick that should
        replace it several times per second.

        This keyed the lookup on INDEX_SPOT_TOKENS, the AngelOne token map. Post-Kite
        those keys never appear in the cache, so the broker branch always missed and
        every index quote silently came from Fyers — which on 2026-08-01 was quoting
        NIFTY 6.49% high. The Kite ticker was running correctly the whole time; nothing
        was reading it.
        """
        now = time.time() if now_ts is None else float(now_ts)

        # Selection is by TRUST first, freshness second. Picking the fresher tick let a
        # continuously-streaming bad feed outrank a briefly-quiet good one: on
        # 2026-08-01 Fyers published a fabricated Saturday session for every index
        # (NIFTY lp=26,230.10 against a true 24,383.60, with its own prev_close field
        # correctly reading 24,383.60) and won purely because it kept ticking.
        #
        # Staleness only means something while the market is open. With the market
        # shut, the last traded price IS the price and does not decay — so a broker
        # tick from an hour ago is still correct, while the old 6s window aged it out
        # and handed the quote to whatever kept streaming.
        try:
            market_open = bool(self.is_market_open())
        except Exception:
            market_open = True
        broker_window = self.INDEX_KITE_PRICE_FRESH_SECONDS if market_open else 86400.0
        broker = self._freshest_cached_index_quote(
            self._index_broker_cache_keys(symbol, token), broker_window, now
        )
        if broker is not None and not str(
            broker[1].get("source") or ""
        ).lower().startswith("kite"):
            broker = None
        fyers = self._freshest_cached_index_quote(
            (FYERS_INDEX_SYMBOLS.get(symbol, ""),),
            self.INDEX_FYERS_PRICE_FRESH_SECONDS,
            now,
        )
        if fyers is not None and not str(
            fyers[1].get("source") or ""
        ).lower().startswith("fyers"):
            fyers = None

        if broker is None and self._fyers_backup_enabled():
            primary_rest = dict((getattr(self, "_ltp_cache", {}) or {}).get(symbol) or {})
            primary_age = self._quote_age_components(primary_rest, now).get("age")
            if (str(primary_rest.get("source", "")).startswith("kite")
                    and float(primary_rest.get("ltp") or 0) > 0
                    and primary_age is not None and 0 <= primary_age <= broker_window):
                broker = (primary_age, primary_rest)

        angel = self._freshest_cached_index_quote(("angel:" + symbol,), broker_window, now) if self._replacement_enabled() else None
        if broker is not None and angel is not None:
            self._flag_quote_divergence(symbol, "kite", float(broker[1]["ltp"]), "angelone", float(angel[1]["ltp"]))

        if broker is not None and fyers is not None:
            broker_ltp = float(broker[1].get("ltp", 0.0) or 0.0)
            fyers_ltp = float(fyers[1].get("ltp", 0.0) or 0.0)
            self._flag_quote_divergence(
                symbol, "kite", broker_ltp, "fyers", fyers_ltp
            )
            # Fresh Kite remains authoritative even when the feeds agree. Choosing the
            # newest packet here caused sub-threshold source flips several times a second,
            # while both callbacks independently mutated one OHLC candle.
            return broker[1], broker[0]

        selected = broker or angel or fyers
        if selected is None:
            return None, None
        return selected[1], selected[0]

    def _emit_arbitrated_index_tick(
        self,
        index_name: str,
        price: float,
        volume: float,
        volume_source: str,
        price_source: str,
        received_at: Optional[float] = None,
        exchange_at: Optional[float] = None,
    ) -> bool:
        """Emit one index tick only after active/standby price-source arbitration.

        Fyers remains the volume provider. While Kite is fresh, a Fyers volume tick is
        emitted with the cached Kite PRICE; a Fyers price-only tick is suppressed. If
        Kite stops, two fresh Fyers packets over a short dwell are required before Fyers
        owns price. A recovering Kite tick takes ownership immediately.
        """
        if self._fyers_backup_enabled():
            return self._emit_three_broker_tick(index_name, price, volume, volume_source, price_source, received_at, exchange_at)
        now = time.time() if received_at is None else float(received_at)
        try:
            value = float(price or 0.0)
            emit_volume = float(volume or 0.0)
        except (TypeError, ValueError):
            return False
        if value <= 0:
            return False
        source = str(price_source or "").lower()
        is_kite = source.startswith("kite")
        is_fyers = source.startswith("fyers")
        is_angel = source.startswith("angel") and self._replacement_enabled()
        if not (is_kite or is_fyers or is_angel):
            return False

        try:
            market_open = bool(self.is_market_open())
        except Exception:
            market_open = True
        source_exchange_at = self._market_timestamp_epoch(exchange_at)
        if market_open and source_exchange_at is not None:
            source_age = max(0.0, now - source_exchange_at)
            source_limit = (
                self.INDEX_KITE_PRICE_FRESH_SECONDS
                if is_kite
                else self.INDEX_FYERS_PRICE_FRESH_SECONDS
            )
            if source_age > source_limit:
                return False

        # Websocket SDKs can replay the closing packet after a reconnect and Fyers has
        # previously streamed synthetic weekend values. Cache those packets for a REF
        # badge, but never stamp their *receipt* minute into the live CandleBuilder.
        if not market_open:
            return False

        broker_window = self.INDEX_KITE_PRICE_FRESH_SECONDS
        broker = self._freshest_cached_index_quote(
            self._index_broker_cache_keys(index_name), broker_window, now
        )
        if broker is not None and not str(
            broker[1].get("source") or ""
        ).lower().startswith("kite"):
            broker = None
        stale_broker = self._freshest_cached_index_quote(
            self._index_broker_cache_keys(index_name), float("inf"), now
        )
        if stale_broker is not None and not str(
            stale_broker[1].get("source") or ""
        ).lower().startswith("kite"):
            stale_broker = None

        secondary_source = "angel_ws" if self._replacement_enabled() else "fyers_ws"
        emit_price = None
        emit_native_price = False
        with self._index_quote_guard():
            routes = getattr(self, "_index_price_route", None)
            if not isinstance(routes, dict):
                routes = {}
                self._index_price_route = routes
            route = routes.setdefault(str(index_name), {})
            session_date = datetime.fromtimestamp(now).date().isoformat()
            if route.get("session_date") != session_date:
                route.clear()
                route["session_date"] = session_date

            if is_kite:
                route.update(
                    {
                        "active_source": "kite_ws",
                        "last_authoritative_price": value,
                        "last_switch_at": now,
                        "fyers_candidate_since": 0.0,
                        "fyers_candidate_ticks": 0,
                    }
                )
                emit_price = value
            elif broker is not None:
                authoritative = float(broker[1].get("ltp", 0.0) or 0.0)
                route.update(
                    {
                        "active_source": "kite_ws",
                        "last_authoritative_price": authoritative,
                        "fyers_candidate_since": 0.0,
                        "fyers_candidate_ticks": 0,
                    }
                )
                # Keep Fyers cumulative volume without allowing its competing price to
                # alter close/high/low. Price-only packets add no information.
                if emit_volume > 0 and authoritative > 0:
                    emit_price = authoritative
            elif str(route.get("active_source") or "") == secondary_source:
                route["last_authoritative_price"] = value
                emit_price = value
            else:
                if not route.get("active_source") and stale_broker is not None:
                    route["active_source"] = "kite_ws"
                    route["last_authoritative_price"] = float(
                        stale_broker[1].get("ltp", 0.0) or 0.0
                    )
                since = float(route.get("fyers_candidate_since", 0.0) or 0.0)
                if since <= 0:
                    since = now
                    route["fyers_candidate_since"] = since
                    route["fyers_candidate_ticks"] = 0
                ticks = int(route.get("fyers_candidate_ticks", 0) or 0) + 1
                route["fyers_candidate_ticks"] = ticks
                ready = bool(
                    ticks >= self.INDEX_PRICE_FAILOVER_MIN_TICKS
                    and now - since >= self.INDEX_PRICE_FAILOVER_HYSTERESIS_SECONDS
                )
                if ready:
                    route.update(
                        {
                            "active_source": secondary_source,
                            "last_authoritative_price": value,
                            "last_switch_at": now,
                            "fyers_candidate_since": 0.0,
                            "fyers_candidate_ticks": 0,
                        }
                    )
                    emit_price = value
                elif emit_volume > 0:
                    # During the short handoff dwell the previous Kite price can carry
                    # the advancing Fyers volume. Its cumulative baseline means no
                    # volume is lost when Fyers becomes active on the next packet.
                    prior = float(route.get("last_authoritative_price", 0.0) or 0.0)
                    if prior > 0:
                        emit_price = prior
            emit_native_price = bool(is_kite or (broker is None
                and route.get("active_source") == secondary_source and source == secondary_source))

        if emit_price is None:
            return False
        self._emit_index_tick(
            index_name,
            float(emit_price),
            emit_volume,
            str(volume_source or ""),
            timestamp=datetime.fromtimestamp(now),
            native_price=emit_native_price,
        )
        return True

    # A Fyers index-volume reading older than this counts as the feed being down.
    # 120s was far too slack for a live market: at 5min bars that is a third of a
    # candle built on a frozen number, and the system would keep reporting healthy
    # volume the whole time. 20s is longer than any normal tick gap but short enough
    # that at most one bar is affected before the fallback engages.
    FYERS_VOLUME_STALE_SECONDS = 20.0
    # The futures fallback hits the broker's quote endpoint, and it is consulted from
    # the tick path, so it is throttled hard. Cumulative volume only ever grows, so a
    # coarse refresh loses nothing but the last few seconds of a minute's delta.
    INDEX_FUTURES_VOLUME_REFRESH_SECONDS = 15.0

    # While in fallback, actively poll Fyers this often so recovery is noticed within
    # seconds. The websocket reconnects on its own, but a dead socket that never
    # reconnects would otherwise leave the system on futures volume indefinitely.
    FYERS_RECOVERY_PROBE_SECONDS = 3.0

    def _emit_index_tick(
        self,
        index_name: str,
        price: float,
        volume,
        source: str,
        timestamp: Optional[datetime] = None,
        *, native_price: bool = True,
    ) -> None:
        """Hand a tick to the candle builder, telling it which feed the volume came from.

        The source label is what stops a Fyers(shares) -> futures(contracts) switch being
        subtracted as if it were one series; the builder rebases instead.
        """
        callback = getattr(self, "on_tick", None)
        if not callable(callback):
            return
        # Decide arity by INSPECTING the callback, not by catching TypeError around the
        # call. A try/except there cannot tell "wrong number of arguments" from a
        # TypeError raised inside the callback body, and the retry would run the tick a
        # second time — double-counting tick_count, re-entering the candle mutation, and
        # replaying it WITHOUT the source label, i.e. without the rebase protection.
        if self._tick_callback_takes_source is None or self._tick_callback_ref is not callback:
            self._tick_callback_ref = callback
            self._tick_callback_takes_source = self._accepts_five_args(callback)
        stamp = timestamp or datetime.now()
        if getattr(callback, "_utema_accepts_native_price", False) is True:
            # A standby volume receipt repeats the owner's cached price. Keep its
            # candle/volume update, but do not advance the native price clock and
            # reject a later-delivered owner tick from an older callback batch.
            callback(index_name, price, volume, stamp, source, native_price=native_price)
        elif self._tick_callback_takes_source:
            callback(index_name, price, volume, stamp, source)
        else:
            callback(index_name, price, volume, stamp)

    @staticmethod
    def _accepts_five_args(callback) -> bool:
        """Whether this callback takes the volume_source argument."""
        import inspect

        try:
            inspect.signature(callback).bind("i", 0.0, 0, None, "")
            return True
        except TypeError:
            return False
        except (ValueError, KeyError):
            # Un-introspectable (a builtin or C callable). Assume the legacy form:
            # passing an argument it cannot take would break every tick.
            return False

    def _index_volume_for_tick(self, index_name: str, now_ts: float):
        """(cumulative_volume, source) to carry on a Kite index tick.

        Kite sends no volume for an index, so a Kite tick would otherwise reset the
        candle's delta to zero. Carry the last Fyers cash total instead.

        Only ever the CASH total. The near-month futures fallback is still resolved by
        get_index_activity — for the health endpoint and the OI read — but it is not
        returned here, because this value is ticked straight into the index candle
        series whose volume column is Fyers cash shares from the history backfill.
        Contracts are a different quantity from constituent shares, not a scale of it
        (measured 0.006 for NIFTY, 0.001 for SENSEX), and nothing downstream can
        separate them once they share a column: backfilled rows carry no volume_basis,
        so the homogeneity guard reads an unlabelled row as "unknown, not a unit" and
        passes a cash window with contract bars at its right edge as homogeneous.

        When Fyers is quiet the carried total simply stops growing, so the delta is 0
        and the bar reads empty until the history backfill fills it. An undercount that
        the profiler already skips is the safe failure; a bar of contracts measured
        against a mean of shares is the one that fabricates a surge.
        """
        carried = float(self._index_cum_volume.get(index_name, 0.0) or 0.0)
        source = str(self._index_cum_volume_source.get(index_name, "") or "")
        fyers_ts = float(self._index_cum_volume_at.get(index_name, 0.0) or 0.0)
        fyers_fresh = carried > 0 and (now_ts - fyers_ts) <= self.FYERS_VOLUME_STALE_SECONDS
        # Derivative participation is independent of cash-volume availability.
        self._request_index_volume_refresh(index_name, now_ts)
        if fyers_fresh:
            return carried, source or "fyers_index"

        # NEVER block here. kiteconnect dispatches on_ticks synchronously on the socket
        # reader thread, so a blocking HTTPS round-trip to Fyers (the feed that is by
        # definition already degraded) plus a Kite quote under a lock would stall ticks
        # for EVERY instrument for the duration. Kick the refresh onto a worker and serve
        # whatever the last one resolved; cumulative volume only grows, so a slightly old
        # total is a small undercount, never a fabricated surge.
        return carried, source

    def _request_index_volume_refresh(self, index_name: str, now_ts: float) -> None:
        """Throttled, off-thread refresh of the cumulative index volume.

        Serialised PER INDEX, not globally. The four indices tick within the same
        instant, and a single shared lock meant the first to arrive won while the other
        three failed a non-blocking acquire — having ALREADY stamped their throttle, so
        they would not try again for another 15s, lose the same race, and repeat.
        Measured with a realistic REST latency, that left 1 of 4 indices ever refreshing;
        the rest carried a cumulative volume of zero all session, which is what set
        volume_available=false everywhere and starved both the entry confidence score and
        the exit volume test.

        The stamp is therefore taken only once a dispatch is certain. Burning the window
        on a refresh that never ran is what turned a lost race into a dead feed.
        """
        last_try = float(self._index_futures_volume_at.get(index_name, 0.0) or 0.0)
        if now_ts - last_try < self.INDEX_FUTURES_VOLUME_REFRESH_SECONDS:
            return
        with self._index_volume_refresh_guard:
            if index_name in self._index_volume_refresh_inflight:
                # Already refreshing this index; leave its throttle untouched so the next
                # tick after it completes can legitimately ask again.
                return
            self._index_volume_refresh_inflight.add(index_name)
            self._index_futures_volume_at[index_name] = now_ts
        try:
            threading.Thread(
                target=self._refresh_index_volume_worker,
                args=(index_name,),
                name=f"index-volume-{index_name}",
                daemon=True,
            ).start()
        except Exception as exc:
            with self._index_volume_refresh_guard:
                self._index_volume_refresh_inflight.discard(index_name)
            logger.debug(f"Index volume refresh dispatch failed for {index_name}: {exc}")

    def _refresh_index_volume_worker(self, index_name: str) -> None:
        try:
            activity = self.get_index_activity(index_name) or {}
            source = str(activity.get("source") or "")
            volume = float(activity.get("volume") or 0.0)
            futures = dict(activity.get("futures") or {})
            if futures.get("available") and float(futures.get("volume") or 0.0) > 0:
                exchange_at = self._exchange_timestamp_from_payload(futures)
                receipt_at = time.time()
                # A REST receipt clock cannot turn yesterday's derivative quote into a
                # current-volume observation.  Snapshot increments are only valid when
                # the broker supplied an exchange/last-trade clock and it is current.
                if exchange_at is not None and abs(receipt_at - exchange_at) <= 90.0:
                    self._participation_store().observe(index_name, "futures", "kite",
                        str(futures.get("tradingsymbol") or ""), exchange_at,
                        float(futures.get("volume") or 0), oi=float(futures.get("oi") or 0),
                        price=float(futures.get("last_price") or 0), expiry=str(futures.get("expiry") or ""),
                        unit=str(futures.get("unit") or "broker_reported_quantity"))
                    self._record_index_futures_volume_snapshot(
                        index_name, futures, now_ts=exchange_at
                    )
                    # Read-only listener (the build-up shadow); it never touches this feed.
                    observer = getattr(self, "futures_observer", None)
                    if callable(observer):
                        try:
                            observer(index_name, exchange_at, float(futures.get("last_price") or 0),
                                     float(futures.get("volume") or 0), float(futures.get("oi") or 0))
                        except Exception as exc:
                            logger.debug(f"Futures observer failed for {index_name}: {exc}")
            if volume <= 0:
                return
            if source == "fyers_index":
                self._index_cum_volume[index_name] = volume
                self._index_cum_volume_source[index_name] = source
                self._index_cum_volume_at[index_name] = time.time()
                self._index_futures_fallback.pop(index_name, None)
            elif source == "kite_index_futures":
                # Recorded for the health endpoint and the OI read, NOT carried onto a
                # tick. An index candle's volume column is Fyers cash shares (the
                # `Prioritizing Fyers history ... to get volume` backfill); ticking
                # near-month CONTRACTS into it puts two quantities ~1000x apart in one
                # column. The homogeneity guard cannot catch that, because backfilled
                # rows carry no volume_basis and it reads an unlabelled row as "unknown,
                # not a unit" — so a window of cash bars with contract bars at its right
                # edge passes as homogeneous. Measured live on 2026-08-14: forming bars
                # of 30-780 contracts against 1m cash bars of 17k-1M shares.
                self._index_futures_fallback[index_name] = {
                    "volume": volume,
                    "oi": float(activity.get("oi") or 0.0),
                    "at": time.time(),
                    "tradingsymbol": str(futures.get("tradingsymbol") or ""),
                    "unit": str(activity.get("volume_unit") or activity.get("unit") or futures.get("unit") or "broker_reported_quantity"),
                }
        except Exception as exc:
            logger.debug(f"Index activity fallback failed for {index_name}: {exc}")
        finally:
            # This worker is off the price/order path; tick collection itself does no I/O.
            try:
                self._participation_store().flush()
            except Exception as exc:
                logger.warning(f"Derivative observations could not be archived: {exc}")
            with self._index_volume_refresh_guard:
                self._index_volume_refresh_inflight.discard(index_name)

    def _participation_store(self):
        from data.derivative_participation import DerivativeParticipation
        # Workers are serialized per index; use a class lock for first publication.
        with self._option_oi_snapshot_file_lock:
            store = getattr(self, "_derivative_participation", None)
            if store is None:
                store = self._derivative_participation = DerivativeParticipation(
                    archive_dir=(Path(os.environ["UTBOT_TEST_DB_PATH"]).parent / "derivative_observations"
                                 if os.environ.get("UTBOT_TEST_DB_PATH") else "data_store/derivative_observations"))
            return store

    def get_derivative_participation(self, instrument, timeframe, as_of):
        return self._participation_store().features(instrument, timeframe, as_of)

    def get_recorded_option_chain(self, instrument, as_of, expiry=""):
        return self._participation_store().recorded_option_chain(instrument, as_of, expiry=expiry)

    def _archive_option_chain(self, instrument, chain, timestamp, quality):
        try:
            self._subscribe_derivative_pairs(instrument, chain)
        except Exception as exc:
            logger.warning(f"Derivative paired stream could not be subscribed for {instrument}: {exc}")
        try:
            timestamp = max(float(timestamp), float(chain.attrs.get("observed_at") or timestamp))
            self._participation_store().record_option_chain(instrument, chain, timestamp, quality)
            self._participation_store().flush()
        except Exception as exc:
            logger.warning(f"Option chain could not be archived for replay: {exc}")

    def _subscribe_derivative_pairs(self, instrument, chain):
        """Warm both exact CE/PE legs from the existing background chain refresh."""
        if not getattr(self, "start_streams", False) or chain is None or chain.empty:
            return
        if getattr(self, "upstox_only", False):
            from engine.expiry_manager import expiry_manager

            frame = expiry_manager.df
            if frame is None or not {"underlying_spot", "call_token", "put_token", "strike"}.issubset(chain):
                return
            spot = float(chain["underlying_spot"].median())
            if not np.isfinite(spot) or spot <= 0:
                return
            strikes = chain.iloc[(chain["strike"] - spot).abs().argsort()[:7]]
            wanted = set(strikes["call_token"]) | set(strikes["put_token"])
            contracts = frame[(frame["name"] == instrument) & frame["token"].isin(wanted)]
            for contract in contracts.to_dict("records"):
                self._register_derivative_contract(instrument, contract)
                self._upstox_stream.watch(contract["token"])
            return
        kite = getattr(self, "kite", None)
        resolver = getattr(kite, "resolve_option", None)
        from intelligence.derivative_context import _expiry
        expiry = _expiry(chain.attrs.get("expiry"))
        if not callable(resolver) or expiry is None:
            return
        cached = (getattr(self, "_ltp_cache", {}) or {}).get(instrument, {})
        spot = float(cached.get("ltp", cached.get("price", 0)) or 0) if isinstance(cached, dict) else 0.0
        if spot <= 0:
            # The observed paired chain supplies an explicit ATM marker when available.
            atm_rows = chain.loc[chain["is_atm"].eq(True)] if "is_atm" in chain else pd.DataFrame()
            if not atm_rows.empty:
                spot = float(atm_rows.iloc[0]["strike"])
            elif "call_ltp" in chain and "put_ltp" in chain:
                distance = (pd.to_numeric(chain.call_ltp, errors="coerce")-
                            pd.to_numeric(chain.put_ltp, errors="coerce")).abs()
                if distance.notna().any():
                    spot = float(chain.loc[distance.idxmin(), "strike"])
        if spot <= 0:
            return
        strikes = sorted(pd.to_numeric(chain["strike"], errors="coerce").dropna().unique(),
                         key=lambda strike: abs(strike-spot))[:11]
        tokens = []
        for strike in strikes:
            pair = [resolver(instrument, float(strike), side, expiry=expiry) for side in ("CE", "PE")]
            if not all(isinstance(row, dict) and row.get("instrument_token")
                       and _expiry(row.get("expiry")) == expiry for row in pair):
                continue
            for side, contract in zip(("CE", "PE"), pair):
                self._register_derivative_contract(instrument, dict(contract, option_type=side))
                tokens.append(str(contract["instrument_token"]))
        if tokens:
            self.subscribe_tokens(tokens)

    def _register_derivative_contract(self, instrument, contract):
        """Keep exact execution identities available to the non-blocking tick path."""
        if isinstance(contract, dict):
            token = str(contract.get("token") or contract.get("instrument_token") or "")
            symbol = str(contract.get("symbol") or contract.get("tradingsymbol") or "")
            expiry = str(contract.get("expiry") or "")
            from intelligence.derivative_context import _expiry
            parsed_expiry = _expiry(expiry)
            if parsed_expiry is not None:
                expiry = parsed_expiry.strftime("%d%b%Y").upper()
            if token and symbol and expiry:
                with self._option_oi_snapshot_file_lock:
                    identities = getattr(self, "_derivative_token_identity", None)
                    if identities is None:
                        identities = self._derivative_token_identity = {}
                    option_type = str(contract.get("option_type") or symbol[-2:]).upper()
                    strike = float(contract.get("strike_val") or contract.get("strike") or 0)
                    if not contract.get("strike_val") and contract.get("instrumenttype") == "OPTIDX":
                        strike /= 100.0  # Execution master stores Angel paise-style strikes.
                    identities[token] = (instrument, f"{symbol}|{expiry}|{token}",
                                         expiry, option_type, strike)
        return contract

    def _record_option_participation_tick(self, token, tick, received_at):
        identity = (getattr(self, "_derivative_token_identity", {}) or {}).get(str(token))
        if not identity or "oi" not in tick:
            return
        quantity = tick.get("volume_traded", tick.get("volume"))
        stamp = self._exchange_timestamp_from_payload(tick)
        if quantity is None or stamp is None or not 0 <= received_at-stamp <= 90:
            return
        instrument, contract, expiry, option_type, strike = identity
        provider = "upstox" if tick.get("source") == "upstox_ws" else "kite"
        self._participation_store().observe(instrument, "options", provider, contract,
            stamp, quantity, oi=tick["oi"], price=tick.get("ltp", tick.get("last_price", 0)), clock="exchange",
            expiry=expiry, option_type=option_type, strike=strike)

    def _record_index_futures_volume_snapshot(
        self,
        index_name: str,
        futures: Dict[str, Any],
        *,
        now_ts: Optional[float] = None,
    ) -> None:
        """Build an isolated one-minute contracts baseline from cumulative FUT volume.

        This series is intentionally never written to index candles. Its ratios compare
        near-month contracts only with earlier minutes of the same contract.
        """
        name = str(index_name or "").upper()
        symbol = str((futures or {}).get("tradingsymbol") or "")
        try:
            cumulative = float((futures or {}).get("volume") or 0.0)
            observed_at = float(now_ts or time.time())
        except (TypeError, ValueError):
            return
        if not name or not symbol or cumulative <= 0 or observed_at <= 0:
            return
        if not hasattr(self, "_index_futures_minute_lock"):
            self._index_futures_minute_lock = threading.RLock()
        if not hasattr(self, "_index_futures_minute_state"):
            self._index_futures_minute_state = {}
        minute = int(observed_at // 60)
        session_day = datetime.fromtimestamp(observed_at).date().isoformat()
        with self._index_futures_minute_lock:
            state = self._index_futures_minute_state.get(name)
            reset = bool(
                not isinstance(state, dict)
                or state.get("tradingsymbol") != symbol
                or state.get("session_day") != session_day
                or cumulative < float(state.get("last_cumulative") or 0.0)
                or minute < int(state.get("minute") or minute)
            )
            if reset:
                self._index_futures_minute_state[name] = {
                    "tradingsymbol": symbol,
                    "session_day": session_day,
                    "minute": minute,
                    "current_volume": 0.0,
                    "completed": deque(maxlen=20),
                    "completed_rates": deque(maxlen=20),
                    "current_seconds": 0.0,
                    "unallocated_volume": 0.0,
                    "last_cumulative": cumulative,
                    "last_at": observed_at,
                }
                return

            last_minute = int(state.get("minute") or minute)
            last_cumulative = float(state.get("last_cumulative") or cumulative)
            delta = max(0.0, cumulative - last_cumulative)
            elapsed = max(0.0, observed_at - float(state.get("last_at") or observed_at))
            if minute != last_minute:
                # The cumulative delta straddles two buckets.  It is unknowable how much
                # belongs to each, even when only one minute was crossed, so never book it
                # into the new minute.  Doing so made the first post-boundary poll look
                # like a genuine one-minute burst.  Keep the completed *known* bucket and
                # rebase the cumulative counter; subsequent same-minute observations are
                # again an exact increment.
                if minute - last_minute == 1:
                    state["completed"].append(
                        max(0.0, float(state.get("current_volume") or 0.0))
                    )
                    seconds = float(state.get("current_seconds") or 0)
                    if seconds >= 30:
                        state.setdefault("completed_rates", deque(maxlen=20)).append(
                            float(state.get("current_volume") or 0) / seconds)
                else:
                    state["completed"].clear()
                    state.setdefault("completed_rates", deque(maxlen=20)).clear()
                state["minute"] = minute
                state["current_volume"] = 0.0
                state["current_seconds"] = 0.0
                state["unallocated_volume"] = float(state.get("unallocated_volume") or 0) + delta
                delta = 0.0
            elif elapsed <= 30:
                state["current_seconds"] = float(state.get("current_seconds") or 0) + elapsed
            else:
                state["unallocated_volume"] = float(state.get("unallocated_volume") or 0) + delta
                delta = 0.0
            state["current_volume"] = float(state.get("current_volume") or 0.0) + delta
            state["last_cumulative"] = cumulative
            state["last_at"] = observed_at

    def get_index_futures_minute_volume_context(
        self, index_name: str, *, now_ts: Optional[float] = None
    ) -> Dict[str, Any]:
        """Return contracts-only current/projected 1m volume against its own median."""
        name = str(index_name or "").upper()
        observed_at = float(now_ts or time.time())
        empty = {
            "available": False,
            "reliable": False,
            "source": ("upstox_index_futures_minute" if getattr(self, "upstox_only", False)
                       else "kite_index_futures_minute"),
            "unit": "broker_reported_quantity",
            "tradingsymbol": "",
            "current_volume": 0.0,
            "baseline_volume": 0.0,
            "ratio": 0.0,
            "projected_ratio": 0.0,
            "completed_samples": 0,
            "age_seconds": None,
        }
        lock = getattr(self, "_index_futures_minute_lock", None)
        states = getattr(self, "_index_futures_minute_state", None)
        if lock is None or not isinstance(states, dict):
            return empty
        with lock:
            state = states.get(name)
            if not isinstance(state, dict):
                return empty
            completed = [
                float(value)
                for value in list(state.get("completed") or [])
                if float(value) >= 0.0
            ]
            current = max(0.0, float(state.get("current_volume") or 0.0))
            last_at = float(state.get("last_at") or 0.0)
            symbol = str(state.get("tradingsymbol") or "")
            bucket_minute = int(state.get("minute") or -1)
            rates = list(state.get("completed_rates") or [])
            seconds = float(state.get("current_seconds") or 0)
            unallocated = float(state.get("unallocated_volume") or 0)
        age = max(0.0, observed_at - last_at) if last_at > 0 else None
        current_minute = int(observed_at // 60)
        # A receipt from the previous minute is not a live observation for the new
        # bucket.  This prevents the clock alone from changing projected participation.
        bucket_current = bucket_minute == current_minute
        positive = [value for value in completed if value > 0]
        baseline_rate = float(np.median(rates)) if rates else 0.0
        baseline = baseline_rate * 60
        ratio = current / baseline if baseline > 0 else 0.0
        projected = (current / seconds) / baseline_rate if seconds > 0 and baseline_rate > 0 else 0.0
        reliable = bool(
            symbol
            and age is not None
            and age <= 30.0
            and bucket_current
            and len(rates) >= 5
            and seconds >= 15
            and baseline > 0
        )
        return {
            **empty,
            "available": bool(symbol and age is not None and age <= 30.0 and bucket_current),
            "reliable": reliable,
            "tradingsymbol": symbol,
            "current_volume": round(current, 4),
            "baseline_volume": round(baseline, 4),
            "ratio": round(ratio, 4),
            "projected_ratio": round(min(projected, 10.0), 4),
            "completed_samples": len(positive),
            "measurement_basis": "sampled_volume_rate",
            "coverage_seconds": seconds,
            "volume_complete": False,
            "unallocated_volume": unallocated,
            "age_seconds": round(age, 2) if age is not None else None,
        }

    def get_index_activity(self, symbol: str) -> Dict[str, Any]:
        """Index participation, with a broker cross-check and a broker fallback.

        Fyers stays PRIMARY for index volume — it is the only source of the cash-market
        quantity every volume threshold in this system was calibrated against. Kite
        cannot supply it at all (an index is not a traded instrument), so the broker
        contributes near-month FUTURES volume and open interest instead: an independent,
        exchange-sourced read on whether there is conviction behind a move.

        The two are different quantities, not two versions of one. `unit` and `source`
        say which is in play, and `comparable_to_thresholds` is False whenever the
        fallback is active — a caller that ignores it will compare contracts against
        thresholds tuned on constituent share volume and be wrong by ~1000x.
        """
        symbol = str(symbol or "").upper()
        now = time.time()
        result: Dict[str, Any] = {
            "symbol": symbol, "volume": 0.0, "source": "none", "unit": "shares",
            "comparable_to_thresholds": False, "fyers_online": False,
            "futures": {}, "confirmation": "unavailable",
            "volume_available": False,
            "volume_source": "none",
            "volume_basis": "unavailable",
            "volume_unit": "shares",
            "volume_comparable": False,
            "volume_unavailable_reason": "no_positive_cash_index_volume",
        }

        if getattr(self, "upstox_only", False):
            self._watch_upstox_futures()
            result["futures"] = self.get_index_futures_minute_volume_context(symbol)
            return result

        cached = (self._ws_cache or {}).get(FYERS_INDEX_SYMBOLS.get(symbol, ""), None)
        fyers_volume, fyers_age = 0.0, None
        if isinstance(cached, dict):
            fyers_volume = float(cached.get("vol", 0.0) or 0.0)
            # The price timestamp is intentionally NOT a fallback. A volume age is
            # meaningful only when a positive cumulative total was observed.
            stamped = float(cached.get("vol_time", 0.0) or 0.0)
            if fyers_volume > 0.0 and stamped > 0.0:
                fyers_age = max(0.0, now - stamped)
        fyers_online = bool(
            fyers_volume > 0.0
            and fyers_age is not None
            and fyers_age <= self.FYERS_VOLUME_STALE_SECONDS
        )
        if not fyers_online:
            # Poll before deciding, so a recoverable feed is not written off for a
            # whole cycle. Cheap, throttled, and it refreshes the same cache the
            # websocket writes — so the very next read sees the recovery.
            probed = self._probe_fyers_volume(symbol, now)
            if probed > 0:
                fyers_volume, fyers_age, fyers_online = probed, 0.0, True
        result["fyers_online"] = fyers_online
        result["fyers_volume_age_seconds"] = round(fyers_age, 1) if fyers_age is not None else None

        futures = {}
        source = getattr(self, "kite", None)
        if source is not None and source.is_connected:
            try:
                futures = source.index_futures_activity(symbol) or {}
            except Exception as exc:
                logger.debug(f"Futures activity lookup failed for {symbol}: {exc}")
        if futures.get("available"):
            exchange_at = self._exchange_timestamp_from_payload(futures)
            # The exchange may update while the REST request is in flight.
            received_at = time.time()
            if exchange_at is None or not 0 <= received_at - exchange_at <= 90:
                futures = {**futures, "available": False, "reason": "stale_or_missing_exchange_timestamp"}
        result["futures"] = futures

        if fyers_online:
            result.update({
                "volume": fyers_volume, "source": "fyers_index", "unit": "shares",
                "comparable_to_thresholds": True,
                "volume_available": True,
                "volume_source": "fyers_index",
                "volume_basis": "cumulative_cash_index_volume",
                "volume_unit": "shares",
                "volume_comparable": True,
                "volume_unavailable_reason": "",
            })
            self._note_volume_source(symbol, "fyers_index")
            if futures.get("available"):
                # Both present: the broker confirms or contradicts direction of
                # participation. Levels are not comparable, so only presence is used.
                result["confirmation"] = (
                    "broker_confirms" if float(futures.get("volume") or 0.0) > 0
                    else "broker_quiet"
                )
            return result

        # Fyers is down or stale. Fall back to the broker so volume-aware logic has
        # SOMETHING real rather than a silent zero — but say loudly that the unit
        # changed, because every calibrated threshold is now meaningless.
        if futures.get("available"):
            result.update({
                "volume": float(futures.get("volume") or 0.0),
                "oi": float(futures.get("oi") or 0.0),
                "source": "kite_index_futures", "unit": str(futures.get("unit") or "broker_reported_quantity"),
                "comparable_to_thresholds": False,
                "confirmation": "fallback_only",
                "volume_available": True,
                "volume_source": "kite_index_futures",
                "volume_basis": "near_month_futures_confirmation",
                "volume_unit": str(futures.get("unit") or "broker_reported_quantity"),
                "volume_comparable": False,
                "volume_unavailable_reason": (
                    "cash_index_volume_unavailable; futures_confirmation_only"
                ),
            })
            self._note_volume_source(symbol, "kite_index_futures")
            self._warn_volume_fallback(symbol, futures)
            return result

        result["confirmation"] = "no_volume_source"
        return result

    def _probe_fyers_volume(self, symbol: str, now: float) -> float:
        """Throttled REST poll to detect Fyers coming back. Returns volume, or 0.0.

        Recovery has to be noticed in seconds, not on whatever cadence the websocket
        happens to reconnect at — the fallback is a different quantity, so every extra
        second on it is another bar the volume series cannot use.
        """
        if self._replacement_enabled():
            # In this route FYERS cash-share volume comes from completed history,
            # not index quotes. Four futile quote probes every 3s consume 80/min,
            # exceeding the Standard 50/min budget and starving real cash history.
            # Existing positive WebSocket volume and price fallback remain usable.
            return 0.0
        if not hasattr(self, "_fyers_probe_at"):
            self._fyers_probe_at = {}
        if now - float(self._fyers_probe_at.get(symbol, 0.0) or 0.0) < self.FYERS_RECOVERY_PROBE_SECONDS:
            return 0.0
        self._fyers_probe_at[symbol] = now
        fyers_symbol = FYERS_INDEX_SYMBOLS.get(symbol)
        if not fyers_symbol or not getattr(self, "fyers", None):
            return 0.0
        try:
            ok, response = self._call_fyers_bounded(
                "quotes",
                {"symbols": fyers_symbol},
                label=f"Fyers volume probe {fyers_symbol}",
            )
            if not ok:
                return 0.0
            if not response or response.get("s") != "ok":
                return 0.0
            values = (response.get("d") or [{}])[0].get("v", {}) or {}
            volume = float(values.get("volume") or values.get("vol_traded_today") or 0.0)
        except Exception as exc:
            logger.debug(f"Fyers volume recovery probe failed for {symbol}: {exc}")
            return 0.0
        if volume <= 0:
            return 0.0
        # Write it back into the same cache the socket feeds, so every other consumer
        # sees the recovery at once rather than each rediscovering it.
        cached = dict(self._ws_cache.get(fyers_symbol) or {})
        # Refresh the VOLUME timestamp only. Stamping time=now while leaving `ltp` at
        # whatever the dead websocket last published makes a frozen price look freshly
        # ticked, and _select_index_ws_quote then serves it as the live index spot for
        # signals and strike selection instead of falling through to the REST tier.
        # This probe knows about volume; it must not vouch for the price.
        try:
            probed_price = float(values.get("lp") or values.get("ltp") or 0.0)
        except (TypeError, ValueError):
            probed_price = 0.0
        cached.update({"vol": volume, "vol_time": now, "source": "fyers_rest_probe"})
        if probed_price and probed_price > 0:
            cached.update({"ltp": probed_price, "time": now})
        self._ws_cache[fyers_symbol] = cached
        if str((getattr(self, "_volume_source_active", {}) or {}).get(symbol) or "") == "kite_index_futures":
            logger.info(
                f"INDEX VOLUME RECOVERED: {symbol} Fyers cash volume is back "
                f"({volume:,.0f}); switching off the futures fallback."
            )
        return volume

    def _note_volume_source(self, symbol: str, source: str) -> None:
        """Record when an index's volume series changes provenance.

        Volume gates work on a RATIO (recent bars / prior bars), which survives a unit
        change perfectly well — as long as the whole window is one unit. A window that
        STRADDLES the switch is the hazard: 3 recent bars of futures contracts over 17
        prior bars of constituent shares gives ~0.006 and reads as THIN, and the
        reverse gives ~175 and reads as a SURGE. Both are fabrications of the join, not
        of the market.
        """
        if not hasattr(self, "_volume_source_changed_at"):
            self._volume_source_changed_at = {}
        if not hasattr(self, "_volume_source_active"):
            self._volume_source_active = {}
        previous = self._volume_source_active.get(symbol)
        if previous == source:
            return
        self._volume_source_active[symbol] = source
        if previous is not None:
            self._volume_source_changed_at[symbol] = time.time()
            logger.warning(
                f"VOLUME SOURCE CHANGED for {symbol}: {previous} -> {source}. "
                "Volume ratios are suppressed until the lookback window is one source."
            )

    def volume_series_is_homogeneous(
        self, symbol: str, timeframe_minutes: float = 5.0, bars: int = 20
    ) -> bool:
        """False while a volume lookback window still spans two different sources."""
        changed_at = float(
            (getattr(self, "_volume_source_changed_at", {}) or {}).get(symbol, 0.0) or 0.0
        )
        if changed_at <= 0:
            return True
        window_seconds = max(1.0, float(timeframe_minutes) * 60.0 * float(bars))
        return (time.time() - changed_at) >= window_seconds

    def _warn_volume_fallback(self, symbol: str, futures: Dict[str, Any]) -> None:
        """Announce the substitution once per 10 minutes per index."""
        if not hasattr(self, "_volume_fallback_alerted"):
            self._volume_fallback_alerted = {}
        now = time.time()
        if now - float(self._volume_fallback_alerted.get(symbol, 0.0) or 0.0) < 600.0:
            return
        self._volume_fallback_alerted[symbol] = now
        unit = str(futures.get("unit") or "broker_reported_quantity")
        unit_label = "broker-reported units" if unit == "broker_reported_quantity" else unit
        logger.warning(
            f"INDEX VOLUME FALLBACK: {symbol} live cumulative cash volume unavailable; using "
            f"{futures.get('tradingsymbol')} futures volume "
            f"({futures.get('volume'):,.0f} {unit_label}, OI {futures.get('oi'):,.0f}) "
            "for confirmation only. Cash-volume gates require independently verified cash bars; the separate "
            "futures-minute proxy uses its own same-unit baseline."
        )

    def _kite_token_for_index(self, symbol: str) -> str:
        """Kite instrument_token for an index spot, as the tick cache keys it."""
        wanted = str(symbol or "").upper()
        for token, name in (getattr(self, "_kite_index_names", {}) or {}).items():
            if name == wanted:
                return str(token)
        return ""

    @staticmethod
    def _cache_epoch(value: Any) -> float:
        if isinstance(value, datetime):
            return value.timestamp()
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _record_contract_tick(
        self,
        token: str,
        price: float,
        received_at: float,
        exchange_at: Optional[float] = None,
    ) -> int:
        """Retain one exact derivative tick for ordered, non-blocking trade checks."""
        token = str(token or "").strip()
        try:
            price = float(price or 0.0)
            received_at = float(received_at or 0.0)
        except (TypeError, ValueError):
            return 0
        if not token or price <= 0 or received_at <= 0:
            return 0

        # Lazy fields keep __new__-based diagnostics/tests compatible without weakening
        # production initialisation above.
        if not hasattr(self, "_contract_tick_history_lock"):
            self._contract_tick_history_lock = threading.Lock()
        with self._contract_tick_history_lock:
            history = getattr(self, "_contract_tick_history", None)
            if not isinstance(history, dict):
                history = {}
                self._contract_tick_history = history
            sequences = getattr(self, "_contract_tick_sequences", None)
            if not isinstance(sequences, dict):
                sequences = {}
                self._contract_tick_sequences = sequences

            active = getattr(self, "_active_contract_tick_tokens", set())
            size = max(16, int(
                getattr(self, "_active_contract_tick_history_size", 32768)
                if token in active
                else getattr(self, "_contract_tick_history_size", 2048)
            ))
            token_limit = max(
                8,
                int(getattr(self, "_contract_tick_history_token_limit", 128) or 128),
            )
            queue = history.get(token)
            if queue is None:
                if len(history) >= token_limit:
                    # Evict least-recently-ticked contract. Frequently ticking active
                    # positions stay resident; stale candidate subscriptions do not grow
                    # memory for the full session.
                    eviction_pool = [key for key in history if key not in active] or list(history)
                    oldest_token = min(
                        eviction_pool,
                        key=lambda key: float(history[key][-1][2]) if history[key] else 0.0,
                    )
                    history.pop(oldest_token, None)
                queue = deque(maxlen=size)
                history[token] = queue
            elif queue.maxlen != size:
                queue = deque(queue, maxlen=size)
                history[token] = queue

            sequence = int(sequences.get(token, 0) or 0) + 1
            sequences[token] = sequence
            try:
                exchange_epoch = self._cache_epoch(exchange_at)
            except Exception:
                exchange_epoch = 0.0
            queue.append((sequence, price, received_at, exchange_epoch))
            return sequence

    def set_active_contract_tick_tokens(self, tokens) -> None:
        """Reserve a longer tick tail for live positions without enlarging all symbols."""
        active = {str(token).strip() for token in tokens if str(token).strip()}
        lock = getattr(self, "_contract_tick_history_lock", None)
        if lock is None:
            self._contract_tick_history_lock = lock = threading.Lock()
        with lock:
            self._active_contract_tick_tokens = active
            history = getattr(self, "_contract_tick_history", {})
            for token, queue in list(history.items()):
                size = (
                    getattr(self, "_active_contract_tick_history_size", 32768)
                    if token in active else getattr(self, "_contract_tick_history_size", 2048)
                )
                size = max(16, int(size))
                if queue.maxlen != size:
                    history[token] = deque(queue, maxlen=size)

    def get_contract_ticks_since(
        self,
        token: str,
        after_sequence: int = 0,
        not_before: float = 0.0,
    ) -> Tuple[List[Tuple[int, float, float, float]], int, bool]:
        """Copy ordered derivative ticks newer than one consumer's cursor.

        Returns (ticks, newest_sequence, overflowed). Copy happens under a short memory
        lock; Scanner evaluates trades after release, so Kite's socket never waits on
        strategy, persistence, notification, or broker work.
        """
        token = str(token or "").strip()
        try:
            after_sequence = max(0, int(after_sequence or 0))
            not_before = max(0.0, float(not_before or 0.0))
        except (TypeError, ValueError):
            after_sequence, not_before = 0, 0.0
        lock = getattr(self, "_contract_tick_history_lock", None)
        history = getattr(self, "_contract_tick_history", None)
        if not token or lock is None or not isinstance(history, dict):
            return [], after_sequence, False
        with lock:
            queue = history.get(token)
            if not queue:
                return [], after_sequence, False
            first_sequence = int(queue[0][0])
            newest_sequence = int(queue[-1][0])
            if newest_sequence <= after_sequence:
                # The 100ms monitor usually already consumed this contract's tail.
                # Avoid scanning up to 32768 retained ticks while holding the socket
                # writer's lock when there is no new observation to evaluate.
                return [], newest_sequence, False
            overflowed = bool(
                after_sequence > 0 and first_sequence > after_sequence + 1
            )
            # The writer allocates each sequence and appends under this same lock.
            # Walk only the unread suffix, then restore its original arrival order.
            # Receipt timestamps can arrive out of order, so only sequence can end
            # the walk; not_before still filters every unread observation.
            ticks = []
            for tick in reversed(queue):
                if int(tick[0]) <= after_sequence:
                    break
                if float(tick[2]) >= not_before:
                    ticks.append(tick)
            ticks.reverse()
            return ticks, newest_sequence, overflowed

    def get_cached_contract_ltp(
        self,
        exchange: str,
        symbol: str,
        token: str,
        max_age_seconds: float = 10.0,
    ) -> Optional[float]:
        """Return a fresh contract mark without starting any REST request."""
        del exchange  # Cache keys are broker token/symbol based.
        now_ts = time.time()
        max_age = max(0.0, float(max_age_seconds or 0.0))
        token = str(token or "")
        symbol = str(symbol or "")
        short_token = token[3:] if token.startswith("999") else token
        for cache, keys in (
            (getattr(self, "_ws_cache", {}) or {}, (token, short_token, symbol)),
            (getattr(self, "_ltp_cache", {}) or {}, (symbol, token, short_token)),
        ):
            for key in dict.fromkeys(key for key in keys if key):
                payload = cache.get(key)
                if not isinstance(payload, dict):
                    continue
                price = float(payload.get("ltp", 0.0) or 0.0)
                upstox = getattr(self, "upstox_only", False)
                if upstox and payload.get("source") not in {"upstox_ws", "upstox_rest"}:
                    continue
                cache_ts = self._cache_epoch(payload.get("last_trade_at") if upstox else payload.get("time"))
                age = now_ts - cache_ts if cache_ts > 0 else float("inf")
                if price > 0 and 0.0 <= age <= max_age:
                    self._last_ltp_time = cache_ts
                    return price
        return None
    
    @retry_on_exception(retries=1, delay=0.1)
    def get_latest_price_by_token(self, token: str, symbol: str, exchange: str = "NSE") -> float:
        """The Master Source for Index Spot Prices — Weekend Precision Fix"""
        def _remember_ltp_time(val: Any) -> float:
            if isinstance(val, dict):
                timing = self._quote_age_components(val)
                self._last_ltp_time = float(timing.get("received_at") or time.time())
                return float(val.get('ltp', 0.0) or 0.0)
            self._last_ltp_time = time.time()
            return val

        if getattr(self, "upstox_only", False):
            from data.upstox_market_data import UPSTOX_INDEX_KEYS
            key = str(token or "")
            if not key.startswith(("NSE_FO|", "BSE_FO|")):
                key = UPSTOX_INDEX_KEYS.get(symbol, key)
            quote = self._upstox_quote(key if "|" in key else "", 2.0)
            max_age = 10.0 if self.is_market_open() else 86400.0
            mark_time = (quote.get("last_trade_at") if key.startswith(("NSE_FO|", "BSE_FO|"))
                         else quote.get("time"))
            if mark_time and -1 <= time.time() - mark_time <= max_age:
                self._last_ltp_time = mark_time
                return quote["ltp"]
            return 0.0

        if symbol == "INDIAVIX":
            upstox_client = getattr(self, "upstox", None)
            if upstox_client is not None:
                try:
                    vix_val = float(upstox_client.get_index_ltp("INDIAVIX") or 0.0)
                    if vix_val > 0:
                        return _remember_ltp_time(vix_val)
                except Exception:
                    pass
            if self._replacement_enabled():
                auxiliary = self._freshest_cached_index_quote(("angel:INDIAVIX",),
                    20.0 if self.is_market_open() else 86400.0, time.time())
                if auxiliary is not None:
                    return _remember_ltp_time(auxiliary[1])
                auxiliary_quote = self._replacement_auxiliary_quote()
                if auxiliary_quote is not None:
                    with self._index_quote_guard():
                        self._ws_cache["angel:INDIAVIX"] = auxiliary_quote
                    return _remember_ltp_time(auxiliary_quote)

        if symbol in INDEX_SPOT_TOKENS:
            now_ts = time.time()
            cached, _age = self._select_index_ws_quote(symbol, token, now_ts)
            if cached is not None:
                return _remember_ltp_time(cached)

        # Non-index contracts may not have redundant source metadata. Index
        # prices already passed the freshness gate above and must fall through
        # to REST/Yahoo when every websocket tick is stale.
        if symbol not in INDEX_SPOT_TOKENS:
            now_ts = time.time()

            def _is_valid_cache(cache_obj: Any) -> bool:
                if not isinstance(cache_obj, dict): return False
                age = now_ts - float(cache_obj.get("time") or 0.0)
                return age <= 29.0

            if token in self._ws_cache and _is_valid_cache(self._ws_cache[token]):
                return _remember_ltp_time(self._ws_cache[token])

            short_token = token[3:] if token.startswith('999') else token
            if short_token in self._ws_cache and _is_valid_cache(self._ws_cache[short_token]):
                return _remember_ltp_time(self._ws_cache[short_token])

            fyers_symbol = fyers_quote_symbol(symbol) or symbol
            if fyers_symbol in self._ws_cache and _is_valid_cache(self._ws_cache[fyers_symbol]):
                return _remember_ltp_time(self._ws_cache[fyers_symbol])
            
        # 2. TIER 1 — THE EXECUTION BROKER (Zerodha Kite).
        # The broker's price is the one orders actually fill against, so it outranks
        # every other feed. Fyers used to be consulted first and on 2026-08-01 it
        # returned NIFTY at 25,965.35 against a true 24,383.60 — a 6.49% error that
        # 2. TIER 1 — THE PRIMARY BROKER (Upstox API v2).
        upstox_ltp = 0.0
        upstox_received_at = 0.0
        upstox_client = getattr(self, "upstox", None)
        if upstox_client is not None and (upstox_client.is_healthy(threshold_seconds=30.0) or upstox_client.failure_duration() < 30.0):
            try:
                ltp = float(upstox_client.get_index_ltp(symbol) or 0.0)
                if ltp > 0:
                    upstox_ltp = ltp
                    upstox_received_at = time.time()
            except Exception as exc:
                logger.debug(f"Upstox REST quote failed for {symbol}: {exc}")

        if upstox_ltp > 0:
            received_at = upstox_received_at or time.time()
            self._last_ltp_time = received_at
            cache_entry = {
                "ltp": upstox_ltp,
                "time": datetime.fromtimestamp(received_at),
                "received_at": received_at,
                "exchange_at": None,
                "source": "upstox",
            }
            with self._index_quote_guard():
                self._ltp_cache[symbol] = cache_entry
            return upstox_ltp

        # If Upstox is down >30s or returns 0, fall through to Angel One fallback below

        # 3. TIER 3 — ANGELONE SMARTAPI INDEX QUOTE.
        # This replaces Yahoo as the normal broker fallback. A short local cache keeps
        # the 5 Hz background LTP worker well below AngelOne request limits while still
        # bounding a SENSEX badge gap to a couple of seconds.
        angel = getattr(self, "angel_index", None)
        if symbol in INDEX_SPOT_TOKENS and angel is not None:
            with self._index_quote_guard():
                angel_cached = dict((self._ltp_cache or {}).get(symbol) or {})
            angel_source = str(angel_cached.get("source") or "").lower()
            angel_cached_at = self._cache_epoch(
                angel_cached.get("received_at") or angel_cached.get("time")
            )
            if (
                angel_source.startswith("angel")
                and float(angel_cached.get("ltp") or 0.0) > 0
                and 0.0 <= time.time() - angel_cached_at <= 2.0
            ):
                return _remember_ltp_time(angel_cached)
            try:
                angel_quote = angel.get_index_quote(symbol)
            except Exception:
                angel_quote = None
            if isinstance(angel_quote, dict):
                angel_ltp = float(angel_quote.get("ltp") or 0.0)
                if angel_ltp > 0:
                    received_at = float(angel_quote.get("received_at") or time.time())
                    previous_close = float(angel_quote.get("previous_close") or 0.0)
                    if previous_close > 0:
                        self._remember_previous_close(
                            symbol, previous_close, "angel_rest"
                        )
                    cache_entry = {
                        **angel_quote,
                        "time": datetime.fromtimestamp(received_at),
                        "received_at": received_at,
                        "source": "angel_rest",
                    }
                    with self._index_quote_guard():
                        self._ltp_cache[symbol] = cache_entry
                    self._last_ltp_time = received_at
                    return angel_ltp

        # 3b. Legacy AngelOne REST. Retired: smart_api is always None now, so this is
        # inert. Left in place rather than excised mid-migration so the surrounding
        # cache/backoff machinery keeps its shape until the file is cleaned up.
        if self.is_connected and self.smart_api:
            if not self._throttle_angel_rest("ltpData"):
                return self._ltp_cache.get(symbol, {}).get('ltp', 0.0)
            try:
                # Double-check cache after acquiring the single-flight lease.
                with self._rest_lock:
                    cached = self._ltp_cache.get(symbol)
                    cached_time = cached.get('time', datetime.min) if isinstance(cached, dict) else datetime.min
                    if isinstance(cached_time, datetime):
                        cached_age = (datetime.now() - cached_time).total_seconds()
                    elif isinstance(cached_time, (int, float)):
                        cached_age = time.time() - float(cached_time)
                    else:
                        cached_age = float("inf")
                    if isinstance(cached, dict) and 0.0 <= cached_age < 5.0:
                        return cached['ltp']

                try:
                    trading_symbol = "INDEX" if symbol in INDEX_SPOT_TOKENS else symbol
                    resp = self.smart_api.ltpData(exchange, trading_symbol, token)
                    if not isinstance(resp, dict):
                        raise RuntimeError(f"Angel REST returned invalid {type(resp).__name__} response")

                    # Check for Invalid Token error
                    if not resp.get('status') and resp.get('errorCode') == 'AG8001':
                        logger.warning("🔄 AngelOne Session Expired (AG8001). Re-authenticating...")
                        if self.connect():
                            resp = self.smart_api.ltpData(exchange, trading_symbol, token)
                        else:
                            return self._ltp_cache.get(symbol, {}).get('ltp', 0.0)

                    if isinstance(resp, dict) and resp.get('status'):
                        self._mark_angel_rest_success()
                    elif isinstance(resp, dict):
                        if self._angel_rest_should_backoff(resp):
                            self._mark_angel_rest_backoff("ltpData", resp)
                        else:
                            logger.warning(f"Angel REST failure for {symbol}: {resp}")

                    if isinstance(resp, dict) and resp.get('status') and resp.get('data'):
                        ltp = float(resp['data'].get('ltp', 0.0))
                        close_px = float(resp['data'].get('close', 0.0))

                        if close_px > 0:
                            self._remember_previous_close(symbol, close_px, "angel_rest")

                        if ltp > 0:
                            received_at = time.time()
                            self._last_ltp_time = received_at
                            cache_entry = {
                                'ltp': ltp,
                                'time': datetime.fromtimestamp(received_at),
                                'received_at': received_at,
                                'exchange_at': None,
                                'source': 'angel',
                            }
                            with self._index_quote_guard():
                                self._ltp_cache[symbol] = cache_entry
                            return ltp
                except Exception as e:
                    if "Invalid Token" in str(e) or "AG8001" in str(e):
                        logger.warning("🔄 AngelOne Exception: Invalid Token. Re-authenticating...")
                        self.connect()
                    else:
                        if self._angel_rest_should_backoff(e):
                            self._mark_angel_rest_backoff("ltpData", e)
                        logger.debug(f"Angel REST LTP fetch failed for {symbol}: {e}")
            finally:
                self._release_angel_rest_quote()
        
        # 4. Optional delayed reference only. AngelOne is now the normal third source;
        # Yahoo stays disabled unless the operator explicitly accepts delayed LTP.
        ticker = YAHOO_TICKERS.get(symbol)
        try:
            from config.settings import get_settings

            yahoo_live_enabled = bool(
                getattr(get_settings(), "yahoo_live_price_fallback_enabled", False)
            )
        except Exception:
            yahoo_live_enabled = False
        if ticker and yahoo_live_enabled:
            try:
                self._yahoo_request_count += 1
                y_ticker = yf.Ticker(ticker)
                y_data = y_ticker.fast_info
                y_ltp = y_data.last_price
                if y_ltp > 0:
                    logger.warning(f"⚠️ EMERGENCY: Using 15min delayed Yahoo data for {symbol}")
                    received_at = time.time()
                    self._last_ltp_time = received_at
                    self._yahoo_last_used_at = self._last_ltp_time
                    cache_entry = {
                        'ltp': y_ltp,
                        'time': datetime.fromtimestamp(received_at),
                        'received_at': received_at,
                        'exchange_at': None,
                        'source': 'yahoo',
                    }
                    with self._index_quote_guard():
                        self._ltp_cache[symbol] = cache_entry
                    return y_ltp
            except Exception as exc:
                logger.debug(f"Yahoo fast_info fallback failed for {symbol}: {exc}")

        # 5. INTERNAL CACHE FALLBACK
        with self._index_quote_guard():
            cache_entry = (getattr(self, "_ltp_cache", {}) or {}).get(symbol, {})
            cache_time = cache_entry.get('time')
            if hasattr(cache_time, 'timestamp'):
                self._last_ltp_time = cache_time.timestamp()
            elif isinstance(cache_time, (int, float)):
                self._last_ltp_time = float(cache_time)
            elif 'received_at' in cache_entry:
                self._last_ltp_time = float(cache_entry['received_at'])
            return cache_entry.get('ltp', 0.0)

    def _get_preopen_fyers_index_quote(
        self,
        symbol: str,
        fyers_symbol: str,
        now_ts: float,
        ttl_seconds: float = 5.0,
    ) -> Optional[Dict[str, Any]]:
        """Use Fyers REST as the pre-open index source.

        AngelOne index WS can emit fresh timestamps with reference/previous
        close values during pre-open. Fyers REST is slower, so this path is
        short-cached and only used before the regular session opens.
        """
        if not getattr(self, "fyers", None) or not fyers_symbol:
            return None

        cache = getattr(self, "_preopen_index_quote_cache", None)
        if cache is None:
            cache = {}
            self._preopen_index_quote_cache = cache

        cached = cache.get(symbol)
        if isinstance(cached, dict):
            age = now_ts - float(cached.get("time") or 0.0)
            if 0 <= age <= ttl_seconds and isinstance(cached.get("quote"), dict):
                quote = dict(cached["quote"])
                quote.update(self._quote_timing_metadata(quote, now_ts))
                return quote

        try:
            values = self._fyers_index_quote_values(symbol)
            ltp = float(values.get("lp", 0.0) or 0.0)
            if ltp > 0:
                received_at = time.time()
                exchange_at = self._exchange_timestamp_from_payload(values)
                previous_close = self._quote_float(values, *PREVIOUS_CLOSE_KEYS)
                if previous_close > 0:
                    self._remember_previous_close(symbol, previous_close, "fyers_rest")
                quote = {
                    "ltp": ltp,
                    "change": float(values.get("ch", 0.0) or 0.0),
                    "change_pct": float(values.get("chp", 0.0) or 0.0),
                    "previous_close": previous_close,
                    "prev_close": previous_close,
                    "source": "fyers_rest_preopen",
                    "received_at": received_at,
                    "exchange_at": exchange_at,
                }
                quote.update(self._quote_timing_metadata(quote, received_at))
                cache[symbol] = {"time": received_at, "quote": quote}
                self._last_ltp_time = received_at
                ltp_cache_entry = {
                    "ltp": ltp,
                    "change": quote["change"],
                    "change_pct": quote["change_pct"],
                    "previous_close": previous_close,
                    "time": datetime.fromtimestamp(received_at),
                    "received_at": received_at,
                    "exchange_at": exchange_at,
                    "source": "fyers_rest_preopen",
                }
                with self._index_quote_guard():
                    self._ltp_cache[symbol] = ltp_cache_entry
                return dict(quote)
        except Exception as exc:
            logger.debug(f"Pre-open Fyers quote fetch failed for {symbol}: {exc}")
        return None

    def get_cached_index_quote_snapshot(
        self,
        symbol: str,
        now_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return the selected index quote from memory without broker/network I/O.

        This is intentionally safe for a fast independent dashboard lane.  It never
        calls ``get_index_quote``, ``get_previous_close`` or a broker adapter.  The
        websocket callbacks and this reader share ``_index_quote_lock``, so source,
        price and timing metadata describe one coherent cache snapshot.
        """
        name = str(symbol or "").upper()
        now = time.time() if now_ts is None else float(now_ts)
        now_dt = datetime.fromtimestamp(now)

        def _cached_float(value: Any) -> float:
            try:
                return float(value or 0.0)
            except (TypeError, ValueError):
                return 0.0

        try:
            session_label = self.get_market_session_label(now_dt)
        except Exception:
            session_label = "UNKNOWN"

        base: Dict[str, Any] = {
            "symbol": name,
            "ltp": 0.0,
            "spot_price": 0.0,
            "change": 0.0,
            "change_points": 0.0,
            "change_pct": 0.0,
            "previous_close": 0.0,
            "prev_close": 0.0,
            "source": "unavailable",
            "quote_source": "unavailable",
            "quote_label": session_label,
            "quote_session": session_label,
            "session": session_label,
            "is_preopen": session_label == "PRE-OPEN",
            "is_fresh": False,
            "is_stale": False,
            "quote_fresh": False,
            "quote_stale": False,
            "freshness": "unavailable",
            "cached_only": True,
            "received_at": None,
            "received_at_epoch": None,
            "exchange_at": None,
            "receipt_timestamp": None,
            "exchange_timestamp": None,
            "exchange_timestamp_epoch": None,
            "quote_timestamp": None,
            "receipt_age_seconds": None,
            "exchange_age_seconds": None,
            "quote_age_seconds": None,
            "freshness_basis": "none",
            "source_role": "unavailable",
            "selected_price_source": "unavailable",
            "candle_price_source": "",
        }
        if name not in INDEX_SPOT_TOKENS:
            return base

        if getattr(self, "upstox_only", False):
            from data.upstox_market_data import UPSTOX_INDEX_KEYS
            key = UPSTOX_INDEX_KEYS.get(name)
            rows = [row for row in (self._ws_cache.get(key), self._ltp_cache.get(key))
                    if row and row["time"] <= now]
            if not rows:
                return base
            row = max(rows, key=lambda item: item["time"])
            price, previous = row["ltp"], float(row.get("previous_close") or 0)
            change = price - previous if previous > 0 else 0
            age = now - row["time"]
            fresh = age <= 5 and session_label == "LIVE"
            source = row["source"]
            base.update(ltp=price, spot_price=price, change=change, change_points=change,
                        change_pct=100 * change / previous if previous > 0 else 0,
                        previous_close=previous, prev_close=previous, source=source, quote_source=source,
                        is_fresh=fresh, is_stale=not fresh, quote_fresh=fresh, quote_stale=not fresh,
                        freshness="fresh" if fresh else "stale", source_role="primary",
                        selected_price_source=source, candle_price_source="upstox_ws",
                        quote_age_seconds=age, received_at=row["received_at"],
                        received_at_epoch=row["received_at"], exchange_at=row["time"],
                        exchange_timestamp_epoch=row["time"], freshness_basis="provider")
            return base

        token = INDEX_SPOT_TOKENS.get(name, ("", ""))[0]
        broker_keys = self._index_broker_cache_keys(name, token)
        fyers_keys = (FYERS_INDEX_SYMBOLS.get(name, ""),)
        angel_keys = ("angel:" + name,) if self._replacement_enabled() else ()
        broker_window = (
            self.INDEX_KITE_PRICE_FRESH_SECONDS
            if session_label == "LIVE"
            else 86400.0
        )

        with self._index_quote_guard():
            angel = self._freshest_cached_index_quote(angel_keys, self.INDEX_FYERS_PRICE_FRESH_SECONDS, now)
            broker = self._freshest_cached_index_quote(broker_keys, broker_window, now)
            if broker is not None and not str(
                broker[1].get("source") or ""
            ).lower().startswith("kite"):
                broker = None
            fyers = self._freshest_cached_index_quote(
                fyers_keys, self.INDEX_FYERS_PRICE_FRESH_SECONDS, now
            )
            if fyers is not None and not str(
                fyers[1].get("source") or ""
            ).lower().startswith("fyers"):
                fyers = None
            rest_cached = dict(
                (getattr(self, "_ltp_cache", {}) or {}).get(name) or {}
            )
            route = dict(
                (getattr(self, "_index_price_route", {}) or {}).get(name) or {}
            )
            previous_close_cached = _cached_float(
                (getattr(self, "_prev_close_global_cache", {}) or {}).get(name, 0.0)
            )
            stale_broker = self._freshest_cached_index_quote(
                broker_keys, float("inf"), now
            )
            if stale_broker is not None and not str(
                stale_broker[1].get("source") or ""
            ).lower().startswith("kite"):
                stale_broker = None
            stale_fyers = self._freshest_cached_index_quote(
                fyers_keys, float("inf"), now
            )
            if stale_fyers is not None and not str(
                stale_fyers[1].get("source") or ""
            ).lower().startswith("fyers"):
                stale_fyers = None

        rest_ltp = _cached_float(rest_cached.get("ltp", 0.0))
        rest_source = str(rest_cached.get("source") or "").lower()
        rest_age = self._quote_age_components(rest_cached, now).get("age")
        rest_limit = (
            self.INDEX_KITE_PRICE_FRESH_SECONDS
            if rest_source.startswith("kite")
            else self.INDEX_FYERS_PRICE_FRESH_SECONDS
        )
        rest_is_live = bool(
            rest_ltp > 0
            and rest_age is not None
            and rest_age <= rest_limit
            and not rest_source.startswith("yahoo")
        )
        primary_rest_is_live = bool(rest_is_live and rest_source.startswith("kite"))
        fyers_rest_is_live = bool(rest_is_live and rest_source.startswith("fyers"))

        selected: Dict[str, Any] = {}
        # Price authority is broker-first across transports, not websocket-first across
        # brokers.  A fresh Kite REST observation is the execution broker's current LTP
        # and must beat a fresh Fyers websocket packet.  This also makes entry-geometry
        # rebasing deterministic when the two feeds differ materially.
        if broker is not None:
            selected = dict(broker[1])
        elif primary_rest_is_live:
            selected = rest_cached
        elif angel is not None:
            selected = dict(angel[1])
        elif fyers is not None:
            selected = dict(fyers[1])
        elif fyers_rest_is_live:
            selected = rest_cached
        elif rest_is_live:
            selected = rest_cached
        else:
            # No live source is available. Keep the deterministic trust order so a
            # delayed vendor packet cannot displace the last execution-broker mark.
            if stale_broker is not None:
                selected = dict(stale_broker[1])
            elif rest_ltp > 0 and rest_source.startswith("kite"):
                selected = rest_cached
            elif stale_fyers is not None:
                selected = dict(stale_fyers[1])
            elif rest_ltp > 0:
                selected = rest_cached

        try:
            ltp = float(selected.get("ltp", 0.0) or 0.0)
        except (TypeError, ValueError):
            ltp = 0.0
        if ltp <= 0:
            return base

        source = str(selected.get("source") or "cached")
        source_lower = source.lower()
        timing = self._quote_timing_metadata(selected, now)
        quote_age = timing.get("quote_age_seconds")
        fresh_limit = (
            self.INDEX_KITE_PRICE_FRESH_SECONDS
            if source_lower.startswith("kite")
            else self.INDEX_FYERS_PRICE_FRESH_SECONDS
        )
        is_fresh = bool(
            quote_age is not None
            and quote_age <= fresh_limit
            and not source_lower.startswith("yahoo")
        )
        previous_close = _cached_float(selected.get("previous_close", 0.0))
        if previous_close <= 0:
            previous_close = previous_close_cached
        change = _cached_float(selected.get("change", 0.0))
        change_pct = _cached_float(selected.get("change_pct", 0.0))
        if previous_close > 0:
            change = ltp - previous_close
            change_pct = change / previous_close * 100.0

        if source_lower.startswith("kite"):
            source_role = "primary"
        elif source_lower.startswith("angel") and self._fyers_backup_enabled():
            source_role = "supplementary"
        elif source_lower.startswith("fyers"):
            source_role = "backup" if self._fyers_backup_enabled() else "failover"
        elif source_lower.startswith("yahoo"):
            source_role = "delayed"
        else:
            source_role = "fallback"
        freshness = "fresh" if is_fresh else (
            "reference" if session_label == "CLOSED" else "stale"
        )
        base.update(
            {
                "ltp": ltp,
                "spot_price": ltp,
                "change": change,
                "change_points": change,
                "change_pct": change_pct,
                "previous_close": previous_close,
                "prev_close": previous_close,
                "source": source,
                "quote_source": source,
                "is_fresh": is_fresh,
                "is_stale": not is_fresh,
                "quote_fresh": is_fresh,
                "quote_stale": not is_fresh,
                "freshness": freshness,
                "source_role": source_role,
                "selected_price_source": source,
                "candle_price_source": str(route.get("active_source") or ""),
            }
        )
        base.update(timing)
        return base

    def get_index_quote(self, symbol: str) -> Dict[str, Any]:
        """Return index LTP plus broker-provided change fields when available."""
        now_dt = datetime.now()
        session_label = self.get_market_session_label(now_dt)
        quote = {
            "ltp": 0.0, 
            "change": 0.0, 
            "change_pct": 0.0, 
            "source": "unknown", 
            "quote_label": session_label,
            "session": session_label,
            "is_preopen": session_label == "PRE-OPEN"
        }
        if getattr(self, "upstox_only", False):
            from data.upstox_market_data import UPSTOX_INDEX_KEYS
            row = self._upstox_quote(UPSTOX_INDEX_KEYS.get(symbol), 2.0)
            if row:
                age = time.time() - row["time"]
                previous = float(row.get("previous_close") or 0.0)
                price = row["ltp"]
                change = price - previous if previous > 0 else 0.0
                fresh = -1 <= age <= 5
                quote.update(row, previous_close=previous, change=change,
                             change_pct=(100 * change / previous if previous > 0 else 0.0),
                             quote_age_seconds=max(0.0, age), is_fresh=fresh, is_stale=not fresh,
                             exchange_at=row["time"])
            return quote

        if symbol not in INDEX_SPOT_TOKENS:
            return quote

        now_ts = time.time()
        token, exchange = INDEX_SPOT_TOKENS.get(symbol, ("", "NSE"))
        fyers_symbol = FYERS_INDEX_SYMBOLS.get(symbol)

        preopen_rest_allowed = self._preopen_rest_quote_allowed(
            dt=now_dt,
            session_label=session_label,
        )
        quote["preopen_settlement"] = preopen_rest_allowed

        if preopen_rest_allowed and fyers_symbol and not self._replacement_enabled():
            preopen_quote = self._get_preopen_fyers_index_quote(symbol, fyers_symbol, now_ts)
            if preopen_quote:
                quote.update(preopen_quote)
                return quote

        cached, _quote_age = self._select_index_ws_quote(symbol, token, now_ts)
        if cached is not None:
            timing = self._quote_timing_metadata(cached, now_ts)
            previous_close = float(cached.get("previous_close", 0.0) or 0.0)
            if previous_close <= 0:
                previous_close = self.get_previous_close(symbol)
            source = str(cached.get("source") or "ws")
            freshness_limit = (
                self.INDEX_KITE_PRICE_FRESH_SECONDS
                if source.lower().startswith("kite")
                else self.INDEX_FYERS_PRICE_FRESH_SECONDS
            )
            effective_age = timing.get("quote_age_seconds")
            is_fresh = bool(
                effective_age is not None and effective_age <= freshness_limit
            )
            quote.update({
                "ltp": float(cached.get("ltp", 0.0) or 0.0),
                "change": float(cached.get("change", 0.0) or 0.0),
                "change_pct": float(cached.get("change_pct", 0.0) or 0.0),
                "previous_close": previous_close,
                "prev_close": previous_close,
                "source": source,
                "is_fresh": is_fresh,
                "is_stale": not is_fresh,
            })
            quote.update(timing)
            self._last_ltp_time = float(timing.get("received_at") or now_ts)
            # A live WebSocket LTP must never be replaced by a slower REST
            # quote merely because the WS payload omitted day-change fields.
            # The scanner uses LTP to build the forming candle and detect UT
            # crossovers, so waiting for/restoring an older REST value here
            # can delay an otherwise real-time intrabar signal.
            if previous_close > 0:
                if not quote["change"]:
                    quote["change"] = quote["ltp"] - previous_close
                if not quote["change_pct"]:
                    quote["change_pct"] = quote["change"] / previous_close * 100.0
            return quote

        if preopen_rest_allowed and fyers_symbol and self._replacement_enabled():
            preopen_quote = self._get_preopen_fyers_index_quote(symbol, fyers_symbol, now_ts)
            if preopen_quote:
                quote.update(preopen_quote)
                return quote

        # No websocket quote passed freshness checks. Resolve REST through the one
        # Kite-first bridge above; the old direct Fyers call here silently bypassed the
        # execution broker and made the dashboard's source order differ from trading.
        ltp = self.get_latest_price_by_token(token, symbol, exchange)
        resolved_at = time.time()
        with self._index_quote_guard():
            cached_source = dict(
                (getattr(self, "_ltp_cache", {}) or {}).get(symbol) or {}
            )
        # A custom/legacy resolver can return a live value without populating the cache.
        # Give that value a receipt clock; absence of cache metadata must not turn a
        # successful live response into a falsely stale badge.
        if float(ltp or 0.0) > 0 and not cached_source:
            cached_source = {
                "ltp": float(ltp),
                "time": datetime.fromtimestamp(resolved_at),
                "received_at": resolved_at,
                "source": "fallback",
            }
        timing = self._quote_timing_metadata(cached_source, time.time())
        quote["ltp"] = float(ltp or 0.0)
        quote["source"] = str(cached_source.get("source") or "fallback")
        quote.update(timing)
        cached_previous_close = float(cached_source.get("previous_close", 0.0) or 0.0)
        prev_close = cached_previous_close or self.get_previous_close(symbol)
        quote["previous_close"] = prev_close
        quote["prev_close"] = prev_close

        # A zero here used to be written straight into the panel. It is not a price: an
        # index does not trade at 0, so it can only mean "nothing answered just now" —
        # before the open, between reconnects, or while both REST paths are timing out.
        # The last known price was sitting in _ltp_cache the whole time and this function
        # read that dict for its `source` and `time` and ignored its `ltp`.
        #
        # Fall back to the last cached trade, then to the previous close, and mark the
        # quote STALE so nothing downstream mistakes it for a live tick. Only when
        # genuinely nothing has ever been seen does it stay 0, and the source says so.
        if quote["ltp"] <= 0:
            recovered = 0.0
            try:
                recovered = float(cached_source.get("ltp", 0.0) or 0.0)
            except (TypeError, ValueError):
                recovered = 0.0
            if recovered > 0:
                quote["ltp"] = recovered
                quote["source"] = f"{quote['source']}_stale"
            elif prev_close > 0:
                quote["ltp"] = prev_close
                quote["source"] = "previous_close"
                quote["quote_age_seconds"] = 0.0
            else:
                quote["source"] = "unavailable"
            quote["is_stale"] = quote["ltp"] > 0

        if quote["ltp"] > 0 and prev_close > 0:
            quote["change"] = quote["ltp"] - prev_close
            quote["change_pct"] = quote["change"] / prev_close * 100.0
        source_lower = str(quote.get("source") or "").lower()
        freshness_limit = (
            self.INDEX_KITE_PRICE_FRESH_SECONDS
            if source_lower.startswith("kite")
            else self.INDEX_FYERS_PRICE_FRESH_SECONDS
        )
        effective_age = quote.get("quote_age_seconds")
        is_fresh = bool(
            quote["ltp"] > 0
            and effective_age is not None
            and effective_age <= freshness_limit
            and not source_lower.startswith("yahoo")
            and source_lower not in {"previous_close", "unavailable"}
            and not source_lower.endswith("_stale")
        )
        quote["is_fresh"] = is_fresh
        quote["is_stale"] = bool(quote["ltp"] > 0 and not is_fresh)
        return quote

    def get_ltp(self, exchange: str, symbol: str, token: str) -> Optional[float]:
        """Unified LTP method used by entire system - Always fetches from AngelOne"""
        price = self.get_latest_price_by_token(token, symbol, exchange)
        return price if price > 0 else None

    @staticmethod
    def _first_positive(values: Dict[str, Any], keys: Tuple[str, ...]) -> float:
        for key in keys:
            try:
                value = float(values.get(key) or 0.0)
            except (TypeError, ValueError, AttributeError):
                value = 0.0
            if np.isfinite(value) and value > 0:
                return value
        return 0.0

    @staticmethod
    def _numeric_payload_evidence(
        values: Dict[str, Any], keys: Tuple[str, ...]
    ) -> Tuple[float, bool]:
        """Return (non-negative value, explicitly supplied).

        ``dict.get(...) or 0`` cannot distinguish a missing field from a broker
        reporting zero.  Liquidity admission needs that distinction: only the
        latter is evidence of an empty contract/book.
        """
        for key in keys:
            if key not in values or values.get(key) is None:
                continue
            try:
                value = float(values.get(key))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            return max(0.0, value), True
        return 0.0, False

    def _touch_quote_from_payload(
        self,
        payload: Dict[str, Any],
        source: str,
        now_ts: float,
        max_age_seconds: float,
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {"usable": False, "source": source, "reason": "invalid_payload"}
        quote_ts = payload.get("time") or payload.get("timestamp") or payload.get("last_update")
        try:
            quote_ts = float(quote_ts or now_ts)
        except (TypeError, ValueError):
            quote_ts = now_ts
        raw_age = (
            now_ts - quote_ts
            if np.isfinite(quote_ts) and np.isfinite(now_ts)
            else float("inf")
        )
        # A materially future-dated source clock is not fresh evidence. Allow only
        # one second of ordinary feed/host skew, matching the final policy's clock
        # tolerance; flooring every negative age to zero let arbitrary future packets
        # remain executable until the wall clock caught up.
        age = max(0.0, raw_age) if raw_age >= -1.0 else float("inf")
        bid = self._first_positive(
            payload,
            (
                "bid",
                "best_bid",
                "bid_price",
                "best_bid_price",
                "bp",
                "bidRate",
                "bidrate",
            ),
        )
        ask = self._first_positive(
            payload,
            (
                "ask",
                "best_ask",
                "ask_price",
                "best_ask_price",
                "ap",
                "askRate",
                "askrate",
                "offer",
                "offer_price",
            ),
        )
        bid_qty, bid_qty_available = self._numeric_payload_evidence(
            payload, ("bid_qty", "bid_size", "bidsize")
        )
        ask_qty, ask_qty_available = self._numeric_payload_evidence(
            payload, ("ask_qty", "ask_size", "asksize")
        )
        oi, oi_available = self._numeric_payload_evidence(
            payload, ("oi", "open_interest", "openInterest")
        )
        volume, volume_available = self._numeric_payload_evidence(
            payload, ("volume", "volume_traded", "vol", "v")
        )
        depth = payload.get("depth") or payload.get("market_depth") or payload.get("best5") or payload.get("bidask")
        if (bid <= 0 or ask <= 0) and depth:
            parsed_depth = self._parse_depth_payload(depth, source)
            bid = bid or float(parsed_depth.get("bid", 0.0) or 0.0)
            ask = ask or float(parsed_depth.get("ask", 0.0) or 0.0)
            if not bid_qty_available:
                bid_qty = float(parsed_depth.get("bid_qty", 0.0) or 0.0)
                bid_qty_available = "bid_qty" in parsed_depth
            if not ask_qty_available:
                ask_qty = float(parsed_depth.get("ask_qty", 0.0) or 0.0)
                ask_qty_available = "ask_qty" in parsed_depth
        result = {
            "bid": bid,
            "ask": ask,
            "ltp": float(payload.get("ltp") or payload.get("lp") or payload.get("last_traded_price") or payload.get("last_price") or 0.0),
            "bid_qty": int(bid_qty),
            "ask_qty": int(ask_qty),
            "bid_qty_available": bool(bid_qty_available),
            "ask_qty_available": bool(ask_qty_available),
            "depth_available": bool(bid_qty_available and ask_qty_available),
            "oi": oi,
            "volume": volume,
            "oi_available": bool(oi_available),
            "volume_available": bool(volume_available),
            "source": source,
            "age_seconds": age,
            "usable": False,
        }
        if bid <= 0 or ask <= 0:
            result["reason"] = "missing_bid_ask"
            return result
        if ask < bid:
            result["reason"] = "crossed_book"
            return result
        if payload.get("usable") is False:
            result["reason"] = str(payload.get("reason") or "provider_rejected_book")
            return result
        if age > max_age_seconds:
            result["reason"] = "stale"
            return result
        spread = ask - bid
        mid = (ask + bid) / 2.0
        result.update({
            "spread": spread,
            "spread_pct": (spread / mid * 100.0) if mid > 0 else 0.0,
            "usable": True,
            "reason": "ok",
        })
        return result

    def get_contract_touch_quote(
        self,
        exchange: str,
        symbol: str,
        token: str,
        max_age_seconds: float = 1.5,
    ) -> Dict[str, Any]:
        """Return best-effort executable touch quote for an option/future contract.

        This is deliberately separate from signal LTP. It is used only by the
        execution layer to choose or shadow a marketable-limit cap.
        """
        now_ts = time.time()
        if getattr(self, "upstox_only", False):
            key = str(token or "")
            if not key.startswith(("NSE_FO|", "BSE_FO|")):
                return {"usable": False, "reason": "unverified_upstox_instrument"}
            quote = self._upstox_quote(key, max_age_seconds)
            now_ts = time.time()
            return self._touch_quote_from_payload(quote, str(quote.get("source") or "upstox"),
                                                  now_ts, max_age_seconds)
        cache_key = f"{exchange}|{symbol}|{token}"
        cached = self._touch_quote_cache.get(cache_key)
        if isinstance(cached, dict):
            try:
                cached_at = float(cached.get("time"))
            except (TypeError, ValueError, OverflowError):
                cached_at = float("nan")
            cache_residence = (
                max(0.0, now_ts - cached_at)
                if np.isfinite(cached_at) and np.isfinite(now_ts)
                else float("inf")
            )
            if isinstance(cached.get("quote"), dict):
                quote = dict(cached["quote"])
                try:
                    source_age = float(quote.get("age_seconds"))
                except (TypeError, ValueError, OverflowError):
                    source_age = float("inf")
                if not np.isfinite(source_age) or source_age < 0:
                    source_age = float("inf")
                # Cache residence is additional age, not a replacement clock.  The
                # old code turned a 1.4-second source quote cached 0.1 seconds ago into
                # a 0.1-second quote and admitted it at a 1.5-second execution limit.
                age = source_age + cache_residence
                quote["age_seconds"] = age
                if 0 <= age <= max_age_seconds:
                    return quote

        keys = [token, str(token), symbol]
        if symbol in FYERS_INDEX_SYMBOLS:
            keys.append(FYERS_INDEX_SYMBOLS.get(symbol, ""))
        for key in keys:
            cached_tick = self._ws_cache.get(key) if key else None
            if not isinstance(cached_tick, dict):
                continue
            quote = self._touch_quote_from_payload(
                cached_tick,
                str(cached_tick.get("source") or "ws_touch"),
                now_ts,
                max_age_seconds,
            )
            if quote.get("usable"):
                self._touch_quote_cache[cache_key] = {"time": now_ts, "quote": quote}
                return quote

        # The broker that will execute the order is authoritative for its own book.
        # A synchronous REST read is acceptable here because callers invoke this only
        # at an execution boundary (scale-ins run on the execution queue), never on the
        # scanner's per-tick pricing path.
        kite_source = getattr(self, "kite", None)
        if (
            kite_source is not None
            and bool(getattr(kite_source, "is_connected", False))
            and callable(getattr(kite_source, "get_depth", None))
        ):
            try:
                depth_payload = kite_source.get_depth(exchange, symbol)
                quote = self._touch_quote_from_payload(
                    depth_payload,
                    "kite_rest_touch",
                    now_ts,
                    max_age_seconds,
                )
                self._touch_quote_cache[cache_key] = {"time": now_ts, "quote": quote}
                # A definitive Kite book is also the safest refusal.  Do not replace
                # an empty/crossed/wide execution-broker book with another venue's
                # superficially better price and then send the order to Kite anyway.
                return quote
            except Exception as exc:
                logger.debug(f"Kite touch quote fetch failed for {symbol}: {exc}")

        if self._replacement_enabled():
            try:
                quote = self._replacement_touch(exchange, symbol, now_ts, max_age_seconds)
            except Exception:
                quote = None
            if quote is not None and (quote.get("usable") or not self._fyers_backup_enabled()):
                self._touch_quote_cache[cache_key] = {"time": now_ts, "quote": quote}
                return quote

        if getattr(self, "fyers", None):
            # Current instrument master is Angel-token based; option Fyers symbols
            # are used only when the caller/cache already provides one.
            fyers_symbol = FYERS_INDEX_SYMBOLS.get(symbol, symbol if ":" in str(symbol) else "")
            if fyers_symbol:
                try:
                    ok, resp = self._call_fyers_bounded(
                        "quotes",
                        {"symbols": fyers_symbol},
                        label=f"Fyers touch quote {fyers_symbol}",
                    )
                    if not ok:
                        resp = None
                    if resp and resp.get("s") == "ok":
                        values = (resp.get("d") or [{}])[0].get("v", {})
                        quote = self._touch_quote_from_payload(
                            values,
                            "fyers_rest_touch",
                            now_ts,
                            max_age_seconds,
                        )
                        self._touch_quote_cache[cache_key] = {"time": now_ts, "quote": quote}
                        return quote
                except Exception as exc:
                    logger.debug(f"Fyers touch quote fetch failed for {symbol}: {exc}")

        quote = {
            "bid": 0.0,
            "ask": 0.0,
            "ltp": 0.0,
            "source": "unavailable",
            "age_seconds": 0.0,
            "usable": False,
            "reason": "unavailable",
        }
        self._touch_quote_cache[cache_key] = {"time": now_ts, "quote": quote}
        return quote

    def get_cached_contract_touch_quote(
        self,
        exchange: str,
        symbol: str,
        token: str,
        max_age_seconds: float = 15.0,
    ) -> Dict[str, Any]:
        """Return a websocket/cache touch without ever issuing a REST request.

        Entry-policy shadow watches can number in the hundreds. Polling a broker REST
        depth endpoint for every hypothetical rejected entry would compete with real
        orders and breach rate limits. Accepted execution boundaries continue to use
        :meth:`get_contract_touch_quote`; the research sampler uses this cache-only
        counterpart after subscribing the watched contract.
        """
        now_ts = time.time()
        cache_key = f"{exchange}|{symbol}|{token}"
        cached = self._touch_quote_cache.get(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("quote"), dict):
            cached_quote = cached["quote"]
            try:
                source_age = float(cached_quote.get("age_seconds"))
                if not np.isfinite(source_age) or source_age < 0:
                    source_age = float("inf")
            except (TypeError, ValueError, OverflowError):
                source_age = float("inf")
            # A wall-clock correction must never make an already-aged quote look
            # younger. Preserve the source age and only add non-negative cache
            # residence time.
            try:
                cached_at = float(cached.get("time"))
            except (TypeError, ValueError, OverflowError):
                cached_at = float("nan")
            cache_residence = (
                max(0.0, now_ts - cached_at)
                if np.isfinite(cached_at) and np.isfinite(now_ts)
                else float("inf")
            )
            age = source_age + cache_residence
            if 0 <= age <= max_age_seconds:
                quote = dict(cached_quote)
                quote["age_seconds"] = age
                quote["usable"] = bool(
                    quote.get("usable")
                    and float(quote.get("bid") or 0.0) > 0
                    and float(quote.get("ask") or 0.0) >= float(quote.get("bid") or 0.0)
                )
                if quote["usable"]:
                    return quote

        last_quote: Dict[str, Any] = {}
        for key in (token, str(token), symbol):
            payload = self._ws_cache.get(key) if key else None
            if not isinstance(payload, dict):
                continue
            quote = self._touch_quote_from_payload(
                payload,
                str(payload.get("source") or "ws_touch"),
                now_ts,
                max_age_seconds,
            )
            last_quote = quote
            if quote.get("usable"):
                self._touch_quote_cache[cache_key] = {"time": now_ts, "quote": quote}
                return quote
        return last_quote or {
            "bid": 0.0,
            "ask": 0.0,
            "source": "cache_unavailable",
            "age_seconds": 0.0,
            "usable": False,
            "reason": "cache_unavailable",
        }

    def get_order_book_snapshot(self, exchange: str, symbol: str, token: str) -> Dict[str, Any]:
        """
        Best-effort bid/ask depth snapshot for OFR.

        This intentionally uses cached feed fields only. If the broker payload
        does not expose depth, callers receive a neutral unavailable result and
        keep using candle-pressure order flow.
        """
        def stamp_age(result: Dict[str, Any], cached_row: Dict[str, Any]) -> Dict[str, Any]:
            """Carry transport freshness with depth so fast decisions can prove it."""
            stamped = dict(result or {})
            try:
                observed_epoch = float(cached_row.get("time") or 0.0)
            except (TypeError, ValueError):
                observed_epoch = 0.0
            if observed_epoch > 0:
                stamped["age_seconds"] = max(0.0, time.time() - observed_epoch)
                stamped["observed_at"] = datetime.fromtimestamp(observed_epoch).isoformat()
            else:
                stamped["age_seconds"] = None
                stamped["observed_at"] = ""
            return stamped

        keys = [token, str(token), symbol, FYERS_INDEX_SYMBOLS.get(symbol, "")]
        for key in keys:
            cached = self._ws_cache.get(key) if key else None
            if not isinstance(cached, dict):
                continue
            depth = cached.get("depth") or cached.get("market_depth") or cached.get("best5") or cached.get("bidask")
            parsed = self._parse_depth_payload(depth, cached.get("source", "ws_depth"))
            if parsed.get("usable"):
                return stamp_age(parsed, cached)
            bid_qty = cached.get("bid_qty") or cached.get("bid_size") or cached.get("bidsize")
            ask_qty = cached.get("ask_qty") or cached.get("ask_size") or cached.get("asksize")
            parsed = self._depth_from_qty(bid_qty, ask_qty, cached.get("source", "ws_depth"))
            if parsed.get("usable"):
                return stamp_age(parsed, cached)
        return {
            "ofr": 1.0,
            "bid_qty": 0,
            "ask_qty": 0,
            "source": "unavailable",
            "usable": False,
            "age_seconds": None,
            "observed_at": "",
        }

    def _parse_depth_payload(self, depth: Any, source: str = "depth") -> Dict[str, Any]:
        if not depth:
            return {"usable": False}
        bids, asks = [], []
        if isinstance(depth, dict):
            bids = depth.get("buy") or depth.get("bids") or depth.get("bid") or []
            asks = depth.get("sell") or depth.get("asks") or depth.get("ask") or []
        elif isinstance(depth, list):
            for row in depth:
                if not isinstance(row, dict):
                    continue
                side = str(row.get("side") or row.get("type") or "").lower()
                if side.startswith("b"):
                    bids.append(row)
                elif side.startswith("a") or side.startswith("s"):
                    asks.append(row)
        bid_qty = self._sum_depth_qty(bids)
        ask_qty = self._sum_depth_qty(asks)
        parsed = self._depth_from_qty(bid_qty, ask_qty, source)
        parsed["bid"] = self._best_depth_price(bids, "bid")
        parsed["ask"] = self._best_depth_price(asks, "ask")
        if parsed.get("bid", 0.0) > 0 and parsed.get("ask", 0.0) > 0:
            parsed["usable"] = True
        return parsed

    @staticmethod
    def _sum_depth_qty(rows: Any) -> float:
        if isinstance(rows, dict):
            rows = [rows]
        total = 0.0
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            qty = row.get("quantity", row.get("qty", row.get("size", row.get("volume", 0))))
            try:
                total += float(qty or 0)
            except (TypeError, ValueError):
                continue
        return total

    @staticmethod
    def _best_depth_price(rows: Any, side: str) -> float:
        if isinstance(rows, dict):
            rows = [rows]
        prices = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            price = row.get("price", row.get("rate", row.get("bp", row.get("ap", 0))))
            try:
                price = float(price or 0.0)
            except (TypeError, ValueError):
                price = 0.0
            if price > 0:
                prices.append(price)
        if not prices:
            return 0.0
        return max(prices) if side == "bid" else min(prices)

    @staticmethod
    def _depth_from_qty(bid_qty: Any, ask_qty: Any, source: str = "depth") -> Dict[str, Any]:
        try:
            bid = float(bid_qty or 0.0)
            ask = float(ask_qty or 0.0)
        except Exception:
            return {"usable": False}
        if bid <= 0 and ask <= 0:
            return {"usable": False}
        ofr = bid / ask if ask > 0 else (bid if bid > 0 else 1.0)
        return {
            "ofr": round(float(ofr), 3),
            "bid_qty": int(bid),
            "ask_qty": int(ask),
            "source": source,
            "usable": True,
        }

    def get_cached_previous_close(self, symbol: str) -> float:
        """Return remembered previous close without broker/network I/O."""
        try:
            return float(
                (getattr(self, "_prev_close_global_cache", {}) or {}).get(symbol, 0.0)
                or 0.0
            )
        except (TypeError, ValueError):
            return 0.0

    def get_previous_close(self, symbol: str) -> float:
        """Fetch official closing price with persistent session caching"""
        if getattr(self, "upstox_only", False):
            from data.upstox_market_data import UPSTOX_INDEX_KEYS

            key = UPSTOX_INDEX_KEYS.get(symbol)
            if not key:
                return 0.0
            ist = ZoneInfo("Asia/Kolkata")
            today = datetime.now(ist).date()
            cached_at = (getattr(self, "_prev_close_cache_time", {}) or {}).get(symbol, 0)
            if (cached_at and datetime.fromtimestamp(cached_at, ist).date() == today
                    and str((getattr(self, "_prev_close_cache_source", {}) or {}).get(symbol, "")).startswith("upstox")):
                return self.get_cached_previous_close(symbol)
            row = self._upstox_quote(key, 300.0)
            close = float(row.get("previous_close") or 0.0)
            if np.isfinite(close) and close > 0:
                return self._remember_previous_close(symbol, close, "upstox_quote")
            candles = self.upstox.get_historical_candles(key, "day", 7, to_date=today - timedelta(days=1))
            if candles:
                close = float(candles[-1][4] or 0.0)
                if np.isfinite(close) and close > 0:
                    return self._remember_previous_close(symbol, close, "upstox_daily")
            return 0.0
        if self._fyers_backup_enabled():
            existing_source = (getattr(self, "_prev_close_cache_source", {}) or {}).get(symbol, "")
            existing_age = time.time() - (getattr(self, "_prev_close_cache_time", {}) or {}).get(symbol, 0)
            if existing_source.startswith(("kite", "angel")) and 0 <= existing_age < 300:
                return float((getattr(self, "_prev_close_global_cache", {}) or {}).get(symbol) or 0)
            primary = self._freshest_cached_index_quote(self._index_broker_cache_keys(symbol),
                self.INDEX_KITE_PRICE_FRESH_SECONDS if self.is_market_open() else 86400.0, time.time())
            if primary is not None and str(primary[1].get("source", "")).startswith("kite"):
                close = float(primary[1].get("previous_close") or 0)
                if close > 0:
                    return self._remember_previous_close(symbol, close, "kite_ws")
            supplementary = self._angel_supplementary_quote(symbol)
            if supplementary is not None and float(supplementary.get("previous_close") or 0) > 0:
                return float(supplementary["previous_close"])
        if not hasattr(self, "_prev_close_global_cache"):
            self._prev_close_global_cache = {}
            self._prev_close_cache_time = {}
            
        if symbol in self._prev_close_global_cache:
            cache_source = getattr(self, "_prev_close_cache_source", {}).get(symbol, "")
            cache_age = time.time() - getattr(self, "_prev_close_cache_time", {}).get(symbol, 0)
            if (
                cache_age < 300
                and (
                    cache_source == "fyers_rest"
                    or symbol not in INDEX_SPOT_TOKENS
                    or not getattr(self, "fyers", None)
                )
            ):
                return self._prev_close_global_cache[symbol]

        if symbol in INDEX_SPOT_TOKENS and getattr(self, "fyers", None):
            try:
                values = self._fyers_index_quote_values(symbol)
                previous_close = self._quote_float(values, *PREVIOUS_CLOSE_KEYS)
                if previous_close > 0:
                    return self._remember_previous_close(symbol, previous_close, "fyers_rest")
            except Exception as exc:
                logger.debug(f"Fyers prev_close fetch error for {symbol}: {exc}")

        angel = getattr(self, "angel_index", None)
        if symbol in INDEX_SPOT_TOKENS and angel is not None:
            try:
                quote = angel.get_index_quote(symbol)
            except Exception:
                quote = None
            if isinstance(quote, dict):
                close_price = float(quote.get("previous_close") or 0.0)
                if close_price > 0:
                    return self._remember_previous_close(
                        symbol, close_price, "angel_rest"
                    )

        # 1. BROKER LIVE FEED (Primary Source)
        if self.is_connected and self.smart_api:
            if symbol in INDEX_SPOT_TOKENS:
                token, exchange = INDEX_SPOT_TOKENS[symbol]
                try:
                    trading_symbol = "INDEX" if symbol in INDEX_SPOT_TOKENS else symbol
                    
                    resp = self._best_effort_angel_rest_call(
                        "previousClose",
                        lambda: self.smart_api.ltpData(exchange, trading_symbol, token),
                    )
                    if isinstance(resp, dict) and resp.get('status') and resp.get('data'):
                        close_price = float(resp['data'].get('close', 0.0))
                        if close_price > 0:
                            return self._remember_previous_close(symbol, close_price, "angel_rest")
                except Exception as e:
                    logger.debug(f"AngelOne prev_close fetch error for {symbol}: {e}")

        return 0.0

    def get_latest_price(self, symbol: str) -> float:
        """Alias for convenience, attempts to find correct token"""
        if symbol in INDEX_SPOT_TOKENS:
            t, e = INDEX_SPOT_TOKENS[symbol]
            return self.get_latest_price_by_token(t, symbol, e)
        return self._ltp_cache.get(symbol, {}).get('ltp', 0.0)

    # ═══════════════════════════════════════════════════════════
    # HISTORICAL & OPTIONS
    # ═══════════════════════════════════════════════════════════

    def _angel_history_rate_limited(self) -> bool:
        """True while the AngelOne history endpoint is inside a rate-limit cooldown."""
        return time.time() < float(getattr(self, "_angel_hist_rate_limited_until", 0.0) or 0.0)

    def _note_angel_history_rate_limit(self, context: str) -> None:
        """Escalating cooldown: 10s doubling per consecutive hit, capped at 120s."""
        streak = int(getattr(self, "_angel_hist_rate_limit_streak", 0) or 0) + 1
        self._angel_hist_rate_limit_streak = streak
        cooldown = min(120.0, 10.0 * (2 ** (streak - 1)))
        self._angel_hist_rate_limited_until = time.time() + cooldown
        logger.warning(
            f"🚨 AngelOne history rate limit ({context}); cooling down {cooldown:.0f}s "
            f"(consecutive hit #{streak})"
        )

    def get_historical_candles_angel(self, token, exchange, interval, days_back=15) -> pd.DataFrame:
        """Fetch historical candles directly from AngelOne SmartApi"""
        if not self.is_connected or not self.smart_api:
            return pd.DataFrame()
        if self._angel_history_rate_limited():
            logger.debug(f"AngelOne history cooldown active; skipping fetch for {token}")
            return pd.DataFrame()

        try:
            # Map interval to AngelOne format
            api_interval = {
                "1min": "ONE_MINUTE",
                "5min": "FIVE_MINUTE",
                "15min": "FIFTEEN_MINUTE"
            }.get(interval, "FIVE_MINUTE")
            
            to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d 09:15")
            
            params = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": api_interval,
                "fromdate": from_date,
                "todate": to_date
            }
            
            # Historical data is deliberately rate-gated and best-effort: it must never
            # flood AngelOne's published 3/s, 180/min history budget. Unpaced fetches
            # produced 189 rate-limit hits in one session, which forced 279 Yahoo
            # fallbacks — and Yahoo reports volume=0 for index tickers, so the whole
            # volume pipeline died. Skipping a request here is safe (cache/fallback
            # covers it); exhausting the budget is not.
            history_limiter = get_shared_history_limiter(getattr(self, "client_id", ""))

            def _history_request():
                api = self.smart_api
                if api is None:
                    return None
                if not history_limiter.try_begin(get_shared_api_limiter(api)):
                    logger.debug(
                        "Skipping AngelOne history request; rate window or "
                        "broker-priority lane is occupied"
                    )
                    return None
                try:
                    return api.getCandleData(params)
                finally:
                    history_limiter.end()

            resp = _history_request()
            if not isinstance(resp, dict):
                return pd.DataFrame()

            # Check for Invalid Token error
            if not resp.get('status') and resp.get('errorCode') == 'AG8001':
                logger.warning("🔄 AngelOne History Session Expired (AG8001). Re-authenticating...")
                if self.connect():
                    resp = _history_request()
                    if not isinstance(resp, dict):
                        return pd.DataFrame()
                else:
                    return pd.DataFrame()

            if resp.get('status') and resp.get('data'):
                df = pd.DataFrame(resp['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

                # Force IST Normalization (Broker → IST Naive)
                # This ensures '03:45 UTC' becomes '09:15 Naive'
                if df.index.tz is not None:
                    df.index = df.index.tz_convert('Asia/Kolkata').tz_localize(None)

                self._angel_hist_rate_limit_streak = 0
                return df
            else:
                err = resp.get('message', 'Unknown Error')
                if "exceeding access rate" in err.lower():
                    self._note_angel_history_rate_limit(f"response for {token}")
                else:
                    logger.debug(f"AngelOne Empty Response for {token}: {err}")
        except Exception as e:
            # The SDK surfaces the plain-text rate-limit body as a JSON parse
            # failure, so the rate-limit signal usually arrives via this branch.
            if "exceeding access rate" in str(e).lower():
                self._note_angel_history_rate_limit(f"exception for {token}")
            else:
                logger.error(f"❌ AngelOne History Fetch Error: {e}")
        return pd.DataFrame()

    @staticmethod
    def _attach_fyers_history_provenance(frame: pd.DataFrame, *, received_at: Optional[datetime] = None) -> pd.DataFrame:
        """Annotate Fyers bars without pretending the REST response is a live receipt."""
        if frame is None or frame.empty:
            return frame
        out = frame.copy()
        stamp = received_at or datetime.now()
        out["volume_basis"] = "fyers_index"
        out["source"] = "fyers_history"
        # Completed History API bars replace price-only live ticks. Without an
        # explicit flag, telemetry restoration carries the tick's False into the
        # backfill, so every resampled bar stays incomplete despite measured volume.
        quantities = pd.to_numeric(
            out.get("volume", pd.Series(index=out.index, dtype=float)), errors="coerce"
        )
        complete = quantities.notna() & np.isfinite(quantities) & quantities.ge(0)
        if "volume_complete" in out:
            complete &= ~out["volume_complete"].eq(False).fillna(False)
        out["volume_complete"] = complete
        # Fyers' candle timestamp is exchange/provider time; receipt_at is local
        # transport time and must remain a separate clock for freshness decisions.
        out["provider_time"] = out.index
        out["receipt_at"] = stamp
        out["tick_count"] = 0
        return out

    @staticmethod
    def _attach_kite_history_provenance(
        frame: pd.DataFrame, *, received_at: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Annotate contract history with its provider and independent receipt clocks."""
        if frame is None or frame.empty:
            return frame
        out = frame.copy()
        out["volume_basis"] = "kite_contract_volume"
        out["source"] = "kite_history"
        out["provider_time"] = out.index
        out["receipt_at"] = received_at or datetime.now()
        out["tick_count"] = 0
        return out

    def _preferred_history_complete(self, frame, interval, days_back, to_date=None):
        """Skip backup requests only for certified, contiguous requested intervals.

        Constituent-share certification can never cover the index's 15:15-15:29
        tail (constituent candles stop at 15:14) or a day with no dated membership
        snapshot. Demanding either one pinned this gate to False, so every index
        request fell through to the FYERS backup even when the certifiable window
        was complete. Only intervals a constituent can certify at all are required.
        """
        from data.market_calendar import load_nse_holidays, load_nse_special_trading_days
        from data.constituent_volume import CONSTITUENT_SESSION_CLOSE, ConstituentVolumeService
        if frame is None or frame.empty or interval not in {"1min", "5min", "15min"}:
            return False
        frame = ConstituentVolumeService.local_frame(frame)
        verified = self._merge_preferred_index_history(frame, pd.DataFrame())
        if verified.empty:
            return False
        target = pd.Timestamp(to_date) if to_date is not None else pd.Timestamp.now(tz="Asia/Kolkata")
        if target.tzinfo is not None:
            target = target.tz_convert("Asia/Kolkata").tz_localize(None)
        period = pd.Timedelta(minutes={"1min": 1, "5min": 5, "15min": 15}[interval])
        holidays, special = set(load_nse_holidays()), set(load_nse_special_trading_days())
        close = pd.Timedelta(hours=CONSTITUENT_SESSION_CLOSE[0], minutes=CONSTITUENT_SESSION_CLOSE[1])
        expected = []
        lower = target-pd.Timedelta(days=max(1, int(days_back)))
        for day in pd.date_range(lower.normalize(), target.normalize(), freq="D"):
            label = day.date().isoformat()
            if label not in special and (day.weekday() >= 5 or label in holidays):
                continue
            opening, closing = day+pd.Timedelta(hours=9, minutes=15), day+close
            expected.extend(stamp for stamp in pd.date_range(opening, closing, freq=period, inclusive="left")
                            if lower <= stamp and stamp+period <= target)
        if not expected:
            return False
        expected = pd.DatetimeIndex(sorted(set(expected)))
        # The newest day is built only up to its last completed minute, so require
        # it just as far as the frame reaches. Every earlier day stays exact, and a
        # day the frame does not reach at all is still required.
        newest = verified.index.max()
        if newest is not None:
            # A background reconciliation may trail by one refresh pass. Never
            # extend that allowance indefinitely or across a missing session.
            refresh = float(getattr(getattr(self, "_replacement_settings", None),
                                    "constituent_history_refresh_seconds", 60.0))
            # A single missing minute can withhold an entire resampled bar. Use
            # the same minute allowance at every timeframe, plus rounding to its
            # last complete slot; do not fabricate that unfinished higher bar.
            grace = pd.Timedelta(seconds=max(90.0, 2 * refresh)) + period - pd.Timedelta(minutes=1)
            if expected.max() - newest > grace:
                return False
            expected = expected[(expected.normalize() != newest.normalize()) | (expected <= newest)]
        if not len(expected) or not expected.isin(verified.index).all():
            return False
        complete = verified.get("volume_complete", pd.Series(False, index=verified.index))
        complete = complete.reindex(expected).fillna(False).eq(True)
        # Invalid certificates must not erase known membership and turn a real
        # coverage failure into an exempt, undated historical interval.
        dated = frame.get("membership_version", pd.Series("", index=frame.index))
        dated = dated.fillna("").astype(str).ne("")
        dated = dated.groupby(dated.index.normalize()).transform("any")
        dated = dated.reindex(expected, fill_value=False)
        return bool(dated.any()) and bool(complete[dated].all())

    def _call_fyers_history_with_backoff(self, data, *, timeout, label):
        """Share history cooldown across indices; never repeat a rejected request."""
        if time.time() < float(getattr(self, "_fyers_history_retry_at", 0.0)):
            return False, None
        ok, response = self._call_fyers_bounded("history", data=data, timeout_seconds=timeout, label=label)
        text = self._angel_rest_error_text(response).lower()
        limited = isinstance(response, dict) and response.get("code") in {-360, 429}
        limited = limited or "requests/minute" in text or "requests/day" in text or "too many requests" in text
        if limited or not ok:
            delay = 60.0 if limited else 30.0
            if "requests/day" in text:
                now = pd.Timestamp.now(tz="Asia/Kolkata")
                delay = max(delay, (now.normalize()+pd.Timedelta(days=1)-now).total_seconds())
            self._fyers_history_retry_at = time.time()+delay
            if limited:
                # Said once, when the refusal happens. Callers re-enter this path every
                # few seconds, so logging there instead produced a stream of identical
                # "returned no data" lines for requests the cooldown never even made.
                logger.warning(
                    f"FYERS history rate-limited ({text[:160] or 'no detail'}) — "
                    f"backing off {delay:.0f}s; certified constituent-share volume and "
                    "Yahoo remain in force meanwhile"
                )
        return ok, response

    def get_historical_candles(
        self, token, exchange, interval, days_back=15, instrument_name="",
        force_refresh: bool = False, to_date: Optional[datetime] = None,
        persist_cache: bool = True,
        kite_call_outcome: Optional[Dict[str, Any]] = None,
        record_kite_history_health: bool = True,
    ) -> pd.DataFrame:
        """
        Unified Historical Data Fetcher — Strictly prioritized for AngelOne.

        `force_refresh` skips BOTH read caches below and goes to the broker. Callers that
        merely want the series keep the default and keep every rate-limit saving; the one
        caller that sets it is the post-close premium back-fill, whose entire purpose is
        to ACQUIRE a day the cache does not have. The option-CSV short-circuit answers on
        file EXISTENCE, not on coverage, so without this a contract whose cache ends at an
        earlier session returns that stale frame, looks like a successful fetch, and the
        day is never written -- which is why 139 archived option files exist and not one
        of them is from August. ``to_date`` pins that back-fill to the ledger's own
        session instead of silently asking for one day relative to process time.

        ``persist_cache=False`` keeps diagnostic/audit callers filesystem read-only.
        Broker results still populate this provider instance's in-memory TTL cache,
        but no candle directory or CSV is created or merged.
        """
        cache_dir = Path("data_store/candles")
        if persist_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{instrument_name}_{interval}.csv"
        df = pd.DataFrame()

        # 0. GLOBAL API CACHE CHECK (Applies to both Fyers and AngelOne)
        now = time.time()
        target_key = to_date.isoformat() if isinstance(to_date, datetime) else "latest"
        cache_key = (
            f"hist_api_{instrument_name}_{exchange}_{token}_{interval}_{int(days_back)}_{target_key}"
        )
        if not hasattr(self, "_api_hist_cache"): self._api_hist_cache = {}
        if not hasattr(self, "_last_api_hist_fetch"): self._last_api_hist_fetch = {}
        
        from config.settings import get_settings
        runtime_settings = get_settings()
        history_ttl = float(getattr(runtime_settings, "history_cache_ttl_seconds", 60.0))
        if (
            str(interval or "").lower() == "1min"
            and str(instrument_name or "").upper() in FYERS_INDEX_SYMBOLS
            and self.is_market_open()
        ):
            # The scanner requests 1m history every ~60s. Reusing the general 240s
            # cache made the newest complete bar predictably 3–5 minutes late.
            history_ttl = min(
                history_ttl,
                45.0,
                float(
                    getattr(
                        runtime_settings,
                        "live_1min_history_cache_ttl_seconds",
                        45.0,
                    )
                    or 45.0
                ),
            )
        last_fetch = self._last_api_hist_fetch.get(cache_key, 0)
        if isinstance(last_fetch, datetime):
            last_fetch = last_fetch.timestamp()
        elif not isinstance(last_fetch, (int, float)):
            last_fetch = 0
        if not force_refresh and now - last_fetch < history_ttl:
            cached_df = self._api_hist_cache.get(cache_key)
            if cached_df is not None:
                logger.debug(f"Using cached API history for {cache_key}")
                return cached_df

        # Historical option candles are heavily reused during backtests. Prefer the
        # local per-symbol CSV cache before touching broker APIs to avoid rate-limit
        # stalls and session-token noise for expired option contracts.
        is_option_symbol = bool(instrument_name) and (str(instrument_name).upper().endswith(("CE", "PE"))
            or any(part in {"CE", "PE"} for part in str(instrument_name).upper().split()))
        if not force_refresh and is_option_symbol and cache_file.exists():
            try:
                cached_df = get_option_history_store().load(cache_file)
                if not cached_df.empty:
                    self._api_hist_cache[cache_key] = cached_df
                    self._last_api_hist_fetch[cache_key] = now
                    logger.info(f"📦 Using Local Option History Cache for {instrument_name}")
                    return cached_df
            except Exception as exc:
                logger.debug(f"Option history cache read failed for {instrument_name}: {exc}")

        if getattr(self, "upstox_only", False):
            from data.upstox_market_data import UPSTOX_INDEX_KEYS
            key = UPSTOX_INDEX_KEYS.get(str(instrument_name).upper(), str(token or ""))
            rows = self.upstox.get_historical_candles(key, interval, days_back, to_date=to_date)
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame([row[:7] for row in rows],
                              columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
            df["timestamp"] = (pd.to_datetime(df["timestamp"], utc=True)
                               .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None))
            df = df.set_index("timestamp").sort_index()
            df = df.loc[~df.index.duplicated(keep="last")]
            df["price_source"] = "upstox_history"
            if "_INDEX|" in key:
                if str(interval).lower() in {"1min", "1m", "1minute"}:
                    requested_at = pd.Timestamp.fromtimestamp(now, tz="Asia/Kolkata").tz_localize(None)
                    df = df.loc[df.index + pd.Timedelta(minutes=1) <= requested_at]
                    if df.empty:
                        return df
                df["volume"] = 0.0
                df["volume_source"] = "unavailable"
            else:
                df["volume_source"] = "upstox_derivative"
            if persist_cache:
                merged = self._save_and_merge_candles(df, cache_file)
                if merged is not None and days_back != 0:
                    df = merged
            self._api_hist_cache[cache_key] = df
            self._last_api_hist_fetch[cache_key] = now
            return df

        # 0b. THE ONLY LIVE HISTORY SOURCE FOR AN OPTION CONTRACT.
        # Everything below this point that could fetch a CE/PE goes through
        # get_historical_candles_angel -> self.smart_api, and self.smart_api has been
        # permanently None since the Kite execution migration (it is assigned in exactly
        # one place in this file, to None). So an option request has had NO reachable
        # broker source since 2026-08-01 -- which is precisely why no option CSV holds a
        # bar newer than 2026-07-31 and every August session archived zero premium.
        # Skipping the caches was never going to be enough on its own; there was nothing
        # behind them to reach.
        #
        # `token` here is the Kite instrument_token, which is what the session event log
        # records and what historical_data() wants.
        if is_option_symbol:
            kite_source = getattr(self, "kite", None)
            if kite_source is not None and getattr(kite_source, "is_connected", False):
                try:
                    history_kwargs = {
                        "days_back": days_back,
                        "to_date": to_date,
                    }
                    # Normal callers keep the historical source's shared live-health
                    # accounting.  Archive repair asks for call-local evidence and opts
                    # out so an expired contract cannot poison that unrelated surface.
                    if kite_call_outcome is not None:
                        history_kwargs["call_outcome"] = kite_call_outcome
                    if not record_kite_history_health:
                        history_kwargs["record_health"] = False
                    kite_df = kite_source.get_historical(
                        int(token), interval, **history_kwargs
                    )
                except (TypeError, ValueError) as exc:
                    logger.debug(f"Kite option history rejected {instrument_name}: {exc}")
                    kite_df = pd.DataFrame()
                except Exception as exc:
                    logger.warning(f"Kite option history failed for {instrument_name}: {exc}")
                    kite_df = pd.DataFrame()
                if kite_df is not None and not kite_df.empty:
                    # scripts/archive_session_data.day_slice keys on a "timestamp"
                    # column, so the index has to carry that name into the CSV or the
                    # archiver cannot see the day it just fetched.
                    kite_df = kite_df.copy()
                    kite_df.index.name = "timestamp"
                    # Defence in depth: _save_and_merge_candles normalises too, but
                    # `df` is also returned to the live caller below without passing
                    # through it when the merge is skipped, and a tz-aware frame there
                    # is the mixed-dtype precondition that froze the scan loop on
                    # 2026-08-19 ("Invalid comparison between dtype=datetime64[us,
                    # tzoffset(None, 19800)] and Timestamp").
                    kite_df = self._naive_kolkata_index(kite_df)
                    kite_df = self._attach_kite_history_provenance(kite_df)
                    # Explicit None test: `merged or kite_df` would evaluate the truth
                    # value of a DataFrame, which raises.
                    if persist_cache:
                        merged = self._save_and_merge_candles(kite_df, cache_file)
                        df = kite_df if merged is None else merged
                    else:
                        df = kite_df
                    self._api_hist_cache[cache_key] = df
                    self._last_api_hist_fetch[cache_key] = now
                    logger.info(
                        f"📈 Kite option history for {instrument_name}: {len(df)} candles"
                    )
                    return df

        preferred_index_history = pd.DataFrame()
        if self._replacement_enabled() and instrument_name in FYERS_INDEX_SYMBOLS:
            # Candidate history stays separate from preserved FYERS archives.
            preferred_index_history = self._request_replacement_history(instrument_name, interval, days_back, to_date,
                persist_cache=persist_cache, force_refresh=force_refresh)
            if not self._fyers_backup_enabled():
                return preferred_index_history
            if self._preferred_history_complete(preferred_index_history, interval, days_back, to_date):
                preferred_index_history = self._merge_preferred_index_history(preferred_index_history, pd.DataFrame())
                if persist_cache:
                    self._save_and_merge_candles(preferred_index_history, cache_file)
                self._api_hist_cache[cache_key] = preferred_index_history
                self._last_api_hist_fetch[cache_key] = now
                return preferred_index_history

        fyers_history_attempted = False
        # FYERS supplies missing backup history/volume after preferred brokers.
        #
        # A refusal already armed the shared cooldown in _call_fyers_history_with_backoff,
        # and re-entering here per index, per timeframe, on every scan tick bought
        # nothing: the call was suppressed anyway, but four INFO/WARNING pairs a second
        # were formatted onto the scanner's own loop and the log read "returned no data"
        # for requests that were never made. A refused request is not a dead source, so
        # this stays a *timing* gate — the cooldown expires, the branch runs again.
        fyers_ready = (bool(getattr(self, 'fyers', None)) and not bool(getattr(self, "_fyers_auth_required", False))
                       and time.time() >= float(getattr(self, "_fyers_history_retry_at", 0.0)))
        if (not fyers_ready and getattr(self, 'fyers', None)
                and instrument_name in FYERS_INDEX_SYMBOLS):
            logger.debug(f"FYERS history for {instrument_name} still in cooldown; "
                         "preferred brokers and Yahoo remain available")
        if fyers_ready and instrument_name in FYERS_INDEX_SYMBOLS:
            try:
                fyers_symbol = FYERS_INDEX_SYMBOLS.get(instrument_name)
                
                if fyers_symbol:
                    preferred_end = (preferred_index_history.index.max()
                                     if not preferred_index_history.empty else None)
                    logger.info(
                        "Fetching FYERS history fallback for {}: interval={}, preferred_rows={}, "
                        "preferred_end={}, force_refresh={}, recovery={}",
                        fyers_symbol, interval, len(preferred_index_history), preferred_end,
                        force_refresh, getattr(self, "_replacement_history_errors", {}).get(instrument_name, "unknown"),
                    )
                    res = "1" if interval == "1min" else ("5" if interval == "5min" else "15")
                    history_target = pd.Timestamp(to_date) if to_date is not None else pd.Timestamp.now(tz="Asia/Kolkata")
                    if history_target.tzinfo is not None:
                        history_target = history_target.tz_convert("Asia/Kolkata").tz_localize(None)
                    range_from = (history_target - timedelta(days=days_back)).strftime("%Y-%m-%d")
                    range_to = history_target.strftime("%Y-%m-%d")
                    
                    data = {
                        "symbol": fyers_symbol,
                        "resolution": res,
                        "date_format": "1",
                        "range_from": range_from,
                        "range_to": range_to,
                        "cont_flag": "1"
                    }
                    # Bounded. This runs on the scanner's event loop, and an unbounded
                    # SDK call here is what stalled the 2026-08-31 open.
                    _hist_timeout = float(
                        getattr(get_settings(), "fyers_history_timeout_seconds", 8.0) or 8.0
                    )
                    fyers_history_attempted = True
                    _ok, response = self._call_fyers_history_with_backoff(
                        data,
                        timeout=_hist_timeout,
                        label=f"fyers.history({fyers_symbol} {res})",
                    )
                    if not _ok:
                        response = None
                    # This is the ONLY source of index volume — AngelOne does not supply it
                    # and the Yahoo tertiary reports 0 for index tickers. Every exit below
                    # used to be silent, so when Fyers stopped returning current-day index
                    # candles on 2026-07-27 the volume pipeline died with nothing in the log
                    # to say why; every volume-gated entry has been dead since. Say why.
                    if not response or response.get('s') != 'ok':
                        logger.warning(
                            f"⚠️ Fyers index history returned no data for {fyers_symbol} "
                            f"({res}m, {range_from}..{range_to}): "
                            f"s={(response or {}).get('s')!r} code={(response or {}).get('code')!r} "
                            f"message={(response or {}).get('message')!r}. "
                            "Completed cash-volume gates abstain unless independently certified "
                            "constituent intervals are available."
                        )
                    else:
                        candles = response.get('candles', [])
                        if not candles:
                            logger.warning(
                                f"⚠️ Fyers index history for {fyers_symbol} returned s=ok but ZERO "
                                f"candles ({res}m, {range_from}..{range_to}); falling through "
                                "to a source that has no index volume."
                            )
                        else:
                            fyers_df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            fyers_df['timestamp'] = pd.to_datetime(fyers_df['timestamp'], unit='s')
                            fyers_df['timestamp'] = fyers_df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
                            fyers_df.set_index('timestamp', inplace=True)
                            if interval == "1min":
                                self._record_index_intrabar_cash_volume(
                                    instrument_name,
                                    fyers_df,
                                    received_at=datetime.now(),
                                )
                            # The newest candle is the minute currently being traded. It
                            # carries a fraction of its eventual volume, and once its
                            # minute closes nothing downstream can tell that fraction from
                            # a genuinely quiet bar.
                            fyers_df = drop_forming_tail_candle(
                                fyers_df, int(res), f"{fyers_symbol} {res}m"
                            )
                            fyers_df = self._attach_fyers_history_provenance(fyers_df)

                            if not fyers_df.empty:
                                nonzero = int((fyers_df['volume'] > 0).sum())
                                if nonzero == 0:
                                    logger.warning(
                                        f"⚠️ Fyers returned {len(fyers_df)} candles for {fyers_symbol} "
                                        "but EVERY volume is zero — the feed is up, the volume field is not."
                                    )
                                else:
                                    logger.debug(
                                        f"Fyers index history {fyers_symbol}: {len(fyers_df)} candles, "
                                        f"{nonzero} with volume, newest {fyers_df.index[-1]}."
                                    )
                                df = fyers_df
                                if persist_cache:
                                    df = self._save_and_merge_candles(df, cache_file)
                                # Cache update moved to bottom
            except Exception as e:
                logger.warning(f"⚠️ Fyers history fetch failed for {instrument_name}: {e}")

        # 2. FALLBACK/PRIMARY FOR OTHERS: BROKER (AngelOne)
        if df.empty and self.is_connected and self.smart_api and token:
            try:
                df = self.get_historical_candles_angel(token, exchange, interval, days_back)
                if not df.empty:
                    if persist_cache:
                        df = self._save_and_merge_candles(df, cache_file)
                    # Cache update moved to bottom
            except Exception as e:
                logger.warning(f"⚠️ AngelOne fetch failed for {instrument_name}: {e}")

        # 3. FALLBACK FOR OTHERS: FYERS (If AngelOne fails)
        if df.empty and fyers_ready and not fyers_history_attempted:
            fyers_symbol = FYERS_INDEX_SYMBOLS.get(instrument_name)
            
            if fyers_symbol:
                try:
                    logger.info(f"Fetching Fyers history for {fyers_symbol}...")
                    # Map interval to Fyers resolution
                    res = "1" if interval == "1min" else ("5" if interval == "5min" else "15")
                    
                    range_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                    range_to = datetime.now().strftime("%Y-%m-%d")
                    
                    data = {
                        "symbol": fyers_symbol,
                        "resolution": res,
                        "date_format": "1",
                        "range_from": range_from,
                        "range_to": range_to,
                        "cont_flag": "1"
                    }
                    # Bounded for the same reason as the index-history call above: this
                    # also runs on the scanner's loop and the SDK takes no timeout.
                    _ok, response = self._call_fyers_history_with_backoff(
                        data,
                        timeout=float(
                            getattr(get_settings(), "fyers_history_timeout_seconds", 8.0) or 8.0
                        ),
                        label=f"fyers.history({fyers_symbol} {res})",
                    )
                    if not _ok:
                        response = None
                    if response and response.get('s') == 'ok':
                        candles = response.get('candles', [])
                        if candles:
                            fyers_df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            fyers_df['timestamp'] = pd.to_datetime(fyers_df['timestamp'], unit='s')
                            fyers_df['timestamp'] = fyers_df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
                            fyers_df.set_index('timestamp', inplace=True)
                            fyers_df = drop_forming_tail_candle(
                                fyers_df, int(res), f"{fyers_symbol} {res}m"
                            )
                            fyers_df = self._attach_fyers_history_provenance(fyers_df)

                            if not fyers_df.empty:
                                df = fyers_df
                                if persist_cache:
                                    df = self._save_and_merge_candles(df, cache_file)
                except Exception as e:
                    logger.warning(f"⚠️ Fyers history fetch failed for {instrument_name}: {e}")

        if self._fyers_backup_enabled() and not preferred_index_history.empty:
            df = self._merge_preferred_index_history(preferred_index_history, df)
            if persist_cache:
                df = self._save_and_merge_candles(df, cache_file)

        # 2. SECONDARY: LOCAL CACHE (If Broker fails or Market is closed)
        if df.empty and cache_file.exists() and not force_refresh:
            # NOT on a forced refresh. This branch is the second way the same stale frame
            # comes back: the caller asked to go and GET a day, every source failed, and
            # handing back the old cache here makes the failure look like a success --
            # the back-fill then counts the contract "filled" and logs a number, while
            # nothing new was written and the archive stays empty. A forced refresh that
            # fetched nothing must return empty and say so.
            try:
                cached_df = self._naive_kolkata_index(
                    pd.read_csv(cache_file, index_col=0, parse_dates=True)
                )
                if not cached_df.empty:
                    df = cached_df
                    logger.info(f"📦 Using Local Cache for {instrument_name}")
            except Exception as exc:
                logger.debug(f"Local candle cache read failed for {instrument_name}: {exc}")

        # 3. TERTIARY: YAHOO FALLBACK
        # If no data or data is from a previous day during a weekday, fallback to Yahoo
        today = datetime.now().date()
        is_stale = False
        if not df.empty:
            last_timestamp = df.index[-1]
            last_date = last_timestamp.date()
            # Before today's first interval closes, the previous session's final
            # candle is the newest possible completed bar. Yahoo cannot repair it.
            interval_minutes = {"1min": 1, "5min": 5, "15min": 15}.get(interval, 5)
            first_close = datetime.combine(today, datetime.min.time()) + timedelta(hours=9, minutes=15+interval_minutes)
            if last_date < today and self.is_market_open() and datetime.now() >= first_close:
                is_stale = True
                logger.info(f"📆 Cache data for {instrument_name} is stale (Last: {last_date}). Triggering Yahoo fallback.")
            elif self.is_market_open() and last_date == today:
                now = datetime.now()
                try:
                    int_mins = int(interval.replace('min', ''))
                except (AttributeError, TypeError, ValueError):
                    int_mins = 5
                threshold_minutes = max(15, 3 * int_mins)
                if (now - last_timestamp).total_seconds() / 60 > threshold_minutes:
                    is_stale = True
                    logger.info(f"🕒 Cache data for {instrument_name} is stale (Last: {last_timestamp}). Triggering Yahoo fallback.")
                
        if df.empty or is_stale:
            ticker = YAHOO_TICKERS.get(instrument_name, "")
            if ticker:
                try:
                    self._yahoo_request_count += 1
                    logger.info(f"📡 Falling back to Yahoo for {instrument_name}...")
                    yahoo_days = min(int(days_back), 7 if interval == "1min" else 60)
                    yahoo_df = yf.Ticker(ticker).history(period=f"{yahoo_days}d", interval=YAHOO_INTERVALS.get(interval, "5m"))
                    if not yahoo_df.empty:
                        self._yahoo_last_used_at = time.time()
                        # Yahoo reports volume=0 for index tickers. This used to be a blind
                        # `df = yahoo_df`, so a stale-but-real frame WITH volume was replaced
                        # by a fresh one WITHOUT it — and volume_ratio then computes 0.0
                        # (not a neutral 1.0), silently disabling every volume-gated entry
                        # and the fast-opposite thrust exit guard. Take Yahoo's prices, but
                        # never let it zero out volume we already had for the same bars.
                        df = self._merge_yahoo_preserving_volume(df, yahoo_df, instrument_name)
                except Exception as e:
                    logger.error(f"❌ Yahoo Fetch Error for {instrument_name}: {e}")

        # Yahoo is delayed last-resort coverage. It may fill absent intervals but
        # cannot overwrite prices already supplied by Kite/Angel for the same bar.
        if not preferred_index_history.empty:
            df = self._merge_preferred_index_history(preferred_index_history, df)

        if df.empty:
            empty_df = pd.DataFrame()
            self._api_hist_cache[cache_key] = empty_df
            self._last_api_hist_fetch[cache_key] = now
            return empty_df
        
        # --- NORMALIZATION & SESSION FILTERING ---
        df.columns = [c.lower() for c in df.columns]
        if 'adj close' in df.columns: df['close'] = df['adj close']
        
        if df.index.tz is not None:
            df.index = df.index.tz_convert('Asia/Kolkata').tz_localize(None)
        else:
            if instrument_name in YAHOO_TICKERS and df.index[-1].hour < 6:
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata').tz_localize(None)

        df = df.between_time('09:15', '15:30')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns: df[col] = df['close'] if col != 'volume' else 0
            
        base_columns = ['open', 'high', 'low', 'close', 'volume']
        telemetry_columns = [
            name
            for name in ('tick_count', 'volume_basis', 'source', 'provider_time', 'receipt_at',
                         'price_provider', 'volume_provider', 'volume_complete', 'membership_version')
            if name in df.columns
        ]
        final_df = df[base_columns + telemetry_columns].copy()
        
        # ALWAYS update cache timestamp to prevent rate-limit spam loops on failures
        self._api_hist_cache[cache_key] = final_df
        self._last_api_hist_fetch[cache_key] = now
        
        return final_df

    @staticmethod
    def _merge_yahoo_preserving_volume(
        prior: pd.DataFrame, yahoo_df: pd.DataFrame, instrument_name: str = ""
    ) -> pd.DataFrame:
        """Take Yahoo's fresher prices without letting it zero out real volume.

        Yahoo reports volume=0 for index tickers. The tertiary fallback used to replace the
        frame outright, so a stale-but-real frame WITH volume lost it to a fresh one
        WITHOUT — and `volume_ratio` then evaluates to 0.0 rather than a neutral 1.0
        (see Scanner._candle_thrust_metrics), which silently disables volume_spike_entry,
        one_way_cascade, market_ltp_shock, late_session_reclaim, the 15m choppy scalp and
        the fast-opposite thrust exit guard.

        Yahoo still wins on price and on any bar we did not already have. Volume is only
        ever carried FORWARD from the prior frame for bars it actually covered; bars only
        Yahoo has keep their zero, which is honest — we have no volume for them.
        """
        out = yahoo_df.copy()
        out.columns = [str(c).lower() for c in out.columns]
        if "volume" not in out.columns:
            out["volume"] = 0
        if prior is None or prior.empty:
            return out

        before = prior.copy()
        before.columns = [str(c).lower() for c in before.columns]
        if "volume" not in before.columns:
            return out

        try:
            prior_vol = pd.to_numeric(before["volume"], errors="coerce").fillna(0)
            prior_vol = prior_vol[~prior_vol.index.duplicated(keep="last")]
            if float(prior_vol.sum()) <= 0:
                return out  # nothing worth preserving

            # Write the coerced series back first: a str/object volume column makes the
            # .loc assignment below raise, and the except would then silently preserve
            # nothing — the very failure this helper exists to prevent.
            new_vol = pd.to_numeric(out["volume"], errors="coerce").fillna(0)
            out["volume"] = new_vol
            carried = prior_vol.reindex(out.index)
            # Only fill where Yahoo gave us nothing and we genuinely had something.
            fill = (new_vol <= 0) & carried.notna() & (carried > 0)
            if bool(fill.any()):
                out.loc[fill, "volume"] = carried[fill]
                logger.info(
                    f"🔁 Yahoo fallback for {instrument_name}: preserved real volume on "
                    f"{int(fill.sum())}/{len(out)} bars (Yahoo reports 0 for indices)."
                )
        except Exception as exc:
            logger.debug(f"Yahoo volume preservation skipped for {instrument_name}: {exc}")
        return out

    _candle_cache_write_lock = threading.RLock()

    def _save_and_merge_candles(self, df: pd.DataFrame, cache_file: Path):
        with self._candle_cache_write_lock:
            return self._merge_candle_cache(df, cache_file)

    def _merge_candle_cache(self, df: pd.DataFrame, cache_file: Path):
        """Merge new candles with existing cache to prevent data loss and fill gaps"""
        if df.empty:
            return

        # Normalise BEFORE branching. This used to sit inside `if cache_file.exists()`,
        # on the premise recorded in _naive_kolkata_index's docstring -- "every live
        # producer already ends in .tz_localize(None)". fc0bf19 added a producer that
        # does not: kite_source.get_historical returns a tz-aware index (kiteconnect
        # hands back +0530 datetimes and data/kite_market_data.py never strips the
        # zone, unlike every Fyers path). A contract fetched through it for the FIRST
        # time skipped the only normalising branch and was written to CSV as
        # "2026-08-19 09:15:00+05:30".
        #
        # Those rows do not stay merely cosmetic. OptionHistoryStore concatenates every
        # contract into one bundle, and pd.concat over mixed naive/aware frames coerces
        # to UTC and drops the zone, so the loader returns 03:45:00 for a bar the CSV
        # records at 09:15:00 -- the whole series shifted 5h30m earlier.
        # engine/signal_processor.py:2098 and :2153 read option premium history through
        # exactly that loader, and it is the evidence base the exit-rule calibrations
        # are drawn from. 56 of 139 contract files were in this state when it was found.
        df = self._naive_kolkata_index(df)

        try:
            if cache_file.exists():
                try:
                    existing_df = self._naive_kolkata_index(
                        pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    )
                except pd.errors.EmptyDataError:
                    existing_df = pd.DataFrame()
                if not existing_df.empty:
                    for frame in (existing_df, df):
                        frame.columns = [c.lower() for c in frame.columns]
                    existing_df = existing_df[~existing_df.index.duplicated(keep='last')].sort_index()
                    df = df[~df.index.duplicated(keep='last')].sort_index()
                    if "volume" in existing_df.columns and "volume" in df.columns:
                        existing_vol = pd.to_numeric(existing_df["volume"], errors="coerce").fillna(0)
                        new_vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
                        shared_idx = df.index.intersection(existing_df.index)
                        preserve_idx = [
                            idx for idx in shared_idx
                            if new_vol.get(idx, 0) <= 0 and existing_vol.get(idx, 0) > 0
                            and not getattr(self, "upstox_only", False)
                        ]
                        if preserve_idx:
                            df.loc[preserve_idx, "volume"] = existing_vol.loc[preserve_idx]
                    # Drop duplicates keeping the LATEST data (important for updates!)
                    merged_df = pd.concat([existing_df, df])
                    merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
                    merged_df.sort_index(inplace=True)
                    # The disk cache is the replay archive. A 5,000-row cap kept
                    # only ~13 sessions of minute data and deleted a 30-session
                    # download on its next refresh. Keep 60 dated sessions, with
                    # the existing 5,000-row floor for less frequent candles.
                    session_dates = merged_df.index.normalize()
                    sessions = session_dates.unique()
                    if len(merged_df) > 5000 and len(sessions) > 60:
                        session_start = int(session_dates.searchsorted(sessions[-60]))
                        start = min(len(merged_df) - 5000, session_start)
                        merged_df = merged_df.iloc[start:]
                    material_cols = [
                        'open', 'high', 'low', 'close', 'volume', 'tick_count',
                        'volume_basis', 'source', 'provider_time', 'price_source',
                    ]
                    compare_cols = [
                        c for c in material_cols
                        if c in existing_df.columns and c in merged_df.columns
                    ]
                    telemetry_schema_unchanged = all(
                        c in existing_df.columns
                        for c in ('tick_count', 'volume_basis', 'source', 'provider_time', 'receipt_at')
                        if c in merged_df.columns
                    )
                    touched = df.index.intersection(merged_df.index)
                    if (
                        compare_cols
                        and telemetry_schema_unchanged
                        and len(merged_df) == len(existing_df)
                        and len(merged_df.index.difference(existing_df.index)) == 0
                        and existing_df.loc[touched, compare_cols].equals(merged_df.loc[touched, compare_cols])
                    ):
                        logger.debug(f"Skipping unchanged candle cache write for {cache_file.name}")
                        # The in-memory response still carries this request's exact
                        # receipt_at. It is transport telemetry, not a material reason
                        # to rewrite thousands of unchanged candle rows every 45s.
                        return merged_df
                    df = merged_df
                    
            # Save the full merged dataset
            with tempfile.NamedTemporaryFile(dir=cache_file.parent, suffix=".tmp", delete=False) as temporary:
                staged = Path(temporary.name)
            try:
                df.to_csv(staged)
                staged.replace(cache_file)
            finally:
                staged.unlink(missing_ok=True)
            get_option_history_store().invalidate(cache_file)
            logger.info(f"💾 Merged and saved data to {cache_file.name} (Total candles: {len(df)})")
        except Exception as e:
            logger.error(f"❌ Failed to merge/save candles to {cache_file.name}: {e}")
        return df

    def chain_matches_execution_expiry(self, index_symbol, chain):
        """Revalidate contract identity even while the transport cache is young."""
        from engine.expiry_manager import expiry_manager
        actual = str(chain.attrs.get("expiry") or "") if chain is not None else ""
        key = "active_monthly" if index_symbol in {"BANKNIFTY", "MIDCPNIFTY"} else "active_weekly"
        expected = expiry_manager.effective_expiry(index_symbol, key)
        if not actual or not expected:
            return not getattr(self, "upstox_only", False)
        actual_date = pd.to_datetime(actual, errors="coerce")
        expected_date = pd.to_datetime(expected, errors="coerce")
        return bool(pd.notna(actual_date) and pd.notna(expected_date) and actual_date.date() == expected_date.date())

    def get_cached_option_chain(
        self,
        index_symbol: str,
        max_age_seconds: float = 10.0,
    ) -> Optional[pd.DataFrame]:
        """Return only a fresh, active-expiry Fyers chain without network I/O."""
        cache_key = f"opt_chain_{index_symbol}"
        cached = (getattr(self, "_opt_chain_cache", {}) or {}).get(cache_key)
        meta = (getattr(self, "_option_chain_meta", {}) or {}).get(index_symbol, {})
        if cached is None or not isinstance(meta, dict):
            return None
        if not self.chain_matches_execution_expiry(index_symbol, cached):
            return None
        allowed = {"fyers", "kite", "angelone"} if self._replacement_enabled() else {"fyers"}
        if getattr(self, "upstox_only", False):
            allowed = {"upstox"}
        if str(meta.get("source") or "").lower() not in allowed or not bool(meta.get("expiry_aligned", False)):
            return None
        fetched_at = float(meta.get("fetched_at") or 0.0)
        age = time.time() - fetched_at if fetched_at > 0 else float("inf")
        if not (0.0 <= age <= max(0.0, float(max_age_seconds or 0.0))):
            return None
        if self._replacement_enabled() and str(meta.get("source")) not in {"fyers", "upstox"}:
            clocks = [self._replacement_epoch(value) for column in ("call_quote_at", "put_quote_at")
                      for value in cached.get(column, pd.Series([None] * len(cached))).tolist()]
            if not clocks or any(stamp is None or not 0 <= time.time()-stamp <= max_age_seconds for stamp in clocks):
                return None
        return cached

    # Only fire the Kite overlay when Fyers genuinely failed to supply OI. Load stays
    # split by default — Fyers carries the chain structure and premiums, Kite carries
    # execution and the tick stream — so neither broker is hammered while the other
    # idles. The overlay is demand-driven, not a second unconditional chain fetch.
    _KITE_OI_OVERLAY_MIN_INTERVAL = 60.0

    @staticmethod
    def _chain_volume_is_usable(chain: "pd.DataFrame") -> bool:
        """Every option leg must retain a measured, finite, nonnegative volume."""
        if chain is None or chain.empty:
            return False
        for side in ("call", "put"):
            column = side + "_volume"
            if column not in chain:
                return False
            values = pd.to_numeric(chain[column], errors="coerce")
            if not (np.isfinite(values) & values.ge(0)).fillna(False).all():
                return False
            available = column + "_available"
            if available in chain and not chain[available].eq(True).fillna(False).all():
                return False
        return True

    def _chain_oi_is_usable(self, chain: "pd.DataFrame") -> bool:
        if chain is None or chain.empty:
            return False
        try:
            calls = pd.to_numeric(chain.get("call_oi", 0), errors="coerce").fillna(0).sum()
            puts = pd.to_numeric(chain.get("put_oi", 0), errors="coerce").fillna(0).sum()
        except Exception:
            return False
        return bool(calls > 0 and puts > 0)

    def _augment_chain_oi_from_kite(
        self, index_symbol: str, chain: "pd.DataFrame", *, expiry: str = ""
    ) -> "pd.DataFrame":
        """Fill open interest from Kite when the Fyers chain has none.

        The 2026-08-01 startup logged `PCR=0, Call Delta=0` against a 123-row Fyers
        chain: the rows arrived, the OI did not, and PCR silently degraded to zero
        rather than reporting itself unavailable. Kite is the broker actually holding
        the position, so its OI is authoritative — but it is only queried when the
        primary source has already failed, keeping the two feeds balanced.
        """
        if self._chain_oi_is_usable(chain):
            return chain
        source = getattr(self, "kite", None)
        if source is None or not source.is_connected:
            return chain
        # Callers pass the Fyers symbol ("NSE:NIFTY50-INDEX"); Kite needs the index root.
        index = FYERS_SYMBOL_TO_INDEX.get(str(index_symbol), str(index_symbol)).upper()
        now = time.time()
        if not hasattr(self, "_last_kite_oi_overlay"):
            self._last_kite_oi_overlay = {}
        if now - float(self._last_kite_oi_overlay.get(index, 0.0) or 0.0) < self._KITE_OI_OVERLAY_MIN_INTERVAL:
            return chain
        self._last_kite_oi_overlay[index] = now
        try:
            spot = float(self._ltp_cache.get(index, {}).get("ltp", 0.0) or 0.0)
            # OI belongs to a contract, not merely a strike.  Resolve the same expiry
            # Fyers selected and pass it through to Kite; an absent/unparseable expiry
            # is safer than a default-nearest overlay during rollover.
            kite_expiry = None
            if expiry:
                for fmt in ("%d%b%Y", "%d-%m-%Y", "%Y-%m-%d"):
                    try:
                        kite_expiry = datetime.strptime(str(expiry), fmt).date()
                        break
                    except ValueError:
                        continue
            if kite_expiry is None:
                return chain
            try:
                kite_chain = source.get_option_chain(
                    index, spot=spot, strikes_each_side=20, expiry=kite_expiry
                )
            except TypeError as exc:
                # An adapter without expiry pinning cannot certify the overlay.
                # Preserve the primary chain without mixing another contract's OI.
                if "expiry" not in str(exc).lower():
                    raise
                return chain
            if kite_chain.empty:
                return chain
            returned_expiry = pd.to_datetime(kite_chain.attrs.get("expiry"), errors="coerce")
            if pd.isna(returned_expiry) or returned_expiry.date() != kite_expiry:
                return chain
            by_strike = {
                float(row["strike"]): row for _, row in kite_chain.iterrows()
            }
            # Use one OI provider for this observation, including uncovered strikes.
            # Mixing a few remaining primary values would corrupt PCR and deltas.
            chain = chain.copy()
            for side in ("call", "put"):
                chain[f"{side}_oi"] = 0.0
                chain[f"{side}_oi_change"] = 0.0
            filled = 0
            for position, strike in chain["strike"].items():
                row = by_strike.get(float(strike))
                if row is None:
                    continue
                chain.at[position, "call_oi"] = float(row["call_oi"])
                chain.at[position, "put_oi"] = float(row["put_oi"])
                for side in ("call", "put"):
                    chain.at[position, f"{side}_oi_change"] = 0.0
                # Carry the live mid so implied volatility is solved from a current
                # two-way price rather than a possibly-stale last trade.
                for side in ("call", "put"):
                    mid = float(row.get(f"{side}_mid") or 0.0)
                    if mid > 0:
                        chain.at[position, f"{side}_mid"] = mid
                filled += 1
            if filled:
                chain.attrs["oi_provider"] = "kite"
                logger.info(
                    f"Filled OI for {filled}/{len(chain)} {index} strikes from Kite "
                    "(Fyers chain returned none) — PCR is live again."
                )
        except Exception as exc:
            logger.warning(f"Kite OI overlay failed for {index}: {exc}")
        return chain

    def get_option_chain(self, index_symbol: str, exchange: str = "NFO") -> Optional[pd.DataFrame]:
        """Fetch a chain for the same active expiry used to select execution contracts."""
        now = time.time()
        cache_key = f"opt_chain_{index_symbol}"
        if not hasattr(self, "_opt_chain_cache"): self._opt_chain_cache = {}
        if not hasattr(self, "_last_opt_chain_fetch"): self._last_opt_chain_fetch = {}

        cached_chain = self._opt_chain_cache.get(cache_key)
        from engine.expiry_manager import expiry_manager
        from config.settings import get_settings
        expiry_type = "active_monthly" if index_symbol in {"BANKNIFTY", "MIDCPNIFTY"} else "active_weekly"
        active = expiry_manager.effective_expiry(index_symbol, expiry_type)
        if not active:
            return pd.DataFrame()
        expiry = pd.to_datetime(active).date().isoformat()
        ttl = max(10.0, float(getattr(get_settings(), "intelligence_cache_ttl_seconds", 30.0)))
        last_fetch = self._last_opt_chain_fetch.get(cache_key, 0.0)
        if (cached_chain is not None and cached_chain.attrs.get("expiry") == expiry
                and 0 <= now - last_fetch < ttl):
            return cached_chain

        client = getattr(self, "upstox", None)
        chain = client.get_option_chain_df(index_symbol, expiry) if client is not None else None
        if chain is None or chain.empty or chain.attrs.get("expiry") != expiry:
            return pd.DataFrame()
        observed = time.time()
        meta = {"source": "upstox", "oi_provider": "upstox", "fetched_at": observed,
                "active_expiry": expiry, "expiry_aligned": True, "fallback": False,
                "strike_count": len(chain), "exchange_clocks_fresh": False,
                "volume_complete": bool(chain[["call_volume_available", "put_volume_available"]].all().all())}
        chain.attrs.update(meta)
        self._opt_chain_cache[cache_key] = chain
        self._last_opt_chain_fetch[cache_key] = observed
        self._option_chain_meta[index_symbol] = meta
        self._archive_option_chain(index_symbol, chain, observed, meta)
        return chain

    def _watch_upstox_futures(self):
        from engine.expiry_manager import expiry_manager
        if not self._upstox_stream or not self.start_streams:
            return
        front = {str(row.get("current_fut_token")): (name, str(row.get("current_fut") or row["current_fut_token"]))
                 for name, row in expiry_manager.expiries.items() if row.get("current_fut_token")}
        self._upstox_front_futures = front
        for key in front:
            self._upstox_stream.watch(key)

    def fetch_instruments(self):
        from engine.expiry_manager import expiry_manager
        expiries = expiry_manager.pre_market_check()
        if getattr(self, "upstox_only", False):
            self._watch_upstox_futures()
        return expiries

    def get_token_for_symbol(self, symbol: str) -> str:
        """Resolve the exact current Upstox symbol for an archive or quote lookup."""
        from engine.expiry_manager import expiry_manager

        frame = expiry_manager.df
        if not getattr(self, "upstox_only", False) or frame is None:
            return ""
        matches = frame.loc[frame["symbol"].eq(symbol), "token"]
        return str(matches.iloc[0]) if len(matches) == 1 else ""

    def get_option_token(self, name: str, strike: float, option_type: str, trade_date: Optional[date] = None, now_dt=None) -> Optional[Dict]:
        from engine.expiry_manager import expiry_manager

        # Expiry Selection Logic:
        # BANKNIFTY/MIDCPNIFTY: Monthly (active_monthly handles T-2 rollover)
        # NIFTY/SENSEX: Always use Weekly (active_weekly handles 2-day rollover)
        if name in {"BANKNIFTY", "MIDCPNIFTY"}:
            expiry_type = "active_monthly"
        else:
            expiry_type = "active_weekly"

        if trade_date is not None:
            # now_dt (full timestamp) lets expiry-day gamma mode apply its intraday cutoff.
            dated = expiry_manager.get_token_for_date(name, strike, option_type, trade_date, expiry_type, now_dt=now_dt)
            if dated:
                return self._register_derivative_contract(name, dated)

        return self._register_derivative_contract(
            name, expiry_manager.get_token(name, strike, option_type, expiry_type))
