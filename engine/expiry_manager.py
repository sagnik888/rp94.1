import asyncio
import gzip
import requests
import pandas as pd
import json
import os
import tempfile
import threading
import time
import re
from datetime import datetime, date, time as dt_time, timedelta
from typing import Dict, List, Optional, Set
from loguru import logger
from pathlib import Path
from zoneinfo import ZoneInfo

class ExpiryManager:
    """
    Intelligent Expiry Resolver for AngelOne
    Resolves Weekly/Monthly expiries, handling holiday shifts and instrument specific rules.
    """
    
    TOKEN_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
    CACHE_DIR = Path("data_store")
    CACHE_FILE = CACHE_DIR / "instruments.json"
    OPTION_ROLLOVER_DAYS = 2  # legacy calendar constant; option rollover now uses
    # option_rollover_working_days (trading-day, holiday-aware) — see _trading_days_until.

    UPSTOX_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
    UPSTOX_CACHE_FILE = CACHE_DIR / "upstox_instruments.json"

    def __init__(self, *, upstox_only=False):
        self.upstox_only = upstox_only
        self.df: Optional[pd.DataFrame] = None
        self.expiries: Dict[str, Dict] = {}
        self.CACHE_DIR.mkdir(exist_ok=True)
        self._loaded_date = date.today()
        self._token_cache: Dict[tuple, Dict] = {}
        self._state_lock = threading.RLock()
        self._refresh_lock = threading.RLock()
        self._bg_download_pending = False
        self._refresh_thread: Optional[threading.Thread] = None
        self._staged_df: Optional[pd.DataFrame] = None
        self._snapshot_generation = 0

    @staticmethod
    def _kite_master_frame() -> Optional[pd.DataFrame]:
        """Kite's contract dump, reshaped into the AngelOne column schema.

        Everything downstream — expiry rolling, gamma windows, get_token, the strike
        fallback — is written against {name, instrumenttype, expiry, symbol, token,
        strike, lotsize}. Translating once here converts the entire contract pipeline
        to Kite without touching that logic.

        Two conventions are preserved deliberately:
          * `expiry` is the %d%b%Y string the expiry map parses ("04AUG2026").
          * `strike` is multiplied by 100, because get_token compares against
            float(strike) * 100 and the closest-strike fallback divides by 100.
        """
        try:
            from config.settings import get_settings
            cfg = get_settings()
            kite_key = str(getattr(cfg, "kite_api_key", "") or "").strip()
            if not kite_key:
                return None
        except Exception:
            return None

        try:
            from data.kite_auth import get_kite_auth
            from data.kite_market_data import KiteDataSource
        except Exception as exc:
            logger.error(f"Kite contract master unavailable (import failed): {exc}")
            return None

        auth = get_kite_auth()
        if auth.kite is None:
            logger.error(
                "Kite contract master unavailable: no session. "
                "Complete the daily login at /api/kite_login."
            )
            return None

        source = KiteDataSource(auth=auth)
        if not source.load_instruments(("NFO", "BFO")):
            return None

        rows = []
        with source._lock:
            dumps = {ex: list(rowset) for ex, rowset in source._instruments.items()}
        for exchange, instruments in dumps.items():
            for row in instruments:
                kind = row.get("instrument_type") or ""
                if kind in {"CE", "PE"}:
                    instrumenttype = "OPTIDX"
                elif kind == "FUT":
                    instrumenttype = "FUTIDX"
                else:
                    continue
                expiry = row.get("expiry")
                if not expiry:
                    continue
                rows.append({
                    "token": str(row.get("instrument_token") or ""),
                    "symbol": row.get("tradingsymbol") or "",
                    "name": row.get("name") or "",
                    "expiry": expiry.strftime("%d%b%Y").upper(),
                    "strike": float(row.get("strike") or 0.0) * 100.0,
                    "lotsize": int(row.get("lot_size") or 0),
                    "instrumenttype": instrumenttype,
                    "exch_seg": exchange,
                })
        if not rows:
            logger.error("Kite contract master produced no index derivatives.")
            return None
        frame = pd.DataFrame(rows)
        logger.success(f"Kite contract master staged: {len(frame)} index derivatives.")
        return frame

    # Kite-derived cache. Deliberately a different file from the AngelOne CACHE_FILE:
    # the two masters use different tradingsymbols and a different strike scale, so
    # mixing them silently resolves contracts the broker will refuse.
    KITE_CACHE_FILE = CACHE_DIR / "kite_instruments.json"

    # Retry pacing for a master we cannot build. Without a Kite session every attempt
    # fails identically, and _ensure_ready() is reached from the option-chain path on
    # every scan — which turned one real problem into a CRITICAL line every few
    # seconds. The cause needs a human (a browser login), so retrying faster than this
    # cannot help; the window is short enough to pick the session up promptly.
    MASTER_RETRY_COOLDOWN_SECONDS = 60.0

    @staticmethod
    def _normalize_upstox_rows(raw_rows):
        from data.upstox_market_data import UPSTOX_INDEX_KEYS

        rows = []
        ist = ZoneInfo("Asia/Kolkata")
        for row in raw_rows:
            name = row.get("underlying_symbol")
            segment = row.get("segment")
            kind = row.get("instrument_type")
            key = str(row.get("instrument_key") or "")
            if (name not in UPSTOX_INDEX_KEYS or row.get("underlying_key") != UPSTOX_INDEX_KEYS[name]
                    or segment not in {"NSE_FO", "BSE_FO"} or not key.startswith(segment + "|")
                    or kind not in {"CE", "PE", "FUT"} or not row.get("trading_symbol")):
                continue
            try:
                expiry = datetime.fromtimestamp(int(row["expiry"]) / 1000, ist).strftime("%d%b%Y").upper()
                lot_size = int(row["lot_size"])
                if lot_size <= 0:
                    continue
                rows.append({
                    "token": key, "symbol": row["trading_symbol"], "name": name, "expiry": expiry,
                    "strike": float(row.get("strike_price") or 0) * 100, "lotsize": lot_size,
                    "instrumenttype": "FUTIDX" if kind == "FUT" else "OPTIDX",
                    "option_type": kind, "tick_size": float(row["tick_size"]) / 100,
                    "exch_seg": "BFO" if segment == "BSE_FO" else "NFO",
                })
            except (KeyError, TypeError, ValueError, OverflowError, OSError):
                continue
        return pd.DataFrame(rows)

    def _download_upstox_master(self, force):
        cache = self.UPSTOX_CACHE_FILE
        try:
            if not force and datetime.fromtimestamp(cache.stat().st_mtime, ZoneInfo("Asia/Kolkata")).date() == datetime.now(ZoneInfo("Asia/Kolkata")).date():
                frame = pd.read_json(cache, orient="records")
                if not frame.empty and frame["token"].str.match(r"^(NSE|BSE)_FO\|[0-9]+$").all():
                    self._staged_df = frame
                    return True
        except (OSError, ValueError, KeyError, AttributeError):
            pass
        try:
            response = requests.get(self.UPSTOX_MASTER_URL, timeout=15)
            response.raise_for_status()
            data = response.content
            rows = json.loads(gzip.decompress(data) if data.startswith(b"\x1f\x8b") else data)
            frame = self._normalize_upstox_rows(rows)
        except (OSError, requests.RequestException, ValueError, TypeError, AttributeError) as exc:
            logger.warning(f"Upstox instrument master unavailable: {exc}")
            return False
        if frame.empty:
            logger.error("Upstox instrument master contains no supported index contracts")
            return False
        self._staged_df = frame
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache.parent, delete=False) as output:
                staged = Path(output.name)
                frame.to_json(output, orient="records")
            try:
                os.replace(staged, cache)
            finally:
                staged.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(f"Could not persist Upstox instrument master: {exc}")
        return True

    @staticmethod
    def _option_type_mask(frame, option_type):
        if "option_type" in frame:
            return frame["option_type"].eq(option_type)
        return frame["symbol"].str.endswith(option_type)

    def upstox_tick_size(self, key):
        contract = self.upstox_contract(key)
        return float(contract["tick_size"]) if contract and "tick_size" in contract else None

    def upstox_contract(self, key, *, require_current=False):
        with self._state_lock:
            frame = self.df
            loaded_date = self._loaded_date
        if frame is None or (require_current and loaded_date != datetime.now(ZoneInfo("Asia/Kolkata")).date()):
            return None
        matches = frame.loc[frame["token"].eq(key)]
        return matches.iloc[0].to_dict() if len(matches) == 1 else None

    def download_master(self, force=False):
        """Load a candidate master without exposing a half-analyzed snapshot."""
        with self._refresh_lock:
            blocked_until = float(getattr(self, "_master_failure_until", 0.0) or 0.0)
            if not force and time.monotonic() < blocked_until:
                return False
            if getattr(self, "upstox_only", False):
                loaded = self._download_upstox_master(force)
                self._master_failure_until = (0.0 if loaded else
                    time.monotonic() + self.MASTER_RETRY_COOLDOWN_SECONDS)
                return loaded
            # Check Kite master if available; otherwise fallback to AngelOne ScripMaster
            frame = self._kite_master_frame()
            if frame is not None and not frame.empty:
                self._master_failure_until = 0.0     # a live session clears the cooldown
                self._staged_df = frame
                try:
                    self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    self.KITE_CACHE_FILE.write_text(
                        frame.to_json(orient="records"), encoding="utf-8"
                    )
                except Exception as exc:
                    logger.debug(f"Could not cache the Kite master: {exc}")
                return True

            # Fallback to AngelOne ScripMaster (backup broker and public master)
            if self._retired_angelone_download(force=force):
                self._master_failure_until = 0.0
                return True

            # Same-day Kite cache only
            if self.KITE_CACHE_FILE.exists():
                cached_on = datetime.fromtimestamp(
                    self.KITE_CACHE_FILE.stat().st_mtime
                ).date()
                if cached_on == date.today():
                    try:
                        cached = pd.DataFrame(
                            json.loads(self.KITE_CACHE_FILE.read_text(encoding="utf-8"))
                        )
                        if not cached.empty:
                            logger.warning(
                                "Using today's cached Kite contract master (no live session)."
                            )
                            self._staged_df = cached
                            return True
                    except Exception as exc:
                        logger.error(f"Kite master cache unreadable: {exc}")
                else:
                    logger.error(
                        f"Kite contract cache is from {cached_on}; refusing to resolve "
                        "contracts from an expired master."
                    )

            self._master_failure_until = time.monotonic() + self.MASTER_RETRY_COOLDOWN_SECONDS
            logger.critical(
                "No contract master available (Kite session missing and AngelOne fallback failed). "
                f"(retrying at most every {self.MASTER_RETRY_COOLDOWN_SECONDS:.0f}s)"
            )
            return False

    def _retired_angelone_download(self, force=False):
        """Dead code kept only for reference; AngelOne is no longer a broker here."""
        if True:
            try:
                if not force and self.CACHE_FILE.exists():
                    file_time = datetime.fromtimestamp(self.CACHE_FILE.stat().st_mtime).date()
                    if file_time == date.today():
                        logger.info("Using cached instrument master.")
                        data = json.loads(self.CACHE_FILE.read_text(encoding="utf-8"))
                        self._staged_df = pd.DataFrame(data)
                        return not self._staged_df.empty

                logger.info("Downloading instrument master from AngelOne...")
                response = requests.get(self.TOKEN_URL, timeout=30)
                response.raise_for_status()
                data = response.json()
                candidate = pd.DataFrame(data)
                if candidate.empty:
                    raise ValueError("instrument master response was empty")
                self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
                fd, tmp_name = tempfile.mkstemp(
                    prefix="instruments_", suffix=".json", dir=str(self.CACHE_DIR)
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        json.dump(data, handle)
                    os.replace(tmp_name, self.CACHE_FILE)
                except Exception:
                    try:
                        os.unlink(tmp_name)
                    except Exception:
                        pass
                    raise
                self._staged_df = candidate
                logger.success("Instrument master downloaded and staged.")
                return True
            except Exception as exc:
                logger.error(f"Failed to download instruments: {exc}")
                csv_path = Path("data_store/instruments.csv")
                if csv_path.exists():
                    age_days = max(
                        0,
                        (datetime.now() - datetime.fromtimestamp(csv_path.stat().st_mtime)).days,
                    )
                    if age_days <= 7:
                        logger.warning(
                            f"Falling back to {csv_path} ({age_days} day(s) old)."
                        )
                        self._staged_df = pd.read_csv(csv_path)
                        return not self._staged_df.empty
                    logger.error(
                        f"Refusing stale fallback instrument CSV ({age_days} days old)."
                    )
                return False

    def analyze_expiries(self):
        """Analyze into locals, then atomically publish the complete snapshot."""
        with self._refresh_lock:
            source = self._staged_df if self._staged_df is not None else self.df
            if source is None or source.empty:
                logger.warning("No data to analyze expiries.")
                return {}
            frame = source.copy()
            resolved = self._build_expiry_map(frame)
            if not resolved:
                logger.error("Instrument master produced no usable expiry snapshot.")
                return {}
            with self._state_lock:
                self.df = frame
                self.expiries = resolved
                self._token_cache = {}
                self._loaded_date = date.today()
                self._snapshot_generation += 1
            self._staged_df = None
            return dict(resolved)

    def _build_expiry_map(self, frame: pd.DataFrame) -> Dict[str, Dict]:
        """Pure expiry calculation for one immutable master candidate."""
        required = {"name", "instrumenttype", "expiry", "symbol", "token"}
        missing = required.difference(frame.columns)
        if missing:
            logger.error(f"Instrument master missing required columns: {sorted(missing)}")
            return {}

        indices = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"]
        frame["expiry_dt"] = pd.to_datetime(
            frame["expiry"], format="%d%b%Y", errors="coerce"
        )
        resolved: Dict[str, Dict] = {}
        for name in indices:
            # 1. Filter Futures
            fut_df = frame[(frame['name'] == name) & (frame['instrumenttype'] == 'FUTIDX')].copy()
            fut_df = fut_df.sort_values('expiry_dt')
            
            # 2. Filter Options
            opt_df = frame[(frame['name'] == name) & (frame['instrumenttype'].isin(['OPTIDX']))].copy()
            opt_df = opt_df.sort_values('expiry_dt')
            
            all_expiries = sorted(opt_df['expiry_dt'].dropna().unique())
            now = pd.Timestamp(date.today())
            
            # Current/Nearest Expiry (Weekly)
            future_expiries = [d for d in all_expiries if d >= now]
            nearest_expiry = future_expiries[0] if len(future_expiries) > 0 else None
            next_weekly = future_expiries[1] if len(future_expiries) > 1 else nearest_expiry
            
            # Determine Monthly Expiries (Last expiry of each month)
            monthly_list = []
            for exp in future_expiries:
                m_list = [d for d in all_expiries if d.month == exp.month and d.year == exp.year]
                if m_list:
                    monthly_list.append(m_list[-1])
            
            unique_monthlies = sorted(list(set(monthly_list)))
            monthly_expiry = unique_monthlies[0] if len(unique_monthlies) > 0 else None
            next_monthly = unique_monthlies[1] if len(unique_monthlies) > 1 else monthly_expiry

            # Rollover in TRADING days (holiday-aware), not calendar days. Roll only on the
            # last trading session before expiry (+ expiry day) when threshold == 1.
            roll_days = self._rollover_working_days()
            today = now.date() if hasattr(now, "date") else now
            wdte_weekly = self._trading_days_until(today, nearest_expiry.date()) if nearest_expiry is not None else 10
            wdte_monthly = self._trading_days_until(today, monthly_expiry.date()) if monthly_expiry is not None else 10

            is_weekly_rollover = wdte_weekly <= roll_days
            is_monthly_rollover = wdte_monthly <= roll_days

            # 3. Determine Final Active Expiries based on rollover rules
            # NIFTY/SENSEX use weekly contracts and roll to next_weekly at T-2.
            final_weekly = next_weekly if is_weekly_rollover else nearest_expiry
            
            # BANKNIFTY/MIDCPNIFTY use monthly contracts and roll to next_monthly at T-2.
            final_monthly = next_monthly if is_monthly_rollover else monthly_expiry

            # 4. Futures Rollover (1 day before expiry rule)
            current_fut_expiry = fut_df.iloc[0]['expiry_dt'] if not fut_df.empty else None
            fut_dte = (current_fut_expiry - now).days if current_fut_expiry else 10
            is_fut_rollover = fut_dte <= 1
            
            fut_idx = 1 if (is_fut_rollover and len(fut_df) > 1) else 0
            if is_fut_rollover and len(fut_df) > 1:
                logger.info(f"🔄 Futures Rollover triggered for {name} (DTE: {fut_dte}). Using next month's contract.")

            # Results Mapping
            resolved[name] = {
                "weekly": nearest_expiry.strftime('%d%b%Y').upper() if nearest_expiry else None,
                "next_weekly": next_weekly.strftime('%d%b%Y').upper() if next_weekly else None,
                "monthly": monthly_expiry.strftime('%d%b%Y').upper() if monthly_expiry else None,
                "next_monthly": next_monthly.strftime('%d%b%Y').upper() if next_monthly else None,
                "active_weekly": final_weekly.strftime('%d%b%Y').upper() if final_weekly else None,
                "active_monthly": final_monthly.strftime('%d%b%Y').upper() if final_monthly else None,
                "current_fut": fut_df.iloc[fut_idx]['symbol'] if not fut_df.empty else None,
                "current_fut_token": fut_df.iloc[fut_idx]['token'] if not fut_df.empty else None,
                "is_monthly_week": nearest_expiry == monthly_expiry if nearest_expiry and monthly_expiry else False,
                "is_weekly_rollover": is_weekly_rollover,
                "is_monthly_rollover": is_monthly_rollover
            }
            
            logger.info(f"Resolved {name}: Weekly={resolved[name]['weekly']}, Monthly={resolved[name]['monthly']}")
            if is_weekly_rollover: logger.info(f"⚠️ {name} Weekly Rollover Active: Using {resolved[name]['active_weekly']}")
            if is_monthly_rollover: logger.info(f"⚠️ {name} Monthly Rollover Active: Using {resolved[name]['active_monthly']}")
        return resolved

    def _ensure_expiry_dates(self):
        if self.df is not None and "expiry_dt" not in self.df.columns:
            self.df["expiry_dt"] = pd.to_datetime(self.df["expiry"], format="%d%b%Y", errors="coerce")

    def _ensure_ready(self) -> bool:
        """Load and analyze the instrument master before token lookups."""
        needs_reload = (
            self.df is None
            or self.df.empty
            or not self.expiries
            or (hasattr(self, "_loaded_date") and date.today() != self._loaded_date)
        )
        if needs_reload:
            usable = self.df is not None and not self.df.empty and bool(self.expiries)
            if usable:
                self._start_background_refresh()
                return not getattr(self, "upstox_only", False)
            # Token/expiry lookups are called from the scanner's asyncio loop.
            # A cold cache must fail closed and warm in the background instead
            # of running the 30-second HTTP download on that latency-sensitive
            # loop.  Synchronous startup callers may still initialize eagerly.
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                self._start_background_refresh()
                return False
            return bool(self.pre_market_check())
        return self.df is not None and not self.df.empty and bool(self.expiries)

    def _start_background_refresh(self) -> bool:
        with self._refresh_lock:
            if self._bg_download_pending:
                return False
            self._bg_download_pending = True

        def _refresh() -> None:
            try:
                if self.download_master(force=True):
                    self.analyze_expiries()
            except Exception as exc:
                logger.warning(f"Background instrument master refresh failed: {exc}")
            finally:
                with self._refresh_lock:
                    self._bg_download_pending = False

        thread = threading.Thread(
            target=_refresh, daemon=True, name="InstrumentMasterRefresh"
        )
        self._refresh_thread = thread
        thread.start()
        return True

    _holiday_cache: Optional[Set[str]] = None
    _special_session_cache: Optional[Set[str]] = None

    @classmethod
    def _load_calendar(cls):
        if cls._holiday_cache is None:
            try:
                from data.market_calendar import load_nse_holidays, load_nse_special_trading_days
                cls._holiday_cache = load_nse_holidays()
                cls._special_session_cache = load_nse_special_trading_days()
            except Exception:
                cls._holiday_cache = set()
                cls._special_session_cache = set()
        return cls._holiday_cache, cls._special_session_cache

    @classmethod
    def _is_trading_day(cls, d: date) -> bool:
        hols, specials = cls._load_calendar()
        ds = d.strftime("%Y-%m-%d")
        if ds in specials:
            return True
        return d.weekday() < 5 and ds not in hols

    @classmethod
    def _trading_days_until(cls, start_date: date, end_date: date) -> int:
        """Number of trading sessions in (start_date, end_date] — holiday/weekend aware.
        0 when start_date is the expiry itself; 1 on the last trading day before it."""
        if end_date <= start_date:
            return 0
        count = 0
        d = start_date + timedelta(days=1)
        while d <= end_date:
            if cls._is_trading_day(d):
                count += 1
            d += timedelta(days=1)
        return count

    @staticmethod
    def _rollover_working_days() -> int:
        try:
            from config.settings import get_settings
            return int(getattr(get_settings(), "option_rollover_working_days", 1) or 1)
        except Exception:
            return 1

    @staticmethod
    def _parse_hhmm(val, default=dt_time(13, 0)) -> dt_time:
        try:
            h, m = str(val or "").split(":")
            return dt_time(int(h), int(m))
        except Exception:
            return default

    def expiry_day_override(self, current_expiry, next_expiry, target_date: date,
                            now_dt=None) -> Optional[str]:
        """Decide whether expiry-day gamma mode overrides the T-2 rollover.

        Pure function of settings + the bar's date/time (no side effects), so it is
        identical in live and backtest. Returns:
          None       -> no override; apply the normal T-2 rollover (unchanged behaviour)
          "current"  -> use the CURRENT (expiring) expiry for max gamma
          "futures"  -> after-cutoff 'futures' mode; caller should route to the index FUT
        """
        try:
            from config.settings import get_settings
            s = get_settings()
        except Exception:
            return None
        if not bool(getattr(s, "expiry_day_trade_current_week", False)):
            return None
        if current_expiry is None or next_expiry is None or current_expiry == next_expiry:
            return None
        try:
            if target_date > current_expiry.date():
                return None
            # Trading days to expiry (holiday-aware): 0 on expiry day, 1 on the last
            # trading session before it.
            dte = self._trading_days_until(target_date, current_expiry.date())
        except Exception:
            return None
        include_t1 = bool(getattr(s, "expiry_day_include_t1", False))
        if dte == 1:
            # Day before expiry: theta still mild, ride the gamma all session (no cutoff).
            return "current" if include_t1 else None
        if dte != 0:
            return None
        # Expiry day (T-0): morning = gamma; midday lull = after-cutoff mode; then the LATE
        # settlement window (default 14:00-15:15) re-engages gamma for the last-hour move.
        cutoff = self._parse_hhmm(getattr(s, "expiry_day_gamma_cutoff", "13:00"))
        now_t = getattr(now_dt, "time", None)
        now_t = now_t() if callable(now_t) else None
        if now_t is None or now_t < cutoff:
            return "current"
        if bool(getattr(s, "expiry_day_late_window", True)):
            lstart = self._parse_hhmm(getattr(s, "expiry_day_late_window_start", "14:00"), dt_time(14, 0))
            lend = self._parse_hhmm(getattr(s, "expiry_day_late_window_end", "15:15"), dt_time(15, 15))
            if lstart <= now_t < lend:
                return "current"
        mode = str(getattr(s, "expiry_day_after_cutoff_mode", "rollover") or "rollover").lower()
        if mode == "futures":
            return "futures"
        if mode == "momentum_current":
            # Keep current-week ATM; the system's existing momentum/quality gates are what
            # filter this down to big, short-tenure moves.
            return "current"
        return None  # "rollover" (safe default) -> normal next-week behaviour

    def in_gamma_window(self, name: str, now_dt=None) -> bool:
        """True when expiry-day gamma mode would have the system trading the CURRENT
        (expiring) expiry right now — used by the theta-crush risk controls (tighter SL,
        early time-exit, momentum gate). Short-circuits to False when the toggle is off so
        it costs nothing in the default configuration."""
        try:
            from config.settings import get_settings
            if not bool(getattr(get_settings(), "expiry_day_trade_current_week", False)):
                return False
        except Exception:
            return False
        try:
            now_dt = now_dt or datetime.now()
            td = now_dt.date()
            self._ensure_expiry_dates()
            # The expiry list depends only on (index, date, instrument master), yet this
            # filtered all ~168k master rows with pyarrow string compares and a per-row
            # .dt.date on every call — ~9% of per-index scan wall time (py-spy, live,
            # 2026-09-23). Cache it per day; a reloaded master is a new object and misses.
            cache = self.__dict__.setdefault("_gamma_expiry_cache", {})
            cache_key = (name, td, id(self.df), len(self.df))
            exps = cache.get(cache_key)
            if exps is None:
                opt = self.df[(self.df["name"] == name)
                              & (self.df["instrumenttype"].isin(["OPTIDX"]))
                              & (self.df["expiry_dt"].dt.date >= td)]
                exps = sorted(opt["expiry_dt"].dropna().unique())
                if name in {"BANKNIFTY", "MIDCPNIFTY"}:
                    monthlies = []
                    for e in exps:
                        same = [d for d in exps if d.month == e.month and d.year == e.year]
                        if same:
                            monthlies.append(same[-1])
                    exps = sorted(set(monthlies))
                if len(cache) > 32:
                    cache.clear()
                cache[cache_key] = exps
            if not exps:
                return False
            cur = pd.Timestamp(exps[0])
            nxt = pd.Timestamp(exps[1]) if len(exps) > 1 else cur
            return self.expiry_day_override(cur, nxt, td, now_dt) == "current"
        except Exception:
            return False

    def _select_option_expiry_for_date(self, name: str, target_date: date, expiry_type: str = "weekly",
                                       now_dt=None) -> Optional[pd.Timestamp]:
        self._ensure_expiry_dates()
        opt_df = self.df[
            (self.df["name"] == name)
            & (self.df["instrumenttype"].isin(["OPTIDX"]))
            & (self.df["expiry_dt"].dt.date >= target_date)
        ].copy()
        if opt_df.empty:
            return None

        all_expiries = sorted(opt_df["expiry_dt"].dropna().unique())
        if expiry_type in ("monthly", "active_monthly"):
            monthlies = []
            for exp in all_expiries:
                same_month = [d for d in all_expiries if d.month == exp.month and d.year == exp.year]
                if same_month:
                    monthlies.append(same_month[-1])
            candidates = sorted(set(monthlies))
        else:
            candidates = all_expiries

        if not candidates:
            return None

        current = candidates[0]
        next_expiry = candidates[1] if len(candidates) > 1 else current
        # Expiry-day gamma mode (gated, default off): may keep the CURRENT expiring expiry
        # during the T-2..T-0 window instead of rolling. "futures" is a routing hint handled
        # upstream, so here it falls through to the normal rollover (rolled next-week option).
        override = self.expiry_day_override(current, next_expiry, target_date, now_dt)
        if override == "current":
            return pd.Timestamp(current)
        wdte = self._trading_days_until(target_date, current.date())
        if wdte <= self._rollover_working_days() and next_expiry != current:
            return pd.Timestamp(next_expiry)
        return pd.Timestamp(current)

    def get_token_for_date(self, name: str, strike: float, option_type: str, target_date: date, expiry_type: str = "weekly", now_dt=None) -> Optional[Dict]:
        """Find an option token for the signal date, respecting weekly/monthly T-2 rollover.

        `now_dt` (the signal's full timestamp) is only used by expiry-day gamma mode to
        apply the intraday cutoff; it is part of the cache key (bucketed before/after the
        cutoff) so a pre-cutoff token is never reused after the cutoff and vice-versa."""
        if not self._ensure_ready():
            return None

        self._ensure_expiry_dates()
        # Resolve the expiry first (gamma mode may pick the current vs rolled expiry based on
        # now_dt), then key the cache on the resolved expiry so before/after-cutoff selections
        # never collide.
        expiry_dt = self._select_option_expiry_for_date(name, target_date, expiry_type, now_dt=now_dt)
        if expiry_dt is None:
            return None
        cache_key = (name, float(strike), option_type, target_date.isoformat(), expiry_type, expiry_dt.isoformat())
        if cache_key in self._token_cache:
            return self._token_cache[cache_key]

        opt_df = self.df[
            (self.df["name"] == name)
            & (self.df["instrumenttype"].isin(["OPTIDX"]))
            & self._option_type_mask(self.df, option_type)
            & (self.df["expiry_dt"] == expiry_dt)
        ].copy()
        if opt_df.empty:
            return None

        expiry_df = opt_df.copy()
        expiry_str = expiry_dt.strftime("%d%b%Y").upper()

        exact = expiry_df[expiry_df["strike"].astype(float) == float(strike) * 100]
        if not exact.empty:
            res = exact.iloc[0].to_dict()
            res["date_aware_expiry"] = expiry_str
            self._token_cache[cache_key] = res
            return res

        expiry_df["strike_val"] = expiry_df["strike"].astype(float) / 100.0
        expiry_df["diff"] = (expiry_df["strike_val"] - float(strike)).abs()
        best_match = expiry_df.loc[expiry_df["diff"].idxmin()]
        if not self._snap_is_sane(name, strike, best_match["strike_val"]):
            return None
        res = best_match.to_dict()
        res["date_aware_expiry"] = expiry_str
        self._token_cache[cache_key] = res
        logger.info(f"Using closest dated strike: {name} {expiry_str} {best_match['strike_val']} {option_type}")
        return res

    # Strike spacing per index, used ONLY to bound the nearest-strike fallback below.
    _STRIKE_INTERVAL = {"NIFTY": 50.0, "BANKNIFTY": 100.0, "SENSEX": 100.0, "MIDCPNIFTY": 25.0}
    # How far the fallback may reach. A missing intermediate strike is one or two steps
    # away; anything further means the REQUESTED value was wrong, not the chain.
    _MAX_STRIKE_SNAP_STEPS = 3

    def _snap_is_sane(self, name: str, requested: float, matched: float) -> bool:
        """Guard the nearest-strike fallback against a wrong-basis request.

        Both resolvers picked idxmin() on |listed - requested| with no bound, so ANY
        number resolved to a real, tradeable contract. A caller passing an option
        PREMIUM (~200) instead of a spot level (~24,700) got the LOWEST listed NIFTY
        strike -- deep ITM, premium in the thousands -- wrapped in a valid symbol and
        token, and the order pipeline proceeded normally.

        The fallback exists to absorb a missing intermediate strike. It must never
        absorb a wrong input.
        """
        try:
            step = self._STRIKE_INTERVAL.get(str(name or "").upper(), 50.0)
            gap = abs(float(matched) - float(requested))
        except (TypeError, ValueError):
            return False
        if gap <= step * self._MAX_STRIKE_SNAP_STEPS:
            return True
        logger.error(
            f"Strike snap REFUSED for {name}: requested {requested} but nearest listed is "
            f"{matched}, {gap:.0f} away ({step:.0f} spacing). That is not a missing "
            f"strike, it is a wrong request -- returning no token rather than buying "
            f"the wrong contract."
        )
        return False

    def effective_expiry(self, name: str, expiry_type: str = "active_weekly"):
        """One contract-expiry authority for execution and option-chain intelligence."""
        value = self.expiries.get(name, {}).get(expiry_type)
        if self.in_gamma_window(name):
            base_type = "weekly" if "weekly" in str(expiry_type) else "monthly"
            value = self.expiries.get(name, {}).get(base_type) or value
        return value

    def get_token(self, name: str, strike: float, option_type: str, expiry_type: str = "weekly") -> Optional[Dict]:
        """Find the specific token for an option strike.

        This is the LIVE (non-dated) path. It normally uses the rolled `active_weekly`/
        `active_monthly` expiry, but expiry-day gamma mode (gated) swaps to the CURRENT
        (expiring) expiry while in the gamma window — judged against wall-clock time, which
        is correct for live. (Backtest uses get_token_for_date with the bar timestamp.)"""
        if not self._ensure_ready(): return None

        expiry_val = self.effective_expiry(name, expiry_type)
        if not expiry_val: return None

        # Cache keyed on the RESOLVED expiry so gamma (current) and normal (rolled) tokens
        # never collide as the window opens/closes intraday.
        cache_key = (name, float(strike), option_type, expiry_type, expiry_val)
        if cache_key in self._token_cache:
            return self._token_cache[cache_key]

        logger.debug(f"Searching for {name} {expiry_type} ({expiry_val}) strike {strike} ({float(strike)*100})")
        
        nifty_opt = self.df[(self.df['name'] == name) & (self.df['expiry'] == expiry_val)]
        
        # Exact match logic
        match = self.df[
            (self.df['name'] == name) & 
            (self.df['expiry'] == expiry_val) & 
            (self.df['strike'].astype(float) == float(strike) * 100) & # AngelOne uses strike * 100
            self._option_type_mask(self.df, option_type)
        ]
        
        if not match.empty:
            res = match.iloc[0].to_dict()
            self._token_cache[cache_key] = res
            return res
            
        # Fallback: Find closest available strike
        if not nifty_opt.empty:
            logger.warning(f"Exact strike {strike} not found for {name} on {expiry_val}. Finding closest available...")
            # Filter by option type too to avoid mismatch
            opt_type_filtered = nifty_opt[self._option_type_mask(nifty_opt, option_type)].copy()
            if not opt_type_filtered.empty:
                opt_type_filtered['strike_val'] = opt_type_filtered['strike'].astype(float) / 100.0
                opt_type_filtered['diff'] = (opt_type_filtered['strike_val'] - float(strike)).abs()
                best_match = opt_type_filtered.loc[opt_type_filtered['diff'].idxmin()]
                if not self._snap_is_sane(name, strike, best_match['strike_val']):
                    return None
                res = best_match.to_dict()
                self._token_cache[cache_key] = res
                logger.info(f"Using closest available strike: {best_match['strike_val']}")
                return res
                
        return None

    def is_expiry_day(self, name: str, target_date: Optional[date] = None) -> bool:
        """Returns True if target_date is the weekly/monthly expiry day for the instrument"""
        dt = target_date or date.today()

        if self.df is not None and not self.df.empty:
            self._ensure_expiry_dates()
            matches = self.df[
                (self.df["name"] == name)
                & (self.df["instrumenttype"].isin(["OPTIDX"]))
                & (self.df["expiry_dt"].dt.date == dt)
            ]
            if not matches.empty:
                return True

        # Check the nearest unrolled contract, not the active T-2 contract.
        if self.expiries and name in self.expiries:
            exp_data = self.expiries[name]
            dt_str = dt.strftime('%d%b%Y').upper()
            expiry_key = "monthly" if name in {"BANKNIFTY", "MIDCPNIFTY"} else "weekly"
            return dt_str == exp_data.get(expiry_key)
                
        # Fallback to hardcoded defaults only if fetch failed
        if name == "NIFTY":
            return dt.weekday() == 3 # Thursday
        elif name == "SENSEX":
            return dt.weekday() == 4 # Friday
        elif name == "BANKNIFTY":
            return dt.weekday() == 2 # Wednesday
        elif name == "MIDCPNIFTY":
            return dt.weekday() == 1 # Tuesday
        return False

    def get_dte(self, name: str, target_date: Optional[date] = None) -> int:
        """Get Days To Expiry (DTE) for the given instrument on a specific date"""
        dt = target_date or date.today()
        expiry_type = "active_monthly" if name in {"BANKNIFTY", "MIDCPNIFTY"} else "active_weekly"
        exp_dt = None
        if target_date is not None and self.df is not None and not self.df.empty:
            selected = self._select_option_expiry_for_date(name, target_date, expiry_type)
            if selected is not None:
                exp_dt = selected.date()
        if exp_dt is None:
            exp_str = (
                self.expiries.get(name, {}).get(expiry_type)
                or self.expiries.get(name, {}).get("weekly")
            )
            if not exp_str:
                return 3
            try:
                exp_dt = datetime.strptime(exp_str, "%d%b%Y").date()
            except Exception:
                return 3
        diff = (exp_dt - dt).days
        return max(0, diff)

    def pre_market_check(self, force: bool = False):
        """Main entry point for daily pre-market analysis"""
        usable = self.df is not None and not self.df.empty and bool(self.expiries)
        stale = date.today() != getattr(self, "_loaded_date", date.today())
        if usable and stale and not force:
            self._start_background_refresh()
            return dict(self.expiries)
        if usable and not stale and not force:
            return dict(self.expiries)
        # Only announce the analysis when it is actually about to do work. This is
        # reached from the per-scan option-chain path, so logging unconditionally
        # printed the banner every few seconds while nothing happened.
        logger.info("═══ PRE-MARKET EXPIRY ANALYSIS ═══")
        if self.download_master(force=force):
            return self.analyze_expiries()
        return {}

def expiry_policy_signature():
    """Invalidate local expiry answers on rollover, gamma or calendar changes."""
    from config.settings import get_settings
    settings = get_settings()
    keys = ("option_rollover_working_days", "expiry_day_trade_current_week", "expiry_day_include_t1",
            "expiry_day_gamma_cutoff", "expiry_day_late_window", "expiry_day_late_window_start",
            "expiry_day_late_window_end", "expiry_day_after_cutoff_mode")
    holidays, special = ExpiryManager._load_calendar()
    return tuple(getattr(settings, key, None) for key in keys), frozenset(holidays), frozenset(special)


class ArchivedExpiryContext:
    """Local contract identity for replay; never loads or changes the live master.

    Dated weekly symbols identify an expired contract without a current login.
    Monthly symbols require explicit expiry metadata. No weekday is invented.
    """

    def __init__(self, data_dir=Path("data_store")):
        self.data_dir = Path(data_dir).resolve()
        self._signature = None
        self._expiries = {}
        self._resolution_cache = {}
        self._lock = threading.RLock()

    @staticmethod
    def _dated_symbol_expiry(name, symbol):
        suffix = str(symbol).upper()[len(name):]
        try:
            # DDMMMYY + strike is ambiguous with Kite's YYMMM + strike
            # without a separately verified strike. Use master metadata for
            # those names; only the compact dated Kite form is unambiguous.
            match = re.fullmatch(r"(\d{2})([1-9OND])(\d{2})\d+(CE|PE)", suffix)
            if match:
                month = {"O": 10, "N": 11, "D": 12}.get(match[2])
                return date(2000 + int(match[1]), month or int(match[2]), int(match[3]))
        except (ValueError, TypeError):
            pass
        return None

    def _load(self):
        paths = [self.data_dir / "kite_instruments.json", self.data_dir / "instruments.json", self.data_dir / "candles"]
        signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else (str(p), None, None) for p in paths)
        with self._lock:
            if signature == self._signature:
                return
            names = ("BANKNIFTY", "MIDCPNIFTY", "NIFTY", "SENSEX")
            symbols = {}
            parsed_expiries = {}
            for path in paths[:2]:
                if not path.exists():
                    continue
                try:
                    rows = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(rows, list):
                        continue
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        symbol = str(row.get("symbol") or row.get("tradingsymbol") or "").upper()
                        name = str(row.get("name") or "").upper()
                        if name not in names or not symbol.startswith(name) or not symbol.endswith(("CE", "PE")):
                            continue
                        value = row.get("expiry")
                        key = str(value)
                        if key not in parsed_expiries:
                            expiry = pd.to_datetime(value, format="%d%b%Y", errors="coerce")
                            if pd.isna(expiry):
                                expiry = pd.to_datetime(value, errors="coerce")
                            parsed_expiries[key] = expiry
                        expiry = parsed_expiries[key]
                        if not pd.isna(expiry):
                            symbols[(name, symbol)] = (expiry.date(), "archived_master")
                except (OSError, ValueError, TypeError):
                    continue
            if paths[2].exists():
                for path in paths[2].glob("*.csv"):
                    symbol = re.sub(r"_\d+min$", "", path.stem).upper()
                    name = next((n for n in names if symbol.startswith(n)), None)
                    if name:
                        expiry = self._dated_symbol_expiry(name, symbol)
                        if expiry is not None:
                            symbols[(name, symbol)] = (expiry, "archived_dated_symbol")
            expiries = {name: {} for name in names}
            for (name, _), (expiry, source) in symbols.items():
                expiries[name][expiry] = source
            self._expiries = expiries
            self._signature = signature
            self._resolution_cache.clear()

    def resolve(self, name, as_of, *, contract_expiry=None, contract_verified=False):
        unknown = {"available": False, "contract_expiry": None, "days_to_expiry": None,
                   "source": "unavailable", "is_expiry_day": False, "in_gamma_window": False}
        try:
            clock = pd.Timestamp(as_of)
            if pd.isna(clock):
                return unknown
            if clock.tzinfo is not None:
                clock = clock.tz_convert("Asia/Kolkata").tz_localize(None)
            day = clock.date()
            if contract_expiry is not None:
                expiry = pd.Timestamp(contract_expiry)
                if not contract_verified or pd.isna(expiry) or expiry.date() < day:
                    return unknown
                return {**unknown, "available": True, "contract_expiry": expiry.date().isoformat(),
                        "days_to_expiry": (expiry.date() - day).days, "source": "verified_option_contract",
                        "is_expiry_day": expiry.date() == day}
            self._load()
            name = str(name or "").upper()
            key = (name, clock, self._signature, expiry_policy_signature())
            with self._lock:
                cached = self._resolution_cache.get(key)
                if cached is not None:
                    return dict(cached)
            sources = self._expiries.get(name, {})
            expiries = sorted(exp for exp in sources if exp >= day)
            if not expiries:
                return unknown
            resolver = ExpiryManager.__new__(ExpiryManager)
            resolver.df = pd.DataFrame({"name": [name] * len(expiries), "instrumenttype": ["OPTIDX"] * len(expiries), "expiry_dt": pd.to_datetime(expiries)})
            selected = resolver._select_option_expiry_for_date(
                name, day, "active_monthly" if name in {"BANKNIFTY", "MIDCPNIFTY"} else "active_weekly", now_dt=clock)
            if selected is None:
                return unknown
            expiry = selected.date()
            result = {"available": True, "contract_expiry": expiry.isoformat(), "days_to_expiry": (expiry - day).days,
                    "source": sources[expiry], "is_expiry_day": day in sources,
                    "in_gamma_window": resolver.in_gamma_window(name, clock)}
            with self._lock:
                if len(self._resolution_cache) >= 256:
                    self._resolution_cache.clear()
                self._resolution_cache[key] = dict(result)
            return result
        except (ValueError, TypeError, OverflowError):
            return unknown


# Singleton for live execution only; historical callers use ArchivedExpiryContext.
expiry_manager = ExpiryManager(upstox_only=True)
