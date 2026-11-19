"""
UT1 Index Trading System — Main Entry Point
═══════════════════════════════════════════════════════════════
Run: python main.py
Dashboard: http://localhost:8001
"""

import sys
import asyncio
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, time as dt_time, timedelta
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# NumPy and SciPy each may reserve a full native worker pool at import time. UT1
# operates on short intraday frames and already parallelizes across instruments;
# large BLAS pools add thread stacks and contention without helping this workload.
# Keep an explicit environment override for research jobs needing wider matrices.
try:
    _numeric_threads = max(1, min(8, int(os.getenv("UT1_NUMERIC_THREADS", "1"))))
except (TypeError, ValueError):
    _numeric_threads = 1
for _thread_env in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, str(_numeric_threads))

import uvicorn
from loguru import logger

from config.settings import get_settings
from data.market_data import MarketDataProvider
from data.candle_builder import CandleBuilder
from engine.multi_timeframe import MultiTimeframeEngine
from engine.signal_manager import SignalManager
from intelligence.intelligence_aggregator import IntelligenceAggregator
from trading.trade_manager import TradeManager, IST
from trading.performance_tracker import PerformanceTracker
from trading.broker import SignalSimBroker
from scanner import Scanner
from notifications import send_desktop_notification
from dashboard.server import app, broadcast
from engine.safe_mode import validate_real_mode_readiness
from engine.runtime_mode import RuntimeMode
import dashboard.server as dashboard_module

# Configure logging
app_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
file_log_level = os.getenv("LOG_FILE_LEVEL", app_log_level).upper()
_TRUTHY_ENV = {"1", "true", "yes", "on"}
_FALSEY_ENV = {"0", "false", "no", "off"}
RUNNING_UNDER_PYTEST = "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ
FILE_LOG_ENABLED = (
    os.getenv("UT1_DISABLE_FILE_LOG", "").strip().lower() not in _TRUTHY_ENV
    and (
        not RUNNING_UNDER_PYTEST
        or os.getenv("UT1_ENABLE_TEST_FILE_LOGS", "").strip().lower() in _TRUTHY_ENV
    )
)
# The launcher tees STDERR to a file so a MemoryError traceback or a Windows fatal
# message outlives the console window. That redirect also swallowed every log line:
# the console went silent right after "Starting Backend Server" while the system was
# in fact scanning normally. Console logging goes to STDOUT, so the operator sees it;
# raw interpreter stderr still lands in the file for the post-mortem.
for _console_stream in (sys.stdout, sys.stderr):
    try:
        _console_stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        # Not a real console (pytest capture, pipe wrapper). Nothing to widen.
        pass
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <level>{message}</level>",
    level=app_log_level,
)
if FILE_LOG_ENABLED:
    logger.add(
        os.getenv("UT1_LOG_FILE", "logs/ut1.log"),
        rotation="00:00",
        retention="3 months",
        level=file_log_level,
    )

def start_angelone_log_cleaner():
    def _cleaner_loop():
        import glob
        import zipfile
        import shutil
        import datetime
        # Folders that failed removal because a file inside is still open
        # (e.g. the AngelOne SDK logger keeps yesterday's app.log handle until
        # restart). Warn ONCE per folder, then skip quietly each hour instead
        # of raising WinError 32 noise all night; retried after next restart.
        locked_skip: set = set()
        while True:
            try:
                now = datetime.datetime.now(IST)
                # Do not run heavy zip operations during active market hours (09:00 to 15:45)
                market_start = datetime.time(9, 0)
                market_end = datetime.time(15, 45)

                if not (market_start <= now.time() <= market_end):
                    today_str = now.strftime("%Y-%m-%d")
                    logs_dir = os.path.join(os.getcwd(), "logs")
                    for folder_path in glob.glob(os.path.join(logs_dir, "202*")):
                        if os.path.isdir(folder_path):
                            folder_name = os.path.basename(folder_path)
                            # Ensure it's a valid date folder and older than today
                            if len(folder_name) == 10 and folder_name < today_str:
                                if folder_path in locked_skip:
                                    continue
                                try:
                                    app_log_path = os.path.join(folder_path, "app.log")
                                    if os.path.exists(app_log_path):
                                        zip_name = f"smartconnect_{folder_name}.log.zip"
                                        zip_path = os.path.join(logs_dir, zip_name)
                                        # Run compression in a separate process to avoid GIL-locking Uvicorn for minutes
                                        import subprocess
                                        import sys
                                        zip_script = f"import zipfile\nwith zipfile.ZipFile(r'{zip_path}', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:\n    zipf.write(r'{app_log_path}', arcname=r'app_{folder_name}.log')"
                                        subprocess.run([sys.executable, "-c", zip_script], check=True)
                                    shutil.rmtree(folder_path)
                                    logger.info(f"Compressed legacy AngelOne log folder: {folder_name}")
                                except subprocess.CalledProcessError as sub_err:
                                    logger.warning(f"Failed to compress log folder {folder_name}: {sub_err}")
                                except (PermissionError, OSError) as lock_err:
                                    locked_skip.add(folder_path)
                                    logger.warning(
                                        f"Log folder {folder_name} is still held open "
                                        f"(likely the SDK logger); skipping until next restart: {lock_err}"
                                    )

                    # Database Expiry Pruning
                    import calendar
                    from data.sqlite_db import DatabaseManager
                    
                    def get_last_thursday(year, month):
                        month_cal = calendar.monthcalendar(year, month)
                        if month_cal[-1][calendar.THURSDAY] != 0:
                            return datetime.date(year, month, month_cal[-1][calendar.THURSDAY])
                        else:
                            return datetime.date(year, month, month_cal[-2][calendar.THURSDAY])
                            
                    last_thursday = get_last_thursday(now.year, now.month)
                    
                    if now.date() >= last_thursday:
                        cleanup_marker = os.path.join(logs_dir, ".last_db_cleanup")
                        current_month_str = now.strftime("%Y-%m")
                        
                        last_cleaned_month = ""
                        if os.path.exists(cleanup_marker):
                            with open(cleanup_marker, "r") as f:
                                last_cleaned_month = f.read().strip()
                                
                        if last_cleaned_month != current_month_str:
                            logger.info(f"Triggering automated Expiry-Based DB Pruning for {current_month_str}...")
                            try:
                                db = DatabaseManager()
                                # Gate the marker on the prune actually succeeding.
                                # cleanup_old_records swallows its own exceptions, so
                                # writing the marker on a normal return marked the month
                                # "done" after a VACUUM that failed on a locked database,
                                # and every remaining hourly tick that month then skipped.
                                if db.cleanup_old_records(days=31):
                                    with open(cleanup_marker, "w") as f:
                                        f.write(current_month_str)
                                else:
                                    logger.warning(
                                        "Monthly DB prune did not complete; leaving the "
                                        "marker unset so the next hourly tick retries."
                                    )
                            except Exception as db_err:
                                logger.error(f"Failed DB Pruning: {db_err}")

            except Exception as e:
                logger.error(f"Error in AngelOne log cleaner: {e}")
            time.sleep(3600) # Check every 1 hour

    threading.Thread(target=_cleaner_loop, daemon=True, name="LogCleaner").start()


def _parse_hhmm(value: str, default: dt_time) -> dt_time:
    try:
        parts = str(value or "").strip().split(":")
        if len(parts) < 2:
            return default
        return dt_time(int(parts[0]), int(parts[1]))
    except Exception:
        return default


def _parse_hhmmss(value: str, default: dt_time) -> dt_time:
    try:
        parts = str(value or "").strip().split(":")
        if len(parts) < 2:
            return default
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) >= 3 else 0
        return dt_time(hour, minute, second)
    except Exception:
        return default


def _setting_enabled(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _FALSEY_ENV:
        return False
    if text in _TRUTHY_ENV:
        return True
    return default


def _ist_at(session_date, clock: dt_time) -> datetime:
    return IST.localize(datetime.combine(session_date, clock))


def _pid_is_running(pid: int) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            process_query_limited_information = 0x1000
            process = ctypes.windll.kernel32.OpenProcess(
                process_query_limited_information, False, int(pid)
            )
            if not process:
                return False
            ctypes.windll.kernel32.CloseHandle(process)
            return True
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _start_tracker_process(
    *,
    since: str,
    until: str,
    interval: int,
    out_dir: Path,
    pid_file: Path,
) -> subprocess.Popen | None:
    try:
        if pid_file.exists():
            try:
                existing_pid = int(pid_file.read_text(encoding="utf-8").strip() or "0")
            except Exception:
                existing_pid = 0
            if _pid_is_running(existing_pid):
                logger.info(f"Live session tracker already running with PID {existing_pid}.")
                return None
            pid_file.unlink(missing_ok=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "tracker_launcher.log"
        log_handle = log_path.open("a", encoding="utf-8")
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "scripts" / "live_session_tracker.py"),
            "--watch",
            "--since",
            since,
            "--until",
            until,
            "--interval",
            str(max(15, int(interval or 60))),
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).parent),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        pid_file.write_text(str(proc.pid), encoding="utf-8")
        logger.info(
            f"Live session tracker started PID {proc.pid}; "
            f"report since {since} until {until} IST; log={log_path}"
        )
        return proc
    except Exception as exc:
        logger.error(f"Failed to start live session tracker: {exc}")
        return None


def start_live_session_tracker_scheduler():
    """Start the read-only live-session bug/regression tracker for market hours.

    Defaults to 09:00-15:40 IST. The child tracker writes rolling JSON/markdown
    reports under runtime_logs/session_tracker and never touches order state.
    """
    enabled = str(os.getenv("LIVE_SESSION_TRACKER_ENABLED", "1")).strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        logger.info("Live session tracker scheduler disabled by LIVE_SESSION_TRACKER_ENABLED.")
        return

    start_at = _parse_hhmm(os.getenv("LIVE_SESSION_TRACKER_START", "09:00"), dt_time(9, 0))
    stop_at = _parse_hhmm(os.getenv("LIVE_SESSION_TRACKER_UNTIL", "15:40"), dt_time(15, 40))
    interval = int(os.getenv("LIVE_SESSION_TRACKER_INTERVAL", "60") or "60")
    root = Path(__file__).parent
    out_dir = root / "runtime_logs" / "session_tracker"
    pid_file = out_dir / "tracker.pid"

    def _scheduler_loop():
        while True:
            now = datetime.now(IST)
            today_start = IST.localize(datetime.combine(now.date(), start_at))
            today_stop = IST.localize(datetime.combine(now.date(), stop_at))
            if start_at <= now.time() <= stop_at:
                _start_tracker_process(
                    since=start_at.strftime("%H:%M"),
                    until=stop_at.strftime("%H:%M"),
                    interval=interval,
                    out_dir=out_dir,
                    pid_file=pid_file,
                )
                sleep_seconds = 60.0
            elif now.time() < start_at:
                sleep_seconds = max(5.0, (today_start - now).total_seconds())
            else:
                tomorrow_start = today_start + timedelta(days=1)
                sleep_seconds = max(60.0, (tomorrow_start - now).total_seconds())

            logger.info(
                f"Live session tracker scheduler sleeping {sleep_seconds:.0f}s "
                f"(window {start_at.strftime('%H:%M')}-{stop_at.strftime('%H:%M')} IST)."
            )
            time.sleep(sleep_seconds)

    threading.Thread(
        target=_scheduler_loop,
        daemon=True,
        name="LiveSessionTrackerScheduler",
    ).start()


def _scanner_liveness_alert_due(
    running: bool, seen_running: bool, elapsed: float, grace: float
) -> bool:
    """Whether a not-running scanner should be escalated.

    Split out of the watchdog loop so it can be tested directly: the loop sleeps a
    floor of 15s between iterations, so a test can never observe a SECOND poll inside
    a reasonable window, and the interesting cases here are all about the second poll
    onwards.

    Silent only while the scanner has never once been seen up and the startup grace is
    still open. A scanner that dies after coming up is reported immediately, which is
    the incident the watchdog exists for.
    """
    if running:
        return False
    return bool(seen_running or elapsed >= grace)


# The two broker-auth alerts that expire on a daily schedule rather than failing at a
# random moment. Only these are re-raised before the open; a mid-session incident still
# gets exactly one alert.
_PREOPEN_AUTH_ALERT_KEYS = ("upstox_token",)


def _preopen_auth_realert_due(
    now: datetime,
    *,
    preopen_start: dt_time,
    session_start: dt_time,
    last_realert_monotonic: float,
    now_monotonic: float,
    every_seconds: float,
) -> bool:
    """Whether a latched broker-auth alert should be raised again before the open.

    ``_alert`` fires once per incident, which is right for a mid-session failure and
    wrong for the daily token expiry. Kite invalidates its access token around 06:00 and
    the Fyers token expires at roughly the same hour, so on 2026-09-16 the operator's
    only Kite alert was delivered at 04:30 and the only Fyers alert at 06:05 -- both
    while they were asleep, and neither repeated. Both logins were finally completed at
    08:41, 34 minutes before the open, by noticing rather than by being told.

    Inside the pre-open window the latch is released on a bounded cadence so the alert
    lands at an hour when the login can still be finished before 09:15. Outside that
    window nothing changes.
    """
    if now.weekday() >= 5:
        return False
    if not (preopen_start <= now.time() < session_start):
        return False
    if last_realert_monotonic <= 0.0:
        return True
    return (now_monotonic - last_realert_monotonic) >= max(60.0, float(every_seconds))


def _accepted_unexecuted_count(scanner) -> int:
    """Signals the gates ACCEPTED this session that never reached the broker.

    Reads the live candidate book rather than the database so the watchdog keeps working
    when the ledger is mid-write. Any failure returns 0, which only costs the count in the
    alert text; the heartbeat still fires.
    """
    total = 0
    try:
        for book in (getattr(scanner, "_session_trade_candidates", {}) or {}).values():
            for candidate in (book or {}).values():
                if not bool(getattr(candidate, "accepted_by_gate", False)):
                    continue
                if getattr(candidate, "broker_order_id", None):
                    continue
                total += 1
    except Exception:
        return 0
    return total


def _auto_off_realert_due(
    *,
    accepted_unexecuted: int,
    last_count: int,
    last_realert_monotonic: float,
    now_monotonic: float,
    every_seconds: float,
) -> bool:
    """Whether the REAL-with-Auto-OFF alert should speak again.

    `_alert` de-duplicates per incident, which on 2026-09-22 meant exactly one CRITICAL at
    09:15:45 and then silence for the rest of the session. The market fell one-sided that
    day, the gates ACCEPTED nine signals -- four of them SHORTs during the fall at 11:00,
    11:35, 12:44 and 13:01 -- and every one was recorded "Accepted and displayed for manual
    execution" with executed=0. No order ever reached the broker and nothing said so again.

    So the alert repeats on two triggers: a new accepted-but-unexecuted signal (the moment
    it costs something), and a slow heartbeat while the condition persists.
    """
    if accepted_unexecuted > max(0, int(last_count)):
        return True
    # The heartbeat is OPT-IN (every_seconds <= 0 disables it). Shipped at 15 minutes on
    # 2026-09-23 it raised 19 desktop notifications between 09:15 and 13:44 on a session the
    # operator was running manually on purpose, with zero accepted signals to report —
    # noise that trains the operator to ignore the one alert that matters.
    if float(every_seconds) <= 0.0 or last_realert_monotonic <= 0.0:
        return False
    return (now_monotonic - last_realert_monotonic) >= max(60.0, float(every_seconds))


def _release_preopen_auth_latches(state: dict) -> list:
    """Drop the de-dup latches for the broker-auth alerts so they can fire again.

    Returns the keys that were actually latched, so a caller can tell a real re-raise
    from a no-op pass where both brokers were already authenticated.
    """
    released = []
    for key in _PREOPEN_AUTH_ALERT_KEYS:
        if state.pop(key, None):
            released.append(key)
    return released


def _apply_runtime_resource_guard(scanner) -> tuple[bool, str]:
    """Latch an entry-only inhibition without changing Auto exit semantics."""
    if (
        str(getattr(scanner, "mode", "") or "").upper() != "REAL"
        or not bool(getattr(scanner, "auto_mode", False))
    ):
        scanner._auto_resource_entry_inhibited = False
        scanner._auto_resource_entry_inhibit_reason = ""
        return False, ""
    try:
        blockers = list(scanner._auto_execution_resource_blockers() or [])
    except Exception as exc:
        blockers = [
            f"Auto resource guard failed to obtain a fresh sample ({exc}); "
            "new exposure is fail-closed"
        ]
    if not blockers:
        scanner._auto_resource_entry_inhibited = False
        scanner._auto_resource_entry_inhibit_reason = ""
        return False, ""

    scanner._auto_resource_entry_inhibited = True
    scanner._auto_resource_entry_inhibit_reason = str(blockers[0])
    return True, "New Auto entries inhibited; " + str(blockers[0])


def _runtime_resource_warning(scanner) -> str:
    """Percentage memory pressure that warns without blocking; '' when there is none.

    The percentages used to block new Auto entries and that block was the operator's
    only prompt to free memory. They now only warn (auto_resource_pct_blocks_entries),
    so the prompt has to survive on its own or the machine drifts toward the headroom
    floor, which does block, with nobody told.
    """
    if (
        str(getattr(scanner, "mode", "") or "").upper() != "REAL"
        or not bool(getattr(scanner, "auto_mode", False))
    ):
        return ""
    warnings = [str(w) for w in (getattr(scanner, "_auto_resource_pressure_warnings", None) or [])]
    if not warnings:
        return ""
    return (
        "Host memory pressure: " + "; ".join(warnings)
        + ". Entries still allowed; they stop only when Windows commit headroom runs out."
        " Close other trading builds and heavy apps now."
    )


def start_runtime_watchdog(scanner: Scanner, settings) -> None:
    """Alert on the silent failure modes that unattended operation cannot tolerate.

    Three real incidents motivated this:
      * the scanner thread died while uvicorn kept serving, so the process looked alive
        for hours with nothing scanning during market hours;
      * Auto Mode is not persisted, so a restart silently reverts to manual and NOTHING
        executes even though signals keep being produced;
      * the Fyers access token expires ~06:00 daily and cannot auto-refresh (broker
        returns code -16, "disabled to comply with SEBI regulations"), so the option
        chain quietly disappears and OI/PCR/Greeks go blank.

    Most checks are detection-only. Critical host memory/commit pressure is the narrow
    exception: it inhibits new Auto exposure while deliberately preserving the Auto
    flag used by software stops, targets, profit locks, and reconciliation.
    """
    if str(os.getenv("RUNTIME_WATCHDOG_ENABLED", "1")).strip().lower() in _FALSEY_ENV:
        logger.info("Runtime watchdog disabled by RUNTIME_WATCHDOG_ENABLED.")
        return

    stale_after = float(os.getenv("RUNTIME_WATCHDOG_STALE_SECONDS", "180") or 180.0)
    check_every = float(os.getenv("RUNTIME_WATCHDOG_INTERVAL_SECONDS", "60") or 60.0)
    start_at = _parse_hhmm(os.getenv("RUNTIME_WATCHDOG_START", "09:15"), dt_time(9, 15))
    stop_at = _parse_hhmm(os.getenv("RUNTIME_WATCHDOG_UNTIL", "15:30"), dt_time(15, 30))
    # Pre-open re-raise window for the daily broker-token alerts. Ends at start_at, so
    # moving RUNTIME_WATCHDOG_START moves the deadline with it.
    preopen_at = _parse_hhmm(
        os.getenv("RUNTIME_WATCHDOG_PREOPEN_START", "08:30"), dt_time(8, 30)
    )
    preopen_every = float(
        os.getenv("RUNTIME_WATCHDOG_PREOPEN_REALERT_SECONDS", "600") or 600.0
    )
    # Optional heartbeat for REAL-with-Auto-OFF, off by default. Every newly accepted but
    # unplaced signal re-raises the alert regardless; set this (seconds) only if you also
    # want reminders while nothing is being accepted.
    auto_off_every = float(
        os.getenv("RUNTIME_WATCHDOG_AUTO_OFF_REALERT_SECONDS", "0") or 0.0
    )

    def _alert(key: str, message: str, state: dict) -> None:
        """CRITICAL + desktop notification, de-duplicated per incident."""
        if state.get(key):
            return
        state[key] = True
        logger.critical(f"WATCHDOG: {message}")
        try:
            scanner.log_event(f"WATCHDOG: {message}", "error")
        except Exception:
            pass
        try:
            send_desktop_notification(f"UT1 WATCHDOG: {message}", "error")
        except Exception as exc:
            logger.debug(f"Watchdog desktop alert unavailable: {exc}")

    # The scanner thread does not set is_running until it has finished coming up, and
    # this loop runs its first check immediately. Restarting inside market hours
    # therefore fired "scanner is NOT running" every single time — observed on
    # 2026-08-07 at 10:02:32.033, the same MILLISECOND the scanner logged that it had
    # started. A CRITICAL plus a desktop alert on every routine restart is how an
    # operator learns to ignore CRITICALs, which is the opposite of what a watchdog
    # covering three real silent-failure incidents is for.
    #
    # A scanner that is genuinely dead on arrival is still caught: the grace only
    # defers the check, it never cancels it.
    startup_grace = float(
        os.getenv("RUNTIME_WATCHDOG_STARTUP_GRACE_SECONDS", "120") or 120.0
    )

    def _loop():
        state: dict = {}
        auto_mode_seen_on = False
        watchdog_started = time.monotonic()
        scanner_seen_running = False
        while True:
            try:
                now = datetime.now(IST)
                in_session = (
                    now.weekday() < 5 and start_at <= now.time() <= stop_at
                )

                # 1. Scanner liveness — the dashboard staying up is NOT proof of life.
                if in_session:
                    last_scan = getattr(scanner, "last_scan_time", None)
                    running = bool(getattr(scanner, "is_running", False))
                    if running:
                        scanner_seen_running = True
                    if _scanner_liveness_alert_due(
                        running,
                        scanner_seen_running,
                        time.monotonic() - watchdog_started,
                        startup_grace,
                    ):
                        _alert(
                            "dead", "scanner is NOT running during market hours", state
                        )
                    elif last_scan is not None:
                        try:
                            last = last_scan if last_scan.tzinfo else IST.localize(last_scan)
                            age = (now - last.astimezone(IST)).total_seconds()
                        except Exception:
                            age = 0.0
                        if age > stale_after:
                            _alert(
                                "stale",
                                f"no scan cycle for {age:.0f}s (threshold {stale_after:.0f}s)",
                                state,
                            )
                        else:
                            state.pop("stale", None)
                            state.pop("dead", None)

                # 2. Auto Mode silently reverting to OFF (it is never persisted).
                auto_now = bool(getattr(scanner, "auto_mode", False))
                if auto_now:
                    auto_mode_seen_on = True
                    state.pop("auto_off", None)
                    state.pop("auto_never_armed", None)
                elif auto_mode_seen_on and in_session:
                    _alert(
                        "auto_off",
                        "Auto Mode reverted to OFF during market hours - signals will "
                        "NOT execute until it is re-enabled",
                        state,
                    )
                elif in_session and str(
                    getattr(scanner, "mode", "") or ""
                ).upper() == "REAL":
                    # The restart case, which the latch above CANNOT see (2026-08-16).
                    # auto_mode_seen_on is per-process, so a process that restarted
                    # mid-session has never seen AUTO on and never trips the branch
                    # above — while the 09:00 premarket worker DOES re-arm REAL after a
                    # restart (completed_dates is in-memory). The result is a dashboard
                    # showing REAL with feeds green and signals printing, zero entries
                    # for the rest of the day, and not one CRITICAL raised. That is
                    # indistinguishable from a quiet market until the close.
                    #
                    # REAL + in-session + AUTO off is worth saying out loud even on a
                    # deliberately manual day: it costs one alert, and the alternative
                    # is a silent full-day outage. _alert de-duplicates on the state key.
                    # Repeat while it is still costing entries. One alert per process was
                    # not enough on 2026-09-22: it spoke at 09:15:45 and never again while
                    # nine accepted signals went unplaced through a one-sided fall.
                    accepted_unexecuted = _accepted_unexecuted_count(scanner)
                    if _auto_off_realert_due(
                        accepted_unexecuted=accepted_unexecuted,
                        last_count=int(state.get("auto_off_accepted_count", 0) or 0),
                        last_realert_monotonic=float(state.get("auto_off_realert_at", 0.0) or 0.0),
                        now_monotonic=time.monotonic(),
                        every_seconds=auto_off_every,
                    ):
                        state.pop("auto_never_armed", None)
                    state["auto_off_accepted_count"] = accepted_unexecuted
                    missed = (
                        f" {accepted_unexecuted} signal(s) have already been accepted and "
                        "displayed for manual execution instead of being placed."
                        if accepted_unexecuted
                        else ""
                    )
                    if not state.get("auto_never_armed"):
                        state["auto_off_realert_at"] = time.monotonic()
                    _alert(
                        "auto_never_armed",
                        "broker mode is REAL during market hours but Auto Mode is OFF - "
                        "signals will NOT execute. A restart brings Auto back only if it "
                        "was ON earlier today (after the day's readiness checks); "
                        "re-enable it on the dashboard if this session is meant to "
                        "trade unattended." + missed,
                        state,
                    )

                # 2a. Pre-open re-raise. Both broker tokens expire on a clock, not on
                # an event, so their alerts are raised hours before anyone is awake to
                # act. Releasing the latch here lets 2b and 3 below speak again while
                # there is still time to log in; if both brokers are healthy the pop is
                # a no-op and nothing is raised.
                if _preopen_auth_realert_due(
                    now,
                    preopen_start=preopen_at,
                    session_start=start_at,
                    last_realert_monotonic=float(state.get("preopen_realert_at", 0.0) or 0.0),
                    now_monotonic=time.monotonic(),
                    every_seconds=preopen_every,
                ):
                    state["preopen_realert_at"] = time.monotonic()
                    released = _release_preopen_auth_latches(state)
                    if released:
                        logger.info(
                            "Pre-open watchdog re-raising broker auth alert(s): "
                            + ", ".join(released)
                        )

                # 2b. Upstox session — daily 24h token check
                try:
                    from data.upstox_auth import get_upstox_auth_status
                    upstox_status = get_upstox_auth_status()
                except Exception as exc:
                    upstox_status = {"authenticated": False, "reason": f"status unavailable: {exc}"}
                if not upstox_status.get("authenticated"):
                    _alert(
                        "upstox_token",
                        f"Upstox login required ({upstox_status.get('reason') or 'no session'}) - "
                        f"Log in at http://{settings.dashboard_host}:{settings.dashboard_port}/api/upstox_login",
                        state,
                    )
                else:
                    state.pop("upstox_token", None)

                status_reader = None
                if RuntimeMode.parse(getattr(scanner, "mode", "")).permits_live_orders:
                    broker = getattr(getattr(scanner, "trades", None), "real_broker", None)
                    status_reader = getattr(broker, "execution_status", None)
                if callable(status_reader) and not status_reader().get("connected"):
                    _alert("angelone_execution", "Angel One execution session unavailable", state)
                else:
                    state.pop("angelone_execution", None)

                # 4. Host memory / Windows commit pressure.  Fail closed only for NEW
                # exposure. Do not change scanner.auto_mode: that same flag controls
                # software exits, so switching it off would strand them behind manual
                # confirmation while the machine is already under stress.
                pressured, pressure_message = _apply_runtime_resource_guard(scanner)
                if pressured:
                    _alert(
                        "resource_pressure",
                        pressure_message,
                        state,
                    )
                else:
                    state.pop("resource_pressure", None)
                warning = "" if pressured else _runtime_resource_warning(scanner)
                if warning:
                    _alert("resource_warning", warning, state)
                else:
                    state.pop("resource_warning", None)
            except Exception as exc:
                logger.debug(f"Runtime watchdog iteration failed: {exc}")
            time.sleep(max(15.0, check_every))

    threading.Thread(target=_loop, daemon=True, name="RuntimeWatchdog").start()
    logger.info(
        f"Runtime watchdog active (stale>{stale_after:.0f}s, "
        f"{start_at.strftime('%H:%M')}-{stop_at.strftime('%H:%M')} IST)."
    )


def _contract_token_for(data_provider, symbol: str) -> str:
    """Resolve an option contract's broker token, or "" when it cannot be found."""
    for attr in ("get_token_for_symbol", "resolve_token", "symbol_token"):
        fn = getattr(data_provider, attr, None)
        if callable(fn):
            try:
                token = str(fn(symbol) or "").strip()
                if token:
                    return token
            except Exception:
                continue
    if getattr(data_provider, "upstox_only", False):
        return ""
    # MarketDataProvider deliberately exposes Kite as a composed source rather than
    # proxying every contract-master method.  The live archive provider has already
    # loaded that source, so use its exact tradingsymbol lookup before falling back to
    # legacy provider shapes.  This is especially useful when an older event omitted
    # its token but the contract is still present in today's broker master.
    kite_source = getattr(data_provider, "kite", None)
    instrument_token = getattr(kite_source, "instrument_token", None)
    if callable(instrument_token):
        exchange = "BFO" if str(symbol or "").upper().startswith("SENSEX") else "NFO"
        try:
            token = str(instrument_token(symbol, exchange) or "").strip()
            if token:
                return token
        except Exception:
            pass
    lookup = getattr(data_provider, "contract_master", None)
    if isinstance(lookup, dict):
        row = lookup.get(symbol) or {}
        return str(row.get("token") or row.get("symboltoken") or "").strip()
    return ""


def _archive_missing_contract_count(value) -> int:
    """Normalize manifest missing-contract coverage across schema generations."""
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 1


_SESSION_ARCHIVE_SCHEMA_VERSION = 3
_SESSION_ARCHIVE_CONTRACT_OWNERSHIP_VERSION = "shadow-outcome-session-date.v1"


def _session_archive_event_identity(root: Path, day: str) -> dict:
    """Identity of the exact event-ledger revision an archive claims to cover."""
    try:
        stat = (
            root / "data_store" / "rtracker" / "events" / f"{day}.jsonl"
        ).stat()
        return {"exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        return {"exists": False, "size": 0, "mtime_ns": 0}


def _session_archive_manifest_complete(root: Path, day: str) -> bool:
    path = root / "data_store" / "session_archive" / day / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        # Schema v3 is the first version that assigns delayed SHADOW_OUTCOME rows to
        # their embedded session date and binds completion to a specific ledger
        # revision. Older "complete" manifests must be rerun once against their
        # already-frozen candles; otherwise foreign contracts and late outcomes remain
        # silently accepted forever.
        if (
            _safe_nonnegative_int(manifest.get("schema_version"))
            < _SESSION_ARCHIVE_SCHEMA_VERSION
            or str(manifest.get("contract_ownership_version") or "")
            != _SESSION_ARCHIVE_CONTRACT_OWNERSHIP_VERSION
            or manifest.get("source_event_identity")
            != _session_archive_event_identity(root, day)
        ):
            return False
        if manifest.get("complete") is not True or bool(
            manifest.get("coverage_errors")
        ):
            return False
        symbols = {
            str(symbol or "").strip()
            for symbol in (manifest.get("traded_symbols_recorded") or [])
            if str(symbol or "").strip()
        }
        contract_count = int(manifest.get("contract_files_archived") or 0)
        spot_count = int(manifest.get("spot_files_archived") or 0)
        missing = _archive_missing_contract_count(
            manifest.get("contract_files_without_day_bars")
        )
        archived_symbols_value = manifest.get("contract_symbols_archived")
        archived_symbols = (
            {
                str(symbol or "").strip()
                for symbol in archived_symbols_value
                if str(symbol or "").strip()
            }
            if isinstance(archived_symbols_value, (list, tuple, set))
            else set()
        )
        missing_symbols = _archive_missing_contract_count(
            manifest.get("missing_contract_symbols")
        )
        exact_contract_coverage = bool(
            not symbols
            or (
                isinstance(archived_symbols_value, (list, tuple, set))
                and symbols.issubset(archived_symbols)
                and missing_symbols == 0
                and contract_count >= len(symbols)
            )
        )
        return bool(
            int(manifest.get("events") or 0) > 0
            and spot_count >= 12
            and missing == 0
            and exact_contract_coverage
        )
    except Exception:
        return False


def _read_session_archive_manifest(root: Path, day: str) -> dict:
    """Best-effort manifest reader shared by completion and repair handling."""
    try:
        value = json.loads(
            (root / "data_store" / "session_archive" / day / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        return dict(value) if isinstance(value, dict) else {}
    except Exception:
        return {}


def _safe_nonnegative_int(value, default: int = 0) -> int:
    """Normalize persisted counters without letting a damaged manifest kill a worker."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return max(0, int(default))


def _session_archive_source_identity(root: Path, day: str, manifest: dict | None = None) -> dict:
    """Identity of the ledger and missing-contract files behind a terminal decision.

    A terminal classification is only valid for the evidence it inspected. Appending
    events, restoring a cache file, or manually repairing an archived candle therefore
    re-queues the day automatically instead of leaving a stale permanent tombstone.
    """
    manifest = manifest if isinstance(manifest, dict) else _read_session_archive_manifest(root, day)

    def _file_identity(path: Path) -> dict:
        try:
            stat = path.stat()
            return {"exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        except OSError:
            return {"exists": False, "size": 0, "mtime_ns": 0}

    ledger = root / "data_store" / "rtracker" / "events" / f"{day}.jsonl"
    symbols_value = manifest.get("missing_contract_symbols") or []
    symbols = sorted(
        {
            str(symbol or "").strip()
            for symbol in symbols_value
            if str(symbol or "").strip()
        }
    ) if isinstance(symbols_value, (list, tuple, set)) else []
    files: dict[str, dict] = {}
    for symbol in symbols:
        files[symbol] = {
            "cache": _file_identity(root / "data_store" / "candles" / f"{symbol}_1min.csv"),
            "archive": _file_identity(
                root / "data_store" / "session_archive" / day / "candles" / f"{symbol}_1min.csv"
            ),
        }
    return {"ledger": _file_identity(ledger), "missing_contract_files": files}


def _session_archive_manifest_terminal(root: Path, day: str) -> bool:
    """Whether an incomplete archive is terminal for its *current* source evidence."""
    manifest = _read_session_archive_manifest(root, day)
    status = manifest.get("repair_status") or {}
    if not isinstance(status, dict) or str(status.get("state") or "").lower() != "terminal_unfillable":
        return False
    recorded = status.get("source_identity")
    return isinstance(recorded, dict) and recorded == _session_archive_source_identity(
        root, day, manifest
    )


def _session_archive_repair_due(root: Path, day: str, now: datetime) -> bool:
    """Apply persisted transient backoff without confusing it with terminal coverage."""
    status = _read_session_archive_manifest(root, day).get("repair_status") or {}
    if not isinstance(status, dict):
        return True
    raw = str(status.get("next_retry_at") or "").strip()
    # Compatibility for manifests written before archive-level backoff was persisted.
    # Their contract rows may all be terminal (or absent for a legacy ledger) while
    # spot coverage still fails, leaving a high archive_failures count but no retry
    # timestamp. Derive the same bounded delay from the last recorded attempt instead
    # of immediately starting another five-minute backlog sweep after every restart.
    if (
        not raw
        and _safe_nonnegative_int(status.get("archive_failures"))
        >= _session_archive_repair_attempt_limit()
    ):
        last_attempt = str(status.get("last_attempt_at") or "").strip()
        if last_attempt:
            try:
                retry_at = datetime.fromisoformat(last_attempt) + timedelta(
                    minutes=_session_archive_retry_backoff_minutes()
                )
                raw = retry_at.isoformat()
            except (TypeError, ValueError, OverflowError):
                pass
    if not raw:
        return True
    try:
        retry_at = datetime.fromisoformat(raw)
        now_ist = IST.localize(now) if now.tzinfo is None else now.astimezone(IST)
        retry_at = IST.localize(retry_at) if retry_at.tzinfo is None else retry_at.astimezone(IST)
        return now_ist >= retry_at
    except (TypeError, ValueError, OverflowError):
        return True


def _session_archive_repair_attempt_limit() -> int:
    """Bound automatic repair retries; a missing expired contract cannot heal by spam."""
    try:
        return max(
            1,
            min(20, int(os.getenv("SESSION_ARCHIVE_REPAIR_MAX_ATTEMPTS", "3") or 3)),
        )
    except (TypeError, ValueError):
        return 3


def _session_archive_retry_backoff_minutes() -> int:
    """Bounded delay shared by persisted and legacy-derived archive backoff."""
    try:
        return max(
            5,
            min(
                24 * 60,
                int(os.getenv("SESSION_ARCHIVE_RETRY_BACKOFF_MINUTES", "360") or 360),
            ),
        )
    except (TypeError, ValueError):
        return 360


def _record_session_archive_repair(
    root: Path,
    day: str,
    *,
    backfill_outcomes: dict | None = None,
    default_reason: str = "archive incomplete",
) -> dict:
    """Persist exact repair evidence without turning transient outages into data loss.

    Retry limits trigger a durable backoff, not a terminal classification. Only an
    outcome explicitly marked both ``terminal_unfillable`` and ``permanent`` may stop
    automatic retries, and that decision is bound to the current ledger/cache identity.
    """
    path = root / "data_store" / "session_archive" / day / "manifest.json"
    manifest = _read_session_archive_manifest(root, day)
    if not manifest:
        manifest = {
            "schema_version": _SESSION_ARCHIVE_SCHEMA_VERSION,
            "contract_ownership_version": _SESSION_ARCHIVE_CONTRACT_OWNERSHIP_VERSION,
            "source_event_identity": _session_archive_event_identity(root, day),
            "date": day,
            "events": 0,
            "complete": False,
            "coverage_errors": [default_reason],
            "missing_contract_symbols": [],
        }

    old = manifest.get("repair_status") or {}
    old = dict(old) if isinstance(old, dict) else {}
    old_attempts = _safe_nonnegative_int(old.get("attempts"))
    archive_failures = _safe_nonnegative_int(old.get("archive_failures")) + 1
    limit = _session_archive_repair_attempt_limit()
    missing_symbols = sorted(
        {
            str(symbol or "").strip()
            for symbol in (manifest.get("missing_contract_symbols") or [])
            if str(symbol or "").strip()
        }
    )
    prior_rows = old.get("missing_contracts") or {}
    prior_rows = prior_rows if isinstance(prior_rows, dict) else {}
    outcomes = backfill_outcomes if isinstance(backfill_outcomes, dict) else {}
    rows: dict[str, dict] = {}
    made_authenticated_attempt = False
    for symbol in missing_symbols:
        prior_row = prior_rows.get(symbol)
        row = dict(prior_row) if isinstance(prior_row, dict) else {}
        outcome = outcomes.get(symbol)
        if isinstance(outcome, dict):
            row.update({key: value for key, value in outcome.items() if value not in (None, "")})
        attempted_symbol = bool(isinstance(outcome, dict) and outcome.get("attempted", True))
        symbol_attempts = _safe_nonnegative_int(row.get("attempts"))
        if attempted_symbol:
            symbol_attempts += 1
            made_authenticated_attempt = True
        if not row:
            row = {"state": "retryable", "reason": default_reason}
        row.setdefault("state", "retryable")
        row.setdefault("reason", default_reason)
        row["attempts"] = symbol_attempts
        permanent = bool(row.get("permanent"))
        if row["state"] == "terminal_unfillable" and not permanent:
            row["state"] = "retryable"
            row["reason"] = f"unconfirmed permanent failure: {row['reason']}"
        if row["state"] == "filled":
            # A successful fetch that did not leave usable day bars is not proof of
            # coverage.  Keep it visible and retryable rather than calling it fixed.
            row["state"] = "retryable"
            row["reason"] = "backfill returned but no usable 1-minute day bars were archived"
        if row["state"] == "retryable" and symbol_attempts >= limit:
            row["state"] = "retry_backoff"
            evidence = str(row.get("reason") or default_reason or "transient gap").strip()
            row["reason"] = (
                f"{evidence} (retry backoff after {symbol_attempts} repair attempts)"
            )[:500]
        rows[symbol] = row

    # Contracts the pre-archive back-fill fetched that the manifest does not list. They
    # never decide the day's state, but their outcome must be kept: unrecorded, an
    # expired contract's "invalid token" was requested again on every cycle (2026-09-25:
    # 19 of the 31 contracts fetched for 2026-08-12, every rotation, all night).
    prior_other = old.get("other_contracts") or {}
    prior_other = prior_other if isinstance(prior_other, dict) else {}
    other_rows: dict[str, dict] = {
        str(symbol): dict(row) for symbol, row in prior_other.items()
        if isinstance(row, dict) and str(symbol) not in rows
    }
    for symbol, outcome in outcomes.items():
        symbol = str(symbol or "").strip()
        if not symbol or symbol in rows or not isinstance(outcome, dict):
            continue
        row = dict(other_rows.get(symbol) or {})
        row.update({key: value for key, value in outcome.items() if value not in (None, "")})
        symbol_attempts = _safe_nonnegative_int(row.get("attempts"))
        if outcome.get("attempted", True):
            symbol_attempts += 1
            made_authenticated_attempt = True
        row["attempts"] = symbol_attempts
        row.setdefault("state", "retryable")
        if row["state"] == "terminal_unfillable" and not bool(row.get("permanent")):
            row["state"] = "retryable"
        other_rows[symbol] = row

    attempts = old_attempts + (1 if made_authenticated_attempt else 0)

    coverage_errors = list(manifest.get("coverage_errors") or [])
    contract_only_errors = all(
        "contract symbol coverage" in str(item).lower()
        or "contract file" in str(item).lower()
        for item in coverage_errors
    ) if coverage_errors else False
    all_missing_terminal = bool(rows) and all(
        str(row.get("state") or "") == "terminal_unfillable" and bool(row.get("permanent"))
        for row in rows.values()
    )
    # Only exact contract-history gaps can be classified terminal automatically.
    # Missing spot files, a failed child process, or a transient filesystem error may
    # recover on the next run and must remain retryable rather than being abandoned
    # after an arbitrary count.
    terminal = all_missing_terminal and (not coverage_errors or contract_only_errors)
    contract_backoff = any(
        str(row.get("state") or "") == "retry_backoff" for row in rows.values()
    )
    # The archive can remain incomplete even when there is no retryable contract row:
    # legacy ledgers may have no contract identity, and sessions whose contracts are
    # all permanently unavailable can still be missing spot/timeframe files.  Those
    # failures previously stayed ``retryable`` forever with no ``next_retry_at``, so
    # the daemon respawned the same doomed child every five minutes after each backlog
    # rotation.  Bound the *archive* retry cadence as well as the per-contract cadence.
    coverage_backoff = not terminal and archive_failures >= limit
    in_backoff = contract_backoff or coverage_backoff
    state = "terminal_unfillable" if terminal else ("retry_backoff" if in_backoff else "retryable")
    now_ist = datetime.now(IST)
    backoff_minutes = _session_archive_retry_backoff_minutes()
    manifest["schema_version"] = max(
        _SESSION_ARCHIVE_SCHEMA_VERSION,
        _safe_nonnegative_int(manifest.get("schema_version")),
    )
    manifest["contract_ownership_version"] = (
        _SESSION_ARCHIVE_CONTRACT_OWNERSHIP_VERSION
    )
    manifest["source_event_identity"] = _session_archive_event_identity(root, day)
    manifest["complete"] = False
    manifest["repair_status"] = {
        "state": state,
        "classification": state,
        "attempts": attempts,
        "archive_failures": archive_failures,
        "max_attempts": limit,
        "last_attempt_at": now_ist.isoformat(),
        "last_error": str(default_reason or "archive incomplete")[:1000],
        "missing_contracts": rows,
        "other_contracts": other_rows,
        "source_identity": _session_archive_source_identity(root, day, manifest),
        "next_retry_at": (
            (now_ist + timedelta(minutes=backoff_minutes)).isoformat()
            if in_backoff and not terminal
            else None
        ),
        "next_action": (
            "manual data recovery required; automatic retries stopped"
            if terminal
            else (
                f"retry incomplete session archive after {backoff_minutes} minute backoff"
                if in_backoff
                else "retry authenticated exact-contract backfill before the next archive run"
            )
        ),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        os.replace(temporary, path)
    except Exception as exc:
        logger.warning(f"Could not persist archive repair status for {day}: {exc}")
    return dict(manifest["repair_status"])


def _next_due_session_archive_day(
    root: Path,
    now: datetime,
    run_at: dt_time,
    attempted: set[str],
) -> str | None:
    """Select each due incomplete ledger once before retrying a failed date.

    Newest-first keeps the latest session available quickly. ``attempted`` makes a
    persistent failure yield to the rest of the backlog; after one complete pass it is
    cleared so unresolved dates remain retryable rather than being abandoned.
    """
    now_ist = IST.localize(now) if now.tzinfo is None else now.astimezone(IST)
    event_dir = root / "data_store" / "rtracker" / "events"
    due_today = now_ist >= _ist_at(now_ist.date(), run_at)
    due: list[str] = []
    try:
        for path in event_dir.glob("????-??-??.jsonl"):
            day = path.stem
            try:
                parsed = datetime.strptime(day, "%Y-%m-%d").date()
            except ValueError:
                continue
            if parsed < now_ist.date() or (parsed == now_ist.date() and due_today):
                if (
                    not _session_archive_manifest_complete(root, day)
                    and not _session_archive_manifest_terminal(root, day)
                    and _session_archive_repair_due(root, day, now_ist)
                ):
                    due.append(day)
    except OSError as exc:
        logger.warning(f"Unable to inspect session ledgers for archive catch-up: {exc}")
        return None

    due = sorted(set(due), reverse=True)
    if not due:
        attempted.clear()
        return None
    pending = [day for day in due if day not in attempted]
    if not pending:
        attempted.clear()
        pending = due
    return pending[0]


def _fetch_contract_history_for_session(
    data_provider,
    session_day: str,
    symbol: str,
    token: str,
    call_outcome: dict | None = None,
) -> bool:
    """Fetch one option contract with the requested session as the API end date."""
    try:
        target_date = datetime.strptime(str(session_day), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return False
    target_end = _ist_at(target_date, dt_time(15, 45))
    df = data_provider.get_historical_candles(
        token,
        "BFO" if str(symbol).upper().startswith("SENSEX") else "NFO",
        "1min",
        days_back=1,
        instrument_name=symbol,
        force_refresh=True,
        to_date=target_end,
        kite_call_outcome=call_outcome,
        record_kite_history_health=False,
    )
    return df is not None and len(df) > 0


def _contract_history_backfill_outcome(
    data_provider,
    filled: bool,
    call_outcome: dict | None = None,
) -> dict:
    """Translate the exact authenticated history result into durable repair evidence.

    Kite returns an empty frame for both an ordinary transient failure and a permanently
    invalid *instrument* token.  Collapsing those outcomes to ``False`` made the archive
    scheduler retry expired contracts forever even though Kite had given a conclusive
    answer.  Keep the permanent match intentionally exact: access-token/session errors,
    throttles, timeouts and successful-but-empty responses remain retryable.
    """
    if filled:
        return {
            "state": "filled",
            "reason": "historical fetch returned usable rows",
            "attempted": True,
        }

    # Prefer evidence carried by this exact call. The provider-wide status remains a
    # compatibility fallback for lightweight providers/tests, but is racy when archive
    # repair and the live scanner fetch history concurrently.
    exact = dict(call_outcome or {})
    if exact:
        last_call_ok = exact.get("ok")
        failure = str(exact.get("reason") or "").strip()
    else:
        status: dict = {}
        source = getattr(data_provider, "kite", None)
        status_fn = getattr(source, "historical_rest_status", None)
        if callable(status_fn):
            try:
                value = status_fn() or {}
                status = dict(value) if isinstance(value, dict) else {}
            except Exception:
                status = {}
        last_call_ok = status.get("kite_historical_rest_last_call_ok")
        failure = str(
            status.get("kite_historical_rest_failure")
            or status.get("kite_historical_rest_last_error")
            or ""
        ).strip()
    normalized = re.sub(r"\s+", " ", failure).strip().lower().rstrip(".")
    normalized = normalized.removeprefix("kiteconnect.exceptions.")
    if last_call_ok is False and normalized == "inputexception: invalid token":
        return {
            "state": "terminal_unfillable",
            "reason": "broker permanently rejected the exact instrument token as invalid",
            "permanent": True,
            "attempted": True,
        }
    if last_call_ok is True:
        return {
            "state": "retryable",
            "reason": "authenticated broker returned zero rows for the closed-session window",
            "attempted": True,
        }
    if failure:
        return {
            "state": "retryable",
            "reason": f"historical broker request failed: {failure}"[:500],
            "attempted": True,
        }
    return {
        "state": "retryable",
        "reason": "historical fetch returned no usable rows without conclusive broker evidence",
        "attempted": True,
    }


def _session_archive_terminal_contracts(root: Path, day: str) -> dict[str, dict]:
    """Terminal per-contract rows whose ledger/cache evidence has not changed."""
    manifest = _read_session_archive_manifest(root, day)
    status = manifest.get("repair_status") or {}
    if not isinstance(status, dict):
        return {}
    recorded = status.get("source_identity")
    if not isinstance(recorded, dict) or recorded != _session_archive_source_identity(
        root, day, manifest
    ):
        return {}
    rows = status.get("missing_contracts") or {}
    rows = dict(rows) if isinstance(rows, dict) else {}
    other = status.get("other_contracts") or {}
    if isinstance(other, dict):
        for symbol, row in other.items():
            rows.setdefault(symbol, row)
    return {
        str(symbol): dict(row)
        for symbol, row in rows.items()
        if isinstance(row, dict)
        and str(row.get("state") or "") == "terminal_unfillable"
        and bool(row.get("permanent"))
    }


def _resolve_session_archive_provider(data_provider):
    """Return (provider, deferred_not_ready, reason) without touching broker state."""
    if data_provider is None:
        # An explicitly unwired scheduler is allowed to archive cache-only data.
        return None, False, ""
    if hasattr(data_provider, "get_historical_candles"):
        return data_provider, False, ""
    if not callable(data_provider):
        return None, False, ""
    try:
        resolved = data_provider()
    except Exception as exc:
        return None, True, f"deferred provider unavailable: {exc}"
    if resolved is None:
        return None, True, "deferred provider not initialized"
    if not hasattr(resolved, "get_historical_candles"):
        return None, True, "deferred provider has no historical-candle API"
    return resolved, False, ""


def start_session_archiver_scheduler(data_provider=None):
    """Archive each session's trades + real option-premium candles after the close.

    Builds the multi-day validation set the offline exit backtester replays. The live
    candle cache only keeps the last-fetched contracts, so without this snapshot a
    traded strike's intraday bars are lost/overwritten (2026-07-24: only ~40% of trades
    were replayable). Runs once per day AFTER market close, never during market hours,
    in a child process so it cannot stall the scanner loop.
    """
    enabled = str(os.getenv("SESSION_ARCHIVE_ENABLED", "1")).strip().lower()
    if enabled in _FALSEY_ENV:
        logger.info("Session archiver disabled by SESSION_ARCHIVE_ENABLED.")
        return

    run_at = _parse_hhmm(os.getenv("SESSION_ARCHIVE_TIME", "15:50"), dt_time(15, 50))
    try:
        provider_retry_seconds = max(
            5.0,
            min(15.0, float(os.getenv("SESSION_ARCHIVE_PROVIDER_RETRY_SECONDS", "10") or 10.0)),
        )
    except (TypeError, ValueError):
        provider_retry_seconds = 10.0
    root = Path(__file__).parent
    script = root / "scripts" / "archive_session_data.py"

    def _manifest_complete(day: str) -> bool:
        return _session_archive_manifest_complete(root, day)

    def _latest_due_event_day(now: datetime, attempted: set[str]) -> str | None:
        """Next incomplete ledger, rotating across the full due backlog."""
        return _next_due_session_archive_day(root, now, run_at, attempted)

    def _archive_loop():
        completed: set[str] = set()
        attempted: set[str] = set()
        last_provider_wait_reason = ""
        announced_market_hours_defer = False
        while True:
            now = datetime.now(IST)
            today_run = _ist_at(now.date(), run_at)
            session_day = _latest_due_event_day(now, attempted)
            if session_day is None:
                target = today_run if now < today_run else _ist_at(now.date() + timedelta(days=1), run_at)
                time.sleep(max(30.0, (target - now).total_seconds()))
                continue
            # Nothing here can succeed before the close, so do not pay for any of it.
            #
            # The child refuses heavy archive IO during market hours and exits non-zero,
            # which this loop reported as "Session archive failed" every five minutes --
            # 73 and 80 times in the 2026-08-24 and 2026-08-25 sessions alone. Worse, the
            # back-fill below runs BEFORE the child and re-fetches EXPIRED option
            # contracts, which is the "Kite historical_data failed: InputException:
            # invalid token" burst in the log (that phrase means the INSTRUMENT token,
            # not the broker session -- the session is fine).
            #
            # Deferring costs nothing: the child would have refused anyway. The day is
            # deliberately NOT added to `attempted`, so it still archives after the close
            # exactly as before -- the fail-closed guard in
            # tests/test_postmarket_archive_fail_closed_2026_08_22.py is untouched.
            try:
                from scripts.archive_session_data import market_hours_now as _market_open
                _defer = bool(_market_open())
            except Exception:
                _defer = False          # never let this guard block archiving
            if _defer:
                if not announced_market_hours_defer:
                    logger.info(
                        f"Session archive for {session_day} deferred until after the close."
                    )
                    announced_market_hours_defer = True
                time.sleep(300.0)
                continue
            announced_market_hours_defer = False

            # BACK-FILL FIRST, in this process, because this is where the broker
            # session lives. The archiver runs as a child with no auth, so it can only
            # copy what the live cache already holds -- and for OPTION contracts the
            # cache held nothing: as of 2026-08-14 it had no CE/PE file newer than
            # 2026-07-31, so every session after that archived index candles and no
            # premium, and became permanently unreplayable. The snapshot existed; the
            # thing it was meant to snapshot did not.
            # Resolved HERE, not at wiring time. The provider is built inside
            # create_system(), which runs on the dashboard bootstrap task well after
            # this scheduler is started, so at wiring time there is nothing to pass.
            # A callable defers the lookup to the moment it is actually needed --
            # after the close, by which point the scanner exists.
            provider, provider_pending, provider_reason = _resolve_session_archive_provider(
                data_provider
            )
            if provider_pending:
                # Startup race: the scheduler thread can reach an overdue ledger before
                # create_system publishes runtime["scanner"]. This is not an archive
                # attempt. Do not rotate away from the newest day and do not run the
                # cache-only child against data we already know is not ready yet.
                if provider_reason != last_provider_wait_reason:
                    logger.info(
                        f"Session archive waiting for runtime data provider: {provider_reason}"
                    )
                    last_provider_wait_reason = provider_reason
                time.sleep(provider_retry_seconds)
                continue
            last_provider_wait_reason = ""
            # Mark only after deferred startup dependencies are ready. A real
            # provider/back-fill/child failure should yield to the rest of the backlog.
            attempted.add(session_day)
            backfill_outcomes: dict[str, dict] = {}
            if provider is not None:
                try:
                    from scripts.archive_session_data import (
                        backfill_contract_history,
                        contract_tokens,
                        load_events,
                        missing_contract_symbols,
                    )

                    # The token comes from the session's OWN event stream. Asking the
                    # provider never worked: _contract_token_for probes four attributes
                    # (get_token_for_symbol / resolve_token / symbol_token /
                    # contract_master) that MarketDataProvider does not have -- it
                    # exposes get_option_token(name, strike, option_type) instead. So the
                    # token was always "", _fetch always returned False, and this
                    # back-fill filled nothing on any session. It stays as a fallback.
                    session_events = load_events(session_day)
                    session_tokens = contract_tokens(session_events, session_day)
                    wanted_symbols = missing_contract_symbols(session_day, session_events)
                    terminal_contracts = _session_archive_terminal_contracts(
                        root, session_day
                    )

                    def _fetch(symbol: str) -> bool:
                        terminal_row = terminal_contracts.get(symbol)
                        if terminal_row is not None:
                            # A different ledger or cache/archive file identity empties
                            # terminal_contracts above and automatically re-enables the
                            # request.  While the evidence is unchanged, do not hammer an
                            # expired instrument token just because another symbol on the
                            # same day is still retryable.
                            backfill_outcomes[symbol] = {
                                **terminal_row,
                                "attempted": False,
                            }
                            return False
                        token = str(session_tokens.get(symbol) or "").strip()
                        if getattr(provider, "upstox_only", False) is True:
                            segment, _, contract_id = token.partition("|")
                            if segment not in {"NSE_FO", "BSE_FO"} or not contract_id.isdecimal():
                                token = ""
                        token = token or _contract_token_for(provider, symbol)
                        if not token:
                            backfill_outcomes[symbol] = {
                                "state": "retryable",
                                "reason": "exact contract token absent from session evidence and provider",
                                "attempted": False,
                            }
                            return False
                        # force_refresh, because the whole point of this call is to GO
                        # AND GET the day. Otherwise get_historical_candles returns the
                        # local option CSV whenever the file merely EXISTS, without
                        # checking that it covers the day asked for -- so a contract
                        # whose cache stops at an earlier session answers instantly with
                        # a non-empty frame, this counts a success, and nothing new is
                        # ever written for the archiver to copy.
                        try:
                            call_outcome: dict = {}
                            filled = _fetch_contract_history_for_session(
                                provider,
                                session_day,
                                symbol,
                                token,
                                call_outcome=call_outcome,
                            )
                        except Exception as exc:
                            backfill_outcomes[symbol] = {
                                "state": "retryable",
                                "reason": f"historical fetch failed: {exc}"[:500],
                                "attempted": True,
                            }
                            return False
                        backfill_outcomes[symbol] = _contract_history_backfill_outcome(
                            provider, filled, call_outcome=call_outcome
                        )
                        return filled

                    filled = backfill_contract_history(
                        session_day, _fetch, events=session_events, quiet=True
                    )
                    for symbol in wanted_symbols:
                        backfill_outcomes.setdefault(symbol, {
                            "state": "retryable",
                            "reason": "backfill produced no classified outcome",
                            "attempted": False,
                        })
                    if filled:
                        logger.info(
                            f"Back-filled {len(filled)} contract histories before "
                            f"archiving {session_day}: {', '.join(filled[:6])}"
                            + (" ..." if len(filled) > 6 else "")
                        )
                except Exception as exc:
                    # Never block the archive: a session that cannot be back-filled must
                    # still archive whatever it does have.
                    logger.warning(f"Contract back-fill skipped for {session_day}: {exc}")
            try:
                result = subprocess.run(
                    [sys.executable, str(script), session_day, "--quiet"],
                    cwd=str(root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    timeout=600,
                    check=True,
                    text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if not _manifest_complete(session_day):
                    raise RuntimeError("archive process returned success but manifest coverage is incomplete")
                logger.info(f"Session archive verified for {session_day}.")
                completed.add(session_day)
            except Exception as exc:
                detail = ""
                if isinstance(exc, subprocess.CalledProcessError):
                    detail = str((exc.stderr or exc.stdout or "").strip())[-500:]
                manifest = _read_session_archive_manifest(root, session_day)
                coverage = manifest.get("coverage_errors") or []
                coverage_reason = "; ".join(
                    str(item).strip() for item in coverage if str(item).strip()
                ) if isinstance(coverage, list) else ""
                reason = coverage_reason[:1000] or detail or str(exc)
                try:
                    repair = _record_session_archive_repair(
                        root,
                        session_day,
                        backfill_outcomes=backfill_outcomes,
                        default_reason=reason,
                    )
                except Exception as repair_exc:
                    # The exception path must never terminate the daemon because a
                    # prior manifest is malformed or temporarily unwritable.
                    logger.exception(
                        f"Could not record session archive repair state for {session_day}: "
                        f"{repair_exc}"
                    )
                    repair = {"state": "retryable", "attempts": 0, "max_attempts": 0}
                if repair.get("state") == "terminal_unfillable":
                    logger.error(
                        f"Session archive for {session_day} is terminal-unfillable after "
                        f"{repair.get('attempts')} repair attempt(s); automatic retries stopped. "
                        f"See data_store/session_archive/{session_day}/manifest.json."
                    )
                else:
                    logger.warning(
                        f"Session archive incomplete for {session_day} "
                        f"(archive failures {repair.get('archive_failures')}/"
                        f"{repair.get('max_attempts')}; authenticated contract repair "
                        f"cycles {repair.get('attempts')}): {exc}"
                        + (f"; child output: {detail}" if detail else "")
                    )
                # Retry the missed ledger instead of marking it complete, but do not
                # spin a failing child process continuously.
                time.sleep(300.0)

    threading.Thread(target=_archive_loop, daemon=True, name="SessionArchiver").start()


def _scanner_trading_day(scanner: Scanner, now_ist: datetime) -> bool:
    checker = getattr(scanner, "_is_trading_day", None)
    if callable(checker):
        try:
            return bool(checker(now_ist))
        except Exception as exc:
            logger.warning(f"Premarket REAL worker trading-day check failed: {exc}")
            return False
    if now_ist.weekday() >= 5:
        return False
    data = getattr(scanner, "data", None)
    try:
        if data is not None and hasattr(data, "is_market_holiday") and data.is_market_holiday(now_ist):
            return False
    except Exception:
        pass
    return True


def _auto_real_mode_attempt(scanner: Scanner, settings, now_ist: datetime | None = None,
                            *, require_running: bool = True) -> tuple[bool, str]:
    """Runtime-only HISTORICAL -> REAL/manual premarket arm.

    This intentionally never enables Auto Mode. It only changes the feed/broker
    operating mode so live data, broker health, warmup, and manual supervision
    are ready before the 09:15 trading gate.
    """
    if not _setting_enabled(getattr(settings, "auto_real_mode_premarket_enabled", True), True):
        return False, "disabled"

    now_ist = now_ist or datetime.now(IST)
    if now_ist.tzinfo is None:
        now_ist = IST.localize(now_ist)
    else:
        now_ist = now_ist.astimezone(IST)

    if require_running and hasattr(scanner, "is_running") and not bool(getattr(scanner, "is_running", False)):
        return False, "scanner_not_running"

    if not _scanner_trading_day(scanner, now_ist):
        return False, "not_trading_day"

    switch_at = _parse_hhmmss(
        getattr(settings, "auto_real_mode_premarket_switch_time", "09:00:00"),
        dt_time(9, 0, 0),
    )
    live_end = _parse_hhmmss(getattr(settings, "real_mode_live_end", "15:30:00"), dt_time(15, 30))
    if now_ist.time() < switch_at:
        return False, "before_switch_time"
    # A switch time set after the close is an operator test of the REAL/manual arm
    # (Auto still needs the day's full readiness, and no entry can pass a closed market).
    if now_ist.time() > live_end and switch_at <= live_end:
        return False, "after_live_end"

    current_mode = str(getattr(scanner, "mode", "") or "").upper()
    if current_mode == "REAL":
        # This worker owns only the 09:00 premarket transition.  Once live
        # trading has started, Auto Mode may have been explicitly authorized by
        # the operator or by the readiness-gated startup arm.  A late worker pass
        # must never revoke that authority between signals.
        live_start = _parse_hhmmss(
            getattr(settings, "real_mode_live_start", "09:15:00"),
            dt_time(9, 15),
        )
        if now_ist.time() >= live_start:
            return True, "already_real"
        if bool(getattr(scanner, "auto_mode", False)):
            auto_disabled = scanner.configure(auto_mode=False, persist_settings=False)
            if not auto_disabled or bool(getattr(scanner, "auto_mode", False)):
                reason = str(
                    getattr(scanner, "_last_configure_block_reason", "")
                    or "auto_mode_disable_blocked"
                )
                logger.warning(f"Premarket REAL/manual arm blocked: {reason}")
                return False, reason
            _keep_operator_auto_on(scanner, now_ist)
        return True, "already_real_manual"
    if current_mode != "HISTORICAL":
        return False, f"unsupported_mode:{current_mode or 'unknown'}"
    kept = getattr(scanner, "_operator_kept_historical_today", None)
    if callable(kept) and kept(now_ist, switch_at):
        # The operator went back to HISTORICAL after today's switch; a restart keeps that.
        return False, "operator_kept_historical"

    # The premarket worker may start while historical Auto Mode is still enabled
    # from a replay.  Disarm it before changing the broker/feed mode so there is no
    # instant in which the REAL singleton can inherit unattended execution.
    was_auto = bool(getattr(scanner, "auto_mode", False))
    auto_disabled = scanner.configure(auto_mode=False, persist_settings=False)
    if not auto_disabled or bool(getattr(scanner, "auto_mode", False)):
        reason = str(
            getattr(scanner, "_last_configure_block_reason", "")
            or "auto_mode_disable_blocked"
        )
        logger.warning(f"Premarket REAL/manual arm blocked: {reason}")
        return False, reason
    if was_auto:
        _keep_operator_auto_on(scanner, now_ist)

    configured = scanner.configure(
        mode="REAL",
        auto_mode=False,
        confirm_real_mode=True,
        real_mode_verification="YES",
        persist_settings=False,
    )
    if configured and str(getattr(scanner, "mode", "") or "").upper() == "REAL":
        message = "Auto premarket arm: switched to REAL mode."
        logger.warning(message)
        try:
            scanner.log_event(message, "system")
        except Exception:
            pass
        return True, "switched_real_manual"

    reason = str(getattr(scanner, "_last_configure_block_reason", "") or "configure_blocked")
    logger.warning(f"Auto premarket REAL arm blocked: {reason}")
    return False, reason


def _keep_operator_auto_on(scanner, now_ist: datetime) -> None:
    """The switch enters REAL with Auto held OFF (Auto needs the day's readiness first).
    The operator's ON is remembered and restored by the scanner's readiness-gated arm, so
    the mode switch never overrides the Auto toggle (2026-09-25 user)."""
    carry = getattr(scanner, "_carry_auto_intent", None)
    if not callable(carry):
        return
    try:
        carry(now_ist.date().isoformat())
        scanner.log_event(
            "Auto Mode was ON: it comes back ON by itself once today's live readiness "
            "checks pass (from 09:15).",
            "system",
        )
    except Exception as exc:
        logger.debug(f"Auto intent not carried over the REAL switch: {exc}")


def start_premarket_real_mode_worker(scanner: Scanner, settings) -> None:
    """Arm REAL/manual mode at 09:00 without revoking live Auto authority.

    The worker always runs and re-reads the toggle and switch time every pass, so the
    dashboard can turn it on or off (or move the time) without a restart.
    """
    if not _setting_enabled(getattr(settings, "auto_real_mode_premarket_enabled", True), True):
        logger.info("Premarket REAL/manual worker idle until enabled from the dashboard.")

    switch_at = _parse_hhmmss(
        getattr(settings, "auto_real_mode_premarket_switch_time", "09:00:00"),
        dt_time(9, 0, 0),
    )
    retry_seconds = max(
        5.0,
        float(getattr(settings, "auto_real_mode_premarket_retry_seconds", 30.0) or 30.0),
    )
    completed_dates: set[str] = set()
    last_reason_by_date: dict[str, str] = {}

    def _scheduler_loop():
        nonlocal switch_at
        while True:
            now = datetime.now(IST)
            session_day = now.date().isoformat()
            if not _setting_enabled(getattr(settings, "auto_real_mode_premarket_enabled", True), True):
                time.sleep(60.0)
                continue
            new_switch = _parse_hhmmss(
                getattr(settings, "auto_real_mode_premarket_switch_time", "09:00:00"),
                dt_time(9, 0, 0),
            )
            if new_switch != switch_at:
                switch_at = new_switch
                completed_dates.discard(session_day)
            today_switch = _ist_at(now.date(), switch_at)

            if session_day in completed_dates:
                tomorrow_switch = _ist_at(now.date() + timedelta(days=1), switch_at)
                sleep_seconds = max(60.0, (tomorrow_switch - now).total_seconds())
            elif now < today_switch:
                sleep_seconds = max(5.0, (today_switch - now).total_seconds())
            else:
                ok, reason = _auto_real_mode_attempt(scanner, settings, now)
                if ok:
                    completed_dates.add(session_day)
                    sleep_seconds = 60.0
                elif reason == "disabled":
                    sleep_seconds = 60.0
                elif reason in {"not_trading_day", "after_live_end", "operator_kept_historical"}:
                    completed_dates.add(session_day)
                    tomorrow_switch = _ist_at(now.date() + timedelta(days=1), switch_at)
                    sleep_seconds = max(60.0, (tomorrow_switch - now).total_seconds())
                else:
                    if last_reason_by_date.get(session_day) != reason:
                        logger.warning(f"Premarket REAL/manual worker waiting: {reason}")
                        last_reason_by_date[session_day] = reason
                    sleep_seconds = retry_seconds

            logger.info(
                "Premarket REAL/manual worker sleeping "
                f"{min(sleep_seconds, 60.0):.0f}s (switch {switch_at.strftime('%H:%M:%S')} IST)."
            )
            # Wake at least once a minute so a dashboard toggle or new time applies.
            time.sleep(min(sleep_seconds, 60.0))

    threading.Thread(
        target=_scheduler_loop,
        daemon=True,
        name="PremarketRealModeWorker",
    ).start()



def create_system():
    """Initialize all system components"""
    settings = get_settings()
    instruments = settings.get_instruments()
    startup_mode_text = str(settings.trading_mode or "").strip().upper()

    def runtime_mode() -> RuntimeMode:
        return RuntimeMode.parse(settings.trading_mode)

    def replay_fallback() -> str:
        return (
            "HISTORICAL"
            if startup_mode_text in {"HISTORICAL", "PAPER", "REAL"}
            else RuntimeMode.REPLAY.value
        )

    risk_summary = settings.risk_configuration_summary()
    logger.info(
        f"Risk configuration: capital={risk_summary['capital_total']:,.0f} "
        f"FUT={risk_summary['capital_fut']:,.0f} OPT={risk_summary['capital_opt']:,.0f} | "
        f"SL FUT={risk_summary['futures_sl_pct']:.3f}% "
        f"OPT={risk_summary['options_sl_pct']:.2f}% | "
        f"absolute cap={risk_summary['max_trade_loss_abs']:,.0f}"
    )
    for warning in risk_summary["warnings"]:
        logger.warning(f"Risk configuration warning: {warning}")
    if runtime_mode().permits_live_orders and not risk_summary["ok"]:
        logger.critical(
            "REAL startup refused by risk configuration: "
            + "; ".join(risk_summary["issues"])
        )
        settings.trading_mode = replay_fallback()

    env_path = Path(__file__).parent / ".env"
    plaintext_secret_keys = set()
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            key, separator, value = raw_line.partition("=")
            if separator and value.strip() and key.strip().upper() in {
                "ANGELONE_API_KEY",
                "ANGELONE_PASSWORD",
                "ANGELONE_TOTP_SECRET",
                "ANGELONE_SECRET_KEY",
                "FYERS_SECRET_KEY",
                "FYERS_PASSWORD",
            }:
                plaintext_secret_keys.add(key.strip().upper())
    if plaintext_secret_keys:
        logger.warning(
            "Broker credentials are stored in local plaintext .env; keep this host single-user "
            "and restrict file ACLs."
        )
    if (
        runtime_mode().permits_live_orders
        and settings.shared_host_mode
        and plaintext_secret_keys
        and not settings.allow_plaintext_broker_secrets
    ):
        logger.critical("REAL startup refused: plaintext broker secrets are forbidden in shared-host mode.")
        settings.trading_mode = replay_fallback()

    logger.info("═" * 60)
    logger.info("🚀 UT1 Index Trading System v2.0")
    logger.info("═" * 60)

    from trading.angelone_broker import AngelOneBroker

    sim_broker = SignalSimBroker()
    real_broker = AngelOneBroker(instance_id=f"UT1:{settings.dashboard_port}")
    data_provider = MarketDataProvider(start_streams=runtime_mode().uses_live_market_data, upstox_only=True)
    data_provider.execution_broker = real_broker
    data_provider.connect()
    data_provider.fetch_instruments()
    if runtime_mode().permits_live_orders:
        readiness = validate_real_mode_readiness(settings, data_provider)
        if readiness.get("blockers") == ["all broker sources unavailable"]:
            for _ in range(12):
                time.sleep(0.25)
                readiness = validate_real_mode_readiness(settings, data_provider)
                if readiness.get("blockers") != ["all broker sources unavailable"]:
                    break
        # Warnings are not blockers, but they were being discarded entirely — including
        # "broker free cash is below the floor for one index lot". See
        # Scanner._surface_readiness_warnings for the full account.
        for warning in readiness.get("warnings") or []:
            logger.warning(f"REAL readiness warning: {warning}")
        if getattr(settings, "production_safe_mode_required", True) and not readiness.get("ok"):
            logger.error(
                "REAL startup refused by safe mode: "
                + "; ".join(readiness.get("blockers") or ["readiness failed"])
            )
            settings.trading_mode = replay_fallback()
            try:
                settings.save_to_env()
            except Exception as exc:
                logger.error(f"Failed to persist HISTORICAL safe-mode fallback: {exc}")
    
    if runtime_mode().permits_live_orders and not (
        real_broker.execution_lease_established() and real_broker.acquire_execution_lease()
    ):
        logger.critical("REAL startup refused: " + real_broker.execution_lease_block_reason())
        settings.trading_mode = replay_fallback()
        try:
            settings.save_to_env()
        except Exception as exc:
            logger.error(f"Failed to persist lease safe-mode fallback: {exc}")

    # Initial broker selection based on settings
    active_broker = real_broker if (runtime_mode().permits_live_orders and real_broker) else sim_broker
    logger.info("Market data: Upstox | Execution: {}",
                "Angel One REAL" if active_broker is real_broker else "simulation")

    # 2. Candle Builder
    candle_builder = CandleBuilder(
        max_candles=2000,
        # 2026-09-25 open: the tick path held the GIL 28% of the time (py-spy, --gil).
        # Publishing the forming 1m row every 0.1s merged it into a 2000-row frame ~10x
        # a second per index: 24.6% of one core per index measured offline, 5.2% at
        # 1.0s/2.0s; one publish still costs ~30ms of pandas on the Kite thread, which
        # was 23% of all process CPU at 1s (11:18 per-thread measurement), hence 2s/3s.
        # Exits and the 100ms trade monitor read the latest TICK price, which still
        # updates on every tick; only the forming candle row and the forming 5m/15m
        # bucket lag, by at most 2s / 3s, against a 30s stabilization window and a 15s
        # S1 forming-bar recompute. Closed bars still publish and resample at once.
        partial_publish_min_interval_seconds=2.0,
        resample_min_interval_seconds=3.0,
    )
    logger.info("✅ Candle Builder")

    # Connect tick streams
    data_provider.on_tick = candle_builder.update_from_tick
    data_provider.start_upstox_stream()

    # 3. UT Bot Multi-TF Engine
    mtf_engine = MultiTimeframeEngine(settings.get_ut_engine_params())
    active_index_count = len(getattr(settings, "active_indices", []) or [])
    scanner_count = active_index_count * 3
    logger.info(
        f"✅ UT1 Multi-TF Engine ({active_index_count} active indices × 3 TFs = {scanner_count} live scanners)"
    )

    # 4. Signal Manager
    signal_manager = SignalManager()
    logger.info("✅ Signal Manager (dedup + grading)")

    # 5. Intelligence Aggregator
    intelligence = IntelligenceAggregator()
    logger.info("✅ Intelligence (Vol/OI/PCR/Greeks/Regime/OrderFlow)")

    # 6. Trade Manager
    trade_manager = TradeManager(
        broker=active_broker,
        max_positions=settings.max_concurrent_positions,
        session_end=settings.ut_session_end,
        product_type=settings.trade_product_type,
        on_notification=send_desktop_notification,
    )
    # Give trade_manager references to both for hot-swapping
    trade_manager.mock_broker = sim_broker
    trade_manager.real_broker = real_broker

    # The live adapter asks this at the moment of every mutation, not when the caller
    # was queued. That is what closes the hot-swap window: a job enqueued while REAL and
    # dispatched after a switch to HISTORICAL asks again here and is refused, and so is
    # anything on a timer (reconcile runs every ~5s in every mode) that reaches the
    # broker after the mode changed underneath it. Bound to the manager rather than to a
    # captured value so it reads the CURRENT mode every time.
    if hasattr(real_broker, "set_execution_authority"):
        real_broker.set_execution_authority(trade_manager._live_orders_permitted)

    # 7. Performance Tracker
    performance = PerformanceTracker()
    logger.info("✅ Performance Tracker")

    # 8. Scanner
    scanner = Scanner(
        data_provider=data_provider,
        candle_builder=candle_builder,
        mtf_engine=mtf_engine,
        signal_manager=signal_manager,
        intelligence=intelligence,
        trade_manager=trade_manager,
        performance=performance,
        instruments_config=instruments,
        trading_mode=settings.trading_mode,
        on_update=broadcast,
        on_notification=send_desktop_notification,
    )
    scanner.configure(
        capital_fut=settings.capital_fut,
        risk_fut_pct=settings.risk_fut_pct,
        capital_opt=settings.capital_opt,
        risk_opt_pct=settings.risk_opt_pct,
    )
    scanner.max_daily_loss_pct = settings.max_daily_loss_pct
    from engine.ut_ema_stream import install_tick_service
    install_tick_service(scanner)
    # Apply the same scheduled/manual policy before any scan or history worker
    # starts. Waiting for the timer lets a mid-session restart replay history with
    # sockets disabled, then discard that work on the delayed REAL transition.
    _auto_real_mode_attempt(scanner, settings, require_running=False)
    logger.info("✅ Scanner (exit-before-entry, 30s data cache)")

    # Set scanner reference for dashboard API
    dashboard_module.scanner_ref = scanner

    logger.info("═" * 60)
    logger.info(f"📊 Dashboard: http://localhost:{settings.dashboard_port}")
    logger.info(f"📊 Mode: {settings.trading_mode.upper()}")
    logger.info(f"📊 Max Daily Loss: {settings.max_daily_loss_pct}%")

    # Say out loud which behaviour-shaping values are ACTUALLY in force, and whether a
    # .env entry is masking the committed default.
    #
    # .env is the real settings store here and it is gitignored, so a value can differ
    # from the code on one machine and nobody can see it. This build has already been
    # bitten twice: .env was once pruned 755 keys -> 99 and the deleted keys silently
    # reverted to DIFFERENT code defaults, and on 2026-09-01 a stabilization change
    # looked applied while .env still pinned the old number. A committed default that
    # never takes effect is worse than no default, because the comment next to it
    # documents a behaviour the system does not have.
    try:
        from config.settings import Settings as _SettingsModel

        _watched = (
            "live_signal_stabilization_seconds",
            "live_signal_stabilization_seconds_5min",
            "live_signal_stabilization_seconds_15min",
            "loss_guard_never_green_cut_seconds",
            "live_5min_quality_min_confidence",
            "index_cooldown_minutes",
            "opposite_flip_confirm_min_seconds",
        )
        for _name in _watched:
            _field = _SettingsModel.model_fields.get(_name)
            if _field is None:
                continue
            _effective = getattr(settings, _name, None)
            _default = _field.default
            if str(_effective) != str(_default):
                logger.warning(
                    f"⚙️  {_name} = {_effective} (env overrides the committed default "
                    f"{_default})"
                )
            else:
                logger.info(f"⚙️  {_name} = {_effective}")
    except Exception as exc:  # never let telemetry block startup
        logger.debug(f"Effective-settings banner skipped: {exc}")

    logger.info("═" * 60)

    return scanner, settings


async def run_scanner(scanner: Scanner):
    """Run scanner in background"""
    await scanner.run()


def main():
    settings = get_settings()
    runtime: dict[str, object] = {}

    async def _bootstrap_system_after_dashboard_start():
        await asyncio.sleep(0.5)
        # Browser is launched directly by Start_UT1.bat using a TCP listener to avoid Python boot delays.
        try:
            dashboard_loop = asyncio.get_running_loop()
            scanner, _ = await asyncio.to_thread(create_system)
            runtime["scanner"] = scanner
            # Lifespan shutdown may begin while create_system is running in its
            # worker thread.  Publish the resource so the shutdown hook can close
            # it, but never launch a scanner thread after ingress has been gated.
            if bool(getattr(dashboard_module, "_dashboard_shutting_down", False)):
                logger.warning("Dashboard shutdown began during bootstrap; scanner launch skipped.")
                return

            async def _threadsafe_broadcast(payload):
                future = asyncio.run_coroutine_threadsafe(broadcast(payload), dashboard_loop)
                await asyncio.wrap_future(future)

            scanner.on_update = _threadsafe_broadcast
            # Immediate trade-panel refresh on entry/exit (thread-safe poke).
            scanner.on_trade_event = dashboard_module.notify_trade_change

            def _run_scanner_loop():
                asyncio.run(scanner.run())

            scanner_thread = threading.Thread(
                target=_run_scanner_loop,
                name="ScannerEventLoop",
                daemon=True,
            )
            # Retain the runner so graceful shutdown can prove the scan/event
            # loop has actually stopped before draining order work and releasing
            # the account execution lease.
            scanner._runner_thread = scanner_thread
            runtime["scanner_thread"] = scanner_thread
            scanner_thread.start()
            start_premarket_real_mode_worker(scanner, settings)
            start_runtime_watchdog(scanner, settings)
        except Exception as exc:
            logger.exception(f"System bootstrap failed after dashboard start: {exc}")

    dashboard_module.startup_hook = lambda: asyncio.create_task(_bootstrap_system_after_dashboard_start())

    async def _graceful_system_shutdown() -> None:
        """Drain broker work, persist scanner state, then release live resources."""
        scanner = runtime.get("scanner")
        await dashboard_module.shutdown_scanner()
        if scanner is None:
            return
        real_broker = getattr(getattr(scanner, "trades", None), "real_broker", None)
        release_lease = getattr(real_broker, "release_execution_lease", None)
        if callable(release_lease):
            try:
                await asyncio.to_thread(release_lease)
                logger.info("✅ Broker account execution lease released.")
            except Exception as exc:
                logger.error(f"Broker lease release failed during shutdown: {exc}")
        telemetry_shutdown = getattr(scanner, "shutdown_telemetry_executor", None)
        if callable(telemetry_shutdown):
            telemetry_shutdown(wait=False)
        lab_executor = getattr(scanner, "_engine_lab_executor", None)
        if lab_executor is not None:
            try:
                lab_executor.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:
                logger.debug(f"Engine-lab executor shutdown skipped: {exc}")
        # The dashboard's dedicated chart pool. Every other executor in this system is
        # shut down here; a new one that is not would keep the process alive on exit.
        chart_executor = getattr(dashboard_module, "_chart_executor", None)
        if chart_executor is not None:
            try:
                chart_executor.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:
                logger.debug(f"Chart executor shutdown skipped: {exc}")
        runtime.clear()

    dashboard_module.shutdown_hook = _graceful_system_shutdown

    # Start the log cleaner thread
    start_angelone_log_cleaner()
    start_live_session_tracker_scheduler()
    # The provider does not exist yet: it is built by create_system() on the bootstrap
    # task that runs after the dashboard starts, and is reachable as scanner.data.
    # Passing the object here referenced a name bound only inside create_system(), which
    # crashed startup with NameError before uvicorn ever came up. The archiver sleeps
    # until after the close, so a getter is resolved in time.
    start_session_archiver_scheduler(
        data_provider=lambda: getattr(runtime.get("scanner"), "data", None)
    )

    logger.info(f"Starting dashboard server on {settings.dashboard_host}:{settings.dashboard_port}")
    uvicorn.run(
        app,
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        log_level="warning",
        ws_max_size=16777216,
    )
    return


if __name__ == "__main__":
    main()
