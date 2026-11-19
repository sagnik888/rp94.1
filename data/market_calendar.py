"""Versioned exchange holiday calendar helpers."""

from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path
from typing import Iterable, Set


NSE_HOLIDAYS_BY_YEAR = {
    2026: {
        "2026-01-15",
        "2026-01-26",
        "2026-03-03",
        "2026-03-26",
        "2026-03-31",
        "2026-04-03",
        "2026-04-14",
        "2026-05-01",
        "2026-05-28",
        "2026-06-26",
        "2026-09-14",
        "2026-10-02",
        "2026-10-20",
        "2026-11-10",
        "2026-11-24",
        "2026-12-25",
    }
}

NSE_SPECIAL_TRADING_DAYS_BY_YEAR = {
    2026: {
        "2026-02-01",
        "2026-11-08",
    }
}

# Index candle session for Diwali Muhurat trading (NSE/CMTR/76941).
# Derivatives have a separate schedule.
INDEX_SPECIAL_SESSION_HOURS = {
    "2026-11-08": (time(14, 20), time(15, 30)),
}


def index_session_bounds(at: datetime) -> tuple[datetime, datetime]:
    opening, closing = INDEX_SPECIAL_SESSION_HOURS.get(at.date().isoformat(), (time(9, 15), time(15, 30)))
    return datetime.combine(at.date(), opening, tzinfo=at.tzinfo), datetime.combine(at.date(), closing, tzinfo=at.tzinfo)


def _normalize_dates(values: Iterable[object]) -> Set[str]:
    dates: Set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        try:
            dates.add(datetime.fromisoformat(text[:10]).strftime("%Y-%m-%d"))
        except ValueError:
            continue
    return dates


def _payload_dates(payload: object) -> Set[str]:
    dates: Set[str] = set()
    if isinstance(payload, dict):
        for values in payload.values():
            if isinstance(values, list):
                dates.update(_normalize_dates(values))
    elif isinstance(payload, list):
        dates.update(_normalize_dates(payload))
    return dates


def _payload_year_keys(payload: object) -> Set[int]:
    years: Set[int] = set()
    if isinstance(payload, dict):
        for key in payload:
            try:
                years.add(int(str(key)[:4]))
            except (TypeError, ValueError):
                continue
    return years


def load_nse_holidays(path: str | Path = "data_store/market_holidays.json") -> Set[str]:
    """Load bundled holidays plus optional user-maintained overrides."""
    holidays: Set[str] = set()
    for year_dates in NSE_HOLIDAYS_BY_YEAR.values():
        holidays.update(year_dates)

    override_path = Path(path)
    if not override_path.exists():
        return holidays

    try:
        payload = json.loads(override_path.read_text(encoding="utf-8"))
    except Exception:
        return holidays

    holidays.update(_payload_dates(payload))
    return holidays


def load_nse_special_trading_days(
    path: str | Path = "data_store/market_special_sessions.json",
) -> Set[str]:
    """Load weekend/holiday dates on which NSE runs a normal trading session."""
    sessions: Set[str] = set()
    for year_dates in NSE_SPECIAL_TRADING_DAYS_BY_YEAR.values():
        sessions.update(year_dates)

    override_path = Path(path)
    if not override_path.exists():
        return sessions

    try:
        payload = json.loads(override_path.read_text(encoding="utf-8"))
    except Exception:
        return sessions

    sessions.update(_payload_dates(payload))
    return sessions


def load_nse_calendar_coverage(
    holiday_path: str | Path = "data_store/market_holidays.json",
    special_path: str | Path = "data_store/market_special_sessions.json",
) -> Set[int]:
    """Return years with an explicit bundled or override exchange calendar."""
    years = set(NSE_HOLIDAYS_BY_YEAR) | set(NSE_SPECIAL_TRADING_DAYS_BY_YEAR)
    for path in (Path(holiday_path), Path(special_path)):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        years.update(_payload_year_keys(payload))
        for date_str in _payload_dates(payload):
            try:
                years.add(int(date_str[:4]))
            except (TypeError, ValueError):
                continue
    return years


# ── Macro-event blackout windows (RBI policy, budget, etc.) ──────────────────
# User-maintained file: data_store/event_blackouts.json
#   [{"date": "2026-08-06", "start": "09:45", "end": "10:30", "label": "RBI policy"}]
# Gated by settings.event_blackout_enabled (default OFF): fresh ENTRIES are
# blocked inside an active window; exits/trailing are never touched.

def load_event_blackout_windows(path: str | Path = "data_store/event_blackouts.json") -> list:
    """Return normalized [{date,start,end,label}] windows; [] when absent/invalid."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    windows = []
    rows = payload if isinstance(payload, list) else payload.get("windows", []) if isinstance(payload, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        date = str(row.get("date") or "").strip()
        start = str(row.get("start") or "").strip()
        end = str(row.get("end") or "").strip()
        try:
            datetime.strptime(date, "%Y-%m-%d")
            start_t = datetime.strptime(start, "%H:%M").time()
            end_t = datetime.strptime(end, "%H:%M").time()
        except (TypeError, ValueError):
            continue
        if end_t <= start_t:
            continue
        windows.append({
            "date": date,
            "start": start,
            "end": end,
            "label": str(row.get("label") or "macro event").strip() or "macro event",
        })
    return windows


def active_event_blackout(now: datetime, windows: list) -> str:
    """Return the active window's label if `now` falls inside one, else ""."""
    if not windows:
        return ""
    date_str = now.strftime("%Y-%m-%d")
    now_t = now.time()
    for win in windows:
        if win.get("date") != date_str:
            continue
        try:
            start_t = datetime.strptime(str(win.get("start")), "%H:%M").time()
            end_t = datetime.strptime(str(win.get("end")), "%H:%M").time()
        except (TypeError, ValueError):
            continue
        if start_t <= now_t <= end_t:
            return str(win.get("label") or "macro event")
    return ""
