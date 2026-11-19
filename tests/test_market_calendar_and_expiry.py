from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from data.market_calendar import (
    load_nse_calendar_coverage,
    load_nse_holidays,
    load_nse_special_trading_days,
)
from data.market_data import MarketDataProvider
from engine.expiry_manager import ExpiryManager
from engine.safe_mode import validate_real_mode_readiness
from scanner import Scanner


def _expiry_rows(name, expiries, instrumenttype="OPTIDX"):
    rows = []
    for expiry in expiries:
        rows.append({
            "name": name,
            "instrumenttype": instrumenttype,
            "expiry": expiry,
            "symbol": f"{name}{expiry}100CE",
            "strike": "10000",
            "token": f"{name}-{expiry}",
        })
    return rows


def test_2026_exchange_calendar_includes_added_holiday_and_special_sunday():
    assert "2026-01-15" in load_nse_holidays()
    assert "2026-02-01" in load_nse_special_trading_days()


def test_market_open_honors_holiday_weekend_and_special_session():
    provider = MarketDataProvider.__new__(MarketDataProvider)
    provider.nse_holidays = load_nse_holidays()
    provider.nse_special_trading_days = load_nse_special_trading_days()

    assert provider.is_market_open(datetime(2026, 1, 15, 10, 0)) is False
    assert provider.is_market_open(datetime(2026, 1, 17, 10, 0)) is False
    assert provider.is_market_open(datetime(2026, 2, 1, 10, 0)) is True
    assert provider.is_market_open(datetime(2026, 2, 1, 8, 0)) is False


def test_unknown_future_calendar_year_fails_closed_until_override(tmp_path):
    missing_holidays = tmp_path / "missing_holidays.json"
    missing_special = tmp_path / "missing_special.json"
    provider = MarketDataProvider.__new__(MarketDataProvider)
    provider.nse_holidays = load_nse_holidays(missing_holidays)
    provider.nse_special_trading_days = load_nse_special_trading_days(missing_special)
    provider.nse_calendar_coverage_years = load_nse_calendar_coverage(
        missing_holidays,
        missing_special,
    )

    assert provider.is_market_open(datetime(2027, 1, 4, 10, 0)) is False

    holidays_file = tmp_path / "market_holidays.json"
    holidays_file.write_text('{"2027": ["2027-01-26"]}', encoding="ascii")
    provider.nse_holidays = load_nse_holidays(holidays_file)
    provider.nse_calendar_coverage_years = load_nse_calendar_coverage(
        holidays_file,
        missing_special,
    )

    assert provider.is_market_open(datetime(2027, 1, 4, 10, 0)) is True
    assert provider.is_market_open(datetime(2027, 1, 26, 10, 0)) is False


def test_weekly_and_monthly_options_roll_one_trading_day_before_expiry(monkeypatch):
    # Rollover is now measured in TRADING days (option_rollover_working_days=1): the system
    # holds the current expiry until the last trading session before it, rolling on T-1
    # (trading) and expiry day — one session later than the old T-2 calendar rule.
    # Isolate from the operator's .env: expiry-day gamma mode keeps the SAME-DAY
    # expiry on T-0 and would legitimately change the expiry-day assertion below.
    from config.settings import get_settings

    monkeypatch.setattr(get_settings(), "expiry_day_trade_current_week", False)
    manager = ExpiryManager()
    manager.df = pd.DataFrame(
        _expiry_rows("NIFTY", ["18JUN2026", "25JUN2026", "02JUL2026"])
        + _expiry_rows("BANKNIFTY", ["25JUN2026", "30JUL2026", "27AUG2026"])
    )
    # NIFTY weekly expiry Thu 2026-06-18:
    # Tue 06-16 = 2 trading sessions out -> still CURRENT (old rule would have rolled here)
    assert manager._select_option_expiry_for_date("NIFTY", date(2026, 6, 16), "weekly").date() == date(2026, 6, 18)
    # Wed 06-17 = last trading session before expiry (T-1) -> ROLL to next week
    assert manager._select_option_expiry_for_date("NIFTY", date(2026, 6, 17), "weekly").date() == date(2026, 6, 25)
    # Expiry day itself -> rolled
    assert manager._select_option_expiry_for_date("NIFTY", date(2026, 6, 18), "weekly").date() == date(2026, 6, 25)

    # BANKNIFTY monthly expiry Thu 2026-06-25:
    # Tue 06-23 = 2 sessions out -> still CURRENT; Wed 06-24 (T-1) -> ROLL to 30JUL
    assert manager._select_option_expiry_for_date("BANKNIFTY", date(2026, 6, 23), "monthly").date() == date(2026, 6, 25)
    assert manager._select_option_expiry_for_date("BANKNIFTY", date(2026, 6, 24), "monthly").date() == date(2026, 7, 30)


def test_expiry_day_detection_uses_expiring_contract_after_rollover():
    manager = ExpiryManager()
    manager.df = pd.DataFrame(_expiry_rows("NIFTY", ["18JUN2026", "25JUN2026"]))
    manager.expiries = {
        "NIFTY": {
            "weekly": "18JUN2026",
            "active_weekly": "25JUN2026",
        }
    }

    assert manager.is_expiry_day("NIFTY", date(2026, 6, 18)) is True
    assert manager.is_expiry_day("NIFTY", date(2026, 6, 19)) is False


def test_scanner_trading_day_allows_declared_special_session():
    scanner = Scanner.__new__(Scanner)
    scanner.data = SimpleNamespace(
        is_special_trading_session=lambda dt: dt.date() == date(2026, 2, 1),
        is_market_holiday=lambda dt: dt.date() == date(2026, 1, 15),
    )

    assert scanner._is_trading_day(datetime(2026, 2, 1, 10, 0)) is True
    assert scanner._is_trading_day(datetime(2026, 1, 17, 10, 0)) is False
    assert scanner._is_trading_day(datetime(2026, 1, 15, 10, 0)) is False


def test_real_mode_safety_allows_special_session_but_blocks_normal_weekend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("", encoding="ascii")
    _use_healthy_upstox_session(monkeypatch)
    settings = SimpleNamespace(
        upstox_api_key="key",
        upstox_api_secret="secret",
    )
    provider = SimpleNamespace(
        is_connected=True,
        get_source_health=lambda: {},
        is_market_holiday=lambda dt: False,
        is_special_trading_session=lambda dt: dt.date() == date(2026, 2, 1),
    )

    special = validate_real_mode_readiness(
        settings, provider, datetime(2026, 2, 1, 10, 0)
    )
    normal_weekend = validate_real_mode_readiness(
        settings, provider, datetime(2026, 2, 8, 10, 0)
    )

    assert special["ok"] is True
    assert "weekend session" in normal_weekend["blockers"]



def _use_healthy_upstox_session(monkeypatch):
    """A valid data session isolates the calendar gates."""
    from datetime import datetime as _dt

    import pytz as _pytz

    class _Auth:
        def status(self):
            now = _dt.now(_pytz.timezone("Asia/Kolkata"))
            return {
                "connected": True, "auth_required": False, "reason": "",
                "user_id": "TEST", "token_updated_at": now.isoformat(),
            }

    monkeypatch.setattr("data.upstox_auth.get_upstox_auth", lambda: _Auth())


def test_real_mode_safety_allows_premarket_arming_window(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("", encoding="ascii")
    _use_healthy_upstox_session(monkeypatch)
    settings = SimpleNamespace(
        upstox_api_key="key",
        upstox_api_secret="secret",
        real_mode_premarket_arm_start="09:00:00",
        real_mode_premarket_settle_end="09:08:00",
        real_mode_premarket_refresh_end="09:14:30",
        real_mode_live_start="09:15:00",
        real_mode_live_end="15:30:00",
    )
    provider = SimpleNamespace(
        is_connected=True,
        get_source_health=lambda: {},
        is_market_holiday=lambda dt: False,
        is_special_trading_session=lambda dt: False,
    )

    before_arm = validate_real_mode_readiness(
        settings, provider, datetime(2026, 6, 16, 8, 59, 59)
    )
    arm_start = validate_real_mode_readiness(
        settings, provider, datetime(2026, 6, 16, 9, 0, 0)
    )
    settle_end = validate_real_mode_readiness(
        settings, provider, datetime(2026, 6, 16, 9, 8, 0)
    )
    refresh_end = validate_real_mode_readiness(
        settings, provider, datetime(2026, 6, 16, 9, 14, 30)
    )
    handoff = validate_real_mode_readiness(
        settings, provider, datetime(2026, 6, 16, 9, 14, 45)
    )
    live_start = validate_real_mode_readiness(
        settings, provider, datetime(2026, 6, 16, 9, 15, 0)
    )

    assert before_arm["ok"] is False
    assert any("outside REAL arming window" in item for item in before_arm["blockers"])
    assert arm_start["ok"] is True
    assert settle_end["ok"] is True
    assert refresh_end["ok"] is True
    assert handoff["ok"] is True
    assert live_start["ok"] is True
    assert any("pre-open settlement window" in item for item in arm_start["warnings"])
    assert any("pre-open settlement window" in item for item in settle_end["warnings"])
    assert any("pre-open analysis window" in item for item in refresh_end["warnings"])
    assert any("market-open handoff window" in item for item in handoff["warnings"])


def test_auto_instrument_selection_routes_iv_momentum_and_late_expiry():
    scanner = Scanner.__new__(Scanner)
    scanner.inst_pref = "AUTO"
    scanner.mode = "HISTORICAL"
    scanner.signal_processor = SimpleNamespace(
        _option_grade_allowed=lambda *args, **kwargs: True
    )
    scanner._historical_expiry_resolver = SimpleNamespace(
        resolve=lambda instrument, as_of: {
            "is_expiry_day": pd.Timestamp(as_of).date() == date(2026, 6, 18),
            "in_gamma_window": False,
        }
    )
    settings = SimpleNamespace(ut_backtest_more_results=True)

    with patch("scanner.get_settings", return_value=settings):
        assert scanner._resolve_instrument_type(
            "NIFTY", "A", "TRENDING", confidence=0.8, iv_percentile=80
        ) == "FUT"
        assert scanner._resolve_instrument_type(
            "NIFTY", "A", "TRENDING", confidence=0.8, iv_percentile=40
        ) == "OPT"
        assert scanner._resolve_instrument_type(
            "NIFTY",
            "C",
            "TRENDING",
            confidence=0.4,
            signal_time=datetime(2026, 6, 18, 14, 15),
            iv_percentile=40,
        ) == "FUT"


def test_auto_instrument_selection_prefers_fut_in_risky_regime_without_premium_momentum():
    scanner = Scanner.__new__(Scanner)
    scanner.inst_pref = "AUTO"
    scanner.mode = "REAL"
    scanner.signal_processor = SimpleNamespace(
        _option_grade_allowed=lambda *args, **kwargs: True
    )
    scanner.expiry = SimpleNamespace(is_expiry_day=lambda *_args: False)
    # Base routing rule under test: lean-to-options OFF (the new lean default is
    # covered separately) so risky-regime-without-premium-momentum -> FUT holds.
    settings = SimpleNamespace(ut_backtest_more_results=True, lean_routing_to_options=False)

    with patch("scanner.get_settings", return_value=settings):
        assert scanner._resolve_instrument_type(
            "NIFTY",
            "A+",
            "RANGING",
            confidence=1.0,
            signal_score=0.25,
            intel_score=0.80,
            adx_value=55.0,
            iv_percentile=40,
        ) == "FUT"
        assert scanner._resolve_instrument_type(
            "NIFTY",
            "A+",
            "RANGING",
            confidence=0.90,
            signal_score=0.50,
            intel_score=0.60,
            adx_value=40.0,
            iv_percentile=40,
        ) == "OPT"
