"""Cached option chains must not be flagged stale merely because we deferred a re-fetch.

IntelligenceAggregator drops the option chain when chain_quality["stale"] is True, so
OI / PCR / Greeks report "OPTIONS DATA UNAVAILABLE". In REAL mode the scan loop sets
allow_fetch=False once a chain is cached (to stay under broker rate limits), and
_get_option_chain_cached used to mark every such deferred read stale=True. The result:
chains were fetched successfully all session, but intelligence never saw one.

`stale` must reflect genuine AGE against option_chain_intelligence_max_age_seconds.
"""
import time

import pandas as pd
import pytest

from scanner import Scanner


def _scanner(age_seconds: float) -> Scanner:
    sc = Scanner.__new__(Scanner)
    chain = pd.DataFrame({"strike": [100, 101, 102], "oi": [1, 2, 3]})
    sc._chain_cache = {"NIFTY": chain}
    sc._chain_quality_cache = {
        "NIFTY": {"source": "fyers", "strike_count": 3, "fallback": False, "expiry_aligned": True}
    }
    sc._last_chain_fetch = {"NIFTY": time.time() - age_seconds}
    return sc


def _authoritative(quality: dict, actual_strikes: int) -> bool:
    """Mirrors IntelligenceAggregator's gate."""
    reported = int(quality.get("strike_count", actual_strikes) or 0)
    usable = quality.get(
        "intelligence_usable",
        min(actual_strikes, reported) >= 3
        and not bool(quality.get("fallback", False))
        and not bool(quality.get("stale", False))
        and bool(quality.get("expiry_aligned", True)),
    )
    return bool(usable) and actual_strikes >= 3


@pytest.mark.parametrize("age", [0.0, 30.0, 120.0, 299.0])
def test_recent_cached_chain_is_not_stale_and_feeds_intelligence(age):
    sc = _scanner(age)
    _, quality = Scanner._get_option_chain_cached(sc, "NIFTY", {}, allow_fetch=False)
    assert quality["stale"] is False
    assert _authoritative(quality, 3) is True


@pytest.mark.parametrize("age", [301.0, 900.0])
def test_genuinely_old_chain_is_stale_and_is_dropped(age):
    sc = _scanner(age)
    _, quality = Scanner._get_option_chain_cached(sc, "NIFTY", {}, allow_fetch=False)
    assert quality["stale"] is True
    assert _authoritative(quality, 3) is False


def test_age_seconds_is_reported():
    sc = _scanner(45.0)
    _, quality = Scanner._get_option_chain_cached(sc, "NIFTY", {}, allow_fetch=False)
    assert 44.0 <= quality["age_seconds"] <= 47.0


def test_threshold_is_configurable():
    from config.settings import Settings

    assert Settings(_env_file=None).option_chain_intelligence_max_age_seconds == 300.0


def test_cached_upstox_snapshot_keeps_oi_without_exchange_leg_clocks_or_volume():
    sc = _scanner(60.0)
    sc._chain_quality_cache["NIFTY"].update(
        source="upstox", volume_complete=False, coverage_complete=True,
        exchange_clocks_fresh=False, fetched_at_epoch=time.time() - 60.0,
    )
    _, quality = sc._get_option_chain_cached("NIFTY", {}, allow_fetch=False)
    assert quality["intelligence_usable"] is True
    assert quality["volume_complete"] is False
    assert quality["exchange_clocks_fresh"] is False
    assert quality["score"] == 0


def test_failed_refresh_does_not_restamp_a_stale_upstox_chain():
    from types import SimpleNamespace

    sc = _scanner(320.0)
    recorded_at = time.time() - 320.0
    sc._chain_quality_cache["NIFTY"].update(source="upstox", fetched_at_epoch=recorded_at)
    sc.data = SimpleNamespace(get_option_chain=lambda *args: pd.DataFrame())
    before, first = sc._get_option_chain_cached("NIFTY", {}, force=True)
    same, next_read = sc._get_option_chain_cached("NIFTY", {})
    assert same is before
    for quality in (first, next_read):
        assert quality["fetched_at_epoch"] == recorded_at
        assert quality["stale"] is True
        assert quality["intelligence_usable"] is False
