"""An absent option chain must not starve independent candle intelligence."""
import asyncio
from types import SimpleNamespace
import pandas as pd
from intelligence.intelligence_aggregator import IntelligenceAggregator
from tests.test_intel_refresh_starvation_2026_10_23 import _source


def test_missing_chain_still_publishes_all_index_timeframes(monkeypatch):
    async def scenario():
        source=_source()
        source._get_option_chain_cached=lambda *a,**k:(None,{})
        seen=[]
        def analyze(**kw):
            seen.append(kw)
            return {"regime":{"regime":"RANGING","candle_count":40},
                    "volume":{"available":True,"current_volume":123.},
                    "data_quality":kw["chain_quality"]}
        source.intel=SimpleNamespace(analyze=analyze)
        real_sleep=asyncio.sleep
        async def poll(seconds):await real_sleep(.002 if seconds>=1 else 0)
        monkeypatch.setattr(asyncio,"sleep",poll)
        worker=asyncio.create_task(source._background_intel_worker())
        async def ready():
            while not all(f"{name}_{tf}" in source._intel_cache
                for name in source.active_indices for tf in ("1min","5min","15min")):
                await real_sleep(.002)
        try:
            await asyncio.wait_for(ready(),2.)
            assert len(seen)>=12
            assert all(row["options_chain"].empty and not row["chain_quality"]["intelligence_usable"] for row in seen)
            assert source._intel_cache["NIFTY_5min"]["volume"]["current_volume"]==123.
        finally:
            source.is_running=False
            worker.cancel()
            await asyncio.gather(worker,return_exceptions=True)
    asyncio.run(scenario())


def test_empty_chain_keeps_measured_volume_and_marks_oi_pcr_unavailable():
    index = pd.date_range("2026-10-30 09:15", periods=40, freq="5min")
    frame = pd.DataFrame({
        "open": range(100, 140), "high": range(101, 141),
        "low": range(99, 139), "close": range(100, 140),
        "volume": [1000] * 40,
    }, index=index)
    result = IntelligenceAggregator().analyze(
        "NIFTY", "5min", frame, options_chain=pd.DataFrame(), spot_price=139,
        strike_interval=50, chain_quality={"intelligence_usable": False},
    )
    assert result["volume"]["available"] is True
    assert result["oi"]["available"] is False
    assert result["pcr"]["available"] is False
    assert result["data_quality"]["reason"] == "Option chain unavailable"


def test_historical_panel_uses_the_last_closed_replay_bar():
    from scanner import Scanner

    scanner = Scanner.__new__(Scanner)
    scanner.candles = SimpleNamespace(get_candles=lambda _, tf: pd.DataFrame(
        {"close": [25000.]}, index=[pd.Timestamp({
            "5min": "2026-11-19 14:15", "1min": "2026-11-19 14:15",
        }[tf])]
    ))
    calls = []
    def lookup(instrument, clock, timeframe, *, panel):
        calls.append((instrument, clock, timeframe, panel))
        return {"pcr": {"primary_pcr": 1.2}, "as_of": clock.isoformat()}
    scanner._get_historical_intel_context_at = lookup
    scanner._live_regime_horizons = lambda *args, as_of, **kwargs: {"day": {"regime": "RANGING"}}

    result = scanner._historical_panel_intel("NIFTY")
    assert calls == [("NIFTY", pd.Timestamp("2026-11-19T14:15:00+05:30"), "5min", True)]
    assert result["pcr"]["primary_pcr"] == 1.2
    assert result["regime_horizons"]["day"]["regime"] == "RANGING"


def test_upstox_option_volume_stays_separate_from_missing_spot_volume():
    frame = pd.DataFrame({name: [25000.] * 40 for name in ("open", "high", "low", "close")},
                         index=pd.date_range("2026-11-19 09:15", periods=40, freq="5min"))
    frame["volume"] = 0.0
    chain = pd.DataFrame({"strike": [24950, 25000, 25050], "call_oi": [100] * 3, "put_oi": [120] * 3,
                          "call_volume": [10] * 3, "put_volume": [20] * 3})
    chain.attrs["expiry"] = "2026-11-23"
    result = IntelligenceAggregator().analyze(
        "NIFTY", "5min", frame, options_chain=chain, spot_price=25000.,
        chain_quality={"source": "upstox", "intelligence_usable": True, "volume_complete": True})
    assert result["volume"]["available"] is False
    assert result["option_volume"]["quantity"] == 90
    assert result["oi"]["available"] and result["pcr"]["primary_pcr"] == 1.2
