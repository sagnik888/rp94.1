"""A history merge must not prevent the live loop from checking signals or exits."""

import asyncio
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

from scanner import Scanner


@pytest.mark.parametrize("warm_history", [False, True])
def test_live_history_merge_yields_to_scanner_loop(monkeypatch, warm_history):
    async def run():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        loop_served = threading.Event()
        calls = []
        phases = []
        frame = pd.DataFrame(
            {"close": range(10)},
            index=pd.date_range("2026-10-07 11:00", periods=10, freq="min"),
        )

        async def no_delay(*_args):
            return None

        monkeypatch.setattr(asyncio, "sleep", no_delay)

        def merge(name, data, timeframe, *, mark_receipt):
            # Simulate a slow merge/lock wait. The scanner must serve another callback
            # before the merge finishes, without relying on scheduler timing limits.
            loop.call_soon_threadsafe(loop_served.set)
            served_during_merge = loop_served.wait(timeout=1.0)
            calls.append((name, data.copy(), timeframe, mark_receipt,
                          threading.get_ident(), served_during_merge))

        scanner = SimpleNamespace(
            mode="REAL", backtest_days=2,
            _api_semaphore=asyncio.Semaphore(1),
            _history_rate_limiter=SimpleNamespace(consume=no_delay),
            data=SimpleNamespace(get_historical_candles=lambda *_a, **_k: frame),
            candles=SimpleNamespace(
                get_candle_count=lambda *_a: 600 if warm_history else 0,
                latest_candle_time=lambda *_a: pd.Timestamp("2026-10-07 11:07"),
                update_candles=merge,
            ),
            _set_candle_history_capacity=lambda **_k: None,
            _record_scan_phase_ms=lambda *a, **k: phases.append((a, k)),
        )
        await Scanner._fetch_one(scanner, "NIFTY", {"token": "1", "exchange": "NSE"}, "1min")
        assert len(calls) == 1
        name, merged, timeframe, receipt, merge_thread, served = calls[0]
        assert served, "history merge blocks signal/exit callbacks on scanner loop"
        assert merge_thread != loop_thread
        assert (name, timeframe, receipt) == ("NIFTY", "1min", False)
        expected = frame.loc["2026-10-07 11:05":] if warm_history else frame
        pd.testing.assert_frame_equal(merged, expected)
        assert phases[0][0][0] == "history_fetch_ms"

    asyncio.run(run())


def test_warm_upstox_minute_refresh_publishes_holes_behind_latest_tick(monkeypatch):
    frame = pd.DataFrame({"close": 100.0},
                         index=pd.date_range("2026-11-19 13:00", periods=30, freq="min"))
    requested_days, merged = [], []

    async def no_delay(*args):
        pass

    def fetch(*args, days_back, **kwargs):
        requested_days.append(days_back)
        return frame

    def merge(name, data, timeframe, *, mark_receipt):
        assert mark_receipt is False
        merged.append(data)

    monkeypatch.setattr(asyncio, "sleep", no_delay)

    async def run():
        scanner = SimpleNamespace(
            mode="REAL", backtest_days=30,
            _api_semaphore=asyncio.Semaphore(1),
            _history_rate_limiter=SimpleNamespace(consume=no_delay),
            data=SimpleNamespace(upstox_only=True, get_historical_candles=fetch),
            candles=SimpleNamespace(get_candle_count=lambda *_: 600,
                                    latest_candle_time=lambda *_: pd.Timestamp("2026-11-19 13:31"),
                                    update_candles=merge),
            _set_candle_history_capacity=lambda **_k: None,
            _record_scan_phase_ms=lambda *a, **k: None,
        )
        await Scanner._fetch_one(scanner, "NIFTY", {"token": "1", "exchange": "NSE"}, "1min",
                                 intraday_refresh=True)

    asyncio.run(run())
    assert requested_days == [0]
    pd.testing.assert_frame_equal(merged[0], frame)
