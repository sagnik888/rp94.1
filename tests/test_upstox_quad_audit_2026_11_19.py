from datetime import date
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from data.market_source_health import build_market_source_snapshot
from data.upstox_market_data import UPSTOX_INDEX_KEYS
from engine.alt_scanners import AltSignalEngine, _nb_supertrend
from engine.combined_scanner import CombinedSignalEngine
from engine.entry_risk_policy import evaluate_entry_risk, projected_stop_risk_rs
from engine.expiry_manager import ExpiryManager, expiry_manager
from engine.scanner_modes import owner
from engine.signal_processor import SignalProcessor
from engine.trade_accounting import estimate_upstox_derivative_fees
from trading import upstox_broker
from trading.order_retry import BrokerOrderRejected
from trading.trade_manager import TradeManager


def test_exit_and_protective_stop_keep_the_resolved_side(monkeypatch):
    broker = upstox_broker.UpstoxBroker(account_id="TEST")
    broker.set_execution_authority(lambda: True)
    monkeypatch.setattr(broker, "has_execution_lease", lambda: True)
    monkeypatch.setattr(upstox_broker, "get_upstox_auth_status", lambda: {"user_id": "TEST"})
    monkeypatch.setattr(upstox_broker, "get_upstox_token", lambda: "mock-token")
    monkeypatch.setattr(expiry_manager, "upstox_tick_size", lambda key: .05)
    reply = Mock(status_code=200)
    reply.json.return_value = {"data": {"order_id": "123"}}
    post = Mock(return_value=reply)
    monkeypatch.setattr(upstox_broker.requests, "post", post)

    contract = "SENSEX 82500 CE 19 NOV 26"
    key = "BSE_FO|888"
    assert broker.close_order(contract, key, 20, "SELL") == "123"
    payload = post.call_args.kwargs["json"]
    assert (payload["transaction_type"], payload["instrument_token"]) == ("SELL", key)
    assert payload["market_protection"] == -1

    assert broker.place_protective_stop(contract, key, 20, "SELL", 200) == "123"
    payload = post.call_args.kwargs["json"]
    assert (payload["transaction_type"], payload["order_type"]) == ("SELL", "SL")
    assert 0 < payload["price"] < payload["trigger_price"]

    broker.set_execution_authority(lambda: False)
    with pytest.raises(BrokerOrderRejected, match="requires REAL"):
        broker.close_order(contract, key, 20, "SELL")
    assert post.call_count == 2


def test_stop_discovery_distinguishes_unavailable_and_multiple_or_wrong_stops(monkeypatch):
    broker = upstox_broker.UpstoxBroker()
    row = {"instrument_token": "BSE_FO|888", "trading_symbol": "SENSEX 82500 CE 19 NOV 26",
           "transaction_type": "SELL", "order_type": "SL", "status": "trigger pending",
           "order_id": "123", "quantity": 20, "trigger_price": 140, "product": "I"}
    read = Mock(side_effect=[None, [], [row, row], [row], [row]])
    monkeypatch.setattr(broker, "_order_book", read)
    inspect = broker.inspect_working_protective_stop
    args = (row["trading_symbol"], row["instrument_token"], "SELL")
    assert inspect(*args, 20)["state"] == "UNKNOWN"
    assert inspect(*args, 20)["state"] == "NONE"
    assert inspect(*args, 20)["state"] == "AMBIGUOUS"
    assert inspect(*args, 40)["state"] == "CONFLICT"
    assert inspect(*args, 20, "INTRADAY") == {"state": "FOUND", "order_id": "123", "trigger_price": 140.0}


def test_upstox_active_contract_master_preserves_real_symbol_and_key():
    raw = {"underlying_symbol": "SENSEX", "underlying_key": UPSTOX_INDEX_KEYS["SENSEX"],
           "segment": "BSE_FO", "instrument_type": "PE", "instrument_key": "BSE_FO|888",
           "trading_symbol": "SENSEX 82500 PE 19 NOV 26", "expiry": 1795084199000,
           "lot_size": 20, "strike_price": 82500, "tick_size": 5}
    frame = ExpiryManager._normalize_upstox_rows([raw])
    row = frame.iloc[0]
    assert (row["symbol"], row["token"]) == (raw["trading_symbol"], raw["instrument_key"])
    assert (row["lotsize"], row["strike"], row["tick_size"]) == (20, 82500 * 100, .05)
    assert ExpiryManager._option_type_mask(frame, "PE").tolist() == [True]
    assert ExpiryManager._option_type_mask(frame, "CE").tolist() == [False]


def test_only_fresh_upstox_quote_can_authorize_entry():
    key = UPSTOX_INDEX_KEYS["NIFTY"]
    provider = SimpleNamespace(upstox_only=True, _ws_cache={
        key: {"time": 99.0, "source": "upstox_ws", "ltp": 25000},
        "NIFTY": {"time": 100.0, "source": "kite_ws", "ltp": 25000},
    }, _ltp_cache={})
    assert build_market_source_snapshot(provider, "NIFTY", {}, {}, 100.0)["entry_eligible"]
    assert not build_market_source_snapshot(provider, "NIFTY", {}, {}, 105.1)["entry_eligible"]


def test_quad_vote_requires_two_and_supertrend_retains_recursive_bands():
    engine = AltSignalEngine(signal_source="quad_prime")
    positions, _ = engine._quad_union_consensus_positions(
        np.array([1, 1, -1, -1]), np.array([0, 1, 0, -1]), np.zeros(4, dtype=int)
    )
    assert positions.tolist() == [0, 1, 1, -1]
    _, upper, lower = _nb_supertrend(np.array([10., 10.]), np.array([12., 14.]), np.array([8., 7.]))
    assert (upper[-1], lower[-1]) == (12., 8.)


def test_opposite_arrows_preserved_as_evidence_but_abstain(monkeypatch):
    engine = AltSignalEngine(signal_source="quad_prime")
    index = pd.date_range("2026-11-19 09:15", periods=32, freq="5min")
    close = np.linspace(100, 130, len(index))
    frame = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                          "close": close, "volume": 100}, index=index)
    buy = SimpleNamespace(bar_index=31, signal_type="BUY", source_engine="utbot")
    sell = SimpleNamespace(bar_index=31, signal_type="SELL", source_engine="ema")
    monkeypatch.setattr(engine, "_merge_union_signals", lambda raw: [buy, sell])
    result = engine.process(frame, instrument="NIFTY", timeframe="5min")
    assert result["signals"] == []
    assert result["conflicted_signals"] == [buy, sell]
    assert result["conflicted_bars"] == [31]


def test_combined_clock_and_owner_scope_with_degraded_quad(monkeypatch):
    engine = CombinedSignalEngine("quad_prime_s1")
    stamps = pd.date_range("2026-11-19 09:15", periods=3, freq="5min")
    df = pd.DataFrame({"close": [1, 2, 3]}, index=stamps)
    approved = SimpleNamespace(source_engine="s1", timestamp=stamps[0], signal_type="BUY")
    unknown = SimpleNamespace(source_engine="custom", timestamp=stamps[0], signal_type="BUY")

    def failed(**kwargs):
        assert "as_of" not in kwargs
        assert list(kwargs["df"].index) == list(stamps[:2])
        raise RuntimeError("feed unavailable")

    def healthy(**kwargs):
        assert list(kwargs["df_15min"].index) == list(stamps[:2])
        return {"signals": [approved, unknown]}

    monkeypatch.setattr(engine.native_engines["quad_prime"], "process", failed)
    monkeypatch.setattr(engine.native_engines["s1"], "process", healthy)
    result = engine.process(df, as_of=stamps[1], df_15min=df)
    assert result["combined_health"] == "DEGRADED"
    assert result["native_entry_eligible"] == {"quad_prime": False, "s1": True}
    assert owner(unknown) == "unknown"
    assert result["native_results"]["quad_prime"]["signals"] == []


def test_merged_replay_keeps_event_and_all_sources():
    signal = SimpleNamespace(timestamp=pd.Timestamp("2026-11-19 10:00"),
                             source_engine="ema", source_engines=["ema", "utbot"])
    merged = SignalProcessor._historical_replay_order([signal], "quad_prime", source_attribution=False)
    assert len(merged) == 1
    assert merged[0].source_engines == ["ema", "utbot"]
    assert len(SignalProcessor._historical_replay_order([signal], "quad_prime")) == 2


def test_risk_policy_rejects_nonfinite_inputs_and_prices_real_multiplier():
    trade = SimpleNamespace(inst_type="FUT", direction="SELL", entry_price=100, current_stop=105,
                            lots=1, lot_size=20, instrument_multiplier=2)
    assert projected_stop_risk_rs(trade) == 200
    trade.lot_size = float("inf")
    assert projected_stop_risk_rs(trade) is None
    trade.lot_size = 20
    assert not evaluate_entry_risk(trade, [], fut_limit_rs=float("nan"), opt_limit_rs=500,
                                   global_limit_rs=1000).allowed


def test_upstox_live_gate_rejects_over_budget_one_lot_even_with_optional_gate_off(monkeypatch):
    import config.settings

    settings = SimpleNamespace(entry_risk_budget_enabled=False, capital_fut=100_000,
                               capital_opt=100_000, capital_total=200_000,
                               risk_fut_pct=5, risk_opt_pct=1, max_daily_loss_pct=5,
                               max_trade_loss_abs=0, slippage_auto_calibration_enabled=False)
    monkeypatch.setattr(config.settings, "get_settings", lambda: settings)
    manager = TradeManager.__new__(TradeManager)
    manager.mode = "REAL"
    manager.broker = upstox_broker.UpstoxBroker()
    manager.open_trades = {}
    manager.closed_trades = []
    fields = dict(inst_type="OPT", direction="LONG", price=200, trailing_stop=185,
                  initial_stop=185, lots=1, lot_size=65, instrument_multiplier=1,
                  instrument="NIFTY", is_recovery=False)
    assert manager._entry_risk_admission(**fields)[0] is False
    assert manager._entry_risk_admission(**{**fields, "trailing_stop": 195})[0] is True
    assert manager._entry_risk_admission(**{**fields, "is_recovery": True})[0] is True


def test_new_upstox_trade_waits_for_reconciled_positions():
    manager = TradeManager.__new__(TradeManager)
    manager.mode = "REAL"
    manager.broker = upstox_broker.UpstoxBroker()
    manager._lock = threading.RLock()
    manager.orphan_positions = {}
    manager.open_trades = {}
    manager._effective_max_positions = lambda: 5
    assert not manager.can_open_trade()[0]
    assert manager.can_open_trade(is_recovery=True)[0]
    manager._live_position_snapshot = (manager.broker, time.monotonic())
    assert manager.can_open_trade()[0]


def test_upstox_round_trip_fees_include_trade_date_and_sell_side_stt():
    pair = dict(entry=200, exit_price=100, quantity=65, inst_type="OPT", instrument="NIFTY",
                direction="SHORT")
    after = estimate_upstox_derivative_fees(**pair, trade_date=date(2026, 4, 1))
    before = estimate_upstox_derivative_fees(**pair, trade_date=date(2026, 3, 31))
    assert after - before == pytest.approx(100 * 65 * .0005)
