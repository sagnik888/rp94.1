from datetime import datetime
from types import SimpleNamespace

from engine.risk_manager import RiskManager


class _Trades:
    def __init__(self, open_trades=None):
        self.closed_trades = []
        self.open_trades = open_trades or {}
        self.closed = []

    def close_trade(self, trade_id, price, reason):
        self.closed.append((trade_id, price, reason))


def _open_trade(**overrides):
    values = {
        "entry_time": datetime.now(),
        "inst_type": "OPT",
        "entry_price": 100.0,
        "current_price": 100.0,
        "unrealized_pnl": 0.0,
        "lots": 20,
        "lot_size": 50,
        "instrument_multiplier": 1.0,
        "trailing_stop": None,
        "direction": "LONG",
        "is_ghost": False,
        "execution_status": "OPEN",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _closed_trade(**overrides):
    values = {
        "id": "closed-1",
        "entry_time": datetime.now(),
        "inst_type": "OPT",
        "pnl": 0.0,
        "is_ghost": False,
        "trading_symbol": "NIFTY31AUG24000CE",
        "grade": "A",
        "data_provenance": "LIVE",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_projected_stop_risk_uses_tighter_live_stop_and_trade_cap():
    manager = RiskManager(_Trades(), 100000.0)
    manager.update_settings(options_sl_pct=15.0, max_trade_loss_abs=5000.0)

    assert manager._projected_stop_risk(_open_trade(lots=1, lot_size=100, trailing_stop=95.0)) == 500.0
    assert manager._projected_stop_risk(_open_trade(lots=10, lot_size=100, trailing_stop=None)) == 5000.0


def test_global_breaker_halts_on_projected_stop_loss_before_mark_to_market_loss():
    trade = _open_trade()
    trades = _Trades({"opt-1": trade})
    manager = RiskManager(trades, 100000.0)
    manager.update_settings(
        max_daily_loss_pct=10.0,
        capital_fut=50000.0,
        capital_opt=50000.0,
        capital_total=100000.0,
        options_sl_pct=15.0,
        max_trade_loss_abs=50000.0,
        circuit_breaker_slippage_bps=0.0,
    )

    assert manager.check_circuit_breaker() is True
    assert manager.daily_loss_breached is True
    assert trades.closed == [("opt-1", 100.0, "GLOBAL_DAILY_LOSS_LIMIT")]


def test_closed_pnl_cache_invalidates_when_pnl_changes_without_count_change():
    trade = _closed_trade()
    trades = _Trades()
    trades.closed_trades = [trade]
    manager = RiskManager(trades, 100000.0)

    assert manager.check_circuit_breaker() is False

    # Reconciliation can correct a closed row in place; list length is unchanged.
    trade.pnl = -5000.0

    assert manager.check_circuit_breaker() is True
    assert manager.daily_loss_breached is True


def test_closed_pnl_cache_invalidates_when_non_live_row_becomes_live():
    trade = _closed_trade(is_ghost=True, pnl=-5000.0)
    trades = _Trades()
    trades.closed_trades = [trade]
    manager = RiskManager(trades, 100000.0)

    assert manager.check_circuit_breaker() is False

    # Risk inclusion fields are part of the fingerprint, not just P&L and count.
    trade.is_ghost = False

    assert manager.check_circuit_breaker() is True
    assert manager.daily_loss_breached is True


def test_broker_only_position_blocks_new_entries_without_triggering_loss_exit():
    trades = _Trades({"owned": _open_trade(lots=1, lot_size=50)})
    trades.orphan_positions = {
        "BANKNIFTY 61000 PE 29 DEC 26": {"netqty": 30, "producttype": "CARRYFORWARD"},
    }
    risk = RiskManager(trades, 1_000_005)
    risk.update_settings(capital_fut=490_000, capital_opt=500_000, risk_opt_pct=9.0)

    assert risk.check_circuit_breaker() is False
    assert not risk.daily_loss_breached and not risk.opt_loss_breached
    assert not risk.can_trade("ANY")
    assert "requires reconciliation" in risk.entry_block_reason("OPT")
    assert trades.closed == []

    trades.orphan_positions.clear()
    assert risk.can_trade("OPT")
    trades.closed_trades.append(_closed_trade(pnl=-40_000))
    assert risk.check_circuit_breaker() is True
    assert trades.closed == [("owned", 100.0, "GLOBAL_DAILY_LOSS_LIMIT")]
