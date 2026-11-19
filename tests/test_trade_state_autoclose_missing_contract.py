from datetime import datetime

from data.sqlite_db import DatabaseManager, db
from trading.trade_manager import IST, TradeManager

# The auto-close applies to SAME-SESSION rows only: a trade that survived a day
# boundary keeps the reconciliation path, because discarding a position that may be
# live at the broker is worse than one noisy alert. A hardcoded date therefore stops
# testing the intended branch the moment the clock rolls past it.
_TODAY = datetime.now(IST)


def test_load_state_closes_open_trade_missing_contract_metadata(monkeypatch):
    with db.write_lock:
        db.conn.execute("DELETE FROM trades")
        db.conn.commit()

    open_trade = {
        "id": "T9999_MISSING_07311234",
        "instrument": "NIFTY 25000 CE",
        "timeframe": "5min",
        "direction": "LONG",
        "entry_price": 100.0,
        "entry_time": _TODAY.strftime("%d %b %H:%M:%S"),
        "trailing_stop": 90.0,
        "current_stop": 90.0,
        "lots": 1,
        "lot_size": 50,
        "grade": "A",
        "entry_timestamp": _TODAY.timestamp(),
        "broker_order_id": "",
        "exit_order_id": "",
        "execution_status": "RECOVERY_REQUIRED",
        "execution_error": "bad contract",
        "pending_execution": 1,
        "pending_exit": 0,
        "broker_quantity": 0,
        "trading_symbol": None,
        "symbol_token": None,
        "inst_type": "OPT",
        "status": "OPEN",
        "current_price": 100.0,
        "entry_spot": 25000.0,
        "exit_price": 0.0,
        "exit_time": None,
        "exit_reason": "",
        "pnl": 0.0,
        "is_ghost": 0,
    }

    db.save_trades([open_trade])

    manager = TradeManager.__new__(TradeManager)
    manager.broker = None
    manager.ui_logger = None
    manager.on_notification = None
    manager.max_positions = 3
    manager.session_end = manager._parse_time("15:25")
    manager.product_type = "CARRYFORWARD"
    manager.open_trades = {}
    manager.closed_trades = []
    manager._trade_counter = 0
    manager._lock = __import__("threading").RLock()
    manager._protection_lock = __import__("threading").RLock()
    manager._state_file = "data_store/trade_state.json"
    manager._last_mark_to_market_save = 0.0
    manager.orphan_positions = {}
    manager.futures_sl_pct = 0.3
    manager.options_sl_pct = 15.0

    manager.load_state()

    assert manager.open_trades == {}
    assert any(t.id == "T9999_MISSING_07311234" and t.status == "CLOSED" for t in manager.closed_trades)


def test_load_state_restores_live_row_and_preserves_unlisted_replay_for_audit(monkeypatch):
    import threading

    broker_symbol = "BANKNIFTY 61000 PE 29 DEC 26"
    active = {
        "id": "T9876_TEST", "instrument": "BANKNIFTY", "timeframe": "5min",
        "direction": "SHORT", "entry_price": 100.0, "entry_timestamp": _TODAY.timestamp(),
        "entry_time": _TODAY.strftime("%d %b %H:%M:%S"),
        "trading_symbol": broker_symbol, "symbol_token": "NSE_FO|9876",
        "data_provenance": "LIVE", "broker_order_id": "owned-order",
        "broker_quantity": 30, "lot_size": 30, "lots": 1, "inst_type": "OPT",
        "status": "OPEN",
    }
    replay = dict(active, id="H_OLD_REPLAY", trading_symbol="BANKNIFTY OLD PE",
                  data_provenance="ENGINE_REPLAY", broker_order_id="", broker_quantity=0)
    monkeypatch.setattr(DatabaseManager, "_tradable_symbols", classmethod(lambda cls: {"OTHER"}))
    monkeypatch.setattr(db, "load_open_trades", lambda: [replay, active])
    monkeypatch.setattr(db, "load_closed_trades", lambda: [])
    manager = TradeManager.__new__(TradeManager)
    manager._lock = threading.RLock()
    manager.open_trades = {}
    manager.closed_trades = []
    manager._trade_counter = 0

    manager.load_state()
    assert manager._trade_state_loaded
    assert list(manager.open_trades) == [active["id"]]
    assert list(manager._restored_replay_open_rows) == [replay["id"]]
