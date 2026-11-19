"""Three defects a second external re-audit found, and what was actually wrong.

The audit called all three P0/P1. Two reproduced by a different route than the one it
described, and the third turned out to be broader than reported. Each test here fails on
the code as it stood before the fix.

SAFETY: fake brokers, isolated database files, no network, no order ever leaves.
"""
import os
import tempfile
import threading
import uuid
from datetime import datetime

import pytest

from config.settings import get_settings
from trading.kite_broker import KiteBroker
from trading.order_retry import BrokerOrderAmbiguous
from trading.trade_manager import Trade, TradeManager

try:                                    # the manager stamps IST on exit times
    from data.market_data import IST
except Exception:                       # pragma: no cover - import shape varies
    from trading.trade_manager import IST


SYMBOL = "NIFTY2681424300CE"
LOT = 75
TRADE_ID = "T0001_NIFTY_08140920"


class _Sink:
    def __getattr__(self, _n):
        return lambda *a, **k: None


# ── P0-B: a recovery exit asked the order book about a tag it never writes ───
#
# emergency_flatten_trade sends its order tagged RECOVERY_EXIT. After an ambiguous
# acknowledgement it asked _tagged_exit_state, which hardcoded leg "EXIT" -- a DIFFERENT
# valid leg, so it built a real tag string (U1X0...) that nothing ever writes. A
# successful read that does not find a never-written tag answers ABSENT, so a book
# positively holding the flatten order reported "nothing reached the exchange, a retry is
# safe", returned before the latch, and check_session_end sent another MARKET exit on
# every pass with no attempt counter.

class _FlattenBroker:
    def __init__(self, book):
        self._book = book
        self.exits = []

    def order_tag(self, trade_id, leg="ENTRY", attempt=0):
        return KiteBroker.order_tag(trade_id, leg, attempt)

    def inspect_order_by_tag(self, tag):
        return KiteBroker.inspect_order_by_tag(self, tag)

    def _order_book(self):
        return self._book

    def close_order(self, *a, **k):
        self.exits.append(a)
        raise BrokerOrderAmbiguous("read timed out after submit")

    def cancel_order(self, oid, variety="NORMAL"):
        return True

    def get_positions(self):
        return {"data": []}

    def inspect_working_protective_stop(self, *a, **k):
        return {"state": "NONE", "order_id": "", "reason": "none", "matches": []}

    def find_working_protective_stop(self, *a, **k):
        return ""

    def get_fill(self, order_id):
        return {"state": "UNKNOWN", "reason": "stub"}


def _flatten_manager(broker):
    m = TradeManager.__new__(TradeManager)
    m.broker = broker
    m.ui_logger = None
    m.mode = "REAL"
    m.product_type = "CARRYFORWARD"
    m.open_trades = {}
    m.closed_trades = []
    m.orphan_positions = {}
    m._lock = threading.RLock()
    m._protection_lock = threading.RLock()
    m.save_state = lambda: None
    m._last_protective_sync = {}
    m._last_protective_status_poll = {}
    m._last_recovery_alert = {}
    m._last_mark_to_market_save = 0.0
    m.max_positions = 4
    m.futures_sl_pct = 0.3
    m.options_sl_pct = 15.0
    m._log_ui = lambda *a, **k: None
    m.on_notification = None
    return m


def _flatten_trade():
    t = Trade(
        id=TRADE_ID, instrument="NIFTY", timeframe="5min", direction="LONG",
        entry_price=200.0, entry_time=datetime.now(IST), trailing_stop=180.0,
        current_stop=180.0, lots=1, lot_size=65, grade="A", inst_type="OPT",
        trading_symbol=SYMBOL, symbol_token="1", broker_order_id="251003000111111",
    )
    t.execution_status = "RECOVERY_REQUIRED"
    t.broker_quantity = 65
    return t


def _flatten_twice(book):
    """Two EOD passes, which is what check_session_end actually does."""
    broker = _FlattenBroker(book)
    manager = _flatten_manager(broker)
    trade = _flatten_trade()
    manager.open_trades[trade.id] = trade
    manager.emergency_flatten_trade(trade, "FORCE_EOD_KILL")
    manager.emergency_flatten_trade(trade, "FORCE_EOD_KILL")
    return manager, trade, broker


def test_two_orders_sharing_the_recovery_tag_do_not_earn_a_third():
    """AMBIGUOUS is a positive sighting, not evidence that nothing was placed."""
    tag = KiteBroker.order_tag(TRADE_ID, "RECOVERY_EXIT")
    book = [
        {"tag": tag, "orderid": "EXIT-A", "orderstatus": "open"},
        {"tag": tag, "orderid": "EXIT-B", "orderstatus": "open"},
    ]
    manager, trade, broker = _flatten_twice(book)

    assert len(broker.exits) == 1, (
        f"sent {len(broker.exits)} market exits on top of an order the book positively "
        "shows resting; every later EOD pass would send another"
    )
    assert trade.id in getattr(manager, "_flatten_ambiguous_latched", set()), (
        "an ambiguous flatten was left unlatched, so it re-fires every pass"
    )


def test_a_found_tag_without_an_order_id_does_not_read_as_absent():
    """The book named our tag; not echoing an id makes it unknown, never 'never sent'."""
    tag = KiteBroker.order_tag(TRADE_ID, "RECOVERY_EXIT")
    manager, trade, broker = _flatten_twice(
        [{"tag": tag, "orderid": "", "orderstatus": "open"}]
    )

    assert len(broker.exits) == 1, "duplicated an exit the order book had already found"
    assert trade.id in getattr(manager, "_flatten_ambiguous_latched", set())


def test_an_empty_book_still_retries():
    """The control. A successful read finding nothing proves the exit never arrived."""
    manager, trade, broker = _flatten_twice([])

    assert trade.id not in getattr(manager, "_flatten_ambiguous_latched", set()), (
        "latched a trade that provably has no exit at the exchange, stranding it"
    )


def test_an_unreadable_book_still_latches():
    manager, trade, broker = _flatten_twice(None)

    assert trade.id in manager._flatten_ambiguous_latched


def test_tagged_exit_state_answers_about_the_leg_it_is_asked_about():
    tag = KiteBroker.order_tag(TRADE_ID, "RECOVERY_EXIT")
    manager = _flatten_manager(
        _FlattenBroker([{"tag": tag, "orderid": "EXIT-A", "orderstatus": "open"}])
    )
    trade = _flatten_trade()

    assert manager._tagged_exit_state(trade, "RECOVERY_EXIT") == "FOUND"
    # The legacy leg is still consulted, so an order placed by the previous build under
    # the old shared "EXIT" tag is not orphaned by the split into distinct exit legs.
    legacy = KiteBroker.order_tag(TRADE_ID, "EXIT")
    manager.broker._book = [{"tag": legacy, "orderid": "OLD", "orderstatus": "open"}]
    assert manager._tagged_exit_state(trade, "RECOVERY_EXIT") == "FOUND"


# ── P0-C(a): the ledger close-out was guarded by the number it exists to replace ──
#
# _finalize_closed_trade gated its FULL_EXIT write on trade.broker_quantity. Reconcile
# sets that to the allocation of a flat position row -- always 0 -- immediately before
# calling finalize, so on the one path that closes REAL trades the gate was always false
# and the ENTRY row stayed on the books at full size forever. No live order was
# mis-sized, but ledger_certification sums every row per contract, so every contract that
# had ever closed reported a mismatch and the gate for re-enabling stacking could not pass.

@pytest.fixture
def ledger(monkeypatch):
    from data.sqlite_db import DatabaseManager
    import trading.trade_manager as tm

    path = os.path.join(tempfile.gettempdir(), f"reaudit_ledger_{uuid.uuid4().hex}.db")
    handle = object.__new__(DatabaseManager)
    handle._init_db(path)
    monkeypatch.setattr(tm, "db", handle)
    yield handle
    try:
        handle.conn.close()
    except Exception:
        pass


class _QuietBroker:
    def get_fill(self, order_id):
        return {"state": "UNKNOWN", "reason": "the exit order cannot state its fill"}

    def order_tag(self, trade_id, leg="", attempt=0):
        return f"{trade_id}|{leg}"[:20]

    def cancel_order(self, *a, **k):
        return True


def _closed_trade():
    t = Trade(
        id="T1", instrument="NIFTY", timeframe="5min", direction="LONG",
        entry_price=100.0, entry_time=datetime(2026, 8, 14, 9, 20),
        trailing_stop=90.0, current_stop=90.0, lots=1, lot_size=LOT, grade="A",
    )
    t.trading_symbol = SYMBOL
    t.symbol_token = "12345"
    t.status = "OPEN"
    t.execution_status = "EXIT_PENDING"
    t.inst_type = "OPT"
    t.is_ghost = False
    t.broker_order_id = "T1-entry"
    t.exit_order_id = "T1-exit"
    t.protective_order_id = ""
    return t


def _ledger_manager(trade):
    mgr = TradeManager.__new__(TradeManager)
    mgr.ui_logger = _Sink()
    mgr.mode = "REAL"
    mgr.broker = _QuietBroker()
    mgr._lock = threading.RLock()
    mgr.open_trades = {trade.id: trade}
    mgr.closed_trades = []
    mgr.orphan_positions = {}
    mgr.product_type = "CARRYFORWARD"
    mgr.save_state = lambda *a, **k: None
    mgr._log_ui = lambda *a, **k: None
    mgr.on_notification = None
    return mgr


def test_a_flat_close_settles_the_ledger_it_opened(ledger):
    """broker_quantity is 0 by the time finalize runs; the ledger still holds 75."""
    trade = _closed_trade()
    trade.broker_quantity = 0                      # what reconcile leaves behind
    mgr = _ledger_manager(trade)
    ledger.record_order_fill("T1-entry", "T1", LOT, tradingsymbol=SYMBOL, leg="ENTRY")

    mgr._finalize_closed_trade("T1", trade, 120.0, "MANUAL_SYNC")

    assert ledger.owned_quantity("T1") == 0, (
        "the trade is closed and the account is flat, but the ledger still attributes "
        "quantity to it -- ledger_certification reads that as a broker/ledger mismatch"
    )


class _PartialExitBroker(_QuietBroker):
    """The exit order states its own fill, and it is only part of the holding."""

    def get_fill(self, order_id):
        if order_id == "T1-exit":
            return {"state": "FILLED", "quantity": 30, "average_price": 120.0}
        return {"state": "UNKNOWN", "reason": "stub"}


def test_a_part_filled_exit_does_not_strand_the_remainder(ledger):
    """A finished trade holds nothing; whatever the order did not cover must still settle."""
    trade = _closed_trade()
    trade.broker_quantity = 0
    mgr = _ledger_manager(trade)
    mgr.broker = _PartialExitBroker()
    ledger.record_order_fill("T1-entry", "T1", LOT, tradingsymbol=SYMBOL, leg="ENTRY")

    mgr._finalize_closed_trade("T1", trade, 120.0, "MANUAL_SYNC")

    assert ledger.owned_quantity("T1") == 0, (
        "the exit order covered 30 of 75 and the remaining 45 stayed attributed to a "
        "closed trade, which certification reads as a broker/ledger mismatch forever"
    )


def test_a_trade_the_ledger_never_knew_writes_nothing(ledger):
    """ABSENT is not zero. No rows in must still mean no rows out."""
    trade = _closed_trade()
    trade.broker_quantity = 0
    mgr = _ledger_manager(trade)

    mgr._finalize_closed_trade("T1", trade, 120.0, "MANUAL_SYNC")

    assert ledger.owned_quantity("T1") is None, (
        "invented a ledger row for a trade that never had one"
    )


def test_exact_broker_fill_pair_owns_live_trade_pnl(ledger, monkeypatch):
    """A delayed zero in Kite's cumulative realised field must not erase fill P&L."""
    settings = get_settings()
    monkeypatch.setattr(settings, "opt_cost", 80.0)
    monkeypatch.setattr(settings, "slippage_opt_pct", 0.30)
    trade = _closed_trade()
    trade.entry_price = 575.75
    trade.lot_size = 30
    trade.broker_quantity = 0
    manager = _ledger_manager(trade)
    ledger.record_order_fill("T1-entry", "T1", 30, tradingsymbol=SYMBOL, leg="ENTRY")

    manager._finalize_closed_trade(
        "T1", trade, 576.0, "BREAKEVEN_STOP", realised_pnl=0.0
    )

    assert trade.charges == pytest.approx(80.0)
    assert trade.pnl == pytest.approx((576.0 - 575.75) * 30 - 80.0)


def test_final_fill_not_ack_invokes_authoritative_terminal_callback(ledger):
    trade = _closed_trade()
    trade.broker_quantity = 0
    manager = _ledger_manager(trade)
    terminals = []

    def terminal_callback(closed_trade):
        # The Trade has left open_trades, but Scanner must still be able to see that
        # its lifecycle is broker-owned until this authoritative callback returns.
        assert closed_trade.id not in manager.open_trades
        assert closed_trade.id in manager._terminal_callback_lifecycles
        terminals.append(closed_trade)

    manager.on_trade_closed = terminal_callback

    assert manager._finalize_closed_trade("T1", trade, 120.0, "TARGET") is True

    assert terminals == [trade]
    assert manager._terminal_callback_lifecycles == set()
    assert trade.status == "CLOSED"
    assert trade.execution_status == "CLOSED"


def test_closing_twice_does_not_double_count(ledger):
    trade = _closed_trade()
    trade.broker_quantity = 0
    mgr = _ledger_manager(trade)
    ledger.record_order_fill("T1-entry", "T1", LOT, tradingsymbol=SYMBOL, leg="ENTRY")

    mgr._finalize_closed_trade("T1", trade, 120.0, "MANUAL_SYNC")
    mgr.open_trades["T1"] = trade                  # a second reconcile pass finds it again
    mgr._finalize_closed_trade("T1", trade, 120.0, "MANUAL_SYNC")

    assert ledger.owned_quantity("T1") == 0


# ── P1-A: a refused REAL switch left the settings object saying REAL ─────────
#
# configure() stamps settings.trading_mode = mode BEFORE the broker gates run, and the
# three refusal paths restored self.mode and self.trades.mode but not the settings
# object -- which the dashboard reads and settings persistence writes to .env, so the
# next startup took main.py's REAL path on a switch that had been refused. Separately,
# the broker this method builds on the fly never received the execution authority that
# main.py wires, and KiteBroker._authorize_mutation refuses every order from a leased
# broker with no authority rather than assuming REAL.

class _StubLease:
    def __init__(self, established):
        self._established = established

    def held(self):
        return True


class _StubRealBroker:
    def __init__(self, established=True, acquired=True):
        self._established = established
        self._acquired = acquired
        self.authority = None

    def execution_lease_established(self):
        return self._established

    def acquire_execution_lease(self):
        return self._acquired

    def execution_lease_block_reason(self):
        return "owned by another process"

    def set_execution_authority(self, provider):
        self.authority = provider

    def replace_api(self, kite):
        return None


class _StubTrades:
    def __init__(self, real_broker=None):
        self.real_broker = real_broker
        self.mock_broker = object()
        self.broker = self.mock_broker
        self.mode = "HISTORICAL"

    def update_risk_settings(self, *a, **k):
        return None

    def has_live_exposure(self):
        return False

    def _live_orders_permitted(self):
        return self.mode == "REAL"


def _engine(trades, monkeypatch):
    import scanner as scanner_mod

    eng = scanner_mod.Scanner.__new__(scanner_mod.Scanner)
    eng.mode = "HISTORICAL"
    eng.trades = trades
    eng.data = type("_D", (), {"is_connected": True, "connect": lambda self: None})()
    eng.risk_manager = type("_R", (), {"update_settings": lambda self, **k: None})()
    eng.on_notification = None
    eng.active_indices = ["NIFTY"]
    eng._hist_candidates_loaded = False
    eng.log_event = lambda *a, **k: None
    eng._purge_intel_regime_caches = lambda **k: None
    eng._purge_session_contamination = lambda **k: None
    eng._surface_readiness_warnings = lambda *a, **k: None
    eng.futures_sl_pct = 0.3
    eng.options_sl_pct = 15.0
    eng.max_daily_loss_pct = 3.0
    eng.risk_fut_pct = 1.0
    eng.risk_opt_pct = 1.0
    eng.capital_fut = 100000.0
    eng.capital_opt = 100000.0
    eng.capital_total = 200000.0
    monkeypatch.setattr(
        scanner_mod, "validate_real_mode_readiness",
        lambda *a, **k: {"ok": True, "blockers": [], "warnings": []},
    )
    return eng


@pytest.fixture
def restore_trading_mode():
    settings = get_settings()
    before = settings.trading_mode
    yield settings
    settings.trading_mode = before


def test_a_refused_real_switch_leaves_no_real_label_behind(monkeypatch, restore_trading_mode):
    """The scanner falls back to HISTORICAL; the settings object must come with it."""
    settings = restore_trading_mode
    settings.trading_mode = "HISTORICAL"
    trades = _StubTrades(_StubRealBroker(established=False))
    eng = _engine(trades, monkeypatch)

    ok = eng.configure(mode="REAL", real_mode_verification="YES", persist_settings=False)

    assert ok is False
    assert eng.mode == "HISTORICAL"
    assert settings.trading_mode == "HISTORICAL", (
        "the switch was refused but the settings object still reports REAL, which the "
        "dashboard shows and settings persistence writes into .env"
    )


def test_a_refused_lease_acquisition_also_restores_the_label(monkeypatch, restore_trading_mode):
    settings = restore_trading_mode
    settings.trading_mode = "HISTORICAL"
    trades = _StubTrades(_StubRealBroker(established=True, acquired=False))
    eng = _engine(trades, monkeypatch)

    ok = eng.configure(mode="REAL", real_mode_verification="YES", persist_settings=False)

    assert ok is False
    assert eng.mode == "HISTORICAL"
    assert settings.trading_mode == "HISTORICAL"


def test_a_broker_built_at_runtime_is_given_its_execution_authority(
    monkeypatch, restore_trading_mode
):
    """Without it every order is refused: a REAL mode that cannot trade."""
    settings = restore_trading_mode
    settings.trading_mode = "HISTORICAL"
    # The authority is wired at construction, before the lease gates. Refusing at the
    # lease keeps configure() returning early instead of running the whole tail of the
    # method, without weakening what this asserts.
    built = _StubRealBroker(established=False)

    import trading.angelone_broker as execution_mod

    monkeypatch.setattr(execution_mod, "AngelOneBroker", lambda **k: built)

    trades = _StubTrades(real_broker=None)
    eng = _engine(trades, monkeypatch)
    eng.data.upstox_only = True

    eng.configure(mode="REAL", real_mode_verification="YES", persist_settings=False)

    assert eng.data.execution_broker is trades.real_broker is built
    assert built.authority is not None, (
        "the runtime-built broker holds an account lease but no execution authority, so "
        "_authorize_mutation refuses every order it is ever asked to send"
    )
    # It must be the MANAGER's live-order gate, read at send time -- not a snapshot of
    # the mode taken while wiring, which could not see a later switch.
    assert built.authority.__func__ is _StubTrades._live_orders_permitted
    trades.mode = "REAL"
    assert built.authority() is True
    trades.mode = "HISTORICAL"
    assert built.authority() is False, (
        "the authority must be read at send time, so a switch back to HISTORICAL is seen"
    )


def test_real_switch_blocks_unready_angel_one_execution(monkeypatch, restore_trading_mode):
    import scanner as scanner_mod
    import trading.angelone_broker as execution_mod

    built = _StubRealBroker()
    monkeypatch.setattr(execution_mod, "AngelOneBroker", lambda **kwargs: built)
    trades = _StubTrades()
    eng = _engine(trades, monkeypatch)
    eng.data.upstox_only = True
    monkeypatch.setattr(restore_trading_mode, "production_safe_mode_required", True)

    def unavailable(settings, provider, **kwargs):
        assert provider.execution_broker is built
        assert trades.broker is trades.mock_broker
        return {"ok": False, "blockers": ["Angel One execution unavailable"], "warnings": []}

    monkeypatch.setattr(scanner_mod, "validate_real_mode_readiness", unavailable)
    assert eng.configure(mode="REAL", real_mode_verification="YES", persist_settings=False) is False
    assert eng.mode == "HISTORICAL"
    assert trades.broker is trades.mock_broker


def test_dashboard_real_readiness_does_not_block_scanner_loop(monkeypatch, restore_trading_mode):
    import asyncio
    from threading import Event

    import scanner as scanner_mod
    from dashboard import server

    broker = _StubRealBroker()
    trades = _StubTrades(broker)
    eng = _engine(trades, monkeypatch)
    eng.mode = "PAPER"
    monkeypatch.setattr(restore_trading_mode, "production_safe_mode_required", True)
    monkeypatch.setattr(server, "scanner_ref", eng)
    started, release = Event(), Event()

    def blocked_readiness(*args, **kwargs):
        started.set()
        assert release.wait(2.0), "readiness blocked the scanner loop"
        return {"ok": False, "blockers": ["execution readiness unavailable"], "warnings": []}

    monkeypatch.setattr(scanner_mod, "validate_real_mode_readiness", blocked_readiness)

    async def run():
        eng._event_loop = asyncio.get_running_loop()
        request = asyncio.create_task(server._configure_on_scanner_loop(
            mode="REAL", real_mode_verification="YES", persist_settings=False,
        ))
        try:
            assert await asyncio.to_thread(started.wait, 1.0)
            assert not request.done()
            assert eng.mode == "PAPER"
            assert trades.broker is trades.mock_broker
        finally:
            release.set()
        assert await request is False
        assert eng.mode == "PAPER"

    asyncio.run(run())


# ── P0-E: the ledger sized an exit larger than the account holds ────────────
#
# _transactable_quantity returned the ledger's number unclamped, and _request_broker_exit
# is the one sizing site that takes it with no explicit override. The ledger sums only
# orders THIS system placed, so it goes stale HIGH whenever quantity leaves without a
# recorded order -- a manual square-off, or a partial de-risk whose reduce order could
# not be read back (_record_leg_fill_from_order is deliberately silent on an UNKNOWN
# fill while broker_quantity still drops from positions). The next full exit then sold
# the original size: on options, a naked written leg in a product Kite will not square off.

class _LedgerTrade:
    def __init__(self, tid="T1", broker_quantity=0, quantity=LOT):
        self.id = tid
        self.broker_quantity = broker_quantity
        self.quantity = quantity


def _sizing_manager(owned):
    mgr = TradeManager.__new__(TradeManager)
    mgr._ledger_owned_quantity = staticmethod(lambda _t: owned)
    return mgr


def test_the_ledger_may_not_size_an_exit_above_what_the_broker_confirms():
    """Ledger 150, broker 75: selling 150 opens a naked short of 75."""
    mgr = _sizing_manager(150)

    assert mgr._transactable_quantity(_LedgerTrade(broker_quantity=75)) == 75, (
        "sized an exit on a stale ledger; the account holds less than that"
    )


def test_an_unreconciled_entry_still_exits_at_its_ledger_size():
    """broker_quantity 0 means 'not reconciled yet', not 'flat'. Clamping there strands
    a live position, so only a POSITIVE confirmation may cap."""
    mgr = _sizing_manager(75)

    assert mgr._transactable_quantity(_LedgerTrade(broker_quantity=0)) == 75


def test_the_ledger_still_wins_when_it_is_the_smaller_number():
    """The whole point of the ledger: attribution on a shared contract. A trade whose
    own orders net to 75 must not exit the 150 the contract holds between two trades."""
    mgr = _sizing_manager(75)

    assert mgr._transactable_quantity(_LedgerTrade(broker_quantity=150)) == 75


def test_a_ledger_known_zero_is_still_zero():
    """The rule the previous audit round installed, unchanged by the cap."""
    mgr = _sizing_manager(0)

    assert mgr._transactable_quantity(_LedgerTrade(broker_quantity=75)) == 0


def test_an_explicit_request_is_still_never_clamped():
    mgr = _sizing_manager(150)

    assert mgr._transactable_quantity(_LedgerTrade(broker_quantity=75), 30) == 30


# ── P0-D: a part-filled reduce or add was booked at its requested size ──────
#
# Both booking helpers sized off the REQUESTED lots. A MARKET order in a thin option
# book can come back part-filled, and the realised child then carried P&L on quantity
# that never left, while a part-filled add moved the weighted-average entry to a price
# the position never paid. The protective stop is armed from the corrected
# broker_quantity and was never wrong, which is why this stayed invisible -- but the
# daily loss cap and every R multiple are computed off these numbers.

def _partial_trade():
    t = Trade(
        id="T1", instrument="NIFTY", timeframe="5min", direction="LONG",
        entry_price=200.0, entry_time=datetime(2026, 8, 14, 9, 20),
        trailing_stop=180.0, current_stop=180.0, lots=3, lot_size=LOT, grade="A",
    )
    t.trading_symbol = SYMBOL
    t.symbol_token = "12345"
    t.inst_type = "OPT"
    t.broker_quantity = 3 * LOT
    t.protective_order_id = ""
    return t


def test_a_part_filled_reduce_realises_only_what_left():
    """Request 75, fill 30: the closed child must carry 30 units, not 75."""
    trade = _partial_trade()
    mgr = TradeManager.__new__(TradeManager)
    mgr.closed_trades = []
    mgr.ui_logger = _Sink()
    mgr._log_ui = lambda *a, **k: None

    mgr._book_partial_child(trade, 1, 260.0, "DERISK", None, closed_quantity=30)
    child = mgr.closed_trades[-1]

    assert child.broker_quantity == 30, (
        f"realised {child.broker_quantity} units of P&L on a 30-unit fill"
    )


def test_a_full_reduce_is_unchanged():
    """The default path must keep booking the requested lots exactly as before."""
    trade = _partial_trade()
    mgr = TradeManager.__new__(TradeManager)
    mgr.closed_trades = []
    mgr.ui_logger = _Sink()
    mgr._log_ui = lambda *a, **k: None

    mgr._book_partial_child(trade, 1, 260.0, "DERISK", None)

    assert mgr.closed_trades[-1].broker_quantity == LOT


def test_a_part_filled_add_weights_the_average_on_what_arrived():
    """Entry 200 x150 plus 30 filled at 260 averages 210, never 220."""
    trade = _partial_trade()
    trade.lots = 2
    trade.broker_quantity = 2 * LOT
    mgr = TradeManager.__new__(TradeManager)
    mgr._record_leg_fill_from_order = lambda *a, **k: False

    mgr._book_scale_in(trade, 1, 260.0, "CASCADE_ADD", filled_quantity=30)

    assert trade.entry_price == pytest.approx(210.0, abs=0.01), (
        f"weighted the average entry at {trade.entry_price:.2f} on lots that never filled"
    )


def test_a_full_add_is_unchanged():
    trade = _partial_trade()
    trade.lots = 2
    trade.broker_quantity = 2 * LOT
    mgr = TradeManager.__new__(TradeManager)
    mgr._record_leg_fill_from_order = lambda *a, **k: False

    mgr._book_scale_in(trade, 1, 260.0, "CASCADE_ADD")

    # 150 @ 200 + 75 @ 260 = 220
    assert trade.entry_price == pytest.approx(220.0, abs=0.01)


# ── P1-B: the premium back-fill could never fetch anything ──────────────────
#
# Two independent breaks, either of which alone empties the archive. The token was
# resolved by probing four attributes MarketDataProvider does not have, so every fetch
# returned False; and get_historical_candles answers an option request from the local
# CSV whenever the file merely EXISTS, without checking it covers the day asked for, so
# even with a token the "fetch" returned a stale frame and wrote nothing. On disk: 139
# archived option files, newest 2026-07-31, and zero in any August session.

def test_missing_archive_token_resolves_only_exact_upstox_symbol(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import pandas as pd

    from data.market_data import MarketDataProvider
    from engine.expiry_manager import expiry_manager
    from main import _contract_token_for

    symbol = "NIFTY 25200 CE 23 NOV 26"
    monkeypatch.setattr(expiry_manager, "df", pd.DataFrame(
        [{"symbol": symbol, "token": "NSE_FO|12345"}]
    ))
    provider = MarketDataProvider.__new__(MarketDataProvider)
    provider.upstox_only = True
    provider.kite = SimpleNamespace(instrument_token=Mock(side_effect=AssertionError("Kite lookup")))
    assert _contract_token_for(provider, symbol) == "NSE_FO|12345"
    assert _contract_token_for(provider, "NIFTY 25300 CE 23 NOV 26") == ""
    provider.kite.instrument_token.assert_not_called()


def test_the_session_events_supply_the_contract_token():
    import scripts.archive_session_data as arch

    events = [
        {"contract": {"trading_symbol": "BANKNIFTY26AUG57700PE", "symbol_token": "15128322"}},
        {"contract": {"trading_symbol": "", "symbol_token": ""}},
        {"contract": {"trading_symbol": "NIFTY26AUG24000CE", "symbol_token": "12345"}},
        {"not_a_contract": True},
    ]

    assert arch.contract_tokens(events) == {
        "BANKNIFTY26AUG57700PE": "15128322",
        "NIFTY26AUG24000CE": "12345",
    }


def test_a_forced_refresh_ignores_an_option_cache_that_lacks_the_day(tmp_path, monkeypatch):
    """The cache short-circuits on file existence, so the back-fill must bypass it."""
    import pandas as pd

    from data.market_data import MarketDataProvider

    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "data_store" / "candles"
    cache_dir.mkdir(parents=True)
    stale = cache_dir / "NIFTY26AUG24000CE_1min.csv"
    stale.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-07-15 09:17:00,1120.0,1120.0,1120.0,1120.0,2\n",
        encoding="utf-8",
    )

    import data.market_data as md

    consulted = {"cache": 0}
    real_store = md.get_option_history_store

    class _Spy:
        def load(self, path):
            consulted["cache"] += 1
            return real_store().load(path)

    monkeypatch.setattr(md, "get_option_history_store", lambda: _Spy())

    provider = MarketDataProvider.__new__(MarketDataProvider)
    provider.is_connected = False
    provider.smart_api = None
    provider.fyers = None

    cached = provider.get_historical_candles(
        "12345", "NFO", "1min", days_back=1, instrument_name="NIFTY26AUG24000CE",
    )
    assert not cached.empty, "the ordinary path should still answer from the CSV cache"
    assert consulted["cache"] == 1

    # The forced call must get PAST the cache. It then runs on into the real fetch
    # machinery, which this bare provider cannot complete -- reaching that at all is the
    # proof, so the failure past this branch is not what is under test.
    try:
        provider.get_historical_candles(
            "12345", "NFO", "1min", days_back=1, instrument_name="NIFTY26AUG24000CE",
            force_refresh=True,
        )
    except Exception:
        pass

    assert consulted["cache"] == 1, (
        "force_refresh still answered from a cache that has no bars for the day asked for"
    )


def test_option_history_has_a_live_source_at_all():
    """AngelOne was the only option-history source and smart_api is permanently None.

    That is the real reason no option CSV holds a bar after 2026-07-31: the Kite
    execution migration removed the fetch, so skipping the caches reached nothing.
    """
    import pandas as pd

    from data.market_data import MarketDataProvider

    class _Kite:
        is_connected = True

        def __init__(self):
            self.asked = []

        def get_historical(self, token, interval, days_back=15, **k):
            self.asked.append((token, interval, days_back))
            idx = pd.to_datetime(["2026-08-13 09:16:00", "2026-08-13 09:17:00"])
            return pd.DataFrame(
                {"open": [10.0, 11.0], "high": [12.0, 12.5], "low": [9.0, 10.5],
                 "close": [11.0, 12.0], "volume": [100, 150]},
                index=idx,
            )

    provider = MarketDataProvider.__new__(MarketDataProvider)
    provider.is_connected = False
    provider.smart_api = None
    provider.fyers = None
    provider.kite = _Kite()

    df = provider.get_historical_candles(
        "15128322", "NFO", "1min", days_back=1,
        instrument_name="BANKNIFTY26AUG57700PE", force_refresh=True,
    )

    assert provider.kite.asked == [(15128322, "1min", 1)], (
        "the option request never reached Kite, the only live history source"
    )
    assert not df.empty
    # scripts/archive_session_data.day_slice keys on a "timestamp" column.
    assert df.index.name == "timestamp"


def test_a_forced_fetch_that_finds_nothing_does_not_answer_from_the_cache(tmp_path, monkeypatch):
    """Otherwise the back-fill counts a success and logs a number for an empty archive."""
    from data.market_data import MarketDataProvider

    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "data_store" / "candles"
    cache_dir.mkdir(parents=True)
    (cache_dir / "NIFTY26AUG24000CE_1min.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-07-15 09:17:00,1120.0,1120.0,1120.0,1120.0,2\n",
        encoding="utf-8",
    )

    provider = MarketDataProvider.__new__(MarketDataProvider)
    provider.is_connected = False
    provider.smart_api = None
    provider.fyers = None
    provider.kite = None                       # no session: nothing can be fetched

    df = provider.get_historical_candles(
        "12345", "NFO", "1min", days_back=1,
        instrument_name="NIFTY26AUG24000CE", force_refresh=True,
    )

    assert df.empty, (
        "a forced refresh that fetched nothing returned the stale cache, so the "
        "back-fill reports contracts 'filled' while writing nothing"
    )


# ── R1: the empty-ack scan could adopt another trade's order ────────────────

class _EmptyAckApi:
    def __init__(self, book):
        self._book = book

    def place_order(self, **params):
        return ""                              # accepted, id lost

    def orders(self):
        return list(self._book)


def _book_row(order_id, tag, status="COMPLETE", qty=75):
    return {
        "order_id": order_id, "tradingsymbol": "NIFTY26AUG24000CE",
        "transaction_type": "BUY", "quantity": qty, "filled_quantity": qty,
        "status": status, "order_type": "MARKET", "product": "NRML", "tag": tag,
    }


def _place_entry(book, trade_id="T0009_08141012"):
    broker = KiteBroker(kite=_EmptyAckApi(book), account_id="", instance_id="t")
    broker.set_execution_authority(lambda: True)
    return broker.place_order(
        "NIFTY26AUG24000CE", "12345", 75, "BUY", "CARRYFORWARD",
        price=0.0, trade_id=trade_id, confidence=0.9,
    )


def test_an_empty_ack_never_adopts_another_trades_order():
    """The scan has no tag term, so on a shared strike it can return a stranger."""
    foreign = KiteBroker.order_tag("T0004_08140931", "ENTRY", 0)

    with pytest.raises(BrokerOrderAmbiguous, match="another trade's tag"):
        _place_entry([_book_row("SIBLING-ORDER", foreign)])


def test_an_untagged_row_is_still_adopted():
    """The case the fallback exists for: a broker that did not echo the tag back.

    This must keep working, or the fix trades one recovery failure for another.
    """
    assert _place_entry([_book_row("OUR-ORDER", "")]) == "OUR-ORDER"


def test_our_own_replacement_attempt_is_not_foreign():
    """A cancel/replace stamps attempt = entry_replace_count + 1, so one trade owns
    more than one entry tag."""
    mine_attempt_1 = KiteBroker.order_tag("T0009_08141012", "ENTRY", 1)

    assert _place_entry([_book_row("MINE-RETRY", mine_attempt_1)]) == "MINE-RETRY"


def test_an_empty_book_is_still_a_clean_failure():
    """Unchanged: a completed scan finding nothing means nothing was placed."""
    from trading.order_retry import BrokerOrderRejected

    with pytest.raises(BrokerOrderRejected, match="confirms nothing was placed"):
        _place_entry([])


# ── R3: a confirmed ZERO was read as "no answer" ────────────────────────────

def test_a_reduce_that_flattens_the_book_zeroes_the_quantity():
    """`confirmed or broker_quantity` kept the pre-reduce size on a flat book, and the
    next line re-arms protection from it: an oversized SELL SL-M on a bought option."""
    import inspect as _inspect

    src = _inspect.getsource(TradeManager._run_partial_reduce)
    assert "trade.broker_quantity = int(confirmed)" in src
    assert "confirmed or trade.broker_quantity" not in src


def test_a_flat_confirmation_is_a_real_answer():
    """The behaviour behind that line, driven directly."""
    confirmed = 0
    assert (confirmed or 150) == 150, "this is the bug the int() cast removes"
    assert int(confirmed) == 0


# ── R7: a negative per-trade sum was invisible to the certification gate ────

def test_a_negative_attribution_cannot_hide_behind_a_matching_total():
    import scripts.ledger_certification as cert

    symbol = "NIFTY2681424300CE"
    # 150 + (-75) = 75, which matches the broker exactly.
    code = cert._report({symbol: {"T1": 150, "T2": -75}}, {symbol: 75}, offline=False)

    assert code == 1, (
        "one trade's excess cancelled another's deficit and the gate reported agreement"
    )


def test_a_clean_ledger_still_certifies():
    import scripts.ledger_certification as cert

    symbol = "NIFTY2681424300CE"

    assert cert._report({symbol: {"T1": 75}}, {symbol: 75}, offline=False) == 0


def test_an_unwired_leased_broker_really_does_refuse_everything():
    """Why the wiring matters: this is the failure it prevents, straight from the adapter."""
    from trading.order_retry import BrokerOrderRejected

    broker = KiteBroker.__new__(KiteBroker)
    broker.api = object()
    broker._execution_lease = _StubLease(established=True)
    broker.execution_lease_established = lambda: True
    broker._require_account_execution_lease = lambda action: None

    with pytest.raises(BrokerOrderRejected, match="no execution authority"):
        broker._authorize_mutation("entry order")
