from datetime import datetime
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from data import market_data
from engine.expiry_manager import expiry_manager
from trading.angelone_broker import AngelOneBroker
from trading.order_retry import BrokerOrderAmbiguous, BrokerOrderRejected
from trading.trade_manager import TradeManager


def test_dashboard_angel_one_badge_distinguishes_standby_from_real_failures():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the dashboard regression")

    app = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "js" / "app.js"
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function badge() {
    const classes = new Set();
    return {
        classes,
        style: {},
        classList: {
            add: value => classes.add(value),
            remove: (...values) => values.forEach(value => classes.delete(value)),
        },
    };
}

const angel = badge();
const upstox = badge();
const label = {textContent: 'CHECKING'};
const elements = {gw_angelone: angel, gw_angelone_state: label, gw_upstox: upstox};
const context = {
    window: {},
    document: {addEventListener() {}, getElementById: id => elements[id]},
    localStorage: {getItem: () => null},
};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), context);

const idle = {provider: 'angelone', configured: true, connected: false, lease_held: false};
const ready = {...idle, connected: true, lease_held: true};
const cases = [
    ['HISTORICAL', idle, 'standby', 'STANDBY'],
    ['LIVE_PAPER', idle, 'standby', 'STANDBY'],
    ['REAL', idle, 'offline', 'DISCONNECTED'],
    ['LIVE_REAL', ready, 'online', 'CONNECTED'],
    ['REAL', {...ready, lease_held: false}, 'offline', 'BLOCKED'],
    ['REAL', {...idle, configured: false}, 'offline', 'NOT CONFIGURED'],
    ['REAL', {}, 'offline', 'UNAVAILABLE'],
    ['REPLAY', ready, 'standby', 'STANDBY'],
];
for (const [mode, broker_status, appearance, text] of cases) {
    context.updateGatewayStatus({upstox_healthy: true, broker_status}, mode);
    assert.deepEqual([...angel.classes], [appearance], mode);
    assert.equal(label.textContent, text, mode);
    assert.deepEqual([...upstox.classes], ['online'], mode);
}
assert.equal(context.isLightweightDashboardDelta({mode: 'REAL'}), false);
"""
    subprocess.run([node, "-e", script, str(app)], check=True, text=True, capture_output=True, timeout=10)


@pytest.fixture
def contracts(monkeypatch):
    catalog = [
        {"name": "SENSEX", "exch_seg": "BFO", "expiry": "26NOV2026", "strike": 8250000,
         "lotsize": 20, "tick_size": 0.05, "instrumenttype": "OPTIDX", "option_type": side,
         "symbol": f"SENSEX 82500 {side} 26 NOV 26", "token": f"BSE_FO|{88 + i}"}
        for i, side in enumerate(("CE", "PE"))
    ]
    native = [
        dict(row, symbol=f"SENSEX26NOV82500{row['option_type']}", token=str(900 + i), tick_size=5)
        for i, row in enumerate(catalog)
    ]
    monkeypatch.setattr(expiry_manager, "df", pd.DataFrame(catalog))
    monkeypatch.setattr(expiry_manager, "_loaded_date", datetime.now(ZoneInfo("Asia/Kolkata")).date())
    monkeypatch.setattr(expiry_manager, "effective_expiry", lambda *_: "26NOV2026")
    return catalog, native


def test_upstox_mode_never_constructs_legacy_data_clients(monkeypatch, contracts):
    def forbidden(*args, **kwargs):
        pytest.fail("Legacy market-data adapter called")

    monkeypatch.setattr(market_data, "KiteDataSource", forbidden)
    monkeypatch.setattr(market_data, "AngelOneIndexQuoteSource", forbidden)
    provider = market_data.MarketDataProvider(start_streams=True, upstox_only=True)
    assert provider.kite is provider.angel_index is None
    assert not provider._prewarm_angelone_fallback()
    provider._upstox_stream = Mock()
    provider._participation_store = Mock()
    provider._upstox_quote = Mock(return_value={"previous_close": 82000})
    provider.upstox.get_option_chain_df = Mock(return_value=pd.DataFrame(
        [{"strike": 82500, "underlying_spot": 82490, "call_token": "BSE_FO|88",
          "put_token": "BSE_FO|89", "call_volume_available": True, "put_volume_available": True}]
    ))
    provider.upstox.get_option_chain_df.return_value.attrs["expiry"] = "2026-11-26"
    assert provider.get_previous_close("SENSEX") == 82000
    chain = provider.get_option_chain("SENSEX")
    assert chain.attrs["source"] == "upstox"
    assert {call.args[0] for call in provider._upstox_stream.watch.call_args_list} == {"BSE_FO|88", "BSE_FO|89"}
    assert provider._participation_store.return_value.record_option_chain.call_count == 1


def test_empty_candle_cache_is_rebuilt_and_merged_atomically(tmp_path):
    provider = market_data.MarketDataProvider.__new__(market_data.MarketDataProvider)
    provider.upstox_only = True
    path = tmp_path / "BANKNIFTY_1min.csv"
    path.touch()
    first = pd.DataFrame({"close": [100.0], "volume": [0.]},
                         index=pd.DatetimeIndex(["2026-11-19 09:15"], name="timestamp"))
    later = first.copy()
    later.index = pd.DatetimeIndex(["2026-11-19 09:16"], name="timestamp")
    provider._save_and_merge_candles(first, path)
    provider._save_and_merge_candles(later, path)
    assert pd.read_csv(path)["close"].tolist() == [100., 100.]
    assert list(tmp_path.iterdir()) == [path]


def test_upstox_index_rest_uses_net_change_when_previous_close_field_missing(monkeypatch):
    from data.upstox_market_data import UpstoxMarketDataClient, UPSTOX_INDEX_KEYS

    client = UpstoxMarketDataClient()
    key = UPSTOX_INDEX_KEYS["NIFTY"]
    reply = Mock()
    reply.json.return_value = {"data": {
        "NSE_INDEX:Nifty 50": {"instrument_token": key, "last_price": 25357.15, "net_change": 129.2,
                                "timestamp": "2026-11-19T13:10:20.823+05:30"}
    }}
    monkeypatch.setattr(client, "_headers", lambda: {})
    monkeypatch.setattr("data.upstox_market_data.requests.get", Mock(return_value=reply))
    assert client.get_quote(key)["previous_close"] == pytest.approx(25227.95)


@pytest.fixture
def broker(monkeypatch, contracts):
    _, native = contracts
    broker = AngelOneBroker(api_key="fixture", client_id="ACCOUNT", password="PIN", totp_secret="fixture")
    broker._contracts_by_identity, broker._native_contracts = broker._normalize_execution_master(native)
    monkeypatch.setattr(broker, "prepare_contracts", lambda: True)
    monkeypatch.setattr(broker, "has_execution_lease", lambda: True)
    monkeypatch.setattr(broker, "_ensure_connected", lambda: True)
    broker.client = Mock()
    broker.client.placeOrderFullResponse.return_value = {"status": True, "data": {"orderid": "1"}}
    broker.set_execution_authority(lambda: True)
    return broker


def test_angel_one_translates_bfo_keys_and_preserves_order_side(broker, contracts):
    row = contracts[0][0]
    symbol, key = row["symbol"], row["token"]
    assert broker.place_order(symbol, key, 20, "BUY", price=180.04) == "1"
    request = broker.client.placeOrderFullResponse.call_args.args[0]
    assert request["exchange"] == "BFO"
    assert (request["tradingsymbol"], request["symboltoken"]) == ("SENSEX26NOV82500CE", "900")
    assert (request["transactiontype"], request["price"]) == ("BUY", "180.0")
    assert broker.close_order(symbol, key, 20, "SELL") == "1"
    assert broker.client.placeOrderFullResponse.call_args.args[0]["transactiontype"] == "SELL"
    assert broker.place_protective_stop(symbol, key, 20, "SELL", 150) == "1"
    stop = broker.client.placeOrderFullResponse.call_args.args[0]
    assert (stop["transactiontype"], stop["ordertype"]) == ("SELL", "STOPLOSS_LIMIT")
    assert 0 < float(stop["price"]) < float(stop["triggerprice"])
    broker.set_execution_authority(lambda: False)
    with pytest.raises(BrokerOrderRejected, match="requires REAL"):
        broker.close_order(symbol, key, 20, "SELL")
    assert broker.client.placeOrderFullResponse.call_count == 3


def test_unknown_acknowledgement_is_ambiguous_and_tag_can_recover(broker, contracts):
    row = contracts[0][0]
    tag = broker.order_tag("trade", "ENTRY")
    broker.client.placeOrderFullResponse.return_value = {"status": False, "errorcode": "AB1004"}
    with pytest.raises(BrokerOrderAmbiguous):
        broker.place_order(row["symbol"], row["token"], 20, "BUY", tag=tag)
    broker.client.orderBook.return_value = {"status": True, "data": [{"orderid": "1", "ordertag": tag}]}
    assert broker.inspect_order_by_tag(tag)["order_id"] == "1"


def test_broker_position_round_trip_keeps_products_separate(broker, contracts):
    upstox, _ = contracts
    position = {"tradingsymbol": "SENSEX26NOV82500CE", "symboltoken": "900", "exchange": "BFO"}
    broker.client.position.return_value = {"status": True, "data": [
        dict(position, netqty="20", producttype="CARRYFORWARD"),
        dict(position, netqty="0", producttype="INTRADAY"),
    ]}
    manager = TradeManager.__new__(TradeManager)
    manager.broker = broker
    manager.product_type = "INTRADAY"
    trade = SimpleNamespace(trading_symbol=upstox[0]["symbol"], symbol_token=upstox[0]["token"])
    row, readable = manager._live_position_for_trade(trade)
    assert readable and row["netqty"] == 0
    broker.client.position.return_value["data"][1]["symboltoken"] = "501"
    broker.client.position.return_value["data"][1]["netqty"] = "20"
    assert broker.get_positions() is None


def test_execution_readiness_waits_for_angel_one_lease_and_positions(broker, monkeypatch):
    broker.client.position.return_value = {"status": True, "data": []}
    assert broker.ready_for_execution()
    broker.client.position.return_value = {"status": False}
    assert not broker.ready_for_execution()
    monkeypatch.setattr(broker, "has_execution_lease", lambda: False)
    monkeypatch.setattr(broker, "acquire_execution_lease", lambda: False)
    reads = broker.client.position.call_count
    assert not broker.ready_for_execution()
    assert broker.client.position.call_count == reads


def test_angel_one_reuses_only_the_current_execution_directory(tmp_path, monkeypatch, contracts):
    import json

    path = tmp_path / "execution.json"
    monkeypatch.setattr(AngelOneBroker, "_MASTER_CACHE", path)
    reply = Mock()
    reply.json.return_value = contracts[1]
    request = Mock(return_value=reply)
    monkeypatch.setattr("trading.angelone_broker.requests.get", request)
    first = AngelOneBroker()
    assert first.prepare_contracts()
    assert request.call_count == 1

    restarted = AngelOneBroker()
    assert restarted.prepare_contracts()
    assert restarted._native_contracts == first._native_contracts
    assert request.call_count == 1

    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["date"] = "2020-01-01"
    path.write_text(json.dumps(saved), encoding="utf-8")
    assert AngelOneBroker().prepare_contracts()
    assert request.call_count == 2


def test_watchdog_checks_angel_one_only_in_live_real_and_rearms(monkeypatch):
    import main
    from data import upstox_auth

    mode_and_health = [
        ("HISTORICAL", False), ("PAPER", False), ("LIVE_REAL", False),
        ("REAL", False), ("REAL", True), ("REAL", False),
        ("REPLAY", False), ("REAL", False),
    ]
    calls, alerts, watchdog = [], [], []
    clock = SimpleNamespace(monotonic=lambda: 100.0)
    index = 0

    def execution_status():
        calls.append(index)
        return {"connected": mode_and_health[index][1]}

    scanner = SimpleNamespace(
        mode=mode_and_health[0][0], is_running=True, auto_mode=False,
        trades=SimpleNamespace(real_broker=SimpleNamespace(execution_status=execution_status)),
        log_event=lambda *args: None,
    )

    def next_check(seconds):
        nonlocal index
        index += 1
        if index == len(mode_and_health):
            raise StopIteration
        scanner.mode = mode_and_health[index][0]

    clock.sleep = next_check
    monkeypatch.setenv("RUNTIME_WATCHDOG_ENABLED", "1")
    monkeypatch.setenv("RUNTIME_WATCHDOG_UNTIL", "15:30")
    monkeypatch.setattr(main, "datetime", SimpleNamespace(
        now=lambda tz: datetime(2026, 11, 19, 18, 27, 29, tzinfo=ZoneInfo("Asia/Kolkata"))))
    monkeypatch.setattr(main, "time", clock)
    monkeypatch.setattr(main, "threading", SimpleNamespace(Thread=lambda **kwargs:
        SimpleNamespace(start=lambda: watchdog.append(kwargs["target"]))))
    monkeypatch.setattr(main, "_apply_runtime_resource_guard", lambda scanner: (False, ""))
    monkeypatch.setattr(upstox_auth, "get_upstox_auth_status", lambda: {"authenticated": True})
    monkeypatch.setattr(main, "send_desktop_notification", lambda message, level: alerts.append((index, message)))

    main.start_runtime_watchdog(scanner, SimpleNamespace(dashboard_host="127.0.0.1", dashboard_port=8500))
    with pytest.raises(StopIteration):
        watchdog[0]()
    assert calls == [2, 3, 4, 5, 7]
    assert alerts == [(i, "UT1 WATCHDOG: Angel One execution session unavailable") for i in (2, 5, 7)]
