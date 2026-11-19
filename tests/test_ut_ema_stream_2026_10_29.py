"""Observed ticks bypass candle publication coalescing without broker side effects."""
import asyncio
import copy
import json
import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pandas as pd
import pytest

from engine.ut_ema_stream import UTEmaTickService, install_tick_service


class Core:
    def __init__(self):
        self._timeline=None
        self.prices=[]
        self.marks=[]
    def process(self,*args,**kwargs):
        self._timeline=object()
    def observe_tick(self,price,stamp):
        self.prices.append(float(price))
        if price != 100.:
            self.marks.append({"entry":price>100,"realtime":True,"observed_at":stamp.isoformat()})
            return self.stream_result()
        return None
    def stream_result(self):
        return {"source_engine":"ut_ema","utema_events":copy.deepcopy(self.marks),"signals":[],"utema_records":{},
                "state":{"current_price":self.prices[-1] if self.prices else 100.}}
    def stream_checkpoint(self):
        return {"prices":list(self.prices),"marks":list(self.marks)}
    def restore_stream_checkpoint(self,payload):
        self._timeline=object()
        self.prices=list(payload["prices"])
        self.marks=list(payload["marks"])


class Adapter:
    signal_source="ut_ema"
    def __init__(self):
        self._native=Core()
        self._utema_lock=threading.RLock()
        self._stream_result=None
    def stream_snapshot(self):
        if getattr(self,"_utema_stream_dirty",False):
            self._stream_result=self._native.stream_result()
            self._stream_result.update(utema_entry_eligible=True,utema_stream_owned=True,
                utema_last_observed_at=self._utema_last_observed_at)
            self._utema_stream_dirty=False
        return copy.deepcopy(self._stream_result)


def host():
    adapter=Adapter()
    frame=pd.DataFrame({"open":[100.,100.],"high":[101.,101.],"low":[99.,99.],"close":[100.,100.]},
        index=pd.date_range("2026-10-29 09:15",periods=2,freq="5min"))
    scanner=NS(mode="REAL",is_running=True,mtf=NS(engines={"NIFTY_5min":adapter},_retired_utema_engines={}),
        candles=NS(get_candles=lambda *a:frame,_price_plausible=lambda *a:True),
        data=NS(on_tick=Mock()),_event_loop=None)
    return scanner,adapter


def test_every_received_observation_reaches_core_in_order(tmp_path):
    scanner,adapter=host()
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    service._schedule=Mock()
    for i,price in enumerate((100.,101.,99.,100.)):
        service.process_observation("NIFTY",price,pd.Timestamp("2026-10-29 09:25")+pd.Timedelta(milliseconds=i),i)
    assert adapter._native.prices==[100.,101.,99.,100.]
    assert len(adapter._native.marks)==2
    assert service.health["NIFTY_5min"]["status"]=="READY"
    assert service._schedule.call_count==2
    assert adapter.stream_snapshot()["state"]["current_price"]==100.


def test_checkpoint_and_recorded_tail_restore_without_duplicate_ticks(tmp_path):
    scanner,adapter=host()
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    at=pd.Timestamp("2026-10-29 09:25")
    service.process_observation("NIFTY",101.,at,1)
    # No native event at100 and no five-second checkpoint elapsed. Its journal
    # record must recover after a process restart, without replaying101 twice.
    service.process_observation("NIFTY",100.,at+pd.Timedelta(milliseconds=1),2)
    checkpoint=json.loads((tmp_path/"NIFTY_5min.json").read_text())
    assert checkpoint["core"]["prices"]==[101.]
    scanner2,adapter2=host()
    service2=UTEmaTickService(scanner2,checkpoint_dir=tmp_path)
    service2.process_observation("NIFTY",102.,at+pd.Timedelta(milliseconds=2),3)
    assert adapter2._native.prices==[101.,100.,102.]
    assert service2.health["NIFTY_5min"]["restored"]


def test_stale_timestamp_cannot_roll_native_state_back(tmp_path):
    scanner,adapter=host()
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    service.process_observation("NIFTY",101.,pd.Timestamp("2026-10-29 09:25:02"),1)
    service.process_observation("NIFTY",99.,pd.Timestamp("2026-10-29 09:25:01"),2)
    assert adapter._native.prices==[101.]
    assert service.health["NIFTY_5min"]["status"]=="STALE_TICK"


def test_waiting_for_history_keeps_all_observations(tmp_path):
    scanner,adapter=host()
    good=scanner.candles.get_candles
    scanner.candles.get_candles=lambda *a:None
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    service.process_observation("NIFTY",101.,pd.Timestamp("2026-10-29 09:25:01"),1)
    assert service.health["NIFTY_5min"]["status"]=="WAITING_HISTORY"
    scanner.candles.get_candles=good
    service.process_observation("NIFTY",99.,pd.Timestamp("2026-10-29 09:25:02"),2)
    assert adapter._native.prices==[101.,99.]


def test_historical_mode_never_advances_live_core(tmp_path):
    scanner,adapter=host()
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    scanner.mode="HISTORICAL"
    assert not service.submit("NIFTY",101.,pd.Timestamp("2026-10-29 09:25"))
    service.process_observation("NIFTY",101.,pd.Timestamp("2026-10-29 09:25"),1)
    assert adapter._native.prices==[]


def test_callback_preserves_candle_contract_and_adds_valid_ticks_only(tmp_path):
    scanner,adapter=host()
    original=scanner.data.on_tick
    service=install_tick_service(scanner,checkpoint_dir=tmp_path)
    service.submit=Mock()
    now=pd.Timestamp("2026-10-29 09:25").to_pydatetime()
    scanner.data.on_tick("NIFTY",101.,3.,now,"cash")
    original.assert_called_once_with("NIFTY",101.,3.,now,"cash")
    service.submit.assert_called_once_with("NIFTY",101.,now)
    assert install_tick_service(scanner) is service
    scanner.candles._price_plausible=lambda *a:False
    scanner.data.on_tick("NIFTY",999999.,3.,now,"cash")
    assert service.submit.call_count==1


def test_close_flushes_processed_native_state_and_stops_accepting_ticks(tmp_path):
    scanner,adapter=host()
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    at=pd.Timestamp("2026-10-29 09:25")
    service.process_observation("NIFTY",101.,at,1)
    service.process_observation("NIFTY",100.,at+pd.Timedelta(milliseconds=1),2)
    service.close()
    assert json.loads((tmp_path/"NIFTY_5min.json").read_text())["core"]["prices"]==[101.,100.]
    assert not service.submit("NIFTY",102.,at)


def test_stream_snapshot_wins_over_old_scan_result(tmp_path):
    from engine.ut_ema_lane import result_for_monitor
    scanner,adapter=host()
    scanner.mtf._result_cache={"NIFTY_5min":{"source_engine":"ut_ema","state":{"current_price":90.}}}
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    service.process_observation("NIFTY",101.,pd.Timestamp("2026-10-29 09:25"),1)
    service.process_observation("NIFTY",100.,pd.Timestamp("2026-10-29 09:25:01"),2)
    result=result_for_monitor(scanner,NS(instrument="NIFTY",timeframe="5min"))
    assert result["state"]["current_price"]==100.


def real_host(monkeypatch):
    from engine.alt_scanners import UTEmaSignalEngine
    import config.settings as config
    monkeypatch.setattr(config,"get_settings",lambda:NS(trading_mode="REAL",scanner_engine="ut_ema"))
    values=[1000.-.05*i for i in range(40)]
    frame=pd.DataFrame({"open":values,"high":[p+.1 for p in values],"low":[p-.1 for p in values],"close":values},
        index=pd.date_range("2026-10-29 09:15",periods=40,freq="5min"))
    adapter=UTEmaSignalEngine()
    scanner=NS(mode="REAL",is_running=True,mtf=NS(engines={"NIFTY_5min":adapter},_retired_utema_engines={}),
        candles=NS(get_candles=lambda *a:frame,_price_plausible=lambda *a:True),data=NS(on_tick=Mock()),_event_loop=None)
    return scanner,adapter,frame


def test_real_engine_sees_crossovers_lost_by_coalesced_snapshots(monkeypatch,tmp_path):
    scanner,adapter,frame=real_host(monkeypatch)
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    at=pd.Timestamp("2026-10-29 12:35")
    for i,price in enumerate((998.05,1000.,997.95)):
        service.process_observation("NIFTY",price,at+pd.Timedelta(milliseconds=i+1),i+1)
    full=adapter.stream_snapshot()
    assert service.health["NIFTY_5min"]["status"]=="READY"
    actual=[mark for mark in full["utema_events"] if mark["realtime"]]
    assert len(actual)>=2
    scanner2,adapter2,frame2=real_host(monkeypatch)
    other=UTEmaTickService(scanner2,checkpoint_dir=tmp_path/"coalesced")
    other.process_observation("NIFTY",997.95,at+pd.Timedelta(milliseconds=3),3)
    coalesced=[mark for mark in adapter2.stream_snapshot()["utema_events"] if mark["realtime"]]
    assert actual!=coalesced


def test_real_checkpoint_tail_matches_uninterrupted_engine(monkeypatch,tmp_path):
    scanner,adapter,frame=real_host(monkeypatch)
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    at=pd.Timestamp("2026-10-29 12:35")
    for i,price in enumerate((998.05,1000.,997.95)):
        service.process_observation("NIFTY",price,at+pd.Timedelta(milliseconds=i+1),i+1)
    # Ordinary unchanged quotes update indicators, but create no event/checkpoint.
    service.process_observation("NIFTY",997.95,at+pd.Timedelta(milliseconds=4),4)
    scanner2,adapter2,frame2=real_host(monkeypatch)
    restored=UTEmaTickService(scanner2,checkpoint_dir=tmp_path)
    next_at=at+pd.Timedelta(milliseconds=5)
    restored.process_observation("NIFTY",997.9,next_at,5)
    service.process_observation("NIFTY",997.9,next_at,5)
    expected,actual=adapter.stream_snapshot(),adapter2.stream_snapshot()
    assert expected["utema_events"]==actual["utema_events"]
    assert expected["utema_records"]==actual["utema_records"]
    assert expected["state"]==actual["state"]


def test_foreign_owner_checkpoint_cannot_seed_index(monkeypatch,tmp_path):
    scanner,adapter,frame=real_host(monkeypatch)
    from engine.ut_ema_engine import UTEmaEngine
    foreign=UTEmaEngine()
    at=pd.Timestamp("2026-10-29 12:35")
    foreign.seed_stream(frame,"SENSEX","5min",at)
    foreign.observe_tick(73000.,at+pd.Timedelta(milliseconds=1))
    tmp_path.mkdir(exist_ok=True)
    (tmp_path/"NIFTY_5min.json").write_text(json.dumps({"core":foreign.stream_checkpoint(),"last_observed_at":at.isoformat()}))
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    service.process_observation("NIFTY",998.05,at+pd.Timedelta(milliseconds=2),1)
    assert adapter._native._timeline.instrument=="NIFTY"
    assert not service.health["NIFTY_5min"].get("restored",False)


def test_cold_retired_owner_resumes_future_observations(monkeypatch,tmp_path):
    scanner,adapter,frame=real_host(monkeypatch)
    scanner.mtf.engines={}
    scanner.mtf._retired_utema_engines={"NIFTY_5min":adapter}
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    service.process_observation("NIFTY",998.05,pd.Timestamp("2026-10-29 12:35:00"),1)
    assert adapter._native._timeline is None
    assert not service.submit("NIFTY",998.05,pd.Timestamp("2026-10-29 12:35:00"))
    trade=NS(instrument="NIFTY",timeframe="5min")
    assert service.monitor_owner(trade,"held",None) is None
    service.process_observation("NIFTY",998.05,pd.Timestamp("2026-10-29 12:35:01"),1)
    service.process_observation("NIFTY",1000.,pd.Timestamp("2026-10-29 12:35:02"),2)
    assert adapter._native._timeline is not None
    assert service.health["NIFTY_5min"]["status"]=="READY"
    assert any(mark["realtime"] for mark in adapter.stream_snapshot()["utema_events"])


def test_missing_bars_pause_then_complete_history_repairs_baseline(monkeypatch,tmp_path):
    scanner,adapter,frame=real_host(monkeypatch)
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    service.process_observation("NIFTY",998.05,pd.Timestamp("2026-10-29 12:35:01"),1)
    service.process_observation("NIFTY",998.2,pd.Timestamp("2026-10-29 12:50:01"),2)
    assert service.health["NIFTY_5min"]["status"]=="ERROR"
    service.process_observation("NIFTY",998.3,pd.Timestamp("2026-10-29 12:50:02"),3)
    assert service.health["NIFTY_5min"]["status"]=="ERROR"
    missing=pd.DataFrame({"open":[998.05]*3,"high":[998.4]*3,"low":[998.]*3,"close":[998.2]*3},
        index=pd.date_range("2026-10-29 12:35",periods=3,freq="5min"))
    complete=pd.concat([frame,missing])
    scanner.candles.get_candles=lambda *a:complete
    service.process_observation("NIFTY",998.3,pd.Timestamp("2026-10-29 12:50:03"),4)
    assert service.health["NIFTY_5min"]["status"]=="READY"
    assert service.health["NIFTY_5min"]["historical_rebuild"]
    assert service.health["NIFTY_5min"]["live_tick_path_restored"] is False
    assert list(tmp_path.glob("*_before_rebuild_*.json"))


def test_upstox_rebuilds_logged_gap_only_from_complete_official_minutes(monkeypatch, tmp_path):
    scanner, adapter, _ = real_host(monkeypatch)
    scanner.data.upstox_only = True
    index = pd.date_range("2026-11-19 09:15", "2026-11-19 13:29", freq="min")
    minute = pd.DataFrame({"open": 998.05, "high": 998.4, "low": 998.0, "close": 998.2,
                           "price_source": "upstox_history"}, index=index)
    five = minute.drop(columns="price_source").resample("5min").first()
    unpublished = minute.copy()
    unpublished.loc["2026-11-19 13:04", "price_source"] = "live_tick"
    frames = {"1min": unpublished, "5min": five}
    scanner.candles.get_candles = lambda instrument, timeframe: frames[timeframe]
    service = UTEmaTickService(scanner, checkpoint_dir=tmp_path)
    before = pd.Timestamp("2026-11-19 12:59:00")
    after = pd.Timestamp("2026-11-19 13:31:17")
    service.process_observation("NIFTY", 998.05, before, 1)
    service.process_observation("NIFTY", 998.20, after, 2)
    assert "expected 2026-11-19T13:00:00, observed 2026-11-19T13:30:00" in service.health["NIFTY_5min"]["detail"]
    service.process_observation("NIFTY", 998.20, after + pd.Timedelta(seconds=1), 3)
    assert service.health["NIFTY_5min"]["status"] == "ERROR"

    frames["1min"] = minute
    service.process_observation("NIFTY", 998.20, after + pd.Timedelta(seconds=2), 4)
    assert service.health["NIFTY_5min"]["status"] == "READY"
    assert service.health["NIFTY_5min"]["historical_rebuild"]
    assert service.health["NIFTY_5min"]["live_tick_path_restored"] is False
    assert all(stamp in adapter._native._stream_times
               for stamp in pd.date_range("2026-11-19 13:00", "2026-11-19 13:25", freq="5min"))


def test_muhurat_history_uses_the_special_session_grid(monkeypatch, tmp_path):
    scanner, _, _ = real_host(monkeypatch)
    scanner.data.upstox_only = True
    index = pd.date_range("2026-11-08 14:20", "2026-11-08 15:29", freq="min")
    minute = pd.DataFrame({"open": 998.05, "high": 998.4, "low": 998.0, "close": 998.2,
                           "price_source": "upstox_history"}, index=index)
    misaligned = minute.drop(columns="price_source").resample("15min").first()
    scanner.candles.get_candles = lambda *args: minute
    service = UTEmaTickService(scanner, checkpoint_dir=tmp_path)
    close = pd.Timestamp("2026-11-08 15:30")

    rebuilt = service._bootstrap_history("NIFTY", "15min", misaligned, close)
    assert list(rebuilt.index) == list(pd.date_range("2026-11-08 14:20", "2026-11-08 15:20", freq="15min"))
    assert not service._history_gap(rebuilt, "15min", close)


def test_corrupt_journal_restore_is_transactional(tmp_path):
    scanner,adapter=host()
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    at=pd.Timestamp("2026-10-29 09:25")
    service.process_observation("NIFTY",101.,at,1)
    service.process_observation("NIFTY",100.,at+pd.Timedelta(milliseconds=1),2)
    journal=next(tmp_path.glob("*.jsonl"))
    with journal.open("ab") as stream:
        stream.write(b'{"broken":true}\n')
    scanner2,adapter2=host()
    adapter2._native.process()
    adapter2._native.observe_tick(98.,at-pd.Timedelta(seconds=1))
    before=adapter2._native.stream_checkpoint()
    restored=UTEmaTickService(scanner2,checkpoint_dir=tmp_path)
    assert not restored._load("NIFTY_5min",adapter2._native)
    assert adapter2._native.stream_checkpoint()==before
    assert "NIFTY_5min" not in restored._last_observed
    assert "NIFTY_5min" not in restored._journal_position
    assert restored.health["NIFTY_5min"]["status"]=="WAITING_HISTORY"


def test_only_current_tick_events_authorize_execution(monkeypatch,tmp_path):
    scanner,adapter,frame=real_host(monkeypatch)
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    at=pd.Timestamp("2026-10-29 12:35")
    service.process_observation("NIFTY",998.05,at+pd.Timedelta(milliseconds=1),1)
    service.process_observation("NIFTY",1000.,at+pd.Timedelta(milliseconds=2),2)
    service.process_observation("NIFTY",997.95,at+pd.Timedelta(milliseconds=3),3)
    current=adapter.stream_snapshot()
    permitted=set(current["utema_executable_event_ids"])
    assert permitted
    for signal in current["signals"]:
        if signal.utema_event_id in permitted:
            assert signal.utema_event["realtime"]
            assert not signal.utema_state.get("exit_time")
    # A restarted owner retains exact native state but no old entry authority.
    scanner2,adapter2,frame2=real_host(monkeypatch)
    restored=UTEmaTickService(scanner2,checkpoint_dir=tmp_path)
    restored.process_observation("NIFTY",997.95,at+pd.Timedelta(milliseconds=4),4)
    assert adapter2.stream_snapshot()["utema_executable_event_ids"]==[]
    # Fresh exit removes authority even on the uninterrupted owner.
    service.process_observation("NIFTY",999.,at+pd.Timedelta(milliseconds=5),5)
    result=adapter.stream_snapshot()
    for identity in result["utema_executable_event_ids"]:
        assert any(identity.startswith(f"{key}|") and not record.get("exit_time")
                   for key,record in result["utema_records"].items())


def test_bootstrap_backlog_cannot_authorize_an_older_entry(monkeypatch,tmp_path):
    scanner,adapter,frame=real_host(monkeypatch)
    scanner.candles.get_candles=lambda *a:None
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    at=pd.Timestamp("2026-10-29 12:35")
    service.process_observation("NIFTY",998.05,at+pd.Timedelta(milliseconds=1),1)
    service.process_observation("NIFTY",1000.,at+pd.Timedelta(milliseconds=2),2)
    scanner.candles.get_candles=lambda *a:frame
    service.process_observation("NIFTY",1000.,at+pd.Timedelta(milliseconds=3),3)
    result=adapter.stream_snapshot()
    assert any(signal.utema_event["realtime"] for signal in result["signals"])
    assert result["utema_executable_event_ids"]==[]


def test_shutdown_drains_accepted_observations_without_execution(tmp_path):
    scanner,adapter=host()
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    service._schedule=Mock(wraps=service._schedule)
    at=pd.Timestamp("2026-10-29 09:25")
    # Hold the worker until every accepted observation is queued.
    with adapter._utema_lock:
        for i in range(20):
            assert service.submit("NIFTY",101.+i,at+pd.Timedelta(milliseconds=i))
        service._closed=True
    service.close()
    assert adapter._native.prices==[101.+i for i in range(20)]
    saved=json.loads((tmp_path/"NIFTY_5min.json").read_text())
    assert saved["core"]["prices"]==adapter._native.prices
    assert not service.submit("NIFTY",200.,at)


def test_event_delivery_paper_base_alignment_and_first_native_exit(monkeypatch,tmp_path):
    """Exercise stream delivery, real lane admission and the real paper monitor."""
    from tests.test_ut_ema_lane_2026_10_29 import host as lane_host,settings
    from tests.test_ut_ema_execution_isolation_2026_10_29 import _monitor
    from engine.ut_ema_engine import Timeline,Params,UTEmaEngine,compute_indicators
    from scanner import Scanner
    import config.settings as config
    scanner,_,_= _monitor(monkeypatch)
    base,processor,rows=lane_host(monkeypatch)
    scanner.__dict__.update(base.__dict__)
    scanner._active_session_entry_candidates=lambda inst,timeframe=None:[row for row in rows
        if row.action=="ENTRY" and (timeframe is None or row.timeframe==timeframe)]
    processor.scanner=scanner
    scanner.signal_processor=processor
    scanner.is_running=True
    scanner.system_power="ON"
    scanner.config={"indices":{"NIFTY":{"strike_interval":50,"lot_size":75}}}
    scanner.user_lots={"NIFTY":2}
    scanner.active_indices=["NIFTY"]
    scanner._refresh_trade_candidates=Mock()
    scanner._check_session_candidates_breaches=lambda inst:Scanner._check_session_candidates_breaches_locked(scanner,inst)
    scanner._run_exit_tick=Mock()
    adapter=NS(signal_source="ut_ema",_utema_lock=threading.RLock(),_stream_result=None)
    scanner.mtf=NS(engines={"NIFTY_5min":adapter},_retired_utema_engines={})
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    monkeypatch.setattr(config,"get_settings",settings)
    timeline=Timeline(Params(),"NIFTY","5min")
    timeline.bar_time=timeline.observed_at=pd.Timestamp("2026-10-29 10:00")
    timeline.bar_index,timeline.live=0,True
    frame=pd.DataFrame([(1000.,1001.,999.,1000.)],columns=["open","high","low","close"],index=[timeline.bar_time])
    def send():
        result=UTEmaEngine().result(timeline,frame,compute_indicators(frame,Params()),1)
        result.update(utema_entry_eligible=True,utema_executable_event_ids=[s.utema_event_id for s in result["signals"] if s.utema_action!="EXIT"])
        adapter._stream_result=result
        asyncio.run(service.deliver("NIFTY","5min",result,timeline.observed_at))
    timeline.enter("UT",1,1000.)
    send()
    assert len(rows)==1 and rows[0].lots==2
    timeline.observed_at+=pd.Timedelta(seconds=1)
    timeline.enter("EMA",1,1000.)
    send()
    assert len(rows)==1 and rows[0].lots==3
    timeline.observed_at+=pd.Timedelta(seconds=1)
    timeline.exit("EMA",1001.,"RSI-50")
    send()
    assert [row.exit_reason for row in rows if row.action=="EXIT"]==["UTEMA_EMA_RSI-50"]
    assert [row.lots for row in rows if row.action=="EXIT"]==[3]
    scanner.trades.open_trade.assert_not_called()


def test_stale_tick_cannot_unlock_failed_stream_or_queued_entry(monkeypatch,tmp_path):
    from engine import ut_ema_lane as lane
    native_stamp=lane.market_stamp
    monkeypatch.setattr(lane,"market_stamp",lambda value=None:native_stamp("2026-10-29 12:36" if value is None else value))
    scanner,adapter,frame=real_host(monkeypatch)
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    scanner._utema_tick_service=service
    scanner.auto_mode=True
    at=pd.Timestamp("2026-10-29 12:35")
    for i,price in enumerate((998.05,1000.,997.95)):
        service.process_observation("NIFTY",price,at+pd.Timedelta(milliseconds=i+1),i+1)
    result=adapter.stream_snapshot()
    identity=result["utema_executable_event_ids"][0]
    queued=next(signal for signal in result["signals"] if signal.utema_event_id==identity)
    lane.state(queued)["applied_event_ids"]=[identity]
    assert lane.submission_authority(scanner,trade=queued)[0]
    service.process_observation("NIFTY",997.9,pd.Timestamp("2026-10-29 12:50:01"),4)
    assert service.health["NIFTY_5min"]["status"]=="ERROR"
    assert not lane.submission_authority(scanner,trade=queued)[0]
    service.process_observation("NIFTY",998.,pd.Timestamp("2026-10-29 12:34:59"),5)
    assert not lane.submission_authority(scanner,trade=queued)[0]
    assert service.health["NIFTY_5min"]["status"]=="ERROR"
    assert adapter.stream_snapshot()["utema_executable_event_ids"]==[]


@pytest.mark.parametrize("change",[{"system_power":"OFF"},{"is_running":False},{"mode":"HISTORICAL"}])
def test_delivery_rechecks_authority_after_waiting_for_lock(monkeypatch,tmp_path,change):
    from engine import ut_ema_lane as lane
    from unittest.mock import AsyncMock
    scanner,adapter=host()
    scanner.system_power="ON"
    scanner._get_session_trade_candidates=Mock(return_value=[])
    scanner._refresh_trade_candidates=Mock()
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    build=Mock(return_value=[])
    coordinate=AsyncMock()
    monkeypatch.setattr(lane,"build_candidates",build)
    monkeypatch.setattr(lane,"coordinate",coordinate)
    async def scenario():
        service._delivery_lock=asyncio.Lock()
        await service._delivery_lock.acquire()
        task=asyncio.create_task(service.deliver("NIFTY","5min",{},pd.Timestamp("2026-10-29 12:35")))
        await asyncio.sleep(0)
        scanner.__dict__.update(change)
        service._delivery_lock.release()
        await task
    asyncio.run(scenario())
    scanner._get_session_trade_candidates.assert_not_called()
    build.assert_not_called()
    coordinate.assert_not_awaited()


def test_bootstrap_requires_complete_closed_history_then_recovers(monkeypatch,tmp_path):
    scanner,adapter,frame=real_host(monkeypatch)
    scanner.candles.get_candles=lambda *a:frame.drop(frame.index[20])
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    at=pd.Timestamp("2026-10-29 12:35:01")
    service.process_observation("NIFTY",998.05,at,1)
    assert service.health["NIFTY_5min"]["status"]=="WAITING_HISTORY"
    assert adapter._native._timeline is None
    assert not getattr(adapter,"_utema_executable_event_ids",set())
    scanner.candles.get_candles=lambda *a:frame
    service.process_observation("NIFTY",998.05,at+pd.Timedelta(milliseconds=1),2)
    assert service.health["NIFTY_5min"]["status"]=="READY"
    assert adapter._native._timeline is not None


def test_history_grid_checks_session_tails_and_respects_host_bounds(monkeypatch,tmp_path):
    scanner,adapter,frame=real_host(monkeypatch)
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    at=pd.Timestamp("2026-10-29 12:35:01")
    assert not service._history_gap(frame,"5min",at)
    assert service._history_gap(frame.iloc[:-1],"5min",at)
    # A short leading warmup is valid; only interior and trailing gaps are holes.
    assert not service._history_gap(frame.iloc[-5:],"5min",at)
    evening=frame.iloc[:4].copy()
    evening.index=pd.date_range("2026-10-29 18:10",periods=4,freq="15min")
    scanner.data.get_index_session_bounds=lambda dt:(pd.Timestamp("2026-10-29 18:10"),pd.Timestamp("2026-10-29 19:00"))
    assert not service._history_gap(evening,"15min",pd.Timestamp("2026-10-29 19:00"))
    assert service._history_gap(evening.iloc[:-1],"15min",pd.Timestamp("2026-10-29 19:00"))


def test_history_rebuild_cannot_ignore_earlier_holes_or_extra_off_grid_bars(monkeypatch,tmp_path):
    scanner,adapter,frame=real_host(monkeypatch)
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    clock=pd.Timestamp("2026-10-29 12:35")
    assert service._history_gap(frame.drop(frame.index[5]),"5min",clock,start=frame.index[-2])
    extra=frame.iloc[[5]].copy()
    extra.index=extra.index+pd.Timedelta(minutes=1)
    assert service._history_gap(pd.concat([frame,extra]).sort_index(),"5min",clock)
    overnight=frame.iloc[[0]].copy()
    overnight.index=pd.DatetimeIndex(["2026-10-29 08:00"])
    assert service._history_gap(pd.concat([overnight,frame]),"5min",clock)


@pytest.mark.parametrize("price,when",[(None,"2026-10-29 12:35"),("bad","2026-10-29 12:35"),
    (100.,None),(100.,pd.NaT),(100.,"bad-time"),(float("nan"),"2026-10-29 12:35")])
def test_invalid_input_cannot_become_a_current_observation(tmp_path,price,when):
    scanner,adapter=host()
    service=UTEmaTickService(scanner,checkpoint_dir=tmp_path)
    assert service.submit("NIFTY",price,when) is False
    assert not service._queue
