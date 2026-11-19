"""Adversarial audit of combined scanner failure and lifecycle boundaries."""
from types import SimpleNamespace as NS
from datetime import datetime, timedelta

import pandas as pd
import pytest

from engine.scanner_modes import COMBINED_MODES
from engine.multi_timeframe import MultiTimeframeEngine
from engine import engine_lab
from tests.test_combined_scanner_modes_2026_10_05 import combined
from tests.test_uts_chart_cache_parity_2026_09_30 import setup, Host, snapshot, entries
from tests.test_ut_smart_live_path_2026_09_26 import live


def frame():
    return pd.DataFrame({'open': 100., 'high': 102., 'low': 99., 'close': 101., 'volume': 1000.},
                        index=pd.date_range('2026-10-02 09:15', periods=40, freq='5min'))


@pytest.fixture(params=['injected native lane outage', ''], ids=['message', 'empty_message'])
def failed_lane(request):
    def fail(**kwargs):
        raise RuntimeError(request.param)
    return fail


@pytest.mark.parametrize('mode', COMBINED_MODES)
def test_total_native_failure_invalidates_pipeline_readiness(combined, monkeypatch, mode, failed_lane):
    combined.scanner_engine = mode
    pipeline = MultiTimeframeEngine()
    pipeline.update_data('NIFTY', '5min', frame())
    assert pipeline.process_instrument('NIFTY').results_5min
    for engine in pipeline.engines['NIFTY_5min'].native_engines.values():
        monkeypatch.setattr(engine, 'process', failed_lane)
    pipeline.require_fresh_result('NIFTY', '5min')
    result = pipeline.process_instrument('NIFTY')
    assert result.timeframe_metadata['5min']['result']['status'] == 'processing_error'
    assert result.results_5min['combined_health'] == 'HALTED'
    assert result.results_5min['signals'] == []
    assert 'NIFTY_5min' not in pipeline._result_cache


@pytest.mark.parametrize('mode', COMBINED_MODES)
def test_engine_lab_counts_native_child_failure(combined, monkeypatch, mode, failed_lane):
    monkeypatch.setattr(engine_lab, 'ENGINE_LAB_MODES', (mode,))
    lab = engine_lab.EngineLab()
    wrapper = lab._engines[f'{mode}|NIFTY|5min']
    monkeypatch.setattr(wrapper.native_engines['s1'], 'process', failed_lane)
    result = lab.process(candles_by_instrument={'NIFTY': {'5min': frame()}}, active_engine=mode)
    assert not lab._result_cache[f'{mode}|NIFTY|5min']['native_results'][COMBINED_MODES[mode]].get('error')
    assert result['rows'][0]['errors'] == 1


@pytest.mark.parametrize('mode', COMBINED_MODES)
def test_engine_lab_retries_failed_companion_without_a_candle_change(combined, monkeypatch, mode, failed_lane):
    monkeypatch.setattr(engine_lab, 'ENGINE_LAB_MODES', (mode,))
    lab = engine_lab.EngineLab()
    wrapper = lab._engines[f'{mode}|NIFTY|5min']
    original = wrapper.native_engines['s1'].process
    calls = []
    def transient(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            return failed_lane(**kwargs)
        return original(**kwargs)
    monkeypatch.setattr(wrapper.native_engines['s1'], 'process', transient)
    candles = {'NIFTY': {'5min': frame()}}
    lab.process(candles_by_instrument=candles, active_engine=mode)
    lab.process(candles_by_instrument=candles, active_engine=mode)
    assert len(calls) == 2
    assert not lab._result_cache[f'{mode}|NIFTY|5min']['native_results']['s1'].get('error')


@pytest.mark.parametrize('case,expected', [
    ('present', 'PRESENT'), ('gone', 'VANISHED'), ('opposite', 'FLIPPED'),
    ('stale', 'UNKNOWN'), ('wrong_instrument', 'UNKNOWN'), ('error', 'UNKNOWN'),
    ('wrong_provenance', 'UNKNOWN'), ('old_computation', 'UNKNOWN'),
])
def test_real_uts_repaint_reader_requires_fresh_owned_pipeline_evidence(combined, case, expected):
    from scanner import Scanner
    from tests.test_ut_smart_live_path_2026_09_26 import _trade_at, _flip, _uts_result
    combined.scanner_engine = 'ut_smart_s1'
    stamp = datetime(2026, 9, 28, 10)
    now = stamp + timedelta(minutes=2)
    candidate = _trade_at(stamp)
    candidate.source_signal_type = 'BUY'
    own = _uts_result(stamp, now, [_flip(stamp)])
    if case == 'gone':
        own['signals'] = []
    elif case == 'opposite':
        own['signals'] = [_flip(stamp, 'SELL')]
    elif case == 'stale':
        own['uts_evaluation']['observed_at'] = now - timedelta(seconds=31)
    elif case == 'wrong_instrument':
        own['instrument'] = 'SENSEX'
    elif case == 'error':
        own['error'] = 'native failure'
    foreign = NS(source_engine='s1', timestamp=stamp, signal_type='BUY')
    envelope = {'signals': [foreign], 'native_results': {
        'ut_smart': own, 's1': {'signals': [foreign]}}}
    scanner = Scanner.__new__(Scanner)
    scanner._cached_results = {'NIFTY_15min': envelope}
    scanner._session_squareoff_clock = lambda: now
    scanner._cached_result_metadata = {'NIFTY_15min': {
        'result_object': dict(envelope) if case == 'wrong_provenance' else envelope,
        'result': {'status': 'ready', 'computed_at_epoch': 99 if case == 'old_computation' else 101}}}
    assert scanner._repaint_signal_status(candidate, stamp, result_newer_than_epoch=100) == expected


@pytest.mark.parametrize('mode,native,lane', [
    ('quad_prime_s1', 's1', 'S1'), ('ut_smart_s1', 's1', 'S1'),
    ('ut_smart_s1', 'ut_smart', 'UTS'),
])
@pytest.mark.parametrize('own_exit', [True, False])
def test_active_monitor_uses_only_owned_native_exit(combined, monkeypatch, mode, native, lane, own_exit):
    from tests.test_s1_lane_2026_09_23 import _monitor, _open_trade
    import scanner as scanner_module
    combined.scanner_engine = mode
    monkeypatch.setattr(scanner_module, 'get_settings', lambda: combined)
    trade = _open_trade(source_engine=native, source_lane=lane)
    prefix = 'uts' if native == 'ut_smart' else 's1'
    signal = NS(source_engine=native, timestamp=trade.signal_time, signal_type='BUY',
        **{f'{prefix}_exit_reason': 'Momentum Exit',
           f'{prefix}_exit_time': trade.signal_time + timedelta(minutes=20), f'{prefix}_exit_price': 0.})
    foreign_native = 's1' if native != 's1' else COMBINED_MODES[mode]
    envelope = {'native_results': {
        native: {'signals': [signal] if own_exit else []},
        foreign_native: {'signals': [signal]}}}
    closed, updated, other = _monitor(monkeypatch, trade, spot=23405., premium=102., engine_result=envelope)
    assert closed == ([f'{lane}_EXIT'] if own_exit else [])
    assert other == []
    if not own_exit:
        assert updated


@pytest.mark.parametrize('lane,native', [('QP', 'quad_prime'), ('S1', 's1'), ('UTS', 'ut_smart')])
def test_sqlite_restart_preserves_engine_owner(combined, tmp_path, lane, native):
    from data.sqlite_db import DatabaseManager
    from trading.trade_manager import Trade
    from engine.scanner_modes import owner, native_result
    from tests.test_s1_system_integration_2026_09_24 import trade
    db = object.__new__(DatabaseManager)
    db._init_db(str(tmp_path / 'owners.sqlite'))
    try:
        original = trade(source_lane=lane)
        db.save_trade(original.to_dict())
        restored = Trade.from_dict(next(row for row in db.load_all_trades() if row['id'] == original.id))
        assert owner(restored) == native
        own = {'signals': []}
        assert native_result(restored, {'native_results': {native: own, 'foreign': {}}}) is own
    finally:
        db.conn.close()


@pytest.mark.parametrize('mode', COMBINED_MODES)
def test_combined_chart_settings_and_corrections_rebuild_without_authority_writes(setup, mode):
    settings, _, candles = setup
    settings.scanner_engine = mode
    settings.chart_rebuild_min_interval_seconds = 10000.
    host = Host(candles)
    before = snapshot(host)
    assert snapshot(host) is before and host.builds == 1
    settings.s1_min_score += 1
    changed = snapshot(host)
    assert changed is not before and host.builds == 2
    host.frame.iloc[-20, host.frame.columns.get_loc('close')] += 10.
    corrected = snapshot(host)
    assert corrected is not changed and host.builds == 3
    assert host._cached_results['NIFTY_15min'] is host.authority
    assert host.mtf._result_cache['NIFTY_15min'] is host.authority
    assert host.authority['state']['position'] == 999
    assert corrected['state']['position'] != 999


def test_combined_uts_chart_closes_terminal_bar_without_waiting_for_cache_floor(setup):
    settings, clock, candles = setup
    settings.scanner_engine = 'ut_smart_s1'
    settings.chart_rebuild_min_interval_seconds = 10000.
    clock[0] = pd.Timestamp('2026-09-30 09:29:59')
    host = Host(candles.loc[:'2026-09-30 09:15'])
    before = snapshot(host)
    assert entries(before) == []
    clock[0] = pd.Timestamp('2026-09-30 09:30')
    after = snapshot(host)
    assert host.builds == 2
    assert ('09:15', 'BUY') in entries(after)
    assert len(before['candles']) == len(after['candles'])


@pytest.mark.parametrize('mode,native', [
    ('quad_prime_s1', 'quad_prime'), ('quad_prime_s1', 's1'), ('ut_smart_s1', 's1'),
])
def test_real_native_candidate_selection_matches_standalone_without_contract_changes(live, monkeypatch, mode, native):
    import engine.signal_processor as processor_module
    import scanner as scanner_module
    from engine.alt_scanners import AltSignalEngine
    from tests.test_s1_engine_2026_09_23 import _frame
    from trading.trade_manager import IST
    candles = _frame()

    def build(composite):
        harness = live(tf='5min', settings_overrides={'s1_min_score': 60})
        settings = processor_module.get_settings()
        settings.scanner_engine = native
        monkeypatch.setattr('config.settings._settings', settings)
        engine = AltSignalEngine(signal_source=native)
        history = engine.process(candles, instrument='NIFTY', timeframe='5min')
        signal = [item for item in history['signals'] if item.timestamp.time() < datetime.strptime('12:30', '%H:%M').time()][-1]
        data = candles.iloc[:candles.index.get_loc(pd.Timestamp(signal.timestamp)) + 2]
        result = engine.process(data, instrument='NIFTY', timeframe='5min')
        assert result['signals'][-1].timestamp == signal.timestamp
        now = signal.timestamp.replace(tzinfo=None) + timedelta(minutes=5, seconds=20)

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return IST.localize(now) if tz else now

        for module in (processor_module, scanner_module):
            monkeypatch.setattr(module, 'datetime', Clock)
        spot = float(data.close.iloc[-1])
        harness.sc.candles = NS(get_candles=lambda instrument, tf: data if tf == '5min' else None,
            get_latest_price=lambda instrument: spot, get_max_timestamp=lambda: now)
        from engine.scanner_modes import scope_suffix
        settings.scanner_engine = mode if composite else native
        suffix = scope_suffix(NS(source_engine=native), settings)
        harness.sc._last_signal_time = {f'NIFTY_5min{suffix}': signal.timestamp - timedelta(seconds=1)}
        weights = NS(tf_weights={'1min': .15, '5min': .4, '15min': .45})
        score, direction = MultiTimeframeEngine._compute_confluence(weights, {'5min': result}, 'NIFTY')
        view = NS(instrument='NIFTY', results_1min=None, results_5min=result, results_15min=None,
            confluence_score=score, confluence_signal=direction, active_signals=result['signals'])
        if composite:
            view.results_5min = {'native_results': {
                native: result, ('s1' if native != 's1' else COMBINED_MODES[mode]): {'signals': [], 'state': None}}}
        args = list(harness.args)
        args[1], args[8], args[9] = view, spot, round(spot / 50) * 50
        return harness.processor.process_best_signal(*args), settings

    expected, _ = build(False)
    actual, settings = build(True)
    assert expected and actual
    assert len(actual) == len(expected)
    for candidate, reference in zip(actual, expected):
        assert candidate.status == 'TRADE SIGNAL'
        assert 0 < candidate.stop < candidate.price < candidate.target
        for field in ('direction', 'inst_type', 'trading_symbol', 'price', 'stop', 'target',
                      'spot_stop', 'entry_spot', 'source_engine', 'rr', 'grade', 'status', 'confidence'):
            assert getattr(candidate, field) == getattr(reference, field), field
        assert candidate.source_lane == ('S1' if native == 's1' else 'QP')
    assert settings.scanner_engine == mode
