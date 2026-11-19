"""Combined base observations retain their standalone dependency and age semantics."""
import math
import numpy as np
import pandas as pd
import pytest

from config.settings import get_settings
from engine.combined_scanner import CombinedSignalEngine, native_mtf_view
from engine.multi_timeframe import MultiTimeframeEngine
from engine.scanner_modes import COMBINED_MODES
from tests.test_combined_s1_forming_reuse_2026_10_05 import _signature
import engine.multi_timeframe as mtf_module
import engine.ut_smart_engine as uts_module


@pytest.fixture(scope="module")
def saved_frames():
    sessions = pd.bdate_range(end="2026-09-30", periods=40)
    frames = {}
    for minutes in (1, 5, 15):
        stamps = [bar for day in sessions for bar in pd.date_range(
            day + pd.Timedelta(hours=9, minutes=15),
            day + pd.Timedelta(hours=15), freq=f"{minutes}min")]
        index = pd.DatetimeIndex(stamps[-900:])
        i = np.arange(len(index))
        close = 25_000 + i / 7 + 35 * np.sin(i / 11)
        frames[f"{minutes}min"] = pd.DataFrame({
            "open": close - 1, "high": close + 4, "low": close - 4,
            "close": close, "volume": 1000 + i % 20,
        }, index=index)
    return frames


def make_pipeline(monkeypatch, mode, saved_frames, *, timeframe="5min", fire="confirmed", intrabar=False):
    settings = get_settings()
    monkeypatch.setattr(settings, "scanner_engine", mode)
    monkeypatch.setattr(settings, "s1_lane_forming_recompute_seconds", 15.)
    monkeypatch.setattr(settings, "uts_lane_allow_intrabar_entries", intrabar)
    monkeypatch.setattr(settings, "live_intrabar_signal_entries", intrabar)
    monkeypatch.setattr(settings, "ut_signal_mode", "realtime" if intrabar else "confirmed")
    clock = [pd.Timestamp("2026-09-30 15:00:01", tz="Asia/Kolkata").timestamp()]
    monkeypatch.setattr(mtf_module.time, "time", lambda: clock[0])
    monkeypatch.setattr(uts_module, "evaluation_clock", lambda as_of=None: pd.Timestamp(
        clock[0], unit="s", tz="UTC").tz_convert("Asia/Kolkata").tz_localize(None))
    pipeline = MultiTimeframeEngine({"signal_mode": fire})
    key = f"NIFTY_{timeframe}"
    engine = pipeline.engines[key]
    assert isinstance(engine, CombinedSignalEngine)
    calls = {native: 0 for native in (COMBINED_MODES[mode], "s1")}
    for native, child in engine.native_engines.items():
        original = child.process

        def counted(*args, _original=original, _native=native, **kwargs):
            calls[_native] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(child, "process", counted)
    pipeline.engines = {key: engine}
    frames = {tf: frame.copy(deep=True) for tf, frame in saved_frames.items()}
    for tf, frame in frames.items():
        pipeline.update_data("NIFTY", tf, frame)
    first = getattr(pipeline.process_instrument("NIFTY"), f"results_{timeframe}")
    assert first and all(not lane.get("error") for lane in first["native_results"].values())
    return pipeline, frames, calls, clock, first


def changed_pass(pipeline, frames, clock, *, timeframe="5min", elapsed=2., **kwargs):
    clock[0] += elapsed
    frame = frames[timeframe].copy(deep=True)
    frame.iloc[-1, frame.columns.get_loc("close")] += 1.
    frame.iloc[-1, frame.columns.get_loc("high")] += 1.
    frames[timeframe] = frame
    pipeline.update_data("NIFTY", timeframe, frame)
    return pipeline.process_instrument("NIFTY", **kwargs)


@pytest.mark.parametrize("mode", COMBINED_MODES)
@pytest.mark.parametrize("timeframe", ("5min", "15min"))
def test_confirmed_combined_base_reuses_on_forming_ticks(monkeypatch, mode, timeframe, saved_frames):
    pipeline, frames, calls, clock, first = make_pipeline(monkeypatch, mode, saved_frames, timeframe=timeframe)
    second = changed_pass(pipeline, frames, clock, timeframe=timeframe)
    base = COMBINED_MODES[mode]
    assert calls[base] == 1
    assert getattr(second, f"results_{timeframe}")["native_results"][base] is first["native_results"][base]


@pytest.mark.parametrize("mode", COMBINED_MODES)
def test_dependency_tick_does_not_replay_combined_base(monkeypatch, mode, saved_frames):
    pipeline, frames, calls, clock, first = make_pipeline(monkeypatch, mode, saved_frames)
    changed_pass(pipeline, frames, clock, timeframe="15min")
    assert calls[COMBINED_MODES[mode]] == 1


@pytest.mark.parametrize("mode", COMBINED_MODES)
def test_base_sizing_change_invalidates_native_observation(monkeypatch, mode, saved_frames):
    pipeline, frames, calls, clock, first = make_pipeline(monkeypatch, mode, saved_frames)
    second = changed_pass(pipeline, frames, clock, capital=200000., risk_pct=2., lots=2, lot_size=50)
    assert calls[COMBINED_MODES[mode]] == 2
    assert calls["s1"] == 2


@pytest.mark.parametrize("mode", COMBINED_MODES)
def test_native_lane_metadata_preserves_original_observation_clock(monkeypatch, mode, saved_frames):
    pipeline, frames, calls, clock, first = make_pipeline(monkeypatch, mode, saved_frames)
    first_meta = dict(pipeline.get_timeframe_metadata("NIFTY", "5min")["result"])
    second = changed_pass(pipeline, frames, clock, elapsed=16.)
    assert calls["s1"] == 2 and calls[COMBINED_MODES[mode]] == 1
    view = native_mtf_view(second, COMBINED_MODES[mode])
    assert view.timeframe_metadata["5min"]["result"]["computed_at_epoch"] == first_meta["computed_at_epoch"]
    assert second.timeframe_metadata["5min"]["result"]["computed_at_epoch"] > first_meta["computed_at_epoch"]


@pytest.mark.parametrize("mode", COMBINED_MODES)
def test_force_fresh_observation_recomputes_both_native_lanes(monkeypatch, mode, saved_frames):
    pipeline, frames, calls, clock, first = make_pipeline(monkeypatch, mode, saved_frames)
    pipeline.require_fresh_result("NIFTY", "5min", newer_than_epoch=clock[0] + 1.)
    changed_pass(pipeline, frames, clock)
    assert calls == {native: 2 for native in calls}


@pytest.mark.parametrize("mode", COMBINED_MODES)
@pytest.mark.parametrize("timeframe", ("1min", "5min", "15min"))
@pytest.mark.parametrize("fire", ("confirmed", "realtime"))
def test_native_base_matches_standalone_cache_sequence(monkeypatch, mode, timeframe, fire, saved_frames):
    pipeline, frames, calls, clock, first = make_pipeline(
        monkeypatch, mode, saved_frames, timeframe=timeframe, fire=fire)
    settings = get_settings()
    base = COMBINED_MODES[mode]
    monkeypatch.setattr(settings, "scanner_engine", base)
    standalone = MultiTimeframeEngine({"signal_mode": fire})
    standalone.engines = {f"NIFTY_{timeframe}": standalone.engines[f"NIFTY_{timeframe}"]}
    for tf, frame in frames.items():
        standalone.update_data("NIFTY", tf, frame)
    before = getattr(standalone.process_instrument("NIFTY"), f"results_{timeframe}")
    assert _signature(before) == _signature(first["native_results"][base])
    monkeypatch.setattr(settings, "scanner_engine", mode)
    combined = changed_pass(pipeline, frames, clock, timeframe=timeframe)
    for tf, frame in frames.items():
        standalone.update_data("NIFTY", tf, frame)
    reference = getattr(standalone.process_instrument("NIFTY"), f"results_{timeframe}")
    native = getattr(combined, f"results_{timeframe}")["native_results"][base]
    assert _signature(native) == _signature(reference)
    assert native.get("uts_evaluation") == reference.get("uts_evaluation")


@pytest.mark.parametrize("mode", COMBINED_MODES)
@pytest.mark.parametrize("mutation", ("closed_correction", "new_bar", "reset_results", "reset_params", "base_parameter"))
def test_base_required_recomputations_preserve_native_invalidation(monkeypatch, mode, mutation, saved_frames):
    pipeline, frames, calls, clock, first = make_pipeline(monkeypatch, mode, saved_frames)
    if mutation == "closed_correction":
        frames["5min"].iloc[-5, frames["5min"].columns.get_loc("close")] += 2.
    elif mutation == "new_bar":
        row = frames["5min"].iloc[[-1]].copy(deep=True)
        row.index += pd.Timedelta(minutes=5)
        frames["5min"] = pd.concat([frames["5min"], row])
        clock[0] += 300.0
    elif mutation == "reset_results":
        pipeline._result_cache.clear()
    elif mutation == "reset_params":
        pipeline.apply_engine_params({"key_value": 1.7, "signal_mode": "confirmed"})
    else:
        native = pipeline.engines["NIFTY_5min"].native_engines[COMBINED_MODES[mode]]
        native.adx_threshold += 1.
    changed_pass(pipeline, frames, clock)
    assert calls[COMBINED_MODES[mode]] == 2


@pytest.mark.parametrize("mode", COMBINED_MODES)
def test_failed_base_lane_retries_without_changed_inputs(monkeypatch, mode, saved_frames):
    pipeline, frames, calls, clock, first = make_pipeline(monkeypatch, mode, saved_frames)
    base = COMBINED_MODES[mode]
    child = pipeline.engines["NIFTY_5min"].native_engines[base]
    original = child.process
    attempts = []

    def temporary_failure(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("base observation unavailable")
        return original(*args, **kwargs)

    monkeypatch.setattr(child, "process", temporary_failure)
    pipeline.require_fresh_result("NIFTY", "5min", newer_than_epoch=clock[0] + 1.)
    failed = changed_pass(pipeline, frames, clock)
    assert failed.results_5min["native_results"][base]["error"]
    clock[0] += 1.
    recovered = pipeline.process_instrument("NIFTY")
    assert len(attempts) == 2
    assert not recovered.results_5min["native_results"][base].get("error")


def test_uts_closed_bar_boundary_requires_new_base_observation(monkeypatch, saved_frames):
    pipeline, frames, calls, clock, first = make_pipeline(monkeypatch, "ut_smart_s1", saved_frames)
    clock[0] = pd.Timestamp("2026-09-30 15:05:01", tz="Asia/Kolkata").timestamp()
    current = pipeline.process_instrument("NIFTY")
    assert calls["ut_smart"] == 2
    assert current.results_5min["native_results"]["ut_smart"]["uts_evaluation"]["last_closed_bar"] == frames["5min"].index[-1]


def test_uts_reuse_preserves_age_then_recomputes_at_existing_freshness_bound(monkeypatch, saved_frames):
    pipeline, frames, calls, clock, first = make_pipeline(monkeypatch, "ut_smart_s1", saved_frames)
    initial = first["native_results"]["ut_smart"]["uts_evaluation"]["observed_at"]
    second = changed_pass(pipeline, frames, clock, elapsed=16.)
    assert second.results_5min["native_results"]["ut_smart"]["uts_evaluation"]["observed_at"] == initial
    clock[0] += 14.
    third = pipeline.process_instrument("NIFTY")
    assert calls["ut_smart"] == 2
    assert third.results_5min["native_results"]["ut_smart"]["uts_evaluation"]["observed_at"] > initial


@pytest.mark.parametrize("timeframe", ("1min", "5min", "15min"))
def test_uts_intrabar_optin_keeps_base_current(monkeypatch, timeframe, saved_frames):
    pipeline, frames, calls, clock, first = make_pipeline(
        monkeypatch, "ut_smart_s1", saved_frames, timeframe=timeframe, intrabar=True)
    changed_pass(pipeline, frames, clock, timeframe=timeframe)
    assert calls["ut_smart"] == 2
