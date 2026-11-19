"""Native policy ownership for scanners that run S1 beside another engine.

The context is task-local and never mutates Settings. Trades persist ownership in
the existing source_lane column; signal sources keep their original meaning.
"""
from contextlib import contextmanager
from contextvars import ContextVar

COMBINED_MODES = {'quad_prime_s1': 'quad_prime', 'ut_smart_s1': 'ut_smart'}
INDEPENDENT_LANE_MODES = frozenset({'s1s2_dual'})
SCANNER_MODES = frozenset({'utbot', 'ema', 'consensus', 'utbot_consensus',
    'quad_union', 'quad_prime', 'adaptive_dual', 's1', 'ut_smart',
    'ut_ema', *COMBINED_MODES, *INDEPENDENT_LANE_MODES})
_native_mode = ContextVar('scanner_native_policy', default=None)


def configured_mode(settings=None):
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()
    return str(getattr(settings, 'scanner_engine', 'utbot') or 'utbot').lower()


def owner(obj):
    lane = str(getattr(obj, 'source_lane', '') or '').upper()
    source = str(getattr(obj, 'source_engine', '') or '').lower()
    s12_lane = str(getattr(obj, 's12_lane', '') or '').upper()
    if lane in {'D1', 'D2'} or s12_lane in {'D1', 'D2'}:
        return (lane if lane in {'D1', 'D2'} else s12_lane).lower()
    if lane == 'S1' or source == 's1':
        return 's1'
    if lane == 'UTS' or source == 'ut_smart':
        return 'ut_smart'
    if lane == 'QP' or source in {'quad_prime', 'utbot', 'ema', 'supertrend', 'consensus', 'multi'}:
        return 'quad_prime'
    return 'unknown'


def scanner_mode(settings=None, obj=None):
    mode = configured_mode(settings)
    if mode in COMBINED_MODES:
        return owner(obj) if obj is not None else (_native_mode.get() or COMBINED_MODES[mode])
    if mode in INDEPENDENT_LANE_MODES and obj is not None:
        lane_owner = owner(obj)
        return lane_owner if lane_owner in {'d1', 'd2'} else mode
    return mode


@contextmanager
def native_policy(mode):
    token = _native_mode.set(mode)
    try:
        yield
    finally:
        _native_mode.reset(token)


def scope_suffix(obj, settings=None):
    mode = configured_mode(settings)
    if mode in INDEPENDENT_LANE_MODES:
        identity = owner(obj) if obj is not None else (_native_mode.get() or mode)
        return '_' + identity
    identity = owner(obj) if obj is not None else (_native_mode.get() or COMBINED_MODES.get(mode, 'quad_prime'))
    return '_' + identity if mode in COMBINED_MODES else ''


def native_result(obj, result):
    if isinstance(result, dict) and 'native_results' in result:
        return result['native_results'].get(owner(obj))
    return result


def same_owner(left, right, settings=None):
    mode = configured_mode(settings)
    if mode not in COMBINED_MODES and mode not in INDEPENDENT_LANE_MODES:
        return True
    identity = owner(left)
    return identity != 'unknown' and identity == owner(right)
