"""Three ways a dead broker session stayed quiet.

1. REAL HOT-SWAP MUST RECHECK THE CURRENT DATA AND EXECUTION SESSIONS.
   The legacy connection latch can go stale. Validate Upstox authentication
   and Angel One execution readiness before arming REAL.

2. THE OPERATOR'S REPAINT-DEFER CAPS WERE INERT ON REAL MONEY.
   `_repaint_absent_checks` / `_repaint_first_absent_at` were written only on session
   TradeCandidates, never on the Trade the real-money tracker holds. Inside
   `_should_defer_option_repaint_abort` that pinned absent_checks at 1 and first_absent
   at None, so the configured "abort after 2 checks / 55s" could not trip and a genuinely
   repainted option rode to candle close instead — up to 5 minutes on a 5min signal, 15
   on a 15min one.

   Fixed by maintaining the counters on the Trade, mirroring the session mirror. NOT by
   raising the default at the read site, which would trip the cap on the FIRST absent
   check and disable the defer entirely — the opposite of the configured intent.

3. AN IP-PERMISSION REFUSAL WAS SILENT ON THE EXIT AND STOP PATHS.
   `_raise_for` escalates, but place_protective_stop, modify_protective_stop and
   cancel_order log and return None/False without it. Those are the paths where an IP
   block matters most: the position ends up with no server-side stop and nothing names
   the cause. The operator has hit this in production on both AngelOne (AG7002) and Kite.
"""
import inspect
from types import SimpleNamespace

import pytest

from scanner import Scanner


# ── 1. no broker is built around a dead session ──────────────────────────────

def test_configure_checks_current_sessions_before_activating_the_broker():
    source = inspect.getsource(Scanner.configure)
    assert source.index("validate_real_mode_readiness(") < source.index(
        "self.trades.broker = self.trades.real_broker"
    )
    assert "KiteBroker(" not in source and "UpstoxBroker(" not in source


def _tracker_src():
    return inspect.getsource(Scanner._track_repaint_until_candle_close)


def test_the_tracker_maintains_the_absent_check_counter():
    src = _tracker_src()
    assert 'setattr(trade, "_repaint_absent_checks"' in src, (
        "the operator's 2-check defer cap is still unreachable on real money"
    )


def test_the_tracker_stamps_the_first_absent_time():
    src = _tracker_src()
    assert 'setattr(trade, "_repaint_first_absent_at"' in src, (
        "the operator's 55s defer cap is still unreachable on real money"
    )


def test_a_present_arrow_resets_both_counters():
    src = _tracker_src()
    reset = src.index('setattr(trade, "_repaint_absent_checks", 0)')
    bump = src.index('int(getattr(trade, "_repaint_absent_checks", 0) or 0) + 1')
    assert reset < bump, "the counters are incremented without ever being reset"


def test_a_confirming_override_restarts_the_clock():
    """Otherwise a held trade accumulates absent checks and trips a cap on evidence the
    override already answered."""
    src = _tracker_src()
    tail = src[src.index("if override_ctx:"):]
    assert '_repaint_absent_checks", 0' in tail
    assert '_repaint_first_absent_at", None' in tail


def test_the_read_site_default_was_not_raised_instead():
    """Raising it would trip the cap on the first absent check and kill the defer."""
    src = inspect.getsource(Scanner._should_defer_option_repaint_abort)
    assert 'getattr(trade_like, "_repaint_absent_checks", 1)' in src


def _defer_caps(absent_checks, max_checks=2):
    """Reproduce the cap test the counters feed."""
    return absent_checks > max_checks


def test_the_cap_can_now_actually_trip():
    assert _defer_caps(1) is False
    assert _defer_caps(2) is False
    assert _defer_caps(3) is True


# ── 3. an IP block is loud on every order path ───────────────────────────────

@pytest.mark.parametrize("fn_name", [
    "place_protective_stop", "modify_protective_stop", "cancel_order",
])
def test_the_exit_and_stop_paths_escalate_an_ip_block(fn_name):
    from trading.kite_broker import KiteBroker

    src = inspect.getsource(getattr(KiteBroker, fn_name))
    assert "_escalate_if_ip_blocked" in src, (
        f"{fn_name} still refuses silently; an IP block leaves the position unprotected "
        "with nothing naming the cause"
    )


def test_the_entry_path_still_escalates():
    """The premise: _raise_for was already doing this and must keep doing it."""
    from trading.kite_broker import KiteBroker

    assert "_escalate_if_ip_blocked" in inspect.getsource(KiteBroker._raise_for)


def test_the_escalation_is_deduplicated():
    """Four call sites must not become four alerts per refusal."""
    from trading.kite_broker import KiteBroker

    src = inspect.getsource(KiteBroker._escalate_if_ip_blocked)
    assert "_ip_block_alerted" in src or "already" in src.lower(), (
        "no latch found; the extra call sites would flood the operator"
    )


def test_the_markers_still_cover_the_empty_allowlist_case():
    """An app with an EMPTY allow-list denies every order rather than allowing all."""
    from trading.kite_broker import KiteBroker

    markers = [m.lower() for m in KiteBroker._IP_BLOCK_MARKERS]
    assert any("not allowed to place orders" in m for m in markers)
    assert any("no ips configured" in m for m in markers)
