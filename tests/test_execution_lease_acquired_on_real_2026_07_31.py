"""The REAL execution lease must be CLAIMED, not merely checked (live bug, 2026-07-31).

A live BANKNIFTY 57200CE entry failed all 7 retries with
"entry order blocked: REAL broker account is owned by another UT1 process" while the
underlying OS lock was FREE and the pid named in the message had exited the previous day.

Cause: the lease was acquired in exactly one place — main.py, at startup, and only when
settings.trading_mode was ALREADY "REAL". A process that reaches REAL through the runtime
mode switch (scanner.configure(mode="REAL"), which is the normal premarket path) hot-swapped
the broker without ever claiming the lease. has_execution_lease() then reported False
forever and execution_lease_block_reason() read stale owner metadata out of the lock file,
naming a dead process.

Second defect: the dynamically-created broker was built with no account_id, so it had NO
lease object at all and every guard failed OPEN — removing the cross-process protection
entirely rather than enforcing it.
"""
import os
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import pytest

from trading.account_execution_lease import AccountExecutionLease
from trading.kite_broker import KiteBroker


def _broker(account_id="TESTACCT1", instance_id="UT1:test"):
    return KiteBroker(kite=None, account_id=account_id, instance_id=instance_id)


def test_lease_is_claimed_on_demand_rather_than_refusing_a_free_lock(tmp_path):
    """The exact live failure: nobody holds the lock, but we never asked for it."""
    b = _broker()
    b._execution_lease.path = tmp_path / "acct.lock"
    assert b.has_execution_lease() is False        # never acquired — the bug's state
    b._require_account_execution_lease("entry order")   # must NOT raise
    assert b.has_execution_lease() is True
    b.release_execution_lease()


def test_a_genuine_live_holder_still_blocks(tmp_path):
    """Claiming on demand must not defeat the protection it exists to provide.

    The holder must be a REAL second process. This test used to hold the lock with
    another lease object inside the pytest process, which is not the thing the lease
    guards against — Windows byte-range locks are per-handle, so that also happens
    whenever ONE process rebuilds its own broker, and treating it as a foreign owner
    is what made a process refuse its own orders. Self-ownership is now allowed; a
    different process must still be refused, which is what this asserts.
    """
    path = tmp_path / "acct.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
                import sys, time
                from pathlib import Path
                sys.path.insert(0, {os.getcwd()!r})
                from trading.account_execution_lease import AccountExecutionLease
                lease = AccountExecutionLease("TESTACCT1", "UT1:holder")
                lease.path = Path({str(path)!r})
                assert lease.acquire()
                print("held", flush=True)
                time.sleep(30)
            """),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        b = _broker()
        b._execution_lease.path = path
        with pytest.raises(PermissionError) as err:
            b._require_account_execution_lease("entry order")
        assert "blocked" in str(err.value)
        assert b.has_execution_lease() is False
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_stale_metadata_from_a_dead_pid_does_not_block(tmp_path):
    """The lock file kept yesterday's owner json; that must not veto a free lock."""
    path = tmp_path / "acct.lock"
    seed = AccountExecutionLease("TESTACCT1", "UT1:8503")
    seed.path = path
    seed.acquire()
    seed.release()                                  # lock free, metadata remains on disk
    b = _broker()
    b._execution_lease.path = path
    b._require_account_execution_lease("entry order")
    assert b.has_execution_lease() is True
    b.release_execution_lease()


def test_broker_without_account_id_has_no_lease_at_all():
    """Documents the fail-OPEN hole that the scanner's dynamic broker fell into."""
    assert _broker(account_id="")._execution_lease is None
    assert _broker(account_id="").has_execution_lease() is True


def test_real_mode_switch_claims_the_lease_and_fails_closed():
    """Structural: configure(mode='REAL') must acquire, and fall back to HISTORICAL."""
    import inspect
    from scanner import Scanner
    body = "\n".join(
        l.split("#", 1)[0] for l in inspect.getsource(Scanner.configure).splitlines()
    )
    assert "acquire_execution_lease" in body, "REAL switch never claims the lease"
    swap = body.split("Broker Hot-Swapped")[0][-1400:]
    assert "acquire" in swap, "the lease must be claimed BEFORE the broker is hot-swapped"
    assert 'self.mode = "HISTORICAL"' in body, "must fail closed when the lease is refused"


def test_angel_one_lease_uses_execution_account(monkeypatch):
    from config.settings import get_settings
    from trading.angelone_broker import AngelOneBroker

    monkeypatch.setattr(get_settings(), "angelone_client_id", " ANGEL-ACCOUNT ")
    broker = AngelOneBroker(instance_id="UT1:8500")

    assert broker.client_id == "ANGEL-ACCOUNT"
    assert broker.execution_lease_established() is True
