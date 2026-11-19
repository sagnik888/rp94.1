"""The contract master must come from Kite, in the schema the expiry logic expects.

Everything downstream of expiry_manager — expiry rolling, gamma windows, get_token and
its closest-strike fallback — is written against the AngelOne column schema. Rather than
rewrite that logic, the Kite dump is translated into the same shape. Two conventions
have to survive the translation or contract lookup silently returns nothing:

  * expiry is the "%d%b%Y" string _build_expiry_map parses ("04AUG2026")
  * strike is stored x100, because get_token compares float(strike) * 100

A wrong strike scale here does not raise: it just fails to match, falls through to the
"closest available strike" path, and quietly trades the wrong contract.
"""
from datetime import date

import pandas as pd
import pytest

from engine.expiry_manager import ExpiryManager


def _kite_row(**over):
    row = {
        "instrument_token": 12345, "tradingsymbol": "NIFTY2680424350CE", "name": "NIFTY",
        "expiry": date(2026, 8, 4), "strike": 24350.0, "instrument_type": "CE",
        "exchange": "NFO", "lot_size": 65,
    }
    row.update(over)
    return row


@pytest.fixture
def staged(monkeypatch):
    """Stage a fake Kite dump through the real translation path."""
    from config.settings import get_settings

    monkeypatch.setattr(get_settings(), "kite_api_key", "fixture-key")

    def _build(rows_by_exchange):
        class _Source:
            def __init__(self, auth=None, kite=None):
                import threading
                self._lock = threading.RLock()
                self._instruments = {}
                self._rows = rows_by_exchange

            def load_instruments(self, exchanges=(), force=False):
                self._instruments = {ex: self._rows.get(ex, []) for ex in exchanges}
                return sum(len(v) for v in self._instruments.values())

        monkeypatch.setattr("data.kite_market_data.KiteDataSource", _Source)
        monkeypatch.setattr(
            "data.kite_auth.get_kite_auth", lambda *a, **k: type("A", (), {"kite": object()})()
        )
        return ExpiryManager._kite_master_frame()
    return _build


def test_frame_carries_the_columns_the_expiry_map_requires(staged):
    frame = staged({"NFO": [_kite_row()], "BFO": []})
    assert frame is not None
    assert {"name", "instrumenttype", "expiry", "symbol", "token"}.issubset(frame.columns)


def test_expiry_is_written_in_the_parsed_string_format(staged):
    frame = staged({"NFO": [_kite_row()], "BFO": []})
    assert frame.iloc[0]["expiry"] == "04AUG2026"
    # The map parses with this exact format; a mismatch yields NaT and no expiries.
    assert pd.to_datetime(frame["expiry"], format="%d%b%Y", errors="coerce").notna().all()


def test_strike_is_scaled_by_100_to_match_get_token(staged):
    """get_token compares against float(strike) * 100; the wrong scale silently misses."""
    frame = staged({"NFO": [_kite_row(strike=24350.0)], "BFO": []})
    assert frame.iloc[0]["strike"] == 2435000.0


def test_option_and_future_types_are_mapped(staged):
    frame = staged({"NFO": [
        _kite_row(instrument_type="CE"),
        _kite_row(instrument_type="PE", tradingsymbol="NIFTY2680424350PE"),
        _kite_row(instrument_type="FUT", tradingsymbol="NIFTY26AUGFUT", strike=0.0),
    ], "BFO": []})
    assert sorted(frame["instrumenttype"].unique()) == ["FUTIDX", "OPTIDX"]
    assert (frame["instrumenttype"] == "FUTIDX").sum() == 1


def test_equity_rows_are_dropped(staged):
    """Only index derivatives belong in this master."""
    frame = staged({"NFO": [_kite_row(), _kite_row(instrument_type="EQ", expiry=None)], "BFO": []})
    assert len(frame) == 1


def test_sensex_comes_from_bfo(staged):
    frame = staged({"NFO": [], "BFO": [
        _kite_row(name="SENSEX", tradingsymbol="SENSEX2680678100CE", strike=78100.0, exchange="BFO")
    ]})
    assert frame.iloc[0]["exch_seg"] == "BFO"
    assert frame.iloc[0]["strike"] == 7810000.0


def test_token_is_the_kite_instrument_token(staged):
    frame = staged({"NFO": [_kite_row(instrument_token=987654)], "BFO": []})
    assert frame.iloc[0]["token"] == "987654"


def test_no_session_returns_none_rather_than_a_bad_master(monkeypatch):
    """A missing session must not silently produce an empty/partial contract set."""
    monkeypatch.setattr(
        "data.kite_auth.get_kite_auth", lambda *a, **k: type("A", (), {"kite": None})()
    )
    assert ExpiryManager._kite_master_frame() is None
