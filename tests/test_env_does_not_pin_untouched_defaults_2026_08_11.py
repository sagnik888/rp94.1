"""A setting added in code was frozen at its first-ever default the moment it persisted.

Settings is a pydantic BaseSettings with ``env_file='.env'``, so the FILE wins over the
class default. save_to_env() then dumped EVERY field, so any value still sitting at its
default got written into .env — turning a code default into a pinned configuration choice
nobody made. Changing that default afterwards did nothing, because the file shadowed it.

Measured 2026-08-11: 681 of 741 keys in .env sat at their dataclass default. Only 60 were
genuine customisations, mostly credentials. Observed concretely when
surge_tick_proxy_enabled was added defaulting False: the running scanner persisted
SURGE_TICK_PROXY_ENABLED='False', and later changing the default to True had no effect.

The fix is narrow on purpose. A field is skipped only when BOTH:
  * it is absent from .env, and
  * its value still equals the class default

Keys already in .env keep being refreshed, so nothing an operator set — or that an
earlier build pinned — changes underneath them. Only NEW defaults stop being written,
which is exactly what makes a default upgradeable again.
"""
import json

import pytest

from config.settings import Settings


class _Recorder:
    """Captures what save_to_env would write, instead of touching the real .env."""

    def __init__(self):
        self.written = {}

    def __call__(self, key, value):
        self.written[key] = value


@pytest.fixture
def capture(monkeypatch, tmp_path):
    import config.settings as cs

    rec = _Recorder()
    monkeypatch.setattr(cs, "_set_env_key_with_retry", rec)
    monkeypatch.chdir(tmp_path)          # so Path('.env') is a scratch file
    (tmp_path / ".env").write_text("", encoding="utf-8")
    return rec


def _existing(monkeypatch, mapping):
    import config.settings as cs

    monkeypatch.setattr(cs.dotenv, "dotenv_values", lambda *a, **k: dict(mapping))


# ── the trap ─────────────────────────────────────────────────────────────────

def test_a_field_at_its_default_is_not_written_when_absent(capture, monkeypatch):
    _existing(monkeypatch, {})
    Settings().save_to_env()
    assert "SURGE_TICK_PROXY_ENABLED" not in capture.written, (
        "an untouched default was pinned into .env, shadowing the code default forever"
    )


def test_most_of_the_file_is_no_longer_rewritten(capture, monkeypatch):
    """681 of 741 keys were being pinned at their default."""
    _existing(monkeypatch, {})
    s = Settings()
    s.save_to_env()
    total = len(getattr(type(s), "model_fields", {}))
    assert len(capture.written) < total * 0.35, (
        f"still writing {len(capture.written)} of {total} fields"
    )


# ── what must still be written ───────────────────────────────────────────────

def test_a_genuinely_customised_value_is_written(capture, monkeypatch):
    _existing(monkeypatch, {})
    s = Settings()
    s.risk_opt_pct = float(s.risk_opt_pct) + 3.0
    s.save_to_env()
    assert "RISK_OPT_PCT" in capture.written
    assert capture.written["RISK_OPT_PCT"] == str(s.risk_opt_pct)


def test_a_key_already_in_env_is_still_refreshed(capture, monkeypatch):
    """An operator's existing line — or one an older build pinned — must not be left
    stale just because the value happens to equal the default."""
    _existing(monkeypatch, {"RISK_OPT_PCT": "9.0"})
    Settings().save_to_env()
    assert "RISK_OPT_PCT" in capture.written


def test_auto_execution_authority_is_never_persisted_on(capture, monkeypatch):
    _existing(monkeypatch, {"AUTO_MODE": "True"})
    settings = Settings()
    settings.auto_mode = True

    settings.save_to_env()

    assert capture.written["AUTO_MODE"] == "False"


def test_secrets_already_present_are_preserved(capture, monkeypatch):
    from config.settings import PROTECTED_SECRET_ENV_KEYS

    key = sorted(PROTECTED_SECRET_ENV_KEYS)[0]
    _existing(monkeypatch, {key: "real-secret-value"})
    s = Settings()
    setattr(s, key.lower(), "")          # blank in memory
    s.save_to_env()
    assert capture.written.get(key) == "real-secret-value", "a live secret was blanked"


def test_inert_settings_are_still_skipped(capture, monkeypatch):
    from config.inert_settings import INERT

    _existing(monkeypatch, {})
    Settings().save_to_env()
    for name in INERT:
        assert name.upper() not in capture.written


def test_runtime_settings_save_preserves_upstox_oauth_rotation(capture, monkeypatch):
    values = {
        "UPSTOX_API_KEY": "rotated-key",
        "UPSTOX_API_SECRET": "rotated-secret",
        "UPSTOX_REDIRECT_URI": "http://localhost:8500/",
        "UPSTOX_USER_ID": "configured-account",
    }
    _existing(monkeypatch, values)
    settings = Settings(upstox_api_key="stale-key", upstox_api_secret="stale-secret")
    settings.save_to_env()
    assert values.keys().isdisjoint(capture.written)


def test_list_values_still_serialise_as_json(capture, monkeypatch):
    _existing(monkeypatch, {})
    s = Settings()
    s.active_indices = ["NIFTY", "BANKNIFTY"]
    s.save_to_env()
    assert json.loads(capture.written["ACTIVE_INDICES"]) == ["NIFTY", "BANKNIFTY"]


# ── the property that makes a default upgradeable ────────────────────────────

def test_changing_a_code_default_now_reaches_a_clean_install(capture, monkeypatch):
    """The whole point: with the key absent from .env, the class default governs — so
    editing it in code actually changes behaviour."""
    _existing(monkeypatch, {})
    Settings().save_to_env()
    assert "SURGE_TICK_PERCENTILE_FLOOR" not in capture.written
    # and a fresh Settings with no such env key reads the code default
    assert Settings().surge_tick_percentile_floor == pytest.approx(97.0)
