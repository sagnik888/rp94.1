"""Unit tests for UpstoxAuth session lifecycle, daily cutoff, and token decoding."""
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from data.upstox_auth import (
    IST,
    UpstoxAuth,
    decode_jwt_expiry,
    most_recent_token_cutoff,
    next_token_expiry,
    read_upstox_env,
)


def test_upstox_reads_fresh_oauth_config(tmp_path, monkeypatch):
    from config import settings
    from data import upstox_auth

    env_file = tmp_path / ".env"
    env_file.write_text("UPSTOX_API_KEY=file-key  # registered app\n"
                        "UPSTOX_API_SECRET='file-secret'\n"
                        "UPSTOX_REDIRECT_URI=http://localhost:8500/\n", encoding="utf-8-sig")
    monkeypatch.setattr(upstox_auth, "ROOT", tmp_path)
    monkeypatch.setattr(settings, "get_settings", lambda: MagicMock(
        upstox_api_key="cached-key", upstox_api_secret="cached-secret",
        upstox_redirect_uri="http://127.0.0.1:8500/api/upstox/callback", upstox_user_id="",
    ))
    for key in ("UPSTOX_API_KEY", "UPSTOX_API_SECRET", "UPSTOX_REDIRECT_URI", "UPSTOX_ACCESS_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    auth = UpstoxAuth(tmp_path / "token.json")

    params = parse_qs(urlsplit(auth.login_url()).query)
    assert params["client_id"] == ["file-key"]
    assert params["redirect_uri"] == ["http://localhost:8500/"]
    assert read_upstox_env()["UPSTOX_API_SECRET"] == "file-secret"

    monkeypatch.setenv("UPSTOX_API_KEY", "deployment-key")
    assert parse_qs(urlsplit(auth.login_url()).query)["client_id"] == ["deployment-key"]


def test_upstox_uses_one_exchange_with_the_authorization_redirect(tmp_path, monkeypatch):
    from data import upstox_auth
    import pytest

    uri = "http://localhost:8500/"
    monkeypatch.setattr(upstox_auth, "read_upstox_env", lambda: {
        "UPSTOX_API_KEY": "configured-key",
        "UPSTOX_API_SECRET": "configured-secret",
        "UPSTOX_REDIRECT_URI": uri,
    })
    auth = UpstoxAuth(tmp_path / "token.json")
    response = MagicMock(status_code=401)
    response.json.return_value = {"errors": [{"errorCode": "UDAPI100068"}]}
    post = MagicMock(return_value=response)
    monkeypatch.setattr(upstox_auth.requests, "post", post)
    monkeypatch.setattr(auth, "has_valid_token", lambda: True)

    params = parse_qs(urlsplit(auth.login_url()).query)
    with pytest.raises(RuntimeError, match="UDAPI100068"):
        auth.complete_login("single-use-code")
    post.assert_called_once()
    assert post.call_args.kwargs["data"]["redirect_uri"] == params["redirect_uri"][0] == uri


class TestUpstoxAuthLifecycle(unittest.TestCase):
    def test_daily_token_cutoffs(self):
        cutoff = most_recent_token_cutoff()
        self.assertEqual(cutoff.tzinfo, IST)
        self.assertEqual(cutoff.hour, 3)
        self.assertEqual(cutoff.minute, 30)

        expiry = next_token_expiry()
        self.assertGreater(expiry, cutoff)
        self.assertEqual((expiry - cutoff).days, 1)

    def test_extract_code_variants(self):
        auth = UpstoxAuth(token_file=Path("temp_nonexistent.json"))
        # Raw code
        self.assertEqual(auth.extract_code("qHZ4HU"), "qHZ4HU")
        # Full redirect URL with query params
        url1 = "http://127.0.0.1:8500/api/upstox/callback?code=qHZ4HU&state=xyz"
        self.assertEqual(auth.extract_code(url1), "qHZ4HU")
        # Direct URL with code at end
        url2 = "http://127.0.0.1:8500/?code=my_auth_code_123"
        self.assertEqual(auth.extract_code(url2), "my_auth_code_123")
        # Empty
        self.assertEqual(auth.extract_code(""), "")

    def test_jwt_expiry_decoding(self):
        # Sample JWT with known expiry
        token = (
            "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ."
            "eyJzdWIiOiI3SkFUWUwiLCJqdGkiOiI2YWZlMTg0ZDVhZTdlMTdlNmY3NGUyZTQiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzk1MDM3MjYxLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3OTUwMzkyMDB9."
            "TOO064x8OJtpb7QMQgbFjXzxwM52rxqKGZcT6V10wr0"
        )
        exp = decode_jwt_expiry(token)
        self.assertIsNotNone(exp)
        self.assertEqual(exp.hour, 3)
        self.assertEqual(exp.minute, 30)

    @patch("data.upstox_auth.UpstoxAuth._verify_live")
    def test_complete_login_with_direct_token(self, mock_verify):
        mock_verify.return_value = (True, "", {"user_id": "7JATYL", "user_name": "UT1 User"})
        test_file = Path("tests/temp_test_upstox_token.json")
        try:
            auth = UpstoxAuth(token_file=test_file)
            token = (
                "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ."
                "eyJzdWIiOiI3SkFUWUwiLCJleHAiOjk5OTk5OTk5OTl9.signature"
            )
            res = auth.complete_login(token)
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["user_id"], "7JATYL")
            self.assertTrue(auth.has_valid_token())
            st = auth.status()
            self.assertTrue(st["connected"])
            self.assertEqual(st["status"], "LIVE")
        finally:
            if test_file.exists():
                test_file.unlink()


if __name__ == "__main__":
    unittest.main()
