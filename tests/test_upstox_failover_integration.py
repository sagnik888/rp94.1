"""Unit tests for Upstox Primary Broker and 30s Angel One Fallback Integration."""
import time
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from trading.upstox_broker import UpstoxBroker
from trading.angelone_broker import AngelOneBroker
from trading.failover_broker import FailoverBroker
from data.upstox_market_data import UpstoxMarketDataClient


class TestUpstoxFailoverIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_upstox = UpstoxBroker(account_id="7JATYL", timeout=5.0)
        self.mock_angel = AngelOneBroker()
        self.failover = FailoverBroker(
            primary=self.mock_upstox,
            fallback=self.mock_angel,
            failover_threshold_seconds=30.0,
        )

    def test_failover_broker_primary_routing(self):
        """When Upstox is healthy, orders route to Upstox."""
        self.assertFalse(self.failover.is_failover_active)
        self.assertEqual(self.failover.active_broker, self.mock_upstox)

        status = self.failover.get_failover_status()
        self.assertEqual(status["primary"], "UPSTOX")
        self.assertEqual(status["fallback"], "ANGELONE")
        self.assertEqual(status["active_broker"], "UPSTOX")
        self.assertFalse(status["failover_active"])

    def test_failover_broker_triggers_after_30s(self):
        """When Upstox failure duration exceeds 30 seconds, failover activates."""
        # Simulate Upstox failure 35 seconds ago
        self.mock_upstox._last_error_time = time.monotonic() - 35.0
        self.mock_upstox._consecutive_failures = 3

        self.assertTrue(self.failover.is_failover_active)
        self.assertEqual(self.failover.active_broker, self.mock_angel)

        status = self.failover.get_failover_status()
        self.assertEqual(status["active_broker"], "ANGELONE")
        self.assertTrue(status["failover_active"])

        # Simulate Upstox recovery
        self.mock_upstox._record_success()
        self.assertFalse(self.failover.is_failover_active)
        self.assertEqual(self.failover.active_broker, self.mock_upstox)

    def test_upstox_broker_lease_management(self):
        """Verify UpstoxBroker lease protocol compatibility."""
        self.assertTrue(self.mock_upstox.execution_lease_established())
        self.assertTrue(self.failover.execution_lease_established())

    def test_upstox_market_data_option_chain_structure(self):
        """Verify Upstox option chain normalizer formats 25 standard columns."""
        client = UpstoxMarketDataClient()
        raw_mock = {
            "status": "success",
            "data": [
                {
                    "expiry": "2026-11-20",
                    "strike_price": 25200.0,
                    "call_options": {
                        "instrument_key": "NSE_FO|12345",
                        "market_data": {"ltp": 150.5, "volume": 10000, "oi": 50000, "close_price": 140.0},
                        "option_greeks": {"iv": 14.5, "delta": 0.52, "gamma": 0.001, "theta": -12.3, "vega": 18.2},
                    },
                    "put_options": {
                        "instrument_key": "NSE_FO|12346",
                        "market_data": {"ltp": 120.0, "volume": 8000, "oi": 45000, "close_price": 115.0},
                        "option_greeks": {"iv": 15.1, "delta": -0.48, "gamma": 0.001, "theta": -11.8, "vega": 17.9},
                    },
                }
            ]
        }
        with patch.object(client, "get_option_chain", return_value=raw_mock):
            df = client.get_option_chain_df("NIFTY")
            self.assertFalse(df.empty)
            self.assertIn("strike", df.columns)
            self.assertIn("call_ltp", df.columns)
            self.assertIn("put_ltp", df.columns)
            self.assertIn("call_iv", df.columns)
            self.assertIn("call_delta", df.columns)
            self.assertIn("put_delta", df.columns)
            self.assertEqual(df.attrs.get("source"), "upstox")
            self.assertEqual(df.attrs.get("expiry"), "2026-11-20")
            self.assertEqual(df.iloc[0]["call_token"], "NSE_FO|12345")
            self.assertNotIn("auto_certified", df.attrs)


if __name__ == "__main__":
    unittest.main()
