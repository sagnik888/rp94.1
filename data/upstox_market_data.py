"""Upstox Market Data Adapter for UT1.

Provides real-time index quotes, option chain with live Greeks,
historical candles, and failure tracking for the 30-second Angel One fallback.
"""
from __future__ import annotations

import math
import threading
import time
import urllib.parse
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
from loguru import logger

from data.upstox_auth import get_upstox_token

# Index key mappings in Upstox
UPSTOX_INDEX_KEYS = {
    "NIFTY": "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
    "MIDCPNIFTY": "NSE_INDEX|NIFTY MID SELECT",
    "SENSEX": "BSE_INDEX|SENSEX",
    "INDIAVIX": "NSE_INDEX|India VIX",
    "INDIA VIX": "NSE_INDEX|India VIX",
}

# Reverse mapping
UPSTOX_KEY_TO_INDEX = {v: k for k, v in UPSTOX_INDEX_KEYS.items()}

UPSTOX_BASE_URL = "https://api.upstox.com"
UPSTOX_LTP_URL = f"{UPSTOX_BASE_URL}/v2/market-quote/ltp"
UPSTOX_QUOTES_URL = f"{UPSTOX_BASE_URL}/v3/market-quote/quotes"
UPSTOX_OPTION_CONTRACT_URL = f"{UPSTOX_BASE_URL}/v2/option/contract"
UPSTOX_OPTION_CHAIN_URL = f"{UPSTOX_BASE_URL}/v2/option/chain"
UPSTOX_INTRADAY_CANDLE_URL = f"{UPSTOX_BASE_URL}/v3/historical-candle/intraday/{{key}}/{{unit}}/{{interval}}"
UPSTOX_HIST_CANDLE_URL = f"{UPSTOX_BASE_URL}/v3/historical-candle/{{key}}/{{unit}}/{{interval}}/{{to}}/{{frm}}"


class UpstoxMarketDataClient:
    """Primary market data client using Upstox API v2 / v3."""

    def __init__(self, timeout: float = 7.0):
        self.timeout = timeout
        self._lock = threading.RLock()
        self._last_success_time = time.monotonic()
        self._last_error_time: Optional[float] = None
        self._consecutive_failures = 0
        self._last_error_reason = ""
        self._contracts_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
        self._chain_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._quote_cache: Dict[str, Dict[str, Any]] = {}

    def _headers(self) -> Dict[str, str]:
        token = get_upstox_token()
        if not token:
            raise ValueError("No Upstox token available in upstox_token.json")
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def _record_success(self) -> None:
        with self._lock:
            self._last_success_time = time.monotonic()
            self._consecutive_failures = 0
            self._last_error_time = None
            self._last_error_reason = ""

    def _record_failure(self, error: Exception | str) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_error_time is None:
                self._last_error_time = now
            self._consecutive_failures += 1
            self._last_error_reason = str(error)

    def reset_health(self) -> None:
        """Manually clear error state upon fresh login."""
        with self._lock:
            self._last_error_time = None
            self._consecutive_failures = 0
            self._last_error_reason = ""

    def is_healthy(self, threshold_seconds: float = 30.0) -> bool:
        """Return True if Upstox has not been continuously failing for > threshold_seconds."""
        with self._lock:
            if self._last_error_time is None:
                return True
            failure_duration = time.monotonic() - self._last_error_time
            # After 30s of failure, allow periodic probe retries every 10s to recover
            if failure_duration >= threshold_seconds and time.monotonic() - getattr(self, "_last_retry_probe", 0) > 10.0:
                self._last_retry_probe = time.monotonic()
                return True
            return failure_duration < threshold_seconds

    def failure_duration(self) -> float:
        with self._lock:
            if self._last_error_time is None:
                return 0.0
            return max(0.0, time.monotonic() - self._last_error_time)

    def get_index_ltp(self, index_name: str) -> Optional[float]:
        """Fetch spot price for NIFTY, BANKNIFTY, MIDCPNIFTY, or SENSEX."""
        key = UPSTOX_INDEX_KEYS.get(index_name.upper())
        if not key:
            return None

        url = f"{UPSTOX_LTP_URL}?instrument_key={urllib.parse.quote(key)}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                for k, v in data.items():
                    price = v.get("last_price")
                    if price is not None:
                        self._record_success()
                        return float(price)
            self._record_failure(f"HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            self._record_failure(e)
            logger.warning(f"Upstox index LTP fetch failed for {index_name}: {e}")
        return None

    def get_quote(self, instrument_key: str) -> Dict[str, Any]:
        """Return a quote for the exact requested key and its provider clock."""
        try:
            response = requests.get(UPSTOX_QUOTES_URL, params={"instrument_key": instrument_key},
                                    headers=self._headers(), timeout=self.timeout)
            response.raise_for_status()
            body = response.json()
            for label, quote in (body.get("data") or {}).items():
                if quote.get("instrument_token", label.replace(":", "|", 1)) != instrument_key:
                    continue
                clock = datetime.fromisoformat(quote["timestamp"])
                if clock.tzinfo is None:
                    return {}
                bid = next(iter((quote.get("depth") or {}).get("buy") or []), {})
                ask = next(iter((quote.get("depth") or {}).get("sell") or []), {})
                ltp = float(quote.get("last_price") or 0)
                if not math.isfinite(ltp) or ltp <= 0:
                    return {}
                close = float(quote.get("prev_close_price") or 0)
                if close <= 0 and quote.get("net_change") is not None:
                    close = ltp - float(quote["net_change"])
                result = {
                    "ltp": ltp, "time": clock.timestamp(), "last_update": clock.timestamp(),
                    "received_at": time.time(), "depth": quote.get("depth"),
                    "source": "upstox_rest",
                    "previous_close": close if math.isfinite(close) and close > 0 else 0.0,
                    "last_trade_at": float(quote.get("last_trade_time") or 0) / 1000,
                }
                for field, value in (("bid", bid.get("price")), ("ask", ask.get("price")),
                                     ("bid_qty", bid.get("quantity")), ("ask_qty", ask.get("quantity")),
                                     ("oi", quote.get("oi")), ("volume", quote.get("volume"))):
                    if value is not None:
                        result[field] = value
                self._record_success()
                return result
        except (ValueError, KeyError, TypeError, AttributeError, requests.RequestException) as exc:
            self._record_failure(exc)
        return {}

    def get_multi_index_quotes(self, indices: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        quotes = {}
        for name in indices or dict.fromkeys(UPSTOX_INDEX_KEYS):
            key = UPSTOX_INDEX_KEYS.get(name.upper())
            if key and (quote := self.get_quote(key)):
                quotes[name.upper()] = {**quote, "symbol": name.upper(), "instrument_key": key,
                                       "timestamp": quote["time"]}
        return quotes

    def get_option_contracts(self, index_name: str) -> List[Dict[str, Any]]:
        """Fetch and cache available option contracts for an index."""
        key = UPSTOX_INDEX_KEYS.get(index_name.upper())
        if not key:
            return []

        now = time.monotonic()
        with self._lock:
            cached = self._contracts_cache.get(index_name)
            if cached and (now - cached[0] < 300.0): # 5-minute cache
                return cached[1]

        url = f"{UPSTOX_OPTION_CONTRACT_URL}?instrument_key={urllib.parse.quote(key)}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
            if resp.status_code == 200:
                contracts = resp.json().get("data", [])
                with self._lock:
                    self._contracts_cache[index_name] = (now, contracts)
                self._record_success()
                return contracts
            self._record_failure(f"HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            self._record_failure(e)
            logger.warning(f"Upstox option contracts fetch failed for {index_name}: {e}")

        cached = self._contracts_cache.get(index_name)
        return cached[1] if cached else []

    def get_active_expiry(self, index_name: str) -> Optional[str]:
        """Get the current active/weekly expiry date for an index (YYYY-MM-DD)."""
        contracts = self.get_option_contracts(index_name)
        if not contracts:
            return None
        today = datetime.now().strftime("%Y-%m-%d")
        valid_expiries = sorted({c["expiry"] for c in contracts if c.get("expiry") and c["expiry"] >= today})
        return valid_expiries[0] if valid_expiries else None

    def get_option_chain(self, index_name: str, expiry_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch the full real-time option chain with Greeks, IV, PCR, volume, OI."""
        key = UPSTOX_INDEX_KEYS.get(index_name.upper())
        if not key:
            return []

        expiry = expiry_date or self.get_active_expiry(index_name)
        if not expiry:
            return []

        cache_key = f"{index_name}_{expiry}"
        now = time.monotonic()
        with self._lock:
            cached = self._chain_cache.get(cache_key)
            if cached and (now - cached[0] < 1.0): # 1-second debounce
                return cached[1]

        url = f"{UPSTOX_OPTION_CHAIN_URL}?instrument_key={urllib.parse.quote(key)}&expiry_date={expiry}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                with self._lock:
                    self._chain_cache[cache_key] = (now, data)
                self._record_success()
                return data
            self._record_failure(f"HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            self._record_failure(e)
            logger.warning(f"Upstox option chain fetch failed for {index_name} ({expiry}): {e}")

        return []

    def get_option_premium_and_greeks(
        self,
        index_name: str,
        strike: float,
        option_type: str,
        expiry_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Find the matching strike in the option chain and return price, Greeks, and liquidity."""
        chain = self.get_option_chain(index_name, expiry_date)
        if not chain:
            return None

        is_call = option_type.upper() in ("CE", "CALL")
        target_strike = round(float(strike), 2)

        # Find closest or exact strike
        best_diff = float("inf")
        best_item = None
        for item in chain:
            s = float(item.get("strike_price") or 0.0)
            diff = abs(s - target_strike)
            if diff < best_diff:
                best_diff = diff
                best_item = item

        if not best_item or best_diff > 50.0:
            return None

        leg_data = best_item.get("call_options" if is_call else "put_options") or {}
        market_data = leg_data.get("market_data") or {}
        greeks = leg_data.get("option_greeks") or {}

        ltp = float(market_data.get("ltp") or 0.0)
        return {
            "instrument_key": leg_data.get("instrument_key", ""),
            "price": ltp,
            "ltp": ltp,
            "bid": float(market_data.get("bid_price") or 0.0),
            "ask": float(market_data.get("ask_price") or 0.0),
            "volume": float(market_data.get("volume") or 0.0),
            "oi": float(market_data.get("oi") or 0.0),
            "prev_oi": float(market_data.get("prev_oi") or 0.0),
            "iv": float(greeks.get("iv") or 0.0),
            "delta": float(greeks.get("delta") or 0.0),
            "gamma": float(greeks.get("gamma") or 0.0),
            "theta": float(greeks.get("theta") or 0.0),
            "vega": float(greeks.get("vega") or 0.0),
            "strike": float(best_item.get("strike_price") or 0.0),
            "spot": float(best_item.get("underlying_spot_price") or 0.0),
            "pcr": float(best_item.get("pcr") or 0.0),
            "source": "upstox_option_chain",
        }

    def get_historical_candles(
        self, index_name: str, interval: str = "5minute", days: int = 5, *, to_date=None,
    ) -> List[List[Any]]:
        """Use dated history for prior sessions and intraday candles for today."""
        key = UPSTOX_INDEX_KEYS.get(index_name.upper(), index_name)
        intervals = {"day": ("days", "1"), "1d": ("days", "1"), "daily": ("days", "1")}
        for minutes in (1, 3, 5, 10, 15, 30, 60):
            for suffix in ("m", "min", "minute"):
                intervals[f"{minutes}{suffix}"] = ("minutes", str(minutes))
        if "|" not in key or interval.lower() not in intervals:
            return []
        unit, count = intervals[interval.lower()]
        quoted_key = urllib.parse.quote(key, safe="")
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        end = to_date.date() if isinstance(to_date, datetime) else to_date or today
        start = end - timedelta(days=max(0, int(days)))
        history_end = min(end, today - timedelta(days=1))
        candles = {}
        urls = []
        window_days = (28 if unit == "minutes" and int(count) <= 15 else
                       85 if unit == "minutes" else 3650)
        while history_end >= start:
            window_start = max(start, history_end - timedelta(days=window_days - 1))
            urls.append(UPSTOX_HIST_CANDLE_URL.format(
                key=quoted_key, unit=unit, interval=count, to=history_end.isoformat(), frm=window_start.isoformat()))
            history_end = window_start - timedelta(days=1)
        if end == today:
            urls.append(UPSTOX_INTRADAY_CANDLE_URL.format(key=quoted_key, unit=unit, interval=count))
        for url in urls:
            try:
                response = requests.get(url, headers=self._headers(), timeout=self.timeout)
                response.raise_for_status()
                for candle in (response.json().get("data") or {}).get("candles", []):
                    if isinstance(candle, list) and len(candle) >= 7:
                        candles[candle[0]] = candle
                self._record_success()
            except (ValueError, requests.RequestException) as exc:
                self._record_failure(exc)
                if "/intraday/" not in url:
                    return []
        return [candles[stamp] for stamp in sorted(candles)]

    def get_option_chain_df(self, index_name: str, expiry_date: Optional[str] = None) -> Optional[Any]:
        """Convert Upstox option chain into standardized pandas DataFrame."""
        import pandas as pd
        raw_chain = self.get_option_chain(index_name, expiry_date)
        if isinstance(raw_chain, dict):
            raw_chain = raw_chain.get("data", [])
        if not raw_chain:
            return None
        rows = []
        for item in raw_chain:
            if expiry_date and item.get("expiry") != expiry_date:
                continue
            strike = float(item.get("strike_price") or 0.0)
            c = item.get("call_options") or {}
            p = item.get("put_options") or {}
            c_mkt = c.get("market_data") or {}
            p_mkt = p.get("market_data") or {}
            c_grk = c.get("option_greeks") or {}
            p_grk = p.get("option_greeks") or {}

            c_oi = float(c_mkt.get("oi") or 0.0)
            c_prev_oi = float(c_mkt.get("prev_oi") or 0.0)
            p_oi = float(p_mkt.get("oi") or 0.0)
            p_prev_oi = float(p_mkt.get("prev_oi") or 0.0)

            rows.append({
                "strike": strike,
                "call_token": str(c.get("instrument_key") or ""),
                "put_token": str(p.get("instrument_key") or ""),
                "call_ltp": float(c_mkt.get("ltp") or 0.0),
                "put_ltp": float(p_mkt.get("ltp") or 0.0),
                "call_oi": c_oi,
                "put_oi": p_oi,
                "call_oi_change": c_oi - c_prev_oi,
                "put_oi_change": p_oi - p_prev_oi,
                "call_volume_available": c_mkt.get("volume") is not None,
                "put_volume_available": p_mkt.get("volume") is not None,
                "call_volume": float(c_mkt.get("volume") or 0.0),
                "put_volume": float(p_mkt.get("volume") or 0.0),
                "call_iv": float(c_grk.get("iv") or 0.0),
                "put_iv": float(p_grk.get("iv") or 0.0),
                "call_delta": float(c_grk.get("delta") or 0.0),
                "put_delta": float(p_grk.get("delta") or 0.0),
                "call_gamma": float(c_grk.get("gamma") or 0.0),
                "put_gamma": float(p_grk.get("gamma") or 0.0),
                "call_theta": float(c_grk.get("theta") or 0.0),
                "put_theta": float(p_grk.get("theta") or 0.0),
                "call_vega": float(c_grk.get("vega") or 0.0),
                "put_vega": float(p_grk.get("vega") or 0.0),
                "call_bid": float(c_mkt.get("bid_price") or 0.0),
                "call_ask": float(c_mkt.get("ask_price") or 0.0),
                "put_bid": float(p_mkt.get("bid_price") or 0.0),
                "put_ask": float(p_mkt.get("ask_price") or 0.0),
                "pcr": float(item.get("pcr") or 0.0),
                "underlying_spot": float(item.get("underlying_spot_price") or 0.0),
            })
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df.sort_values(by="strike", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.attrs["source"] = "upstox"
        df.attrs["provider"] = "upstox"
        df.attrs["expiry"] = expiry_date or str(raw_chain[0].get("expiry") or "")
        return df

