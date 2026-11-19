"""Upstox API v2 Live Broker Execution.

Implements the unified Broker interface with Upstox API v2:
- Order placement (MARKET, LIMIT)
- Protective stops (SL-M, SL)
- Order modification and cancellation with confirmation
- Position book reconciliation normalized to system standards
- Margin preflight checks
- Fail-closed error handling (distinguishing ambiguous vs rejected)
- Session, authority, and execution lease checks.
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from trading.broker import Broker
from trading.order_retry import BrokerOrderAmbiguous, BrokerOrderRejected
from trading.account_execution_lease import AccountExecutionLease, lease_owner_description
from data.upstox_auth import get_upstox_auth_status, get_upstox_token

UPSTOX_BASE_URL = "https://api.upstox.com"
UPSTOX_ORDER_PLACE = "https://api-hft.upstox.com/v2/order/place"
UPSTOX_ORDER_MODIFY = "https://api-hft.upstox.com/v2/order/modify"
UPSTOX_ORDER_CANCEL = "https://api-hft.upstox.com/v2/order/cancel"
UPSTOX_ORDER_BOOK = f"{UPSTOX_BASE_URL}/v2/order/retrieve-all"
UPSTOX_ORDER_DETAILS = f"{UPSTOX_BASE_URL}/v2/order/details"
UPSTOX_POSITIONS = f"{UPSTOX_BASE_URL}/v2/portfolio/short-term-positions"
UPSTOX_FUNDS = f"{UPSTOX_BASE_URL}/v2/user/get-funds-and-margin"

class UpstoxBroker(Broker):
    """Upstox API v2 Live Execution Broker."""

    def __init__(self, account_id: str = "", instance_id: str = "UT1", timeout: float = 7.0):
        super().__init__()
        self.timeout = timeout
        self.account_id = str(account_id or "").strip()
        self._execution_lease = (
            AccountExecutionLease(self.account_id, instance_id)
            if self.account_id
            else None
        )
        self._execution_authority = None
        self._lock = threading.RLock()
        self._last_success_time = time.monotonic()
        self._last_error_time: Optional[float] = None
        self._consecutive_failures = 0
        self._last_error_reason = ""

    def acquire_execution_lease(self) -> bool:
        if self._execution_lease is None:
            return False
        return self._execution_lease.acquire()

    def execution_lease_established(self) -> bool:
        return self._execution_lease is not None

    def release_execution_lease(self) -> None:
        if self._execution_lease is not None:
            self._execution_lease.release()

    def has_execution_lease(self) -> bool:
        lease = self._execution_lease
        return bool(lease and lease.held)

    def execution_lease_block_reason(self) -> str:
        lease = self._execution_lease
        if lease is None:
            return "Upstox account ID is required for the execution lease"
        if lease.held:
            return ""
        return "REAL broker account is owned by " + lease_owner_description(lease.owner_metadata)

    def set_execution_authority(self, provider) -> None:
        self._execution_authority = provider

    def _mutation_headers(self) -> Dict[str, str]:
        if not callable(self._execution_authority) or not self._execution_authority():
            raise BrokerOrderRejected("Upstox order mutation requires REAL mode")
        if not self.has_execution_lease():
            raise BrokerOrderRejected(self.execution_lease_block_reason())
        if get_upstox_auth_status().get("user_id") != self.account_id:
            raise BrokerOrderRejected("Upstox session does not match the execution lease account")
        return self._headers()

    @staticmethod
    def order_tag(trade_id: str, leg: str = "ENTRY", attempt: int = 0) -> str:
        identity = f"{trade_id}|{leg}|{attempt}".encode()
        return "UT" + hashlib.blake2s(identity, digest_size=9).hexdigest()

    @staticmethod
    def _instrument_key(token: str) -> str:
        key = str(token or "")
        segment, separator, exchange_token = key.partition("|")
        if separator and segment in {"NSE_FO", "BSE_FO"} and exchange_token.isdecimal():
            return key
        raise BrokerOrderRejected("Upstox order requires a verified NSE_FO or BSE_FO instrument key")

    @staticmethod
    def _transaction_side(side: str) -> str:
        side = str(side).upper()
        if side not in {"BUY", "SELL"}:
            raise BrokerOrderRejected("Upstox order requires BUY or SELL")
        return side

    @staticmethod
    def _tick_price(instrument_key: str, price: float, side: str) -> float:
        from engine.expiry_manager import expiry_manager

        tick = expiry_manager.upstox_tick_size(instrument_key)
        if tick is None or not math.isfinite(tick) or tick <= 0:
            raise BrokerOrderRejected("Upstox tick size unavailable for exact contract")
        ticks = price / tick
        rounded = math.floor(ticks + 1e-9) if side == "BUY" else math.ceil(ticks - 1e-9)
        return round(max(1, rounded) * tick, 4)

    def _headers(self) -> Dict[str, str]:
        token = get_upstox_token()
        if not token:
            raise BrokerOrderRejected("Upstox access token missing or expired; login required")
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
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

    def is_healthy(self, threshold_seconds: float = 30.0) -> bool:
        """Returns False if Upstox has been continuously failing for > 30s."""
        with self._lock:
            if self._last_error_time is None:
                return True
            return (time.monotonic() - self._last_error_time) < threshold_seconds

    def failure_duration(self) -> float:
        with self._lock:
            if self._last_error_time is None:
                return 0.0
            return max(0.0, time.monotonic() - self._last_error_time)

    def _map_product(self, product_type: str) -> str:
        p = str(product_type).upper()
        if p in ("INTRADAY", "MIS", "I"):
            return "I"
        return "D"

    _product = _map_product

    def place_order(
        self,
        symbol: str,
        token: str,
        qty: int,
        side: str,
        product_type: str = "INTRADAY",
        price: float = 0.0,
        touch_quote: Optional[Dict] = None,
        tag: str = "ut1",
        **kwargs: Any,
    ) -> str:
        """Place an order via Upstox API v2."""
        qty = int(qty)
        if qty <= 0:
            raise BrokerOrderRejected(f"Invalid quantity {qty} for {symbol}")

        instrument_token = self._instrument_key(token)
        price = float(price)
        if not math.isfinite(price) or price < 0.0:
            raise BrokerOrderRejected(f"Invalid order price for {symbol}")
        order_type = "MARKET" if price == 0.0 else "LIMIT"
        transaction_type = self._transaction_side(side)
        product = self._map_product(product_type)

        payload = {
            "quantity": qty,
            "product": product,
            "validity": "DAY",
            "price": self._tick_price(instrument_token, price, transaction_type) if price else 0.0,
            "tag": str(tag or "ut1")[:20],
            "instrument_token": instrument_token,
            "order_type": order_type,
            "transaction_type": transaction_type,
            "disclosed_quantity": 0,
            "trigger_price": 0.0,
            "is_amo": False,
        }

        if order_type == "MARKET":
            payload["market_protection"] = -1
        return self._submit_order(payload, symbol)

    def _submit_order(self, payload: Dict[str, Any], symbol: str) -> str:
        headers = self._mutation_headers()
        try:
            resp = requests.post(UPSTOX_ORDER_PLACE, headers=headers, json=payload, timeout=self.timeout)
        except requests.Timeout as exc:
            self._record_failure(exc)
            # Timeout during placement is ambiguous: the order may have reached the exchange
            raise BrokerOrderAmbiguous(f"Upstox place_order timed out for {symbol}: {exc}") from exc
        except Exception as exc:
            self._record_failure(exc)
            raise BrokerOrderAmbiguous(f"Upstox place_order transport error for {symbol}: {exc}") from exc

        if resp.status_code == 200:
            try:
                order_id = str((resp.json().get("data") or {}).get("order_id") or "")
            except (ValueError, AttributeError):
                order_id = ""
            if order_id:
                self._record_success()
                logger.info(f"Upstox order placed: {symbol} {payload['transaction_type']} {payload['quantity']} -> order_id={order_id}")
                return order_id

        err_body = resp.text[:300]
        self._record_failure(f"HTTP {resp.status_code}: {err_body}")

        # Distinguish definitive reject from ambiguous failure
        if resp.status_code in (400, 422):
            raise BrokerOrderRejected(f"Upstox rejected order for {symbol}: {err_body}")
        raise BrokerOrderAmbiguous(f"Upstox order status uncertain (HTTP {resp.status_code}): {err_body}")

    def close_order(
        self,
        symbol: str,
        token: str,
        qty: int,
        side: str,
        product_type: str = "INTRADAY",
        price: float = 0.0,
        tag: str = "ut1_exit",
        **kwargs: Any,
    ) -> str:
        """Close an existing position by sending an opposing market/limit order."""
        return self.place_order(
            symbol=symbol,
            token=token,
            qty=qty,
            side=side,
            product_type=product_type,
            price=price,
            tag=tag,
            **kwargs,
        )

    def get_positions(self) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """Fetch positions and normalize to the system's expected format."""
        try:
            resp = requests.get(UPSTOX_POSITIONS, headers=self._headers(), timeout=self.timeout)
            if resp.status_code == 200:
                body = resp.json()
                raw_positions = body.get("data")
                if body.get("status") != "success" or not isinstance(raw_positions, list):
                    return None
                rows = []
                for p in raw_positions:
                    net_qty = int(float(p.get("quantity") or 0))
                    rows.append({
                        "tradingsymbol": p.get("trading_symbol", ""),
                        "symboltoken": p.get("instrument_token") or p.get("instrument_key", ""),
                        "buyavgprice": float(p.get("buy_price") or 0.0),
                        "sellavgprice": float(p.get("sell_price") or 0.0),
                        "ltp": float(p.get("last_price") or 0.0),
                        "netqty": net_qty,
                        "buyqty": int(float(p.get("buy_quantity") or 0)),
                        "sellqty": int(float(p.get("sell_quantity") or 0)),
                        "buyprice": float(p.get("buy_price") or 0.0),
                        "sellprice": float(p.get("sell_price") or 0.0),
                        "pnl": float(p.get("pnl") or 0.0),
                        "realised": float(p.get("realised_profit") or 0.0),
                        "unrealised": float(p.get("unrealised_profit") or 0.0),
                        "product": p.get("product", ""),
                        "producttype": "INTRADAY" if p.get("product") == "I" else "CARRYFORWARD",
                    })
                self._record_success()
                return {"data": rows}
            self._record_failure(f"HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as exc:
            self._record_failure(exc)
            logger.warning(f"Upstox get_positions failed: {exc}")
        return None

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Fetch status and fill details for an order."""
        if not order_id:
            return {"status": "UNKNOWN", "orderstatus": "UNKNOWN"}

        url = f"{UPSTOX_ORDER_DETAILS}?order_id={urllib.parse.quote(order_id)}"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
            if resp.status_code == 200:
                latest = resp.json().get("data")
                if isinstance(latest, dict) and str(latest.get("order_id") or "") == order_id:
                    raw_status = str(latest.get("status") or "").upper()
                    # Normalise to standard keys
                    status_map = {
                        "COMPLETE": "complete",
                        "REJECTED": "rejected",
                        "CANCELLED": "cancelled",
                        "TRIGGER PENDING": "trigger pending",
                        "OPEN": "open",
                        "PUT ORDER REQ RECEIVED": "open",
                    }
                    norm_status = status_map.get(raw_status, raw_status.lower())
                    result = {
                        "order_id": order_id,
                        "orderstatus": norm_status,
                        "status": norm_status,
                        "filledshares": int(float(latest.get("filled_quantity") or 0)),
                        "filled_quantity": int(float(latest.get("filled_quantity") or 0)),
                        "price": float(latest.get("average_price") or 0.0),
                        "average_price": float(latest.get("average_price") or 0.0),
                        "trigger_price": float(latest.get("trigger_price") or 0.0),
                        "status_message": str(latest.get("status_message") or ""),
                        "transaction_type": latest.get("transaction_type", ""),
                        "tradingsymbol": latest.get("trading_symbol", ""),
                    }
                    self._record_success()
                    return result
            self._record_failure(f"HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as exc:
            self._record_failure(exc)
            logger.warning(f"Upstox get_order_status failed for {order_id}: {exc}")

        return {"status": "UNKNOWN", "orderstatus": "UNKNOWN"}

    @classmethod
    def _stop_parameters(cls, key: str, symbol: str, side: str, trigger_price: float) -> Dict[str, Any]:
        trigger = float(trigger_price)
        if not math.isfinite(trigger) or trigger <= 0:
            raise BrokerOrderRejected("Upstox protective stop requires a positive trigger")
        # Reuse the exchange-valid stop-limit prices used by the existing risk policy.
        from trading.kite_broker import KiteBroker
        kind = "OPTCE" if any(part in {"CE", "PE"} for part in symbol.upper().split()) else symbol
        params = {"price": 0.0, **KiteBroker._protective_stop_params(kind, side, trigger)}
        params["trigger_price"] = cls._tick_price(key, trigger, side)
        if params["price"]:
            params["price"] = cls._tick_price(key, params["price"], side)
        return params

    def place_protective_stop(
        self, symbol: str, token: str, qty: int, side: str, trigger_price: float,
        product_type: str = "INTRADAY", tag: str = "", **kwargs: Any,
    ) -> Optional[str]:
        if int(qty) <= 0:
            raise BrokerOrderRejected("Upstox protective stop requires positive quantity")
        side = self._transaction_side(side)
        payload = {
            "quantity": int(qty), "product": self._map_product(product_type),
            "validity": "DAY", "tag": tag[:20], "instrument_token": self._instrument_key(token),
            "transaction_type": side, "disclosed_quantity": 0, "is_amo": False,
            **self._stop_parameters(self._instrument_key(token), symbol, side, trigger_price),
        }
        return self._submit_order(payload, symbol)

    def modify_protective_stop(
        self,
        order_id: str,
        symbol: str,
        token: str,
        qty: int,
        side: str,
        trigger_price: float,
        product_type: str = "INTRADAY",
        **kwargs: Any,
    ) -> bool:
        """Modify trigger price / quantity of a working protective stop."""
        if not order_id:
            return False

        payload = {
            "order_id": order_id,
            "quantity": int(qty),
            "validity": "DAY",
            **self._stop_parameters(self._instrument_key(token), symbol, self._transaction_side(side), trigger_price),
            "disclosed_quantity": 0,
        }
        headers = self._mutation_headers()
        try:
            resp = requests.put(UPSTOX_ORDER_MODIFY, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                self._record_success()
                logger.info(f"Upstox stop modified: {order_id} -> trigger={trigger_price}")
                return True
            self._record_failure(f"HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as exc:
            self._record_failure(exc)
            logger.error(f"Upstox modify_protective_stop error for {order_id}: {exc}")
        return False

    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> bool:
        """Cancel an open or trigger-pending order."""
        if not order_id:
            return False
        url = f"{UPSTOX_ORDER_CANCEL}?order_id={urllib.parse.quote(order_id)}"
        headers = self._mutation_headers()
        try:
            resp = requests.delete(url, headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                self._record_success()
                logger.info(f"Upstox order cancelled: {order_id}")
                return True
            self._record_failure(f"HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as exc:
            self._record_failure(exc)
            logger.warning(f"Upstox cancel_order error for {order_id}: {exc}")
        return False

    def cancel_order_confirmed(
        self,
        order_id: str,
        variety: str = "NORMAL",
        timeout_seconds: float = 2.0,
        poll_seconds: float = 0.1,
    ) -> bool:
        """Cancel order and poll until confirmation or timeout."""
        self.cancel_order(order_id, variety=variety)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            st = self.get_order_status(order_id)
            if st.get("orderstatus") in ("cancelled", "rejected", "complete"):
                return True
            time.sleep(poll_seconds)
        return False

    def _order_book(self) -> Optional[list]:
        try:
            response = requests.get(UPSTOX_ORDER_BOOK, headers=self._headers(), timeout=self.timeout)
            response.raise_for_status()
            body = response.json()
            if body.get("status") == "success" and isinstance(body.get("data"), list):
                return body["data"]
        except (requests.RequestException, ValueError, BrokerOrderRejected):
            pass
        return None

    def inspect_order_by_tag(self, tag: str) -> Dict[str, Any]:
        orders = self._order_book() if tag else None
        if orders is None:
            return {"state": "UNKNOWN"}
        matches = [row for row in orders if row.get("tag") == tag]
        if not matches:
            return {"state": "ABSENT"}
        if len(matches) != 1 or not matches[0].get("order_id"):
            return {"state": "AMBIGUOUS", "matches": matches}
        row = matches[0]
        return {"state": "FOUND", "order_id": row["order_id"], "order": row,
                "status": row.get("status", ""), "matches": matches}

    @staticmethod
    def _matches_contract(row: dict, symbol: str, token: str, side: str) -> bool:
        return (bool(token) and token == (row.get("instrument_token") or row.get("instrument_key"))
                and symbol == row.get("trading_symbol")
                and side.upper() == row.get("transaction_type"))

    def find_working_protective_stop(
        self, symbol: str, token: str, side: str, quantity: int = 0,
    ) -> Optional[str]:
        return self.inspect_working_protective_stop(symbol, token, side, quantity).get("order_id")

    def inspect_working_protective_stop(
        self, symbol: str, token: str, side: str, quantity: int,
        product_type: Optional[str] = None, trigger_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        orders = self._order_book()
        if orders is None:
            return {"state": "UNKNOWN"}
        matches = [row for row in orders
                   if self._matches_contract(row, symbol, token, side)
                   and str(row.get("status") or "").lower() in {"open", "trigger pending"}
                   and row.get("order_type") in {"SL", "SL-M"}]
        if not matches:
            return {"state": "NONE"}
        if len(matches) != 1:
            return {"state": "AMBIGUOUS"}
        row = matches[0]
        try:
            trigger = float(row.get("trigger_price") or 0)
            if (not row.get("order_id") or not math.isfinite(trigger) or trigger <= 0
                    or quantity <= 0 or int(row.get("quantity") or 0) != quantity
                    or int(row.get("filled_quantity") or 0) != 0
                    or (product_type and row.get("product") != self._map_product(product_type))):
                return {"state": "CONFLICT"}
        except (ValueError, TypeError):
            return {"state": "CONFLICT"}
        return {"state": "FOUND", "order_id": row["order_id"], "trigger_price": trigger}

    def inspect_recent_entry_order(
        self, symbol: str, token: str, side: str, quantity: int = 0, **kwargs: Any,
    ) -> Dict[str, Any]:
        orders = self._order_book()
        if orders is None:
            return {"state": "UNKNOWN"}
        matches = [row for row in orders
                   if self._matches_contract(row, symbol, token, side)
                   and row.get("order_type") in {"LIMIT", "MARKET"}
                   and str(row.get("status") or "").lower() not in {"rejected", "cancelled"}
                   and (not quantity or row.get("quantity") == quantity)]
        if not matches:
            return {"state": "ABSENT"}
        # A shape match could belong to a different trade. Exact tags may recover it.
        return {"state": "AMBIGUOUS"}

    def preflight_entry_margin(self, **kwargs: Any) -> Dict[str, Any]:
        """Query Upstox available funds and margin."""
        try:
            resp = requests.get(UPSTOX_FUNDS, headers=self._headers(), timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                equity = data.get("equity", {})
                avail = float(equity.get("available_margin") or 0.0)
                used = float(equity.get("used_margin") or 0.0)
                return {
                    "available_margin": avail,
                    "used_margin": used,
                    "status": "OK",
                    "broker": "UPSTOX",
                }
        except Exception as exc:
            logger.warning(f"Upstox preflight_entry_margin failed: {exc}")

        return {
            "available_margin": None,
            "used_margin": 0.0,
            "status": "UNAVAILABLE",
            "broker": "UPSTOX",
        }
