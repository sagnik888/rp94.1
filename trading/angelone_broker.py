"""Angel One SmartAPI order execution and account reconciliation."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import tempfile
import threading
import time
from zoneinfo import ZoneInfo

from loguru import logger
import pyotp
import requests

from trading.account_execution_lease import AccountExecutionLease, lease_owner_description
from trading.broker import Broker
from trading.order_retry import BrokerOrderAmbiguous, BrokerOrderRejected, broker_rejection, is_terminal_rejection
from trading.secure_logging import install_smartapi_log_redaction, redact_sensitive_text

try:
    from SmartApi import SmartConnect
except ImportError:
    SmartConnect = None


class AngelOneBroker(Broker):
    """Translate verified Upstox contracts at the SmartAPI order boundary."""

    _MASTER_CACHE = Path("data_store/cache/angelone_execution_master.json")
    _SUPPORTED_SEGMENTS = {"NIFTY": "NFO", "BANKNIFTY": "NFO", "MIDCPNIFTY": "NFO", "SENSEX": "BFO"}

    def __init__(
        self, api_key="", client_id="", password="", totp_secret="", *, instance_id="UT1",
    ):
        super().__init__()
        from config.settings import get_settings

        settings = get_settings()
        self.api_key = api_key or settings.angelone_api_key
        self.client_id = str(client_id or settings.angelone_client_id).strip().upper()
        self.password = password or settings.angelone_password
        self.totp_secret = totp_secret or settings.angelone_totp_secret
        self._lease = AccountExecutionLease(self.client_id, instance_id) if self.client_id else None
        self._execution_authority = None
        self._lock = threading.RLock()
        self.client = None
        self._auth_time = 0.0
        self._auth_retry_at = 0.0
        self.is_connected = False
        self._master_lock = threading.Lock()
        self._master_day = None
        self._master_retry_at = 0.0
        self._native_contracts = {}
        self._contracts_by_identity = {}

    def execution_lease_established(self):
        return self._lease is not None

    def has_execution_lease(self):
        return bool(self._lease and self._lease.held)

    def acquire_execution_lease(self):
        return bool(self._lease and self._lease.acquire())

    def release_execution_lease(self):
        if self._lease:
            self._lease.release()

    def execution_lease_block_reason(self):
        if not self._lease:
            return "Angel One client ID is required for the execution lease"
        if self._lease.held:
            return ""
        return "Angel One account owned by " + lease_owner_description(self._lease.owner_metadata)

    def set_execution_authority(self, provider):
        self._execution_authority = provider

    def execution_status(self):
        return {"provider": "angelone", "connected": self.is_connected,
                "configured": all((self.api_key, self.client_id, self.password, self.totp_secret)),
                "lease_held": self.has_execution_lease()}

    def _ensure_connected(self):
        with self._lock:
            now = time.monotonic()
            if self.client is not None and self.is_connected and now - self._auth_time < 6 * 3600:
                return True
            if SmartConnect is None or not self.execution_status()["configured"] or now < self._auth_retry_at:
                return False
            self._auth_retry_at = now + 30
            self.is_connected = False
            install_smartapi_log_redaction()
            try:
                smart = SmartConnect(api_key=self.api_key)
                # SDK exceptions may include request headers and login payloads.
                totp = pyotp.TOTP(self.totp_secret.replace(" ", "")).now()
                session = smart.generateSession(self.client_id, self.password, totp) or {}
                payload = session.get("data") or {}
                if session.get("status") is True and str(payload.get("clientcode") or "").upper() == self.client_id:
                    self.client = smart
                    self._auth_time = now
                    self.is_connected = True
                    logger.info("Angel One execution session authenticated")
                else:
                    logger.warning("Angel One execution login refused ({})", str(session.get("errorcode") or "account unverified"))
            except Exception as exc:
                logger.warning("Angel One execution login unavailable ({})", type(exc).__name__)
            return self.is_connected

    @staticmethod
    def _product(value):
        product = str(value or "").upper()
        if product in {"I", "MIS", "INTRADAY"}:
            return "INTRADAY"
        if product in {"D", "NRML", "CARRYFORWARD"}:
            return "CARRYFORWARD"
        return product

    @staticmethod
    def _identity(contract):
        return (contract["exch_seg"], contract["name"], contract["expiry"],
                contract["option_type"], float(contract["strike"]) if contract["option_type"] != "FUT" else 0.0)

    @classmethod
    def _normalize_execution_master(cls, rows):
        groups, native = {}, {}
        for row in rows:
            name = str(row.get("name") or "").upper()
            segment = str(row.get("exch_seg") or "").upper()
            if cls._SUPPORTED_SEGMENTS.get(name) != segment:
                continue
            kind = row.get("instrumenttype")
            symbol = str(row.get("symbol") or "")
            token = str(row.get("token") or "")
            side = "FUT" if kind == "FUTIDX" and symbol.endswith("FUT") else symbol[-2:]
            if not token.isdecimal() or not symbol.startswith(name):
                continue
            if not (side == "FUT" and kind == "FUTIDX" or kind == "OPTIDX" and side in {"CE", "PE"}):
                continue
            try:
                contract = {
                    "name": name, "exch_seg": segment, "option_type": side, "token": token, "symbol": symbol,
                    "expiry": datetime.strptime(row["expiry"], "%d%b%Y").strftime("%d%b%Y").upper(),
                    "strike": float(row["strike"]), "lot_size": int(row["lotsize"]),
                    "tick_size": float(row["tick_size"]) / 100.0,
                }
                if (contract["lot_size"] <= 0 or not math.isfinite(contract["tick_size"])
                        or contract["tick_size"] <= 0
                        or side != "FUT" and not (math.isfinite(contract["strike"]) and contract["strike"] > 0)):
                    continue
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            groups.setdefault(cls._identity(contract), []).append(contract)
            native.setdefault((segment, symbol, token), []).append(contract)
        return ({key: rows[0] for key, rows in groups.items() if len(rows) == 1},
                {key: rows[0] for key, rows in native.items() if len(rows) == 1})

    def prepare_contracts(self):
        """Load SmartAPI order symbols and token IDs for the current trading date."""
        from data.angelone_market_data import ANGELONE_SCRIP_MASTER_URL

        day = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
        with self._master_lock:
            if self._master_day == day:
                return True
            if time.monotonic() < self._master_retry_at:
                return False
            self._master_retry_at = time.monotonic() + 60
            by_identity, by_native = {}, {}
            try:
                saved = json.loads(self._MASTER_CACHE.read_text(encoding="utf-8"))
                if saved.get("date") == day and saved.get("source") == ANGELONE_SCRIP_MASTER_URL:
                    by_identity, by_native = self._normalize_execution_master(saved["contracts"])
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                pass
            if not by_identity or not by_native:
                try:
                    response = requests.get(ANGELONE_SCRIP_MASTER_URL, timeout=18)
                    response.raise_for_status()
                    rows = [
                        row for row in response.json()
                        if isinstance(row, dict)
                        and self._SUPPORTED_SEGMENTS.get(str(row.get("name") or "").upper()) == str(row.get("exch_seg") or "").upper()
                        and row.get("instrumenttype") in {"OPTIDX", "FUTIDX"}
                    ]
                    by_identity, by_native = self._normalize_execution_master(rows)
                except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
                    logger.warning("Angel One order contract directory unavailable ({})", type(exc).__name__)
                    return False
                if not by_identity or not by_native:
                    return False
                self._write_execution_master_cache(day, ANGELONE_SCRIP_MASTER_URL, rows)
            self._contracts_by_identity = by_identity
            self._native_contracts = by_native
            self._master_day = day
            return True

    def _write_execution_master_cache(self, day, source, rows):
        path = self._MASTER_CACHE
        staged = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
                staged = Path(output.name)
                json.dump({"date": day, "source": source, "contracts": rows}, output, separators=(",", ":"))
            staged.replace(path)
        except (OSError, TypeError, ValueError):
            logger.warning("Angel One order contract directory cache could not be saved")
        finally:
            if staged is not None:
                try:
                    staged.unlink(missing_ok=True)
                except OSError:
                    logger.debug("Could not remove temporary Angel One order contract directory")

    def _resolve_contract(self, symbol, key):
        if not self.prepare_contracts():
            return None
        from engine.expiry_manager import expiry_manager

        token = str(key or "")
        if "|" in token:
            upstox = expiry_manager.upstox_contract(token, require_current=True)
            if not upstox or upstox["symbol"] != symbol:
                return None
            contract = self._contracts_by_identity.get(self._identity(upstox))
            if contract and contract["lot_size"] == upstox["lotsize"]:
                return contract
            return None
        # A broker-owned legacy position retains the exact native symbol and token.
        matches = [row for (segment, name, native), row in self._native_contracts.items()
                   if token == native and symbol == name]
        return matches[0] if len(matches) == 1 else None

    def _canonical_contract(self, symbol, token, exchange, frame):
        if not self._native_contracts:
            return symbol, token
        native = self._native_contracts.get((exchange, symbol, token))
        if native is None or frame is None:
            return symbol, token
        segment, name, expiry, side, strike = self._identity(native)
        matches = frame[(frame["exch_seg"] == segment) & (frame["name"] == name)
                        & (frame["expiry"] == expiry) & (frame["option_type"] == side)
                        & (frame["lotsize"] == native["lot_size"])]
        if side != "FUT":
            matches = matches[matches["strike"] == strike]
        if len(matches) != 1:
            return symbol, token
        row = matches.iloc[0]
        return str(row["symbol"]), str(row["token"])

    def _read(self, operation):
        if not self._ensure_connected():
            return None
        try:
            response = getattr(self.client, operation)() or {}
            if response.get("status") is True:
                return response
            if (response.get("errorcode") or response.get("errorCode")) in {"AG8001", "AG8002", "AG8003", "AB1010"}:
                self.is_connected = False
        except Exception as exc:
            logger.warning("Angel One {} unavailable ({})", operation, type(exc).__name__)
        return None

    def ready_for_execution(self):
        owned = self.has_execution_lease()
        if not owned and not self.acquire_execution_lease():
            return False
        try:
            return self.get_positions() is not None and self.prepare_contracts()
        finally:
            if not owned:
                self.release_execution_lease()

    def available_funds(self):
        if not self.is_connected:
            return None
        response = self._read("rmsLimit")
        try:
            value = float((response or {})["data"]["availablecash"])
            return value if math.isfinite(value) and value >= 0 else None
        except (KeyError, ValueError, TypeError):
            return None

    def _authorize_mutation(self):
        if not callable(self._execution_authority) or not self._execution_authority():
            raise BrokerOrderRejected("Angel One order mutation requires REAL mode", terminal=True)
        if not self.has_execution_lease():
            raise BrokerOrderRejected(self.execution_lease_block_reason(), terminal=True)
        if not self._ensure_connected():
            raise BrokerOrderRejected("Angel One execution session unavailable", terminal=True)

    @staticmethod
    def order_tag(trade_id, leg="ENTRY", attempt=0):
        key = f"{trade_id}|{leg}|{attempt}".encode()
        return "UT" + hashlib.blake2s(key, digest_size=8).hexdigest()

    @staticmethod
    def _ticks(value, tick, *, round_up):
        ticks = value / tick
        return math.ceil(ticks - 1e-9) if round_up else math.floor(ticks + 1e-9)

    def _order_parameters(self, symbol, token, qty, side, product_type, *, price=0.0, trigger=None, tag=""):
        contract = self._resolve_contract(symbol, token)
        if contract is None:
            raise BrokerOrderRejected("Exact Angel One order contract unavailable", terminal=True)
        qty, price, side = int(qty), float(price), str(side).upper()
        if qty <= 0 or qty % contract["lot_size"] or side not in {"BUY", "SELL"} or not math.isfinite(price) or price < 0:
            raise BrokerOrderRejected("Invalid Angel One order side, price, or lot quantity", terminal=True)
        product = self._product(product_type)
        if product not in {"CARRYFORWARD", "INTRADAY"}:
            raise BrokerOrderRejected("Unsupported Angel One derivative product", terminal=True)
        tick = contract["tick_size"]
        order = {
            "variety": "NORMAL", "tradingsymbol": contract["symbol"], "symboltoken": contract["token"],
            "transactiontype": side, "exchange": contract["exch_seg"], "producttype": product,
            "ordertype": "LIMIT" if price else "MARKET", "duration": "DAY",
            "price": str(round(self._ticks(price, tick, round_up=side == "SELL") * tick, 4)) if price else "0",
            "quantity": str(qty),
        }
        if tag:
            order["ordertag"] = str(tag)[:19]
        if trigger is not None:
            trigger = float(trigger)
            if not math.isfinite(trigger) or trigger <= tick:
                raise BrokerOrderRejected("Invalid Angel One protective trigger", terminal=True)
            trigger_ticks = self._ticks(trigger, tick, round_up=side == "SELL")
            buffer_pct = .05 if trigger < 10 else .03 if trigger <= 100 else .02 if trigger <= 500 else .01
            gap = max(1, math.ceil(trigger_ticks * buffer_pct))
            limit_ticks = max(1, trigger_ticks - gap) if side == "SELL" else trigger_ticks + gap
            order.update(variety="STOPLOSS", ordertype="STOPLOSS_LIMIT",
                         triggerprice=str(round(trigger_ticks * tick, 4)),
                         price=str(round(limit_ticks * tick, 4)))
        return order

    def _submit(self, order):
        self._authorize_mutation()
        try:
            result = self.client.placeOrderFullResponse(order)
            if isinstance(result, dict):
                order_id = str((result.get("data") or {}).get("orderid") or "")
                if result.get("status") is True and order_id:
                    return order_id
                if not order_id:
                    reason = redact_sensitive_text(broker_rejection(result))
                    code = str(result.get("errorcode") or result.get("errorCode") or "")
                    if is_terminal_rejection(reason) or code in {"AB1008", "AB1018"}:
                        raise BrokerOrderRejected(reason, code=code, terminal=True)
        except BrokerOrderRejected:
            raise
        except Exception as exc:
            raise BrokerOrderAmbiguous(f"Angel One order acknowledgement unavailable ({type(exc).__name__})") from exc
        raise BrokerOrderAmbiguous("Angel One order acknowledgement missing")

    def place_order(self, symbol, token, qty, side, product_type="INTRADAY", price=0.0, *, tag="", **kwargs):
        self._authorize_mutation()
        return self._submit(self._order_parameters(symbol, token, qty, side, product_type, price=price, tag=tag))

    def close_order(self, symbol, token, qty, side, product_type="INTRADAY", price=0.0, *, tag="", **kwargs):
        return self.place_order(symbol, token, qty, side, product_type, price, tag=tag)

    def place_protective_stop(self, symbol, token, qty, side, trigger_price, product_type="INTRADAY", *, tag="", **kwargs):
        self._authorize_mutation()
        return self._submit(self._order_parameters(symbol, token, qty, side, product_type, trigger=trigger_price, tag=tag))

    def modify_protective_stop(self, order_id, symbol, token, qty, side, trigger_price, product_type="INTRADAY", **kwargs):
        self._authorize_mutation()
        order = self._order_parameters(symbol, token, qty, side, product_type, trigger=trigger_price)
        order["orderid"] = order_id
        self._authorize_mutation()
        try:
            response = self.client.modifyOrder(order) or {}
            return response.get("status") is True
        except Exception:
            return False

    def cancel_order(self, order_id, variety="NORMAL"):
        self._authorize_mutation()
        try:
            return (self.client.cancelOrder(order_id, variety) or {}).get("status") is True
        except Exception:
            return False

    def cancel_order_confirmed(self, order_id, variety="NORMAL", timeout_seconds=2.0, poll_seconds=.2):
        self.cancel_order(order_id, variety)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.get_order_status(order_id).get("orderstatus") in {"cancelled", "canceled", "complete", "rejected"}:
                return True
            time.sleep(poll_seconds)
        return False

    def get_positions(self):
        response = self._read("position")
        if response is None or "data" not in response:
            return None
        rows = response["data"]
        if rows is None:
            return {"data": []}
        if not isinstance(rows, list) or rows and not self.prepare_contracts():
            return None

        from engine.expiry_manager import expiry_manager

        result = []
        frame = expiry_manager.df
        for raw in rows:
            if not isinstance(raw, dict) or not all(k in raw for k in ("tradingsymbol", "producttype", "netqty")):
                return None
            try:
                quantity = int(raw["netqty"])
            except (TypeError, ValueError):
                return None
            symbol = str(raw["tradingsymbol"])
            token = str(raw.get("symboltoken") or "")
            exchange = str(raw.get("exchange") or "").upper()
            canonical_symbol, key = self._canonical_contract(symbol, token, exchange, frame)
            if (quantity and exchange in {"NFO", "BFO"} and symbol.startswith(("NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY"))
                    and "|" not in key):
                # Until the masters agree, a same-contract position may simply have
                # different names. Reject the snapshot instead of asserting flatness.
                return None
            result.append(dict(raw, tradingsymbol=canonical_symbol, symboltoken=key, netqty=quantity,
                               producttype=self._product(raw["producttype"]),
                               execution_symbol=symbol, execution_token=token))
        return {"data": result}

    def _order_book(self):
        response = self._read("orderBook")
        if response is None or "data" not in response:
            return None
        data = response["data"]
        return [] if data is None else data if isinstance(data, list) else None

    def get_order_status(self, order_id):
        for row in self._order_book() or []:
            if str(row.get("orderid") or "") == str(order_id):
                state = str(row.get("orderstatus") or row.get("status") or "UNKNOWN").lower()
                return dict(row, status=state, orderstatus=state, order_id=str(order_id),
                            average_price=row.get("averageprice") or 0.0,
                            filled_quantity=row.get("filledshares") or 0)
        return {"status": "UNKNOWN", "orderstatus": "UNKNOWN"}

    def inspect_order_by_tag(self, tag):
        rows = self._order_book()
        if rows is None or not tag:
            return {"state": "UNKNOWN"}
        matches = [r for r in rows if r.get("ordertag") == tag]
        if not matches:
            return {"state": "ABSENT"}
        if len(matches) != 1 or not matches[0].get("orderid"):
            return {"state": "AMBIGUOUS", "matches": matches}
        row = matches[0]
        return {"state": "FOUND", "order_id": row["orderid"], "order": row,
                "status": row.get("orderstatus") or row.get("status") or "", "matches": matches}

    def inspect_recent_entry_order(self, symbol, token, side, quantity=0, **kwargs):
        contract = self._resolve_contract(symbol, token)
        rows = self._order_book() if contract else None
        if rows is None:
            return {"state": "UNKNOWN"}
        candidates = [row for row in rows if row.get("tradingsymbol") == contract["symbol"]
                      and str(row.get("symboltoken")) == contract["token"]
                      and row.get("exchange") == contract["exch_seg"]
                      and row.get("transactiontype") == side.upper()
                      and row.get("variety") == "NORMAL"
                      and str(row.get("orderstatus") or row.get("status") or "").lower() != "rejected"]
        return {"state": "AMBIGUOUS" if candidates else "ABSENT"}

    def inspect_working_protective_stop(self, symbol, token, side, quantity=0, product_type=None, trigger_price=None):
        contract = self._resolve_contract(symbol, token)
        rows = self._order_book() if contract else None
        if rows is None:
            return {"state": "UNKNOWN"}
        matches = [r for r in rows if r.get("tradingsymbol") == contract["symbol"]
                   and str(r.get("symboltoken")) == contract["token"] and r.get("exchange") == contract["exch_seg"]
                   and r.get("transactiontype") == side.upper() and r.get("variety") == "STOPLOSS"
                   and str(r.get("orderstatus") or r.get("status") or "").lower() in {"open", "trigger pending"}]
        if not matches:
            return {"state": "NONE"}
        if len(matches) != 1:
            return {"state": "AMBIGUOUS"}
        row = matches[0]
        try:
            trigger = float(row["triggerprice"])
            if (not math.isfinite(trigger) or trigger <= 0 or not row.get("orderid")
                    or quantity <= 0 or int(row["quantity"]) != quantity or int(row.get("filledshares") or 0)
                    or product_type and self._product(row.get("producttype")) != self._product(product_type)):
                return {"state": "CONFLICT"}
        except (ValueError, TypeError, KeyError):
            return {"state": "CONFLICT"}
        desired = float(trigger_price or 0)
        state = "FOUND_STALE_TRIGGER" if desired and abs(trigger - desired) > contract["tick_size"] else "FOUND"
        return {"state": state, "order_id": row["orderid"], "trigger_price": trigger}
