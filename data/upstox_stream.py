"""Upstox V3 stream with a bounded set of recent derivative subscriptions."""
import json
import math
import threading
import time
from urllib.parse import urlparse

import requests
from loguru import logger
from websockets.sync.client import connect

from data.upstox_auth import get_upstox_token
from data.upstox_market_data import UPSTOX_INDEX_KEYS

_AUTHORIZE = "https://api.upstox.com/v3/feed/market-data-feed/authorize"


class UpstoxStream:
    def __init__(self, on_quote, timeout):
        self.on_quote = on_quote
        self.timeout = timeout
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._contracts = {}

    def watch(self, key):
        if not str(key).startswith(("NSE_FO|", "BSE_FO|")):
            return
        with self._lock:
            self._contracts[key] = time.monotonic()
            if len(self._contracts) > 128:
                self._contracts.pop(min(self._contracts, key=self._contracts.get))

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="UpstoxMarketFeed")
            self._thread.start()

    def stop(self, timeout):
        self._stop.set()
        thread = self._thread
        if thread:
            thread.join(timeout)
        return not thread or not thread.is_alive()

    def _run(self):
        from upstox_client.feeder.proto.MarketDataFeedV3_pb2 import FeedResponse

        delay = 2
        while not self._stop.is_set():
            token = get_upstox_token()
            if not token:
                self._stop.wait(2)
                continue
            try:
                response = requests.get(_AUTHORIZE, headers={"Authorization": f"Bearer {token}"},
                                        timeout=self.timeout)
                response.raise_for_status()
                uri = response.json()["data"]["authorized_redirect_uri"]
                target = urlparse(uri)
                if target.scheme != "wss" or not target.hostname or not target.hostname.endswith(".upstox.com"):
                    raise ValueError("Invalid feed host")
                with connect(uri, open_timeout=self.timeout, close_timeout=1) as socket:
                    delay = 2
                    self._read(socket, token, FeedResponse)
            except Exception as exc:
                # An exception can include the single-use URL and must not be logged verbatim.
                logger.warning(f"Upstox market feed disconnected ({type(exc).__name__})")
            self._stop.wait(delay)
            delay = min(delay * 2, 30)

    def _read(self, socket, token, response_type):
        indices = set(UPSTOX_INDEX_KEYS.values())
        subscribed = set()
        next_sync = 0.0
        last_packet = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_sync:
                if token != get_upstox_token():
                    return
                with self._lock:
                    keys = indices | set(self._contracts)
                for method, changed in (("sub", keys - subscribed), ("unsub", subscribed - keys)):
                    if changed:
                        socket.send(json.dumps({"guid": f"{method}-{time.time_ns()}", "method": method,
                                                "data": {"mode": "full", "instrumentKeys": sorted(changed)}}).encode())
                subscribed = keys
                next_sync = now + 1.0
            try:
                packet = socket.recv(timeout=1)
            except TimeoutError:
                if now - last_packet > 45.0:
                    return
                continue
            last_packet = now
            if isinstance(packet, bytes):
                for key, quote in self.decode(response_type.FromString(packet)):
                    if key in subscribed:
                        self.on_quote(key, quote)

    @staticmethod
    def decode(message):
        observed_at = message.currentTs / 1000
        if observed_at <= 0:
            return
        for key, feed in message.feeds.items():
            if not feed.HasField("fullFeed"):
                continue
            full = feed.fullFeed
            market = full.HasField("marketFF")
            if not market and not full.HasField("indexFF"):
                continue
            row = full.marketFF if market else full.indexFF
            if not row.HasField("ltpc") or not math.isfinite(row.ltpc.ltp) or row.ltpc.ltp <= 0:
                continue
            quote = {"ltp": row.ltpc.ltp, "previous_close": row.ltpc.cp, "time": observed_at,
                     "received_at": time.time(), "last_trade_at": row.ltpc.ltt / 1000,
                     "exchange_timestamp": observed_at, "source": "upstox_ws"}
            if market:
                quote.update(oi=row.oi, volume=row.vtt)
                if row.marketLevel.bidAskQuote:
                    touch = row.marketLevel.bidAskQuote[0]
                    quote.update(bid=touch.bidP, ask=touch.askP, bid_qty=touch.bidQ, ask_qty=touch.askQ)
            yield key, quote
