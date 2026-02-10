# execution/mt5_bridge.py
from __future__ import annotations

import os
import uuid
import logging
import requests
from dataclasses import dataclass
from typing import Optional

from config.settings import Config

logger = logging.getLogger(__name__)

# Reuse your existing Order DTO if present, else define a minimal one
try:
    from execution.paper_trader import Order  # your existing DTO
except Exception:
    @dataclass
    class Order:
        symbol: str
        side: str       # "buy" or "sell"
        qty: float      # interpreted as MT5 lots for FX
        price: float    # not used by EA, included for parity
        ts: Optional[object] = None
        
MT5_SUFFIX = os.getenv("MT5_SYMBOL_SUFFIX", "")
def yf_to_mt5(sym: str) -> str:
    s = sym.replace("=X", "")  # strip Yahoo FX suffix
    return f"{s}{MT5_SUFFIX}"



class MT5BridgeBroker:
    """
    Broker adapter that pushes commands to the local HTTP bridge.
    The MT5 Executor EA polls /next and executes in MT5.
    """

    def __init__(self):
        base = os.getenv("BRIDGE_URL", "http://127.0.0.1:5000").rstrip("/")
        self.enqueue_url = f"{base}/enqueue"
        self.event_url   = f"{base}/event"   # not used here; EA posts to it
        self.session = requests.Session()
        logger.info(f"MT5BridgeBroker → {self.enqueue_url}")

    # Account/equity not tracked here (MT5 owns that). Provide stubs:
    def equity(self) -> float:
        return float(getattr(Config, "INITIAL_CAPITAL", 10_000))

    def _buying_power_now(self) -> float:
        return self.equity()

    def get_position(self, symbol: str) -> float:
        # Could be implemented via a request/response pattern later
        return 0.0

    def mark_price(self, symbol: str, price: float) -> None:
        return None

    def _to_line(self, order: Order) -> str:
        """
        Convert order → flat line understood by EA:
          cmd=place;order_ref=<uuid>;symbol=EURUSD;side=BUY;lots=0.10
        """
        side = "BUY" if order.side.lower() == "buy" else "SELL"
        order_ref = f"py-{uuid.uuid4().hex[:8]}"
        lots = float(order.qty)
        mt5_symbol = yf_to_mt5(order.symbol)

        line = (
            f"cmd=place;"
            f"order_ref={order_ref};"
            f"symbol={mt5_symbol};"
            f"side={side};"
            f"lots={lots:.2f}"
        )
        return line

    def submit_order(self, order: Order) -> str:
        """
        Enqueue the order for MT5 EA to pull.
        """
        line = self._to_line(order)
        try:
            r = self.session.post(
                self.enqueue_url, data=line.encode("utf-8"),
                headers={"Content-Type": "text/plain"}, timeout=5
            )
            if r.status_code // 100 == 2:
                logger.info(f"MT5BridgeBroker queued -> {line}")
                return "mt5bridge-queued"
            else:
                logger.error(f"Bridge enqueue failed {r.status_code}: {r.text}")
                return "mt5bridge-error"
        except Exception as e:
            logger.error(f"Bridge enqueue exception: {e}")
            return "mt5bridge-error"