# execution/paper_trader.py

from __future__ import annotations

import os
import logging
import json
import zmq
from dataclasses import dataclass
from typing import Optional, Any

from config.settings import Config

logger = logging.getLogger(__name__)

# --- Try to import Alpaca Trading + Market Data. Fall back to PaperBroker if not available.
HAVE_ALPACA = False
try:
    # Trading (orders & account)
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.common.exceptions import APIError

    # Market data (latest trade)
    from alpaca.data.historical import (
        StockHistoricalDataClient,
        CryptoHistoricalDataClient,
    )
    from alpaca.data.requests import (
        StockLatestTradeRequest,
        CryptoLatestTradeRequest,
    )
    HAVE_ALPACA = True
except Exception:
    HAVE_ALPACA = False   # we will log if user asked for Alpaca

try:
    from execution.mt5_bridge import MT5BridgeBroker
    HAVE_MT5_BRIDGE = True
except Exception:
    HAVE_MT5_BRIDGE = False
    
# --------------------------
# Order DTO used by live_runner
# --------------------------
@dataclass
class Order:
    symbol: str            # "AAPL", "MSFT", "BTC/USD" (for crypto), etc.
    side: str              # "buy" | "sell"
    qty: float             # positive units (shares/coins)
    price: float           # upstream/fallback price used for sizing and paper fills
    ts: Optional[Any] = None   # caller may pass pd.Timestamp.utcnow(); we keep it generic


# --------------------------
# PaperBroker (in-process)
# --------------------------
class PaperBroker:
    """
    Minimal in-process broker:
      - equity & positions tracking
      - clamps order notional to a configurable portion of 'buying power' (here BP = equity)
      - simple fills at provided price (no slippage, no mark-to-market PnL)
    """

    def __init__(self, starting_equity: float = Config.INITIAL_CAPITAL):
        self._equity = float(starting_equity)
        self._positions = {}  # symbol -> qty
        logger.info("Using in-process PaperBroker.")

    # --- account state ---
    def equity(self) -> float:
        return float(self._equity)

    def _buying_power_now(self) -> float:
        # For paper mode, use equity as BP.
        return float(self._equity)

    def get_position(self, symbol: str) -> float:
        return float(self._positions.get(symbol, 0.0))

    def mark_price(self, symbol: str, price: float) -> None:
        # No-op in this simple paper broker.
        return None

    # --- order flow ---
    def submit_order(self, order: Order) -> str:
        px = float(order.price)
        if px <= 0:
            logger.error(f"PaperBroker: invalid price for {order.symbol}: {px}")
            return "paper-skip-bad-price"

        # Build the maximum allowed notional using the same config rules as live
        bp = self._buying_power_now()
        eq = self.equity()
        max_notional = _effective_max_notional(bp, eq)

        requested_notional = abs(order.qty) * px
        qty_final = float(order.qty)

        if requested_notional > max_notional:
            qty_final = max_notional / px
            logger.warning(
                f"PaperBroker: clamp for {order.symbol}. "
                f"requested={requested_notional:.2f} > max={max_notional:.2f}. "
                f"qty {order.qty:.4f} -> {qty_final:.4f}"
            )
            if qty_final <= 0:
                return "paper-skip-qty-0"

        # Instant "fill"
        signed = qty_final if order.side.lower() == "buy" else -qty_final
        self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + signed

        # Naive cash impact (no unrealized PnL)
        notional = qty_final * px
        if order.side.lower() == "buy":
            self._equity -= notional
        else:
            self._equity += notional

        oid = f"paper-{order.symbol}"
        logger.info(f"Paper fill {order.side} {qty_final:.4f} {order.symbol} @ ~{px:.4f}")
        return oid


# --------------------------
# AlpacaBroker (alpaca-py v2)
# --------------------------
def _is_paper_base_url(url: str) -> bool:
    u = (url or "").lower().strip()
    return "paper-api.alpaca.markets" in u


class AlpacaBroker:
    """
    Alpaca live/paper broker:
      - Uses TradingClient for orders/account
      - Uses Historical Data clients for latest trade price
      - Clamps order notional based on .env caps (fraction of BP, absolute caps, per-trade caps)
      - Equities: integer shares + GTC; Crypto: fractional + GTC
      - Falls back to upstream order.price if MD lookup fails
    """

    def __init__(self):
        if not HAVE_ALPACA:
            raise RuntimeError("alpaca-py is not installed. Try: pip install alpaca-py")

        key = (Config.ALPACA_API_KEY or "").strip()
        sec = (Config.ALPACA_SECRET_KEY or "").strip()
        if not key or not sec:
            raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")

        paper_flag = True if _is_paper_base_url(getattr(Config, "ALPACA_BASE_URL", "")) else False
        self.client = TradingClient(api_key=key, secret_key=sec, paper=paper_flag)

        # Market data clients
        self.stocks_md = StockHistoricalDataClient(api_key=key, secret_key=sec)
        self.crypto_md = CryptoHistoricalDataClient(api_key=key, secret_key=sec)

        logger.info("Using Alpaca (alpaca-py) Trading API v2.")

    # --- account state ---
    def equity(self) -> float:
        acct = self.client.get_account()
        return float(acct.equity)

    def _buying_power_now(self) -> float:
        acct = self.client.get_account()
        return float(acct.buying_power)

    def get_position(self, symbol: str) -> float:
        try:
            p = self.client.get_open_position(symbol)
            return float(p.qty)
        except Exception:
            return 0.0

    def mark_price(self, symbol: str, price: float) -> None:
        # Not required; Alpaca keeps account/equity server-side.
        return None

    # --- helpers ---
    def _asset_info(self, symbol: str):
        try:
            asset = self.client.get_asset(symbol)
            asset_class = str(asset.asset_class).lower()  # 'us_equity', 'crypto', etc.
            shortable = bool(getattr(asset, "shortable", False))
            tradable = bool(getattr(asset, "tradable", False))
            return asset_class, shortable, tradable
        except Exception:
            return None, False, False

    def _latest_price(self, symbol: str, fallback: float) -> float:
        """
        Latest trade via market data clients; fallback to provided price if needed.
        """
        asset_class, _, tradable = self._asset_info(symbol)
        if not tradable:
            logger.error(f"Alpaca: {symbol} not tradable or not found. Using fallback price.")
            return float(fallback)

        try:
            if asset_class == "crypto":
                req = CryptoLatestTradeRequest(symbol_or_symbols=symbol)
                out = self.crypto_md.get_crypto_latest_trade(req)
                price = getattr(out.get(symbol), "price", None) if isinstance(out, dict) else getattr(out, "price", None)
                return float(price) if price is not None else float(fallback)
            else:
                req = StockLatestTradeRequest(symbol_or_symbols=symbol)
                out = self.stocks_md.get_stock_latest_trade(req)
                price = getattr(out.get(symbol), "price", None) if isinstance(out, dict) else getattr(out, "price", None)
                return float(price) if price is not None else float(fallback)
        except Exception as e:
            logger.warning(f"Alpaca MD latest trade failed for {symbol}: {e}. Using fallback price.")
            return float(fallback)

    # --- order flow ---
    def submit_order(self, order: Order) -> str:
        # Price to size with
        px = self._latest_price(order.symbol, fallback=float(order.price))
        if px <= 0:
            logger.error(f"Alpaca: bad price {px} for {order.symbol}")
            return "alpaca-skip-bad-price"

        # Re-check BP and equity just-in-time
        bp = self._buying_power_now()
        eq = self.equity()

        # Determine the maximum allowed notional from config & broker state
        max_notional = _effective_max_notional(bp, eq)

        requested_notional = abs(order.qty) * px
        qty = float(order.qty)

        if requested_notional > max_notional:
            qty = max_notional / px
            logger.warning(
                f"Alpaca: clamp for {order.symbol}. "
                f"requested={requested_notional:.2f} > max={max_notional:.2f}. "
                f"qty {order.qty:.4f} -> {qty:.4f}"
            )

        # Asset rules
        asset_class, shortable, tradable = self._asset_info(order.symbol)
        if not tradable:
            logger.error(f"Alpaca: {order.symbol} not tradable. Skipping.")
            return "alpaca-skip-not-tradable"

        # Equities → whole shares, GTC; Crypto → fractional ok, GTC
        if asset_class != "crypto":
            qty_final = int(qty)
            if qty_final < 1:
                logger.info(f"Alpaca: qty < 1 share after clamp for {order.symbol}; skipping.")
                return "alpaca-skip-tiny"
            tif = TimeInForce.GTC
        else:
            qty_final = float(qty)
            tif = TimeInForce.GTC

        # Shortability check for equities
        if asset_class != "crypto" and order.side.lower() == "sell" and not shortable:
            logger.info(f"Alpaca: {order.symbol} not shortable. Skipping short.")
            return "alpaca-skip-not-shortable"

        side_enum = OrderSide.BUY if order.side.lower() == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=order.symbol,
            qty=qty_final,
            side=side_enum,
            time_in_force=tif,
        )

        try:
            o = self.client.submit_order(req)
            logger.info(f"Live executed {order.side} {qty_final} {order.symbol} @ ~{px:.4f}")
            return str(o.id)
        except APIError as e:
            logger.error(f"Alpaca APIError: {e}")
            return "alpaca-error"


# --------------------------
# Helper: compute effective max notional from config
# --------------------------
def _effective_max_notional(buying_power: float, equity: float) -> float:
    """
    Combine the configured caps to determine the maximum notional allowed
    for a single order *right now*.

    Priority: we take the MIN of all active caps so the most conservative wins.
    If none are set, default to 95% of broker BP.
    """
    limits = []

    # 1) Fraction of broker BP (default 0.95 if unset)
    frac = getattr(Config, "MAX_LIVE_CAPITAL_FRACTION", 0.95)
    try:
        if frac and float(frac) > 0:
            limits.append(float(frac) * float(buying_power))
    except Exception:
        pass

    # 2) Absolute live capital cap ($)
    cap_abs = getattr(Config, "MAX_LIVE_CAPITAL", None)
    try:
        if cap_abs and float(cap_abs) > 0:
            limits.append(float(cap_abs))
    except Exception:
        pass

    # 3) Per-trade absolute cap ($)
    per_trade_abs = getattr(Config, "PER_TRADE_NOTIONAL_CAP", None)
    try:
        if per_trade_abs and float(per_trade_abs) > 0:
            limits.append(float(per_trade_abs))
    except Exception:
        pass

    # 4) Per-trade fraction of equity
    per_trade_frac = getattr(Config, "PER_TRADE_EQUITY_FRACTION", None)
    try:
        if per_trade_frac and float(per_trade_frac) > 0:
            limits.append(float(per_trade_frac) * float(equity))
    except Exception:
        pass

    if not limits:
        return 0.95 * float(buying_power)
    return min(limits)

class MT5BridgeBroker:
    """
    Sends trade instructions to an MQL5 'Executor EA' via ZeroMQ PUSH socket.
    The EA handles all MT5-specific execution, SL/TP, and confirmations.
    """

    def __init__(self):
        self.host = getattr(Config, "MT5_BRIDGE_HOST", "127.0.0.1")
        self.port = int(getattr(Config, "MT5_BRIDGE_PORT", 5555))
        ctx = zmq.Context()
        self.socket = ctx.socket(zmq.PUSH)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        logger.info(f"Using MT5BridgeBroker via ZeroMQ at tcp://{self.host}:{self.port}")

    def equity(self) -> float:
        # No equity tracking here; MT5 EA can report back if we implement pull.
        return 0.0

    def _buying_power_now(self) -> float:
        return 0.0  # Not tracked here; all sizing is handled in Python before sending.

    def get_position(self, symbol: str) -> float:
        return 0.0  # Could be implemented via a request/response pattern.

    def mark_price(self, symbol: str, price: float) -> None:
        return None

    def submit_order(self, order: Order) -> str:
        try:
            msg = {
                "symbol": order.symbol,
                "action": order.side.upper(),
                "lot": order.qty,
                "price": order.price,
            }
            self.socket.send_json(msg)
            logger.info(f"MT5BridgeBroker: sent order -> {msg}")
            return "mt5bridge-sent"
        except Exception as e:
            logger.error(f"MT5BridgeBroker send error: {e}")
            return "mt5bridge-error"
        
        
# --------------------------
# Broker chooser (imported by live_runner)
# --------------------------
def choose_broker():
    """
    Choose broker: Alpaca, MT5 bridge, or Paper.
    """
    broker_choice = str(getattr(Config, "BROKER", "")).lower()

    if broker_choice == "mt5_bridge":
        if not HAVE_MT5_BRIDGE:
            logger.error("BROKER=mt5_bridge but MT5 bridge not importable. Falling back to PaperBroker.")
            return PaperBroker(starting_equity=Config.INITIAL_CAPITAL)
        return MT5BridgeBroker()

    use_alpaca = broker_choice == "alpaca" or (
        broker_choice == "" and str(getattr(Config, "USE_ALPACA", "false")).lower() in {"1", "true", "yes"}
    )
    if use_alpaca:
        if not HAVE_ALPACA:
            logger.error("USE_ALPACA=true but alpaca-py not installed; falling back to PaperBroker.")
            return PaperBroker(starting_equity=Config.INITIAL_CAPITAL)
        try:
            return AlpacaBroker()
        except Exception as e:
            logger.error(f"Failed to initialize AlpacaBroker: {e}. Falling back to PaperBroker.")
            return PaperBroker(starting_equity=Config.INITIAL_CAPITAL)

    return PaperBroker(starting_equity=Config.INITIAL_CAPITAL)