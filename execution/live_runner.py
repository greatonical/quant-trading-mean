# execution/live_runner.py
from __future__ import annotations

import os
import sys
import time
import pandas as pd
import yfinance as yf

# Ensure project root on path
_THIS = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.settings import Config
from utils.logger import get_logger
from strategies.mean_reversion import MeanReversionStrategy
from collections import deque, defaultdict
from datetime import datetime, timedelta, timezone

# Reuse the existing Order DTO and broker selector (Paper/Alpaca fallback)
from execution.paper_trader import Order, choose_broker

log = get_logger(__name__)



def _now_utc():
    return datetime.now(timezone.utc)

class TradeLimiter:
    """
    Simple in-memory limiter:
      - rolling global trades/minute gate
      - per-symbol per-bar max
      - per-symbol cooldown after an order
      - concurrent open-position cap (estimated from target signals)
    """
    def __init__(self):
        self.max_open_positions = int(float(os.getenv("MAX_OPEN_POSITIONS", os.getenv("MAX_POSITIONS", 5))))
        self.max_trades_per_min = int(float(os.getenv("MAX_TRADES_PER_MINUTE", 3)))
        self.max_trades_per_bar = int(float(os.getenv("MAX_TRADES_PER_BAR", 1)))
        self.cooldown_secs      = int(float(os.getenv("ENTRY_COOLDOWN_SECONDS", 90)))
        self.exec_on_close_only = str(os.getenv("EXECUTE_ON_BAR_CLOSE_ONLY", "1")).lower() in {"1","true","yes"}

        # rolling window of timestamps for last 60s trades (global)
        self._recent_trades = deque()   # deque[datetime]
        # per-symbol bar key -> count
        self._bar_counts = defaultdict(int)   # {(symbol, bar_key): count}
        # per-symbol cooldown until timestamp
        self._cooldown_until = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
        # per-symbol last seen completed bar key to help dedupe
        self._last_bar_key = {}  # symbol -> (bar_start_time, bar_close_time)

    def _purge_old(self, now):
        cutoff = now - timedelta(seconds=60)
        while self._recent_trades and self._recent_trades[0] < cutoff:
            self._recent_trades.popleft()

    def bar_key(self, df):
        """
        Return a key that identifies the last *completed* bar (safe for live):
        We'll use the DataFrame index (assumed tz-aware or naive but consistent).
        """
        if len(df) == 1:
            # only one row: treat it as 'completed' for safety
            t = df.index[-1]
            return (str(t),)
        else:
            # use the second-to-last row as completed bar
            t = df.index[-2]
            return (str(t),)

    def can_trade(self, symbol: str, df, open_symbols_count: int) -> tuple[bool, str]:
        now = _now_utc()
        self._purge_old(now)

        # 1) concurrent open-position cap (estimated): if we already have too many signals to be non-zero
        if open_symbols_count >= self.max_open_positions:
            return (False, f"SKIP: MAX_OPEN_POSITIONS reached ({open_symbols_count}/{self.max_open_positions}).")

        # 2) rolling global per-minute cap
        if len(self._recent_trades) >= self.max_trades_per_min:
            return (False, f"SKIP: MAX_TRADES_PER_MINUTE reached ({len(self._recent_trades)}/{self.max_trades_per_min}).")

        # 3) per-symbol cooldown
        if now < self._cooldown_until[symbol]:
            wait = int((self._cooldown_until[symbol] - now).total_seconds())
            return (False, f"SKIP: cooldown active for {symbol} ({wait}s left).")

        # 4) per-bar gating (and "bar close only")
        bk = self.bar_key(df)
        if self.exec_on_close_only:
            # Ensure we only trade once per completed bar up to max_trades_per_bar
            count = self._bar_counts[(symbol, bk)]
            if count >= self.max_trades_per_bar:
                return (False, f"SKIP: MAX_TRADES_PER_BAR for {symbol} on bar {bk}.")
        return (True, "OK")

    def notify_placed(self, symbol: str, df):
        now = _now_utc()
        self._recent_trades.append(now)
        # mark cooldown
        self._cooldown_until[symbol] = now + timedelta(seconds=self.cooldown_secs)
        # increment per-bar count
        bk = self.bar_key(df)
        self._bar_counts[(symbol, bk)] += 1

    def notify_new_bar(self, symbol: str, df):
        """Optional: clear earlier bar count when bar advances."""
        bk = self.bar_key(df)
        last = self._last_bar_key.get(symbol)
        if last != bk:
            # new bar -> nothing to do; counts are per key anyway
            self._last_bar_key[symbol] = bk


def fetch_latest(symbol: str, interval: str, period: str = "60d") -> pd.DataFrame:
    """Pull recent bars for live signals (yfinance intraday max=60d)."""
    t = yf.Ticker(symbol)
    df = t.history(period=period, interval=interval)
    if df is None or df.empty:
        raise RuntimeError(f"No live data for {symbol}")
    return df.dropna()


def pick_broker():
    """
    Honor BROKER env, else fall back to choose_broker().
    BROKER=mt5_bridge -> use the local HTTP bridge to MT5.
    """
    broker_env = os.getenv("BROKER", "").strip().lower()
    if broker_env == "mt5_bridge":
        from execution.mt5_bridge import MT5BridgeBroker
        return MT5BridgeBroker()
    # default: Alpaca or Paper, as implemented previously
    return choose_broker()


# -----------------------------
# FX helpers for MT5 bridge
# -----------------------------
def is_fx_symbol(sym: str) -> bool:
    s = sym.upper()
    return s.endswith("=X") or ("/" in s) or s in {
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
        "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"
    }

MT5_MIN_LOTS = Config.MT5_MIN_LOTS
MT5_MAX_LOTS = Config.MT5_MAX_LOTS
LOTS_PER_100USD = Config.LOTS_PER_100USD

# MT5_MIN_LOTS = float(os.getenv("MT5_MIN_LOTS", "0.01"))
# MT5_MAX_LOTS = float(os.getenv("MT5_MAX_LOTS_PER_TRADE", "0.02"))
# LOTS_PER_100USD = float(os.getenv("MT5_LOTS_PER_100USD", "0.01"))

def clamp_lots(x: float) -> float:
    if x < MT5_MIN_LOTS:
        return MT5_MIN_LOTS
    if x > MT5_MAX_LOTS:
        return MT5_MAX_LOTS
    return x

def dollars_to_lots(notional_usd: float) -> float:
    """
    Intuitive scale: if LOTS_PER_100USD=0.01, then $100 notional -> 0.01 lots.
    """
    lots_raw = (max(notional_usd, 0.0) / 100.0) * LOTS_PER_100USD
    return clamp_lots(lots_raw)


def main():
    # Symbols for live trading
    symbols = [s.strip() for s in str(Config.LIVE_SYMBOLS).split(",") if s.strip()]
    broker = pick_broker()
    broker_name = broker.__class__.__name__

    # Strategy parameters (match your env-simple config)
    strat = MeanReversionStrategy(
        lookback=Config.LOOKBACK,
        z_entry=Config.Z_ENTRY,
        z_exit=Config.Z_EXIT,
        stop_loss_pct=Config.STOP_LOSS,
        take_profit_pct=Config.TAKE_PROFIT,
        sizing_method=Config.SIZING_METHOD,
        atr_period=Config.ATR_PERIOD,
        atr_mult=Config.ATR_MULT,
        use_rsi=bool(int(getattr(Config, "USE_RSI", 1))),  # 1/0 in .env
    )

    poll_secs = int(float(getattr(Config, "LIVE_POLL_SECONDS", 60)))
    limiter = TradeLimiter()
    log.info(f"Starting live loop ({poll_secs}s). Ctrl+C to stop.")

    while True:
        try:
            for sym in symbols:
                # --- data & signal ---
                df = fetch_latest(sym, Config.TIMEFRAME)
                res = strat.generate(df)

                last = res.iloc[-2] if len(res) > 1 else res.iloc[-1]
                z = float(last.get("Z", float("nan")))
                r = float(last.get("RSI", float("nan")))
                sig = int(last.get("position", 0))           # -1/0/+1 signal
                sz  = float(last.get("size", 0.0))           # 0..1 fraction size
                price = float(last["Close"])
                log.info(f"{sym} z={z:.2f} rsi={r:.1f} pos={sig} size={sz:.3f}")

                # --- translate position to order delta ---
                 # --- translate position to order delta ---
                broker.mark_price(sym, price)

                # Count how many symbols currently want a non-zero position (target signals)
                # We approximate "open positions" by desired non-zero target across all symbols in this pass.
                # First compute this symbol's desired before we build the global count:
                eq = float(broker.equity())
                target_dollars = sz * sig * eq
                qty = abs(target_dollars) / max(price, 1e-8)

                equity_like = sym.isalpha()
                if equity_like:
                    qty = float(int(qty))

                desired_qty = qty if sig > 0 else (-qty if sig < 0 else 0.0)

                # Build an estimated open-symbols count for limiter:
                # (We do it simple: count this symbol + whatever we tracked as open last iteration is overkill;
                # so here we derive only with current symbol = 1 if desired non-zero else 0. For stricter control,
                # persist a set of non-zero desireds across the loop.)
                open_symbols_count = 1 if abs(desired_qty) > 0 else 0

                # --- trade limiter gate ---
                ok, reason = limiter.can_trade(sym, df, open_symbols_count)
                if not ok:
                    log.info(f"{sym} {reason} Would have targeted qty={desired_qty:.4f} at {price:.5f}.")
                    continue

                # Quantity logic depends on venue & asset:
                if broker_name == "MT5BridgeBroker" and is_fx_symbol(sym):
                    # FX via MT5: compute *lots* from dollars using the env scale
                    lots = dollars_to_lots(abs(target_dollars))
                    desired_qty = lots if sig > 0 else (-lots if sig < 0 else 0.0)
                else:
                    # Equities/crypto: share/coin-like quantity from dollars
                    qty = abs(target_dollars) / max(price, 1e-8)
                    equity_like = sym.isalpha()
                    if equity_like:
                        qty = float(int(qty))  # whole shares for equities
                        if qty < 1 and sig != 0:
                            log.info(f"{sym} target < 1 share; skipping trade.")
                            continue
                    desired_qty = qty if sig > 0 else (-qty if sig < 0 else 0.0)

                # Position reconciliation (MT5 bridge get_position() is a stub → 0.0 for now)
                current_qty = float(broker.get_position(sym))
                delta = desired_qty - current_qty

                if abs(delta) > 1e-6:
                    side = "buy" if delta > 0 else "sell"
                    ord = Order(
                        symbol=sym,
                        side=side,
                        qty=abs(delta),      # MT5 bridge: qty == lots; Alpaca/Paper: shares
                        price=price,
                        ts=pd.Timestamp.utcnow(),
                    )
                    oid = broker.submit_order(ord)
                    limiter.notify_placed(sym, df)
                    log.info(f"{sym} {side} {abs(delta):.4f} @ {price:.5f} (eq={eq:.2f}) -> {oid}")
                else:
                    log.info(f"{sym} no change. pos={current_qty:.4f} price={price:.5f}")

            time.sleep(poll_secs)

        except KeyboardInterrupt:
            log.info("Stopped by user.")
            break
        except Exception as e:
            log.exception(f"Live loop error: {e}")
            time.sleep(poll_secs)


if __name__ == "__main__":
    main()