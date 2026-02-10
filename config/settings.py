# config/settings.py
import os
from dotenv import load_dotenv

# Load environment variables from ".env"
load_dotenv()

class Config:
    """
    Central configuration for the trading system.
    Reads from .env and falls back to safe defaults.
    Includes backward compatibility for older variable names.
    """

    # === API Configuration ===
    USE_ALPACA = os.getenv("USE_ALPACA", "false").lower() == "true"
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    ALPACA_API_KEY        = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY     = os.getenv("ALPACA_SECRET_KEY")
    ALPACA_BASE_URL       = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    

    # === Trading Parameters ===
    INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", 10_000))
    RISK_PER_TRADE  = float(os.getenv("RISK_PER_TRADE", 0.02))
    MAX_POSITIONS   = int(os.getenv("MAX_POSITIONS", 5))

    # === Mean Reversion Parameters ===
    LOOKBACK  = int(float(os.getenv("LOOKBACK", 20)))
    Z_ENTRY     = float(os.getenv("Z_ENTRY", 2.0))
    Z_EXIT      = float(os.getenv("Z_EXIT", 0.0))
    STOP_LOSS    = float(os.getenv("STOP_LOSS", 0.02))
    TAKE_PROFIT  = float(os.getenv("TAKE_PROFIT", 0.04))

    # === Transaction Costs (basis points) ===
    FEE_BPS      = float(os.getenv("FEE_BPS", os.getenv("FEES_BPS", 1.0)))
    SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", 0.5))

    # === Data Parameters ===
    FOREX_PAIRS  = os.getenv("FOREX_PAIRS", "EURUSD=X,GBPUSD=X,USDJPY=X,AUDUSD=X,USDCAD=X").split(",")
    LIVE_SYMBOLS = os.getenv("LIVE_SYMBOLS", "EURUSD=X,GBPUSD=X")
    TIMEFRAME    = os.getenv("TIMEFRAME", os.getenv("INTERVAL", "1h"))
    PERIOD       = os.getenv("PERIOD", "6mo")
    DATA_DIR     = os.getenv("DATA_DIR", "data")
    REPORTS_DIR  = os.getenv("REPORTS_DIR", "reports")
    CACHE        = os.getenv("CACHE", "true").lower() == "true"
    
    # MetaTrader 5
    MT5_MIN_LOTS = float(os.getenv("MT5_MIN_LOTS", "0.01"))
    MT5_MAX_LOTS = float(os.getenv("MT5_MAX_LOTS_PER_TRADE", "0.02"))
    LOTS_PER_100USD = float(os.getenv("MT5_LOTS_PER_100USD", "0.01"))

    # === Live Execution Parameters ===
    LIVE_POLL_SECONDS = int(os.getenv("LIVE_POLL_SECONDS", 60))  # poll every 60 seconds in live mode

    # === Position Sizing ===
    SIZING_METHOD = os.getenv("SIZING_METHOD", "fixed")  # "fixed" or "atr"
    ATR_PERIOD    = int(os.getenv("ATR_PERIOD", 14))
    ATR_MULT      = float(os.getenv("ATR_MULT", 1.0))

    # === Walk-Forward Optimization Parameters ===
    TRAIN_BARS = int(os.getenv("TRAIN_BARS", 300))
    TEST_BARS  = int(os.getenv("TEST_BARS", 60))
    
    # Bridge
    BRIDGE_HOST   = os.environ.get("BRIDGE_HOST", "127.0.0.1")
    BRIDGE_PORT   = int(os.environ.get("BRIDGE_PORT", "5000"))
    REPORTS_DIR   = os.environ.get("REPORTS_DIR", "reports")
    EVENTS_CSV    = os.path.join(REPORTS_DIR, "live_events.csv")
    MAX_QUEUE_LEN = int(os.environ.get("BRIDGE_MAX_QUEUE", "1000"))
    
    # --- Live capital caps (optional) ---
    MAX_LIVE_CAPITAL_FRACTION = float(os.getenv("MAX_LIVE_CAPITAL_FRACTION", 0.95))  # default 95% of BP
    try:
        MAX_LIVE_CAPITAL = float(os.getenv("MAX_LIVE_CAPITAL", "").strip())  # absolute $ cap
    except Exception:
        MAX_LIVE_CAPITAL = None

    try:
        PER_TRADE_NOTIONAL_CAP = float(os.getenv("PER_TRADE_NOTIONAL_CAP", "").strip())
    except Exception:
        PER_TRADE_NOTIONAL_CAP = None

    try:
        PER_TRADE_EQUITY_FRACTION = float(os.getenv("PER_TRADE_EQUITY_FRACTION", "").strip())
    except Exception:
        PER_TRADE_EQUITY_FRACTION = None

    @staticmethod
    def periods_per_year_from_timeframe(tf: str) -> float:
        """
        Annualization factor for Sharpe-like stats.
        Forex trades 24/7 — use these approximations:
          1d  -> 365
          1h  -> 24 * 365
          30m -> 2 * 24 * 365
          15m -> 4 * 24 * 365
        Default 252 (equity-like) for unknown frames.
        """
        tf = tf.lower().strip()
        if tf == "1d":  return 365.0
        if tf == "1h":  return 24.0 * 365.0
        if tf == "30m": return 2.0 * 24.0 * 365.0
        if tf == "15m": return 4.0 * 24.0 * 365.0
        return 252.0