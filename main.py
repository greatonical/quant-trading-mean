# main.py
from __future__ import annotations
import pandas as pd


# --- Dynamic project-root resolver (run from anywhere) ---
import os, sys
_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(_THIS_FILE)  # main.py sits at the root already
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# ---------------------------------------------------------

from config.settings import Config
from utils.logger import get_logger
from utils.data_loader import DataLoader
from strategies.mean_reversion import MeanReversionStrategy
from backtests.backtest_engine import BacktestEngine

log = get_logger(__name__)

def run_symbol(symbol: str, loader: DataLoader, strategy: MeanReversionStrategy, engine: BacktestEngine):
    df = loader.download(symbol, period=Config.PERIOD, interval=Config.TIMEFRAME, force_refresh=False)
    if df.empty:
        log.error(f"{symbol}: no data, skipping.")
        return None, None

    res = strategy.generate(df)
    metrics = engine.metrics(res)
    log.info(f"Results for {symbol}:")
    for k, v in metrics.items():
        log.info(f"  {k}: {v}")
    return res, metrics

def main():
    log.info("🚀 Mean Reversion — Starting")
    loader = DataLoader()
    strategy = MeanReversionStrategy(
        lookback=Config.LOOKBACK,
        z_entry=Config.Z_ENTRY,
        z_exit=Config.Z_EXIT,
        fee_bps=Config.FEE_BPS,
        slippage_bps=Config.SLIPPAGE_BPS
    )
    engine = BacktestEngine(initial_capital=Config.INITIAL_CAPITAL, timeframe=Config.TIMEFRAME)

    all_metrics = []
    first_df = None
    first_symbol = None

    for sym in Config.FOREX_PAIRS:
        log.info(f"=== {sym} ===")
        df_res, m = run_symbol(sym, loader, strategy, engine)
        if df_res is None:
            continue
        all_metrics.append({"Symbol": sym, **m})
        if first_df is None:
            first_df = df_res
            first_symbol = sym

    if first_df is not None:
        engine.plot(first_df, symbol=first_symbol)
        try:
            path = engine.quantstats_report(first_df, first_symbol)
            log.info(f"QuantStats report (first symbol) → {path}")
        except Exception as e:
            log.warning(f"QuantStats report skipped: {e}")

    if all_metrics:
        log.info("📋 SUMMARY")
        summary = pd.DataFrame(all_metrics)
        with pd.option_context('display.max_columns', None):
            print(summary.to_string(index=False))

    log.info("✅ Done.")

if __name__ == "__main__":
    main()
