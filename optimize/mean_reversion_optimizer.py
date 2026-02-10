# optimize/mean_reversion_optimizer.py
from __future__ import annotations

# --- Dynamic project-root resolver (run from anywhere) ---
import os, sys
_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))  # .../quant-trading-mean
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# ---------------------------------------------------------

import argparse
import itertools
import math
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.settings import Config
from utils.logger import get_logger
from utils.data_loader import DataLoader
from strategies.mean_reversion import MeanReversionStrategy

log = get_logger(__name__)

# ------------------------------
# Metrics helper
# ------------------------------
def compute_metrics_from_returns(
    rets: pd.Series, initial_capital: float, annualizer: float
) -> Dict[str, float]:
    rets = rets.dropna().astype(float)
    if rets.empty:
        return {
            "total_return": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "bars": 0,
            "final_equity": np.nan,
        }

    equity = (1 + rets).cumprod()
    final_equity = initial_capital * equity.iloc[-1]

    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    max_dd = dd.min()

    m = rets.mean()
    s = rets.std(ddof=0)
    ann_ret = m * annualizer
    ann_vol = s * math.sqrt(annualizer) if s > 0 else np.nan
    sharpe = (ann_ret / ann_vol) if (ann_vol and ann_vol > 0) else np.nan

    return {
        "total_return": equity.iloc[-1] - 1.0,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "bars": int(len(rets)),
        "final_equity": float(final_equity),
    }

# ------------------------------
# Strategy run (supports SL/TP)
# ------------------------------
def run_strategy(df: pd.DataFrame,
                 lookback: int,
                 z_entry: float,
                 z_exit: float,
                 stop_loss_pct: float,
                 take_profit_pct: float) -> pd.DataFrame:
    strat = MeanReversionStrategy(
        lookback=lookback,
        z_entry=z_entry,
        z_exit=z_exit,
        fee_bps=Config.FEE_BPS,
        slippage_bps=Config.SLIPPAGE_BPS,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )
    return strat.generate(df)

# ------------------------------
# Grid optimization (now includes SL/TP)
# ------------------------------
def optimize_grid(
    symbol: str,
    df: pd.DataFrame,
    lookbacks: List[int],
    z_entries: List[float],
    z_exits: List[float],
    stop_losses: List[float],
    take_profits: List[float],
    metric: str = "sharpe",
    train_frac: float = 1.0,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Grid search over (lookback, z_entry, z_exit, stop_loss_pct, take_profit_pct).
    If train_frac < 1.0, split time-wise into train/test; optimize on train, report best on test.
    Returns (results_train, results_test_for_best_or_none).
    """
    if not (0.5 <= train_frac <= 1.0):
        raise ValueError("train_frac should be between 0.5 and 1.0")
    split_idx = int(len(df) * train_frac)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy() if train_frac < 1.0 else None

    annualizer = Config.periods_per_year_from_timeframe(Config.TIMEFRAME)
    results = []

    # Cartesian product of parameter grid
    for lb, ze, zx, sl, tp in itertools.product(lookbacks, z_entries, z_exits, stop_losses, take_profits):
        # Sanity: z_exit should typically be closer to mean than z_entry (optional, not enforced)
        try:
            bt = run_strategy(df_train, lb, ze, zx, sl, tp)
            metrics = compute_metrics_from_returns(bt["strategy_ret"], Config.INITIAL_CAPITAL, annualizer)
            results.append({
                "symbol": symbol,
                "lookback": lb,
                "z_entry": ze,
                "z_exit": zx,
                "stop_loss": sl,
                "take_profit": tp,
                **metrics
            })
        except Exception as e:
            log.warning(f"Param set failed (lb={lb}, z_entry={ze}, z_exit={zx}, sl={sl}, tp={tp}): {e}")

    results_df = pd.DataFrame(results)

    metric = metric.lower()
    if metric not in {"sharpe", "ann_return", "total_return", "max_drawdown"}:
        raise ValueError("metric must be one of: sharpe, ann_return, total_return, max_drawdown")

    if results_df.empty:
        return results_df, None

    if metric == "max_drawdown":
        best_row = results_df.loc[results_df["max_drawdown"].idxmax()]
    else:
        best_row = results_df.loc[results_df[metric].idxmax()]

    test_df = None
    if df_test is not None and len(df_test) > 10:
        bt_test = run_strategy(
            df_test,
            int(best_row["lookback"]),
            float(best_row["z_entry"]),
            float(best_row["z_exit"]),
            float(best_row["stop_loss"]),
            float(best_row["take_profit"])
        )
        test_metrics = compute_metrics_from_returns(bt_test["strategy_ret"], Config.INITIAL_CAPITAL, annualizer)
        test_df = pd.DataFrame([{
            "symbol": symbol,
            "lookback": int(best_row["lookback"]),
            "z_entry": float(best_row["z_entry"]),
            "z_exit": float(best_row["z_exit"]),
            "stop_loss": float(best_row["stop_loss"]),
            "take_profit": float(best_row["take_profit"]),
            **{f"test_{k}": v for k, v in test_metrics.items()}
        }])

    return results_df, test_df

# ------------------------------
# Helpers: save CSV & heatmaps
# ------------------------------
def _safe_filename(s: str) -> str:
    return s.replace("/", "_").replace("=", "_")

def save_results_and_plots(
    symbol: str,
    results_df: pd.DataFrame,
    best_params: Tuple[int, float, float, float, float],
    out_dir: str = Config.REPORTS_DIR,
    heatmap_fixed: Tuple[int, float] | None = None,  # (lookback, stop_loss) slice
):
    os.makedirs(out_dir, exist_ok=True)
    base = f"opt_meanrev_{_safe_filename(symbol)}"
    csv_path = os.path.join(out_dir, f"{base}.csv")
    results_df.to_csv(csv_path, index=False)
    log.info(f"Saved grid results CSV → {csv_path}")

    # We have 5 params; for heatmaps fix 2 of them (lookback, stop_loss) and plot z_entry vs z_exit per take_profit
    if heatmap_fixed is None:
        # default: pick mode of lookback and stop_loss
        lb = int(results_df["lookback"].mode().iloc[0])
        sl = float(results_df["stop_loss"].mode().iloc[0])
    else:
        lb, sl = heatmap_fixed

    slice_df = results_df[(results_df["lookback"] == lb) & (results_df["stop_loss"] == sl)].copy()
    if slice_df.empty:
        log.warning("No rows for heatmap slice; skipping heatmaps.")
        return

    for metric in ["sharpe", "ann_return", "total_return"]:
        # We’ll produce one heatmap per take_profit value
        for tp in sorted(slice_df["take_profit"].unique()):
            sub = slice_df[slice_df["take_profit"] == tp]
            if sub.empty:
                continue
            pivot = sub.pivot_table(index="z_entry", columns="z_exit", values=metric, aggfunc="mean")
            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca()
            im = ax.imshow(pivot.values, aspect="auto", origin="lower")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
            ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
            ax.set_xlabel("z_exit")
            ax.set_ylabel("z_entry")
            ax.set_title(f"{symbol} — {metric} (lb={lb}, SL={sl:.2%}, TP={tp:.2%})")
            fig.colorbar(im, ax=ax)
            out_png = os.path.join(out_dir, f"{base}_heatmap_{metric}_lb{lb}_sl{int(sl*10000)}bp_tp{int(tp*10000)}bp.png")
            fig.tight_layout()
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            log.info(f"Saved heatmap → {out_png}")

    lb, ze, zx, sl2, tp2 = best_params
    log.info(f"Best params → lookback={lb}, z_entry={ze}, z_exit={zx}, stop_loss={sl2:.2%}, take_profit={tp2:.2%}")

# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Optimize Mean Reversion parameters (grid search incl. SL/TP).")
    ap.add_argument("--symbol", required=True, help="Ticker (e.g., EURUSD=X, AAPL)")
    ap.add_argument("--metric", default="sharpe", help="sharpe | ann_return | total_return | max_drawdown")
    ap.add_argument("--train-frac", type=float, default=1.0, help="Train fraction (0.5–1.0). Remainder is out-of-sample.")
    ap.add_argument("--lookbacks", default="10,15,20,30,40,60", help="Comma-separated lookbacks")
    ap.add_argument("--z-entries", default="1.5,2.0,2.5,3.0", help="Comma-separated z_entry values")
    ap.add_argument("--z-exits", default="0.0,0.2,0.5,0.8", help="Comma-separated z_exit values")
    ap.add_argument("--stop-losses", default="0.01,0.02,0.03", help="Comma-separated stop-loss % (decimals)")
    ap.add_argument("--take-profits", default="0.02,0.03,0.05", help="Comma-separated take-profit % (decimals)")
    ap.add_argument("--interval", default=None, help="1h, 1d, etc. Defaults to Config.TIMEFRAME")
    ap.add_argument("--period", default=None, help="6mo, 1y, etc. Defaults to Config.PERIOD")
    args = ap.parse_args()

    symbol = args.symbol
    metric = args.metric.lower()
    train_frac = float(args.train_frac)
    lbs = [int(x) for x in args.lookbacks.split(",")]
    zes = [float(x) for x in args.z_entries.split(",")]
    zxs = [float(x) for x in args.z_exits.split(",")]
    sls = [float(x) for x in args.stop_losses.split(",")]
    tps = [float(x) for x in args.take_profits.split(",")]
    interval = args.interval or Config.TIMEFRAME
    period = args.period or Config.PERIOD

    loader = DataLoader()
    df = loader.download(symbol, period=period, interval=interval, force_refresh=False)

    results_df, test_df = optimize_grid(
        symbol=symbol,
        df=df,
        lookbacks=lbs,
        z_entries=zes,
        z_exits=zxs,
        stop_losses=sls,
        take_profits=tps,
        metric=metric,
        train_frac=train_frac,
    )

    if results_df.empty:
        log.warning("No optimization results to save.")
        return

    # Pick best row by requested metric (train)
    if metric == "max_drawdown":
        best_row = results_df.loc[results_df["max_drawdown"].idxmax()]
    else:
        best_row = results_df.loc[results_df[metric].idxmax()]

    best_params = (
        int(best_row["lookback"]),
        float(best_row["z_entry"]),
        float(best_row["z_exit"]),
        float(best_row["stop_loss"]),
        float(best_row["take_profit"]),
    )

    save_results_and_plots(
        symbol=symbol,
        results_df=results_df,
        best_params=best_params,
        out_dir=Config.REPORTS_DIR,
        heatmap_fixed=(best_params[0], best_params[3]),  # fix lookback & stop-loss
    )

    if test_df is not None:
        log.info("Out-of-sample (test) metrics for best params:")
        with pd.option_context('display.max_columns', None):
            print(test_df.to_string(index=False))

if __name__ == "__main__":
    main()
