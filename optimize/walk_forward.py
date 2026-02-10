from __future__ import annotations

# --- Dynamic project-root resolver (run from anywhere) ---
import os, sys
_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))  # .../quant-trading-mean
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# ---------------------------------------------------------

import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.settings import Config
from utils.logger import get_logger
from utils.data_loader import DataLoader
from strategies.mean_reversion import MeanReversionStrategy
from optimize.mean_reversion_optimizer import compute_metrics_from_returns

log = get_logger(__name__)


def run_strategy(df: pd.DataFrame,
                 lookback: int,
                 z_entry: float,
                 z_exit: float,
                 stop_loss_pct: float,
                 take_profit_pct: float,
                 use_rsi: bool = True) -> pd.DataFrame:
    strat = MeanReversionStrategy(
        lookback=lookback,
        z_entry=z_entry,
        z_exit=z_exit,
        fee_bps=Config.FEE_BPS,
        slippage_bps=Config.SLIPPAGE_BPS,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        sizing_method=Config.SIZING_METHOD,
        atr_period=Config.ATR_PERIOD,
        atr_mult=Config.ATR_MULT,
        use_rsi=use_rsi,
    )
    return strat.generate(df)


def select_best(df_train: pd.DataFrame,
                lookbacks: List[int],
                z_entries: List[float],
                z_exits: List[float],
                stop_losses: List[float],
                take_profits: List[float],
                metric: str) -> Tuple[int, float, float, float, float, Dict[str, float]]:
    """
    Brute-force grid on training window only. Skips combos with too few usable bars.
    """
    ann = Config.periods_per_year_from_timeframe(Config.TIMEFRAME)
    best = None
    best_metrics = None
    MIN_BARS = 5  # lenient so we don't discard sparse windows entirely

    for lb in lookbacks:
        for ze in z_entries:
            for zx in z_exits:
                for sl in stop_losses:
                    for tp in take_profits:
                        try:
                            bt = run_strategy(df_train, lb, ze, zx, sl, tp, use_rsi=True)
                            sr = bt["strategy_ret"].dropna()
                            if len(sr) < MIN_BARS:
                                continue
                            m = compute_metrics_from_returns(sr, Config.INITIAL_CAPITAL, ann)
                            if best is None:
                                best = (lb, ze, zx, sl, tp); best_metrics = m
                            else:
                                if metric == "max_drawdown":
                                    if m["max_drawdown"] > best_metrics["max_drawdown"]:
                                        best = (lb, ze, zx, sl, tp); best_metrics = m
                                else:
                                    if m[metric] > best_metrics[metric]:
                                        best = (lb, ze, zx, sl, tp); best_metrics = m
                        except Exception:
                            continue

    if best is None:
        raise RuntimeError("No valid parameter set found on training window.")
    return (*best, best_metrics)


def stitch_forward_returns(windows: List[Dict]) -> pd.Series:
    parts = []
    for w in windows:
        sr = w.get("test_returns", None)
        if sr is not None and not sr.empty:
            parts.append(sr)
    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def plot_wfo_equity(forward_rets: pd.Series, out_path: str, title: str):
    eq = (1 + forward_rets.fillna(0.0)).cumprod()
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.plot(eq.index, eq.values, label="Walk-Forward Equity")
    ax.axhline(y=1.0, linestyle='--', alpha=0.5, label='Start')
    ax.set_title(title)
    ax.set_ylabel("Equity (start=1.0)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def walk_forward(
    df: pd.DataFrame,
    train_bars: int,
    test_bars: int,
    lookbacks: List[int],
    z_entries: List[float],
    z_exits: List[float],
    stop_losses: List[float],
    take_profits: List[float],
    metric: str = "sharpe",
    max_windows: Optional[int] = None,
    auto_relax: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    total = len(df)
    if train_bars + test_bars > total:
        raise ValueError("Not enough bars for one train+test window.")

    idx = 0
    windows = []
    window_count = 0

    while idx + train_bars + test_bars <= total:
        tr = df.iloc[idx : idx + train_bars]
        te = df.iloc[idx + train_bars : idx + train_bars + test_bars]
        if len(tr) < 10 or len(te) < 5:
            break

        # Warm-up requirement
        warmup = max(max(lookbacks), Config.ATR_PERIOD) + 5
        if len(tr) < warmup + 10:
            log.warning(f"Window {window_count}: training slice too short (need ≥ {warmup+10} bars). Skipping.")
            idx += test_bars
            window_count += 1
            if max_windows and window_count >= max_windows:
                break
            continue

        try:
            lb, ze, zx, sl, tp, train_metrics = select_best(tr, lookbacks, z_entries, z_exits, stop_losses, take_profits, metric)
        except Exception as e:
            log.warning(f"Window {window_count}: optimization failed, skipping. {e}")
            idx += test_bars
            window_count += 1
            if max_windows and window_count >= max_windows:
                break
            continue

        # Out-of-sample test
        try:
            bt_test = run_strategy(te, lb, ze, zx, sl, tp, use_rsi=True)
            test_rets = bt_test["strategy_ret"].dropna()
            oos_entries = ((bt_test["position"] != 0) & (bt_test["position"].shift(1).fillna(0) == 0)).sum()
            log.info(f"Window {window_count}: OOS entries={int(oos_entries)}, returns_len={len(test_rets)} "
                     f"(lb={lb}, ze={ze}, zx={zx}, sl={sl}, tp={tp})")

            # Save the first test slice if debugging
            if window_count == 0 and os.getenv("DEBUG_WFO", "0") == "1":
                os.makedirs(Config.REPORTS_DIR, exist_ok=True)
                debug_path = os.path.join(Config.REPORTS_DIR, "wfo_test_window0_debug.csv")
                bt_test[['Close','Z','RSI','position','size','strategy_ret']].to_csv(debug_path)
                log.info(f"Saved test-window0 debug to {debug_path}")

            # Auto-relax if zero entries
            if oos_entries == 0 and auto_relax:
                ze_relaxed = max(0.6, ze * 0.7)  # reduce threshold
                log.info(f"Window {window_count}: Auto-relaxing (RSI OFF, z_entry {ze}→{ze_relaxed}). Retesting...")
                bt_test = run_strategy(te, lb, ze_relaxed, zx, sl, tp, use_rsi=False)
                test_rets = bt_test["strategy_ret"].dropna()
                oos_entries = ((bt_test["position"] != 0) & (bt_test["position"].shift(1).fillna(0) == 0)).sum()
                log.info(f"Window {window_count}: OOS entries after relax={int(oos_entries)}, returns_len={len(test_rets)}")

        except Exception as e:
            log.warning(f"Window {window_count}: test failed, skipping. {e}")
            test_rets = pd.Series(dtype=float)

        ann = Config.periods_per_year_from_timeframe(Config.TIMEFRAME)
        test_metrics = compute_metrics_from_returns(test_rets, Config.INITIAL_CAPITAL, ann)

        windows.append({
            "window": window_count,
            "train_start": tr.index[0], "train_end": tr.index[-1],
            "test_start": te.index[0],  "test_end":  te.index[-1],
            "lookback": lb, "z_entry": ze, "z_exit": zx, "stop_loss": sl, "take_profit": tp,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "test_returns": test_rets,
        })

        idx += test_bars
        window_count += 1
        if max_windows and window_count >= max_windows:
            break

    if not windows:
        raise RuntimeError("No walk-forward windows produced results.")

    params_rows, metrics_rows = [], []
    for w in windows:
        params_rows.append({
            "window": w["window"],
            "train_start": w["train_start"], "train_end": w["train_end"],
            "test_start":  w["test_start"],  "test_end":  w["test_end"],
            "lookback": w["lookback"], "z_entry": w["z_entry"], "z_exit": w["z_exit"],
            "stop_loss": w["stop_loss"], "take_profit": w["take_profit"],
        })
        tm = w["test_metrics"]
        metrics_rows.append({
            "window": w["window"],
            "Final Equity": tm.get("final_equity", np.nan),
            "Total Return": tm.get("total_return", np.nan),
            "Annualized Return": tm.get("ann_return", np.nan),
            "Annualized Volatility": tm.get("ann_vol", np.nan),
            "Sharpe Ratio": tm.get("sharpe", np.nan),
            "Max Drawdown": tm.get("max_drawdown", np.nan),
            "Bars": tm.get("bars", 0),
        })

    params_df = pd.DataFrame(params_rows)
    metrics_df = pd.DataFrame(metrics_rows)

    forward_rets = stitch_forward_returns(windows)
    return params_df, metrics_df, forward_rets


def main():
    ap = argparse.ArgumentParser(description="Walk-Forward Optimization for Mean Reversion (with SL/TP, sizing, auto-relax).")
    ap.add_argument("--symbol", required=True, help="Ticker (e.g., EURUSD=X)")
    ap.add_argument("--interval", default=None, help="1h, 1d, etc. Defaults to Config.TIMEFRAME")
    ap.add_argument("--period", default=None, help="6mo, 1y, 60d, etc. Defaults to Config.PERIOD")
    ap.add_argument("--train-bars", type=int, default=2000, help="Bars in training window")
    ap.add_argument("--test-bars",  type=int, default=300,  help="Bars in testing window")
    ap.add_argument("--lookbacks", default="10,15,20,30,40,60", help="Comma-separated lookbacks")
    ap.add_argument("--z-entries", default="1.5,2.0,2.5,3.0", help="Comma-separated z_entry values")
    ap.add_argument("--z-exits",   default="0.0,0.2,0.5,0.8", help="Comma-separated z_exit values")
    ap.add_argument("--stop-losses",   default="0.01,0.02,0.03", help="Comma-separated stop-loss % (decimals)")
    ap.add_argument("--take-profits",  default="0.02,0.03,0.05", help="Comma-separated take-profit % (decimals)")
    ap.add_argument("--metric", default="sharpe", help="sharpe | ann_return | total_return | max_drawdown")
    ap.add_argument("--max-windows", type=int, default=None, help="Optional cap on number of rolling windows")
    ap.add_argument("--auto-relax", action="store_true",
                    help="If a test window has 0 entries, retry with RSI OFF and reduced z_entry.")
    args = ap.parse_args()

    symbol = args.symbol
    interval = args.interval or Config.TIMEFRAME
    period = args.period or Config.PERIOD

    lookbacks = [int(x) for x in args.lookbacks.split(",")]
    z_entries = [float(x) for x in args.z_entries.split(",")]
    z_exits   = [float(x) for x in args.z_exits.split(",")]
    stop_losses  = [float(x) for x in args.stop_losses.split(",")]
    take_profits = [float(x) for x in args.take_profits.split(",")]

    loader = DataLoader()
    df = loader.download(symbol, period=period, interval=interval, force_refresh=False)
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")

    params_df, metrics_df, forward_rets = walk_forward(
        df=df,
        train_bars=int(args.train_bars),
        test_bars=int(args.test_bars),
        lookbacks=lookbacks,
        z_entries=z_entries,
        z_exits=z_exits,
        stop_losses=stop_losses,
        take_profits=take_profits,
        metric=args.metric.lower(),
        max_windows=args.max_windows,
        auto_relax=bool(args.auto_relax),
    )

    os.makedirs(Config.REPORTS_DIR, exist_ok=True)
    base = f"wfo_{symbol.replace('/','_').replace('=','_')}"
    params_path  = os.path.join(Config.REPORTS_DIR, f"{base}_params.csv")
    metrics_path = os.path.join(Config.REPORTS_DIR, f"{base}_metrics.csv")
    equity_path  = os.path.join(Config.REPORTS_DIR, f"{base}_equity.png")
    fwdrets_path = os.path.join(Config.REPORTS_DIR, f"{base}_forward_returns.csv")

    params_df.to_csv(params_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    forward_rets.to_csv(fwdrets_path, header=["forward_strategy_ret"])
    plot_wfo_equity(forward_rets, equity_path, title=f"WFO Equity — {symbol}")

    log.info(f"Saved WFO params → {params_path}")
    log.info(f"Saved WFO test metrics → {metrics_path}")
    log.info(f"Saved WFO forward returns → {fwdrets_path}")
    log.info(f"Saved WFO equity plot → {equity_path}")


if __name__ == "__main__":
    main()