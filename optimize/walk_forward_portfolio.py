from __future__ import annotations

# --- project root resolver ---
import os, sys
_THIS = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# -----------------------------

from typing import List, Dict, Tuple, Optional
import argparse
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
    ann = Config.periods_per_year_from_timeframe(Config.TIMEFRAME)
    best = None
    best_metrics = None
    MIN_BARS = 5

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
                                best, best_metrics = (lb, ze, zx, sl, tp), m
                            else:
                                if metric == "max_drawdown":
                                    if m["max_drawdown"] > best_metrics["max_drawdown"]:
                                        best, best_metrics = (lb, ze, zx, sl, tp), m
                                else:
                                    if m[metric] > best_metrics[metric]:
                                        best, best_metrics = (lb, ze, zx, sl, tp), m
                        except Exception:
                            continue

    if best is None:
        raise RuntimeError("No valid parameter set found on training window.")
    return (*best, best_metrics)


def _forward_slice(df: pd.DataFrame, idx: int, train_bars: int, test_bars: int):
    tr = df.iloc[idx: idx + train_bars]
    te = df.iloc[idx + train_bars: idx + train_bars + test_bars]
    return tr, te


def _equal_weight(n: int) -> np.ndarray:
    if n <= 0: return np.array([])
    w = np.ones(n, dtype=float) / n
    return w


def _inv_vol_weights(returns_df: pd.DataFrame, eps: float = 1e-8) -> np.ndarray:
    # annualization is irrelevant for relative weights; use sample stdev
    vol = returns_df.std(ddof=0).replace(0, np.nan)
    inv = 1.0 / (vol + eps)
    inv = inv.fillna(0.0)
    if inv.sum() == 0:
        return _equal_weight(returns_df.shape[1])
    w = inv / inv.sum()
    return w.values


def plot_equity(eq: pd.Series, title: str, out_path: str):
    fig = plt.figure(figsize=(10,5))
    ax = plt.gca()
    ax.plot(eq.index, eq.values, label="WFP Equity")
    ax.axhline(y=1.0, linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("Equity (start=1.0)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def walk_forward_portfolio(
    data_map: Dict[str, pd.DataFrame],
    train_bars: int,
    test_bars: int,
    lookbacks: List[int],
    z_entries: List[float],
    z_exits: List[float],
    stop_losses: List[float],
    take_profits: List[float],
    metric: str = "sharpe",
    weight_rule: str = "inv_vol",  # "equal" | "inv_vol"
    max_windows: Optional[int] = None,
    auto_relax: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    For each rolling window:
      - For each symbol: optimize on train, test on test → get OOS returns.
      - Stack OOS returns across symbols and compute per-window weights.
      - Produce a portfolio return series per window, then stitch all OOS windows.
    """
    symbols = list(data_map.keys())
    lengths = [len(df) for df in data_map.values()]
    total = min(lengths)  # align windows conservatively on shortest asset

    # align by index intersection to avoid mismatched timestamps
    common_index = None
    for df in data_map.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    data_map = {sym: df.loc[common_index].copy() for sym, df in data_map.items()}
    total = len(common_index)

    if train_bars + test_bars > total:
        raise ValueError("Not enough bars for one train+test window across all assets.")

    idx = 0
    window_count = 0
    params_rows, metrics_rows = []
    stitched = []

    while idx + train_bars + test_bars <= total:
        # Build slices for all symbols
        train_slices = {}
        test_slices  = {}
        for sym, df in data_map.items():
            tr, te = _forward_slice(df, idx, train_bars, test_bars)
            train_slices[sym] = tr
            test_slices[sym]  = te

        # Warm-up check against largest lookback/ATR
        warmup = max(max(lookbacks), Config.ATR_PERIOD) + 5
        if any(len(tr) < warmup + 10 for tr in train_slices.values()):
            log.warning(f"WFP Window {window_count}: training slice too short. Skipping.")
            idx += test_bars
            window_count += 1
            if max_windows and window_count >= max_windows: break
            continue

        # Per-symbol optimization on training, then test
        per_sym_rets = {}
        per_sym_params = {}
        for sym in symbols:
            tr = train_slices[sym]
            te = test_slices[sym]
            try:
                lb, ze, zx, sl, tp, _ = select_best(tr, lookbacks, z_entries, z_exits, stop_losses, take_profits, metric)
                bt_test = run_strategy(te, lb, ze, zx, sl, tp, use_rsi=True)
                sr = bt_test["strategy_ret"].dropna()

                # Auto-relax if OOS has zero entries
                entries = ((bt_test["position"] != 0) & (bt_test["position"].shift(1).fillna(0) == 0)).sum()
                if entries == 0 and auto_relax:
                    ze_relaxed = max(0.6, ze * 0.7)
                    bt_test = run_strategy(te, lb, ze_relaxed, zx, sl, tp, use_rsi=False)
                    sr = bt_test["strategy_ret"].dropna()

                per_sym_rets[sym] = sr
                per_sym_params[sym] = dict(lookback=lb, z_entry=ze, z_exit=zx, stop_loss=sl, take_profit=tp)
            except Exception as e:
                log.warning(f"WFP Window {window_count} {sym}: failed ({e}). Using zero returns.")
                per_sym_rets[sym] = pd.Series(0.0, index=te.index)
                per_sym_params[sym] = dict(lookback=np.nan, z_entry=np.nan, z_exit=np.nan, stop_loss=np.nan, take_profit=np.nan)

        # stack returns into DataFrame aligned by time
        ret_df = pd.DataFrame({sym: per_sym_rets[sym].reindex(test_slices[sym].index).fillna(0.0) for sym in symbols})
        # compute weights within this window
        if weight_rule == "equal":
            w = _equal_weight(len(symbols))
        else:
            w = _inv_vol_weights(ret_df)

        # portfolio return for this window
        port_ret = (ret_df.values @ w)
        port_ret = pd.Series(port_ret, index=ret_df.index, name="portfolio_ret")
        stitched.append(port_ret)

        # bookkeeping rows
        # params: one row per symbol per window
        for sym in symbols:
            row = {"window": window_count, "symbol": sym,
                   "train_start": train_slices[sym].index[0], "train_end": train_slices[sym].index[-1],
                   "test_start":  test_slices[sym].index[0],  "test_end":  test_slices[sym].index[-1],
                   "weight_rule": weight_rule, "weight": float(w[symbols.index(sym)])}
            row.update(per_sym_params[sym])
            params_rows.append(row)

        # metrics for the window (portfolio-level)
        ann = Config.periods_per_year_from_timeframe(Config.TIMEFRAME)
        m = compute_metrics_from_returns(port_ret.dropna(), Config.INITIAL_CAPITAL, ann)
        metrics_rows.append({"window": window_count, **m})

        idx += test_bars
        window_count += 1
        if max_windows and window_count >= max_windows: break

    if not stitched:
        raise RuntimeError("WFP produced no windows.")

    fwd_port = pd.concat(stitched).sort_index()
    fwd_port = fwd_port[~fwd_port.index.duplicated(keep="first")]

    params_df = pd.DataFrame(params_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    return params_df, metrics_df, fwd_port


def main():
    ap = argparse.ArgumentParser(description="Walk-Forward Portfolio (multi-asset).")
    ap.add_argument("--symbols", required=True, help="Comma-separated tickers, e.g. EURUSD=X,GBPUSD=X,USDJPY=X")
    ap.add_argument("--interval", default=None, help="1h, 1d, etc. Defaults to Config.TIMEFRAME")
    ap.add_argument("--period", default=None, help="60d for intraday; 2y for 1d+. Defaults to Config.PERIOD")
    ap.add_argument("--train-bars", type=int, default=2000)
    ap.add_argument("--test-bars",  type=int, default=300)
    ap.add_argument("--lookbacks", default="10,15,20,30")
    ap.add_argument("--z-entries", default="1.0,1.5,2.0")
    ap.add_argument("--z-exits",   default="0.0,0.2,0.5")
    ap.add_argument("--stop-losses",  default="0.01,0.02")
    ap.add_argument("--take-profits", default="0.02,0.04")
    ap.add_argument("--metric", default="sharpe")
    ap.add_argument("--weight-rule", default="inv_vol", choices=["equal","inv_vol"])
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--auto-relax", action="store_true")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    interval = args.interval or Config.TIMEFRAME
    period = args.period or Config.PERIOD

    loader = DataLoader()
    data_map = {}
    for sym in symbols:
        df = loader.download(sym, period=period, interval=interval, force_refresh=False)
        data_map[sym] = df

    params_df, metrics_df, fwd_port = walk_forward_portfolio(
        data_map=data_map,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        lookbacks=[int(x) for x in args.lookbacks.split(",")],
        z_entries=[float(x) for x in args.z_entries.split(",")],
        z_exits=[float(x) for x in args.z_exits.split(",")],
        stop_losses=[float(x) for x in args.stop_losses.split(",")],
        take_profits=[float(x) for x in args.take_profits.split(",")],
        metric=args.metric,
        weight_rule=args.weight_rule,
        max_windows=args.max_windows,
        auto_relax=args.auto_relax,
    )

    os.makedirs(Config.REPORTS_DIR, exist_ok=True)
    base = "wfp_" + "_".join([s.replace('=','_').replace('/','_') for s in symbols])
    ppath  = os.path.join(Config.REPORTS_DIR, f"{base}_params.csv")
    mpath  = os.path.join(Config.REPORTS_DIR, f"{base}_metrics.csv")
    rpath  = os.path.join(Config.REPORTS_DIR, f"{base}_returns.csv")
    epath  = os.path.join(Config.REPORTS_DIR, f"{base}_equity.png")

    params_df.to_csv(ppath, index=False)
    metrics_df.to_csv(mpath, index=False)
    fwd_port.to_csv(rpath, header=["portfolio_ret"])
    eq = (1 + fwd_port.fillna(0)).cumprod()
    plot_equity(eq, f"WFP Equity — {', '.join(symbols)}", epath)

    log.info(f"WFP params  → {ppath}")
    log.info(f"WFP metrics → {mpath}")
    log.info(f"WFP returns → {rpath}")
    log.info(f"WFP equity  → {epath}")


if __name__ == "__main__":
    main()