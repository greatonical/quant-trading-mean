# optimize/portfolio_optimizer.py
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
import os

from config.settings import Config
from utils.logger import get_logger
from utils.data_loader import DataLoader
from strategies.mean_reversion import MeanReversionStrategy

log = get_logger(__name__)

# ------------------------------
# Helpers: weight generators
# ------------------------------
def equal_weight(n: int) -> np.ndarray:
    """Equal weight vector that sums to 1 (no shorting)."""
    return np.ones(n) / n

def inv_vol_weight(returns: pd.DataFrame) -> np.ndarray:
    """
    Inverse-volatility weights (Risk Parity lite): w_i ∝ 1/σ_i.
    - Uses sample std of each column.
    - No shorting; normalizes to 1.
    """
    vol = returns.std(ddof=0).replace(0, np.nan)
    w = 1.0 / vol
    w = w.fillna(0.0).values
    if w.sum() == 0:
        return equal_weight(returns.shape[1])
    w = np.maximum(w, 0)
    return w / w.sum()

def dirichlet_weights(n_assets: int, n_samples: int = 5000, alpha: float = 1.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Sample nonnegative weight vectors that sum to 1 using a Dirichlet distribution.
    alpha=1 -> uniform over the simplex. Returns array shape (n_samples, n_assets).
    """
    rng = rng or np.random.default_rng()
    return rng.dirichlet(alpha=np.ones(n_assets) * alpha, size=n_samples)

# ------------------------------
# Metrics
# ------------------------------
def annualizer_from_interval(interval: str) -> float:
    return Config.periods_per_year_from_timeframe(interval)

def portfolio_stats(returns: pd.DataFrame, w: np.ndarray, annualizer: float) -> Dict[str, float]:
    """
    Compute portfolio mean/vol/sharpe & total/ann return for given weights (w).
    'returns' must be a DataFrame of aligned per-bar strategy returns for assets.
    """
    # Ensure shape
    R = returns.values  # shape (T, N)
    w = np.asarray(w).reshape(-1, 1)  # (N, 1)
    # Per-bar portfolio returns
    rp = (R @ w).ravel()  # (T,)
    rp = pd.Series(rp, index=returns.index)
    # Stats
    mu = rp.mean()
    sd = rp.std(ddof=0)
    ann_ret = mu * annualizer
    ann_vol = sd * np.sqrt(annualizer) if sd > 0 else 0.0
    sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0.0
    total_return = (1 + rp).prod() - 1.0
    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "total_return": float(total_return)
    }

# ------------------------------
# Core: build returns matrix
# ------------------------------
def build_returns_matrix(symbols: List[str], period: str, interval: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    For each symbol: download data, run mean-reversion strategy with Config params,
    and return a single DataFrame with aligned strategy returns columns (one per symbol).
    Also returns per-symbol metrics (ann_return, ann_vol, sharpe, total_return).
    """
    loader = DataLoader()
    strat = MeanReversionStrategy(
        lookback=Config.LOOKBACK,
        z_entry=Config.Z_ENTRY,
        z_exit=Config.Z_EXIT,
        fee_bps=Config.FEE_BPS,
        slippage_bps=Config.SLIPPAGE_BPS,
        stop_loss_pct=Config.STOP_LOSS,
        take_profit_pct=Config.TAKE_PROFIT
    )

    returns_map: Dict[str, pd.Series] = {}
    metrics_map: Dict[str, Dict[str, float]] = {}
    ann = annualizer_from_interval(interval)

    for sym in symbols:
        try:
            df = loader.download(sym, period=period, interval=interval, force_refresh=False)
            res = strat.generate(df)
            sr = res["strategy_ret"].dropna()
            if sr.empty:
                log.warning(f"{sym}: empty strategy returns; skipping.")
                continue
            returns_map[sym] = sr
            # Per-symbol stats (standalone)
            metrics_map[sym] = portfolio_stats(sr.to_frame(sym), np.array([1.0]), ann)
        except Exception as e:
            log.error(f"{sym}: {e}")

    if not returns_map:
        raise RuntimeError("No valid returns; please check symbols or data availability.")

    # Align by time index (inner join on timestamps)
    returns_df = pd.concat(returns_map.values(), axis=1, join="inner")
    returns_df.columns = list(returns_map.keys())
    returns_df = returns_df.sort_index()
    return returns_df, metrics_map

# ------------------------------
# Max-Sharpe via random search (Dirichlet)
# ------------------------------
def max_sharpe_dirichlet(returns: pd.DataFrame, n_samples: int = 5000, alpha: float = 1.0, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Sample lots of long-only weight vectors on the simplex and pick the highest Sharpe.
    """
    rng = rng or np.random.default_rng()
    W = dirichlet_weights(returns.shape[1], n_samples=n_samples, alpha=alpha, rng=rng)  # (S, N)
    ann = annualizer_from_interval(Config.TIMEFRAME)
    best_w = None
    best_stats = {"sharpe": -np.inf}
    for w in W:
        stats = portfolio_stats(returns, w, ann)
        if stats["sharpe"] > best_stats.get("sharpe", -np.inf):
            best_stats = stats
            best_w = w
    return best_w, best_stats

# ------------------------------
# Scatter plot helper
# ------------------------------
def scatter_portfolios(returns: pd.DataFrame, out_path: str, n_cloud: int = 1500):
    """
    Draw a scatter of random portfolios (ann_vol vs ann_return), with Sharpe color.
    Also mark equal-weight and inverse-vol solutions.
    """
    ann = annualizer_from_interval(Config.TIMEFRAME)
    W = dirichlet_weights(returns.shape[1], n_samples=n_cloud, alpha=1.0)
    vols, rets, sharpes = [], [], []
    for w in W:
        s = portfolio_stats(returns, w, ann)
        vols.append(s["ann_vol"])
        rets.append(s["ann_return"])
        sharpes.append(s["sharpe"])

    fig = plt.figure(figsize=(9, 6))
    ax = plt.gca()
    sc = ax.scatter(vols, rets, c=sharpes, alpha=0.6)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Sharpe")

    # Mark EW and INVVOL
    ew = equal_weight(returns.shape[1])
    ews = portfolio_stats(returns, ew, ann)
    ax.scatter([ews["ann_vol"]], [ews["ann_return"]], marker="x", s=120, label="Equal Weight")

    iv = inv_vol_weight(returns)
    ivs = portfolio_stats(returns, iv, ann)
    ax.scatter([ivs["ann_vol"]], [ivs["ann_return"]], marker="^", s=120, label="Inv-Vol")

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Portfolio Cloud (Random Long-Only Weights)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Combine multiple strategy return streams and optimize portfolio weights.")
    ap.add_argument("--symbols", required=False, default=None, help="Comma-separated tickers; default uses Config.FOREX_PAIRS")
    ap.add_argument("--interval", default=None, help="Data interval (e.g., 1h, 1d). Defaults to Config.TIMEFRAME")
    ap.add_argument("--period", default=None, help="yfinance period (e.g., 6mo, 1y). Defaults to Config.PERIOD")
    ap.add_argument("--samples", type=int, default=5000, help="Number of random portfolios to sample for Max-Sharpe")
    args = ap.parse_args()

    symbols = args.symbols.split(",") if args.symbols else Config.FOREX_PAIRS
    interval = args.interval or Config.TIMEFRAME
    period = args.period or Config.PERIOD
    n_samples = int(args.samples)

    log.info(f"Building returns matrix for: {symbols} (interval={interval}, period={period})")
    returns_df, per_symbol = build_returns_matrix(symbols, period=period, interval=interval)

    # Save per-symbol stats
    os.makedirs(Config.REPORTS_DIR, exist_ok=True)
    sym_stats_path = os.path.join(Config.REPORTS_DIR, "portfolio_symbol_stats.csv")
    pd.DataFrame.from_dict(per_symbol, orient="index").to_csv(sym_stats_path)
    log.info(f"Saved per-symbol stats → {sym_stats_path}")

    # Equal-weight
    ew = equal_weight(returns_df.shape[1])
    ew_stats = portfolio_stats(returns_df, ew, annualizer_from_interval(interval))
    log.info(f"Equal-Weight: {ew_stats}")

    # Inverse-vol
    iv = inv_vol_weight(returns_df)
    iv_stats = portfolio_stats(returns_df, iv, annualizer_from_interval(interval))
    log.info(f"Inverse-Vol: {iv_stats}")

    # Max-Sharpe via random search
    best_w, best_stats = max_sharpe_dirichlet(returns_df, n_samples=n_samples)
    log.info(f"Max-Sharpe (random search): {best_stats}")

    # Save weights table
    weights_df = pd.DataFrame({
        "symbol": returns_df.columns,
        "equal_weight": ew,
        "inv_vol": iv,
        "max_sharpe": best_w
    })
    weights_path = os.path.join(Config.REPORTS_DIR, "portfolio_weights.csv")
    weights_df.to_csv(weights_path, index=False)
    log.info(f"Saved weight table → {weights_path}")

    # Save scatter cloud
    cloud_path = os.path.join(Config.REPORTS_DIR, "portfolio_cloud.png")
    scatter_portfolios(returns_df, cloud_path)
    log.info(f"Saved portfolio cloud → {cloud_path}")

    # Save combined portfolio equity (for the max-sharpe solution)
    rp = (returns_df.values @ best_w).ravel()
    eq = pd.Series((1 + rp)).cumprod()
    eq.index = returns_df.index
    eq_path = os.path.join(Config.REPORTS_DIR, "portfolio_equity.csv")
    eq.to_csv(eq_path, header=["equity"])
    log.info(f"Saved portfolio equity path → {eq_path}")

if __name__ == "__main__":
    main()
