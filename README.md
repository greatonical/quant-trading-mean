# Mean-Reversion Trading Framework

A Python research-to-execution framework for systematic **mean-reversion** strategies on FX and other liquid markets. It keeps strategy research and live execution in one codebase, so a signal that works in a backtest runs through the same logic in production.

## What it does

- **Mean-reversion signals** — z-score based entries and exits: trade when price deviates far enough from its rolling mean to expect reversion, with configurable entry/exit thresholds.
- **Position sizing** — switchable between fixed sizing and ATR-based (volatility-adjusted) sizing via a single env var.
- **Backtesting** — single-symbol and multi-symbol backtests on historical price data, with sized returns (not just raw signal returns).
- **Parameter optimization** — search strategy parameters (including SL/TP) against the **Sharpe ratio**, with a train/test split to avoid fitting on the whole sample.
- **Walk-forward validation** — rolling train/test evaluation so results reflect out-of-sample behavior rather than a single lucky fit.
- **Portfolio optimization** — optimize across multiple FX pairs at once, sampling thousands of parameter combinations.
- **Grid search** — exhaustive parameter sweeps under the same optimizer.

## Stack

Python · pandas · numpy · yfinance (data) · pandas_ta (indicators, pure-Python, no TA-Lib/C toolchain) · matplotlib / plotly / quantstats (reporting & analytics) · Jupyter (optional notebooks).

## Project structure

```
strategies/   # signal logic (mean-reversion, z-score)
optimize/      # parameter search, walk-forward, grid + portfolio optimizers
execution/     # order/execution layer
bridge/        # data / platform bridge
config/        # configuration
utils/         # shared helpers
notebooks/     # research notebooks
main.py        # multi-symbol entry point
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # then set SIZING_METHOD (fixed | atr) and FOREX_PAIRS
```

Backtest one symbol:
```bash
python -m backtests.run_backtest --symbol EURUSD=X --interval 1h
```

Multi-symbol run (uses FOREX_PAIRS from .env):
```bash
python main.py
```

Optimize parameters (Sharpe, 70/30 train/test):
```bash
python -m optimize.mean_reversion_optimizer --symbol EURUSD=X --metric sharpe --train-frac 0.7
```

Portfolio optimization across symbols:
```bash
python -m optimize.portfolio_optimizer --symbols EURUSD=X,GBPUSD=X,USDJPY=X --samples 8000
```

Grid search:
```bash
python -m optimize.grid_search --symbol EURUSD=X --metric sharpe --train-frac 0.7
```

## Notes

This is a research framework: results shown in backtesting are historical and not a guarantee of live performance. Configuration (data source, pairs, sizing, risk) is driven through `.env` — see `ENV_README.md` and `COMMANDS-README.md` for the full reference.
