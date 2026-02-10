# 📖 Environment Configuration Guide

This document explains each variable in your `.env` file:  
- **What it does**  
- **Where it attaches (which module uses it)**  
- **Why it matters (impact on trading)**  

---

## 🔹 DATA & GENERAL
| Variable | Used In | Description | Why It Matters |
|----------|---------|-------------|----------------|
| `SYMBOL` | Backtests, Data Fetch | Default single symbol for quick tests when you don’t override with CLI or `LIVE_SYMBOLS`. | Ensures your scripts always have a fallback symbol. |
| `FOREX_PAIRS` | Backtests, Portfolio | List of FX pairs (Yahoo format with `=X`) used for training/testing. | Defines your research universe. |
| `LIVE_SYMBOLS` | `live_runner.py` | Comma-separated list of symbols to trade in live mode. | Controls which assets get real-time signals. |
| `TIMEFRAME` | Strategy, Data Fetch | Bar interval for signals. Supports `1m, 5m, 15m, 1h, 1d`. | Smaller = more trades, bigger = slower signals. |
| `PERIOD` | Data Fetch | Lookback window for yfinance history. E.g. `60d` (intraday max). | Controls how much historical context your strategy sees. |
| `TRAIN_BARS`, `TEST_BARS` | Backtests | How many bars are used in train/test splits. | Affects backtest quality and validation. |
| `DATA_DIR`, `REPORTS_DIR` | File I/O | Where datasets and reports are saved. | Keeps output organized. |
| `CACHE` | Data Utils | Enable/disable local caching of yfinance data. | Saves API calls and speeds up reruns. |

---

## 🔹 STRATEGY PARAMETERS
| Variable | Used In | Description | Why It Matters |
|----------|---------|-------------|----------------|
| `LOOKBACK` | Mean Reversion | Window length (bars) for z-score calculation. | Short = reactive, Long = smoother. |
| `Z_ENTRY` | Mean Reversion | Z-score threshold to **enter** trade. (e.g. `0.8`) | Higher = fewer trades, stronger signals. |
| `Z_EXIT` | Mean Reversion | Z-score threshold to **exit** trade. (e.g. `0.0`). | Controls when you take profits/flatten. |
| `STOP_LOSS` | Risk Logic | Fractional stop-loss (0.01 = 1%). | Caps max downside per trade. |
| `TAKE_PROFIT` | Risk Logic | Fractional take-profit (0.02 = 2%). | Locks in upside when hit. |
| `USE_RSI` | Strategy Filter | `1/0` toggle for RSI filter. | Adds momentum filter to reduce false entries. |
| `RSI_PERIOD` | RSI Calc | Lookback for RSI. Default = 14. | Shorter = choppier signals. |
| `RSI_OVERBOUGHT` / `RSI_OVERSOLD` | RSI Calc | Thresholds for long/short filter. | Adds confirmation to mean reversion. |

---

## 🔹 SIZING
| Variable | Used In | Description | Why It Matters |
|----------|---------|-------------|----------------|
| `SIZING_METHOD` | Strategy → Broker | `fixed` = fixed fraction; `atr` = volatility-based. | Defines how trade sizes are chosen. |
| `RISK_PER_TRADE` | Sizer | Fraction of equity at risk per trade. (e.g. `0.02` = 2%). | Central risk knob. |
| `ATR_PERIOD`, `ATR_MULT` | ATR Sizer | Only used when `SIZING_METHOD=atr`. | Scales position size by volatility. |

---

## 🔹 COSTS & CAPITAL
| Variable | Used In | Description | Why It Matters |
|----------|---------|-------------|----------------|
| `INITIAL_CAPITAL` | PaperBroker / MT5 stub | Starting cash for paper/live calcs. | Baseline account equity. |
| `FEES_BPS` | Execution | Commission per trade in basis points (1.0 = 0.01%). | Models transaction costs. |
| `SLIPPAGE_BPS` | Execution | Simulated slippage in basis points. | Models realistic execution. |
| `MAX_POSITIONS` | Portfolio Manager | Max concurrent open trades. | Prevents overexposure. |

---

## 🔹 EXECUTION
| Variable | Used In | Description | Why It Matters |
|----------|---------|-------------|----------------|
| `USE_ALPACA` | Broker Selector | `true/false` toggle for Alpaca. | Switch between Alpaca and other brokers. |
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL` | AlpacaBroker | Live/paper trading API credentials. | Required if using Alpaca. |
| `BROKER` | `live_runner.py` | e.g. `mt5_bridge` | Chooses broker backend. |
| `MT5_BRIDGE_HOST`, `MT5_BRIDGE_PORT` | MT5BridgeBroker | HTTP bridge config for EA. | Where Python pushes orders. |
| `MT5_SYMBOL_SUFFIX` | Symbol Mapper | Adds broker suffix like `.m` if needed. | Ensures symbols match MT5 broker. |
| `MT5_MIN_LOTS` | MT5BridgeBroker | Minimum allowed trade size. | Prevents too-small orders. |
| `MT5_MAX_LOTS_PER_TRADE` | MT5BridgeBroker | Cap per single order. | Protects against oversized lots. |
| `MT5_LOTS_PER_100USD` | MT5BridgeBroker | Scaling: how many lots per $100 equity. | Ensures lot sizing respects account balance. |
| `LIVE_POLL_SECONDS` | Live Runner | Sleep between loops. | Controls how often new trades are checked. |

---

## 🔹 OPTIMIZATION / DEBUG
| Variable | Used In | Description | Why It Matters |
|----------|---------|-------------|----------------|
| `DEBUG_WFO` | WFO Runs | Print extra diagnostics. | Debugging only. |
| `AUTO_RELAX` | WFO/WFP | Auto-adjust thresholds if no trades. | Prevents “dead” test slices. |
| `WFP_WEIGHT_RULE` | Walk-Forward Portfolio | `equal` or `inv_vol`. | Chooses weighting scheme. |

---

## 🔹 LOGGING
| Variable | Used In | Description | Why It Matters |
|----------|---------|-------------|----------------|
| `LOG_LEVEL` | Logger | e.g. `INFO`, `DEBUG`. | Controls verbosity. |
| `LOG_TO_FILE` | Logger | 1/0 toggle. | Enable log persistence. |
| `LOG_FILE` | Logger | Path to log file. | Keeps history of runs. |

---

## 🔹 LIVE CAPITAL CAPS
These apply only in **live trading**. The system takes the **minimum** of these caps:  
| Variable | Description | Example |
|----------|-------------|---------|
| `MAX_LIVE_CAPITAL_FRACTION` | Max fraction of broker’s buying power allowed. | 0.1 → 10% of margin used |
| `MAX_LIVE_CAPITAL` | Absolute USD cap. | 100 → no trade bigger than $100 |
| `PER_TRADE_NOTIONAL_CAP` | Per-trade absolute USD cap. | 50 → max $50 per trade |
| `PER_TRADE_EQUITY_FRACTION` | Max fraction of equity per trade. | 0.15 → 15% of equity |
