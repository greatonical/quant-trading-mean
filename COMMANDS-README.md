# Quant Trading Mean-Reversion — Commands Cheat Sheet

This file lists all the important commands used during development, testing, and live trading.  
Comments (`# ...`) explain what each command does.

---

## 1. Environment Setup (macOS/Linux)

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Alpaca SDK (for equities/crypto trading)
pip install alpaca-py

# 4. (Optional) Upgrade yfinance + websockets if warnings appear
pip install --upgrade yfinance websockets

# 5. Copy and configure environment variables
cp .env.example .env
# Then open `.env` and edit values (API keys, symbols, broker, etc.)
```

---

## 2. Backtesting

```bash
# Run a single-symbol backtest (EURUSD=X, 1h candles)
python -m backtests.run_backtest --symbol EURUSD=X --interval 1h
```

---

## 3. Grid Search (parameter sweep)

```bash
# Example grid search for optimal strategy params
python -m optimize.grid_search --symbol EURUSD=X --interval 1h --period 6mo \
  --lookbacks 10,20,30 --z-entries 1.0,1.5,2.0 --z-exits 0.0,0.2 \
  --stop-losses 0.01,0.02 --take-profits 0.02,0.04 \
  --metric sharpe --out reports/grid_results.csv
```

---

## 4. Walk-Forward Optimization (WFO)

```bash
# Rolling-window optimization and testing
python -m optimize.walk_forward \
  --symbol EURUSD=X \
  --interval 1h \
  --period 1y \
  --train-bars 1500 \
  --test-bars 300 \
  --lookbacks 10,20,30 \
  --z-entries 1.5,2.0 \
  --z-exits 0.0,0.2 \
  --stop-losses 0.01,0.02 \
  --take-profits 0.02,0.04 \
  --metric sharpe \
  --max-windows 10
```

---

## 5. Live Trading with Alpaca

```bash
# Run live trading loop with Alpaca (requires USE_ALPACA=true in .env)
python3 -m execution.live_runner
```

`.env` must contain:

```ini
USE_ALPACA=true
ALPACA_API_KEY=yourkey
ALPACA_SECRET_KEY=yoursecret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

---

## 6. Live Trading with MetaTrader 5 Bridge

### 6.1 Start Bridge Server

```bash
# Start the Flask bridge server (provides /enqueue, /next, /event endpoints)
python -m bridge.server
```

---

### 6.2 Run Live Runner with MT5 Bridge

```bash
# Run live trading loop, broker set to mt5_bridge
BROKER=mt5_bridge BRIDGE_URL=http://127.0.0.1:5000 \
python3 -m execution.live_runner
```

> In MT5:  
> - Attach `EA_Executor.mq5` to a chart  
> - Enable **Algo Trading**  
> - Allow `http://127.0.0.1:5000` in **Tools → Options → Expert Advisors → Allow WebRequest**

---

### 6.3 Smoke Test with Curl

```bash
# Send a manual order into the bridge queue
curl -X POST --data-binary 'cmd=place;order_ref=smoke1;symbol=EURUSD;side=BUY;lots=0.10' \
     -H 'Content-Type: text/plain' http://127.0.0.1:5000/enqueue
# EA should ACK then FILLED in bridge logs and MT5 terminal
```

---

## 7. Logs & Debugging

```bash
# Tail application logs
tail -f logs/app.log

# View bridge logs (shows dequeue and EA events)
python -m bridge.server

# Check EA callback events (CSV file)
tail -f bridge/live_events.csv
```

---

## 8. Cleanups

```bash
# Remove cached historical data
rm -f data/*.csv

# Rebuild environment if broken
deactivate || true
rm -rf .venv
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

---

## 9. One-Liner Recap

```bash
# Live trading (Alpaca)
python3 -m execution.live_runner

# Live trading (MT5 Bridge)
BROKER=mt5_bridge BRIDGE_URL=http://127.0.0.1:5000 python3 -m execution.live_runner

# Start bridge server
python -m bridge.server

# Smoke test order
curl -X POST --data-binary 'cmd=place;order_ref=smoke1;symbol=EURUSD;side=BUY;lots=0.10' \
     -H 'Content-Type: text/plain' http://127.0.0.1:5000/enqueue

# Backtest
python -m backtests.run_backtest --symbol EURUSD=X --interval 1h

# Walk-forward optimization
python -m optimize.walk_forward --symbol EURUSD=X --interval 1h --period 1y ...
```

---

✅ With these commands, you can **set up, backtest, optimize, run live (Alpaca or MT5), and debug**.  
This file is safe to copy-paste as your project’s `README.md`.
