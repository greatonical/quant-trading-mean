# If you haven’t already:
pip install -r requirements.txt
cp .env.example .env

# (Optional) choose sizing style in .env:
# SIZING_METHOD=fixed
# or
# SIZING_METHOD=atr

# Backtest one symbol
python -m backtests.run_backtest --symbol EURUSD=X --interval 1h

# Multi-symbol main (uses FOREX_PAIRS from .env)
python main.py

# Optimize params including SL/TP (sized returns now)
python -m optimize.mean_reversion_optimizer --symbol EURUSD=X --metric sharpe --train-frac 0.7

# Portfolio optimization across multiple symbols (sized returns)
python -m optimize.portfolio_optimizer --symbols EURUSD=X,GBPUSD=X,USDJPY=X --samples 8000

# uses the powerful mean_reversion_optimizer under the hood
python -m optimize.grid_search --symbol EURUSD=X --metric sharpe --train-frac 0.7


# Example: 2000 bars train, 500 bars test on 1h data
python -m optimize.walk_forward \
  --symbol EURUSD=X \
  --interval 1h \
  --period 2y \
  --train-bars 2000 \
  --test-bars 500 \
  --lookbacks 10,15,20,30 \
  --z-entries 1.5,2.0,2.5 \
  --z-exits 0.0,0.2,0.5 \
  --stop-losses 0.01,0.02 \
  --take-profits 0.02,0.04 \
  --metric sharpe

