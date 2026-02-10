# optimize/grid_search.py
from __future__ import annotations

# --- Dynamic project-root resolver (run from anywhere) ---
import os, sys
_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))  # .../quant-trading-mean
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# ---------------------------------------------------------

import argparse
from optimize.mean_reversion_optimizer import main as mr_opt_main

def main():
    """
    Thin wrapper that delegates to mean_reversion_optimizer.main()
    so you can run:  python -m optimize.grid_search --symbol EURUSD=X ...
    """
    mr_opt_main()

if __name__ == "__main__":
    main()
