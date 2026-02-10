from __future__ import annotations
import os
import pandas as pd
import yfinance as yf

from config.settings import Config
from utils.logger import get_logger

log = get_logger(__name__)


class DataLoader:
    def __init__(self, data_dir: str = Config.DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _cache_path(self, symbol: str, period: str, interval: str) -> str:
        fname = f"{symbol.replace('=','_').replace('/','_')}_{interval}_{period}.csv"
        return os.path.join(self.data_dir, fname)

    def download(self, symbol: str, period: str, interval: str, force_refresh: bool = False) -> pd.DataFrame:
        path = self._cache_path(symbol, period, interval)
        if os.path.exists(path) and not force_refresh and Config.CACHE:
            log.info(f"Loading cached: {path}")
            return pd.read_csv(path, index_col=0, parse_dates=True)

        # Yahoo keeps intraday (<=1h) only ~60d — auto-clamp to avoid empty results
        intraday = {"1m","2m","5m","15m","30m","60m","90m","1h"}
        if interval.lower() in intraday and period not in {"1d","5d","60d"}:
            log.info(f"[DataLoader] Adjusting period {period} → 60d for {interval} (Yahoo intraday limit).")
            period = "60d"

        log.info(f"Downloading {symbol} period={period} interval={interval} ...")
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval)

        if df is None or df.empty:
            raise ValueError(f"No data for {symbol} ({period} {interval})")

        df = df.dropna()
        df.to_csv(path)
        log.info(f"Cached → {path}")
        return df