from __future__ import annotations
import numpy as np
import pandas as pd

from config.settings import Config


class PositionSizer:
    """
    Produces a per‑bar 'size' (0..1): fraction of equity to allocate while in a trade.

    Methods:
      - 'fixed': size = min(1, RISK_PER_TRADE / stop_loss_pct)
      - 'atr'  : size = min(1, RISK_PER_TRADE / ((ATR_MULT * ATR) / entry_price))

    Vectorized, constant per trade_id, 0 when flat.
    """

    def __init__(self,
                 method: str = Config.SIZING_METHOD,
                 risk_per_trade: float = Config.RISK_PER_TRADE,
                 stop_loss_pct: float = Config.STOP_LOSS,
                 atr_mult: float = Config.ATR_MULT):
        self.method = method.lower()
        self.risk_per_trade = float(risk_per_trade)
        self.stop_loss_pct = float(stop_loss_pct)
        self.atr_mult = float(atr_mult)

    def compute(self,
                df: pd.DataFrame,
                trade_id_col: str = "trade_id",
                entry_price_col: str = "entry_price",
                atr_col: str | None = "ATR") -> pd.Series:
        """
        Returns a Series 'size' aligned with df.index, 0 when flat.
        Vectorized: compute one constant size per trade and broadcast.
        """
        size = pd.Series(0.0, index=df.index)

        if trade_id_col not in df or entry_price_col not in df:
            return size

        trade_id = df[trade_id_col]
        in_trade = trade_id.notna()
        if not in_trade.any():
            return size

        if self.method not in {"fixed", "atr"}:
            raise ValueError("SIZING_METHOD must be 'fixed' or 'atr'")

        if self.method == "fixed":
            stop_pct = max(self.stop_loss_pct, 1e-6)
            s_const = self.risk_per_trade / stop_pct
            s_const = max(min(s_const, 1.0), 0.0)
            size.loc[in_trade] = s_const
            return size

        # ATR‑based sizing
        if atr_col is None or atr_col not in df:
            return size

        atr_entry = df.groupby(trade_id)[atr_col].transform("first")
        ep = df[entry_price_col]
        pct_atr = (self.atr_mult * atr_entry) / ep
        pct_atr = pct_atr.replace([np.inf, -np.inf], np.nan)

        s = (self.risk_per_trade / pct_atr).clip(lower=0.0, upper=1.0)
        size = s.where(in_trade, 0.0).fillna(0.0)
        return size