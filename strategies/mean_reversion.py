from __future__ import annotations
import os
import numpy as np
import pandas as pd

from config.settings import Config
from utils.indicators import zscore, rsi, bollinger_bands, sma, atr
from utils.position_sizer import PositionSizer


class MeanReversionStrategy:
    """
    Vectorized mean‑reversion strategy with:
      • Z‑score entries/exits (with optional RSI gates)
      • Vectorized SL/TP + mean reversion exits
      • Position sizing (fixed or ATR‑based)
      • Transaction costs
    """

    def __init__(self,
                 lookback: int = Config.LOOKBACK,
                 z_entry: float = Config.Z_ENTRY,
                 z_exit: float = Config.Z_EXIT,
                 fee_bps: float = Config.FEE_BPS,
                 slippage_bps: float = Config.SLIPPAGE_BPS,
                 stop_loss_pct: float = Config.STOP_LOSS,
                 take_profit_pct: float = Config.TAKE_PROFIT,
                 sizing_method: str = Config.SIZING_METHOD,
                 atr_period: int = Config.ATR_PERIOD,
                 atr_mult: float = Config.ATR_MULT,
                 use_rsi: bool = True):
        self.lookback = int(lookback)
        self.z_entry  = float(z_entry)
        self.z_exit   = float(z_exit)
        self.cost_bps = (float(fee_bps) + float(slippage_bps)) / 10_000.0
        self.stop_loss_pct   = float(stop_loss_pct)
        self.take_profit_pct = float(take_profit_pct)
        self.sizer = PositionSizer(method=sizing_method,
                                   risk_per_trade=Config.RISK_PER_TRADE,
                                   stop_loss_pct=self.stop_loss_pct,
                                   atr_mult=atr_mult)
        self.atr_period = int(atr_period)
        self.atr_mult   = float(atr_mult)
        self.use_rsi    = bool(use_rsi)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # === Indicators ===
        out['Z']   = zscore(out['Close'], window=self.lookback)
        out['RSI'] = rsi(out['Close'], window=14)
        out['BBU'], out['BBM'], out['BBL'] = bollinger_bands(out['Close'], window=self.lookback, std_dev=2.0)
        out['SMA'] = sma(out['Close'], window=self.lookback)
        out['ATR'] = atr(out['High'], out['Low'], out['Close'], window=self.atr_period)

        # === Entry/exit conditions (Z-score core) ===
        long_entry_mask   = out['Z'] < -self.z_entry
        long_exit_mask    = out['Z'] >  self.z_exit
        short_entry_mask  = out['Z'] >  self.z_entry
        short_exit_mask   = out['Z'] < -self.z_exit

        # Optional RSI gates
        if self.use_rsi:
            long_entry_mask   &= out['RSI'] <= 70.0
            short_entry_mask  &= out['RSI'] >= 30.0

        # === State machine (preserve explicit exits) ===
        pos_signal = pd.Series(np.nan, index=out.index, dtype=float)
        pos_signal[long_entry_mask]  =  1.0
        pos_signal[short_entry_mask] = -1.0
        # Exits must not overwrite entry bars: give entries precedence
        exit_mask = (long_exit_mask | short_exit_mask) & ~(long_entry_mask | short_entry_mask)
        pos_signal[exit_mask] = 0.0
        out['position_pre'] = pos_signal.ffill().fillna(0.0).astype(int)

        # === Trade IDs & entry prices ===
        prev_pos = out['position_pre'].shift(1).fillna(0).astype(int)
        entry_flag = (out['position_pre'] != 0) & (prev_pos == 0)
        trade_id = entry_flag.cumsum()
        trade_id = trade_id.where(out['position_pre'] != 0, np.nan)  # NaN when flat
        out['trade_id'] = trade_id
        out['entry_price'] = out.groupby('trade_id')['Close'].transform('first')

        # === Vectorized risk exits (SL/TP + mean reversion exit) ===
        pnl_long  = out['Close'] / out['entry_price'] - 1.0
        pnl_short = out['entry_price'] / out['Close'] - 1.0
        is_long   = out['position_pre'] ==  1
        is_short  = out['position_pre'] == -1

        long_stop  = is_long  & (pnl_long  <= -self.stop_loss_pct)
        long_take  = is_long  & (pnl_long  >=  self.take_profit_pct)
        short_stop = is_short & (pnl_short <= -self.stop_loss_pct)
        short_take = is_short & (pnl_short >=  self.take_profit_pct)

        z_exit_long  = is_long  & (out['Z'] >  self.z_exit)
        z_exit_short = is_short & (out['Z'] < -self.z_exit)

        combined_exit = (long_stop | long_take | short_stop | short_take | z_exit_long | z_exit_short).astype(bool)

        # First exit in each trade (dtype‑safe)
        ce_int  = combined_exit.astype(int)
        ce_rank = ce_int.groupby(out['trade_id']).cumsum()
        first_exit_flag = combined_exit & (ce_rank == 1)
        out['exit_flag'] = first_exit_flag.fillna(False)

        # Deactivate positions after first exit per trade
        exit_cum = out.groupby('trade_id')['exit_flag'].cumsum().fillna(0)
        active = (exit_cum == 0)
        out['position'] = np.where(active, out['position_pre'], 0).astype(int)

        # === Position sizing (vectorized; constant per trade) ===
        out['size'] = self.sizer.compute(out, trade_id_col='trade_id',
                                         entry_price_col='entry_price', atr_col='ATR')

        # === Returns & costs (no look‑ahead) ===
        out['ret'] = out['Close'].pct_change().fillna(0.0)
        out['trade_change'] = out['position'].diff().fillna(out['position']).abs()
        out['cost'] = out['trade_change'] * self.cost_bps * out['size'].fillna(0.0)
        out['strategy_ret'] = (out['position'].shift(1).fillna(0)
                               * out['size'].shift(1).fillna(0.0)
                               * out['ret']) - out['cost']

        # === Diagnostics (opt‑in with DEBUG_WFO=1) ===
        if os.getenv("DEBUG_WFO", "0") == "1":
            le_raw = (out['Z'] < -self.z_entry).sum()
            se_raw = (out['Z'] >  self.z_entry).sum()
            le     = long_entry_mask.sum()
            se     = short_entry_mask.sum()
            realized_entries = ((out['position'] != 0) & (out['position'].shift(1).fillna(0) == 0)).sum()
            print(f"[STRAT-DIAG] z_entry={self.z_entry}, z_exit={self.z_exit}, lookback={self.lookback} | "
                  f"raw_le={int(le_raw)}, raw_se={int(se_raw)}, le={int(le)}, se={int(se)}, "
                  f"realized_entries={int(realized_entries)} | size_mean={out['size'].mean():.3f}")

        return out.drop(columns=['position_pre'])