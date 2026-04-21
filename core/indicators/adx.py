"""Average Directional Index indicator."""

import numpy as np
import pandas as pd

from .base import Indicator
from .registry import register_indicator


@register_indicator
class ADX(Indicator):
    kind = "adx"
    required_columns = ("high", "low", "close")

    def __init__(self, period=14, name=None):
        self.period = period
        super().__init__(name=name)

    def default_name(self):
        return f"adx_{self.period}"

    def compute(self, df):
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)

        true_range = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        tr_smooth = true_range.ewm(alpha=1 / self.period, min_periods=self.period).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1 / self.period, min_periods=self.period).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1 / self.period, min_periods=self.period).mean()

        tr_safe = tr_smooth.replace(0.0, np.nan)
        plus_di = 100.0 * plus_dm_smooth / tr_safe
        minus_di = 100.0 * minus_dm_smooth / tr_safe

        di_sum = (plus_di + minus_di).replace(0.0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        adx = dx.ewm(alpha=1 / self.period, min_periods=self.period).mean()
        return {
            self.name: adx,
            f"{self.name}_plus_di": plus_di,
            f"{self.name}_minus_di": minus_di,
        }