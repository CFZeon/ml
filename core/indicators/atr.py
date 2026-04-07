"""Average True Range indicator."""

import pandas as pd

from .base import Indicator
from .registry import register_indicator


@register_indicator
class ATR(Indicator):
    kind = "atr"
    required_columns = ("high", "low", "close")

    def __init__(self, period=14, name=None):
        self.period = period
        super().__init__(name=name)

    def default_name(self):
        return f"atr_{self.period}"

    def compute(self, df):
        high = df["high"]
        low = df["low"]
        close = df["close"]
        true_range = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        return {self.name: true_range.ewm(alpha=1 / self.period, min_periods=self.period).mean()}