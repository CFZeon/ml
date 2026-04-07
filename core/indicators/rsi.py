"""Relative Strength Index indicator."""

from .base import Indicator
from .registry import register_indicator


@register_indicator
class RSI(Indicator):
    kind = "rsi"
    required_columns = ("close",)

    def __init__(self, period=14, name=None):
        self.period = period
        super().__init__(name=name)

    def default_name(self):
        return f"rsi_{self.period}"

    def compute(self, df):
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / self.period, min_periods=self.period).mean()
        avg_loss = loss.ewm(alpha=1 / self.period, min_periods=self.period).mean()
        rs = avg_gain / avg_loss
        return {self.name: 100 - 100 / (1 + rs)}