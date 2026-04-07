"""MACD indicator."""

from .base import Indicator
from .registry import register_indicator


@register_indicator
class MACD(Indicator):
    kind = "macd"
    required_columns = ("close",)

    def __init__(self, fast=12, slow=26, signal=9, name=None):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        super().__init__(name=name)

    def default_name(self):
        return f"macd_{self.fast}_{self.slow}_{self.signal}"

    def compute(self, df):
        fast_ema = df["close"].ewm(span=self.fast, min_periods=self.fast).mean()
        slow_ema = df["close"].ewm(span=self.slow, min_periods=self.slow).mean()
        line = fast_ema - slow_ema
        signal_line = line.ewm(span=self.signal, min_periods=self.signal).mean()
        return {
            f"{self.name}_line": line,
            f"{self.name}_signal": signal_line,
            f"{self.name}_hist": line - signal_line,
        }