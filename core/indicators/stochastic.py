"""Stochastic oscillator indicator."""

from .base import Indicator
from .registry import register_indicator


@register_indicator
class StochasticOscillator(Indicator):
    kind = "stochastic"
    required_columns = ("high", "low", "close")

    def __init__(self, period=14, smooth_k=3, d_period=3, name=None):
        self.period = period
        self.smooth_k = smooth_k
        self.d_period = d_period
        super().__init__(name=name)

    def default_name(self):
        return f"stoch_{self.period}_{self.smooth_k}_{self.d_period}"

    def compute(self, df):
        highest_high = df["high"].rolling(self.period).max()
        lowest_low = df["low"].rolling(self.period).min()
        raw_k = 100.0 * (df["close"] - lowest_low) / (highest_high - lowest_low)
        k = raw_k.rolling(self.smooth_k).mean()
        d = k.rolling(self.d_period).mean()
        return {
            f"{self.name}_k": k,
            f"{self.name}_d": d,
        }