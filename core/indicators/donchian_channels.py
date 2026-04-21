"""Donchian channel indicator."""

from .base import Indicator
from .registry import register_indicator


@register_indicator
class DonchianChannels(Indicator):
    kind = "donchian"
    required_columns = ("high", "low")

    def __init__(self, period=20, name=None):
        self.period = period
        super().__init__(name=name)

    def default_name(self):
        return f"donchian_{self.period}"

    def compute(self, df):
        upper = df["high"].rolling(self.period).max()
        lower = df["low"].rolling(self.period).min()
        mid = (upper + lower) / 2.0
        return {
            f"{self.name}_upper": upper,
            f"{self.name}_lower": lower,
            f"{self.name}_mid": mid,
        }