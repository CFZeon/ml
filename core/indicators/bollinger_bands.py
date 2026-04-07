"""Bollinger Bands indicator."""

from .base import Indicator
from .registry import register_indicator


def _slug(value):
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace("-", "n").replace(".", "p")


@register_indicator
class BollingerBands(Indicator):
    kind = "bollinger"
    required_columns = ("close",)

    def __init__(self, period=20, num_std=2.0, name=None):
        self.period = period
        self.num_std = num_std
        super().__init__(name=name)

    def default_name(self):
        return f"bb_{self.period}_{_slug(self.num_std)}"

    def compute(self, df):
        mid = df["close"].rolling(self.period).mean()
        std = df["close"].rolling(self.period).std()
        upper = mid + self.num_std * std
        lower = mid - self.num_std * std
        width = upper - lower
        return {
            f"{self.name}_upper": upper,
            f"{self.name}_lower": lower,
            f"{self.name}_pctb": (df["close"] - lower) / width,
            f"{self.name}_bw": width / mid,
        }