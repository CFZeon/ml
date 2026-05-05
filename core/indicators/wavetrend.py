"""WaveTrend oscillator indicator."""

import numpy as np

from .base import Indicator
from .registry import register_indicator


@register_indicator
class WaveTrendOscillator(Indicator):
    kind = "wavetrend"
    required_columns = ("high", "low", "close")

    def __init__(self, channel_length=10, average_length=21, signal_length=4, name=None):
        self.channel_length = int(channel_length)
        self.average_length = int(average_length)
        self.signal_length = int(signal_length)
        super().__init__(name=name)

    def default_name(self):
        return f"wt_{self.channel_length}_{self.average_length}_{self.signal_length}"

    def params(self):
        return {
            "channel_length": self.channel_length,
            "average_length": self.average_length,
            "signal_length": self.signal_length,
        }

    def compute(self, df):
        hlc3 = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
        esa = hlc3.ewm(span=self.channel_length, adjust=False, min_periods=self.channel_length).mean()
        deviation = (hlc3 - esa).abs().ewm(
            span=self.channel_length,
            adjust=False,
            min_periods=self.channel_length,
        ).mean()
        ci = (hlc3 - esa).div((0.015 * deviation).replace(0.0, np.nan))
        wt1 = ci.ewm(span=self.average_length, adjust=False, min_periods=self.average_length).mean()
        wt2 = wt1.rolling(self.signal_length, min_periods=self.signal_length).mean()
        spread = wt1 - wt2
        return {
            f"{self.name}_wt1": wt1,
            f"{self.name}_wt2": wt2,
            f"{self.name}_spread": spread,
        }