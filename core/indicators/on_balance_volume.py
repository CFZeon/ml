"""On-balance volume indicator."""

import numpy as np

from .base import Indicator
from .registry import register_indicator


@register_indicator
class OnBalanceVolume(Indicator):
    kind = "obv"
    required_columns = ("close", "volume")

    def default_name(self):
        return "obv"

    def compute(self, df):
        direction = np.sign(df["close"].diff()).fillna(0.0)
        obv = (direction * df["volume"].astype(float)).cumsum()
        return {self.name: obv}