"""Combined derivatives interaction indicator."""

import numpy as np
import pandas as pd

from .base import Indicator
from .registry import register_indicator


@register_indicator
class DerivativesCombined(Indicator):
    kind = "derivatives_combined"
    required_columns = ("close",)

    def __init__(
        self,
        funding_prefix="funding_ctx",
        oi_prefix="oi_ctx",
        funding_window=9,
        oi_change_horizon="4h",
        volatility_window=20,
        name=None,
    ):
        self.funding_prefix = funding_prefix
        self.oi_prefix = oi_prefix
        self.funding_window = int(funding_window)
        self.oi_change_horizon = str(oi_change_horizon)
        self.volatility_window = int(volatility_window)
        super().__init__(name=name)

    def default_name(self):
        return "deriv_combo"

    def params(self):
        return {
            "funding_prefix": self.funding_prefix,
            "oi_prefix": self.oi_prefix,
            "funding_window": self.funding_window,
            "oi_change_horizon": self.oi_change_horizon,
            "volatility_window": self.volatility_window,
        }

    def _oi_change_label(self):
        return self.oi_change_horizon.strip().lower().replace(" ", "")

    def validate(self, df):
        super().validate(df)
        required = [
            f"{self.funding_prefix}_delta",
            f"{self.funding_prefix}_z_{self.funding_window}",
            f"{self.oi_prefix}_log_change_{self._oi_change_label()}",
            f"{self.oi_prefix}_trend_spread",
            f"{self.oi_prefix}_pressure_vs_notional_volume",
        ]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"{self.kind} requires columns {missing}, got {list(df.columns)}")

    def compute(self, df):
        close = df["close"].astype(float)
        returns = close.pct_change().fillna(0.0)
        realized_vol = returns.rolling(self.volatility_window, min_periods=2).std(ddof=0).fillna(0.0)
        funding_delta = df[f"{self.funding_prefix}_delta"].astype(float).fillna(0.0)
        funding_z = df[f"{self.funding_prefix}_z_{self.funding_window}"].astype(float).fillna(0.0)
        oi_log_change = df[f"{self.oi_prefix}_log_change_{self._oi_change_label()}"]
        oi_log_change = oi_log_change.astype(float).fillna(0.0)
        oi_trend_spread = df[f"{self.oi_prefix}_trend_spread"].astype(float).fillna(0.0)
        oi_pressure = df[f"{self.oi_prefix}_pressure_vs_notional_volume"].astype(float).fillna(0.0)

        outputs = {
            f"{self.name}_funding_delta_x_return": funding_delta * returns,
            f"{self.name}_oi_log_change_x_vol": oi_log_change * realized_vol,
            f"{self.name}_price_up_oi_trend_up": ((returns > 0.0) & (oi_trend_spread > 0.0)).astype(float),
            f"{self.name}_price_down_oi_trend_up": ((returns < 0.0) & (oi_trend_spread > 0.0)).astype(float),
            f"{self.name}_funding_delta_return_sign": np.sign(funding_delta) * np.sign(returns),
            f"{self.name}_crowding_proxy": funding_z * oi_log_change,
            f"{self.name}_funding_x_oi_pressure": funding_delta * oi_pressure,
        }
        return outputs