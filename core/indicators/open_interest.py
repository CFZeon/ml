"""Open-interest non-level context indicator."""

import numpy as np
import pandas as pd

from . import _derivatives_binance as derivatives_support
from .base import Indicator
from .registry import register_indicator


@register_indicator
class OpenInterestContext(Indicator):
    kind = "open_interest"
    required_columns = ("close", "volume")

    def __init__(
        self,
        symbol=None,
        period="5m",
        change_horizon="4h",
        trend_short_span=12,
        trend_long_span=48,
        warmup="2D",
        max_age="30m",
        min_coverage=0.5,
        cache_dir=".cache",
        allow_exact_matches=True,
        name=None,
    ):
        self.symbol = symbol
        self.period = str(period)
        self.change_horizon = str(change_horizon)
        self.trend_short_span = int(trend_short_span)
        self.trend_long_span = int(trend_long_span)
        self.warmup = str(warmup)
        self.max_age = str(max_age)
        self.min_coverage = float(min_coverage)
        self.cache_dir = cache_dir
        self.allow_exact_matches = bool(allow_exact_matches)
        self._last_report = {}
        if self.trend_short_span < 1:
            raise ValueError("trend_short_span must be >= 1")
        if self.trend_long_span < self.trend_short_span:
            raise ValueError("trend_long_span must be >= trend_short_span")
        super().__init__(name=name)

    def default_name(self):
        return "oi_ctx"

    def params(self):
        return {
            "symbol": self.symbol,
            "period": self.period,
            "change_horizon": self.change_horizon,
            "trend_short_span": self.trend_short_span,
            "trend_long_span": self.trend_long_span,
            "warmup": self.warmup,
            "max_age": self.max_age,
            "min_coverage": self.min_coverage,
            "cache_dir": self.cache_dir,
            "allow_exact_matches": self.allow_exact_matches,
        }

    def describe(self, outputs):
        metadata = super().describe(outputs)
        metadata.update({"alignment": dict(self._last_report)})
        return metadata

    def _change_label(self):
        return self.change_horizon.strip().lower().replace(" ", "")

    def compute(self, df):
        symbol = derivatives_support.resolve_symbol(df, self.symbol)
        base_index = pd.DatetimeIndex(df.index)
        aligned_index = derivatives_support.ensure_utc_index(base_index)
        warmup = pd.Timedelta(self.warmup)
        period_delta = derivatives_support.period_to_timedelta(self.period)
        horizon_delta = pd.Timedelta(self.change_horizon)
        horizon_steps = max(1, int(np.ceil(horizon_delta / period_delta)))
        start_dt = aligned_index.min() - warmup
        end_dt = aligned_index.max() + pd.Timedelta(milliseconds=1)

        open_interest_frame = derivatives_support.fetch_open_interest_history(
            symbol,
            self.period,
            start_dt,
            end_dt,
            cache_dir=self.cache_dir,
        )
        open_interest_context = open_interest_frame.copy()
        oi_notional = pd.to_numeric(open_interest_context.get("sumOpenInterestValue"), errors="coerce")
        oi_notional = oi_notional.where(oi_notional > 0.0)
        log_oi_notional = np.log(oi_notional)
        open_interest_context["oi_log_change"] = log_oi_notional - log_oi_notional.shift(horizon_steps)
        open_interest_context["oi_trend_spread"] = (
            log_oi_notional.ewm(span=self.trend_short_span, adjust=False).mean()
            - log_oi_notional.ewm(span=self.trend_long_span, adjust=False).mean()
        )
        open_interest_context["oi_notional_delta"] = pd.to_numeric(
            open_interest_context.get("sumOpenInterestValue"), errors="coerce"
        ).diff()

        aligned, report = derivatives_support.align_context_frame(
            base_index,
            open_interest_context,
            value_columns=["oi_log_change", "oi_trend_spread", "oi_notional_delta"],
            max_age=self.max_age,
            allow_exact_matches=self.allow_exact_matches,
        )
        if report["coverage"] < self.min_coverage:
            raise RuntimeError(
                f"Open-interest coverage too low for {symbol}: {report['coverage']:.2%} < {self.min_coverage:.2%}"
            )

        oi_log_change = aligned["oi_log_change"].astype(float)
        oi_trend_spread = aligned["oi_trend_spread"].astype(float)
        traded_notional = df["close"].astype(float) * df["volume"].astype(float)
        oi_pressure = aligned["oi_notional_delta"].astype(float).div(traded_notional.replace(0.0, np.nan))
        oi_pressure = oi_pressure.replace([np.inf, -np.inf], np.nan)
        oi_age_minutes = aligned["source_age"] / pd.Timedelta(minutes=1)
        change_label = self._change_label()

        self._last_report = {
            **report,
            "symbol": symbol,
            "period": self.period,
            "change_horizon": self.change_horizon,
            "horizon_steps": horizon_steps,
            "trend_short_span": self.trend_short_span,
            "trend_long_span": self.trend_long_span,
            "window_start": start_dt,
            "window_end": end_dt,
        }

        outputs = {
            f"{self.name}_log_change_{change_label}": oi_log_change,
            f"{self.name}_trend_spread": oi_trend_spread,
            f"{self.name}_pressure_vs_notional_volume": oi_pressure,
            f"{self.name}_age_minutes": oi_age_minutes,
        }
        return {column: pd.Series(series, index=base_index) for column, series in outputs.items()}