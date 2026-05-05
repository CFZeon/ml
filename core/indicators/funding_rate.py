"""Funding-rate first-derivative context indicator."""

import numpy as np
import pandas as pd

from . import _derivatives_binance as derivatives_support
from .base import Indicator
from .registry import register_indicator


@register_indicator
class FundingRateContext(Indicator):
    kind = "funding_rate"
    required_columns = ("close",)

    def __init__(
        self,
        symbol=None,
        rolling_window=9,
        warmup="10D",
        max_age="8h",
        zscore_threshold=2.0,
        min_coverage=0.5,
        cache_dir=".cache",
        allow_exact_matches=True,
        name=None,
    ):
        self.symbol = symbol
        self.rolling_window = int(rolling_window)
        self.warmup = str(warmup)
        self.max_age = str(max_age)
        self.zscore_threshold = float(zscore_threshold)
        self.min_coverage = float(min_coverage)
        self.cache_dir = cache_dir
        self.allow_exact_matches = bool(allow_exact_matches)
        self._last_report = {}
        super().__init__(name=name)

    def default_name(self):
        return "funding_ctx"

    def params(self):
        return {
            "symbol": self.symbol,
            "rolling_window": self.rolling_window,
            "warmup": self.warmup,
            "max_age": self.max_age,
            "zscore_threshold": self.zscore_threshold,
            "min_coverage": self.min_coverage,
            "cache_dir": self.cache_dir,
            "allow_exact_matches": self.allow_exact_matches,
        }

    def describe(self, outputs):
        metadata = super().describe(outputs)
        metadata.update({"alignment": dict(self._last_report)})
        return metadata

    def compute(self, df):
        symbol = derivatives_support.resolve_symbol(df, self.symbol)
        base_index = pd.DatetimeIndex(df.index)
        aligned_index = derivatives_support.ensure_utc_index(base_index)
        warmup = pd.Timedelta(self.warmup)
        start_dt = aligned_index.min() - warmup
        end_dt = aligned_index.max() + pd.Timedelta(milliseconds=1)

        funding_frame = derivatives_support.fetch_funding_history(
            symbol,
            start_dt,
            end_dt,
            cache_dir=self.cache_dir,
        )
        funding_context = funding_frame.copy()
        funding_context["funding_delta"] = (
            pd.to_numeric(funding_context.get("funding_rate"), errors="coerce")
            .diff()
            .fillna(0.0)
        )

        aligned, report = derivatives_support.align_context_frame(
            base_index,
            funding_context,
            value_columns=["funding_delta"],
            max_age=self.max_age,
            allow_exact_matches=self.allow_exact_matches,
        )
        if report["coverage"] < self.min_coverage:
            raise RuntimeError(
                f"Funding-rate coverage too low for {symbol}: {report['coverage']:.2%} < {self.min_coverage:.2%}"
            )

        funding_delta = aligned["funding_delta"].astype(float)
        funding_abs_delta = funding_delta.abs()
        funding_mean = funding_delta.rolling(self.rolling_window, min_periods=1).mean()
        funding_z = derivatives_support.rolling_zscore(
            funding_delta,
            self.rolling_window,
            min_periods=min(3, self.rolling_window),
        )
        funding_age_hours = aligned["source_age"] / pd.Timedelta(hours=1)
        observed_event = aligned["source_age"].eq(pd.Timedelta(0)).astype(float)
        extreme_pos = ((funding_delta > 0.0) & (funding_z.abs() >= self.zscore_threshold)).astype(float)
        extreme_neg = ((funding_delta < 0.0) & (funding_z.abs() >= self.zscore_threshold)).astype(float)

        self._last_report = {
            **report,
            "symbol": symbol,
            "rolling_window": self.rolling_window,
            "window_start": start_dt,
            "window_end": end_dt,
        }

        outputs = {
            f"{self.name}_delta": funding_delta,
            f"{self.name}_abs_delta": funding_abs_delta,
            f"{self.name}_mean_{self.rolling_window}": funding_mean,
            f"{self.name}_z_{self.rolling_window}": funding_z.fillna(0.0),
            f"{self.name}_age_hours": funding_age_hours,
            f"{self.name}_extreme_pos": extreme_pos,
            f"{self.name}_extreme_neg": extreme_neg,
            f"{self.name}_observed_event": observed_event,
        }
        return {column: pd.Series(series, index=base_index) for column, series in outputs.items()}