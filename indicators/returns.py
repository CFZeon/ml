"""Past-only return features for quick baseline experiments."""

from __future__ import annotations

import pandas as pd

KIND = "returns"
REQUIRED_COLUMNS = ("close",)
LOOKAHEAD_SAFE = True
STATEFUL = False
DESCRIPTION = "Lagged close-to-close returns over configurable lookback windows."


def compute(df: pd.DataFrame, periods=(1, 3, 6)) -> pd.DataFrame:
    close = pd.to_numeric(df["close"], errors="coerce")
    outputs: dict[str, pd.Series] = {}
    for raw_period in tuple(periods):
        period = int(raw_period)
        if period < 1:
            raise ValueError("returns periods must be >= 1")
        outputs[f"returns_{period}"] = close.pct_change(period)
    return pd.DataFrame(outputs, index=df.index)
