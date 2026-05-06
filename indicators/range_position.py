"""Example custom indicator that measures where price sits within a rolling range."""

from __future__ import annotations

import numpy as np
import pandas as pd

KIND = "range_position"
REQUIRED_COLUMNS = ("high", "low", "close")
LOOKAHEAD_SAFE = True
STATEFUL = False
DESCRIPTION = "Rolling range position and normalized range width using only past bars."


def compute(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    resolved_window = int(window)
    if resolved_window < 2:
        raise ValueError("range_position window must be >= 2")

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    rolling_high = high.rolling(resolved_window, min_periods=resolved_window).max()
    rolling_low = low.rolling(resolved_window, min_periods=resolved_window).min()
    range_width = (rolling_high - rolling_low).replace(0.0, np.nan)
    outputs = {
        f"range_position_{resolved_window}": (close - rolling_low).div(range_width),
        f"range_position_{resolved_window}_width": range_width.div(close.replace(0.0, np.nan)),
    }
    return pd.DataFrame(outputs, index=df.index)
