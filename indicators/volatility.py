"""Past-only rolling volatility features for baseline experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

KIND = "volatility"
REQUIRED_COLUMNS = ("close",)
LOOKAHEAD_SAFE = True
STATEFUL = False
DESCRIPTION = "Rolling realized volatility features computed from lagged close returns."


def compute(
    df: pd.DataFrame,
    window: int = 24,
    min_periods: int | None = None,
    annualize: bool = False,
    periods_per_year: int = 24 * 365,
) -> pd.DataFrame:
    resolved_window = int(window)
    if resolved_window < 2:
        raise ValueError("volatility window must be >= 2")
    resolved_min_periods = resolved_window if min_periods is None else int(min_periods)
    if resolved_min_periods < 1:
        raise ValueError("volatility min_periods must be >= 1")

    returns = pd.to_numeric(df["close"], errors="coerce").pct_change()
    rolling_volatility = returns.rolling(window=resolved_window, min_periods=resolved_min_periods).std()
    outputs = {
        f"volatility_{resolved_window}": rolling_volatility,
        f"volatility_{resolved_window}_pct_rank": rolling_volatility.rolling(
            window=resolved_window,
            min_periods=max(2, resolved_window // 2),
        ).rank(pct=True),
    }
    if annualize:
        outputs[f"volatility_{resolved_window}_annualized"] = rolling_volatility * float(np.sqrt(periods_per_year))
    return pd.DataFrame(outputs, index=df.index)
