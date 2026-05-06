"""User-facing funding-rate indicator wrapper with the simple compute(df) contract."""

from __future__ import annotations

import pandas as pd

KIND = "funding_rate"
REQUIRED_COLUMNS = ("close",)
LOOKAHEAD_SAFE = True
STATEFUL = False
DESCRIPTION = "Aligned Binance funding-rate context features for futures experiments."


def compute(df: pd.DataFrame, **params) -> pd.DataFrame:
    from core.indicators.funding_rate import FundingRateContext

    indicator = FundingRateContext(name=KIND, **params)
    return indicator.run(df).to_frame()
