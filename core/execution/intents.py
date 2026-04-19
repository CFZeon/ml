from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class OrderIntent:
    timestamp: pd.Timestamp
    side: str | None
    order_type: str
    time_in_force: str
    requested_position: float
    previous_position: float
    requested_order_quantity: float
    requested_notional: float
    execution_price: float
    participation_cap: float
    min_fill_ratio: float
    max_order_age_bars: int
    cancel_replace_bars: int

    def to_dict(self):
        payload = asdict(self)
        payload["timestamp"] = self.timestamp
        return payload


__all__ = ["OrderIntent"]