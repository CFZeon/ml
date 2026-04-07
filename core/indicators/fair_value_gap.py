"""ICT-style Fair Value Gap indicator."""

import pandas as pd

from .base import Indicator
from .registry import register_indicator


@register_indicator
class FairValueGap(Indicator):
    kind = "fvg"
    required_columns = ("high", "low", "close")

    def __init__(self, min_gap_pct=0.0, name=None):
        self.min_gap_pct = min_gap_pct
        super().__init__(name=name)

    def default_name(self):
        return "fvg"

    def describe(self, outputs):
        metadata = super().describe(outputs)
        metadata.update({
            "definition": "ICT-style 3-candle gap: bullish when high[t-2] < low[t], bearish when low[t-2] > high[t]",
            "selection": "Nearest active gap by midpoint distance to current close",
            "fill_state_encoding": {
                0: "no active gap",
                1: "open",
                2: "partially filled",
            },
        })
        return metadata

    def _select_active_gap(self, gaps, price):
        if not gaps:
            return None
        return min(gaps, key=lambda gap: (abs(price - gap["mid"]), -gap["birth_pos"]))

    def _record_gap_state(self, outputs, prefix, position, gap, price, active_count):
        outputs[f"{prefix}_top"].iloc[position] = gap["top"] if gap else float("nan")
        outputs[f"{prefix}_bottom"].iloc[position] = gap["bottom"] if gap else float("nan")
        outputs[f"{prefix}_size"].iloc[position] = gap["size"] if gap else float("nan")
        outputs[f"{prefix}_size_pct"].iloc[position] = gap["size_pct"] if gap else float("nan")
        outputs[f"{prefix}_age"].iloc[position] = position - gap["birth_pos"] if gap else float("nan")
        outputs[f"{prefix}_distance_pct"].iloc[position] = ((price - gap["mid"]) / price) if gap and price else float("nan")
        outputs[f"{prefix}_fill_state"].iloc[position] = gap["fill_state"] if gap else 0.0
        outputs[f"{prefix}_active_count"].iloc[position] = active_count

    def compute(self, df):
        index = df.index
        high = df["high"].astype(float).to_numpy()
        low = df["low"].astype(float).to_numpy()
        close = df["close"].astype(float).to_numpy()
        size = len(df)

        output_names = [
            f"{self.name}_bull_top",
            f"{self.name}_bull_bottom",
            f"{self.name}_bull_size",
            f"{self.name}_bull_size_pct",
            f"{self.name}_bull_age",
            f"{self.name}_bull_distance_pct",
            f"{self.name}_bull_fill_state",
            f"{self.name}_bull_active_count",
            f"{self.name}_bear_top",
            f"{self.name}_bear_bottom",
            f"{self.name}_bear_size",
            f"{self.name}_bear_size_pct",
            f"{self.name}_bear_age",
            f"{self.name}_bear_distance_pct",
            f"{self.name}_bear_fill_state",
            f"{self.name}_bear_active_count",
        ]
        outputs = {
            name: pd.Series(float("nan"), index=index, dtype=float)
            for name in output_names
        }
        outputs[f"{self.name}_bull_fill_state"] = pd.Series(0.0, index=index, dtype=float)
        outputs[f"{self.name}_bull_active_count"] = pd.Series(0.0, index=index, dtype=float)
        outputs[f"{self.name}_bear_fill_state"] = pd.Series(0.0, index=index, dtype=float)
        outputs[f"{self.name}_bear_active_count"] = pd.Series(0.0, index=index, dtype=float)

        active_bulls = []
        active_bears = []

        for position in range(size):
            current_low = low[position]
            current_high = high[position]
            current_close = close[position]

            next_bulls = []
            for gap in active_bulls:
                if current_low <= gap["bottom"]:
                    continue
                gap["fill_state"] = 2.0 if current_low < gap["top"] else 1.0
                next_bulls.append(gap)
            active_bulls = next_bulls

            next_bears = []
            for gap in active_bears:
                if current_high >= gap["top"]:
                    continue
                gap["fill_state"] = 2.0 if current_high > gap["bottom"] else 1.0
                next_bears.append(gap)
            active_bears = next_bears

            if position >= 2:
                bull_bottom = high[position - 2]
                bull_top = low[position]
                bull_size = bull_top - bull_bottom
                bull_size_pct = bull_size / current_close if current_close else 0.0
                if bull_bottom < bull_top and bull_size_pct >= self.min_gap_pct:
                    active_bulls.append({
                        "top": bull_top,
                        "bottom": bull_bottom,
                        "mid": (bull_top + bull_bottom) / 2,
                        "size": bull_size,
                        "size_pct": bull_size_pct,
                        "birth_pos": position,
                        "fill_state": 1.0,
                    })

                bear_bottom = high[position]
                bear_top = low[position - 2]
                bear_size = bear_top - bear_bottom
                bear_size_pct = bear_size / current_close if current_close else 0.0
                if bear_bottom < bear_top and bear_size_pct >= self.min_gap_pct:
                    active_bears.append({
                        "top": bear_top,
                        "bottom": bear_bottom,
                        "mid": (bear_top + bear_bottom) / 2,
                        "size": bear_size,
                        "size_pct": bear_size_pct,
                        "birth_pos": position,
                        "fill_state": 1.0,
                    })

            selected_bull = self._select_active_gap(active_bulls, current_close)
            selected_bear = self._select_active_gap(active_bears, current_close)
            self._record_gap_state(
                outputs,
                f"{self.name}_bull",
                position,
                selected_bull,
                current_close,
                float(len(active_bulls)),
            )
            self._record_gap_state(
                outputs,
                f"{self.name}_bear",
                position,
                selected_bear,
                current_close,
                float(len(active_bears)),
            )

        return outputs