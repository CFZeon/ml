"""Causal liquidity input resolution for bar-based and L2 execution paths."""

from __future__ import annotations

import pandas as pd

from ..slippage import OrderBookImpactModel


def _is_orderbook_model(slippage_model):
    if slippage_model is None:
        return False
    if isinstance(slippage_model, str):
        normalized = slippage_model.strip().lower().replace("-", "_")
        return normalized in {"orderbook", "order_book"}
    return isinstance(slippage_model, OrderBookImpactModel)


def _coerce_volume(volume, index):
    if volume is None:
        return None
    if isinstance(volume, pd.Series):
        series = volume.reindex(index)
    else:
        series = pd.Series(volume, index=index)
    return pd.to_numeric(series, errors="coerce").astype(float)


def _resolve_snapshot_time_column(frame):
    for candidate in ["snapshot_time", "timestamp", "ts"]:
        if candidate in frame.columns:
            return candidate
    return None


class LiquidityInputResolver:
    def __init__(self, liquidity_lag_bars=1):
        self.liquidity_lag_bars = max(0, int(1 if liquidity_lag_bars is None else liquidity_lag_bars))

    def _resolve_bar_volume(self, volume, index):
        series = _coerce_volume(volume, index)
        if series is None:
            return None, {
                "liquidity_source": "none",
                "liquidity_lag_bars": int(self.liquidity_lag_bars),
                "ex_post_liquidity_rows": 0,
            }

        if self.liquidity_lag_bars < 1:
            ex_post_rows = int(series.notna().sum())
            raise ValueError(
                "liquidity_lag_bars must be at least 1 for bar-volume inputs; "
                f"received {self.liquidity_lag_bars} with {ex_post_rows} ex-post rows"
            )

        return series.shift(self.liquidity_lag_bars).fillna(0.0), {
            "liquidity_source": "lagged_bar_volume",
            "liquidity_lag_bars": int(self.liquidity_lag_bars),
            "ex_post_liquidity_rows": 0,
        }

    def _resolve_orderbook_depth(self, orderbook_depth, index):
        if orderbook_depth is None:
            return None, {
                "liquidity_source": "orderbook_depth",
                "liquidity_lag_bars": 0,
                "ex_post_liquidity_rows": 0,
            }

        frame = orderbook_depth.copy() if isinstance(orderbook_depth, pd.DataFrame) else pd.DataFrame(orderbook_depth)
        execution_index = pd.DatetimeIndex(index)
        snapshot_column = _resolve_snapshot_time_column(frame)

        if snapshot_column is not None:
            snapshot_times = pd.to_datetime(frame[snapshot_column], utc=True, errors="coerce")
            if isinstance(frame.index, pd.DatetimeIndex) and frame.index.equals(execution_index):
                ex_post_mask = snapshot_times > frame.index
                ex_post_rows = int(ex_post_mask.fillna(False).sum())
                if ex_post_rows > 0:
                    raise ValueError(
                        "orderbook_depth contains snapshots after the simulated order timestamp "
                        f"({ex_post_rows} ex-post rows)"
                    )
                aligned = frame.drop(columns=[snapshot_column], errors="ignore").copy()
                aligned.index = execution_index
            else:
                working = frame.copy()
                working["_snapshot_time"] = snapshot_times
                working = working.dropna(subset=["_snapshot_time"]).sort_values("_snapshot_time")
                aligned = pd.merge_asof(
                    pd.DataFrame({"execution_time": execution_index}),
                    working.drop(columns=[snapshot_column], errors="ignore"),
                    left_on="execution_time",
                    right_on="_snapshot_time",
                    direction="backward",
                )
                aligned.index = execution_index
                aligned = aligned.drop(columns=["execution_time", "_snapshot_time"], errors="ignore")
            return aligned, {
                "liquidity_source": "orderbook_depth",
                "liquidity_lag_bars": 0,
                "ex_post_liquidity_rows": 0,
            }

        if isinstance(frame.index, pd.DatetimeIndex):
            aligned = frame.sort_index().reindex(execution_index, method="ffill")
            aligned.index = execution_index
            return aligned, {
                "liquidity_source": "orderbook_depth",
                "liquidity_lag_bars": 0,
                "ex_post_liquidity_rows": 0,
            }

        raise ValueError(
            "orderbook_depth must use a DatetimeIndex or include a snapshot_time/timestamp column"
        )

    def resolve(self, index, volume=None, orderbook_depth=None, slippage_model=None):
        execution_index = pd.DatetimeIndex(index)
        if _is_orderbook_model(slippage_model) and orderbook_depth is not None:
            resolved_orderbook, diagnostics = self._resolve_orderbook_depth(orderbook_depth, execution_index)
            return {
                "volume": None,
                "orderbook_depth": resolved_orderbook,
                "diagnostics": diagnostics,
            }

        resolved_volume, diagnostics = self._resolve_bar_volume(volume, execution_index)
        return {
            "volume": resolved_volume,
            "orderbook_depth": orderbook_depth,
            "diagnostics": diagnostics,
        }


def resolve_liquidity_inputs(index, volume=None, orderbook_depth=None, slippage_model=None, liquidity_lag_bars=1):
    resolver = LiquidityInputResolver(liquidity_lag_bars=liquidity_lag_bars)
    return resolver.resolve(index=index, volume=volume, orderbook_depth=orderbook_depth, slippage_model=slippage_model)