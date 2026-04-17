Volume-Dependent Slippage Model

**Priority**: Critical  
**Status**: Open  
**Decision**: Support L2/order book data in the interface for future use. Per-bar estimation model. ADV window configurable separately, defaults to ATR window.

### Architecture

```
SlippageModel (Protocol)
├── FlatSlippageModel          — current behavior (backward compat)
├── SquareRootImpactModel      — σ × √(V / ADV), using OHLCV volume
└── OrderBookImpactModel       — future: uses L2 depth data via optional adapter
```

### Steps

1. **Create `core/slippage.py`**
   - Define `SlippageModel` protocol with method:
     ```python
     def estimate(self, trade_notional: pd.Series, volume: pd.Series,
                  volatility: pd.Series, price: pd.Series,
                  orderbook_depth: pd.DataFrame | None = None) -> pd.Series
     ```
   - Implement `FlatSlippageModel(rate: float)` — returns `rate` for every bar (current behavior).
   - Implement `SquareRootImpactModel(adv_window: int = 14, base_impact_bps: float = 5.0)`:
     - ADV = `volume.rolling(adv_window).mean()`
     - Per-bar volatility = `price.pct_change().rolling(adv_window).std()`
     - Participation rate = `trade_notional / (ADV × price)`, capped at 1.0
     - Impact = `base_impact_bps × volatility × sqrt(participation_rate)`
     - Floor: at least `base_impact_bps / 10000` per bar with any turnover.
   - Implement `OrderBookImpactModel` stub that raises `NotImplementedError("L2 data adapter not yet available")` with the interface accepting `orderbook_depth: pd.DataFrame` (columns: `bid_depth_usd`, `ask_depth_usd`, `mid_price`).

2. **Modify `core/backtest.py`**
   - Add import of slippage models.
   - In `run_backtest()`: accept `slippage_model: str | SlippageModel | None = None` parameter.
     - If `slippage_model` is a string (`"flat"`, `"sqrt_impact"`, `"orderbook"`), instantiate the corresponding model.
     - If `slippage_model is None`, fall back to `FlatSlippageModel(slippage_rate)` for backward compatibility.
   - In `_run_pandas_backtest()` and `_run_vectorbt_backtest()`:
     - When using a non-flat model, compute per-bar slippage by calling `model.estimate(...)` instead of `turnover * slippage_rate`.
     - Pass `volume` from the close series context (requires passing volume data through).

3. **Modify `core/pipeline.py` → `BacktestStep`**
   - Pass `raw_data["volume"]` and the configured `slippage_model` from `backtest` config section to `run_backtest()`.
   - Config key: `backtest.slippage_model: "sqrt_impact"` (default: `"flat"` for backward compat).
   - Config key: `backtest.slippage_adv_window: 14` (default: matches ATR window from features config if not set).

4. **Add tests**
   - `test_slippage_sqrt_model_increases_with_volume_ratio()` — verify impact increases when trade size / ADV increases.
   - `test_slippage_flat_model_matches_legacy_behavior()` — regression test.
   - `test_slippage_orderbook_raises_not_implemented()` — interface contract.
   - `test_backtest_with_sqrt_slippage_produces_lower_returns()` — integration.

### Acceptance Criteria

- `SquareRootImpactModel` consumes the `volume` column from OHLCV data.
- Per-bar slippage varies with volume and volatility.
- Default behavior unchanged when `slippage_model` is not specified.
- `OrderBookImpactModel` interface exists but raises `NotImplementedError` when called without L2 data.