 Kelly Sizing Decontamination

**Priority**: High  
**Status**: Partially mitigated  
**Decision**: Per-fold OOS estimation with shrinkage. Most recent fold's estimates for final signals. Minimum trade count threshold for Kelly eligibility.

### Steps

1. **Modify `_estimate_trade_outcome_stats()` in `core/pipeline.py`**
   - Accept optional `shrinkage_alpha` parameter (default: 0.5 — balanced between fold-local and pooled).
   - When fold-local and pooled estimates are both available:
     ```python
     shrunk_win = alpha * fold_win + (1 - alpha) * pooled_win
     shrunk_loss = alpha * fold_loss + (1 - alpha) * pooled_loss
     ```
   - Institutional standard: half-Kelly with shrinkage = 0.5 is the conservative default at most quant firms. Bridgewater and AQR use shrinkage estimators for position sizing inputs.

2. **Add minimum trade count gate in `_build_signal_state()`**
   - Config key: `signals.min_trades_for_kelly: 30` (default: 30 — statistical minimum for stable mean/variance estimation).
   - When OOS trade count < threshold, replace Kelly with flat fractional sizing: `size = fraction × direction`.
   - Log a warning when this fallback activates.

3. **Add `max_kelly_fraction` cap**
   - Config key: `signals.max_kelly_fraction: 0.5` (default: 0.5 — half-Kelly is the institutional standard; full Kelly is never used in practice due to estimation error).
   - Clip final Kelly output to `min(kelly_output, max_kelly_fraction)`.

4. **Per-fold validation-set estimation**
   - In `TrainModelsStep`, when a validation set exists, use validation-set trade outcomes for fold-level sizing (already partially done).
   - When no validation set: use the **prior fold's** OOS test outcomes (truly out-of-sample, one fold behind). This requires storing fold outcomes across iterations.
   - Institutional standard: Renaissance and D.E. Shaw reportedly use rolling OOS windows for sizing calibration, never contemporaneous training data.

5. **Add tests**
   - `test_kelly_fallback_to_flat_below_trade_threshold()` — verify flat sizing when < 30 trades.
   - `test_kelly_shrinkage_blends_fold_and_pooled()` — verify shrinkage formula.
   - `test_kelly_capped_at_max_fraction()` — verify cap.

### Acceptance Criteria

- Kelly estimates use shrinkage between fold-local and pooled OOS outcomes.
- Below 30 OOS trades, Kelly degrades to flat fractional sizing with a warning.
- `max_kelly_fraction` prevents oversized positions.