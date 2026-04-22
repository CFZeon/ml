# ISSUES

## High-Impact Findings

- Critical | Backtesting / Evaluation: CPCV is being summarized as if it produces one tradable out-of-sample track, but the implementation produces multiple overlapping path-specific signal streams and then averages their results. `SignalsStep` emits `paths` for CPCV, `BacktestStep` backtests each path independently, and `_summarize_path_backtests()` reports `aggregate_mode="mean"` while averaging Sharpe, net profit, confidence intervals, and even p-values. Averaged p-values and averaged confidence bounds are not valid statistical objects, and mean path Sharpe is not the Sharpe of any executable strategy. This can make a non-tradable ensemble of mutually inconsistent paths look robust.

- Critical | AutoML / Leakage: fold-level sizing and signal policy are contaminated by prior out-of-sample trade outcomes. In `TrainModelsStep`, `prior_oos_trade_outcomes = pd.concat(oos_trade_outcomes)` is used before each new fold to set Kelly inputs, trade counts, and policy calibration. Under CPCV, `_iter_validation_splits()` consumes `cpcv_split()`, and `cpcv_split()` enumerates block combinations rather than strictly chronological folds. That means a later-processed fold can inherit trade outcomes from future calendar periods. The classifier may be trained causally while the sizing and threshold logic are not.

- Critical | Deployment Realism: `execution_policy={"adapter": "nautilus"}` does not deliver Nautilus-grade execution simulation. `core/execution/nautilus_adapter.py` is a stub boundary with no matching engine, while the tests still exercise `run_backtest(..., engine="pandas")`. Queue position, order acknowledgement latency, cancel/replace races, websocket/API desynchronization, and book-level matching remain approximated by deterministic bar logic. The backtest-to-live gap is therefore still structural, not residual.

- High | Data Integrity / Crypto Microstructure: there is no explicit wash-trading, spoofing, self-trade, or venue-manipulation detection anywhere in `core/`. `core/data_quality.py` only flags OHLC inconsistencies, return/range spikes, nonpositive volume, quote-volume mismatches, and trade-count anomalies. In crypto, manipulated prints can survive those checks and become both features and labels, especially on thinner symbols or stress windows. That inflates apparent signal quality because the model can learn exchange artifacts instead of transferable edge.

- High | Feature Engineering / Timestamp Validity: derivatives context features are as-of joined and then forward-filled with no explicit time-to-live. In `build_futures_context_feature_block()`, mark, premium, and funding frames are aligned with `_asof_reindex(...)` and then `.ffill()` is applied. During API gaps, venue outages, or stale secondary feeds, old derivatives state is presented as current information across many bars. That is a hidden timestamp misalignment that is especially dangerous around regime changes.

- High | Cross-Venue / Fragmented Liquidity: the spot reference overlay is a simple median of venue closes and volumes in `_build_spot_overlay()`, not a liquidity-weighted executable composite. Partial venue coverage is explicitly allowed to remain advisory in `build_spot_reference_validation()`, and the test suite asserts that partial coverage can still pass promotion. For a retail trader executing on Binance, this can validate signals using stale or non-executable external prints and understate fragmentation risk.

- High | Out-of-Sample / Robustness: drift tooling exists, but the main pipeline does not actually run drift detection or drift-triggered retraining. `core/drift.py` provides `DriftMonitor`, but the implementation is only surfaced through exports and registry gating; the research pipeline steps never compute a drift report or act on it. The system can therefore claim drift awareness while remaining scheduled-retrain-only in practice.

- High | Evaluation Metrics: the main trading objective is still driven by point estimates first. In `_build_objective_diagnostics()`, `risk_adjusted_after_costs` uses point-estimate Sharpe by default, and the confidence lower bound only matters if `objective_use_confidence_lower_bound=True`. This means selection is still primarily chasing noisy validation point estimates, with significance acting as a later gate rather than the optimized quantity. In a multi-trial AutoML loop, that is exactly how noise wins early.

- Moderate | Data Preprocessing: duplicate market bars are silently collapsed with `keep="first"` in `_prepare_frame()` and `_merge_frames()`. If Binance restates a bar, if cache content differs from a refetch, or if multiple sources disagree, the pipeline keeps the first seen row with no reconciliation or exception path. That preserves potentially stale data while the downstream contract and lineage layers still see a structurally valid frame.

- Moderate | AutoML / Multiple Testing: the effective hypothesis count is materially larger than the reported trial count. The search space jointly varies label definition, barrier parameters, feature selection, model family, calibration regularization, and validation fraction. After the study, the same validation slice is reused again for fragility checks and promotion gating. White RC / SPA reduce some of the damage, but they do not erase the fact that the validation set has been reused for both search and post-search acceptance.

- Moderate | Consumer Hardware / Operations: `run_automl_study()` executes full candidate pipelines per trial, then may rerun completed-trial diagnostics, local perturbation fragility checks, validation-holdout checks, locked-holdout checks, and replication cohorts. For a retail user on consumer hardware, this makes frequent multi-symbol retraining slow enough that models can be stale before they are revalidated. A slow but statistically careful pipeline is still a deployment risk if the market state changes faster than the retrain cadence.

## Failure Modes

- The strategy can look robust because averaged CPCV path metrics smooth away path dependence, while live trading only realizes one chronological path with one sequence of fills.

- Kelly sizing can look calibrated because future fold trade outcomes leak into fold-level thresholding and sizing, then collapse when deployed strictly forward in time.

- Derivatives-context features can look stable because stale funding, mark, or premium data are forward-filled through gaps, letting the model trade on dead state during outages.

- Cross-venue confirmation can look disciplined while still being anchored to a simple median overlay with partial venue coverage and no manipulation filter, which is not the same thing as executable price discovery.

- Execution resilience can look tested because downtime and halt scenarios are replayed against a pandas surrogate, but live outcomes will depend on queue position, acknowledgement timing, and microstructure that the current adapter does not simulate.

## Prior Note

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