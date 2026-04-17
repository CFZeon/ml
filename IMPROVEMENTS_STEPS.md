# IMPROVEMENTS_STEPS — Detailed Implementation Plans

Each section provides: context, architectural decisions, step-by-step implementation plan, files to modify/create, and acceptance criteria.

---

## C3. Volume-Dependent Slippage Model [implemented]

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

---

## H4. Two-Stage Holdout for AutoML [implemented]

**Priority**: High  
**Status**: Open  
**Decision**: Two-stage holdout (search → validation → final holdout). DSR threshold configurable per study with generous default (0.3). Enable Optuna MedianPruner.

### Architecture

```
Full data
├── [60%] Search window — Optuna runs inner WF/CPCV here
├── [20%] Validation holdout — used for trial RANKING (replaces inner OOS for selection)
└── [20%] Final holdout — touched once for the winning trial
```

### Steps

1. **Modify `core/automl.py` → `_resolve_locked_holdout_plan()`**
   - Rename to `_resolve_holdout_plan()`.
   - Add `validation_fraction` (default 0.2) alongside existing `locked_holdout_fraction` (default 0.2).
   - Compute three splits: `search_end`, `validation_end`, `holdout_start`.
   - Return plan dict with keys: `search_rows`, `validation_rows`, `holdout_rows`, `validation_start_timestamp`, `holdout_start_timestamp`.

2. **Modify `run_automl_study()`**
   - After each trial runs on the search window, re-run the best config on the **validation window** using a single walk-forward split (train=search, test=validation).
   - Store the validation-window metrics in the trial record as `validation_metrics`.
   - Change trial ranking: use validation-window metrics for selection instead of inner-search OOS metrics.
   - Add `minimum_dsr_threshold` config (default: 0.3 — generous, avoids rejecting marginal strategies prematurely). Trials with DSR below this threshold are penalized to `-inf` in ranking.

3. **Add Optuna pruning**
   - Instantiate `optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)` when creating the study.
   - After each trial completes, report the objective value as an intermediate result. Optuna prunes trials that underperform the running median.
   - Config key: `automl.enable_pruning: true` (default: true).

4. **Modify `_evaluate_locked_holdout()`**
   - Now only operates on the final 20% holdout.
   - Receives the best trial's config **after** validation-window ranking.
   - Add `holdout_warning` flag: if holdout Sharpe CI lower bound < 0, emit a warning.

5. **Update study report**
   - Add `validation_holdout` section alongside `locked_holdout` in the study summary.
   - Report both validation-window and final-holdout metrics.

6. **Add tests**
   - `test_two_stage_holdout_validation_used_for_ranking()` — verify trial selection uses validation metrics, not search-window metrics.
   - `test_dsr_threshold_rejects_low_dsr_trials()` — verify trials below threshold are penalized.
   - `test_optuna_pruner_reduces_trial_count()` — verify that clearly bad trials are stopped early.

### Acceptance Criteria

- Trial selection uses validation-window metrics, not inner-search OOS metrics.
- DSR threshold is configurable; trials below it are rejected.
- Optuna MedianPruner reduces average trial runtime.
- Final holdout is only evaluated once on the winning trial.

---

## H5. Kelly Sizing Decontamination

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

---

## H6. Transaction Cost Sensitivity Analysis

**Priority**: High  
**Status**: Open  
**Decision**: Sweep both fee_rate and slippage_rate. Results as subkey of backtest output. Full sweep table.

### Steps

1. **Add `_cost_sensitivity_sweep()` in `core/backtest.py`**
   ```python
   def _cost_sensitivity_sweep(close, signals, equity, base_fee_rate, base_slippage_rate,
                                multipliers=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
                                **backtest_kwargs) -> dict:
   ```
   - For each multiplier, rerun backtest with `fee_rate = base_fee_rate × mult` and `slippage_rate = base_slippage_rate × mult`.
   - Collect per-level: `net_profit_pct`, `sharpe_ratio`, `profit_factor`, `total_trades`, `win_rate`.
   - Compute `breakeven_multiplier`: interpolate to find the multiplier where `net_profit_pct` crosses zero.
   - Compute `cost_margin_of_safety = breakeven_multiplier / 1.0`.

2. **Wire into `_summarize_backtest()`**
   - After computing main metrics, if `cost_sensitivity` config is enabled:
     ```python
     result["cost_sensitivity"] = _cost_sensitivity_sweep(...)
     ```
   - Config key: `backtest.cost_sensitivity.enabled: true` (default: false — opt-in to avoid slowing every backtest).
   - Config key: `backtest.cost_sensitivity.multipliers: [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]`.

3. **Output format**
   ```python
   {
       "cost_sensitivity": {
           "enabled": True,
           "base_fee_rate": 0.001,
           "base_slippage_rate": 0.0,
           "breakeven_multiplier": 2.3,
           "cost_margin_of_safety": 2.3,
           "sweep": [
               {"multiplier": 0.5, "fee_rate": 0.0005, "slippage_rate": 0.0, "net_profit_pct": 0.12, "sharpe_ratio": 3.2, ...},
               {"multiplier": 1.0, "fee_rate": 0.001, "slippage_rate": 0.0, "net_profit_pct": 0.08, "sharpe_ratio": 2.1, ...},
               ...
           ]
       }
   }
   ```

4. **Add tests**
   - `test_cost_sweep_breakeven_between_levels()` — verify breakeven interpolation.
   - `test_cost_sweep_monotonic_decline()` — higher costs → lower returns.
   - `test_cost_sweep_disabled_by_default()` — no extra computation unless opted in.

### Acceptance Criteria

- Full sweep table with configurable multipliers.
- Breakeven multiplier computed by interpolation.
- Both fee_rate and slippage_rate scaled.
- Stored as `backtest["cost_sensitivity"]` subkey.

---

## M1. Drift Detection (ADWIN / PSI / KS)

**Priority**: Medium  
**Status**: Open  
**Decision**: Monitor features, predictions, and realized PnL. Batch mode. Follow institutional standards for cooldown/thresholds.

### Architecture

```
core/drift.py
├── DriftDetector (Protocol)
│   ├── ADWINDetector        — streaming concept drift (River library)
│   ├── PSIDetector          — batch Population Stability Index
│   └── KSDetector           — batch Kolmogorov-Smirnov test
├── DriftMonitor
│   ├── monitor_features()   — PSI per feature column
│   ├── monitor_predictions()— KL divergence of predicted class probabilities
│   └── monitor_performance()— ADWIN on rolling accuracy / rolling Sharpe
└── DriftReport
```

### Steps

1. **Create `core/drift.py`**
   - `PSIDetector(n_bins=10, threshold=0.2)`:
     - Compute PSI between reference distribution (training) and current distribution (recent window).
     - PSI > 0.1 = "moderate drift", PSI > 0.2 = "significant drift" (institutional standard from Evidently AI, NannyML).
   - `KSDetector(significance=0.05)`:
     - Two-sample KS test between training and recent feature distributions.
     - Returns p-value per feature; flag features where p < significance.
   - `ADWINDetector(delta=0.002)`:
     - Wrap `river.drift.ADWIN` for streaming detection.
     - If `river` not installed, fall back to a simple rolling-window mean-shift test.

2. **Implement `DriftMonitor`**
   ```python
   class DriftMonitor:
       def __init__(self, reference_features, reference_predictions, config):
           ...

       def check(self, current_features, current_predictions, current_performance):
           """Return DriftReport with per-feature PSI, prediction KL, performance drift."""
           ...
   ```
   - Batch mode: called every `check_interval` bars (default: 100 bars for 1h data = ~4 days).
   - Cooldown: minimum 500 bars (default) between retrain triggers. Institutional standard: avoid reacting to transient noise; Two Sigma reportedly uses 2-week cooldown.
   - Minimum sample: at least 200 bars in the current window before drift checks activate.

3. **Add `DriftMonitorStep` to pipeline**
   - Runs after `TrainModelsStep`.
   - Stores `drift_report` in pipeline state.
   - Config section: `drift: {enabled: true, check_interval: 100, cooldown_bars: 500, min_samples: 200, psi_threshold: 0.2}`.

4. **Add retrain triggering logic**
   - `should_retrain(drift_report) -> bool`: returns True if:
     - \>30% of features have PSI > threshold, OR
     - Rolling accuracy ADWIN fires, OR
     - Rolling Sharpe drops below 0 for > cooldown_bars.
   - Returns a `RetrainRecommendation(reason, urgency, features_drifted, bars_since_last_retrain)`.

5. **Add tests**
   - `test_psi_detects_distribution_shift()` — synthetic shifted normal → PSI > 0.2.
   - `test_adwin_detects_mean_shift()` — sequence with abrupt mean change.
   - `test_cooldown_prevents_rapid_retrain()` — two drift events within cooldown → only one trigger.
   - `test_drift_monitor_batch_mode()` — check called at intervals, not per bar.

### Acceptance Criteria

- PSI computed per feature between training and recent windows.
- ADWIN monitors rolling accuracy/Sharpe.
- Prediction distribution drift monitored via KL divergence.
- Cooldown of 500 bars (configurable) between retrain triggers.
- Minimum 200 bars before drift checks activate.

---

## M2. Local Model Registry

**Priority**: Medium  
**Status**: Open  
**Decision**: Purely local file system. Enforce feature schema compatibility. Follow institutional standards for retention.

### Architecture

```
core/registry.py
├── ModelArtifact (dataclass)
│   ├── model_id: str (UUID)
│   ├── symbol: str
│   ├── created_at: datetime
│   ├── stage: str ("challenger" | "champion" | "archived")
│   ├── feature_columns: list[str]
│   ├── feature_schema_hash: str
│   ├── config_snapshot: dict
│   ├── training_metrics: dict
│   ├── validation_metrics: dict
│   ├── dsr_score: float
│   ├── pbo_score: float
│   └── model_path: Path
├── LocalModelRegistry
│   ├── register(model, metadata) -> artifact_id
│   ├── load(artifact_id) -> (model, metadata)
│   ├── promote(artifact_id, stage)
│   ├── get_champion(symbol) -> artifact_id | None
│   ├── rollback(symbol) -> artifact_id
│   ├── list_versions(symbol) -> list[ModelArtifact]
│   └── validate_schema(artifact_id, feature_columns) -> bool
└── Storage: joblib serialization + Parquet metadata index
```

### Steps

1. **Create `core/registry.py`**
   - `ModelArtifact` dataclass with all metadata fields.
   - `LocalModelRegistry(root_dir=".cache/registry")`:
     - Metadata index: Parquet file at `{root_dir}/index.parquet` with one row per artifact.
     - Model files: `{root_dir}/models/{artifact_id}.joblib`.
     - `register()`: serialize model with `joblib.dump()`, compute SHA-256 hash of the file, write metadata row.
     - `load()`: validate hash, `joblib.load()`.
     - `promote()`: change stage from "challenger" to "champion". Previous champion → "archived".
     - `rollback()`: find most recent "archived" artifact for the symbol, promote it back to "champion".
     - `validate_schema()`: compare incoming feature column list against artifact's stored `feature_columns`. Return False if mismatch.

2. **Feature schema hashing**
   - `feature_schema_hash = hashlib.sha256(json.dumps(sorted(feature_columns)).encode()).hexdigest()[:16]`
   - At inference time, compute hash of incoming features. If it doesn't match the champion's hash, raise `SchemaValidationError`.

3. **Retention policy**
   - Keep last 10 versions per symbol (institutional standard: MLflow default is unlimited, but disk-constrained setups typically keep 5-20). Configurable via `registry.max_versions_per_symbol: 10`.
   - Auto-archive versions beyond the limit (oldest first).
   - Never auto-delete — only archive. Manual `purge()` method for cleanup.

4. **Wire into pipeline**
   - After `TrainModelsStep`, if `registry.enabled: true`:
     - Register the final model + meta-model as a new artifact.
     - If DSR > threshold and PBO < 0.5, auto-promote to "champion".
     - Otherwise, keep as "challenger".

5. **Replace pickle in `core/models.py`**
   - Deprecate `save_model()` / `load_model()` with a warning redirecting to the registry.
   - Keep them functional for backward compatibility but use `joblib` internally instead of `pickle`.

6. **Add tests**
   - `test_register_and_load_round_trip()` — model fidelity after serialization.
   - `test_promote_demotes_previous_champion()` — stage transitions.
   - `test_schema_validation_rejects_mismatch()` — wrong feature set → error.
   - `test_rollback_restores_previous_champion()` — rollback flow.
   - `test_retention_policy_archives_old_versions()` — 11th version archives the 1st.

### Acceptance Criteria

- Models stored with `joblib`, indexed in Parquet.
- Feature schema hash validated at inference time.
- Champion/challenger/archived stage lifecycle.
- Last 10 versions retained per symbol.
- `pickle`-based store deprecated with warning.

---

## M3. Structural Break Detection (SADF / GSADF / CUSUM)

**Priority**: Medium  
**Status**: Open  
**Decision**: Detection runs on everything (log prices, feature distributions, prediction residuals). Follow institutional standards for bubble response and critical values.

### Steps

1. **Create `core/structural_breaks.py`**
   - `sadf_test(series, min_window=None, significance=0.05, max_lags=None)`:
     - For each expanding window from `min_window` to `len(series)`, run ADF test.
     - Return: `sup_adf_stat`, `critical_values` (from AFML Table 17.1 for crypto-appropriate sample sizes), `is_explosive: bool`.
     - Default `min_window = max(20, len(series) // 10)`.
   - `gsadf_test(series, min_window=None, significance=0.05)`:
     - Double loop: vary both start and end of the ADF window.
     - Return: `gsadf_stat`, `is_explosive`, `explosive_periods: list[tuple[start, end]]`.
     - This is computationally expensive — add an optional `step` parameter (default: 1) to skip windows.
   - `cusum_detect(series, threshold=5.0, drift=0.0, direction="both")`:
     - Standard CUSUM. Return index positions where cumulative sum exceeds threshold.
     - `direction`: "up", "down", or "both" (detect both positive and negative shifts).

2. **Integrate as `StructuralBreakStep`**
   - Runs after `detect_regimes`, before `build_labels`.
   - Monitors:
     - **Log prices**: SADF/GSADF on `np.log(close)` — detects bubbles.
     - **Feature distributions**: CUSUM on rolling mean of top-5 features by importance — detects feature regime shifts.
     - **Prediction residuals**: CUSUM on rolling prediction accuracy — detects model degradation.
   - Store results in `pipeline.state["structural_breaks"]`.

3. **Institutional standard for bubble response**
   - **Do not hard-override signals** in the research pipeline. Instead:
     - Add a `bubble_regime` binary feature column (1 when SADF detects explosive behavior).
     - The model learns whether to trade during bubbles.
     - For live deployment (future scope): a configurable hard kill-switch flag `structural_breaks.halt_on_bubble: false`.
   - Rationale: firms like Man AHL and Winton use structural break detection as an **input feature**, not a kill switch, in research. Kill switches are reserved for risk overlays in production.

4. **Critical values for crypto**
   - AFML Table 17.1 provides critical values for standard ADF with trend. For crypto (where bubbles are frequent and short-lived), use the 5% critical value as default significance.
   - For GSADF: use Monte Carlo simulated critical values (pre-compute for standard sample sizes and cache).

5. **Add tests**
   - `test_sadf_detects_exponential_growth()` — synthetic exponential series → `is_explosive = True`.
   - `test_sadf_rejects_random_walk()` — Brownian motion → `is_explosive = False`.
   - `test_cusum_detects_mean_shift()` — series with abrupt level change.
   - `test_gsadf_identifies_bubble_periods()` — synthetic bubble + crash → explosive_periods non-empty.

### Acceptance Criteria

- SADF and GSADF detect explosive behavior in log price series.
- CUSUM detects mean shifts in feature distributions.
- Results stored as feature columns for the model to consume.
- No hard signal override in research mode.

---

## M4. Signal Half-Life / Alpha Decay

**Priority**: Medium  
**Status**: Open  
**Decision**: Follow institutional standards (WorldQuant / Millennium approach).

### Steps

1. **Add `estimate_signal_half_life()` in `core/backtest.py`**
   ```python
   def estimate_signal_half_life(signal_returns, max_lag=50):
       """Estimate the lag at which signal alpha decays to 50%.

       Uses the Ornstein-Uhlenbeck mean-reversion half-life method:
       regress y(t) - y(t-1) on y(t-1) and estimate λ = -log(2) / β.

       Returns dict with half_life_bars, decay_constant, r_squared.
       """
   ```
   - Institutional standard (WorldQuant, AFML Ch. 5): fit AR(1) model to cumulative signal returns. Half-life = `−ln(2) / θ` where `θ` is the AR(1) coefficient.
   - If the AR(1) coefficient is positive (momentum signal), report half-life as `∞` and flag as "non-decaying".
   - Also compute autocorrelation at lags 1..max_lag and report the lag at which autocorrelation first drops below 0.5.

2. **Compute per signal type**
   - In `TrainModelsStep`, after generating OOS signals, compute:
     - Half-life of **raw directional signal** (model predictions).
     - Half-life of **meta-filtered signal** (after meta-model gating).
     - Half-life of **sized signal** (after Kelly sizing).
   - Store in `training["signal_decay"]`:
     ```python
     {
         "raw_direction": {"half_life_bars": 8, "decay_constant": -0.087, "r_squared": 0.45},
         "meta_filtered": {"half_life_bars": 12, ...},
         "sized": {"half_life_bars": 15, ...},
     }
     ```

3. **Use half-life informatively**
   - Institutional standard: half-life estimates do **not** feed back into the holding period or execution strategy in the research phase. They are a diagnostic metric.
   - Report half-life alongside backtest results.
   - If half-life < 2 bars: warn "Signal decays faster than execution latency; results may not be replicable in live trading."
   - If half-life > holding_bars × 2: warn "Signal persists much longer than holding period; consider increasing holding_bars."

4. **Add tests**
   - `test_half_life_mean_reverting_series()` — synthetic OU process with known half-life.
   - `test_half_life_momentum_signal_returns_inf()` — trending signal → infinite half-life.
   - `test_half_life_computed_per_signal_type()` — integration with pipeline.

### Acceptance Criteria

- Half-life estimated via AR(1) method for all three signal types.
- Stored in training summary.
- Warnings emitted for signals that decay faster than execution or persist far beyond holding period.

---

## M5. Walk-Forward Fold Stability Analysis

**Priority**: Medium  
**Status**: Open  
**Decision**: CV threshold = 0.5 (moderate). Trigger warning on instability. Per-fold backtest Sharpe computed.

### Steps

1. **Modify `TrainModelsStep` in `core/pipeline.py`**
   - After computing `fold_metrics`, add per-fold backtest Sharpe:
     - For each fold, if OOS continuous signals exist, run a mini-backtest on the test window.
     - Store `fold_backtest_sharpe` in each fold's metrics dict.
   - Compute stability metrics:
     ```python
     fold_values = [m["directional_accuracy"] for m in fold_metrics if m.get("directional_accuracy") is not None]
     stability = {
         "directional_accuracy": {
             "mean": np.mean(fold_values),
             "std": np.std(fold_values, ddof=1),
             "min": np.min(fold_values),
             "max": np.max(fold_values),
             "cv": np.std(fold_values, ddof=1) / max(np.mean(fold_values), 1e-12),
             "is_stable": cv < 0.5,
         },
         # Same for f1_macro, log_loss, brier_score, fold_backtest_sharpe
     }
     ```

2. **Emit warning on instability**
   - After stability computation:
     ```python
     if not stability["directional_accuracy"]["is_stable"]:
         warnings.warn(
             f"Fold-to-fold directional accuracy CV={stability['directional_accuracy']['cv']:.2f} "
             f"exceeds stability threshold 0.5. Strategy performance is unstable across time periods.",
             RuntimeWarning,
         )
     ```

3. **Store in training summary**
   - `training["fold_stability"] = stability`

4. **Add tests**
   - `test_fold_stability_stable_strategy()` — uniform fold metrics → `is_stable = True`.
   - `test_fold_stability_unstable_strategy()` — wildly varying fold metrics → `is_stable = False`, warning emitted.

### Acceptance Criteria

- Std, min, max, CV reported for each key metric across folds.
- Per-fold backtest Sharpe computed and included in stability analysis.
- Warning emitted when CV > 0.5.

---

## M6. Pandas Backtest Discrete Trade Modeling

**Priority**: Medium  
**Status**: Open  
**Decision**: Follow institutional standards (discrete events for TBL, continuous for fixed-horizon).

### Steps

1. **Add `_run_pandas_discrete_backtest()` in `core/backtest.py`**
   - Model entry/exit as discrete events from signal changes.
   - At each signal onset (position goes from 0 → ±X):
     - Record entry price = execution price at signal bar + delay.
     - Hold for `holding_bars` or until signal reverses (whichever comes first).
     - Exit: record exit price, compute PnL for the trade.
   - Support overlapping trades: a new signal replaces the current position (close old, open new). Institutional standard: most event-driven backtests treat new signals as position flips, not overlapping holds.
   - Use signal size (Kelly fraction) for the trade size.

2. **Config integration**
   - `backtest.trade_model: "continuous"` (default) or `"discrete"`.
   - When `trade_model = "auto"`: use `"discrete"` for triple-barrier labels, `"continuous"` for fixed-horizon labels.

3. **Modify `_build_trade_ledger()`**
   - For discrete mode, build the ledger directly from the event-based entry/exit pairs rather than from position sign changes.

4. **Add tests**
   - `test_discrete_fewer_trades_than_continuous()` — discrete model produces fewer trades.
   - `test_discrete_matches_vbt_for_simple_signals()` — compare to VBT from_orders output.
   - `test_auto_mode_selects_discrete_for_tbl()` — triple-barrier → discrete.

### Acceptance Criteria

- Discrete backtest models entry/exit events, not continuous rebalancing.
- Trade count matches actual model predictions.
- Default is `"auto"` which selects based on label type.

---

## M7. Liquidity Filtering / Symbol Eligibility

**Priority**: Medium  
**Status**: Open  
**Decision**: Follow institutional standards.

### Steps

1. **Add `filter_by_liquidity()` in `core/data.py`**
   ```python
   def filter_by_liquidity(data, min_adv_usd=1_000_000, min_trading_days=90,
                            min_bar_volume_pct=0.01):
       """Filter bars and validate symbol eligibility.

       Parameters
       ----------
       min_adv_usd : float — minimum average daily quote volume in USD.
       min_trading_days : float — minimum listing age.
       min_bar_volume_pct : float — drop bars where volume < pct of ADV (outlier bars).

       Returns filtered data and eligibility report.
       """
   ```
   - Use `quote_volume` (USDT-denominated) for ADV — institutional standard for crypto since base volume is in different units across symbols.
   - Per-bar filtering: remove bars where `quote_volume < min_bar_volume_pct × ADV`. These are exchange downtime or extremely thin liquidity bars.
   - Symbol-level rejection: if median daily quote volume < `min_adv_usd`, reject the entire symbol with a clear error message.

2. **Wire into `FetchDataStep`**
   - After data fetch, before feature computation:
     ```python
     if config.get("liquidity_filter", {}).get("enabled", False):
         data, eligibility = filter_by_liquidity(data, **config["liquidity_filter"])
         pipeline.state["liquidity_eligibility"] = eligibility
     ```
   - Config: `data.liquidity_filter: {enabled: false, min_adv_usd: 1000000, min_trading_days: 90}`.

3. **Add tests**
   - `test_liquidity_rejects_low_volume_symbol()` — synthetic data with volume < threshold.
   - `test_liquidity_passes_high_volume_symbol()` — BTCUSDT-level volume.
   - `test_per_bar_filtering_removes_thin_bars()` — bars with 0 volume dropped.

### Acceptance Criteria

- Symbol rejected if median daily quote volume < threshold.
- Individual thin-volume bars filtered.
- Report stored in pipeline state.

---

## M8. Benchmark Comparison

**Priority**: Medium  
**Status**: Partially implemented  
**Decision**: Follow institutional standards.

### Steps

1. **Auto-generate buy-and-hold benchmark in `BacktestStep`**
   - Compute `benchmark_returns = raw_data["close"].pct_change().fillna(0.0)`.
   - Pass to `run_backtest()` as `benchmark_returns` automatically (unless user provides a custom benchmark).

2. **Add alpha decomposition to `_summarize_backtest()`**
   - Compute:
     - `alpha = annualized(strategy_return - beta × benchmark_return)`
     - `beta = cov(strat, bench) / var(bench)`
     - `information_ratio = alpha / tracking_error`
     - `tracking_error = std(strategy_return - benchmark_return) × sqrt(periods_per_year)`
   - Store under `backtest["benchmark_analysis"]`.

3. **Add simple trend-following baseline (optional)**
   - Config: `backtest.trend_baseline: {enabled: false, fast_window: 50, slow_window: 200}`.
   - SMA crossover: long when fast > slow, else flat.
   - Report as a second benchmark if enabled.

4. **Add tests**
   - `test_benchmark_auto_generated_buy_and_hold()` — buy-and-hold benchmark present by default.
   - `test_alpha_beta_computation()` — known returns → expected alpha/beta.

### Acceptance Criteria

- Buy-and-hold benchmark auto-generated and passed to backtest.
- Alpha, beta, information ratio, tracking error reported.
- Bootstrap CI includes p_value_gt_benchmark.

---

## M9. Walk-Forward Embargo Default Fix

**Priority**: Medium  
**Status**: Partially addressed  
**Decision**: Follow AFML recommendation (gap = max_holding). Apply to walk-forward only. Warn when gap=0 explicitly set.

### Steps

1. **Modify `_iter_validation_splits()` in `core/pipeline.py`**
   - When `validation_method == "walk_forward"`:
     - If `gap` is not explicitly set in config, auto-infer: `gap = max_holding` from label config (for triple-barrier) or `gap = horizon` (for fixed-horizon).
     - If `gap` is explicitly set to 0, emit:
       ```python
       warnings.warn(
           "Walk-forward gap=0 with overlapping labels risks information leakage. "
           "Consider setting gap >= max_holding.",
           RuntimeWarning,
       )
       ```

2. **Update `walk_forward_split()` docstring**
   - Note that `gap=0` is deprecated for overlapping-label scenarios.
   - Document: "For triple-barrier labels with max_holding=H, set gap >= H to prevent serial correlation leakage (AFML Ch. 7)."

3. **Add test**
   - `test_walk_forward_auto_embargo_uses_max_holding()` — verify gap auto-set from config.
   - `test_walk_forward_gap_zero_emits_warning()` — explicit gap=0 → warning.

### Acceptance Criteria

- Walk-forward gap auto-inferred from label config when not set.
- Warning emitted when gap=0 is explicitly used.
- CPCV embargo unchanged (already correct).

---

## M10. Sequential Bootstrap for RF Training

**Priority**: Medium  
**Status**: Open  
**Decision**: Follow institutional standards. Use sample-weight approximation with concurrency detection. Warn on high-concurrency RF without sequential bootstrap.

### Steps

1. **Modify `train_model()` in `core/models.py`**
   - Accept optional `labels` parameter for concurrency detection.
   - When `model_type="rf"` and labels show high concurrency (mean uniqueness < 0.7):
     - Use `max_samples` parameter of sklearn RF (available since 0.22): set `max_samples` to the number of unique samples from `sequential_bootstrap()`.
     - Alternatively: create a custom bootstrap index array via `sequential_bootstrap()` and subsample the training data explicitly.
   - Institutional standard: AFML Ch. 4.5.2 recommends sequential bootstrapping for any bagged estimator with concurrent labels. The sample-weight approach is an approximation that doesn't fully solve the OOB inflation problem.

2. **Implementation approach: subsampled training**
   - Rather than subclassing RF, use a simpler approach:
     ```python
     if model_type == "rf" and mean_uniqueness < 0.7:
         bootstrap_indices = sequential_bootstrap(labels, close, n_samples=len(X))
         X_boot = X.iloc[bootstrap_indices]
         y_boot = y.iloc[bootstrap_indices]
         model.fit(X_boot, y_boot, sample_weight=sw[bootstrap_indices] if sw is not None else None)
     ```
   - Set `bootstrap=False` in the RF constructor to prevent double-bootstrapping.

3. **Seed `sequential_bootstrap()`**
   - Fix the `np.random.choice` call to use a seeded `np.random.default_rng()`:
     ```python
     def sequential_bootstrap(labels, close, n_samples=None, random_state=42):
         rng = np.random.default_rng(random_state)
         ...
         drawn.append(rng.choice(n_labels, p=probs))
     ```

4. **Also apply to GBM**
   - For GBM with `subsample < 1.0`: apply sequential bootstrap to the subsampled indices. However, GBM's built-in subsampling is per-tree and per-iteration — we cannot easily inject custom indices.
   - Practical compromise: rely on `sample_weight` (uniqueness weights) for GBM. This is the institutional standard — AFML specifically targets bagged estimators, not boosted ones.

5. **Add warning**
   - When RF is selected and mean uniqueness < 0.7 and sequential bootstrap is disabled:
     ```python
     warnings.warn(
         f"RandomForest with concurrent labels (mean_uniqueness={mean_uniq:.2f}). "
         "Sequential bootstrapping recommended. Enable via model.sequential_bootstrap: true.",
         RuntimeWarning,
     )
     ```

6. **Add tests**
   - `test_sequential_bootstrap_reduces_correlation()` — bootstrap samples have higher average uniqueness than random samples.
   - `test_rf_sequential_bootstrap_flag()` — config `model.sequential_bootstrap: true` → sequential indices used.
   - `test_sequential_bootstrap_seed_reproducible()` — same seed → same indices.

### Acceptance Criteria

- Sequential bootstrap integrated for RF when labels are concurrent.
- `sequential_bootstrap()` uses a seeded RNG.
- GBM uses sample weights only (no bootstrap injection).
- Warning emitted when RF used with concurrent labels without sequential bootstrap.

---

## L1. Drawdown Kill Switch and Position Limits

**Priority**: Low  
**Status**: Open  
**Decision**: Percentage-based. Resume after cooldown. Portfolio equity drawdown only.

### Steps

1. **Add `_apply_risk_overlay()` in `core/backtest.py`**
   ```python
   def _apply_risk_overlay(position, equity_curve, config):
       """Zero out positions when drawdown exceeds threshold.

       config keys:
           max_drawdown_halt: float (e.g., -0.20 = halt at 20% drawdown)
           halt_cooldown_bars: int (default: 50 — resume after cooldown if drawdown recovers)
           halt_mode: "permanent" | "cooldown" (default: "cooldown")
       """
   ```
   - Track running drawdown from peak equity.
   - When drawdown exceeds threshold: set `position = 0` for `halt_cooldown_bars` bars.
   - If `halt_mode = "permanent"`: zero all remaining positions.
   - If `halt_mode = "cooldown"`: resume after cooldown if drawdown has recovered to < threshold / 2.

2. **Wire into both backtest engines**
   - Apply `_apply_risk_overlay()` to the position series before passing to `_run_pandas_backtest()` or `_run_vectorbt_backtest()`.

3. **Add tests**
   - `test_kill_switch_zeros_positions_at_threshold()`.
   - `test_cooldown_resumes_after_recovery()`.
   - `test_permanent_halt_never_resumes()`.

---

## L2. Data Freshness Detection

**Priority**: Low  
**Status**: Open

### Steps

1. **Add `check_data_freshness()` in `core/data.py`**
   ```python
   def check_data_freshness(data, expected_interval, max_staleness_multiplier=2.0):
       """Check if the most recent bar is within expected recency.

       Returns dict with is_fresh, last_bar_time, expected_next_bar, staleness_seconds.
       """
   ```
   - Compute expected interval from the data's median bar spacing.
   - If `now - last_bar_time > expected_interval × max_staleness_multiplier`: `is_fresh = False`.

2. **Wire into `FetchDataStep`**
   - After fetching data, run freshness check.
   - If stale, emit warning but don't halt (research mode). In future live mode: halt signal generation.

---

## L3. Missing Candle / Gap Handling

**Priority**: Low  
**Status**: Open

### Steps

1. **Modify `_detect_and_report_gaps()` in `core/data.py`**
   - Add `gap_fill_policy` parameter: `"ffill"` (forward-fill), `"drop_window"`, `"flag"`.
   - `"ffill"`: forward-fill OHLCV values and mark `is_gap_filled = True` column.
   - `"drop_window"`: drop N bars around each gap (N = rolling_window).
   - `"flag"`: add `has_gap` boolean column; features downstream can check this.
   - Default: `"ffill"` (institutional standard — most quant systems forward-fill and flag).

2. **Add `gap_filled` column to output**
   - Downstream feature builders can optionally exclude gap-filled bars from rolling statistics.

---

## L4. Crypto-Native Indicators

**Priority**: Low  
**Status**: Open

### Steps

1. **Add indicators under `core/indicators/`**
   - `vwap.py` — Volume-Weighted Average Price.
   - `funding_momentum.py` — Rolling funding rate momentum (uses futures context data).
   - `open_interest_change.py` — OI change rate from futures context.
   - `taker_flow.py` — Taker buy/sell ratio momentum (already partially in features via `taker_buy_ratio`).

2. **Add feature extractors in `core/features.py`**
   - Register each new indicator in `INDICATOR_FEATURE_EXTRACTORS`.

---

## L5. Hyperparameter Sensitivity Analysis

**Priority**: Low  
**Status**: Open

### Steps

1. **Add `_hyperparameter_sensitivity()` in `core/automl.py`**
   - After study completion, use `optuna.importance.get_param_importances(study)` to rank hyperparameters.
   - For the top-3 most important parameters, compute partial dependence: performance as a function of each parameter while holding others fixed.
   - Flag parameters where a 10% perturbation changes objective by > 20% — sign of overfitting.

2. **Store in study summary**
   - `automl["sensitivity"] = {"param_importances": {...}, "fragile_params": [...]}`.

---

## L6. Per-Regime Performance Decomposition

**Priority**: Low  
**Status**: Open

### Steps

1. **Add `_regime_performance_decomposition()` in `core/backtest.py`**
   ```python
   def _regime_performance_decomposition(strat_ret, position, regime_series):
       """Break down performance by regime label.

       Returns dict: {regime_id: {sharpe, win_rate, max_drawdown, exposure_rate, bar_count}}.
       """
   ```
   - For each unique regime value, compute metrics only on bars where that regime is active.

2. **Wire into `BacktestStep`**
   - If `pipeline.state["regimes"]` exists, compute decomposition and store in `backtest["regime_performance"]`.

---

## L7. Safe Model Serialization

**Priority**: Low  
**Status**: Open

### Steps

1. **Replace `pickle` with `joblib` in `core/models.py`**
   - Change `save_model()` to use `joblib.dump()`.
   - Change `load_model()` to use `joblib.load()`.
   - Add SHA-256 hash of the serialized file to the metadata dict.
   - At load time, verify hash before deserializing.
   - `joblib` is already a transitive dependency of `scikit-learn`.

2. **Add `requirements.txt` entry**
   - `joblib` (already included via sklearn, but make explicit).

3. **Future**: migrate to `skops.io` for truly safe serialization (Phase 2).

---

## L8. Unit Tests for Core Statistical Functions

**Priority**: Low  
**Status**: Open

### Steps

1. **Create `tests/test_core_statistics.py`**
   - `test_fractional_diff_known_output()` — compare against hand-computed or `fracdiff` package output for d=0.5 on a known series.
   - `test_triple_barrier_hand_computed()` — 10-bar series with known barriers, verify labels.
   - `test_sample_weights_no_overlap()` — non-overlapping labels → all weights = 1.0.
   - `test_sample_weights_full_overlap()` — fully concurrent labels → weights < 1.0.
   - `test_kelly_fraction_known_values()` — p=0.6, avg_win=2, avg_loss=1, fraction=1.0 → k=0.1.
   - `test_walk_forward_split_coverage()` — all indices covered by test sets.
   - `test_cpcv_split_all_combos()` — correct number of splits for given n_blocks.
   - `test_check_stationarity_stationary_series()` — white noise → stationary.
   - `test_check_stationarity_random_walk()` — cumulative sum → not stationary.

---

## L9. Feature Schema Validation

**Priority**: Low  
**Status**: Open

### Steps

1. **Store feature column list with model artifact**
   - In `TrainModelsStep`, add `last_selected_columns` to the training summary (already done).
   - In `save_model()` / registry: persist the column list.

2. **At inference time, validate**
   - In `SignalsStep` fallback path: compare incoming columns to stored `last_selected_columns`.
   - If mismatch: raise `FeatureSchemaError` with diff of missing/extra columns.

---

## L10. Portfolio-Level Risk Controls

**Priority**: Low  
**Status**: Open (noted as open question in AGENTS.md)

### Steps

1. **Create `core/portfolio.py`**
   - `PortfolioRiskOverlay(max_gross_exposure=3.0, max_net_exposure=1.5, max_portfolio_drawdown=-0.15, max_correlation_exposure=0.8)`.
   - Accept per-symbol position targets; output adjusted positions.
   - Cap total gross exposure (sum of absolute positions across symbols).
   - Cap net exposure (sum of signed positions).
   - If portfolio drawdown exceeds threshold, scale all positions proportionally.

2. **Wire into a multi-symbol pipeline orchestrator** (future scope — v2).

---

## NEW Issues (From Audit)

### NEW-C1. Walk-Forward Gap Default

**Merged with M9** — see M9 above.

### NEW-H1. Class-Balance Weight Overcorrection

**Priority**: High  

### Steps

1. **Modify `_combine_class_balance_weights()` in `core/pipeline.py`**
   - Add a `max_weight_multiplier` cap (default: 10.0). No single sample gets more than 10× the median weight.
   - Institutional standard: Man AHL and Winton cap sample weights to prevent gradient distortion.
   ```python
   combined = uniqueness_weights * class_balance
   median_weight = combined.median()
   combined = combined.clip(upper=max_weight_multiplier * median_weight)
   ```

2. **Config key**: `model.max_sample_weight_multiplier: 10.0`.

3. **For GBM**: use the combined weight approach (no `class_weight` parameter available).
   For RF: use the combined weight approach with the cap (do not additionally set `class_weight`).
   For logistic: use the combined weight approach (do not set `class_weight='balanced'` — already the case).

4. **Add test**
   - `test_weight_cap_prevents_extreme_multiplier()` — 95/5 class split → no weight > 10× median.

---

### NEW-H2. Meta-Model Inner Splits Lack CPCV Option

**Priority**: Medium  

### Steps

1. **Modify `_train_inner_meta_model()` in `core/pipeline.py`**
   - Add `meta_cv_method` config key (default: `"walk_forward"`).
   - When `meta_cv_method = "walk_forward"`: current behavior.
   - Walk-forward is acceptable for the inner meta-model because:
     - The meta-model operates on a much smaller feature space (6-14 features vs 50-100 primary features).
     - Overfitting risk is lower.
     - CPCV for the inner loop would be computationally expensive.
   - Institutional standard: meta-models use standard walk-forward with purging. CPCV is reserved for the outer loop.

2. **Ensure inner purging uses label-aware cutoffs**
   - Already done: `_purge_overlapping_training_rows()` is called. Verify it also uses `test_intervals` in CPCV mode.

---

### NEW-M1. Feature Importance Stability

**Priority**: Medium

### Steps

1. **Add `_feature_importance_stability()` in `core/models.py`**
   ```python
   def _feature_importance_stability(fold_importances):
       """Compute rank correlation of feature importances across folds.

       Returns Spearman correlation matrix and average pairwise correlation.
       Low average correlation (<0.3) indicates unstable feature importance.
       """
   ```

2. **Wire into `summarize_feature_block_diagnostics()`**
   - Compute cross-fold Spearman rank correlation of feature importance vectors.
   - Flag features whose rank varies by more than 50% of total features across folds.

---

### NEW-M2. Drawdown Kill Switch

**Merged with L1** — see L1 above.

### NEW-M3. HMM Gaussian Assumption

**Priority**: Low  

### Steps

1. **Add Student-t HMM option**
   - `hmmlearn` only supports Gaussian emissions. For Student-t:
     - Use `pomegranate` library which supports arbitrary distributions, OR
     - Pre-transform features with a rank-based normalization (rank → inverse normal CDF) before feeding to Gaussian HMM. This is the institutional standard workaround.
   - Config: `regime.hmm_preprocessing: "rank_normalize"` (default: `None`).

2. **Rank normalization**
   ```python
   from scipy.stats import rankdata, norm
   ranked = rankdata(features, axis=0) / (len(features) + 1)
   normalized = norm.ppf(ranked)
   ```
   - Apply per-column on the training slice, store the rank mapping, apply same mapping on test (use quantile mapping from train distribution).

---

### NEW-L1. Sequential Bootstrap Seed Fix

**Priority**: Low  

### Steps

1. **Modify `sequential_bootstrap()` in `core/labeling.py`**
   - Replace `np.random.choice` with `rng.choice`:
     ```python
     def sequential_bootstrap(labels, close, n_samples=None, random_state=42):
         rng = np.random.default_rng(random_state)
         ...
         drawn.append(rng.choice(n_labels, p=probs))
     ```
   - This makes results reproducible without relying on global numpy state.

---

## Implementation Priority Order

| Phase | Items | Rationale |
|-------|-------|-----------|
| **Phase 1** (immediate) | C3, H4, H6, M9, NEW-H1 | Fix result-invalidating or performance-inflating issues |
| **Phase 2** (short-term) | H5, M5, M8, M10, NEW-L1 | Improve validation quality and decontaminate sizing |
| **Phase 3** (medium-term) | M1, M2, M3, M4, M6 | Deployability infrastructure |
| **Phase 4** (longer-term) | L1-L10, NEW-H2, NEW-M1, NEW-M3 | Polish and completeness |
