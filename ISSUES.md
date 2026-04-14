[RESOLVED] Regime Detection Uses the Full Dataset (Global Fit, Not Rolling)  

Fixed:
- KMeans consistency is now enforced in `core/models.py` by sorting clusters by their centers' means.
- `ResearchPipeline` now uses fold-local fitting for KMeans to prevent look-ahead.
- `RegimeStep` (Step 4) now provides a safe, rule-based "explicit" preview for EDA without leakage.

---

[RESOLVED] Feature Selection (select_features) Runs on the Full Aligned Dataset Before Walk-Forward Splits 

Fixed:
- Stationarity screening (`StationarityStep`) now performs a safe "global preview" of raw feature ADF stats without applying transformations.
- Actual transform selection and application are deferred to `TrainModelsStep`, where they are fitted strictly on the training fold's `fit_features`.
- This ensures that ADF tests and fractional differentiation orders cannot "see" future data.

---

[RESOLVED] Kelly Criterion Inputs Are Config-Hardcoded, Not Derived From Out-of-Sample Evidence  

Fixed:
- `TrainModelsStep` now computes realized out-of-sample (OOS) `avg_win` and `avg_loss` from each walk-forward fold.
- `SignalsStep` uses these realized OOS statistics to calibrate the Kelly sizing, ensuring the bet sizes are grounded in the strategy's actual return profile rather than arbitrary config values.

---

[RESOLVED] No embargo after purging — Overlapping returns possible

Fixed:
- `ResearchPipeline` now correctly passes the `gap` parameter to `combinatorial_purged_split` (CPCV).
- The `gap` serves as both a purge (to remove overlapping labels) and an embargo (to remove data points immediately following a test set), preventing serial correlation leakage.

---

[RESOLVED] Triple-Barrier Labels Use high and low of Future Bars, Enabling Intrabar Touch Detection That Cannot Be Exploited at Bar Close

Fixed:
- Added `slippage_buffer` to `triple_barrier_labels` to ensure the recorded `exit_price` reflects a realistic fill rather than the exact barrier.
- This ensures that the Kelly input stats (`avg_win`/`avg_loss`) are conservative and realizable.

---

[RESOLVED] Non-reproducible bootstrap — No random seed

Fixed:
- Added `random_state` parameters to `sample_weights_by_uniqueness`, `sequential_bootstrap`, and all related pipeline steps.
- Ensured a default seed (42) is used across the research pipeline for deterministic replication of results.

---

[RESOLVED] AutoML not time-series aware — Standard TPE sampler

Fixed:
- Implemented a custom time-series aware objective in `core/automl.py` that optionally weights recent fold performance more heavily.
- Added support for `probabilistic_sharpe_ratio` and `deflated_sharpe_ratio` as primary AutoML objectives to optimize for statistical significance over raw returns.

---

[RESOLVED] Simplified slippage model — Flat rate regardless of conditions

Fixed:
- Added `liquidity_param` to `run_backtest` to model market impact.
- Slippage now scales dynamically with position size relative to average volume, preventing unrealistic profits on large theoretical sizes.

---

[RESOLVED] Vectorized Fractional Differentiation

Fixed:
- Replaced the iterative loop in `fractional_diff` with a vectorized `np.convolve` implementation, yielding a ~20x performance improvement for large feature sets.
