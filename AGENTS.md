# Project Guidelines

## Objective

Build a self-adaptive, regime-aware AutoML trading system for Binance crypto with a research-first architecture that can retrain frequently and run rolling-window backtests across symbols, timeframes, equity assumptions, and custom data.

Treat these as fixed product requirements unless the user changes them explicitly:

- Support both Binance spot and futures through shared abstractions.
- Keep indicators modular and attachable.
- Make feature extraction from indicator outputs a first-class layer.
- Train direction/classification models.
- Keep one model per symbol.
- Share timeframe analysis logic across symbols rather than hard-forking the system by timeframe.
- Support scheduled retraining and drift-triggered retraining.
- Do not build a custom backtesting or execution engine.
- Support arbitrary timestamped tabular custom data joined with Binance market data.

## Architecture

Keep the system split into clean layers:

1. Data adapters
2. Indicator library
3. Feature graph / feature builders
4. Stationarity and transformation layer
5. Labeling layer (triple-barrier, fixed-horizon, trend-scanning)
6. Sampling layer (sample weights, sequential bootstrapping)
7. Regime detection layer
8. Training and AutoML pipeline (primary model + meta-labeling model)
9. Model registry and artifact store
10. Inference and signal generation
11. Bet sizing / position sizing layer
12. Risk and portfolio controls
13. Backtest / execution adapter

Design expectations:

- Indicators must be plug-in units that produce named outputs plus metadata.
- Feature builders must be able to consume raw OHLCV, indicator outputs, and joined custom data.
- Derived features from indicators must be easy to add: lags, slopes, distances, z-scores, ranks, crossovers, rolling statistics, regime-conditioned transforms, cross-timeframe aggregates, and fractionally differentiated series.
- All features entering the model must pass stationarity checks (ADF test). If a feature is non-stationary, apply fractional differentiation to preserve maximum memory while achieving stationarity, or apply other appropriate transforms.
- Fractional differentiation must be a built-in feature transformation, not an afterthought. It preserves long-memory and mean-reversion information that integer differencing destroys.
- Raw data must stay immutable. Derived features, labels, train/test splits, and model artifacts must be reproducible and versioned.
- All workflows must be config-driven. Avoid hard-coded symbol, timeframe, or feature assumptions.

## Backtesting And Validation

Use an existing engine rather than implementing one from scratch.

- Default to VectorBT first for v1 because it fits high-throughput research, parameter sweeps, multi-symbol comparisons, and rolling-window experiments well.
- Preserve an execution adapter boundary so NautilusTrader can replace or complement the research engine later if research-to-live parity becomes a core requirement.
- Backtests must support rolling and expanding windows with configurable train length, test length, step size, gap, and embargo.
- Backtests must support different equity assumptions, position sizing modes, fees, slippage, and symbol-specific constraints.
- Backtests must run across symbols, shared timeframe studies, and custom datasets without rewriting the pipeline.
- Report metrics per fold, per symbol, per timeframe, and in aggregate.

Validation rules:

- Never use random cross-validation for trading data.
- Use walk-forward or time-series splits.
- If labels use forward horizons or overlapping outcomes, use purging and embargo to reduce leakage.
- Weight samples by average uniqueness when labels overlap in time. Use sequential bootstrapping for any ensemble model (random forest, bagged classifiers) to avoid inflating accuracy through redundant correlated samples.
- Treat probability calibration and threshold selection as separate from model fitting.
- Optimize for trading-aware objectives, not raw accuracy alone.
- Track signal half-life and alpha decay per signal type. A signal that decays within a few bars needs different execution logic than one that persists for many bars.

## Binance-Specific Caveats

Respect exchange constraints in both research assumptions and live-facing logic.

- Binance enforces symbol filters such as tick size, lot size, min notional, notional bounds, market lot size, max order counts, and position limits.
- Binance enforces request-weight and order-count rate limits. Historical backfills and retraining jobs must batch requests carefully and prefer streaming updates where appropriate.
- Keep all internal timestamps in UTC.
- Binance klines support many intervals, but startTime and endTime are interpreted in UTC even if a timezone parameter is supplied for interval interpretation.
- Spot and futures must not share the same account/execution assumptions.
- Futures introduces leverage, funding, liquidation risk, and mark/index price semantics that must remain isolated from spot logic.
- Data ingestion must account for delistings, halted symbols, missing candles, exchange outages, and contract specification changes.

## Regime Awareness And Adaptation

Regime awareness must be explicit rather than implied.

- Support a regime detector or regime labels as a separate layer.
- Allow either regime-specific models or regime-specific thresholds without changing the rest of the pipeline.
- Monitor feature drift, prediction drift, and realized performance drift.
- ADWIN or an equivalent detector is acceptable for drift-triggered retraining, but retraining decisions must also use minimum sample thresholds and cooldown periods.
- Structural break tests (CUSUM, SADF/GSADF) may optionally complement reactive drift detection for proactive regime change identification.
- Keep champion and challenger model artifacts. Do not replace a deployed model without out-of-sample evidence and rollback support.

## AutoML Rules

- AutoML must search across feature sets, model families, and training windows in a reproducible way.
- Prefer strong simple baselines first: logistic regression, tree ensembles, gradient boosting, and calibrated classifiers.
- Add complexity only if it survives walk-forward validation after costs.
- Support a two-model architecture: a primary model for directional prediction and a meta-labeling model that learns whether the primary signal is worth acting on and how to size the bet.
- The classifier must support a no-trade / abstain class. Forcing a directional call on every bar creates unnecessary turnover and cost drag.
- Keep regime modeling, label definition, thresholding, and sizing rules separable.
- Avoid combinatorial explosion across indicators, features, windows, and hyperparameters. Modularity is required; unconstrained search is not.
- Store enough metadata to rerun any historical experiment exactly.

## Labeling Rules

- Use the triple-barrier method as the primary labeling strategy. Three concurrent barriers — profit-taking, stop-loss, and a vertical time barrier — produce labels that reflect realistic trade outcomes.
- Support fixed-horizon labeling as a simpler fallback for rapid prototyping.
- The labeling layer must be pluggable: triple-barrier, fixed-horizon, and trend-scanning must be interchangeable without changing downstream pipeline code.
- Label outcomes must record both the label class and the barrier that was hit, so meta-labeling and bet sizing have full context.

## Sampling Rules

- When labels overlap in time (as triple-barrier labels typically do), compute sample uniqueness based on the concurrency of label spans.
- Weight samples by their average uniqueness so that the model does not overfit to redundant information from overlapping outcomes.
- Use sequential bootstrapping for any bagged or ensemble model. Standard random bootstrapping on concurrent labels inflates apparent accuracy.
- Sample weight computation must be deterministic and reproducible given the same label set.

## Bet Sizing And Position Sizing

- Translate calibrated classifier probabilities into position sizes using the fractional Kelly criterion (half-Kelly or a configurable fraction).
- Bet sizing must be a separate layer from signal generation. The primary model predicts direction; the meta-labeling model (or a sizing rule) determines how much capital to allocate.
- The sizing layer must respect Binance symbol filters (lot size, min notional) after computing the raw Kelly fraction.
- Support configurable Kelly fractions and caps to account for estimation error in predicted probabilities.

## Data And Feature Rules

- Custom data must enter through a timestamped tabular contract with explicit availability timestamps.
- Joins must be point-in-time safe. Never let future-known rows leak into earlier decisions.
- Missing data policy must be explicit per feature source.
- Timeframe analysis should be shareable across symbols so the system can compare which time resolutions generalize best before training per-symbol models.
- Keep feature schemas stable across retrains or version them when they change.

## Retraining And Operations

- Support both scheduled retraining and drift-triggered retraining.
- Every retrain must log symbol, timeframe, data window, feature schema version, label definition, model family, hyperparameters, validation metrics, and artifact location.
- Persist raw data and derived artifacts in portable storage formats such as Parquet plus metadata.
- Make experiments resumable and comparable across symbols and dates.
- Keep research, paper, and live environments isolated.
- Hard risk controls must exist outside the model: max position, max leverage, max loss, exposure caps, kill switch, and data freshness checks.

## Confirmed Scope

- Market scope: both Binance spot and futures.
- Model target: direction/classification.
- Execution engine requirement: use an existing engine, not a custom one.
- Engine preference: VectorBT first, with an adapter boundary that leaves room for NautilusTrader later.
- Retraining policy: scheduled plus drift-triggered.
- Symbol policy: one model per symbol.
- Timeframe policy: timeframe analysis is shared across symbols.
- Custom data policy: arbitrary timestamped tabular data must be joinable.
- Labeling: triple-barrier primary, fixed-horizon fallback, trend-scanning supported.
- Meta-labeling: required as a core capability (secondary model for signal filtering and bet sizing).
- Sampling: sample weights by uniqueness and sequential bootstrapping required for ensemble models.
- Bet sizing: calibrated probability to fractional Kelly criterion.
- Abstain class: the classifier must support a no-trade / abstain class.
- Alpha decay: track signal half-life per signal type as a first-class metric.
- Stationarity: all features must pass ADF stationarity checks; fractional differentiation is a built-in transform.
- Structural breaks: optional complement to ADWIN (CUSUM, SADF/GSADF).
- Cost model: global flat fee model for research phase.

## Important Caveats Often Missed

- Leakage control is usually the biggest hidden failure mode in trading ML systems.
- Accuracy can improve while trading performance worsens if thresholds, costs, and class imbalance are ignored.
- Symbol-level models can still overfit shared timeframe research if the timeframe study is not validated on held-out periods.
- Futures research is misleading if funding, mark-price behavior, and liquidation constraints are omitted.
- Fast retraining is only valuable if feature generation, labeling, and artifact versioning are deterministic.
- Binance filters can invalidate trades that look fine in a naive backtest.
- A regime-aware system still needs risk overlays; regime labels alone do not make execution safe.

## Open Questions

- Is live trading a v1 requirement, or is the first milestone research and backtesting only?
- For futures, do funding, mark price, and liquidation logic belong in the first implementation wave or later?
- Should regime detection be rule-based, unsupervised, or jointly learned with the classifier?
- Is a portfolio allocator likely to be added later even though v1 is one model per symbol?

## Implementation status (progress so far)

- **Summary**: A minimal, runnable scaffold implementing the core pipeline has been added and tested locally. The indicator layer has been split into a dedicated `core/indicators/` package with one file per indicator, still using a config-driven modular registry that produces structured outputs plus metadata, and now includes a Fair Value Gap indicator. A reusable stepwise `ResearchPipeline` abstraction has also been added so future model pipelines can share orchestration without copying the end-to-end flow in `example.py`.
- **Created files**:
	- `requirements.txt` — dependency manifest (pandas, numpy, scikit-learn, statsmodels, requests)
	- `core/data.py` — `fetch_binance_vision()` (monthly ZIP download + caching)
	- `core/indicators/` — modular indicator package with `base.py`, `registry.py`, and one file per indicator (`rsi.py`, `macd.py`, `bollinger_bands.py`, `atr.py`, `fair_value_gap.py`)
	- `core/pipeline.py` — reusable stepwise `ResearchPipeline` abstraction plus pluggable pipeline steps for fetch, indicators, features, stationarity, regime detection, labels, alignment, weighting, training, signals, and backtest
	- `core/features.py` — `fractional_diff()`, `check_stationarity()`, `build_features()`
	- `core/labeling.py` — `triple_barrier_labels()`, `fixed_horizon_labels()`, `sample_weights_by_uniqueness()`, `sequential_bootstrap()`
	- `core/models.py` — `walk_forward_split()`, `train_model()`, `train_meta_model()`, `evaluate_model()`, `detect_regime()`, `save_model()`/`load_model()`
	- `core/backtest.py` — `kelly_fraction()`, `run_backtest()` (pandas-based adapter)
	- `core/__init__.py` — re-exports for `from core import ...`
	- `example.py` — end-to-end demonstration pipeline using the reusable stepwise pipeline abstraction
	- `example_fvg.py` — end-to-end demonstration pipeline using the reusable stepwise pipeline abstraction plus config-built FVG features

- **Test run (local) — 2026-04-06**:
	- Dataset: BTCUSDT 1h, 2024-01-01 → 2024-04-01 (2,184 bars)
	- Features: ~52 (includes fractional-diff `close_fracdiff`)
	- Labels: triple-barrier → 2,136 labels (1,133 long, 978 short, 25 abstain)
	- Walk-forward CV (3 folds): avg accuracy ≈ 0.455, avg F1 ≈ 0.281
	- Backtest (pandas adapter): total return 23.89%, Sharpe 8.48, max drawdown -5.26%, trades 43, win rate 61.21%
	- Fractional differentiation made `close` stationary (ADF p=0.000142)
	- FVG example (`example_fvg.py`): ~100 features, 168 aligned samples, avg accuracy ≈ 0.476, avg F1 ≈ 0.330, backtest return 7.39%, Sharpe 4.11, max drawdown -7.06%, trades 15, win rate 58.54%

- **How to run locally**:

```bash
python -m pip install -r requirements.txt
python example.py
python example_fvg.py
```

- **Next steps (planned)**:
	- Add model registry (MLflow/W&B adapter), scheduled retraining, and drift-triggered retraining (ADWIN).
	- Integrate VectorBT as an optional backtest engine via an adapter boundary.
	- Add unit tests and CI for core modules and a reproducible experiment runner.
	- Replace pickle-based artifact store with a proper registry (optional).

If you'd like, I can now add unit tests or integrate VectorBT — which should I do next?
