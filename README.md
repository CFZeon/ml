# Regime-Aware Crypto AutoML Research Stack

This repository is a research-first trading stack for Binance crypto that keeps model training, feature generation, validation, and execution assumptions explicit.

Trade-ready example configs now fail closed on stale context and missing futures funding coverage instead of silently zero-filling unknown state.
They also fail closed on conflicting duplicate market bars instead of silently keeping the first timestamp collision.
Trade-ready configs now also build one blocking pre-training data-certification verdict that binds market gaps, quarantine disposition, context TTL breaches, and configured reference-validation coverage into a single gate.
Trade-ready and AutoML training paths now run a blocking baseline-vs-prefix lookahead certification on the causal feature surface before model fitting.
Example builders now mark backtests as `research_only` by default; trade-ready runs must opt into event-style execution plus explicit stress scenarios. The default trade-ready certification path still fails closed when Nautilus is unavailable, while `example_trade_ready_automl.py --smoke` now degrades explicitly into a labeled research-only surrogate path for local feedback.
Trade-ready monitoring now also defaults to a finite policy profile instead of leaving freshness, fill-quality, slippage-drift, and signal-decay thresholds at `inf` or `None`.

The current codebase is built around these constraints:

- one model per symbol
- shared timeframe analysis across symbols
- classification and meta-labeling workflows
- point-in-time-safe custom data joins
- walk-forward and CPCV-style validation instead of random CV
- execution-aware backtests with Binance filter enforcement
- futures-aware research with funding, mark-price valuation, and a liquidation-capable margin model

## What Is Implemented

The repo now includes:

- Binance spot and futures data loaders with cacheing, retry/backoff, and integrity reporting
- modular indicators in `core/indicators/` including RSI, MACD, Bollinger Bands, ATR, Fair Value Gap, ADX, stochastic oscillator, on-balance volume, and Donchian channels
- feature engineering with fractional differentiation and ADF-based stationarity screening
- triple-barrier, fixed-horizon, and trend-scanning labels
- uniqueness weighting and sequential bootstrap support
- explicit regime detection and context features
- AutoML with validation/holdout separation, CPCV/PBO diagnostics, DSR reporting, White RC / Hansen SPA post-selection inference, fragility checks, and stability gating
- a baseline-vs-prefix lookahead provocation harness for detecting future-informed features, labels, probabilities, signals, and execution inputs, with automatic blocking feature-surface certification on trade-ready and AutoML runs
- a causal liquidity resolver that shifts bar-volume inputs, validates L2 snapshot timestamps, and records liquidity provenance in backtest outputs
- explicit order-intent and execution-policy objects for backtests, with a legacy full-fill parity path and a default event-style partial-fill/cancel flow
- a tiered execution cost stack with proxy, L2 depth-curve, and fill-event attribution modes plus stress sweeps
- feature admission and retirement governance with venue-specific tagging, transform/lineage metadata, robustness screens, and ablation-based promotion diagnostics
- a dedicated regime layer with explicit instrument-state, market-state, and cross-asset-state inputs, provenance reports, and endogenous-vs-context stability ablations
- a pre-feature data-quality quarantine layer that flags, nulls, drops, or winsorizes suspicious bars and records structured anomaly reports
- a unified trade-ready data-certification report that consolidates market-bar integrity, quarantine status, context TTL breaches, and configured cross-venue reference validation into one promotion-facing verdict
- historical universe snapshots that gate cross-symbol studies by listing status, minimum history, and liquidity, plus lifecycle-aware backtest actions for halts and delists
- explicit cross-stage embargoes between AutoML search, validation, and locked holdout windows so label horizons and execution delays cannot bleed across stage boundaries
- safe persistent artifact storage with Parquet/JSON caches and skops-backed model bundles that verify hashes and feature schema before load
- a local registry with immutable version manifests, champion/challenger rollback flows, offline promotion decisions, and drift-report attachments
- drift monitoring with PSI/KS feature checks, prediction KL divergence, and ADWIN-backed performance drift hooks with minimum-sample and cooldown guardrails
- venue-failure scenario schedules for downtime, stale marks, halts, leverage caps, forced deleveraging, and reproducible stress-matrix replays
- operations-centric monitoring for raw-data freshness, custom-data TTL breaches, L2 snapshot age, feature schema drift, fill-quality deterioration, slippage drift, and inference latency/backlog
- pandas and vectorbt execution adapters behind a shared order-validation contract
- futures backtests with funding, mark-price valuation, leverage-bracket caps, isolated/cross margin modes, and liquidation events

## Repository Layout

- `core/data.py`: Binance data access, symbol filters, point-in-time custom joins, futures contract metadata, leverage bracket adapters
- `core/context.py`: futures context and cross-asset context loaders
- `core/features.py`: feature construction, family metadata, stationarity screening, supervised selection
- `core/labeling.py`: event labeling and uniqueness weighting helpers
- `core/models.py`: model training, diagnostics, validation helpers
- `core/regime.py`: regime feature construction, provenance tracking, endogenous-vs-context ablations, and HMM / explicit regime detection
- `core/scenarios.py`: venue downtime, stale-mark, halt, deleveraging schedules plus stress-matrix replay helpers
- `core/automl.py`: Optuna-backed search, ranking, holdout logic, and overfitting diagnostics
- `core/backtest.py`: execution-aware backtests, slippage models, and futures margin/liquidation simulation
- `core/monitoring.py`: operational health reports, schema checks, execution-quality drift summaries, and local artifact emission
- `core/execution/costs.py`: proxy, depth-aware, and fill-aware execution cost models plus fill-event attribution
- `core/reference_data.py`: generic reference-overlay feature adapters for future multi-exchange feeds
- `core/feature_governance.py`: feature metadata, admission/retirement rules, portability summaries, and promotion-gate diagnostics
- `core/data_quality.py`: bad-print detection, quarantine actions, and structured anomaly reporting before features and labels
- `core/universe.py`: historical universe snapshots, eligibility gates, and symbol lifecycle handling for halts/delists
- `core/storage.py`: shared JSON/Parquet persistence helpers and SHA-256 verification primitives for safe caches and model manifests
- `core/drift.py`: batch and streaming drift detection plus retraining guardrails
- `core/readiness.py`: operator-facing deploy/hold report that binds promotion status, monitoring health, drift state, backend availability, and rollback readiness
- `core/registry/`: immutable version manifests, local registry index, and champion/challenger promotion flows
- `core/execution/intents.py`: order-intent data structures emitted before execution simulation
- `core/execution/policies.py`: execution adapter and fill-policy resolution for backtests
- `core/execution/nautilus_adapter.py`: NautilusTrader adapter boundary with explicit unavailability errors and research-only surrogate metadata
- `core/execution/liquidity.py`: causal bar-volume and order-book liquidity input resolution
- `core/stat_tests.py`: White Reality Check, Hansen SPA, and aligned candidate return-matrix helpers
- `core/pipeline.py`: stepwise research pipeline orchestration
- `core/lookahead.py`: baseline-plus-prefix replay audit for lookahead bias provocation
- `example_active_spot.py`, `example_active_futures.py`: runnable active-trading demos
- `example_trend_volume_spot.py`, `example_trend_breakout_futures.py`: runnable expanded-indicator demos
- `example.py`, `example_custom_data.py`, `example_futures.py`, `example_fvg.py`, `example_synthetic_derivatives.py`, `example_automl.py`, `example_local_certification_automl.py`, `example_trade_ready_automl.py`: runnable end-to-end examples and smoke/integration demos
- `example_drift_retraining_cycle.py`: deterministic champion/challenger drift orchestration example with rollback
- `tests/`: regression coverage for validation, joins, execution semantics, AutoML governance, and futures behavior

## Example Guide

If your goal is to build your own scenario instead of just running the shipped demos, read `HOW_TO_USE.md` first and then copy `example_test_case_template.py`.

If you want examples that reliably place trades under the current pipeline contracts, start with the active demos:

- `example_active_spot.py`: spot workflow using cross-asset context, CPCV training, and a more permissive signal policy so the demo produces executed spot trades instead of mostly abstaining; add `--local-certification` to reuse the same case under the strict local certification runtime
- `example_active_futures.py`: USDT-M futures workflow using mark-price valuation, isolated-margin account rules, and an active long/short signal policy on the pandas execution adapter; add `--local-certification` to reuse the same case under the strict local certification runtime

The rest of the examples serve different purposes:

- `example.py`: baseline end-to-end spot research workflow with conservative settings; add `--local-certification` to switch the same data case into the strict local certification runtime
- `example_custom_data.py`: point-in-time-safe custom-data join example; add `--local-certification` to certify the same joined feature surface under strict local runtime guards
- `example_futures.py`: conservative futures example using real Binance data and the liquidation-aware adapter; add `--local-certification` to run it under the strict local certification runtime
- `example_trend_volume_spot.py`: spot example that blends RSI/MACD/Bollinger/ATR with ADX, stochastic, OBV, and Donchian features; add `--local-certification` to promote the same scenario into the strict local certification runtime
- `example_trend_breakout_futures.py`: futures example focused on ADX plus Donchian trend-breakout context layered onto the existing futures pipeline; add `--local-certification` to promote the same scenario into the strict local certification runtime
- `example_fvg.py`: Fair Value Gap feature example; useful as a feature smoke test and may legitimately abstain; add `--local-certification` when you want the same scenario to run under strict local certification runtime defaults
- `example_synthetic_derivatives.py`: offline synthetic derivatives/integration example; may also abstain depending on the generated regime path
- `example_local_certification_automl.py`: strict local certification path with locked holdout, replication, fail-closed data handling, and a hard local Nautilus requirement; intended for paper or pre-capital certification on consumer hardware
- `example_trade_ready_automl.py`: hardened AutoML certification profile with locked holdout, replication cohorts, DSR/PBO diagnostics, blocking pre-training feature-surface lookahead certification, and promotion-readiness reporting; the default run still fails closed if Nautilus is unavailable, while `--smoke` executes an explicitly reduced-power research-surrogate path for local feedback
- `example_drift_retraining_cycle.py`: deterministic registry and drift example showing scheduled retraining, challenger promotion, rollback, and the final operator deploy/hold decision
- `example_automl.py`: constrained AutoML smoke/demo path kept for short runtime feedback; any post-selection rebuild is explicitly labeled as a research refit artifact, not untouched OOS evidence

The end-to-end remediation program for making the repo trade-ready is tracked in `TRADE_READY_REMEDIATION_PLAN.md`.

The user-facing entry points are now intentionally separated:

- `example_automl.py`: research-only demo
- `example_local_certification_automl.py`: strict local certification on consumer hardware
- `example_trade_ready_automl.py`: stricter operator-facing certification and promotion path

For the builder-based real-data demos, local certification is also available as a shared runtime switch instead of a separate script. These examples now accept `--local-certification`:

- `example.py`
- `example_futures.py`
- `example_custom_data.py`
- `example_active_spot.py`
- `example_active_futures.py`
- `example_trend_volume_spot.py`
- `example_trend_breakout_futures.py`
- `example_fvg.py`
- `example_test_case_template.py`

That switch applies the strict local-certification runtime profile, requires a local Nautilus installation, and fails closed rather than silently downgrading to research mode.

The operator path is now distinct from the certification path: certify a candidate with `example_local_certification_automl.py` or `example_trade_ready_automl.py`, then hand off the promoted champion to `example_drift_retraining_cycle.py` or `ResearchPipeline.inspect_deployment_readiness(...)` for the final deploy-versus-hold decision.
The hardened trade-ready path now also auto-applies the `trade_ready` monitoring profile, which binds finite thresholds for freshness, custom-data fallback, fill quality, slippage drift, and signal-decay deterioration.
`example_utils.py` now also exposes `build_trade_ready_runtime_overrides(...)`, which centralizes the fail-closed trade-ready defaults for market-data gaps, duplicate-bar handling, quarantine blocking, and futures funding coverage.
Trade-ready runs now also inherit a binding significance floor: the runtime enables significance by default, requires a minimum observation count for statistical-significance payloads, and the trade-ready AutoML profile reports explicit underpowered-evidence reasons instead of silently treating missing confidence bounds as a generic gate failure.

The shared example builders in `example_utils.py` now enable strict context-missing and futures-funding coverage policies by default. If a cross-asset leader goes stale, a futures context feed ages out, or an expected funding event is missing, the example path stops with an explicit gate failure instead of treating that unknown state as a tradable zero.
They also set `data.duplicate_policy = "fail"`, so conflicting duplicate bars or restated timestamp collisions stop the run instead of being silently collapsed.
They also set `data.futures_context.recent_stats_availability_lag = "period_close"`, so Binance recent-stat context is aligned to publication-safe timestamps rather than the raw interval it summarizes.
They also default to `backtest.evaluation_mode = "research_only"`. Only the hardened trade-ready AutoML path opts into `trade_ready` evaluation with event-style execution and an explicit stress matrix.
When you do opt into `trade_ready`, the shared runtime now defaults `data.gap_policy = "fail"`, keeps `data.duplicate_policy = "fail"`, blocks on any quarantined rows, and upgrades futures funding coverage to strict mode unless you explicitly mark the run as research-only with `backtest.research_only_override = true`.
The hardened trade-ready AutoML profile now also sets a minimum statistical-evidence floor: certification runs require at least 64 significance observations, while `--smoke` keeps a visibly reduced 32-observation floor so local feedback stays distinct from certification-grade evidence.
The hardened trade-ready AutoML example now also enables a blocking data-certification contract before training. That contract fails closed on market gaps, data-quality quarantine, context TTL breaches, and missing or failing configured reference validation.
Trade-ready and AutoML runs now also auto-enable a blocking lookahead replay over the causal feature surface before training. You can still widen the audit explicitly, but the default blocking surface is limited to pre-training features so label maturation does not generate false positives.
The hardened trade-ready AutoML override now also enables replication cohorts by default, so a candidate must survive alternate windows or sibling-symbol cohorts before it can present as promotion-ready.
The default trade-ready example now uses a stronger certification budget with higher validation-trade, replication-coverage, and post-selection requirements. If you choose `python example_trade_ready_automl.py --smoke`, the script declares the run as reduced-power and keeps that distinction visible in the AutoML summary.
Surrogate execution remains research-only. `example_trade_ready_automl.py` still exits early on the default certification path when Nautilus is unavailable, but `python example_trade_ready_automl.py --smoke` now makes that downgrade explicitly and labels the run as a research surrogate in the summary.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Quick Start

Run the active spot example:

```bash
python example_active_spot.py
```

Run the same active spot example under the shared local-certification runtime:

```bash
python example_active_spot.py --local-certification
```

Open the guided onboarding doc:

```bash
python -c "from pathlib import Path; print(Path('HOW_TO_USE.md').resolve())"
```

Run the copy-and-edit template:

```bash
python example_test_case_template.py
```

Run the copy-and-edit template under the shared local-certification runtime:

```bash
python example_test_case_template.py --local-certification
```

Run the active futures example:

```bash
python example_active_futures.py
```

Run the baseline research example:

```bash
python example.py
```

Run the futures example with mark-price valuation and the liquidation-aware margin model:

```bash
python example_futures.py
```

Run the custom-data example:

```bash
python example_custom_data.py
```

Run the expanded-indicator spot example:

```bash
python example_trend_volume_spot.py
```

Run the expanded-indicator futures example:

```bash
python example_trend_breakout_futures.py
```

Run the test suite:

```bash
python -m pytest
```

Run the hardened AutoML example:

```bash
python example_trade_ready_automl.py
```

Run the local certification AutoML example:

```bash
python example_local_certification_automl.py
```

Run the drift retraining example:

```bash
python example_drift_retraining_cycle.py
```

## Futures Margin Model

P2-12 added a futures account overlay on top of the existing execution adapter.

It supports:

- `isolated` and `cross` margin modes
- mark-price-driven unrealized PnL tracking
- maintenance-margin and margin-ratio monitoring
- leverage-bracket-based exposure caps
- liquidation events with explicit liquidation fees
- reporting of realized leverage, warning-bar counts, funding paid/received, and liquidation summaries

The main configuration surface lives under `backtest.futures_account`.

Example:

```python
"backtest": {
	"engine": "pandas",
	"valuation_price": "mark",
	"apply_funding": True,
	"allow_short": True,
	"leverage": 3.0,
	"futures_account": {
		"enabled": True,
		"margin_mode": "isolated",
		"warning_margin_ratio": 0.8,
		"leverage_brackets_data": {
			"symbol": "BTCUSDT",
			"brackets": [
				{
					"bracket": 1,
					"initial_leverage": 20.0,
					"notional_floor": 0.0,
					"notional_cap": 50000.0,
					"maint_margin_ratio": 0.02,
					"cum": 0.0,
				}
			],
		},
	},
}
```

Supported bracket sources:

- `leverage_brackets_data`: inline normalized bracket payload in config
- `leverage_brackets_path`: JSON file containing either normalized rows or Binance-style bracket payloads
- `use_signed_leverage_brackets: true`: fetch from Binance's signed leverage-bracket endpoint and cache locally

If you use the signed endpoint, provide credentials via config or environment variables:

- `BINANCE_API_KEY` / `BINANCE_API_SECRET`
- or `BINANCE_FAPI_KEY` / `BINANCE_FAPI_SECRET`

Public contract metadata is loaded from futures `exchangeInfo` and exposed separately from spot/futures execution filters.

## Validation Philosophy

The repo does not treat headline accuracy as sufficient evidence.

Current safeguards include:

- walk-forward and CPCV validation paths
- purging-aware training flow for overlapping labels
- fold-local regime and stationarity fitting
- regime provenance and endogenous-vs-context ablation reports with stability-based promotion gates
- feature admission summaries that combine stationarity, rolling-sign stability, regime robustness, perturbation sensitivity, and retirement filters
- prefix-only replay audits that compare baseline outputs against truncated reruns at sampled decision timestamps
- holdout-aware AutoML promotion
- explicit search/validation and validation/holdout gaps derived from label horizon, signal delay, and embargo settings
- safe artifact loads that fail closed on hash mismatches or feature-schema drift
- local registry decisions that preserve immutable version manifests while tracking promotion and rollback state separately
- drift guardrails that require minimum sample thresholds, cooldown windows, and multiple confirming signals before retrain recommendations
- venue-failure scenario replays that stress downtime, stale-mark, and halt policies before promotion
- operational monitoring artifacts that can gate promotion on freshness, schema, execution, and latency health
- deflated Sharpe and PBO diagnostics
- fold-stability reporting with optional rejection gates
- execution-aware backtests with slippage, fees, and Binance constraint handling

## Notes

- The liquidation-aware futures simulator is validated on the included linear futures workflows and synthetic tests. If you add inverse-contract research, provide explicit contract metadata and bracket inputs rather than assuming spot-like quantity semantics.
- The vectorbt adapter remains available for standard execution-aware backtests. The futures account overlay uses the pandas path because liquidation and margin-state transitions are simulated explicitly.
- Custom data joins now fail closed by default unless availability semantics are explicit.

## Current Status

The core V1.5 hardening stack now includes locked holdouts, signal-policy separation, lookahead provocation, causal liquidity inputs, post-selection inference, event-style execution simulation, microstructure-aware costs, venue-portability plus feature-admission governance, a provenance-aware regime layer redesign, pre-feature data-quality quarantine, historical universe snapshots for survivorship-aware cross-symbol research, explicit cross-stage embargoes between search, validation, and locked holdout windows, safer persistent storage for data/context caches plus model artifacts, a local registry plus drift-governed promotion flow, venue-failure stress scenarios, and operations-centric monitoring artifacts.

The implementation backlog in `IMPROVEMENTS.md` is now complete for the listed V1.5 remediation items.