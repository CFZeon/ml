# GAPS - Current-State Priority Backlog

This document is the current verified gap backlog for the repository after reconciling:

- the current codebase,
- the passing test suite,
- repository memory notes, and
- external professional standards.

It intentionally does not repeat older backlog items that are already implemented. The goal is to capture the remaining gaps that still matter for research integrity, execution realism, and production readiness.

## Scope and Method

- The priority order below reflects the latest verified audit, not older historical gap lists.
- Each item includes the professional standard, the concrete repo evidence, a detailed implementation plan, acceptance criteria, and a test plan.
- The highest priority items are the ones most likely to distort results or create false confidence in the research outputs.

## Professional Standards Used

- Nested model selection and selection-bias control:
  https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- Time-aware validation and explicit train-test gaps:
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Sequential bootstrapping for overlapping financial labels:
  https://hudsonthames.org/bagging-in-financial-machine-learning-sequential-bootstrapping-python/
- Lookahead-bias detection patterns for trading systems:
  https://www.freqtrade.io/en/stable/lookahead-analysis/
- Point-in-time joins and historical feature reconstruction:
  https://docs.feast.dev/getting-started/concepts/point-in-time-joins
- Exchange order filters and symbol constraints:
  https://developers.binance.com/docs/binance-spot-api-docs/filters
- Data and prediction drift monitoring:
  https://docs.evidentlyai.com/metrics/explainer_drift
- Streaming drift detection with ADWIN:
  https://riverml.xyz/dev/api/drift/ADWIN/
- Model registry lifecycle, versioning, immutability, and archive semantics:
  https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models?view=azureml-api-2
- Serialization safety warning for pickle:
  https://docs.python.org/3/library/pickle.html

## Already Implemented and Not Repeated Here

- Deflated Sharpe Ratio and CPCV-style overfitting diagnostics are already integrated into AutoML selection reporting.
- A three-stage search, validation, and locked-holdout flow already exists.
- CPCV is already available and is the default validation path in current repo memory.
- Fold-local regime fitting, fold-local stationarity transforms, and fallback-signal-window hardening are already in place.
- Stationary-bootstrap significance reporting already exists in the backtest layer.
- Dynamic slippage models already exist.
- Kelly sizing decontamination is already in place.
- P0-1 is complete: AutoML promotion now uses an explicit selection-policy gate stack with complexity, gap, fragility, and holdout-aware eligibility checks.
- P0-2 is complete: the default AutoML objective is now trading-first (`risk_adjusted_after_costs`) and classification metrics are treated as gates and diagnostics.
- P0-3 is complete: RandomForest training is now concurrency-aware and can use deterministic sequential bootstrapping under high label overlap.
- P0-4 is complete: pandas and vectorbt now share a pre-engine execution contract and parity tests cover simple deterministic scenarios.
- P0-5 is complete: Binance filter parsing and centralized order validation now cover materially relevant research-phase rules beyond the earlier partial subset.
- P0-6 is complete: execution and valuation price normalization no longer future-backfill by default; causal fill policies and diagnostics are now explicit.
- P1-7 is complete: Binance data fetching now supports bounded retry/backoff, explicit gap policies, and structured integrity reports surfaced into pipeline state.
- P1-8 is complete: point-in-time custom joins now fail closed by default, support TTL-style freshness constraints, and surface join audit diagnostics.
- P1-9 is complete: feature-family provenance, family diagnostics, and endogenous-only detection are now first-class outputs across the pipeline.
- P1-10 is complete: trial-return alignment and CPCV/PBO diagnostics now preserve missingness by default and enforce explicit overlap policies.
- P1-11 is complete: fold-stability diagnostics are now computed in training artifacts, surfaced in reporting, and usable as an AutoML promotion gate.
- P2-12 is complete: futures backtests now support contract metadata, leverage-bracket inputs, isolated/cross margin modes, mark-price margin tracking, and liquidation events with explicit reporting.

The gaps below start from that hardened baseline.

## Priority Order

### P2 - Futures Realism and Operating Model

13. Deployment governance is still incomplete: no implemented drift-monitoring loop, no local registry workflow, no champion-challenger promotion flow, and raw pickle persistence is still present.

---

## P2-13. Implement Deployment Governance: Drift Monitoring, Local Registry, Promotion Flow, and Safe Artifacts

**Why this is P2**

- The repo is already beyond toy-research scaffolding, but it still lacks the operating model needed to retrain, compare, promote, and roll back models safely.
- This is the largest remaining gap between research code and a controlled production workflow.

**Professional standard**

- Evidently-style drift monitoring uses explicit reference versus current distributions and dataset-level drift rules.
- River's ADWIN gives a streaming detector with clear significance and window semantics.
- Model registries use immutable versions, mutable tags and descriptions, archive states, and explicit lineage from run to artifact.
- Python's own documentation warns that pickle is not secure and should only be unpickled from trusted sources.

**Repo evidence**

- `core/models.py::save_model()` and `load_model()` still use raw pickle.
- There is no registry module, no version manifest, no champion-challenger lifecycle, and no rollback workflow.
- No drift-monitoring module exists in the codebase.

**Repo touchpoints**

- `core/models.py`
- `core/automl.py`
- `core/pipeline.py`
- `requirements.txt`
- `tests/test_automl_holdout_objective.py`

**Implementation plan**

1. Replace raw artifact persistence with a local registry layout.
   - Add a registry folder structure such as `artifacts/registry/<symbol>/<model_name>/<version>/`.
   - Store: model artifact, config snapshot, feature schema, label spec, dataset window, metrics summary, lineage metadata, and a manifest file.

2. Make model versions immutable.
   - Only descriptions, tags, and status should be mutable after registration.
   - Promotion should create or update a status tag such as `candidate`, `champion`, `archived`, or `rejected`.

3. Add champion-challenger promotion logic.
   - Promotion should require locked-holdout pass, stability pass, and registry metadata completeness.
   - Challenger deployment should preserve the previous champion for rollback.

4. Add drift monitoring primitives.
   - Batch drift: feature drift and prediction drift using KS, PSI, Wasserstein, or JS-based tests.
   - Streaming drift: ADWIN on calibrated probabilities, realized edge, or trade-win process.
   - Retraining triggers should require minimum sample thresholds and cooldown windows.

5. Add safe artifact verification.
   - Replace raw pickle persistence with `joblib` as an immediate step, plus artifact hashing.
   - Keep a path open for `skops` or another safer format later.
   - Verify hash integrity before loading.

6. Add schema validation at load time.
   - The registry should store the exact ordered feature schema and training-time feature-family metadata.
   - Inference should fail closed on schema mismatch.

7. Integrate registry writes with AutoML outputs.
   - The best selected trial should be registrable as a versioned artifact with its evaluation lineage, not just a returned Python object.

**Acceptance criteria**

- Models can be registered with immutable versioned metadata and archived without deletion.
- The repo supports champion and challenger status assignments with rollback capability.
- Drift metrics and retraining triggers are stored in a structured report.
- Raw pickle-only persistence is removed from the main model-storage path.

**Test plan**

- Add a registry test that creates a model version and verifies manifest completeness.
- Add a rollback test where a challenger is archived and the previous champion remains loadable.
- Add drift tests covering both batch drift and ADWIN-triggered alerts.
- Add a hash-verification test proving tampered artifacts are rejected at load time.

---

## Recommended Execution Order

1. Complete P2-13 before any scheduled or drift-triggered retraining workflow is treated as production-capable.

## Exit Condition for Calling the Research Stack "Professionally Defensible"

The repository should only be considered professionally defensible for research-to-deployment promotion when all of the following are true:

- AutoML defaults to a trading-first objective and rejects fragile winners.
- RandomForest training is concurrency-aware.
- Backtest engines share one execution contract.
- Binance constraints are centrally validated.
- No future-backfill remains in price normalization.
- Data completeness and point-in-time joins fail closed by default.
- Overfitting diagnostics use observed-only overlaps.
- Fold stability gates are enforced.
- Futures leverage can liquidate.
- Models are versioned, drift-monitored, and safely loadable.
