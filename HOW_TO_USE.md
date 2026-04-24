# How To Use This Repository

This repository has outgrown the point where reading `README.md` and one example file is enough to understand how to build your own case. The practical way to use it now is:

1. Pick the closest example family.
2. Start from the shared config builders in `example_utils.py`.
3. Apply a small override block instead of rewriting a full config.
4. Use the offline seeding path when you want deterministic tests.

If you want the shortest path to your own runnable scenario, start from `example_test_case_template.py`.

## Start Here

Use this map instead of scanning every example manually.

| Goal | Start with | Why |
| --- | --- | --- |
| Spot workflow that usually places trades | `example_active_spot.py` | Uses a more permissive signal policy than the conservative baseline |
| Futures workflow that usually places trades | `example_active_futures.py` | Uses the pandas futures adapter with long/short execution and margin rules |
| Conservative spot research baseline | `example.py` | Smallest real-data baseline for the full research path |
| Conservative futures research baseline | `example_futures.py` | Shows mark-price valuation, funding, and liquidation-aware futures setup |
| Attach custom exogenous data | `example_custom_data.py` | Demonstrates point-in-time-safe custom data joins |
| Run an offline deterministic case | `example_synthetic_derivatives.py` | Seeds `pipeline.state` directly and avoids network fetches |
| Run trade-ready AutoML research | `example_trade_ready_automl.py` | Uses the hardened AutoML profile with locked holdout, replication cohorts, binding selection gates, and promotion-readiness reporting; if Nautilus is unavailable, it fails closed unless you explicitly enable a labeled research-only override |
| Run drift-governed retraining flow | `example_drift_retraining_cycle.py` | Shows champion/challenger registration, scheduled retraining, and hybrid rollback |
| Run AutoML smoke/demo path | `example_automl.py` | Shows the searchable config surface quickly, but intentionally disables promotion-safe controls |
| Explore FVG-specific features | `example_fvg.py` | Narrow feature example; useful as a feature smoke test |
| Explore wider indicator families | `example_trend_volume_spot.py` and `example_trend_breakout_futures.py` | Show how to widen the indicator stack without changing the rest of the pipeline |

## Recommended Workflow

For a new research case, do not start by copying one of the largest inline configs from older examples.

Start from the shared builders in `example_utils.py`:

- `build_spot_research_config(...)`
- `build_futures_research_config(...)`
- `clone_config_with_overrides(base_config, overrides)`
- `build_custom_data_entry(...)`
- `seed_offline_pipeline_state(...)`

That gives you one stable base config plus a short diff that contains only your experiment-specific changes.

The shared builders now include strict context and funding integrity guardrails by default. If cross-asset context goes stale or a futures funding event is missing, the pipeline fails closed with an explicit gate error instead of converting that unknown state into zeros.
They also set `data.duplicate_policy = "fail"`, so conflicting duplicate market bars raise immediately instead of being silently de-duplicated.
They also set `data.futures_context.recent_stats_availability_lag = "period_close"`, so recent Binance futures statistics are indexed at publication-safe timestamps instead of the interval they summarize.
They also default to `backtest.evaluation_mode = "research_only"`, so a normal builder-based example is explicitly research-grade unless you promote it to a trade-ready evaluation profile.
The hardened trade-ready AutoML override also enables replication cohorts by default, so promotion-readiness is checked on alternate windows or sibling symbols instead of a single holdout narrative.
That hardened trade-ready config still requires a real Nautilus backend. The shipped `example_trade_ready_automl.py` runner now fails closed unless you explicitly set `backtest.research_only_override = true` for a labeled research-only fallback.

## Hardened Defaults Migration

The remediation cycle changed several defaults that older local configs may have relied on.

1. AutoML now defaults to the hardened promotion policy profile. Locked-holdout and post-selection failures block promotion unless you explicitly opt into `automl.policy_profile = "legacy_permissive"`.
2. Context features now default to `preserve_missing` semantics with unknown-state indicators instead of zero-filling stale or missing context. If you intentionally want legacy smoke-test behavior, set `compat.legacy_missing_semantics = true` or override `features.context_missing_policy.mode = "zero_fill"`.
3. Trade-ready backtests now default to strict futures funding coverage and fail closed on surrogate execution. If you want a surrogate study, keep `backtest.evaluation_mode = "research_only"` or set `backtest.research_only_override = true` explicitly.
4. Custom `features.builders` now trigger an automatic lookahead provocation audit before training. Builder leaks block training in blocking mode and still surface as promotion blockers in advisory mode.
5. Objective evidence gates are materially stricter by default and can optionally require lower confidence bounds for Sharpe and net profit.

## Build A New Real-Data Case

The normal pattern is:

```python
from core import ATR, BollingerBands, MACD, RSI, ResearchPipeline
from example_utils import build_spot_research_config, clone_config_with_overrides

base_config = build_spot_research_config(
    symbol="BTCUSDT",
    interval="1h",
    start="2024-01-01",
    end="2024-06-01",
    indicators=[RSI(14), MACD(), BollingerBands(20), ATR(14)],
    context_symbols=["ETHUSDT", "SOLUSDT"],
)

config = clone_config_with_overrides(
    base_config,
    {
        "signals": {
            "threshold": 0.0,
            "edge_threshold": 0.0,
            "meta_threshold": 0.5,
        },
        "backtest": {
            "signal_delay_bars": 1,
        },
    },
)

pipeline = ResearchPipeline(config)
```

That is now the pattern used by the main examples.

What to change first:

1. `data.symbol`, `data.interval`, `data.start`, `data.end`
2. `indicators`
3. `labels`
4. `signals`
5. `backtest`

What not to remove casually:

1. `universe` when you use cross-asset context
2. `data.market` for futures cases
3. `valuation_price`, `apply_funding`, `funding_missing_policy`, and `futures_account` for realistic futures backtests

## Build A Futures Case

Use `build_futures_research_config(...)` instead of switching a spot config by hand. The futures builder already carries the important futures defaults:

- `data.market = "um_futures"`
- futures cross-asset context market
- mark-price valuation
- funding application
- strict funding-event coverage checks for research backtests
- long/short enablement
- a liquidation-aware futures account block on the pandas engine

If you intentionally want the old permissive smoke-test behavior, override these explicitly:

1. set `features.context_missing_policy.mode` to `zero_fill`
2. remove or relax `features.futures_context_ttl` and `features.cross_asset_context_ttl`
3. set `backtest.funding_missing_policy.mode` to `zero_fill`
4. set `data.duplicate_policy` to `warn` or `flag`
5. set `data.futures_context.recent_stats_availability_lag` to `none` only if you intentionally accept non-causal recent-stat alignment

If you want a trade-ready evaluation instead of a research-only one, add these explicitly:

1. set `backtest.evaluation_mode` to `trade_ready`
2. set `backtest.execution_profile` to `trade_ready_event_driven`
3. use a real Nautilus execution policy such as `{"adapter": "nautilus", ...}` and do not set `force_simulation`
4. provide `backtest.scenario_matrix` with named stress cases such as `downtime`, `stale_mark`, and `halt`
5. set `backtest.required_stress_scenarios` so the promotion gate knows which cases are mandatory

If you only want a surrogate execution study, keep `backtest.evaluation_mode = "research_only"` and set `execution_policy.force_simulation = true` explicitly. If you start from the hardened trade-ready example, you must also set `backtest.research_only_override = true` before it will downgrade to that labeled fallback path.

Minimal pattern:

```python
from core import ATR, MACD, RSI, ResearchPipeline
from example_utils import build_futures_research_config, clone_config_with_overrides

config = build_futures_research_config(
    symbol="BTCUSDT",
    interval="1h",
    start="2024-01-01",
    end="2024-04-01",
    indicators=[RSI(14), MACD(), ATR(14)],
    context_symbols=["ETHUSDT"],
)

config = clone_config_with_overrides(
    config,
    {
        "model": {
            "cv_method": "walk_forward",
            "n_splits": 4,
            "train_size": 360,
            "test_size": 96,
            "gap": 3,
        },
        "signals": {
            "policy_mode": "theory_only",
        },
    },
)

pipeline = ResearchPipeline(config)
```

## Build A Custom-Data Case

Do not inject ad hoc columns directly into market bars. Use the custom-data contract.

Pattern:

```python
from example_utils import build_custom_data_entry, build_spot_research_config

custom_entry = build_custom_data_entry(
    "macro_feed",
    frame=custom_frame,
    timestamp_column="timestamp",
    availability_column="available_at",
    value_columns=["macro_score", "macro_regime"],
    prefix="exo",
    max_feature_age="6h",
)

config = build_spot_research_config(
    symbol="BTCUSDT",
    interval="1h",
    start="2024-01-01",
    end="2024-03-01",
    indicators=indicators,
    context_symbols=["ETHUSDT"],
    custom_data=[custom_entry],
)
```

Requirements for custom data:

1. The data must be timestamped.
2. The data must include an explicit availability timestamp.
3. You should declare the feature columns you want to expose.
4. Missing-data policy should be intentional, not accidental.

Use `example_custom_data.py` as the working reference.

## Build An Offline Or Deterministic Case

If you want repeatable tests, do not rely on live downloads. Seed the pipeline state directly.

Pattern:

```python
from core import ResearchPipeline
from example_utils import seed_offline_pipeline_state

pipeline = ResearchPipeline(config)
seed_offline_pipeline_state(
    pipeline,
    raw_data=raw_data,
    futures_context=futures_context,
    cross_asset_context={"ETHUSDT": eth_data},
    symbol_filters={"tick_size": 0.1, "step_size": 0.001, "min_notional": 10.0},
)
```

This is the right path when you need:

1. deterministic regression tests
2. integration tests without network access
3. synthetic stress scenarios
4. narrow reproductions for bug reports

Use `example_synthetic_derivatives.py` as the working reference.

## Run Drift-Governed Retraining

Use `example_drift_retraining_cycle.py` when you want the operational path from drift signal to challenger decision.

The important runtime pieces are:

1. a registry store with an existing champion
2. reference features and current runtime features/predictions
3. a scheduled retraining window flag
4. a challenger-training callback that returns a fully formed candidate payload
5. a rollback policy for critical degradation

`ResearchPipeline.run_drift_retraining_cycle(...)` is the pipeline-facing wrapper around the tested orchestration function. It will reuse current pipeline state for `X`, OOS probabilities, equity-curve performance, and operational monitoring when those are already populated.

Consumer-hardware scheduling assumptions:

1. keep `scheduled_window_open` coarse, typically daily or weekly rather than every bar
2. set drift guardrails with a non-trivial `min_samples` and `cooldown_bars`
3. retrain one symbol at a time and keep the challenger search constrained
4. use `hybrid` rollback mode so only critical degradation auto-rolls back without human review

That keeps the drift loop resumable and comparable without turning every small wobble into a full retrain.

## Turn A Scenario Into A Test Case

There are three useful test styles in this repository.

### 1. Unit Test A Small Primitive

Use this when the behavior belongs to one function or one narrow module.

Good targets:

- labeling helpers
- governance rules
- drift checks
- data-contract validation
- execution cost models

Look at these tests for patterns:

- `tests/test_data_contracts.py`
- `tests/test_data_quality_quarantine.py`
- `tests/test_microstructure_cost_models.py`

### 2. Pipeline Smoke Test With Offline Data

Use this when you want to prove that a full configuration still runs end to end.

Pattern:

```python
def test_my_case_smoke():
    pipeline = ResearchPipeline(config)
    seed_offline_pipeline_state(pipeline, raw_data=raw_data)

    pipeline.run_indicators()
    pipeline.build_features()
    pipeline.build_labels()
    aligned = pipeline.align_data()

    assert not aligned["X"].empty
```

Look at these tests for realistic end-to-end patterns:

- `tests/test_derivatives_context_pipeline.py`
- `tests/test_drift_retraining_workflow.py`
- `tests/test_execution_adapter_parity.py`

### 3. Adversarial Regression Test

Use this when you are trying to prove that the pipeline rejects leakage, stale data, bad contracts, or unsafe promotions.

Good references:

- `tests/test_lookahead_provocation.py`
- `tests/test_cross_stage_embargo.py`
- `tests/test_feature_admission_policy.py`
- `tests/test_promotion_gate_binding.py`

The right question for these tests is not "does it run?" but "does it fail closed under the bad condition I care about?"

## Which Files Matter Most

If you only read five files to start building your own cases, read these:

1. `example_test_case_template.py`
2. `example_utils.py`
3. `example_active_spot.py` or `example_active_futures.py`
4. `example_custom_data.py` if you have exogenous inputs
5. `example_synthetic_derivatives.py` if you want deterministic tests

## Common Mistakes

1. Copying an older full config block instead of using a builder plus overrides. That creates drift between examples immediately.
2. Forgetting `availability_column` when joining custom data. That turns a point-in-time-safe design into leakage.
3. Using the vectorbt path and expecting full futures-account liquidation semantics. The explicit futures account model lives on the pandas adapter.
4. Treating zero-trade outputs as automatically broken. Some conservative examples are intentionally allowed to abstain.
5. Removing the `universe` snapshot while still asking for cross-asset context. That breaks the causal symbol-eligibility contract.
6. Running AutoML as if it were the default onboarding path. It is a good advanced demo, not the best first entrypoint.
7. Treating `example_automl.py` as promotion-safe. The hardened path is `example_trade_ready_automl.py`; the old script is intentionally a short smoke/demo workflow.

## Suggested First Runs

If you are new to the repo, run these in order:

```bash
python example_active_spot.py
python example_active_futures.py
python example_custom_data.py
python example_test_case_template.py
python -m pytest tests/test_derivatives_context_pipeline.py
```

That sequence shows:

1. a normal spot case
2. a normal futures case
3. a point-in-time-safe exogenous-data case
4. the recommended copy-and-edit template
5. what a deterministic regression-style test looks like