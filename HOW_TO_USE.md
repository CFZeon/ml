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
| Run AutoML | `example_automl.py` | Shows the searchable config surface and Optuna study settings |
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
3. `valuation_price`, `apply_funding`, and `futures_account` for realistic futures backtests

## Build A Futures Case

Use `build_futures_research_config(...)` instead of switching a spot config by hand. The futures builder already carries the important futures defaults:

- `data.market = "um_futures"`
- futures cross-asset context market
- mark-price valuation
- funding application
- long/short enablement
- a liquidation-aware futures account block on the pandas engine

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