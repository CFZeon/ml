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
| Run strict local certification AutoML | `example_local_certification_automl.py` | Keeps fail-closed data policies, locked holdout, replication, and an explicit `local_certification_surrogate` fallback when Nautilus is unavailable; suitable for paper or pre-capital evidence, not live-capital release |
| Run trade-ready AutoML research | `example_trade_ready_automl.py` | Uses the stronger certification profile with locked holdout, replication cohorts, blocking feature-surface lookahead certification, binding selection gates, and promotion-readiness reporting; add `--smoke` for a visibly reduced-power run that still requires Nautilus, and read `oos_evidence.class`, `execution_evidence.class`, plus `funding_coverage_status` before any Sharpe or net-profit number |
| Run drift-governed retraining flow | `example_drift_retraining_cycle.py` | Shows champion/challenger registration, a paper-validation loop, a kill-switch / drawdown gate, scheduled retraining, hybrid rollback, and the final operator deploy/hold decision |
| Run research-only AutoML with locked holdout separation | `example_automl.py` | Keeps the public surrogate path research-only, but now preserves a contiguous validation replay, a final locked holdout, SPA-based post-selection inference, and a minimum significance floor before any post-selection refit |
| Run the old unsafe AutoML smoke path | `example_automl.py --research-demo` | Restores the fast demo mode that disables locked holdout and selection gates |
| Explore FVG-specific features | `example_fvg.py` | Narrow feature example; useful as a feature smoke test |
| Explore wider indicator families | `example_trend_volume_spot.py` and `example_trend_breakout_futures.py` | Show how to widen the indicator stack without changing the rest of the pipeline |

The builder-based real-data examples now also accept `--local-certification` when the scenario is suitable for the strict local certification runtime. That shared switch is available on:

- `example.py`
- `example_futures.py`
- `example_custom_data.py`
- `example_active_spot.py`
- `example_active_futures.py`
- `example_trend_volume_spot.py`
- `example_trend_breakout_futures.py`

Those shared builders also keep execution costs finite by default: the public research examples use `backtest.slippage_rate = 0.0002` and `backtest.slippage_model = "sqrt_impact"` unless you explicitly override them.
- `example_fvg.py`
- `example_test_case_template.py`

That path keeps the strict local-certification gates and stays explicit about execution evidence. It uses Nautilus when available and otherwise resolves to `local_certification_surrogate` instead of silently downgrading to the research path. The surrogate path is still non-event-driven and remains ineligible for live-capital release.

## Recommended Workflow

For a new research case, do not start by copying one of the largest inline configs from older examples.

Start from the shared builders in `example_utils.py`:

- `build_spot_research_config(...)`
- `build_futures_research_config(...)`
- `build_research_demo_runtime_overrides(market="spot" | "um_futures")`
- `build_local_certification_runtime_overrides(market="spot" | "um_futures")`
- `build_trade_ready_runtime_overrides(market="spot" | "um_futures")`
- `build_local_certification_automl_overrides(...)`
- `parse_local_certification_args(description)`
- `prepare_example_runtime_config(...)`
- `clone_config_with_overrides(base_config, overrides)`
- `build_custom_data_entry(...)`
- `seed_offline_pipeline_state(...)`

That gives you one stable base config plus a short diff that contains only your experiment-specific changes.

The shared builders now include strict context and funding integrity guardrails by default. If cross-asset context goes stale or a futures funding event is missing, the pipeline fails closed with an explicit gate error instead of converting that unknown state into zeros.
They also set `data.duplicate_policy = "fail"`, so conflicting duplicate market bars raise immediately instead of being silently de-duplicated.
They also set `data.futures_context.recent_stats_availability_lag = "period_close"`, so recent Binance futures statistics are indexed at publication-safe timestamps instead of the interval they summarize.
Those context z-scores are also trailing and prefix-invariant at cutoffs, so normalized derivatives features stay causal when you replay a fold boundary locally.
They also default to `backtest.evaluation_mode = "research_only"`, so a normal builder-based example is explicitly research-grade unless you promote it to a trade-ready evaluation profile.
If you promote a config to local certification, use `build_local_certification_runtime_overrides(...)` as the short diff.
If you promote a config to `trade_ready`, use `build_trade_ready_runtime_overrides(...)` as the short diff. Those shared helpers set fail-closed gap handling, duplicate-bar blocking, quarantine blocking, and strict futures funding coverage in one place instead of re-stating those guards per example.
If you want an example script to expose the same path on its CLI, use `parse_local_certification_args(...)` plus `prepare_example_runtime_config(...)` and keep the runtime mode explicit in the printed summary.
Trade-ready and local-certification runtime resolution now also force statistical significance back on and apply a minimum observation floor. If you want a research-grade exception, switch the config to `backtest.evaluation_mode = "research_only"`; capital-facing modes now reject `backtest.research_only_override = true`.
Local certification now also uses an explicit `local_certification` monitoring profile instead of reusing research defaults, and both capital-facing profiles emit a blocking `monitoring_gate_report` when required telemetry is missing or fallback assumptions appear.
The hardened trade-ready profile now also builds a single data-certification verdict before training. That report binds market gaps, data-quality quarantine, context TTL breaches, and configured reference validation into one blocking trade-ready gate.
Trade-ready and AutoML runs now auto-enable a blocking baseline-vs-prefix lookahead replay over the causal feature surface before training. If you widen the audit manually, keep in mind that labels are meant to mature with future bars and are not part of the default blocking surface.
The hardened trade-ready AutoML override now defaults to a stronger certification budget: more validation-trade evidence, wider replication coverage, and heavier post-selection diagnostics than the smoke path.
The hardened trade-ready AutoML override also enables replication cohorts by default, so promotion-readiness is checked on alternate windows or sibling symbols instead of a single holdout narrative.
The hardened trade-ready AutoML profile now also separates statistical evidence floors by power profile: certification uses a 64-observation significance floor, while `--smoke` keeps a visibly lower 32-observation floor and surfaces underpowered-evidence reasons in the summary.
The printed AutoML summary now starts with `oos_evidence.class` and `oos_evidence.evidence_stack_complete`, so you can see whether the run exercised the full adversarial OOS stack before reading Sharpe, return, or trade counts.
The same summary now also prints `execution_evidence.class`, `execution_evidence.execution_mode`, and `promotion_execution_ready`. If that line says `research_surrogate`, the run is research-only execution evidence even if the strategy metrics are attractive.
The same summary now also prints `funding_coverage_status`, missing-event count, and whether the funding gate passed. On futures cases, treat `fallback` as research-only leniency and `strict` as the certification-capable path.
The same summary now also prints the monitoring envelope verdict, missing metrics, and blocking reasons. In capital-facing modes, treat any missing telemetry there as a release blocker rather than a cosmetic warning.
If you intentionally choose `python example_trade_ready_automl.py --smoke`, the script prints `reduced power: True` before running and keeps that label in the resulting AutoML summary together with `capital_evidence_contract.requested_mode`, `effective_mode`, and `capital_path_eligible`.
That hardened trade-ready config still fails closed without a real Nautilus backend on the default certification path, and `--smoke` now fails closed the same way. Use `example_automl.py` when you want the safer research-only surrogate path with locked holdout separation plus SPA-based post-selection inference, use `example_automl.py --research-demo` when you intentionally want the old unsafe smoke mode, or use `build_research_demo_runtime_overrides(...)` when you are wiring your own surrogate config.
For AutoML certification-capable modes, CPCV or purged temporal search, search-stage embargo, validation-to-holdout gap, locked holdout, post-selection inference, and replication are now treated as one OOS evidence stack. If any of those controls are disabled, the run aborts before optimization instead of soft-disclaiming the result later.
It also auto-applies the `trade_ready` monitoring profile, so the certification path no longer inherits the permissive research monitoring defaults.

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

If you do set `backtest.funding_missing_policy.mode` to `zero_fill`, expect the resulting backtest summary to show `funding_coverage_status = "fallback"` so the missing-event leniency stays visible.

If you want a capital-facing evaluation instead of a research-only one, add these explicitly:

1. set `backtest.evaluation_mode` to `local_certification` for paper or pre-capital certification, or `trade_ready` for the stricter operator-facing path
2. use a real Nautilus execution policy such as `{"adapter": "nautilus", ...}` and do not set `force_simulation` for `trade_ready`; for `local_certification`, a real Nautilus backend is preferred, but the explicit surrogate local-certification path is still admissible for paper or pre-capital gating and will remain non-live-capital evidence
3. enable `reference_data` and configure blocking coverage/divergence rules for the venues you trust
4. enable `data_certification` if your config does not already do so and keep it in blocking mode for trade-ready runs
5. provide `backtest.scenario_matrix` with named stress cases such as `downtime`, `stale_mark`, and `halt`
6. set `backtest.required_stress_scenarios` so the promotion gate knows which cases are mandatory
7. set `monitoring.policy_profile` to `trade_ready` if you want the config to declare the binding monitoring profile explicitly; the pipeline will now auto-apply that profile for `local_certification` and `trade_ready` runs when you omit it
8. leave `backtest.research_only_override` unset for capital-facing runs; if you intentionally want research-grade leniency, switch back to `backtest.evaluation_mode = "research_only"`
9. set `backtest.significance.min_observations` only if you intentionally want a different evidence floor; otherwise the runtime will apply the trade-ready default and the AutoML summary will report underpowered significance explicitly

If you only want a surrogate execution study, keep `backtest.evaluation_mode = "research_only"` and set `execution_policy.force_simulation = true` explicitly. The default `example_trade_ready_automl.py` run still fails closed without Nautilus, and `python example_trade_ready_automl.py --smoke` no longer downgrades around that requirement; otherwise use `example_automl.py` for the safer locked-holdout surrogate path, `example_automl.py --research-demo` for the old unsafe smoke mode, or your own explicitly research-only config.
Capital-release evaluation now requires `execution_evidence.class == "event_driven_certification"`. Surrogate execution remains available as either research-only evidence or explicit `local_certification_surrogate` evidence, but neither path is live-capital eligible.

If you want the shortest strict path before the operator-facing workflow, use `python example_local_certification_automl.py`. That entrypoint does not silently downgrade when Nautilus is unavailable; it keeps the same fail-closed certification gates but switches to the explicit `local_certification_surrogate` runtime so the resulting evidence remains clearly labeled as non-event-driven and pre-capital only.
If you want the same local-certification runtime on a non-AutoML real-data demo, run that example with `--local-certification` instead of creating a separate script first.

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
The example now builds a paper-calibration report first with `ResearchPipeline.inspect_paper_trading_calibration(...)`, attaches that report to the champion artifact, computes an operational-limits report with `ResearchPipeline.inspect_operational_limits(...)`, and then ends with `ResearchPipeline.inspect_deployment_readiness(...)`, which turns the promoted champion, champion age, current monitoring state, drift outcome, backend status, rollback availability, attached paper evidence, and kill-switch status into one explicit deploy-or-hold verdict.

The important runtime pieces are:

1. a registry store with an existing champion
2. reference features and current runtime features/predictions
3. a scheduled retraining window flag
4. a challenger-training callback that returns a fully formed candidate payload
5. paper or shadow-live observations that can be aggregated into a paper report
6. an operational-limits policy with a live drawdown threshold and armed kill switch
7. a model-freshness policy or the default 28-day champion TTL
8. a rollback policy for critical degradation

`ResearchPipeline.run_drift_retraining_cycle(...)` is the pipeline-facing wrapper around the tested orchestration function. It will reuse current pipeline state for `X`, OOS probabilities, equity-curve performance, and operational monitoring when those are already populated.
`ResearchPipeline.inspect_paper_trading_calibration(...)` is the paper-validation wrapper. It aggregates paper or shadow-live observations into the same calibration contract used by the readiness gates and stores the resulting `paper_calibration` report in pipeline state.
`ResearchPipeline.inspect_operational_limits(...)` is the kill-switch wrapper. It normalizes kill-switch readiness plus current equity drawdown into an `operational_limits` report, defaults the drawdown ceiling to 10%, and marks the report as breached when the running strategy falls through that limit.
`ResearchPipeline.inspect_deployment_readiness(...)` is the follow-up operator gate. It reuses current pipeline monitoring and drift state, then checks seven blocking surfaces before a deploy decision is considered ready:

1. the target version is an approved champion
2. the current champion is still inside the configured model TTL
3. operational monitoring is healthy enough for deployment
4. drift is not actively demanding a retrain, or the latest drift-triggered retrain already produced a promoted challenger
5. the required execution backend is available
6. an attached paper-calibration report clears the paper stage
7. at least one rollback candidate is archived and ready

For `micro_capital` and `scaled_capital`, the operational-limits report must also stay green: the kill switch must be armed and the current drawdown must remain above the configured loss threshold. By default, deployment readiness also expires the promoted champion after 28 days and surfaces `model_expired` until a fresh version is promoted.

Consumer-hardware scheduling assumptions:

1. keep `scheduled_window_open` coarse, typically daily or weekly rather than every bar
2. set drift guardrails with a non-trivial `min_samples`, `cooldown_bars`, and a finite `max_bars_between_retrain` cadence; the default is 672 bars
3. retrain one symbol at a time and keep the challenger search constrained
4. use `hybrid` rollback mode so only critical degradation auto-rolls back without human review

That keeps the drift loop resumable and comparable without turning every small wobble into a full retrain.

The intended operator handoff is now:

1. certify the strategy candidate with `example_trade_ready_automl.py`
2. register and promote the approved champion
3. aggregate paper or shadow-live observations through `inspect_paper_trading_calibration(...)` or `example_drift_retraining_cycle.py`
4. attach the resulting paper report to the promoted champion artifact
5. arm the kill switch and evaluate live drawdown through `inspect_operational_limits(...)`
6. run scheduled or drift-triggered retraining through `example_drift_retraining_cycle.py`, keeping `max_bars_between_retrain` finite
7. call `inspect_deployment_readiness(...)` before any live-facing deploy decision and treat `model_expired` as a hard hold until a fresh champion is promoted

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
7. Treating `example_automl.py` as promotion-safe. The default script now keeps a locked holdout, a lightweight selection freeze, and SPA-based post-selection inference, but execution is still surrogate and the result is still research-only. The hardened path is `example_trade_ready_automl.py`; use `example_automl.py --research-demo` only when you intentionally want the old unsafe smoke workflow.

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