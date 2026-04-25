# Trade-Ready Remediation Plan

This plan turns the repo from a research-capable stack into a trade-ready workflow one bounded item at a time.

Implementation rule:

1. Research one item in isolation.
2. Implement only that item.
3. Update docs and examples for that item.
4. Validate it.
5. Then move to the next item.

## Item 1 — Safe-By-Default AutoML Governance

- Status: Complete
- Goal: Make the easiest AutoML path use locked holdout, binding selection gates, DSR/PBO diagnostics, and binding post-selection inference.
- Why it matters: A user following the default example path should not be able to mistake a demo winner for a promotion-safe model.
- Scope:
  - add a hardened AutoML profile helper
  - add a trade-ready AutoML example
  - relabel the old AutoML script as a smoke/demo workflow
  - update README and HOW_TO_USE onboarding
- Success criteria:
  - the hardened example reports promotion readiness explicitly
  - the demo example is clearly labeled non-promotion-safe

## Item 2 — Fail-Closed Missing-Data And Staleness Semantics

- Status: Complete
- Goal: Stop converting stale context, missing funding, and unknown state into economically valid zeros.
- Why it matters: Unknown state is currently allowed to masquerade as flat state, which inflates robustness and regime awareness.
- Scope:
  - add explicit missing-data policies for futures funding and context features
  - preserve missing/stale companion flags instead of collapsing them into zeros
  - block or quarantine training/backtests when missing-state coverage breaches thresholds
- Success criteria:
  - missing funding can no longer silently improve futures PnL
  - stale context is surfaced as unknown state, not tradable signal state

## Item 3 — Tradeability Gating For Execution And Stress Realism

- Status: Complete
- Goal: Separate research-valid results from trade-valid results with binding execution and stress requirements.
- Why it matters: A strategy can currently look strong while never surviving queue uncertainty, venue outages, or deleveraging stress.
- Scope:
  - require explicit trade-ready stress scenarios for promotion
  - tighten execution-realism reporting in examples and docs
  - make research-only versus trade-ready evaluation modes explicit
- Success criteria:
  - a result cannot present as trade-ready without passing execution and stress gates
  - users can tell immediately whether a backtest is only research-grade

## Item 4 — Drift-Governed Promotion And Retraining Loop

- Status: Complete
- Goal: Wire drift-triggered retraining and champion/challenger evaluation into the main operational path.
- Why it matters: Monitoring artifacts alone do not create a live-safe system if degrading models are never re-evaluated automatically.
- Scope:
  - integrate drift-cycle orchestration into the pipeline/runtime path
  - expose champion/challenger and rollback flow in examples
  - document the consumer-hardware retraining runbook and scheduling assumptions
- Success criteria:
  - the repo has a documented and runnable path from drift signal to retrain decision to rollback/promotion outcome

## Item 5 — Conflict-Aware Market Data Integrity

- Status: Complete
- Goal: Fail closed on conflicting duplicate bars and restated history instead of silently keeping the first row.
- Why it matters: A structurally valid frame built from unresolved timestamp collisions can produce reproducible but incorrect features, labels, and backtests.
- Scope:
  - distinguish exact duplicate bars from conflicting duplicate bars during data preparation and merge
  - emit integrity metadata that records duplicate conflict counts and affected timestamps
  - fail builder and example paths closed when conflicting duplicates are detected in trade-oriented workflows
- Success criteria:
  - exact duplicate rows can be de-duplicated deterministically
  - conflicting duplicate rows raise or block instead of being silently collapsed

## Item 6 — Causal Availability Modeling For Futures Context

- Status: Complete
- Goal: Stop treating Binance futures recent-stat endpoints as if they are actionable at the timestamp they summarize.
- Why it matters: Endpoint timestamps are not the same thing as live availability; period aggregates can leak into the first bar where a retail user could not yet know them.
- Scope:
  - attach explicit availability timestamps to recent futures context endpoints
  - shift recent-stat alignment to decision-safe availability rather than raw economic interval timestamps
  - surface availability policy in docs and examples so users do not assume endpoint timestamps are causal
- Success criteria:
  - recent-stat features are aligned using availability-safe timestamps
  - tests cover the availability lag behavior for recent futures context inputs

## Item 7 — Binding Replication For Trade-Ready AutoML

- Status: Complete
- Goal: Require the hardened AutoML path to prove robustness on alternate cohorts instead of one validation slice plus one holdout.
- Why it matters: A strategy can survive one historical narrative and still fail immediately when the window or nearby symbol changes.
- Scope:
  - enable replication cohorts in the hardened AutoML profile by default
  - bind replication as a promotion gate in trade-ready examples and docs
  - expose the replication outcome clearly in example summaries and onboarding docs
- Success criteria:
  - the trade-ready AutoML profile runs with replication enabled by default
  - a candidate cannot present as trade-ready when replication coverage or pass rate fails

## Item 8 — Fail-Closed Trade-Ready Execution Backend

- Status: Complete
- Goal: Make trade-ready execution require a real event-driven backend instead of allowing the bar surrogate to masquerade as deployment-ready.
- Why it matters: Queue position, acknowledgement latency, and order-book path dependence are dominant live failure modes and are not modeled by the surrogate.
- Scope:
  - make the hardened trade-ready example require a real Nautilus backend instead of `force_simulation=True`
  - relabel surrogate execution paths as research-only in docs and examples
  - tighten messaging so users can see immediately whether they are using a promotable backend or a surrogate
- Success criteria:
  - the hardened trade-ready example fails closed when Nautilus is unavailable
  - surrogate execution remains available only through explicitly research-grade example paths

## Phase 2 — Trade-Ready Certification Hardening

## Item 9 — Global Pre-Training Lookahead Certification

- Status: Complete
- Goal: Make the standard trade-ready and AutoML paths run a blocking lookahead provocation audit before model training, even when no custom feature builder is present.
- Why it matters: Leakage should be caught on the normal path, not only when a user adds a custom builder. The easiest promotion-oriented workflow must audit the causal feature surface by default.
- Scope:
  - wire the baseline-vs-prefix replay harness into `TrainModelsStep`
  - auto-enable the guard for trade-ready and AutoML runs, while keeping explicit config overrides available
  - expose the guard result in training summaries and promotion gates
  - update the hardened AutoML example and onboarding docs to show the new blocking audit
- Success criteria:
  - trade-ready and AutoML runs emit a `lookahead_guard` report in training summaries
  - future-shifted feature builders are rejected automatically in blocking mode
  - the default blocking audit surface is limited to causal pre-training features so standard label maturation does not trigger false positives

## Item 10 — Trade-Ready Data Certification

- Status: Complete
- Goal: Turn data-quality, gap handling, and reference coverage from diagnostics into a single trade-ready certification contract.
- Why it matters: Promotion-safe modeling still depends on whether the underlying bars, context, and reference feeds were complete, conflict-free, and publication-safe.
- Scope:
  - bind gap handling, quarantine disposition, duplicate-conflict detection, and reference coverage into one certification report
  - require trade-ready examples to surface certification status prominently before training and backtesting
  - standardize fail-closed thresholds so research-grade leniency cannot masquerade as trade-ready data quality
- Success criteria:
  - trade-ready runs report one explicit data-certification verdict
  - incomplete or conflicting market/context/reference data block trade-ready evaluation by default

Implemented notes:

- `core/pipeline.py` now builds a unified `data_certification` report from market integrity, data-quality quarantine, context TTL, and configured reference validation, and blocks in blocking mode before training.
- `core/automl.py` now carries `data_certification` into promotion gates and summarized artifacts.
- `example_trade_ready_automl.py`, `README.md`, and `HOW_TO_USE.md` now surface the certification verdict explicitly.

## Item 11 — Stronger Trade-Ready Certification Profile

- Status: Complete
- Goal: Replace the current low-power hardened example profile with a certification profile that produces meaningful rejection power on consumer hardware.
- Why it matters: A profile that mostly checks whether the stack can finish is not strong enough to support a deployment decision.
- Scope:
  - raise minimum evidence requirements for validation trades, replication coverage, and post-selection diagnostics
  - separate smoke/demo budgets from certification budgets in examples and helper builders
  - expose the reduced-power path explicitly when a user chooses a local smoke run
- Success criteria:
  - the certification example declares when it is running with reduced statistical power
  - promotion-safe defaults require materially stronger evidence than the smoke path

Implemented notes:

- `example_utils.py` now exposes explicit `certification` and `smoke` trade-ready AutoML power profiles, with stronger certification defaults for validation trade counts, replication coverage, DSR track length, objective gates, and post-selection bootstrap depth.
- `example_trade_ready_automl.py` now defaults to the certification profile, accepts `--smoke` for an explicitly reduced-power run, and surfaces that profile selection before and after AutoML execution.
- `core/automl.py`, `README.md`, and `HOW_TO_USE.md` now carry the trade-ready profile metadata so reduced-power runs remain visibly distinct from certification-grade runs.

## Item 12 — Operational Trade-Ready Path

- Status: Complete
- Goal: Connect promotion-safe research outputs to an operator-facing runbook for scheduled retraining, drift response, and deployment readiness checks.
- Why it matters: A model that is statistically acceptable once is still not trade-ready unless the operational control loop is explicit and reproducible.
- Scope:
  - define the handoff from certified model artifact to monitored runtime configuration
  - surface mandatory operational checks for data freshness, drift state, backend availability, and rollback readiness
  - update examples/docs so the operator path is distinct from the research/demo path
- Success criteria:
  - the repo exposes a documented path from certified backtest to monitored runtime deployment decision

Implemented notes:

- `core/readiness.py` now provides `build_deployment_readiness_report(...)`, a blocking operator-facing deploy/hold verdict that binds promotion status, operational monitoring, drift state, backend availability, and rollback readiness.
- `ResearchPipeline.inspect_deployment_readiness(...)` now exposes that report on the pipeline surface and persists it under `pipeline.state['deployment_readiness']`.
- `example_drift_retraining_cycle.py`, `README.md`, and `HOW_TO_USE.md` now show the full handoff from champion promotion and rollback flow to the final operator deploy-or-hold decision.
  - missing operational prerequisites block promotion-ready messaging by default

## Phase 3 — Default-Critical Remediation

The earlier phases made the pipeline promotion-aware and operationally inspectable.
The remaining gaps in `ISSUES.md` are about default-path ambiguity: the repo can still produce research-valid evidence that a retail user could mistake for trade-ready evidence.

Implementation rule for this phase:

1. research one item in isolation
2. implement only that item
3. update `README.md`, `HOW_TO_USE.md`, and affected examples for that item
4. validate it
5. then move to the next item

## Item 13 — Fail-Closed Trade-Ready Backend Requirement

- Status: Complete
- Goal: Stop `example_trade_ready_automl.py` from silently downgrading a trade-ready certification run into a weaker research-only experiment.
- Why it matters: the current fallback disables locked holdout and overfitting controls exactly on the path users are most likely to run locally.
- Scope:
  - require a real Nautilus backend for the certification profile
  - make the local surrogate path a separate, explicitly research-only example path instead of an automatic downgrade
  - surface deterministic failure reasons in example output and tests
- Success criteria:
  - certification mode fails closed when Nautilus is unavailable
  - no trade-ready example disables locked holdout, post-selection, or replication controls as part of a local fallback

Implemented notes:

- `core/pipeline.py` now enforces that `backtest.evaluation_mode = "trade_ready"` must use a Nautilus execution adapter with `force_simulation = false` unless the caller explicitly opts into `execution_profile = "research_surrogate"` and `research_only_override = true`.
- `example_trade_ready_automl.py` now fails closed when Nautilus is unavailable instead of downgrading itself into a weaker research-only experiment.
- `example_automl.py`, `README.md`, and `HOW_TO_USE.md` now describe the demo workflow as the explicit research-only surrogate path.
- Regression coverage now passes on `tests/test_trade_ready_execution_fail_closed.py` and `tests/test_trade_ready_example_profiles.py`.

## Item 14 — Binding Trade-Ready Monitoring Policy

- Status: Complete
- Goal: Replace the current mostly advisory operational-monitoring defaults with a finite trade-ready policy profile.
- Why it matters: telemetry without blocking thresholds satisfies neither model-risk monitoring nor live-trading control expectations.
- Scope:
  - add a hardened trade-ready monitoring policy with finite thresholds for latency, queue backlog, slippage drift, data lag, L2 age, and signal decay
  - auto-apply that policy in trade-ready examples and pipeline trade-ready mode
  - ensure readiness and promotion artifacts surface which thresholds were active
- Success criteria:
  - trade-ready runs fail or hold when required monitoring metrics are missing or breached
  - readiness artifacts distinguish research monitoring from trade-ready monitoring explicitly

Implemented notes:

- `core/monitoring.py` now exposes `resolve_monitoring_policy(...)` plus a finite `trade_ready` profile for freshness, fallback usage, fill quality, slippage drift, inference backlog, and signal-decay thresholds.
- `core/pipeline.py` now auto-applies that `trade_ready` monitoring profile whenever `backtest.evaluation_mode = "trade_ready"`, even if the caller omits an explicit monitoring block.
- `example_trade_ready_automl.py`, `example_utils.py`, `README.md`, and `HOW_TO_USE.md` now surface the active monitoring profile and monitoring health explicitly in the hardened trade-ready path.
- Regression coverage now passes on the new focused monitoring-policy tests in `tests/test_operations_monitoring.py` and `tests/test_operational_trade_ready_path.py`.

## Item 15 — Shared Trade-Ready Data And Funding Defaults

- Status: Complete
- Goal: Make trade-ready builders and runtime helpers fail closed on missing funding coverage, market-data gaps, and quarantined bars without relying on per-example overrides.
- Why it matters: safety-critical defaults should live in shared configuration surfaces, not only in one hardened script.
- Scope:
  - move strict trade-ready funding and data-integrity defaults into reusable config helpers
  - ensure trade-ready mode resolves strict funding coverage unless the user explicitly opts into research-only behavior
  - make gap/quarantine behavior visible in examples and onboarding docs
- Success criteria:
  - trade-ready configs inherit strict funding/data defaults from shared builders
  - missing funding events and quarantined market bars block trade-ready workflows consistently across examples

Implemented notes:

- `core/pipeline.py` now resolves strict funding coverage from `backtest.evaluation_mode = "trade_ready"` and auto-applies fail-closed `data.gap_policy`, `data.duplicate_policy`, and `data_quality.block_on_quarantine` defaults unless the caller explicitly marks the run as research-only with `backtest.research_only_override = true`.
- `example_utils.py` now exposes `build_trade_ready_runtime_overrides(...)`, a reusable helper that centralizes the trade-ready data and futures-funding defaults instead of repeating them inline per example.
- `example_trade_ready_automl.py`, `README.md`, and `HOW_TO_USE.md` now use and document that shared helper, and the trade-ready example prints the active data-policy guardrails alongside the monitoring profile.
- Regression coverage now passes on `tests/test_funding_coverage_gate.py` and `tests/test_trade_ready_example_profiles.py`.

## Item 16 — Statistical Evidence Floor For Trade-Ready Runs

- Status: Complete
- Goal: Tighten the minimum statistical evidence required before a trade-ready result can present as credible on consumer hardware.
- Why it matters: a result that survives a surrogate simulator with only a handful of effective observations is not trade-ready evidence.
- Scope:
  - add a trade-ready significance profile with stricter minimum observations and explicit low-sample failure reasons
  - bind those checks into trade-ready examples and summaries
  - keep smoke/demo runs available, but visibly separate them from certification-grade evidence
- Success criteria:
  - trade-ready runs report when significance is unavailable or too underpowered to support promotion claims

Implemented notes:

- `core/pipeline.py` now auto-enables a minimum trade-ready significance profile in runtime backtest kwargs, including a default minimum observation floor unless `backtest.research_only_override = true` is set for an explicit research-grade exception.
- `core/backtest.py` now records `observation_count`, `min_observations`, and `underpowered` in `statistical_significance`, so short-sample runs explain why significance is unavailable instead of returning a bare disabled flag.
- `core/automl.py`, `example_utils.py`, and `example_trade_ready_automl.py` now bind explicit significance-availability and observation-count gates into the trade-ready AutoML profiles, with certification at 64 observations and `--smoke` at a visibly weaker 32-observation floor.
- `README.md`, `HOW_TO_USE.md`, and the AutoML summary output now surface underpowered-evidence reasons directly.
- Regression coverage now passes on `tests/test_data_backtest_adapter.py`, `tests/test_objective_gate_thresholds.py`, `tests/test_objective_gate_confidence_bounds.py`, and `tests/test_trade_ready_example_profiles.py`.
  - docs and examples separate smoke feedback from certification evidence without ambiguity
