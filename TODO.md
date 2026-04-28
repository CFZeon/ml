# TODO

## Scope

This file translates `REMEDIATION.md` into the current repository state.
It is not a restatement of the remediation document. It is the execution plan
for the remaining gaps after reviewing the codebase on 2026-04-28.

## Current State

The repo already has substantial work in place:

1. Trade-ready data defaults, data certification, and fail-closed runtime guards already exist.
2. Locked holdout, CPCV-style validation, replication cohorts, DSR/PBO, and promotion gating already exist.
3. Paper calibration, live calibration, and staged release readiness already exist.
4. The remaining P0 gap is not the absence of controls. It is the absence of clean evidence separation and a discoverable strict local certification path.

## External Research Distilled Into Engineering Rules

1. Nested model selection must keep hyperparameter search inside the inner loop and untouched evaluation in the outer loop.
2. `TimeSeriesSplit.gap` is only a boundary exclusion. It is not a replacement for purging overlapping labels and embargo.
3. CPCV is useful because it generates a distribution of backtest paths, not one flattering path.
4. DSR and minimum track record logic must be bound into promotion when many trials are searched.
5. Event-driven local backtesting is feasible on consumer hardware with Nautilus `BacktestNode` or `BacktestEngine.reset()`.
6. Bar-based execution must respect close-time execution semantics, conservative fill assumptions, and explicit limitations.
7. Flat fees are not sufficient for capital-facing evidence. Slippage, latency, liquidity, queue position, and order-type behavior must be explicit.

## Remaining Workstreams

### P0

1. Separate search evidence from post-selection refit artifacts.
2. Add a strict local certification entrypoint that does not silently downgrade to the research demo.
3. Normalize the AutoML summary contract so selection, holdout, replication, and promotion fields are always present and typed enough for downstream tooling.
4. Update stale focused AutoML tests so they state when they require permissive research semantics instead of relying on hardened defaults implicitly.

### P1

5. Expand evidence tagging so every capital-facing backtest payload carries an explicit evidence class.
6. Tighten local certification execution defaults and make the execution-profile matrix explicit in code and docs.
7. Add a first-class paper shadow artifact schema rather than relying only on calibration summaries.

### P2

8. Make local certification and trade-ready example classification visible in docs and tests.
9. Add explicit micro-capital default caps and kill-switch policy checks where they are still implicit.

## Step-By-Step Plan

1. Add an AutoML contract module that normalizes and validates summary payloads before persistence and return.
2. Preserve backward-compatible legacy fields in the AutoML summary, but add explicit evidence sections:
   `selection_evidence`, `validation_replay_evidence`, `locked_holdout_evidence`, `replication_evidence`, `refit_artifact`.
3. Add explicit `evidence_class` labels to AutoML backtest summaries and refit artifacts.
4. Stop `AutoMLStep.run()` from mutating `pipeline.config` and clearing downstream state by default.
5. Add an explicit `ResearchPipeline.refit_selected_candidate(...)` flow that produces a labeled `post_selection_refit` artifact.
6. Rework `example_automl.py` so any refit stage is printed as research-only post-selection output, not untouched OOS evidence.
7. Add `build_local_certification_runtime_overrides(...)` and `build_local_certification_automl_overrides(...)` in `example_utils.py`.
8. Add `example_local_certification_automl.py` as the discoverable strict local path.
9. Make local certification fail closed when Nautilus is unavailable instead of silently downgrading to the research demo.
10. Update `README.md` and `HOW_TO_USE.md` so the three user-facing paths are obvious:
    research demo, local certification, operator-facing trade-ready path.
11. Add focused regression tests for evidence separation and the local certification profile.
12. Update the stale AutoML holdout objective tests so the permissive scenarios opt into permissive research semantics explicitly.
13. Run the focused AutoML and example-profile regression slice.
14. Fix any regressions introduced by the new summary contract or explicit refit path.

## Acceptance Slice For This Iteration

1. `run_automl()` no longer rewrites the pipeline into a post-selection backtest state.
2. The demo script cannot present a refit without labeling it as a research refit artifact.
3. A strict local certification example exists and is surfaced in the top-level docs.
4. The AutoML summary always contains normalized `selection_outcome`, `locked_holdout`, `replication`, and `promotion_eligibility_report` sections.
5. Focused AutoML and example-profile tests are green on the touched slice.