# Remediation Plan To Raise The Minimum Bar For Risking Capital

## Purpose

This document replaces the earlier remediation notes with a plan tied to the current audit findings.

The goal is not to prove the strategy is profitable. The goal is to raise the minimum engineering, statistical, and operational bar high enough that:

1. research-only evidence cannot be mistaken for deployment-safe evidence
2. leakage and experiment contamination are harder to introduce than to avoid
3. the system abstains cleanly when evidence is weak instead of forcing a winner
4. a retail operator on consumer hardware can reach a bounded pre-capital certification workflow

Until the items below are implemented and validated, the system should still be treated as research-only.

## Minimum Bar Definition

The minimum bar for risking money in this repository is reached only when all of the following are true:

1. every AutoML run has immutable experiment lineage and cannot silently mix trials across incompatible datasets, code revisions, or search spaces
2. temporal gaps, embargoes, and holdout boundaries are owned by a single split mechanism and are fully auditable row-by-row
3. selection, freeze, locked holdout, and post-selection inference are deterministic, binding, and green in focused regression tests
4. the model-selection path and the tradable replay path are explicitly separated and both must pass their designated gates
5. trade-ready mode requires an event-driven backend and a paper or shadow-live calibration stage before any capital is released
6. significance, generalization, and replication thresholds are high enough that a short lucky sample cannot pass certification
7. missing data, stale context, funding gaps, and historical universe uncertainty fail closed in trade-ready workflows
8. live deployment readiness is decided by a capital-release ladder, not a single backtest summary
## External Research Used

This plan incorporates isolated research from:

1. Optuna `create_study` behavior: existing studies are returned when `load_if_exists=True`, which is useful for resumability but unsafe as a default for clean experiments
2. scikit-learn `TimeSeriesSplit`: the `gap` parameter is the single exclusion mechanism between train and test and should not be double-applied by separate masking logic
3. QuantConnect fill-model documentation: realistic fills require explicit modeling of partial fills, stale fills, and volume-constrained execution; built-in backtests still simplify live brokerage behavior
4. internal audit criteria already captured in [ISSUES.md](ISSUES.md)

The remediation below translates those principles into repo-specific engineering work.

## Guiding Rules

1. Fail closed in trade-ready mode.
2. Fresh experiments are the default. Resume is explicit and validated.
3. One component owns each temporal boundary.
4. If no candidate is good enough, the correct outcome is abstain.
5. Statistical corrections do not rescue a misspecified simulator.
6. Backtest, paper, and live are three different evidence classes and must stay separate.

## Workstream Map

| ID | Workstream | Primary Problem | Priority |
|---|---|---|---|
| WS-01 | Experiment lineage and study isolation | stale or incompatible Optuna trials contaminate selection | P0 |
| WS-02 | Temporal split ownership | staged validation likely double-applies gap/embargo | P0 |
| WS-03 | Selection and holdout governance reliability | strict policies can still crash instead of abstaining cleanly | P0 |
| WS-04 | Selection architecture and metric-source clarity | CPCV is diagnostic while replay drives tradable scoring | P1 |
| WS-05 | Search-space decomposition and replication | one study is searching economic specifications, not just parameters | P1 |
| WS-06 | Data, feature, and missingness causality | stale context, funding gaps, and lookahead protection are still uneven across paths | P1 |
| WS-07 | Execution realism and pre-capital calibration | surrogate backtests are too easy to over-trust | P0 |
| WS-08 | Statistical evidence floor and sizing discipline | current thresholds are too weak for capital decisions | P1 |
| WS-09 | Universe, portability, and survivorship controls | single-asset success can still be cherry-picked | P2 |
| WS-10 | Capital-release ladder and operations | no single run should authorize real money | P0 |

## Repository State Snapshot (2026-04-27)

This document is a remediation plan, not a statement that the repo is missing every control from zero. The current codebase already contains several important partial controls that must now be finished, bound everywhere, and made non-bypassable.

| Workstream | Status | Current Evidence | Remaining Blocker |
|---|---|---|---|
| WS-01 | Completed | [core/automl.py](core/automl.py) now binds manifest-derived experiment identity, resume policy, lineage validation, and study storage context, with focused coverage in [tests/test_automl_experiment_manifest.py](tests/test_automl_experiment_manifest.py) and [tests/test_automl_resume_guard.py](tests/test_automl_resume_guard.py). | Residual operational caveat: Windows SQLite cleanup still needs tolerant test harnesses. |
| WS-02 | Completed | Explicit audited split objects now own gap accounting across [core/automl.py](core/automl.py) and [core/pipeline.py](core/pipeline.py), with coverage in [tests/test_temporal_split_row_accounting.py](tests/test_temporal_split_row_accounting.py), [tests/test_explicit_split_bypass_gap.py](tests/test_explicit_split_bypass_gap.py), and [tests/test_cross_stage_embargo.py](tests/test_cross_stage_embargo.py). | Residual risk: additional split families should reuse the same explicit-split path rather than reintroducing implicit gap logic. |
| WS-03 | Completed | Selection now abstains cleanly when no eligible candidate survives policy, and promotion reporting remains bound in [core/automl.py](core/automl.py) with coverage in [tests/test_automl_selection_abstention.py](tests/test_automl_selection_abstention.py). | Residual risk: objective-policy changes should keep the abstention path explicit rather than raising late runtime failures. |
| WS-04 | Completed | Validation contracts and metric-source reporting are explicit in [core/automl.py](core/automl.py), [core/pipeline.py](core/pipeline.py), and [example_utils.py](example_utils.py), with coverage in [tests/test_validation_contract_resolution.py](tests/test_validation_contract_resolution.py), [tests/test_metric_source_reporting.py](tests/test_metric_source_reporting.py), and [tests/test_cpcv_replay_dual_gate.py](tests/test_cpcv_replay_dual_gate.py). | Residual risk: future summary shapes should keep validation-source fields first-class. |
| WS-05 | Completed | Certification profiles now freeze thesis-space search and enforce family caps before wasted trials in [core/automl.py](core/automl.py) and [example_utils.py](example_utils.py), with coverage in [tests/test_trade_ready_profiles_do_not_search_thesis_space.py](tests/test_trade_ready_profiles_do_not_search_thesis_space.py) and [tests/test_max_trials_per_family_precheck.py](tests/test_max_trials_per_family_precheck.py). | Residual risk: new search-space families must be classified into thesis/model-family/hyperparameter tiers before activation. |
| WS-06 | Completed | Lookahead guard defaults and feature-availability metadata are now universal across model-training paths in [core/pipeline.py](core/pipeline.py), [core/features.py](core/features.py), and [core/feature_governance.py](core/feature_governance.py), with coverage in [tests/test_global_lookahead_guard_default.py](tests/test_global_lookahead_guard_default.py) and [tests/test_feature_availability_contract.py](tests/test_feature_availability_contract.py). | Residual risk: future feature builders must continue emitting availability metadata rather than silently degrading to timeless features. |
| WS-07 | Completed | Paper or shadow-live calibration artifacts, readiness gating, and deployment evidence persistence now exist in [core/readiness.py](core/readiness.py) and [core/registry/store.py](core/registry/store.py), with coverage in [tests/test_trade_ready_requires_paper_report.py](tests/test_trade_ready_requires_paper_report.py) and [tests/test_live_calibration_report.py](tests/test_live_calibration_report.py). | Residual risk: live calibration thresholds remain policy-tunable and should be tightened with production evidence. |
| WS-08 | Completed | Statistical significance now tracks effective bet counts and trade-ready Kelly sizing auto-caps until calibration evidence is green in [core/backtest.py](core/backtest.py), [core/automl.py](core/automl.py), [core/pipeline.py](core/pipeline.py), and [example_utils.py](example_utils.py), with coverage in [tests/test_effective_bet_count_significance.py](tests/test_effective_bet_count_significance.py), [tests/test_underpowered_is_blocking.py](tests/test_underpowered_is_blocking.py), and [tests/test_kelly_disabled_when_uncalibrated.py](tests/test_kelly_disabled_when_uncalibrated.py). | Residual risk: certification thresholds still need periodic recalibration by strategy cadence. |
| WS-09 | Completed | Certification portability is now an explicit contract layered over replication cohorts in [core/automl.py](core/automl.py) and declared by trade-ready profiles in [example_utils.py](example_utils.py), with coverage in [tests/test_portability_contract.py](tests/test_portability_contract.py), [tests/test_demo_universe_not_allowed_in_certification.py](tests/test_demo_universe_not_allowed_in_certification.py), and [tests/test_replication_validator.py](tests/test_replication_validator.py). | Residual risk: venue-level replay cohorts are still future work when broader reference datasets become available. |
| WS-10 | Completed | Deployment readiness now reports explicit capital-release stages, release blockers, and manual-ack-gated advancement in [core/readiness.py](core/readiness.py) and [core/pipeline.py](core/pipeline.py), with coverage in [tests/test_release_stage_transitions.py](tests/test_release_stage_transitions.py), [tests/test_release_blockers_surface.py](tests/test_release_blockers_surface.py), and [tests/test_micro_capital_requires_manual_ack.py](tests/test_micro_capital_requires_manual_ack.py). | Residual risk: live operator runbooks and human approvals remain process controls outside the code path. |

Read the remaining work below as finish-or-bind work where status is `Partial`, not as a claim that the repository has none of the relevant machinery.

## WS-01: Experiment Lineage And Study Isolation

### Problem

The current study identity is too weak. `_resolve_study_name(...)` in [core/automl.py](core/automl.py#L133) names studies primarily by symbol, interval, objective, and optional schema version. `_build_study_storage_path(...)` in [core/automl.py](core/automl.py#L512) stores them under a predictable SQLite path, and `run_automl_study(...)` in [core/automl.py](core/automl.py#L3198) calls Optuna with `load_if_exists=True`.

That means a later run can silently reuse completed trials generated from:

1. a different dataset window
2. different exchange filters or funding history
3. a changed search space
4. changed code paths in pipeline or backtest logic

This is unacceptable for capital decisions.

### Target State

Every experiment has a unique identity derived from immutable lineage, and resuming a study is explicit and validated.

### Required Code Changes

1. Replace implicit study identity with an `experiment_manifest`.
2. Compute and persist the following hashes before `optuna.create_study(...)`:
   1. `data_lineage_hash`: symbol universe, interval, start/end timestamps, raw-data source metadata, funding source metadata, reference feeds, universe snapshot IDs
   2. `feature_schema_hash`: feature builder config, schema version, transform chain config, custom data schema, lookahead guard config
   3. `objective_hash`: objective name, objective gates, significance settings, promotion policy profile
   4. `code_revision`: git SHA if available, else explicit version string
   5. `search_space_hash`: exact AutoML search space after defaults are resolved
3. Introduce a single `experiment_id = stable_hash(manifest)`.
4. Change storage layout from a shared flat DB path to:

```text
.cache/automl/
  experiments/
    <experiment_id>/
      study.db
      manifest.json
      lineage.json
      summary.json
```

5. Change `automl.resume_mode` to one of:
   1. `never` (default)
   2. `if_manifest_matches`
   3. `force`
6. When `resume_mode == "if_manifest_matches"`, load the manifest from disk and compare field-by-field before study reuse.
7. When any lineage field differs, refuse to resume and emit a structured remediation message containing:
   1. stored experiment id
   2. current experiment id
   3. mismatched fields
   4. exact storage path to archive or delete
8. Persist manifest fields to both Optuna study-level metadata and each trial `user_attrs` so that copied or exported studies remain auditable.

### Recommended Implementation Shape

1. Add `_build_experiment_manifest(base_config, automl_config, state_bundle)` in [core/automl.py](core/automl.py).
2. Add `_experiment_storage_dir(manifest)` in [core/automl.py](core/automl.py).
3. Add `_validate_resume_manifest(storage_dir, manifest, resume_mode)`.
4. Change `_resolve_study_name(...)` to produce a human-readable label only, not the true identity.
5. Add `summary["experiment_manifest"]`, `summary["experiment_id"]`, and `summary["resume_mode"]` to the AutoML output.

### Tests

1. `tests/test_automl_experiment_manifest.py`
   1. manifest hash changes when search space changes
   2. manifest hash changes when data window changes
   3. manifest hash changes when schema version changes
2. `tests/test_automl_resume_guard.py`
   1. `resume_mode=never` creates a fresh study
   2. `resume_mode=if_manifest_matches` resumes only when manifests match exactly
   3. mismatched manifests raise a deterministic error with mismatch details
3. `tests/test_automl_trial_user_attrs.py`
   1. every completed trial carries experiment lineage attrs

### Acceptance Criteria

1. A fresh run never reuses old trials unless the user explicitly asked for resume.
2. A resumed run is impossible if lineage differs.
3. Every study is reconstructable from manifest plus artifacts alone.

## WS-02: Temporal Split Ownership

### Problem

The staged temporal path in `_execute_temporal_split_candidate(...)` in [core/automl.py](core/automl.py#L717) pre-masks aligned rows around the stage boundary, then the downstream validation loop still calls
`walk_forward_split(...)` with `gap=model_config.get("gap", 0)` in [core/pipeline.py](core/pipeline.py#L409).

That creates a credible double-gap risk.

Even if the current logic sometimes lands on an acceptable slice, the ownership model is wrong: one layer edits the dataset, another layer applies the gap again.

### Target State

Temporal boundaries are defined once and consumed exactly once.

### Required Code Changes

1. Stop representing staged validation by mutating `X`, `y`, and `labels_aligned` plus inferred train/test row counts.
2. Replace that mechanism with explicit split objects.
3. Introduce a canonical split schema:

```python
{
    "split_id": "validation_stage_0",
    "train_index": np.ndarray,
    "test_index": np.ndarray,
    "gap_index": np.ndarray,
    "gap_bars": int,
    "source": "staged_holdout_plan",
    "timestamp_bounds": {
        "train_end": ...,
        "test_start": ...,
    },
}
```

4. Add `model.explicit_splits` support in `_iter_validation_splits(...)` in [core/pipeline.py](core/pipeline.py).
5. When `explicit_splits` is present, bypass `walk_forward_split(...)` and `cpcv_split(...)` entirely for that stage.
6. Move all gap calculation into one function, for example `_resolve_temporal_boundaries(...)`, and require it to return:
   1. search rows
   2. search-validation gap rows
   3. validation rows
   4. validation-holdout gap rows
   5. holdout rows
   6. total-accounted rows
7. Add a hard assertion that accounted rows equal raw rows unless leading warm-up bars are explicitly declared in the manifest.
8. Add a `gap_audit` payload to training and backtest summaries.
### Recommended Implementation Shape

1. In [core/automl.py](core/automl.py), `_resolve_holdout_plan(...)` should generate explicit split definitions, not only timestamps.
2. `_execute_temporal_split_candidate(...)` should attach those explicit splits to candidate config instead of pre-masking aligned state.
3. `_iter_validation_splits(...)` in [core/pipeline.py](core/pipeline.py) should prefer `explicit_splits` when provided.
4. Remove the need to set `n_splits=1`, `train_size`, and `test_size` for staged evaluation.

### Tests

1. `tests/test_temporal_split_row_accounting.py`
   1. search + gaps + validation + holdout add up exactly
2. `tests/test_explicit_split_bypass_gap.py`
   1. explicit splits do not apply an extra `gap`
3. `tests/test_gap_audit_payload.py`
   1. every staged run reports the exact gap ownership and counts
4. extend [tests/test_automl_holdout_objective.py](tests/test_automl_holdout_objective.py)
   1. assert exact row counts on search, validation, gap, and holdout slices

### Acceptance Criteria

1. No staged run depends on implicit row counts to recover split geometry.
2. Gap ownership is unique and auditable.
3. The focused holdout tests stop failing because of boundary ambiguity.

## WS-03: Selection And Holdout Governance Reliability

### Problem

The repo already contains substantial governance machinery: selection-policy summaries, freeze metadata, locked-holdout accounting, promotion gates, and focused regression coverage in [tests/test_automl_holdout_objective.py](tests/test_automl_holdout_objective.py). The remaining failure shape is still wrong: when no eligible candidate survives the configured policy, `run_automl_study(...)` in [core/automl.py](core/automl.py#L3330) raises `RuntimeError("AutoML found no eligible trial under the configured selection policy")`.

For a capital-facing system, this is the wrong failure shape. The selection layer should:

1. explain why every candidate failed
2. return an abstention outcome when evidence is insufficient
3. freeze the winner exactly once when one exists
4. never crash because the policy did its job

### Target State

Selection produces one of three deterministic outcomes:

1. `selected`
2. `abstain_no_eligible_trial`
3. `error_invalid_configuration`

### Required Code Changes

1. Replace the current runtime error path with a structured abstention summary.
2. Introduce a `SelectionOutcome` payload:

```python
{
    "status": "abstain_no_eligible_trial",
    "eligible_trial_count": 0,
    "completed_trial_count": 12,
    "rejected_trial_count": 12,
    "top_rejection_reasons": [...],
    "selection_freeze": None,
    "promotion_ready": False,
}
```

3. Preserve full per-trial rejection diagnostics in the final summary even when nothing is selected.
4. Split configuration errors from evidence failures.
5. Make `selection_freeze` an explicit object that is created only after:
   1. candidate ranking on allowed metrics
   2. freeze hash generation
   3. single locked-holdout access
6. Add an access counter around locked-holdout evaluation and assert it remains exactly one for selected candidates.
7. Ensure `candidate_hash`, `selection_freeze`, `holdout_consulted_for_selection`, and `evaluated_after_freeze` are always present in the summary.
8. Add deterministic ordering for rejection reasons so tests remain stable.

### Recommended Implementation Shape

1. Refactor `_build_trial_selection_report(...)` in [core/automl.py](core/automl.py) to return a normalized `eligibility_report` with:
   1. `blocking_failures`
   2. `advisory_failures`
   3. measured values
   4. policy thresholds
2. Add `_finalize_selection_outcome(...)` after trial aggregation.
3. Make `run_automl_study(...)` return abstention summaries instead of throwing when no eligible trial exists.

### Tests

1. extend [tests/test_automl_holdout_objective.py](tests/test_automl_holdout_objective.py)
   1. existing tests must pass green under strict defaults
   2. add one case for `abstain_no_eligible_trial`
2. `tests/test_automl_selection_outcomes.py`
   1. one valid candidate produces `selected`
   2. all rejected candidates produce `abstain_no_eligible_trial`
   3. bad config produces `error_invalid_configuration`
3.
`tests/test_locked_holdout_single_access.py`
   1. locked holdout is evaluated once, after freeze, never during selection

### Acceptance Criteria

1. No focused governance test fails.
2. No “no eligible trial” case crashes the study.
3. The summary always explains the selection outcome in machine-readable form.

## WS-04: Selection Architecture And Metric-Source Clarity

### Problem

The repository has two valuable but different validation views:

1. CPCV path diagnostics
2. executable walk-forward replay

The current pipeline makes CPCV partly diagnostic while replayed walk-forward paths can become the tradable signal source in [core/pipeline.py](core/pipeline.py#L4406), with CPCV significance explicitly described as diagnostic-only in [core/pipeline.py](core/pipeline.py#L1521).

That is not inherently wrong, but the selection architecture must state exactly which path is used for ranking, which is used for vetoes, and which is used for the final tradable summary.

### Target State

The repository uses a declared multi-stage validation contract.

### Required Code Changes

1. Introduce `automl.validation_contract` with explicit stages:
   1. `search_ranker`
   2. `contiguous_validation`
   3. `locked_holdout`
   4. optional `replication`
2. For trade-ready profiles, use the following default contract:
   1. `search_ranker = cpcv`
   2. `contiguous_validation = walk_forward_replay`
   3. `locked_holdout = single_access_contiguous`
   4. `replication = required`
3. Keep CPCV for ranking robustness and PBO-style diagnostics.
4. Keep contiguous replay for operational realism.
5. Require both the CPCV ranker and contiguous replay to pass their own gates.
6. Add explicit fields to every training and selection summary:
   1. `selection_metric_source`
   2. `diagnostic_metric_source`
   3. `tradable_metric_source`
   4. `all_required_sources_passed`
7. Add warning or blocking behavior when a summary mixes sources without naming them.

### Recommended Implementation Shape

1. Add `_resolve_validation_contract(...)` in [core/automl.py](core/automl.py).
2. Extend `_resolve_primary_training_payload(...)` to return named metric-source metadata.
3. Add a `validation_sources` block to selection reports and backtest summaries.
4. Update examples so they print the selection source and tradable source separately.

### Tests

1. `tests/test_validation_contract_resolution.py`
2. `tests/test_metric_source_reporting.py`
3. `tests/test_cpcv_replay_dual_gate.py`
   1. CPCV pass + replay fail must block trade-ready promotion
   2. replay pass + CPCV fail must block trade-ready promotion

### Acceptance Criteria

1. Every report makes source-of-truth explicit.
2. No trade-ready decision is based on one source while quietly displaying another.
## WS-05: Search-Space Decomposition And Replication

### Problem

The default AutoML search space in [core/automl.py](core/automl.py#L35) searches over economic specification choices as well as model hyperparameters, including labels, lag structure, model family, and other structural degrees of freedom.

DSR, PBO, SPA, and White Reality Check help, but they do not turn one broad structural search on one historical narrative into strong causal evidence.

### Target State

Economic thesis selection, model-family selection, and parameter tuning are separated into different experiment layers.

### Required Code Changes

1. Split the current search space into three tiers:
   1. `thesis_space`: label family, label horizon, regime mode, feature family, execution assumptions
   2. `model_family_space`: logistic, tree ensemble, gradient boosting, calibrated wrapper
   3. `hyperparameter_space`: only model-specific parameters and thresholds
2. Prohibit `thesis_space` search in a certification run.
3. Require each certification experiment to freeze:
   1. label family
   2. barrier logic
   3. feature family
   4. execution profile
4. Add `experiment_family_id` so related challenger runs can be grouped without mixing them into one study.
5. Bind replication by default in trade-ready profiles.
6. Define replication cohorts as explicit contiguous windows or sibling assets, not random subsamples.
7. Add `max_trials_per_family` enforcement earlier in the study lifecycle, not only at summary time.

### Recommended Certification Defaults

For consumer hardware, define two visible power profiles:

1. `smoke`
   1. allowed only for workflow debugging
   2. cannot produce `capital_release_eligible=True`
2. `certification`
   1. fixed thesis
   2. bounded model family list
   3. replication required
   4. full significance and holdout gates enabled

### Tests

1. `tests/test_trade_ready_profiles_do_not_search_thesis_space.py`
2. `tests/test_replication_is_binding.py`
3. `tests/test_max_trials_per_family_precheck.py`

### Acceptance Criteria

1. Certification runs cannot vary economic thesis parameters.
2. Replication failure blocks promotion by default.
3. Smoke runs are visibly labeled and cannot authorize capital release.

## WS-06: Data, Feature, And Missingness Causality

### Problem

The repo already contains part of this layer. In trade-ready mode, [core/pipeline.py](core/pipeline.py) now auto-enables blocking lookahead audits for AutoML and trade-ready paths, resolves strict funding-missing handling, upgrades gap, duplicate, and quarantine defaults, and emits a unified `data_certification` report that can block promotion.

The remaining data-causality and missingness risks are:

1. lookahead guard is not universal on all research paths
2. context missingness can collapse to economically neutral zeros
3. funding gaps can become zero carry if not handled strictly
4. data gaps and quarantined anomalies can remain warnings rather than blockers

### Target State

Every feature and context input is point-in-time safe, missingness is represented explicitly, and trade-ready mode blocks on uncertified data.

### Required Code Changes

1. Make lookahead provocation default on all model-training paths, not only AutoML, custom builders, or trade-ready mode.
2. Add an explicit feature-availability contract. Every feature block should carry:
   1. `event_timestamp`
   2. `available_timestamp`
   3. `source`
   4. `join_mode`
3. Reject any feature whose `available_timestamp > decision_timestamp`.
4. Replace zero-fill defaults for context and funding in trade-ready mode with `preserve_missing` plus gating.
5. Add a unified `data_certification` report that includes:
   1. data gap policy outcome
   2. quarantine counts and blocked rows
   3. funding coverage
   4. context TTL and unknown-state rate
   5. reference data coverage
6. Make `data_certification` a blocking promotion gate in trade-ready workflows.
7. Require explicit historical universe snapshots for any multi-symbol experiment.
8. For single-symbol strategies, require either sibling-asset replication or alternate contiguous regime replication before capital release.

### Recommended Implementation Shape

1. Add `FeatureAvailabilityFrame` support in [core/features.py](core/features.py) and [core/context.py](core/context.py).
2. Extend [core/pipeline.py](core/pipeline.py) so `FeaturesStep` and context loaders emit
availability metadata.
3. Centralize missingness policy resolution in one place and emit it in summaries.
4. Extend the existing lookahead report so it names which artifact failed and why.

### Tests

1. `tests/test_global_lookahead_guard_default.py`
2. `tests/test_feature_availability_contract.py`
3. `tests/test_trade_ready_data_certification_gate.py`
4. `tests/test_funding_missing_blocks_trade_ready.py`
5. `tests/test_context_unknown_state_rate.py`

### Acceptance Criteria

1. Trade-ready runs fail closed on uncertified data.
2. Missingness is observable as missingness, not silently converted into alpha.
3. Lookahead audit always runs before a capital-facing certification decision.
## WS-07: Execution Realism And Pre-Capital Calibration

### Problem

This is partially addressed already. [example_trade_ready_automl.py](example_trade_ready_automl.py) now hard-fails without a real Nautilus backend instead of silently falling back, and [core/backtest.py](core/backtest.py) explicitly reports surrogate execution limitations such as `bar_surrogate_only`, `no_queue_position_model`, and `no_event_driven_ack_latency` whenever the adapter boundary is unavailable.

That is directionally correct, but it is still not enough to risk money after a single historical backtest because there is still no paper or shadow-live calibration artifact or release gate.

### Target State

Trade-ready mode is impossible without:

1. an event-driven backend
2. paper or shadow-live evidence
3. calibration of fill, fee, and slippage error versus expected behavior

### Required Code Changes

1. Keep surrogate execution available only for research.
2. Add a pre-capital evidence store:

```text
.cache/deployment/
  <deployment_candidate_id>/
    paper_metrics.json
    slippage_calibration.json
    fill_quality.json
    readiness.json
```

3. Add `deployment_candidate_id = stable_hash(experiment_id + frozen_candidate_hash + execution_profile)`.
4. Introduce `build_live_calibration_report(...)` that compares paper or shadow-live results with certified expectations:
   1. realized slippage vs modeled slippage
   2. fill ratio vs modeled fill ratio
   3. order rejection rate
   4. stale data incidents
   5. latency percentile breaches
5. Add hard release gates such as:
   1. no missing market or funding data breaches in paper window
   2. realized slippage error within configured tolerance
   3. fill ratio degradation within tolerance
   4. no kill-switch triggers in paper window
6. Add explicit maker or passive-order restrictions if queue modeling is absent.
7. Do not permit Kelly sizing above a capped fraction until paper calibration passes.

### Recommended Capital-Release Ladder

1. `research_certified`
   1. historical certification only
   2. no live capital allowed
2. `paper_verified`
   1. 4 to 8 weeks paper or shadow-live
   2. all monitoring thresholds live and binding
   3. live calibration report green
3. `micro_capital`
   1. fixed notional or capped quarter-Kelly maximum
   2. daily stop-loss and kill-switch enabled
4. `scaled_capital`
   1. only after micro-capital window stays within modeled drawdown and slippage error budgets

### Tests

1. `tests/test_trade_ready_requires_paper_report.py`
2. `tests/test_live_calibration_report.py`
3. `tests/test_capital_release_ladder.py`
4. `tests/test_kelly_cap_before_paper_green.py`

### Acceptance Criteria

1. No candidate can be marked `capital_release_eligible=True` without a green paper verification stage.
2. Surrogate-only results are visibly tagged `research_only=True`.

## WS-08: Statistical Evidence Floor And Sizing Discipline

### Problem

The current evidence floor is too low for risking money. Even if DSR, PBO, SPA, White Reality Check, and bootstrap intervals are implemented correctly, they are only meaningful if:

1. the simulator is faithful enough
2. the number of effectively independent bets is large enough
3. the holdout is large and recent enough
4. the sizing layer does not amplify uncalibrated model error

### Target State

Trade-ready profiles use stricter thresholds and treat low-power results as abstentions.

### Required Code Changes

1. Distinguish bar count from effective trade or bet count in all significance logic.
2. Add `effective_bet_count` to backtest and selection summaries.
3. Make significance unavailable when only bar count is sufficient but effective bet count is not.
4. Introduce stricter certification defaults, for example:
   1. `minimum_dsr_threshold >= 0.60`
   2. `max_generalization_gap <= 0.10 to 0.15` depending on strategy cadence
   3. `max_param_fragility <= 0.10`
   4. `min_validation_trade_count >= 75` for intraday and `>= 40` only for slower strategies
   5. `min_locked_holdout_trade_count >= 50`
   6. `Sharpe_CI_lower > 0` net of costs
   7. `PBO <= 0.20`
   8. `SPA` or `White RC` pass under certification profile
5. Add a certification-only rule that Kelly sizing is disabled or capped until:
   1. calibration error is under threshold
   2. paper verification is green
6. Add explicit `underpowered_reason` fields
instead of treating unavailable significance as neutral.

### Recommended Implementation Shape

1. Extend significance payloads in [core/backtest.py](core/backtest.py).
2. Extend objective gates in [core/automl.py](core/automl.py) to consume effective-bet counts and CI lower bounds.
3. Extend promotion policy to treat missing or underpowered significance as blocking in certification mode.
### Tests

1. `tests/test_effective_bet_count_significance.py`
2. `tests/test_certification_stat_thresholds.py`
3. `tests/test_underpowered_is_blocking.py`
4. `tests/test_kelly_disabled_when_uncalibrated.py`

### Acceptance Criteria

1. Underpowered certification runs abstain.
2. Kelly sizing cannot magnify uncertified results.

## WS-09: Universe, Portability, And Survivorship Controls

### Problem

Even with historical universe support in [core/universe.py](core/universe.py#L265), a user can still cherry-pick one asset, one venue, or one regime and obtain a visually strong result.

### Target State

Capital-facing certification requires at least one portability check beyond the original narrative.

### Required Code Changes

1. Add `portability_contract` to certification profiles.
2. Require at least one of:
   1. sibling-asset replication
   2. alternate contiguous regime replication
   3. alternate venue replay when data exists
3. Add portability summaries to promotion reports:
   1. pass or fail
   2. number of cohorts attempted
   3. number of cohorts passed
   4. degradation versus primary cohort
4. Disallow synthetic universe defaults outside demo examples.

### Tests

1. `tests/test_portability_contract.py`
2. `tests/test_demo_universe_not_allowed_in_certification.py`

### Acceptance Criteria

1. A single cherry-picked asset cannot pass certification without another supporting cohort.

## WS-10: Capital-Release Ladder And Operations

### Problem

The repo now contains real operator-facing readiness components: finite trade-ready monitoring thresholds, monitoring artifacts, and blocking deployment-readiness logic in [core/readiness.py](core/readiness.py). The remaining gap is that deployment still collapses to a ready-or-hold verdict without an explicit capital-release stage, paper-verification state, `release_blockers`, or manual advancement control.

### Target State

Capital release is a staged operational verdict with explicit blockers, not a boolean inferred from one summary.

### Required Code Changes

1. Introduce `capital_release_stage` with allowed values:
   1. `research_only`
   2. `research_certified`
   3. `paper_verified`
   4. `micro_capital`
   5. `scaled_capital`
2. Extend deployment readiness to include:
   1. monitoring policy health
   2. backend availability
   3. paper verification result
   4. rollback readiness
   5. operational limits and kill switch status
3. Add a `release_blockers` array to readiness reports.
4. Add a `release_runbook.md` section to docs describing exactly how operators advance stages.
5. Require manual operator acknowledgment before moving from `paper_verified` to `micro_capital`.

### Tests

1. `tests/test_release_stage_transitions.py`
2. `tests/test_release_blockers_surface.py`
3. `tests/test_micro_capital_requires_manual_ack.py`

### Acceptance Criteria

1. No output can imply “ready to trade” without stating the release stage.
2. The default stage after historical certification remains `research_certified`, not `micro_capital`.

## Rollout Order

### Phase 0: Stop Silent Contamination

1. WS-01 experiment lineage and resume guards
2. WS-02 explicit split ownership
3. WS-03 abstention-based selection outcomes

Reason: these fix the parts most likely to generate a polished but invalid result.

### Phase 1: Make Validation Contract Explicit

1. WS-04 validation-contract and metric-source reporting
2. WS-05 search-space decomposition

Reason: this turns a collection of strong ideas into a controlled certification workflow.
### Phase 2: Harden Data And Execution

1. WS-06 data certification and universal lookahead guard
2. WS-07 event-driven plus paper calibration
3. WS-08 stricter significance and sizing discipline

Reason: this is the minimum package needed before any money discussion is credible.

### Phase 3: Portability And Operations

1. WS-09 portability contract
2. WS-10 capital-release ladder

Reason: these are what separate a strong backtest stack from an operational trading workflow.

## CI And Validation Matrix

Run the following focused checks in CI now, split between existing coverage that should already stay green and missing coverage that should be added as the remediation lands.

Existing focused coverage to keep green:

1. `python -m pytest tests/test_automl_holdout_objective.py -q`
2. `python -m pytest tests/test_cross_stage_embargo.py -q`
3. `python -m pytest tests/test_pipeline_lookahead_guard_wiring.py -q`
4. `python -m pytest tests/test_funding_coverage_gate.py -q`
5. `python -m pytest tests/test_feature_portability_gates.py -q`
6. `python -m pytest tests/test_cross_venue_reference_validation.py -q`
7. `python -m pytest tests/test_operations_monitoring.py -q`
8. `python -m pytest tests/test_operational_trade_ready_path.py -q`

Missing focused coverage to add:

9. `python -m pytest tests/test_automl_experiment_manifest.py -q`
10. `python -m pytest tests/test_automl_resume_guard.py -q`
11. `python -m pytest tests/test_temporal_split_row_accounting.py -q`
12. `python -m pytest tests/test_explicit_split_bypass_gap.py -q`
13. `python -m pytest tests/test_automl_selection_outcomes.py -q`
14. `python -m pytest tests/test_validation_contract_resolution.py -q`
15. `python -m pytest tests/test_metric_source_reporting.py -q`
16. `python -m pytest tests/test_replication_is_binding.py -q`
17. `python -m pytest tests/test_global_lookahead_guard_default.py -q`
18. `python -m pytest tests/test_trade_ready_data_certification_gate.py -q`
19. `python -m pytest tests/test_trade_ready_requires_paper_report.py -q`
20. `python -m pytest tests/test_effective_bet_count_significance.py -q`
21. `python -m pytest tests/test_release_stage_transitions.py -q`

Add three smoke integration profiles:

1. research-only surrogate smoke: must succeed but stay `research_only=True`
2. trade-ready certification without event-driven backend: must fail closed
3. paper-verified certification mock: must remain blocked from capital release unless all readiness and calibration gates pass

## Documentation Changes

Update the following after implementation:

1. [README.md](README.md)
   1. add a “Research vs Certification vs Capital Release” section
   2. state that certification is not live approval
2. [HOW_TO_USE.md](HOW_TO_USE.md)
   1. add experiment manifest and resume guidance
   2. add validation-contract explanation
   3. add paper-verification runbook
3. [example_automl.py](example_automl.py)
   1. keep it explicitly labeled as the research-only surrogate path
   2. print `research_only` and `capital_release_stage`
4. [example_trade_ready_automl.py](example_trade_ready_automl.py)
   1. preserve the current hard fail when Nautilus is unavailable
   2. print validation sources, experiment id, release stage, and blockers
5. [example_drift_retraining_cycle.py](example_drift_retraining_cycle.py)
   1. show how abstention and release blockers prevent promotion

## What “Safe Enough To Risk Money” Means Here

Even after all of the above is implemented, the correct interpretation is still narrow:

1. the system is safe enough to risk small, explicitly capped capital under a staged release plan
2. it is not proof of durable alpha
3. it is not a guarantee against regime break, venue degradation, or operator error

The minimum acceptable operational path is:

1. historical certification passes under the explicit validation contract
2. paper or shadow-live verification passes for 4 to 8 weeks
3. micro-capital run passes with live monitoring and no blocker events
4. only then does scaled capital become eligible

If that ladder is not acceptable, the system should remain research-only.

## Definition Of Done

This remediation is complete only when:

1. focused governance tests are green
2. study reuse is explicit and lineage-validated
3. staged gaps are owned by explicit split definitions
4. no-eligible-trial outcomes abstain instead of crashing
5. selection-source and tradable-source metrics are both visible and binding
6. certification runs use frozen thesis definitions and replication gates
7. trade-ready mode remains fail closed without an event-driven backend and additionally blocks on paper verification
8. significance is based on effective bets and can block underpowered runs
9. every deployment-facing summary includes `capital_release_stage` and `release_blockers`

Only then is the repository’s minimum bar high enough to discuss risking money at all.