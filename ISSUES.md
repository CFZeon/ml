Two-Stage Holdout for AutoML

**Priority**: High  
**Status**: Open  
**Decision**: Two-stage holdout (search → validation → final holdout). DSR threshold configurable per study with generous default (0.3). Enable Optuna MedianPruner.

### Architecture

```
Full data
├── [60%] Search window — Optuna runs inner WF/CPCV here
├── [20%] Validation holdout — used for trial RANKING (replaces inner OOS for selection)
└── [20%] Final holdout — touched once for the winning trial
```

### Steps

1. **Modify `core/automl.py` → `_resolve_locked_holdout_plan()`**
   - Rename to `_resolve_holdout_plan()`.
   - Add `validation_fraction` (default 0.2) alongside existing `locked_holdout_fraction` (default 0.2).
   - Compute three splits: `search_end`, `validation_end`, `holdout_start`.
   - Return plan dict with keys: `search_rows`, `validation_rows`, `holdout_rows`, `validation_start_timestamp`, `holdout_start_timestamp`.

2. **Modify `run_automl_study()`**
   - After each trial runs on the search window, re-run the best config on the **validation window** using a single walk-forward split (train=search, test=validation).
   - Store the validation-window metrics in the trial record as `validation_metrics`.
   - Change trial ranking: use validation-window metrics for selection instead of inner-search OOS metrics.
   - Add `minimum_dsr_threshold` config (default: 0.3 — generous, avoids rejecting marginal strategies prematurely). Trials with DSR below this threshold are penalized to `-inf` in ranking.

3. **Add Optuna pruning**
   - Instantiate `optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)` when creating the study.
   - After each trial completes, report the objective value as an intermediate result. Optuna prunes trials that underperform the running median.
   - Config key: `automl.enable_pruning: true` (default: true).

4. **Modify `_evaluate_locked_holdout()`**
   - Now only operates on the final 20% holdout.
   - Receives the best trial's config **after** validation-window ranking.
   - Add `holdout_warning` flag: if holdout Sharpe CI lower bound < 0, emit a warning.

5. **Update study report**
   - Add `validation_holdout` section alongside `locked_holdout` in the study summary.
   - Report both validation-window and final-holdout metrics.

6. **Add tests**
   - `test_two_stage_holdout_validation_used_for_ranking()` — verify trial selection uses validation metrics, not search-window metrics.
   - `test_dsr_threshold_rejects_low_dsr_trials()` — verify trials below threshold are penalized.
   - `test_optuna_pruner_reduces_trial_count()` — verify that clearly bad trials are stopped early.

### Acceptance Criteria

- Trial selection uses validation-window metrics, not inner-search OOS metrics.
- DSR threshold is configurable; trials below it are rejected.
- Optuna MedianPruner reduces average trial runtime.
- Final holdout is only evaluated once on the winning trial.