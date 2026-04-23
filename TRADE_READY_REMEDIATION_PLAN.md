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
