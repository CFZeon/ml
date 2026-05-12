# 2026-05-11 Institutional Research Memo

## Purpose

This memo translates the seven issues from the 2026-05-11 adversarial audit into implementation guidance that matches institutional control expectations while staying realistic for a retail trader running on consumer hardware.

## Research Basis

This write-up is based on isolated review of:

- NIST AI Risk Management Framework: govern, map, measure, and manage model risk instead of treating missing evidence as acceptable evidence.
- BCBS 239 principles for effective risk data aggregation and risk reporting: risk data and control outputs must be accurate, complete, timely, and adaptable under stress.
- Repo research notes already captured in repository memory, especially around routing runtime binding, regime-layer redesign, execution-realism gating, cross-stage embargoes, and drift/retraining governance.

## Retail Constraint Set

The target system is not a bank stack. The right translation for retail on consumer hardware is:

- One authoritative CPU-friendly decision path.
- Deterministic outputs across reruns.
- Fail-closed semantics in capital-facing modes.
- Compact audit artifacts in JSON or Parquet, not heavy distributed infrastructure.
- Simple detectors and bounded loops over bars and specialists, not cluster-scale streaming systems.

## Governing Design Rules

- Unknown is not pass.
- Diagnostic-only controls must not be presented as execution evidence.
- Routing, regime detection, and drift monitoring must share a stable semantic contract across retrains.
- Research and promotion should consume the same causal objects whenever capital claims are being made.
- Consumer-hardware constraints should remove unnecessary complexity, not remove binding controls.

## 1. Bind Router Decisions Into Executed PnL

### Institutional Readout

Institutional model governance does not accept a split between the policy that is described and the policy that is actually charged to PnL. If routing is a capital-relevant decision, then route choice, switch timing, and switch cost must live inside the execution path that produces backtest returns and promotion evidence.

### Best Solution

- Make the router state machine part of the backtest execution path, not a replay-only summary attached afterward.
- Compute positions from the selected specialist or fallback model at each decision timestamp.
- Apply router switch friction, blocked switches, and route-induced model changes directly to holdings and fills.
- Keep replay traces, but downgrade them to audit artifacts rather than the basis for routing claims.
- Ensure promotion gates read only the backtest payload produced by executed routing, not an auxiliary trace.

### Consumer-Hardware Implementation

- Precompute per-specialist predictions, probabilities, and health inputs once per fold.
- Run a lightweight per-bar router loop across cached arrays instead of launching a separate simulator per specialist.
- Persist only a compact decision trace: timestamp, selected model id, confidence, reason, blocked-switch reason, and applied switch cost.
- Keep the number of live-eligible specialists small and certified. Retail hardware should not be routing across dozens of models.

### What Not To Build

- Do not add a full exchange simulator just to make routing binding.
- Do not build a reinforcement-learning router.
- Do not duplicate one execution path for research and another for promotion.

### Acceptance Standard

- Disabling the router must change both the decision trace and the equity curve.
- Router switching costs must reduce realised returns, not only appear in a report.
- Promotion eligibility for routing must become impossible when only replay diagnostics are available.

## 2. Treat Warm And Unavailable Regime States As Safe-Mode Inputs

### Institutional Readout

Institutional controls distinguish between a known low-confidence state and an unknown state. Warmup, missing observations, and detector fallback are safe-mode conditions. They should not be handled as ordinary inputs to route selection.

### Best Solution

- Promote regime state to an explicit tri-state contract: known, warm, unavailable.
- While the detector is warm or unavailable, route only to a certified fallback model or to no-trade.
- Require explicit confidence and persistence thresholds only after warmup completes.
- Replace lexical or model-id tie-breaking with an explicit safe-priority policy.
- Log every warm-state decision so the operator can quantify how often the router is operating outside full information.

### Consumer-Hardware Implementation

- Use boolean masks and small counters, not probabilistic ensembles or online smoothing layers.
- Expose a single config for `warm_mode = fallback_only | no_trade`.
- Keep `warmup_bars`, `min_confidence`, and `min_persistence_bars` deterministic and symbol-specific.
- Prefer a global fallback that is already certified over trying to infer a specialist during detector warmup.

### What Not To Build

- Do not let the router infer “probably okay” behaviour from missing-health defaults.
- Do not let warm states degrade into alphabetical or id-order routing.
- Do not add hidden heuristics that are hard to audit later.

### Acceptance Standard

- Warm or unavailable regime states never cause a specialist switch unless that policy is explicitly certified.
- Tie situations always resolve to a documented safe default.
- Router traces report the share of decisions taken in warm or unavailable mode.

## 3. Make Sparse-Evidence Router Gates Block Capital-Facing Promotion

### Institutional Readout

A control with insufficient evidence is not a passed control. In capital-facing systems, sparse evidence should map to unknown or blocked, not green.

### Best Solution

- Change router-stability gate semantics from boolean pass/fail to `passed`, `failed`, or `unknown`.
- In trade-ready and local-certification modes, treat `unknown` as blocking.
- Require a minimum decision count before switch-rate or stability metrics can clear eligibility.
- Keep research-only mode advisory, but make the advisory status explicit in the summary.
- Separate “not applicable because router disabled” from “unknown because evidence too thin.”

### Consumer-Hardware Implementation

- This is a cheap control change. It is mostly policy wiring and report semantics.
- Persist decision counts and switch counts in every path summary.
- Default thresholds should be proportional to expected routing frequency, not arbitrary raw counts copied across timeframes.

### What Not To Build

- Do not solve this with a larger routing model.
- Do not average sparse paths until they look statistically dense.
- Do not hide `unknown` inside `passed_with_warning` semantics.

### Acceptance Standard

- A router with too few decisions cannot become promotion-ready.
- Post-selection reports distinguish failed, passed, and unknown router evidence.
- Research summaries remain available without pretending they are certification outputs.

## 4. Expand Lookahead Certification To The Full Causal Surface

### Institutional Readout

Institutional validation does not stop at raw feature generation if later stages can still inject forward dependence. Regime construction, labels, signal thresholds, and sizing logic all sit on the capital path and must be causally certified.

### Best Solution

- Expand the lookahead audit beyond `build_features` to cover:
  - regime observation and state construction
  - label generation
  - signal thresholding
  - sizing inputs and trade-outcome-derived statistics
- Run audits on fold-local objects, not only on global preview artifacts.
- Report which stages were checked, which were skipped, and why.
- Fail capital-facing promotion when any required stage is missing from the audit.
- Keep the audit focused on whether the output at time `t` changes when future rows are withheld.

### Consumer-Hardware Implementation

- Use sampled replay rather than exhaustive replay.
- Audit a small deterministic set of timestamps per fold and hash or compare the resulting outputs for each audited artifact.
- Reuse existing fold-local helpers so the audit path does not drift from the training path.
- Store only compact audit diffs and offending columns, not full duplicated matrices.

### What Not To Build

- Do not brute-force replay every timestamp on a laptop if sampled replay identifies violations accurately enough.
- Do not create a separate alternate pipeline just for auditing.
- Do not keep advisory defaults in trade-ready paths.

### Acceptance Standard

- Any forward dependence introduced after feature generation still causes the guard to fail.
- Capital-facing summaries show full stage coverage, not just a generic passed flag.
- Missing audit coverage resolves to blocking status, not silent success.

## 5. Treat Specialist Fallback Share As Concentration Risk

### Institutional Readout

If a specialist system silently routes a large share of decisions into a generalist fallback, then the system is no longer behaving as a true specialist architecture. That is concentration risk disguised as graceful degradation.

### Best Solution

- Enable fallback-share limits by default in capital-facing modes.
- Require minimum distinct regimes and minimum sample depth before a specialist bundle can be considered specialist-ready.
- Track fallback share by fold, by regime, and over time.
- If fallback share breaches threshold, either:
  - block promotion, or
  - explicitly relabel the candidate as generalist-with-regime-features rather than specialist.
- Treat unseen-regime traffic as a first-class governance metric, not a debugging field.

### Consumer-Hardware Implementation

- Use simple counters and ratios.
- Keep one certified global fallback model; retail infrastructure does not need layered fallback trees.
- Add explicit summary fields that operators can inspect quickly: fallback rows, fallback share, unseen regimes, and specialist coverage.
- Bound the number of regimes and specialists so coverage remains statistically meaningful on retail data windows.

### What Not To Build

- Do not mask poor specialist coverage by averaging everything into one aggregate OOS score.
- Do not create multiple nested fallback chains.
- Do not allow specialists to remain eligible when most live traffic goes to fallback.

### Acceptance Standard

- A specialist candidate with excessive fallback share cannot become promotion-ready.
- Reports surface whether the candidate is truly specialist or effectively generalist.
- Unseen-regime usage is visible in both training and post-selection outputs.

## 6. Replace Ordinal Regime IDs With Stable Semantic Contracts

### Institutional Readout

Institutional routing and monitoring require stable state meaning across versions. Integer cluster ids from latent models are not a stable contract. The live system needs regime labels that remain interpretable and comparable after retraining.

### Best Solution

- Define specialist compatibility on semantic tags, not ordinal latent-state ids.
- Prefer native or rule-based detectors with interpretable outputs for live routing: trend, volatility, liquidity, structural-break states.
- If HMM or other latent detectors remain in the stack, map raw states into versioned semantic descriptors using observable properties such as realised volatility, trend strength, turnover, or break intensity.
- Persist the semantic mapping in the detector manifest and require compatibility checks to reference detector type, schema version, and semantic label.
- Keep HMMs as research tools or auxiliary features if their semantics cannot be kept stable enough for live routing.

### Consumer-Hardware Implementation

- Native detectors already fit the retail hardware constraint better than repeated HMM fitting.
- Use deterministic thresholds, rolling statistics, and small state machines instead of EM refits in the live loop.
- Keep the semantic taxonomy compact. Retail operators benefit more from stable labels like `trend_up_high_vol` than from a five-state latent clustering that changes meaning each retrain.

### What Not To Build

- Do not keep routing keyed to `0`, `1`, `2` without a semantic layer.
- Do not try to stabilise latent ids only by sorting cluster means.
- Do not hide detector-version drift from the routing compatibility logic.

### Acceptance Standard

- Specialist compatibility survives retraining when the same semantic market state recurs.
- Detector manifests expose both raw model metadata and semantic label definitions.
- Routing reports remain interpretable across promoted versions.

## 7. Upgrade Drift Monitoring From Mean-Probability KL To Decision-Aware Drift

### Institutional Readout

Institutional drift governance separates several questions:

- Did the input distribution change?
- Did the score distribution change?
- Did the action policy change?
- Did realised outcomes degrade?

Averaging each probability column and comparing only those means is too coarse for that job.

### Best Solution

- Replace mean-column KL as the primary score-drift metric with fixed-bin divergence over:
  - predicted class distribution
  - maximum probability or confidence
  - class margin or decision edge
  - action rate and abstain rate
- Add calibration-drift monitoring when realised outcomes are available.
- Keep performance drift detectors such as ADWIN or EWMA on realised trade outcomes, but treat them as one signal in a multi-signal stack rather than the sole institutional trigger.
- Separate operator actions into `observe`, `recalibrate`, `reroute`, and `retrain` depending on which drift family fired.
- Keep TTL-style freshness forcing as a distinct governance channel, not a proxy for drift quality.

### Consumer-Hardware Implementation

- Use fixed 10-bin histograms, PSI, Jensen-Shannon divergence, or similar bounded CPU-cheap metrics.
- Track action-rate drift directly from executed signals. That is often more useful for retail than an elaborate latent drift model.
- Use rolling windows sized from bars per day and realised trade rate so thresholds remain comparable across 5m, 1h, and 4h systems.
- Persist compact per-window summaries rather than full probability matrices.

### What Not To Build

- Do not rely on a single drift scalar.
- Do not use heavyweight online density estimation if fixed-bin summaries solve the operational problem.
- Do not trigger retraining solely because score means moved slightly while actions and realised edge stayed stable.

### Acceptance Standard

- Drift reports explain which layer moved: inputs, scores, actions, or realised outcomes.
- Retraining recommendations become more specific and less noisy.
- Retail operators can inspect the drift report without needing a separate analytics stack.

## Recommended Implementation Order

For this repository, the safest order is:

1. Bind router decisions into executed PnL.
2. Make warm states and sparse-evidence gates fail closed in capital-facing modes.
3. Expand the lookahead guard to all capital-relevant stages.
4. Enforce fallback-share and regime-coverage controls by default.
5. Stabilise regime semantics for routing compatibility.
6. Upgrade drift monitoring to decision-aware multi-signal reporting.

## Final Position

Institutional quality for a retail operator does not mean copying institutional infrastructure. It means copying institutional control logic:

- one authoritative path,
- explicit unknown states,
- stable semantics,
- bounded fallback behaviour,
- and monitoring that distinguishes real degradation from noisy summaries.

Those controls are feasible on consumer hardware. The main requirement is discipline at the abstraction boundary, not more compute.