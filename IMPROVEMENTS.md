# IMPROVEMENTS - Adversarial Audit And Gap Ranking

## Scope

- Benchmarked this repository against publicly documented practices from AQR, Man Group, Two Sigma, Microsoft Qlib, QuantConnect, NautilusTrader, and Binance exchange docs.
- Standard used: point-in-time data integrity, independent replication, time-aware validation, realistic execution, explicit regime handling, online/offline separation, and closed-loop monitoring.
- Relative verdict: this repo is materially stronger than the average retail quant stack. Relative to public institutional standards, it is still not deployable. The biggest remaining risks are not missing features; they are execution surrogates, non-binding governance, single-venue truth, and operations gaps.
- Not the first place I would attack: CPCV and embargo logic, post-selection inference, gap handling, historical universe snapshots, and futures funding and liquidation simulation are already better than most hobby codebases.

## P0 - Could Invalidate Results

- Execution realism is still a surrogate. [core/execution/nautilus_adapter.py](core/execution/nautilus_adapter.py) is only an adapter boundary, while the canonical fill logic still lives in [core/backtest.py](core/backtest.py). Public baselines such as NautilusTrader and QuantConnect treat queue position, book type, latency, price protection, partial fills, and timestamp semantics as first-class venue mechanics. If fills are wrong, every downstream Sharpe, Kelly size, fragility score, and drift threshold is downstream wrong.
- The repo computes more governance than it enforces. [core/pipeline.py](core/pipeline.py) emits feature portability and regime stability promotion gates, but [core/automl.py](core/automl.py) eligibility currently binds feature admission and operational health, not portability or regime stability. This is a classic false-comfort pattern: the report looks institutionally hardened while the actual winner can still be venue-specific or regime-fragile.
- The data layer is not yet data as code. [core/data_quality.py](core/data_quality.py) catches local OHLCV anomalies, but there are no hard source contracts, schema compatibility gates, lineage attestations, or upstream change-management checks across raw market data, futures context, and custom data. Two Sigma’s published standard is versioned transforms, CI/CD, data contracts, lineage, and proactive anomaly detection. Without that, silent vendor changes can contaminate both training and validation without triggering a hard failure.
- Single-venue truth is still treated as market truth. [core/data.py](core/data.py) and [core/context.py](core/context.py) are Binance-centric, and nothing in the default research path reconciles Binance prints against other venues, filters wash-trading-like distortions, or validates whether volume and trade-count anomalies are exchange-local artifacts. That means AutoML can learn Binance microstructure pathologies instead of portable alpha.
- Signal half-life is absent. There is no signal half-life or alpha-decay computation anywhere in core or tests. Man Group’s public regime work and institutional practice more broadly treat timing horizon as part of the signal definition, not a reporting afterthought. A signal whose edge decays inside the execution lag can still look profitable in bar-based backtests and then die the moment it touches live latency.
- Out-of-sample governance is still too local. AQR’s public standard is theory plus replication across other periods, places, and related implementations. This repo deserves credit for DSR, PBO, White RC, and Hansen SPA, but the winning model is still selected inside a single-symbol, single-venue study defined by one pipeline instance in [core/automl.py](core/automl.py). Repeated search on one local market can still promote structured noise that survives one holdout.

## P1 - Major Weaknesses

- Promotion logic compares incomparable metrics. In [core/registry/manifest.py](core/registry/manifest.py), selection value comes from validation. In [core/registry/store.py](core/registry/store.py), the challenger is compared against the champion’s locked holdout score. That is not an apples-to-apples contest. Registry promotion can therefore reject or approve a model for bookkeeping reasons rather than true superiority.
- Drift handling is advisory, not operational. [core/drift.py](core/drift.py) can recommend retraining, but there is no durable closed-loop process that retrains, re-evaluates a challenger, promotes it, and rolls back safely under degraded live conditions. QuantConnect and Qlib both document explicit scheduling and online/offline separation. Here the logic largely stops at recommendation.
- The search surface is still large enough to manufacture edge. [core/automl.py](core/automl.py) searches label geometry, fractional differencing, rolling windows, squeeze thresholds, feature-selection thresholds, regime counts, model family, model hyperparameters, and calibration hyperparameters. The repo has meaningful overfitting controls, but those controls cannot fully rescue a wide search performed on one venue-symbol-history tuple.
- Data-quality controls are mostly univariate heuristics. [core/data_quality.py](core/data_quality.py) flags OHLC inconsistencies, spikes, nonpositive volume, quote-volume inconsistencies, and trade-count anomalies. It does not model quote and trade sequencing, mark and index sanity, funding-print anomalies, spoof-like liquidity behavior, or cross-source reconciliation. In crypto, that is where bad data stops looking obviously bad.
- Live and paper execution are not a config flip away. [core/execution/nautilus_adapter.py](core/execution/nautilus_adapter.py) is still a placeholder boundary, not a working execution stack. Qlib’s public workflow and QuantConnect’s public docs both separate offline experiments from online serving or trading more explicitly. This repo still has a backtest-first architecture with a live gap in the middle.
- Portfolio reality is under-modeled. The repo is intentionally one-model-per-symbol, but there is no portfolio allocator or cross-symbol capital coordination in the research path. A set of individually attractive symbol models can therefore pass independently while being impossible to run together under shared capital, leverage, and exposure limits.
- Many safeguards are demonstrated on stylized fixtures. The test suite is broad and thoughtful, but much of the evidence is synthetic, deterministic, and unit-scoped. That proves awareness of the risk. It does not prove the whole stack remains correct under months of real Binance outages, delistings, regime transitions, and custom-data defects interacting at once.

## P2 - Important But Secondary

- Futures realism is better than average but still exchange-idealized. [core/backtest.py](core/backtest.py) models funding, mark-based valuation, leverage brackets, margin modes, and liquidation. It still does not get you all the way to exchange behavior under ADL, insurance-fund effects, portfolio margin, or multi-position cross-margin contagion.
- Scenario realism exists but is not central to model promotion. The repo has downtime and venue-failure scenarios, yet selection still centers validation metrics and holdout outcomes more than stress survival. That leaves room for strategies that are statistically attractive but operationally brittle.
- Monitoring exists more as artifacts than control loops. [core/monitoring.py](core/monitoring.py) can summarize freshness, fill-ratio deterioration, and health gates. What is still missing is a continuously running system that blocks trading, de-risks positions, or forces fallback behavior when those metrics break.
- The repo still relies heavily on bar abstractions. NautilusTrader’s public documentation is explicit that data granularity changes queue dynamics, stop execution, and fill-path ordering. As long as most research remains bar-driven, any strategy that depends on intrabar precision should be treated as suspect by default.

## Components Creating False Confidence

- DSR, PBO, White RC, and Hansen SPA are real improvements. They do not rescue unrealistic fills, venue-local artifacts, or missing signal-decay analysis.
- Feature portability and regime stability diagnostics exist. Because [core/automl.py](core/automl.py) does not bind them during eligibility, they currently behave more like audit metadata than hard governance.
- Registry, monitoring, and drift modules exist. Because promotion is metric-misaligned and retraining is not closed-loop, they do not yet add up to production governance.
- Futures margin and liquidation simulation exist. Without a real event-driven execution engine, those controls can still sit on top of unrealistic entry and exit assumptions.
- A large test suite exists. Much of it validates isolated invariants, not full-chain adversarial behavior under real venue messiness.

## Failure Modes That Can Still Produce Attractive Backtests

- The model learns Binance-specific manipulation or venue-local liquidity behavior that disappears when market microstructure changes.
- The strategy looks strong because next-bar execution and surrogate fills over-credit liquidity that would not have been accessible live.
- The selected model survives one holdout but fails immediately in the next regime because decay speed was never measured.
- A venue-specific context feature dominates the edge, but portability is reported rather than enforced.
- A challenger is promoted or blocked because validation and locked-holdout metrics are mixed in registry logic.
- An upstream schema or feed change shifts meaning without tripping a hard contract failure, contaminating both training and validation.
- Multiple individually profitable symbol models cannot be run together once capital, leverage, and correlation are shared.
- Drift is detected only after performance damage, and there is no live control loop to cut risk or roll back automatically.

## Public Benchmark Basis

- AQR: public emphasis on independent replication, theory or story, and expecting materially worse out-of-sample performance than backtests.
- Man Group: public regime work uses explicit state variables, similarity scoring, exclusion windows, and regime-timing logic rather than vague regime awareness.
- Two Sigma: public data-as-code standard emphasizes data contracts, lineage, CI/CD, ownership, and proactive anomaly detection.
- QuantConnect: public reality-modeling docs treat fills, fees, slippage, buying power, brokerage rules, and walk-forward scheduling as explicit system components.
- NautilusTrader: public backtesting docs treat order-book granularity, timestamp conventions, latency, queue heuristics, and partial fills as core execution semantics.
- Binance: public filter docs show that exchange validity depends on side-aware percent-price, market lot size, min and max notional, amendment limits, and position constraints, not just tick and step rounding.

## External Sources Consulted

- https://www.aqr.com/Insights/Perspectives/The-Replication-Crisis-That-Wasnt
- https://www.man.com/insights/regimes-systematic-models-power-of-prediction
- https://www.twosigma.com/articles/treating-data-as-code-at-two-sigma/
- https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/key-concepts
- https://www.quantconnect.com/docs/v2/writing-algorithms/optimization/walk-forward-optimization
- https://nautilustrader.io/docs/latest/concepts/backtesting/
- https://qlib.readthedocs.io/en/latest/component/workflow.html
- https://developers.binance.com/docs/binance-spot-api-docs/filters