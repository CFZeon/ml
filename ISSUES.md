# ISSUES

## Audit Findings

- Critical | Data Integrity / Feature Engineering: exogenous context is still allowed to become stale but numerically valid. In `core/context.py`, `build_cross_asset_context_feature_block()` does `data.reindex(base_data.index).ffill()` with no TTL, `build_multi_timeframe_context_feature_block()` as-of joins higher-timeframe bars and then `.ffill()`, and futures-context features are TTL-masked but later collapsed with `.fillna(0.0)`. Missing or stale state is therefore not cleanly separable from genuine zero values. That can make the model look regime-aware while it is actually trading on dead feeds, sparse coverage, or outage artifacts.

- Critical | Backtesting / Futures Economics: missing funding is priced as zero. Funding is reindexed and `.fillna(0.0)` in both the pipeline path and the backtest path. During API gaps or incomplete funding history, the system undercharges carry exactly when funding can dominate short-horizon edge. That directly inflates futures PnL and distorts model ranking.

- Critical | Deployment Realism: `execution_policy.adapter="nautilus"` is still a boundary, not a matching engine. `core/execution/nautilus_adapter.py` only exposes availability metadata; when Nautilus is unavailable the backtest either raises or requires `force_simulation=true`, which falls back to the repo’s own bar surrogate. The backtest itself records `no_queue_position_model`, `no_event_driven_ack_latency`, and `no_order_book_matching_engine` as explicit limitations. A strategy can therefore post attractive execution-aware metrics without ever facing the dominant live failure modes.

- High | AutoML / Multiple Testing: the strongest post-selection falsification is non-binding by default. `_resolve_overfitting_control()` sets `post_selection.require_pass=False`, so White Reality Check and Hansen SPA can be computed, reported, and still not invalidate the selected winner. In a repeated-trial AutoML loop, that is a clean path to selecting noise with an institutional-looking audit trail.

- High | Data Integrity / Preprocessing: duplicate bars are silently collapsed with `keep="first"` in `_prepare_frame()` and `_merge_frames()` in `core/data.py`. If Binance restates data, cache and refetch disagree, or two sources collide, the first row wins with no reconciliation or hard failure. That preserves stale data while downstream contract checks still see a structurally valid frame.

- High | Data Integrity / Crypto Microstructure: `core/data_quality.py` does not attempt wash-trading, spoofing, self-trade, or venue-manipulation detection. It flags OHLC inconsistencies, spikes, nonpositive volume, quote-volume mismatch, and trade-count anomalies, but manipulated prints can still survive into both features and labels. In crypto, that often means the model learns exchange pathology instead of transferable edge.

- High | Exchange Fragmentation / Reference Validation: spot cross-venue validation is allowed to remain advisory. `build_spot_reference_validation()` keeps `promotion_pass=True` under partial venue coverage and only blocks divergence when configured blocking; the tests explicitly assert that partial coverage can still pass. For a retail Binance trader, this means fragmented or incomplete external price discovery can appear as a validation badge rather than a hard stop.

- High | Deployment / Regime Handling: drift tooling exists, but the retraining loop is not wired into the main pipeline. `run_drift_retraining_cycle()` lives in `core/orchestration.py` and is exported, but it is not invoked by `ResearchPipeline` or the AutoML study path. The repo can generate monitoring and drift artifacts while still behaving operationally like a manual or scheduled system once conditions change.

- Moderate | Backtesting / Stress Realism: venue downtime, stale marks, halts, forced deleveraging, and lifecycle events are opt-in scenario inputs, not baseline selection conditions. `run_backtest()` can model them, but AutoML still optimizes the unstressed path unless the user explicitly runs a scenario matrix. That makes robustness easy to overstate because the stress distribution is optional paperwork.

- Moderate | Deployment Realism / Execution Style: the default bar engine only supports market or aggressive execution. Passive and limit-order behavior is explicitly unsupported in `_build_execution_contract()`. Any edge that depends on maker placement, queue priority, or spread capture is therefore not actually validated by this stack.

- Moderate | Evaluation / Decay Robustness: signal-decay diagnostics become advisory when realized trade count is low. In `core/signal_decay.py`, low sample size downgrades the gate to `advisory` rather than blocking. That avoids false negatives, but it also lets thin evidence produce a “looks monitored” posture on strategies that barely traded.

- Moderate | Missing-Data Semantics: across `core/context.py`, `core/pipeline.py`, and `core/backtest.py`, absent features, absent funding, absent benchmark returns, and absent liquidity are frequently converted to `0.0`. Zero is a valid economic state, not a generic unknown token. This compresses uncertainty into the feature and cost space and makes the system appear more stable than the data actually are.

- Moderate | Retail Deployability: the study path reruns full candidate pipelines and may then add validation holdout, locked holdout, replication, post-selection inference, and monitoring. On consumer hardware, that creates a real risk that retraining finishes on yesterday’s market. A statistically careful pipeline is still operationally wrong if the retrain latency is longer than the market-state half-life.

## Failure Modes

- The model looks regime-aware because cross-asset and futures-context features are filled through gaps, but live trading reacts to stale or dead state.

- The strategy looks profitable on futures because missing funding prints are treated as zero carry instead of unknown or adverse carry.

- The execution layer looks realistic because order intents, partial fills, and participation caps exist, while the live-dominant frictions still have no matching engine behind them.

- The research looks statistically disciplined because White RC, SPA, holdouts, and monitoring appear in reports, but the strongest rejection test can still be non-binding.

- Cross-venue validation can make a fragmented or manipulated market look safer than it is because incomplete venue coverage still passes by default.

- Data integrity can look clean after duplicate collapsing even when the underlying source delivered conflicting bars.

- A black-swan-sensitive strategy can survive selection because crash, deleveraging, halt, and downtime scenarios are not part of the default objective path.

- A retail user can mistake “promotion-ready metadata exists” for “the system is deployable,” even though the code itself records material execution limitations.