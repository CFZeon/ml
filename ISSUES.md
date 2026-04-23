# ISSUES

## Audit Findings

### Data Integrity & Preprocessing

- Critical | Duplicate timestamp collisions are silently collapsed in `core/data.py` via `duplicated(keep="first")` before quarantine runs. If Binance restates history, cached and refetched windows disagree, or two sources collide, the conflict disappears instead of failing fast. That converts provenance corruption into apparently clean data.

- Critical | Derivatives "recent stats" have no publication-lag model. `openInterestHist`, `takerlongshortRatio`, `globalLongShortAccountRatio`, and `basis` are fetched in `core/context.py` by endpoint timestamp and merged with backward as-of alignment as if they are actionable immediately at that timestamp. For a retail operator on consumer hardware, that can leak end-of-period aggregates into the first bar where they could not yet be known live.

- Critical | Context missingness can still be converted into tradable numeric states. The library default resolves `context_missing_policy` to `zero_fill`; `build_cross_asset_context_feature_block()` forward-fills when TTL is unset, and `build_multi_timeframe_context_feature_block()` ends with `.fillna(0.0)`. Missing, stale, and genuinely flat states can therefore become the same feature value.

- Critical | Futures funding is only safe when strict coverage is explicitly configured. Base backtest and model utilities still reindex missing funding to `0.0`, so API gaps or incomplete funding history undercharge one of the main short-horizon futures costs exactly when market stress is highest.

- High | Exchange downtime and missing candles are permissive by default. `fetch_binance_bars(..., gap_policy="warn")` allows incomplete windows to stay in research with warnings rather than invalidating the run. In 24/7 crypto, outages and contract interruptions are part of the data-generating process, not harmless nuisance events.

- High | Cross-venue validation can still pass on partial external coverage. `core/reference_data.py` only blocks partial reference coverage if configured to do so, so Binance-local distortions can survive the integrity layer while still carrying a "validated" report.

- High | The quarantine layer does not address crypto-specific manipulation modes. `core/data_quality.py` checks OHLC consistency, spikes, volume, quote-volume mismatch, and trade-count anomalies, but not wash trading, spoofing, self-trade bursts, or fabricated liquidity.

### Feature Engineering

- Critical | The highest-signal contextual features are timestamp-aligned, not causally modeled. That is not enough for endpoints whose release time lags their economic interval. A feature can be point-in-time safe on paper and still be unavailable at the actual decision time.

- High | The feature layer can manufacture stability from missingness. Zero-filled or forward-filled context creates long flat runs that the model can interpret as regime information, even when the true source is outage, sparse coverage, or stale leader data.

- High | AutoML does not just tune estimators; it searches feature construction, label construction, and regime specification simultaneously. Lags, fractional differencing, rolling windows, squeeze quantiles, barrier sizes, holding periods, regime count, feature-selection thresholds, model family, and hyperparameters are all in the search surface. On one symbol and one timeframe, that is a large specification search relative to the sample size.

### AutoML & Validation

- High | Replication is optional and off by default, including in the hardened trade-ready profile. A candidate can clear search, one contiguous validation slice, and one locked holdout without proving it survives alternate windows or related symbols.

- High | The temporal validation leg is a single contiguous split (`n_splits=1` in the pre-holdout replay). If search, validation, and holdout all sit inside the same dominant market narrative, the winner can still be a regime-specific accident with good DSR/PBO paperwork.

- High | The fragility check is parameter-local only. It perturbs neighboring override values, not data revisions, timestamp jitter, fee shocks, missing-bar patterns, or liquidity shocks. A model can look stable to configuration nudges and still be brittle to the perturbations that happen live.

- High | The statistical corrections are conditional on the same historical path family. DSR, PBO, post-selection inference, and bootstrap lower bounds do reduce naive search bias, but they do not rescue omitted-variable errors like stale context, missing funding, or unrealistic fills. They can certify the wrong simulator.

### Backtesting & Evaluation

- Critical | "Event-driven" execution is still not a real matching engine in the standard setup. `core/execution/nautilus_adapter.py` is only a boundary, `requirements.txt` does not ship NautilusTrader, and the documented trade-ready example explicitly sets `force_simulation=True`. In a normal retail environment, that means the system falls back to the repo's deterministic bar surrogate.

- Critical | The fallback engine explicitly lacks the frictions that kill live crypto strategies: `no_queue_position_model`, `no_event_driven_ack_latency`, and `no_order_book_matching_engine`. Bar-level partial fills are deterministic functions of bar volume and participation caps, not of queue position, spread dynamics, or order-book state.

- High | Passive or limit behavior is not falsifiable in the default engine. The bar executor rejects non-market and non-aggressive order types, so any thesis that depends on maker rebates, spread capture, or queue priority cannot be tested here.

- High | Stress testing is scenario enumeration, not adversarial market replay. The trade-ready example checks three named scenarios once (`downtime`, `stale_mark`, `halt`), but liquidation cascades, funding spikes, API throttling, cross-venue dislocations, and correlated exchange failures are outside the default objective surface.

- High | Confidence intervals are only as good as the execution model underneath them. A stationary-bootstrap Sharpe lower bound from a bar surrogate can still be materially too optimistic once live fills, queue loss, and endpoint latency are introduced.

### Deployment Realism

- High | The main AutoML and trade-ready study paths do not automatically run drift-triggered retraining. Drift tooling exists, but it is an explicit orchestration call rather than an always-on control loop. The system can look monitored while operational behavior remains manual.

- High | Consumer-hardware latency is part of research validity here, not just ops hygiene. Repeated candidate training, bootstrap significance, post-selection checks, and context fetches can finish after the state that justified the trade has already moved. A nominally adaptive system that reacts late is still stale.

- High | The documented trade-ready path is promotion-safe only if the operator respects the rejection signal. On a standard install it is still likely to end in surrogate execution or `promotion ok : False`. A retail user who treats the best trial as deployable anyway will be trading a model that the system itself did not certify.

## Failure Modes

- The strategy looks profitable because missing funding during data gaps is priced as zero carry.

- It looks regime-aware because missing or stale context is numerically transformed into stable feature plateaus.

- It looks point-in-time safe because timestamps are monotone, while derivatives-context endpoints are still consumed without a publication-lag model.

- It looks cross-venue validated even when external venue coverage is only partial.

- It looks statistically robust because DSR, PBO, and bootstrap lower bounds are positive, even though the live execution process was never modeled.

- It looks trade-ready because the execution policy says `nautilus`, while the actual backend is still a bar surrogate on a normal retail install.

- It looks stable because the winner survives nearby hyperparameter perturbations, but small data revisions, bar drops, fee shocks, or liquidity shocks were never part of the fragility test.

- It survives one validation slice and one locked holdout, then fails live because replication across alternate windows or symbols was never required.
