# TODO

### High Priority: Correctness & Leakage Prevention
1. [x] Refine AutoML Objective: Use stable Probabilistic Sharpe Ratio (PSR) for optimization; current DSR objective is non-stationary for TPE.
2. [ ] Eliminate Global Stationarity "Preview": Implement strict fold-local ADF testing and transform selection to ensure zero leakage.
3. [ ] Strengthen Meta-Model Purging: Ensure purging gap covers label autocorrelation and event horizons to prevent meta-label leakage.

### Medium Priority: Model Robustness & Science
4. [ ] Integrate KPSS test: Add to stationarity screening as a counter-balance to ADF to prevent over-differencing and signal destruction.
5. [ ] Implement Clustered Feature Importance (CFI): Properly handle importance diagnostics for groups of redundant or highly correlated features.
6. [ ] Implement HMM or Markov-Switching Regimes: Replace unstable KMeans clustering with transition-aware regime modeling.

### Low Priority: Data Quality & Efficiency
7. [ ] Reduce Fracdiff Data Attrition: Use 'same' mode or padding to preserve initial samples and minimize data loss.
8. [ ] Handle Data Gaps in Indicators: Implement time-weighted indicators or window rejection to handle non-contiguous price data.
