# TODO

1. [x] Implement Combinatorial Purged Cross-Validation (CPCV).
2. [x] Fix the AutoML Objective (Multiple Testing Correction / Deflated Sharpe Ratio).
3. [x] Hard-Code Labeling Parameters (remove from AutoML search space).
4. [x] Strengthen Metalabeling (ensure meta-model is only trained on OOS primary predictions).
5. [x] Vectorize Fracdiff (use `scipy.signal.lfilter` or robust loop for performance).
6. [x] Fix Path-Dependency Tie-Breaking Bias (default `triple_barrier_labels` to SL when both barriers hit).
7. [x] Implement Kelly Shrinkage / Half-Kelly Caps (prevent "Blow-up" risk from raw OOS estimates).
8. [ ] Strengthen Meta-Model Purging (ensure purging gap covers label autocorrelation/horizon).
9. [ ] Eliminate Global Stationarity "Preview" (strict fold-local ADF testing and transform selection).
10. [ ] Handle Data Gaps in Indicators (implement time-weighted indicators or window rejection).
11. [ ] Reduce Fracdiff Data Attrition (use 'same' mode or padding to preserve initial samples).
12. [ ] Implement HMM or Markov-Switching Regimes (replace unstable KMeans clustering).
