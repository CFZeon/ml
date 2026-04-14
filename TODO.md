# TODO

1. [x] Implement Combinatorial Purged Cross-Validation (CPCV).
2. [ ] Fix the AutoML Objective (Multiple Testing Correction / Deflated Sharpe Ratio).
3. [ ] Hard-Code Labeling Parameters (remove from AutoML search space).
4. [ ] Strengthen Metalabeling (ensure meta-model is only trained on OOS primary predictions).
5. [ ] Vectorize Fracdiff (use `scipy.signal.lfilter` for performance).
