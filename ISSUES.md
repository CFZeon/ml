Summary Table: Priority for Retail Capital Trust
# Issue Severity Impact on Profitability Fix Difficulty
1 Research example has no holdout/selection gates | CRITICAL Results are unreliable; may be 30-70% overfit | Config change only
2 Post-selection inference disabled by default | HIGH | Cannot distinguish skill from luck with 25 trials | Config change only
3 No paper-trading validation loop | HIGH | No way to verify model works in real-time before capital | Major feature
4 No kill switch / drawdown gate | HIGH | Unbounded loss potential | Moderate feature
5 Insufficient data for statistical significance | HIGH | 4 months hourly = ~Sharpe ±1.0 CI | User decision
6 Default slippage rate = 0 | MEDIUM-HIGH | Inflates returns by 5-20% depending on pair | Config change only
7 Rolling z-score leaks ~20 test bars | MEDIUM | ~1-3% metric inflation | Code fix
8 No model TTL / forced expiry | MEDIUM | Stale model risk | Moderate feature
9 Trade-ready path requires Nautilus | MEDIUM | Retail locked out of hardened path | Design decision

Final Assessment
Rating: Research-Only

The architecture is excellent — it demonstrates awareness of virtually every institutional concern (purging, embargo, CPCV, post-selection inference, execution realism gates, locked holdouts, replication tests, stress scenarios). The code quality is high and the layered evaluation modes (research → certification → trade-ready) are well-designed.

But the accessible path for a retail trader produces unreliable results by default. The research example disables every safeguard that would make the output trustworthy. A retail user running example_automl.py gets a best-trial Sharpe that is:

Not holdout-validated
Not corrected for multiple testing
Uses zero slippage
Based on insufficient data duration
To make this system trustworthy for retail capital allocation, a user must:

Enable locked_holdout_enabled: True with at least 20% holdout
Enable overfitting_control.post_selection.enabled: True
Use at minimum 6+ months of data (ideally 12+)
Set slippage_rate ≥ 0.0003 (or use SquareRootImpactModel)
Only trust strategies where the holdout Sharpe CI lower bound > 0
Paper-trade for at least 1 month before committing capital
Implement a manual kill switch at -10% drawdown
The system is not lying to you — it labels everything as research_only and the trade-ready path explicitly fails closed. The problem is that a retail user may not understand that the research output is not evidence of profitability.