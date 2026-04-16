No Statistical Significance Testing on Backtest Results

- **What**: The pipeline reports metrics (Sharpe, net profit, win rate) as point estimates. No confidence intervals, no bootstrap distributions, no hypothesis tests.
- **Industry standard**: At minimum, block-bootstrap the equity curve to produce confidence intervals on Sharpe. Better: use the stationary bootstrap (Politis & Romano) which preserves serial correlation. Report p-values for Sharpe > 0 and for Sharpe > benchmark. AQR and Man Group publish bootstrapped confidence intervals as standard practice.
- **Why it matters**: A Sharpe of 1.5 computed from 3 months of hourly data has wide confidence intervals. Without them, there is no way to distinguish signal from noise. The reported Sharpe 8.48 from the initial test run (43 trades, 3 months) almost certainly has a 95% CI that includes 0.
- **Gap in repo**: `_summarize_backtest()` returns scalar metrics only. No bootstrapping, no CI, no hypothesis testing.
- **File**: [core/backtest.py](core/backtest.py)