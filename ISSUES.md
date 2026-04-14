[RESOLVED] Regime Detection Uses the Full Dataset (Global Fit, Not Rolling)  

Fixed:
- KMeans consistency is now enforced in `core/models.py` by sorting clusters by their centers' means.
- `ResearchPipeline` now uses fold-local fitting for KMeans to prevent look-ahead.
- `RegimeStep` (Step 4) now provides a safe, rule-based "explicit" preview for EDA without leakage.

---

Feature Selection (select_features) Runs on the Full Aligned Dataset Before Walk-Forward Splits 

 core/pipeline.py — select_features() step commentary from example.py (Step 6b): 

 "global preselection disabled; supervised MI filtering runs inside each walk-forward fold" 

 The code says this runs inside each fold, which is correct in principle. However, screen_features_for_stationarity (the ADF screening step, Step 3) runs on the full features dataframe before any fold 
 splitting. The fit_features parameter exists to fix this, but the pipeline only uses it if explicitly wired. Stationarity screening on the full dataset means ADF tests can "see" the distribution of future 
 data when deciding which transform to apply to early-period data — a subtle but real form of look-ahead.

 Triple-Barrier Labels Use high and low of Future Bars, Enabling Intrabar Touch Detection That Cannot Be Exploited at Bar Close  

core/labeling.py  

for timestamp, high_price, low_price, close_price in zip(...):  
    hit_pt = bool(high_price >= upper)  
    hit_sl = bool(low_price <= lower)  

The code checks whether the high of each future bar touched the PT barrier, or the low touched the SL. This is the correct implementation for triple-barrier labeling. However, when barrier_tie_break="sl" (the  
 default) and both are hit on the same bar, the label is -1 (stop loss). This is conservative and fine.  

The subtle flaw: the exit price is set to upper or lower (the barrier level), not the actual execution price, and this exit price is recorded in gross_return which feeds the backtest's win/loss statistics but  
 is not the price used in the actual backtest engine. The backtest uses signals + next-bar returns, not exit_price. So gross_return in the label frame is effectively decorative — it doesn't affect PnL, but it  
 does affect the avg_win / avg_loss inputs to the Kelly criterion if these are computed from label statistics rather than realized backtest returns.

 Kelly Criterion Inputs Are Config-Hardcoded, Not Derived From Out-of-Sample Evidence  

example.py  

"signals": {  
    "avg_win": 0.02,  
    "avg_loss": 0.02,  
    "fraction": 0.5,  
    ...  
}  

core/backtest.py — kelly_fraction()  

def kelly_fraction(prob_win, avg_win, avg_loss, fraction=0.5):  
    b = avg_win / avg_loss  # b = 1.0 when avg_win == avg_loss  
    k = (prob_win * b - q) / b  

avg_win and avg_loss are injected from config (both 0.02), making b=1. The Kelly fraction therefore depends only on prob_win. This is mathematically consistent if the config values actually reflect the  
strategy's true win/loss sizes, but since they're hardcoded rather than estimated from OOS fold metrics, the sizing is not grounded in the actual strategy's return profile. With b=1, Kelly degenerates to  
2*prob_win - 1, which only makes sense if the average win and loss are exactly equal. Any deviation in reality means the position sizing is wrong.

Summary of Critical Issues
Look-ahead bias in regime detection — Statistics computed on full dataset before walk-forward splits
Look-ahead bias in stationarity screening — Transform selection uses future data
Triple-barrier exit price mismatch — Barrier levels ≠ executable prices
Hardcoded Kelly parameters — Not derived from OOS evidence
No embargo after purging — Overlapping returns possible
Non-reproducible bootstrap — No random seed
AutoML not time-series aware — Standard TPE sampler
Simplified slippage model — Flat rate regardless of conditions