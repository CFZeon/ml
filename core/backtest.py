"""Bet sizing and backtesting."""

import numpy as np
import pandas as pd


# ───────────────────────────────────────────────────────────────────────────
# Kelly criterion
# ───────────────────────────────────────────────────────────────────────────

def kelly_fraction(prob_win, avg_win, avg_loss, fraction=0.5):
    """Fractional Kelly position size.

    Parameters
    ----------
    prob_win : float   – estimated probability of a winning trade
    avg_win  : float   – average win magnitude  (positive)
    avg_loss : float   – average loss magnitude  (positive)
    fraction : float   – Kelly fraction (0.5 = half-Kelly)

    Returns float in [0, 1].
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1 - prob_win
    k = (prob_win * b - q) / b
    return max(0.0, min(k, 1.0)) * fraction


# ───────────────────────────────────────────────────────────────────────────
# Backtest engine  (simple pandas – swap for VectorBT / NautilusTrader later)
# ───────────────────────────────────────────────────────────────────────────

def run_backtest(close, signals, equity=10_000.0, fee_rate=0.001):
    """Vectorised backtest on categorical signals.

    Parameters
    ----------
    close    : pd.Series  – price series (aligned with signals)
    signals  : pd.Series  – +1 long, -1 short, 0 flat
    equity   : float      – starting capital
    fee_rate : float      – one-way fee

    Returns dict with metrics and equity curve.
    """
    returns = close.pct_change().fillna(0)

    # position acts on the NEXT bar
    position = signals.shift(1).fillna(0)

    # cost on every position change
    trades = position.diff().fillna(0).abs()
    costs = trades * fee_rate

    strat_ret = position * returns - costs
    eq_curve = equity * (1 + strat_ret).cumprod()

    # ── metrics ──────────────────────────────────────────────────────────
    total_ret = eq_curve.iloc[-1] / equity - 1
    sharpe = (strat_ret.mean() / strat_ret.std() * np.sqrt(252 * 24)
              if strat_ret.std() > 0 else 0.0)
    peak = eq_curve.cummax()
    max_dd = ((eq_curve - peak) / peak).min()
    n_trades = int(trades.gt(0).sum()) // 2
    active = strat_ret[position != 0]
    win_rate = float(active.gt(0).mean()) if len(active) > 0 else 0.0

    return {
        "total_return": round(total_ret, 4),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_dd, 4),
        "total_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "equity_curve": eq_curve,
    }
