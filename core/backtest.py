"""Bet sizing and backtesting."""

import numpy as np
import pandas as pd


_SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


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


def _infer_periods_per_year(index):
    if len(index) < 2:
        return 0.0

    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return 0.0

    seconds = deltas.median().total_seconds()
    if seconds <= 0:
        return 0.0

    return _SECONDS_PER_YEAR / seconds


def _round_metric(value, digits=4):
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return round(float(value), digits)
    return value


def _safe_ratio(numerator, denominator, default=0.0):
    if denominator == 0:
        if numerator > 0:
            return float("inf")
        return default
    return numerator / denominator


def _max_drawdown_duration(equity_curve, peak):
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        return 0, pd.Timedelta(0)

    underwater = equity_curve < peak
    max_bars = 0
    max_duration = pd.Timedelta(0)
    current_start = None
    current_bars = 0

    for timestamp, is_underwater in underwater.items():
        if is_underwater:
            if current_start is None:
                current_start = timestamp
                current_bars = 1
            else:
                current_bars += 1
            continue

        if current_start is not None:
            current_duration = timestamp - current_start
            if current_bars > max_bars:
                max_bars = current_bars
                max_duration = current_duration
            current_start = None
            current_bars = 0

    if current_start is not None:
        current_duration = equity_curve.index[-1] - current_start
        if current_bars > max_bars:
            max_bars = current_bars
            max_duration = current_duration

    return max_bars, max_duration


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
    prev_equity = eq_curve.shift(1).fillna(equity)
    pnl = eq_curve - prev_equity
    fees_paid = (prev_equity * costs).sum()

    # ── metrics ──────────────────────────────────────────────────────────
    total_ret = eq_curve.iloc[-1] / equity - 1
    periods_per_year = _infer_periods_per_year(close.index)
    annualization = np.sqrt(periods_per_year) if periods_per_year > 0 else 0.0
    volatility = strat_ret.std()
    sharpe = (strat_ret.mean() / volatility * annualization
              if volatility > 0 and annualization > 0 else 0.0)
    downside = strat_ret.where(strat_ret < 0, 0.0)
    downside_vol = downside.std()
    sortino = (strat_ret.mean() / downside_vol * annualization
               if downside_vol > 0 and annualization > 0 else 0.0)
    peak = eq_curve.cummax()
    max_dd = ((eq_curve - peak) / peak).min()
    max_dd_amount = abs((eq_curve - peak).min())
    max_dd_bars, max_dd_duration = _max_drawdown_duration(eq_curve, peak)
    n_trades = int(trades.gt(0).sum()) // 2
    active = strat_ret[position != 0]
    active_pnl = pnl[position != 0]
    winners = active_pnl[active_pnl > 0]
    losers = active_pnl[active_pnl < 0]
    gross_profit = winners.sum()
    gross_loss = abs(losers.sum())
    win_rate = float(active_pnl.gt(0).mean()) if len(active_pnl) > 0 else 0.0
    avg_win = winners.mean() if len(winners) > 0 else 0.0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.0
    expectancy = active_pnl.mean() if len(active_pnl) > 0 else 0.0
    expectancy_pct = active.mean() if len(active) > 0 else 0.0
    profit_factor = _safe_ratio(gross_profit, gross_loss)
    exposure_rate = float(position.ne(0).mean()) if len(position) > 0 else 0.0

    elapsed_years = 0.0
    if len(close.index) > 1:
        elapsed_years = (close.index[-1] - close.index[0]).total_seconds() / _SECONDS_PER_YEAR
    cagr = ((eq_curve.iloc[-1] / equity) ** (1 / elapsed_years) - 1
            if elapsed_years > 0 and eq_curve.iloc[-1] > 0 else 0.0)
    calmar = _safe_ratio(cagr, abs(max_dd))

    return {
        "starting_equity": _round_metric(equity, 2),
        "ending_equity": _round_metric(eq_curve.iloc[-1], 2),
        "net_profit": _round_metric(eq_curve.iloc[-1] - equity, 2),
        "net_profit_pct": _round_metric(total_ret, 4),
        "gross_profit": _round_metric(gross_profit, 2),
        "gross_loss": _round_metric(gross_loss, 2),
        "fees_paid": _round_metric(fees_paid, 2),
        "total_return": _round_metric(total_ret, 4),
        "cagr": _round_metric(cagr, 4),
        "sharpe_ratio": _round_metric(sharpe, 2),
        "sortino_ratio": _round_metric(sortino, 2),
        "calmar_ratio": _round_metric(calmar, 2),
        "annualized_volatility": _round_metric(volatility * annualization, 4),
        "max_drawdown": _round_metric(max_dd, 4),
        "max_drawdown_amount": _round_metric(max_dd_amount, 2),
        "max_drawdown_duration": max_dd_duration,
        "max_drawdown_duration_bars": max_dd_bars,
        "exposure_rate": _round_metric(exposure_rate, 4),
        "profit_factor": _round_metric(profit_factor, 2),
        "avg_win": _round_metric(avg_win, 2),
        "avg_loss": _round_metric(avg_loss, 2),
        "expectancy": _round_metric(expectancy, 2),
        "expectancy_pct": _round_metric(expectancy_pct, 6),
        "total_trades": n_trades,
        "win_rate": _round_metric(win_rate, 4),
        "equity_curve": eq_curve,
    }
