"""Bet sizing and backtesting."""

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

try:  # pragma: no cover - optional dependency exercised in integration tests
    import vectorbt as vbt
    from vectorbt.portfolio.enums import Direction, SizeType
except ImportError:  # pragma: no cover
    vbt = None
    Direction = None
    SizeType = None


_SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


# ───────────────────────────────────────────────────────────────────────────
# Kelly criterion and Statistical Robustness
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


def probabilistic_sharpe_ratio(observed_sr, skewness, kurtosis, n_observations,
                              benchmark_sr=0.0):
    """Calculate the Probabilistic Sharpe Ratio (PSR).

    PSR adjusts the Sharpe Ratio for non-normality (skewness and kurtosis)
    and sample length. It represents the probability that the true SR
    is greater than the benchmark.
    """
    if n_observations <= 2:
        return 0.5
    
    # Calculate the standard deviation of the Sharpe Ratio
    # Note: observed_sr should be non-annualized here for the formula, 
    # but often used with annualized SR and adjusted n. 
    # Here we use the standard formula for SR estimation error.
    variance_sr = (1.0 - skewness * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr**2) / (n_observations - 1.0)
    sr_std = np.sqrt(max(0.0, variance_sr))
    
    if sr_std <= 0:
        return 1.0 if observed_sr > benchmark_sr else 0.0
        
    return float(norm.cdf((observed_sr - benchmark_sr) / sr_std))


def deflated_sharpe_ratio(observed_sr, sr_variance, n_trials, skewness, kurtosis,
                          n_observations):
    """Calculate the Deflated Sharpe Ratio (DSR).

    DSR accounts for selection bias (multiple testing) by using the 
    Expected Maximum Sharpe Ratio under the null hypothesis as the benchmark.
    """
    if n_trials <= 1:
        return probabilistic_sharpe_ratio(observed_sr, skewness, kurtosis, n_observations)

    # Euler-Mascheroni constant
    emc = 0.57721566490153286
    
    # Expected maximum Sharpe Ratio under null (False Strategy Theorem)
    # n_trials should ideally be the number of independent trials.
    expected_max_sr = np.sqrt(sr_variance) * (
        (1.0 - emc) * norm.ppf(1.0 - 1.0 / n_trials) +
        emc * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    )
    
    return probabilistic_sharpe_ratio(
        observed_sr, skewness, kurtosis, n_observations, 
        benchmark_sr=expected_max_sr
    )


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


def _round_down_to_step(values, step):
    if step is None or step <= 0:
        return values
    return np.floor(values / step) * step


def _normalize_price_series(price, tick_size=None):
    series = pd.Series(price, copy=False).astype(float)
    if tick_size is not None and tick_size > 0:
        series = pd.Series(_round_down_to_step(series.to_numpy(), tick_size), index=series.index, dtype=float)
    return series.replace(0.0, np.nan).ffill().bfill()


def _normalize_position_targets(signals, leverage=1.0, allow_short=True):
    target = pd.Series(signals, copy=False).astype(float).fillna(0.0) * float(leverage)
    if allow_short:
        return target.clip(-abs(float(leverage)), abs(float(leverage)))
    return target.clip(0.0, abs(float(leverage)))


def _vectorbt_trade_ledger(portfolio, index):
    readable = portfolio.trades.records_readable
    if readable.empty:
        return pd.DataFrame(columns=["entry_time", "exit_time", "direction", "bars", "entry_price", "exit_price", "return_pct"])

    entry_time = pd.to_datetime(readable["Entry Timestamp"], utc=True)
    exit_time = pd.to_datetime(readable["Exit Timestamp"], utc=True)
    entry_loc = index.get_indexer(entry_time)
    exit_loc = index.get_indexer(exit_time)
    bars = np.where((entry_loc >= 0) & (exit_loc >= 0), exit_loc - entry_loc + 1, np.nan)

    ledger = pd.DataFrame(
        {
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": readable["Direction"].map({"Long": 1, "Short": -1}).fillna(0).astype(int),
            "bars": pd.Series(bars, dtype="float").fillna(0).astype(int),
            "entry_price": pd.to_numeric(readable["Avg Entry Price"], errors="coerce"),
            "exit_price": pd.to_numeric(readable["Avg Exit Price"], errors="coerce"),
            "return_pct": pd.to_numeric(readable["Return"], errors="coerce"),
        }
    )
    return ledger


def _compute_funding_cash(position, funding_rates, equity_curve):
    if funding_rates is None:
        return pd.Series(0.0, index=position.index, dtype=float)

    funding = pd.Series(funding_rates, index=position.index).reindex(position.index).fillna(0.0).astype(float)
    prev_equity = equity_curve.shift(1).fillna(equity_curve.iloc[0])
    return prev_equity * (-position.astype(float) * funding)


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


def _build_trade_ledger(strat_ret, position, execution_series):
    trades = []
    current_sign = 0.0
    entry_time = None
    entry_price = None
    segment_returns = []
    previous_timestamp = None

    for timestamp in strat_ret.index:
        sign_now = float(np.sign(position.loc[timestamp]))
        if current_sign == 0.0:
            if sign_now != 0.0:
                current_sign = sign_now
                entry_time = timestamp
                entry_price = float(execution_series.loc[timestamp])
                segment_returns = [float(strat_ret.loc[timestamp])]
        elif sign_now == current_sign:
            segment_returns.append(float(strat_ret.loc[timestamp]))
        else:
            if segment_returns:
                trade_return = float(np.prod(1.0 + np.asarray(segment_returns, dtype=float)) - 1.0)
                exit_time = previous_timestamp if previous_timestamp is not None else timestamp
                exit_price = float(execution_series.loc[exit_time])
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "direction": int(np.sign(current_sign)),
                        "bars": int(len(segment_returns)),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return_pct": trade_return,
                    }
                )

            if sign_now != 0.0:
                current_sign = sign_now
                entry_time = timestamp
                entry_price = float(execution_series.loc[timestamp])
                segment_returns = [float(strat_ret.loc[timestamp])]
            else:
                current_sign = 0.0
                entry_time = None
                entry_price = None
                segment_returns = []

        previous_timestamp = timestamp

    if current_sign != 0.0 and segment_returns:
        trade_return = float(np.prod(1.0 + np.asarray(segment_returns, dtype=float)) - 1.0)
        exit_time = strat_ret.index[-1]
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "direction": int(np.sign(current_sign)),
                "bars": int(len(segment_returns)),
                "entry_price": entry_price,
                "exit_price": float(execution_series.loc[exit_time]),
                "return_pct": trade_return,
            }
        )

    if not trades:
        return pd.DataFrame(columns=["entry_time", "exit_time", "direction", "bars", "entry_price", "exit_price", "return_pct"])
    return pd.DataFrame(trades)


def _summarize_backtest(equity_curve, strat_ret, position, execution_series, equity,
                        fees_paid, slippage_paid, signal_delay_bars, trade_ledger,
                        funding_pnl=0.0, engine="pandas"):
    prev_equity = equity_curve.shift(1).fillna(equity)
    pnl = equity_curve - prev_equity

    total_ret = equity_curve.iloc[-1] / equity - 1
    periods_per_year = _infer_periods_per_year(equity_curve.index)
    annualization = np.sqrt(periods_per_year) if periods_per_year > 0 else 0.0
    volatility = strat_ret.std()
    sharpe = (strat_ret.mean() / volatility * annualization
              if volatility > 0 and annualization > 0 else 0.0)
    
    # Statistical moments for DSR/PSR
    sk = float(skew(strat_ret)) if len(strat_ret) > 2 else 0.0
    kurt = float(kurtosis(strat_ret)) if len(strat_ret) > 2 else 0.0
    psr = probabilistic_sharpe_ratio(sharpe, sk, kurt, len(strat_ret))

    downside = strat_ret.where(strat_ret < 0, 0.0)
    downside_vol = downside.std()
    sortino = (strat_ret.mean() / downside_vol * annualization
               if downside_vol > 0 and annualization > 0 else 0.0)
    peak = equity_curve.cummax()
    max_dd = ((equity_curve - peak) / peak).min()
    max_dd_amount = abs((equity_curve - peak).min())
    max_dd_bars, max_dd_duration = _max_drawdown_duration(equity_curve, peak)

    sign_changed = np.sign(position) != np.sign(position.shift(1).fillna(0.0))
    opened_trade = position.ne(0.0) & (position.shift(1).fillna(0.0).eq(0.0) | sign_changed)
    n_trades = int(opened_trade.sum())
    active_mask = position.abs() > 1e-12
    active = strat_ret[active_mask]
    active_pnl = pnl[active_mask]
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
    avg_position_size = float(position.abs().mean()) if len(position) > 0 else 0.0
    total_turnover = float(position.diff().abs().fillna(position.abs()).sum()) if len(position) > 0 else 0.0

    trade_returns = trade_ledger["return_pct"] if not trade_ledger.empty else pd.Series(dtype=float)
    trade_winners = trade_returns[trade_returns > 0]
    trade_losers = trade_returns[trade_returns < 0]
    trade_profit_factor = _safe_ratio(float(trade_winners.sum()), abs(float(trade_losers.sum())))
    trade_win_rate = float(trade_returns.gt(0).mean()) if len(trade_returns) > 0 else 0.0
    avg_trade_return_pct = float(trade_returns.mean()) if len(trade_returns) > 0 else 0.0
    avg_trade_bars = float(trade_ledger["bars"].mean()) if not trade_ledger.empty else 0.0

    elapsed_years = 0.0
    if len(equity_curve.index) > 1:
        elapsed_years = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / _SECONDS_PER_YEAR
    cagr = ((equity_curve.iloc[-1] / equity) ** (1 / elapsed_years) - 1
            if elapsed_years > 0 and equity_curve.iloc[-1] > 0 else 0.0)
    calmar = _safe_ratio(cagr, abs(max_dd))

    funding_paid = max(-float(funding_pnl), 0.0)
    funding_received = max(float(funding_pnl), 0.0)

    return {
        "engine": engine,
        "starting_equity": _round_metric(equity, 2),
        "ending_equity": _round_metric(equity_curve.iloc[-1], 2),
        "net_profit": _round_metric(equity_curve.iloc[-1] - equity, 2),
        "net_profit_pct": _round_metric(total_ret, 4),
        "gross_profit": _round_metric(gross_profit, 2),
        "gross_loss": _round_metric(gross_loss, 2),
        "fees_paid": _round_metric(fees_paid, 2),
        "slippage_paid": _round_metric(slippage_paid, 2),
        "funding_pnl": _round_metric(funding_pnl, 2),
        "funding_paid": _round_metric(funding_paid, 2),
        "funding_received": _round_metric(funding_received, 2),
        "total_return": _round_metric(total_ret, 4),
        "cagr": _round_metric(cagr, 4),
        "sharpe_ratio": _round_metric(sharpe, 2),
        "probabilistic_sharpe_ratio": _round_metric(psr, 4),
        "skewness": _round_metric(sk, 4),
        "kurtosis": _round_metric(kurt, 4),
        "sortino_ratio": _round_metric(sortino, 2),
        "calmar_ratio": _round_metric(calmar, 2),
        "annualized_volatility": _round_metric(volatility * annualization, 4),
        "max_drawdown": _round_metric(max_dd, 4),
        "max_drawdown_amount": _round_metric(max_dd_amount, 2),
        "max_drawdown_duration": max_dd_duration,
        "max_drawdown_duration_bars": max_dd_bars,
        "exposure_rate": _round_metric(exposure_rate, 4),
        "average_position_size": _round_metric(avg_position_size, 4),
        "signal_delay_bars": signal_delay_bars,
        "total_turnover": _round_metric(total_turnover, 4),
        "profit_factor": _round_metric(profit_factor, 2),
        "avg_win": _round_metric(avg_win, 2),
        "avg_loss": _round_metric(avg_loss, 2),
        "expectancy": _round_metric(expectancy, 2),
        "expectancy_pct": _round_metric(expectancy_pct, 6),
        "total_trades": n_trades,
        "win_rate": _round_metric(win_rate, 4),
        "active_bar_win_rate": _round_metric(win_rate, 4),
        "closed_trades": int(len(trade_ledger)),
        "trade_win_rate": _round_metric(trade_win_rate, 4),
        "avg_trade_return_pct": _round_metric(avg_trade_return_pct, 6),
        "avg_trade_bars": _round_metric(avg_trade_bars, 2),
        "trade_profit_factor": _round_metric(trade_profit_factor, 2),
        "trade_ledger": trade_ledger,
        "equity_curve": equity_curve,
    }


def _run_vectorbt_backtest(close, position, equity, fee_rate, slippage_rate,
                           execution_prices, signal_delay_bars, allow_short,
                           symbol_filters=None, funding_rates=None):
    if vbt is None or Direction is None or SizeType is None:
        raise ImportError("vectorbt is not installed")

    symbol_filters = dict(symbol_filters or {})
    tick_size = symbol_filters.get("tick_size")
    step_size = symbol_filters.get("step_size")
    max_size = symbol_filters.get("max_qty")
    min_notional = symbol_filters.get("min_notional")

    valuation_series = _normalize_price_series(close, tick_size=tick_size)
    execution_series = _normalize_price_series(execution_prices if execution_prices is not None else close, tick_size=tick_size)
    min_size = None
    if min_notional is not None and float(min_notional) > 0:
        min_size = (float(min_notional) / execution_series.replace(0.0, np.nan)).fillna(0.0)

    portfolio = vbt.Portfolio.from_orders(
        close=valuation_series,
        size=position,
        size_type=SizeType.TargetPercent,
        direction=Direction.Both if allow_short else Direction.LongOnly,
        price=execution_series,
        fees=fee_rate,
        slippage=slippage_rate,
        min_size=min_size,
        max_size=max_size,
        size_granularity=step_size,
        init_cash=equity,
        freq=valuation_series.index.to_series().diff().median(),
    )

    base_equity = pd.Series(portfolio.value(), index=valuation_series.index, dtype=float)
    funding_cash = _compute_funding_cash(position, funding_rates, base_equity)
    adjusted_equity = base_equity + funding_cash.cumsum()
    adjusted_returns = adjusted_equity.pct_change().fillna(0.0)
    trade_ledger = _vectorbt_trade_ledger(portfolio, valuation_series.index)
    fees_paid = float(portfolio.orders.records_readable["Fees"].sum()) if not portfolio.orders.records_readable.empty else 0.0
    slippage_paid = float((adjusted_equity.shift(1).fillna(equity) * position.diff().abs().fillna(position.abs()) * slippage_rate).sum())

    return _summarize_backtest(
        equity_curve=adjusted_equity,
        strat_ret=adjusted_returns,
        position=position,
        execution_series=execution_series,
        equity=equity,
        fees_paid=fees_paid,
        slippage_paid=slippage_paid,
        signal_delay_bars=signal_delay_bars,
        trade_ledger=trade_ledger,
        funding_pnl=float(funding_cash.sum()),
        engine="vectorbt",
    )


def _run_pandas_backtest(close, position, equity, fee_rate, slippage_rate,
                         execution_prices, signal_delay_bars, funding_rates=None):
    valuation_series = pd.Series(close, copy=False).astype(float)
    execution_series = valuation_series if execution_prices is None else pd.Series(execution_prices, index=valuation_series.index).reindex(valuation_series.index).astype(float)
    returns = valuation_series.pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    fees = turnover * fee_rate
    slippage = turnover * slippage_rate
    funding_returns = pd.Series(0.0, index=position.index, dtype=float)
    if funding_rates is not None:
        funding_returns = -position * pd.Series(funding_rates, index=position.index).reindex(position.index).fillna(0.0)

    strat_ret = position * returns + funding_returns - fees - slippage
    equity_curve = equity * (1.0 + strat_ret).cumprod()
    trade_ledger = _build_trade_ledger(strat_ret, position, execution_series)
    prev_equity = equity_curve.shift(1).fillna(equity)
    return _summarize_backtest(
        equity_curve=equity_curve,
        strat_ret=strat_ret,
        position=position,
        execution_series=execution_series,
        equity=equity,
        fees_paid=float((prev_equity * fees).sum()),
        slippage_paid=float((prev_equity * slippage).sum()),
        signal_delay_bars=signal_delay_bars,
        trade_ledger=trade_ledger,
        funding_pnl=float((prev_equity * funding_returns).sum()),
        engine="pandas",
    )


# ───────────────────────────────────────────────────────────────────────────
# Backtest engine adapter  (VectorBT first, pandas fallback)
# ───────────────────────────────────────────────────────────────────────────

def run_backtest(close, signals, equity=10_000.0, fee_rate=0.001, slippage_rate=0.0,
                 execution_prices=None, signal_delay_bars=1, engine="vectorbt",
                 market="spot", leverage=1.0, allow_short=None, symbol_filters=None,
                 funding_rates=None, volume=None, liquidity_param=0.0):
    """Run a backtest through the configured execution adapter.

    Signals are shifted forward by *signal_delay_bars* before being applied to 
    execution prices. A delay of 1 means a signal generated at the close of 
    bar T is executed at the close (or open) of bar T+1.

    Market impact is optionally modeled if *volume* and *liquidity_param* are
    provided. Effective slippage increases as position size increases relative
    to the average volume.

    Parameters
    ----------
    close            : pd.Series  – mark-to-market or valuation price series
    signals          : pd.Series  – target portfolio weights before execution delay
    equity           : float      – starting capital
    fee_rate         : float      – one-way fee
    slippage_rate    : float      – one-way baseline slippage estimate
    execution_prices : pd.Series or None – optional execution price series, e.g. next-bar open
    signal_delay_bars: int        – bars to delay signal application before execution
    engine           : str        – "vectorbt" (default) or "pandas"
    market           : str        – "spot", "um_futures", or "cm_futures"
    leverage         : float      – target leverage multiplier
    allow_short      : bool or None – whether short positions are permitted
    symbol_filters   : dict or None – exchange filters (min_qty, step_size, etc.)
    funding_rates    : pd.Series or None – funding rate series for futures
    volume           : pd.Series or None – trade volume for market impact modeling
    liquidity_param  : float      – sensitivity of slippage to trade size (0 = none)
    """
    close = pd.Series(close, copy=False).astype(float)
    signal_series = pd.Series(signals, index=close.index).reindex(close.index).fillna(0.0).astype(float)
    signal_delay_bars = max(0, int(signal_delay_bars))

    if signal_delay_bars == 0:
        import warnings
        warnings.warn(
            "signal_delay_bars=0 detected. This assumes instantaneous execution at "
            "the signal bar's close/valuation price, which is usually a lookahead leak.",
            UserWarning,
            stacklevel=2
        )

    allow_short = (market or "spot") != "spot" if allow_short is None else bool(allow_short)
    position = _normalize_position_targets(
        signal_series.shift(signal_delay_bars).fillna(0.0),
        leverage=leverage,
        allow_short=allow_short,
    )

    # Effective slippage modeling (Market Impact)
    effective_slippage = float(slippage_rate)
    if liquidity_param > 0 and volume is not None:
        avg_vol = volume.rolling(24).mean().reindex(close.index).ffill().fillna(volume.mean())
        # Turnover is the change in target weights * equity
        turnover_nominal = position.diff().abs().fillna(position.abs()) * float(equity)
        # Ratio of trade size to average volume
        impact_ratio = (turnover_nominal / (avg_vol * close).replace(0, np.nan)).fillna(0.0)
        # Effective slippage = baseline + liquidity_param * impact_ratio
        # We take the mean impact for simplicity in the flat-rate engine, 
        # but institutional models apply this per-trade.
        effective_slippage += float(liquidity_param * impact_ratio.mean())

    selected_engine = (engine or "vectorbt").lower()
    if selected_engine == "vectorbt":
        try:
            return _run_vectorbt_backtest(
                close=close,
                position=position,
                equity=equity,
                fee_rate=fee_rate,
                slippage_rate=effective_slippage,
                execution_prices=execution_prices,
                signal_delay_bars=signal_delay_bars,
                allow_short=allow_short,
                symbol_filters=symbol_filters,
                funding_rates=funding_rates,
            )
        except ImportError:
            selected_engine = "pandas"

    return _run_pandas_backtest(
        close=close,
        position=position,
        equity=equity,
        fee_rate=fee_rate,
        slippage_rate=effective_slippage,
        execution_prices=execution_prices,
        signal_delay_bars=signal_delay_bars,
        funding_rates=funding_rates,
    )
