"""Bet sizing and backtesting."""

import numpy as np
import pandas as pd

from .slippage import FlatSlippageModel, OrderBookImpactModel, SquareRootImpactModel

try:  # pragma: no cover - optional dependency exercised in integration tests
    import vectorbt as vbt
    from vectorbt.portfolio.enums import Direction, SizeType
except ImportError:  # pragma: no cover
    vbt = None
    Direction = None
    SizeType = None


_SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
_DEFAULT_SIGNIFICANCE_CONFIG = {
    "enabled": True,
    "method": "stationary_bootstrap",
    "bootstrap_samples": 500,
    "confidence_level": 0.95,
    "mean_block_length": None,
    "random_state": 42,
    "min_observations": 8,
}


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


def _align_numeric_series(values, index, fill_value=0.0):
    if values is None:
        return pd.Series(fill_value, index=index, dtype=float)
    if isinstance(values, pd.Series):
        series = values.reindex(index)
    else:
        series = pd.Series(values, index=index)
    return pd.to_numeric(series, errors="coerce").fillna(fill_value).astype(float)


def _align_numeric_frame(frame, index):
    if frame is None:
        return None
    aligned = frame.reindex(index).copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame, index=index)
    for column in aligned.columns:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
    return aligned


def _resolve_slippage_model(slippage_model, slippage_rate):
    if slippage_model is None:
        return FlatSlippageModel(rate=slippage_rate)

    if isinstance(slippage_model, str):
        aliases = {
            "flat": "flat",
            "sqrt-impact": "sqrt_impact",
            "sqrt_impact": "sqrt_impact",
            "square-root-impact": "sqrt_impact",
            "square_root_impact": "sqrt_impact",
            "orderbook": "orderbook",
            "order_book": "orderbook",
        }
        resolved_name = aliases.get(slippage_model.strip().lower(), slippage_model.strip().lower())
        if resolved_name == "flat":
            return FlatSlippageModel(rate=slippage_rate)
        if resolved_name == "sqrt_impact":
            return SquareRootImpactModel()
        if resolved_name == "orderbook":
            return OrderBookImpactModel()
        raise ValueError("Unsupported slippage_model. Choose from ['flat', 'sqrt_impact', 'orderbook']")

    if not hasattr(slippage_model, "estimate"):
        raise TypeError("slippage_model must be None, a supported string alias, or implement estimate(...)")
    return slippage_model


def _estimate_slippage_rates(position, equity, valuation_series, execution_series,
                             slippage_rate, slippage_model=None, volume=None,
                             funding_rates=None, orderbook_depth=None):
    position = pd.Series(position, index=valuation_series.index, copy=False).reindex(valuation_series.index).fillna(0.0).astype(float)
    turnover = position.diff().abs().fillna(position.abs()).astype(float)
    model = _resolve_slippage_model(slippage_model, slippage_rate)

    if not isinstance(model, FlatSlippageModel) and volume is None:
        raise ValueError("volume is required when using a non-flat slippage model")

    aligned_volume = _align_numeric_series(volume, execution_series.index, fill_value=0.0).clip(lower=0.0)
    aligned_funding = _align_numeric_series(funding_rates, valuation_series.index, fill_value=0.0)
    gross_returns = position * valuation_series.pct_change().fillna(0.0) - position * aligned_funding
    gross_equity = float(equity) * (1.0 + gross_returns).cumprod()
    prev_equity = gross_equity.shift(1).fillna(float(equity))
    trade_notional = prev_equity * turnover
    volatility_window = max(1, int(getattr(model, "adv_window", 14)))
    volatility = execution_series.pct_change().rolling(volatility_window).std()

    slippage_rates = model.estimate(
        trade_notional=trade_notional,
        volume=aligned_volume,
        volatility=volatility,
        price=execution_series,
        orderbook_depth=_align_numeric_frame(orderbook_depth, execution_series.index),
    )
    slippage_rates = _align_numeric_series(slippage_rates, execution_series.index, fill_value=0.0).clip(lower=0.0)
    return slippage_rates.where(turnover > 0.0, 0.0), turnover


def _annualized_sharpe(returns, annualization):
    if annualization <= 0 or len(returns) < 2:
        return 0.0
    volatility = float(np.std(returns, ddof=1))
    if volatility <= 0:
        return 0.0
    return float(np.mean(returns) / volatility * annualization)


def _annualized_sortino(returns, annualization):
    if annualization <= 0 or len(returns) < 2:
        return 0.0
    downside = np.where(np.asarray(returns, dtype=float) < 0.0, returns, 0.0)
    downside_vol = float(np.std(downside, ddof=1))
    if downside_vol <= 0:
        return 0.0
    return float(np.mean(returns) / downside_vol * annualization)


def _compute_total_return(equity_curve, starting_equity):
    if len(equity_curve) == 0 or starting_equity <= 0:
        return 0.0
    return float(equity_curve[-1] / starting_equity - 1.0)


def _compute_max_drawdown_from_equity(equity_curve):
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    return float(np.min((equity_curve - peak) / peak))


def _compute_cagr_from_equity(equity_curve, starting_equity, elapsed_years):
    if len(equity_curve) == 0 or elapsed_years <= 0 or starting_equity <= 0 or equity_curve[-1] <= 0:
        return 0.0
    return float((equity_curve[-1] / starting_equity) ** (1.0 / elapsed_years) - 1.0)


def _compute_calmar(cagr, max_drawdown):
    return float(_safe_ratio(cagr, abs(max_drawdown)))


def _resolve_significance_config(significance):
    if significance is None:
        return dict(_DEFAULT_SIGNIFICANCE_CONFIG)
    if isinstance(significance, bool):
        return {**_DEFAULT_SIGNIFICANCE_CONFIG, "enabled": significance}
    if not isinstance(significance, dict):
        raise TypeError("significance must be None, a bool, or a dict")
    return {**_DEFAULT_SIGNIFICANCE_CONFIG, **dict(significance)}


def _default_mean_block_length(sample_size):
    if sample_size <= 1:
        return 1
    return max(2, min(sample_size, int(round(sample_size ** (1.0 / 3.0)))))


def _stationary_bootstrap_indices(sample_size, mean_block_length, rng):
    if sample_size <= 0:
        return np.array([], dtype=int)

    mean_block_length = max(1, int(mean_block_length))
    restart_probability = min(1.0, 1.0 / float(mean_block_length))
    indices = np.empty(sample_size, dtype=int)
    indices[0] = int(rng.integers(0, sample_size))

    for loc in range(1, sample_size):
        if rng.random() < restart_probability:
            indices[loc] = int(rng.integers(0, sample_size))
        else:
            indices[loc] = (indices[loc - 1] + 1) % sample_size

    return indices


def _build_bootstrap_interval(samples, confidence_level, digits=4):
    finite = np.asarray(samples, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None

    alpha = (1.0 - float(confidence_level)) / 2.0
    lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
    return {
        "lower": _round_metric(float(lower), digits),
        "upper": _round_metric(float(upper), digits),
        "confidence_level": float(confidence_level),
    }


def _centered_bootstrap_p_value(samples, observed_estimate, threshold):
    finite = np.asarray(samples, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or observed_estimate is None or threshold is None:
        return None

    observed_estimate = float(observed_estimate)
    threshold = float(threshold)
    deviations = finite - observed_estimate
    test_statistic = observed_estimate - threshold
    equal_mask = np.isclose(deviations, test_statistic, rtol=1e-9, atol=1e-12)
    tail_probability = float(np.mean(deviations > test_statistic) + 0.5 * np.mean(equal_mask))
    return max(0.0, min(1.0, tail_probability))


def _align_benchmark_returns(benchmark_returns, index):
    if benchmark_returns is None:
        return None

    if isinstance(benchmark_returns, pd.Series):
        aligned = pd.Series(benchmark_returns, copy=False).reindex(index)
    else:
        aligned = pd.Series(benchmark_returns, index=index)

    if aligned.isna().any():
        raise ValueError("benchmark_returns must cover every backtest timestamp")

    return aligned.astype(float)


def _compute_significance_metrics(strat_ret, equity, periods_per_year, elapsed_years,
                                  sharpe, sortino, calmar, total_ret, max_dd,
                                  significance=None, benchmark_returns=None,
                                  benchmark_sharpe=None):
    config = _resolve_significance_config(significance)
    payload = {
        "enabled": bool(config.get("enabled", True)),
        "method": str(config.get("method", "stationary_bootstrap")),
        "bootstrap_samples": int(config.get("bootstrap_samples", 500)),
        "confidence_level": float(config.get("confidence_level", 0.95)),
        "mean_block_length": None,
        "random_state": config.get("random_state"),
        "benchmark_sharpe_ratio": None,
        "metrics": {},
    }
    if not payload["enabled"]:
        payload["reason"] = "disabled"
        return payload

    if payload["bootstrap_samples"] <= 0:
        raise ValueError("bootstrap_samples must be greater than zero")
    if not 0.0 < payload["confidence_level"] < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    method = payload["method"].lower()
    if method not in {"stationary", "stationary_bootstrap"}:
        raise ValueError(f"unsupported significance method: {payload['method']}")
    payload["method"] = "stationary_bootstrap"

    strategy_returns = pd.Series(strat_ret, copy=False).astype(float)
    benchmark_series = _align_benchmark_returns(benchmark_returns, strategy_returns.index)

    finite_mask = np.isfinite(strategy_returns.to_numpy())
    if benchmark_series is not None:
        finite_mask &= np.isfinite(benchmark_series.to_numpy())

    strategy_returns = strategy_returns.loc[finite_mask]
    benchmark_series = benchmark_series.loc[finite_mask] if benchmark_series is not None else None

    min_observations = max(2, int(config.get("min_observations", 8)))
    if len(strategy_returns) < min_observations:
        payload["enabled"] = False
        payload["reason"] = "insufficient_observations"
        return payload

    mean_block_length = config.get("mean_block_length")
    if mean_block_length is None:
        mean_block_length = _default_mean_block_length(len(strategy_returns))
    mean_block_length = max(1, int(mean_block_length))
    payload["mean_block_length"] = mean_block_length

    annualization = np.sqrt(periods_per_year) if periods_per_year > 0 else 0.0
    rng = np.random.default_rng(config.get("random_state"))
    strategy_values = strategy_returns.to_numpy(dtype=float)
    benchmark_values = benchmark_series.to_numpy(dtype=float) if benchmark_series is not None else None

    bootstrap_metrics = {
        "sharpe_ratio": np.empty(payload["bootstrap_samples"], dtype=float),
        "sortino_ratio": np.empty(payload["bootstrap_samples"], dtype=float),
        "calmar_ratio": np.empty(payload["bootstrap_samples"], dtype=float),
        "net_profit_pct": np.empty(payload["bootstrap_samples"], dtype=float),
        "max_drawdown": np.empty(payload["bootstrap_samples"], dtype=float),
    }
    benchmark_sharpes = np.empty(payload["bootstrap_samples"], dtype=float) if benchmark_values is not None else None

    for sample_idx in range(payload["bootstrap_samples"]):
        sampled_idx = _stationary_bootstrap_indices(len(strategy_values), mean_block_length, rng)
        sampled_returns = strategy_values[sampled_idx]
        sampled_equity = equity * np.cumprod(1.0 + sampled_returns)
        sampled_total_ret = _compute_total_return(sampled_equity, equity)
        sampled_max_dd = _compute_max_drawdown_from_equity(sampled_equity)
        sampled_cagr = _compute_cagr_from_equity(sampled_equity, equity, elapsed_years)

        bootstrap_metrics["sharpe_ratio"][sample_idx] = _annualized_sharpe(sampled_returns, annualization)
        bootstrap_metrics["sortino_ratio"][sample_idx] = _annualized_sortino(sampled_returns, annualization)
        bootstrap_metrics["calmar_ratio"][sample_idx] = _compute_calmar(sampled_cagr, sampled_max_dd)
        bootstrap_metrics["net_profit_pct"][sample_idx] = sampled_total_ret
        bootstrap_metrics["max_drawdown"][sample_idx] = sampled_max_dd

        if benchmark_sharpes is not None:
            benchmark_sharpes[sample_idx] = _annualized_sharpe(benchmark_values[sampled_idx], annualization)

    observed_metrics = {
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "net_profit_pct": float(total_ret),
        "max_drawdown": float(max_dd),
    }

    observed_benchmark_sharpe = None
    if benchmark_series is not None:
        observed_benchmark_sharpe = _annualized_sharpe(benchmark_values, annualization)
        payload["benchmark_sharpe_ratio"] = _round_metric(observed_benchmark_sharpe, 4)
    elif benchmark_sharpe is not None:
        observed_benchmark_sharpe = float(benchmark_sharpe)
        payload["benchmark_sharpe_ratio"] = _round_metric(observed_benchmark_sharpe, 4)

    for metric_name, point_estimate in observed_metrics.items():
        metric_payload = {
            "point_estimate": _round_metric(point_estimate, 4),
            "confidence_interval": _build_bootstrap_interval(
                bootstrap_metrics[metric_name],
                payload["confidence_level"],
                digits=4,
            ),
        }
        if metric_name == "sharpe_ratio":
            metric_payload["p_value_gt_zero"] = _round_metric(
                _centered_bootstrap_p_value(bootstrap_metrics[metric_name], point_estimate, 0.0),
                6,
            )
            if benchmark_sharpes is not None:
                observed_diff = point_estimate - float(observed_benchmark_sharpe)
                bootstrap_diffs = bootstrap_metrics[metric_name] - benchmark_sharpes
                metric_payload["p_value_gt_benchmark"] = _round_metric(
                    _centered_bootstrap_p_value(bootstrap_diffs, observed_diff, 0.0),
                    6,
                )
            elif payload["benchmark_sharpe_ratio"] is not None:
                metric_payload["p_value_gt_benchmark"] = _round_metric(
                    _centered_bootstrap_p_value(
                        bootstrap_metrics[metric_name],
                        point_estimate,
                        float(payload["benchmark_sharpe_ratio"]),
                    ),
                    6,
                )
        payload["metrics"][metric_name] = metric_payload

    return payload


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
                        funding_pnl=0.0, engine="pandas", significance=None,
                        benchmark_returns=None, benchmark_sharpe=None):
    prev_equity = equity_curve.shift(1).fillna(equity)
    pnl = equity_curve - prev_equity

    total_ret = equity_curve.iloc[-1] / equity - 1
    periods_per_year = _infer_periods_per_year(equity_curve.index)
    annualization = np.sqrt(periods_per_year) if periods_per_year > 0 else 0.0
    volatility = strat_ret.std()
    sharpe = _annualized_sharpe(strat_ret.to_numpy(dtype=float), annualization)
    sortino = _annualized_sortino(strat_ret.to_numpy(dtype=float), annualization)
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
    cagr = _compute_cagr_from_equity(equity_curve.to_numpy(dtype=float), equity, elapsed_years)
    calmar = _compute_calmar(cagr, max_dd)

    significance_metrics = _compute_significance_metrics(
        strat_ret=strat_ret,
        equity=equity,
        periods_per_year=periods_per_year,
        elapsed_years=elapsed_years,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        total_ret=total_ret,
        max_dd=max_dd,
        significance=significance,
        benchmark_returns=benchmark_returns,
        benchmark_sharpe=benchmark_sharpe,
    )

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
        "statistical_significance": significance_metrics,
        "trade_ledger": trade_ledger,
        "equity_curve": equity_curve,
    }


def _run_vectorbt_backtest(close, position, equity, fee_rate, slippage_rate,
                           execution_prices, signal_delay_bars, allow_short,
                           symbol_filters=None, funding_rates=None,
                           significance=None, benchmark_returns=None,
                           benchmark_sharpe=None, volume=None,
                           slippage_model=None, orderbook_depth=None):
    if vbt is None or Direction is None or SizeType is None:
        raise ImportError("vectorbt is not installed")

    symbol_filters = dict(symbol_filters or {})
    tick_size = symbol_filters.get("tick_size")
    step_size = symbol_filters.get("step_size")
    max_size = symbol_filters.get("max_qty")
    min_notional = symbol_filters.get("min_notional")

    valuation_series = _normalize_price_series(close, tick_size=tick_size)
    execution_series = _normalize_price_series(execution_prices if execution_prices is not None else close, tick_size=tick_size)
    slippage_rates, turnover = _estimate_slippage_rates(
        position=position,
        equity=equity,
        valuation_series=valuation_series,
        execution_series=execution_series,
        slippage_rate=slippage_rate,
        slippage_model=slippage_model,
        volume=volume,
        funding_rates=funding_rates,
        orderbook_depth=orderbook_depth,
    )
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
        slippage=slippage_rates,
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
    slippage_paid = float((adjusted_equity.shift(1).fillna(equity) * turnover * slippage_rates).sum())

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
        significance=significance,
        benchmark_returns=benchmark_returns,
        benchmark_sharpe=benchmark_sharpe,
    )


def _run_pandas_backtest(close, position, equity, fee_rate, slippage_rate,
                         execution_prices, signal_delay_bars, funding_rates=None,
                         significance=None, benchmark_returns=None,
                         benchmark_sharpe=None, volume=None,
                         slippage_model=None, orderbook_depth=None):
    valuation_series = pd.Series(close, copy=False).astype(float)
    execution_series = valuation_series if execution_prices is None else pd.Series(execution_prices, index=valuation_series.index).reindex(valuation_series.index).astype(float)
    returns = valuation_series.pct_change().fillna(0.0)
    slippage_rates, turnover = _estimate_slippage_rates(
        position=position,
        equity=equity,
        valuation_series=valuation_series,
        execution_series=execution_series,
        slippage_rate=slippage_rate,
        slippage_model=slippage_model,
        volume=volume,
        funding_rates=funding_rates,
        orderbook_depth=orderbook_depth,
    )
    fees = turnover * fee_rate
    slippage = turnover * slippage_rates
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
        significance=significance,
        benchmark_returns=benchmark_returns,
        benchmark_sharpe=benchmark_sharpe,
    )


# ───────────────────────────────────────────────────────────────────────────
# Backtest engine adapter  (VectorBT first, pandas fallback)
# ───────────────────────────────────────────────────────────────────────────

def run_backtest(close, signals, equity=10_000.0, fee_rate=0.001, slippage_rate=0.0,
                 execution_prices=None, signal_delay_bars=1, engine="vectorbt",
                 market="spot", leverage=1.0, allow_short=None, symbol_filters=None,
                 funding_rates=None, significance=None, benchmark_returns=None,
                 benchmark_sharpe=None, volume=None, slippage_model=None,
                 orderbook_depth=None):
    """Run a backtest through the configured execution adapter.

    Parameters
    ----------
    close            : pd.Series  – mark-to-market or valuation price series
    signals          : pd.Series  – target portfolio weights before execution delay
    equity           : float      – starting capital
    fee_rate         : float      – one-way fee
    slippage_rate    : float      – one-way flat slippage rate used by the legacy/default model
    execution_prices : pd.Series or None – optional execution price series, e.g. next-bar open
    signal_delay_bars: int        – bars to delay signal application before execution
    engine           : str        – "vectorbt" (default) or "pandas"
    market           : str        – "spot", "um_futures", or "cm_futures"
    leverage         : float      – exposure multiplier applied to target weights
    allow_short      : bool|None  – defaults to False for spot, True for futures
    symbol_filters   : dict|None  – Binance execution filters (tick size, lot size, min notional)
    funding_rates    : pd.Series|None – futures funding rates aligned to close index; applied on funding timestamps only
    significance     : bool|dict|None – stationary-bootstrap significance settings; enabled by default
    benchmark_returns: pd.Series|array|None – optional benchmark returns aligned to backtest index for Sharpe comparison
    benchmark_sharpe : float|None – optional benchmark Sharpe ratio threshold when no benchmark return series is supplied
    volume           : pd.Series|array|None – bar volume aligned to the backtest index; required for non-flat slippage models
    slippage_model   : str|object|None – one of {"flat", "sqrt_impact", "orderbook"} or a custom estimator implementing estimate(...)
    orderbook_depth  : pd.DataFrame|None – optional L2 depth frame for future order-book-aware slippage models

    Returns dict with metrics and equity curve.
    """
    close = pd.Series(close, copy=False).astype(float)
    signal_series = pd.Series(signals, index=close.index).reindex(close.index).fillna(0.0).astype(float)
    benchmark_returns = _align_benchmark_returns(benchmark_returns, close.index)
    signal_delay_bars = max(0, int(signal_delay_bars))
    allow_short = (market or "spot") != "spot" if allow_short is None else bool(allow_short)
    position = _normalize_position_targets(
        signal_series.shift(signal_delay_bars).fillna(0.0),
        leverage=leverage,
        allow_short=allow_short,
    )

    selected_engine = (engine or "vectorbt").lower()
    if selected_engine == "vectorbt":
        try:
            return _run_vectorbt_backtest(
                close=close,
                position=position,
                equity=equity,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                execution_prices=execution_prices,
                signal_delay_bars=signal_delay_bars,
                allow_short=allow_short,
                symbol_filters=symbol_filters,
                funding_rates=funding_rates,
                significance=significance,
                benchmark_returns=benchmark_returns,
                benchmark_sharpe=benchmark_sharpe,
                volume=volume,
                slippage_model=slippage_model,
                orderbook_depth=orderbook_depth,
            )
        except ImportError:
            selected_engine = "pandas"

    return _run_pandas_backtest(
        close=close,
        position=position,
        equity=equity,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        execution_prices=execution_prices,
        signal_delay_bars=signal_delay_bars,
        funding_rates=funding_rates,
        significance=significance,
        benchmark_returns=benchmark_returns,
        benchmark_sharpe=benchmark_sharpe,
        volume=volume,
        slippage_model=slippage_model,
        orderbook_depth=orderbook_depth,
    )
