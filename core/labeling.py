"""Labeling (triple-barrier, fixed-horizon) and sampling (weights, bootstrap)."""

import warnings

import numpy as np
import pandas as pd


# ───────────────────────────────────────────────────────────────────────────
# Labeling
# ───────────────────────────────────────────────────────────────────────────

def triple_barrier_labels(close, volatility, high=None, low=None, pt_sl=(2.0, 2.0),
                          max_holding=10, min_return=0.0, cost_rate=0.0,
                          barrier_tie_break="sl",
                          entry_prices=None, start_offset=0,
                          missing_future_policy="drop",
                          volatility_fit_boundary=None,
                          volatility_window=None):
    """Apply triple-barrier labeling.

    Three concurrent barriers decide the label per bar:
      • profit-taking  (upper) = entry * (1 + pt_mult × vol)
      • stop-loss      (lower) = entry * (1 − sl_mult × vol)
      • vertical time  = max_holding bars

    Parameters
    ----------
    close : pd.Series      – close prices (datetime index)
    volatility : pd.Series  – per-bar volatility **in decimal** (e.g. 0.02 = 2 %)
    high, low : pd.Series or None – intrabar high/low used for path-aware barrier checks
    pt_sl : tuple(float,float) – (profit-taking multiplier, stop-loss multiplier)
    max_holding : int       – bars until vertical barrier
    min_return : float      – returns below this at the time barrier → label 0 (abstain)
    cost_rate : float       – round-trip cost buffer applied to label thresholds
    barrier_tie_break : str – when pt/sl hit in the same bar, choose "sl" or "pt"
    entry_prices : pd.Series or None – execution prices (e.g. open[T+delay]); when
        provided barriers are anchored to the actual fill price, not close[T].
    start_offset : int      – bars between signal bar and execution bar; the future
        window starts at this offset so label horizons align to real holding periods.

    Returns
    -------
    pd.DataFrame  index=t0, columns=[t1, label, barrier]
        label ∈ {-1, 0, +1};  barrier ∈ {"pt", "sl", "time"}
    """
    pt_m, sl_m = pt_sl
    start_offset = max(0, int(start_offset))
    missing_future_policy = str(missing_future_policy or "drop").strip().lower()
    if missing_future_policy != "drop":
        raise ValueError(f"Unsupported missing_future_policy={missing_future_policy!r}")
    high = close if high is None else high.reindex(close.index)
    low = close if low is None else low.reindex(close.index)
    if barrier_tie_break not in {"sl", "pt", "conservative"}:
        raise ValueError("barrier_tie_break must be one of {'sl', 'pt', 'conservative'}")
    if barrier_tie_break in {"sl", "pt"}:
        warnings.warn(
            "barrier_tie_break='sl'/'pt' assumes intrabar execution path that is not observable in kline OHLC data",
            UserWarning,
        )

    if volatility_fit_boundary is not None:
        boundary = max(1, int(volatility_fit_boundary))
        window = max(2, int(volatility_window or max_holding))
        train_close = pd.Series(close.iloc[:boundary], copy=False).astype(float)
        fold_volatility = train_close.pct_change(fill_method=None).rolling(window, min_periods=max(2, window // 2)).std()
        volatility = fold_volatility.reindex(close.index).ffill()
    else:
        volatility = pd.Series(volatility, copy=False).reindex(close.index)

    _entry_series = entry_prices if entry_prices is not None else close
    rows = []
    dropped_incomplete_future_windows = 0
    tie_count = 0

    # Ensure we never index beyond the close series
    for i in range(len(close) - max_holding - start_offset):
        # Entry price: execution price at bar i+start_offset
        entry = _entry_series.iloc[i + start_offset]
        vol = volatility.iloc[i]
        if pd.isna(entry) or not np.isfinite(entry) or entry <= 0:
            continue
        if pd.isna(vol) or vol <= 0:
            continue

        upper = entry * (1 + pt_m * vol + cost_rate)
        lower = entry * (1 - sl_m * vol - cost_rate)
        # Future window begins at the execution bar (bar i+start_offset)
        future_close = close.iloc[i + start_offset: i + start_offset + max_holding]
        future_high = high.iloc[i + start_offset: i + start_offset + max_holding]
        future_low = low.iloc[i + start_offset: i + start_offset + max_holding]
        if (
            len(future_close) < max_holding
            or len(future_high) < max_holding
            or len(future_low) < max_holding
            or future_close.isna().any()
            or future_high.isna().any()
            or future_low.isna().any()
        ):
            dropped_incomplete_future_windows += 1
            continue

        label = None
        barrier = None
        t1 = None
        exit_price = None
        for timestamp, high_price, low_price, close_price in zip(
            future_close.index,
            future_high.to_numpy(),
            future_low.to_numpy(),
            future_close.to_numpy(),
        ):
            hit_pt = bool(high_price >= upper)
            hit_sl = bool(low_price <= lower)
            if hit_pt and hit_sl:
                tie_count += 1
                if barrier_tie_break == "pt":
                    label, barrier, exit_price = 1, "pt", upper
                elif barrier_tie_break == "conservative":
                    label, barrier, exit_price = 0, "tie", close_price
                else:
                    label, barrier, exit_price = -1, "sl", lower
                t1 = timestamp
                break
            if hit_pt:
                label, barrier, t1, exit_price = 1, "pt", timestamp, upper
                break
            if hit_sl:
                label, barrier, t1, exit_price = -1, "sl", timestamp, lower
                break

        if t1 is not None:
            gross_return = (exit_price - entry) / entry
        else:
            end_price = future_close.iloc[-1] if len(future_close) else entry
            gross_return = (end_price - entry) / entry
            effective_move = max(abs(gross_return) - cost_rate, 0.0)
            label = 0 if effective_move < min_return else (1 if gross_return > 0 else -1)
            barrier = "time"
            t1 = future_close.index[-1] if len(future_close) else close.index[i]
            exit_price = end_price

        rows.append(
            {
                "t0": close.index[i],
                "t1": t1,
                "label": label,
                "barrier": barrier,
                "entry_price": entry,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "cost_rate": cost_rate,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(
            columns=[
                "t1",
                "label",
                "barrier",
                "entry_price",
                "exit_price",
                "gross_return",
                "cost_rate",
            ]
        )
        result.index.name = "t0"
    else:
        result = result.set_index("t0")
    result.attrs["integrity_report"] = {
        "missing_future_policy": missing_future_policy,
        "dropped_incomplete_future_windows": int(dropped_incomplete_future_windows),
        "retained_label_rows": int(len(result)),
        "tie_count": int(tie_count),
    }
    return result


def fixed_horizon_labels(close, horizon=5, threshold=0.0, cost_rate=0.0,
                         entry_prices=None, start_offset=0):
    """Label by forward return over *horizon* bars.

    Returns DataFrame with columns [label, t1, forward_return].
    label ∈ {-1, 0, +1} (0 = below threshold → abstain).

    Parameters
    ----------
    entry_prices : pd.Series or None – execution price series; when provided,
        returns are computed from entry_prices[T+start_offset] to
        entry_prices[T+start_offset+horizon] instead of close[T].
    start_offset : int – execution delay in bars.
    """
    start_offset = max(0, int(start_offset))
    _prices = entry_prices if entry_prices is not None else close
    # entry_value[T] = _prices[T + start_offset]
    entry_value = _prices.shift(-start_offset) if start_offset > 0 else pd.Series(_prices, copy=True)
    # exit_value[T] = _prices[T + start_offset + horizon]
    exit_value = entry_value.shift(-horizon)
    fwd = (exit_value - entry_value) / entry_value.replace(0, np.nan)
    effective_threshold = float(threshold) + float(cost_rate)
    label = pd.Series(0, index=close.index)
    label[fwd > effective_threshold] = 1
    label[fwd < -effective_threshold] = -1
    t1_series = pd.Series(_prices.index, index=_prices.index).shift(-(start_offset + horizon))
    result = pd.DataFrame({
        "label": label,
        "t1": t1_series,
        "forward_return": fwd,
        "gross_return": fwd,
        "cost_rate": cost_rate,
    })
    return result.dropna(subset=["t1"])


def _linear_trend_t_value(values):
    """Return the t-statistic and slope of a simple linear trend fit."""
    y = np.asarray(values, dtype=float).reshape(-1)
    if len(y) < 3 or np.isnan(y).any():
        return 0.0, 0.0

    x = np.arange(len(y), dtype=float)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0:
        return 0.0, 0.0

    slope = float(np.dot(x_centered, y_centered) / denominator)
    intercept = float(y.mean() - slope * x.mean())
    residuals = y - (intercept + slope * x)
    degrees_of_freedom = len(y) - 2
    if degrees_of_freedom <= 0:
        return 0.0, slope

    sigma2 = float(np.dot(residuals, residuals) / degrees_of_freedom)
    if sigma2 <= 0:
        return 0.0, slope

    slope_se = float(np.sqrt(sigma2 / denominator))
    if slope_se <= 0:
        return 0.0, slope

    return float(slope / slope_se), slope


def trend_scanning_labels(close, min_horizon=8, max_horizon=48, step=4,
                          min_t_value=1.5, min_return=0.0, cost_rate=0.0,
                          price_transform="log",
                          entry_prices=None, start_offset=0):
    """Label by the strongest forward trend over a range of horizons.

    For each timestamp, fit simple linear trends over candidate forward windows and
    select the horizon with the largest absolute t-statistic. This is typically
    less noisy than forcing one fixed forecast horizon when trends evolve over
    uneven durations.

    Returns a DataFrame with columns:
        label, t1, trend_t_value, trend_slope, trend_horizon, forward_return
    label ∈ {-1, 0, +1}; 0 means the trend was not statistically/economically strong enough.

    Parameters
    ----------
    entry_prices : pd.Series or None – execution price series; return is computed
        from entry_prices[T+start_offset] rather than close[T].
    start_offset : int – execution delay in bars.
    """
    close = pd.Series(close, copy=False).astype(float)
    start_offset = max(0, int(start_offset))
    _entry_series = pd.Series(entry_prices, copy=False).astype(float) if entry_prices is not None else close
    min_horizon = max(2, int(min_horizon))
    max_horizon = max(min_horizon, int(max_horizon))
    step = max(1, int(step))
    min_t_value = float(min_t_value)
    min_return = float(min_return)
    cost_rate = float(cost_rate)

    if price_transform == "log":
        transformed = np.log(close.where(close > 0))
    elif price_transform == "price":
        transformed = close.copy()
    else:
        raise ValueError(f"Unsupported price_transform={price_transform!r}")

    rows = []
    last_start = len(close) - min_horizon - start_offset
    for i in range(max(0, last_start)):
        # Execution entry at bar i+start_offset
        entry_price = float(_entry_series.iloc[i + start_offset])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        best = None
        for horizon in range(min_horizon, max_horizon + 1, step):
            end = i + start_offset + horizon
            if end >= len(close):
                break

            # Trend is measured over the forward price path starting at execution bar
            window = transformed.iloc[i + start_offset: end + 1]
            if window.isna().any():
                continue

            trend_t_value, slope = _linear_trend_t_value(window.to_numpy())
            if best is None or abs(trend_t_value) > abs(best["trend_t_value"]):
                exit_price = float(close.iloc[end])
                best = {
                    "t1": close.index[end],
                    "trend_t_value": float(trend_t_value),
                    "trend_slope": float(slope),
                    "trend_horizon": int(horizon),
                    "exit_price": exit_price,
                    "forward_return": float((exit_price - entry_price) / entry_price),
                }

        if best is None:
            continue

        effective_move = max(abs(best["forward_return"]) - cost_rate, 0.0)
        label = 0
        if abs(best["trend_t_value"]) >= min_t_value and effective_move >= min_return:
            label = 1 if best["trend_t_value"] > 0 else -1

        rows.append(
            {
                "t0": close.index[i],
                "t1": best["t1"],
                "label": label,
                "barrier": "trend_scan",
                "entry_price": entry_price,
                "exit_price": best["exit_price"],
                "gross_return": best["forward_return"],
                "forward_return": best["forward_return"],
                "trend_t_value": best["trend_t_value"],
                "trend_slope": best["trend_slope"],
                "trend_horizon": best["trend_horizon"],
                "cost_rate": cost_rate,
            }
        )

    return pd.DataFrame(rows).set_index("t0")


# ───────────────────────────────────────────────────────────────────────────
# Sample weights & sequential bootstrap
# ───────────────────────────────────────────────────────────────────────────

def _indicator_matrix(labels, timestamps):
    """Build boolean indicator matrix (n_times × n_labels).

    Entry [t, j] = 1 when label j's [t0, t1] span covers timestamp t.
    """
    n_t = len(timestamps)
    n_l = len(labels)
    mat = np.zeros((n_t, n_l), dtype=np.int8)
    t_idx = {t: i for i, t in enumerate(timestamps)}

    for j, (t0, row) in enumerate(labels.iterrows()):
        s = t_idx.get(t0)
        e = t_idx.get(row["t1"])
        if s is not None and e is not None:
            mat[s: e + 1, j] = 1
    return mat


def sample_weights_by_uniqueness(labels, close):
    """Weight each label by its average uniqueness (inverse of concurrency).

    Parameters
    ----------
    labels : pd.DataFrame  – output of triple_barrier_labels (index=t0, col t1)
    close  : pd.Series     – full close-price series (used as timestamp source)

    Returns
    -------
    pd.Series – one weight per label, indexed like *labels*
    """
    if labels is None or len(labels) == 0:
        return pd.Series(dtype=float, name="sample_weight")

    close = pd.Series(close, copy=False)
    if close.empty:
        return pd.Series(1.0, index=labels.index, name="sample_weight", dtype=float)

    mat = _indicator_matrix(labels, close.index)
    conc = mat.sum(axis=1).clip(min=1)
    weights = []
    for j in range(mat.shape[1]):
        mask = mat[:, j] == 1
        weights.append((1.0 / conc[mask]).mean() if mask.any() else 1.0)
    return pd.Series(weights, index=labels.index, name="sample_weight")


def sequential_bootstrap(labels, close, n_samples=None, random_state=None):
    """Draw sample indices that maximise average uniqueness.

    Returns np.ndarray of drawn label indices (integers into *labels*).
    """
    if labels is None or len(labels) == 0:
        return np.asarray([], dtype=int)

    mat = _indicator_matrix(labels, close.index)
    n_labels = mat.shape[1]
    if n_samples is None:
        n_samples = n_labels

    rng = np.random.default_rng(random_state)
    drawn = []
    for _ in range(n_samples):
        uniq = np.zeros(n_labels)
        for j in range(n_labels):
            cols = drawn + [j]
            sub = mat[:, cols]
            c = sub.sum(axis=1).clip(min=1)
            mask = sub[:, -1] == 1
            uniq[j] = (1.0 / c[mask]).mean() if mask.any() else 0.0
        probs = uniq / uniq.sum() if uniq.sum() > 0 else np.ones(n_labels) / n_labels
        drawn.append(int(rng.choice(n_labels, p=probs)))

    return np.asarray(drawn, dtype=int)
