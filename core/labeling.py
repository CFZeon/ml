"""Labeling (triple-barrier, fixed-horizon) and sampling (weights, bootstrap)."""

import numpy as np
import pandas as pd


# ───────────────────────────────────────────────────────────────────────────
# Labeling
# ───────────────────────────────────────────────────────────────────────────

def triple_barrier_labels(close, volatility, high=None, low=None, pt_sl=(2.0, 2.0),
                          max_holding=10, min_return=0.0, cost_rate=0.0,
                          barrier_tie_break="sl"):
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

    Returns
    -------
    pd.DataFrame  index=t0, columns=[t1, label, barrier]
        label ∈ {-1, 0, +1};  barrier ∈ {"pt", "sl", "time"}
    """
    pt_m, sl_m = pt_sl
    high = close if high is None else high.reindex(close.index)
    low = close if low is None else low.reindex(close.index)
    rows = []

    for i in range(len(close) - max_holding):
        entry = close.iloc[i]
        vol = volatility.iloc[i]
        if pd.isna(vol) or vol <= 0:
            continue

        upper = entry * (1 + pt_m * vol + cost_rate)
        lower = entry * (1 - sl_m * vol - cost_rate)
        future_close = close.iloc[i + 1: i + 1 + max_holding]
        future_high = high.iloc[i + 1: i + 1 + max_holding]
        future_low = low.iloc[i + 1: i + 1 + max_holding]

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
                if barrier_tie_break == "pt":
                    label, barrier, exit_price = 1, "pt", upper
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

    return pd.DataFrame(rows).set_index("t0")


def fixed_horizon_labels(close, horizon=5, threshold=0.0, cost_rate=0.0):
    """Label by forward return over *horizon* bars.

    Returns DataFrame with columns [label, t1, forward_return].
    label ∈ {-1, 0, +1} (0 = below threshold → abstain).
    """
    fwd = close.pct_change(horizon).shift(-horizon)
    effective_threshold = float(threshold) + float(cost_rate)
    label = pd.Series(0, index=close.index)
    label[fwd > effective_threshold] = 1
    label[fwd < -effective_threshold] = -1
    result = pd.DataFrame({
        "label": label,
        "t1": close.index.to_series().shift(-horizon),
        "forward_return": fwd,
        "gross_return": fwd,
        "cost_rate": cost_rate,
    })
    return result.dropna(subset=["t1"])


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
    mat = _indicator_matrix(labels, close.index)
    conc = mat.sum(axis=1).clip(min=1)
    weights = []
    for j in range(mat.shape[1]):
        mask = mat[:, j] == 1
        weights.append((1.0 / conc[mask]).mean() if mask.any() else 1.0)
    return pd.Series(weights, index=labels.index, name="sample_weight")


def sequential_bootstrap(labels, close, n_samples=None):
    """Draw sample indices that maximise average uniqueness.

    Returns np.ndarray of drawn label indices (integers into *labels*).
    """
    mat = _indicator_matrix(labels, close.index)
    n_labels = mat.shape[1]
    if n_samples is None:
        n_samples = n_labels

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
        drawn.append(np.random.choice(n_labels, p=probs))

    return np.array(drawn)
