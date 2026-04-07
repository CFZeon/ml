"""Labeling (triple-barrier, fixed-horizon) and sampling (weights, bootstrap)."""

import numpy as np
import pandas as pd


# ───────────────────────────────────────────────────────────────────────────
# Labeling
# ───────────────────────────────────────────────────────────────────────────

def triple_barrier_labels(close, volatility, pt_sl=(2.0, 2.0),
                          max_holding=10, min_return=0.0):
    """Apply triple-barrier labeling.

    Three concurrent barriers decide the label per bar:
      • profit-taking  (upper) = entry * (1 + pt_mult × vol)
      • stop-loss      (lower) = entry * (1 − sl_mult × vol)
      • vertical time  = max_holding bars

    Parameters
    ----------
    close : pd.Series      – close prices (datetime index)
    volatility : pd.Series  – per-bar volatility **in decimal** (e.g. 0.02 = 2 %)
    pt_sl : tuple(float,float) – (profit-taking multiplier, stop-loss multiplier)
    max_holding : int       – bars until vertical barrier
    min_return : float      – returns below this at the time barrier → label 0 (abstain)

    Returns
    -------
    pd.DataFrame  index=t0, columns=[t1, label, barrier]
        label ∈ {-1, 0, +1};  barrier ∈ {"pt", "sl", "time"}
    """
    pt_m, sl_m = pt_sl
    rows = []

    for i in range(len(close) - max_holding):
        entry = close.iloc[i]
        vol = volatility.iloc[i]
        if pd.isna(vol) or vol <= 0:
            continue

        upper = entry * (1 + pt_m * vol)
        lower = entry * (1 - sl_m * vol)
        future = close.iloc[i + 1: i + 1 + max_holding]

        pt_idx = future[future >= upper]
        sl_idx = future[future <= lower]
        pt_time = pt_idx.index[0] if len(pt_idx) else None
        sl_time = sl_idx.index[0] if len(sl_idx) else None

        if pt_time and sl_time:
            if pt_time <= sl_time:
                label, barrier, t1 = 1, "pt", pt_time
            else:
                label, barrier, t1 = -1, "sl", sl_time
        elif pt_time:
            label, barrier, t1 = 1, "pt", pt_time
        elif sl_time:
            label, barrier, t1 = -1, "sl", sl_time
        else:
            end_price = future.iloc[-1] if len(future) else entry
            ret = (end_price - entry) / entry
            label = 0 if abs(ret) < min_return else (1 if ret > 0 else -1)
            barrier = "time"
            t1 = future.index[-1] if len(future) else close.index[i]

        rows.append({"t0": close.index[i], "t1": t1, "label": label, "barrier": barrier})

    return pd.DataFrame(rows).set_index("t0")


def fixed_horizon_labels(close, horizon=5, threshold=0.0):
    """Label by forward return over *horizon* bars.

    Returns DataFrame with columns [label, t1, forward_return].
    label ∈ {-1, 0, +1} (0 = below threshold → abstain).
    """
    fwd = close.pct_change(horizon).shift(-horizon)
    label = pd.Series(0, index=close.index)
    label[fwd > threshold] = 1
    label[fwd < -threshold] = -1
    result = pd.DataFrame({
        "label": label,
        "t1": close.index.to_series().shift(-horizon),
        "forward_return": fwd,
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
