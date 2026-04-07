"""Training, meta-labeling, walk-forward CV, regime detection, model store."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.cluster import KMeans


# ───────────────────────────────────────────────────────────────────────────
# Walk-forward splits
# ───────────────────────────────────────────────────────────────────────────

def walk_forward_split(X, n_splits=3, train_size=None, test_size=None,
                       gap=0, expanding=False):
    """Yield (train_idx, test_idx) arrays – never shuffled.

    Parameters
    ----------
    X : array-like        – only length is used
    n_splits : int
    train_size, test_size : int or None (auto-sized from n_splits)
    gap : int             – embargo rows between train and test
    expanding : bool      – if True train_start is always 0
    """
    n = len(X)
    if test_size is None:
        test_size = n // (n_splits + 1)
    if train_size is None:
        train_size = test_size * 2

    for i in range(n_splits):
        test_end = n - (n_splits - i - 1) * test_size
        test_start = test_end - test_size
        train_end = test_start - gap
        train_start = 0 if expanding else max(0, train_end - train_size)

        if train_start >= train_end or test_start >= test_end:
            continue

        yield (np.arange(train_start, train_end),
               np.arange(test_start, min(test_end, n)))


# ───────────────────────────────────────────────────────────────────────────
# Model catalogue
# ───────────────────────────────────────────────────────────────────────────

_MODELS = {
    "rf": lambda: RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=42, n_jobs=-1),
    "gbm": lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=4, random_state=42),
    "logistic": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42),
}


def train_model(X, y, sample_weight=None, model_type="rf"):
    """Train a classifier.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series           – labels (may include 0 = abstain)
    sample_weight : pd.Series or None
    model_type : str        – "rf" | "gbm" | "logistic"

    Returns the fitted model.
    """
    if model_type not in _MODELS:
        raise ValueError(f"Unknown model_type={model_type!r}. Choose from {list(_MODELS)}")
    model = _MODELS[model_type]()
    sw = sample_weight.values if sample_weight is not None else None
    model.fit(X, y, sample_weight=sw)
    return model


# ───────────────────────────────────────────────────────────────────────────
# Meta-labeling
# ───────────────────────────────────────────────────────────────────────────

def train_meta_model(primary_preds, X, y_true, sample_weight=None):
    """Train a meta-labeling model.

    The meta model learns *whether the primary prediction is correct* and
    outputs a probability used for bet sizing.

    y_meta = 1 if primary_pred == y_true, else 0.
    """
    y_meta = (np.asarray(primary_preds) == np.asarray(y_true)).astype(int)
    X_meta = X.copy()
    X_meta["primary_pred"] = primary_preds
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    sw = sample_weight.values if sample_weight is not None else None
    model.fit(X_meta, y_meta, sample_weight=sw)
    return model


# ───────────────────────────────────────────────────────────────────────────
# Evaluation
# ───────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X, y):
    """Return dict of classification metrics."""
    preds = model.predict(X)
    m = {
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "f1_macro": round(float(f1_score(y, preds, average="macro", zero_division=0)), 4),
    }
    if hasattr(model, "predict_proba"):
        try:
            m["log_loss"] = round(
                float(log_loss(y, model.predict_proba(X), labels=model.classes_)),
                4,
            )
        except Exception:
            pass
    return m


# ───────────────────────────────────────────────────────────────────────────
# Regime detection  (simple KMeans – swap for HMM/ADWIN later)
# ───────────────────────────────────────────────────────────────────────────

def detect_regime(features, n_regimes=2):
    """Cluster rows into *n_regimes* regimes using KMeans.

    Parameters
    ----------
    features : pd.DataFrame – numeric columns (e.g. volatility, volume_trend)

    Returns pd.Series of regime labels (ints), indexed like *features*.
    """
    clean = features.dropna()
    normed = (clean - clean.mean()) / clean.std().replace(0, 1)
    km = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    return pd.Series(km.fit_predict(normed), index=clean.index, name="regime")


# ───────────────────────────────────────────────────────────────────────────
# Artifact store  (pickle-based – swap for MLflow/W&B later)
# ───────────────────────────────────────────────────────────────────────────

def save_model(model, path, metadata=None):
    """Persist model + metadata dict to *path*."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump({"model": model, "metadata": metadata or {}}, f)


def load_model(path):
    """Return (model, metadata) from a previously saved artifact."""
    with open(path, "rb") as f:
        art = pickle.load(f)
    return art["model"], art["metadata"]
