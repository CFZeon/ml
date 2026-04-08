"""Training, meta-labeling, walk-forward CV, regime detection, model store."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

def build_model(model_type="rf", model_params=None):
    """Create a configured model instance from type and parameter dict."""
    model_params = dict(model_params or {})

    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=int(model_params.get("n_estimators", 200)),
            max_depth=model_params.get("max_depth"),
            min_samples_leaf=int(model_params.get("min_samples_leaf", 1)),
            class_weight=model_params.get("class_weight", "balanced"),
            random_state=int(model_params.get("random_state", 42)),
            n_jobs=int(model_params.get("n_jobs", -1)),
        )

    if model_type == "gbm":
        return GradientBoostingClassifier(
            n_estimators=int(model_params.get("n_estimators", 200)),
            learning_rate=float(model_params.get("learning_rate", 0.1)),
            max_depth=int(model_params.get("max_depth", 4)),
            subsample=float(model_params.get("subsample", 1.0)),
            min_samples_leaf=int(model_params.get("min_samples_leaf", 1)),
            random_state=int(model_params.get("random_state", 42)),
        )

    if model_type == "logistic":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(model_params.get("c", model_params.get("C", 1.0))),
                        max_iter=int(model_params.get("max_iter", 1000)),
                        class_weight=model_params.get("class_weight", "balanced"),
                        random_state=int(model_params.get("random_state", 42)),
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown model_type={model_type!r}. Choose from ['rf', 'gbm', 'logistic']")


def train_model(X, y, sample_weight=None, model_type="rf", model_params=None):
    """Train a classifier.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series           – labels (may include 0 = abstain)
    sample_weight : pd.Series or None
    model_type : str        – "rf" | "gbm" | "logistic"

    Returns the fitted model.
    """
    model = build_model(model_type=model_type, model_params=model_params)
    sw = sample_weight.values if sample_weight is not None else None
    fit_kwargs = {}
    if sw is not None:
        fit_kwargs["sample_weight"] = sw
        if isinstance(model, Pipeline):
            fit_kwargs["model__sample_weight"] = sw
            fit_kwargs.pop("sample_weight", None)

    model.fit(X, y, **fit_kwargs)
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


def _resolve_estimator(model):
    if isinstance(model, Pipeline):
        return model.named_steps.get("model", model)
    return model


def get_feature_importance(model, feature_names):
    """Return a feature-importance series when the estimator exposes one."""
    estimator = _resolve_estimator(model)

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float))
        if values.ndim > 1:
            values = values.mean(axis=0)
        values = values.ravel()
    else:
        return pd.Series(dtype=float)

    if len(values) != len(feature_names):
        return pd.Series(dtype=float)

    return pd.Series(values, index=feature_names, dtype=float).sort_values(ascending=False)


def compute_feature_block_diagnostics(model, X_train, X_test, y_test, feature_blocks, baseline_metrics=None):
    """Measure feature-block contribution using native importance and OOS ablation."""
    baseline_metrics = dict(baseline_metrics or evaluate_model(model, X_test, y_test))
    feature_blocks = dict(feature_blocks or {})
    native_importance = get_feature_importance(model, list(X_test.columns))
    block_columns = {}

    for column in X_test.columns:
        block_name = feature_blocks.get(column, "unknown")
        block_columns.setdefault(block_name, []).append(column)

    train_medians = X_train.median(numeric_only=True)
    blocks = []
    for block_name, columns in sorted(block_columns.items()):
        ablated_X = X_test.copy()
        replacement = train_medians.reindex(columns).fillna(0.0).to_numpy()
        ablated_X.loc[:, columns] = replacement
        ablated_metrics = evaluate_model(model, ablated_X, y_test)

        block_info = {
            "block": block_name,
            "feature_count": len(columns),
            "native_importance": round(float(native_importance.reindex(columns).fillna(0.0).sum()), 6),
            "accuracy_drop": round(
                float(baseline_metrics.get("accuracy", 0.0) - ablated_metrics.get("accuracy", 0.0)),
                6,
            ),
            "f1_drop": round(
                float(baseline_metrics.get("f1_macro", 0.0) - ablated_metrics.get("f1_macro", 0.0)),
                6,
            ),
            "log_loss_increase": None,
        }
        if "log_loss" in baseline_metrics and "log_loss" in ablated_metrics:
            block_info["log_loss_increase"] = round(
                float(ablated_metrics["log_loss"] - baseline_metrics["log_loss"]),
                6,
            )
        blocks.append(block_info)

    blocks.sort(
        key=lambda item: (
            item.get("f1_drop", 0.0),
            item.get("accuracy_drop", 0.0),
            item.get("native_importance", 0.0),
        ),
        reverse=True,
    )

    top_features = []
    if not native_importance.empty:
        for feature, importance in native_importance.head(10).items():
            top_features.append(
                {
                    "feature": feature,
                    "block": feature_blocks.get(feature, "unknown"),
                    "native_importance": round(float(importance), 6),
                }
            )

    return {
        "baseline_metrics": baseline_metrics,
        "blocks": blocks,
        "top_features": top_features,
    }


def summarize_feature_block_diagnostics(fold_diagnostics):
    """Aggregate block diagnostics across walk-forward folds."""
    if not fold_diagnostics:
        return {"summary": [], "top_features": [], "folds": []}

    block_totals = {}
    feature_totals = {}
    for fold_index, diagnostics in enumerate(fold_diagnostics):
        for block_info in diagnostics.get("blocks", []):
            block_name = block_info["block"]
            summary = block_totals.setdefault(
                block_name,
                {
                    "block": block_name,
                    "feature_count": block_info["feature_count"],
                    "folds": 0,
                    "native_importance": [],
                    "accuracy_drop": [],
                    "f1_drop": [],
                    "log_loss_increase": [],
                },
            )
            summary["feature_count"] = max(summary["feature_count"], block_info["feature_count"])
            summary["folds"] += 1
            summary["native_importance"].append(block_info.get("native_importance", 0.0))
            summary["accuracy_drop"].append(block_info.get("accuracy_drop", 0.0))
            summary["f1_drop"].append(block_info.get("f1_drop", 0.0))
            if block_info.get("log_loss_increase") is not None:
                summary["log_loss_increase"].append(block_info["log_loss_increase"])

        for feature_info in diagnostics.get("top_features", []):
            feature_name = feature_info["feature"]
            feature_total = feature_totals.setdefault(
                feature_name,
                {
                    "feature": feature_name,
                    "block": feature_info["block"],
                    "native_importance": [],
                },
            )
            feature_total["native_importance"].append(feature_info["native_importance"])

    summary_rows = []
    for block_name, totals in block_totals.items():
        summary_rows.append(
            {
                "block": block_name,
                "feature_count": totals["feature_count"],
                "folds": totals["folds"],
                "avg_native_importance": round(float(np.mean(totals["native_importance"])), 6),
                "avg_accuracy_drop": round(float(np.mean(totals["accuracy_drop"])), 6),
                "avg_f1_drop": round(float(np.mean(totals["f1_drop"])), 6),
                "avg_log_loss_increase": (
                    round(float(np.mean(totals["log_loss_increase"])), 6)
                    if totals["log_loss_increase"]
                    else None
                ),
            }
        )

    summary_rows.sort(
        key=lambda item: (
            item.get("avg_f1_drop", 0.0),
            item.get("avg_accuracy_drop", 0.0),
            item.get("avg_native_importance", 0.0),
        ),
        reverse=True,
    )

    top_features = []
    for feature_name, totals in sorted(
        feature_totals.items(),
        key=lambda item: np.mean(item[1]["native_importance"]),
        reverse=True,
    )[:10]:
        top_features.append(
            {
                "feature": feature_name,
                "block": totals["block"],
                "avg_native_importance": round(float(np.mean(totals["native_importance"])), 6),
            }
        )

    return {
        "summary": summary_rows,
        "top_features": top_features,
        "folds": fold_diagnostics,
    }


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
