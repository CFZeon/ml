"""Training, meta-labeling, walk-forward CV, regime detection, model store."""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss
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

def build_model(model_type="gbm", model_params=None):
    """Create a configured model instance from type and parameter dict."""
    model_params = dict(model_params or {})

    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=int(model_params.get("n_estimators", 200)),
            max_depth=model_params.get("max_depth"),
            min_samples_leaf=int(model_params.get("min_samples_leaf", 1)),
            class_weight=model_params.get("class_weight", None),
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
                        class_weight=model_params.get("class_weight", None),
                        random_state=int(model_params.get("random_state", 42)),
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown model_type={model_type!r}. Choose from ['rf', 'gbm', 'logistic']")


def train_model(X, y, sample_weight=None, model_type="gbm", model_params=None):
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

class ConstantProbabilityModel:
    """Minimal classifier that returns a fixed positive-class probability."""

    def __init__(self, positive_probability=0.5):
        self.positive_probability = float(np.clip(positive_probability, 0.0, 1.0))
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        count = len(X)
        positive = np.full(count, self.positive_probability, dtype=float)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])

    def predict(self, X):
        label = 1 if self.positive_probability >= 0.5 else 0
        return np.full(len(X), label, dtype=int)


class BinaryProbabilityCalibrator:
    """Platt-style calibrator on top of a raw binary probability series."""

    def __init__(self, model):
        self.model = model

    @staticmethod
    def _build_features(probabilities):
        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        logits = np.log(probs / (1.0 - probs))
        return np.column_stack([probs, logits])

    def predict_proba(self, probabilities):
        features = self._build_features(probabilities)
        return self.model.predict_proba(features)

    def transform(self, probabilities):
        return self.predict_proba(probabilities)[:, 1]


def fit_binary_probability_calibrator(probabilities, y_true, sample_weight=None, model_params=None):
    """Fit a Platt-style calibrator for binary probabilities."""
    model_params = dict(model_params or {})
    target = pd.Series(y_true).astype(int)
    sw = None
    if sample_weight is not None:
        raw_weights = sample_weight.values if isinstance(sample_weight, pd.Series) else sample_weight
        sw = np.asarray(raw_weights, dtype=float)

    if target.nunique() < 2:
        if sw is not None and float(sw.sum()) > 0.0:
            positive_rate = float(np.average(target, weights=sw))
        else:
            positive_rate = float(target.mean()) if len(target) else 0.5
        return ConstantProbabilityModel(positive_probability=positive_rate)

    features = BinaryProbabilityCalibrator._build_features(probabilities)
    model = LogisticRegression(
        C=float(model_params.get("c", model_params.get("C", 1.0))),
        max_iter=int(model_params.get("max_iter", 1000)),
        class_weight=model_params.get("class_weight"),
        random_state=int(model_params.get("random_state", 42)),
    )
    fit_kwargs = {"sample_weight": sw} if sw is not None else {}
    model.fit(features, target, **fit_kwargs)
    return BinaryProbabilityCalibrator(model)


def apply_binary_probability_calibrator(calibrator, probabilities):
    """Apply a previously fitted binary probability calibrator."""
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if calibrator is None:
        return values
    if hasattr(calibrator, "transform"):
        return calibrator.transform(values)
    return calibrator.predict_proba(values)[:, 1]


def predict_probability_frame(model, X, ordered_classes=(-1, 0, 1)):
    """Return aligned class probabilities for the requested class order."""
    probabilities = model.predict_proba(X)
    classes = getattr(model, "classes_", None)

    if classes is None and isinstance(model, Pipeline):
        estimator = model.named_steps.get("model")
        classes = getattr(estimator, "classes_", None)

    if classes is None:
        raise ValueError("Model does not expose classes_ for probability alignment")

    frame = pd.DataFrame(probabilities, index=X.index, columns=list(classes), dtype=float)
    for class_label in ordered_classes:
        if class_label not in frame.columns:
            frame[class_label] = 0.0
    return frame.loc[:, list(ordered_classes)]


def build_meta_feature_frame(primary_preds, primary_probabilities, context=None):
    """Build a compact, probability-aware meta-label feature frame.

    Parameters
    ----------
    primary_preds : array-like
        Directional predictions from the primary model.
    primary_probabilities : pd.DataFrame
        Per-class probability frame aligned to primary_preds.
    context : pd.DataFrame or None
        Optional regime / volatility features to merge into the meta frame.
        Columns are prefixed with ``ctx_`` and NaN-filled to guard against
        alignment gaps.
    """
    prediction_series = pd.Series(primary_preds, index=primary_probabilities.index)
    probability_frame = primary_probabilities.copy()

    meta = pd.DataFrame(index=probability_frame.index)
    meta["primary_pred"] = prediction_series.astype(float)
    meta["prob_short"] = probability_frame.get(-1, 0.0)
    meta["prob_flat"] = probability_frame.get(0, 0.0)
    meta["prob_long"] = probability_frame.get(1, 0.0)
    meta["direction_edge"] = meta["prob_long"] - meta["prob_short"]

    predicted_class_prob = []
    for timestamp, prediction in prediction_series.items():
        if prediction in probability_frame.columns:
            predicted_class_prob.append(float(probability_frame.loc[timestamp, prediction]))
        else:
            predicted_class_prob.append(0.0)
    meta["predicted_class_prob"] = predicted_class_prob

    if context is not None:
        context_df = context if isinstance(context, pd.DataFrame) else pd.DataFrame(context)
        numeric_ctx = (
            context_df
            .select_dtypes(include=[np.number])
            .reindex(meta.index)
            .fillna(0.0)
        )
        for col in numeric_ctx.columns:
            meta[f"ctx_{col}"] = numeric_ctx[col].values

    return meta


def build_trade_outcome_frame(primary_preds, labels):
    """Build realized trade outcomes for a set of directional predictions.

    The resulting frame is aligned to ``primary_preds`` and estimates the net
    trade return implied by taking the predicted direction over the label event.
    When label return columns are unavailable, an empty frame is returned.
    """
    prediction_series = pd.Series(primary_preds, copy=False)
    if labels is None or prediction_series.empty:
        return pd.DataFrame(index=prediction_series.index)

    label_frame = labels.reindex(prediction_series.index).copy()
    if label_frame.empty:
        return pd.DataFrame(index=prediction_series.index)

    return_column = None
    for candidate in ["gross_return", "forward_return"]:
        if candidate in label_frame.columns:
            return_column = candidate
            break

    if return_column is None:
        return pd.DataFrame(index=prediction_series.index)

    trade_direction = prediction_series.apply(lambda value: 1.0 if value > 0 else (-1.0 if value < 0 else 0.0))
    gross_return = pd.to_numeric(label_frame[return_column], errors="coerce")
    cost_rate = pd.Series(0.0, index=prediction_series.index, dtype=float)
    if "cost_rate" in label_frame.columns:
        cost_rate = pd.to_numeric(label_frame["cost_rate"], errors="coerce").reindex(prediction_series.index).fillna(0.0)

    gross_trade_return = trade_direction * gross_return
    net_trade_return = (gross_trade_return - cost_rate).where(trade_direction.ne(0.0), 0.0)
    profitable = (trade_direction.ne(0.0) & net_trade_return.gt(0.0)).astype(int)

    return pd.DataFrame(
        {
            "trade_direction": trade_direction.astype(float),
            "gross_trade_return": gross_trade_return.astype(float),
            "net_trade_return": net_trade_return.astype(float),
            "profitable": profitable.astype(int),
            "trade_taken": trade_direction.ne(0.0).astype(int),
        },
        index=prediction_series.index,
    )


def build_execution_outcome_frame(primary_preds, valuation_prices, execution_prices=None,
                                  holding_bars=1, signal_delay_bars=1,
                                  fee_rate=0.0, slippage_rate=0.0,
                                  funding_rates=None, cutoff_timestamp=None):
    """Build execution-aligned trade outcomes for a directional prediction series.

    Outcomes are computed using the same delayed, bar-by-bar return semantics as
    the backtest adapter rather than label-specific barrier returns. This keeps
    profitability targets and Kelly inputs aligned to executed strategy PnL.
    """
    prediction_series = pd.Series(primary_preds, copy=False)
    if prediction_series.empty:
        return pd.DataFrame(index=prediction_series.index)

    valuation_series = pd.Series(valuation_prices, copy=False).astype(float)
    if execution_prices is None:
        execution_series = valuation_series
    else:
        execution_series = pd.Series(execution_prices, copy=False).reindex(valuation_series.index).astype(float)

    funding_series = None
    if funding_rates is not None:
        funding_series = pd.Series(funding_rates, copy=False).reindex(valuation_series.index).fillna(0.0).astype(float)

    holding_bars = max(1, int(holding_bars))
    signal_delay_bars = max(0, int(signal_delay_bars))
    cutoff = pd.Timestamp(cutoff_timestamp) if cutoff_timestamp is not None else None
    execution_returns = execution_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    round_trip_cost = 2.0 * (float(fee_rate) + float(slippage_rate))

    rows = []
    index_positions = valuation_series.index.get_indexer(prediction_series.index)
    for timestamp, prediction, base_loc in zip(
        prediction_series.index,
        prediction_series.to_numpy(),
        index_positions,
    ):
        direction = 1.0 if prediction > 0 else (-1.0 if prediction < 0 else 0.0)
        row = {
            "trade_direction": float(direction),
            "entry_time": pd.NaT,
            "exit_time": pd.NaT,
            "entry_price": np.nan,
            "exit_price": np.nan,
            "gross_trade_return": (0.0 if direction == 0.0 else np.nan),
            "net_trade_return": (0.0 if direction == 0.0 else np.nan),
            "profitable": (0 if direction == 0.0 else np.nan),
            "trade_taken": 0,
            "outcome_available": (1 if direction == 0.0 else 0),
        }

        if direction == 0.0 or base_loc < 0:
            rows.append(row)
            continue

        active_start = int(base_loc + signal_delay_bars)
        active_end = int(active_start + holding_bars - 1)
        if active_start >= len(valuation_series) or active_end >= len(valuation_series):
            rows.append(row)
            continue

        entry_time = valuation_series.index[active_start]
        exit_time = valuation_series.index[active_end]
        if cutoff is not None and exit_time > cutoff:
            rows.append(row)
            continue

        price_returns = direction * execution_returns.iloc[active_start: active_end + 1].to_numpy(dtype=float)
        gross_trade_return = float(np.prod(1.0 + price_returns) - 1.0)

        funding_trade_return = 0.0
        if funding_series is not None:
            funding_bar_returns = -direction * funding_series.iloc[active_start: active_end + 1].to_numpy(dtype=float)
            funding_trade_return = float(np.prod(1.0 + price_returns + funding_bar_returns) - 1.0) - gross_trade_return

        net_trade_return = gross_trade_return + funding_trade_return - round_trip_cost
        entry_price = execution_series.iloc[active_start]
        exit_price = execution_series.iloc[active_end]
        if not np.isfinite(entry_price):
            entry_price = np.nan
        if not np.isfinite(exit_price):
            exit_price = np.nan

        row.update(
            {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_trade_return": gross_trade_return,
                "net_trade_return": net_trade_return,
                "profitable": int(net_trade_return > 0.0),
                "trade_taken": 1,
                "outcome_available": 1,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, index=prediction_series.index)


def train_meta_model(primary_preds, primary_probabilities, y_true, labels=None,
                     trade_outcomes=None, sample_weight=None, model_params=None,
                     context=None):
    """Train a meta-labeling model on primary predictions and probabilities.

    The preferred target is trade profitability after costs, derived from the
    realized execution-aligned outcomes. If profitability data is unavailable,
    the model falls back to label-derived outcomes and then raw directional
    correctness.
    """
    model_params = dict(model_params or {})
    prediction_series = pd.Series(primary_preds, index=y_true.index)
    if trade_outcomes is None:
        trade_outcomes = build_trade_outcome_frame(prediction_series, labels)

    if trade_outcomes is not None and not trade_outcomes.empty and trade_outcomes["profitable"].notna().any():
        y_meta = pd.to_numeric(
            trade_outcomes["profitable"].reindex(prediction_series.index),
            errors="coerce",
        )
    else:
        y_meta = (prediction_series == pd.Series(y_true, index=y_true.index)).astype(float)

    X_meta = build_meta_feature_frame(prediction_series, primary_probabilities, context=context)
    valid_mask = X_meta.notna().all(axis=1) & y_meta.notna()

    sw = None
    if sample_weight is not None:
        sample_weight_series = pd.Series(sample_weight, index=X_meta.index)
        valid_mask &= sample_weight_series.notna()
        sw = sample_weight_series.loc[valid_mask].to_numpy(dtype=float)

    X_meta = X_meta.loc[valid_mask]
    y_meta = y_meta.loc[valid_mask].astype(int)
    if y_meta.nunique() < 2:
        if sw is not None and sw.sum() > 0:
            positive_rate = float(np.average(y_meta, weights=sw))
        else:
            positive_rate = float(y_meta.mean()) if len(y_meta) else 0.5
        return ConstantProbabilityModel(positive_probability=positive_rate)

    model = Pipeline(
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
    fit_kwargs = {"model__sample_weight": sw} if sw is not None else {}
    model.fit(X_meta, y_meta, **fit_kwargs)
    return model


# ───────────────────────────────────────────────────────────────────────────
# Evaluation
# ───────────────────────────────────────────────────────────────────────────

def _binary_expected_calibration_error(y_true, positive_probability, n_bins=10):
    """Return a simple expected calibration error for binary probabilities."""
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    probability = np.clip(np.asarray(positive_probability, dtype=float).reshape(-1), 0.0, 1.0)

    valid_mask = np.isfinite(truth) & np.isfinite(probability)
    if not valid_mask.any():
        return np.nan

    truth = truth[valid_mask]
    probability = probability[valid_mask]
    if len(probability) == 0:
        return np.nan

    bins = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
    assignments = np.digitize(probability, bins[1:-1], right=True)
    total = float(len(probability))
    calibration_error = 0.0

    for bin_idx in range(len(bins) - 1):
        bin_mask = assignments == bin_idx
        if not bin_mask.any():
            continue
        avg_confidence = float(probability[bin_mask].mean())
        avg_accuracy = float(truth[bin_mask].mean())
        calibration_error += (float(bin_mask.sum()) / total) * abs(avg_confidence - avg_accuracy)

    return calibration_error


def _evaluate_directional_probability_quality(probability_frame, y_true):
    """Score directional probabilities on the non-abstain subset only."""
    y_series = pd.Series(y_true, index=probability_frame.index)
    directional_mask = y_series.ne(0)
    if not directional_mask.any():
        return {}

    directional_frame = probability_frame.loc[directional_mask, [-1, 1]].copy()
    row_sums = directional_frame.sum(axis=1).replace(0.0, np.nan)
    valid_mask = row_sums.notna() & directional_frame.notna().all(axis=1)
    if not valid_mask.any():
        return {}

    directional_frame = directional_frame.loc[valid_mask].div(row_sums.loc[valid_mask], axis=0)
    directional_truth = y_series.loc[directional_mask].loc[valid_mask]
    binary_truth = directional_truth.eq(1).astype(int)
    positive_probability = directional_frame[1].clip(1e-6, 1.0 - 1e-6)

    metrics = {
        "directional_log_loss": round(
            float(
                log_loss(
                    binary_truth,
                    np.column_stack([1.0 - positive_probability.to_numpy(), positive_probability.to_numpy()]),
                    labels=[0, 1],
                )
            ),
            4,
        ),
        "directional_brier_score": round(
            float(brier_score_loss(binary_truth, positive_probability.to_numpy())),
            4,
        ),
        "directional_calibration_error": round(
            float(_binary_expected_calibration_error(binary_truth, positive_probability.to_numpy())),
            4,
        ),
    }
    metrics["log_loss"] = metrics["directional_log_loss"]
    metrics["brier_score"] = metrics["directional_brier_score"]
    metrics["calibration_error"] = metrics["directional_calibration_error"]
    return metrics

def evaluate_model(model, X, y):
    """Return dict of classification metrics."""
    preds = model.predict(X)
    y_series = pd.Series(y, index=X.index if hasattr(X, "index") else None)
    pred_series = pd.Series(preds, index=y_series.index)
    m = {
        "accuracy": round(float(accuracy_score(y_series, pred_series)), 4),
        "f1_macro": round(float(f1_score(y_series, pred_series, average="macro", zero_division=0)), 4),
        "prediction_coverage": round(float(pred_series.ne(0).mean()), 4),
        "label_abstain_rate": round(float(y_series.eq(0).mean()), 4),
    }

    directional_mask = y_series.ne(0)
    if directional_mask.any():
        m["directional_accuracy"] = round(
            float(accuracy_score(y_series.loc[directional_mask], pred_series.loc[directional_mask])),
            4,
        )
        m["directional_f1_macro"] = round(
            float(
                f1_score(
                    y_series.loc[directional_mask],
                    pred_series.loc[directional_mask],
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        )
    if hasattr(model, "predict_proba"):
        try:
            probability_frame = predict_probability_frame(model, X)
            m.update(_evaluate_directional_probability_quality(probability_frame, y_series))
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
def _coalesce_regime_signal(features, include_terms, exclude_terms=None,
                            fallback_column=None, reference_features=None):
    exclude_terms = tuple(term.lower() for term in (exclude_terms or ()))
    reference = features if reference_features is None else reference_features.reindex(columns=features.columns)
    selected_columns = []
    for column in features.columns:
        lower = column.lower()
        if any(term in lower for term in include_terms) and not any(term in lower for term in exclude_terms):
            selected_columns.append(column)

    if not selected_columns and fallback_column is not None and fallback_column in features.columns:
        selected_columns = [fallback_column]

    if not selected_columns:
        return pd.Series(0.0, index=features.index, dtype=float)

    selected = features[selected_columns].apply(pd.to_numeric, errors="coerce")
    reference_selected = reference[selected_columns].apply(pd.to_numeric, errors="coerce")
    reference_mean = reference_selected.mean()
    reference_std = reference_selected.std().replace(0, 1)
    standardized = (selected - reference_mean) / reference_std
    return standardized.mean(axis=1).fillna(0.0)


def _bucket_regime_signal(series, lower_quantile=0.33, upper_quantile=0.67,
                          invert=False, reference_series=None):
    values = -pd.Series(series, copy=False) if invert else pd.Series(series, copy=False)
    reference = values if reference_series is None else (-pd.Series(reference_series, copy=False) if invert else pd.Series(reference_series, copy=False))
    clean = reference.dropna()
    if clean.empty:
        return pd.Series(0, index=values.index, dtype=int)

    lower = float(clean.quantile(lower_quantile))
    upper = float(clean.quantile(upper_quantile))
    bucket = pd.Series(0, index=values.index, dtype=int)
    bucket[values <= lower] = -1
    bucket[values >= upper] = 1
    return bucket


def _detect_explicit_regime(features, config=None, fit_features=None):
    config = dict(config or {})
    clean = features.dropna()
    if clean.empty:
        return pd.DataFrame(columns=["trend_regime", "volatility_regime", "liquidity_regime", "regime"])

    reference = clean if fit_features is None else fit_features.reindex(columns=features.columns).dropna()
    if reference.empty:
        reference = clean

    trend_score = _coalesce_regime_signal(
        clean,
        include_terms=("trend", "ret_", "return", "momentum"),
        exclude_terms=("vol", "volume", "liquid"),
        reference_features=reference,
    )
    volatility_score = _coalesce_regime_signal(
        clean,
        include_terms=("vol", "range", "atr", "dispersion"),
        exclude_terms=("volume", "liquid"),
        reference_features=reference,
    )
    liquidity_score = _coalesce_regime_signal(
        clean,
        include_terms=("liquid", "volume", "turnover", "trade"),
        exclude_terms=("illiquid",),
        reference_features=reference,
    )
    illiquidity_score = _coalesce_regime_signal(
        clean,
        include_terms=("illiquid", "amihud"),
        reference_features=reference,
    )
    liquidity_score = liquidity_score - illiquidity_score

    trend_reference = _coalesce_regime_signal(
        reference,
        include_terms=("trend", "ret_", "return", "momentum"),
        exclude_terms=("vol", "volume", "liquid"),
        reference_features=reference,
    )
    volatility_reference = _coalesce_regime_signal(
        reference,
        include_terms=("vol", "range", "atr", "dispersion"),
        exclude_terms=("volume", "liquid"),
        reference_features=reference,
    )
    liquidity_reference = _coalesce_regime_signal(
        reference,
        include_terms=("liquid", "volume", "turnover", "trade"),
        exclude_terms=("illiquid",),
        reference_features=reference,
    )
    illiquidity_reference = _coalesce_regime_signal(
        reference,
        include_terms=("illiquid", "amihud"),
        reference_features=reference,
    )
    liquidity_reference = liquidity_reference - illiquidity_reference

    lower_quantile = float(config.get("lower_quantile", 0.33))
    upper_quantile = float(config.get("upper_quantile", 0.67))
    liquidity_invert = bool(config.get("liquidity_invert", False))

    trend_regime = _bucket_regime_signal(
        trend_score,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        reference_series=trend_reference,
    )
    volatility_regime = _bucket_regime_signal(
        volatility_score,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        reference_series=volatility_reference,
    )
    liquidity_regime = _bucket_regime_signal(
        liquidity_score,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        invert=liquidity_invert,
        reference_series=liquidity_reference,
    )

    composite = (
        (trend_regime + 1) * 9
        + (volatility_regime + 1) * 3
        + (liquidity_regime + 1)
    ).astype(int)

    return pd.DataFrame(
        {
            "trend_regime": trend_regime.astype(int),
            "volatility_regime": volatility_regime.astype(int),
            "liquidity_regime": liquidity_regime.astype(int),
            "regime": composite,
        },
        index=clean.index,
    )


def _detect_hmm_regime(features, n_regimes=2, config=None, fit_features=None):
    """Gaussian HMM-based regime detection with stable state ordering.

    States are sorted by ascending L1 norm of their mean vectors so that state 0
    is the most neutral and the last state is the most deviant. This prevents the
    cross-fold label-flipping that makes raw KMeans cluster IDs inconsistent.

    Parameters
    ----------
    features : pd.DataFrame – numeric columns describing regime state
    n_regimes : int         – number of hidden states
    config : dict or None   – optional HMM settings:
        covariance_type (default "diag"), n_iter (default 100),
        tol (default 1e-3), random_state (default 42)
    fit_features : pd.DataFrame or None – reference training slice used to fit
        the scaler and HMM transition/emission parameters.
    """
    try:
        from hmmlearn.hmm import GaussianHMM  # deferred – not a hard runtime dep
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "hmmlearn is required for method='hmm'. "
            "Install it with: pip install hmmlearn>=0.3"
        ) from exc

    config = dict(config or {})
    clean = features.dropna()
    if clean.empty:
        return pd.Series(dtype=int, name="regime")

    reference = clean if fit_features is None else fit_features.reindex(columns=features.columns).dropna()
    if reference.empty:
        reference = clean

    n_states = max(1, min(int(n_regimes), len(reference)))
    if n_states == 1:
        return pd.Series(0, index=clean.index, name="regime", dtype=int)

    covariance_type = config.get("covariance_type", "diag")
    n_iter = int(config.get("n_iter", 100))
    tol = float(config.get("tol", 1e-3))
    random_state = int(config.get("random_state", 42))

    scaler = StandardScaler()
    scaler.fit(reference)
    normed_reference = scaler.transform(reference)
    normed = scaler.transform(clean)

    hmm_model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
    )
    try:
        hmm_model.fit(normed_reference)
        raw_labels = hmm_model.predict(normed)
    except Exception:  # noqa: BLE001 – fall back gracefully on convergence failures
        return pd.Series(0, index=clean.index, name="regime", dtype=int)

    # Stable ordering: sort states by ascending L1 norm of state means.
    # State 0 = most neutral; last state = most extreme. This keeps regime
    # semantics consistent across folds even though HMM init is random.
    norms = np.linalg.norm(hmm_model.means_, ord=1, axis=1)
    sort_order = np.argsort(norms)  # old_label -> rank position
    remap = np.empty(n_states, dtype=int)
    for new_label, old_label in enumerate(sort_order):
        remap[old_label] = new_label

    return pd.Series(remap[raw_labels], index=clean.index, name="regime", dtype=int)


def detect_regime(features, n_regimes=2, method="hmm", config=None, fit_features=None):
    """Detect market regimes from a feature frame.

    Parameters
    ----------
    features : pd.DataFrame – numeric columns describing regime state
    n_regimes : int         – number of regimes / hidden states
    method : str            – "hmm" (default), "explicit", or "kmeans" (deprecated)
    config : dict or None   – optional method-specific settings
    fit_features : pd.DataFrame or None – reference slice used to fit scaler,
        HMM parameters, quantiles, or clusters before applying to ``features``.

    Returns a pd.Series (hmm/kmeans) or a pd.DataFrame (explicit).
    """
    method = (method or "hmm").lower()
    if method == "explicit":
        return _detect_explicit_regime(features, config=config, fit_features=fit_features)
    if method == "hmm":
        return _detect_hmm_regime(features, n_regimes=n_regimes, config=config, fit_features=fit_features)

    if method == "kmeans":
        warnings.warn(
            "detect_regime method='kmeans' is deprecated. KMeans cluster IDs are "
            "arbitrary across folds and degrade model quality through inconsistent "
            "regime labels. Use method='hmm' (default) or method='explicit' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    clean = features.dropna()
    reference = clean if fit_features is None else fit_features.reindex(columns=features.columns).dropna()
    if clean.empty:
        return pd.Series(dtype=int, name="regime")
    if reference.empty:
        reference = clean

    n_clusters = max(1, min(int(n_regimes), len(reference), len(clean)))
    if n_clusters == 1:
        return pd.Series(0, index=clean.index, name="regime", dtype=int)

    scaler = StandardScaler()
    scaler.fit(reference)
    normed_reference = scaler.transform(reference)
    normed = scaler.transform(clean)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(normed_reference)
    return pd.Series(km.predict(normed), index=clean.index, name="regime")


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
