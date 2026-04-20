"""Training, meta-labeling, validation splitters, and model store."""

import hashlib
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .execution import resolve_liquidity_inputs
from .features import ENDOGENOUS_FEATURE_FAMILIES, resolve_feature_family
from .labeling import sequential_bootstrap
from .regime import detect_regime
from .slippage import _estimate_reference_trade_slippage_rates
from .storage import file_sha256, read_json, write_json

try:  # pragma: no cover - exercised through save/load tests
    from skops.io import dump as skops_dump
    from skops.io import get_untrusted_types as skops_get_untrusted_types
    from skops.io import load as skops_load
except ImportError:  # pragma: no cover
    skops_dump = None
    skops_get_untrusted_types = None
    skops_load = None


# ───────────────────────────────────────────────────────────────────────────
# Validation splits
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


def cpcv_split(X, n_blocks=4, test_blocks=None, embargo=0):
    """Yield CPCV train/test index arrays plus split metadata.

    Parameters
    ----------
    X : array-like
        Only the length is used.
    n_blocks : int
        Number of contiguous blocks to partition the sample into.
    test_blocks : int or None
        Number of blocks to designate as the test path in each combination.
        Defaults to half the available blocks, rounded down.
    embargo : int
        Number of rows to embargo immediately after each test block.
    """
    n = len(X)
    if n < 2:
        return

    block_count = max(2, min(int(n_blocks), n))
    blocks = [
        np.asarray(block, dtype=int)
        for block in np.array_split(np.arange(n, dtype=int), block_count)
        if len(block) > 0
    ]
    block_count = len(blocks)
    if block_count < 2:
        return

    if test_blocks is None:
        test_block_count = max(1, block_count // 2)
    else:
        test_block_count = int(test_blocks)
    test_block_count = max(1, min(test_block_count, block_count - 1))

    embargo = max(0, int(embargo))

    for split_number, test_block_ids in enumerate(combinations(range(block_count), test_block_count)):
        test_parts = [blocks[block_id] for block_id in test_block_ids]
        test_idx = np.concatenate(test_parts)

        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_idx] = True

        embargo_mask = np.zeros(n, dtype=bool)
        for block_id in test_block_ids:
            block = blocks[block_id]
            embargo_start = int(block[-1]) + 1
            embargo_end = min(n, embargo_start + embargo)
            if embargo_start < embargo_end:
                embargo_mask[embargo_start:embargo_end] = True

        train_mask = ~(test_mask | embargo_mask)
        train_idx = np.flatnonzero(train_mask)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        yield (
            train_idx,
            test_idx,
            {
                "split_id": f"cpcv_{split_number}",
                "validation_method": "cpcv",
                "block_count": int(block_count),
                "test_block_count": int(test_block_count),
                "train_blocks": tuple(block_id for block_id in range(block_count) if block_id not in test_block_ids),
                "test_blocks": tuple(test_block_ids),
                "embargo_rows": int((embargo_mask & ~test_mask).sum()),
            },
        )


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
            bootstrap=bool(model_params.get("bootstrap", True)),
            oob_score=bool(model_params.get("oob_score", False)),
            max_samples=model_params.get("max_samples"),
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


def _as_weight_series(sample_weight, index):
    if sample_weight is None:
        return None
    if isinstance(sample_weight, pd.Series):
        return sample_weight.reindex(index).astype(float)
    return pd.Series(np.asarray(sample_weight, dtype=float), index=index, dtype=float)


def _resolve_rf_sampling_report(model_params, sampling_metadata):
    model_params = dict(model_params or {})
    sampling_metadata = dict(sampling_metadata or {})
    seq_config = dict(sampling_metadata.get("sequential_bootstrap") or {})
    random_state = seq_config.get("random_state", model_params.get("random_state", 42))
    mean_uniqueness = sampling_metadata.get("mean_uniqueness")
    mean_uniqueness = float(mean_uniqueness) if mean_uniqueness is not None else None
    uniqueness_threshold = float(seq_config.get("uniqueness_threshold", 0.90))
    enabled = bool(seq_config.get("enabled", True))
    high_concurrency = bool(mean_uniqueness is not None and mean_uniqueness < uniqueness_threshold)
    return {
        "sequential_bootstrap_enabled": enabled,
        "sequential_bootstrap_used": False,
        "reason": "not_applicable",
        "warning": None,
        "mean_uniqueness": mean_uniqueness,
        "uniqueness_threshold": uniqueness_threshold,
        "high_concurrency": high_concurrency,
        "random_state": int(random_state) if random_state is not None else None,
        "bootstrap_sample_size": int(seq_config.get("n_samples")) if seq_config.get("n_samples") is not None else None,
    }


def train_model(
    X,
    y,
    sample_weight=None,
    model_type="gbm",
    model_params=None,
    sampling_metadata=None,
    return_report=False,
    emit_warnings=True,
):
    """Train a classifier.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series           – labels (may include 0 = abstain)
    sample_weight : pd.Series or None
    model_type : str        – "rf" | "gbm" | "logistic"
    sampling_metadata : dict or None
        Optional fold-local label metadata and uniqueness diagnostics used to
        activate sequential bootstrapping for RandomForest training.

    Returns the fitted model.
    """
    model_params = dict(model_params or {})
    sampling_report = {
        "sequential_bootstrap_enabled": False,
        "sequential_bootstrap_used": False,
        "reason": "not_applicable",
        "warning": None,
        "mean_uniqueness": None,
        "uniqueness_threshold": None,
        "high_concurrency": False,
        "random_state": model_params.get("random_state", 42),
        "bootstrap_sample_size": None,
    }

    X_train = X
    y_train = y
    sw_series = _as_weight_series(sample_weight, X.index)
    fit_model_params = dict(model_params)

    if model_type == "rf":
        sampling_report = _resolve_rf_sampling_report(model_params, sampling_metadata)
        if sampling_report["high_concurrency"]:
            if sampling_report["sequential_bootstrap_enabled"]:
                labels = (sampling_metadata or {}).get("labels")
                close = (sampling_metadata or {}).get("close")
                n_samples = sampling_report["bootstrap_sample_size"] or len(X)
                if labels is None or close is None or len(labels) == 0 or len(close) == 0:
                    sampling_report["reason"] = "missing_sampling_metadata"
                    sampling_report["warning"] = (
                        "RandomForest received high-concurrency labels without label metadata for sequential bootstrapping."
                    )
                else:
                    bootstrap_idx = sequential_bootstrap(
                        labels,
                        close,
                        n_samples=n_samples,
                        random_state=sampling_report["random_state"],
                    )
                    if len(bootstrap_idx) > 0:
                        X_train = X.iloc[bootstrap_idx]
                        y_train = y.iloc[bootstrap_idx]
                        if sw_series is not None:
                            sw_series = sw_series.iloc[bootstrap_idx]
                        fit_model_params["bootstrap"] = False
                        fit_model_params["oob_score"] = False
                        sampling_report["sequential_bootstrap_used"] = True
                        sampling_report["reason"] = "high_concurrency_resampled"
                        sampling_report["bootstrap_sample_size"] = int(len(bootstrap_idx))
                    else:
                        sampling_report["reason"] = "empty_bootstrap_sample"
            else:
                sampling_report["reason"] = "disabled_on_high_concurrency"
                sampling_report["warning"] = (
                    "RandomForest is training on high-concurrency labels with sequential bootstrap disabled."
                )
        else:
            sampling_report["reason"] = "uniqueness_above_threshold"

        if sampling_report["warning"] and emit_warnings:
            warnings.warn(sampling_report["warning"], RuntimeWarning)

    model = build_model(model_type=model_type, model_params=fit_model_params)
    sw = sw_series.values if sw_series is not None else None
    fit_kwargs = {}
    if sw is not None:
        fit_kwargs["sample_weight"] = sw
        if isinstance(model, Pipeline):
            fit_kwargs["model__sample_weight"] = sw
            fit_kwargs.pop("sample_weight", None)

    model.fit(X_train, y_train, **fit_kwargs)
    if return_report:
        return model, sampling_report
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
                                  funding_rates=None, cutoff_timestamp=None,
                                  equity=10_000.0, volume=None,
                                  slippage_model=None, orderbook_depth=None,
                                  liquidity_lag_bars=1):
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
    fee_cost_rate = 2.0 * float(fee_rate)
    liquidity_inputs = resolve_liquidity_inputs(
        index=execution_series.index,
        volume=volume,
        orderbook_depth=orderbook_depth,
        slippage_model=slippage_model,
        liquidity_lag_bars=liquidity_lag_bars,
    )
    slippage_bar_rates = _estimate_reference_trade_slippage_rates(
        equity=equity,
        execution_series=execution_series,
        slippage_rate=slippage_rate,
        slippage_model=slippage_model,
        volume=liquidity_inputs["volume"],
        orderbook_depth=liquidity_inputs["orderbook_depth"],
    )

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
            "entry_fee_rate": (0.0 if direction == 0.0 else np.nan),
            "exit_fee_rate": (0.0 if direction == 0.0 else np.nan),
            "fee_cost_rate": (0.0 if direction == 0.0 else np.nan),
            "entry_slippage_rate": (0.0 if direction == 0.0 else np.nan),
            "exit_slippage_rate": (0.0 if direction == 0.0 else np.nan),
            "slippage_cost_rate": (0.0 if direction == 0.0 else np.nan),
            "round_trip_cost_rate": (0.0 if direction == 0.0 else np.nan),
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

        entry_slippage_rate = float(slippage_bar_rates.iloc[active_start])
        exit_slippage_rate = float(slippage_bar_rates.iloc[active_end])
        slippage_cost_rate = entry_slippage_rate + exit_slippage_rate
        round_trip_cost_rate = fee_cost_rate + slippage_cost_rate
        net_trade_return = gross_trade_return + funding_trade_return - round_trip_cost_rate
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
                "entry_fee_rate": float(fee_rate),
                "exit_fee_rate": float(fee_rate),
                "fee_cost_rate": fee_cost_rate,
                "entry_slippage_rate": entry_slippage_rate,
                "exit_slippage_rate": exit_slippage_rate,
                "slippage_cost_rate": slippage_cost_rate,
                "round_trip_cost_rate": round_trip_cost_rate,
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


def _replace_columns_with_train_medians(frame, columns, train_medians):
    if not columns:
        return frame

    updated = frame.copy()
    replacement = train_medians.reindex(columns).fillna(0.0).to_numpy()
    updated.loc[:, columns] = replacement
    return updated


def _family_bundle_specs(present_families):
    present = set(present_families)
    endogenous_families = {family for family in present if family in ENDOGENOUS_FEATURE_FAMILIES}
    specs = []

    if endogenous_families:
        specs.append(("endogenous_only", endogenous_families))
    if "futures_context" in present:
        specs.append(("endogenous_plus_futures", endogenous_families | {"futures_context"}))
    if "cross_asset" in present:
        specs.append(("endogenous_plus_cross_asset", endogenous_families | {"cross_asset"}))
    if "custom_exogenous" in present:
        specs.append(("endogenous_plus_custom", endogenous_families | {"custom_exogenous"}))
    if present:
        specs.append(("full_context", present))

    deduped = []
    seen = set()
    for name, families in specs:
        family_key = tuple(sorted(families))
        if not families or family_key in seen:
            continue
        seen.add(family_key)
        deduped.append((name, families))
    return deduped


def compute_feature_family_diagnostics(model, X_train, X_test, y_test, feature_blocks, baseline_metrics=None):
    """Measure feature-family contribution and context bundle dependence."""
    baseline_metrics = dict(baseline_metrics or evaluate_model(model, X_test, y_test))
    feature_blocks = dict(feature_blocks or {})
    feature_families = {
        column: resolve_feature_family(feature_blocks.get(column, "unknown"))
        for column in X_test.columns
    }
    native_importance = get_feature_importance(model, list(X_test.columns))
    family_columns = {}

    for column in X_test.columns:
        family = feature_families.get(column, "unknown")
        family_columns.setdefault(family, []).append(column)

    train_medians = X_train.median(numeric_only=True)
    families = []
    for family_name, columns in sorted(family_columns.items()):
        ablated_metrics = evaluate_model(
            model,
            _replace_columns_with_train_medians(X_test, columns, train_medians),
            y_test,
        )
        family_info = {
            "family": family_name,
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
            family_info["log_loss_increase"] = round(
                float(ablated_metrics["log_loss"] - baseline_metrics["log_loss"]),
                6,
            )
        families.append(family_info)

    families.sort(
        key=lambda item: (
            item.get("f1_drop", 0.0),
            item.get("accuracy_drop", 0.0),
            item.get("native_importance", 0.0),
        ),
        reverse=True,
    )

    bundles = []
    for bundle_name, allowed_families in _family_bundle_specs(family_columns):
        kept_columns = [
            column
            for column in X_test.columns
            if feature_families.get(column, "unknown") in allowed_families
        ]
        masked_columns = [column for column in X_test.columns if column not in kept_columns]
        bundle_metrics = baseline_metrics if not masked_columns else evaluate_model(
            model,
            _replace_columns_with_train_medians(X_test, masked_columns, train_medians),
            y_test,
        )
        bundles.append(
            {
                "bundle": bundle_name,
                "families": sorted(allowed_families),
                "feature_count": len(kept_columns),
                "accuracy": round(float(bundle_metrics.get("accuracy", 0.0)), 6),
                "f1_macro": round(float(bundle_metrics.get("f1_macro", 0.0)), 6),
                "accuracy_drop_vs_full": round(
                    float(baseline_metrics.get("accuracy", 0.0) - bundle_metrics.get("accuracy", 0.0)),
                    6,
                ),
                "f1_drop_vs_full": round(
                    float(baseline_metrics.get("f1_macro", 0.0) - bundle_metrics.get("f1_macro", 0.0)),
                    6,
                ),
            }
        )

    bundles.sort(
        key=lambda item: (
            item.get("f1_drop_vs_full", 0.0),
            item.get("accuracy_drop_vs_full", 0.0),
        )
    )

    selected_families = sorted(family_columns)
    return {
        "baseline_metrics": baseline_metrics,
        "families": families,
        "bundles": bundles,
        "selected_families": selected_families,
        "endogenous_only_selected": bool(selected_families) and set(selected_families).issubset(ENDOGENOUS_FEATURE_FAMILIES),
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


def summarize_feature_family_diagnostics(fold_diagnostics):
    """Aggregate feature-family diagnostics across validation folds."""
    if not fold_diagnostics:
        return {
            "summary": [],
            "bundles": [],
            "folds": [],
            "selected_families": [],
            "endogenous_only_selected_any_fold": False,
            "endogenous_only_selected_all_folds": False,
        }

    family_totals = {}
    bundle_totals = {}
    selected_families = set()
    endogenous_flags = []

    for diagnostics in fold_diagnostics:
        selected_families.update(diagnostics.get("selected_families", []))
        endogenous_flags.append(bool(diagnostics.get("endogenous_only_selected")))

        for family_info in diagnostics.get("families", []):
            family_name = family_info["family"]
            summary = family_totals.setdefault(
                family_name,
                {
                    "family": family_name,
                    "feature_count": family_info["feature_count"],
                    "folds": 0,
                    "native_importance": [],
                    "accuracy_drop": [],
                    "f1_drop": [],
                    "log_loss_increase": [],
                },
            )
            summary["feature_count"] = max(summary["feature_count"], family_info["feature_count"])
            summary["folds"] += 1
            summary["native_importance"].append(family_info.get("native_importance", 0.0))
            summary["accuracy_drop"].append(family_info.get("accuracy_drop", 0.0))
            summary["f1_drop"].append(family_info.get("f1_drop", 0.0))
            if family_info.get("log_loss_increase") is not None:
                summary["log_loss_increase"].append(family_info["log_loss_increase"])

        for bundle_info in diagnostics.get("bundles", []):
            bundle_name = bundle_info["bundle"]
            summary = bundle_totals.setdefault(
                bundle_name,
                {
                    "bundle": bundle_name,
                    "families": bundle_info.get("families", []),
                    "feature_count": bundle_info["feature_count"],
                    "folds": 0,
                    "accuracy": [],
                    "f1_macro": [],
                    "accuracy_drop_vs_full": [],
                    "f1_drop_vs_full": [],
                },
            )
            summary["families"] = bundle_info.get("families", summary["families"])
            summary["feature_count"] = max(summary["feature_count"], bundle_info["feature_count"])
            summary["folds"] += 1
            summary["accuracy"].append(bundle_info.get("accuracy", 0.0))
            summary["f1_macro"].append(bundle_info.get("f1_macro", 0.0))
            summary["accuracy_drop_vs_full"].append(bundle_info.get("accuracy_drop_vs_full", 0.0))
            summary["f1_drop_vs_full"].append(bundle_info.get("f1_drop_vs_full", 0.0))

    summary_rows = []
    for family_name, totals in family_totals.items():
        summary_rows.append(
            {
                "family": family_name,
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

    bundle_rows = []
    for bundle_name, totals in bundle_totals.items():
        bundle_rows.append(
            {
                "bundle": bundle_name,
                "families": totals["families"],
                "feature_count": totals["feature_count"],
                "folds": totals["folds"],
                "avg_accuracy": round(float(np.mean(totals["accuracy"])), 6),
                "avg_f1_macro": round(float(np.mean(totals["f1_macro"])), 6),
                "avg_accuracy_drop_vs_full": round(float(np.mean(totals["accuracy_drop_vs_full"])), 6),
                "avg_f1_drop_vs_full": round(float(np.mean(totals["f1_drop_vs_full"])), 6),
            }
        )

    bundle_rows.sort(
        key=lambda item: (
            item.get("avg_f1_drop_vs_full", 0.0),
            item.get("avg_accuracy_drop_vs_full", 0.0),
        )
    )

    return {
        "summary": summary_rows,
        "bundles": bundle_rows,
        "folds": fold_diagnostics,
        "selected_families": sorted(selected_families),
        "endogenous_only_selected_any_fold": any(endogenous_flags),
        "endogenous_only_selected_all_folds": all(endogenous_flags),
    }

# ───────────────────────────────────────────────────────────────────────────
# Safe artifact store
# ───────────────────────────────────────────────────────────────────────────

_SAFE_MODEL_FORMAT_VERSION = 1


def _artifact_paths(path):
    target = Path(path)
    if target.suffix in {".json", ".skops", ".pkl", ".pickle"}:
        base = target.with_suffix("")
    else:
        base = target
    manifest_path = base.with_suffix(".json")
    model_path = base.with_suffix(".skops")
    return manifest_path, model_path


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    return value


def _coerce_feature_schema(metadata, feature_schema=None):
    metadata = dict(metadata or {})
    schema = dict(feature_schema or metadata.get("feature_schema") or {})
    explicit_columns = schema.get("feature_order") or metadata.get("feature_order") or metadata.get("feature_columns")
    if explicit_columns is None:
        explicit_columns = metadata.get("last_selected_columns")
    if explicit_columns is None:
        explicit_columns = metadata.get("required_columns")
    if explicit_columns is not None:
        schema["feature_order"] = list(explicit_columns)
        schema.setdefault("required_columns", list(explicit_columns))

    required_columns = schema.get("required_columns")
    if required_columns is not None:
        schema["required_columns"] = list(required_columns)

    schema_version = schema.get("schema_version") or metadata.get("schema_version") or metadata.get("feature_schema_version")
    if schema_version is not None:
        schema["schema_version"] = schema_version
    return schema


def _build_model_manifest(metadata, feature_schema, model_path, trusted_types, serializer):
    clean_metadata = _json_ready(metadata or {})
    manifest = {
        "format_version": _SAFE_MODEL_FORMAT_VERSION,
        "serializer": serializer,
        "model_path": model_path.name,
        "model_sha256": file_sha256(model_path) if model_path.exists() else None,
        "feature_schema": _json_ready(feature_schema or {}),
        "metadata": clean_metadata,
        "trusted_types": list(trusted_types or []),
        "sklearn_version": sklearn_version,
    }
    if serializer == "recipe_only":
        manifest["retrain_recipe"] = clean_metadata.get("retrain_recipe") or clean_metadata.get("training_recipe")
    return manifest


def _verify_model_hash(model_path, manifest):
    expected_hash = manifest.get("model_sha256")
    if not expected_hash:
        return
    actual_hash = file_sha256(model_path)
    if actual_hash != expected_hash:
        raise ValueError(f"Model artifact hash verification failed for {model_path}")


def _verify_feature_schema(feature_schema, expected_feature_columns=None):
    if not expected_feature_columns:
        return

    expected = list(expected_feature_columns)
    required = list((feature_schema or {}).get("required_columns") or [])
    if required and expected != required:
        raise ValueError(
            "Feature schema mismatch: expected feature order "
            f"{required} but received {expected}"
        )

    feature_order = list((feature_schema or {}).get("feature_order") or [])
    if feature_order and expected != feature_order:
        raise ValueError(
            "Feature order mismatch: artifact requires "
            f"{feature_order} but received {expected}"
        )


def save_model(model, path, metadata=None, feature_schema=None):
    """Persist a sklearn-compatible model using skops plus a JSON manifest."""
    manifest_path, model_path = _artifact_paths(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    feature_schema = _coerce_feature_schema(metadata, feature_schema=feature_schema)

    if skops_dump is None or skops_get_untrusted_types is None:
        raise ImportError(
            "Safe model persistence requires skops. Install it with `python -m pip install skops`."
        )

    serializer = "skops"
    trusted_types = []
    try:
        skops_dump(model, model_path)
        trusted_types = list(skops_get_untrusted_types(file=str(model_path)))
    except Exception as exc:
        model_path.unlink(missing_ok=True)
        retrain_recipe = dict(metadata or {}).get("retrain_recipe") or dict(metadata or {}).get("training_recipe")
        if retrain_recipe is None:
            raise RuntimeError(
                "Model could not be safely serialized with skops and no retrain recipe was supplied"
            ) from exc
        serializer = "recipe_only"

    manifest = _build_model_manifest(
        metadata=metadata,
        feature_schema=feature_schema,
        model_path=model_path,
        trusted_types=trusted_types,
        serializer=serializer,
    )
    write_json(manifest_path, manifest)
    return manifest_path


def load_model(path, expected_feature_columns=None):
    """Return (model, metadata) from a manifest-verified safe artifact bundle."""
    manifest_path, model_path = _artifact_paths(path)
    manifest = read_json(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"Model manifest not found at {manifest_path}")

    serializer = manifest.get("serializer", "skops")
    feature_schema = dict(manifest.get("feature_schema") or {})
    _verify_feature_schema(feature_schema, expected_feature_columns=expected_feature_columns)

    metadata = dict(manifest.get("metadata") or {})
    metadata.setdefault("feature_schema", feature_schema)
    metadata.setdefault("artifact_manifest", manifest_path)

    if serializer == "recipe_only":
        raise RuntimeError(
            "This artifact stores only a retraining recipe because the model could not be safely serialized"
        )

    if skops_load is None:
        raise ImportError(
            "Safe model loading requires skops. Install it with `python -m pip install skops`."
        )

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    _verify_model_hash(model_path, manifest)
    trusted_types = list(manifest.get("trusted_types") or [])
    model = skops_load(str(model_path), trusted=trusted_types)
    return model, metadata
