"""Regime-aware feature engineering and walk-forward training helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .models import predict_probability_frame, train_model, walk_forward_split


def _coerce_regime_frame(regime_data, index=None):
    if regime_data is None:
        frame = pd.DataFrame()
    elif isinstance(regime_data, pd.Series):
        column_name = regime_data.name or "regime"
        frame = regime_data.to_frame(name=column_name)
    else:
        frame = pd.DataFrame(regime_data).copy()
    if index is not None:
        frame = frame.reindex(index)
    return frame


def summarize_regime_coverage(regime_data, regime_column="regime", config=None):
    regime_frame = _coerce_regime_frame(regime_data)
    config = dict(config or {})
    max_dominant_share = float(config.get("max_dominant_share", 0.8))
    min_distinct_regimes = int(config.get("min_distinct_regimes", 2))

    if regime_frame.empty:
        return {
            "available_rows": 0,
            "distinct_regimes": 0,
            "dominant_regime": None,
            "dominant_share": None,
            "regime_distribution": {},
            "coverage_ok": False,
            "reasons": ["regime_data_unavailable"],
        }

    target_column = regime_column if regime_column in regime_frame.columns else regime_frame.columns[0]
    labels = pd.Series(regime_frame[target_column], copy=False).dropna()
    if labels.empty:
        return {
            "available_rows": 0,
            "distinct_regimes": 0,
            "dominant_regime": None,
            "dominant_share": None,
            "regime_distribution": {},
            "coverage_ok": False,
            "reasons": ["regime_labels_missing"],
        }

    distribution = labels.value_counts(normalize=True, dropna=True)
    dominant_regime = distribution.index[0]
    dominant_share = float(distribution.iloc[0])
    distinct_regimes = int(labels.nunique())
    reasons = []
    if distinct_regimes < min_distinct_regimes:
        reasons.append("insufficient_regime_diversity")
    if dominant_share > max_dominant_share:
        reasons.append("dominant_regime_exceeds_threshold")
    return {
        "available_rows": int(len(labels)),
        "distinct_regimes": distinct_regimes,
        "dominant_regime": dominant_regime,
        "dominant_share": dominant_share,
        "regime_distribution": {str(key): float(value) for key, value in distribution.items()},
        "coverage_ok": not reasons,
        "reasons": reasons,
    }


@dataclass
class RegimeAwareFeatureFrame:
    frame: pd.DataFrame
    regime_columns: list[str] = field(default_factory=list)
    normalized_columns: list[str] = field(default_factory=list)
    interaction_columns: list[str] = field(default_factory=list)


def build_regime_aware_feature_frame(X, regime_data, config=None):
    config = dict(config or {})
    base = pd.DataFrame(X).copy()
    base = base.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    regime_frame = _coerce_regime_frame(regime_data, index=base.index)
    if regime_frame.empty:
        return RegimeAwareFeatureFrame(frame=base)

    max_dummy_cardinality = int(config.get("max_dummy_cardinality", 16))
    max_interaction_features = int(config.get("max_interaction_features", 4))
    max_interaction_regimes = int(config.get("max_interaction_regimes", 6))
    add_interactions = bool(config.get("regime_interactions", True))

    numeric_regime = regime_frame.select_dtypes(include=[np.number]).copy()
    regime_state_columns = []
    for column in numeric_regime.columns:
        engineered_name = f"regime_state__{column}"
        base[engineered_name] = numeric_regime[column].fillna(0.0).astype(float)
        regime_state_columns.append(engineered_name)

    dummy_frames = []
    dummy_columns = []
    for column in regime_frame.columns:
        non_null = regime_frame[column].dropna()
        if non_null.empty or int(non_null.nunique()) > max_dummy_cardinality:
            continue
        encoded = regime_frame[column].fillna("missing").astype(str)
        dummies = pd.get_dummies(encoded, prefix=f"regime__{column}", dtype=float)
        dummy_frames.append(dummies)
        dummy_columns.extend(list(dummies.columns))
    if dummy_frames:
        base = pd.concat([base] + dummy_frames, axis=1)

    normalized_columns = []
    volatility_candidates = [
        column for column in numeric_regime.columns
        if any(term in column.lower() for term in ("vol", "atr", "range", "dispersion"))
    ]
    return_like_columns = [
        column for column in base.columns
        if any(term in column.lower() for term in ("ret", "return", "momentum", "slope"))
        and not column.startswith(("regime_state__", "regime__", "vol_norm__", "cond__"))
    ]
    if volatility_candidates and return_like_columns:
        volatility_scale = numeric_regime[volatility_candidates[0]].abs().replace(0.0, np.nan).fillna(1.0)
        volatility_scale = volatility_scale.clip(lower=1e-6)
        for column in return_like_columns:
            engineered_name = f"vol_norm__{column}"
            base[engineered_name] = pd.to_numeric(base[column], errors="coerce").fillna(0.0) / volatility_scale
            normalized_columns.append(engineered_name)

    interaction_columns = []
    if add_interactions and dummy_columns:
        candidate_columns = [
            column for column in base.columns
            if column not in dummy_columns
            and not column.startswith(("regime_state__", "vol_norm__", "cond__"))
        ][:max_interaction_features]
        for feature_name in candidate_columns:
            feature_values = pd.to_numeric(base[feature_name], errors="coerce").fillna(0.0)
            for dummy_name in dummy_columns[:max_interaction_regimes]:
                interaction_name = f"cond__{feature_name}__{dummy_name}"
                base[interaction_name] = feature_values * base[dummy_name]
                interaction_columns.append(interaction_name)

    return RegimeAwareFeatureFrame(
        frame=base,
        regime_columns=regime_state_columns + dummy_columns,
        normalized_columns=normalized_columns,
        interaction_columns=interaction_columns,
    )


def _evaluate_regime_aware_predictions(y_true, predictions):
    y_series = pd.Series(y_true, index=predictions.index if isinstance(predictions, pd.Series) else None)
    pred_series = pd.Series(predictions, index=y_series.index)
    metrics = {
        "accuracy": round(float(accuracy_score(y_series, pred_series)), 4),
        "f1_macro": round(float(f1_score(y_series, pred_series, average="macro", zero_division=0)), 4),
        "prediction_coverage": round(float(pred_series.ne(0).mean()), 4),
    }
    directional_mask = y_series.ne(0)
    if directional_mask.any():
        metrics["directional_accuracy"] = round(
            float(accuracy_score(y_series.loc[directional_mask], pred_series.loc[directional_mask])),
            4,
        )
    return metrics


class RegimeAwareModelBundle:
    def __init__(
        self,
        *,
        strategy,
        model=None,
        fallback_model=None,
        specialist_models=None,
        feature_config=None,
        feature_columns=None,
        regime_column="regime",
        ordered_classes=(-1, 0, 1),
    ):
        self.strategy = str(strategy)
        self.model = model
        self.fallback_model = fallback_model
        self.specialist_models = dict(specialist_models or {})
        self.feature_config = dict(feature_config or {})
        self.feature_columns = list(feature_columns or [])
        self.regime_column = regime_column
        self.ordered_classes = tuple(ordered_classes)

    def _transform_feature_strategy(self, X, regime_data):
        feature_result = build_regime_aware_feature_frame(X, regime_data, config=self.feature_config)
        return feature_result.frame.reindex(columns=self.feature_columns, fill_value=0.0)

    def predict(self, X, regime_data):
        predictions, _ = self.predict_with_report(X, regime_data)
        return predictions.to_numpy()

    def predict_proba(self, X, regime_data):
        _, probability_frame, _ = self.predict_with_probability_report(X, regime_data)
        return probability_frame

    def predict_with_report(self, X, regime_data):
        predictions, _, report = self.predict_with_probability_report(X, regime_data)
        return predictions, report

    def predict_with_probability_report(self, X, regime_data):
        X_frame = pd.DataFrame(X).copy()
        regime_frame = _coerce_regime_frame(regime_data, index=X_frame.index)
        if self.strategy == "feature":
            transformed = self._transform_feature_strategy(X_frame, regime_frame)
            predictions = pd.Series(self.model.predict(transformed), index=X_frame.index)
            probabilities = predict_probability_frame(self.model, transformed, ordered_classes=self.ordered_classes)
            report = {"strategy": self.strategy, "fallback_rows": 0, "unseen_regimes": []}
            return predictions, probabilities, report

        if self.strategy != "specialist":
            raise ValueError(f"Unknown regime-aware strategy={self.strategy!r}")

        labels = pd.Series(
            regime_frame[self.regime_column] if self.regime_column in regime_frame.columns else regime_frame.iloc[:, 0],
            index=X_frame.index,
        ) if not regime_frame.empty else pd.Series(np.nan, index=X_frame.index)
        predictions = pd.Series(0, index=X_frame.index, dtype=int)
        probabilities = pd.DataFrame(0.0, index=X_frame.index, columns=list(self.ordered_classes), dtype=float)
        fallback_rows = 0
        unseen_regimes = []

        for regime_value, row_index in labels.groupby(labels).groups.items():
            model = self.specialist_models.get(regime_value)
            if model is None:
                model = self.fallback_model
                fallback_rows += int(len(row_index))
                unseen_regimes.append(regime_value)
            group_X = X_frame.loc[row_index]
            predictions.loc[row_index] = model.predict(group_X)
            probabilities.loc[row_index, :] = predict_probability_frame(
                model,
                group_X,
                ordered_classes=self.ordered_classes,
            ).to_numpy()

        missing_rows = labels.index[labels.isna()]
        if len(missing_rows) > 0:
            fallback_rows += int(len(missing_rows))
            fallback_predictions = self.fallback_model.predict(X_frame.loc[missing_rows])
            predictions.loc[missing_rows] = fallback_predictions
            probabilities.loc[missing_rows, :] = predict_probability_frame(
                self.fallback_model,
                X_frame.loc[missing_rows],
                ordered_classes=self.ordered_classes,
            ).to_numpy()
            unseen_regimes.append("missing")

        report = {
            "strategy": self.strategy,
            "fallback_rows": int(fallback_rows),
            "unseen_regimes": sorted({str(value) for value in unseen_regimes}),
        }
        return predictions, probabilities, report


def train_regime_aware_model(
    X,
    y,
    regime_data,
    *,
    strategy="feature",
    model_type="gbm",
    model_params=None,
    feature_config=None,
    regime_column="regime",
    min_samples_per_regime=40,
    sample_weight=None,
):
    X_frame = pd.DataFrame(X).copy()
    y_series = pd.Series(y, index=X_frame.index)
    regime_frame = _coerce_regime_frame(regime_data, index=X_frame.index)
    strategy = str(strategy).lower()

    if strategy == "feature":
        feature_result = build_regime_aware_feature_frame(X_frame, regime_frame, config=feature_config)
        model = train_model(
            feature_result.frame,
            y_series,
            sample_weight=sample_weight,
            model_type=model_type,
            model_params=model_params,
        )
        bundle = RegimeAwareModelBundle(
            strategy="feature",
            model=model,
            feature_config=feature_config,
            feature_columns=list(feature_result.frame.columns),
            regime_column=regime_column,
        )
        report = {
            "strategy": "feature",
            "feature_columns": list(feature_result.frame.columns),
            "regime_columns": list(feature_result.regime_columns),
            "normalized_columns": list(feature_result.normalized_columns),
            "interaction_columns": list(feature_result.interaction_columns),
            "regime_alignment": "inference_aligned_input",
        }
        return bundle, report

    if strategy != "specialist":
        raise ValueError("strategy must be 'feature' or 'specialist'")

    if regime_frame.empty:
        raise ValueError("specialist strategy requires regime_data")

    target_column = regime_column if regime_column in regime_frame.columns else regime_frame.columns[0]
    labels = pd.Series(regime_frame[target_column], index=X_frame.index)
    fallback_model = train_model(
        X_frame,
        y_series,
        sample_weight=sample_weight,
        model_type=model_type,
        model_params=model_params,
    )
    specialist_models = {}
    specialist_rows = {}
    skipped_regimes = {}
    for regime_value, row_index in labels.groupby(labels).groups.items():
        regime_X = X_frame.loc[row_index]
        regime_y = y_series.loc[row_index]
        if len(regime_X) < int(min_samples_per_regime):
            skipped_regimes[str(regime_value)] = "minimum_samples_not_met"
            continue
        if regime_y.nunique() < 2:
            skipped_regimes[str(regime_value)] = "single_class_regime"
            continue
        regime_weight = None
        if sample_weight is not None:
            regime_weight = pd.Series(sample_weight, index=X_frame.index).loc[row_index]
        specialist_models[regime_value] = train_model(
            regime_X,
            regime_y,
            sample_weight=regime_weight,
            model_type=model_type,
            model_params=model_params,
        )
        specialist_rows[str(regime_value)] = int(len(regime_X))

    bundle = RegimeAwareModelBundle(
        strategy="specialist",
        fallback_model=fallback_model,
        specialist_models=specialist_models,
        regime_column=target_column,
    )
    report = {
        "strategy": "specialist",
        "trained_regimes": [str(value) for value in specialist_models],
        "trained_rows_by_regime": specialist_rows,
        "skipped_regimes": skipped_regimes,
        "fallback_enabled": True,
        "regime_alignment": "inference_aligned_input",
    }
    return bundle, report


def train_regime_aware_walk_forward(
    X,
    y,
    regime_data,
    *,
    strategy="feature",
    model_type="gbm",
    model_params=None,
    feature_config=None,
    regime_column="regime",
    min_samples_per_regime=40,
    coverage_config=None,
    sample_weight=None,
    n_splits=3,
    train_size=None,
    test_size=None,
    gap=0,
    expanding=False,
):
    X_frame = pd.DataFrame(X).copy()
    y_series = pd.Series(y, index=X_frame.index)
    regime_frame = _coerce_regime_frame(regime_data, index=X_frame.index)

    folds = []
    oos_predictions = []
    oos_probabilities = []
    last_model = None

    for fold_number, (train_idx, test_idx) in enumerate(
        walk_forward_split(
            X_frame,
            n_splits=n_splits,
            train_size=train_size,
            test_size=test_size,
            gap=gap,
            expanding=expanding,
        )
    ):
        X_train = X_frame.iloc[train_idx]
        X_test = X_frame.iloc[test_idx]
        y_train = y_series.iloc[train_idx]
        y_test = y_series.iloc[test_idx]
        regime_train = regime_frame.iloc[train_idx]
        regime_test = regime_frame.iloc[test_idx]
        weight_train = None
        if sample_weight is not None:
            weight_train = pd.Series(sample_weight, index=X_frame.index).iloc[train_idx]

        train_coverage = summarize_regime_coverage(regime_train, regime_column=regime_column, config=coverage_config)
        test_coverage = summarize_regime_coverage(regime_test, regime_column=regime_column, config=coverage_config)

        bundle, training_report = train_regime_aware_model(
            X_train,
            y_train,
            regime_train,
            strategy=strategy,
            model_type=model_type,
            model_params=model_params,
            feature_config=feature_config,
            regime_column=regime_column,
            min_samples_per_regime=min_samples_per_regime,
            sample_weight=weight_train,
        )
        predictions, probabilities, inference_report = bundle.predict_with_probability_report(X_test, regime_test)
        metrics = _evaluate_regime_aware_predictions(y_test, predictions)

        folds.append(
            {
                "fold": fold_number,
                "train_index": X_train.index,
                "test_index": X_test.index,
                "coverage": {"train": train_coverage, "test": test_coverage},
                "training_report": training_report,
                "inference_report": inference_report,
                "metrics": metrics,
            }
        )
        oos_predictions.append(predictions)
        oos_probabilities.append(probabilities)
        last_model = bundle

    if last_model is None:
        raise RuntimeError("No walk-forward splits were generated for regime-aware training")

    combined_predictions = pd.concat(oos_predictions).sort_index()
    combined_probabilities = pd.concat(oos_probabilities).sort_index().reindex(combined_predictions.index)
    return {
        "strategy": strategy,
        "folds": folds,
        "last_model": last_model,
        "oos_predictions": combined_predictions,
        "oos_probabilities": combined_probabilities,
        "coverage_summary": {
            "train_ok_share": float(np.mean([fold["coverage"]["train"]["coverage_ok"] for fold in folds])),
            "test_ok_share": float(np.mean([fold["coverage"]["test"]["coverage_ok"] for fold in folds])),
        },
    }


__all__ = [
    "RegimeAwareFeatureFrame",
    "RegimeAwareModelBundle",
    "build_regime_aware_feature_frame",
    "summarize_regime_coverage",
    "train_regime_aware_model",
    "train_regime_aware_walk_forward",
]