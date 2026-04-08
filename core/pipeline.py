"""Stepwise research pipeline abstraction for reusable model workflows."""

import copy

import numpy as np
import pandas as pd

from .automl import run_automl_study
from .backtest import kelly_fraction, run_backtest
from .data import fetch_binance_vision
from .features import build_features, check_stationarity
from .indicators import run_indicators
from .labeling import (
    fixed_horizon_labels,
    sample_weights_by_uniqueness,
    triple_barrier_labels,
)
from .models import (
    detect_regime,
    evaluate_model,
    train_meta_model,
    train_model,
    walk_forward_split,
)


def _default_regime_features(pipeline):
    data = pipeline.require("data")
    return pd.DataFrame(
        {
            "vol_20": data["close"].pct_change().rolling(20).std(),
            "vol_60": data["close"].pct_change().rolling(60).std(),
        }
    ).dropna()


def _default_stationarity_specs(pipeline):
    specs = [{"name": "close", "source": "data", "column": "close"}]
    features = pipeline.state.get("features")
    if features is not None and "close_fracdiff" in features.columns:
        specs.append({"name": "close_fracdiff", "source": "features", "column": "close_fracdiff"})
    return specs


def _positive_class_probability(model, X, positive_class=1):
    probabilities = model.predict_proba(X)
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model")
        classes = getattr(estimator, "classes_", None)

    if probabilities.ndim == 1:
        return probabilities

    if probabilities.shape[1] == 1:
        if classes is not None and len(classes) == 1 and classes[0] == positive_class:
            return np.ones(len(X))
        return np.zeros(len(X))

    if classes is not None and positive_class in classes:
        positive_index = list(classes).index(positive_class)
        return probabilities[:, positive_index]

    return probabilities[:, -1]


class PipelineStep:
    name = "step"

    def run(self, pipeline):
        raise NotImplementedError


class FetchDataStep(PipelineStep):
    name = "fetch_data"

    def run(self, pipeline):
        config = pipeline.section("data")
        data = fetch_binance_vision(**config)
        pipeline.state["raw_data"] = data
        pipeline.state["data"] = data.copy()
        return data


class IndicatorsStep(PipelineStep):
    name = "run_indicators"

    def run(self, pipeline):
        data = pipeline.require("data")
        indicator_run = run_indicators(data, pipeline.section("indicators"))
        pipeline.state["indicator_run"] = indicator_run
        pipeline.state["data"] = indicator_run.frame
        return indicator_run


class AutoMLStep(PipelineStep):
    name = "run_automl"

    def run(self, pipeline):
        config = pipeline.section("automl")
        if not config.get("enabled"):
            return None

        summary = run_automl_study(
            pipeline,
            pipeline_class=ResearchPipeline,
            trial_step_classes=[
                FeaturesStep,
                RegimeStep,
                LabelsStep,
                AlignDataStep,
                SampleWeightsStep,
                TrainModelsStep,
                SignalsStep,
                BacktestStep,
            ],
        )

        best_overrides = copy.deepcopy(summary.get("best_overrides", {}))
        for section, values in best_overrides.items():
            current = pipeline.config.get(section, {})
            if isinstance(current, dict) and isinstance(values, dict):
                merged = dict(current)
                merged.update(values)
                pipeline.config[section] = merged
            else:
                pipeline.config[section] = values

        for key in [
            "features",
            "stationarity",
            "regime_features",
            "regimes",
            "labels",
            "X",
            "y",
            "labels_aligned",
            "sample_weights",
            "training",
            "signals",
            "backtest",
        ]:
            pipeline.state.pop(key, None)

        pipeline.state["automl"] = summary
        return summary


class FeaturesStep(PipelineStep):
    name = "build_features"

    def run(self, pipeline):
        data = pipeline.require("data")
        config = pipeline.section("features")
        features = build_features(
            data,
            lags=config.get("lags"),
            frac_diff_d=config.get("frac_diff_d"),
        )

        for builder in config.get("builders", []):
            built = builder(pipeline, features.copy())
            if built is not None:
                features = built

        pipeline.state["features"] = features
        return features


class StationarityStep(PipelineStep):
    name = "check_stationarity"

    def run(self, pipeline):
        config = pipeline.section("stationarity")
        specs = config.get("series") or _default_stationarity_specs(pipeline)
        results = {}

        for spec in specs:
            source = pipeline.require(spec.get("source", "data"))
            column = spec["column"]
            series = source[column]
            results[spec.get("name", column)] = check_stationarity(series.dropna())

        pipeline.state["stationarity"] = results
        return results


class RegimeStep(PipelineStep):
    name = "detect_regimes"

    def run(self, pipeline):
        features = pipeline.require("features")
        config = pipeline.section("regime")
        builder = config.get("builder") or _default_regime_features
        regime_features = builder(pipeline)
        regimes = detect_regime(regime_features, n_regimes=config.get("n_regimes", 2))

        aligned_features = features.loc[regimes.index].copy()
        aligned_features[config.get("column_name", "regime")] = regimes.values

        pipeline.state["regime_features"] = regime_features
        pipeline.state["regimes"] = regimes
        pipeline.state["features"] = aligned_features
        return {"regime_features": regime_features, "regimes": regimes}


class LabelsStep(PipelineStep):
    name = "build_labels"

    def run(self, pipeline):
        data = pipeline.require("data")
        config = pipeline.section("labels")
        label_kind = config.get("kind", "triple_barrier")

        if label_kind == "triple_barrier":
            volatility_builder = config.get("volatility_builder")
            if volatility_builder is None:
                window = config.get("volatility_window", 24)
                volatility = data["close"].pct_change().rolling(window).std()
            else:
                volatility = volatility_builder(pipeline)

            labels = triple_barrier_labels(
                close=data["close"],
                volatility=volatility,
                pt_sl=config.get("pt_sl", (2.0, 2.0)),
                max_holding=config.get("max_holding", 24),
                min_return=config.get("min_return", 0.0),
            )
        elif label_kind == "fixed_horizon":
            labels = fixed_horizon_labels(
                close=data["close"],
                horizon=config.get("horizon", 5),
                threshold=config.get("threshold", 0.0),
            )
        else:
            raise ValueError(f"Unsupported label kind={label_kind!r}")

        pipeline.state["labels"] = labels
        return labels


class AlignDataStep(PipelineStep):
    name = "align_data"

    def run(self, pipeline):
        features = pipeline.require("features")
        labels = pipeline.require("labels")
        label_column = pipeline.section("labels").get("label_column", "label")

        common = features.index.intersection(labels.index)
        X = features.loc[common].copy()
        y = labels.loc[common, label_column].copy()
        labels_aligned = labels.loc[common].copy()

        mask = X.notna().all(axis=1) & y.notna()
        X = X.loc[mask]
        y = y.loc[mask]
        labels_aligned = labels_aligned.loc[mask]

        pipeline.state["X"] = X
        pipeline.state["y"] = y
        pipeline.state["labels_aligned"] = labels_aligned
        return {"X": X, "y": y, "labels_aligned": labels_aligned}


class SampleWeightsStep(PipelineStep):
    name = "compute_sample_weights"

    def run(self, pipeline):
        labels_aligned = pipeline.require("labels_aligned")
        raw_data = pipeline.require("raw_data")
        weights = sample_weights_by_uniqueness(labels_aligned, raw_data["close"])
        pipeline.state["sample_weights"] = weights
        return weights


class TrainModelsStep(PipelineStep):
    name = "train_models"

    def run(self, pipeline):
        X = pipeline.require("X")
        y = pipeline.require("y")
        weights = pipeline.require("sample_weights")
        config = pipeline.section("model")

        fold_metrics = []
        last_model = None
        last_meta = None
        oos_predictions = []
        oos_meta_prob = []

        for fold, (train_idx, test_idx) in enumerate(
            walk_forward_split(
                X,
                n_splits=config.get("n_splits", 3),
                train_size=config.get("train_size"),
                test_size=config.get("test_size"),
                gap=config.get("gap", 0),
                expanding=config.get("expanding", False),
            )
        ):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            w_train = weights.iloc[train_idx]

            model = train_model(
                X_train,
                y_train,
                sample_weight=w_train,
                model_type=config.get("type", "rf"),
                model_params=config.get("params"),
            )
            metrics = evaluate_model(model, X_test, y_test)
            metrics["fold"] = fold

            train_primary_preds = model.predict(X_train)
            test_primary_preds = pd.Series(model.predict(X_test), index=X_test.index)
            meta_model = train_meta_model(train_primary_preds, X_train, y_train, sample_weight=w_train)

            X_meta_test = X_test.copy()
            X_meta_test["primary_pred"] = test_primary_preds.values
            meta_prob_test = pd.Series(_positive_class_probability(meta_model, X_meta_test), index=X_test.index)

            fold_metrics.append(metrics)
            last_model = model
            last_meta = meta_model
            oos_predictions.append(test_primary_preds)
            oos_meta_prob.append(meta_prob_test)

        if last_model is None or last_meta is None:
            raise RuntimeError("No walk-forward folds were generated; adjust split sizes.")

        avg_accuracy = sum(metric["accuracy"] for metric in fold_metrics) / len(fold_metrics)
        avg_f1 = sum(metric["f1_macro"] for metric in fold_metrics) / len(fold_metrics)
        oos_predictions = pd.concat(oos_predictions).sort_index()
        oos_meta_prob = pd.concat(oos_meta_prob).sort_index().reindex(oos_predictions.index)
        training = {
            "fold_metrics": fold_metrics,
            "avg_accuracy": avg_accuracy,
            "avg_f1_macro": avg_f1,
            "last_model": last_model,
            "last_meta": last_meta,
            "oos_predictions": oos_predictions,
            "oos_meta_prob": oos_meta_prob,
        }
        pipeline.state["training"] = training
        return training


class SignalsStep(PipelineStep):
    name = "generate_signals"

    def run(self, pipeline):
        training = pipeline.require("training")
        config = pipeline.section("signals")

        prediction_series = training.get("oos_predictions")
        meta_prob_series = training.get("oos_meta_prob")
        if prediction_series is None or meta_prob_series is None:
            X = pipeline.require("X")
            model = training["last_model"]
            meta_model = training["last_meta"]
            predictions = pd.Series(model.predict(X), index=X.index)
            X_meta = X.copy()
            X_meta["primary_pred"] = predictions.values
            meta_prob_series = pd.Series(_positive_class_probability(meta_model, X_meta), index=X.index)
            prediction_series = predictions

        prediction_series = prediction_series.sort_index()
        meta_prob_series = meta_prob_series.reindex(prediction_series.index)

        continuous = pd.Series(0.0, index=prediction_series.index)
        for timestamp, prediction in prediction_series.items():
            if prediction == 0:
                continue
            size = kelly_fraction(
                prob_win=meta_prob_series.loc[timestamp],
                avg_win=config.get("avg_win", 0.02),
                avg_loss=config.get("avg_loss", 0.02),
                fraction=config.get("fraction", 0.5),
            )
            continuous.loc[timestamp] = prediction * size

        threshold = config.get("threshold", 0.05)
        signals = continuous.apply(
            lambda value: 1 if value > threshold else (-1 if value < -threshold else 0)
        )

        result = {
            "predictions": prediction_series,
            "meta_prob": meta_prob_series,
            "continuous_signals": continuous,
            "signals": signals,
        }
        pipeline.state["signals"] = result
        return result


class BacktestStep(PipelineStep):
    name = "run_backtest"

    def run(self, pipeline):
        raw_data = pipeline.require("raw_data")
        signals = pipeline.require("signals")["signals"]
        config = pipeline.section("backtest")
        backtest = run_backtest(
            close=raw_data["close"].loc[signals.index],
            signals=signals,
            equity=config.get("equity", 10_000.0),
            fee_rate=config.get("fee_rate", 0.001),
        )
        pipeline.state["backtest"] = backtest
        return backtest


DEFAULT_STEPS = [
    FetchDataStep,
    IndicatorsStep,
    AutoMLStep,
    FeaturesStep,
    StationarityStep,
    RegimeStep,
    LabelsStep,
    AlignDataStep,
    SampleWeightsStep,
    TrainModelsStep,
    SignalsStep,
    BacktestStep,
]


class ResearchPipeline:
    """Reusable stepwise research pipeline backed by config dicts."""

    def __init__(self, config, steps=None):
        self.config = dict(config)
        step_classes = steps or DEFAULT_STEPS
        self.steps = [step() if isinstance(step, type) else step for step in step_classes]
        self.step_map = {step.name: step for step in self.steps}
        self.state = {}
        self.step_results = {}

    def section(self, name):
        return self.config.get(name, {})

    def require(self, key):
        if key not in self.state:
            raise RuntimeError(f"Pipeline state {key!r} is not available yet")
        return self.state[key]

    def run_step(self, name):
        result = self.step_map[name].run(self)
        self.step_results[name] = result
        return result

    def fetch_data(self):
        return self.run_step("fetch_data")

    def run_indicators(self):
        return self.run_step("run_indicators")

    def build_features(self):
        return self.run_step("build_features")

    def run_automl(self):
        return self.run_step("run_automl")

    def check_stationarity(self):
        return self.run_step("check_stationarity")

    def detect_regimes(self):
        return self.run_step("detect_regimes")

    def build_labels(self):
        return self.run_step("build_labels")

    def align_data(self):
        return self.run_step("align_data")

    def compute_sample_weights(self):
        return self.run_step("compute_sample_weights")

    def train_models(self):
        return self.run_step("train_models")

    def generate_signals(self):
        return self.run_step("generate_signals")

    def run_backtest(self):
        return self.run_step("run_backtest")

    def run(self):
        for step in self.steps:
            self.run_step(step.name)
        return self.state["backtest"]