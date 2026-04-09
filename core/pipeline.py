"""Stepwise research pipeline abstraction for reusable model workflows."""

import copy

import numpy as np
import pandas as pd

from .automl import run_automl_study
from .backtest import kelly_fraction, run_backtest
from .data import fetch_binance_vision
from .features import build_feature_set, check_stationarity, screen_features_for_stationarity, select_features
from .indicators import run_indicators
from .labeling import (
    fixed_horizon_labels,
    sample_weights_by_uniqueness,
    triple_barrier_labels,
)
from .models import (
    ConstantProbabilityModel,
    build_meta_feature_frame,
    compute_feature_block_diagnostics,
    detect_regime,
    evaluate_model,
    predict_probability_frame,
    summarize_feature_block_diagnostics,
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


def _resolve_signal_holding_bars(pipeline, signal_config):
    configured = signal_config.get("holding_bars")
    if configured is not None:
        return max(1, int(configured))

    label_config = pipeline.section("labels")
    if label_config.get("kind") == "fixed_horizon":
        return max(1, int(label_config.get("horizon", 1)))
    return max(1, int(label_config.get("max_holding", 1)))


def _apply_holding_period(event_weights, holding_bars):
    weights = pd.Series(event_weights, copy=False).astype(float)
    if holding_bars <= 1 or weights.empty:
        return weights.clip(-1.0, 1.0)

    held = pd.Series(0.0, index=weights.index)
    for lag in range(holding_bars):
        shifted = weights.shift(lag).fillna(0.0)
        held = held.where(held.abs() >= shifted.abs(), shifted)
    return held.clip(-1.0, 1.0)


def _train_inner_meta_model(X_train, y_train, sample_weights, model_config):
    inner_predictions = []
    inner_probabilities = []
    inner_truth = []
    inner_weights = []
    min_train_rows = max(50, min(100, len(X_train) // 3))
    min_test_rows = 20
    binary_primary = model_config.get("binary_primary", True)

    for inner_train_idx, inner_test_idx in walk_forward_split(
        X_train,
        n_splits=model_config.get("meta_n_splits", max(2, min(3, model_config.get("n_splits", 3)))),
        gap=model_config.get("gap", 0),
        expanding=model_config.get("expanding", False),
    ):
        if len(inner_train_idx) < min_train_rows or len(inner_test_idx) < min_test_rows:
            continue

        X_inner_train = X_train.iloc[inner_train_idx]
        y_inner_train = y_train.iloc[inner_train_idx]
        w_inner_train = sample_weights.iloc[inner_train_idx]
        X_inner_test = X_train.iloc[inner_test_idx]

        if binary_primary:
            binary_mask = y_inner_train.ne(0)
            X_inner_train = X_inner_train.loc[binary_mask]
            y_inner_train = y_inner_train.loc[binary_mask]
            w_inner_train = w_inner_train.loc[binary_mask]
            if len(X_inner_train) < min_train_rows:
                continue

        inner_model = train_model(
            X_inner_train,
            y_inner_train,
            sample_weight=w_inner_train,
            model_type=model_config.get("type", "gbm"),
            model_params=model_config.get("params"),
        )

        inner_pred = pd.Series(inner_model.predict(X_inner_test), index=X_inner_test.index)
        inner_prob = predict_probability_frame(inner_model, X_inner_test)
        inner_predictions.append(inner_pred)
        inner_probabilities.append(inner_prob)
        inner_truth.append(y_train.iloc[inner_test_idx])
        inner_weights.append(sample_weights.iloc[inner_test_idx])

    if not inner_predictions:
        return ConstantProbabilityModel(positive_probability=0.5)

    meta_predictions = pd.concat(inner_predictions).sort_index()
    meta_probabilities = pd.concat(inner_probabilities).sort_index()
    meta_truth = pd.concat(inner_truth).sort_index().reindex(meta_predictions.index)
    meta_weights = pd.concat(inner_weights).sort_index().reindex(meta_predictions.index).fillna(1.0)

    return train_meta_model(
        meta_predictions,
        meta_probabilities,
        meta_truth,
        sample_weight=meta_weights,
        model_params=model_config.get("meta_params"),
    )


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
                FeatureSelectionStep,
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
            "feature_selection",
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
        indicator_run = pipeline.state.get("indicator_run")
        config = pipeline.section("features")
        feature_set = build_feature_set(
            data,
            lags=config.get("lags"),
            frac_diff_d=config.get("frac_diff_d"),
            indicator_run=indicator_run,
            rolling_window=config.get("rolling_window", 20),
            squeeze_quantile=config.get("squeeze_quantile", 0.2),
        )
        features = feature_set.frame
        feature_blocks = dict(feature_set.feature_blocks)

        for builder in config.get("builders", []):
            built = builder(pipeline, features.copy())
            if built is not None:
                features = built

            feature_blocks = {
                column: feature_blocks.get(column, config.get("custom_block_name", "custom"))
                for column in features.columns
            }

        screening_config = dict(pipeline.section("stationarity"))
        screening_config.setdefault("enabled", True)
        screening_config.setdefault("rolling_window", config.get("rolling_window", 20))
        screening_config.setdefault("frac_diff_d", config.get("frac_diff_d"))
        screening_result = screen_features_for_stationarity(
            features,
            feature_blocks=feature_blocks,
            config=screening_config,
        )

        pipeline.state["raw_features"] = features
        pipeline.state["feature_blocks_raw"] = feature_blocks
        pipeline.state["feature_screening"] = screening_result.report
        pipeline.state["feature_blocks"] = screening_result.feature_blocks
        pipeline.state["features"] = screening_result.frame
        return screening_result.frame


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

        if "feature_screening" in pipeline.state:
            results["feature_screening"] = pipeline.state["feature_screening"]

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
        feature_blocks = dict(pipeline.state.get("feature_blocks", {}))
        feature_blocks[config.get("column_name", "regime")] = "regime"
        pipeline.state["feature_blocks"] = feature_blocks
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


class FeatureSelectionStep(PipelineStep):
    name = "select_features"

    def run(self, pipeline):
        X = pipeline.require("X")
        y = pipeline.require("y")
        feature_blocks = pipeline.state.get("feature_blocks", {})
        config = pipeline.section("feature_selection")

        result = select_features(X, y, feature_blocks=feature_blocks, config=config)
        pipeline.state["X"] = result.frame.reindex(X.index).loc[X.index.intersection(result.frame.index)]
        pipeline.state["feature_blocks"] = result.feature_blocks
        pipeline.state["feature_selection"] = result.report
        return result


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
        feature_blocks = pipeline.state.get("feature_blocks", {})
        binary_primary = config.get("binary_primary", True)

        fold_metrics = []
        fold_block_diagnostics = []
        last_model = None
        last_meta = None
        oos_predictions = []
        oos_probabilities = []
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

            X_train_primary = X_train
            y_train_primary = y_train
            w_train_primary = w_train
            if binary_primary:
                binary_mask = y_train.ne(0)
                X_train_primary = X_train.loc[binary_mask]
                y_train_primary = y_train.loc[binary_mask]
                w_train_primary = w_train.loc[binary_mask]

            model = train_model(
                X_train_primary,
                y_train_primary,
                sample_weight=w_train_primary,
                model_type=config.get("type", "gbm"),
                model_params=config.get("params"),
            )
            metrics = evaluate_model(model, X_test, y_test)
            metrics["fold"] = fold
            test_primary_probs = predict_probability_frame(model, X_test)
            block_diagnostics = compute_feature_block_diagnostics(
                model,
                X_train_primary,
                X_test,
                y_test,
                feature_blocks=feature_blocks,
                baseline_metrics=metrics,
            )
            fold_block_diagnostics.append(block_diagnostics)

            test_primary_preds = pd.Series(model.predict(X_test), index=X_test.index)
            meta_model = _train_inner_meta_model(X_train, y_train, w_train, config)

            X_meta_test = build_meta_feature_frame(test_primary_preds, test_primary_probs)
            meta_prob_test = pd.Series(_positive_class_probability(meta_model, X_meta_test), index=X_test.index)

            fold_metrics.append(metrics)
            last_model = model
            last_meta = meta_model
            oos_predictions.append(test_primary_preds)
            oos_probabilities.append(test_primary_probs)
            oos_meta_prob.append(meta_prob_test)

        if last_model is None or last_meta is None:
            raise RuntimeError("No walk-forward folds were generated; adjust split sizes.")

        avg_accuracy = sum(metric["accuracy"] for metric in fold_metrics) / len(fold_metrics)
        avg_f1 = sum(metric["f1_macro"] for metric in fold_metrics) / len(fold_metrics)
        oos_predictions = pd.concat(oos_predictions).sort_index()
        oos_probabilities = pd.concat(oos_probabilities).sort_index().reindex(oos_predictions.index)
        oos_meta_prob = pd.concat(oos_meta_prob).sort_index().reindex(oos_predictions.index)
        feature_diagnostics = summarize_feature_block_diagnostics(fold_block_diagnostics)

        # Estimate avg_win / avg_loss from OOS label outcomes
        oos_avg_win = None
        oos_avg_loss = None
        try:
            labels_aligned = pipeline.require("labels_aligned")
            raw_data = pipeline.require("raw_data")
            close_all = raw_data["close"]
            oos_labels = labels_aligned.loc[labels_aligned.index.intersection(oos_predictions.index)]
            if not oos_labels.empty and "t1" in oos_labels.columns:
                entry_prices = close_all.reindex(oos_labels.index)
                exit_idx = pd.DatetimeIndex(oos_labels["t1"])
                exit_prices = close_all.reindex(exit_idx)
                realized_returns = (exit_prices.values - entry_prices.values) / entry_prices.values
                realized_returns = pd.Series(realized_returns, index=oos_labels.index).dropna()
                wins = realized_returns[realized_returns > 0]
                losses = realized_returns[realized_returns < 0]
                if len(wins) > 0:
                    oos_avg_win = float(wins.mean())
                if len(losses) > 0:
                    oos_avg_loss = float(losses.abs().mean())
        except Exception:
            pass

        training = {
            "fold_metrics": fold_metrics,
            "avg_accuracy": avg_accuracy,
            "avg_f1_macro": avg_f1,
            "last_model": last_model,
            "last_meta": last_meta,
            "oos_predictions": oos_predictions,
            "oos_probabilities": oos_probabilities,
            "oos_meta_prob": oos_meta_prob,
            "feature_block_diagnostics": feature_diagnostics,
            "oos_avg_win": oos_avg_win,
            "oos_avg_loss": oos_avg_loss,
        }
        pipeline.state["training"] = training
        return training


class SignalsStep(PipelineStep):
    name = "generate_signals"

    def run(self, pipeline):
        training = pipeline.require("training")
        config = pipeline.section("signals")

        prediction_series = training.get("oos_predictions")
        probability_frame = training.get("oos_probabilities")
        meta_prob_series = training.get("oos_meta_prob")
        if prediction_series is None or meta_prob_series is None or probability_frame is None:
            X = pipeline.require("X")
            model = training["last_model"]
            meta_model = training["last_meta"]
            predictions = pd.Series(model.predict(X), index=X.index)
            probability_frame = predict_probability_frame(model, X)
            X_meta = build_meta_feature_frame(predictions, probability_frame)
            meta_prob_series = pd.Series(_positive_class_probability(meta_model, X_meta), index=X.index)
            prediction_series = predictions

        prediction_series = prediction_series.sort_index()
        probability_frame = probability_frame.reindex(prediction_series.index).fillna(0.0)
        meta_prob_series = meta_prob_series.reindex(prediction_series.index)

        direction = prediction_series.apply(lambda value: 1.0 if value > 0 else (-1.0 if value < 0 else 0.0))
        direction_edge = probability_frame[1] - probability_frame[-1]
        confidence = direction_edge.abs().clip(0.0, 1.0)

        avg_win = config.get("avg_win", 0.02)
        avg_loss = config.get("avg_loss", 0.02)
        if training.get("oos_avg_win") is not None:
            avg_win = training["oos_avg_win"]
        if training.get("oos_avg_loss") is not None:
            avg_loss = training["oos_avg_loss"]

        kelly_size = meta_prob_series.apply(
            lambda prob: kelly_fraction(
                prob_win=prob,
                avg_win=avg_win,
                avg_loss=avg_loss,
                fraction=config.get("fraction", 0.5),
            )
        )

        event_signals = direction * kelly_size
        event_signals = event_signals.where(direction.ne(0.0), 0.0)
        event_signals = event_signals.where(confidence >= config.get("edge_threshold", 0.05), 0.0)
        event_signals = event_signals.where(meta_prob_series >= config.get("meta_threshold", 0.55), 0.0)
        event_signals = event_signals.where(event_signals.abs() >= config.get("threshold", 0.03), 0.0)

        holding_bars = _resolve_signal_holding_bars(pipeline, config)
        continuous = _apply_holding_period(event_signals, holding_bars)
        signals = continuous.apply(lambda value: 1 if value > 1e-12 else (-1 if value < -1e-12 else 0))

        result = {
            "predictions": prediction_series,
            "primary_probabilities": probability_frame,
            "meta_prob": meta_prob_series,
            "direction_edge": direction_edge,
            "confidence": confidence,
            "kelly_size": kelly_size,
            "event_signals": event_signals,
            "holding_bars": holding_bars,
            "continuous_signals": continuous,
            "signals": signals,
            "avg_win_used": avg_win,
            "avg_loss_used": avg_loss,
        }
        pipeline.state["signals"] = result
        return result


class BacktestStep(PipelineStep):
    name = "run_backtest"

    def run(self, pipeline):
        raw_data = pipeline.require("raw_data")
        signal_state = pipeline.require("signals")
        config = pipeline.section("backtest")
        positions = signal_state["continuous_signals"] if config.get("use_continuous_positions", True) else signal_state["signals"]
        backtest = run_backtest(
            close=raw_data["close"].loc[positions.index],
            signals=positions,
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
    FeatureSelectionStep,
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

    def select_features(self):
        return self.run_step("select_features")

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