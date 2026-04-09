"""Stepwise research pipeline abstraction for reusable model workflows."""

import copy
from itertools import product

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
    apply_binary_probability_calibrator,
    ConstantProbabilityModel,
    build_meta_feature_frame,
    compute_feature_block_diagnostics,
    detect_regime,
    evaluate_model,
    fit_binary_probability_calibrator,
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


def _split_train_validation_window(X_train, y_train, sample_weights, model_config):
    validation_size = model_config.get("validation_size")
    validation_fraction = float(model_config.get("validation_fraction", 0.2))
    min_fit_size = int(model_config.get("min_fit_size", 250))
    min_validation_size = int(model_config.get("min_validation_size", 100))

    n_rows = len(X_train)
    if validation_size is None:
        validation_size = int(round(n_rows * validation_fraction))
    validation_size = max(min_validation_size, int(validation_size))
    validation_size = min(validation_size, max(0, n_rows - min_fit_size))

    if validation_size < min_validation_size or (n_rows - validation_size) < min_fit_size:
        return X_train, y_train, sample_weights, None, None, None

    fit_end = n_rows - validation_size
    return (
        X_train.iloc[:fit_end],
        y_train.iloc[:fit_end],
        sample_weights.iloc[:fit_end],
        X_train.iloc[fit_end:],
        y_train.iloc[fit_end:],
        sample_weights.iloc[fit_end:],
    )


def _estimate_outcome_stats(labels, close, default_win, default_loss):
    if labels is None or labels.empty:
        return float(default_win), float(default_loss)

    if "forward_return" in labels.columns:
        realized_returns = pd.Series(labels["forward_return"], index=labels.index).dropna()
    elif "t1" in labels.columns:
        entry_prices = close.reindex(labels.index)
        exit_prices = close.reindex(pd.DatetimeIndex(labels["t1"]))
        realized_returns = pd.Series(
            (exit_prices.values - entry_prices.values) / entry_prices.values,
            index=labels.index,
        ).dropna()
    else:
        return float(default_win), float(default_loss)

    wins = realized_returns[realized_returns > 0]
    losses = realized_returns[realized_returns < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else float(default_win)
    avg_loss = float(losses.abs().mean()) if len(losses) > 0 else float(default_loss)
    return avg_win, avg_loss


def _calibrate_binary_probability_series(probabilities, target, sample_weight=None, calibrator_config=None):
    probability_series = pd.Series(probabilities, index=target.index if hasattr(target, "index") else None, dtype=float)
    target_series = pd.Series(target, index=probability_series.index if probability_series.index is not None else None)
    mask = probability_series.notna() & target_series.notna()

    if mask.sum() < 30 or target_series.loc[mask].nunique() < 2:
        return probability_series.clip(0.0, 1.0), None

    aligned_weights = None
    if sample_weight is not None:
        aligned_weights = pd.Series(sample_weight, index=target_series.index).loc[mask]

    calibrator = fit_binary_probability_calibrator(
        probability_series.loc[mask].to_numpy(),
        target_series.loc[mask].astype(int),
        sample_weight=aligned_weights,
        model_params=calibrator_config,
    )
    calibrated = pd.Series(
        apply_binary_probability_calibrator(calibrator, probability_series.to_numpy()),
        index=probability_series.index,
        dtype=float,
    ).clip(0.0, 1.0)
    return calibrated, calibrator


def _calibrate_primary_probability_frame(probability_frame, y_true, sample_weight=None, calibrator_config=None):
    frame = probability_frame.copy()
    if 1 not in frame.columns or -1 not in frame.columns:
        return frame, None
    if 0 in frame.columns and frame[0].abs().sum() > 1e-12:
        return frame, None

    target = pd.Series(y_true, index=frame.index)
    binary_mask = target.isin([-1, 1])
    if binary_mask.sum() < 30 or target.loc[binary_mask].nunique() < 2:
        return frame, None

    aligned_weights = None
    if sample_weight is not None:
        aligned_weights = pd.Series(sample_weight, index=target.index).loc[binary_mask]

    calibrator = fit_binary_probability_calibrator(
        frame.loc[binary_mask, 1].to_numpy(),
        target.loc[binary_mask].eq(1).astype(int),
        sample_weight=aligned_weights,
        model_params=calibrator_config,
    )
    calibrated_long = pd.Series(
        apply_binary_probability_calibrator(calibrator, frame[1].to_numpy()),
        index=frame.index,
        dtype=float,
    ).clip(0.0, 1.0)
    frame[1] = calibrated_long
    frame[-1] = (1.0 - calibrated_long).clip(0.0, 1.0)
    if 0 in frame.columns:
        frame[0] = 0.0
    return frame, calibrator


def _apply_primary_probability_calibrator(probability_frame, calibrator):
    if calibrator is None:
        return probability_frame.copy()

    frame = probability_frame.copy()
    calibrated_long = pd.Series(
        apply_binary_probability_calibrator(calibrator, frame[1].to_numpy()),
        index=frame.index,
        dtype=float,
    ).clip(0.0, 1.0)
    frame[1] = calibrated_long
    frame[-1] = (1.0 - calibrated_long).clip(0.0, 1.0)
    if 0 in frame.columns:
        frame[0] = 0.0
    return frame


def _resolve_signal_threshold_grid(signal_config):
    def _grid(key, fallback):
        values = signal_config.get(key)
        if values is None:
            values = fallback
        return sorted({round(float(value), 6) for value in values})

    threshold_default = float(signal_config.get("threshold", 0.03))
    edge_default = float(signal_config.get("edge_threshold", 0.05))
    meta_default = float(signal_config.get("meta_threshold", 0.55))
    fraction_default = float(signal_config.get("fraction", 0.5))

    return {
        "threshold": _grid("threshold_grid", [0.0, 0.005, 0.01, threshold_default, 0.03]),
        "edge_threshold": _grid("edge_threshold_grid", [0.0, 0.03, edge_default, 0.08]),
        "meta_threshold": _grid("meta_threshold_grid", [0.5, meta_default, 0.6, 0.65]),
        "fraction": _grid("fraction_grid", [0.25, fraction_default, 0.75]),
    }


def _build_signal_state(prediction_series, probability_frame, meta_prob_series, signal_config, avg_win, avg_loss, holding_bars):
    direction = prediction_series.apply(lambda value: 1.0 if value > 0 else (-1.0 if value < 0 else 0.0))
    direction_edge = probability_frame[1] - probability_frame[-1]
    confidence = direction_edge.abs().clip(0.0, 1.0)
    kelly_size = meta_prob_series.apply(
        lambda prob: kelly_fraction(
            prob_win=prob,
            avg_win=avg_win,
            avg_loss=avg_loss,
            fraction=signal_config.get("fraction", 0.5),
        )
    )

    event_signals = direction * kelly_size
    event_signals = event_signals.where(direction.ne(0.0), 0.0)
    event_signals = event_signals.where(confidence >= signal_config.get("edge_threshold", 0.05), 0.0)
    event_signals = event_signals.where(meta_prob_series >= signal_config.get("meta_threshold", 0.55), 0.0)
    event_signals = event_signals.where(event_signals.abs() >= signal_config.get("threshold", 0.03), 0.0)

    continuous = _apply_holding_period(event_signals, holding_bars)
    signals = continuous.apply(lambda value: 1 if value > 1e-12 else (-1 if value < -1e-12 else 0))
    return {
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
        "tuned_params": {
            "threshold": float(signal_config.get("threshold", 0.03)),
            "edge_threshold": float(signal_config.get("edge_threshold", 0.05)),
            "meta_threshold": float(signal_config.get("meta_threshold", 0.55)),
            "fraction": float(signal_config.get("fraction", 0.5)),
        },
    }


def _score_signal_state(backtest, signal_config):
    score = float(backtest.get("net_profit_pct", 0.0))
    score += float(signal_config.get("tuning_weight_sharpe", 0.03)) * float(backtest.get("sharpe_ratio", 0.0))

    profit_factor = backtest.get("profit_factor", 0.0)
    if not np.isfinite(profit_factor):
        profit_factor = signal_config.get("tuning_profit_factor_cap", 5.0)
    score += float(signal_config.get("tuning_weight_profit_factor", 0.02)) * min(
        float(profit_factor),
        float(signal_config.get("tuning_profit_factor_cap", 5.0)),
    )
    score -= float(signal_config.get("tuning_weight_drawdown", 0.5)) * abs(float(backtest.get("max_drawdown", 0.0)))

    min_trades = int(signal_config.get("tuning_min_trades", 3))
    total_trades = int(backtest.get("total_trades", 0))
    if total_trades < min_trades:
        score -= float(signal_config.get("tuning_trade_penalty", 0.01)) * (min_trades - total_trades)
    return score


def _tune_signal_parameters(validation_close, validation_execution_prices, prediction_series, probability_frame, meta_prob_series, signal_config, backtest_config, avg_win, avg_loss, holding_bars):
    if validation_close is None or len(validation_close) < 25:
        default_config = dict(signal_config)
        state = _build_signal_state(
            prediction_series,
            probability_frame,
            meta_prob_series,
            default_config,
            avg_win,
            avg_loss,
            holding_bars,
        )
        backtest = run_backtest(
            close=validation_close.loc[state["continuous_signals"].index] if validation_close is not None else prediction_series,
            signals=state["continuous_signals"],
            equity=backtest_config.get("equity", 10_000.0),
            fee_rate=backtest_config.get("fee_rate", 0.001),
            slippage_rate=backtest_config.get("slippage_rate", 0.0),
            execution_prices=(
                validation_execution_prices.loc[state["continuous_signals"].index]
                if validation_execution_prices is not None
                else None
            ),
        ) if validation_close is not None else None
        return {
            "params": state["tuned_params"],
            "signal_state": state,
            "backtest": backtest,
            "score": _score_signal_state(backtest, signal_config) if backtest is not None else None,
        }

    threshold_grid = _resolve_signal_threshold_grid(signal_config)
    best = None
    for threshold, edge_threshold, meta_threshold, fraction in product(
        threshold_grid["threshold"],
        threshold_grid["edge_threshold"],
        threshold_grid["meta_threshold"],
        threshold_grid["fraction"],
    ):
        candidate_config = dict(signal_config)
        candidate_config.update(
            {
                "threshold": threshold,
                "edge_threshold": edge_threshold,
                "meta_threshold": meta_threshold,
                "fraction": fraction,
            }
        )
        state = _build_signal_state(
            prediction_series,
            probability_frame,
            meta_prob_series,
            candidate_config,
            avg_win,
            avg_loss,
            holding_bars,
        )
        backtest = run_backtest(
            close=validation_close.loc[state["continuous_signals"].index],
            signals=state["continuous_signals"],
            equity=backtest_config.get("equity", 10_000.0),
            fee_rate=backtest_config.get("fee_rate", 0.001),
            slippage_rate=backtest_config.get("slippage_rate", 0.0),
            execution_prices=(
                validation_execution_prices.loc[state["continuous_signals"].index]
                if validation_execution_prices is not None
                else None
            ),
        )
        score = _score_signal_state(backtest, signal_config)
        if best is None or score > best["score"]:
            best = {
                "params": state["tuned_params"],
                "signal_state": state,
                "backtest": backtest,
                "score": score,
            }

    return best


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
        backtest_config = pipeline.section("backtest")
        label_kind = config.get("kind", "triple_barrier")
        cost_rate = config.get("cost_rate")
        if cost_rate is None:
            cost_rate = 2.0 * (
                float(backtest_config.get("fee_rate", 0.0))
                + float(backtest_config.get("slippage_rate", 0.0))
            )

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
                high=data.get("high"),
                low=data.get("low"),
                pt_sl=config.get("pt_sl", (2.0, 2.0)),
                max_holding=config.get("max_holding", 24),
                min_return=config.get("min_return", 0.0),
                cost_rate=float(cost_rate),
                barrier_tie_break=config.get("barrier_tie_break", "sl"),
            )
        elif label_kind == "fixed_horizon":
            labels = fixed_horizon_labels(
                close=data["close"],
                horizon=config.get("horizon", 5),
                threshold=config.get("threshold", 0.0),
                cost_rate=float(cost_rate),
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
        config = pipeline.section("feature_selection")

        report = {
            "enabled": bool(config.get("enabled", True)),
            "mode": "fold_local",
            "input_features": int(X.shape[1]),
            "selected_features": int(X.shape[1]),
            "max_features": config.get("max_features"),
            "min_mi_threshold": float(config.get("min_mi_threshold", 0.0)),
            "note": "Supervised feature selection is applied inside each walk-forward fold.",
        }
        pipeline.state["feature_selection"] = report
        return type("FeatureSelectionPreview", (), {"report": report})()


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
        labels_aligned = pipeline.require("labels_aligned")
        raw_data = pipeline.require("raw_data")
        config = pipeline.section("model")
        selection_config = pipeline.section("feature_selection")
        signal_config = pipeline.section("signals")
        backtest_config = pipeline.section("backtest")
        feature_blocks = pipeline.state.get("feature_blocks", {})
        binary_primary = config.get("binary_primary", True)
        holding_bars = _resolve_signal_holding_bars(pipeline, signal_config)
        default_avg_win = float(signal_config.get("avg_win", 0.02))
        default_avg_loss = float(signal_config.get("avg_loss", 0.02))
        close_all = raw_data["close"]

        fold_metrics = []
        fold_block_diagnostics = []
        fold_feature_selection = []
        fold_signal_tuning = []
        last_model = None
        last_meta = None
        last_primary_calibrator = None
        last_meta_calibrator = None
        last_selected_columns = list(X.columns)
        last_signal_params = {
            "threshold": float(signal_config.get("threshold", 0.03)),
            "edge_threshold": float(signal_config.get("edge_threshold", 0.05)),
            "meta_threshold": float(signal_config.get("meta_threshold", 0.55)),
            "fraction": float(signal_config.get("fraction", 0.5)),
        }
        last_avg_win = default_avg_win
        last_avg_loss = default_avg_loss
        oos_predictions = []
        oos_probabilities = []
        oos_meta_prob = []
        oos_direction_edge = []
        oos_confidence = []
        oos_kelly_size = []
        oos_event_signals = []
        oos_continuous_signals = []
        oos_signals = []

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

            X_fit, y_fit, w_fit, X_val, y_val, w_val = _split_train_validation_window(
                X_train,
                y_train,
                w_train,
                config,
            )

            fold_feature_blocks = dict(feature_blocks)
            selected_columns = list(X_fit.columns)
            selection_result = None
            if selection_config.get("enabled", True):
                selection_result = select_features(
                    X_fit,
                    y_fit,
                    feature_blocks=fold_feature_blocks,
                    config=selection_config,
                )
                if not selection_result.frame.empty:
                    selected_columns = [column for column in selection_result.frame.columns if column in X.columns]
                    fold_feature_blocks = selection_result.feature_blocks

            X_fit_model = X_fit.loc[:, selected_columns]
            X_train_model = X_train.loc[:, selected_columns]
            X_test_model = X_test.loc[:, selected_columns]
            X_val_model = X_val.loc[:, selected_columns] if X_val is not None else None

            fold_feature_selection.append(
                {
                    "fold": fold,
                    "input_features": int(X_train.shape[1]),
                    "selected_features": int(len(selected_columns)),
                    "top_mi_scores": (selection_result.report.get("top_mi_scores", {}) if selection_result is not None else {}),
                }
            )

            X_train_primary = X_fit_model
            y_train_primary = y_fit
            w_train_primary = w_fit
            if binary_primary:
                binary_mask = y_fit.ne(0)
                X_train_primary = X_fit_model.loc[binary_mask]
                y_train_primary = y_fit.loc[binary_mask]
                w_train_primary = w_fit.loc[binary_mask]

            model = train_model(
                X_train_primary,
                y_train_primary,
                sample_weight=w_train_primary,
                model_type=config.get("type", "gbm"),
                model_params=config.get("params"),
            )

            metrics = evaluate_model(model, X_test_model, y_test)
            metrics["fold"] = fold

            test_primary_preds = pd.Series(model.predict(X_test_model), index=X_test_model.index)
            test_primary_probs_raw = predict_probability_frame(model, X_test_model)
            test_primary_probs = test_primary_probs_raw.copy()

            primary_calibrator = None
            meta_calibrator = None
            meta_model = _train_inner_meta_model(X_fit_model, y_fit, w_fit, config)

            fold_avg_win, fold_avg_loss = _estimate_outcome_stats(
                labels_aligned.loc[y_fit.index],
                close_all,
                default_avg_win,
                default_avg_loss,
            )

            tuned_signal_params = dict(last_signal_params)
            tuning_backtest = None
            tuning_score = None
            if X_val_model is not None and not X_val_model.empty:
                val_primary_preds = pd.Series(model.predict(X_val_model), index=X_val_model.index)
                val_primary_probs_raw = predict_probability_frame(model, X_val_model)
                val_primary_probs, primary_calibrator = _calibrate_primary_probability_frame(
                    val_primary_probs_raw,
                    y_val,
                    sample_weight=w_val,
                    calibrator_config=config.get("calibration_params"),
                )

                X_meta_val = build_meta_feature_frame(val_primary_preds, val_primary_probs_raw)
                val_meta_prob_raw = pd.Series(
                    _positive_class_probability(meta_model, X_meta_val),
                    index=X_val_model.index,
                )
                val_meta_prob, meta_calibrator = _calibrate_binary_probability_series(
                    val_meta_prob_raw,
                    (val_primary_preds == y_val).astype(int),
                    sample_weight=w_val,
                    calibrator_config=config.get("meta_calibration_params"),
                )

                tuning = _tune_signal_parameters(
                    validation_close=close_all.loc[X_val_model.index],
                    validation_execution_prices=raw_data["open"].loc[X_val_model.index] if backtest_config.get("use_open_execution", True) and "open" in raw_data.columns else None,
                    prediction_series=val_primary_preds,
                    probability_frame=val_primary_probs,
                    meta_prob_series=val_meta_prob,
                    signal_config=signal_config,
                    backtest_config=backtest_config,
                    avg_win=fold_avg_win,
                    avg_loss=fold_avg_loss,
                    holding_bars=holding_bars,
                )
                if tuning is not None:
                    tuned_signal_params = dict(tuning["params"])
                    tuning_backtest = tuning["backtest"]
                    tuning_score = tuning["score"]

            test_primary_probs = _apply_primary_probability_calibrator(test_primary_probs_raw, primary_calibrator)

            block_diagnostics = compute_feature_block_diagnostics(
                model,
                X_train_primary,
                X_test_model,
                y_test,
                feature_blocks=fold_feature_blocks,
                baseline_metrics=metrics,
            )
            fold_block_diagnostics.append(block_diagnostics)

            X_meta_test = build_meta_feature_frame(test_primary_preds, test_primary_probs_raw)
            meta_prob_test_raw = pd.Series(
                _positive_class_probability(meta_model, X_meta_test),
                index=X_test_model.index,
            )
            meta_prob_test = pd.Series(
                apply_binary_probability_calibrator(meta_calibrator, meta_prob_test_raw.to_numpy()),
                index=X_test_model.index,
                dtype=float,
            ).clip(0.0, 1.0) if meta_calibrator is not None else meta_prob_test_raw.clip(0.0, 1.0)

            fold_signal_config = dict(signal_config)
            fold_signal_config.update(tuned_signal_params)
            signal_state = _build_signal_state(
                test_primary_preds,
                test_primary_probs,
                meta_prob_test,
                fold_signal_config,
                fold_avg_win,
                fold_avg_loss,
                holding_bars,
            )

            fold_metrics.append(metrics)
            fold_signal_tuning.append(
                {
                    "fold": fold,
                    "params": tuned_signal_params,
                    "score": tuning_score,
                    "backtest": (
                        {
                            "net_profit_pct": tuning_backtest.get("net_profit_pct"),
                            "sharpe_ratio": tuning_backtest.get("sharpe_ratio"),
                            "max_drawdown": tuning_backtest.get("max_drawdown"),
                            "total_trades": tuning_backtest.get("total_trades"),
                        }
                        if tuning_backtest is not None
                        else None
                    ),
                }
            )
            last_model = model
            last_meta = meta_model
            last_primary_calibrator = primary_calibrator
            last_meta_calibrator = meta_calibrator
            last_selected_columns = list(selected_columns)
            last_signal_params = dict(tuned_signal_params)
            last_avg_win = fold_avg_win
            last_avg_loss = fold_avg_loss
            oos_predictions.append(test_primary_preds)
            oos_probabilities.append(test_primary_probs)
            oos_meta_prob.append(meta_prob_test)
            oos_direction_edge.append(signal_state["direction_edge"])
            oos_confidence.append(signal_state["confidence"])
            oos_kelly_size.append(signal_state["kelly_size"])
            oos_event_signals.append(signal_state["event_signals"])
            oos_continuous_signals.append(signal_state["continuous_signals"])
            oos_signals.append(signal_state["signals"])

        if last_model is None or last_meta is None:
            raise RuntimeError("No walk-forward folds were generated; adjust split sizes.")

        avg_accuracy = sum(metric["accuracy"] for metric in fold_metrics) / len(fold_metrics)
        avg_f1 = sum(metric["f1_macro"] for metric in fold_metrics) / len(fold_metrics)
        oos_predictions = pd.concat(oos_predictions).sort_index()
        oos_probabilities = pd.concat(oos_probabilities).sort_index().reindex(oos_predictions.index)
        oos_meta_prob = pd.concat(oos_meta_prob).sort_index().reindex(oos_predictions.index)
        oos_direction_edge = pd.concat(oos_direction_edge).sort_index().reindex(oos_predictions.index)
        oos_confidence = pd.concat(oos_confidence).sort_index().reindex(oos_predictions.index)
        oos_kelly_size = pd.concat(oos_kelly_size).sort_index().reindex(oos_predictions.index)
        oos_event_signals = pd.concat(oos_event_signals).sort_index().reindex(oos_predictions.index)
        oos_continuous_signals = pd.concat(oos_continuous_signals).sort_index().reindex(oos_predictions.index)
        oos_signals = pd.concat(oos_signals).sort_index().reindex(oos_predictions.index)
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
            "last_primary_calibrator": last_primary_calibrator,
            "last_meta_calibrator": last_meta_calibrator,
            "last_selected_columns": last_selected_columns,
            "last_signal_params": last_signal_params,
            "last_avg_win": last_avg_win,
            "last_avg_loss": last_avg_loss,
            "oos_predictions": oos_predictions,
            "oos_probabilities": oos_probabilities,
            "oos_meta_prob": oos_meta_prob,
            "oos_direction_edge": oos_direction_edge,
            "oos_confidence": oos_confidence,
            "oos_kelly_size": oos_kelly_size,
            "oos_event_signals": oos_event_signals,
            "oos_continuous_signals": oos_continuous_signals,
            "oos_signals": oos_signals,
            "feature_block_diagnostics": feature_diagnostics,
            "feature_selection": {
                "enabled": bool(selection_config.get("enabled", True)),
                "mode": "fold_local",
                "avg_selected_features": round(float(np.mean([row["selected_features"] for row in fold_feature_selection])), 2),
                "folds": fold_feature_selection,
            },
            "signal_tuning": fold_signal_tuning,
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

        if training.get("oos_continuous_signals") is not None:
            result = {
                "predictions": training["oos_predictions"],
                "primary_probabilities": training["oos_probabilities"],
                "meta_prob": training["oos_meta_prob"],
                "direction_edge": training["oos_direction_edge"],
                "confidence": training["oos_confidence"],
                "kelly_size": training["oos_kelly_size"],
                "event_signals": training["oos_event_signals"],
                "holding_bars": _resolve_signal_holding_bars(pipeline, config),
                "continuous_signals": training["oos_continuous_signals"],
                "signals": training["oos_signals"],
                "avg_win_used": training.get("oos_avg_win") if training.get("oos_avg_win") is not None else training.get("last_avg_win", config.get("avg_win", 0.02)),
                "avg_loss_used": training.get("oos_avg_loss") if training.get("oos_avg_loss") is not None else training.get("last_avg_loss", config.get("avg_loss", 0.02)),
                "signal_tuning": training.get("signal_tuning", []),
                "tuned_params": training.get("last_signal_params", {}),
            }
            pipeline.state["signals"] = result
            return result

        prediction_series = training.get("oos_predictions")
        probability_frame = training.get("oos_probabilities")
        meta_prob_series = training.get("oos_meta_prob")
        if prediction_series is None or meta_prob_series is None or probability_frame is None:
            X = pipeline.require("X")
            selected_columns = training.get("last_selected_columns") or list(X.columns)
            X_model = X.loc[:, selected_columns]
            model = training["last_model"]
            meta_model = training["last_meta"]
            predictions = pd.Series(model.predict(X_model), index=X_model.index)
            probability_frame_raw = predict_probability_frame(model, X_model)
            probability_frame = _apply_primary_probability_calibrator(
                probability_frame_raw,
                training.get("last_primary_calibrator"),
            )
            X_meta = build_meta_feature_frame(predictions, probability_frame_raw)
            meta_prob_raw = pd.Series(_positive_class_probability(meta_model, X_meta), index=X_model.index)
            meta_prob_series = pd.Series(
                apply_binary_probability_calibrator(
                    training.get("last_meta_calibrator"),
                    meta_prob_raw.to_numpy(),
                ),
                index=X_model.index,
                dtype=float,
            ).clip(0.0, 1.0) if training.get("last_meta_calibrator") is not None else meta_prob_raw.clip(0.0, 1.0)
            prediction_series = predictions

        prediction_series = prediction_series.sort_index()
        probability_frame = probability_frame.reindex(prediction_series.index).fillna(0.0)
        meta_prob_series = meta_prob_series.reindex(prediction_series.index)

        avg_win = config.get("avg_win", 0.02)
        avg_loss = config.get("avg_loss", 0.02)
        if training.get("oos_avg_win") is not None:
            avg_win = training["oos_avg_win"]
        if training.get("oos_avg_loss") is not None:
            avg_loss = training["oos_avg_loss"]

        holding_bars = _resolve_signal_holding_bars(pipeline, config)
        signal_config = dict(config)
        signal_config.update(training.get("last_signal_params", {}))
        result = _build_signal_state(
            prediction_series,
            probability_frame,
            meta_prob_series,
            signal_config,
            avg_win,
            avg_loss,
            holding_bars,
        )
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
            slippage_rate=config.get("slippage_rate", 0.0),
            execution_prices=(raw_data["open"].loc[positions.index] if config.get("use_open_execution", True) and "open" in raw_data.columns else None),
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