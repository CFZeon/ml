"""High-level experiment runner for config-driven research workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from core import ResearchPipeline

from .config import ResolvedExperimentConfig, load_experiment_config


@dataclass
class ExperimentResult:
    """Artifacts and summary metadata returned by ``run_experiment``."""

    name: str
    status: str
    config: dict[str, Any]
    pipeline: ResearchPipeline
    warnings: list[str]
    artifacts: dict[str, Any]
    config_path: Path | None = None


def _emit(message: str = "", *, quiet: bool) -> None:
    if not quiet:
        print(message)


def _apply_hook(
    hooks: Mapping[str, Callable[[ResearchPipeline, Any], Any]] | None,
    hook_name: str,
    pipeline: ResearchPipeline,
    artifact: Any,
) -> Any:
    if not hooks:
        return artifact
    hook = hooks.get(hook_name)
    if hook is None:
        return artifact
    updated = hook(pipeline, artifact)
    return artifact if updated is None else updated


def _section(title: str, *, quiet: bool) -> None:
    separator = "=" * 78
    _emit(f"\n{separator}\n{title}\n{separator}", quiet=quiet)


def _format_metric(value: Any, *, digits: int = 4, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(numeric):
        return "n/a"
    if percent:
        return f"{numeric:.2%}"
    return f"{numeric:.{digits}f}"


def _format_indicator_specs(indicators: list[Any]) -> list[str]:
    labels: list[str] = []
    for spec in indicators:
        if isinstance(spec, str):
            labels.append(spec)
            continue
        if isinstance(spec, Mapping):
            kind = spec.get("kind", spec.get("indicator", "unknown"))
            params = dict(spec.get("params") or {})
            if spec.get("name"):
                params = {"name": spec.get("name"), **params}
            if params:
                labels.append(f"{kind}({params})")
            else:
                labels.append(str(kind))
            continue
        labels.append(getattr(spec, "kind", type(spec).__name__))
    return labels


def _resolve_model_choice(*, config: dict[str, Any], automl: dict[str, Any] | None, training: dict[str, Any]) -> str:
    if automl:
        best_overrides = dict(automl.get("best_overrides") or {})
        model_override = dict(best_overrides.get("model") or {})
        if model_override.get("type"):
            return str(model_override["type"])
        best_params = dict(automl.get("best_params") or {})
        for key in ["model.type", "type", "model"]:
            if key in best_params:
                return str(best_params[key])
    if training.get("model_type"):
        return str(training["model_type"])
    return str((config.get("model") or {}).get("type", "unknown"))


def _print_automl_bundle_summary(automl: dict[str, Any], *, quiet: bool) -> None:
    bundle = dict(automl.get("best_bundle_lineage") or {})
    if not bundle:
        return

    primary_detector = dict(bundle.get("primary_detector") or {})
    router = dict(bundle.get("router") or {})
    specialists = list(bundle.get("specialist_model_ids") or [])
    detector_name = primary_detector.get("name") or "n/a"
    detector_type = primary_detector.get("type") or "n/a"

    _emit(f"Bundle        : {bundle.get('bundle_name') or 'n/a'}", quiet=quiet)
    if bundle.get("bundle_description"):
        _emit(f"Bundle desc   : {bundle.get('bundle_description')}", quiet=quiet)
    _emit(
        "Bundle path   : "
        f"detector={detector_name}:{detector_type}  "
        f"router={router.get('type', 'n/a')}  "
        f"specialists={specialists if specialists else ['fallback_only']}",
        quiet=quiet,
    )


def _print_overview(resolved: ResolvedExperimentConfig, *, quiet: bool) -> None:
    config = resolved.config
    data = dict(config.get("data") or {})
    experiment = dict(config.get("experiment") or {})
    _section(f"Experiment: {resolved.name}", quiet=quiet)
    _emit(f"Config path   : {resolved.config_path or 'inline'}", quiet=quiet)
    _emit(f"Data source   : {experiment.get('data_source', 'binance_bars')}", quiet=quiet)
    _emit(f"Market        : {data.get('market', 'spot')}", quiet=quiet)
    _emit(f"Symbol        : {data.get('symbol')}", quiet=quiet)
    _emit(f"Interval      : {data.get('interval')}", quiet=quiet)
    _emit(f"Window        : {data.get('start')} -> {data.get('end')}", quiet=quiet)
    _emit(f"Quick mode    : {resolved.quick_mode}", quiet=quiet)
    _emit(f"Indicators    : {_format_indicator_specs(list(config.get('indicators') or []))}", quiet=quiet)
    _emit(
        "Validation    : "
        f"{(config.get('model') or {}).get('cv_method', 'cpcv')}  "
        f"labels={(config.get('labels') or {}).get('kind', 'triple_barrier')}  "
        f"backtest={(config.get('backtest') or {}).get('engine', 'vectorbt')}",
        quiet=quiet,
    )


def _print_data_summary(data: pd.DataFrame, *, quiet: bool) -> None:
    _section("Data", quiet=quiet)
    _emit(f"Rows          : {len(data)}", quiet=quiet)
    if len(data.index):
        _emit(f"Range         : {data.index[0]} -> {data.index[-1]}", quiet=quiet)
    _emit(f"Columns       : {list(data.columns)}", quiet=quiet)


def _print_indicator_summary(indicator_run: Any, *, quiet: bool) -> None:
    _section("Indicators", quiet=quiet)
    results = list(getattr(indicator_run, "results", []) or [])
    _emit(f"Computed      : {[result.kind for result in results]}", quiet=quiet)
    _emit(f"Output cols   : {len(getattr(indicator_run, 'frame', pd.DataFrame()).columns)}", quiet=quiet)


def _print_feature_summary(features: pd.DataFrame, stationarity: dict[str, Any], *, quiet: bool) -> None:
    _section("Features", quiet=quiet)
    _emit(f"Feature count : {features.shape[1]}", quiet=quiet)
    for feature_name in ["close", "close_fracdiff"]:
        stats = dict(stationarity.get(feature_name) or {})
        if stats:
            _emit(
                f"{feature_name:<13}: stationary={stats.get('stationary')}  p={_format_metric(stats.get('p_value'), digits=6)}",
                quiet=quiet,
            )
    screening = dict((stationarity.get("feature_screening") or {}).get("summary") or {})
    if screening:
        _emit(
            "Screening     : "
            f"screened={screening.get('screened_feature_count', 0)}/{screening.get('total_features', 0)}  "
            f"transformed={screening.get('transformed_features', 0)}  "
            f"dropped={screening.get('dropped_features', 0)}",
            quiet=quiet,
        )


def _print_regime_summary(regimes: Any, *, quiet: bool) -> None:
    _section("Regimes", quiet=quiet)
    if isinstance(regimes, pd.DataFrame):
        counts = regimes["regime"].value_counts().to_dict() if "regime" in regimes.columns else {}
        _emit(f"Columns       : {list(regimes.columns)}", quiet=quiet)
    else:
        counts = pd.Series(regimes).value_counts().to_dict()
    _emit(f"Counts        : {counts}", quiet=quiet)


def _print_label_summary(labels: pd.DataFrame, *, quiet: bool) -> None:
    _section("Labels", quiet=quiet)
    _emit(f"Rows          : {len(labels)}", quiet=quiet)
    if "label" in labels.columns:
        _emit(f"Distribution  : {labels['label'].value_counts().to_dict()}", quiet=quiet)
    if "barrier" in labels.columns:
        _emit(f"Barriers      : {labels['barrier'].value_counts().to_dict()}", quiet=quiet)


def _print_alignment_summary(aligned: dict[str, Any], *, quiet: bool) -> None:
    _section("Aligned Matrix", quiet=quiet)
    X = aligned["X"]
    y = aligned.get("y")
    _emit(f"Samples       : {len(X)}", quiet=quiet)
    _emit(f"Feature count : {X.shape[1]}", quiet=quiet)
    if y is not None:
        _emit(f"Label mix     : {pd.Series(y).value_counts().to_dict()}", quiet=quiet)


def _print_training_summary(training: dict[str, Any], *, config: dict[str, Any], automl: dict[str, Any] | None, quiet: bool) -> None:
    _section("Training", quiet=quiet)
    validation = dict(training.get("validation") or {})
    validation_method = str(validation.get("method") or (config.get("model") or {}).get("cv_method", "cpcv"))
    _emit(f"Model         : {_resolve_model_choice(config=config, automl=automl, training=training)}", quiet=quiet)
    _emit(f"Validation    : {validation_method}", quiet=quiet)
    if validation:
        if validation_method == "cpcv":
            _emit(
                "Split config  : "
                f"splits={validation.get('split_count')}  n_blocks={validation.get('n_blocks')}  "
                f"test_blocks={validation.get('test_blocks')}  embargo={validation.get('embargo_bars')}",
                quiet=quiet,
            )
        else:
            _emit(
                "Split config  : "
                f"splits={validation.get('split_count')}  gap={validation.get('gap')}  expanding={validation.get('expanding')}",
                quiet=quiet,
            )
    selected_columns = list(training.get("last_selected_columns") or [])
    _emit(f"Selected feats: {selected_columns[:15]}{' ...' if len(selected_columns) > 15 else ''}", quiet=quiet)
    _emit(f"Feature count : {len(selected_columns)}", quiet=quiet)
    _emit(f"Avg accuracy  : {_format_metric(training.get('avg_accuracy'))}", quiet=quiet)
    _emit(f"Avg macro F1  : {_format_metric(training.get('avg_f1_macro'))}", quiet=quiet)
    if training.get("avg_log_loss") is not None:
        _emit(f"Avg log loss  : {_format_metric(training.get('avg_log_loss'))}", quiet=quiet)
    if training.get("avg_directional_accuracy") is not None:
        _emit(f"Dir accuracy  : {_format_metric(training.get('avg_directional_accuracy'))}", quiet=quiet)
    feature_adaptation = dict(training.get("feature_adaptation") or {})
    if feature_adaptation:
        last_manifest = dict(feature_adaptation.get("last_manifest") or {})
        _emit(
            "Feature adapt : "
            f"enabled={bool(feature_adaptation.get('enabled', False))}  "
            f"applied={bool(feature_adaptation.get('applied_in_any_fold', False))}  "
            f"deferred={bool(feature_adaptation.get('deferred_runtime', False))}",
            quiet=quiet,
        )
        _emit(
            "Adapt policy  : "
            f"adapter={last_manifest.get('adapter_type', 'identity')}  "
            f"scaling={feature_adaptation.get('requested_scaling_mode', 'identity')}  "
            f"selection={feature_adaptation.get('requested_selection_mode', 'identity')}",
            quiet=quiet,
        )


def _print_backtest_summary(backtest: dict[str, Any], *, quiet: bool) -> None:
    _section("Backtest", quiet=quiet)
    _emit(f"Engine        : {backtest.get('engine')}", quiet=quiet)
    _emit(f"Sharpe        : {_format_metric(backtest.get('sharpe_ratio'))}", quiet=quiet)
    _emit(f"Net return    : {_format_metric(backtest.get('net_profit_pct'), percent=True)}", quiet=quiet)
    _emit(f"Max drawdown  : {_format_metric(backtest.get('max_drawdown'), percent=True)}", quiet=quiet)
    _emit(f"Trades        : {backtest.get('total_trades', 'n/a')}", quiet=quiet)
    if backtest.get("win_rate") is not None:
        _emit(f"Win rate      : {_format_metric(backtest.get('win_rate'), percent=True)}", quiet=quiet)
    router_report = dict(backtest.get("router_stability_report") or {})
    if router_report.get("enabled"):
        _emit(
            "Router        : "
            f"switches={router_report.get('switch_count', 'n/a')}  "
            f"switch_rate={_format_metric(router_report.get('switch_rate'), percent=True)}  "
            f"blocked={router_report.get('blocked_switch_count', 'n/a')}  "
            f"controls={router_report.get('configured_control_count', 'n/a')}",
            quiet=quiet,
        )


def _collect_warnings(*, aligned: dict[str, Any] | None, training: dict[str, Any] | None, backtest: dict[str, Any] | None) -> list[str]:
    warnings: list[str] = []
    sample_count = 0 if not aligned else int(len(aligned.get("X", [])))
    if sample_count and sample_count < 250:
        warnings.append(f"Small modeling sample: {sample_count} aligned rows. Extend the window before trusting fine-grained metrics.")

    training = dict(training or {})
    backtest = dict(backtest or {})
    trade_count = int(backtest.get("total_trades") or 0)
    sharpe_ratio = backtest.get("sharpe_ratio")
    if sharpe_ratio is not None:
        try:
            if float(sharpe_ratio) >= 3.0 and trade_count < 30:
                warnings.append(
                    "Too-good-to-be-true profile: Sharpe is very high relative to the trade count. Treat this as unstable until you widen the sample."
                )
        except (TypeError, ValueError):
            pass

    if sample_count and trade_count and trade_count / sample_count > 0.15:
        warnings.append("High turnover strategy: trade count is high relative to the modeled sample. Review costs and signal thresholds.")

    try:
        avg_accuracy = float(training.get("avg_accuracy")) if training.get("avg_accuracy") is not None else None
    except (TypeError, ValueError):
        avg_accuracy = None
    if avg_accuracy is not None and avg_accuracy >= 0.7 and sample_count < 500:
        warnings.append(
            "Classification accuracy is unusually high for a short sample. Re-check regime stability and extend the evaluation window."
        )

    metric_warnings = list((backtest.get("metric_qualification") or {}).get("warnings") or [])
    warnings.extend(str(item) for item in metric_warnings)
    return warnings


def _print_warnings(warnings: list[str], *, quiet: bool) -> None:
    if not warnings:
        return
    _section("Warnings", quiet=quiet)
    for warning in warnings:
        _emit(f"- {warning}", quiet=quiet)


def _run_manual_experiment(
    pipeline: ResearchPipeline,
    *,
    config: dict[str, Any],
    quiet: bool,
    hooks: Mapping[str, Callable[[ResearchPipeline, Any], Any]] | None,
) -> tuple[ResearchPipeline, dict[str, Any], list[str], str]:
    artifacts: dict[str, Any] = {}
    pipeline_state = getattr(pipeline, "state", {}) or {}

    data = pipeline_state.get("data")
    if data is None:
        data = pipeline.fetch_data()
    data = _apply_hook(hooks, "after_fetch_data", pipeline, data)
    artifacts["data"] = data
    _print_data_summary(data, quiet=quiet)

    indicator_run = pipeline.run_indicators()
    indicator_run = _apply_hook(hooks, "after_run_indicators", pipeline, indicator_run)
    artifacts["indicator_run"] = indicator_run
    _print_indicator_summary(indicator_run, quiet=quiet)

    features = pipeline.build_features()
    features = _apply_hook(hooks, "after_build_features", pipeline, features)
    stationarity = pipeline.check_stationarity()
    stationarity = _apply_hook(hooks, "after_check_stationarity", pipeline, stationarity)
    artifacts["features"] = features
    artifacts["stationarity"] = stationarity
    _print_feature_summary(features, stationarity, quiet=quiet)

    regime_result = pipeline.detect_regimes()
    regime_result = _apply_hook(hooks, "after_detect_regimes", pipeline, regime_result)
    artifacts["regimes"] = regime_result
    _print_regime_summary(regime_result.get("regimes"), quiet=quiet)

    labels = pipeline.build_labels()
    labels = _apply_hook(hooks, "after_build_labels", pipeline, labels)
    artifacts["labels"] = labels
    _print_label_summary(labels, quiet=quiet)

    aligned = pipeline.align_data()
    aligned = _apply_hook(hooks, "after_align_data", pipeline, aligned)
    artifacts["aligned"] = aligned
    _print_alignment_summary(aligned, quiet=quiet)

    artifacts["feature_selection"] = pipeline.select_features()
    artifacts["sample_weights"] = pipeline.compute_sample_weights()

    training = pipeline.train_models()
    training = _apply_hook(hooks, "after_train_models", pipeline, training)
    artifacts["training"] = training
    _print_training_summary(training, config=config, automl=None, quiet=quiet)

    signals = pipeline.generate_signals()
    backtest = pipeline.run_backtest()
    signals = _apply_hook(hooks, "after_generate_signals", pipeline, signals)
    backtest = _apply_hook(hooks, "after_run_backtest", pipeline, backtest)
    artifacts["signals"] = signals
    artifacts["backtest"] = backtest
    _print_backtest_summary(backtest, quiet=quiet)

    warnings = _collect_warnings(aligned=aligned, training=training, backtest=backtest)
    _print_warnings(warnings, quiet=quiet)
    return pipeline, artifacts, warnings, "completed"


def _run_automl_experiment(pipeline: ResearchPipeline, *, config: dict[str, Any], quiet: bool) -> tuple[ResearchPipeline, dict[str, Any], list[str], str]:
    artifacts: dict[str, Any] = {}

    data = pipeline.fetch_data()
    artifacts["data"] = data
    _print_data_summary(data, quiet=quiet)

    indicator_run = pipeline.run_indicators()
    artifacts["indicator_run"] = indicator_run
    _print_indicator_summary(indicator_run, quiet=quiet)

    _section("AutoML", quiet=quiet)
    automl = pipeline.run_automl()
    artifacts["automl"] = automl
    _emit(f"Objective     : {automl.get('objective')}", quiet=quiet)
    _emit(f"Selection     : {automl.get('selection_metric')} ({automl.get('selection_mode')})", quiet=quiet)
    _emit(f"Trials        : {automl.get('trial_count')}", quiet=quiet)
    _emit(f"Best value    : {_format_metric(automl.get('best_value'))}", quiet=quiet)
    _print_automl_bundle_summary(automl, quiet=quiet)

    if not automl.get("best_overrides"):
        selection_outcome = dict(automl.get("selection_outcome") or {})
        warnings = [
            "No AutoML candidate passed the configured selection gates.",
            *[str(reason) for reason in selection_outcome.get("top_rejection_reasons") or []],
        ]
        _print_warnings(warnings, quiet=quiet)
        return pipeline, artifacts, warnings, "no_candidate"

    refit = pipeline.refit_selected_candidate(automl)
    artifacts["refit"] = refit
    refit_pipeline = refit["pipeline"]

    features = refit_pipeline.state["features"]
    stationarity = refit_pipeline.state["stationarity"]
    aligned = {
        "X": refit_pipeline.state["X"],
        "y": refit_pipeline.state["y"],
        "labels": refit_pipeline.state.get("labels_aligned"),
    }
    training = refit["training"]
    backtest = refit["backtest"]

    artifacts["features"] = features
    artifacts["stationarity"] = stationarity
    artifacts["aligned"] = aligned
    artifacts["training"] = training
    artifacts["signals"] = refit["signals"]
    artifacts["backtest"] = backtest

    _print_feature_summary(features, stationarity, quiet=quiet)
    _print_regime_summary(refit_pipeline.state["regimes"], quiet=quiet)
    _print_label_summary(refit_pipeline.state["labels"], quiet=quiet)
    _print_alignment_summary(aligned, quiet=quiet)
    _print_training_summary(training, config=config, automl=automl, quiet=quiet)
    _print_backtest_summary(backtest, quiet=quiet)

    warnings = _collect_warnings(aligned=aligned, training=training, backtest=backtest)
    _print_warnings(warnings, quiet=quiet)
    return refit_pipeline, artifacts, warnings, "completed"


def run_experiment(
    config_source: str | Path | Mapping[str, Any] | ResolvedExperimentConfig,
    *,
    quick: bool = False,
    quiet: bool = False,
    pipeline: ResearchPipeline | None = None,
    hooks: Mapping[str, Callable[[ResearchPipeline, Any], Any]] | None = None,
) -> ExperimentResult:
    """Run a full research experiment from a user-facing config mapping or YAML file."""

    resolved = (
        config_source
        if isinstance(config_source, ResolvedExperimentConfig)
        else load_experiment_config(config_source, quick=quick)
    )
    _print_overview(resolved, quiet=quiet)

    controller_pipeline = pipeline or ResearchPipeline(resolved.config)
    automl_enabled = bool((resolved.config.get("automl") or {}).get("enabled", False))
    if automl_enabled:
        active_pipeline, artifacts, warnings, status = _run_automl_experiment(
            controller_pipeline,
            config=resolved.config,
            quiet=quiet,
        )
    else:
        active_pipeline, artifacts, warnings, status = _run_manual_experiment(
            controller_pipeline,
            config=resolved.config,
            quiet=quiet,
            hooks=hooks,
        )

    return ExperimentResult(
        name=resolved.name,
        status=status,
        config=resolved.config,
        pipeline=active_pipeline,
        warnings=warnings,
        artifacts=artifacts,
        config_path=resolved.config_path,
    )
