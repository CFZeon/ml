"""Shared helpers for the repository example scripts."""

from math import isfinite

import pandas as pd


def _format_metric(value, digits=4, percent=False, money=False):
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not isfinite(numeric):
        return str(value)
    if money:
        return f"${numeric:,.2f}"
    if percent:
        return f"{numeric:.2%}"
    return f"{numeric:.{digits}f}"


def print_section(sep, step, title):
    print(f"\n{sep}\nStep {step} · {title}\n{sep}")


def print_stationarity_summary(stationarity):
    for feature_name in ["close", "close_fracdiff"]:
        stats = stationarity.get(feature_name)
        if not isinstance(stats, dict):
            continue
        print(
            f"  {feature_name:<13}: stationary={stats.get('stationary')}  "
            f"p={_format_metric(stats.get('p_value'), digits=6)}"
        )

    screening = stationarity.get("feature_screening", {}).get("summary", {})
    if screening:
        print(
            "  screened     : "
            f"{screening.get('screened_feature_count', 0)}/{screening.get('total_features', 0)}  "
            f"transformed={screening.get('transformed_features', 0)}  "
            f"dropped={screening.get('dropped_features', 0)}"
        )
        transform_usage = screening.get("transform_usage")
        if transform_usage:
            print(f"  transforms   : {transform_usage}")


def print_regime_summary(regimes):
    if isinstance(regimes, pd.DataFrame):
        print(f"  regime cols  : {list(regimes.columns)}")
        regime_counts = regimes["regime"].value_counts().to_dict() if "regime" in regimes.columns else {}
    else:
        regime_counts = pd.Series(regimes).value_counts().to_dict()
    print(f"  regime counts: {regime_counts}")


def print_label_summary(labels):
    print(f"  label rows   : {len(labels)}")
    if "label" in labels.columns:
        print(f"  distribution : {labels['label'].value_counts().to_dict()}")
    if "barrier" in labels.columns:
        print(f"  barriers     : {labels['barrier'].value_counts().to_dict()}")
    if "trend_horizon" in labels.columns:
        print(
            "  trend horizon: "
            f"median={_format_metric(labels['trend_horizon'].median(), digits=2)}  "
            f"max={_format_metric(labels['trend_horizon'].max(), digits=2)}"
        )


def print_alignment_summary(aligned):
    X = aligned["X"]
    y = aligned.get("y")
    print(f"  samples      : {len(X)}")
    print(f"  feature count: {X.shape[1]}")
    if y is not None:
        print(f"  label mix    : {pd.Series(y).value_counts().to_dict()}")


def print_feature_selection_summary(selection):
    report = getattr(selection, "report", {})
    print("  preview mode : supervised MI filtering runs inside each walk-forward fold")
    print(
        "  configured   : "
        f"max_features={report.get('max_features') or 'auto'}  "
        f"min_mi={report.get('min_mi_threshold')}"
    )


def print_weight_summary(weights):
    print(
        "  weight stats : "
        f"min={_format_metric(weights.min())}  "
        f"max={_format_metric(weights.max())}  "
        f"mean={_format_metric(weights.mean())}"
    )


def print_training_summary(training):
    for metric in training.get("fold_metrics", []):
        parts = [
            f"fold {metric.get('fold')}",
            f"acc={_format_metric(metric.get('accuracy'))}",
            f"f1={_format_metric(metric.get('f1_macro'))}",
        ]
        if metric.get("directional_accuracy") is not None:
            parts.append(f"dir_acc={_format_metric(metric.get('directional_accuracy'))}")
        if metric.get("directional_f1_macro") is not None:
            parts.append(f"dir_f1={_format_metric(metric.get('directional_f1_macro'))}")
        if metric.get("log_loss") is not None:
            parts.append(f"log_loss={_format_metric(metric.get('log_loss'))}")
        print(f"  {'  '.join(parts)}")

    print(f"  avg accuracy : {_format_metric(training.get('avg_accuracy'))}")
    print(f"  avg f1       : {_format_metric(training.get('avg_f1_macro'))}")
    if training.get("avg_directional_accuracy") is not None:
        print(f"  avg dir acc  : {_format_metric(training.get('avg_directional_accuracy'))}")
    if training.get("avg_directional_f1_macro") is not None:
        print(f"  avg dir f1   : {_format_metric(training.get('avg_directional_f1_macro'))}")
    if training.get("avg_log_loss") is not None:
        print(f"  avg log loss : {_format_metric(training.get('avg_log_loss'))}")
    if training.get("avg_brier_score") is not None:
        print(f"  avg brier    : {_format_metric(training.get('avg_brier_score'))}")
    if training.get("avg_calibration_error") is not None:
        print(f"  avg calib    : {_format_metric(training.get('avg_calibration_error'))}")

    feature_selection = training.get("feature_selection", {})
    if feature_selection:
        print(
            "  avg selected : "
            f"{_format_metric(feature_selection.get('avg_selected_features'), digits=2)}"
        )
    fallback_scope = training.get("fallback_inference", {})
    if fallback_scope:
        print(
            "  fallback     : "
            f"mode={fallback_scope.get('mode')}  "
            f"safe_rows={fallback_scope.get('aligned_safe_row_count', 0)}  "
            f"last_train_end={fallback_scope.get('last_fold_train_end')}"
        )
    if training.get("last_signal_params"):
        print(f"  tuned signals: {training['last_signal_params']}")

    block_diag = training.get("feature_block_diagnostics", {})
    if block_diag.get("summary"):
        print("  top blocks   :")
        for block in block_diag["summary"][:5]:
            print(
                "    "
                f"{block['block']}: f1_drop={_format_metric(block.get('avg_f1_drop'), digits=6)}  "
                f"acc_drop={_format_metric(block.get('avg_accuracy_drop'), digits=6)}  "
                f"native={_format_metric(block.get('avg_native_importance'), digits=6)}"
            )


def print_signal_summary(signal_result, allow_short=True):
    signal_source = signal_result.get("signal_source")
    if signal_source is not None:
        print(f"  source       : {signal_source}")
    fallback_scope = signal_result.get("fallback_scope", {})
    if signal_source is not None and signal_source.startswith("post_final_training_fallback"):
        print(
            "  fallback     : "
            f"scored={fallback_scope.get('scored_row_count', fallback_scope.get('aligned_safe_row_count', 0))}  "
            f"excluded={fallback_scope.get('excluded_row_count', 0)}"
        )
    signals = pd.Series(signal_result["signals"], copy=False)
    executable_signals = signals if allow_short else signals.clip(lower=0)
    print(
        "  signal mix   : "
        f"long={int((executable_signals == 1).sum())}  "
        f"short={int((executable_signals == -1).sum())}  "
        f"flat={int((executable_signals == 0).sum())}"
    )
    if not allow_short and int((signals == -1).sum()) > 0:
        print(f"  clipped shorts: {int((signals == -1).sum())}")
    continuous = signal_result.get("continuous_signals")
    if continuous is not None:
        continuous_series = pd.Series(continuous, copy=False)
        executable_continuous = continuous_series if allow_short else continuous_series.clip(lower=0.0)
        print(f"  avg abs size : {_format_metric(executable_continuous.abs().mean())}")


def print_backtest_summary(backtest):
    for label, key, formatter in [
        ("engine", "engine", None),
        ("start equity", "starting_equity", "money"),
        ("end equity", "ending_equity", "money"),
        ("net profit", "net_profit", "money"),
        ("net return", "net_profit_pct", "percent"),
        ("gross profit", "gross_profit", "money"),
        ("gross loss", "gross_loss", "money"),
        ("funding pnl", "funding_pnl", "money"),
        ("fees paid", "fees_paid", "money"),
        ("slippage", "slippage_paid", "money"),
        ("sharpe ratio", "sharpe_ratio", None),
        ("sortino", "sortino_ratio", None),
        ("calmar", "calmar_ratio", None),
        ("CAGR", "cagr", "percent"),
        ("volatility", "annualized_volatility", "percent"),
        ("max drawdown", "max_drawdown", "percent"),
        ("profit factor", "profit_factor", None),
        ("expectancy", "expectancy", "money"),
        ("avg win", "avg_win", "money"),
        ("avg loss", "avg_loss", "money"),
        ("exposure", "exposure_rate", "percent"),
        ("signal delay", "signal_delay_bars", None),
        ("trades", "total_trades", None),
        ("closed trades", "closed_trades", None),
        ("win rate", "win_rate", "percent"),
        ("trade win rt", "trade_win_rate", "percent"),
    ]:
        if key not in backtest:
            continue
        value = backtest.get(key)
        if formatter == "money":
            rendered = _format_metric(value, money=True)
        elif formatter == "percent":
            rendered = _format_metric(value, percent=True)
        else:
            rendered = _format_metric(value)
        print(f"  {label:<12}: {rendered}")

    if "max_drawdown_amount" in backtest:
        print(f"  dd amount    : {_format_metric(backtest.get('max_drawdown_amount'), money=True)}")
    if "max_drawdown_duration_bars" in backtest:
        print(
            "  dd duration  : "
            f"{backtest.get('max_drawdown_duration_bars')} bars "
            f"({backtest.get('max_drawdown_duration')})"
        )


def print_automl_summary(automl):
    print(f"  study name   : {automl.get('study_name')}")
    print(f"  objective    : {automl.get('objective')}")
    print(f"  selection    : {automl.get('selection_metric')} ({automl.get('selection_mode')})")
    print(f"  trials       : {automl.get('trial_count')}")
    print(f"  best value   : {_format_metric(automl.get('best_value'))}")
    if automl.get("best_value_raw") is not None:
        print(f"  raw value    : {_format_metric(automl.get('best_value_raw'))}")
    print(f"  best params  : {automl.get('best_params')}")

    best_training = automl.get("best_training", {})
    if best_training:
        print(
            "  best train   : "
            f"dir_acc={_format_metric(best_training.get('avg_directional_accuracy'))}  "
            f"acc={_format_metric(best_training.get('avg_accuracy'))}  "
            f"log_loss={_format_metric(best_training.get('avg_log_loss'))}"
        )

    best_backtest = automl.get("best_backtest", {})
    if best_backtest:
        print(
            "  best backtest: "
            f"sharpe={_format_metric(best_backtest.get('sharpe_ratio'))}  "
            f"net={_format_metric(best_backtest.get('net_profit_pct'), percent=True)}  "
            f"mdd={_format_metric(best_backtest.get('max_drawdown'), percent=True)}  "
            f"trades={best_backtest.get('total_trades')}"
        )

    best_overfitting = automl.get("best_overfitting", {}).get("deflated_sharpe", {})
    if best_overfitting:
        print(
            "  best DSR     : "
            f"{_format_metric(best_overfitting.get('deflated_sharpe_ratio'))}  "
            f"obs_sr={_format_metric(best_overfitting.get('observed_sharpe_ratio'))}  "
            f"bench_sr={_format_metric(best_overfitting.get('benchmark_sharpe_ratio'))}"
        )

    diagnostics = automl.get("overfitting_diagnostics", {})
    pbo = diagnostics.get("pbo", {})
    if pbo.get("enabled"):
        print(
            "  CPCV PBO     : "
            f"pbo={_format_metric(pbo.get('probability_of_backtest_overfitting'))}  "
            f"splits={pbo.get('split_count')}  "
            f"lambda_med={_format_metric(pbo.get('lambda_median'))}"
        )
    elif pbo:
        print(f"  CPCV PBO     : disabled ({pbo.get('reason')})")

    locked_holdout = automl.get("locked_holdout") or {}
    if locked_holdout.get("enabled"):
        holdout_backtest = locked_holdout.get("backtest") or {}
        print(
            "  holdout      : "
            f"rows={locked_holdout.get('aligned_holdout_rows')}  "
            f"range={locked_holdout.get('start_timestamp')} -> {locked_holdout.get('end_timestamp')}"
        )
        if holdout_backtest:
            print(
                "  holdout bt   : "
                f"sharpe={_format_metric(holdout_backtest.get('sharpe_ratio'))}  "
                f"net={_format_metric(holdout_backtest.get('net_profit_pct'), percent=True)}  "
                f"trades={holdout_backtest.get('total_trades')}"
            )