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


def _format_confidence_interval(interval, digits=4, percent=False, money=False):
    if not isinstance(interval, dict):
        return None
    lower = interval.get("lower")
    upper = interval.get("upper")
    confidence_level = interval.get("confidence_level")
    if lower is None or upper is None or confidence_level is None:
        return None
    rendered_lower = _format_metric(lower, digits=digits, percent=percent, money=money)
    rendered_upper = _format_metric(upper, digits=digits, percent=percent, money=money)
    return f"{confidence_level:.0%} [{rendered_lower}, {rendered_upper}]"


def print_section(sep, step, title):
    print(f"\n{sep}\nStep {step} · {title}\n{sep}")


def build_example_universe_config(
    primary_symbol,
    *,
    context_symbols=None,
    market="spot",
    snapshot_timestamp=None,
    min_history_days=30,
    min_liquidity=1_000_000.0,
    requested_symbol_policy="error",
):
    """Build a self-contained historical-universe config for runnable examples.

    Examples that request cross-asset context now need an explicit universe snapshot.
    Keeping the snapshot inline avoids hidden cache dependencies while still exercising
    the same eligibility path as the production pipeline.
    """

    resolved_snapshot = pd.Timestamp(snapshot_timestamp or "2024-01-01", tz="UTC")
    symbols = [str(primary_symbol), *[str(symbol) for symbol in (context_symbols or [])]]
    ordered_symbols = list(dict.fromkeys(symbols))
    base_liquidity = max(float(min_liquidity) * 10.0, 10_000_000.0)

    snapshot_symbols = []
    for index, symbol in enumerate(ordered_symbols):
        snapshot_symbols.append(
            {
                "symbol": symbol,
                "market": market,
                "status": "TRADING",
                "listing_start": "2020-01-01T00:00:00Z",
                "avg_daily_quote_volume": float(base_liquidity - (index * float(min_liquidity))),
            }
        )

    return {
        "market": market,
        "snapshots": [
            {
                "snapshot_timestamp": resolved_snapshot.isoformat().replace("+00:00", "Z"),
                "market": market,
                "symbols": snapshot_symbols,
            }
        ],
        "requested_symbol_policy": requested_symbol_policy,
        "min_history_days": int(min_history_days),
        "min_liquidity": float(min_liquidity),
    }


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
    print("  preview mode : supervised MI filtering runs inside each validation split")
    print(
        "  configured   : "
        f"max_features={report.get('max_features') or 'auto'}  "
        f"min_mi={report.get('min_mi_threshold')}"
    )
    family_summary = report.get("input_family_summary", {})
    if family_summary.get("selected_family_counts"):
        print(f"  families     : {family_summary['selected_family_counts']}")


def print_weight_summary(weights):
    print(
        "  weight stats : "
        f"min={_format_metric(weights.min())}  "
        f"max={_format_metric(weights.max())}  "
        f"mean={_format_metric(weights.mean())}"
    )


def print_training_summary(training):
    validation = training.get("validation", {})
    if validation:
        if validation.get("method") == "cpcv":
            print(
                "  validation   : "
                f"cpcv  splits={validation.get('split_count')}  "
                f"n_blocks={validation.get('n_blocks')}  "
                f"test_blocks={validation.get('test_blocks')}  "
                f"embargo={validation.get('embargo_bars')}"
            )
        else:
            print(
                "  validation   : "
                f"walk_forward  splits={validation.get('split_count')}  "
                f"gap={validation.get('gap')}"
            )

    for metric in training.get("fold_metrics", []):
        split_label = metric.get("split_id") or f"fold {metric.get('fold')}"
        parts = [
            split_label,
            f"acc={_format_metric(metric.get('accuracy'))}",
            f"f1={_format_metric(metric.get('f1_macro'))}",
        ]
        if metric.get("directional_accuracy") is not None:
            parts.append(f"dir_acc={_format_metric(metric.get('directional_accuracy'))}")
        if metric.get("directional_f1_macro") is not None:
            parts.append(f"dir_f1={_format_metric(metric.get('directional_f1_macro'))}")
        if metric.get("log_loss") is not None:
            parts.append(f"log_loss={_format_metric(metric.get('log_loss'))}")
        if metric.get("test_blocks") is not None:
            parts.append(f"test_blocks={metric.get('test_blocks')}")
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
        if feature_selection.get("selected_families"):
            print(f"  families     : {feature_selection.get('selected_families')}")
        if feature_selection.get("avg_selected_family_counts"):
            print(f"  avg fam cnt  : {feature_selection.get('avg_selected_family_counts')}")
        if feature_selection.get("endogenous_only_selected_all_folds"):
            print("  family mode  : endogenous-only selected in every fold")
    bootstrap = training.get("bootstrap", {})
    if bootstrap:
        print(
            "  bootstrap    : "
            f"used={bootstrap.get('used_in_any_fold')}  "
            f"warnings={bootstrap.get('warning_count', 0)}"
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

    fold_stability = training.get("fold_stability", {})
    if fold_stability.get("metrics"):
        primary_metric = fold_stability.get("primary_metric")
        primary_stats = fold_stability.get("metrics", {}).get(primary_metric, {})
        if primary_metric and primary_stats:
            print(
                "  stability    : "
                f"{primary_metric} cv={_format_metric(primary_stats.get('cv'), digits=6)}  "
                f"range=[{_format_metric(primary_stats.get('min'), digits=6)}, {_format_metric(primary_stats.get('max'), digits=6)}]"
            )
        if fold_stability.get("worst_fold_sharpe") is not None:
            print(f"  worst sharpe : {_format_metric(fold_stability.get('worst_fold_sharpe'), digits=6)}")
        if fold_stability.get("worst_fold_net_profit_pct") is not None:
            print(f"  worst return : {_format_metric(fold_stability.get('worst_fold_net_profit_pct'), digits=6, percent=True)}")
        if fold_stability.get("max_drawdown_dispersion") is not None:
            print(f"  dd dispersion: {_format_metric(fold_stability.get('max_drawdown_dispersion'), digits=6)}")
        if fold_stability.get("policy_enabled"):
            print(f"  stability ok : {fold_stability.get('passed')}  reasons={fold_stability.get('reasons', [])}")

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

    family_diag = training.get("feature_family_diagnostics", {})
    if family_diag.get("summary"):
        print("  top families :")
        for family in family_diag["summary"][:5]:
            print(
                "    "
                f"{family['family']}: f1_drop={_format_metric(family.get('avg_f1_drop'), digits=6)}  "
                f"acc_drop={_format_metric(family.get('avg_accuracy_drop'), digits=6)}  "
                f"native={_format_metric(family.get('avg_native_importance'), digits=6)}"
            )
    if family_diag.get("bundles"):
        print("  family bundles:")
        for bundle in family_diag["bundles"][:4]:
            print(
                "    "
                f"{bundle['bundle']}: f1_drop={_format_metric(bundle.get('avg_f1_drop_vs_full'), digits=6)}  "
                f"acc_drop={_format_metric(bundle.get('avg_accuracy_drop_vs_full'), digits=6)}  "
                f"families={bundle.get('families')}"
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
    paths = signal_result.get("paths") or []
    if paths:
        long_counts = []
        short_counts = []
        flat_counts = []
        avg_abs_sizes = []
        for path in paths:
            signals = pd.Series(path["signals"], copy=False)
            executable_signals = signals if allow_short else signals.clip(lower=0)
            long_counts.append(float((executable_signals == 1).sum()))
            short_counts.append(float((executable_signals == -1).sum()))
            flat_counts.append(float((executable_signals == 0).sum()))
            continuous = path.get("continuous_signals")
            if continuous is not None:
                continuous_series = pd.Series(continuous, copy=False)
                executable_continuous = continuous_series if allow_short else continuous_series.clip(lower=0.0)
                avg_abs_sizes.append(float(executable_continuous.abs().mean()))

        print(f"  path count   : {len(paths)}")
        print(
            "  avg signal mix: "
            f"long={_format_metric(sum(long_counts) / len(long_counts), digits=2)}  "
            f"short={_format_metric(sum(short_counts) / len(short_counts), digits=2)}  "
            f"flat={_format_metric(sum(flat_counts) / len(flat_counts), digits=2)}"
        )
        if avg_abs_sizes:
            print(f"  avg abs size : {_format_metric(sum(avg_abs_sizes) / len(avg_abs_sizes))}")
        return

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
    if backtest.get("path_count"):
        print(
            "  validation   : "
            f"{backtest.get('validation_method')}  "
            f"aggregate={backtest.get('aggregate_mode')}  "
            f"paths={backtest.get('path_count')}"
        )

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
    if backtest.get("account_model") == "futures_margin":
        print(
            "  futures acct : "
            f"mode={backtest.get('futures_margin_mode')}  "
            f"brackets={backtest.get('futures_bracket_count')}"
        )
        print(
            "  liquidations : "
            f"count={backtest.get('liquidation_event_count')}  "
            f"fees={_format_metric(backtest.get('liquidation_fee_paid'), money=True)}"
        )
        print(
            "  margin risk  : "
            f"max_ratio={_format_metric(backtest.get('max_margin_ratio'), digits=6)}  "
            f"warn={backtest.get('bars_above_margin_warning')} bars "
            f"({_format_metric(backtest.get('bars_above_margin_warning_rate'), percent=True)})"
        )
        print(
            "  leverage use : "
            f"avg={_format_metric(backtest.get('avg_realized_leverage'), digits=6)}  "
            f"max={_format_metric(backtest.get('max_realized_leverage'), digits=6)}  "
            f"capped={backtest.get('leverage_cap_adjustments')}"
        )
    metric_ranges = backtest.get("metric_ranges") or {}
    if metric_ranges:
        for key, label in [("net_profit_pct", "net range"), ("sharpe_ratio", "sharpe rng"), ("max_drawdown", "mdd range")]:
            stats = metric_ranges.get(key)
            if not stats:
                continue
            percent = key in {"net_profit_pct", "max_drawdown"}
            print(
                f"  {label:<12}: "
                f"min={_format_metric(stats.get('min'), percent=percent)}  "
                f"med={_format_metric(stats.get('median'), percent=percent)}  "
                f"max={_format_metric(stats.get('max'), percent=percent)}"
            )

    significance = backtest.get("statistical_significance") or {}
    if significance.get("enabled"):
        print(
            "  stats        : "
            f"{significance.get('method')}  "
            f"samples={significance.get('bootstrap_samples')}  "
            f"ci={significance.get('confidence_level', 0.95):.0%}  "
            f"block={significance.get('mean_block_length')}"
        )
        if significance.get("aggregate_mode") is not None:
            print(
                "  stats agg    : "
                f"{significance.get('aggregate_mode')}  "
                f"paths={significance.get('path_count')}"
            )
        if significance.get("benchmark_sharpe_ratio") is not None:
            print(f"  bench sharpe : {_format_metric(significance.get('benchmark_sharpe_ratio'))}")

        stats_metrics = significance.get("metrics") or {}
        for label, key, formatter in [
            ("sharpe ci", "sharpe_ratio", None),
            ("sortino ci", "sortino_ratio", None),
            ("calmar ci", "calmar_ratio", None),
            ("net ret ci", "net_profit_pct", "percent"),
            ("mdd ci", "max_drawdown", "percent"),
        ]:
            metric = stats_metrics.get(key)
            if not metric:
                continue
            interval = _format_confidence_interval(
                metric.get("confidence_interval"),
                digits=4,
                percent=formatter == "percent",
            )
            if interval is None:
                continue
            suffix = []
            if metric.get("p_value_gt_zero") is not None:
                suffix.append(f"p>0={_format_metric(metric.get('p_value_gt_zero'), digits=6)}")
            if metric.get("p_value_gt_benchmark") is not None:
                suffix.append(f"p>bench={_format_metric(metric.get('p_value_gt_benchmark'), digits=6)}")
            suffix_text = f"  {'  '.join(suffix)}" if suffix else ""
            print(f"  {label:<12}: {interval}{suffix_text}")
    elif significance.get("reason"):
        print(f"  stats        : unavailable ({significance.get('reason')})")


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

    best_objective = automl.get("best_objective_diagnostics") or {}
    if best_objective:
        parts = [
            f"score={_format_metric(best_objective.get('raw_score'))}",
        ]
        if best_objective.get("primary_metric"):
            parts.append(
                f"{best_objective['primary_metric']}={_format_metric(best_objective['components'].get(best_objective['primary_metric']))}"
            )
        if best_objective.get("primary_metric_source"):
            parts.append(f"source={best_objective.get('primary_metric_source')}")
        if best_objective.get("benchmark_reference") not in (None, 0.0):
            parts.append(f"benchmark={_format_metric(best_objective.get('benchmark_reference'))}")
        print(f"  objective det: {'  '.join(parts)}")

        gates = best_objective.get("classification_gates") or {}
        if gates.get("enabled"):
            failed = gates.get("failed") or []
            print(
                "  objective gate: "
                f"passed={gates.get('passed')}  "
                f"failed={failed if failed else 'none'}"
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

    validation_holdout = automl.get("validation_holdout") or {}
    if validation_holdout.get("enabled"):
        validation_backtest = validation_holdout.get("backtest") or {}
        print(
            "  validation   : "
            f"rows={validation_holdout.get('aligned_validation_rows')}  "
            f"range={validation_holdout.get('start_timestamp')} -> {validation_holdout.get('end_timestamp')}"
        )
        if validation_backtest:
            print(
                "  validation bt: "
                f"sharpe={_format_metric(validation_backtest.get('sharpe_ratio'))}  "
                f"net={_format_metric(validation_backtest.get('net_profit_pct'), percent=True)}  "
                f"trades={validation_backtest.get('total_trades')}"
            )

    locked_holdout = automl.get("locked_holdout") or {}
    if locked_holdout.get("enabled"):
        holdout_backtest = locked_holdout.get("backtest") or {}
        print(
            "  final holdout: "
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
        if locked_holdout.get("holdout_warning"):
            print("  holdout warn : Sharpe CI lower bound is below zero")