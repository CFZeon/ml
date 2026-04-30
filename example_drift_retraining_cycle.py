"""Deterministic drift-retraining demo with champion promotion and rollback.

Usage
-----
    python example_drift_retraining_cycle.py
"""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from core import (
    LocalRegistryStore,
    ResearchPipeline,
    build_model,
    create_promotion_eligibility_report,
    finalize_promotion_eligibility_report,
    resolve_canonical_promotion_score,
    upsert_promotion_gate,
)
from example_utils import (
    print_deployment_readiness_summary,
    print_operational_limits_summary,
    print_paper_calibration_summary,
    print_section,
)


def _fit_logistic_model():
    X = pd.DataFrame({"f1": [0.0, 1.0, 0.0, 1.0], "f2": [1.0, 1.0, 0.0, 0.0]})
    y = pd.Series([0, 1, 0, 1])
    model = build_model("logistic", {"c": 1.0})
    model.fit(X, y)
    return model, list(X.columns)


def _make_eligibility_report(score_value):
    score = resolve_canonical_promotion_score(
        locked_holdout_report={"raw_objective_value": score_value},
        selection_value=score_value,
    )
    report = create_promotion_eligibility_report(score_basis=score["basis"], score_value=score["value"])
    for group, name in (
        ("selection", "feature_admission"),
        ("selection", "feature_portability"),
        ("selection", "regime_stability"),
        ("selection", "operational_health"),
        ("post_selection", "locked_holdout"),
        ("post_selection", "locked_holdout_gap"),
        ("post_selection", "execution_realism"),
        ("post_selection", "stress_realism"),
    ):
        report = upsert_promotion_gate(report, group=group, name=name, passed=True)
    return finalize_promotion_eligibility_report(report)


def _make_drift_inputs(current_periods=240):
    reference_index = pd.date_range("2026-10-01", periods=300, freq="1h", tz="UTC")
    current_index = pd.date_range("2026-11-01", periods=current_periods, freq="1h", tz="UTC")
    reference_features = pd.DataFrame(
        {
            "alpha": np.random.default_rng(1).normal(0.0, 1.0, len(reference_index)),
            "beta": np.random.default_rng(2).normal(0.0, 1.0, len(reference_index)),
        },
        index=reference_index,
    )
    current_features = pd.DataFrame(
        {
            "alpha": np.random.default_rng(3).normal(3.0, 1.0, len(current_index)),
            "beta": np.random.default_rng(4).normal(3.0, 1.0, len(current_index)),
        },
        index=current_index,
    )
    reference_predictions = pd.DataFrame({"p0": 0.8, "p1": 0.2}, index=reference_index)
    current_predictions = pd.DataFrame({"p0": 0.2, "p1": 0.8}, index=current_index)
    performance = pd.Series(
        np.r_[np.full(len(current_index) // 2, 0.6), np.full(len(current_index) - (len(current_index) // 2), -0.4)],
        index=current_index,
    )
    return reference_features, current_features, reference_predictions, current_predictions, performance


def _register_champion(store, symbol, score_value):
    model, feature_columns = _fit_logistic_model()
    report = _make_eligibility_report(score_value)
    version_id = store.register_version(
        model,
        symbol=symbol,
        feature_columns=feature_columns,
        training_summary={"avg_f1_macro": 0.75},
        validation_summary={"raw_objective_value": score_value, "promotion_ready": True},
        promotion_eligibility_report=report,
    )
    store.promote(version_id, "champion", symbol=symbol, decision={"approved": True, "reasons": ["approved"]})
    return version_id


def _make_challenger_payload(score_value, *, monitoring_report=None):
    model, feature_columns = _fit_logistic_model()
    report = _make_eligibility_report(score_value)
    return {
        "model": model,
        "feature_columns": feature_columns,
        "training_summary": {"avg_f1_macro": 0.80},
        "validation_summary": {"raw_objective_value": score_value, "promotion_ready": True},
        "promotion_eligibility_report": report,
        "sample_count": 320,
        "monitoring_report": dict(monitoring_report or {"healthy": True, "reasons": []}),
    }


def _make_paper_observations(*, periods=35, mode="paper"):
    index = pd.date_range("2026-12-01", periods=periods, freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": index,
            "mode": mode,
            "trade_count": np.full(periods, 12.0),
            "modeled_slippage_bps": np.full(periods, 1.8),
            "realized_slippage_bps": np.linspace(1.86, 1.92, periods),
            "modeled_fill_ratio": np.full(periods, 0.94),
            "realized_fill_ratio": np.linspace(0.915, 0.905, periods),
            "data_breach": np.zeros(periods, dtype=int),
            "funding_breach": np.zeros(periods, dtype=int),
            "kill_switch_trigger": np.zeros(periods, dtype=int),
        }
    )


def _make_runtime_equity_curve(*, periods=24, final_drawdown=0.04):
    index = pd.date_range("2027-01-01", periods=periods, freq="1h", tz="UTC")
    peak_segment = np.linspace(1.0, 1.12, periods // 2, endpoint=True)
    trough_segment = np.linspace(1.12, 1.12 * (1.0 - final_drawdown), periods - len(peak_segment), endpoint=True)
    values = np.r_[peak_segment, trough_segment]
    return pd.Series(values, index=index, dtype=float)


def main():
    sep = "=" * 60
    symbol = "BTCUSDT"
    registry_root = Path(".cache") / "registry" / "drift_cycle_demo"
    if registry_root.exists():
        shutil.rmtree(registry_root)
    registry_root.mkdir(parents=True, exist_ok=True)

    reference_features, current_features, reference_predictions, current_predictions, performance = _make_drift_inputs()
    store = LocalRegistryStore(root_dir=registry_root)
    champion_id = _register_champion(store, symbol, 0.12)

    pipeline = ResearchPipeline({"data": {"symbol": symbol, "interval": "1h"}})
    pipeline.state["X"] = current_features
    pipeline.state["training"] = {"oos_probabilities": current_predictions}
    pipeline.state["backtest"] = {"equity_curve": (1.0 + performance).cumprod()}
    pipeline.state["operational_monitoring"] = {"healthy": True, "reasons": []}

    print_section(sep, 1, "Promoting an approved challenger")
    promoted = pipeline.run_drift_retraining_cycle(
        store=store,
        reference_features=reference_features,
        reference_predictions=reference_predictions,
        bars_since_last_retrain=800,
        scheduled_window_open=True,
        train_challenger=lambda: _make_challenger_payload(0.18),
    )
    print(f"  previous champion : {champion_id}")
    print(f"  retrain status    : {promoted['retrain_status']}")
    print(f"  candidate version : {promoted['candidate_version_id']}")
    print(f"  new champion      : {store.get_champion(symbol)['version_id']}")
    print(f"  promotion reasons : {promoted['promotion_decision'].get('reasons')}")

    print_section(sep, 2, "Paper validation loop")
    paper_report = pipeline.inspect_paper_trading_calibration(
        certified_expectations={"modeled_slippage_bps": 1.8, "modeled_fill_ratio": 0.94},
        paper_observations=_make_paper_observations(),
    )
    print_paper_calibration_summary(paper_report)
    current_champion = store.get_champion(symbol)["version_id"]
    paper_path = store.attach_paper_report(current_champion, paper_report, symbol=symbol)
    pipeline.state.pop("paper_calibration", None)
    print(f"  attached to  : {current_champion}")
    print(f"  report path  : {paper_path}")

    print_section(sep, 3, "Micro-capital release gate")
    green_limits = pipeline.inspect_operational_limits(
        operational_limits={"healthy": True, "kill_switch_ready": True},
        equity_curve=_make_runtime_equity_curve(final_drawdown=0.04),
    )
    print_operational_limits_summary(green_limits)
    readiness = pipeline.inspect_deployment_readiness(
        store=store,
        backend_status={"adapter": "nautilus", "available": True, "reasons": []},
        release_request={"requested_stage": "micro_capital", "manual_acknowledged": True},
    )
    print_deployment_readiness_summary(readiness)

    print_section(sep, 4, "Model TTL expiry hold")
    current_champion_row = store.get_champion(symbol)
    freshness_anchor = (
        pd.Timestamp(current_champion_row.get("promoted_at"))
        if pd.notna(current_champion_row.get("promoted_at"))
        else pd.Timestamp(current_champion_row.get("created_at"))
    )
    stale_readiness = pipeline.inspect_deployment_readiness(
        store=store,
        backend_status={"adapter": "nautilus", "available": True, "reasons": []},
        release_request={"requested_stage": "micro_capital", "manual_acknowledged": True},
        policy={
            "max_model_age_days": 28,
            "warn_model_age_days": 21,
            "as_of_timestamp": freshness_anchor + pd.Timedelta(days=35),
        },
    )
    print_deployment_readiness_summary(stale_readiness)

    print_section(sep, 5, "Kill switch breach and rollback")
    breached_limits = pipeline.inspect_operational_limits(
        operational_limits={"healthy": True, "kill_switch_ready": True},
        equity_curve=_make_runtime_equity_curve(final_drawdown=0.14),
    )
    print_operational_limits_summary(breached_limits)
    pipeline.state["operational_monitoring"] = {"healthy": True, "reasons": []}
    rollback_result = pipeline.run_drift_retraining_cycle(
        store=store,
        reference_features=reference_features,
        reference_predictions=reference_predictions,
        bars_since_last_retrain=900,
        scheduled_window_open=True,
        train_challenger=lambda: _make_challenger_payload(0.08),
        rollback_policy={"mode": "hybrid"},
    )
    print(f"  retrain status    : {rollback_result['retrain_status']}")
    print(f"  candidate version : {rollback_result['candidate_version_id']}")
    print(f"  rollback status   : {rollback_result['rollback'].get('status')}")
    print(f"  rollback reasons  : {rollback_result['rollback'].get('reasons')}")
    print(f"  restored champion : {rollback_result['rollback'].get('restored_version_id')}")
    print(f"  current champion  : {store.get_champion(symbol)['version_id']}")

    print_section(sep, 6, "Operator hold decision after rollback")
    blocked_readiness = pipeline.inspect_deployment_readiness(
        store=store,
        backend_status={"adapter": "nautilus", "available": True, "reasons": []},
        release_request={"requested_stage": "micro_capital", "manual_acknowledged": True},
    )
    print_deployment_readiness_summary(blocked_readiness)

    print(f"\n{sep}\nDrift retraining example complete. Registry root: {registry_root}\n{sep}")


if __name__ == "__main__":
    main()