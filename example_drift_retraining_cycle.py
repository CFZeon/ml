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
from example_utils import print_deployment_readiness_summary, print_section


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

    print_section(sep, 2, "Operator deployment readiness")
    readiness = pipeline.inspect_deployment_readiness(
        store=store,
        backend_status={"adapter": "nautilus", "available": True, "reasons": []},
    )
    print_deployment_readiness_summary(readiness)

    print_section(sep, 3, "Rejecting a challenger and rolling back")
    pipeline.state["operational_monitoring"] = {"healthy": False, "reasons": ["feature_schema"]}
    rollback_result = pipeline.run_drift_retraining_cycle(
        store=store,
        reference_features=reference_features,
        reference_predictions=reference_predictions,
        bars_since_last_retrain=900,
        scheduled_window_open=True,
        train_challenger=lambda: _make_challenger_payload(0.08, monitoring_report={"healthy": False, "reasons": ["feature_schema"]}),
        current_monitoring_report=pipeline.state["operational_monitoring"],
        rollback_policy={"mode": "hybrid", "critical_reasons": ["feature_schema"]},
    )
    print(f"  retrain status    : {rollback_result['retrain_status']}")
    print(f"  candidate version : {rollback_result['candidate_version_id']}")
    print(f"  rollback status   : {rollback_result['rollback'].get('status')}")
    print(f"  restored champion : {rollback_result['rollback'].get('restored_version_id')}")
    print(f"  current champion  : {store.get_champion(symbol)['version_id']}")

    print_section(sep, 4, "Operator hold decision after rollback")
    blocked_readiness = pipeline.inspect_deployment_readiness(
        store=store,
        backend_status={"adapter": "nautilus", "available": True, "reasons": []},
    )
    print_deployment_readiness_summary(blocked_readiness)

    print(f"\n{sep}\nDrift retraining example complete. Registry root: {registry_root}\n{sep}")


if __name__ == "__main__":
    main()