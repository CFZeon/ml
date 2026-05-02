"""Run a constrained Optuna-backed AutoML example on the research pipeline.

Usage
-----
    python example_automl.py
    python example_automl.py --research-demo

The default run keeps a locked holdout plus selection-policy gates enabled so
the AutoML search remains separated from untouched OOS evaluation, while still
using research-only surrogate execution. Pass --research-demo only when you
explicitly want the faster unsafe smoke path that disables those controls.
For capital-facing execution evidence, use example_local_certification_automl.py
or example_trade_ready_automl.py with a real Nautilus backend.
"""

import argparse

from pathlib import Path

from core import ATR, BollingerBands, MACD, RSI, ResearchPipeline
from example_utils import (
    build_spot_research_config,
    clone_config_with_overrides,
    print_alignment_summary,
    print_automl_summary,
    print_backtest_summary,
    print_feature_selection_summary,
    print_label_summary,
    print_regime_summary,
    print_section,
    print_signal_summary,
    print_stationarity_summary,
    print_training_summary,
    print_weight_summary,
)


def main():
    args = parse_args()
    sep = "=" * 60
    if args.research_demo:
        print("This run is the explicit fast research-demo path. It disables locked holdout and selection gates.")
        print("Use the default example_automl.py run when you want selection-safe OOS separation on the surrogate backend.")
    else:
        print("This run keeps locked holdout and selection gates enabled by default.")
        print("It also requires a minimum statistical-power floor before treating any candidate as evidence-bearing.")
        print("Execution evidence remains research-only because this entrypoint still uses the surrogate backend.")
    print("Use example_local_certification_automl.py for the strict local certification path.")
    print("Use example_trade_ready_automl.py only when you want the stricter operator-facing certification path with a real Nautilus backend.")

    storage_name = "example_automl_research_demo_v5.db" if args.research_demo else "example_automl_v5.db"
    automl_storage = Path(".cache") / "automl" / storage_name
    automl_storage.parent.mkdir(parents=True, exist_ok=True)
    if automl_storage.exists():
        automl_storage.unlink()

    config = build_example_automl_config(automl_storage=automl_storage, research_demo=args.research_demo)

    pipeline = ResearchPipeline(config)

    print_section(sep, 1, "Fetching BTCUSDT spot data")
    data = pipeline.fetch_data()
    print(f"  rows         : {len(data)}")
    print(f"  range        : {data.index[0]} -> {data.index[-1]}")

    print_section(sep, 2, "Running indicators")
    indicator_run = pipeline.run_indicators()
    print(f"  indicators   : {[result.kind for result in indicator_run.results]}")

    print_section(sep, 3, "Running AutoML search")
    automl = pipeline.run_automl()
    print_automl_summary(automl)

    if not automl.get("best_overrides"):
        selection_outcome = dict(automl.get("selection_outcome") or {})
        print_section(sep, 4, "Selection outcome")
        print("  result       : no candidate passed the research selection gates")
        print(f"  status       : {selection_outcome.get('status', 'unknown')}")
        print(f"  reasons      : {selection_outcome.get('top_rejection_reasons') or ['no_eligible_trial_under_selection_policy']}")
        print("  next step    : widen the sample, relax the search space, or switch to --research-demo for the explicit unsafe smoke path.")
        return

    print_section(sep, 4, "Building explicit research refit artifact")
    print("  warning      : this refit is a post-selection research artifact, not untouched OOS evidence.")
    refit = pipeline.refit_selected_candidate(automl)
    refit_pipeline = refit["pipeline"]

    features = refit_pipeline.state["features"]
    print(f"  feature count: {features.shape[1]}")
    stationarity = refit_pipeline.state["stationarity"]
    print_stationarity_summary(stationarity)

    print_section(sep, 5, "Previewing regime features")
    regimes = refit_pipeline.state["regimes"]
    print("  mode         : disabled for this compact AutoML demo workflow")
    print_regime_summary(regimes)

    print_section(sep, 6, "Building labels and aligning research matrix")
    labels = refit_pipeline.state["labels"]
    print_label_summary(labels)
    aligned = {
        "X": refit_pipeline.state["X"],
        "y": refit_pipeline.state["y"],
        "labels": refit_pipeline.state["labels_aligned"],
    }
    print_alignment_summary(aligned)

    print_section(sep, 7, "Previewing feature-selection and weighting")
    selection = refit_pipeline.state["feature_selection"]
    print_feature_selection_summary(selection)
    weights = refit_pipeline.state["sample_weights"]
    print_weight_summary(weights)

    print_section(sep, 8, "CPCV training")
    training = refit["training"]
    print_training_summary(training)

    print_section(sep, 9, "Generating signals")
    signals = refit["signals"]
    print_signal_summary(signals, allow_short=False)

    print_section(sep, 10, "Backtesting")
    backtest = refit["backtest"]
    print_backtest_summary(backtest)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the research-only AutoML example.")
    parser.add_argument(
        "--research-demo",
        action="store_true",
        help=(
            "Use the faster unsafe smoke path that disables locked holdout and selection gates. "
            "The default run keeps those controls enabled."
        ),
    )
    return parser.parse_args()


def build_example_automl_config(*, automl_storage, research_demo=False):
    symbol = "BTCUSDT"
    interval = "1h"
    start = "2024-01-01"
    end = "2024-05-01" if research_demo else "2024-07-01"
    context_symbols = ["ETHUSDT"]

    config = build_spot_research_config(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        indicators=[RSI(14), MACD(), BollingerBands(20), ATR(14)],
        context_symbols=context_symbols,
    )
    config = clone_config_with_overrides(
        config,
        {
            "data": {
                "futures_context": {"enabled": False},
            },
            "features": {
                "schema_version": "indicator_aware_v7_example_workflow",
            },
            "regime": {
                "enabled": False,
            },
            "signals": {
                "policy_mode": "validation_calibrated",
            },
            "backtest": {
                "signal_delay_bars": 2,
            },
            "automl": {
                "enabled": True,
                "seed": 42,
                "validation_fraction": 0.2,
                "minimum_dsr_threshold": None,
                "enable_pruning": False,
                "objective": "sharpe_ratio",
                "storage": str(automl_storage),
                "search_space": {
                    "features": {
                        "lags": {"type": "categorical", "choices": ["1,3,6", "1,4,12"]},
                        "frac_diff_d": {"type": "categorical", "choices": [0.4, 0.6]},
                        "rolling_window": {"type": "categorical", "choices": [20, 28]},
                        "squeeze_quantile": {"type": "categorical", "choices": [0.15, 0.2]},
                    },
                    "feature_selection": {
                        "enabled": {"type": "categorical", "choices": [True]},
                        "max_features": {"type": "categorical", "choices": [48, 64]},
                        "min_mi_threshold": {"type": "categorical", "choices": [0.0, 0.0005]},
                    },
                    "labels": {
                        "min_return": {"type": "categorical", "choices": [0.0005, 0.001]},
                    },
                    "regime": {
                        "n_regimes": {"type": "categorical", "choices": [2]},
                    },
                    "model": {
                        "type": {"type": "categorical", "choices": ["gbm"]},
                        "gap": {"type": "categorical", "choices": [24]},
                    },
                },
            },
        },
    )
    if research_demo:
        return clone_config_with_overrides(
            config,
            {
                "automl": {
                    "n_trials": 2,
                    "locked_holdout_enabled": False,
                    "study_name": "BTCUSDT_1h_example_automl_research_demo_v5",
                    "selection_policy": {
                        "enabled": False,
                    },
                    "overfitting_control": {
                        "enabled": False,
                        "deflated_sharpe": {"enabled": False},
                        "pbo": {"enabled": False},
                        "post_selection": {"enabled": False},
                    },
                },
                "example_runtime": {
                    "mode": "research_demo",
                },
            },
        )
    return clone_config_with_overrides(
        config,
        {
            "backtest": {
                "significance": {
                    "enabled": True,
                    "bootstrap_samples": 300,
                    "confidence_level": 0.95,
                    "random_state": 42,
                    "min_observations": 64,
                    "min_effective_bets": 64,
                },
            },
            "automl": {
                "n_trials": 2,
                "locked_holdout_enabled": True,
                "locked_holdout_fraction": 0.2,
                "policy_profile": "legacy_permissive",
                "study_name": "BTCUSDT_1h_example_automl_v5",
                "selection_policy": {
                    "enabled": True,
                    "gate_modes": {
                        "locked_holdout": "blocking",
                        "locked_holdout_gap": "blocking",
                    },
                    "max_generalization_gap": 0.35,
                    "min_validation_trade_count": 5,
                    "require_locked_holdout_pass": True,
                    "min_locked_holdout_score": 0.0,
                    "max_trials_per_model_family": 4,
                    "require_fold_stability_pass": False,
                },
                "objective_gates": {
                    "enabled": True,
                    "min_trade_count": 10,
                    "min_effective_bet_count": 16,
                    "require_statistical_significance": True,
                    "min_significance_observations": 64,
                    "min_sharpe_ci_lower": 0.0,
                },
                "overfitting_control": {
                    "enabled": True,
                    "deflated_sharpe": {"enabled": False},
                    "pbo": {"enabled": False},
                    "post_selection": {
                        "enabled": True,
                        "require_pass": True,
                        "pass_rule": "spa",
                        "alpha": 0.05,
                        "max_candidates": 8,
                        "correlation_threshold": 0.9,
                        "min_overlap_fraction": 0.5,
                        "min_overlap_observations": 12,
                        "overlap_policy": "strict_intersection",
                        "bootstrap_samples": 300,
                        "random_state": 42,
                    },
                },
            },
            "example_runtime": {
                "mode": "research_only_locked_holdout_power_gated",
            },
        },
    )


if __name__ == "__main__":
    main()