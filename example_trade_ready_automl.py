"""Run the hardened AutoML research profile intended for trade-ready promotion checks.

Usage
-----
    python example_trade_ready_automl.py

This example keeps the search budget constrained for consumer hardware, but it
does enable the controls that a promotion-safe workflow needs: locked holdout,
replication cohorts, selection gating, DSR/PBO diagnostics, binding post-selection
inference, and explicit stress scenarios for the post-selection backtest.
The base config requires a real Nautilus execution backend for the trade-ready
evaluation path. When Nautilus is unavailable, the script now fails closed unless
you explicitly opt into a research-only surrogate override.
It may still report `promotion ok : False` if the locked holdout, execution,
replication, or stress gates reject the candidate.
"""

from pathlib import Path

from core import ATR, BollingerBands, MACD, RSI, ResearchPipeline
from core.execution import NAUTILUS_AVAILABLE
from example_utils import (
    build_spot_research_config,
    build_trade_ready_automl_overrides,
    clone_config_with_overrides,
    print_automl_summary,
    print_section,
)


def build_trade_ready_example_config(*, automl_storage):
    symbol = "BTCUSDT"
    interval = "1h"
    start = "2024-01-01"
    end = "2024-05-01"
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
                "schema_version": "indicator_aware_v7_trade_ready_profile",
            },
            "signals": {
                "policy_mode": "validation_calibrated",
            },
            "backtest": {
                "engine": "pandas",
                "signal_delay_bars": 2,
                "evaluation_mode": "trade_ready",
                "execution_profile": "trade_ready_event_driven",
                "research_only_override": False,
                "required_stress_scenarios": ["downtime", "stale_mark", "halt"],
                "execution_policy": {
                    "adapter": "nautilus",
                    "time_in_force": "IOC",
                    "participation_cap": 0.10,
                    "min_fill_ratio": 0.25,
                },
                "scenario_matrix": {
                    "downtime": {
                        "events": [{"event_type": "downtime", "timestamp": "2024-03-05T12:00:00Z"}],
                        "policy": {"downtime_action": "freeze"},
                    },
                    "stale_mark": {
                        "events": [{"event_type": "stale_mark", "timestamp": "2024-03-12T12:00:00Z"}],
                        "policy": {"stale_mark_action": "reject"},
                    },
                    "halt": {
                        "events": [{"event_type": "halt", "start": "2024-03-19T12:00:00Z", "end": "2024-03-19T14:00:00Z"}],
                        "policy": {},
                    },
                },
            },
        },
    )
    config = clone_config_with_overrides(
        config,
        build_trade_ready_automl_overrides(
            storage_path=automl_storage,
            study_name="BTCUSDT_1h_trade_ready_automl_v1",
            n_trials=2,
            search_space={
                "features": {
                    "lags": {"type": "categorical", "choices": ["1,4,12"]},
                    "frac_diff_d": {"type": "categorical", "choices": [0.4]},
                    "rolling_window": {"type": "categorical", "choices": [20]},
                    "squeeze_quantile": {"type": "categorical", "choices": [0.2]},
                },
                "feature_selection": {
                    "enabled": {"type": "categorical", "choices": [True]},
                    "max_features": {"type": "categorical", "choices": [64]},
                    "min_mi_threshold": {"type": "categorical", "choices": [0.0005]},
                },
                "labels": {
                    "pt_mult": {"type": "categorical", "choices": [1.5]},
                    "sl_mult": {"type": "categorical", "choices": [2.0]},
                    "max_holding": {"type": "categorical", "choices": [24]},
                    "min_return": {"type": "categorical", "choices": [0.0005]},
                    "volatility_window": {"type": "categorical", "choices": [24]},
                    "barrier_tie_break": {"type": "categorical", "choices": ["sl"]},
                },
                "regime": {
                    "n_regimes": {"type": "categorical", "choices": [2]},
                },
                "model": {
                    "type": {"type": "categorical", "choices": ["gbm"]},
                    "gap": {"type": "categorical", "choices": [24]},
                    "validation_fraction": {"type": "categorical", "choices": [0.2]},
                    "meta_n_splits": {"type": "categorical", "choices": [2]},
                    "params": {
                        "gbm": {
                            "n_estimators": {"type": "categorical", "choices": [400]},
                            "learning_rate": {"type": "categorical", "choices": [0.05]},
                            "max_depth": {"type": "categorical", "choices": [3]},
                            "subsample": {"type": "categorical", "choices": [0.7]},
                            "min_samples_leaf": {"type": "categorical", "choices": [3]},
                        },
                    },
                },
            },
            extra_automl_fields={
                "seed": 42,
            },
        ),
    )
    return config


def prepare_trade_ready_runtime_config(config, *, nautilus_available=NAUTILUS_AVAILABLE):
    if nautilus_available:
        return config, False

    if not bool((config.get("backtest") or {}).get("research_only_override", False)):
        raise RuntimeError(
            "Trade-ready example requires NautilusTrader or backtest.research_only_override=true for an explicit research-only fallback."
        )

    runtime_config = clone_config_with_overrides(
        config,
        {
            "backtest": {
                "evaluation_mode": "research_only",
                "execution_profile": "research_surrogate",
                "execution_policy": {
                    "force_simulation": True,
                },
            },
            "automl": {
                "locked_holdout_enabled": False,
                "minimum_dsr_threshold": None,
                "selection_policy": {
                    "enabled": False,
                },
                "overfitting_control": {
                    "enabled": False,
                    "deflated_sharpe": {"enabled": False},
                    "pbo": {"enabled": False},
                    "post_selection": {"enabled": False, "require_pass": False},
                },
            },
        },
    )
    return runtime_config, True


def main():
    sep = "=" * 60

    automl_storage = Path(".cache") / "automl" / "example_trade_ready_automl_v1.db"
    automl_storage.parent.mkdir(parents=True, exist_ok=True)
    if automl_storage.exists():
        automl_storage.unlink()

    config = build_trade_ready_example_config(automl_storage=automl_storage)
    try:
        config, using_research_fallback = prepare_trade_ready_runtime_config(config)
    except RuntimeError as exc:
        print(str(exc))
        return
    if using_research_fallback:
        print("NautilusTrader is unavailable; running the hardened AutoML profile in research-only fallback mode.")
        print("Promotion readiness will remain non-deployable until you rerun with a real Nautilus backend.")

    pipeline = ResearchPipeline(config)

    print_section(sep, 1, "Fetching BTCUSDT spot data")
    data = pipeline.fetch_data()
    print(f"  rows         : {len(data)}")
    print(f"  range        : {data.index[0]} -> {data.index[-1]}")

    print_section(sep, 2, "Running indicators")
    indicator_run = pipeline.run_indicators()
    print(f"  indicators   : {[result.kind for result in indicator_run.results]}")

    print_section(sep, 3, "Running trade-ready AutoML profile")
    automl = pipeline.run_automl()
    print_automl_summary(automl)

    print_section(sep, 4, "Trade-ready interpretation")
    print("  This profile is allowed to reject the winner.")
    print("  A best trial is not automatically a trade-ready trial.")
    print("  Replication must also pass on alternate cohorts before promotion can succeed.")
    if using_research_fallback:
        print("  This run used the research-only surrogate fallback because Nautilus was unavailable.")
        print("  The locked holdout was also disabled so the local smoke run can complete on the full sample.")
        print("  DSR/PBO/post-selection gates were relaxed for this fallback run.")
        print("  Re-run with a real Nautilus backend to get a true trade-ready promotion verdict.")
    else:
        print("  Trade-ready evaluation now requires explicit stress cases and a real Nautilus backend.")
    print("  Check 'promotion ok' and 'promotion why' before treating the model as deployable.")


if __name__ == "__main__":
    main()