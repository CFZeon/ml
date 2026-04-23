"""Run the hardened AutoML research profile intended for trade-ready promotion checks.

Usage
-----
    python example_trade_ready_automl.py

This example keeps the search budget constrained for consumer hardware, but it
does enable the controls that a promotion-safe workflow needs: locked holdout,
replication cohorts, selection gating, DSR/PBO diagnostics, binding post-selection
inference, and explicit stress scenarios for the post-selection backtest.
It requires a real Nautilus execution backend for the trade-ready evaluation path.
It may still report `promotion ok : False` if the locked holdout, execution,
replication, or stress gates reject the candidate.
"""

from pathlib import Path

from core import ATR, BollingerBands, MACD, RSI, ResearchPipeline
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
                "required_stress_scenarios": ["downtime", "stale_mark", "halt"],
                "execution_policy": {
                    "adapter": "nautilus",
                    "time_in_force": "IOC",
                    "participation_cap": 1.0,
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
            n_trials=4,
            extra_automl_fields={
                "seed": 42,
            },
        ),
    )
    return config


def main():
    sep = "=" * 60

    automl_storage = Path(".cache") / "automl" / "example_trade_ready_automl_v1.db"
    automl_storage.parent.mkdir(parents=True, exist_ok=True)
    if automl_storage.exists():
        automl_storage.unlink()

    config = build_trade_ready_example_config(automl_storage=automl_storage)

    pipeline = ResearchPipeline(config)

    print_section(sep, 1, "Running trade-ready AutoML profile")
    automl = pipeline.run_automl()
    print_automl_summary(automl)

    print_section(sep, 2, "Trade-ready interpretation")
    print("  This profile is allowed to reject the winner.")
    print("  A best trial is not automatically a trade-ready trial.")
    print("  Replication must also pass on alternate cohorts before promotion can succeed.")
    print("  Trade-ready evaluation now requires explicit stress cases and a real Nautilus backend.")
    print("  If you want a surrogate execution study, keep it in research_only mode instead.")
    print("  Check 'promotion ok' and 'promotion why' before treating the model as deployable.")


if __name__ == "__main__":
    main()