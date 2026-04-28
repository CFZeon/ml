"""Run the accessible local-certification AutoML profile.

Usage
-----
    python example_local_certification_automl.py

This example is the strict local certification path between the lightweight
research demo and the fully trade-ready operator workflow. It is intended for
consumer hardware, but it still requires a local Nautilus installation and does
not silently downgrade to the research demo.
"""

from pathlib import Path

from core import ATR, BollingerBands, MACD, RSI, ResearchPipeline
from core.execution import NAUTILUS_AVAILABLE
from example_utils import (
    build_default_certification_scenario_matrix,
    build_local_certification_automl_overrides,
    build_local_certification_runtime_overrides,
    build_spot_research_config,
    clone_config_with_overrides,
    print_automl_summary,
    print_data_certification_summary,
    print_section,
)


def build_local_certification_example_config(*, automl_storage):
    symbol = "BTCUSDT"
    interval = "1h"
    start = "2024-01-01"
    end = "2024-06-01"
    context_symbols = ["ETHUSDT"]

    scenario_matrix = build_default_certification_scenario_matrix()
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
        clone_config_with_overrides(
            build_local_certification_runtime_overrides(market="spot"),
            {
                "data": {
                    "futures_context": {"enabled": False},
                },
                "features": {
                    "schema_version": "indicator_aware_v7_local_certification",
                },
                "regime": {
                    "enabled": False,
                },
                "signals": {
                    "policy_mode": "validation_calibrated",
                },
                "backtest": {
                    "signal_delay_bars": 2,
                    "significance": {
                        "bootstrap_samples": 400,
                        "min_observations": 48,
                    },
                    "scenario_matrix": scenario_matrix,
                },
            },
        ),
    )
    config = clone_config_with_overrides(
        config,
        build_local_certification_automl_overrides(
            storage_path=automl_storage,
            study_name="BTCUSDT_1h_local_certification_automl_v1",
            seed=42,
        ),
    )
    config["example_runtime"] = {
        "mode": "local_certification",
        "risk_level": "paper_or_pre_capital_only",
    }
    return config


def prepare_local_certification_runtime_config(config, *, nautilus_available=NAUTILUS_AVAILABLE):
    if nautilus_available:
        return config
    raise RuntimeError(
        "Local certification requires a local Nautilus installation. "
        "Install/configure NautilusTrader and rerun example_local_certification_automl.py, "
        "or use example_automl.py for the explicit research-only demo."
    )


def main():
    sep = "=" * 60
    automl_storage = Path(".cache") / "automl" / "example_local_certification_automl_v1.db"
    automl_storage.parent.mkdir(parents=True, exist_ok=True)
    if automl_storage.exists():
        automl_storage.unlink()

    config = build_local_certification_example_config(automl_storage=automl_storage)
    try:
        config = prepare_local_certification_runtime_config(config)
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    pipeline = ResearchPipeline(config)
    profile = dict((config.get("automl") or {}).get("trade_ready_profile") or {})

    print_section(sep, 1, "Local certification profile")
    print(f"  profile      : {profile.get('name')}")
    print(f"  n_trials     : {profile.get('n_trials')}")
    print(f"  eval mode    : {config.get('backtest', {}).get('evaluation_mode')}")
    print(f"  exec profile : {config.get('backtest', {}).get('execution_profile')}")
    print("  note         : this run is certification-grade evidence for paper or micro-capital gating, not a live release.")
    print("  funding note : futures cases must keep 'funding cov' at strict before you treat any PnL as certification-grade.")
    print("  monitor note : 'op envelope' must pass with no missing telemetry before local certification can be treated as admissible.")

    print_section(sep, 2, "Fetching BTCUSDT spot data")
    data = pipeline.fetch_data()
    print(f"  rows         : {len(data)}")
    print(f"  range        : {data.index[0]} -> {data.index[-1]}")

    print_section(sep, 3, "Running indicators")
    indicator_run = pipeline.run_indicators()
    print(f"  indicators   : {[result.kind for result in indicator_run.results]}")

    print_section(sep, 4, "Data certification preview")
    certification = pipeline.inspect_data_certification()
    print_data_certification_summary(certification)

    print_section(sep, 5, "Running local certification AutoML")
    automl = pipeline.run_automl()
    print_automl_summary(automl)


if __name__ == "__main__":
    main()