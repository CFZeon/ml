"""Run the hardened AutoML research profile intended for trade-ready promotion checks.

Usage
-----
    python example_trade_ready_automl.py
    python example_trade_ready_automl.py --smoke

The default run uses the stronger certification profile. `--smoke` switches to
an explicitly reduced-power local feedback profile that is still useful for
debugging the control flow but is not sufficient promotion evidence.
The certification path requires a real Nautilus execution backend for the
trade-ready evaluation path and fails closed when that backend is unavailable.
The explicit `--smoke` path can downgrade itself into a clearly labeled
research-only surrogate run when Nautilus is unavailable locally.
"""

import argparse
from pathlib import Path

from core import ATR, BollingerBands, MACD, RSI, ResearchPipeline
from core.execution import NAUTILUS_AVAILABLE
from example_utils import (
    build_default_certification_scenario_matrix,
    build_spot_research_config,
    build_trade_ready_runtime_overrides,
    build_trade_ready_automl_overrides,
    clone_config_with_overrides,
    print_data_certification_summary,
    print_automl_summary,
    print_section,
)


def build_trade_ready_example_config(*, automl_storage):
    return _build_trade_ready_example_config(automl_storage=automl_storage, power_profile="certification")


def _build_trade_ready_example_config(*, automl_storage, power_profile):
    power_profile = "smoke" if str(power_profile).strip().lower() == "smoke" else "certification"
    significance_min_observations = 32 if power_profile == "smoke" else 64
    scenario_matrix = build_default_certification_scenario_matrix()
    symbol = "BTCUSDT"
    interval = "1h"
    start = "2024-01-01"
    end = "2024-05-01" if power_profile == "smoke" else "2024-07-01"
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
        clone_config_with_overrides(
            build_trade_ready_runtime_overrides(market="spot"),
            {
                "data": {
                    "futures_context": {"enabled": False},
                },
                "reference_data": {
                    "enabled": True,
                    "spot": {
                        "venues": ["coinbase", "kraken"],
                        "partial_coverage_mode": "blocking",
                        "divergence_mode": "blocking",
                        "min_coverage_ratio": 0.95,
                    },
                },
                "data_certification": {
                    "enabled": True,
                    "require_reference_validation": True,
                },
                "monitoring": {
                    "policy_profile": "trade_ready",
                },
                "features": {
                    "schema_version": "indicator_aware_v7_trade_ready_profile",
                    "lookahead_guard": {
                            "enabled": False,
                        "mode": "blocking",
                        "decision_sample_size": 16,
                        "min_prefix_rows": 160,
                        "step_names": ["build_features"],
                        "artifact_names": ["features"],
                    },
                },
                "signals": {
                    "policy_mode": "validation_calibrated",
                },
                "backtest": {
                    "engine": "pandas",
                    "signal_delay_bars": 2,
                    "significance": {
                        "bootstrap_samples": 300 if power_profile == "smoke" else 1000,
                        "min_observations": significance_min_observations,
                    },
                    "required_stress_scenarios": ["downtime", "stale_mark", "halt"],
                    "execution_policy": {
                        "adapter": "nautilus",
                        "time_in_force": "IOC",
                        "participation_cap": 1.0,
                    },
                    "scenario_matrix": scenario_matrix,
                },
            },
        ),
    )
    config = clone_config_with_overrides(
        config,
        build_trade_ready_automl_overrides(
            storage_path=automl_storage,
            study_name=f"BTCUSDT_1h_trade_ready_automl_{power_profile}_v1",
            profile=power_profile,
            extra_automl_fields={
                "seed": 42,
            },
        ),
    )
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Run the trade-ready AutoML certification example.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the explicitly reduced-power smoke profile instead of the default certification profile.",
    )
    return parser.parse_args()


def prepare_trade_ready_runtime_config(config, *, nautilus_available=NAUTILUS_AVAILABLE):
    trade_ready_profile = dict((config.get("automl") or {}).get("trade_ready_profile") or {})
    reduced_power = bool(trade_ready_profile.get("reduced_power", False))
    if nautilus_available:
        return config

    if reduced_power:
        return clone_config_with_overrides(
            config,
            {
                "regime": {
                    "enabled": False,
                },
                "data_quality": {
                    "block_on_quarantine": False,
                },
                "backtest": {
                    "evaluation_mode": "research_only",
                    "execution_profile": "research_surrogate",
                    "research_only_override": True,
                    "required_stress_scenarios": [],
                    "execution_policy": {
                        "adapter": "nautilus",
                        "force_simulation": True,
                        "time_in_force": "IOC",
                        "participation_cap": 1.0,
                    },
                },
                "automl": {
                    "locked_holdout_enabled": False,
                    "selection_policy": {
                        "enabled": False,
                    },
                    "overfitting_control": {
                        "enabled": False,
                        "deflated_sharpe": {"enabled": False},
                        "pbo": {"enabled": False},
                        "post_selection": {"enabled": False},
                    },
                        "replication": {
                            "enabled": False,
                        },
                },
                "example_runtime": {
                    "mode": "research_surrogate",
                    "note": (
                        "Nautilus is unavailable, so the explicit smoke profile downgraded "
                        "to a research-only surrogate execution study."
                    ),
                },
            },
        )

    raise RuntimeError(
        "Trade-ready certification requires a real Nautilus backend. "
        "Install/configure NautilusTrader and rerun example_trade_ready_automl.py, "
        "or use example_local_certification_automl.py for the strict local certification path, "
        "or example_automl.py for the explicit research-only surrogate path."
    )


def main():
    args = parse_args()
    power_profile = "smoke" if args.smoke else "certification"
    sep = "=" * 60

    automl_storage = Path(".cache") / "automl" / f"example_trade_ready_automl_{power_profile}_v1.db"
    automl_storage.parent.mkdir(parents=True, exist_ok=True)
    if automl_storage.exists():
        automl_storage.unlink()

    config = _build_trade_ready_example_config(automl_storage=automl_storage, power_profile=power_profile)
    try:
        config = prepare_trade_ready_runtime_config(config)
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(2) from exc
    trade_ready_profile = dict((config.get("automl") or {}).get("trade_ready_profile") or {})
    example_runtime = dict(config.get("example_runtime") or {})

    pipeline = ResearchPipeline(config)
    monitoring_config = dict(config.get("monitoring") or {})
    data_config = dict(config.get("data") or {})
    data_quality_config = dict(config.get("data_quality") or {})
    significance_config = dict((config.get("backtest") or {}).get("significance") or {})

    print_section(sep, 1, "Trade-ready profile")
    print(f"  profile      : {trade_ready_profile.get('name')}")
    print(f"  reduced power: {bool(trade_ready_profile.get('reduced_power', False))}")
    print(f"  n_trials     : {trade_ready_profile.get('n_trials')}")
    print(f"  validation bt: {trade_ready_profile.get('min_validation_trade_count')} min trades")
    print(f"  monitoring   : {monitoring_config.get('policy_profile', 'trade_ready')}")
    print(
        "  stats floor  : "
        f"min_obs={significance_config.get('min_observations', 'default')}  "
        f"gate_obs={trade_ready_profile.get('min_significance_observations')}"
    )
    print(
        "  data policy  : "
        f"gap={data_config.get('gap_policy', 'fail')}  "
        f"duplicate={data_config.get('duplicate_policy', 'fail')}  "
        f"quarantine_block={bool(data_quality_config.get('block_on_quarantine', True))}"
    )
    if trade_ready_profile.get("reduced_power", False):
        print("  note         : this is a reduced-power smoke run and not sufficient promotion evidence.")
    if example_runtime:
        print(f"  runtime mode : {example_runtime.get('mode', 'unknown')}")
        print(f"  runtime note : {example_runtime.get('note', '')}")

    print_section(sep, 2, "Fetching BTCUSDT spot data")
    data = pipeline.fetch_data()
    print(f"  rows         : {len(data)}")
    print(f"  range        : {data.index[0]} -> {data.index[-1]}")

    print_section(sep, 3, "Running data-quality checks")
    clean = pipeline.check_data_quality()
    print(f"  clean rows   : {len(clean)}")

    print_section(sep, 4, "Running indicators")
    indicator_run = pipeline.run_indicators()
    print(f"  indicators   : {[result.kind for result in indicator_run.results]}")

    print_section(sep, 5, "Certifying pre-training data")
    pipeline.build_features()
    data_certification = pipeline.inspect_data_certification()
    print_data_certification_summary(data_certification)

    print_section(sep, 6, "Running trade-ready AutoML profile")
    automl = pipeline.run_automl()
    print_automl_summary(automl)

    print_section(sep, 7, "Trade-ready interpretation")
    print("  This profile is allowed to reject the winner.")
    print("  A best trial is not automatically a trade-ready trial.")
    print("  A blocking data-certification contract now runs before training and backtesting.")
    print("  A blocking lookahead replay now runs on the feature surface before training.")
    print("  Replication must also pass on alternate cohorts before promotion can succeed.")
    if trade_ready_profile.get("reduced_power", False):
        print("  This run used the reduced-power smoke profile, so a pass is still not certification-grade evidence.")
    print("  Trade-ready evaluation now requires explicit stress cases, reference validation, and a real Nautilus backend.")
    if example_runtime:
        print("  This specific run used the explicit smoke-only research surrogate fallback because Nautilus was unavailable.")
    print("  Use example_automl.py when you need the explicit research-only surrogate path.")
    print("  Check 'promotion ok' and 'promotion why' before treating the model as deployable.")


if __name__ == "__main__":
    main()