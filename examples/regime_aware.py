"""Regime-aware example using the new config-driven entrypoint."""

from pathlib import Path

from experiments import run_experiment
from example_utils import print_phase_zero_contract_summary


if __name__ == "__main__":
    result = run_experiment(Path(__file__).resolve().parents[1] / "configs" / "btc_regime_aware.yaml")
    print_phase_zero_contract_summary(result)
