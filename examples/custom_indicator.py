"""Custom-indicator example. Edit indicators/range_position.py and rerun this file."""

from pathlib import Path

from experiments import run_experiment


if __name__ == "__main__":
    run_experiment(Path(__file__).resolve().parents[1] / "configs" / "btc_custom_indicator.yaml")
