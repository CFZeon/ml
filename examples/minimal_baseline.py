"""Minimal baseline example: single symbol, two indicators, CPCV validation."""

from pathlib import Path

from experiments import run_experiment


if __name__ == "__main__":
    run_experiment(Path(__file__).resolve().parents[1] / "configs" / "btc_baseline.yaml")
