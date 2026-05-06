"""CLI entrypoint for config-driven research experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from core import list_available_indicators
from experiments import load_experiment_config, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a config-driven research experiment.")
    parser.add_argument("--config", type=Path, help="Path to a YAML experiment config.")
    parser.add_argument("--quick", action="store_true", help="Apply config.quick_overrides for a shorter debug run.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output to warnings only.")
    parser.add_argument("--list-indicators", action="store_true", help="Print the available indicator kinds and exit.")
    parser.add_argument("--validate-only", action="store_true", help="Validate and expand the config without running the pipeline.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_indicators:
        for kind in list_available_indicators():
            print(kind)
        return 0

    if args.config is None:
        raise SystemExit("--config is required unless --list-indicators is used")

    resolved = load_experiment_config(args.config, quick=args.quick)
    if args.validate_only:
        print(f"Validated: {resolved.name}")
        print(f"Config path: {resolved.config_path}")
        print(f"Indicators: {resolved.config.get('indicators')}")
        return 0

    result = run_experiment(resolved, quiet=args.quiet)
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
