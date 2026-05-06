"""Shared entry helpers for root example scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping

from experiments import ResolvedExperimentConfig, load_experiment_config, run_experiment
from core import ResearchPipeline
from example_utils import prepare_example_runtime_config


def parse_example_args(description: str, *, include_local_certification: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Apply the example's quick_overrides for a shorter smoke run.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce example output to the runner's essential summaries.",
    )
    if include_local_certification:
        parser.add_argument(
            "--local-certification",
            action="store_true",
            help=(
                "Run this example with the strict local-certification runtime profile. "
                "Uses a real local Nautilus backend when available, otherwise falls back to an explicit surrogate "
                "certification mode that remains paper or pre-capital only."
            ),
        )
    return parser.parse_args()


def resolve_repo_path(*parts: str) -> Path:
    return Path(__file__).resolve().parent.joinpath(*parts)


def run_example(
    config_source: str | Path | Mapping[str, Any] | ResolvedExperimentConfig,
    *,
    market: str = "spot",
    local_certification: bool = False,
    quick: bool = False,
    quiet: bool = False,
    nautilus_available: bool = True,
    example_name: str = "example",
    pipeline: ResearchPipeline | None = None,
    hooks: Mapping[str, Callable[[ResearchPipeline, Any], Any]] | None = None,
):
    resolved = (
        config_source
        if isinstance(config_source, ResolvedExperimentConfig)
        else load_experiment_config(config_source, quick=quick)
    )

    config = resolved.config
    if local_certification:
        config = prepare_example_runtime_config(
            config,
            market=market,
            local_certification=True,
            nautilus_available=nautilus_available,
            example_name=example_name,
        )
        resolved = ResolvedExperimentConfig(
            name=resolved.name,
            raw_config=resolved.raw_config,
            config=config,
            config_path=resolved.config_path,
            quick_mode=resolved.quick_mode,
        )

    return run_experiment(resolved, quiet=quiet, pipeline=pipeline, hooks=hooks)