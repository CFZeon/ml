"""Config-driven AutoML example that compares orchestration bundles."""

from example_entrypoints import parse_example_args, resolve_repo_path, run_example


def main():
    args = parse_example_args(
        "Run the orchestration-bundle AutoML demo.",
        include_local_certification=False,
    )
    run_example(
        resolve_repo_path("configs", "btc_regime_bundle_automl.yaml"),
        market="spot",
        quick=args.quick,
        quiet=args.quiet,
        example_name="example_regime_bundle_automl.py",
    )


if __name__ == "__main__":
    main()