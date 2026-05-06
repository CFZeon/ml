"""Baseline spot research example via the shared config-driven runner."""

from core.execution import NAUTILUS_AVAILABLE

from example_entrypoints import parse_example_args, resolve_repo_path, run_example


def main():
    args = parse_example_args("Run the baseline spot research example.")
    run_example(
        resolve_repo_path("configs", "btc_baseline.yaml"),
        market="spot",
        local_certification=args.local_certification,
        quick=args.quick,
        quiet=args.quiet,
        nautilus_available=NAUTILUS_AVAILABLE,
        example_name="example.py",
    )


if __name__ == "__main__":
    main()
