"""Active futures example via the shared config-driven runner."""

from core.execution import NAUTILUS_AVAILABLE

from example_entrypoints import parse_example_args, resolve_repo_path, run_example


def main():
    args = parse_example_args("Run the active futures demo.")
    run_example(
        resolve_repo_path("configs", "btc_active_futures.yaml"),
        market="um_futures",
        local_certification=args.local_certification,
        quick=args.quick,
        quiet=args.quiet,
        nautilus_available=NAUTILUS_AVAILABLE,
        example_name="example_active_futures.py",
    )


if __name__ == "__main__":
    main()