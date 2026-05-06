"""Copy-and-edit research template backed by the shared runner."""

from core.execution import NAUTILUS_AVAILABLE

from example_entrypoints import parse_example_args, resolve_repo_path, run_example


def main():
    args = parse_example_args("Run the copy-and-edit research template.")
    run_example(
        resolve_repo_path("configs", "template_case.yaml"),
        market="spot",
        local_certification=args.local_certification,
        quick=args.quick,
        quiet=args.quiet,
        nautilus_available=NAUTILUS_AVAILABLE,
        example_name="example_test_case_template.py",
    )


if __name__ == "__main__":
    main()