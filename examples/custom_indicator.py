"""Custom-indicator example via the shared config-driven runner."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.execution import NAUTILUS_AVAILABLE
from example_entrypoints import parse_example_args, resolve_repo_path, run_example


def main():
    args = parse_example_args("Run the custom indicator example.")
    run_example(
        resolve_repo_path("configs", "btc_custom_indicator.yaml"),
        market="spot",
        local_certification=args.local_certification,
        quick=args.quick,
        quiet=args.quiet,
        nautilus_available=NAUTILUS_AVAILABLE,
        example_name="examples/custom_indicator.py",
    )


if __name__ == "__main__":
    main()
