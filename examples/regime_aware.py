"""Regime-aware example using the detector-backed config-driven entrypoint."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from example_entrypoints import parse_example_args, run_example
from example_utils import print_phase_zero_contract_summary


if __name__ == "__main__":
    args = parse_example_args("Run the regime-aware example.", include_local_certification=False)
    result = run_example(
        "configs/btc_regime_aware.yaml",
        market="spot",
        quick=args.quick,
        quiet=args.quiet,
        example_name="examples/regime_aware.py",
    )
    print_phase_zero_contract_summary(result, quiet=args.quiet)
