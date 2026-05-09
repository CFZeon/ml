"""Filtered-HMM regime example using the detector-backed replay runtime."""

from core.execution import NAUTILUS_AVAILABLE

from example_entrypoints import parse_example_args, resolve_repo_path, run_example
from example_utils import print_phase_zero_contract_summary


def _print_hmm_summary(result, *, quiet: bool):
    if quiet:
        return

    config = dict(result.config or {})
    regime = dict(config.get("regime") or {})
    detectors = [dict(detector) for detector in list(regime.get("detectors") or []) if isinstance(detector, dict)]
    primary = detectors[0] if detectors else {}
    params = dict(primary.get("params") or {})
    pipeline_state = dict(getattr(getattr(result, "pipeline", None), "state", {}) or {})
    runtime = dict(pipeline_state.get("regime_detection") or {})
    manifests = list(runtime.get("detector_manifests") or [])
    primary_manifest = manifests[0] if manifests else None
    posterior_mode = "n/a" if primary_manifest is None else dict(primary_manifest.metadata or {}).get("posterior_mode", "n/a")

    print("\nFiltered HMM")
    print(
        "Configured    : "
        f"detector={primary.get('name', 'n/a')} "
        f"states={params.get('n_regimes', params.get('state_count', 'n/a'))} "
        f"covariance={params.get('covariance_type', 'diag')}"
    )
    print(
        "Runtime       : "
        f"type={runtime.get('detector_type', 'n/a')} "
        f"method={runtime.get('method', 'n/a')} "
        f"posterior={posterior_mode}"
    )


def main():
    args = parse_example_args("Run the filtered-HMM regime example.")
    result = run_example(
        resolve_repo_path("configs", "btc_filtered_hmm.yaml"),
        market="spot",
        local_certification=args.local_certification,
        quick=args.quick,
        quiet=args.quiet,
        nautilus_available=NAUTILUS_AVAILABLE,
        example_name="example_filtered_hmm.py",
    )
    _print_hmm_summary(result, quiet=args.quiet)
    print_phase_zero_contract_summary(result, quiet=args.quiet)


if __name__ == "__main__":
    main()