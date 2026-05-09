"""Regime-orchestration example using detector and router config sections."""

from core.execution import NAUTILUS_AVAILABLE

from example_entrypoints import parse_example_args, resolve_repo_path, run_example
from example_utils import print_phase_zero_contract_summary


def _resolve_primary_detector_name(regime):
    detectors = [dict(detector) for detector in list(regime.get("detectors") or []) if isinstance(detector, dict)]
    ensemble = dict(regime.get("ensemble") or {})
    primary_name = str(ensemble.get("primary_detector", "") or "").strip()
    if primary_name:
        return primary_name
    for detector in detectors:
        if detector.get("primary", False):
            return str(detector.get("name") or detector.get("type") or "detector")
    if detectors:
        return str(detectors[0].get("name") or detectors[0].get("type") or "detector")
    return "n/a"


def _print_orchestration_summary(result, *, quiet: bool):
    if quiet:
        return

    config = result.config
    regime = dict(config.get("regime") or {})
    router = dict(config.get("router") or {})
    model_library = dict(config.get("model_library") or {})
    regime_aware = dict((config.get("model") or {}).get("regime_aware") or {})
    pipeline_state = dict(getattr(getattr(result, "pipeline", None), "state", {}) or {})
    runtime = dict(pipeline_state.get("regime_detection") or {})
    manifests = list(runtime.get("detector_manifests") or [])
    primary_manifest = manifests[0] if manifests else None
    detectors = [
        str(detector.get("name") or detector.get("type") or "detector")
        for detector in list(regime.get("detectors") or [])
    ]
    specialists = [
        str(spec.get("model_id") or spec.get("estimator") or "specialist")
        for spec in list(model_library.get("specialists") or [])
    ]

    print("\nOrchestration")
    print(f"Configured    : detectors={detectors}")
    print(f"Primary       : {_resolve_primary_detector_name(regime)}")
    if primary_manifest is not None:
        posterior_mode = dict(primary_manifest.metadata or {}).get("posterior_mode", "n/a")
        print(
            "Runtime       : "
            f"type={primary_manifest.detector_type} "
            f"method={runtime.get('method', 'n/a')} "
            f"posterior={posterior_mode}"
        )
    else:
        print(f"Runtime       : method={runtime.get('method', regime.get('method', 'n/a'))}")
    print(f"Model path    : strategy={regime_aware.get('strategy', 'n/a')}")
    print(f"Router        : {router.get('type', 'n/a')}")
    print(f"Specialists   : {specialists}")


def main():
    args = parse_example_args("Run the regime orchestration example.")
    result = run_example(
        resolve_repo_path("configs", "btc_regime_aware.yaml"),
        market="spot",
        local_certification=args.local_certification,
        quick=args.quick,
        quiet=args.quiet,
        nautilus_available=NAUTILUS_AVAILABLE,
        example_name="example_regime_orchestration.py",
    )
    _print_orchestration_summary(result, quiet=args.quiet)
    print_phase_zero_contract_summary(result, quiet=args.quiet)


if __name__ == "__main__":
    main()