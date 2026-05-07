"""Regime-orchestration example using detector and router config sections."""

from core.execution import NAUTILUS_AVAILABLE

from example_entrypoints import parse_example_args, resolve_repo_path, run_example


def _print_orchestration_summary(result, *, quiet: bool):
    if quiet:
        return

    config = result.config
    regime = dict(config.get("regime") or {})
    router = dict(config.get("router") or {})
    model_library = dict(config.get("model_library") or {})
    compatibility = dict(regime.get("compatibility_adapter") or {})
    regime_aware = dict((config.get("model") or {}).get("regime_aware") or {})
    detectors = [
        str(detector.get("name") or detector.get("type") or "detector")
        for detector in list(regime.get("detectors") or [])
    ]
    specialists = [
        str(spec.get("model_id") or spec.get("estimator") or "specialist")
        for spec in list(model_library.get("specialists") or [])
    ]

    print("\nOrchestration")
    print(f"Detectors     : {detectors}")
    print(f"Primary       : {compatibility.get('primary_detector', 'n/a')}")
    print(f"Legacy bridge : method={regime.get('method', 'n/a')} strategy={regime_aware.get('strategy', 'n/a')}")
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


if __name__ == "__main__":
    main()