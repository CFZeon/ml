# Examples

These examples are the main user-facing entrypoints for the research framework.

Quick start:

```bash
python run.py --config configs/btc_baseline.yaml
python run.py --config configs/btc_baseline.yaml --quick
python run.py --list-indicators
```

Runnable examples:

```bash
python examples/minimal_baseline.py
python examples/multi_feature_funding.py
python examples/regime_aware.py
python examples/custom_indicator.py
python examples/walk_forward.py
```

Workflow:

1. Copy one of the YAML files from `configs/`.
2. Change the symbol, dates, indicators, or model settings.
3. If you want a new feature, copy `indicators/range_position.py` and keep the same `compute(df)` contract.
4. Run the config through `python run.py --config ...`.
5. Use `--quick` while iterating on indicator code, then rerun the full window before trusting the metrics.