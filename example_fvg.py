"""End-to-end pipeline using a config-driven Fair Value Gap indicator setup.

Usage
-----
    python example_fvg.py
    python example_fvg.py --local-certification
"""

import pandas as pd

from core.execution import NAUTILUS_AVAILABLE
from example_entrypoints import parse_example_args, run_example
from example_utils import build_spot_research_config, clone_config_with_overrides


def build_fvg_regime_features(pipeline):
    data = pipeline.require("data")
    features = pipeline.require("features")
    return pd.DataFrame(
        {
            "vol_20": data["close"].pct_change().rolling(20).std(),
            "fvg_gap_imbalance": features["fvg_main_gap_imbalance"],
            "fvg_distance_spread": features["fvg_main_distance_spread"],
        }
    ).dropna()


def main():
    args = parse_example_args("Run the Fair Value Gap feature example.")
    end = "2024-03-01" if args.quick else "2024-06-01"
    base_config = build_spot_research_config(
        symbol="BTCUSDT",
        interval="1h",
        start="2024-01-01",
        end=end,
        indicators=[
            {"kind": "rsi", "params": {"period": 14}},
            {"kind": "atr", "params": {"period": 14}},
            {"kind": "fvg", "params": {"name": "fvg_main", "min_gap_pct": 0.0005}},
        ],
        context_symbols=[],
    )
    config = clone_config_with_overrides(
        base_config,
        {
            "experiment": {
                "name": "fvg_spot",
                "description": "End-to-end spot pipeline using a config-driven Fair Value Gap indicator setup.",
            },
            "data": {
                "futures_context": {"enabled": False},
            },
            "features": {
                "context_timeframes": [],
            },
            "regime": {"method": "hmm", "builder": build_fvg_regime_features},
            "quick_overrides": {
                "data": {
                    "end": "2024-02-15",
                },
                "regime": {"enabled": False},
                "model": {
                    "type": "logistic",
                    "cv_method": "walk_forward",
                    "n_splits": 1,
                    "train_size": 240,
                    "test_size": 48,
                    "gap": 6,
                },
            },
            "signals": {
                "policy_mode": "frozen_manual",
                "avg_win": 0.04,
                "avg_loss": 0.01,
                "shrinkage_alpha": 0.5,
                "fraction": 0.75,
                "min_trades_for_kelly": 30,
                "max_kelly_fraction": 0.5,
                "threshold": 0.0,
                "edge_threshold": 0.0,
                "meta_threshold": 0.0,
                "profitability_threshold": 0.0,
                "expected_edge_threshold": 0.0,
                "sizing_mode": "expected_utility",
                "tuning_min_trades": 5,
            },
        },
    )
    run_example(
        config,
        market="spot",
        local_certification=args.local_certification,
        quick=args.quick,
        quiet=args.quiet,
        nautilus_available=NAUTILUS_AVAILABLE,
        example_name="example_fvg.py",
    )


if __name__ == "__main__":
    main()