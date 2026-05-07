"""End-to-end pipeline using a config-driven Fair Value Gap indicator setup.

Usage
-----
    python example_fvg.py
    python example_fvg.py --local-certification
"""

import pandas as pd

from core.execution import NAUTILUS_AVAILABLE
from example_entrypoints import parse_example_args, run_example


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
    config = {
        "experiment": {
            "name": "fvg_spot",
            "description": "End-to-end spot pipeline using a config-driven Fair Value Gap indicator setup.",
        },
        "data": {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "start": "2024-01-01",
            "end": end,
            "futures_context": {"enabled": False},
        },
        "indicators": [
            {"kind": "rsi", "params": {"period": 14}},
            {"kind": "atr", "params": {"period": 14}},
            {"kind": "fvg", "params": {"name": "fvg_main", "min_gap_pct": 0.0005}},
        ],
        "features": {
            "lags": [1, 3, 6],
            "frac_diff_d": 0.4,
            "rolling_window": 20,
            "squeeze_quantile": 0.2,
        },
        "feature_selection": {"enabled": True, "max_features": 96, "min_mi_threshold": 0.0005},
        "regime": {"method": "hmm", "builder": build_fvg_regime_features},
        "labels": {
            "kind": "triple_barrier",
            "pt_sl": (2.0, 2.0),
            "max_holding": 24,
            "min_return": 0.001,
            "volatility_window": 24,
            "barrier_tie_break": "sl",
        },
        "model": {
            "type": "gbm",
            "cv_method": "cpcv",
            "n_blocks": 4,
            "test_blocks": 2,
            "validation_fraction": 0.2,
            "meta_n_splits": 2,
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
        "backtest": {
            "equity": 10_000,
            "fee_rate": 0.001,
            "slippage_rate": 0.0002,
            "slippage_model": "sqrt_impact",
            "engine": "vectorbt",
            "use_open_execution": True,
            "signal_delay_bars": 2,
        },
    }
    run_example(
        config,
        market="spot",
        local_certification=args.local_certification,
        quiet=args.quiet,
        nautilus_available=NAUTILUS_AVAILABLE,
        example_name="example_fvg.py",
    )


if __name__ == "__main__":
    main()