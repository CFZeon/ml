# Regime-Aware Crypto AutoML Research Stack

This repository is a research-first trading stack for Binance crypto that keeps model training, feature generation, validation, and execution assumptions explicit.

The current codebase is built around these constraints:

- one model per symbol
- shared timeframe analysis across symbols
- classification and meta-labeling workflows
- point-in-time-safe custom data joins
- walk-forward and CPCV-style validation instead of random CV
- execution-aware backtests with Binance filter enforcement
- futures-aware research with funding, mark-price valuation, and a liquidation-capable margin model

## What Is Implemented

The repo now includes:

- Binance spot and futures data loaders with cacheing, retry/backoff, and integrity reporting
- modular indicators in `core/indicators/`
- feature engineering with fractional differentiation and ADF-based stationarity screening
- triple-barrier, fixed-horizon, and trend-scanning labels
- uniqueness weighting and sequential bootstrap support
- explicit regime detection and context features
- AutoML with validation/holdout separation, CPCV/PBO diagnostics, DSR reporting, fragility checks, and stability gating
- a baseline-vs-prefix lookahead provocation harness for detecting future-informed features, labels, probabilities, signals, and execution inputs
- a causal liquidity resolver that shifts bar-volume inputs, validates L2 snapshot timestamps, and records liquidity provenance in backtest outputs
- pandas and vectorbt execution adapters behind a shared order-validation contract
- futures backtests with funding, mark-price valuation, leverage-bracket caps, isolated/cross margin modes, and liquidation events

## Repository Layout

- `core/data.py`: Binance data access, symbol filters, point-in-time custom joins, futures contract metadata, leverage bracket adapters
- `core/context.py`: futures context and cross-asset context loaders
- `core/features.py`: feature construction, family metadata, stationarity screening, supervised selection
- `core/labeling.py`: event labeling and uniqueness weighting helpers
- `core/models.py`: model training, diagnostics, validation helpers
- `core/automl.py`: Optuna-backed search, ranking, holdout logic, and overfitting diagnostics
- `core/backtest.py`: execution-aware backtests, slippage models, and futures margin/liquidation simulation
- `core/execution/liquidity.py`: causal bar-volume and order-book liquidity input resolution
- `core/pipeline.py`: stepwise research pipeline orchestration
- `core/lookahead.py`: baseline-plus-prefix replay audit for lookahead bias provocation
- `example.py`, `example_fvg.py`, `example_custom_data.py`, `example_futures.py`, `example_automl.py`: runnable end-to-end examples
- `tests/`: regression coverage for validation, joins, execution semantics, AutoML governance, and futures behavior

## Installation

```bash
python -m pip install -r requirements.txt
```

## Quick Start

Run the baseline research example:

```bash
python example.py
```

Run the futures example with mark-price valuation and the liquidation-aware margin model:

```bash
python example_futures.py
```

Run the test suite:

```bash
python -m pytest
```

## Futures Margin Model

P2-12 added a futures account overlay on top of the existing execution adapter.

It supports:

- `isolated` and `cross` margin modes
- mark-price-driven unrealized PnL tracking
- maintenance-margin and margin-ratio monitoring
- leverage-bracket-based exposure caps
- liquidation events with explicit liquidation fees
- reporting of realized leverage, warning-bar counts, funding paid/received, and liquidation summaries

The main configuration surface lives under `backtest.futures_account`.

Example:

```python
"backtest": {
	"engine": "pandas",
	"valuation_price": "mark",
	"apply_funding": True,
	"allow_short": True,
	"leverage": 3.0,
	"futures_account": {
		"enabled": True,
		"margin_mode": "isolated",
		"warning_margin_ratio": 0.8,
		"leverage_brackets_data": {
			"symbol": "BTCUSDT",
			"brackets": [
				{
					"bracket": 1,
					"initial_leverage": 20.0,
					"notional_floor": 0.0,
					"notional_cap": 50000.0,
					"maint_margin_ratio": 0.02,
					"cum": 0.0,
				}
			],
		},
	},
}
```

Supported bracket sources:

- `leverage_brackets_data`: inline normalized bracket payload in config
- `leverage_brackets_path`: JSON file containing either normalized rows or Binance-style bracket payloads
- `use_signed_leverage_brackets: true`: fetch from Binance's signed leverage-bracket endpoint and cache locally

If you use the signed endpoint, provide credentials via config or environment variables:

- `BINANCE_API_KEY` / `BINANCE_API_SECRET`
- or `BINANCE_FAPI_KEY` / `BINANCE_FAPI_SECRET`

Public contract metadata is loaded from futures `exchangeInfo` and exposed separately from spot/futures execution filters.

## Validation Philosophy

The repo does not treat headline accuracy as sufficient evidence.

Current safeguards include:

- walk-forward and CPCV validation paths
- purging-aware training flow for overlapping labels
- fold-local regime and stationarity fitting
- prefix-only replay audits that compare baseline outputs against truncated reruns at sampled decision timestamps
- holdout-aware AutoML promotion
- deflated Sharpe and PBO diagnostics
- fold-stability reporting with optional rejection gates
- execution-aware backtests with slippage, fees, and Binance constraint handling

## Notes

- The liquidation-aware futures simulator is validated on the included linear futures workflows and synthetic tests. If you add inverse-contract research, provide explicit contract metadata and bracket inputs rather than assuming spot-like quantity semantics.
- The vectorbt adapter remains available for standard execution-aware backtests. The futures account overlay uses the pandas path because liquidation and margin-state transitions are simulated explicitly.
- Custom data joins now fail closed by default unless availability semantics are explicit.

## Current Status

The remaining open backlog item is deployment governance: drift monitoring, local model registry workflow, champion-challenger promotion, and safer artifact persistence.