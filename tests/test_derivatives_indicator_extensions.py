from unittest import mock

import numpy as np
import pandas as pd

from core.indicators.derivatives_combined import DerivativesCombined
from core.indicators.funding_rate import FundingRateContext
from core.indicators.open_interest import OpenInterestContext


def _base_frame(index):
    close = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)
    volume = pd.Series(np.linspace(1_000.0, 1_500.0, len(index)), index=index)
    return pd.DataFrame({"close": close, "volume": volume}, index=index)


def test_funding_indicator_does_not_leak_future_events():
    index = pd.date_range("2026-01-01 23:45:00", periods=34, freq="15min", tz="UTC")
    data = _base_frame(index)
    funding_frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-02 00:00:00+00:00", "2026-01-02 08:00:00+00:00"]
            ),
            "funding_rate": [0.0010, -0.0005],
            "funding_mark_price": [100.0, 101.0],
        }
    )

    with mock.patch(
        "core.indicators.funding_rate.derivatives_support.fetch_funding_history",
        return_value=funding_frame,
    ):
        result = FundingRateContext(
            symbol="BTCUSDT",
            rolling_window=3,
            warmup="1D",
            max_age="8h",
            min_coverage=0.0,
        ).run(data).to_frame()

    assert pd.isna(result.loc[pd.Timestamp("2026-01-01 23:45:00+00:00"), "funding_ctx_delta"])
    assert result.loc[pd.Timestamp("2026-01-02 07:45:00+00:00"), "funding_ctx_delta"] == 0.0
    assert result.loc[pd.Timestamp("2026-01-02 08:00:00+00:00"), "funding_ctx_delta"] == -0.0015
    assert result.loc[pd.Timestamp("2026-01-02 08:00:00+00:00"), "funding_ctx_observed_event"] == 1.0
    assert result.loc[pd.Timestamp("2026-01-02 07:45:00+00:00"), "funding_ctx_observed_event"] == 0.0


def test_open_interest_indicator_aligns_backward_asof_to_base_bars():
    index = pd.date_range("2026-02-01 10:00:00", periods=3, freq="15min", tz="UTC")
    data = _base_frame(index)
    oi_frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-02-01 09:55:00+00:00",
                    "2026-02-01 10:05:00+00:00",
                    "2026-02-01 10:20:00+00:00",
                ]
            ),
            "sumOpenInterest": [1000.0, 1010.0, 1030.0],
            "sumOpenInterestValue": [50_000.0, 50_500.0, 51_500.0],
        }
    )

    with mock.patch(
        "core.indicators.open_interest.derivatives_support.fetch_open_interest_history",
        return_value=oi_frame,
    ):
        result = OpenInterestContext(
            symbol="BTCUSDT",
            period="5m",
            change_horizon="10m",
            trend_short_span=2,
            trend_long_span=3,
            warmup="1D",
            max_age="30m",
            min_coverage=0.0,
        ).run(data).to_frame()

    expected_log_change = np.log(51_500.0 / 50_000.0)
    expected_pressure = 1_000.0 / (data.loc[index[2], "close"] * data.loc[index[2], "volume"])
    assert pd.isna(result.loc[index[0], "oi_ctx_log_change_10m"])
    assert pd.isna(result.loc[index[1], "oi_ctx_log_change_10m"])
    assert np.isclose(result.loc[index[2], "oi_ctx_log_change_10m"], expected_log_change)
    assert np.isfinite(result.loc[index[1], "oi_ctx_trend_spread"])
    assert np.isclose(result.loc[index[2], "oi_ctx_pressure_vs_notional_volume"], expected_pressure)
    assert result.loc[index[0], "oi_ctx_age_minutes"] == 5.0
    assert result.loc[index[1], "oi_ctx_age_minutes"] == 10.0
    assert result.loc[index[2], "oi_ctx_age_minutes"] == 10.0


def test_combined_indicator_emits_finite_interaction_features():
    index = pd.date_range("2026-03-01 00:00:00", periods=8, freq="15min", tz="UTC")
    frame = _base_frame(index)
    frame["funding_ctx_delta"] = [0.0, 0.0, 0.0002, 0.0002, -0.0001, -0.0001, 0.0003, 0.0003]
    frame["funding_ctx_z_3"] = [0.0, 0.0, 1.5, 1.0, -1.2, -1.0, 2.1, 1.7]
    frame["oi_ctx_log_change_4h"] = [0.0, 0.0, 0.02, 0.03, -0.01, -0.005, 0.04, 0.01]
    frame["oi_ctx_trend_spread"] = [0.0, 0.001, 0.01, 0.015, -0.004, -0.002, 0.02, 0.01]
    frame["oi_ctx_pressure_vs_notional_volume"] = [0.0, 0.005, 0.01, -0.004, -0.002, 0.008, -0.003, 0.002]

    result = DerivativesCombined(
        funding_prefix="funding_ctx",
        oi_prefix="oi_ctx",
        funding_window=3,
        oi_change_horizon="4h",
        volatility_window=3,
    ).run(frame).to_frame()

    assert not result.empty
    assert result.columns.tolist() == [
        "deriv_combo_funding_delta_x_return",
        "deriv_combo_oi_log_change_x_vol",
        "deriv_combo_price_up_oi_trend_up",
        "deriv_combo_price_down_oi_trend_up",
        "deriv_combo_funding_delta_return_sign",
        "deriv_combo_crowding_proxy",
        "deriv_combo_funding_x_oi_pressure",
    ]
    assert np.isfinite(result.to_numpy()).all()