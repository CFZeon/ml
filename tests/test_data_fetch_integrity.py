import io
import tempfile
import unittest
import zipfile
from unittest import mock

import pandas as pd
import requests

from core import ResearchPipeline
from core.data import _COLUMNS, _download_archive, _symbol_filters_cache_path, fetch_binance_symbol_filters, fetch_binance_vision
from core.storage import read_json, write_parquet_frame


def _archive_bytes(rows):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        csv_payload = pd.DataFrame(rows, columns=_COLUMNS).to_csv(index=False, header=False).encode("utf-8")
        zf.writestr("BTCUSDT-1h-2024-01.csv", csv_payload)
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, status_code, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class DataFetchIntegrityTest(unittest.TestCase):
    def test_download_archive_retries_on_timeout_before_success(self):
        session = mock.Mock()
        session.get.side_effect = [
            requests.Timeout("timed out"),
            _FakeResponse(
                200,
                content=_archive_bytes(
                    [[1704067200000, "100", "101", "99", "100.5", "10", 1704070799999, "1000", "12", "4", "400", "0"]]
                ),
            ),
        ]

        with mock.patch("core.data.time.sleep") as sleep_mock:
            frame, report = _download_archive(
                "https://example.com/archive.zip",
                session,
                retry_policy={"max_retries": 1, "backoff_factor": 0.0, "backoff_jitter": 0.0},
            )

        self.assertEqual(session.get.call_count, 2)
        self.assertEqual(int(report["retry_count"]), 1)
        self.assertEqual(report["status"], "downloaded")
        self.assertEqual(len(frame), 1)
        sleep_mock.assert_not_called()

    def test_fetch_binance_vision_returns_structured_integrity_report_for_gaps(self):
        index = pd.to_datetime(["2024-01-01 00:00:00+00:00", "2024-01-01 02:00:00+00:00"])
        frame = pd.DataFrame(
            {
                "open": [100.0, 102.0],
                "high": [101.0, 103.0],
                "low": [99.0, 101.0],
                "close": [100.5, 102.5],
                "volume": [10.0, 11.0],
                "quote_volume": [1005.0, 1127.5],
                "trades": [12, 13],
                "taker_buy_base_vol": [4.0, 4.5],
                "taker_buy_quote_vol": [400.0, 460.0],
            },
            index=index,
        )

        load_report = {
            "period_kind": "monthly",
            "period_key": "2024-01",
            "window_start": pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
            "window_end": pd.Timestamp("2024-01-01 03:00:00", tz="UTC"),
            "expected_rows": 3,
            "observed_rows": 2,
            "missing_rows": 1,
            "missing_segments": [
                {
                    "start": pd.Timestamp("2024-01-01 01:00:00", tz="UTC"),
                    "end": pd.Timestamp("2024-01-01 01:00:00", tz="UTC"),
                    "count": 1,
                }
            ],
            "status": "incomplete",
            "used_cache": False,
            "refreshed": True,
            "retry_count": 0,
            "downloads": [],
            "dropped_window": False,
        }
        session_factory = mock.MagicMock()
        session_factory.return_value.__enter__.return_value = mock.Mock()

        with mock.patch("core.data.requests.Session", session_factory), mock.patch("core.data._load_period", return_value=(frame, load_report)):
            result, report = fetch_binance_vision(
                symbol="BTCUSDT",
                interval="1h",
                start="2024-01-01",
                end="2024-01-01 03:00:00",
                cache_dir=None,
                gap_policy="flag",
                return_report=True,
            )

        self.assertEqual(len(result), 2)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(int(report["missing_rows"]), 1)
        self.assertEqual(int(report["periods"][0]["missing_rows"]), 1)
        self.assertEqual(result.attrs["integrity_report"]["status"], "incomplete")

    def test_gap_policy_fail_raises_when_missing_windows_remain(self):
        frame = pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [10.0],
                "quote_volume": [1005.0],
                "trades": [12],
                "taker_buy_base_vol": [4.0],
                "taker_buy_quote_vol": [400.0],
            },
            index=pd.to_datetime(["2024-01-01 00:00:00+00:00"]),
        )
        load_report = {
            "period_kind": "monthly",
            "period_key": "2024-01",
            "window_start": pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
            "window_end": pd.Timestamp("2024-01-01 03:00:00", tz="UTC"),
            "expected_rows": 3,
            "observed_rows": 1,
            "missing_rows": 2,
            "missing_segments": [],
            "status": "incomplete",
            "used_cache": False,
            "refreshed": True,
            "retry_count": 0,
            "downloads": [],
            "dropped_window": False,
        }
        session_factory = mock.MagicMock()
        session_factory.return_value.__enter__.return_value = mock.Mock()

        with mock.patch("core.data.requests.Session", session_factory), mock.patch("core.data._load_period", return_value=(frame, load_report)):
            with self.assertRaisesRegex(RuntimeError, "Incomplete data window"):
                fetch_binance_vision(
                    symbol="BTCUSDT",
                    interval="1h",
                    start="2024-01-01",
                    end="2024-01-01 03:00:00",
                    cache_dir=None,
                    gap_policy="fail",
                )

    def test_fetch_data_step_surfaces_integrity_report(self):
        index = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0, 102.0],
                "close": [100.5, 101.5, 102.5, 103.5],
                "volume": [10.0, 11.0, 12.0, 13.0],
                "quote_volume": [1005.0, 1116.5, 1230.0, 1345.5],
                "trades": [100, 101, 102, 103],
                "taker_buy_base_vol": [4.0, 4.1, 4.2, 4.3],
                "taker_buy_quote_vol": [400.0, 410.0, 420.0, 430.0],
            },
            index=index,
        )
        report = {"status": "complete", "missing_rows": 0, "periods": []}
        pipeline = ResearchPipeline({"data": {"symbol": "BTCUSDT", "interval": "1h", "start": "2024-01-01", "end": "2024-01-01 04:00:00", "gap_policy": "flag"}})

        with mock.patch("core.pipeline.fetch_binance_bars", return_value=(frame, report)), mock.patch("core.pipeline.fetch_binance_symbol_filters", return_value={}):
            result = pipeline.fetch_data()

        self.assertEqual(len(result), 4)
        self.assertEqual(pipeline.state["data_integrity_report"], report)
        self.assertIn("data_lineage", pipeline.state)
        self.assertTrue(pipeline.state["data_lineage"]["source_groups"]["market_data"])

    def test_symbol_filters_loader_rewrites_legacy_cache_as_json(self):
        legacy_frame = pd.DataFrame(
            [
                {
                    "symbol": "BTCUSDT",
                    "tick_size": 0.1,
                    "step_size": 0.001,
                    "min_notional": 10.0,
                }
            ]
        )
        payload = {
            "symbol": "BTCUSDT",
            "filters": [
                {"filterType": "PRICE_FILTER", "minPrice": "10", "maxPrice": "1000000", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "50", "stepSize": "0.001"},
                {"filterType": "MIN_NOTIONAL", "minNotional": "10", "applyToMarket": True, "avgPriceMins": 5},
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = _symbol_filters_cache_path(temp_dir, "spot", "BTCUSDT")
            write_parquet_frame(cache_path, legacy_frame)

            with mock.patch("core.data._fetch_exchange_info_symbol_payload", return_value=payload):
                filters = fetch_binance_symbol_filters("BTCUSDT", market="spot", cache_dir=temp_dir)

            self.assertIsInstance(filters, dict)
            self.assertAlmostEqual(float(filters["tick_size"]), 0.01, places=6)
            cached = read_json(cache_path)
            self.assertIsInstance(cached, dict)
            self.assertEqual(cached["symbol"], "BTCUSDT")


if __name__ == "__main__":
    unittest.main()