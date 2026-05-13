import unittest

import pandas as pd

from core import ResearchPipeline, build_monitoring_report
from core.pipeline import _build_pipeline_operational_monitoring
from example_local_certification_automl import build_local_certification_example_config


class LocalCertificationMonitoringDefaultsTest(unittest.TestCase):
    def test_research_profile_uses_finite_advisory_thresholds(self):
        report = build_monitoring_report(policy={"policy_profile": "research"})

        policy = report["policy"]
        self.assertEqual(policy["policy_profile"], "research")
        self.assertEqual(policy["max_data_lag"], "24h")
        self.assertEqual(policy["max_l2_snapshot_age"], "1h")
        self.assertEqual(float(policy["max_slippage_drift"]), 0.01)
        self.assertEqual(float(policy["max_fallback_assumption_rate"]), 0.25)
        self.assertEqual(float(policy["max_fill_ratio_deterioration"]), 0.5)
        self.assertEqual(float(policy["max_inference_p95_ms"]), 2000.0)
        self.assertEqual(int(policy["max_queue_backlog"]), 100)

    def test_local_certification_profile_uses_finite_thresholds(self):
        report = build_monitoring_report(policy={"policy_profile": "local_certification"})

        policy = report["policy"]
        self.assertEqual(policy["policy_profile"], "local_certification")
        self.assertEqual(policy["max_data_lag"], "4h")
        self.assertEqual(policy["max_l2_snapshot_age"], "15min")
        self.assertEqual(float(policy["max_fill_ratio_deterioration"]), 0.15)
        self.assertEqual(float(policy["max_slippage_drift"]), 0.002)
        self.assertEqual(float(policy["max_inference_p95_ms"]), 500.0)
        self.assertEqual(int(policy["max_queue_backlog"]), 0)
        self.assertEqual(float(policy["min_signal_decay_net_edge_at_delay"]), 0.0)
        self.assertEqual(float(policy["max_fallback_assumption_rate"]), 0.0)

    def test_local_certification_example_uses_local_monitoring_profile(self):
        config = build_local_certification_example_config(automl_storage="local-cert-monitoring.db")

        self.assertEqual(config["monitoring"]["policy_profile"], "local_certification")

    def test_pipeline_local_certification_binds_profile_even_if_research_requested(self):
        index = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "backtest": {"evaluation_mode": "local_certification"},
                "monitoring": {"policy_profile": "research"},
            }
        )
        pipeline.state["raw_data"] = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]}, index=index)

        report = _build_pipeline_operational_monitoring(
            pipeline,
            expected_feature_columns=["feature_a"],
            actual_feature_columns=["feature_a"],
            backtest_reports=[
                {
                    "execution_evidence": {
                        "class": "event_driven_certification",
                        "execution_mode": "nautilus_l1",
                        "promotion_execution_ready": True,
                    },
                    "funding_coverage_report": {
                        "coverage_status": "strict",
                        "promotion_pass": True,
                    },
                }
            ],
            signal_decay_report={
                "promotion_pass": True,
                "gate_mode": "blocking",
                "trade_count": 12,
                "half_life_bars": 3,
                "net_edge_at_effective_delay": 0.01,
            },
            inference_latencies_ms=[10.0, 12.0, 14.0],
            queue_backlog=[0, 0, 0],
            scope="training",
        )

        self.assertEqual(report["policy"]["policy_profile"], "local_certification")

    def test_pipeline_monitoring_carries_deployment_profile_and_resource_telemetry(self):
        index = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        pipeline = ResearchPipeline(
            {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "monitoring": {"deployment_profile": "consumer_laptop"},
                "features": {"context_timeframes": ["4h"]},
            }
        )
        pipeline.state["raw_data"] = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]}, index=index)
        pipeline.state["resource_telemetry"] = {
            "peak_rss_mb": 4096.0,
            "model_load_latency_ms": 900.0,
            "storage_footprint_mb": 1024.0,
        }
        pipeline.state["replay_benchmark"] = {
            "run_count": 2,
            "throughput_bars_per_sec": 300.0,
            "latency_p95_ms": 20.0,
            "latency_p99_ms": 24.0,
            "memory_spike_mb": 4300.0,
            "restart_recovery_time_ms": 30_000.0,
            "deterministic_pass": True,
        }

        report = _build_pipeline_operational_monitoring(
            pipeline,
            expected_feature_columns=["feature_a"],
            actual_feature_columns=["feature_a"],
            inference_latencies_ms=[10.0, 12.0, 14.0],
            scope="training",
        )

        envelope = report["operating_envelope"]
        self.assertEqual(envelope["deployment_profile"], "consumer_laptop")
        self.assertEqual(envelope["resource_telemetry"]["symbol_count"], 1)
        self.assertEqual(envelope["resource_telemetry"]["timeframe_count"], 2)
        self.assertTrue(envelope["replay_benchmark"]["observed"])


if __name__ == "__main__":
    unittest.main()