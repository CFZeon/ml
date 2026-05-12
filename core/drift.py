"""Batch and streaming drift detection helpers."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised through fallback-aware tests
    from river.drift import ADWIN as RiverADWIN
except ImportError:  # pragma: no cover
    RiverADWIN = None


def _safe_series(value):
    if value is None:
        return pd.Series(dtype=float)
    if isinstance(value, pd.Series):
        return pd.Series(value, copy=False)
    return pd.Series(value)


def _safe_frame(value):
    if value is None:
        return pd.DataFrame()
    return pd.DataFrame(value).copy()


def _coerce_regime_label_series(value):
    if value is None:
        return pd.Series(dtype=float)
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return pd.Series(dtype=float)
        column = "regime" if "regime" in value.columns else value.columns[0]
        return pd.Series(value[column], copy=False)
    return _safe_series(value)


def _coerce_interval_to_timedelta(value):
    if value in (None, "", False):
        return None
    if isinstance(value, pd.Timedelta):
        return value

    text = str(value).strip()
    if not text:
        return None
    try:
        return pd.Timedelta(text)
    except Exception:
        pass

    if len(text) < 2 or not text[:-1].isdigit():
        return None
    amount = int(text[:-1])
    unit = text[-1]
    if unit == "m":
        return pd.Timedelta(minutes=amount)
    if unit == "h":
        return pd.Timedelta(hours=amount)
    if unit == "d":
        return pd.Timedelta(days=amount)
    if unit == "w":
        return pd.Timedelta(days=7 * amount)
    if unit == "M":
        return pd.Timedelta(days=30 * amount)
    return None


def _resolve_drift_policy(config=None):
    configured = dict(config or {})
    resolved = {
        "psi_threshold": 0.2,
        "psi_feature_share_threshold": 0.3,
        "ks_significance": 0.05,
        "prediction_kl_threshold": 0.1,
        "score_distribution_tv_threshold": 0.25,
        "confidence_psi_threshold": 0.2,
        "confidence_ks_threshold": 0.25,
        "margin_psi_threshold": 0.2,
        "margin_ks_threshold": 0.25,
        "action_rate_delta_threshold": 0.15,
        "abstain_rate_delta_threshold": 0.10,
        "regime_psi_threshold": 0.2,
        "regime_total_variation_threshold": 0.25,
        "regime_transition_threshold": 0.2,
        "cooldown_days": 21.0,
        "cooldown_bars": None,
        "max_days_between_retrain": 28.0,
        "max_bars_between_retrain": None,
        "min_sample_days": (200.0 / 24.0),
        "min_samples": None,
        "expected_trades_per_day": None,
        "min_expected_trades": None,
        "min_drift_signals": 2,
        "adwin_delta": 0.002,
        "interval": None,
        "post_retrain_warmup_required": True,
        "post_retrain_warmup_mode": "paper",
    }
    resolved.update(configured)

    interval = resolved.get("interval") or resolved.get("bar_interval")
    interval_delta = _coerce_interval_to_timedelta(interval)
    bars_per_day = None
    if interval_delta is not None and interval_delta > pd.Timedelta(0):
        bars_per_day = (24.0 * 60.0 * 60.0) / float(interval_delta.total_seconds())

    cooldown_bars = resolved.get("cooldown_bars")
    if cooldown_bars is None:
        if bars_per_day is not None:
            cooldown_bars = int(max(1, math.ceil(float(resolved.get("cooldown_days", 21.0)) * bars_per_day)))
        else:
            cooldown_bars = 500

    min_samples = resolved.get("min_samples")
    if min_samples is None:
        if bars_per_day is not None:
            min_samples = int(
                max(1, math.ceil(float(resolved.get("min_sample_days", (200.0 / 24.0))) * bars_per_day))
            )
        else:
            min_samples = 200

    trade_rate_min_samples = None
    expected_trades_per_day = resolved.get("expected_trades_per_day")
    min_expected_trades = resolved.get("min_expected_trades")
    if (
        bars_per_day is not None
        and expected_trades_per_day not in (None, 0, 0.0)
        and min_expected_trades not in (None, 0, 0.0)
    ):
        trade_rate_min_samples = int(
            max(
                1,
                math.ceil((float(min_expected_trades) / float(expected_trades_per_day)) * bars_per_day),
            )
        )
        min_samples = max(int(min_samples), trade_rate_min_samples)

    max_bars_between_retrain = resolved.get("max_bars_between_retrain")
    if max_bars_between_retrain is None:
        if bars_per_day is not None:
            max_bars_between_retrain = int(
                max(1, math.ceil(float(resolved.get("max_days_between_retrain", 28.0)) * bars_per_day))
            )
        else:
            max_bars_between_retrain = 24 * 28

    resolved["interval"] = interval
    resolved["bars_per_day"] = None if bars_per_day is None else float(bars_per_day)
    resolved["cooldown_bars"] = int(max(0, int(cooldown_bars)))
    resolved["min_samples"] = int(max(1, int(min_samples)))
    resolved["max_bars_between_retrain"] = int(max(1, int(max_bars_between_retrain)))
    resolved["trade_rate_min_samples"] = None if trade_rate_min_samples is None else int(trade_rate_min_samples)
    return resolved


def _population_stability_index(reference, current, bins=10, epsilon=1e-6):
    reference_series = _safe_series(reference).dropna().astype(float)
    current_series = _safe_series(current).dropna().astype(float)
    if reference_series.empty or current_series.empty:
        return None

    quantiles = np.linspace(0.0, 1.0, int(max(2, bins)) + 1)
    edges = np.unique(np.nanquantile(reference_series, quantiles))
    if len(edges) < 2:
        edges = np.array([reference_series.min(), reference_series.max() + epsilon])
    if len(edges) < 2 or not np.isfinite(edges).all():
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference_series, bins=edges)
    current_counts, _ = np.histogram(current_series, bins=edges)
    reference_share = np.maximum(reference_counts / max(1, reference_counts.sum()), epsilon)
    current_share = np.maximum(current_counts / max(1, current_counts.sum()), epsilon)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def _ks_statistic(reference, current):
    reference_series = np.sort(_safe_series(reference).dropna().astype(float).to_numpy())
    current_series = np.sort(_safe_series(current).dropna().astype(float).to_numpy())
    if len(reference_series) == 0 or len(current_series) == 0:
        return None, None

    combined = np.sort(np.concatenate([reference_series, current_series]))
    reference_cdf = np.searchsorted(reference_series, combined, side="right") / len(reference_series)
    current_cdf = np.searchsorted(current_series, combined, side="right") / len(current_series)
    statistic = float(np.max(np.abs(reference_cdf - current_cdf)))
    n_eff = (len(reference_series) * len(current_series)) / (len(reference_series) + len(current_series))
    p_value = float(min(1.0, 2.0 * math.exp(-2.0 * n_eff * statistic * statistic)))
    return statistic, p_value


def _kl_divergence(reference, current, epsilon=1e-6):
    reference_frame = _safe_frame(reference)
    current_frame = _safe_frame(current)
    if reference_frame.empty or current_frame.empty:
        return None
    common_columns = [column for column in reference_frame.columns if column in current_frame.columns]
    if not common_columns:
        return None
    ref = reference_frame[common_columns].mean(axis=0).to_numpy(dtype=float)
    cur = current_frame[common_columns].mean(axis=0).to_numpy(dtype=float)
    ref = np.maximum(ref, epsilon)
    cur = np.maximum(cur, epsilon)
    ref = ref / ref.sum()
    cur = cur / cur.sum()
    return float(np.sum(cur * np.log(cur / ref)))


def _normalize_probability_columns(frame):
    resolved = _safe_frame(frame)
    if resolved.empty:
        return resolved
    numeric = resolved.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    return numeric


def _prediction_label_series(frame):
    probabilities = _normalize_probability_columns(frame)
    if probabilities.empty:
        return pd.Series(dtype=object)
    labels = probabilities.idxmax(axis=1)
    return pd.Series(labels, index=probabilities.index, copy=False)


def _prediction_confidence_series(frame):
    probabilities = _normalize_probability_columns(frame)
    if probabilities.empty:
        return pd.Series(dtype=float)
    return probabilities.max(axis=1).astype(float)


def _prediction_margin_series(frame):
    probabilities = _normalize_probability_columns(frame)
    if probabilities.empty:
        return pd.Series(dtype=float)
    sorted_probs = np.sort(probabilities.to_numpy(dtype=float), axis=1)
    if sorted_probs.shape[1] == 1:
        margins = sorted_probs[:, 0]
    else:
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    return pd.Series(margins, index=probabilities.index, dtype=float)


def _label_distribution(series):
    labels = _safe_series(series).dropna()
    if labels.empty:
        return {}
    distribution = labels.value_counts(normalize=True, dropna=True)
    return {str(key): float(value) for key, value in distribution.items()}


def _infer_action_state(series):
    labels = _safe_series(series)
    if labels.empty:
        return pd.DataFrame(columns=["active", "abstain"])
    normalized = labels.astype(str).str.strip().str.lower()
    abstain_mask = normalized.isin({"0", "0.0", "abstain", "no_trade", "hold", "flat", "neutral"})
    active_mask = (~abstain_mask) & normalized.ne("nan")
    return pd.DataFrame({"active": active_mask.astype(float), "abstain": abstain_mask.astype(float)}, index=labels.index)


def _distribution_total_variation(reference_distribution, current_distribution):
    if not reference_distribution or not current_distribution:
        return None
    keys = set(reference_distribution) | set(current_distribution)
    if not keys:
        return None
    return float(
        0.5 * sum(abs(float(reference_distribution.get(key, 0.0)) - float(current_distribution.get(key, 0.0))) for key in keys)
    )


def _categorical_distribution_shift(reference, current, epsilon=1e-6):
    reference_series = _coerce_regime_label_series(reference).dropna()
    current_series = _coerce_regime_label_series(current).dropna()
    if reference_series.empty or current_series.empty:
        return None

    categories = sorted(set(reference_series.tolist()) | set(current_series.tolist()))
    if not categories:
        return None

    reference_distribution = {}
    current_distribution = {}
    for category in categories:
        reference_distribution[str(category)] = float((reference_series == category).mean())
        current_distribution[str(category)] = float((current_series == category).mean())

    psi = float(
        sum(
            (max(current_distribution[key], epsilon) - max(reference_distribution[key], epsilon))
            * math.log(max(current_distribution[key], epsilon) / max(reference_distribution[key], epsilon))
            for key in reference_distribution
        )
    )
    total_variation = _distribution_total_variation(reference_distribution, current_distribution)
    return {
        "psi": psi,
        "total_variation": total_variation,
        "reference_distribution": reference_distribution,
        "current_distribution": current_distribution,
    }


def _categorical_transition_distribution(series):
    labels = _coerce_regime_label_series(series).dropna().tolist()
    if len(labels) < 2:
        return {}

    transitions = {}
    total = 0
    for left, right in zip(labels[:-1], labels[1:]):
        key = f"{left}->{right}"
        transitions[key] = transitions.get(key, 0) + 1
        total += 1
    if total <= 0:
        return {}
    return {key: float(count / total) for key, count in transitions.items()}


class ADWINDetector:
    """Streaming mean-shift detector with a River ADWIN hook and fallback path."""

    def __init__(self, delta=0.002, fallback_window=40, fallback_zscore=2.5):
        self.delta = float(delta)
        self.fallback_window = int(max(10, fallback_window))
        self.fallback_zscore = float(fallback_zscore)
        self._detector = RiverADWIN(delta=self.delta) if RiverADWIN is not None else None
        self._history = deque(maxlen=self.fallback_window * 4)

    @property
    def using_river(self):
        return self._detector is not None

    def update(self, value):
        if value is None or not np.isfinite(value):
            return {"drift_detected": False, "method": "ignored_invalid"}

        if self._detector is not None:
            self._detector.update(float(value))
            drift_detected = bool(getattr(self._detector, "drift_detected", False))
            return {
                "drift_detected": drift_detected,
                "method": "river_adwin",
                "value": float(value),
            }

        self._history.append(float(value))
        if len(self._history) < self.fallback_window * 2:
            return {"drift_detected": False, "method": "fallback_warmup", "value": float(value)}

        history = np.asarray(self._history, dtype=float)
        left = history[-(self.fallback_window * 2):-self.fallback_window]
        right = history[-self.fallback_window:]
        left_mean = float(np.mean(left))
        right_mean = float(np.mean(right))
        left_std = float(np.std(left, ddof=1)) if len(left) > 1 else 0.0
        threshold = self.fallback_zscore * max(left_std, 1e-6)
        drift_detected = abs(right_mean - left_mean) > threshold
        return {
            "drift_detected": bool(drift_detected),
            "method": "fallback_mean_shift",
            "value": float(value),
            "left_mean": left_mean,
            "right_mean": right_mean,
            "threshold": threshold,
        }


class DriftMonitor:
    def __init__(self, reference_features, reference_predictions=None, reference_regimes=None, config=None):
        self.reference_features = _safe_frame(reference_features)
        self.reference_predictions = _safe_frame(reference_predictions)
        self.reference_regimes = _coerce_regime_label_series(reference_regimes)
        self.config = _resolve_drift_policy(config)
        self.performance_detector = ADWINDetector(delta=self.config["adwin_delta"])

    def check(self, current_features, current_predictions=None, current_performance=None,
              current_regimes=None, bars_since_last_retrain=None):
        current_features = _safe_frame(current_features)
        current_predictions = _safe_frame(current_predictions)
        inferred_reference_regimes = self.reference_regimes
        if inferred_reference_regimes.empty and "regime" in self.reference_features.columns:
            inferred_reference_regimes = _coerce_regime_label_series(self.reference_features["regime"])
        inferred_current_regimes = _coerce_regime_label_series(current_regimes)
        if inferred_current_regimes.empty and "regime" in current_features.columns:
            inferred_current_regimes = _coerce_regime_label_series(current_features["regime"])

        sample_count = int(
            len(current_features)
            or len(current_predictions)
            or len(inferred_current_regimes)
            or len(_safe_series(current_performance))
        )

        feature_reports = {}
        feature_alerts = []
        common_columns = [
            column for column in self.reference_features.columns
            if column in current_features.columns and pd.api.types.is_numeric_dtype(current_features[column])
        ]
        for column in common_columns:
            psi = _population_stability_index(self.reference_features[column], current_features[column])
            ks_stat, ks_p_value = _ks_statistic(self.reference_features[column], current_features[column])
            alert = bool(
                (psi is not None and psi >= float(self.config["psi_threshold"]))
                or (ks_p_value is not None and ks_p_value < float(self.config["ks_significance"]))
            )
            feature_reports[column] = {
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_p_value": ks_p_value,
                "alert": alert,
            }
            if alert:
                feature_alerts.append(column)

        feature_drift_share = float(len(feature_alerts) / len(common_columns)) if common_columns else 0.0
        feature_drift = bool(feature_drift_share >= float(self.config["psi_feature_share_threshold"]))

        prediction_kl = _kl_divergence(self.reference_predictions, current_predictions)
        reference_labels = _prediction_label_series(self.reference_predictions)
        current_labels = _prediction_label_series(current_predictions)
        class_distribution_tv = _distribution_total_variation(
            _label_distribution(reference_labels),
            _label_distribution(current_labels),
        )
        reference_confidence = _prediction_confidence_series(self.reference_predictions)
        current_confidence = _prediction_confidence_series(current_predictions)
        confidence_psi = _population_stability_index(reference_confidence, current_confidence)
        confidence_ks, confidence_ks_p = _ks_statistic(reference_confidence, current_confidence)
        reference_margin = _prediction_margin_series(self.reference_predictions)
        current_margin = _prediction_margin_series(current_predictions)
        margin_psi = _population_stability_index(reference_margin, current_margin)
        margin_ks, margin_ks_p = _ks_statistic(reference_margin, current_margin)

        score_drift = bool(
            (prediction_kl is not None and prediction_kl >= float(self.config["prediction_kl_threshold"]))
            or (class_distribution_tv is not None and class_distribution_tv >= float(self.config["score_distribution_tv_threshold"]))
            or (confidence_psi is not None and confidence_psi >= float(self.config["confidence_psi_threshold"]))
            or (confidence_ks is not None and confidence_ks >= float(self.config["confidence_ks_threshold"]))
            or (margin_psi is not None and margin_psi >= float(self.config["margin_psi_threshold"]))
            or (margin_ks is not None and margin_ks >= float(self.config["margin_ks_threshold"]))
        )
        prediction_drift = score_drift

        reference_action_state = _infer_action_state(reference_labels)
        current_action_state = _infer_action_state(current_labels)
        reference_action_rate = None if reference_action_state.empty else float(reference_action_state["active"].mean())
        current_action_rate = None if current_action_state.empty else float(current_action_state["active"].mean())
        reference_abstain_rate = None if reference_action_state.empty else float(reference_action_state["abstain"].mean())
        current_abstain_rate = None if current_action_state.empty else float(current_action_state["abstain"].mean())
        action_rate_delta = None
        abstain_rate_delta = None
        if reference_action_rate is not None and current_action_rate is not None:
            action_rate_delta = float(current_action_rate - reference_action_rate)
        if reference_abstain_rate is not None and current_abstain_rate is not None:
            abstain_rate_delta = float(current_abstain_rate - reference_abstain_rate)
        action_drift = bool(
            (action_rate_delta is not None and abs(action_rate_delta) >= float(self.config["action_rate_delta_threshold"]))
            or (abstain_rate_delta is not None and abs(abstain_rate_delta) >= float(self.config["abstain_rate_delta_threshold"]))
        )

        regime_distribution = _categorical_distribution_shift(inferred_reference_regimes, inferred_current_regimes)
        reference_transition = _categorical_transition_distribution(inferred_reference_regimes)
        current_transition = _categorical_transition_distribution(inferred_current_regimes)
        regime_transition_tv = _distribution_total_variation(reference_transition, current_transition)
        regime_drift = bool(
            regime_distribution is not None
            and (
                float(regime_distribution.get("psi") or 0.0) >= float(self.config["regime_psi_threshold"])
                or float(regime_distribution.get("total_variation") or 0.0) >= float(self.config["regime_total_variation_threshold"])
                or float(regime_transition_tv or 0.0) >= float(self.config["regime_transition_threshold"])
            )
        )

        performance_updates = []
        performance_drift = False
        for value in _safe_series(current_performance).dropna().tolist():
            update = self.performance_detector.update(value)
            performance_updates.append(update)
            performance_drift = performance_drift or bool(update.get("drift_detected", False))

        max_bars_between_retrain = self.config.get("max_bars_between_retrain")
        if max_bars_between_retrain is not None:
            max_bars_between_retrain = int(max_bars_between_retrain)
            if max_bars_between_retrain <= 0:
                max_bars_between_retrain = None
        model_ttl_expired = bool(
            max_bars_between_retrain is not None
            and bars_since_last_retrain is not None
            and int(bars_since_last_retrain) >= max_bars_between_retrain
        )
        cooldown_active = (
            bars_since_last_retrain is not None
            and int(bars_since_last_retrain) < int(self.config["cooldown_bars"])
            and not model_ttl_expired
        )
        enough_samples = sample_count >= int(self.config["min_samples"])
        evidence_count = int(feature_drift) + int(score_drift) + int(action_drift) + int(regime_drift) + int(performance_drift)
        sufficient_evidence = evidence_count >= int(self.config["min_drift_signals"])
        should_retrain = bool(enough_samples and not cooldown_active and (sufficient_evidence or model_ttl_expired))

        reasons = []
        if model_ttl_expired:
            reasons.append("model_ttl_expired")
        if score_drift:
            reasons.append("score_drift_detected")
        if action_drift:
            reasons.append("action_drift_detected")
        if not enough_samples:
            reasons.append("minimum_samples_not_met")
        if cooldown_active:
            reasons.append("cooldown_active")
        if enough_samples and not sufficient_evidence and not model_ttl_expired:
            reasons.append("insufficient_drift_evidence")
        if should_retrain:
            reasons.append("retrain_recommended")

        return {
            "sample_count": sample_count,
            "feature_reports": feature_reports,
            "feature_alerts": feature_alerts,
            "feature_drift_share": feature_drift_share,
            "feature_drift": feature_drift,
            "prediction_kl_divergence": prediction_kl,
            "score_report": {
                "mean_probability_kl_divergence": prediction_kl,
                "class_distribution_total_variation": class_distribution_tv,
                "confidence_psi": confidence_psi,
                "confidence_ks_statistic": confidence_ks,
                "confidence_ks_p_value": confidence_ks_p,
                "margin_psi": margin_psi,
                "margin_ks_statistic": margin_ks,
                "margin_ks_p_value": margin_ks_p,
            },
            "score_drift": score_drift,
            "prediction_drift": prediction_drift,
            "action_report": {
                "reference_class_distribution": _label_distribution(reference_labels),
                "current_class_distribution": _label_distribution(current_labels),
                "reference_action_rate": reference_action_rate,
                "current_action_rate": current_action_rate,
                "action_rate_delta": action_rate_delta,
                "reference_abstain_rate": reference_abstain_rate,
                "current_abstain_rate": current_abstain_rate,
                "abstain_rate_delta": abstain_rate_delta,
            },
            "action_drift": action_drift,
            "regime_report": {
                "distribution": regime_distribution,
                "reference_transition_distribution": reference_transition,
                "current_transition_distribution": current_transition,
                "transition_total_variation": regime_transition_tv,
            },
            "regime_drift": regime_drift,
            "performance_updates": performance_updates,
            "performance_drift": performance_drift,
            "interval": self.config.get("interval"),
            "bars_per_day": self.config.get("bars_per_day"),
            "cooldown_bars": int(self.config["cooldown_bars"]),
            "min_samples": int(self.config["min_samples"]),
            "model_ttl_expired": model_ttl_expired,
            "max_bars_between_retrain": max_bars_between_retrain,
            "trade_rate_min_samples": self.config.get("trade_rate_min_samples"),
            "cooldown_active": cooldown_active,
            "minimum_samples_met": enough_samples,
            "evidence_count": evidence_count,
            "recommendation": {
                "should_retrain": should_retrain,
                "reasons": reasons,
                "bars_since_last_retrain": bars_since_last_retrain,
                "max_bars_between_retrain": max_bars_between_retrain,
                "model_ttl_expired": model_ttl_expired,
            },
        }


def evaluate_drift_guardrails(drift_report, policy=None):
    policy = _resolve_drift_policy(policy)
    minimum_samples = int(policy.get("min_samples", 200))
    cooldown_bars = int(policy.get("cooldown_bars", 500))
    minimum_signal_count = int(policy.get("min_drift_signals", 2))
    report = dict(drift_report or {})
    sample_count = int(report.get("sample_count") or 0)
    evidence_count = int(report.get("evidence_count") or 0)
    bars_since_last_retrain = report.get("recommendation", {}).get("bars_since_last_retrain")
    model_ttl_expired = bool(report.get("model_ttl_expired", False))
    max_bars_between_retrain = int(policy.get("max_bars_between_retrain", report.get("max_bars_between_retrain") or (24 * 28)))
    cooldown_active = bars_since_last_retrain is not None and int(bars_since_last_retrain) < cooldown_bars and not model_ttl_expired
    approved = bool(
        sample_count >= minimum_samples
        and (evidence_count >= minimum_signal_count or model_ttl_expired)
        and not cooldown_active
        and bool(report.get("recommendation", {}).get("should_retrain", False))
    )
    reasons = []
    if model_ttl_expired:
        reasons.append("model_ttl_expired")
    if sample_count < minimum_samples:
        reasons.append("minimum_samples_not_met")
    if evidence_count < minimum_signal_count and not model_ttl_expired:
        reasons.append("insufficient_drift_evidence")
    if cooldown_active:
        reasons.append("cooldown_active")
    if approved:
        reasons.append("approved")
    return {
        "approved": approved,
        "reasons": reasons,
        "sample_count": sample_count,
        "evidence_count": evidence_count,
        "interval": policy.get("interval"),
        "bars_per_day": policy.get("bars_per_day"),
        "cooldown_bars": int(cooldown_bars),
        "min_samples": int(minimum_samples),
        "model_ttl_expired": model_ttl_expired,
        "max_bars_between_retrain": max_bars_between_retrain,
        "trade_rate_min_samples": policy.get("trade_rate_min_samples"),
    }


__all__ = [
    "ADWINDetector",
    "DriftMonitor",
    "evaluate_drift_guardrails",
]