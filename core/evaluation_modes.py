from __future__ import annotations

from dataclasses import asdict, dataclass


VALID_EVALUATION_MODES = {"research_only", "local_certification", "trade_ready"}


@dataclass(frozen=True, slots=True)
class EvaluationModeResolution:
    requested_mode: str
    effective_mode: str
    is_research_only: bool
    is_capital_facing: bool
    research_only_override: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_evaluation_mode(backtest_config=None) -> EvaluationModeResolution:
    config = dict(backtest_config or {})
    requested_mode = str(config.get("evaluation_mode", "research_only")).strip().lower() or "research_only"
    if requested_mode not in VALID_EVALUATION_MODES:
        allowed = ", ".join(sorted(VALID_EVALUATION_MODES))
        raise ValueError(
            f"Unsupported backtest.evaluation_mode={requested_mode!r}. Expected one of: {allowed}."
        )

    research_only_override = bool(config.get("research_only_override", False))
    if research_only_override and requested_mode != "research_only":
        raise ValueError(
            "backtest.research_only_override=true is only valid when backtest.evaluation_mode='research_only'."
        )

    effective_mode = "research_demo" if requested_mode == "research_only" else requested_mode
    return EvaluationModeResolution(
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        is_research_only=requested_mode == "research_only",
        is_capital_facing=requested_mode in {"local_certification", "trade_ready"},
        research_only_override=research_only_override,
    )


__all__ = ["EvaluationModeResolution", "VALID_EVALUATION_MODES", "resolve_evaluation_mode"]