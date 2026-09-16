from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RoutingThresholds(BaseModel):
    high_reward: float = 0.80
    low_reward: float = 0.30
    rl_spread_min: float = 0.30
    all_high_spread_max: float = 0.10
    error_rate_high: float = 0.50


def route_case(
    metrics: dict[str, Any],
    *,
    integrity: dict[str, Any] | None = None,
    thresholds: RoutingThresholds | None = None,
    split: str = "train",
) -> dict[str, str]:
    threshold = thresholds or RoutingThresholds()
    if split == "holdout":
        return {"route": "heldout_eval", "reason": "holdout is never promoted into training pools"}
    if integrity is not None and not integrity.get("ok", False):
        return {"route": "quarantine", "reason": ",".join(integrity.get("errors") or ["integrity_failed"])}
    if metrics.get("environment_error_rate", 0.0) >= threshold.error_rate_high:
        return {"route": "backtest_gap", "reason": "environment/backtest errors dominate"}
    if metrics.get("parse_error_rate", 0.0) >= threshold.error_rate_high:
        return {"route": "sft_format", "reason": "high JSON parse error rate"}
    if metrics.get("grounding_error_rate", 0.0) >= threshold.error_rate_high:
        return {"route": "sft_grounding", "reason": "high grounding/citation error rate"}
    if metrics.get("tool_error_rate", 0.0) >= threshold.error_rate_high:
        return {"route": "sft_tool", "reason": "high model-caused tool error rate"}
    if (
        metrics.get("min_reward", 0.0) >= threshold.high_reward
        and metrics.get("reward_spread", 0.0) <= threshold.all_high_spread_max
    ):
        return {"route": "all_high_eval", "reason": "already solved with low rollout variance"}
    if metrics.get("any_success") and metrics.get("reward_spread", 0.0) >= threshold.rl_spread_min:
        return {"route": "rl_main", "reason": "learnable within-group reward variation"}
    if metrics.get("max_step_hit_rate", 0.0) >= threshold.error_rate_high:
        return {"route": "long_trajectory", "reason": "frequent max-step termination"}
    if (
        metrics.get("max_reward", 0.0) < threshold.high_reward
        and metrics.get("reward_spread", 0.0) <= threshold.all_high_spread_max
    ):
        return {"route": "sft_curriculum", "reason": "consistently weak but structurally clean"}
    if metrics.get("max_reward", 0.0) < threshold.low_reward:
        return {"route": "more_probe", "reason": "all-low group needs diagnosis"}
    return {"route": "more_probe", "reason": "insufficient evidence for promotion"}
