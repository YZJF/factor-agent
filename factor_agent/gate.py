"""Extract must pass before evolve. Stage B IR never flows back here."""

from __future__ import annotations

from factor_agent.schemas import FactorSpec, RewardResult
from factor_agent.spec.expr import expr_ok

EXTRACT_REWARD_MIN = 0.7


def passed_extract_gate(
    reward: RewardResult,
    pred: FactorSpec | None,
    min_reward: float = EXTRACT_REWARD_MIN,
) -> bool:
    if pred is None:
        return False
    ok, _ = expr_ok(pred.expr)
    return reward.reward >= min_reward and not reward.active_caps and ok
