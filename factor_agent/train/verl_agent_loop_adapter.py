"""Version-stable token buffer used by a verl AgentLoop integration.

The installed verl version owns generation and `AgentLoopOutput`. Keeping this
buffer free of verl imports lets us test the critical token/mask contract before
binding it to a specific upstream release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from factor_agent.train.rollout_correction import validate_response_mask


@dataclass
class FactorAgentLoopBuffer:
    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    rollout_log_probs: list[float] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)

    def append_model(self, token_ids: list[int], log_probs: list[float] | None = None, *, step: int) -> None:
        if log_probs is not None and len(log_probs) != len(token_ids):
            raise ValueError("model token/log-prob lengths differ")
        self.response_ids.extend(token_ids)
        self.response_mask.extend([1] * len(token_ids))
        self.rollout_log_probs.extend(log_probs or [0.0] * len(token_ids))
        self.segments.append({"type": "model", "step": step, "token_ids": token_ids, "mask": 1})

    def append_environment(self, token_ids: list[int], *, segment_type: str, step: int) -> None:
        self.response_ids.extend(token_ids)
        self.response_mask.extend([0] * len(token_ids))
        self.rollout_log_probs.extend([0.0] * len(token_ids))
        self.segments.append({"type": segment_type, "step": step, "token_ids": token_ids, "mask": 0})

    def validate(self, training_log_probs: list[float] | None = None) -> None:
        validate_response_mask(
            self.response_ids,
            self.response_mask,
            rollout_log_probs=self.rollout_log_probs,
            training_log_probs=training_log_probs,
        )

    def as_verl_fields(self, *, max_response_length: int | None = None) -> dict[str, Any]:
        limit = max_response_length or len(self.response_ids)
        response_ids = self.response_ids[:limit]
        response_mask = self.response_mask[:limit]
        rollout_log_probs = self.rollout_log_probs[:limit]
        validate_response_mask(
            response_ids,
            response_mask,
            rollout_log_probs=rollout_log_probs,
        )
        return {
            "prompt_ids": self.prompt_ids,
            "response_ids": response_ids,
            "response_mask": response_mask,
            "response_logprobs": rollout_log_probs,
            "token_trace": {"segments": self.segments, "response_mask": response_mask},
        }


def rollout_correction_overrides(*, enabled: bool, threshold: float = 2.0) -> list[str]:
    if not enabled:
        return []
    if threshold <= 1.0:
        raise ValueError("rollout correction threshold must be greater than 1")
    return [
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "algorithm.rollout_correction.rollout_is=sequence",
        f"algorithm.rollout_correction.rollout_is_threshold={threshold}",
        "algorithm.rollout_correction.rollout_is_batch_normalize=False",
        "algorithm.rollout_correction.bypass_mode=False",
    ]
