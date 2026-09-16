from __future__ import annotations

import math
from statistics import mean
from typing import Any


def validate_response_mask(
    response_ids: list[int],
    response_mask: list[int],
    *,
    rollout_log_probs: list[float] | None = None,
    training_log_probs: list[float] | None = None,
) -> None:
    if len(response_ids) != len(response_mask):
        raise ValueError("response_ids and response_mask lengths differ")
    if not response_ids:
        raise ValueError("empty response")
    if any(value not in {0, 1} for value in response_mask):
        raise ValueError("response_mask must contain only 0/1")
    if not any(response_mask):
        raise ValueError("response_mask contains no model-generated tokens")
    for name, values in (
        ("rollout_log_probs", rollout_log_probs),
        ("training_log_probs", training_log_probs),
    ):
        if values is not None and len(values) != len(response_ids):
            raise ValueError(f"{name} length differs from response_ids")


def build_response_mask(segments: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    """Flatten token segments; only model-generated segments participate in loss."""

    response_ids: list[int] = []
    response_mask: list[int] = []
    for segment in segments:
        token_ids = [int(token) for token in segment.get("token_ids") or []]
        is_model = segment.get("type") == "model"
        response_ids.extend(token_ids)
        response_mask.extend([1 if is_model else 0] * len(token_ids))
    validate_response_mask(response_ids, response_mask)
    return response_ids, response_mask


def sequence_tis(
    rollout_log_probs: list[float],
    training_log_probs: list[float],
    response_mask: list[int],
    *,
    threshold: float = 2.0,
) -> dict[str, float | bool]:
    if threshold <= 1.0:
        raise ValueError("TIS threshold must be greater than 1")
    response_ids = list(range(len(response_mask)))
    validate_response_mask(
        response_ids,
        response_mask,
        rollout_log_probs=rollout_log_probs,
        training_log_probs=training_log_probs,
    )
    differences = [
        training - rollout
        for rollout, training, mask in zip(rollout_log_probs, training_log_probs, response_mask)
        if mask
    ]
    log_ratio = sum(differences)
    raw_ratio = math.exp(max(-50.0, min(50.0, log_ratio)))
    lower = 1.0 / threshold
    clipped_ratio = min(threshold, max(lower, raw_ratio))
    return {
        "log_ratio": log_ratio,
        "raw_ratio": raw_ratio,
        "clipped_ratio": clipped_ratio,
        "was_clipped": clipped_ratio != raw_ratio,
        "mean_abs_logprob_delta": mean(abs(value) for value in differences),
    }


def correction_diagnostics(
    samples: list[dict[str, Any]],
    *,
    threshold: float = 2.0,
    enable_delta: float = 0.02,
) -> dict[str, Any]:
    results = [
        sequence_tis(
            sample["rollout_log_probs"],
            sample["training_log_probs"],
            sample["response_mask"],
            threshold=threshold,
        )
        for sample in samples
    ]
    mean_delta = mean(float(item["mean_abs_logprob_delta"]) for item in results) if results else 0.0
    clip_rate = sum(bool(item["was_clipped"]) for item in results) / len(results) if results else 0.0
    return {
        "sample_count": len(results),
        "mean_abs_logprob_delta": mean_delta,
        "clip_rate": clip_rate,
        "threshold": threshold,
        "recommended": bool(results) and mean_delta >= enable_delta,
        "reason": "systematic rollout/training log-prob gap" if results and mean_delta >= enable_delta else "gap below enable threshold",
    }
