from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev
from typing import Any


def compute_group_metrics(records: list[dict[str, Any]], *, high_reward: float = 0.8) -> dict[str, Any]:
    if not records:
        return {
            "k": 0,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "mean_reward": 0.0,
            "reward_std": 0.0,
            "reward_spread": 0.0,
            "success_at_k": 0.0,
            "any_success": False,
            "illegal_tool_call_rate": 0.0,
        }
    rewards = [float(record.get("reward", 0.0)) for record in records]
    tool_calls = sum(int(record.get("tool_calls", 0)) for record in records)
    illegal_calls = sum(int(record.get("illegal_tool_calls", 0)) for record in records)
    caps: Counter[str] = Counter()
    for record in records:
        caps.update(record.get("active_caps") or [])
    rate = lambda predicate: sum(1 for record in records if predicate(record)) / len(records)
    successes = sum(reward >= high_reward for reward in rewards)
    return {
        "k": len(records),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "mean_reward": mean(rewards),
        "reward_std": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "reward_spread": max(rewards) - min(rewards),
        "success_at_k": successes / len(records),
        "any_success": successes > 0,
        "parse_error_rate": rate(lambda item: item.get("parse_error") or "invalid_json_cap" in (item.get("active_caps") or [])),
        "grounding_error_rate": rate(
            lambda item: item.get("grounding_error")
            or bool({"invalid_citation_cap", "hallucinated_expr_cap"} & set(item.get("active_caps") or []))
        ),
        "illegal_tool_call_rate": illegal_calls / tool_calls if tool_calls else 0.0,
        "tool_error_rate": rate(lambda item: int(item.get("tool_error_model", 0)) > 0),
        "environment_error_rate": rate(lambda item: int(item.get("tool_error_environment", 0)) > 0),
        "spec_drift_rate": rate(lambda item: item.get("spec_drift") or "spec_drift_cap" in (item.get("active_caps") or [])),
        "backtest_failure_rate": rate(lambda item: item.get("backtest_failed", False)),
        "max_step_hit_rate": rate(lambda item: item.get("max_step_hit", False)),
        "hard_cap_distribution": dict(caps),
    }
