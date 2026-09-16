from __future__ import annotations

from collections.abc import Iterable
from statistics import median
from typing import Any

from factor_agent.schemas import RewardResult

TASK_SUCCESS_MIN = 0.7
GROUNDING_MIN = 1.0

HEADLINE_KEYS = ("task_success_rate", "evidence_grounding_rate", "illegal_tool_call_rate")


def task_success(result: RewardResult) -> bool:
    """End-to-end success: parses, executes, every field matches, no cap fired."""

    return (
        not result.active_caps
        and result.reward >= TASK_SUCCESS_MIN
        and result.subscores.fields >= 1.0
        and result.subscores.executable >= 1.0
    )


def evidence_grounded(result: RewardResult) -> bool:
    """Every expression parameter located inside a valid citation."""

    return (
        result.subscores.grounding >= GROUNDING_MIN
        and "invalid_citation_cap" not in result.active_caps
    )


def summarize(
    results: list[RewardResult],
    *,
    tool_audits: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    audits = list(tool_audits or [])
    calls = sum(int(item.get("tool_calls", 0)) for item in audits)
    illegal = sum(int(item.get("illegal_tool_calls", 0)) for item in audits)
    tool_summary = {
        "tool_calls": float(calls),
        "illegal_tool_calls": float(illegal),
        "illegal_tool_call_rate": illegal / calls if calls else 0.0,
    }
    if not results:
        return {
            "n": 0,
            "mean_reward": 0.0,
            "parse_ok": 0.0,
            "field_mean": 0.0,
            "grounding_mean": 0.0,
            "gate_pass_rate": 0.0,
            "hallucination_rate": 0.0,
            "task_success_rate": 0.0,
            "evidence_grounding_rate": 0.0,
            **tool_summary,
        }
    n = len(results)
    return {
        "n": float(n),
        "mean_reward": sum(r.reward for r in results) / n,
        "parse_ok": sum("invalid_json_cap" not in r.active_caps for r in results) / n,
        "field_mean": sum(r.subscores.fields for r in results) / n,
        "grounding_mean": sum(r.subscores.grounding for r in results) / n,
        "gate_pass_rate": sum(r.reward >= 0.7 and not r.active_caps for r in results) / n,
        "hallucination_rate": sum("hallucinated_expr_cap" in r.active_caps for r in results) / n,
        "task_success_rate": sum(task_success(r) for r in results) / n,
        "evidence_grounding_rate": sum(evidence_grounded(r) for r in results) / n,
        **tool_summary,
    }


def summarize_by_group(
    results: list[RewardResult],
    groups: dict[str, str],
) -> dict[str, Any]:
    """Held-out metrics per institution/series group, plus the across-group median.

    A single prolific institution would otherwise dominate the headline average,
    so the per-group median is what we report alongside the pooled numbers.
    """

    buckets: dict[str, list[RewardResult]] = {}
    for result in results:
        buckets.setdefault(groups.get(result.case_id, result.case_id), []).append(result)
    per_group = {name: summarize(bucket) for name, bucket in sorted(buckets.items())}
    medians = {
        f"median_{key}": median([summary[key] for summary in per_group.values()])
        for key in ("task_success_rate", "evidence_grounding_rate", "mean_reward")
        if per_group
    }
    return {
        "group_count": len(per_group),
        "pooled": summarize(results),
        "per_group": per_group,
        **medians,
    }


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Before/after deltas in percentage points for the headline rates."""

    delta = {}
    for key in HEADLINE_KEYS:
        start = float(before.get(key, 0.0))
        end = float(after.get(key, 0.0))
        delta[key] = {
            "before": round(start * 100, 1),
            "after": round(end * 100, 1),
            "delta_pp": round((end - start) * 100, 1),
        }
    return delta
