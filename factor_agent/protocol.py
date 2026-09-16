"""Illegal tool-call taxonomy shared by stage A and stage B rollouts.

A call is illegal when the *call itself* breaks the tool contract. A legal call
that returns a negative verdict (``validate_expr`` reporting an operator outside
the whitelist) is not illegal — that failure is already priced into the reward
subscores, and double counting it would make the illegal-call rate track
expression quality instead of protocol compliance.
"""

from __future__ import annotations

from typing import Any

UNKNOWN_TOOL = "unknown_tool"
UNPARSABLE_ARGUMENTS = "unparsable_arguments"
SCHEMA_INVALID = "schema_invalid"
BUDGET_EXCEEDED = "budget_exceeded"
FROZEN_FIELD_MUTATION = "frozen_field_mutation"
MULTI_TOOL_PER_STEP = "multi_tool_per_step"

ILLEGAL_KINDS = (
    UNKNOWN_TOOL,
    UNPARSABLE_ARGUMENTS,
    SCHEMA_INVALID,
    BUDGET_EXCEEDED,
    FROZEN_FIELD_MUTATION,
    MULTI_TOOL_PER_STEP,
)


def tool_names(schemas: list[dict[str, Any]]) -> set[str]:
    return {str(schema.get("function", schema).get("name", "")) for schema in schemas}


def classify_call(
    *,
    name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    allowed: set[str],
) -> str | None:
    """Return the illegal-call kind, or None when the call honoured the contract."""

    if name not in allowed:
        return UNKNOWN_TOOL
    if "_raw" in arguments:
        return UNPARSABLE_ARGUMENTS
    if result.get("ok"):
        return None
    if result.get("source") == "environment":
        return None
    error = str(result.get("error", ""))
    if "max evaluation rounds" in error:
        return BUDGET_EXCEEDED
    if "invalid FactorSpec" in error:
        return SCHEMA_INVALID
    return None


def audit(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one trajectory's observations into protocol-compliance counters."""

    total = len(observations)
    by_kind: dict[str, int] = {}
    for observation in observations:
        kind = observation.get("illegal_kind")
        if kind:
            by_kind[kind] = by_kind.get(kind, 0) + 1
    illegal = sum(by_kind.values())
    return {
        "tool_calls": total,
        "illegal_tool_calls": illegal,
        "illegal_tool_call_rate": illegal / total if total else 0.0,
        "illegal_by_kind": by_kind,
        "environment_failures": sum(
            1
            for observation in observations
            if not observation.get("ok") and observation.get("source") == "environment"
        ),
    }


def mark_multi_tool_steps(observations: list[dict[str, Any]]) -> None:
    """Flag every call in a step that emitted more than one tool call."""

    per_step: dict[Any, int] = {}
    for observation in observations:
        per_step[observation.get("step")] = per_step.get(observation.get("step"), 0) + 1
    for observation in observations:
        if per_step.get(observation.get("step"), 0) > 1 and not observation.get("illegal_kind"):
            observation["illegal_kind"] = MULTI_TOOL_PER_STEP
            observation["illegal"] = True
