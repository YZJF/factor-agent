"""Declaration vs implementation. Recorded in diagnostics; not a weight yet."""

from __future__ import annotations

from factor_agent.schemas import FactorSpec

_FREQ_RANK = {"daily": 0, "weekly": 1, "monthly": 2}


def declaration_issues(spec: FactorSpec) -> list[str]:
    issues: list[str] = []
    expr = spec.expr.upper()
    if spec.neutralization in {"industry", "industry_size"} and "INDUSTRY_NEUTRALIZE" not in expr:
        issues.append("neutralization_missing_in_expr")
    if spec.neutralization == "none" and "INDUSTRY_NEUTRALIZE" in expr:
        issues.append("neutralize_in_expr_but_field_none")
    if _FREQ_RANK[spec.rebalance] < _FREQ_RANK[spec.frequency]:
        issues.append("rebalance_faster_than_frequency")
    return issues
