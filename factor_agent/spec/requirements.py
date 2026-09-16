"""Asset-level requirements: every expression parameter is declared and located.

The verifier prices grounding as a subscore; these checks are the harder gate
applied when an annotation is promoted into a `case / report / verifier / gold`
asset or replayed as an SFT trace. An annotation that carries a 20-day window
without a chunk- or page-level citation for that 20 is not a usable label.
"""

from __future__ import annotations

from factor_agent.documents.normalize import normalize_chinese_numbers
from factor_agent.schemas import FactorSpec
from factor_agent.spec.expr import numbers_in_text


def expr_parameters(spec: FactorSpec) -> set[int]:
    return numbers_in_text(spec.expr)


def declared_parameters(spec: FactorSpec) -> set[int]:
    return set(spec.windows)


def undeclared_parameters(spec: FactorSpec) -> set[int]:
    return expr_parameters(spec) - declared_parameters(spec)


def evidence_text(spec: FactorSpec) -> str:
    return normalize_chinese_numbers("\n".join(span.quote for span in spec.evidence))


def uncited_parameters(spec: FactorSpec) -> set[int]:
    present = numbers_in_text(evidence_text(spec))
    return (expr_parameters(spec) | declared_parameters(spec)) - present


def unlocated_spans(spec: FactorSpec) -> list[str]:
    """Spans that cannot be resolved back to a chunk plus a page or char range."""

    return [
        span.chunk_id or "<no chunk_id>"
        for span in spec.evidence
        if not span.chunk_id.strip()
        or (span.page_start is None and span.char_start is None)
    ]


def missing_requirements(spec: FactorSpec) -> list[str]:
    problems: list[str] = []
    if not spec.evidence:
        problems.append("no evidence span")
    undeclared = undeclared_parameters(spec)
    if undeclared:
        problems.append(f"expression parameters not declared in windows: {sorted(undeclared)}")
    uncited = uncited_parameters(spec)
    if uncited:
        problems.append(f"parameters without a citation: {sorted(uncited)}")
    unlocated = unlocated_spans(spec)
    if unlocated:
        problems.append(f"spans without chunk/page locator: {unlocated}")
    return problems
