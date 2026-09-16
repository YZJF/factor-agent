"""Rule verifier: gold spec vs prediction. No LLM judge."""

from __future__ import annotations

from factor_agent.documents.normalize import normalize_chinese_numbers
from factor_agent.parse import parse_spec
from factor_agent.schemas import (
    ACTIVE_CAPS,
    WEIGHTS,
    FactorCase,
    FactorSpec,
    RewardResult,
    Subscores,
)
from factor_agent.spec.consistency import declaration_issues
from factor_agent.spec.expr import expr_ok, numbers_in_text
from factor_agent.spec.requirements import evidence_text, missing_requirements

FIELD_KEYS = ("universe", "frequency", "neutralization", "rebalance")


def _field_score(pred: FactorSpec, gold: FactorSpec) -> float:
    hits = sum(getattr(pred, k) == getattr(gold, k) for k in FIELD_KEYS)
    expr_hit = 1.0 if _norm_expr(pred.expr) == _norm_expr(gold.expr) else 0.0
    return (hits + expr_hit) / (len(FIELD_KEYS) + 1)


def _norm_expr(expr: str) -> str:
    return "".join(expr.split()).lower()


def _citation_errors(pred: FactorSpec, report: str) -> list[str]:
    errors = [
        f"{citation.chunk_id}: quote not found"
        for citation in pred.evidence
        if not citation.quote.strip() or citation.quote not in report
    ]
    errors += [
        "<no chunk_id>: citation is not anchored to a chunk"
        for citation in pred.evidence
        if not citation.chunk_id.strip()
    ]
    return errors


def _grounding(pred: FactorSpec) -> float:
    """Parameters must appear inside the spans the model actually cited."""

    nums = set(pred.windows) | numbers_in_text(pred.expr)
    if not pred.evidence:
        return 0.0
    if not nums:
        return 1.0
    present = numbers_in_text(evidence_text(pred))
    return sum(n in present for n in nums) / len(nums)


def _report_support(pred: FactorSpec, report: str) -> float:
    """Permissive check used only to separate fabrication from a missing citation."""

    nums = set(pred.windows) | numbers_in_text(pred.expr)
    if not nums:
        return 1.0
    present = numbers_in_text(normalize_chinese_numbers(report))
    return sum(n in present for n in nums) / len(nums)


def _hallucinated(pred: FactorSpec, gold: FactorSpec, report: str) -> bool:
    if _norm_expr(pred.expr) == _norm_expr(gold.expr):
        return False
    ok, _ = expr_ok(pred.expr)
    if not ok:
        return False
    if pred.universe != gold.universe:
        return True
    return _report_support(pred, report) < 0.3


def score_prediction(
    case: FactorCase,
    solution_str: str,
    pred: FactorSpec | None = None,
) -> RewardResult:
    caps: list[str] = []
    reasons: dict[str, str] = {}
    parsed = pred if pred is not None else parse_spec(solution_str)
    issues: list[str] = []

    if parsed is None:
        caps.append("invalid_json_cap")
        reasons["invalid_json_cap"] = "could not parse FactorSpec JSON"
        raw = 0.0
        subs = Subscores()
    else:
        issues = declaration_issues(parsed)
        schema_ok = 1.0
        exe_ok, exe_msg = expr_ok(parsed.expr)
        executable = 1.0 if exe_ok else 0.0
        if not exe_ok:
            caps.append("illegal_expr_cap")
            reasons["illegal_expr_cap"] = exe_msg
        citation_errors = _citation_errors(parsed, case.report)
        if citation_errors:
            caps.append("invalid_citation_cap")
            reasons["invalid_citation_cap"] = "; ".join(citation_errors)
        fields = _field_score(parsed, case.gold)
        grounding = _grounding(parsed)
        if _hallucinated(parsed, case.gold, case.report):
            caps.append("hallucinated_expr_cap")
            reasons["hallucinated_expr_cap"] = "expr ungrounded vs report/gold"
        subs = Subscores(
            schema_ok=schema_ok,
            executable=executable,
            fields=fields,
            grounding=grounding,
        )
        raw = (
            WEIGHTS["schema_ok"] * schema_ok
            + WEIGHTS["executable"] * executable
            + WEIGHTS["fields"] * fields
            + WEIGHTS["grounding"] * grounding
        )

    cap = min((ACTIVE_CAPS[c] for c in caps), default=1.0)
    final = min(raw, cap)
    return RewardResult(
        case_id=case.case_id,
        reward=final,
        raw_reward=raw,
        subscores=subs,
        active_caps=caps,
        cap_reasons=reasons,
        diagnostics={
            "weights": WEIGHTS,
            "declaration_issues": issues,
            "citation_count": len(parsed.evidence) if parsed is not None else 0,
            "requirement_issues": missing_requirements(parsed) if parsed is not None else [],
            "as_of_bars": 1,
        },
    )


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """verl-style hook: (output, gold json) -> float."""
    import json

    payload = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if "report" in payload and "gold" in payload:
        case = FactorCase.model_validate(payload)
    else:
        case = FactorCase(
            case_id=payload.get("case_id", "unknown"),
            report=payload.get("report", ""),
            gold=FactorSpec.model_validate(payload["gold"] if "gold" in payload else payload),
        )
    return score_prediction(case, solution_str).reward
