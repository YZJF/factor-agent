from __future__ import annotations

from typing import Any

from factor_agent.gate import passed_extract_gate
from factor_agent.schemas import FactorCase
from factor_agent.score.verifier import score_prediction


def check_case_integrity(case: FactorCase) -> dict[str, Any]:
    replay = score_prediction(case, case.gold.model_dump_json(), pred=case.gold)
    errors: list[str] = []
    if not passed_extract_gate(replay, case.gold):
        errors.append("gold_replay_failed")
    if case.document and case.document.content_hash == "":
        errors.append("missing_document_hash")
    if case.gold.evidence and any(citation.quote not in case.report for citation in case.gold.evidence):
        errors.append("gold_citation_not_in_report")
    return {
        "ok": not errors,
        "errors": errors,
        "gold_reward": replay.reward,
        "case_id": case.case_id,
    }
