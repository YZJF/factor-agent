"""Build an injected-error eval set from gold. Tune the verifier, not train RM."""

from __future__ import annotations

from pathlib import Path

from factor_agent.io import dump_dicts
from factor_agent.schemas import FactorCase
from factor_agent.score.inject import INJECT_KINDS, corrupt
from factor_agent.score.verifier import score_prediction


def inject_rows(cases: list[FactorCase]) -> list[dict]:
    rows: list[dict] = []
    for case in cases:
        gold_r = score_prediction(case, case.gold.model_dump_json())
        rows.append(
            {
                "case_id": case.case_id,
                "kind": "gold",
                "reward": gold_r.reward,
                "active_caps": gold_r.active_caps,
                "pred": case.gold.model_dump(),
            }
        )
        for kind in INJECT_KINDS:
            pred = corrupt(case.gold, kind)
            result = score_prediction(case, pred.model_dump_json(), pred=pred)
            rows.append(
                {
                    "case_id": case.case_id,
                    "kind": kind,
                    "reward": result.reward,
                    "active_caps": result.active_caps,
                    "pred": pred.model_dump(),
                }
            )
    return rows


def build_inject_eval(cases: list[FactorCase], out_path: str | Path) -> int:
    rows = inject_rows(cases)
    dump_dicts(out_path, rows)
    return len(rows)
