from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from factor_agent.schemas import FactorCase


def case_from_dict(raw: dict[str, Any]) -> FactorCase:
    """Accept FactorCase or shorthand {id, spec} rows."""
    data = dict(raw)
    if "gold" not in data and "spec" in data:
        data["gold"] = data["spec"]
    if "case_id" not in data:
        data["case_id"] = data.get("id") or data.get("caseId") or "unknown"
    return FactorCase.model_validate(data)


def load_jsonl(path: str | Path) -> list[FactorCase]:
    cases: list[FactorCase] = []
    p = Path(path)
    if not p.exists():
        return cases
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(case_from_dict(json.loads(line)))
    return cases


def dump_jsonl(path: str | Path, cases: list[FactorCase]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(case.model_dump_json() + "\n")


def dump_dicts(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
