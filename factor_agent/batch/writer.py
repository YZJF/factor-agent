from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from factor_agent.batch.manifest import BatchManifest, ManifestEntry
from factor_agent.documents.models import ReportSnapshot
from factor_agent.schemas import FactorCase
from factor_agent.spec.requirements import missing_requirements
from factor_agent.split import assert_no_leak, group_key


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write(path: Path, payload: Any) -> str:
    content = _json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def write_batch(
    cases: list[FactorCase],
    snapshots: dict[str, ReportSnapshot],
    out_dir: str | Path,
    *,
    batch_id: str,
    verifier_spec: dict[str, Any] | None = None,
    require_evidence: bool = True,
) -> BatchManifest:
    assert_no_leak(cases)
    root = Path(out_dir)
    entries: list[ManifestEntry] = []
    score_spec = verifier_spec or {
        "weights": {"schema_ok": 0.25, "executable": 0.25, "fields": 0.40, "grounding": 0.10},
        "caps": {"invalid_json_cap": 0.0, "illegal_expr_cap": 0.15, "hallucinated_expr_cap": 0.35},
    }
    for case in cases:
        if not case.document or case.document.doc_id not in snapshots:
            raise ValueError(f"missing report snapshot for case {case.case_id}")
        if require_evidence:
            problems = missing_requirements(case.gold)
            if problems:
                raise ValueError(f"case {case.case_id} is not a usable label: {problems}")
        case_root = root / case.case_id
        paths = {
            "case": case_root / "case.json",
            "report": case_root / "report_snapshot.json",
            "verifier": case_root / "verifier_spec.json",
            "gold": case_root / "gold.json",
        }
        hashes = {
            "case": _write(paths["case"], case.model_dump(exclude={"gold", "report"})),
            "report": _write(paths["report"], snapshots[case.document.doc_id].model_dump()),
            "verifier": _write(paths["verifier"], score_spec),
            "gold": _write(paths["gold"], {"case_id": case.case_id, "gold": case.gold.model_dump()}),
        }
        entries.append(
            ManifestEntry(
                case_id=case.case_id,
                group_id=group_key(case),
                split=case.split,
                case_path=str(paths["case"]),
                report_snapshot_path=str(paths["report"]),
                verifier_spec_path=str(paths["verifier"]),
                gold_path=str(paths["gold"]),
                content_hashes=hashes,
            )
        )
    manifest = BatchManifest(batch_id=batch_id, entries=entries)
    _write(root / "manifest.json", manifest.model_dump())
    return manifest
