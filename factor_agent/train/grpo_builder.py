"""GRPO prompt rows. Reward is computed at train time via verifier.compute_score."""

from __future__ import annotations

import json
from pathlib import Path

from factor_agent.batch.manifest import BatchManifest, ManifestEntry
from factor_agent.documents.tools import ReportToolRuntime
from factor_agent.evolve.run import evolve_tool_schemas
from factor_agent.io import dump_dicts
from factor_agent.prompts import EVOLVE_SYSTEM, EXTRACT_SYSTEM, extract_user
from factor_agent.schemas import FactorCase
from factor_agent.split import assert_no_leak


def to_grpo_row(case: FactorCase, entry: ManifestEntry | None = None) -> dict:
    extra_info = {
        "case_id": case.case_id,
        "group_id": case.group_id,
        "split": case.split,
        "stage": "extract",
        "adapter": "extract_lora",
    }
    if entry is not None:
        extra_info.update(
            {
                "case_path": entry.case_path,
                "report_snapshot_path": entry.report_snapshot_path,
                "verifier_spec_path": entry.verifier_spec_path,
                "gold_path": entry.gold_path,
                "content_hashes": entry.content_hashes,
            }
        )
    return {
        "data_source": "factor_extract",
        "prompt": [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": extract_user(case.report)},
        ],
        "reward_model": {
            "ground_truth": json.dumps(case.model_dump(), ensure_ascii=False),
            "name": "factor_rule_verifier",
        },
        "tools": ReportToolRuntime.schemas(),
        "extra_info": extra_info,
    }


def to_evolve_grpo_row(case: FactorCase, entry: ManifestEntry | None = None) -> dict:
    row = to_grpo_row(case, entry)
    row["data_source"] = "factor_evolve"
    row["prompt"] = [
        {"role": "system", "content": EVOLVE_SYSTEM},
        {
            "role": "user",
            "content": f"Seed FactorSpec:\n{case.gold.model_dump_json()}\nImprove it without changing frozen fields.",
        },
    ]
    row["tools"] = evolve_tool_schemas()
    row["extra_info"]["stage"] = "evolve"
    row["extra_info"]["adapter"] = "evolve_lora"
    row["extra_info"]["seed"] = case.gold.model_dump()
    row["extra_info"]["agent_name"] = "factor_evolve_agent"
    return row


def build_grpo(
    cases: list[FactorCase],
    out_path: str | Path,
    manifest: BatchManifest | None = None,
    *,
    stage: str = "extract",
) -> int:
    assert_no_leak(cases)
    entries = {entry.case_id: entry for entry in manifest.entries} if manifest else {}
    row_builder = to_evolve_grpo_row if stage == "evolve" else to_grpo_row
    rows = [row_builder(c, entries.get(c.case_id)) for c in cases if c.split == "train"]
    dump_dicts(out_path, rows)
    return len(rows)
