from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from factor_agent.trajectory import Trajectory


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class RolloutStore:
    def __init__(self, root: str | Path, *, run_id: str):
        self.run_dir = Path(root) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        trajectory: Trajectory,
        *,
        score: dict[str, Any],
        prompt: list[dict[str, Any]] | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        rollout_log_probs: list[float] | None = None,
        training_log_probs: list[float] | None = None,
        rollout_id: str | None = None,
    ) -> dict[str, Any]:
        identifier = rollout_id or str(uuid4())
        artifact_dir = self.run_dir / trajectory.case_id / identifier
        prompt_payload = prompt or [
            {"role": step.role, "content": step.content}
            for step in trajectory.steps
            if step.role in {"system", "user"}
        ]
        tool_audit = dict(trajectory.extras.get("tool_audit") or {})
        record = {
            "run_id": self.run_dir.name,
            "rollout_id": identifier,
            "case_id": trajectory.case_id,
            "stage": trajectory.stage,
            "reward": float(score.get("reward", trajectory.reward or 0.0)),
            "active_caps": score.get("active_caps", []),
            "tool_calls": int(tool_audit.get("tool_calls", 0)),
            "illegal_tool_calls": int(tool_audit.get("illegal_tool_calls", 0)),
            "illegal_by_kind": tool_audit.get("illegal_by_kind", {}),
            "prompt_hash": _stable_hash(prompt_payload),
            "tool_schema_hash": _stable_hash(tool_schemas or []),
            "trajectory_path": str(artifact_dir / "trajectory.json"),
            "score_path": str(artifact_dir / "score.json"),
            "metadata": metadata or {},
        }
        _atomic_json(artifact_dir / "trajectory.json", trajectory.model_dump())
        _atomic_json(artifact_dir / "score.json", score)
        _atomic_json(
            artifact_dir / "metadata.json",
            {
                **record,
                "prompt": prompt_payload,
                "tool_schemas": tool_schemas or [],
                "rollout_log_probs": rollout_log_probs,
                "training_log_probs": training_log_probs,
            },
        )
        with (self.run_dir / "scores.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
