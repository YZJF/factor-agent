from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def summarize_run(
    records: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    classifications: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rewards = [float(record.get("reward", 0.0)) for record in records]
    caps: Counter[str] = Counter()
    for record in records:
        caps.update(record.get("active_caps") or [])
    routes = Counter(item["route"] for item in (classifications or []))
    illegal_kinds: Counter[str] = Counter()
    for record in records:
        illegal_kinds.update(record.get("illegal_by_kind") or {})
    tool_calls = sum(int(record.get("tool_calls", 0)) for record in records)
    illegal_calls = sum(int(record.get("illegal_tool_calls", 0)) for record in records)
    summary = {
        "rollout_count": len(records),
        "case_count": len({record.get("case_id") for record in records}),
        "mean_reward": mean(rewards) if rewards else 0.0,
        "reward_std": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "tool_calls": tool_calls,
        "illegal_tool_calls": illegal_calls,
        "illegal_tool_call_rate": illegal_calls / tool_calls if tool_calls else 0.0,
        "illegal_tool_call_kinds": dict(illegal_kinds),
        "hard_cap_distribution": dict(caps),
        "route_distribution": dict(routes),
    }
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["# Factor Agent Run Report", ""]
    for key, value in summary.items():
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict) else value
        lines.append(f"- `{key}`: {rendered}")
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def summarize_scores_file(run_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "scores.jsonl"
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return summarize_run(records, out_dir)
