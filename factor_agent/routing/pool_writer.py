from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def write_pools(classifications: list[dict[str, Any]], out_dir: str | Path) -> dict[str, int]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for classification in classifications:
        pools[classification["route"]].append(classification)
    counts: dict[str, int] = {}
    for route, rows in pools.items():
        path = root / f"{route}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        counts[route] = len(rows)
    (root / "summary.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return counts
