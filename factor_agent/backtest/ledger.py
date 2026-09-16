from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class LedgerEntry(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    namespace: str
    case_id: str
    expression_hash: str
    period: str
    config_hash: str
    request: dict[str, Any]
    result: dict[str, Any]
    created_at: float = Field(default_factory=time)


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class BacktestLedger:
    """Append-only ledger; entries are never updated in place."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.entries: list[LedgerEntry] = []
        if self.path and self.path.exists():
            self.entries = [
                LedgerEntry.model_validate_json(line)
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def append(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(entry.model_dump_json() + "\n")

    def cached(self, *, namespace: str, expression_hash: str, period: str, config_hash: str) -> LedgerEntry | None:
        return next(
            (
                entry
                for entry in reversed(self.entries)
                if entry.namespace == namespace
                and entry.expression_hash == expression_hash
                and entry.period == period
                and entry.config_hash == config_hash
                and entry.result.get("ok")
            ),
            None,
        )
