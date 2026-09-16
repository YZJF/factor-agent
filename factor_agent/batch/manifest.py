from __future__ import annotations

from pydantic import BaseModel, Field


class ManifestEntry(BaseModel):
    case_id: str
    group_id: str
    split: str
    case_path: str
    report_snapshot_path: str
    verifier_spec_path: str
    gold_path: str
    content_hashes: dict[str, str] = Field(default_factory=dict)


class BatchManifest(BaseModel):
    batch_id: str
    entries: list[ManifestEntry] = Field(default_factory=list)
    version: str = "factor_batch_v1"
