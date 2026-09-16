from __future__ import annotations

from pydantic import BaseModel, Field

from factor_agent.schemas import DocumentMeta


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    char_start: int
    char_end: int


class ReportSnapshot(BaseModel):
    meta: DocumentMeta
    text: str
    chunks: list[DocumentChunk] = Field(default_factory=list)
    version: str = "report_snapshot_v1"
