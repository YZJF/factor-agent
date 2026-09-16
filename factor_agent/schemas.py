"""Versioned task schemas for document-grounded factor extraction."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

Frequency = Literal["daily", "weekly", "monthly"]
Neutralization = Literal["none", "industry", "size", "industry_size"]


class EvidenceSpan(BaseModel):
    """A stable citation into the normalized research report."""

    chunk_id: str
    quote: str
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None


class DocumentMeta(BaseModel):
    """Document identity used for grouping, provenance, and leak checks."""

    doc_id: str
    source_path: str | None = None
    source_format: Literal["text", "markdown", "pdf"] = "text"
    title: str = ""
    institution: str = ""
    series: str = ""
    published_at: date | None = None
    content_hash: str = ""


class FactorSpec(BaseModel):
    name: str
    universe: str
    frequency: Frequency
    neutralization: Neutralization
    rebalance: Frequency
    expr: str
    windows: list[int] = Field(default_factory=list)
    evidence: list[EvidenceSpan] = Field(default_factory=list)


class FactorCase(BaseModel):
    """One training/eval item. gold is hidden from the agent prompt."""

    case_id: str
    report: str
    gold: FactorSpec
    split: Literal["train", "holdout"] = "train"
    max_steps: int = 4
    version: str = "factor_case_v1"
    notes: str = ""
    group_id: str = ""
    document: DocumentMeta | None = None


class Subscores(BaseModel):
    schema_ok: float = 0.0
    executable: float = 0.0
    fields: float = 0.0
    grounding: float = 0.0


class RewardResult(BaseModel):
    case_id: str
    reward: float
    raw_reward: float
    subscores: Subscores
    active_caps: list[str] = Field(default_factory=list)
    cap_reasons: dict[str, str] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class BacktestResult(BaseModel):
    ok: bool
    metric: float | None = None
    metric_name: str = "Information_Ratio_with_cost"
    period: Literal["search", "score", "eval"] = "search"
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str = ""
    error_source: Literal["model", "environment"] | None = None
    cached: bool = False
    run_id: str = ""


class EvolveSubscores(BaseModel):
    validity: float = 0.0
    no_drift: float = 0.0
    protocol: float = 0.0
    backtest_success: float = 0.0
    performance: float = 0.0
    robustness: float = 0.0
    novelty: float = 0.0


class EvolveRewardResult(BaseModel):
    reward: float
    raw_reward: float
    subscores: EvolveSubscores
    score_deltas: list[float] = Field(default_factory=list)
    active_caps: list[str] = Field(default_factory=list)
    cap_reasons: dict[str, str] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


# Hard caps: hit one and the final score cannot exceed this.
ACTIVE_CAPS = {
    "invalid_json_cap": 0.0,
    "illegal_expr_cap": 0.15,
    "invalid_citation_cap": 0.20,
    "hallucinated_expr_cap": 0.35,
}

WEIGHTS = {
    "schema_ok": 0.25,
    "executable": 0.25,
    "fields": 0.40,
    "grounding": 0.10,
}
