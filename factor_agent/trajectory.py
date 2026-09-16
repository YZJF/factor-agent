from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Step(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_name: str | None = None
    tool_ok: bool | None = None
    tool_call_id: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    source: Literal["model", "environment"] | None = None
    response_mask: int = 1


class Trajectory(BaseModel):
    case_id: str
    stage: Literal["extract", "evolve"]
    steps: list[Step] = Field(default_factory=list)
    reward: float | None = None
    extras: dict[str, Any] = Field(default_factory=dict)
    parsed_actions: list[dict[str, Any]] = Field(default_factory=list)
    tool_observations: list[dict[str, Any]] = Field(default_factory=list)
    final_text: str = ""
