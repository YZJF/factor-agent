"""Search / score / eval time windows. Agent mutates on search, reward uses score."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DateWindows(BaseModel):
    search_start: str = "2018-01-01"
    search_end: str = "2021-12-31"
    score_start: str = "2022-01-01"
    score_end: str = "2023-12-31"
    eval_start: str = "2024-01-01"
    eval_end: str = "2024-12-31"
    mix_search: float = Field(default=0.3, ge=0.0, le=1.0)

    def search(self) -> tuple[str, str]:
        return self.search_start, self.search_end

    def score(self) -> tuple[str, str]:
        return self.score_start, self.score_end

    def eval(self) -> tuple[str, str]:
        return self.eval_start, self.eval_end


DEFAULT_WINDOWS = DateWindows()
