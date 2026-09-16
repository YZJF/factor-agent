"""Optional rLLM adapters. Import rllm only when you train."""

from __future__ import annotations

from factor_agent.io import load_jsonl
from factor_agent.schemas import FactorCase
from factor_agent.score.verifier import score_prediction


def score_episode(task: dict, solution_str: str) -> float:
    case = FactorCase.model_validate(task["case"])
    return score_prediction(case, solution_str).reward


def load_cases(path: str) -> list[dict]:
    return [{"case": c.model_dump()} for c in load_jsonl(path)]
