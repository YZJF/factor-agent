"""case -> extract -> rule score -> gate -> (optional) evolve.

Stage B IR does not flow back to stage A.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from factor_agent.evolve.periods import DateWindows, DEFAULT_WINDOWS
from factor_agent.evolve.run import run_evolve
from factor_agent.extract.run import run_extract
from factor_agent.gate import EXTRACT_REWARD_MIN, passed_extract_gate
from factor_agent.providers.base import CompleteFn
from factor_agent.schemas import FactorCase, FactorSpec, RewardResult
from factor_agent.score.verifier import score_prediction
from factor_agent.trajectory import Trajectory


def run_case(
    case: FactorCase,
    complete: CompleteFn,
    *,
    extract_reward_min: float = EXTRACT_REWARD_MIN,
    evolve: bool = False,
    backtest: Callable[..., dict[str, Any]] | None = None,
    windows: DateWindows = DEFAULT_WINDOWS,
) -> dict[str, Any]:
    pred, extract_traj = run_extract(case, complete)
    reward: RewardResult = score_prediction(case, extract_traj.steps[-1].content, pred=pred)
    passed = passed_extract_gate(reward, pred, extract_reward_min)
    evolve_traj: Trajectory | None = None
    if evolve and passed and pred is not None and backtest is not None:
        evolve_traj = run_evolve(pred, complete, backtest, windows)
    return {
        "case_id": case.case_id,
        "pred": pred.model_dump() if pred else None,
        "reward": reward.model_dump(),
        "passed_gate": passed,
        "extract": extract_traj.model_dump(),
        "evolve": evolve_traj.model_dump() if evolve_traj else None,
    }
