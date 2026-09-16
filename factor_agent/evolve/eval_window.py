"""Frozen eval-window report: seed vs evolved factor, after cost.

Nothing in training ever touches this window. Stage B optimises on `search` and
is rewarded on the hidden `score` window; this module is the only place the
`eval` window is read, and it runs offline against finished checkpoints.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import median
from typing import Any

from pydantic import BaseModel, Field

from factor_agent.backtest.client import BacktestClient
from factor_agent.evolve.periods import DEFAULT_WINDOWS, DateWindows
from factor_agent.evolve.run import evaluate_spec
from factor_agent.schemas import FactorSpec

REPORTED_METRICS = (
    "Information_Ratio_with_cost",
    "annualized_return_with_cost",
    "turnover",
    "max_drawdown",
)


class EvalWindowRow(BaseModel):
    case_id: str
    seed_metric: float | None = None
    evolved_metric: float | None = None
    delta: float | None = None
    ok: bool = False
    seed_metrics: dict[str, float] = Field(default_factory=dict)
    evolved_metrics: dict[str, float] = Field(default_factory=dict)
    error: str = ""


class EvalWindowReport(BaseModel):
    window: tuple[str, str]
    n: int = 0
    n_scored: int = 0
    median_seed_metric: float | None = None
    median_evolved_metric: float | None = None
    median_delta: float | None = None
    improved_share: float = 0.0
    rows: list[EvalWindowRow] = Field(default_factory=list)


def evaluate_eval_window(
    pairs: list[tuple[str, FactorSpec, FactorSpec]],
    backend: Callable[..., dict[str, Any]] | BacktestClient,
    windows: DateWindows = DEFAULT_WINDOWS,
) -> EvalWindowReport:
    """Score (case_id, seed, evolved) triples on the frozen eval window."""

    rows: list[EvalWindowRow] = []
    for case_id, seed, evolved in pairs:
        seed_result = evaluate_spec(
            backend, seed, case_id=case_id, period="eval", windows=windows
        )
        evolved_result = evaluate_spec(
            backend, evolved, case_id=case_id, period="eval", windows=windows
        )
        ok = (
            seed_result.ok
            and evolved_result.ok
            and seed_result.metric is not None
            and evolved_result.metric is not None
        )
        rows.append(
            EvalWindowRow(
                case_id=case_id,
                seed_metric=seed_result.metric,
                evolved_metric=evolved_result.metric,
                delta=(evolved_result.metric - seed_result.metric) if ok else None,
                ok=ok,
                seed_metrics={
                    key: value
                    for key, value in seed_result.metrics.items()
                    if key in REPORTED_METRICS
                },
                evolved_metrics={
                    key: value
                    for key, value in evolved_result.metrics.items()
                    if key in REPORTED_METRICS
                },
                error="; ".join(filter(None, [seed_result.error, evolved_result.error])),
            )
        )
    scored = [row for row in rows if row.ok]
    return EvalWindowReport(
        window=windows.eval(),
        n=len(rows),
        n_scored=len(scored),
        median_seed_metric=median([row.seed_metric for row in scored]) if scored else None,
        median_evolved_metric=median([row.evolved_metric for row in scored]) if scored else None,
        median_delta=median([row.delta for row in scored]) if scored else None,
        improved_share=(
            sum(row.delta > 0 for row in scored) / len(scored) if scored else 0.0
        ),
        rows=rows,
    )
