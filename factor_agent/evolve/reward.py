from __future__ import annotations

from collections import Counter

from factor_agent.evolve.drift import spec_drift
from factor_agent.schemas import (
    BacktestResult,
    EvolveRewardResult,
    EvolveSubscores,
    FactorSpec,
)
from factor_agent.spec.expr import expr_ok

WEIGHTS = {
    "validity": 0.15,
    "no_drift": 0.15,
    "protocol": 0.10,
    "backtest_success": 0.10,
    "performance": 0.35,
    "robustness": 0.10,
    "novelty": 0.05,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_evolution(
    seed: FactorSpec,
    candidates: list[FactorSpec],
    score_results: list[BacktestResult],
    *,
    baseline_metric: float | None,
    parsed_actions: list[dict],
    robustness: float = 0.5,
) -> EvolveRewardResult:
    caps: list[str] = []
    reasons: dict[str, str] = {}
    valid = [expr_ok(candidate.expr)[0] for candidate in candidates]
    drift = [field for candidate in candidates for field in spec_drift(seed, candidate)]
    step_counts = Counter(action.get("step") for action in parsed_actions)
    multi_tool_steps = sorted(str(step) for step, count in step_counts.items() if count > 1)
    if multi_tool_steps:
        caps.append("multi_tool_per_step_cap")
        reasons["multi_tool_per_step_cap"] = f"multiple tools in steps: {multi_tool_steps}"
    if valid and not all(valid):
        caps.append("illegal_expr_cap")
        reasons["illegal_expr_cap"] = "one or more candidate expressions are illegal"
    if drift:
        caps.append("spec_drift_cap")
        reasons["spec_drift_cap"] = f"frozen fields changed: {sorted(set(drift))}"
    successful = [result for result in score_results if result.ok and result.metric is not None]
    deltas: list[float] = []
    best = baseline_metric
    for result in score_results:
        if not result.ok or result.metric is None or best is None:
            deltas.append(0.0)
            continue
        delta = result.metric - best
        deltas.append(delta)
        best = max(best, result.metric)
    best_delta = max(deltas, default=0.0)
    subscores = EvolveSubscores(
        validity=sum(valid) / len(valid) if valid else 0.0,
        no_drift=0.0 if drift else 1.0,
        protocol=0.0 if multi_tool_steps else 1.0,
        backtest_success=len(successful) / len(score_results) if score_results else 0.0,
        performance=_clamp(0.5 + best_delta / 2.0),
        robustness=_clamp(robustness),
        novelty=(
            sum(candidate.expr.replace(" ", "") != seed.expr.replace(" ", "") for candidate in candidates)
            / len(candidates)
            if candidates
            else 0.0
        ),
    )
    raw = sum(getattr(subscores, key) * weight for key, weight in WEIGHTS.items())
    cap_values = {"multi_tool_per_step_cap": 0.0, "illegal_expr_cap": 0.15, "spec_drift_cap": 0.30}
    cap = min((cap_values[name] for name in caps), default=1.0)
    return EvolveRewardResult(
        reward=min(raw, cap),
        raw_reward=raw,
        subscores=subscores,
        score_deltas=deltas,
        active_caps=caps,
        cap_reasons=reasons,
        diagnostics={
            "weights": WEIGHTS,
            "baseline_metric": baseline_metric,
            "best_metric": best,
            "environment_failures": sum(
                not result.ok and result.error_source == "environment" for result in score_results
            ),
        },
    )
