"""Stage B: multi-turn mutation with visible search feedback and hidden score reward."""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

from factor_agent.backtest.client import BacktestClient
from factor_agent.evolve.drift import spec_drift
from factor_agent.evolve.reward import score_evolution
from factor_agent.evolve.periods import DateWindows, DEFAULT_WINDOWS
from factor_agent.prompts import EVOLVE_SYSTEM
from factor_agent.protocol import (
    FROZEN_FIELD_MUTATION,
    audit,
    classify_call,
    mark_multi_tool_steps,
    tool_names,
)
from factor_agent.schemas import BacktestResult, FactorSpec
from factor_agent.spec.expr import expr_ok
from factor_agent.trajectory import Step, Trajectory


def evaluate_factor(
    spec: FactorSpec,
    backtest: Callable[..., dict[str, Any]],
    windows: DateWindows = DEFAULT_WINDOWS,
    *,
    period: str = "score",
) -> dict[str, Any]:
    start, end = getattr(windows, period)()
    try:
        return backtest(spec, start=start, end=end)
    except TypeError:
        return backtest(spec)


def _as_backtest_result(raw: dict[str, Any], period: str) -> BacktestResult:
    if "ok" in raw:
        return BacktestResult.model_validate({**raw, "period": period})
    metrics = (raw.get("data") or {}).get("metrics") or raw.get("metrics") or raw
    value = metrics.get("Information_Ratio_with_cost")
    if value is None:
        value = metrics.get("IR")
    return BacktestResult(
        ok=value is not None,
        metric=float(value) if value is not None else None,
        period=period,
        metrics={key: float(item) for key, item in metrics.items() if isinstance(item, (int, float))},
        error="" if value is not None else "Information_Ratio_with_cost missing",
        error_source=None if value is not None else "environment",
    )


def evaluate_spec(
    backend: Callable[..., dict[str, Any]] | BacktestClient,
    spec: FactorSpec,
    *,
    case_id: str,
    period: str,
    windows: DateWindows,
) -> BacktestResult:
    start, end = getattr(windows, period)()
    if isinstance(backend, BacktestClient):
        return backend.evaluate(spec, case_id=case_id, period=period, start=start, end=end)
    try:
        raw = backend(spec, start=start, end=end)
    except Exception as exc:
        return BacktestResult(ok=False, period=period, error=str(exc), error_source="environment")
    return _as_backtest_result(raw, period)


def _robustness_score(
    backend: Callable[..., dict[str, Any]] | BacktestClient,
    spec: FactorSpec,
    *,
    case_id: str,
    windows: DateWindows,
    reference: BacktestResult,
) -> tuple[float, dict[str, Any]]:
    if not reference.ok or reference.metric is None:
        return 0.5, {}
    start, end = windows.score()
    if isinstance(backend, BacktestClient):
        lag = backend.evaluate(
            spec, case_id=case_id, period="score", start=start, end=end, shift_bars=1
        )
        shuffled = backend.evaluate(
            spec, case_id=case_id, period="score", start=start, end=end, shuffle=True
        )
    else:
        try:
            lag = _as_backtest_result(backend(spec, start=start, end=end, shift_bars=1), "score")
            shuffled = _as_backtest_result(backend(spec, start=start, end=end, shuffle=True), "score")
        except TypeError:
            return 0.5, {"status": "backend_does_not_support_contrasts"}
    scale = max(abs(reference.metric), 0.1)
    lag_score = (
        max(0.0, 1.0 - abs(reference.metric - lag.metric) / scale)
        if lag.ok and lag.metric is not None
        else 0.0
    )
    shuffle_score = (
        max(0.0, 1.0 - abs(shuffled.metric) / scale)
        if shuffled.ok and shuffled.metric is not None
        else 0.0
    )
    return (lag_score + shuffle_score) / 2.0, {
        "lag_one": lag.model_dump(),
        "shuffle_cross_section": shuffled.model_dump(),
    }


def evolve_tool_schemas() -> list[dict[str, Any]]:
    spec_schema = FactorSpec.model_json_schema()
    return [
        {
            "type": "function",
            "function": {
                "name": "validate_expr",
                "description": "Validate a complete mutated FactorSpec before backtesting.",
                "parameters": {
                    "type": "object",
                    "properties": {"spec": spec_schema},
                    "required": ["spec"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "evaluate_factor",
                "description": "Backtest one complete FactorSpec on the visible search window.",
                "parameters": {
                    "type": "object",
                    "properties": {"spec": spec_schema},
                    "required": ["spec"],
                },
            },
        },
    ]


def _candidate(payload: dict[str, Any], seed: FactorSpec) -> FactorSpec | None:
    candidate_payload = payload.get("spec", payload)
    if "factor_expr" in candidate_payload:
        candidate_payload = {
            **seed.model_dump(),
            "name": candidate_payload.get("factor_name", seed.name),
            "expr": candidate_payload["factor_expr"],
        }
    try:
        return FactorSpec.model_validate(candidate_payload)
    except Exception:
        return None


def run_evolve(
    seed: FactorSpec,
    complete: Callable[..., Any],
    backtest: Callable[..., dict[str, Any]] | BacktestClient,
    windows: DateWindows = DEFAULT_WINDOWS,
    *,
    max_rounds: int = 4,
) -> Trajectory:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": EVOLVE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Seed spec:\n{seed.model_dump_json()}\n"
                f"Search window: {windows.search()}\n"
                f"Score window (reward): {windows.score()}\n"
                "Propose one mutation, then we will backtest."
            ),
        },
    ]
    steps = [Step(role="system", content=EVOLVE_SYSTEM), Step(role="user", content=messages[1]["content"])]
    actions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    candidates: list[FactorSpec] = []
    score_results: list[BacktestResult] = []
    baseline = evaluate_spec(backtest, seed, case_id=seed.name, period="score", windows=windows)
    final_text = ""
    evaluations = 0
    action_budget = max_rounds * 3
    allowed = tool_names(evolve_tool_schemas())
    for step_index in range(action_budget):
        try:
            raw = complete(messages, tools=evolve_tool_schemas())
        except TypeError:
            raw = complete(messages)
        if isinstance(raw, str):
            content, tool_calls = raw, []
        else:
            content = str(raw.get("content") or "")
            tool_calls = list(raw.get("tool_calls") or [])
        steps.append(Step(role="assistant", content=content))
        if not tool_calls:
            final_text = content
            break
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        for call_index, call in enumerate(tool_calls):
            function = call.get("function", call)
            name = str(function.get("name", ""))
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            call_id = str(call.get("id") or f"evolve-{step_index}-{call_index}")
            candidate = _candidate(arguments, seed)
            action = {"step": step_index, "tool_call_id": call_id, "name": name, "arguments": arguments}
            actions.append(action)
            if candidate is None:
                result: dict[str, Any] = {"ok": False, "error": "invalid FactorSpec", "source": "model"}
            elif name == "validate_expr":
                ok, message = expr_ok(candidate.expr)
                result = {"ok": ok, "message": message, "source": "model"}
            elif name == "evaluate_factor":
                if evaluations >= max_rounds:
                    result = {"ok": False, "error": "max evaluation rounds reached", "source": "model"}
                else:
                    visible = evaluate_spec(backtest, candidate, case_id=seed.name, period="search", windows=windows)
                    hidden = evaluate_spec(backtest, candidate, case_id=seed.name, period="score", windows=windows)
                    candidates.append(candidate)
                    score_results.append(hidden)
                    evaluations += 1
                    result = visible.model_dump(exclude={"run_id"})
                    result["source"] = visible.error_source or "model"
            else:
                result = {"ok": False, "error": f"unknown tool: {name}", "source": "model"}
            illegal_kind = classify_call(
                name=name, arguments=arguments, result=result, allowed=allowed
            )
            if illegal_kind is None and candidate is not None and spec_drift(seed, candidate):
                illegal_kind = FROZEN_FIELD_MUTATION
            observation = {
                "step": step_index,
                "tool_call_id": call_id,
                "name": name,
                "ok": bool(result.get("ok")),
                "source": result.get("source", "model"),
                "illegal": illegal_kind is not None,
                "illegal_kind": illegal_kind,
                "content": result,
            }
            observations.append(observation)
            encoded = json.dumps(result, ensure_ascii=False)
            steps.append(
                Step(
                    role="tool",
                    content=encoded,
                    tool_name=name,
                    tool_ok=observation["ok"],
                    tool_call_id=call_id,
                    tool_arguments=arguments,
                    source=observation["source"],
                    response_mask=0,
                )
            )
            messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": encoded})
        if evaluations >= max_rounds:
            break
    robustness = 0.5
    contrast_details: dict[str, Any] = {}
    successful_indexes = [
        index for index, result in enumerate(score_results) if result.ok and result.metric is not None
    ]
    if successful_indexes:
        best_index = max(successful_indexes, key=lambda index: score_results[index].metric or float("-inf"))
        robustness, contrast_details = _robustness_score(
            backtest,
            candidates[best_index],
            case_id=seed.name,
            windows=windows,
            reference=score_results[best_index],
        )
    evolve_reward = score_evolution(
        seed,
        candidates,
        score_results,
        baseline_metric=baseline.metric if baseline.ok else None,
        parsed_actions=actions,
        robustness=robustness,
    )
    mark_multi_tool_steps(observations)
    return Trajectory(
        case_id=seed.name,
        stage="evolve",
        steps=steps,
        reward=evolve_reward.reward,
        extras={
            "seed": seed.model_dump(),
            "windows": windows.model_dump(),
            "baseline_score": baseline.model_dump(),
            "hidden_score_results": [result.model_dump() for result in score_results],
            "contrast_results": contrast_details,
            "evolve_reward": evolve_reward.model_dump(),
            "max_rounds": max_rounds,
            "max_step_hit": evaluations >= max_rounds and not final_text,
            "tool_audit": audit(observations),
        },
        parsed_actions=actions,
        tool_observations=observations,
        final_text=final_text,
    )
