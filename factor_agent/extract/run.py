"""Stage A: report -> FactorSpec. Provider is injected; no training here."""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

from factor_agent.documents.tools import ReportToolRuntime
from factor_agent.parse import parse_spec
from factor_agent.prompts import EXTRACT_SYSTEM, extract_user
from factor_agent.protocol import audit, classify_call, mark_multi_tool_steps, tool_names
from factor_agent.schemas import FactorCase, FactorSpec
from factor_agent.score.verifier import score_prediction
from factor_agent.trajectory import Step, Trajectory


def run_extract(
    case: FactorCase,
    complete: Callable[[list[dict[str, str]]], str],
) -> tuple[FactorSpec | None, Trajectory]:
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": extract_user(case.report)},
    ]
    text = complete(messages)
    pred = parse_spec(text)
    reward = score_prediction(case, text, pred=pred)
    traj = Trajectory(
        case_id=case.case_id,
        stage="extract",
        steps=[Step(role="assistant", content=text)],
        reward=reward.reward,
    )
    return pred, traj


def run_extract_with_tools(
    case: FactorCase,
    complete: Callable[..., Any],
    runtime: ReportToolRuntime,
    *,
    max_steps: int | None = None,
) -> tuple[FactorSpec | None, Trajectory]:
    """Run a permissive long-report loop and retain every model/tool action."""

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": extract_user(case.report)},
    ]
    steps = [
        Step(role="system", content=EXTRACT_SYSTEM),
        Step(role="user", content=messages[-1]["content"]),
    ]
    actions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    final_text = ""
    limit = max_steps or case.max_steps
    allowed = tool_names(runtime.schemas())
    for step_index in range(limit):
        try:
            raw = complete(messages, tools=runtime.schemas())
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
        for call_index, tool_call in enumerate(tool_calls):
            function = tool_call.get("function", tool_call)
            name = str(function.get("name", ""))
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            call_id = str(tool_call.get("id") or f"extract-{step_index}-{call_index}")
            action = {"step": step_index, "tool_call_id": call_id, "name": name, "arguments": arguments}
            result = runtime.execute(name, arguments)
            illegal_kind = classify_call(
                name=name, arguments=arguments, result=result, allowed=allowed
            )
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
            actions.append(action)
            observations.append(observation)
            steps.append(
                Step(
                    role="tool",
                    content=json.dumps(result, ensure_ascii=False),
                    tool_name=name,
                    tool_ok=observation["ok"],
                    tool_call_id=call_id,
                    tool_arguments=arguments,
                    source=observation["source"],
                    response_mask=0,
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
    pred = parse_spec(final_text)
    reward = score_prediction(case, final_text, pred=pred)
    mark_multi_tool_steps(observations)
    trajectory = Trajectory(
        case_id=case.case_id,
        stage="extract",
        steps=steps,
        reward=reward.reward,
        parsed_actions=actions,
        tool_observations=observations,
        final_text=final_text,
        extras={
            "max_steps": limit,
            "max_step_hit": not bool(final_text),
            "tool_audit": audit(observations),
        },
    )
    return pred, trajectory
