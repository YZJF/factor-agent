"""Optional verl AgentLoop binding for stage-B factor evolution."""

from __future__ import annotations

import json
import os
import re
from typing import Any
from uuid import uuid4

from factor_agent.backtest.client import BacktestClient, BacktestConfig
from factor_agent.evolve.reward import score_evolution
from factor_agent.evolve.run import _candidate, evolve_tool_schemas
from factor_agent.evolve.periods import DateWindows
from factor_agent.schemas import FactorSpec
from factor_agent.train.verl_agent_loop_adapter import FactorAgentLoopBuffer

try:  # pragma: no cover - exercised in the external verl environment
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
    from verl.workers.rollout.replica import TokenOutput

    VERL_AVAILABLE = True
except ImportError:  # local unit tests intentionally do not depend on verl
    VERL_AVAILABLE = False

    class AgentLoopBase:  # type: ignore[no-redef]
        pass

    def register(name):  # type: ignore[no-redef]
        return lambda cls: cls


_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_text_tool_calls(text: str) -> tuple[list[dict[str, Any]], str | None]:
    matches = _TOOL_CALL.findall(text)
    calls: list[dict[str, Any]] = []
    for raw in matches:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            return [], str(exc)
        function = payload.get("function", payload)
        calls.append(
            {
                "id": payload.get("id") or uuid4().hex,
                "name": function.get("name"),
                "arguments": function.get("arguments") or function.get("parameters") or {},
            }
        )
    return calls, None


@register("factor_evolve_agent")
class FactorEvolveAgentLoop(AgentLoopBase):
    """Generate model tokens in verl; execute factor tools between turns."""

    def __init__(self, *args, **kwargs):
        if not VERL_AVAILABLE:
            raise RuntimeError("FactorEvolveAgentLoop requires an installed verl")
        super().__init__(*args, **kwargs)
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns or 12
        self.response_length = self.rollout_config.response_length
        self.windows = DateWindows()
        self.client = BacktestClient(
            BacktestConfig(endpoint=os.environ.get("FACTOR_BACKTEST_URL", "http://localhost:8001/backtest")),
            namespace=os.environ.get("VERL_RUN_ID", "verl"),
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> "AgentLoopOutput":
        messages = [dict(message) for message in kwargs["raw_prompt"]]
        extra = dict(kwargs.get("extra_info") or {})
        seed_payload = extra.get("seed") or extra.get("gold")
        if seed_payload is None and extra.get("gold_path"):
            with open(extra["gold_path"], encoding="utf-8") as handle:
                seed_payload = json.load(handle)["gold"]
        seed = FactorSpec.model_validate(seed_payload)
        prompt_ids = await self.apply_chat_template(messages, tools=evolve_tool_schemas())
        buffer = FactorAgentLoopBuffer(prompt_ids=list(prompt_ids))
        baseline = self.client.evaluate(
            seed,
            case_id=extra.get("case_id", seed.name),
            period="score",
            start=self.windows.score_start,
            end=self.windows.score_end,
        )
        candidates = []
        score_results = []
        actions = []
        final_text = ""
        request_id = uuid4().hex
        for step in range(self.max_assistant_turns):
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=list(prompt_ids) + buffer.response_ids,
                sampling_params=sampling_params,
            )
            generated = list(output.token_ids)
            buffer.append_model(generated, list(output.log_probs) if output.log_probs else None, step=step)
            text = self.tokenizer.decode(generated, skip_special_tokens=False)
            calls, parse_error = parse_text_tool_calls(text)
            messages.append({"role": "assistant", "content": text})
            if parse_error:
                observation = {"ok": False, "error": parse_error, "source": "model"}
            elif not calls:
                final_text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                break
            else:
                observation = {}
                for call in calls:
                    actions.append({"step": step, **call})
                    candidate = _candidate(call["arguments"], seed)
                    if candidate is None:
                        observation = {"ok": False, "error": "invalid FactorSpec", "source": "model"}
                    elif call["name"] == "validate_expr":
                        from factor_agent.spec.expr import expr_ok

                        ok, message = expr_ok(candidate.expr)
                        observation = {"ok": ok, "message": message, "source": "model"}
                    elif call["name"] == "evaluate_factor":
                        visible = self.client.evaluate(
                            candidate,
                            case_id=extra.get("case_id", seed.name),
                            period="search",
                            start=self.windows.search_start,
                            end=self.windows.search_end,
                        )
                        hidden = self.client.evaluate(
                            candidate,
                            case_id=extra.get("case_id", seed.name),
                            period="score",
                            start=self.windows.score_start,
                            end=self.windows.score_end,
                        )
                        candidates.append(candidate)
                        score_results.append(hidden)
                        observation = visible.model_dump(exclude={"run_id"})
                    else:
                        observation = {"ok": False, "error": f"unknown tool {call['name']}", "source": "model"}
            tool_message = {
                "role": "tool",
                "tool_call_id": calls[-1]["id"] if calls else f"error-{step}",
                "name": calls[-1]["name"] if calls else "tool_error",
                "content": json.dumps(observation, ensure_ascii=False),
            }
            messages.append(tool_message)
            tool_ids = await self.apply_chat_template([tool_message], remove_system_prompt=True)
            remaining = self.response_length - len(buffer.response_ids)
            buffer.append_environment(list(tool_ids)[:remaining], segment_type="tool_observation", step=step)
            if len(buffer.response_ids) >= self.response_length:
                break
        reward = score_evolution(
            seed,
            candidates,
            score_results,
            baseline_metric=baseline.metric if baseline.ok else None,
            parsed_actions=actions,
        )
        fields = buffer.as_verl_fields(max_response_length=self.response_length)
        return AgentLoopOutput(
            prompt_ids=fields["prompt_ids"],
            response_ids=fields["response_ids"],
            response_mask=fields["response_mask"],
            response_logprobs=fields["response_logprobs"],
            reward_score=reward.reward,
            num_turns=len(buffer.segments) + 1,
            metrics={"reward": reward.reward, "raw_reward": reward.raw_reward},
            extra_fields={
                "reward_extra_info": {
                    "active_caps_json": json.dumps(reward.active_caps),
                    "subscores_json": reward.subscores.model_dump_json(),
                },
                "token_trace": fields["token_trace"],
                "final_text": final_text,
            },
        )
