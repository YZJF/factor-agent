"""Turn gold cases into SFT rows: report -> gold JSON."""

from __future__ import annotations

import json
from pathlib import Path

from factor_agent.documents.tools import ReportToolRuntime
from factor_agent.evolve.run import evolve_tool_schemas
from factor_agent.gate import passed_extract_gate
from factor_agent.io import dump_dicts
from factor_agent.prompts import EVOLVE_SYSTEM, EXTRACT_SYSTEM, extract_user
from factor_agent.schemas import FactorCase
from factor_agent.score.verifier import score_prediction
from factor_agent.split import assert_no_leak, group_key
from factor_agent.spec.expr import expr_ok


def to_sft_row(case: FactorCase) -> dict:
    gold_json = case.gold.model_dump_json()
    reward = score_prediction(case, gold_json, pred=case.gold)
    if not passed_extract_gate(reward, case.gold):
        raise ValueError(f"gold replay failed for {case.case_id}: {reward.model_dump()}")
    messages: list[dict] = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": extract_user(case.report)},
    ]
    message_loss_mask = [0, 0]
    if case.gold.evidence:
        citation = case.gold.evidence[0]
        call_id = f"{case.case_id}-read"
        if citation.page_start is not None:
            function = {
                "name": "read_report_pages",
                "arguments": json.dumps(
                    {"page_start": citation.page_start, "page_end": citation.page_end or citation.page_start}
                ),
            }
        else:
            function = {
                "name": "search_report",
                "arguments": json.dumps({"query": citation.quote[:32], "limit": 5}, ensure_ascii=False),
            }
        messages.extend(
            [
                {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": function}]},
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": function["name"],
                    "content": json.dumps({"ok": True, "evidence": [item.model_dump() for item in case.gold.evidence]}, ensure_ascii=False),
                },
            ]
        )
        message_loss_mask.extend([1, 0])
    validate_id = f"{case.case_id}-validate"
    messages.extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": validate_id,
                        "type": "function",
                        "function": {
                            "name": "validate_factor_spec",
                            "arguments": json.dumps({"spec": case.gold.model_dump()}, ensure_ascii=False),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": validate_id,
                "name": "validate_factor_spec",
                "content": '{"ok": true, "expr_ok": true, "citation_errors": []}',
            },
            {"role": "assistant", "content": gold_json},
        ]
    )
    message_loss_mask.extend([1, 0, 1])
    return {
        "data_source": "factor_extract",
        "prompt": messages[:2],
        "response": gold_json,
        "messages": messages,
        "tools": ReportToolRuntime.schemas(),
        "enable_thinking": False,
        "message_loss_mask": message_loss_mask,
        "extra_info": {
            "case_id": case.case_id,
            "group_id": group_key(case),
            "split": case.split,
            "stage": "extract",
            "adapter": "extract_lora",
        },
    }


def to_evolve_sft_row(case: FactorCase) -> dict:
    """Protocol warmup for the evolve adapter. There is no gold mutation.

    Replay: validate the seed, then emit the same spec. This teaches frozen
    fields and one-tool-per-turn without inventing a fake IR improvement.
    GRPO is where real backtest reward starts.
    """

    gold_json = case.gold.model_dump_json()
    reward = score_prediction(case, gold_json, pred=case.gold)
    if not passed_extract_gate(reward, case.gold):
        raise ValueError(f"evolve seed failed extract gate for {case.case_id}")
    ok, message = expr_ok(case.gold.expr)
    if not ok:
        raise ValueError(f"evolve seed expression is illegal for {case.case_id}: {message}")
    spec_payload = json.dumps({"spec": case.gold.model_dump()}, ensure_ascii=False)
    validate_id = f"{case.case_id}-validate-expr"
    messages = [
        {"role": "system", "content": EVOLVE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Seed spec:\n{gold_json}\n"
                "Keep universe, frequency, and neutralization. "
                "Validate the expression, then return the FactorSpec JSON."
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": validate_id,
                    "type": "function",
                    "function": {"name": "validate_expr", "arguments": spec_payload},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": validate_id,
            "name": "validate_expr",
            "content": json.dumps({"ok": True, "message": "ok", "source": "model"}),
        },
        {"role": "assistant", "content": gold_json},
    ]
    return {
        "data_source": "factor_evolve",
        "prompt": messages[:2],
        "response": gold_json,
        "messages": messages,
        "tools": evolve_tool_schemas(),
        "enable_thinking": False,
        "message_loss_mask": [0, 0, 1, 0, 1],
        "extra_info": {
            "case_id": case.case_id,
            "group_id": group_key(case),
            "split": case.split,
            "stage": "evolve",
            "seed": case.gold.model_dump(),
            "adapter": "evolve_lora",
        },
    }


def build_sft(cases: list[FactorCase], out_path: str | Path, *, stage: str = "extract") -> int:
    assert_no_leak(cases)
    if stage == "extract":
        rows = [to_sft_row(c) for c in cases if c.split == "train"]
    elif stage == "evolve":
        rows = [to_evolve_sft_row(c) for c in cases if c.split == "train"]
    else:
        raise ValueError(f"unknown SFT stage: {stage}")
    dump_dicts(out_path, rows)
    return len(rows)
