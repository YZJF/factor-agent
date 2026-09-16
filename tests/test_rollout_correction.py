import pytest

from factor_agent.schemas import FactorCase, FactorSpec
from factor_agent.train.grpo_builder import to_evolve_grpo_row
from factor_agent.train.rollout_correction import (
    build_response_mask,
    correction_diagnostics,
    sequence_tis,
    validate_response_mask,
)
from factor_agent.train.verl_agent_loop_adapter import (
    FactorAgentLoopBuffer,
    rollout_correction_overrides,
)
from factor_agent.train.verl_factor_agent_loop import parse_text_tool_calls


def test_response_mask_excludes_tool_tokens():
    ids, mask = build_response_mask(
        [
            {"type": "model", "token_ids": [1, 2]},
            {"type": "tool_observation", "token_ids": [3, 4, 5]},
            {"type": "model", "token_ids": [6]},
        ]
    )
    assert ids == [1, 2, 3, 4, 5, 6]
    assert mask == [1, 1, 0, 0, 0, 1]


def test_sequence_tis_masks_tools_and_clips():
    result = sequence_tis(
        rollout_log_probs=[-1.0, 0.0, -1.0],
        training_log_probs=[0.0, 100.0, 0.0],
        response_mask=[1, 0, 1],
        threshold=2.0,
    )
    assert result["raw_ratio"] > 2.0
    assert result["clipped_ratio"] == 2.0
    assert result["was_clipped"] is True


def test_correction_is_recommended_only_for_measured_gap():
    diagnostics = correction_diagnostics(
        [
            {
                "rollout_log_probs": [-1.0, 0.0],
                "training_log_probs": [-0.9, 10.0],
                "response_mask": [1, 0],
            }
        ],
        enable_delta=0.05,
    )
    assert diagnostics["recommended"] is True


def test_loop_buffer_and_overrides():
    buffer = FactorAgentLoopBuffer(prompt_ids=[10])
    buffer.append_model([1, 2], [-0.2, -0.3], step=0)
    buffer.append_environment([3], segment_type="tool_observation", step=0)
    fields = buffer.as_verl_fields()
    assert fields["response_mask"] == [1, 1, 0]
    assert any("rollout_is=sequence" in item for item in rollout_correction_overrides(enabled=True))
    with pytest.raises(ValueError):
        validate_response_mask([1], [0])


def test_evolve_row_and_text_tool_parser():
    spec = FactorSpec(
        name="factor",
        universe="CSI500",
        frequency="daily",
        neutralization="none",
        rebalance="weekly",
        expr="-TS_PCTCHANGE($close, 20)",
        windows=[20],
    )
    row = to_evolve_grpo_row(FactorCase(case_id="case", report="report 20", gold=spec))
    assert row["extra_info"]["agent_name"] == "factor_evolve_agent"
    calls, error = parse_text_tool_calls(
        '<tool_call>{"name":"evaluate_factor","arguments":{"spec":{}}}</tool_call>'
    )
    assert error is None
    assert calls[0]["name"] == "evaluate_factor"
