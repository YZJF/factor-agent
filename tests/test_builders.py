from factor_agent.schemas import FactorCase, FactorSpec
from factor_agent.train.grpo_builder import to_evolve_grpo_row, to_grpo_row
from factor_agent.train.lora import ADAPTERS, grpo_overrides
from factor_agent.train.sft_builder import to_evolve_sft_row, to_sft_row
from factor_agent.verifier import compute_score


def _case() -> FactorCase:
    gold = FactorSpec(
        name="reversal_20d",
        universe="CSI500",
        frequency="daily",
        neutralization="industry",
        rebalance="weekly",
        expr="INDUSTRY_NEUTRALIZE(-TS_PCTCHANGE($close, 20), $industry)",
        windows=[20],
    )
    return FactorCase(
        case_id="demo",
        report="CSI500 20-day reversal, industry-neutral. INDUSTRY_NEUTRALIZE(-TS_PCTCHANGE($close, 20), $industry)",
        gold=gold,
    )


def test_sft_row_has_prompt_and_response():
    row = to_sft_row(_case())
    assert row["prompt"][0]["role"] == "system"
    assert "CSI500" in row["response"]


def test_grpo_row_ground_truth_scores_high():
    case = _case()
    row = to_grpo_row(case)
    score = compute_score(
        row["data_source"],
        case.gold.model_dump_json(),
        row["reward_model"]["ground_truth"],
    )
    assert score >= 0.85
    assert row["extra_info"]["adapter"] == "extract_lora"


def test_evolve_sft_is_protocol_warmup_not_a_fake_ir():
    row = to_evolve_sft_row(_case())
    assert row["data_source"] == "factor_evolve"
    assert row["extra_info"]["adapter"] == "evolve_lora"
    assert row["message_loss_mask"] == [0, 0, 1, 0, 1]
    names = [
        call["function"]["name"]
        for message in row["messages"]
        if message.get("tool_calls")
        for call in message["tool_calls"]
    ]
    assert names == ["validate_expr"]
    assert "evaluate_factor" not in names
    assert row["response"] == _case().gold.model_dump_json()


def test_evolve_grpo_does_not_reuse_extract_adapter_name():
    row = to_evolve_grpo_row(_case())
    assert row["extra_info"]["adapter"] == "evolve_lora"
    assert row["extra_info"]["stage"] == "evolve"


def test_lora_checkpoints_are_disjoint():
    assert ADAPTERS["extract"]["grpo_dir"] != ADAPTERS["evolve"]["grpo_dir"]
    flags = grpo_overrides(adapter_path="checkpoints/evolve_lora/sft")
    assert any("evolve_lora/sft" in item for item in flags)
