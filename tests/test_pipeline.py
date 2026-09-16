from factor_agent.pipeline import run_case
from factor_agent.providers.echo import echo_complete
from factor_agent.schemas import FactorCase, FactorSpec


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
    report = "CSI500 20-day reversal, industry-neutral, weekly rebalance. Expression INDUSTRY_NEUTRALIZE(-TS_PCTCHANGE($close, 20), $industry)."
    return FactorCase(case_id="demo", report=report, gold=gold)


def test_echo_pipeline_passes_gate():
    case = _case()
    out = run_case(case, echo_complete(case), extract_reward_min=0.7)
    assert out["passed_gate"] is True
    assert out["pred"]["universe"] == "CSI500"
    assert out["evolve"] is None
