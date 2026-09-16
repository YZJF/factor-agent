from factor_agent.inject import corrupt
from factor_agent.parse import parse_spec
from factor_agent.schemas import FactorCase, FactorSpec
from factor_agent.verifier import score_prediction


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


def test_gold_scores_high():
    case = _case()
    r = score_prediction(case, case.gold.model_dump_json())
    assert r.reward >= 0.85
    assert not r.active_caps


def test_bad_json_is_zero():
    r = score_prediction(_case(), "not json")
    assert r.reward == 0.0
    assert "invalid_json_cap" in r.active_caps


def test_illegal_expr_capped():
    case = _case()
    bad = corrupt(case.gold, "illegal_expr")
    r = score_prediction(case, bad.model_dump_json())
    assert r.reward <= 0.15


def test_parse_fence():
    spec = parse_spec("```json\n" + _case().gold.model_dump_json() + "\n```")
    assert spec is not None
    assert spec.universe == "CSI500"
