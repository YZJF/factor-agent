from factor_agent.inject import INJECT_KINDS, corrupt
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


def test_each_inject_kind_scores_below_gold():
    case = _case()
    gold_r = score_prediction(case, case.gold.model_dump_json())
    assert gold_r.reward >= 0.85
    for kind in INJECT_KINDS:
        pred = corrupt(case.gold, kind)
        r = score_prediction(case, pred.model_dump_json(), pred=pred)
        assert r.reward < gold_r.reward, kind


def test_hallucinated_is_capped():
    case = _case()
    pred = corrupt(case.gold, "hallucinated_expr")
    r = score_prediction(case, pred.model_dump_json(), pred=pred)
    assert "hallucinated_expr_cap" in r.active_caps
    assert r.reward <= 0.35
