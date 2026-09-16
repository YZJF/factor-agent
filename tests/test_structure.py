import pytest

from factor_agent.evolve.drift import spec_drift
from factor_agent.evolve.periods import DateWindows
from factor_agent.gate import passed_extract_gate
from factor_agent.schemas import FactorCase, FactorSpec, RewardResult, Subscores
from factor_agent.spec.consistency import declaration_issues
from factor_agent.spec.expr import ALLOWED_FUNCS, ALLOWED_VARS, AS_OF_BARS
from factor_agent.split import assert_no_leak, leak_ids


def _spec(**kwargs) -> FactorSpec:
    base = dict(
        name="reversal_20d",
        universe="CSI500",
        frequency="daily",
        neutralization="industry",
        rebalance="weekly",
        expr="INDUSTRY_NEUTRALIZE(-TS_PCTCHANGE($close, 20), $industry)",
        windows=[20],
    )
    base.update(kwargs)
    return FactorSpec.model_validate(base)


def test_whitelist_counts():
    assert len(ALLOWED_FUNCS) == 14
    assert len(ALLOWED_VARS) == 9
    assert AS_OF_BARS == 1


def test_declaration_flags_missing_neutralize():
    issues = declaration_issues(_spec(expr="-TS_PCTCHANGE($close, 20)"))
    assert "neutralization_missing_in_expr" in issues


def test_declaration_clean_when_aligned():
    assert declaration_issues(_spec()) == []


def test_spec_drift_frozen_fields():
    seed = _spec()
    moved = _spec(universe="CSI300")
    assert spec_drift(seed, moved) == ["universe"]
    assert spec_drift(seed, seed) == []


def test_date_windows_do_not_overlap_eval():
    w = DateWindows()
    assert w.score_end < w.eval_start
    assert w.search_end < w.score_start


def test_gate_rejects_capped():
    pred = _spec()
    reward = RewardResult(
        case_id="x",
        reward=0.9,
        raw_reward=0.9,
        subscores=Subscores(),
        active_caps=["hallucinated_expr_cap"],
    )
    assert passed_extract_gate(reward, pred) is False


def test_leak_detects_same_id():
    gold = _spec()
    cases = [
        FactorCase(case_id="dup", report="x 20", gold=gold, split="train"),
        FactorCase(case_id="dup", report="x 20", gold=gold, split="holdout"),
    ]
    assert leak_ids(cases) == {"dup"}
    with pytest.raises(ValueError, match="leak"):
        assert_no_leak(cases)
