from factor_agent.documents.ingest import ingest_document
from factor_agent.documents.tools import ReportToolRuntime
from factor_agent.eval.metrics import compare, summarize, summarize_by_group
from factor_agent.extract.run import run_extract_with_tools
from factor_agent.protocol import (
    MULTI_TOOL_PER_STEP,
    SCHEMA_INVALID,
    UNKNOWN_TOOL,
    audit,
    classify_call,
)
from factor_agent.schemas import DocumentMeta, EvidenceSpan, FactorCase, FactorSpec
from factor_agent.score.verifier import score_prediction
from factor_agent.spec.requirements import missing_requirements
from factor_agent.split import group_key

QUOTE = "过去二十(20)个交易日"  # CJK fixture: Chinese numerals must ground windows=[20].
ALLOWED = {"search_report", "validate_factor_spec"}


def _spec(*, evidence: bool = True) -> FactorSpec:
    return FactorSpec(
        name="reversal_20d",
        universe="CSI500",
        frequency="daily",
        neutralization="industry",
        rebalance="weekly",
        expr="INDUSTRY_NEUTRALIZE(-TS_PCTCHANGE($close, 20), $industry)",
        windows=[20],
        evidence=(
            [EvidenceSpan(chunk_id="p1-c0000", quote=QUOTE, page_start=1, char_start=3)]
            if evidence
            else []
        ),
    )


def _case(spec: FactorSpec) -> FactorCase:
    return FactorCase(
        case_id="demo",
        report=f"中证500内{QUOTE}反转，行业中性，周度调仓。",
        gold=spec,
    )


def test_chinese_numerals_ground_the_window():
    spec = _spec()
    result = score_prediction(_case(spec), spec.model_dump_json())
    assert result.subscores.grounding == 1.0
    assert not missing_requirements(spec)


def test_missing_citation_zeroes_grounding_without_a_cap():
    spec = _spec(evidence=False)
    result = score_prediction(_case(_spec()), spec.model_dump_json())
    assert result.subscores.grounding == 0.0
    assert "invalid_citation_cap" not in result.active_caps
    assert missing_requirements(spec) == [
        "no evidence span",
        "parameters without a citation: [20]",
    ]


def test_parameter_outside_the_cited_span_is_not_grounded():
    spec = _spec()
    spec.windows = [20, 60]
    result = score_prediction(_case(_spec()), spec.model_dump_json())
    assert result.subscores.grounding == 0.5


def test_classify_call_separates_protocol_from_verdict():
    assert classify_call(name="nope", arguments={}, result={}, allowed=ALLOWED) == UNKNOWN_TOOL
    assert (
        classify_call(
            name="validate_factor_spec",
            arguments={"spec": {}},
            result={"ok": False, "error": "invalid FactorSpec: missing expr", "source": "model"},
            allowed=ALLOWED,
        )
        == SCHEMA_INVALID
    )
    # A well-formed call whose verdict is negative stays legal.
    assert (
        classify_call(
            name="validate_factor_spec",
            arguments={"spec": {}},
            result={"ok": False, "expr_ok": False, "source": "model"},
            allowed=ALLOWED,
        )
        is None
    )
    assert (
        classify_call(
            name="search_report",
            arguments={},
            result={"ok": False, "error": "qlib down", "source": "environment"},
            allowed=ALLOWED,
        )
        is None
    )


def test_extract_loop_audits_unknown_tools_and_parallel_calls(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("中证500过去二十个交易日反转，行业中性。", encoding="utf-8")
    snapshot = ingest_document(report, doc_id="doc-1")
    case = FactorCase(case_id="c1", document=snapshot.meta, report=snapshot.text, gold=_spec())
    replies = iter(
        [
            {
                "content": "",
                "tool_calls": [
                    {"id": "a", "function": {"name": "grep_report", "arguments": "{}"}},
                    {"id": "b", "function": {"name": "search_report", "arguments": '{"query":"反转"}'}},
                ],
            },
            {"content": _spec().model_dump_json(), "tool_calls": []},
        ]
    )

    _, trajectory = run_extract_with_tools(
        case, lambda messages, tools=None: next(replies), ReportToolRuntime(snapshot)
    )
    kinds = [item["illegal_kind"] for item in trajectory.tool_observations]
    assert kinds == [UNKNOWN_TOOL, MULTI_TOOL_PER_STEP]
    assert trajectory.extras["tool_audit"]["illegal_tool_call_rate"] == 1.0


def test_audit_counts_environment_failures_separately():
    summary = audit(
        [
            {"step": 0, "ok": False, "source": "environment", "illegal_kind": None},
            {"step": 1, "ok": True, "source": "model", "illegal_kind": None},
        ]
    )
    assert summary == {
        "tool_calls": 2,
        "illegal_tool_calls": 0,
        "illegal_tool_call_rate": 0.0,
        "illegal_by_kind": {},
        "environment_failures": 1,
    }


def test_headline_metrics_and_percentage_point_compare():
    good = score_prediction(_case(_spec()), _spec().model_dump_json())
    bad = score_prediction(_case(_spec()), "not json")
    before = summarize([good, bad, bad, bad], tool_audits=[{"tool_calls": 10, "illegal_tool_calls": 2}])
    after = summarize([good, good, good, bad], tool_audits=[{"tool_calls": 10, "illegal_tool_calls": 1}])
    assert before["task_success_rate"] == 0.25
    assert after["task_success_rate"] == 0.75
    delta = compare(before, after)
    assert delta["task_success_rate"]["delta_pp"] == 50.0
    assert delta["illegal_tool_call_rate"]["delta_pp"] == -10.0


def test_group_key_falls_back_to_institution_and_series():
    document = DocumentMeta(doc_id="d1", institution="BrokerA", series="PriceVolumeWeekly")
    case = FactorCase(case_id="c1", report="x", gold=_spec(), document=document)
    assert group_key(case) == "BrokerA::PriceVolumeWeekly"
    case.group_id = "explicit"
    assert group_key(case) == "explicit"


def test_group_report_medians_do_not_follow_the_biggest_institution():
    good = score_prediction(_case(_spec()), _spec().model_dump_json())
    bad = score_prediction(_case(_spec()), "not json")
    results = [good, bad, bad, bad]
    groups = {"demo": "BrokerA::PriceVolumeWeekly"}
    report = summarize_by_group(results, groups)
    assert report["group_count"] == 1
    assert report["pooled"]["n"] == 4.0
