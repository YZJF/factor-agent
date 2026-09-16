from pathlib import Path

import pytest

from factor_agent.batch.writer import write_batch
from factor_agent.documents.ingest import ingest_document
from factor_agent.documents.tools import ReportToolRuntime
from factor_agent.extract.run import run_extract_with_tools
from factor_agent.schemas import DocumentMeta, EvidenceSpan, FactorCase, FactorSpec
from factor_agent.split import assert_no_leak
from factor_agent.train.sft_builder import to_sft_row


QUOTE = "二十(20)日反转"  # CJK fixture: Chinese-numeral normalization must keep grounding.


def _spec(chunk_id: str = "p0-c0000") -> FactorSpec:
    return FactorSpec(
        name="reversal_20d",
        universe="CSI500",
        frequency="daily",
        neutralization="industry",
        rebalance="weekly",
        expr="INDUSTRY_NEUTRALIZE(-TS_PCTCHANGE($close, 20), $industry)",
        windows=[20],
        evidence=[
            EvidenceSpan(chunk_id=chunk_id, quote=QUOTE, char_start=5, char_end=5 + len(QUOTE))
        ],
    )


def test_markdown_ingestion_and_tools(tmp_path: Path):
    report = tmp_path / "report.md"
    report.write_text("中证500二十日反转，行业中性。", encoding="utf-8")
    snapshot = ingest_document(report, doc_id="doc-1")
    assert "二十(20)日" in snapshot.text
    runtime = ReportToolRuntime(snapshot)
    result = runtime.search_report("反转")
    assert result["matches"]
    spec = _spec(result["matches"][0]["chunk_id"])
    assert runtime.validate_factor_spec(spec.model_dump())["ok"] is True


def test_group_level_leak_is_rejected():
    document = DocumentMeta(doc_id="same-report")
    train = FactorCase(case_id="a", group_id="same", document=document, report="x", gold=_spec(), split="train")
    holdout = FactorCase(case_id="b", group_id="same", document=document, report="y", gold=_spec(), split="holdout")
    with pytest.raises(ValueError, match="group leak"):
        assert_no_leak([train, holdout])


def test_four_file_batch_manifest(tmp_path: Path):
    report = tmp_path / "report.md"
    report.write_text("中证500二十日反转，行业中性。", encoding="utf-8")
    snapshot = ingest_document(report, doc_id="doc-1")
    case = FactorCase(
        case_id="case-1",
        group_id="doc-1",
        document=snapshot.meta,
        report=snapshot.text,
        gold=_spec(snapshot.chunks[0].chunk_id),
    )
    manifest = write_batch([case], {"doc-1": snapshot}, tmp_path / "batch", batch_id="batch-1")
    assert len(manifest.entries) == 1
    entry = manifest.entries[0]
    assert Path(entry.gold_path).exists()
    assert Path(entry.verifier_spec_path).exists()


def test_tool_extract_loop_and_sft_masks(tmp_path: Path):
    report = tmp_path / "report.md"
    report.write_text("中证500二十日反转，行业中性。", encoding="utf-8")
    snapshot = ingest_document(report, doc_id="doc-1")
    spec = _spec(snapshot.chunks[0].chunk_id)
    case = FactorCase(case_id="case-1", document=snapshot.meta, report=snapshot.text, gold=spec)
    replies = iter(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search-1",
                        "type": "function",
                        "function": {"name": "search_report", "arguments": '{"query":"反转"}'},
                    }
                ],
            },
            {"content": spec.model_dump_json(), "tool_calls": []},
        ]
    )

    def complete(messages, tools=None):
        assert tools
        return next(replies)

    pred, trajectory = run_extract_with_tools(case, complete, ReportToolRuntime(snapshot))
    assert pred is not None
    assert trajectory.tool_observations[0]["ok"] is True
    assert any(step.role == "tool" and step.response_mask == 0 for step in trajectory.steps)
    sft = to_sft_row(case)
    assert len(sft["messages"]) == len(sft["message_loss_mask"])
    assert sft["message_loss_mask"][-2:] == [0, 1]
    assert sft["tools"]
    assert sft["enable_thinking"] is False
