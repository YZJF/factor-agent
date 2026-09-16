from __future__ import annotations

from typing import Any

from factor_agent.documents.models import ReportSnapshot
from factor_agent.schemas import FactorSpec
from factor_agent.spec.consistency import declaration_issues
from factor_agent.spec.expr import expr_ok


class ReportToolRuntime:
    """Read-only tools exposed to long-document extraction rollouts."""

    def __init__(self, snapshot: ReportSnapshot):
        self.snapshot = snapshot

    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_report",
                    "description": "Search normalized report chunks for a literal query.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_report_pages",
                    "description": "Read chunks overlapping a page range.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "page_start": {"type": "integer"},
                            "page_end": {"type": "integer"},
                        },
                        "required": ["page_start", "page_end"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_factor_spec",
                    "description": "Validate FactorSpec schema, expression whitelist, declarations, and citations.",
                    "parameters": {
                        "type": "object",
                        "properties": {"spec": {"type": "object"}},
                        "required": ["spec"],
                    },
                },
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search_report":
            return self.search_report(str(arguments.get("query", "")), int(arguments.get("limit", 5)))
        if name == "read_report_pages":
            return self.read_report_pages(int(arguments["page_start"]), int(arguments["page_end"]))
        if name == "validate_factor_spec":
            return self.validate_factor_spec(arguments.get("spec", {}))
        return {"ok": False, "error": f"unknown tool: {name}", "source": "model"}

    def search_report(self, query: str, limit: int = 5) -> dict[str, Any]:
        needle = query.strip().lower()
        matches = [
            chunk.model_dump()
            for chunk in self.snapshot.chunks
            if needle and needle in chunk.text.lower()
        ][: max(1, min(limit, 20))]
        return {"ok": True, "query": query, "matches": matches}

    def read_report_pages(self, page_start: int, page_end: int) -> dict[str, Any]:
        chunks = [
            chunk.model_dump()
            for chunk in self.snapshot.chunks
            if chunk.page_start is not None and page_start <= chunk.page_start <= page_end
        ]
        return {"ok": True, "page_start": page_start, "page_end": page_end, "chunks": chunks}

    def validate_factor_spec(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            spec = FactorSpec.model_validate(payload)
        except Exception as exc:
            return {"ok": False, "error": f"invalid FactorSpec: {exc}", "source": "model"}
        expression_ok, expression_message = expr_ok(spec.expr)
        known_chunks = {chunk.chunk_id: chunk for chunk in self.snapshot.chunks}
        citation_errors = [
            evidence.chunk_id
            for evidence in spec.evidence
            if evidence.chunk_id not in known_chunks
            or evidence.quote not in known_chunks[evidence.chunk_id].text
        ]
        issues = declaration_issues(spec)
        return {
            "ok": expression_ok and not citation_errors,
            "expr_ok": expression_ok,
            "expr_message": expression_message,
            "declaration_issues": issues,
            "citation_errors": citation_errors,
        }
