"""Research-report ingestion, indexing, and read-only extraction tools."""

from factor_agent.documents.ingest import ingest_document
from factor_agent.documents.models import DocumentChunk, ReportSnapshot
from factor_agent.documents.tools import ReportToolRuntime

__all__ = ["DocumentChunk", "ReportSnapshot", "ReportToolRuntime", "ingest_document"]
