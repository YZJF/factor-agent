from __future__ import annotations

import hashlib
from pathlib import Path

from factor_agent.documents.models import DocumentChunk, ReportSnapshot
from factor_agent.documents.normalize import normalize_chinese_numbers
from factor_agent.schemas import DocumentMeta


def _read_pdf(path: Path) -> tuple[str, list[tuple[int, str]]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional PDF input
        raise RuntimeError("PDF ingestion requires `pip install pypdf`") from exc
    pages = [(index + 1, page.extract_text() or "") for index, page in enumerate(PdfReader(path).pages)]
    return "\n\n".join(text for _, text in pages), pages


def _make_chunks(text: str, *, chunk_chars: int, overlap: int, page: int | None = None) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunks.append(
            DocumentChunk(
                chunk_id=f"p{page or 0}-c{len(chunks):04d}",
                text=text[start:end],
                page_start=page,
                page_end=page,
                char_start=start,
                char_end=end,
            )
        )
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def ingest_document(
    path: str | Path,
    *,
    doc_id: str | None = None,
    title: str = "",
    institution: str = "",
    series: str = "",
    chunk_chars: int = 1800,
    overlap: int = 200,
) -> ReportSnapshot:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        raw_text, pages = _read_pdf(source)
        normalized_pages = [(page, normalize_chinese_numbers(text)) for page, text in pages]
        text = "\n\n".join(value for _, value in normalized_pages)
        chunks = [
            chunk
            for page, value in normalized_pages
            for chunk in _make_chunks(value, chunk_chars=chunk_chars, overlap=overlap, page=page)
        ]
        source_format = "pdf"
    else:
        raw_text = source.read_text(encoding="utf-8")
        text = normalize_chinese_numbers(raw_text)
        chunks = _make_chunks(text, chunk_chars=chunk_chars, overlap=overlap)
        source_format = "markdown" if suffix in {".md", ".markdown"} else "text"
    digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    meta = DocumentMeta(
        doc_id=doc_id or digest[:16],
        source_path=str(source),
        source_format=source_format,
        title=title,
        institution=institution,
        series=series,
        content_hash=digest,
    )
    return ReportSnapshot(meta=meta, text=text, chunks=chunks)
