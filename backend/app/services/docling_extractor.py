"""
Optional Docling layout/OCR extraction for scanned PDF tables.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from io import BytesIO
from typing import Any, Dict, List

from .pdf_extractor import _clean_line

logger = logging.getLogger(__name__)

EXTRACTION_ENGINE_DOCLING_SCANNED = "docling_scanned"
_CONVERTER_CACHE: Dict[str, Any] = {}
_DEFAULT_TIMEOUT_SEC = float(os.getenv("DOCLING_PDF_TIMEOUT_SEC", "45"))


def is_docling_scanned_enabled() -> bool:
    return os.getenv("SOP_DOCLING_SCANNED_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_docling_available() -> bool:
    try:
        import docling  # noqa: F401

        return True
    except Exception:
        return False


def _get_converter() -> Any:
    cached = _CONVERTER_CACHE.get("scanned")
    if cached is not None:
        return cached

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        force_backend_text=False,
        document_timeout=_DEFAULT_TIMEOUT_SEC if _DEFAULT_TIMEOUT_SEC > 0 else None,
        generate_page_images=True,
    )
    cached = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    _CONVERTER_CACHE["scanned"] = cached
    return cached


def _table_rows(table: Any) -> tuple[List[List[str]], int | None]:
    rows: List[List[str]] = []
    try:
        grid = table.data.grid
    except Exception:
        return [], None
    for row in grid:
        cells = [_clean_line(getattr(cell, "text", "") or "") for cell in row]
        if any(cells):
            rows.append(cells)
    header_rows = None
    try:
        if hasattr(table, "num_header_rows"):
            header_rows = int(table.num_header_rows)
    except Exception:
        header_rows = None
    return rows, header_rows


def extract_scanned_pdf_docling_elements(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    if not is_docling_scanned_enabled():
        raise RuntimeError("Docling scanned extraction is disabled.")
    if not is_docling_available():
        raise RuntimeError("Docling is not installed.")

    from docling.datamodel.base_models import ConversionStatus
    from docling_core.types.doc.document import ListItem, SectionHeaderItem, TableItem, TextItem, TitleItem
    from docling_core.types.doc.labels import DocItemLabel
    from docling_core.types.io import DocumentStream
    from ..utils.table_blocks import infer_header_row_count, normalize_table_rows

    converter = _get_converter()
    stream = DocumentStream(name="scanned_upload.pdf", stream=BytesIO(pdf_bytes))

    def run_convert() -> Any:
        return converter.convert(stream)

    if _DEFAULT_TIMEOUT_SEC > 0:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run_convert)
            try:
                result = future.result(timeout=_DEFAULT_TIMEOUT_SEC + 30)
            except FuturesTimeoutError as exc:
                raise TimeoutError(f"Docling scanned PDF conversion exceeded {_DEFAULT_TIMEOUT_SEC}s") from exc
    else:
        result = run_convert()

    status = getattr(result, "status", None)
    if status in {ConversionStatus.FAILURE}:
        raise RuntimeError(f"Docling scanned conversion failed with status={status}")

    doc = getattr(result, "document", None)
    if doc is None:
        raise RuntimeError("Docling returned no document for scanned PDF.")

    elements: List[Dict[str, Any]] = []
    list_buffer: List[str] = []

    def flush_list() -> None:
        if not list_buffer:
            return
        elements.append({"type": "text", "style": "paragraph", "content": "\n".join(list_buffer)})
        list_buffer.clear()

    for item, _level in doc.iterate_items():
        if isinstance(item, TableItem):
            flush_list()
            rows, explicit_headers = _table_rows(item)
            rows = normalize_table_rows(rows)
            if rows:
                elements.append({
                    "type": "table",
                    "content": rows,
                    "header_rows": infer_header_row_count(rows, explicit_headers),
                })
            continue

        if isinstance(item, ListItem):
            text = _clean_line(getattr(item, "text", "") or "")
            if text:
                list_buffer.append(text)
            continue

        flush_list()

        if isinstance(item, (TitleItem, SectionHeaderItem)):
            text = _clean_line(getattr(item, "text", "") or "")
            if text:
                elements.append({"type": "text", "style": "heading", "content": text})
            continue

        label = getattr(item, "label", None)
        if label in {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.PICTURE, DocItemLabel.FORMULA}:
            continue
        if isinstance(item, TextItem) or label in {
            DocItemLabel.PARAGRAPH,
            DocItemLabel.TEXT,
            DocItemLabel.CAPTION,
            DocItemLabel.FOOTNOTE,
            DocItemLabel.CODE,
        }:
            text = _clean_line(getattr(item, "text", "") or "")
            if text:
                elements.append({"type": "text", "style": "paragraph", "content": text})

    flush_list()
    if not elements:
        raise RuntimeError("Docling scanned conversion produced no elements.")
    logger.info("[docling] scanned PDF elements=%s tables=%s", len(elements), sum(1 for e in elements if e.get("type") == "table"))
    return elements
