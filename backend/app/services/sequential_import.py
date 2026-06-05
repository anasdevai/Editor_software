"""
Reading-order extraction for SOP uploads.

This keeps the existing import API stable while adding the Cybrain-style
``elements`` payload used for better PDF/OCR rendering.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Tuple

import pdfplumber

from .pdf_extractor import (
    _clean_line,
    _is_key_value_line,
    _is_likely_heading,
    _pdf_is_scanned,
    _run_ocr_lines_on_page,
    sanitize_extracted_text,
)
from .sop_metadata_extractor import strip_invalid_control_chars

logger = logging.getLogger(__name__)


def elements_to_plain_text(elements: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for el in elements or []:
        if not isinstance(el, dict):
            continue
        if el.get("type") == "text":
            text = str(el.get("content") or "").strip()
            if text:
                parts.append(text)
        elif el.get("type") == "table":
            for row in el.get("content") or []:
                cells = [str(c).strip() for c in row or [] if str(c).strip()]
                if cells:
                    parts.append(" | ".join(cells))
    return "\n\n".join(parts).strip()


def elements_to_blocks(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from ..utils.table_blocks import table_block_from_rows

    blocks: List[Dict[str, Any]] = []
    for el in elements or []:
        if not isinstance(el, dict):
            continue
        etype = str(el.get("type") or "").lower()
        if etype == "text":
            text = _clean_line(str(el.get("content") or ""))
            if not text:
                continue
            style = str(el.get("style") or "paragraph").lower()
            if style == "heading" or _is_likely_heading(text):
                blocks.append({"type": "section_heading", "text": text, "level": 2})
            elif _is_key_value_line(text):
                key, _, value = text.partition(":")
                blocks.append({"type": "two_column_row", "left": _clean_line(key), "right": _clean_line(value)})
            else:
                blocks.append({"type": "paragraph", "text": text})
        elif etype == "table":
            rows = el.get("content") or []
            if isinstance(rows, list) and rows:
                table = table_block_from_rows(rows, header_rows=el.get("header_rows"))
                if table:
                    blocks.append(table)
    return blocks


def _table_rows(table) -> List[List[str]]:
    from ..utils.table_blocks import normalize_table_rows

    try:
        extracted = table.extract() or []
    except Exception:
        extracted = []
    rows: List[List[str]] = []
    for row in extracted:
        normalized = [_clean_line(cell or "") for cell in row or []]
        if any(normalized):
            rows.append(normalized)
    return normalize_table_rows(rows)


def _text_lines_to_elements(lines: List[str]) -> List[Dict[str, Any]]:
    from ..utils.table_blocks import table_block_from_paragraph_text

    elements: List[Dict[str, Any]] = []
    table = table_block_from_paragraph_text("\n".join(lines))
    if table:
        return [{"type": "table", "content": table["rows"], "header_rows": table.get("header_rows", 0)}]
    for raw in lines:
        line = _clean_line(raw)
        if not line:
            continue
        style = "heading" if _is_likely_heading(line) else "paragraph"
        elements.append({"type": "text", "style": style, "content": line})
    return elements


def _extract_scanned_pdf_elements(file_bytes: bytes) -> List[Dict[str, Any]]:
    try:
        from .docling_extractor import extract_scanned_pdf_docling_elements, is_docling_scanned_enabled

        if is_docling_scanned_enabled():
            elements = extract_scanned_pdf_docling_elements(file_bytes)
            if elements:
                return elements
    except Exception as exc:
        logger.warning("[sequential-import] Docling scanned extraction unavailable; falling back: %s", exc)

    elements: List[Dict[str, Any]] = []
    try:
        import fitz  # type: ignore

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            raw_text = ""
            try:
                tp = page.get_textpage_ocr(flags=fitz.TEXT_PRESERVE_WHITESPACE, full=True)
                raw_text = tp.extractTEXT()
            except Exception:
                raw_text = page.get_text("text") or ""
            elements.extend(_text_lines_to_elements(raw_text.splitlines()))
        doc.close()
    except Exception as exc:
        logger.warning("[sequential-import] fitz OCR failed; falling back to pytesseract: %s", exc)

    if elements:
        return elements

    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page_num, _page in enumerate(pdf.pages, start=1):
                elements.extend(_text_lines_to_elements(_run_ocr_lines_on_page(file_bytes, page_num)))
    except Exception as exc:
        logger.exception("[sequential-import] scanned PDF OCR fallback failed: %s", exc)
    return elements


def _extract_native_pdf_elements(file_bytes: bytes) -> List[Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []
    line_y_tolerance = 3.0
    para_gap_threshold = 14.0

    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                tables = list(page.find_tables() or [])
                tables.sort(key=lambda t: float(t.bbox[1]))
                table_bboxes = [(float(t.bbox[0]), float(t.bbox[1]), float(t.bbox[2]), float(t.bbox[3])) for t in tables]

                def word_in_table(word: Dict[str, Any]) -> bool:
                    wx0 = float(word["x0"])
                    wtop = float(word["top"])
                    wx1 = float(word["x1"])
                    wbot = float(word["bottom"])
                    return any(
                        wx0 >= tx0 - 2 and wx1 <= tx1 + 2 and wtop >= ttop - 2 and wbot <= tbot + 2
                        for tx0, ttop, tx1, tbot in table_bboxes
                    )

                words = page.extract_words(x_tolerance=3, y_tolerance=3) or []
                buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
                for word in words:
                    if word_in_table(word):
                        continue
                    bucket = int(round(float(word["top"]) / line_y_tolerance) * line_y_tolerance)
                    buckets[bucket].append(word)

                events: List[Tuple[float, str, Any]] = []
                for bucket_top in sorted(buckets.keys()):
                    row_words = sorted(buckets[bucket_top], key=lambda w: float(w["x0"]))
                    line = " ".join(w["text"] for w in row_words).strip()
                    if line:
                        events.append((float(bucket_top), "line", line))
                for table in tables:
                    events.append((float(table.bbox[1]), "table", table))
                events.sort(key=lambda e: e[0])

                buffer: List[str] = []
                last_line_top: float | None = None

                def flush_para() -> None:
                    if not buffer:
                        return
                    text = _clean_line(" ".join(buffer))
                    if text:
                        style = "heading" if _is_likely_heading(text) else "paragraph"
                        elements.append({"type": "text", "style": style, "content": text})
                    buffer.clear()

                for top, kind, obj in events:
                    if kind == "table":
                        flush_para()
                        last_line_top = None
                        rows = _table_rows(obj)
                        if rows:
                            from ..utils.table_blocks import infer_header_row_count

                            elements.append({
                                "type": "table",
                                "content": rows,
                                "header_rows": infer_header_row_count(rows),
                            })
                    else:
                        if last_line_top is not None and top - last_line_top > para_gap_threshold:
                            flush_para()
                        buffer.append(str(obj))
                        last_line_top = top
                flush_para()
    except Exception as exc:
        logger.exception("[sequential-import] native PDF extraction failed: %s", exc)
    return elements


def extract_sequential_upload(raw: bytes, filename: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, bool]:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        scanned = _pdf_is_scanned(raw)
        elements = _extract_scanned_pdf_elements(raw) if scanned else _extract_native_pdf_elements(raw)
        blocks = elements_to_blocks(elements)
        from .document_structure import refine_blocks

        text = sanitize_extracted_text(elements_to_plain_text(elements))
        return elements, refine_blocks(blocks, text), strip_invalid_control_chars(text), scanned

    if name.endswith(".docx"):
        from .pdf_extractor import extract_docx_bytes

        blocks, text = extract_docx_bytes(raw)
        elements = [
            {"type": "text", "style": "heading" if _is_likely_heading(str(b.get("text") or "")) else "paragraph", "content": str(b.get("text") or "")}
            for b in blocks
            if isinstance(b, dict) and str(b.get("text") or "").strip()
        ]
        return elements, blocks, strip_invalid_control_chars(text), False

    if name.endswith((".txt", ".md", ".csv", ".json")):
        text = strip_invalid_control_chars(raw.decode("utf-8", errors="replace"))
        elements = _text_lines_to_elements(text.splitlines())
        blocks = elements_to_blocks(elements)
        from .document_structure import refine_blocks

        return elements, refine_blocks(blocks, text), text, False

    raise ValueError("Unsupported file type")
