"""
Reading-order extraction for SOP uploads.

This keeps the existing import API stable while adding the Cybrain-style
``elements`` payload used for better PDF/OCR rendering.
"""
from __future__ import annotations

import logging
import re
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
EXTRACTION_ENGINE_SEQUENTIAL = "sequential"
_RAPIDOCR_ENGINE: Any | None = None


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


def _cluster_ocr_words_to_rows(words: List[Dict[str, Any]], y_tolerance: float = 8.0) -> List[List[Dict[str, Any]]]:
    rows: List[List[Dict[str, Any]]] = []
    for word in sorted(words, key=lambda w: (float(w.get("top", 0)), float(w.get("x0", 0)))):
        placed = False
        word_mid = (float(word.get("top", 0)) + float(word.get("bottom", word.get("top", 0)))) / 2
        for row in rows:
            row_mid = sum(
                (float(w.get("top", 0)) + float(w.get("bottom", w.get("top", 0)))) / 2 for w in row
            ) / max(len(row), 1)
            if abs(word_mid - row_mid) <= y_tolerance:
                row.append(word)
                placed = True
                break
        if not placed:
            rows.append([word])
    return [sorted(row, key=lambda w: float(w.get("x0", 0))) for row in rows]


def _derive_column_anchors(candidate_rows: List[List[Dict[str, Any]]]) -> List[float]:
    starts: List[float] = []
    for row in candidate_rows:
        if len(row) >= 2:
            starts.extend(float(w.get("x0", 0)) for w in row[:12])
    anchors: List[float] = []
    for start in sorted(starts):
        if not anchors or abs(start - anchors[-1]) > 28:
            anchors.append(start)
        else:
            anchors[-1] = (anchors[-1] + start) / 2
    return anchors[:12]


def _row_to_cells(row: List[Dict[str, Any]], anchors: List[float]) -> List[str]:
    if len(anchors) < 2:
        return [" ".join(str(w.get("text", "")).strip() for w in row if str(w.get("text", "")).strip())]
    cells = [[] for _ in anchors]
    for word in row:
        x0 = float(word.get("x0", 0))
        idx = min(range(len(anchors)), key=lambda i: abs(x0 - anchors[i]))
        cells[idx].append(str(word.get("text", "")).strip())
    return [_clean_line(" ".join(parts)) for parts in cells]


def _positioned_words_to_elements(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from ..utils.table_blocks import infer_header_row_count, normalize_table_rows

    rows = _cluster_ocr_words_to_rows(words)
    if not rows:
        return []

    elements: List[Dict[str, Any]] = []
    line_buffer: List[str] = []
    table_rows: List[List[Dict[str, Any]]] = []

    def flush_lines() -> None:
        if not line_buffer:
            return
        elements.extend(_text_lines_to_elements(line_buffer))
        line_buffer.clear()

    def flush_table() -> None:
        if len(table_rows) < 2:
            for row in table_rows:
                line = _clean_line(" ".join(str(w.get("text", "")) for w in row))
                if line:
                    line_buffer.append(line)
            table_rows.clear()
            return
        anchors = _derive_column_anchors(table_rows)
        grid = normalize_table_rows([_row_to_cells(row, anchors) for row in table_rows])
        nonempty_cols = [
            idx
            for idx in range(len(grid[0]) if grid else 0)
            if sum(1 for row in grid if idx < len(row) and row[idx]) >= 2
        ]
        if len(grid) >= 2 and len(nonempty_cols) >= 2:
            flush_lines()
            trimmed = [[row[idx] if idx < len(row) else "" for idx in nonempty_cols] for row in grid]
            elements.append({
                "type": "table",
                "content": trimmed,
                "header_rows": infer_header_row_count(trimmed),
                "source": "ocr_position_table",
            })
        else:
            for row in table_rows:
                line = _clean_line(" ".join(str(w.get("text", "")) for w in row))
                if line:
                    line_buffer.append(line)
        table_rows.clear()

    for row in rows:
        line = _clean_line(" ".join(str(w.get("text", "")) for w in row))
        if not line:
            continue
        # Treat rows with multiple separated word anchors as possible table rows.
        wide_gaps = 0
        for left, right in zip(row, row[1:]):
            gap = float(right.get("x0", 0)) - float(left.get("x1", left.get("x0", 0)))
            if gap >= 18:
                wide_gaps += 1
        looks_tabular = len(row) >= 2 and (wide_gaps >= 2 or any(ch in line for ch in ["|", "\t"]))
        if looks_tabular:
            table_rows.append(row)
            continue
        flush_table()
        line_buffer.append(line)

    flush_table()
    flush_lines()
    return elements


def _table_signature(element: Dict[str, Any]) -> str:
    if not isinstance(element, dict) or element.get("type") != "table":
        return ""
    rows = element.get("content") or element.get("rows") or []
    cells: List[str] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, list):
            continue
        for cell in row:
            text = re.sub(r"[^a-z0-9]+", "", str(cell or "").lower())
            if text:
                cells.append(text)
    return "|".join(cells[:80])


def _merge_missing_table_elements(base: List[Dict[str, Any]], supplemental: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not supplemental:
        return base
    merged = list(base)
    seen = {_table_signature(el) for el in merged if el.get("type") == "table"}
    for el in supplemental:
        if not isinstance(el, dict) or el.get("type") != "table":
            continue
        sig = _table_signature(el)
        if not sig or sig in seen:
            continue
        seen.add(sig)
        merged.append(el)
    return merged


def _extract_docling_table_elements(file_bytes: bytes) -> List[Dict[str, Any]]:
    try:
        from .docling_extractor import extract_scanned_pdf_docling_elements, is_docling_scanned_enabled

        if not is_docling_scanned_enabled():
            return []
        elements = extract_scanned_pdf_docling_elements(file_bytes)
        return [el for el in elements if isinstance(el, dict) and el.get("type") == "table"]
    except Exception as exc:
        logger.warning("[sequential-import] Docling image/table supplement failed: %s", exc)
        return []


def _get_rapidocr_engine() -> Any | None:
    global _RAPIDOCR_ENGINE
    if _RAPIDOCR_ENGINE is not None:
        return _RAPIDOCR_ENGINE
    try:
        from rapidocr import EngineType, RapidOCR

        _RAPIDOCR_ENGINE = RapidOCR(params={
            "Det.engine_type": EngineType.TORCH,
            "Cls.engine_type": EngineType.TORCH,
            "Rec.engine_type": EngineType.TORCH,
        })
        return _RAPIDOCR_ENGINE
    except Exception as exc:
        logger.warning("[sequential-import] RapidOCR torch engine unavailable: %s", exc)
        return None


def _rapidocr_image_to_elements(image_bytes: bytes) -> List[Dict[str, Any]]:
    engine = _get_rapidocr_engine()
    if engine is None:
        return []
    try:
        result = engine(image_bytes)
    except Exception as exc:
        logger.warning("[sequential-import] RapidOCR image OCR failed: %s", exc)
        return []

    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    boxes = [] if boxes is None else boxes
    texts = [] if texts is None else texts
    scores = [] if scores is None else scores
    words: List[Dict[str, Any]] = []
    for idx, text in enumerate(texts):
        clean = _clean_line(str(text or ""))
        if not clean:
            continue
        try:
            score = float(scores[idx])
        except Exception:
            score = 1.0
        if score < 0.35:
            continue
        try:
            pts = boxes[idx]
            xs = [float(pt[0]) for pt in pts]
            ys = [float(pt[1]) for pt in pts]
        except Exception:
            continue
        words.append({
            "x0": min(xs),
            "top": min(ys),
            "x1": max(xs),
            "bottom": max(ys),
            "text": clean,
        })
    return _positioned_words_to_elements(words)


def _extract_embedded_image_elements(file_bytes: bytes) -> List[Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []
    try:
        import fitz  # type: ignore

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        seen_xrefs: set[int] = set()
        for page in doc:
            for image_info in page.get_images(full=True):
                if not image_info:
                    continue
                xref = int(image_info[0])
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    extracted = doc.extract_image(xref)
                    image_bytes = extracted.get("image")
                except Exception:
                    image_bytes = None
                if not image_bytes:
                    continue
                image_elements = _rapidocr_image_to_elements(image_bytes)
                if image_elements:
                    elements.extend(image_elements)
        doc.close()
    except Exception as exc:
        logger.warning("[sequential-import] embedded image OCR extraction failed: %s", exc)
    return elements


def _extract_fitz_ocr_position_elements(file_bytes: bytes) -> List[Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []
    try:
        import fitz  # type: ignore

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            words_raw = []
            try:
                textpage = page.get_textpage_ocr(flags=fitz.TEXT_PRESERVE_WHITESPACE, full=True)
                words_raw = page.get_text("words", textpage=textpage, sort=True) or []
            except Exception:
                words_raw = page.get_text("words", sort=True) or []

            words: List[Dict[str, Any]] = []
            for item in words_raw:
                if len(item) < 5:
                    continue
                text = _clean_line(str(item[4]))
                if text:
                    words.append({
                        "x0": float(item[0]),
                        "top": float(item[1]),
                        "x1": float(item[2]),
                        "bottom": float(item[3]),
                        "text": text,
                    })
            page_elements = _positioned_words_to_elements(words)
            if page_elements:
                elements.extend(page_elements)
        doc.close()
    except Exception as exc:
        logger.warning("[sequential-import] fitz OCR position table reconstruction failed: %s", exc)
    return elements


def _extract_scanned_pdf_elements(file_bytes: bytes) -> List[Dict[str, Any]]:
    docling_elements: List[Dict[str, Any]] = []
    try:
        from .docling_extractor import extract_scanned_pdf_docling_elements, is_docling_scanned_enabled

        if is_docling_scanned_enabled():
            docling_elements = extract_scanned_pdf_docling_elements(file_bytes)
            if any(el.get("type") == "table" for el in docling_elements):
                return docling_elements
    except Exception as exc:
        logger.warning("[sequential-import] Docling scanned extraction unavailable; falling back: %s", exc)

    positioned_elements = _extract_fitz_ocr_position_elements(file_bytes)
    if any(el.get("type") == "table" for el in positioned_elements):
        return positioned_elements
    if docling_elements:
        return docling_elements

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
                    if row_words:
                        events.append((float(bucket_top), "line", row_words))
                for table in tables:
                    events.append((float(table.bbox[1]), "table", table))
                events.sort(key=lambda e: e[0])

                positioned_segment: List[Dict[str, Any]] = []

                def flush_positioned_segment() -> None:
                    if not positioned_segment:
                        return
                    segment_elements = _positioned_words_to_elements(positioned_segment)
                    if segment_elements:
                        elements.extend(segment_elements)
                    else:
                        lines = []
                        for row in _cluster_ocr_words_to_rows(positioned_segment, y_tolerance=line_y_tolerance):
                            line = _clean_line(" ".join(str(w.get("text", "")) for w in row))
                            if line:
                                lines.append(line)
                        elements.extend(_text_lines_to_elements(lines))
                    positioned_segment.clear()

                for top, kind, obj in events:
                    if kind == "table":
                        flush_positioned_segment()
                        rows = _table_rows(obj)
                        if rows:
                            from ..utils.table_blocks import infer_header_row_count

                            elements.append({
                                "type": "table",
                                "content": rows,
                                "header_rows": infer_header_row_count(rows),
                            })
                    else:
                        positioned_segment.extend(obj)
                flush_positioned_segment()
    except Exception as exc:
        logger.exception("[sequential-import] native PDF extraction failed: %s", exc)

    # Native PDFs can still contain image-only tables or screenshots. pdfplumber
    # will treat the file as native and skip OCR, so supplement only missing
    # table elements from Docling/position OCR without replacing normal text.
    table_count = sum(1 for el in elements if el.get("type") == "table")
    embedded_image_tables = [el for el in _extract_embedded_image_elements(file_bytes) if el.get("type") == "table"]
    elements = _merge_missing_table_elements(elements, embedded_image_tables)
    if sum(1 for el in elements if el.get("type") == "table") == table_count:
        position_tables = [el for el in _extract_fitz_ocr_position_elements(file_bytes) if el.get("type") == "table"]
        elements = _merge_missing_table_elements(elements, position_tables)
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
        from .docx_extractor import extract_docx_elements

        elements = extract_docx_elements(raw)
        blocks = elements_to_blocks(elements)
        from .document_structure import refine_blocks

        text = sanitize_extracted_text(elements_to_plain_text(elements))
        blocks = refine_blocks(blocks, text)
        return elements, blocks, strip_invalid_control_chars(text), False

    if name.endswith((".txt", ".md", ".csv", ".json")):
        text = strip_invalid_control_chars(raw.decode("utf-8", errors="replace"))
        elements = _text_lines_to_elements(text.splitlines())
        blocks = elements_to_blocks(elements)
        from .document_structure import refine_blocks

        return elements, refine_blocks(blocks, text), text, False

    raise ValueError("Unsupported file type")
