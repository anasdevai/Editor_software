"""
Shared structure detection for SOP uploads (TXT, PDF text, DOCX paragraphs, OCR).
Produces typed blocks and a hierarchical structured_document JSON payload.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .pdf_extractor import (
    _clean_line,
    _is_bullet_item,
    _is_key_value_line,
    _is_likely_heading,
    _is_numbered_heading,
    _is_numbered_item,
    _to_heading_level,
    sanitize_extracted_text,
)


def _merge_split_sop_id_lines(lines: List[str]) -> List[str]:
    """OCR often splits 'SOP ID: SOP-XXX' across lines."""
    merged: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().upper() == "SOP" and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if re.match(r"(?i)^ID\s*:", nxt):
                merged.append(f"SOP {nxt}")
                i += 2
                continue
        merged.append(line)
        i += 1
    return merged


def structure_lines_to_blocks(lines: List[str]) -> List[Dict[str, Any]]:
    """Convert normalized lines into typed extraction blocks."""
    lines = [_clean_line(ln) for ln in _merge_split_sop_id_lines(lines) if _clean_line(ln)]
    blocks: List[Dict[str, Any]] = []
    para_buffer: List[str] = []

    def flush_paragraph() -> None:
        if not para_buffer:
            return
        text = _clean_line(" ".join(para_buffer))
        if text:
            blocks.append({"type": "paragraph", "text": text})
        para_buffer.clear()

    _sop_id_line = re.compile(
        r"(?i)^(?:SOP\s*ID|Document\s*ID|Dokumenten?\s*ID)\s*:\s*\S+"
    )

    i = 0
    while i < len(lines):
        line = lines[i]
        if _sop_id_line.match(line) or _is_key_value_line(line):
            flush_paragraph()
            key, value = line.split(":", 1)
            blocks.append({"type": "two_column_row", "left": _clean_line(key), "right": _clean_line(value)})
            i += 1
            continue
        if _is_numbered_heading(line) or _is_likely_heading(line):
            flush_paragraph()
            level = _to_heading_level(line)
            block_type = "section_heading" if level <= 2 else "heading"
            blocks.append({"type": block_type, "text": line, "level": level})
            i += 1
            continue
        if _is_bullet_item(line):
            flush_paragraph()
            items: List[str] = []
            while i < len(lines):
                row = lines[i]
                if not _is_bullet_item(row):
                    break
                items.append(_clean_line(re.sub(r"^[-*•]\s+", "", row)))
                i += 1
            if items:
                blocks.append({"type": "bullet_list", "items": items})
            continue
        if _is_numbered_item(line):
            flush_paragraph()
            items = []
            while i < len(lines):
                row = lines[i]
                if not _is_numbered_item(row):
                    break
                items.append(_clean_line(re.sub(r"^\d+[\)\.]\s+", "", row)))
                i += 1
            if items:
                blocks.append({"type": "numbered_list", "items": items})
            continue
        para_buffer.append(line)
        i += 1

    flush_paragraph()
    return blocks


def structure_blocks_from_text(text: str) -> List[Dict[str, Any]]:
    """Structure plain text using line-based heuristics (TXT, DOCX, flattened PDF)."""
    text = sanitize_extracted_text(text or "")
    if not text.strip():
        return []
    lines: List[str] = []
    for raw in text.splitlines():
        chunk = _clean_line(raw)
        if not chunk:
            continue
        if "\n" in raw and len(chunk) > 80:
            lines.extend(_clean_line(x) for x in raw.splitlines() if _clean_line(x))
        else:
            lines.append(chunk)
    return structure_lines_to_blocks(lines)


def _needs_restructure(blocks: List[Dict[str, Any]]) -> bool:
    if not blocks:
        return True
    if len(blocks) == 1:
        only = blocks[0]
        if only.get("type") == "paragraph" and "\n" in str(only.get("text", "")):
            return True
    typed = sum(
        1
        for b in blocks
        if str(b.get("type", "")).lower()
        in {"section_heading", "heading", "two_column_row", "bullet_list", "numbered_list", "table"}
    )
    return typed == 0 and len(blocks) <= 3


def refine_blocks(blocks: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """Re-run structure detection when upstream extraction returned coarse paragraphs."""
    if not _needs_restructure(blocks):
        return blocks
    structured = structure_blocks_from_text(text)
    return structured or blocks


def enrich_metadata_text(text: str, blocks: Optional[List[Dict[str, Any]]]) -> str:
    """
    Reconstruct labeled lines from two_column_row blocks so metadata rules
    see Title/Version/Date even when OCR split labels across lines.
    """
    parts = [sanitize_extracted_text(text or "")]
    for block in blocks or []:
        btype = str(block.get("type", "")).lower()
        if btype == "two_column_row":
            left = str(block.get("left", "")).strip()
            right = str(block.get("right", "")).strip()
            if left and right:
                parts.append(f"{left}: {right}")
        elif btype in {"section_heading", "heading"}:
            value = str(block.get("text", "")).strip()
            if value:
                parts.append(value)
    return "\n".join(p for p in parts if p).strip()


def blocks_to_structured_document(
    blocks: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build hierarchical JSON: metadata + sections/subsections + content nodes."""
    meta = metadata or {}
    doc: Dict[str, Any] = {
        "sop_id": meta.get("sop_id") or "",
        "title": meta.get("title") or "",
        "version": meta.get("version") or "",
        "date": meta.get("date") or meta.get("effective_date") or "",
        "department": meta.get("department") or "",
        "status": meta.get("status") or "",
        "sections": [],
    }
    current_section: Optional[Dict[str, Any]] = None
    current_subsection: Optional[Dict[str, Any]] = None

    def ensure_section(title: str, level: int) -> Dict[str, Any]:
        nonlocal current_section, current_subsection
        sec = {
            "title": title,
            "level": level,
            "subsections": [],
            "content": [],
        }
        doc["sections"].append(sec)
        current_section = sec
        current_subsection = None
        return sec

    def ensure_subsection(title: str, level: int) -> Dict[str, Any]:
        nonlocal current_subsection
        if current_section is None:
            ensure_section("Document Body", 1)
        sub = {"title": title, "level": level, "content": []}
        current_section["subsections"].append(sub)
        current_subsection = sub
        return sub

    def append_content(node: Dict[str, Any]) -> None:
        if current_subsection is not None:
            current_subsection["content"].append(node)
        elif current_section is not None:
            current_section["content"].append(node)
        else:
            if not doc["sections"]:
                ensure_section("Document Body", 1)
            current_section["content"].append(node)

    for block in blocks or []:
        btype = str(block.get("type", "")).lower()
        if btype in {"section_heading", "heading"}:
            title = str(block.get("text", "")).strip()
            level = int(block.get("level") or 2)
            if level <= 2:
                ensure_section(title, level)
            else:
                ensure_subsection(title, level)
        elif btype == "two_column_row":
            append_content(
                {
                    "type": "key_value",
                    "label": block.get("left", ""),
                    "value": block.get("right", ""),
                }
            )
        elif btype == "paragraph":
            append_content({"type": "paragraph", "text": block.get("text", "")})
        elif btype in {"bullet_list", "numbered_list"}:
            append_content(
                {
                    "type": btype,
                    "items": list(block.get("items") or []),
                }
            )
        elif btype == "table":
            append_content({"type": "table", "rows": block.get("rows") or []})

    return doc
