"""
Normalize extracted table grids for TipTap / ProseMirror.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

_CELL_BREAK = re.compile(r"\s{2,}|\t")
_HEADER_LABEL = re.compile(
    r"^(?:no\.?|#|item|step|phase|date|version|revision|status|title|name|role|"
    r"description|requirement|reference|id|sop|action|owner|department|remarks?|comments?)\b",
    re.IGNORECASE,
)


def clean_table_cell(value: Any) -> str:
    text = str(value if value is not None else "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(text.split()).strip()


def normalize_table_rows(rows: List[List[Any]]) -> List[List[str]]:
    cleaned: List[List[str]] = []
    max_cols = 0
    for row in rows or []:
        if not isinstance(row, (list, tuple)):
            continue
        cells = [clean_table_cell(c) for c in row]
        if not any(cells):
            continue
        cleaned.append(cells)
        max_cols = max(max_cols, len(cells))
    if not max_cols or not cleaned:
        return []
    out: List[List[str]] = []
    for row in cleaned:
        padded = list(row) + [""] * (max_cols - len(row))
        out.append(padded[:max_cols])
    return out


def infer_header_row_count(rows: List[List[str]], explicit: int | None = None) -> int:
    if not rows:
        return 0
    if explicit is not None:
        try:
            return max(0, min(len(rows) - 1 if len(rows) > 1 else len(rows), int(explicit)))
        except (TypeError, ValueError):
            pass
    if len(rows) == 1:
        return 0

    def row_stats(row: List[str]) -> tuple[float, int]:
        nonempty = [c for c in row if c]
        if not nonempty:
            return 0.0, 0
        avg_len = sum(len(c) for c in nonempty) / len(nonempty)
        label_hits = sum(1 for c in nonempty if _HEADER_LABEL.search(c))
        return avg_len, label_hits

    avg0, labels0 = row_stats(rows[0])
    avg1, labels1 = row_stats(rows[1])
    if len(rows) >= 3 and avg0 < 42 and avg1 < 42:
        avg2, _ = row_stats(rows[2])
        if avg2 > max(avg0, avg1) * 1.2:
            if labels0 >= max(1, len(rows[0]) // 3) and labels1 >= max(1, len(rows[1]) // 3):
                return 2
            return 1
    if labels0 >= max(2, len(rows[0]) // 2):
        return 1
    if avg0 < 40 and avg1 > avg0 * 1.3:
        return 1
    if avg0 > 80 and avg1 > 80:
        return 0
    return 0


def table_block_from_rows(rows: List[List[Any]], *, header_rows: int | None = None) -> Dict[str, Any] | None:
    normalized = normalize_table_rows(rows)
    if not normalized:
        return None
    return {"type": "table", "rows": normalized, "header_rows": infer_header_row_count(normalized, header_rows)}


def paragraph_text_looks_like_table(text: str) -> bool:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    col_counts: List[int] = []
    for line in lines[:40]:
        parts = [p.strip() for p in _CELL_BREAK.split(line) if p.strip()]
        if len(parts) >= 2:
            col_counts.append(len(parts))
    if len(col_counts) < 2:
        return False
    dominant = max(set(col_counts), key=col_counts.count)
    return col_counts.count(dominant) >= max(2, len(col_counts) * 0.6) and dominant >= 2


def table_block_from_paragraph_text(text: str) -> Dict[str, Any] | None:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    rows: List[List[str]] = []
    for line in lines:
        parts = [p.strip() for p in _CELL_BREAK.split(line) if p.strip()]
        if len(parts) >= 2:
            rows.append(parts)
    return table_block_from_rows(rows)
