"""
Rule-based + optional LLM fallback extraction of SOP metadata from raw PDF/OCR text.
Supports German and English; does not rely on fixed line indices.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

_INVALID_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def strip_invalid_control_chars(text: str) -> str:
    """Remove null/control chars unsafe for JSONB while preserving Unicode."""
    if text is None:
        return ""
    return _INVALID_CONTROL_CHARS.sub("", str(text))


def sanitize_text_for_metadata_extraction(text: str) -> str:
    """
    Strip markdown/noise so labeled-field regexes match PDF-to-markdown imports.
    Does not remove alphanumeric content inside titles.
    """
    if not text:
        return ""
    text = strip_invalid_control_chars(text)
    out_lines: List[str] = []
    for raw in text.splitlines():
        ln = raw.replace("\ufeff", "").strip()
        ln = re.sub(r"^#{1,6}\s*", "", ln)
        ln = re.sub(r"\*{1,3}|_{1,3}|`+", "", ln)
        ln = re.sub(r"\s+", " ", ln).strip()
        out_lines.append(ln)
    return "\n".join(out_lines)


_SOP_TOKEN = r"SOP(?:[-/][A-Z0-9]+){2,}"
_SOP_ID_LINE = re.compile(
    rf"(?i)(?:SOP\s*(?:ID|Nr\.?|Number)?|Dokumenten?\s*(?:ID|Nr\.?))\s*(?:\+\s*(?:Titel|Title))?\s*[:#]?\s*({_SOP_TOKEN})"
)
_SOP_ID_GENERIC = re.compile(rf"\b({_SOP_TOKEN})\b", re.I)

# Title on same row as label (may be empty if continued on next lines)
_TITLE_SAME_LINE = re.compile(
    r"(?im)^\s*(?:SOP\s*)?(?:Title|Titel|Betreff|Document\s*Title|Bezeichnung|SOP\s*Title)\s*[:#]?\s*(.*)$"
)

_INLINE_TITLE = re.compile(
    r"(?im)^\s*(?:Title|Titel|Betreff)\s*[:#]\s*(.+?)\s*$"
)

_COMBINED_SOP_TITLE = re.compile(
    rf"(?im)^\s*(?:SOP\s*ID\s*\+\s*(?:Titel|Title)|SOP\s*(?:ID|Nr\.?)\s*(?:/|\+|-)\s*(?:Titel|Title))\s*[:#]?\s*{_SOP_TOKEN}\s*(?:[-–—:]\s*)?(.+?)\s*$"
)

_NEXT_FIELD_LINE = re.compile(
    r"(?i)^(SOP\s*ID|Document\s*(?:ID|Nr\.?)|Dokumenten?\s*(?:ID|Nr)|Version|Revision|Rev\.?|"
    r"Abteilung|Department|Bereich|Datum|Date|Effective|Gültig|Page|Seite|Ausgabe|Stand|"
    r"Scope|Geltungsbereich|Purpose|Zweck)\s*[:#]?"
)

_TITLE_METADATA_LINE = re.compile(
    r"(?i)^\s*(?:"
    r"Effective\s*Date|Gültig\s*ab|Gueltig\s*ab|Date|Datum|Version|Revision|Rev\.?|"
    r"Department|Abteilung|Status|Author|Approved\s*by|Review\s*Date|"
    r"Nächste\s*Prüfung|Prüfdatum|Überprüfung"
    r")\b"
)

_TITLE_SECTION_HEADING = re.compile(
    r"(?i)^\s*(?:Zweck|Purpose|Scope|Geltungsbereich|Definitions?|Definitionen|"
    r"Verantwortlichkeiten|Responsibilities)\b"
)

_FLATTENED_LABELS = [
    "SOP ID",
    "Document ID",
    "Dokumenten ID",
    "Title",
    "Titel",
    "Version",
    "Revision",
    "Rev.",
    "Status",
    "Department",
    "Abteilung",
    "Bereich",
    "Effective Date",
    "Review Date",
    "Date",
    "Datum",
]

_TITLE_BAD_VERBS = re.compile(
    r"(?i)\b(verifies?|approves?|performs?|ensures?|records?|collects?|"
    r"validates?|grants?|applies?|restores?|coordinates?|executes?)\b"
)

# Explicit document-version labels (line-based). Order: most specific first.
_EXPLICIT_VERSION_LINE_PATTERNS: List[tuple[str, re.Pattern]] = [
    (
        "docver",
        re.compile(
            r"(?im)^\s*(?:Dokument-Version|Document\s*Version|Dokument\s*Revision)\s*[:#]\s*(.+?)\s*$"
        ),
    ),
    (
        "docver2",
        re.compile(r"(?im)^\s*Document\s*Revision\s*[:#]\s*(.+?)\s*$"),
    ),
    ("version", re.compile(r"(?im)^\s*Version\s*[:#]\s*(.+?)\s*$")),
    ("revision", re.compile(r"(?im)^\s*(?:Revision|Rev\.?)\s*[:#]\s*(.+?)\s*$")),
    ("ausgabe", re.compile(r"(?im)^\s*(?:Ausgabe|Ausgabestand)\s*[:#]\s*(.+?)\s*$")),
    ("stand_colon", re.compile(r"(?im)^\s*Stand\s*[:#]\s*(.+?)\s*$")),
    # "Version 1", "Revision 02" (no colon)
    ("version_word", re.compile(r"(?im)^\s*Version\s+(\d+(?:\.\d+)*)\s*$")),
    (
        "revision_word",
        re.compile(r"(?im)^\s*(?:Revision|Rev\.?)\s+(0*\d+(?:\.\d+)*)\s*$"),
    ),
    # "Stand 2.1" (keyword Stand often used without colon in DE templates)
    ("stand_word", re.compile(r"(?im)^\s*Stand\s+(\d+(?:\.\d+)*)\s*$")),
]

_VERSION_PATTERNS_CELL = [
    re.compile(r"(?i)(?:^|\n|\s)Version\s*(?:number|nr\.?)?\s*[:#]?\s*(V\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?)\b"),
    # Capture full V1 / V2.1 — do not strip the V prefix (tables often use "V1" in cells).
    re.compile(r"(?i)\b(V\d+(?:\.\d+)?)\b"),
    re.compile(r"(?i)(?:Revision|Rev\.?)\s*[:#]?\s*0*(\d+(?:\.\d+)?)\b"),
    re.compile(r"(?i)\bRevision\s+0*(\d+)\b"),
    re.compile(r"(?i)(?:Ausgabe|Ausgabestand|Stand)\s*[:#]?\s*([0-9]+(?:\.[0-9]+)?)\b"),
    re.compile(r"(?i)(?:Version|Revision|Rev\.?)\s*[:#]?\s*([0-9]+(?:\.[0-9]+)?(?:\.[0-9]+)?)\b"),
]

_VERSION_LABEL = re.compile(
    r"(?i)(?:Version|Revision|Rev\.?|Ausgabe|Ausgabestand|Stand)\s*[:#]?\s*([0-9]+(?:\.[0-9]+)?(?:\.[0-9]+)?)"
)

_DATE_LABEL = re.compile(
    r"(?i)(?:Datum|Date|Effective\s*Date|Gültig(?:keit)?|Gueltig(?:keit)?|Gültig\s*ab|Gueltig\s*ab|Freigabedatum)\s*[:#]?\s*"
    r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})"
)

_DEPT_LABEL = re.compile(
    r"(?i)(?:Abteilung|Department|Bereich|Organisation(?:seinheit)?)\s*[:#]?\s*(.+?)(?:\n|$)"
)

_REVIEW_DATE_LABEL = re.compile(
    r"(?i)(?:Review\s*Date|Nächste\s*Prüfung|Prüfdatum|Überprüfung)\s*[:#]?\s*"
    r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})"
)

_STATUS_LABEL = re.compile(
    r"(?im)^\s*(?:Status|Freigabestatus|Dokumentstatus)\s*[:#]?\s*([^\n|]+)"
)

_GENERIC_DATE = re.compile(
    r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{4}|\d{4}-\d{2}-\d{2})\b"
)

_CATEGORY_KEYWORDS = [
    (re.compile(r"(?i)\b(notfall|notfallzugriff|emergency|break[-\s]?glass)\b"), "Emergency / Notfall"),
    (re.compile(r"(?i)\b(firewall|network|netzwerk)\b"), "Network Security"),
    (re.compile(r"(?i)\b(ki|ai|predictive\s+maintenance)\b"), "AI / Production Systems"),
    (re.compile(r"(?i)\bcapa\b"), "CAPA"),
    (re.compile(r"(?i)\b(abweichung|deviation)\b"), "Deviation"),
    (re.compile(r"(?i)\baudit\b"), "Audit"),
]

_DEPT_KEYWORDS = [
    (re.compile(r"(?i)\b(it|informationstechnik|information\s+technology)\b"), "IT"),
    (re.compile(r"(?i)\b(qa|qc|quality\s*assurance|qualitätssicherung)\b"), "QA"),
    (re.compile(r"(?i)\b(qualität|quality)\b(?!\s*assurance)"), "Quality"),
    (re.compile(r"(?i)\b(produktion|production|manufacturing|herstellung|fertigung)\b"), "Production"),
    (re.compile(r"(?i)\b(technik|technical|engineering)\b"), "Technical"),
    (re.compile(r"(?i)\boperations\b|\bbetrieb\b"), "Operations"),
    (re.compile(r"(?i)\b(regulatory|regulatorik)\b"), "Regulatory"),
]

_DEPT_FROM_SOP_SEGMENT = {
    "IT": "IT",
    "QA": "QA",
    "QC": "QA",
    "QS": "QA",
    "PROD": "Production",
    "PRD": "Production",
    "MFG": "Production",
    "OPS": "Operations",
    "TEC": "Technical",
    "TEK": "Technical",
    "ENG": "Engineering",
}

_STATUS_VALUE_PATTERNS = [
    (re.compile(r"(?i)\b(effective|freigegeben)\b"), "effective"),
    (re.compile(r"(?i)\b(under\s*review|in\s*review|prüfung|pruefung)\b"), "under_review"),
    (re.compile(r"(?i)\b(approved)\b"), "approved"),
    (re.compile(r"(?i)\b(obsolete|obsolet)\b"), "obsolete"),
    (re.compile(r"(?i)\b(draft|entwurf)\b"), "draft"),
]


def _normalize_date(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        return s
    m = re.match(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"
    m = re.match(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{2})$", s)
    if m:
        d, mo, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = 2000 + y2 if y2 < 70 else 1900 + y2
        return f"{y:04d}-{mo:02d}-{d:02d}"
    return ""


def _find_sop_id(text: str) -> str:
    explicit = _extract_labeled_value(text, ["SOP ID", "SOP Nr.", "SOP Number", "Document ID", "Dokumenten ID"])
    if explicit:
        m_explicit = _SOP_ID_GENERIC.search(explicit)
        if m_explicit:
            return m_explicit.group(1).strip().upper()
    m = _SOP_ID_LINE.search(text)
    if m:
        return m.group(1).strip().upper()
    m = _SOP_ID_GENERIC.search(text)
    if m:
        return m.group(1).strip().upper()
    return ""


def _label_stop_pattern(exclude_labels: Optional[List[str]] = None) -> re.Pattern:
    exclude = {x.lower() for x in (exclude_labels or [])}
    labels = [lbl for lbl in _FLATTENED_LABELS if lbl.lower() not in exclude]
    escaped = "|".join(re.escape(lbl) for lbl in labels)
    return re.compile(rf"(?i)\b(?:{escaped})\s*[:#]")


def _extract_labeled_value(
    text: str,
    labels: List[str],
    *,
    stop_labels: Optional[List[str]] = None,
) -> str:
    if not text:
        return ""
    label_pat = "|".join(re.escape(lbl) for lbl in labels)
    m = re.search(rf"(?is)\b(?:{label_pat})\s*[:#]\s*", text)
    if not m:
        return ""
    tail = text[m.end() :]
    stop_re = _label_stop_pattern(stop_labels)
    m_stop = stop_re.search(tail)
    raw = tail[: m_stop.start()] if m_stop else tail
    raw = raw.splitlines()[0] if "\n" in raw else raw
    return re.sub(r"\s+", " ", raw).strip()


def _invalid_title(title: str, sop_id: str) -> bool:
    if not title or len(title.strip()) < 3:
        return True
    t = title.strip()
    tl = t.lower()
    if _TITLE_METADATA_LINE.match(t) or _TITLE_SECTION_HEADING.match(t):
        return True
    if tl.startswith("sop id") or tl.startswith("document id") or tl.startswith("dokument"):
        return True
    if "##" in t or (t.startswith("*") and t.endswith("*") and len(t) < 80):
        return True
    if t.endswith("."):
        return True
    if _TITLE_BAD_VERBS.search(t):
        return True
    if len(t) > 120:
        return True
    if sop_id:
        compact_t = re.sub(r"\s+", "", t).upper()
        compact_id = re.sub(r"\s+", "", sop_id).upper()
        if compact_id and compact_t == compact_id:
            return True
        if sop_id in t and len(t) <= len(sop_id) + 12:
            return True
    return False


def _invalid_department(dept: str) -> bool:
    if not dept or len(dept.strip()) < 2:
        return True
    d = dept.strip()
    dl = d.lower()
    if re.fullmatch(r"\([^)]*\)", d):
        return True
    if dl in ("scope", "purpose", "zweck", "geltungsbereich") or "(scope)" in dl:
        return True
    if "scope" in dl and len(d) < 48:
        return True
    return False


def _find_title(text: str, sop_id: str) -> str:
    explicit = _extract_labeled_value(
        text,
        ["Title", "Titel", "Document Title", "SOP Title", "Betreff", "Bezeichnung"],
    )
    explicit = _clean_title_value(explicit)
    if explicit and not _invalid_title(explicit, sop_id):
        return explicit

    lines = text.splitlines()
    m = _COMBINED_SOP_TITLE.search(text)
    if m:
        combined = _clean_title_value(m.group(1))
        if combined and not _invalid_title(combined, sop_id):
            return combined

    for i, line in enumerate(lines):
        m = _TITLE_SAME_LINE.match(line)
        if m:
            same = _clean_title_value(m.group(1))
            if len(same) > 2 and not _invalid_title(same, sop_id):
                return same
            continuation: List[str] = []
            for j in range(i + 1, len(lines)):
                ln = lines[j].strip()
                if not ln:
                    break
                if _NEXT_FIELD_LINE.match(ln):
                    break
                continuation.append(ln)
            if continuation:
                merged = _clean_title_value(" ".join(continuation))
                if not _invalid_title(merged, sop_id):
                    return merged

        m = _INLINE_TITLE.match(line)
        if m:
            t = _clean_title_value(m.group(1))
            if len(t) > 2 and not _invalid_title(t, sop_id):
                return t

    for ln in lines[:40]:
        sl = ln.strip()
        if not sl:
            continue
        if re.match(r"^[-*•]\s+", sl):
            continue
        if sop_id and sl.strip().upper() == sop_id.upper():
            continue
        if sop_id and sop_id in sl and len(sl) <= len(sop_id) + 3:
            continue
        if re.match(r"^[\d.]+\s+[A-ZÄÖÜ]", sl):
            continue
        if len(sl) < 10:
            continue
        if re.match(
            r"^(version|revision|page|seite|datum|date|effective\s*date|department|abteilung|status|author|approved\s*by|review\s*date|scope|purpose|zweck|geltungsbereich|definitions?|definitionen|verantwortlichkeiten|responsibilities)\b",
            sl,
            re.I,
        ):
            continue
        if "**" in sl or sl.startswith("#"):
            continue
        cand = _clean_title_value(sl)
        if _invalid_title(cand, sop_id):
            continue
        return cand
    return ""


def _clean_title_value(s: str) -> str:
    t = re.sub(r"\s+", " ", (s or "").strip())
    t = re.sub(r"^[:\-\s]+", "", t)
    t = re.sub(r"^[^\wÄÖÜäöü(]+", "", t)
    t = re.sub(r"\*{1,3}|_{1,3}", "", t)
    t = re.split(
        r"\s+\|\s*(?:Status|Version|Revision|Rev\.?|Department|Abteilung|Date|Datum|Effective|Gültig|Gueltig)\b",
        t,
        maxsplit=1,
        flags=re.I,
    )[0]
    return t.strip()


def _sop_numeric_suffix(sop_id: str) -> str:
    if not sop_id:
        return ""
    m = re.search(r"(?i)SOP-[A-Z0-9]+-(\d+(?:\.\d+)?)\s*$", sop_id.strip())
    return m.group(1) if m else ""


def _looks_like_date_token(s: str) -> bool:
    return bool(re.match(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$", s.strip()))


def _looks_like_section_number_line(line: str) -> bool:
    return bool(re.match(r"^\s*\d{1,2}\.\s+[A-Za-zÄÖÜ]", line.strip()))


def _false_positive_version(display: str, sop_id: str, source_line: str) -> bool:
    if not display or _looks_like_date_token(display):
        return True
    if _looks_like_section_number_line(source_line):
        return True
    d = display.strip()
    suffix = _sop_numeric_suffix(sop_id)
    if suffix and d == suffix and sop_id and sop_id in source_line.replace(" ", ""):
        # Trailing SOP number (e.g. 003 from SOP-IT-003), not document revision
        if re.search(re.escape(sop_id.replace(" ", "")), source_line.replace(" ", ""), re.I):
            return True
    # Pure 3-digit match same as SOP suffix only when line is mostly the ID
    if suffix and d == suffix and len(source_line) < len(sop_id) + 15:
        if sop_id.split("-")[-1].lstrip("0") == d.lstrip("0") or sop_id.endswith(d):
            return True
    return False


def _version_sort_key(display: str) -> tuple:
    """Lexicographic tuple so 10 > 9; supports V2.1, 1.0, Rev. 02."""
    if not display:
        return (-1,)
    s = display.strip()
    m = re.match(r"(?i)^V(\d+)(?:\.(\d+))?(?:\.(\d+))?$", s)
    if m:
        return (2, int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0))
    m = re.match(r"(?i)^Rev\.?\s*0*(\d+)(?:\.(\d+))?$", s)
    if m:
        return (1, int(m.group(1)), int(m.group(2) or 0))
    m = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?$", s)
    if m:
        return (0, int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0))
    return (0, hash(s) % 10000,)


def _canonicalize_version_token(raw: str, label_kind: str) -> str:
    """Normalize captured substring to clean display (V1, 1.0, Rev. 02)."""
    if not raw:
        return ""
    s = raw.strip().strip(":*#.- ")
    s = re.split(
        r"\s+\|\s*(?:Status|Department|Abteilung|Date|Datum|Effective|Gültig|Gueltig|Title|Titel)\b",
        s,
        maxsplit=1,
        flags=re.I,
    )[0]
    m_first = re.match(r"(?i)^(V\s*\d+(?:\.\d+)*|0*\d+(?:\.\d+)*|Rev\.?\s*0*\d+(?:\.\d+)?)\b", s)
    if m_first:
        s = m_first.group(1)
    s = re.sub(r"\s+", "", s)
    if not s:
        return ""
    m = re.match(r"(?i)^V(\d+(?:\.\d+)*)$", s)
    if m:
        return f"V{m.group(1)}"
    if label_kind in ("revision", "revision_word") and re.match(r"^0*\d+(?:\.\d+)*$", s) and len(s) <= 6:
        if "." in s:
            return s.lstrip("0") or "0"
        n = int(s.lstrip("0") or "0")
        return f"Rev. {n:02d}"
    if re.match(r"^\d+(?:\.\d+)*$", s):
        return s
    return s


def _parse_version_candidate(cell: str, sop_id: str, source_line: str) -> Optional[str]:
    """Extract one version display string from arbitrary cell/line text."""
    if not cell:
        return None
    line = source_line or cell
    s = str(cell).strip()
    for pat in _VERSION_PATTERNS_CELL:
        m = pat.search(s)
        if m:
            g = m.group(m.lastindex if m.lastindex else 1)
            disp = _canonicalize_version_token(g, "generic")
            if disp and not _false_positive_version(disp, sop_id, line):
                return disp
    vm = re.search(r"(?i)\b(V\d+(?:\.\d+)?)\b", s)
    if vm:
        disp = _canonicalize_version_token(vm.group(1), "generic")
        if not _false_positive_version(disp, sop_id, line):
            return disp
    vm = re.search(r"(?i)\bV\s*(\d+(?:\.\d+)?)\b", s)
    if vm:
        disp = f"V{vm.group(1)}"
        if not _false_positive_version(disp, sop_id, line):
            return disp
    return None


def _explicit_label_versions(text: str, sop_id: str, max_lines: Optional[int]) -> List[tuple[tuple, str]]:
    """Collect (sort_key, display) from explicit labeled lines."""
    out: List[tuple[tuple, str]] = []
    lines = text.splitlines()
    limit = len(lines) if max_lines is None else min(len(lines), max_lines)
    for i in range(limit):
        line = lines[i]
        if _looks_like_section_number_line(line):
            continue
        for kind, pat in _EXPLICIT_VERSION_LINE_PATTERNS:
            m = pat.search(line)
            if not m:
                continue
            raw = (m.group(m.lastindex if m.lastindex else 1) or "").strip()
            lk = "revision" if kind in ("revision", "revision_word") else "version"
            disp = _canonicalize_version_token(raw, lk)
            if disp and not _false_positive_version(disp, sop_id, line):
                out.append((_version_sort_key(disp), disp))
            break
    return out


def _best_version_from_tables(blocks: Optional[List[Dict[str, Any]]], sop_id: str) -> Optional[str]:
    """Change-history / revision tables: prefer latest date row, else highest version."""
    if not blocks:
        return None
    history_pool: List[tuple[tuple, str, Optional[str]]] = []
    other_pool: List[tuple[tuple, str]] = []

    for block in blocks:
        if str(block.get("type", "")).lower() != "table":
            continue
        rows = block.get("rows") or []
        if len(rows) < 2:
            continue
        header = [str(c or "").lower() for c in (rows[0] or [])]
        is_history = any(
            any(
                k in h
                for k in (
                    "änd",
                    "chang",
                    "hist",
                    "histor",
                    "änderung",
                    "änderungshistorie",
                    "document history",
                )
            )
            for h in header
        )
        ver_cols: List[int] = []
        date_cols: List[int] = []
        for i, h in enumerate(header):
            # Avoid matching "Nr." row-index columns; require version/revision semantics.
            if any(k in h for k in ("revision", "version", "ausgabe", "stand", "dokument")):
                ver_cols.append(i)
            elif re.search(r"\b(rev|ver|version)\b", h) and any(
                x in h for x in ("nr", "no", "num", "#")
            ):
                ver_cols.append(i)
            elif re.match(r"^\s*(rev|ver)\.?\s*$", h.strip()):
                ver_cols.append(i)
            if any(k in h for k in ("datum", "date", "effective", "freigabe", "approved")):
                date_cols.append(i)

        data_rows = rows[1:]

        if is_history and ver_cols:
            for row in data_rows:
                row_str = " | ".join(str(c or "") for c in (row or []))
                best_cell = ""
                for ci in ver_cols:
                    if ci < len(row):
                        cell = str(row[ci] or "").strip()
                        if cell:
                            best_cell = cell
                            break
                if not best_cell:
                    continue
                disp = _parse_version_candidate(best_cell, sop_id, row_str)
                if not disp:
                    continue
                row_date: Optional[str] = None
                for di in date_cols:
                    if di < len(row):
                        row_date = _normalize_date(str(row[di] or "").strip()) or None
                        if row_date:
                            break
                history_pool.append((_version_sort_key(disp), disp, row_date))
        elif ver_cols:
            for row in data_rows:
                row_str = " | ".join(str(c or "") for c in (row or []))
                for ci in ver_cols:
                    if ci < len(row):
                        disp = _parse_version_candidate(str(row[ci] or ""), sop_id, row_str)
                        if disp:
                            other_pool.append((_version_sort_key(disp), disp))
        elif is_history:
            for row in data_rows:
                row_str = " | ".join(str(c or "") for c in (row or []))
                for cell in row or []:
                    disp = _parse_version_candidate(str(cell), sop_id, row_str)
                    if disp:
                        history_pool.append((_version_sort_key(disp), disp, None))

    if history_pool:
        with_dates = [c for c in history_pool if c[2]]
        if with_dates:
            best = max(with_dates, key=lambda x: x[2] or "")
            return best[1]
        return max(history_pool, key=lambda x: x[0])[1]
    if other_pool:
        return max(other_pool, key=lambda x: x[0])[1]
    return None


def _version_near_sop_id(text: str, sop_id: str) -> Optional[str]:
    if not sop_id:
        return None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if sop_id not in line:
            continue
        window = "\n".join(lines[max(0, i - 4) : min(len(lines), i + 6)])
        m = re.search(
            r"(?i)(?:Version|Revision|Rev\.?|Dokument-Version|Stand|Ausgabe)\s*[:#]?\s*(V?\d+(?:\.\d+)?|\d+(?:\.\d+)?)\b",
            window,
        )
        if m:
            disp = _canonicalize_version_token(m.group(1), "version")
            if disp and not _false_positive_version(disp, sop_id, window):
                return disp
    return None


def _generic_version_scan(text: str, sop_id: str) -> Optional[str]:
    """Last resort: labeled flexible patterns on full text (no leading V-first bias)."""
    found: List[tuple[tuple, str]] = []
    lines = text.splitlines()
    for line in lines:
        if _looks_like_section_number_line(line):
            continue
        m = re.search(
            r"(?i)(?:Dokument-Version|Document\s*Version|Version|Revision|Rev\.?|Ausgabe|Stand)\s*[:#]\s*(.+?)(?:\s{2,}|$)",
            line,
        )
        if m:
            disp = _canonicalize_version_token(m.group(1), "revision")
            if disp and not _false_positive_version(disp, sop_id, line):
                found.append((_version_sort_key(disp), disp))
    if found:
        return max(found, key=lambda x: x[0])[1]
    for pat in _VERSION_PATTERNS_CELL:
        m = pat.search(text)
        if m:
            g = m.group(m.lastindex if m.lastindex else 1)
            disp = _canonicalize_version_token(g, "generic")
            if disp and not _false_positive_version(disp, sop_id, text[:200]):
                return disp
    m = _VERSION_LABEL.search(text)
    if m:
        disp = _canonicalize_version_token(m.group(1), "generic")
        if disp and not _false_positive_version(disp, sop_id, text[:200]):
            return disp
    return None


def _pick_best_explicit(pairs: List[tuple[tuple, str]]) -> str:
    if not pairs:
        return ""
    return max(pairs, key=lambda x: x[0])[1]


def _find_version(text: str, blocks: Optional[List[Dict[str, Any]]], sop_id: str) -> str:
    """
    Document revision from PDF/text (not DB workflow version).
    Priority: explicit labeled lines on first page → change-history tables →
    explicit labels in remainder → near SOP ID → generic scan.
    """
    text = text or ""
    lines = text.splitlines()
    head_n = 150
    head = "\n".join(lines[:head_n])

    explicit_head = _explicit_label_versions(head, sop_id, None)
    if explicit_head:
        return _pick_best_explicit(explicit_head)

    tb = _best_version_from_tables(blocks, sop_id)
    if tb:
        return tb

    if len(lines) > head_n:
        rest = "\n".join(lines[head_n:])
        explicit_rest = _explicit_label_versions(rest, sop_id, None)
        if explicit_rest:
            return _pick_best_explicit(explicit_rest)

    near = _version_near_sop_id(text, sop_id)
    if near:
        return near

    gen = _generic_version_scan(text, sop_id)
    if gen:
        return gen
    return ""


def _dates_from_tables(blocks: Optional[List[Dict[str, Any]]]) -> tuple[str, str]:
    if not blocks:
        return "", ""
    eff = rev = ""
    for block in blocks:
        if str(block.get("type", "")).lower() != "table":
            continue
        rows = block.get("rows") or []
        if len(rows) < 2:
            continue
        header = [str(c or "").lower() for c in (rows[0] or [])]
        date_cols: List[int] = []
        eff_cols: List[int] = []
        rev_cols: List[int] = []
        for i, h in enumerate(header):
            if any(k in h for k in ("datum", "date", "effective", "gültig", "freigabe")):
                date_cols.append(i)
            if any(k in h for k in ("review", "prüf", "überprüf", "nächste")):
                rev_cols.append(i)
            if any(k in h for k in ("änd", "chang", "hist", "revision", "version")):
                eff_cols.append(i)
        data_rows = rows[1:]
        for row in reversed(data_rows):
            for idx in date_cols:
                if idx < len(row):
                    d = _normalize_date(str(row[idx] or "").strip())
                    if d and not eff:
                        eff = d
            for idx in rev_cols:
                if idx < len(row):
                    d = _normalize_date(str(row[idx] or "").strip())
                    if d and not rev:
                        rev = d
        if eff_cols and not eff:
            for row in reversed(data_rows):
                for j, cell in enumerate(row or []):
                    for dt in _GENERIC_DATE.findall(str(cell)):
                        nd = _normalize_date(dt)
                        if nd:
                            eff = nd
                            break
                    if eff:
                        break
                if eff:
                    break
    return eff, rev


def _find_dates(text: str, blocks: Optional[List[Dict[str, Any]]]) -> tuple[str, str]:
    effective = ""
    exp_effective = _extract_labeled_value(text, ["Effective Date", "Gültig ab", "Gueltig ab", "Freigabedatum"])
    if exp_effective:
        effective = _normalize_date(exp_effective)
    m = _DATE_LABEL.search(text)
    if m and not effective:
        effective = _normalize_date(m.group(1))
    review = ""
    exp_review = _extract_labeled_value(text, ["Review Date", "Nächste Prüfung", "Prüfdatum", "Überprüfung"])
    if exp_review:
        review = _normalize_date(exp_review)
    m = _REVIEW_DATE_LABEL.search(text)
    if m and not review:
        review = _normalize_date(m.group(1))

    teff, trev = _dates_from_tables(blocks)
    if not effective and teff:
        effective = teff
    if not review and trev:
        review = trev

    if not effective:
        dates = _GENERIC_DATE.findall(text)
        if dates:
            effective = _normalize_date(dates[0])
    if not review and len(_GENERIC_DATE.findall(text)) > 1:
        second = _GENERIC_DATE.findall(text)
        if len(second) > 1:
            review = _normalize_date(second[1])
    return effective, review


def _department_from_sop_id(sop_id: str) -> str:
    if not sop_id:
        return ""
    m = re.search(r"(?i)SOP-([A-Z]{2,10})-", sop_id)
    if not m:
        return ""
    code = m.group(1).upper()
    return _DEPT_FROM_SOP_SEGMENT.get(code, "")


def _find_department(text: str, sop_id: str) -> str:
    explicit = _extract_labeled_value(
        text,
        ["Department", "Abteilung", "Bereich", "Organisationseinheit", "Organisation"],
    )
    if explicit:
        dept = _clean_title_value(explicit)
        if dept and not _invalid_department(dept):
            return dept[:160]

    m = _DEPT_LABEL.search(text)
    if m:
        dept = m.group(1).strip()
        dept = re.split(
            r"\s{2,}|\n|\s+\|\s*(?:Status|Version|Revision|Rev\.?|Date|Datum|Effective|Gültig|Gueltig|Title|Titel)\b|"
            r"\b(?:Status|Version|Title|Effective\s*Date|Review\s*Date|Date|SOP\s*ID)\s*[:#]",
            dept,
            maxsplit=1,
            flags=re.I,
        )[0].strip()
        dept = _clean_title_value(dept)
        if dept and not _invalid_department(dept):
            return dept[:160]
    inferred = _department_from_sop_id(sop_id)
    if inferred:
        return inferred
    for rx, label in _DEPT_KEYWORDS:
        if rx.search(text):
            return label
    return ""


def _find_category(text: str, title: str) -> str:
    blob = f"{title}\n{text[:3000]}"
    for rx, label in _CATEGORY_KEYWORDS:
        if rx.search(blob):
            return label
    return ""


def _normalize_status(value: str) -> str:
    if not value:
        return ""
    v = str(value).strip()
    for rx, normalized in _STATUS_VALUE_PATTERNS:
        if rx.search(v):
            return normalized
    compact = re.sub(r"[\s-]+", "_", v.lower())
    return compact if compact in {"draft", "under_review", "effective", "obsolete", "approved"} else ""


def _find_status(text: str) -> str:
    explicit = _extract_labeled_value(text, ["Status", "Freigabestatus", "Dokumentstatus"])
    if explicit:
        normalized = _normalize_status(explicit)
        if normalized:
            return normalized

    m = _STATUS_LABEL.search(text or "")
    if m:
        normalized = _normalize_status(m.group(1))
        if normalized:
            return normalized
    # fallback scan if label parser misses line format
    for line in (text or "").splitlines()[:120]:
        if re.search(r"(?i)\bstatus\b", line):
            normalized = _normalize_status(line)
            if normalized:
                return normalized
    return ""


def _rules_extract(text: str, blocks: Optional[List[Dict[str, Any]]]) -> Dict[str, str]:
    text = text or ""
    sop_id = _find_sop_id(text)
    title = _find_title(text, sop_id)
    if _invalid_title(title, sop_id):
        title = ""
    if not title:
        for ln in text.splitlines()[:35]:
            sl = ln.strip()
            if len(sl) < 12:
                continue
            if re.match(r"^[-*•]\s+", sl):
                continue
            if re.match(r"^\d+(?:\.\d+)*\s+(?:SECTION|Section|Kapitel)\b", sl, re.I):
                continue
            c = _clean_title_value(sl)
            if not _invalid_title(c, sop_id):
                title = c
                break
    version = _find_version(text, blocks, sop_id)
    effective, review = _find_dates(text, blocks)
    status = _find_status(text)
    department = _find_department(text, sop_id)
    category = _find_category(text, title)

    if not department:
        department = _find_department(text[:6000], sop_id)

    primary_date = effective or ""

    return {
        "sop_id": sop_id,
        "title": title,
        "version": version,
        "date": primary_date,
        "type": "SOP",
        "category": category,
        "department": department,
        "status": status,
        "_effective_date": effective,
        "_review_date": review,
    }


def _needs_llm(d: Dict[str, str], text: str) -> bool:
    if len(text.strip()) < 80:
        return False
    missing_core = not (d.get("sop_id") or "").strip() or not (d.get("title") or "").strip()
    return missing_core


def _llm_extract(text: str) -> Optional[Dict[str, str]]:
    try:
        from chatbot.llm.provider import create_openai_client, get_local_llm_config
    except Exception:
        return None

    snippet = text.strip()[:14000]
    cfg = get_local_llm_config()
    model = os.getenv("LOCAL_LLM_METADATA_MODEL") or cfg.model
    client = create_openai_client()
    prompt = (
        "You extract structured metadata from SOP (Standard Operating Procedure) document text. "
        "The text may be German or English. Return ONLY a compact JSON object with these keys:\n"
        'sop_id, title, version, date, type, category, department\n'
        "- sop_id: identifier like SOP-IT-003 if present, else empty string\n"
        "- title: document title only, no prefix labels\n"
        "- version: revision string like 1.0 or 2 if present\n"
        "- date: one ISO date yyyy-mm-dd for main effective/issue date if found, else empty\n"
        '- type: always the string "SOP"\n'
        "- category: short label e.g. Emergency, QA, CAPA if inferable, else empty\n"
        "- department: e.g. IT, QA, Production — infer from text if explicit label missing\n"
        "Do not wrap in markdown. No explanation.\n\n---\n" + snippet
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = ((response.choices[0].message.content if response.choices else "") or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        return {
            "sop_id": str(data.get("sop_id") or "").strip(),
            "title": str(data.get("title") or "").strip(),
            "version": str(data.get("version") or "").strip(),
            "date": str(data.get("date") or "").strip(),
            "type": "SOP",
            "category": str(data.get("category") or "").strip(),
            "department": str(data.get("department") or "").strip(),
        }
    except Exception:
        return None


def _merge_rule_and_llm(rule: Dict[str, str], llm: Optional[Dict[str, str]]) -> Dict[str, str]:
    out = {k: v for k, v in rule.items() if not k.startswith("_")}
    if not llm:
        return out
    for key in ("sop_id", "title", "version", "date", "category", "department"):
        if not (out.get(key) or "").strip() and (llm.get(key) or "").strip():
            out[key] = llm[key]
    if not out.get("type"):
        out["type"] = "SOP"
    return out


def extract_sop_metadata_from_text(
    text: str,
    blocks: Optional[List[Dict[str, Any]]] = None,
    *,
    use_llm_fallback: bool = True,
) -> Dict[str, str]:
    """
    Returns canonical keys: sop_id, title, version, date, type, category, department.
    """
    text = sanitize_text_for_metadata_extraction(text or "")
    rule = _rules_extract(text, blocks)
    llm_part = None
    if use_llm_fallback and _needs_llm(rule, text):
        llm_part = _llm_extract(text)
    merged = _merge_rule_and_llm(rule, llm_part)

    eff = rule.get("_effective_date") or ""
    rev = rule.get("_review_date") or ""
    if not merged.get("date") and eff:
        merged["date"] = eff
    merged.setdefault("type", "SOP")

    # Normalize date field when LLM returned non-ISO
    if merged.get("date"):
        merged["date"] = _normalize_date(merged["date"]) or merged["date"]

    eff_iso = eff or merged.get("date") or ""
    rev_iso = rev or ""

    return {
        "sop_id": merged.get("sop_id") or "",
        "title": merged.get("title") or "",
        "version": merged.get("version") or "",
        "date": merged.get("date") or eff_iso,
        "type": merged.get("type") or "SOP",
        "category": merged.get("category") or "",
        "department": merged.get("department") or "",
        "status": _normalize_status(merged.get("status") or rule.get("status") or ""),
        "effective_date": eff_iso,
        "review_date": rev_iso,
    }


def to_frontend_sop_metadata(structured: Dict[str, Any]) -> Dict[str, Any]:
    """Map extractor output into sopMetadata shape used by the editor."""
    eff = (structured.get("effective_date") or structured.get("date") or "").strip()
    rev_d = (structured.get("review_date") or "").strip()
    return {
        "documentId": structured.get("sop_id") or "",
        "title": structured.get("title") or "",
        "department": structured.get("department") or "",
        "docType": structured.get("type") or "SOP",
        "category": structured.get("category") or "",
        "sopVersion": structured.get("version") or "",
        "sopStatus": _normalize_status(structured.get("status") or ""),
        "status": _normalize_status(structured.get("status") or ""),
        "effectiveDate": eff,
        "reviewDate": rev_d,
    }
