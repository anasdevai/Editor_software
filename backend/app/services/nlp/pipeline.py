from __future__ import annotations

import logging
import re
from typing import Any

from app.services.sop_metadata_extractor import extract_sop_metadata_from_text

logger = logging.getLogger(__name__)


def _safe_ratio(count: int, total: int) -> float:
    return round(count / max(total, 1), 3)


def detect_language(text: str) -> dict[str, Any]:
    """Lightweight language/script detection that keeps startup dependency-safe."""
    text = text or ""
    total = max(len(text), 1)
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04FF")
    cjk = sum(1 for c in text if "\u4E00" <= c <= "\u9FFF")

    de_hits = len(
        re.findall(
            r"\b(und|die|der|das|ist|ein|eine|mit|von|bei|auf|werden|durch|oder)\b",
            text,
            re.I,
        )
    )
    en_hits = len(
        re.findall(
            r"\b(the|and|for|with|from|this|that|shall|must|will|which|when|where)\b",
            text,
            re.I,
        )
    )
    primary = "de" if de_hits > en_hits else "en"
    if arabic / total > 0.3:
        script_type = "arabic_urdu"
    elif cyrillic / total > 0.3:
        script_type = "cyrillic"
    elif cjk / total > 0.3:
        script_type = "cjk"
    elif (arabic / total > 0.1 and latin / total > 0.1) or (
        cyrillic / total > 0.1 and latin / total > 0.1
    ):
        script_type = "mixed"
    else:
        script_type = "latin"

    words = re.findall(r"\b\w+\b", text)
    technical_tokens = len(re.findall(r"\b[A-Z0-9-]{3,}\b", text))
    is_bilingual = de_hits >= 3 and en_hits >= 3
    return {
        "lang_code": primary,
        "primary_language": primary,
        "iso_code": primary,
        "confidence": 0.72 if de_hits or en_hits else 0.5,
        "is_bilingual": is_bilingual,
        "bilingual_pair": ["de", "en"] if is_bilingual else [],
        "script_type": script_type,
        "technical_language_density": _safe_ratio(technical_tokens, len(words)),
        "has_spacy_support": False,
    }


def detect_writing_style(text: str) -> dict[str, Any]:
    lines = (text or "").splitlines()
    total = max(len(lines), 1)
    table_lines = sum(1 for line in lines if line.count("|") >= 2)
    bullet_lines = sum(1 for line in lines if re.match(r"^\s*[-*•]\s+", line))
    form_lines = sum(1 for line in lines if re.match(r"^[A-Z][A-Za-z\s]+:\s*\S+", line))
    deep_decimal = bool(re.search(r"^\s*\d+\.\d+\.\d+\s+", text or "", re.M))
    standard_decimal = bool(re.search(r"^\s*\d+\.\d+\s+", text or "", re.M))
    legal_article = bool(re.search(r"^\s*Article\s+[IVX\d]", text or "", re.M | re.I))

    primary_style = "FREE_PROSE"
    strategy = "STRATEGY_H_SEMANTIC"
    if legal_article:
        primary_style = "LEGAL_CLAUSE"
        strategy = "STRATEGY_B_LEGAL_STRUCTURE"
    elif deep_decimal or standard_decimal:
        primary_style = "ISO_DECIMAL_NUMBERED"
        strategy = "STRATEGY_A_DECIMAL_NUMBERED"
    elif table_lines / total > 0.3:
        primary_style = "TABLE_DOMINANT"
        strategy = "STRATEGY_E_TABLE_AWARE"
    elif bullet_lines / total > 0.4:
        primary_style = "BULLET_ONLY"
        strategy = "STRATEGY_D_HEADER_BULLET"
    elif form_lines / total > 0.3:
        primary_style = "FORM_BASED"
        strategy = "STRATEGY_F_FIELD_VALUE"

    return {
        "primary_style": primary_style,
        "numbering_type": "decimal" if deep_decimal or standard_decimal else "simple",
        "table_density": round(table_lines / total, 3),
        "bullet_ratio": round(bullet_lines / total, 3),
        "form_field_ratio": round(form_lines / total, 3),
        "chunking_strategy_selected": strategy,
    }


def classify_tone_profile(text: str) -> dict[str, Any]:
    text = text or ""
    sentences = [s for s in re.split(r"[.!?]+", text) if len(s.strip()) > 10]
    total_sentences = max(len(sentences), 1)
    shall = len(re.findall(r"\bshall\b", text, re.I))
    must = len(re.findall(r"\b(must|required|mandatory)\b", text, re.I))
    should = len(re.findall(r"\b(should|recommended)\b", text, re.I))
    may = len(re.findall(r"\b(may|can|permitted)\b", text, re.I))
    modal_total = max(shall + must + should + may, 1)
    shall_ratio = shall / modal_total
    passive_ratio = len(
        re.findall(r"\b(is|are|was|were|be|been|being)\s+\w+ed\b", text, re.I)
    ) / total_sentences

    primary_tone = "technical_descriptive"
    if shall_ratio > 0.55 and passive_ratio > 0.25:
        primary_tone = "highly_formal_regulatory"
    elif passive_ratio > 0.5:
        primary_tone = "clinical_evidence_based"

    return {
        "primary_tone": primary_tone,
        "sub_tone_modifiers": ["AUDIT_HEAVY"]
        if re.search(r"\b(audit|traceability|evidence|record|log)\b", text, re.I)
        else [],
        "formality_level": "highly_formal" if "formal" in primary_tone else "standard",
        "authority_level": "high" if "formal" in primary_tone else "medium",
        "compliance_weight": "mandatory" if shall_ratio > 0.4 else "recommended",
        "tone_signals": {
            "SHALL_ratio": round(shall_ratio, 3),
            "passive_ratio": round(passive_ratio, 3),
        },
    }


def auto_discover_domain_rules(text: str) -> str:
    keywords = {
        "IT_OT_Infrastructure": ["firewall", "network", "vpn", "ot", "access", "vlan"],
        "Quality_Management": ["capa", "deviation", "gmp", "qa", "quality", "compliance"],
        "Manufacturing": ["production", "machine", "filling", "maintenance"],
        "Clinical": ["patient", "clinical", "trial", "protocol", "consent"],
    }
    lower = (text or "").lower()
    scores = {
        domain: sum(1 for word in words if re.search(rf"\b{re.escape(word)}", lower))
        for domain, words in keywords.items()
    }
    return max(scores, key=scores.get) if any(scores.values()) else "Quality_Management"


def adaptive_semantic_chunk(text: str, style_info: dict[str, Any]) -> list[dict[str, Any]]:
    lines = (text or "").splitlines()
    chunks: list[dict[str, Any]] = []
    current_title = "Intro"
    current_lines: list[str] = []
    current_start = 0
    header_pattern = re.compile(
        r"^\s*(?:#+\s*|\*+\s*|\d+(?:\.\d+)*[\.)]\s*)?"
        r"(Purpose|Scope|Procedure|Responsibilities|Definitions|Records|References|"
        r"Zweck|Ziel|Geltungsbereich|Verantwortung|Vorgehen|Ablauf)\b",
        re.I,
    )

    def section_type(title: str) -> str:
        lower = title.lower()
        mapping = {
            "purpose": ["purpose", "objective", "zweck", "ziel"],
            "scope": ["scope", "geltungsbereich"],
            "definitions": ["definition", "begriffe"],
            "responsibilities": ["responsib", "verantwort"],
            "procedure": ["procedure", "process", "vorgehen", "ablauf"],
            "records": ["record", "documentation", "protokoll"],
            "references": ["reference", "referenz"],
        }
        for label, aliases in mapping.items():
            if any(alias in lower for alias in aliases):
                return label
        return "general"

    def flush(end: int) -> None:
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append(
                {
                    "chunk_id": f"chunk-{len(chunks) + 1}",
                    "title": current_title,
                    "section_title": current_title,
                    "section_type": section_type(current_title),
                    "chunk_type": "header" if current_title != "Intro" else "intro",
                    "start_line": current_start,
                    "end_line": end,
                    "is_generic": False,
                    "content": content,
                }
            )

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped and header_pattern.match(stripped):
            flush(idx - 1)
            current_title = stripped[:120]
            current_lines = [line]
            current_start = idx
        else:
            current_lines.append(line)
    flush(len(lines) - 1)

    if not chunks and (text or "").strip():
        chunks.append(
            {
                "chunk_id": "chunk-1",
                "title": "Document",
                "section_title": "Document",
                "section_type": "general",
                "chunk_type": "text",
                "start_line": 0,
                "end_line": len(lines),
                "is_generic": True,
                "content": text.strip(),
            }
        )
    return chunks


def extract_workflow_steps(text: str) -> list[str]:
    action_verbs = (
        "Ensure|Perform|Record|Maintain|Check|Verify|Submit|Review|Approve|Create|"
        "Update|Install|Test|Inspect|Monitor|Sicherstellen|Prüfen|Ueberwachen|"
        "Überwachen|Dokumentieren|Einhalten|Konfigurieren|Validieren|Freigeben"
    )
    steps = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if len(stripped) > 10 and re.search(rf"\b({action_verbs})\b", stripped, re.I):
            steps.append(stripped)
    return steps[:15]


def detect_risks_and_hazards(text: str) -> list[str]:
    risk_keywords = [
        "danger",
        "warning",
        "caution",
        "hazard",
        "risk",
        "critical",
        "safety",
        "failure",
        "error",
        "contamination",
        "security",
        "breach",
        "Gefahr",
        "Warnung",
        "kritisch",
        "Ausfall",
        "unbefugt",
    ]
    found = {word for word in risk_keywords if re.search(rf"\b{re.escape(word)}", text or "", re.I)}
    return sorted(found)


def extract_compliance_references(text: str) -> list[str]:
    patterns = [
        r"\bISO\s*\d+\b",
        r"\bIEC\s*\d+\b",
        r"\bBSI\b",
        r"\bNIST\b",
        r"\b\d+\s*CFR\s*Part\s*\d+\b",
        r"\bGDPR\b",
        r"\bHIPAA\b",
        r"\bAnnex\s*\d+\b",
        r"\bG[A-Z]P\b",
        r"\bMFA\b",
    ]
    refs: set[str] = set()
    for pattern in patterns:
        refs.update(match.strip() for match in re.findall(pattern, text or "", re.I))
    return sorted(refs)


def extract_roles(text: str) -> list[str]:
    roles = re.findall(
        r"\b(QA|QC|CISO|Admin|Operator|Supervisor|Owner|User|Manager|Reviewer|Approver|Technician)\b",
        text or "",
        re.I,
    )
    return sorted({role.upper() if len(role) <= 4 else role.title() for role in roles})


def validate_section_order(sections: list[str]) -> dict[str, Any]:
    canonical = ["PURPOSE", "SCOPE", "RESPONSIBILITIES", "PROCEDURE", "RECORDS"]
    present = {section.upper() for section in sections}
    return {"canonical_order": canonical, "missing_sections": [s for s in canonical if s not in present]}


def analyze_sop_text(text: str) -> dict[str, Any]:
    """Detect an SOP profile using local deterministic NLP heuristics."""
    text = text or ""
    lang_info = detect_language(text)
    style_info = detect_writing_style(text)
    tone_info = classify_tone_profile(text)
    chunks = adaptive_semantic_chunk(text, style_info)
    metadata = extract_sop_metadata_from_text(text, use_llm_fallback=False)
    domain = auto_discover_domain_rules(text)
    sections = [chunk["title"] for chunk in chunks if chunk.get("section_type") != "general"]
    section_types = [chunk["section_type"] for chunk in chunks if chunk.get("section_type") != "general"]
    gaps = validate_section_order(section_types)["missing_sections"]
    workflow = extract_workflow_steps(text)
    compliance = extract_compliance_references(text)
    risks = detect_risks_and_hazards(text)
    roles = extract_roles(text)

    words = re.findall(r"\b\w+\b", text)
    quality_score = max(20.0, 100.0 - (len(gaps) * 8.0))
    return {
        "sop_id": metadata.get("sop_id", ""),
        "title": metadata.get("title", ""),
        "version": metadata.get("version", ""),
        "domain": domain,
        "sections": sections,
        "style_profile": {
            "primary_style": style_info["primary_style"],
            "primary_tone": tone_info["primary_tone"],
            "formality_level": tone_info["formality_level"],
            "strictness_level": "high"
            if tone_info.get("tone_signals", {}).get("SHALL_ratio", 0) > 0.4
            else "moderate",
            "numbering_type": style_info["numbering_type"],
            "format_pattern": style_info["primary_style"].lower().replace("_", " "),
            "compliance_weight": tone_info["compliance_weight"],
            "language": lang_info,
            "quality_score": round(quality_score, 1),
            "roles": roles,
        },
        "roles": roles,
        "workflow": workflow,
        "compliance": compliance,
        "risks": risks,
        "gaps": gaps,
        "chunks": chunks[:12],
        "stats": {"characters": len(text), "words": len(words), "chunks": len(chunks)},
    }

