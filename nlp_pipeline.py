
"""
Industry-Level Dynamic SOP NLP Pipeline
======================================

Purpose:
- Dynamically analyze any SOP or controlled procedure.
- Detect document information, writing style, roles, workflow, compliance elements,
  risks/gaps, terminology, structure patterns, and client SOP profile.
- Generate profile.md content for client-specific style learning.

Design principles:
- Works with standard Python first.
- Uses optional NLP libraries when available.
- Every important detection includes evidence and confidence where possible.
- No hardcoded single-SOP behavior. Rule banks are extensible.

Main entry points:
- analyze_sop_industry_level(text, client_name="Client")
- generate_profile_md(client_profile)
- analyze_sop_file(input_path, output_json_path=None, output_profile_path=None)
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except Exception:  # pragma: no cover
    LANGDETECT_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except Exception:  # pragma: no cover
    TEXTSTAT_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:  # pragma: no cover
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SBERT_AVAILABLE = True
except Exception:  # pragma: no cover
    SBERT_AVAILABLE = False


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

_SPACY_CACHE: Dict[str, Any] = {}
_SBERT_CACHE: Dict[str, Any] = {}

SPACY_MODEL_MAP = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
}


def _safe_round(value: float, ndigits: int = 3) -> float:
    try:
        return round(float(value), ndigits)
    except Exception:
        return 0.0


def _normalise_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalise_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")


def _split_sentences(text: str) -> List[str]:
    # Keeps SOP-style fragments but avoids tiny noise snippets.
    parts = re.split(r"(?<=[.!?])\s+|\n(?=\s*(?:[-*•]|\d+[.)]|[A-Z][A-Za-z ]{2,}:))", text or "")
    return [_normalise_space(p) for p in parts if len(_normalise_space(p).split()) >= 3]


def _split_lines(text: str) -> List[str]:
    return [line.rstrip() for line in (text or "").splitlines()]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_/-]*", text or "")


def _lower_tokens(text: str) -> List[str]:
    return [t.lower() for t in _tokenize(text)]


def _keyword_hits(text: str, keywords: Sequence[str]) -> List[str]:
    lower = (text or "").lower()
    hits = []
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw.lower()) + r"\b", lower):
            hits.append(kw)
    return sorted(set(hits), key=lambda x: x.lower())


def _extract_evidence(text: str, keywords: Sequence[str], max_items: int = 3) -> List[str]:
    evidence = []
    sentences = _split_sentences(text)
    for s in sentences:
        sl = s.lower()
        if any(kw.lower() in sl for kw in keywords):
            evidence.append(s[:350])
        if len(evidence) >= max_items:
            break
    return evidence


def _confidence_from_hits(hit_count: int, possible: int = 5, base: float = 0.25, cap: float = 0.95) -> float:
    return _safe_round(min(cap, base + (hit_count / max(possible, 1)) * (cap - base)), 3)


def _get_spacy_doc(text: str, lang_code: str = "en") -> Any:
    if not SPACY_AVAILABLE:
        return None
    model_name = SPACY_MODEL_MAP.get(lang_code[:2], "en_core_web_sm")
    if model_name not in _SPACY_CACHE:
        try:
            _SPACY_CACHE[model_name] = spacy.load(model_name)
        except Exception:
            _SPACY_CACHE[model_name] = None
    model = _SPACY_CACHE.get(model_name)
    if not model:
        return None
    try:
        return model(text[:900000])
    except Exception:
        return None


def _get_sbert() -> Any:
    if not SBERT_AVAILABLE:
        return None
    if "default" not in _SBERT_CACHE:
        try:
            _SBERT_CACHE["default"] = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _SBERT_CACHE["default"] = None
    return _SBERT_CACHE.get("default")


@dataclass
class EvidenceItem:
    value: str
    confidence: float
    evidence: List[str]
    source: str = "rule"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SectionChunk:
    title: str
    label: str
    content: str
    start_line: int
    end_line: int
    level: int = 0
    confidence: float = 0.75

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Knowledge banks
# -----------------------------------------------------------------------------

REGULATORY_STANDARDS = {
    "ISO 9001": [r"\bISO\s*9001\b", "QMS_general"],
    "ISO 13485": [r"\bISO\s*13485\b", "medical_device_QMS"],
    "ISO 17025": [r"\bISO\s*17025\b", "laboratory"],
    "ISO 14001": [r"\bISO\s*14001\b", "environmental_EHS"],
    "ISO 45001": [r"\bISO\s*45001\b", "occupational_health_safety"],
    "IEC 62443": [r"\bIEC\s*62443\b", "IT_OT_security"],
    "21 CFR Part 11": [r"\b21\s*CFR\s*Part\s*11\b", "pharma_IT_csv"],
    "21 CFR Part 210/211": [r"\b21\s*CFR\s*Part\s*(210|211)\b", "pharma_GMP"],
    "EU GMP": [r"\bEU\s*GMP\b|\bEudraLex\b|\bAnnex\s*11\b|\bAnnex\s*15\b", "pharma_GMP"],
    "ICH Q9": [r"\bICH\s*Q9\b", "quality_risk_management"],
    "ICH Q10": [r"\bICH\s*Q10\b", "pharma_quality_system"],
    "GxP": [r"\bGxP\b|\bGMP\b|\bGDP\b|\bGLP\b", "life_sciences_GxP"],
    "HACCP": [r"\bHACCP\b", "food_safety"],
    "NIST": [r"\bNIST\b|\bSP\s*800[- ]\d+\b", "cybersecurity"],
    "SOC 2": [r"\bSOC\s*2\b", "security_compliance"],
    "HIPAA": [r"\bHIPAA\b", "healthcare_privacy"],
    "GDPR": [r"\bGDPR\b", "privacy"],
}

DOMAIN_BANKS = {
    "QA_QMS": [
        "quality", "qms", "qa", "qc", "audit", "nonconformance", "non-conformance", "deviation",
        "capa", "change control", "document control", "training", "record", "controlled copy",
        "effective date", "revision", "approval", "quality assurance", "quality control",
    ],
    "pharma_GMP": [
        "gmp", "batch", "lot", "release", "qualified person", "qp", "validation", "qualification",
        "cleanroom", "contamination", "sterile", "oos", "oot", "lims", "manufacturing record",
        "master batch record", "data integrity", "alcoa", "alcoa+",
    ],
    "IT_OT_security": [
        "it", "ot", "server", "firewall", "access control", "authentication", "authorization", "mfa",
        "sso", "vpn", "rbac", "active directory", "iam", "pam", "service account", "scada", "plc",
        "sps", "backup", "restore", "patch", "vulnerability", "incident response", "cybersecurity",
    ],
    "production_manufacturing": [
        "production", "manufacturing", "line clearance", "operator", "technician", "equipment",
        "machine", "work order", "assembly", "maintenance", "calibration", "batch production",
    ],
    "laboratory": [
        "laboratory", "sample", "specimen", "assay", "reagent", "calibration", "method validation",
        "uncertainty", "chain of custody", "lab analyst", "test method",
    ],
    "EHS": [
        "safety", "hazard", "ppe", "risk assessment", "spill", "waste", "environmental", "emergency",
        "injury", "incident", "near miss", "lockout", "tagout", "loto",
    ],
    "HR_training": [
        "training", "competency", "employee", "onboarding", "hr", "attendance", "trainer",
        "qualification matrix", "training record",
    ],
    "finance_admin": [
        "invoice", "purchase order", "approval matrix", "expense", "finance", "procurement", "vendor",
    ],
}

DEPARTMENT_BANKS = {
    "Quality Assurance": ["qa", "quality assurance", "quality manager", "document control", "qms", "compliance"],
    "Quality Control": ["qc", "quality control", "qc analyst", "laboratory analyst", "lab analyst"],
    "Information Technology": ["it", "it admin", "system administrator", "network", "cybersecurity", "service desk"],
    "Operational Technology": ["ot", "scada", "plc", "sps", "industrial control", "production network"],
    "Production": ["production", "operator", "line supervisor", "manufacturing", "plant", "shop floor"],
    "Engineering": ["engineering", "maintenance", "calibration", "equipment owner", "utilities"],
    "Regulatory Affairs": ["regulatory", "regulatory affairs", "submission", "health authority"],
    "Warehouse / Logistics": ["warehouse", "inventory", "shipment", "logistics", "dispatch"],
    "EHS": ["ehs", "safety officer", "environmental", "health and safety"],
    "Human Resources": ["hr", "human resources", "training coordinator", "people operations"],
}

SOP_TYPE_BANKS = {
    "Policy SOP": ["policy", "governance", "principle", "shall apply", "organization-wide"],
    "Procedure SOP": ["procedure", "process", "step", "workflow", "method", "how to"],
    "Work Instruction": ["work instruction", "wi", "operator shall", "technician shall", "step-by-step"],
    "Validation / Qualification SOP": ["validation", "qualification", "iq", "oq", "pq", "csv", "computer system validation"],
    "Document Control SOP": ["document control", "controlled copy", "revision", "effective date", "change history"],
    "Training SOP": ["training", "competency", "training record", "qualification matrix"],
    "Deviation / Incident SOP": ["deviation", "incident", "nonconformance", "near miss", "event report"],
    "CAPA SOP": ["capa", "corrective action", "preventive action", "root cause", "effectiveness check"],
    "Audit SOP": ["audit", "audit finding", "observation", "audit plan", "audit report"],
    "Access Control SOP": ["access control", "user access", "privilege", "rbac", "account provisioning", "password"],
    "Backup / Restore SOP": ["backup", "restore", "recovery", "disaster recovery", "rto", "rpo"],
    "Change Control SOP": ["change control", "change request", "impact assessment", "change approval"],
    "EHS SOP": ["ppe", "hazard", "safety", "spill", "emergency", "lockout", "tagout"],
}

SECTION_ALIASES = {
    "document_information": ["document information", "sop information", "title", "sop id", "document number", "version"],
    "purpose": ["purpose", "objective", "aim", "intent", "zweck", "ziel"],
    "scope": ["scope", "applicability", "coverage", "geltungsbereich", "anwendungsbereich"],
    "definitions": ["definitions", "terms", "glossary", "abbreviations", "begriffe", "definitionen"],
    "responsibilities": ["responsibilities", "roles", "accountabilities", "raci", "ownership", "verantwortlichkeiten"],
    "procedure": ["procedure", "process", "method", "steps", "workflow", "work instruction", "verfahren", "ablauf"],
    "approval": ["approval", "sign-off", "authorization", "approver", "freigabe", "genehmigung"],
    "review": ["review", "periodic review", "annual review", "management review", "überprüfung"],
    "incident": ["incident", "event", "near miss", "security event", "reportable event"],
    "deviation": ["deviation", "nonconformance", "non-conformance", "exception", "abweichung"],
    "capa": ["capa", "corrective action", "preventive action", "effectiveness check", "root cause"],
    "audit": ["audit", "audit finding", "observation", "inspection", "audit trail"],
    "records": ["records", "documentation", "logs", "forms", "evidence", "retention", "aufzeichnungen"],
    "references": ["references", "related documents", "standards", "source documents", "referenzen"],
    "revision_history": ["revision", "change history", "version history", "amendment", "änderungshistorie"],
    "training": ["training", "competency", "qualification", "training records"],
    "risk_controls": ["risk", "control", "mitigation", "hazard", "fmea", "criticality"],
    "escalation": ["escalation", "escalate", "notify", "management notification"],
}

ROLE_BANKS = {
    "QA": ["qa", "quality assurance", "quality manager", "document controller", "qms owner", "compliance officer"],
    "QC": ["qc", "quality control", "qc analyst", "laboratory analyst"],
    "IT": ["it", "it admin", "system administrator", "network administrator", "service desk", "cybersecurity"],
    "OT": ["ot", "automation engineer", "scada engineer", "plc engineer", "sps engineer"],
    "Production": ["production", "production manager", "line supervisor", "operator", "plant manager"],
    "Technician": ["technician", "maintenance technician", "operator", "engineer", "analyst"],
    "Reviewer": ["reviewer", "review", "technical reviewer", "qa reviewer", "process owner"],
    "Approver": ["approver", "approve", "approval", "authorized person", "signatory", "department head"],
    "Owner": ["owner", "process owner", "system owner", "document owner", "business owner"],
    "Auditor": ["auditor", "internal auditor", "lead auditor", "audit team"],
    "CAPA Owner": ["capa owner", "action owner", "responsible owner", "corrective action owner"],
    "EHS": ["ehs", "safety officer", "health and safety", "environmental officer"],
}

ACTION_VERBS = [
    "verify", "ensure", "perform", "execute", "document", "record", "review", "approve", "submit",
    "notify", "escalate", "investigate", "assess", "classify", "close", "release", "archive",
    "train", "retain", "monitor", "control", "validate", "authorize", "implement", "check",
]

MODAL_TERMS = {
    "mandatory": ["shall", "must", "is required to", "are required to", "mandatory", "muss"],
    "recommended": ["should", "recommended", "sollte"],
    "permissive": ["may", "can", "permitted", "allowed"],
    "prohibited": ["shall not", "must not", "may not", "prohibited", "not permitted", "darf nicht"],
}

FLOW_DEFINITIONS = {
    "approval_flow": [
        "draft", "prepare", "submit", "review", "approve", "authorize", "release", "effective", "archive",
    ],
    "incident_flow": [
        "detect", "identify", "report", "notify", "classify", "contain", "investigate", "escalate", "resolve", "close",
    ],
    "review_flow": [
        "schedule", "review", "assess", "update", "approve", "record", "periodic review", "annual review",
    ],
    "capa_flow": [
        "deviation", "root cause", "investigate", "corrective action", "preventive action", "implement", "effectiveness", "verify", "close",
    ],
    "audit_flow": [
        "plan", "conduct", "finding", "observation", "report", "capa", "follow-up", "close"],
    "change_control_flow": [
        "request", "impact assessment", "risk assessment", "review", "approve", "implement", "verify", "close"],
}

REQUIRED_ELEMENTS_BY_TYPE = {
    "default": ["purpose", "scope", "responsibilities", "procedure", "records", "approval", "revision_history"],
    "CAPA SOP": ["purpose", "scope", "responsibilities", "procedure", "capa", "records", "approval", "review"],
    "Deviation / Incident SOP": ["purpose", "scope", "responsibilities", "procedure", "incident", "deviation", "escalation", "records", "approval"],
    "Audit SOP": ["purpose", "scope", "responsibilities", "procedure", "audit", "records", "capa", "approval"],
    "Access Control SOP": ["purpose", "scope", "responsibilities", "procedure", "approval", "risk_controls", "records", "review"],
    "Validation / Qualification SOP": ["purpose", "scope", "responsibilities", "procedure", "risk_controls", "records", "approval", "review"],
    "Document Control SOP": ["purpose", "scope", "responsibilities", "procedure", "approval", "revision_history", "records", "references"],
}

CONTROL_KEYWORDS = [
    "control", "verification", "check", "approval", "review", "audit trail", "record", "evidence", "segregation",
    "access restriction", "mfa", "dual approval", "independent review", "effectiveness check", "monitoring",
]

ESCALATION_KEYWORDS = [
    "escalate", "escalation", "notify", "inform", "report to", "management", "department head",
    "qa manager", "critical", "overdue", "unresolved", "severity", "immediate",
]

TIMING_PATTERN = re.compile(
    r"\b(within\s+\d+\s+(?:minute|minutes|hour|hours|day|days|working day|working days|business day|business days)|"
    r"no later than\s+[^.;\n]+|"
    r"every\s+\d+\s+(?:month|months|year|years|day|days)|"
    r"(?:daily|weekly|monthly|quarterly|annually|yearly|immediately|before|after)\b)",
    re.I,
)

TRACE_ID_PATTERN = re.compile(r"\b(?:SOP|WI|POL|FORM|FRM|DEV|CAPA|AUD|NCR|CC|CR|DEC|TRN)-?[A-Z0-9]{2,}[-/]?\d{2,6}\b", re.I)


# -----------------------------------------------------------------------------
# Stage 1: Language and structural detection
# -----------------------------------------------------------------------------

def detect_language(text: str) -> Dict[str, Any]:
    text = text or ""
    lang_code = "en"
    confidence = 0.50
    alternatives = []

    if LANGDETECT_AVAILABLE and text.strip():
        try:
            probs = detect_langs(text[:50000])
            if probs:
                lang_code = probs[0].lang
                confidence = _safe_round(probs[0].prob, 3)
                alternatives = [{"lang": p.lang, "probability": _safe_round(p.prob, 3)} for p in probs[:3]]
        except Exception:
            pass

    total_chars = max(len(text), 1)
    latin = sum(1 for c in text if c.isascii() and c.isalpha()) / total_chars
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06FF") / total_chars
    cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04FF") / total_chars
    cjk = sum(1 for c in text if "\u4E00" <= c <= "\u9FFF") / total_chars

    en_markers = len(re.findall(r"\b(the|and|shall|must|procedure|scope|purpose|approval|review)\b", text, re.I))
    de_markers = len(re.findall(r"\b(und|der|die|das|muss|zweck|geltungsbereich|freigabe|verfahren)\b", text, re.I))
    is_bilingual = en_markers >= 4 and de_markers >= 4

    return {
        "primary_language": lang_code,
        "confidence": confidence,
        "alternatives": alternatives,
        "is_bilingual": is_bilingual,
        "bilingual_pair": ["en", "de"] if is_bilingual else [],
        "script_ratios": {
            "latin": _safe_round(latin, 3),
            "arabic_urdu": _safe_round(arabic, 3),
            "cyrillic": _safe_round(cyrillic, 3),
            "cjk": _safe_round(cjk, 3),
        },
        "processing_mode": "bilingual_parallel" if is_bilingual else "standard",
    }


def detect_sections(text: str) -> List[Dict[str, Any]]:
    lines = _split_lines(text)
    heading_regexes = [
        re.compile(r"^\s*(\d+(?:\.\d+)*)(?:[.)])?\s+(.{2,120})$"),
        re.compile(r"^\s*#{1,6}\s+(.{2,120})$"),
        re.compile(r"^\s*([A-Z][A-Z0-9 /&()\-]{3,120})\s*$"),
        re.compile(r"^\s*([A-Z][A-Za-z0-9 /&()\-]{2,80}):\s*$"),
    ]

    headings: List[Tuple[int, str, int, float]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 140:
            continue
        matched_title = None
        level = 0
        conf = 0.60
        for rgx in heading_regexes:
            m = rgx.match(stripped)
            if not m:
                continue
            if len(m.groups()) == 2:
                level = m.group(1).count(".") + 1
                matched_title = m.group(2).strip(" :.-")
                conf = 0.85
            else:
                matched_title = m.group(1).strip(" :.-")
                conf = 0.72
            break
        if matched_title:
            headings.append((idx, matched_title, level, conf))

    if not headings:
        return [SectionChunk("Document", "general", text, 0, max(len(lines) - 1, 0), confidence=0.40).to_dict()]

    chunks: List[SectionChunk] = []
    for i, (line_no, title, level, conf) in enumerate(headings):
        end = headings[i + 1][0] - 1 if i + 1 < len(headings) else len(lines) - 1
        content = "\n".join(lines[line_no:end + 1]).strip()
        label, label_conf = classify_section_title(title)
        chunks.append(SectionChunk(title=title, label=label, content=content, start_line=line_no, end_line=end, level=level, confidence=max(conf, label_conf)).to_dict())
    return chunks


def classify_section_title(title: str) -> Tuple[str, float]:
    norm = _normalise_space(title).lower()
    best_label = "general"
    best_score = 0.0
    for label, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            alias_l = alias.lower()
            if alias_l == norm or alias_l in norm:
                score = 0.95 if alias_l == norm else 0.82
            else:
                score = _token_overlap(norm, alias_l)
            if score > best_score:
                best_label = label
                best_score = score
    if best_score < 0.25:
        return "general", 0.35
    return best_label, _safe_round(best_score, 3)


def _token_overlap(a: str, b: str) -> float:
    a_set = set(re.findall(r"[a-z0-9]{3,}", a.lower()))
    b_set = set(re.findall(r"[a-z0-9]{3,}", b.lower()))
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / max(len(a_set | b_set), 1)


# -----------------------------------------------------------------------------
# Stage 2: Document information and domain
# -----------------------------------------------------------------------------

def _detect_from_bank(text: str, bank: Dict[str, Sequence[str]], max_items: int = 5) -> List[EvidenceItem]:
    results: List[EvidenceItem] = []
    lower = text.lower()
    for label, keywords in bank.items():
        hits = []
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw.lower()) + r"\b", lower):
                hits.append(kw)
        if hits:
            results.append(EvidenceItem(
                value=label,
                confidence=_confidence_from_hits(len(hits), possible=6, base=0.35),
                evidence=_extract_evidence(text, hits, max_items=2),
                source="keyword_bank",
            ))
    return sorted(results, key=lambda x: x.confidence, reverse=True)[:max_items]


def detect_regulatory_standards(text: str) -> List[Dict[str, Any]]:
    found = []
    for standard, (pattern, domain) in REGULATORY_STANDARDS.items():
        matches = re.findall(pattern, text or "", re.I)
        if matches:
            found.append({
                "standard": standard,
                "mapped_domain": domain,
                "confidence": 0.95,
                "evidence": _extract_evidence(text, [standard.split()[0], standard], max_items=2),
            })
    return found


def detect_document_information(text: str, sections: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    sections = sections or detect_sections(text)
    standards = detect_regulatory_standards(text)
    domain_candidates = _detect_from_bank(text, DOMAIN_BANKS, max_items=6)
    dept_candidates = _detect_from_bank(text, DEPARTMENT_BANKS, max_items=6)
    sop_type_candidates = _detect_from_bank(text, SOP_TYPE_BANKS, max_items=6)

    if standards:
        for st in standards:
            mapped = st["mapped_domain"]
            if not any(c.value == mapped for c in domain_candidates):
                domain_candidates.append(EvidenceItem(mapped, 0.90, st.get("evidence", []), "regulatory_standard"))
        domain_candidates = sorted(domain_candidates, key=lambda x: x.confidence, reverse=True)[:6]

    explicit_ids = sorted(set(TRACE_ID_PATTERN.findall(text or "")))
    title = _guess_title(text, sections)
    version = _first_match(text, [r"\b(?:version|revision|rev\.?|v)\s*[:#-]?\s*([A-Z0-9.\-]+)\b"])
    effective_date = _first_match(text, [r"\b(?:effective date|date effective|valid from)\s*[:#-]?\s*([A-Za-z0-9, ./-]{6,30})"])

    primary_domain = domain_candidates[0].value if domain_candidates else "UNKNOWN"
    primary_department = dept_candidates[0].value if dept_candidates else "UNKNOWN"
    primary_sop_type = sop_type_candidates[0].value if sop_type_candidates else "Procedure SOP" if _has_any(text, ["procedure", "scope", "purpose"]) else "UNKNOWN"

    return {
        "title": title,
        "document_ids": explicit_ids,
        "version_or_revision": version,
        "effective_date": effective_date,
        "sop_type": _evidence_value(primary_sop_type, sop_type_candidates),
        "category": _evidence_value(_infer_category(primary_domain, primary_sop_type), domain_candidates),
        "department": _evidence_value(primary_department, dept_candidates),
        "domain": _evidence_value(primary_domain, domain_candidates),
        "standards": standards,
        "all_sop_type_candidates": [c.to_dict() for c in sop_type_candidates],
        "all_department_candidates": [c.to_dict() for c in dept_candidates],
        "all_domain_candidates": [c.to_dict() for c in domain_candidates],
    }


def _evidence_value(value: str, candidates: Sequence[EvidenceItem]) -> Dict[str, Any]:
    for c in candidates:
        if c.value == value:
            return c.to_dict()
    return {"value": value, "confidence": 0.40 if value != "UNKNOWN" else 0.10, "evidence": [], "source": "inference"}


def _infer_category(domain: str, sop_type: str) -> str:
    if domain in {"QA_QMS", "pharma_GMP", "life_sciences_GxP", "medical_device_QMS"}:
        return "Quality / Compliance"
    if domain in {"IT_OT_security", "cybersecurity", "security_compliance", "pharma_IT_csv"}:
        return "IT / Security / CSV"
    if domain in {"production_manufacturing", "laboratory"}:
        return "Operations / Technical"
    if "CAPA" in sop_type or "Deviation" in sop_type or "Audit" in sop_type:
        return "Quality Event Management"
    if domain == "EHS":
        return "EHS"
    return domain or "General SOP"


def _guess_title(text: str, sections: Sequence[Dict[str, Any]]) -> Optional[str]:
    first_lines = [l.strip() for l in _split_lines(text)[:20] if l.strip()]
    for rgx in [r"\btitle\s*[:#-]\s*(.+)", r"\bSOP\s*Title\s*[:#-]\s*(.+)"]:
        for line in first_lines:
            m = re.search(rgx, line, re.I)
            if m:
                return m.group(1).strip()[:180]
    if sections:
        t = sections[0].get("title") or ""
        if len(t.split()) >= 2:
            return t[:180]
    for line in first_lines:
        if 3 <= len(line.split()) <= 16 and not re.search(r"^(version|date|prepared|approved|page)\b", line, re.I):
            return line[:180]
    return None


def _first_match(text: str, patterns: Sequence[str]) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text or "", re.I)
        if m:
            return _normalise_space(m.group(1))[:80]
    return None


def _has_any(text: str, keywords: Sequence[str]) -> bool:
    lower = (text or "").lower()
    return any(k.lower() in lower for k in keywords)


# -----------------------------------------------------------------------------
# Stage 3: Writing style, tone, modal verbs, complexity
# -----------------------------------------------------------------------------

def detect_writing_style(text: str) -> Dict[str, Any]:
    lines = _split_lines(text)
    sentences = _split_sentences(text)
    tokens = _tokenize(text)
    total_lines = max(len(lines), 1)
    total_sentences = max(len(sentences), 1)
    total_tokens = max(len(tokens), 1)

    bullet_count = sum(1 for l in lines if re.match(r"^\s*(?:[-*•]|[a-zA-Z]\)|\d+[.)])\s+", l))
    numbered_count = sum(1 for l in lines if re.match(r"^\s*\d+(?:\.\d+)*[.)]?\s+", l))
    table_count = sum(1 for l in lines if l.count("|") >= 2 or re.search(r"\t", l))
    form_count = sum(1 for l in lines if re.match(r"^\s*[A-Za-z][A-Za-z ]{2,40}:\s*.+", l))
    headings = detect_sections(text)

    modal_counts = {group: sum(len(re.findall(r"\b" + re.escape(term) + r"\b", text, re.I)) for term in terms) for group, terms in MODAL_TERMS.items()}
    action_counts = {verb: len(re.findall(r"\b" + re.escape(verb) + r"\b", text, re.I)) for verb in ACTION_VERBS}
    action_counts = {k: v for k, v in action_counts.items() if v > 0}

    avg_sentence_length = sum(len(s.split()) for s in sentences) / total_sentences
    avg_line_words = sum(len(l.split()) for l in lines if l.strip()) / max(sum(1 for l in lines if l.strip()), 1)
    passive_count = len(re.findall(r"\b(?:is|are|was|were|be|been|being)\s+\w+(?:ed|en)\b", text, re.I))
    imperative_count = sum(1 for s in sentences if re.match(r"^(?:" + "|".join(ACTION_VERBS) + r")\b", s, re.I))
    acronym_count = len(re.findall(r"\b[A-Z]{2,}\b", text or ""))
    nominalisation_count = len(re.findall(r"\b\w+(?:tion|ment|ance|ence|ity)\b", text or "", re.I))
    contractions = len(re.findall(r"\b\w+'(?:t|re|ve|ll|d|m)\b", text or "", re.I))

    if modal_counts["mandatory"] >= modal_counts["recommended"] + modal_counts["permissive"] and modal_counts["mandatory"] > 0:
        directive_wording = "mandatory/controlled"
    elif imperative_count / total_sentences > 0.25:
        directive_wording = "imperative/action-led"
    elif modal_counts["recommended"] > modal_counts["mandatory"]:
        directive_wording = "guidance/recommendation-led"
    else:
        directive_wording = "descriptive/mixed"

    if avg_sentence_length >= 24 or nominalisation_count / total_tokens > 0.06:
        writing_complexity = "high"
    elif avg_sentence_length >= 16:
        writing_complexity = "medium"
    else:
        writing_complexity = "low_to_medium"

    formality_score = 0.0
    formality_score += min(modal_counts["mandatory"] / max(total_sentences, 1), 1.0) * 0.30
    formality_score += min(passive_count / max(total_sentences, 1), 1.0) * 0.20
    formality_score += min(nominalisation_count / max(total_tokens / 100, 1), 1.0) * 0.20
    formality_score += (0.15 if contractions == 0 else 0.0)
    formality_score += (0.15 if len(headings) >= 4 else 0.0)

    if formality_score >= 0.70:
        formality = "highly_formal"
    elif formality_score >= 0.45:
        formality = "formal"
    else:
        formality = "standard"

    if directive_wording.startswith("mandatory") and formality in {"formal", "highly_formal"}:
        tone = "formal_regulatory"
    elif directive_wording.startswith("imperative"):
        tone = "instructional_procedural"
    elif acronym_count / total_tokens > 0.07:
        tone = "technical_operational"
    else:
        tone = "mixed_descriptive"

    return {
        "formality": {"value": formality, "score": _safe_round(formality_score, 3)},
        "tone": tone,
        "directive_wording": directive_wording,
        "modal_verbs": modal_counts,
        "action_verb_counts": action_counts,
        "writing_complexity": writing_complexity,
        "complexity_signals": {
            "avg_sentence_length": _safe_round(avg_sentence_length, 2),
            "avg_line_words": _safe_round(avg_line_words, 2),
            "passive_sentence_ratio": _safe_round(passive_count / total_sentences, 3),
            "imperative_sentence_ratio": _safe_round(imperative_count / total_sentences, 3),
            "acronym_density": _safe_round(acronym_count / total_tokens, 3),
            "nominalisation_density": _safe_round(nominalisation_count / total_tokens, 3),
            "contractions_count": contractions,
        },
        "structure_signals": {
            "section_count": len(headings),
            "bullet_count": bullet_count,
            "numbered_step_count": numbered_count,
            "table_like_line_count": table_count,
            "form_field_line_count": form_count,
            "primary_format": _infer_primary_format(bullet_count, numbered_count, table_count, form_count, len(headings)),
        },
    }


def _infer_primary_format(bullets: int, numbered: int, tables: int, forms: int, sections: int) -> str:
    if tables >= max(5, numbered, bullets):
        return "table_dominant"
    if numbered >= 3 and sections >= 3:
        return "controlled_numbered_sop"
    if bullets >= 5:
        return "checklist_or_bullet_sop"
    if forms >= 5:
        return "form_based_sop"
    if sections >= 3:
        return "sectioned_sop"
    return "free_prose_or_short_instruction"


# -----------------------------------------------------------------------------
# Stage 4: Roles and RACI-style detection
# -----------------------------------------------------------------------------

def extract_roles_raci(text: str, sections: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    sections = sections or detect_sections(text)
    role_results = {}
    lower = text.lower()

    responsibility_sections = [s for s in sections if s.get("label") == "responsibilities"]
    responsibility_text = "\n".join(s.get("content", "") for s in responsibility_sections) or text

    for role, variants in ROLE_BANKS.items():
        hits = [v for v in variants if re.search(r"\b" + re.escape(v.lower()) + r"\b", lower)]
        if not hits:
            continue
        evidence = _extract_evidence(responsibility_text, hits + ACTION_VERBS, max_items=4) or _extract_evidence(text, hits, max_items=3)
        actions = _extract_role_actions(text, hits)
        raci = _infer_raci_category(role, actions, evidence)
        role_results[role] = {
            "detected": True,
            "confidence": _confidence_from_hits(len(hits) + len(actions), possible=8, base=0.45),
            "matched_terms": sorted(set(hits)),
            "responsibility_actions": actions,
            "raci_category": raci,
            "evidence": evidence,
        }

    # Extract capitalized custom roles not covered by bank.
    custom_roles = extract_custom_roles(text)
    for r in custom_roles:
        key = r["role"]
        if key not in role_results:
            role_results[key] = r

    missing_expected_roles = []
    for expected in ["QA", "Reviewer", "Approver"]:
        if expected not in role_results:
            missing_expected_roles.append(expected)

    return {
        "roles": role_results,
        "detected_role_count": len(role_results),
        "missing_expected_roles": missing_expected_roles,
        "raci_summary": _summarise_raci(role_results),
    }


def _extract_role_actions(text: str, role_terms: Sequence[str]) -> List[str]:
    actions = []
    sentences = _split_sentences(text)
    for s in sentences:
        sl = s.lower()
        if any(rt.lower() in sl for rt in role_terms):
            for verb in ACTION_VERBS:
                if re.search(r"\b" + re.escape(verb) + r"(?:s|ed|ing)?\b", sl):
                    actions.append(verb)
    return sorted(set(actions))[:12]


def _infer_raci_category(role: str, actions: Sequence[str], evidence: Sequence[str]) -> str:
    joined = " ".join(actions).lower() + " " + " ".join(evidence).lower()
    if role in {"Approver"} or re.search(r"\b(approve|authorize|sign|release)\b", joined):
        return "Accountable/Approver"
    if role in {"Reviewer", "QA", "QC", "Auditor"} or re.search(r"\b(review|verify|audit|assess)\b", joined):
        return "Reviewer/Verifier"
    if role in {"Technician", "Production", "IT", "OT"} or re.search(r"\b(perform|execute|implement|record|submit)\b", joined):
        return "Responsible/Executor"
    if re.search(r"\b(notify|inform|consult)\b", joined):
        return "Consulted/Informed"
    return "Mentioned/Unclear"


def extract_custom_roles(text: str) -> List[Dict[str, Any]]:
    patterns = [
        r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+(?:shall|must|will|is responsible for|is accountable for)\b",
        r"\b(?:role|owner|responsible person)\s*[:#-]\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b",
    ]
    ignored = {"The", "This", "Procedure", "Purpose", "Scope", "Records"}
    found = []
    for pat in patterns:
        for m in re.finditer(pat, text or "", re.I):
            role = _normalise_space(m.group(1))
            if role in ignored or len(role) < 3:
                continue
            ev = _extract_evidence(text, [role], max_items=2)
            found.append({
                "detected": True,
                "confidence": 0.55,
                "matched_terms": [role],
                "responsibility_actions": _extract_role_actions(text, [role]),
                "raci_category": "Custom Role / Needs Review",
                "evidence": ev,
            })
    # deduplicate
    output = []
    seen = set()
    for item in found:
        role = item["matched_terms"][0]
        if role.lower() not in seen:
            seen.add(role.lower())
            item["role"] = role
            output.append(item)
    return output[:15]


def _summarise_raci(role_results: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    summary = defaultdict(list)
    for role, details in role_results.items():
        summary[details.get("raci_category", "Mentioned/Unclear")].append(role)
    return {k: sorted(v) for k, v in summary.items()}


# -----------------------------------------------------------------------------
# Stage 5: Workflows
# -----------------------------------------------------------------------------

def extract_workflows(text: str, sections: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    sections = sections or detect_sections(text)
    sentences = _split_sentences(text)
    section_text_by_label = defaultdict(str)
    for s in sections:
        section_text_by_label[s.get("label", "general")] += "\n" + s.get("content", "")

    workflows = {}
    for flow_name, stages in FLOW_DEFINITIONS.items():
        preferred_text = _workflow_scope_text(flow_name, text, section_text_by_label)
        detected_stages = []
        for stage in stages:
            ev = _extract_evidence(preferred_text, [stage], max_items=2)
            if ev:
                detected_stages.append({
                    "stage": stage,
                    "confidence": _confidence_from_hits(len(ev), possible=2, base=0.55),
                    "evidence": ev,
                    "timing": extract_timing_mentions(" ".join(ev)),
                    "roles": _roles_in_text(" ".join(ev)),
                })
        sequence_score = len(detected_stages) / max(len(stages), 1)
        workflows[flow_name] = {
            "detected": sequence_score >= 0.18,
            "confidence": _safe_round(min(0.95, 0.25 + sequence_score), 3),
            "sequence_score": _safe_round(sequence_score, 3),
            "stages_detected": detected_stages,
            "missing_core_stages": _missing_core_flow_stages(flow_name, [s["stage"] for s in detected_stages]),
            "has_timing": any(x.get("timing") for x in detected_stages),
            "has_role_assignment": any(x.get("roles") for x in detected_stages),
        }
    return workflows


def _workflow_scope_text(flow_name: str, full_text: str, section_text_by_label: Dict[str, str]) -> str:
    mapping = {
        "approval_flow": ["approval", "review", "revision_history", "procedure"],
        "incident_flow": ["incident", "deviation", "escalation", "procedure"],
        "review_flow": ["review", "approval", "revision_history", "procedure"],
        "capa_flow": ["capa", "deviation", "audit", "procedure"],
        "audit_flow": ["audit", "capa", "procedure"],
        "change_control_flow": ["procedure", "approval", "risk_controls", "review"],
    }
    chunks = [section_text_by_label.get(lbl, "") for lbl in mapping.get(flow_name, [])]
    scoped = "\n".join(c for c in chunks if c.strip())
    return scoped if len(scoped.split()) >= 25 else full_text


def _missing_core_flow_stages(flow_name: str, detected: Sequence[str]) -> List[str]:
    core = {
        "approval_flow": ["submit", "review", "approve", "release"],
        "incident_flow": ["report", "classify", "investigate", "escalate", "close"],
        "review_flow": ["review", "update", "approve", "record"],
        "capa_flow": ["root cause", "corrective action", "effectiveness", "close"],
        "audit_flow": ["plan", "conduct", "finding", "report", "close"],
        "change_control_flow": ["request", "impact assessment", "approve", "implement", "verify"],
    }
    detected_l = {d.lower() for d in detected}
    return [s for s in core.get(flow_name, []) if s.lower() not in detected_l]


def extract_timing_mentions(text: str) -> List[str]:
    return sorted(set(_normalise_space(m.group(0)) for m in TIMING_PATTERN.finditer(text or "")))[:20]


def _roles_in_text(text: str) -> List[str]:
    roles = []
    lower = (text or "").lower()
    for role, variants in ROLE_BANKS.items():
        if any(re.search(r"\b" + re.escape(v.lower()) + r"\b", lower) for v in variants):
            roles.append(role)
    return sorted(set(roles))


# -----------------------------------------------------------------------------
# Stage 6: Compliance elements and risk/gap detection
# -----------------------------------------------------------------------------

def detect_compliance_elements(text: str, sections: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    sections = sections or detect_sections(text)
    labels = [s.get("label") for s in sections]
    standards = detect_regulatory_standards(text)
    trace_ids = sorted(set(m.group(0) for m in TRACE_ID_PATTERN.finditer(text or "")))
    timing_mentions = extract_timing_mentions(text)
    controls = _keyword_hits(text, CONTROL_KEYWORDS)
    escalation = _keyword_hits(text, ESCALATION_KEYWORDS)
    records = _keyword_hits(text, ["record", "log", "evidence", "form", "retention", "audit trail", "documentation"])
    training = _keyword_hits(text, ["training", "competency", "qualified", "qualification", "training record"])
    data_integrity = _keyword_hits(text, ["alcoa", "data integrity", "audit trail", "electronic signature", "part 11", "access control"])

    return {
        "standards_detected": standards,
        "traceability_ids": trace_ids,
        "section_labels_detected": sorted(set(l for l in labels if l and l != "general")),
        "timing_mentions": timing_mentions,
        "control_terms": controls,
        "escalation_terms": escalation,
        "recordkeeping_terms": records,
        "training_terms": training,
        "data_integrity_terms": data_integrity,
        "compliance_strength_score": _score_compliance_strength(standards, trace_ids, timing_mentions, controls, records, labels),
    }


def _score_compliance_strength(standards, trace_ids, timing, controls, records, labels) -> Dict[str, Any]:
    score = 0
    score += min(len(standards), 2) * 12
    score += min(len(trace_ids), 5) * 4
    score += min(len(timing), 5) * 4
    score += min(len(controls), 6) * 5
    score += min(len(records), 5) * 4
    score += min(len(set(labels) & {"purpose", "scope", "responsibilities", "procedure", "approval", "records"}), 6) * 5
    score = min(score, 100)
    grade = "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D"
    return {"score": score, "grade": grade}


def detect_risks_and_gaps(
    text: str,
    document_info: Dict[str, Any],
    sections: List[Dict[str, Any]],
    roles: Dict[str, Any],
    workflows: Dict[str, Any],
    compliance: Dict[str, Any],
) -> Dict[str, Any]:
    labels = {s.get("label") for s in sections}
    sop_type = document_info.get("sop_type", {}).get("value", "default")
    required = REQUIRED_ELEMENTS_BY_TYPE.get(sop_type, REQUIRED_ELEMENTS_BY_TYPE["default"])

    gaps = {
        "missing_information": [],
        "missing_approvals": [],
        "missing_timing": [],
        "missing_escalation_logic": [],
        "missing_controls": [],
        "role_gaps": [],
        "workflow_gaps": [],
        "structure_gaps": [],
    }

    # Required sections / information.
    for req in required:
        if req not in labels:
            gaps["missing_information"].append(_gap(req, "Required SOP section or element not detected", "high" if req in {"purpose", "scope", "procedure", "responsibilities"} else "medium"))

    if not document_info.get("title"):
        gaps["missing_information"].append(_gap("title", "Document title could not be detected", "medium"))
    if not document_info.get("version_or_revision"):
        gaps["missing_information"].append(_gap("version_or_revision", "Version/revision information is missing or unclear", "medium"))

    # Approval logic.
    approval_flow = workflows.get("approval_flow", {})
    has_approval_section = "approval" in labels
    has_approver_role = "Approver" in roles.get("roles", {}) or _has_any(text, ["approved by", "approver", "authorized by", "sign-off"])
    if not has_approval_section and not has_approver_role:
        gaps["missing_approvals"].append(_gap("approval_authority", "No clear approval section or approver authority detected", "high"))
    if approval_flow.get("detected") and approval_flow.get("missing_core_stages"):
        for stage in approval_flow["missing_core_stages"]:
            gaps["missing_approvals"].append(_gap(stage, f"Approval flow is missing stage: {stage}", "medium"))

    # Timing rules.
    timing = compliance.get("timing_mentions", [])
    flow_needs_timing = _has_any(text, ["approve", "review", "incident", "deviation", "capa", "escalate", "audit", "periodic"])
    if flow_needs_timing and not timing:
        gaps["missing_timing"].append(_gap("timing", "No clear timing/SLA/frequency found for controlled actions", "high"))
    for flow_name, wf in workflows.items():
        if wf.get("detected") and not wf.get("has_timing") and flow_name in {"incident_flow", "capa_flow", "approval_flow", "review_flow"}:
            gaps["missing_timing"].append(_gap(flow_name, f"{flow_name} detected but no step-level timing found", "medium"))

    # Escalation.
    risk_context = _has_any(text, ["incident", "deviation", "critical", "overdue", "failure", "nonconformance", "security event", "capa"])
    has_escalation = bool(compliance.get("escalation_terms"))
    if risk_context and not has_escalation:
        gaps["missing_escalation_logic"].append(_gap("escalation", "Risk/event context detected but escalation path is missing", "high"))

    # Controls.
    if len(compliance.get("control_terms", [])) < 2:
        gaps["missing_controls"].append(_gap("controls", "Not enough verification/control language detected", "medium"))
    if _has_any(text, ["capa", "corrective action", "preventive action"]) and not _has_any(text, ["effectiveness", "verify effectiveness", "effectiveness check"]):
        gaps["missing_controls"].append(_gap("capa_effectiveness", "CAPA is mentioned but effectiveness check is missing", "high"))
    if _has_any(text, ["access", "user", "privilege", "password"]) and not _has_any(text, ["periodic review", "access review", "mfa", "least privilege", "rbac"]):
        gaps["missing_controls"].append(_gap("access_control", "Access-related SOP lacks strong access control/review language", "high"))

    # Role gaps.
    detected_roles = roles.get("roles", {})
    for expected in ["QA", "Reviewer", "Approver"]:
        if expected not in detected_roles:
            gaps["role_gaps"].append(_gap(expected, f"Expected role not clearly detected: {expected}", "medium"))
    if sop_type in {"CAPA SOP", "Deviation / Incident SOP"} and "CAPA Owner" not in detected_roles and not _has_any(text, ["action owner", "owner"]):
        gaps["role_gaps"].append(_gap("owner", "No CAPA/action owner clearly assigned", "high"))

    # Workflow gaps.
    for flow_name, wf in workflows.items():
        if wf.get("detected") and wf.get("missing_core_stages"):
            severity = "high" if flow_name in {"incident_flow", "capa_flow"} else "medium"
            for stage in wf["missing_core_stages"][:5]:
                gaps["workflow_gaps"].append(_gap(f"{flow_name}:{stage}", f"{flow_name} missing core stage: {stage}", severity))

    # Structure.
    if len(sections) < 3:
        gaps["structure_gaps"].append(_gap("sectioning", "Document has weak section structure for a controlled SOP", "medium"))

    flat = [item for group in gaps.values() for item in group]
    risk_score = _calculate_risk_score(flat)
    return {
        "gaps": gaps,
        "risk_score": risk_score,
        "gap_count": len(flat),
        "critical_focus_areas": _top_focus_areas(flat),
    }


def _gap(code: str, message: str, severity: str) -> Dict[str, Any]:
    return {"code": code, "message": message, "severity": severity}


def _calculate_risk_score(gaps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    weights = {"low": 2, "medium": 5, "high": 9, "critical": 12}
    raw = sum(weights.get(g.get("severity", "medium"), 5) for g in gaps)
    score = min(100, raw)
    if score >= 70:
        level = "high"
    elif score >= 35:
        level = "medium"
    elif score > 0:
        level = "low"
    else:
        level = "controlled"
    return {"score": score, "level": level}


def _top_focus_areas(gaps: Sequence[Dict[str, Any]]) -> List[str]:
    high = [g["message"] for g in gaps if g.get("severity") in {"critical", "high"}]
    return high[:5]


# -----------------------------------------------------------------------------
# Stage 7: Terminology, style suggestions, profile system
# -----------------------------------------------------------------------------

def extract_terminology(text: str) -> Dict[str, Any]:
    acronyms = sorted(set(re.findall(r"\b[A-Z]{2,}(?:/[A-Z]{2,})?\b", text or "")))
    trace_ids = sorted(set(m.group(0) for m in TRACE_ID_PATTERN.finditer(text or "")))
    domain_terms = []
    for terms in DOMAIN_BANKS.values():
        domain_terms.extend(_keyword_hits(text, terms))
    controlled_terms = _keyword_hits(text, [
        "shall", "must", "approval", "effective date", "revision", "deviation", "capa", "audit trail",
        "root cause", "effectiveness check", "controlled copy", "retention", "training record", "risk assessment",
    ])
    definitions = extract_definitions(text)
    return {
        "acronyms": acronyms[:80],
        "traceability_ids": trace_ids[:100],
        "domain_terms": sorted(set(domain_terms))[:100],
        "controlled_terms": sorted(set(controlled_terms))[:100],
        "definitions": definitions,
    }


def extract_definitions(text: str) -> List[Dict[str, str]]:
    results = []
    patterns = [
        r"^\s*([A-Z][A-Za-z0-9 /-]{1,40})\s*[:=-]\s*(.{5,200})$",
        r"\b([A-Z][A-Za-z0-9 /-]{1,40})\s+means\s+(.{5,200}?)(?:\.|;|\n)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text or "", re.I | re.M):
            term = _normalise_space(m.group(1))
            definition = _normalise_space(m.group(2))
            if len(term.split()) <= 6 and len(definition.split()) >= 3:
                results.append({"term": term, "definition": definition[:300]})
    # deduplicate
    out = []
    seen = set()
    for r in results:
        k = r["term"].lower()
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out[:40]


def generate_style_suggestions(style: Dict[str, Any], gaps: Dict[str, Any]) -> List[Dict[str, str]]:
    suggestions = []
    formality = style.get("formality", {}).get("value")
    modal = style.get("modal_verbs", {})
    if modal.get("mandatory", 0) == 0:
        suggestions.append({"area": "Directive wording", "suggestion": "Use controlled mandatory wording such as shall/must for required actions."})
    if style.get("writing_complexity") == "high":
        suggestions.append({"area": "Writing complexity", "suggestion": "Split long sentences into shorter actor-action-object instructions."})
    if style.get("structure_signals", {}).get("section_count", 0) < 4:
        suggestions.append({"area": "Structure", "suggestion": "Add standard SOP sections: Purpose, Scope, Responsibilities, Procedure, Records, Approval, Revision History."})
    if formality in {"standard"}:
        suggestions.append({"area": "Formality", "suggestion": "Increase formal SOP tone by reducing conversational language and adding clear responsibilities."})
    if gaps.get("risk_score", {}).get("level") in {"medium", "high"}:
        suggestions.append({"area": "Compliance gaps", "suggestion": "Resolve high-priority missing approvals, timing, escalation, and control gaps before release."})
    return suggestions


def build_client_profile(
    analysis: Dict[str, Any],
    client_name: str = "Client",
    existing_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    existing_profile = existing_profile or {}
    doc = analysis.get("document_information", {})
    style = analysis.get("writing_style", {})
    terminology = analysis.get("terminology", {})
    workflows = analysis.get("workflows", {})

    profile = {
        "client_name": client_name,
        "profile_version": "1.0",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "detected_domains": _merge_unique(existing_profile.get("detected_domains", []), [doc.get("domain", {}).get("value")]),
        "detected_departments": _merge_unique(existing_profile.get("detected_departments", []), [doc.get("department", {}).get("value")]),
        "detected_sop_types": _merge_unique(existing_profile.get("detected_sop_types", []), [doc.get("sop_type", {}).get("value")]),
        "preferred_style": {
            "formality": style.get("formality", {}).get("value"),
            "tone": style.get("tone"),
            "directive_wording": style.get("directive_wording"),
            "writing_complexity": style.get("writing_complexity"),
            "primary_format": style.get("structure_signals", {}).get("primary_format"),
        },
        "modal_language": style.get("modal_verbs", {}),
        "common_sections": analysis.get("structure_patterns", {}).get("section_labels", []),
        "terminology": {
            "acronyms": terminology.get("acronyms", []),
            "controlled_terms": terminology.get("controlled_terms", []),
            "domain_terms": terminology.get("domain_terms", []),
        },
        "workflow_patterns": {
            name: {
                "detected": wf.get("detected"),
                "stages": [s.get("stage") for s in wf.get("stages_detected", [])],
            }
            for name, wf in workflows.items()
        },
        "rewrite_rules": _generate_rewrite_rules(style, analysis.get("risks_gaps", {})),
    }
    return profile


def _merge_unique(a: Sequence[Any], b: Sequence[Any]) -> List[Any]:
    out = []
    for item in list(a) + list(b):
        if item and item != "UNKNOWN" and item not in out:
            out.append(item)
    return out


def _generate_rewrite_rules(style: Dict[str, Any], gaps: Dict[str, Any]) -> List[str]:
    rules = [
        "Use clear actor-action-object sentences.",
        "Preserve controlled terms, acronyms, IDs, version numbers, and timing commitments.",
        "Do not invent approvals, deadlines, or responsibilities that are not present in the source.",
        "Flag missing compliance elements instead of silently adding them.",
    ]
    if style.get("directive_wording") == "mandatory/controlled":
        rules.append("Prefer shall/must for mandatory steps and may/should only for optional or recommended actions.")
    if gaps.get("gap_count", 0) > 0:
        rules.append("Before final rewrite, list unresolved gaps for owner confirmation.")
    return rules


def generate_profile_md(client_profile: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# {client_profile.get('client_name', 'Client')} SOP Profile")
    lines.append("")
    lines.append(f"Generated: {client_profile.get('generated_at', '')}")
    lines.append(f"Profile version: {client_profile.get('profile_version', '1.0')}")
    lines.append("")
    lines.append("## Detected Document Context")
    lines.append(f"- Domains: {', '.join(client_profile.get('detected_domains', [])) or 'Not detected'}")
    lines.append(f"- Departments: {', '.join(client_profile.get('detected_departments', [])) or 'Not detected'}")
    lines.append(f"- SOP Types: {', '.join(client_profile.get('detected_sop_types', [])) or 'Not detected'}")
    lines.append("")
    style = client_profile.get("preferred_style", {})
    lines.append("## Writing Style")
    for key in ["formality", "tone", "directive_wording", "writing_complexity", "primary_format"]:
        lines.append(f"- {key.replace('_', ' ').title()}: {style.get(key) or 'Not detected'}")
    lines.append("")
    lines.append("## Modal Language")
    for k, v in client_profile.get("modal_language", {}).items():
        lines.append(f"- {k.title()}: {v}")
    lines.append("")
    lines.append("## Common Sections")
    for section in client_profile.get("common_sections", []):
        lines.append(f"- {section}")
    if not client_profile.get("common_sections"):
        lines.append("- Not enough section data detected")
    lines.append("")
    lines.append("## Terminology")
    terms = client_profile.get("terminology", {})
    lines.append(f"- Acronyms: {', '.join(terms.get('acronyms', [])[:40]) or 'None detected'}")
    lines.append(f"- Controlled Terms: {', '.join(terms.get('controlled_terms', [])[:40]) or 'None detected'}")
    lines.append(f"- Domain Terms: {', '.join(terms.get('domain_terms', [])[:40]) or 'None detected'}")
    lines.append("")
    lines.append("## Workflow Patterns")
    for name, wf in client_profile.get("workflow_patterns", {}).items():
        status = "detected" if wf.get("detected") else "not detected"
        stages = ", ".join(wf.get("stages", [])) or "No stages detected"
        lines.append(f"- {name}: {status}; stages: {stages}")
    lines.append("")
    lines.append("## Rewrite Rules")
    for rule in client_profile.get("rewrite_rules", []):
        lines.append(f"- {rule}")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Stage 8: End-to-end analysis
# -----------------------------------------------------------------------------

def analyze_sop_industry_level(
    text: str,
    client_name: str = "Client",
    existing_profile: Optional[Dict[str, Any]] = None,
    include_profile_md: bool = True,
) -> Dict[str, Any]:
    text = text or ""
    language = detect_language(text)
    sections = detect_sections(text)
    document_info = detect_document_information(text, sections)
    writing_style = detect_writing_style(text)
    roles = extract_roles_raci(text, sections)
    workflows = extract_workflows(text, sections)
    compliance = detect_compliance_elements(text, sections)
    risks_gaps = detect_risks_and_gaps(text, document_info, sections, roles, workflows, compliance)
    terminology = extract_terminology(text)
    structure_patterns = {
        "section_count": len(sections),
        "section_labels": sorted(set(s.get("label") for s in sections if s.get("label"))),
        "sections": sections,
    }

    result = {
        "pipeline_version": "industry_sop_nlp_v1.0",
        "analysis_timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "document_information": document_info,
        "language": language,
        "writing_style": writing_style,
        "roles_raci": roles,
        "workflows": workflows,
        "compliance_elements": compliance,
        "risks_gaps": risks_gaps,
        "terminology": terminology,
        "structure_patterns": structure_patterns,
        "style_suggestions": [],
    }
    result["style_suggestions"] = generate_style_suggestions(writing_style, risks_gaps)
    profile = build_client_profile(result, client_name=client_name, existing_profile=existing_profile)
    result["client_profile"] = profile
    if include_profile_md:
        result["profile_md"] = generate_profile_md(profile)
    return result


# Backward-compatible alias names for easy integration.
def analyze_sop(text: str, client_name: str = "Client") -> Dict[str, Any]:
    return analyze_sop_industry_level(text, client_name=client_name)


def analyze_document(text: str, client_name: str = "Client") -> Dict[str, Any]:
    return analyze_sop_industry_level(text, client_name=client_name)


def analyze_sop_file(
    input_path: str,
    output_json_path: Optional[str] = None,
    output_profile_path: Optional[str] = None,
    client_name: str = "Client",
) -> Dict[str, Any]:
    text = Path(input_path).read_text(encoding="utf-8", errors="ignore")
    analysis = analyze_sop_industry_level(text, client_name=client_name, include_profile_md=True)
    if output_json_path:
        Path(output_json_path).write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_profile_path:
        Path(output_profile_path).write_text(analysis.get("profile_md", ""), encoding="utf-8")
    return analysis


# -----------------------------------------------------------------------------
# CLI usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze SOP dynamically using industry-level NLP rules.")
    parser.add_argument("input", help="Path to SOP text file")
    parser.add_argument("--client-name", default="Client", help="Client/profile name")
    parser.add_argument("--json", default=None, help="Output JSON path")
    parser.add_argument("--profile", default=None, help="Output profile.md path")
    args = parser.parse_args()

    result = analyze_sop_file(args.input, args.json, args.profile, client_name=args.client_name)
    print(json.dumps({
        "pipeline_version": result["pipeline_version"],
        "sop_type": result["document_information"].get("sop_type"),
        "department": result["document_information"].get("department"),
        "domain": result["document_information"].get("domain"),
        "risk_score": result["risks_gaps"].get("risk_score"),
        "gap_count": result["risks_gaps"].get("gap_count"),
    }, indent=2, ensure_ascii=False))
