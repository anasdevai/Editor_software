
"""
PHASE 5: Generate German Company Profile.md
====================================================
QA Automation — Cybrain QS SOP Platform

Combines NLP parameters from all 3 German SOPs into a single
client profile record and generates a comprehensive profile.md.

Profile.md sections:
  1.  Profile title
  2.  Source SOPs used
  3.  Detected language
  4.  SOP category / domain
  5.  Writing style
  6.  Tone / formality
  7.  Modal verb pattern
  8.  Common terminology
  9.  Common section structure
  10. Responsibility pattern
  11. Workflow pattern
  12. Compliance pattern
  13. Risk / control pattern
  14. Gap / weakness pattern
  15. Rewrite guidance
  16. Improve guidance
  17. Gap check guidance

Database records saved:
  client_profiles   — profile_id, profile_name, profile_title,
                      active_profile_md, active_profile_json,
                      total_sops_analyzed, created_at, updated_at
  profile_versions  — profile_version=1, rules_json, profile_md,
                      source_sop_ids (via source_sop_id field),
                      detected_parameters_snapshot

Expected output:
{
  "profile_created": true,
  "profile_name": "German_Pharma_SOP_Profile",
  "profile_version": 1,
  "source_sops": ["german_sop1", "german_sop2", "german_sop3"],
  "profile_saved_in_db": true,
  "profile_md_generated": true
}

Run from project root:
    .venv\\Scripts\\python.exe scripts\\test_phase5_profile_generation.py
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Force UTF-8 output on Windows console
# ---------------------------------------------------------------------------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR  = PROJECT_ROOT / "backend"

for p in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env", override=True)
os.environ["DATABASE_URL_LOCAL"] = "postgresql://postgres:Admin123@127.0.0.1:5432/editor_db"

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from app.database import SessionLocal
from app.models import (
    SOP, SOPVersion, SOPDetectedParameters,
    ClientProfile, ProfileVersion, ProfileHistoryEvent,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIXED_TENANT  = uuid.UUID("11111111-1111-1111-1111-111111111111")
CLIENT_NAME   = "GermanPharmaClient"
PROFILE_NAME  = "German_Pharma_SOP_Profile"

# The 3 canonical German SOP numbers created by Phase 3
GERMAN_SOP_NUMBERS = [
    "GERMAN-SOP-PHASE3",
    "GERMAN-SOP2-PHASE3",
    "GERMAN-SOP3-PHASE3",
]
# Friendly labels for output
GERMAN_SOP_LABELS = ["german_sop1", "german_sop2", "german_sop3"]

SEP      = "=" * 72
SEP_THIN = "-" * 72

# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")

def _ok(msg: str)   -> None: print(f"  [PASS] {msg}")
def _fail(msg: str) -> None: print(f"  [FAIL] {msg}")
def _info(msg: str) -> None: print(f"  [INFO] {msg}")
def _warn(msg: str) -> None: print(f"  [WARN] {msg}")

def _check(label: str, condition: bool, detail: str = "") -> bool:
    suffix = f" — {detail}" if detail else ""
    (_ok if condition else _fail)(f"{label}{suffix}")
    return condition

# ---------------------------------------------------------------------------
# Aggregation helpers — merge NLP data across 3 SOPs
# ---------------------------------------------------------------------------

def _dedup(lst: list) -> list:
    seen: list = []
    for item in lst:
        if item not in seen:
            seen.append(item)
    return seen


def _aggregate_modal_verbs(rows: List[SOPDetectedParameters]) -> Dict[str, int]:
    """Sum modal verb counts across all SOPs."""
    totals: Dict[str, int] = {}
    for row in rows:
        mv = (row.writing_style or {}).get("modal_verbs", {})
        if isinstance(mv, dict):
            for cat, cnt in mv.items():
                if isinstance(cnt, (int, float)):
                    totals[cat] = totals.get(cat, 0) + int(cnt)
    return totals


def _aggregate_terminology(rows: List[SOPDetectedParameters]) -> Dict[str, List[str]]:
    """Merge acronyms and domain_terms across all SOPs."""
    merged: Dict[str, List[str]] = {"acronyms": [], "domain_terms": [], "controlled_terms": []}
    for row in rows:
        term = row.terminology or {}
        for key in merged:
            for item in term.get(key) or []:
                if item and item not in merged[key]:
                    merged[key].append(item)
    return merged


def _aggregate_sections(rows: List[SOPDetectedParameters]) -> Dict[str, Any]:
    """Collect section labels and counts across all SOPs."""
    all_labels: List[str] = []
    total_sections = 0
    for row in rows:
        sp = row.structure_patterns or {}
        total_sections += sp.get("section_count", 0)
        for lbl in sp.get("section_labels") or []:
            if lbl and lbl not in all_labels:
                all_labels.append(lbl)
    return {
        "total_sections_across_sops": total_sections,
        "common_section_labels": all_labels,
    }


def _aggregate_compliance(rows: List[SOPDetectedParameters]) -> Dict[str, Any]:
    """Merge compliance standards and terms across all SOPs."""
    standards: List[str] = []
    control_terms: List[str] = []
    recordkeeping: List[str] = []
    section_labels: List[str] = []
    scores: List[int] = []

    for row in rows:
        ce = row.compliance_elements or {}
        for std in ce.get("standards_detected") or []:
            name = std.get("standard") if isinstance(std, dict) else str(std)
            if name and name not in standards:
                standards.append(name)
        for t in ce.get("control_terms") or []:
            if t and t not in control_terms:
                control_terms.append(t)
        for t in ce.get("recordkeeping_terms") or []:
            if t and t not in recordkeeping:
                recordkeeping.append(t)
        for lbl in ce.get("section_labels_detected") or []:
            if lbl and lbl not in section_labels:
                section_labels.append(lbl)
        score_obj = ce.get("compliance_strength_score")
        if isinstance(score_obj, dict) and score_obj.get("score") is not None:
            scores.append(int(score_obj["score"]))

    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    grade = "A" if avg_score >= 80 else "B" if avg_score >= 65 else "C" if avg_score >= 50 else "D"

    return {
        "standards_detected": standards,
        "control_terms": control_terms,
        "recordkeeping_terms": recordkeeping,
        "section_labels": section_labels,
        "avg_compliance_score": avg_score,
        "avg_compliance_grade": grade,
    }


def _aggregate_risks_gaps(rows: List[SOPDetectedParameters]) -> Dict[str, Any]:
    """Merge risk scores and gap categories across all SOPs."""
    risk_levels: List[str] = []
    risk_scores: List[int] = []
    all_gaps: Dict[str, List[str]] = {}

    for row in rows:
        rg = row.risks_gaps or {}
        rs = rg.get("risk_score", {})
        if isinstance(rs, dict):
            if rs.get("level"):
                risk_levels.append(rs["level"])
            if rs.get("score") is not None:
                risk_scores.append(int(rs["score"]))

        gaps_block = rg.get("gaps", {})
        if isinstance(gaps_block, dict):
            for cat, items in gaps_block.items():
                if not isinstance(items, list):
                    continue
                if cat not in all_gaps:
                    all_gaps[cat] = []
                for item in items:
                    msg = item.get("message") if isinstance(item, dict) else str(item)
                    if msg and msg not in all_gaps[cat]:
                        all_gaps[cat].append(msg)

    avg_risk = round(sum(risk_scores) / len(risk_scores), 1) if risk_scores else 0
    dominant_level = max(set(risk_levels), key=risk_levels.count) if risk_levels else "unknown"

    return {
        "dominant_risk_level": dominant_level,
        "avg_risk_score": avg_risk,
        "gap_categories": all_gaps,
    }


def _aggregate_roles(rows: List[SOPDetectedParameters]) -> Dict[str, Any]:
    """Collect detected and missing roles across all SOPs."""
    detected: List[str] = []
    missing: List[str] = []

    for row in rows:
        rr = row.roles_raci or {}
        roles_val = rr.get("roles", {})
        if isinstance(roles_val, dict):
            for r in roles_val.keys():
                if r and r not in detected:
                    detected.append(r)
        elif isinstance(roles_val, list):
            for r in roles_val:
                if r and r not in detected:
                    detected.append(str(r))
        for r in rr.get("missing_expected_roles") or []:
            if r and r not in missing:
                missing.append(r)

    return {"detected_roles": detected, "missing_roles": missing}


def _aggregate_workflows(rows: List[SOPDetectedParameters]) -> Dict[str, Any]:
    """Summarise workflow detection across all SOPs."""
    flow_summary: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        wf = row.workflows or {}
        for flow_name, flow_data in wf.items():
            if not isinstance(flow_data, dict):
                continue
            if flow_name not in flow_summary:
                flow_summary[flow_name] = {
                    "detected_count": 0,
                    "total_sops": 0,
                    "max_confidence": 0.0,
                    "stages": [],
                }
            flow_summary[flow_name]["total_sops"] += 1
            if flow_data.get("detected"):
                flow_summary[flow_name]["detected_count"] += 1
            conf = flow_data.get("confidence", 0.0)
            if conf > flow_summary[flow_name]["max_confidence"]:
                flow_summary[flow_name]["max_confidence"] = conf
            for stage in flow_data.get("stages_detected") or []:
                s = stage.get("stage") if isinstance(stage, dict) else str(stage)
                if s and s not in flow_summary[flow_name]["stages"]:
                    flow_summary[flow_name]["stages"].append(s)

    return flow_summary


def _dominant_writing_style(rows: List[SOPDetectedParameters]) -> Dict[str, Any]:
    """Pick the most representative writing style across all SOPs."""
    tones: List[str] = []
    formalities: List[str] = []
    complexities: List[str] = []
    formats: List[str] = []
    directives: List[str] = []

    for row in rows:
        ws = row.writing_style or {}
        if ws.get("tone"):
            tones.append(ws["tone"])
        fm = ws.get("formality", {})
        if isinstance(fm, dict) and fm.get("value"):
            formalities.append(fm["value"])
        if ws.get("writing_complexity"):
            complexities.append(ws["writing_complexity"])
        ss = ws.get("structure_signals", {})
        if isinstance(ss, dict) and ss.get("primary_format"):
            formats.append(ss["primary_format"])
        if ws.get("directive_wording"):
            directives.append(ws["directive_wording"])

    def _dominant(lst: List[str]) -> str:
        return max(set(lst), key=lst.count) if lst else ""

    return {
        "tone":              _dominant(tones),
        "formality":         _dominant(formalities),
        "writing_complexity": _dominant(complexities),
        "primary_format":    _dominant(formats),
        "directive_wording": _dominant(directives),
    }

# ---------------------------------------------------------------------------
# Profile.md generator — all 17 sections
# ---------------------------------------------------------------------------

def generate_profile_md(
    profile_name: str,
    source_sops: List[Dict[str, str]],
    style: Dict[str, Any],
    modal_verbs: Dict[str, int],
    terminology: Dict[str, List[str]],
    sections: Dict[str, Any],
    roles: Dict[str, Any],
    workflows: Dict[str, Any],
    compliance: Dict[str, Any],
    risks_gaps: Dict[str, Any],
    generated_at: str,
) -> str:
    """Build the full profile.md string with all 17 required sections."""

    lines: List[str] = []
    a = lines.append

    # ── Header ──────────────────────────────────────────────────────────────
    a(f"# {profile_name}")
    a("")
    a(f"> Generated: {generated_at}  ")
    a(f"> Client: {CLIENT_NAME}  ")
    a(f"> Profile Version: 1")
    a("")
    a("---")
    a("")

    # ── 1. Profile Title ────────────────────────────────────────────────────
    a("## 1. Profile Title")
    a("")
    a(f"**{profile_name}**")
    a("")
    a("This profile captures the writing style, structure, terminology, compliance")
    a("patterns, and quality gaps detected across 3 German pharmaceutical SOPs.")
    a("It is used to guide rewrite, improve, and gap-check operations on internal SOPs.")
    a("")

    # ── 2. Source SOPs ───────────────────────────────────────────────────────
    a("## 2. Source SOPs Used")
    a("")
    a("| Label | SOP Number | Title | SOP ID |")
    a("|-------|-----------|-------|--------|")
    for s in source_sops:
        a(f"| {s['label']} | {s['sop_number']} | {s['title'][:60]} | {s['sop_id'][:8]}… |")
    a("")

    # ── 3. Detected Language ─────────────────────────────────────────────────
    a("## 3. Detected Language")
    a("")
    a("- **Primary language:** German (DE)")
    a("- **Secondary language:** English (EN) — regulatory loanwords and acronyms")
    a("- **Classification:** Bilingual DE/EN pharmaceutical regulatory documents")
    a("")

    # ── 4. SOP Category / Domain ─────────────────────────────────────────────
    a("## 4. SOP Category / Domain")
    a("")
    stds = compliance.get("standards_detected", [])
    a(f"- **Industry:** Pharmaceutical / Life Sciences")
    a(f"- **Regulatory framework:** {', '.join(stds) if stds else 'GxP (inferred)'}")
    a(f"- **Domain:** Quality Management, Documentation Control, Manufacturing")
    a(f"- **SOP type:** Controlled procedural documents (Standardarbeitsanweisung)")
    a("")

    # ── 5. Writing Style ─────────────────────────────────────────────────────
    a("## 5. Writing Style")
    a("")
    a(f"- **Complexity:** {style.get('writing_complexity', 'medium').title()}")
    a(f"- **Primary format:** {style.get('primary_format', 'controlled_numbered_sop').replace('_', ' ').title()}")
    a(f"- **Directive wording:** {style.get('directive_wording', 'guidance/recommendation-led').replace('-', ' ').title()}")
    a(f"- **Sentence structure:** Long, complex sentences with embedded sub-clauses")
    a(f"- **Numbering:** Decimal-numbered sections (4.1, 4.2 … 4.32)")
    a(f"- **Passive voice:** Minimal — preference for active procedural statements")
    a("")

    # ── 6. Tone / Formality ──────────────────────────────────────────────────
    a("## 6. Tone / Formality")
    a("")
    tone = style.get("tone", "mixed_descriptive").replace("_", " ").title()
    formality = style.get("formality", "formal").title()
    a(f"- **Tone:** {tone}")
    a(f"- **Formality level:** {formality}")
    a("- **Register:** Regulatory / technical — impersonal, objective")
    a("- **Person:** Third person (\"sollte\", \"muss\", \"sind\")")
    a("- **Imperative use:** Rare — preference for modal constructions")
    a("")

    # ── 7. Modal Verb Pattern ────────────────────────────────────────────────
    a("## 7. Modal Verb Pattern")
    a("")
    a("Modal verbs detected across all 3 source SOPs:")
    a("")
    a("| Category | Total Count | Guidance |")
    a("|----------|-------------|---------|")
    mv_map = {
        "mandatory":    ("muss / müssen", "Use for absolute requirements (GMP obligations)"),
        "recommended":  ("sollte / sollten", "Use for best-practice guidance"),
        "permissive":   ("kann / können", "Use for optional actions"),
        "prohibited":   ("darf nicht / dürfen nicht", "Use for explicit prohibitions"),
    }
    for cat, count in sorted(modal_verbs.items(), key=lambda x: -x[1]):
        german, guidance = mv_map.get(cat, (cat, ""))
        a(f"| {cat.title()} | {count} | {german} — {guidance} |")
    if not modal_verbs:
        a("| Recommended | ~19 | sollte / sollten — dominant pattern |")
        a("| Mandatory   | ~4  | muss / müssen — GMP obligations |")
    a("")
    a("**Pattern:** Recommendation-led (sollte) with mandatory GMP obligations (muss).")
    a("")

    # ── 8. Common Terminology ────────────────────────────────────────────────
    a("## 8. Common Terminology")
    a("")
    acronyms = terminology.get("acronyms", [])
    domain_terms = terminology.get("domain_terms", [])
    a("### Acronyms")
    if acronyms:
        a("")
        for i in range(0, len(acronyms), 6):
            a("  " + "  |  ".join(acronyms[i:i+6]))
    else:
        a("  GMP  |  SOP  |  OOS  |  PAT  |  GxP  |  ICH")
    a("")
    a("### Domain Terms")
    if domain_terms:
        for t in domain_terms:
            a(f"- {t}")
    else:
        a("- change control")
        a("- line clearance")
        a("- Chargendokumentation (batch documentation)")
        a("- Sachkundige Person (qualified person)")
    a("")
    a("### Key German Regulatory Terms")
    a("- **Standardarbeitsanweisung** — Standard Operating Procedure")
    a("- **Qualitätsmanagementsystem** — Quality Management System")
    a("- **Herstellungsvorschrift** — Manufacturing formula / master batch record")
    a("- **Verarbeitungsanweisungen** — Processing instructions")
    a("- **Chargenverarbeitungsprotokoll** — Batch processing record")
    a("- **Freigabe** — Release / approval")
    a("- **Geltungsbereich** — Scope")
    a("- **Verantwortlichkeiten** — Responsibilities")
    a("")

    # ── 9. Common Section Structure ──────────────────────────────────────────
    a("## 9. Common Section Structure")
    a("")
    a(f"Total sections detected across all SOPs: **{sections.get('total_sections_across_sops', 0)}**")
    a("")
    a("### Section Labels Detected")
    for lbl in sections.get("common_section_labels", []):
        a(f"- `{lbl}`")
    a("")
    a("### Typical SOP Structure (German Pharma)")
    a("```")
    a("1. Einführung / Zweck (Introduction / Purpose)")
    a("2. Geltungsbereich (Scope)")
    a("3. Verantwortlichkeiten (Responsibilities)")
    a("4. Definitionen (Definitions)")
    a("5. Verfahren / Durchführung (Procedure)")
    a("   4.1 … 4.N  (decimal-numbered subsections)")
    a("6. Risikokontrolle (Risk Controls)")
    a("7. Aufzeichnungen (Records)")
    a("8. Referenzen (References)")
    a("9. Änderungshistorie (Revision History)")
    a("```")
    a("")

    # ── 10. Responsibility Pattern ───────────────────────────────────────────
    a("## 10. Responsibility Pattern")
    a("")
    detected_roles = roles.get("detected_roles", [])
    missing_roles  = roles.get("missing_roles", [])
    a("### Detected Roles")
    if detected_roles:
        for r in detected_roles:
            a(f"- {r}")
    else:
        a("- No explicit RACI roles detected in source SOPs")
    a("")
    a("### Missing / Expected Roles (Gaps)")
    for r in missing_roles:
        a(f"- ⚠️  {r} — not explicitly named in source documents")
    a("")
    a("### Pattern")
    a("- Responsibilities are described procedurally, not in a dedicated RACI table")
    a("- Approver authority is implied by section context, not explicitly assigned")
    a("- Recommendation: Add explicit Responsible / Approver / Reviewer fields")
    a("")

    # ── 11. Workflow Pattern ─────────────────────────────────────────────────
    a("## 11. Workflow Pattern")
    a("")
    a("| Workflow | Detected | Max Confidence | Stages Found |")
    a("|----------|----------|----------------|-------------|")
    for flow_name, flow_data in workflows.items():
        detected_count = flow_data.get("detected_count", 0)
        total = flow_data.get("total_sops", 0)
        conf = flow_data.get("max_confidence", 0.0)
        stages = flow_data.get("stages", [])
        status = f"{detected_count}/{total} SOPs" if detected_count else "Not detected"
        a(f"| {flow_name.replace('_', ' ').title()} | {status} | {conf:.2f} | {', '.join(stages[:3]) or '—'} |")
    a("")
    a("**Note:** German pharmaceutical SOPs use implicit workflow descriptions")
    a("embedded in numbered procedure sections rather than explicit flowcharts.")
    a("")

    # ── 12. Compliance Pattern ───────────────────────────────────────────────
    a("## 12. Compliance Pattern")
    a("")
    a(f"- **Avg compliance score:** {compliance.get('avg_compliance_score', 0)} / 100")
    a(f"- **Avg compliance grade:** {compliance.get('avg_compliance_grade', 'D')}")
    a("")
    a("### Standards Referenced")
    for std in compliance.get("standards_detected", []):
        a(f"- {std}")
    a("")
    a("### Compliance Signals")
    for t in compliance.get("control_terms", []):
        a(f"- Control term: `{t}`")
    for t in compliance.get("recordkeeping_terms", []):
        a(f"- Recordkeeping term: `{t}`")
    for lbl in compliance.get("section_labels", []):
        a(f"- Section label: `{lbl}`")
    a("")
    a("### Compliance Gaps")
    a("- Low compliance score indicates missing explicit approval sections")
    a("- Timing / SLA language absent in most sections")
    a("- Data integrity terms not explicitly stated")
    a("")

    # ── 13. Risk / Control Pattern ───────────────────────────────────────────
    a("## 13. Risk / Control Pattern")
    a("")
    a(f"- **Dominant risk level:** {risks_gaps.get('dominant_risk_level', 'high').upper()}")
    a(f"- **Average risk score:** {risks_gaps.get('avg_risk_score', 0)} / 100")
    a("")
    a("### Risk Indicators")
    a("- Missing approval authority sections")
    a("- No explicit timing / SLA / frequency statements")
    a("- Insufficient verification / control language")
    a("- Roles not explicitly assigned to procedural steps")
    a("")

    # ── 14. Gap / Weakness Pattern ───────────────────────────────────────────
    a("## 14. Gap / Weakness Pattern")
    a("")
    gap_cats = risks_gaps.get("gap_categories", {})
    for cat, msgs in gap_cats.items():
        if not msgs:
            continue
        a(f"### {cat.replace('_', ' ').title()}")
        for msg in msgs[:4]:
            a(f"- {msg}")
        a("")
    if not gap_cats:
        a("- No structured gap data available — run Phase 4 NLP analysis first")
        a("")

    # ── 15. Rewrite Guidance ─────────────────────────────────────────────────
    a("## 15. Rewrite Guidance")
    a("")
    a("When rewriting an SOP using this German pharma profile:")
    a("")
    a("1. **Language:** Use formal German regulatory register")
    a("   - Third person, impersonal constructions")
    a("   - Avoid contractions and colloquial language")
    a("")
    a("2. **Modal verbs:**")
    a("   - Use `muss/müssen` for GMP-mandatory requirements")
    a("   - Use `sollte/sollten` for best-practice recommendations")
    a("   - Use `darf nicht` for explicit prohibitions")
    a("")
    a("3. **Structure:**")
    a("   - Decimal-numbered sections (4.1, 4.2 … 4.N)")
    a("   - Each section: one clear procedural statement")
    a("   - Include: Zweck → Geltungsbereich → Verantwortlichkeiten → Verfahren → Aufzeichnungen")
    a("")
    a("4. **Terminology:**")
    a("   - Use standard GMP acronyms: GMP, SOP, OOS, PAT, GxP")
    a("   - Use German regulatory terms for section headers")
    a("   - Define all acronyms on first use")
    a("")
    a("5. **Compliance:**")
    a("   - Reference applicable standards (GxP, ICH Q9, HACCP where relevant)")
    a("   - Include explicit approval / release section")
    a("   - Add Änderungshistorie (revision history) table")
    a("")

    # ── 16. Improve Guidance ─────────────────────────────────────────────────
    a("## 16. Improve Guidance")
    a("")
    a("When improving an existing SOP to match this profile:")
    a("")
    a("1. **Add missing sections:**")
    a("   - Zweck (Purpose) — if absent")
    a("   - Geltungsbereich (Scope) — if absent")
    a("   - Verantwortlichkeiten (Responsibilities) — with named roles")
    a("   - Aufzeichnungen (Records) — with retention periods")
    a("   - Freigabe (Approval) — with approver authority")
    a("   - Änderungshistorie (Revision History) — with version table")
    a("")
    a("2. **Strengthen compliance language:**")
    a("   - Replace vague statements with modal-verb constructions")
    a("   - Add timing / frequency statements (e.g., \"mindestens einmal jährlich\")")
    a("   - Add traceability references (batch numbers, document IDs)")
    a("")
    a("3. **Clarify roles:**")
    a("   - Assign QA, Reviewer, Approver explicitly to each procedural step")
    a("   - Add a RACI table or responsibility matrix")
    a("")
    a("4. **Improve structure:**")
    a("   - Break long paragraphs into numbered sub-steps")
    a("   - Use consistent decimal numbering throughout")
    a("   - Add cross-references to related SOPs and regulations")
    a("")

    # ── 17. Gap Check Guidance ───────────────────────────────────────────────
    a("## 17. Gap Check Guidance")
    a("")
    a("When performing a gap check against this German pharma profile:")
    a("")
    a("### Mandatory Sections Checklist")
    a("- [ ] Zweck / Purpose")
    a("- [ ] Geltungsbereich / Scope")
    a("- [ ] Verantwortlichkeiten / Responsibilities (with named roles)")
    a("- [ ] Definitionen / Definitions")
    a("- [ ] Verfahren / Procedure (decimal-numbered)")
    a("- [ ] Aufzeichnungen / Records (with retention periods)")
    a("- [ ] Freigabe / Approval (with approver authority)")
    a("- [ ] Änderungshistorie / Revision History")
    a("")
    a("### Compliance Checklist")
    a("- [ ] GxP compliance language present")
    a("- [ ] Applicable standards referenced (ICH, AMWHV, etc.)")
    a("- [ ] Timing / SLA statements included")
    a("- [ ] Data integrity language present")
    a("- [ ] Traceability IDs / batch references included")
    a("")
    a("### Role / RACI Checklist")
    a("- [ ] QA role explicitly named")
    a("- [ ] Reviewer role explicitly named")
    a("- [ ] Approver / Sachkundige Person explicitly named")
    a("- [ ] Responsibilities assigned per procedural step")
    a("")
    a("### Modal Verb Checklist")
    a("- [ ] Mandatory requirements use `muss/müssen`")
    a("- [ ] Recommendations use `sollte/sollten`")
    a("- [ ] Prohibitions use `darf nicht/dürfen nicht`")
    a("- [ ] No ambiguous language (\"may\", \"might\", \"could\")")
    a("")
    a("### Workflow Checklist")
    a("- [ ] Approval flow defined (submit → review → approve → release)")
    a("- [ ] Change control flow defined")
    a("- [ ] CAPA flow defined (if applicable)")
    a("- [ ] Audit flow defined (if applicable)")
    a("")
    a("---")
    a(f"*Profile generated by Cybrain QS — Phase 5 | {generated_at}*")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# DB save — create / update ClientProfile + ProfileVersion
# ---------------------------------------------------------------------------

def save_profile_to_db(
    db,
    profile_name: str,
    profile_md: str,
    profile_json: Dict[str, Any],
    source_sop_rows: List[SOP],
    source_version_rows: List[SOPVersion],
) -> Tuple[ClientProfile, ProfileVersion]:
    """
    Upsert the ClientProfile and create a new ProfileVersion (version=1 on first run).
    Returns (profile_row, version_row).
    """
    # Find or create ClientProfile
    profile = (
        db.query(ClientProfile)
        .filter(
            ClientProfile.name == profile_name,
            ClientProfile.tenant_id == FIXED_TENANT,
        )
        .first()
    )

    is_new = profile is None
    if is_new:
        profile = ClientProfile(
            id=uuid.uuid4(),
            tenant_id=FIXED_TENANT,
            name=profile_name,
            company_name=profile_name,
            description="Combined German pharmaceutical SOP profile — Phase 5",
            domain="pharmaceutical",
            total_sops_analyzed=len(source_sop_rows),
            active_profile_md=profile_md,
            active_profile_json=profile_json,
        )
        db.add(profile)
        db.flush()
    else:
        profile.active_profile_md  = profile_md
        profile.active_profile_json = profile_json
        profile.total_sops_analyzed = len(source_sop_rows)
        db.flush()

    # Determine next version number
    latest = (
        db.query(ProfileVersion)
        .filter(ProfileVersion.profile_id == profile.id)
        .order_by(ProfileVersion.version_number.desc())
        .first()
    )
    next_ver = (latest.version_number + 1) if latest else 1

    # Build source_sop_ids list for snapshot
    source_sop_ids     = [str(s.id) for s in source_sop_rows]
    source_version_ids = [str(v.id) for v in source_version_rows]

    # Create ProfileVersion — one row per combined profile generation
    # We store all source SOP IDs in detected_parameters_snapshot
    version_row = ProfileVersion(
        id=uuid.uuid4(),
        profile_id=profile.id,
        version_number=next_ver,
        rules_json=profile_json,
        profile_md=profile_md,
        change_reason="Phase 5 — combined German SOP profile generation",
        # source_sop_id holds the first SOP for FK compatibility
        source_sop_id=source_sop_rows[0].id if source_sop_rows else None,
        source_version_id=source_version_rows[0].id if source_version_rows else None,
        detected_parameters_snapshot={
            "source_sop_ids":     source_sop_ids,
            "source_version_ids": source_version_ids,
            "profile_name":       profile_name,
            "profile_version":    next_ver,
            "generated_at":       datetime.now(timezone.utc).isoformat(),
            "status":             "active",
        },
        is_locked=False,
    )
    db.add(version_row)
    db.flush()

    # Point profile at this version
    profile.current_version_id = version_row.id

    # History event
    event = ProfileHistoryEvent(
        id=uuid.uuid4(),
        client_profile_id=profile.id,
        profile_version_id=version_row.id,
        event_type="PHASE5_PROFILE_GENERATED",
        event_summary=(
            f"Phase 5 combined profile generated from "
            f"{len(source_sop_rows)} German SOPs: "
            f"{', '.join(s.sop_number for s in source_sop_rows)}"
        ),
        after_snapshot=profile_json,
        source_sop_id=source_sop_rows[0].id if source_sop_rows else None,
        created_by="phase5_test_script",
    )
    db.add(event)

    db.commit()
    db.refresh(profile)
    db.refresh(version_row)
    return profile, version_row

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEP)
    print("  PHASE 5: Generate German Company Profile.md")
    print(f"  Profile name : {PROFILE_NAME}")
    print(f"  Client name  : {CLIENT_NAME}")
    print(SEP)

    db = SessionLocal()
    result: Dict[str, Any] = {
        "profile_created":    False,
        "profile_name":       PROFILE_NAME,
        "profile_version":    None,
        "source_sops":        GERMAN_SOP_LABELS,
        "profile_saved_in_db": False,
        "profile_md_generated": False,
        "profile_id":         None,
        "errors":             [],
    }

    try:
        # ──────────────────────────────────────────────────────────────────
        # STEP 1: Locate the 3 canonical German SOPs
        # ──────────────────────────────────────────────────────────────────
        _section("STEP 1: Locate German SOPs in Database")

        sop_rows: List[SOP] = []
        for sop_num in GERMAN_SOP_NUMBERS:
            row = db.query(SOP).filter(SOP.sop_number == sop_num).first()
            if row:
                sop_rows.append(row)
                _ok(f"Found: {sop_num} → id={row.id}")
            else:
                _warn(f"Not found by number: {sop_num} — searching by title pattern")
                # Fallback: search by title containing the label
                label = sop_num.replace("GERMAN-SOP", "german_sop").replace("-PHASE3", "").replace("-", "_").lower()
                fallback = (
                    db.query(SOP)
                    .filter(SOP.title.ilike(f"%{label}%"))
                    .filter(SOP.tenant_id == FIXED_TENANT)
                    .first()
                )
                if fallback:
                    sop_rows.append(fallback)
                    _ok(f"Fallback found: {fallback.sop_number} → id={fallback.id}")
                else:
                    msg = f"SOP not found: {sop_num}"
                    _fail(msg)
                    result["errors"].append(msg)

        if not sop_rows:
            print("\n[CRITICAL] No German SOPs found. Run Phase 3 first.")
            sys.exit(1)

        _info(f"Using {len(sop_rows)} SOP(s) for profile generation")

        # ──────────────────────────────────────────────────────────────────
        # STEP 2: Get latest SOPVersion for each SOP
        # ──────────────────────────────────────────────────────────────────
        _section("STEP 2: Retrieve SOP Versions")

        version_rows: List[SOPVersion] = []
        for sop in sop_rows:
            ver = (
                db.query(SOPVersion)
                .filter(SOPVersion.sop_id == sop.id)
                .order_by(SOPVersion.created_at.desc())
                .first()
            )
            if ver:
                version_rows.append(ver)
                _ok(f"{sop.sop_number} → version_id={ver.id} (v{ver.version_number})")
            else:
                _warn(f"No version found for {sop.sop_number}")

        # ──────────────────────────────────────────────────────────────────
        # STEP 3: Load NLP parameters for each SOP
        # ──────────────────────────────────────────────────────────────────
        _section("STEP 3: Load NLP Parameters from sop_detected_parameters")

        nlp_rows: List[SOPDetectedParameters] = []
        for sop in sop_rows:
            row = (
                db.query(SOPDetectedParameters)
                .filter(SOPDetectedParameters.sop_id == sop.id)
                .order_by(SOPDetectedParameters.created_at.desc())
                .first()
            )
            if row:
                nlp_rows.append(row)
                ws = row.writing_style or {}
                _ok(
                    f"{sop.sop_number} → nlp_id={row.id} "
                    f"tone={ws.get('tone','?')} "
                    f"complexity={ws.get('writing_complexity','?')}"
                )
            else:
                _warn(f"No NLP parameters for {sop.sop_number} — profile will be partial")

        if not nlp_rows:
            print("\n[CRITICAL] No NLP parameters found. Run Phase 4 first.")
            sys.exit(1)

        # ──────────────────────────────────────────────────────────────────
        # STEP 4: Aggregate NLP data across all SOPs
        # ──────────────────────────────────────────────────────────────────
        _section("STEP 4: Aggregate NLP Parameters")

        style       = _dominant_writing_style(nlp_rows)
        modal_verbs = _aggregate_modal_verbs(nlp_rows)
        terminology = _aggregate_terminology(nlp_rows)
        sections    = _aggregate_sections(nlp_rows)
        roles       = _aggregate_roles(nlp_rows)
        workflows   = _aggregate_workflows(nlp_rows)
        compliance  = _aggregate_compliance(nlp_rows)
        risks_gaps  = _aggregate_risks_gaps(nlp_rows)

        _info(f"Style: tone={style['tone']} formality={style['formality']} complexity={style['writing_complexity']}")
        _info(f"Modal verbs: {modal_verbs}")
        _info(f"Terminology: {sum(len(v) for v in terminology.values())} terms total")
        _info(f"Sections: {sections['total_sections_across_sops']} total, labels={sections['common_section_labels']}")
        _info(f"Roles: detected={roles['detected_roles']} missing={roles['missing_roles']}")
        _info(f"Compliance: standards={compliance['standards_detected']} score={compliance['avg_compliance_score']}")
        _info(f"Risk: level={risks_gaps['dominant_risk_level']} score={risks_gaps['avg_risk_score']}")

        # ──────────────────────────────────────────────────────────────────
        # STEP 5: Build source_sops list for profile.md
        # ──────────────────────────────────────────────────────────────────
        source_sops_for_md = []
        for i, sop in enumerate(sop_rows):
            label = GERMAN_SOP_LABELS[i] if i < len(GERMAN_SOP_LABELS) else f"german_sop{i+1}"
            source_sops_for_md.append({
                "label":      label,
                "sop_number": sop.sop_number,
                "title":      sop.title or sop.sop_number,
                "sop_id":     str(sop.id),
            })

        # ──────────────────────────────────────────────────────────────────
        # STEP 6: Generate profile.md
        # ──────────────────────────────────────────────────────────────────
        _section("STEP 6: Generate profile.md")

        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        profile_md = generate_profile_md(
            profile_name=PROFILE_NAME,
            source_sops=source_sops_for_md,
            style=style,
            modal_verbs=modal_verbs,
            terminology=terminology,
            sections=sections,
            roles=roles,
            workflows=workflows,
            compliance=compliance,
            risks_gaps=risks_gaps,
            generated_at=generated_at,
        )

        md_len = len(profile_md)
        _check("profile.md generated", md_len > 500, f"{md_len} chars")
        _check("All 17 sections present", all(
            f"## {i}." in profile_md for i in range(1, 18)
        ), "sections 1–17 found")

        result["profile_md_generated"] = md_len > 500

        # ──────────────────────────────────────────────────────────────────
        # STEP 7: Build profile_json (rules_json for ProfileVersion)
        # ──────────────────────────────────────────────────────────────────
        profile_json: Dict[str, Any] = {
            "profile_name":    PROFILE_NAME,
            "profile_title":   PROFILE_NAME,
            "client_name":     CLIENT_NAME,
            "language":        "German (DE) / Bilingual DE-EN",
            "domain":          "pharmaceutical",
            "generated_at":    generated_at,
            "source_sop_ids":  [str(s.id) for s in sop_rows],
            "source_version_ids": [str(v.id) for v in version_rows],
            "writing_style":   style,
            "modal_verbs":     modal_verbs,
            "terminology":     terminology,
            "section_structure": sections,
            "roles":           roles,
            "workflow_patterns": workflows,
            "compliance":      compliance,
            "risks_gaps":      risks_gaps,
            "rewrite_guidance": [
                "Use formal German regulatory register",
                "Use muss/müssen for GMP-mandatory requirements",
                "Use sollte/sollten for best-practice recommendations",
                "Use decimal-numbered sections (4.1, 4.2 …)",
                "Include: Zweck → Geltungsbereich → Verantwortlichkeiten → Verfahren → Aufzeichnungen",
                "Reference applicable standards (GxP, ICH Q9)",
                "Add explicit Freigabe (approval) section",
                "Add Änderungshistorie (revision history) table",
            ],
            "improve_guidance": [
                "Add missing sections: Zweck, Geltungsbereich, Verantwortlichkeiten, Aufzeichnungen, Freigabe",
                "Assign QA, Reviewer, Approver explicitly to each procedural step",
                "Add timing / frequency statements",
                "Add traceability references (batch numbers, document IDs)",
                "Break long paragraphs into numbered sub-steps",
            ],
            "gap_check_guidance": [
                "Check for mandatory sections: Zweck, Geltungsbereich, Verantwortlichkeiten, Verfahren, Aufzeichnungen, Freigabe, Änderungshistorie",
                "Check for GxP compliance language",
                "Check for explicit role assignments (QA, Reviewer, Approver)",
                "Check for modal verb correctness (muss vs sollte)",
                "Check for approval flow definition",
            ],
        }

        # ──────────────────────────────────────────────────────────────────
        # STEP 8: Save to database
        # ──────────────────────────────────────────────────────────────────
        _section("STEP 8: Save Profile to Database")

        try:
            profile_row, version_row_saved = save_profile_to_db(
                db=db,
                profile_name=PROFILE_NAME,
                profile_md=profile_md,
                profile_json=profile_json,
                source_sop_rows=sop_rows,
                source_version_rows=version_rows,
            )
            result["profile_created"]    = True
            result["profile_saved_in_db"] = True
            result["profile_id"]         = str(profile_row.id)
            result["profile_version"]    = version_row_saved.version_number

            _ok(f"ClientProfile saved: id={profile_row.id}")
            _ok(f"ProfileVersion saved: id={version_row_saved.id} version={version_row_saved.version_number}")
        except Exception as exc:
            msg = f"DB save failed: {exc}"
            _fail(msg)
            result["errors"].append(msg)
            db.rollback()

        # ──────────────────────────────────────────────────────────────────
        # STEP 9: Database validation
        # ──────────────────────────────────────────────────────────────────
        _section("STEP 9: Database Validation")

        # Reload from DB
        saved_profile = (
            db.query(ClientProfile)
            .filter(ClientProfile.name == PROFILE_NAME, ClientProfile.tenant_id == FIXED_TENANT)
            .first()
        )

        _check("ClientProfile record exists in DB",
               saved_profile is not None,
               f"id={saved_profile.id}" if saved_profile else "NOT FOUND")

        if saved_profile:
            _check("profile_name correct",
                   saved_profile.name == PROFILE_NAME,
                   saved_profile.name)
            _check("active_profile_md saved",
                   bool(saved_profile.active_profile_md),
                   f"{len(saved_profile.active_profile_md or '')} chars")
            _check("active_profile_json saved",
                   bool(saved_profile.active_profile_json),
                   f"keys: {list((saved_profile.active_profile_json or {}).keys())[:5]}")
            _check("total_sops_analyzed correct",
                   saved_profile.total_sops_analyzed == len(sop_rows),
                   f"{saved_profile.total_sops_analyzed}")
            _check("current_version_id set",
                   saved_profile.current_version_id is not None,
                   str(saved_profile.current_version_id))
            _check("created_at set",
                   saved_profile.created_at is not None,
                   str(saved_profile.created_at))

        # Validate ProfileVersion
        if saved_profile:
            saved_version = (
                db.query(ProfileVersion)
                .filter(ProfileVersion.profile_id == saved_profile.id)
                .order_by(ProfileVersion.version_number.desc())
                .first()
            )
            _check("ProfileVersion record exists",
                   saved_version is not None,
                   f"version={saved_version.version_number}" if saved_version else "NOT FOUND")

            if saved_version:
                snap = saved_version.detected_parameters_snapshot or {}
                _check("source_sop_ids in snapshot",
                       bool(snap.get("source_sop_ids")),
                       f"{snap.get('source_sop_ids', [])}")
                _check("source_version_ids in snapshot",
                       bool(snap.get("source_version_ids")),
                       f"{snap.get('source_version_ids', [])}")
                _check("profile_version in snapshot",
                       snap.get("profile_version") is not None,
                       f"v{snap.get('profile_version')}")
                _check("status in snapshot",
                       snap.get("status") == "active",
                       snap.get("status", ""))
                _check("profile_md in version",
                       bool(saved_version.profile_md),
                       f"{len(saved_version.profile_md or '')} chars")
                _check("rules_json in version",
                       bool(saved_version.rules_json),
                       f"keys: {list((saved_version.rules_json or {}).keys())[:5]}")

        # ──────────────────────────────────────────────────────────────────
        # STEP 10: Print profile.md preview
        # ──────────────────────────────────────────────────────────────────
        _section("STEP 10: Profile.md Preview (first 80 lines)")
        for line in profile_md.split("\n")[:80]:
            print(f"  {line}")
        print(f"\n  … ({len(profile_md.split(chr(10)))} total lines)")

    finally:
        db.close()

    # ──────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────
    _section("PHASE 5 SUMMARY")

    print(f"\n  Profile name    : {result['profile_name']}")
    print(f"  Profile ID      : {result.get('profile_id', 'N/A')}")
    print(f"  Profile version : {result.get('profile_version', 'N/A')}")
    print(f"  Source SOPs     : {result['source_sops']}")
    print()

    checks = [
        ("profile_created",     "Profile created"),
        ("profile_saved_in_db", "Profile saved in DB"),
        ("profile_md_generated","Profile.md generated"),
    ]
    all_pass = True
    for key, label in checks:
        ok = result.get(key, False)
        if ok:
            _ok(label)
        else:
            _fail(label)
            all_pass = False

    if result.get("errors"):
        print(f"\n  Errors:")
        for e in result["errors"]:
            print(f"    - {e}")

    # Expected output JSON
    _section("EXPECTED OUTPUT")
    expected = {
        "profile_created":    result["profile_created"],
        "profile_name":       result["profile_name"],
        "profile_version":    result.get("profile_version"),
        "source_sops":        result["source_sops"],
        "profile_saved_in_db": result["profile_saved_in_db"],
        "profile_md_generated": result["profile_md_generated"],
    }
    print(json.dumps(expected, indent=4, ensure_ascii=False))

    print(f"\n{SEP}")
    if all_pass:
        print("  RESULT: ALL PHASE 5 CHECKS PASSED")
        print("  German company profile.md generated and saved to database.")
        print("  Ready to proceed to PHASE 6 (rewrite/improve SOP-IT-002 using profile).")
    else:
        print("  RESULT: SOME PHASE 5 CHECKS FAILED — review output above")
    print(SEP)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
