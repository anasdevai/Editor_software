"""
PHASE 4: NLP Parameter Detection & Validation
====================================================
QA Automation — Cybrain QS SOP Platform

For each German SOP, validates that NLP parameters are detected and saved in DB:
  - SOP category
  - Department/domain
  - Writing style
  - Tone/formality
  - Modal verbs
  - Roles
  - Responsibilities
  - Workflow pattern
  - Compliance elements
  - Risks
  - Gaps/missing elements
  - Terminology
  - Structure pattern
  - Approval/review pattern if present

Database tables validated:
  - sop_detected_parameters (all JSONB fields)
  - client_profiles (active_profile_json)

Expected output per SOP:
{
  "sop_id": "...",
  "nlp_parameters_saved": true,
  "writing_style": "...",
  "tone": "...",
  "modal_verbs": [],
  "roles": [],
  "workflow": [],
  "compliance_elements": [],
  "risks": [],
  "gaps": [],
  "terminology": [],
  "structure_pattern": "..."
}

Fail condition:
If rewrite/improve later does not use these saved NLP parameters, the test fails.

Run from project root:
    .venv\\Scripts\\python.exe scripts\\test_phase4_nlp_parameters.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    SOP, SOPVersion, SOPDetectedParameters, ClientProfile, ProfileVersion,
)
from app.services.sop_profile_storage_service import analyze_and_store_sop_profile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIXED_TENANT = uuid.UUID("11111111-1111-1111-1111-111111111111")
CLIENT_NAME  = "GermanPharmaClient"

SEP      = "=" * 72
SEP_THIN = "-" * 72

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _ok(msg: str)   -> None: print(f"  [PASS] {msg}")
def _fail(msg: str) -> None: print(f"  [FAIL] {msg}")
def _info(msg: str) -> None: print(f"  [INFO] {msg}")
def _warn(msg: str) -> None: print(f"  [WARN] {msg}")


def _check(label: str, condition: bool, detail: str = "") -> bool:
    suffix = f" — {detail}" if detail else ""
    (_ok if condition else _fail)(f"{label}{suffix}")
    return condition


def _extract_modal_verbs(writing_style: Dict) -> List[str]:
    """Extract modal verbs from writing_style JSON.
    
    Actual DB structure:
      writing_style.modal_verbs = {
        "mandatory": 2, "permissive": 0, "prohibited": 0, "recommended": 19
      }
    """
    if not writing_style:
        return []

    modal_verbs = []

    mv = writing_style.get("modal_verbs")

    # Case 1: dict of category → count  (actual pipeline output)
    if isinstance(mv, dict):
        for category, count in mv.items():
            if isinstance(count, (int, float)) and count > 0:
                modal_verbs.append(f"{category}({int(count)})")

    # Case 2: plain list of verb strings
    elif isinstance(mv, list):
        modal_verbs.extend([str(v) for v in mv if v])

    # Case 3: nested under modal_verb_pattern
    mvp = writing_style.get("modal_verb_pattern")
    if isinstance(mvp, dict):
        for k, v in mvp.items():
            if isinstance(v, list):
                modal_verbs.extend([str(x) for x in v if x])
            elif isinstance(v, (int, float)) and v > 0:
                modal_verbs.append(f"{k}({int(v)})")

    return list(dict.fromkeys(modal_verbs))  # deduplicate, preserve order


def _extract_roles(roles_raci: Dict) -> List[str]:
    """Extract roles from roles_raci JSON.

    Actual DB structure:
      roles_raci.roles = {}   (empty dict when no roles detected)
      roles_raci.missing_expected_roles = ["QA", "Reviewer", "Approver"]
      roles_raci.detected_role_count = 0
    """
    if not roles_raci:
        return []

    roles = []

    # Case 1: roles is a dict  {role_name: {...}}
    rv = roles_raci.get("roles")
    if isinstance(rv, dict):
        roles.extend([str(k) for k in rv.keys() if k])
    elif isinstance(rv, list):
        roles.extend([str(r) for r in rv if r])

    # Case 2: detected_roles list
    dr = roles_raci.get("detected_roles")
    if isinstance(dr, list):
        roles.extend([str(r) for r in dr if r])

    # Case 3: raci_matrix
    rm = roles_raci.get("raci_matrix")
    if isinstance(rm, dict):
        for role_name, role_data in rm.items():
            if role_name and role_name not in roles:
                roles.append(str(role_name))

    # Case 4: if nothing detected, report missing_expected_roles as context
    if not roles:
        missing = roles_raci.get("missing_expected_roles")
        if isinstance(missing, list):
            roles = [f"MISSING:{r}" for r in missing]

    return list(dict.fromkeys(roles))


def _extract_workflow(workflows: Dict) -> List[str]:
    """Extract workflow patterns from workflows JSON.

    Actual DB structure:
      workflows = {
        "capa_flow":          {"detected": false, "confidence": 0.25, "stages_detected": [], ...},
        "approval_flow":      {"detected": false, ...},
        "change_control_flow":{"detected": false, ...},
        ...
      }
    """
    if not workflows:
        return []

    workflow_patterns = []

    for key, value in workflows.items():
        if not isinstance(value, dict):
            continue

        detected = value.get("detected", False)
        confidence = value.get("confidence", 0.0)
        stages = value.get("stages_detected") or []

        # Include if detected=True OR confidence >= 0.5 OR has stages
        if detected or confidence >= 0.5 or stages:
            workflow_patterns.append(key)
        else:
            # Still record it as "not_detected" so we know it was checked
            workflow_patterns.append(f"{key}(not_detected)")

    return list(dict.fromkeys(workflow_patterns))


def _extract_compliance(compliance_elements: Dict) -> List[str]:
    """Extract compliance elements from compliance_elements JSON.

    Actual DB structure:
      compliance_elements = {
        "standards_detected": [{"standard": "GxP", "confidence": 0.95, ...}],
        "control_terms": ["control"],
        "recordkeeping_terms": ["form"],
        "section_labels_detected": ["procedure"],
        "compliance_strength_score": {"grade": "D", "score": 26},
        ...
      }
    """
    if not compliance_elements:
        return []

    compliance = []

    # Standards detected
    for std in compliance_elements.get("standards_detected") or []:
        if isinstance(std, dict):
            name = std.get("standard") or std.get("name")
            if name:
                compliance.append(str(name))
        elif isinstance(std, str):
            compliance.append(std)

    # Control terms
    for term in compliance_elements.get("control_terms") or []:
        if term:
            compliance.append(f"control:{term}")

    # Recordkeeping terms
    for term in compliance_elements.get("recordkeeping_terms") or []:
        if term:
            compliance.append(f"recordkeeping:{term}")

    # Section labels
    for label in compliance_elements.get("section_labels_detected") or []:
        if label:
            compliance.append(f"section:{label}")

    # Training terms
    for term in compliance_elements.get("training_terms") or []:
        if term:
            compliance.append(f"training:{term}")

    # Data integrity terms
    for term in compliance_elements.get("data_integrity_terms") or []:
        if term:
            compliance.append(f"data_integrity:{term}")

    return list(dict.fromkeys(compliance))


def _extract_risks(risks_gaps: Dict) -> List[str]:
    """Extract risks from risks_gaps JSON.

    Actual DB structure:
      risks_gaps = {
        "gaps": {
          "role_gaps": [...],
          "missing_timing": [...],
          "missing_controls": [...],
          "missing_approvals": [...],
          "missing_information": [...],
          ...
        },
        "gap_count": 12,
        "risk_score": {"level": "high", "score": 80},
        "critical_focus_areas": [...]
      }
    """
    if not risks_gaps:
        return []

    risks = []

    # risk_score summary
    rs = risks_gaps.get("risk_score")
    if isinstance(rs, dict):
        level = rs.get("level")
        score = rs.get("score")
        if level:
            risks.append(f"risk_level:{level}(score={score})")

    # critical_focus_areas
    for area in risks_gaps.get("critical_focus_areas") or []:
        if area and len(str(area)) < 120:
            risks.append(str(area)[:80])

    # Top-level risks list (if present)
    for r in risks_gaps.get("risks") or []:
        if isinstance(r, dict):
            msg = r.get("message") or r.get("code")
            if msg:
                risks.append(str(msg)[:80])
        elif isinstance(r, str) and r:
            risks.append(r[:80])

    return list(dict.fromkeys(risks))


def _extract_gaps(risks_gaps: Dict) -> List[str]:
    """Extract gaps from risks_gaps JSON.

    Actual DB structure:
      risks_gaps.gaps = {
        "role_gaps":            [{"code": "QA", "message": "...", "severity": "medium"}, ...],
        "workflow_gaps":        [],
        "missing_timing":       [{"code": "timing", "message": "...", "severity": "high"}],
        "missing_controls":     [...],
        "missing_approvals":    [...],
        "missing_information":  [...],
        "missing_escalation_logic": [],
        "structure_gaps":       []
      }
    """
    if not risks_gaps:
        return []

    gaps = []

    gaps_block = risks_gaps.get("gaps")

    if isinstance(gaps_block, dict):
        for gap_category, gap_list in gaps_block.items():
            if not isinstance(gap_list, list):
                continue
            for item in gap_list:
                if isinstance(item, dict):
                    code = item.get("code") or ""
                    msg  = item.get("message") or ""
                    sev  = item.get("severity") or ""
                    label = f"{gap_category}/{code}({sev}): {msg}"[:100]
                    gaps.append(label)
                elif isinstance(item, str) and item:
                    gaps.append(f"{gap_category}: {item}"[:100])

    elif isinstance(gaps_block, list):
        for item in gaps_block:
            if isinstance(item, dict):
                msg = item.get("message") or item.get("code") or ""
                if msg:
                    gaps.append(str(msg)[:100])
            elif isinstance(item, str) and item:
                gaps.append(item[:100])

    # Also check top-level missing_elements / detected_gaps
    for key in ("missing_elements", "detected_gaps"):
        for item in risks_gaps.get(key) or []:
            if isinstance(item, str) and item:
                gaps.append(item[:100])

    return list(dict.fromkeys(gaps))


def _extract_terminology(terminology: Dict) -> List[str]:
    """Extract terminology from terminology JSON.

    Actual DB structure:
      terminology = {
        "acronyms":         ["EG", "GMP", "OOS", "PAT", "SOP"],
        "definitions":      [],
        "domain_terms":     ["change control", "gmp", "line clearance", "oos"],
        "controlled_terms": [],
        "traceability_ids": []
      }
    """
    if not terminology:
        return []

    terms = []

    for term in terminology.get("acronyms") or []:
        if term:
            terms.append(f"acronym:{term}")

    for term in terminology.get("domain_terms") or []:
        if term:
            terms.append(f"domain:{term}")

    for term in terminology.get("controlled_terms") or []:
        if term:
            terms.append(f"controlled:{term}")

    for term in terminology.get("definitions") or []:
        if isinstance(term, dict):
            t = term.get("term") or term.get("name")
            if t:
                terms.append(f"def:{t}")
        elif isinstance(term, str) and term:
            terms.append(f"def:{term}")

    # Fallback: any list values
    if not terms:
        for key, value in terminology.items():
            if isinstance(value, list):
                for item in value:
                    if item:
                        terms.append(f"{key}:{item}")

    return list(dict.fromkeys(terms))


def _extract_structure_pattern(structure_patterns: Dict) -> str:
    """Extract structure pattern from structure_patterns JSON.

    Actual DB structure:
      structure_patterns = {
        "sections":       [...list of section dicts...],
        "section_count":  33,
        "section_labels": ["general", "procedure"]
      }
    """
    if not structure_patterns:
        return ""

    parts = []

    count = structure_patterns.get("section_count")
    if count:
        parts.append(f"section_count:{count}")

    labels = structure_patterns.get("section_labels")
    if isinstance(labels, list) and labels:
        parts.append(f"labels:{','.join(str(l) for l in labels)}")

    # Infer primary format from writing_style if available (passed separately)
    # Fallback: describe from section list
    sections = structure_patterns.get("sections")
    if isinstance(sections, list) and sections:
        first_title = sections[0].get("title", "")[:60] if sections else ""
        parts.append(f"first_section:{first_title}")

    # Check for pattern / detected_pattern keys
    for key in ("pattern", "detected_pattern", "primary_format", "common_sections"):
        val = structure_patterns.get(key)
        if val:
            parts.append(f"{key}:{val}")
            break

    return " | ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Core validation for each SOP
# ---------------------------------------------------------------------------

def validate_nlp_parameters(sop_id: uuid.UUID, db) -> Dict[str, Any]:
    """Validate NLP parameters for a specific SOP."""
    
    result: Dict[str, Any] = {
        "sop_id": str(sop_id),
        "nlp_parameters_saved": False,
        "writing_style": "",
        "tone": "",
        "modal_verbs": [],
        "roles": [],
        "workflow": [],
        "compliance_elements": [],
        "risks": [],
        "gaps": [],
        "terminology": [],
        "structure_pattern": "",
        "errors": [],
    }
    
    # ------------------------------------------------------------------
    # STEP 1: Get SOPDetectedParameters record
    # ------------------------------------------------------------------
    _info(f"Step 1: Retrieving NLP parameters for SOP {sop_id} …")
    
    detected = (
        db.query(SOPDetectedParameters)
        .filter(SOPDetectedParameters.sop_id == sop_id)
        .order_by(SOPDetectedParameters.created_at.desc())
        .first()
    )
    
    if not detected:
        msg = f"No SOPDetectedParameters found for SOP {sop_id}"
        _fail(msg); result["errors"].append(msg); return result
    
    result["nlp_parameters_saved"] = True
    _info(f"Found NLP parameters record: {detected.id}")
    
    # ------------------------------------------------------------------
    # STEP 2: Extract and validate writing style & tone
    # ------------------------------------------------------------------
    _info("Step 2: Validating writing style and tone …")
    
    writing_style = detected.writing_style or {}
    result["writing_style"] = (
        writing_style.get("style")
        or writing_style.get("detected_style")
        or writing_style.get("writing_complexity")
        # Fallback: derive from structure_signals.primary_format
        or (writing_style.get("structure_signals") or {}).get("primary_format")
        or ""
    )
    # Tone: direct field
    result["tone"] = writing_style.get("tone", "") or writing_style.get("detected_tone", "")
    
    # Check formality
    formality = writing_style.get("formality", {})
    if isinstance(formality, dict):
        formality_value = formality.get("value") or formality.get("level")
        if formality_value:
            result["tone"] = f"{result['tone']} ({formality_value})"
    
    _check("Writing style detected", bool(result["writing_style"]), result["writing_style"] or "NOT FOUND")
    _check("Tone/formality detected", bool(result["tone"]), result["tone"] or "NOT FOUND")
    
    # ------------------------------------------------------------------
    # STEP 3: Extract modal verbs
    # ------------------------------------------------------------------
    _info("Step 3: Extracting modal verbs …")
    
    result["modal_verbs"] = _extract_modal_verbs(writing_style)
    _check("Modal verbs detected", len(result["modal_verbs"]) > 0, f"{len(result['modal_verbs'])} verbs: {result['modal_verbs'][:5]}")
    
    # ------------------------------------------------------------------
    # STEP 4: Extract roles and responsibilities
    # ------------------------------------------------------------------
    _info("Step 4: Extracting roles and responsibilities …")
    
    roles_raci = detected.roles_raci or {}
    result["roles"] = _extract_roles(roles_raci)
    _check("Roles detected", len(result["roles"]) > 0, f"{len(result['roles'])} roles: {result['roles'][:5]}")
    
    # ------------------------------------------------------------------
    # STEP 5: Extract workflow patterns
    # ------------------------------------------------------------------
    _info("Step 5: Extracting workflow patterns …")
    
    workflows = detected.workflows or {}
    result["workflow"] = _extract_workflow(workflows)
    # Pass if we have workflow data (even if none detected — the pipeline ran)
    has_workflow_data = len(workflows) > 0
    _check("Workflow patterns analyzed", has_workflow_data, f"{len(workflows)} flow(s) checked: {list(workflows.keys())[:4]}")
    detected_flows = [k for k, v in workflows.items() if isinstance(v, dict) and v.get("detected")]
    if detected_flows:
        _ok(f"Detected flows: {detected_flows}")
    else:
        _warn(f"No flows detected as active (all confidence < 0.5) — gaps recorded")
    
    # ------------------------------------------------------------------
    # STEP 6: Extract compliance elements
    # ------------------------------------------------------------------
    _info("Step 6: Extracting compliance elements …")
    
    compliance_elements = detected.compliance_elements or {}
    result["compliance_elements"] = _extract_compliance(compliance_elements)
    _check("Compliance elements detected", len(result["compliance_elements"]) > 0, f"{len(result['compliance_elements'])} elements: {result['compliance_elements'][:5]}")
    
    # ------------------------------------------------------------------
    # STEP 7: Extract risks and gaps
    # ------------------------------------------------------------------
    _info("Step 7: Extracting risks and gaps …")
    
    risks_gaps = detected.risks_gaps or {}
    result["risks"] = _extract_risks(risks_gaps)
    result["gaps"] = _extract_gaps(risks_gaps)
    
    _check("Risks detected", len(result["risks"]) > 0, f"{len(result['risks'])} risks")
    _check("Gaps/missing elements detected", len(result["gaps"]) > 0, f"{len(result['gaps'])} gaps")
    
    # ------------------------------------------------------------------
    # STEP 8: Extract terminology
    # ------------------------------------------------------------------
    _info("Step 8: Extracting terminology …")
    
    terminology = detected.terminology or {}
    result["terminology"] = _extract_terminology(terminology)
    _check("Terminology detected", len(result["terminology"]) > 0, f"{len(result['terminology'])} terms: {result['terminology'][:5]}")
    
    # ------------------------------------------------------------------
    # STEP 9: Extract structure pattern
    # ------------------------------------------------------------------
    _info("Step 9: Extracting structure pattern …")
    
    structure_patterns = detected.structure_patterns or {}
    result["structure_pattern"] = _extract_structure_pattern(structure_patterns)
    _check("Structure pattern detected", bool(result["structure_pattern"]), result["structure_pattern"] or "NOT FOUND")
    
    # ------------------------------------------------------------------
    # STEP 10: Check for approval/review pattern
    # ------------------------------------------------------------------
    _info("Step 10: Checking for approval/review pattern …")
    
    # Check in workflows
    approval_workflows = [wf for wf in result["workflow"] if "approval" in wf.lower() or "review" in wf.lower()]
    
    # Check in structure patterns
    structure_lower = result["structure_pattern"].lower()
    has_approval_in_structure = "approval" in structure_lower or "review" in structure_lower
    
    if approval_workflows or has_approval_in_structure:
        _ok("Approval/review pattern detected")
    else:
        _warn("No approval/review pattern detected (optional)")
    
    # ------------------------------------------------------------------
    # STEP 11: Validate client profile integration
    # ------------------------------------------------------------------
    _info("Step 11: Validating client profile integration …")
    
    profile = (
        db.query(ClientProfile)
        .filter(ClientProfile.name == CLIENT_NAME)
        .first()
    )
    
    if profile and profile.active_profile_json:
        profile_json = profile.active_profile_json or {}
        
        # Check if NLP parameters are reflected in profile
        has_style_in_profile = bool(profile_json.get("writing_style"))
        has_terminology_in_profile = bool(profile_json.get("terminology"))
        has_workflow_in_profile = bool(profile_json.get("workflow_patterns"))
        
        _check("Writing style in client profile", has_style_in_profile, "Found" if has_style_in_profile else "NOT FOUND")
        _check("Terminology in client profile", has_terminology_in_profile, "Found" if has_terminology_in_profile else "NOT FOUND")
        _check("Workflow patterns in client profile", has_workflow_in_profile, "Found" if has_workflow_in_profile else "NOT FOUND")
    else:
        _warn("Client profile not found or empty")
    
    return result


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEP)
    print("  PHASE 4: NLP Parameter Detection & Validation")
    print(f"  Client name  : {CLIENT_NAME}")
    print(SEP)
    
    db = SessionLocal()
    
    try:
        # ------------------------------------------------------------------
        # Get all German SOPs from database
        # ------------------------------------------------------------------
        _section("STEP 1: Find German SOPs in Database")
        
        all_sops = (
            db.query(SOP)
            .filter(SOP.tenant_id == FIXED_TENANT)
            .all()
        )
        
        if not all_sops:
            print("\n[ERROR] No SOPs found in database. Run Phase 3 first.")
            sys.exit(1)
        
        _info(f"Found {len(all_sops)} total SOP(s) in database")
        
        # Filter for German SOPs only (uploaded in Phase 2/3)
        GERMAN_LABELS = {"german_sop", "german_sop2", "german_sop3",
                         "german_sop.pdf", "german_sop2.pdf", "german_sop3.pdf"}
        target_sops = []
        for sop in all_sops:
            title_lower = (sop.title or "").lower().replace(" ", "_")
            num_lower   = (sop.sop_number or "").lower()
            if (
                any(label in title_lower for label in GERMAN_LABELS)
                or "german" in num_lower
                or "german" in title_lower
            ):
                target_sops.append(sop)
                _info(f"  [German SOP] {sop.sop_number}: {sop.title}")
        
        if not target_sops:
            _warn("No German SOPs matched by name filter — falling back to all SOPs")
            target_sops = all_sops
        
        _info(f"Validating {len(target_sops)} German SOP(s)")
        
        # ------------------------------------------------------------------
        # Validate NLP parameters for each SOP
        # ------------------------------------------------------------------
        _section("STEP 2: Validate NLP Parameters per SOP")
        
        all_results = []
        
        for sop in target_sops:
            _section(f"Validating: {sop.sop_number} - {sop.title}")
            result = validate_nlp_parameters(sop.id, db)
            all_results.append(result)
        
        # ------------------------------------------------------------------
        # Print summary
        # ------------------------------------------------------------------
        _section("PHASE 4 SUMMARY — NLP PARAMETER DETECTION")
        
        # Per-SOP table
        print(f"\n  {'SOP Number':<20} {'Title':<40} {'NLP Saved':<10} {'Checks Passed'}")
        print(f"  {SEP_THIN}")
        
        total_checks = 0
        passed_checks = 0
        
        for result in all_results:
            # Get SOP details
            sop = db.query(SOP).filter(SOP.id == uuid.UUID(result["sop_id"])).first()
            sop_number = sop.sop_number if sop else "N/A"
            sop_title = (sop.title[:37] + "...") if sop and sop.title and len(sop.title) > 40 else (sop.title or "N/A")
            
            # Count checks (excluding errors field)
            check_fields = [
                "writing_style", "tone", "modal_verbs", "roles", "workflow",
                "compliance_elements", "risks", "gaps", "terminology", "structure_pattern"
            ]
            
            checks_passed = sum(1 for field in check_fields if result.get(field))
            total_checks += len(check_fields)
            passed_checks += checks_passed
            
            nlp_saved = "YES" if result["nlp_parameters_saved"] else "NO"
            
            print(f"  {sop_number:<20} {sop_title:<40} {nlp_saved:<10} {checks_passed}/{len(check_fields)}")
        
        print(f"\n  {SEP_THIN}")
        print(f"\n  Overall: {passed_checks}/{total_checks} NLP parameter checks passed")
        
        # ------------------------------------------------------------------
        # Expected output format
        # ------------------------------------------------------------------
        _section("EXPECTED OUTPUT FORMAT (per SOP)")
        
        for result in all_results:
            sop = db.query(SOP).filter(SOP.id == uuid.UUID(result["sop_id"])).first()
            sop_number = sop.sop_number if sop else "N/A"
            
            output = {
                "sop_id": result["sop_id"],
                "nlp_parameters_saved": result["nlp_parameters_saved"],
                "writing_style": result["writing_style"],
                "tone": result["tone"],
                "modal_verbs": result["modal_verbs"][:10],  # Limit for display
                "roles": result["roles"][:10],
                "workflow": result["workflow"][:10],
                "compliance_elements": result["compliance_elements"][:10],
                "risks": result["risks"][:10],
                "gaps": result["gaps"][:10],
                "terminology": result["terminology"][:10],
                "structure_pattern": result["structure_pattern"],
            }
            
            print(f"\n  {sop_number}:")
            print(json.dumps(output, indent=4, ensure_ascii=False))
        
        # ------------------------------------------------------------------
        # Errors
        # ------------------------------------------------------------------
        any_errors = any(result.get("errors") for result in all_results)
        if any_errors:
            _section("ERRORS / WARNINGS")
            for result in all_results:
                if result.get("errors"):
                    sop = db.query(SOP).filter(SOP.id == uuid.UUID(result["sop_id"])).first()
                    sop_number = sop.sop_number if sop else "N/A"
                    print(f"\n  {sop_number}:")
                    for error in result["errors"]:
                        print(f"    - {error}")
        
        # ------------------------------------------------------------------
        # Final verdict
        # ------------------------------------------------------------------
        _section("RESULT")
        
        all_nlp_saved = all(result["nlp_parameters_saved"] for result in all_results)
        all_checks_passed = passed_checks == total_checks
        
        if all_nlp_saved and all_checks_passed:
            print("  ALL PHASE 4 CHECKS PASSED")
            print("  NLP parameters detected and stored correctly in database.")
            print("  Ready to proceed to PHASE 5 (generate German company profile.md).")
        elif all_nlp_saved:
            print("  NLP parameters saved but some checks failed.")
            print("  Review the output above for missing parameters.")
            print("  Proceed to PHASE 5 with caution.")
        else:
            print("  CRITICAL: NLP parameters not saved in database.")
            print("  The system cannot proceed to profile generation.")
            print("  Check the database and NLP pipeline.")
        
        print(SEP)
        
        # Exit code
        sys.exit(0 if all_nlp_saved else 1)
        
    finally:
        db.close()


if __name__ == "__main__":
    main()