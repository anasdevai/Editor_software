import logging
import os
import sys
from typing import Any, Dict, Optional
from uuid import UUID
from sqlalchemy.orm import Session

# Try importing from root nlp_pipeline, fallback to app.services.nlp.pipeline
try:
    import nlp_pipeline
    HAS_ROOT_NLP = True
except ImportError:
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        import nlp_pipeline
        HAS_ROOT_NLP = True
    except ImportError:
        HAS_ROOT_NLP = False

from app.models import SOP, SOPDetectedParameters, ClientProfile, ProfileVersion, ProfileHistoryEvent, ProfileSuggestion

logger = logging.getLogger(__name__)


def _find_existing_profile_for_sop(db: Session, sop_id: UUID, tenant_id: UUID) -> Optional[ClientProfile]:
    """
    Look up the ClientProfile already linked to this SOP via SOPDetectedParameters.
    This is the authoritative lookup — each SOP has exactly one profile.
    """
    existing_param = (
        db.query(SOPDetectedParameters)
        .filter(
            SOPDetectedParameters.sop_id == sop_id,
            SOPDetectedParameters.client_profile_id.isnot(None),
        )
        .order_by(SOPDetectedParameters.created_at.desc())
        .first()
    )
    if existing_param and existing_param.client_profile_id:
        profile = db.query(ClientProfile).filter(
            ClientProfile.id == existing_param.client_profile_id,
            ClientProfile.tenant_id == tenant_id,
        ).first()
        if profile:
            return profile
    return None


def cleanup_profile_for_deleted_sop(db: Session, sop_id: UUID) -> Dict[str, Any]:
    """
    Stage deletion of the per-SOP profile data when an SOP is deleted.

    This does not commit. The caller owns the surrounding SOP-delete transaction.
    A ClientProfile is deleted only when no other SOPDetectedParameters row still
    points at it, so older/shared profiles are not removed accidentally.
    """
    summary: Dict[str, Any] = {
        "deleted_profile_ids": [],
        "retained_profile_ids": [],
        "deleted_suggestions": 0,
        "detected_rows": 0,
    }

    detected_rows = (
        db.query(SOPDetectedParameters)
        .filter(SOPDetectedParameters.sop_id == sop_id)
        .all()
    )
    summary["detected_rows"] = len(detected_rows)

    profile_ids = []
    for row in detected_rows:
        if row.client_profile_id and row.client_profile_id not in profile_ids:
            profile_ids.append(row.client_profile_id)

    summary["deleted_suggestions"] += db.query(ProfileSuggestion).filter(
        ProfileSuggestion.sop_id == sop_id
    ).delete(synchronize_session=False)

    for profile_id in profile_ids:
        still_used = (
            db.query(SOPDetectedParameters.id)
            .filter(
                SOPDetectedParameters.client_profile_id == profile_id,
                SOPDetectedParameters.sop_id != sop_id,
            )
            .first()
        )
        if still_used:
            summary["retained_profile_ids"].append(str(profile_id))
            continue

        summary["deleted_suggestions"] += db.query(ProfileSuggestion).filter(
            ProfileSuggestion.profile_id == profile_id
        ).delete(synchronize_session=False)
        db.query(ProfileHistoryEvent).filter(
            ProfileHistoryEvent.client_profile_id == profile_id
        ).delete(synchronize_session=False)
        db.query(ProfileVersion).filter(
            ProfileVersion.profile_id == profile_id
        ).delete(synchronize_session=False)

        profile = db.query(ClientProfile).filter(ClientProfile.id == profile_id).first()
        if profile:
            db.delete(profile)
            summary["deleted_profile_ids"].append(str(profile_id))

    db.query(SOPDetectedParameters).filter(
        SOPDetectedParameters.sop_id == sop_id
    ).delete(synchronize_session=False)

    logger.info("[profile] cleanup staged for deleted SOP %s: %s", sop_id, summary)
    return summary


def analyze_and_store_sop_profile(
    db: Session,
    sop_id: UUID,
    sop_version_id: UUID,
    text: str,
    client_name: str = "Client",
    source_filename: str = "",
    user_id: Optional[UUID] = None,
) -> None:
    """
    Analyses an SOP via the NLP pipeline and stores:
      - SOPDetectedParameters  (always one per upload)
      - ClientProfile          (one PER SOP — created on first upload, updated on re-upload)

    Each SOP always has its own isolated ClientProfile row.
    The profile_name is based on SOP metadata so it is human-readable in the UI.
    Lookup is done via SOPDetectedParameters.sop_id → client_profile_id so that
    re-uploads of the same SOP always update the SAME profile row, never merge
    with a different SOP's profile.
    """
    try:
        sop = db.query(SOP).filter(SOP.id == sop_id).first()
        if not sop:
            logger.warning(f"SOP {sop_id} not found, skipping profile analysis.")
            return

        tenant_id = sop.tenant_id

        # ── 1. Run NLP analysis ──────────────────────────────────────────────
        if HAS_ROOT_NLP:
            analysis = nlp_pipeline.analyze_sop_industry_level(
                text,
                client_name=client_name,
                include_profile_md=True,
            )
        else:
            from app.services.nlp.pipeline import analyze_sop_text
            analysis = analyze_sop_text(text)

        # ── 2. Extract profile_json / profile_md from pipeline output ────────
        built_profile_json = analysis.get("client_profile") or {}
        built_profile_md   = analysis.get("profile_md")   or ""

        # Fallback: generate profile_md from the built profile json
        if not built_profile_md and built_profile_json and HAS_ROOT_NLP:
            try:
                built_profile_md = nlp_pipeline.generate_profile_md(built_profile_json)
            except Exception as md_err:
                logger.warning(f"generate_profile_md fallback failed: {md_err}")

        if not built_profile_md:
            built_profile_md = (
                f"# {client_name} SOP Profile\n\n"
                f"Profile generated from SOP: {source_filename}\n"
            )

        # ── 3. Build a human-readable unique profile name ────────────────────
        if client_name == "German_Pharma_SOP_Profile":
            profile_name = client_name
        elif sop.sop_number:
            profile_name = f"{sop.sop_number} - {sop.title or source_filename or str(sop_id)}"
        elif sop.title:
            profile_name = sop.title
        else:
            profile_name = source_filename or f"SOP {sop_id}"

        # ── 4. Find or create the ClientProfile FOR THIS SOP ─────────────────
        # Primary lookup: via SOPDetectedParameters.sop_id link (authoritative)
        profile = _find_existing_profile_for_sop(db, sop_id, tenant_id)

        if not profile:
            # Secondary lookup: by name (for legacy rows created before this fix)
            profile = (
                db.query(ClientProfile)
                .filter(
                    ClientProfile.name == profile_name,
                    ClientProfile.tenant_id == tenant_id,
                )
                .first()
            )
            # Make sure this named profile is actually linked to THIS sop (not another)
            if profile:
                linked_param = (
                    db.query(SOPDetectedParameters)
                    .filter(
                        SOPDetectedParameters.client_profile_id == profile.id,
                        SOPDetectedParameters.sop_id != sop_id,
                    )
                    .first()
                )
                if linked_param:
                    # This profile belongs to a DIFFERENT sop → don't reuse it
                    profile = None

        if not profile:
            # First time this SOP is analysed — create a brand-new profile
            profile = ClientProfile(
                tenant_id=tenant_id,
                name=profile_name,
                company_name=client_name,
                total_sops_analyzed=0,
                active_profile_json={},
                active_profile_md="",
            )
            db.add(profile)
            db.flush()  # get profile.id
            logger.info(
                f"[profile] Created NEW ClientProfile for sop_id={sop_id} "
                f"name='{profile_name}'"
            )
        else:
            # Re-upload of the same SOP — update the existing profile
            logger.info(
                f"[profile] Updating existing ClientProfile id={profile.id} "
                f"for sop_id={sop_id} name='{profile_name}'"
            )
            # Keep name in sync if SOP metadata changed
            profile.name = profile_name
            profile.company_name = client_name

        # ── 5. Save SOPDetectedParameters ────────────────────────────────────
        detected = SOPDetectedParameters(
            sop_id=sop_id,
            sop_version_id=sop_version_id,
            client_profile_id=profile.id,
            client_name=client_name,
            source_filename=source_filename,
            analysis_json=analysis,
            document_information=analysis.get("document_information"),
            writing_style=analysis.get("writing_style"),
            roles_raci=analysis.get("roles_raci"),
            workflows=analysis.get("workflows"),
            compliance_elements=analysis.get("compliance_elements"),
            risks_gaps=analysis.get("risks_gaps"),
            terminology=analysis.get("terminology"),
            structure_patterns=analysis.get("structure_patterns"),
            style_suggestions=analysis.get("style_suggestions"),
            readiness_check=analysis.get("readiness_check"),
        )
        db.add(detected)

        # ── 6. Update profile JSON & markdown ────────────────────────────────
        # For this SOP's OWN profile, we always use the freshly-built data.
        # No cross-SOP merging: each profile reflects exactly the one SOP it belongs to.
        existing_json = profile.active_profile_json or {}

        if existing_json and profile.total_sops_analyzed > 0:
            # Re-upload: merge to keep any manually-added rewrite_rules, but
            # overwrite all NLP-derived fields with the latest analysis.
            merged_json = dict(existing_json)
            merged_json.update(built_profile_json)
            # Preserve manually-added rewrite_rules
            if existing_json.get("rewrite_rules"):
                merged_json["rewrite_rules"] = _deduplicate_list(
                    built_profile_json.get("rewrite_rules", []) +
                    existing_json.get("rewrite_rules", [])
                )
        else:
            merged_json = built_profile_json

        merged_json["profile_version"] = existing_json.get("profile_version", "3.0")

        profile.active_profile_json = merged_json

        # Regenerate profile_md from the merged profile
        if HAS_ROOT_NLP and merged_json:
            try:
                profile.active_profile_md = nlp_pipeline.generate_profile_md(merged_json)
            except Exception as regen_err:
                logger.warning(f"profile_md regen failed: {regen_err}")
                profile.active_profile_md = built_profile_md
        else:
            profile.active_profile_md = built_profile_md

        profile.total_sops_analyzed = (profile.total_sops_analyzed or 0) + 1
        profile.current_version_id = None  # versioning disabled

        logger.info(
            f"[profile] Stored profile for sop_id={sop_id} "
            f"profile_id={profile.id} "
            f"profile_md_length={len(profile.active_profile_md)} chars "
            f"total_sops_analyzed={profile.total_sops_analyzed}"
        )

        # ── 7. Store style suggestions (dedup per profile+sop+rule) ──────────
        for sugg in analysis.get("style_suggestions", []):
            rule_text = sugg.get("suggestion") or sugg.get("suggested_rule") or ""
            if not rule_text:
                continue
            exists = db.query(ProfileSuggestion).filter(
                ProfileSuggestion.profile_id == profile.id,
                ProfileSuggestion.sop_id == sop_id,
                ProfileSuggestion.suggested_rule == rule_text,
            ).first()
            if not exists:
                db.add(ProfileSuggestion(
                    tenant_id=tenant_id,
                    profile_id=profile.id,
                    sop_id=sop_id,
                    suggestion_type=sugg.get("area") or sugg.get("suggestion_type") or "style",
                    suggested_rule=rule_text,
                    evidence_json=sugg,
                    confidence=0.8,
                    status="pending",
                ))

        db.commit()
        logger.info(f"[profile] Committed profile update for sop_id={sop_id}")

    except Exception:
        logger.exception("Failed to analyze and store SOP profile")
        db.rollback()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _deduplicate_list(lst: list) -> list:
    seen = []
    for item in lst:
        if item not in seen:
            seen.append(item)
    return seen


def merge_analysis_into_profile(existing_profile: Dict[str, Any], new_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges a new NLP pipeline analysis result into an existing profile JSON.
    Uses analysis["client_profile"] as the new data source.
    """
    merged = dict(existing_profile)
    new_prof = new_analysis.get("client_profile") or {}

    # 1. Basic metadata
    merged["client_name"] = (
        new_prof.get("client_name")
        or existing_profile.get("client_name")
        or new_analysis.get("client_name", "Client")
    )

    # 2. Lists of domains / departments / sop_types
    for key in ["detected_domains", "detected_departments", "detected_sop_types"]:
        merged[key] = _deduplicate_list(
            existing_profile.get(key, []) + new_prof.get(key, [])
        )

    # 3. preferred_style
    ex_style = existing_profile.get("preferred_style", {}) or {}
    n_style  = new_prof.get("preferred_style", {}) or {}
    merged_style = dict(ex_style)
    for k in ["formality", "tone", "directive_wording", "writing_complexity", "primary_format"]:
        val = n_style.get(k) or ex_style.get(k)
        if val:
            merged_style[k] = val
    merged["preferred_style"] = merged_style

    # 4. modal_language
    ex_modal, n_modal = existing_profile.get("modal_language", {}) or {}, new_prof.get("modal_language", {}) or {}
    merged_modal = dict(ex_modal)
    for k, v in n_modal.items():
        if isinstance(v, list) and isinstance(ex_modal.get(k), list):
            merged_modal[k] = _deduplicate_list(ex_modal[k] + v)
        else:
            merged_modal[k] = v
    merged["modal_language"] = merged_modal

    # 5. common_sections
    merged["common_sections"] = _deduplicate_list(
        existing_profile.get("common_sections", []) + new_prof.get("common_sections", [])
    )

    # 6. terminology
    ex_term = existing_profile.get("terminology", {}) or {}
    n_term  = new_prof.get("terminology", {}) or new_analysis.get("terminology", {}) or {}
    merged_term = {}
    for sub_key in ["acronyms", "controlled_terms", "domain_terms"]:
        merged_term[sub_key] = _deduplicate_list(
            (ex_term.get(sub_key) or []) + (n_term.get(sub_key) or [])
        )
    ex_phrases = ex_term.get("key_phrases", []) or []
    n_phrases  = n_term.get("key_phrases",  []) or []
    phrase_counts: Dict[str, int] = {}
    for p in ex_phrases + n_phrases:
        if isinstance(p, dict) and p.get("phrase"):
            phrase_counts[p["phrase"]] = phrase_counts.get(p["phrase"], 0) + p.get("count", 1)
    merged_term["key_phrases"] = [
        {"phrase": k, "count": v}
        for k, v in sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
    ]
    merged["terminology"] = merged_term

    # 7. compliance_elements
    ex_comp = existing_profile.get("compliance_elements", {}) or {}
    n_comp  = new_analysis.get("compliance_elements", {}) or {}
    merged_comp: Dict[str, Any] = {}
    for c_key in set(list(ex_comp.keys()) + list(n_comp.keys())):
        ex_val, n_val = ex_comp.get(c_key, []), n_comp.get(c_key, [])
        if isinstance(ex_val, list) and isinstance(n_val, list):
            merged_comp[c_key] = _deduplicate_list(ex_val + n_val)
        else:
            merged_comp[c_key] = n_val or ex_val
    merged["compliance_elements"] = merged_comp

    # 8. document_content_profile
    new_dcp = new_prof.get("document_content_profile") or {}
    merged["document_content_profile"] = new_dcp or merged.get("document_content_profile", {})

    # 9. workflow_patterns
    ex_wf = existing_profile.get("workflow_patterns", {}) or {}
    n_wf  = new_prof.get("workflow_patterns", {}) or {}
    merged_wf = dict(ex_wf)
    for k, v in n_wf.items():
        ex_v = ex_wf.get(k, {}) or {}
        merged_wf[k] = {
            "detected": v.get("detected") or ex_v.get("detected", False),
            "status":   v.get("status")   or ex_v.get("status", "not_detected"),
            "confidence": v.get("confidence") or ex_v.get("confidence", 0.0),
            "stages": _deduplicate_list(ex_v.get("stages", []) + v.get("stages", [])),
            "missing_core_stages": _deduplicate_list(
                ex_v.get("missing_core_stages", []) + v.get("missing_core_stages", [])
            ),
        }
    merged["workflow_patterns"] = merged_wf

    # 10. rewrite_rules
    merged["rewrite_rules"] = _deduplicate_list(
        existing_profile.get("rewrite_rules", []) + new_prof.get("rewrite_rules", [])
    )

    # 11. roles_raci
    ex_roles = existing_profile.get("roles_raci", {}) or {}
    n_roles  = new_analysis.get("roles_raci", {}) or new_prof.get("roles_raci", {}) or {}
    r_dict_ex = ex_roles.get("roles", ex_roles) if isinstance(ex_roles.get("roles"), dict) else ex_roles
    r_dict_n  = n_roles.get("roles", n_roles)   if isinstance(n_roles.get("roles"),  dict) else n_roles
    for role, spec in r_dict_n.items():
        if not isinstance(spec, dict):
            continue
        if role in r_dict_ex:
            r_dict_ex[role] = {
                "detected": spec.get("detected") or r_dict_ex[role].get("detected", False),
                "status":   spec.get("status")   or r_dict_ex[role].get("status", "not_detected"),
                "confidence": max(spec.get("confidence", 0.0), r_dict_ex[role].get("confidence", 0.0)),
                "matched_terms": _deduplicate_list(r_dict_ex[role].get("matched_terms", []) + spec.get("matched_terms", [])),
                "responsibility_actions": _deduplicate_list(r_dict_ex[role].get("responsibility_actions", []) + spec.get("responsibility_actions", [])),
                "raci_category": spec.get("raci_category") or r_dict_ex[role].get("raci_category"),
                "evidence": _deduplicate_list(r_dict_ex[role].get("evidence", []) + spec.get("evidence", []))[:20],
            }
        else:
            r_dict_ex[role] = spec
    merged["roles_raci"] = r_dict_ex

    # 12. risks_gaps
    ex_risks = existing_profile.get("risks_gaps", {}) or {}
    n_risks  = new_analysis.get("risks_gaps", {}) or new_prof.get("risks_gaps", {}) or {}
    ex_gaps  = ex_risks.get("gaps", {}) or {}
    n_gaps   = n_risks.get("gaps",  {}) or {}
    merged_gaps: Dict[str, Any] = {}
    for g_key in set(list(ex_gaps.keys()) + list(n_gaps.keys())):
        merged_gaps[g_key] = _deduplicate_list(ex_gaps.get(g_key, []) + n_gaps.get(g_key, []))
    merged["risks_gaps"] = {
        "gaps": merged_gaps,
        "risk_score": n_risks.get("risk_score") or ex_risks.get("risk_score") or {"score": 0.0, "level": "low"},
        "gap_count": sum(len(v) for v in merged_gaps.values()),
        "critical_focus_areas": _deduplicate_list(
            ex_risks.get("critical_focus_areas", []) + n_risks.get("critical_focus_areas", [])
        ),
    }

    # 13. structure_patterns
    ex_struct = existing_profile.get("structure_patterns", {}) or {}
    n_struct  = new_analysis.get("structure_patterns", {}) or new_prof.get("structure_patterns", {}) or {}
    merged["structure_patterns"] = {
        "section_labels": _deduplicate_list(ex_struct.get("section_labels", []) + n_struct.get("section_labels", [])),
        "primary_format": n_struct.get("primary_format") or ex_struct.get("primary_format") or "standard",
        "section_count": max(ex_struct.get("section_count", 0), n_struct.get("section_count", 0)),
        "headings_detected": _deduplicate_list(ex_struct.get("headings_detected", []) + n_struct.get("headings_detected", [])),
    }

    # 14. writing_style (always use latest)
    merged["writing_style"] = new_analysis.get("writing_style", {}) or new_prof.get("writing_style", {})

    return merged


def get_profile_context_for_llm(db: Session, client_profile_id: UUID) -> Dict[str, Any]:
    """Returns a compact dict from active_profile_json for LLM context."""
    profile = db.query(ClientProfile).filter(ClientProfile.id == client_profile_id).first()
    if not profile or not profile.active_profile_json:
        return {}
    return {
        "client_name": profile.company_name or profile.name,
        "terminology": profile.active_profile_json.get("terminology", []),
        "writing_style": profile.active_profile_json.get("writing_style", {}),
    }
