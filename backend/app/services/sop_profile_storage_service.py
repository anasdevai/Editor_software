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
    # Attempt to add parent dir to path
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

def analyze_and_store_sop_profile(
    db: Session,
    sop_id: UUID,
    sop_version_id: UUID,
    text: str,
    client_name: str = "Client",
    source_filename: str = "",
    user_id: Optional[UUID] = None
) -> None:
    """
    Analyzes an SOP using the NLP pipeline and stores the detected parameters
    and profile history. Never fails the upload process (errors are caught and logged).
    """
    try:
        sop = db.query(SOP).filter(SOP.id == sop_id).first()
        if not sop:
            logger.warning(f"SOP {sop_id} not found, skipping profile analysis.")
            return

        tenant_id = sop.tenant_id

        # 1. Run NLP analysis
        if HAS_ROOT_NLP:
            analysis = nlp_pipeline.analyze_sop_industry_level(text, client_name=client_name)
        else:
            from app.services.nlp.pipeline import analyze_sop_text
            analysis = analyze_sop_text(text)

        # 2. Save SOPDetectedParameters
        detected = SOPDetectedParameters(
            sop_id=sop_id,
            sop_version_id=sop_version_id,
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
        
        # 3. Find or create ClientProfile
        profile = db.query(ClientProfile).filter(
            ClientProfile.name == client_name, 
            ClientProfile.tenant_id == tenant_id
        ).first()
        
        is_new_profile = False
        if not profile:
            profile = ClientProfile(
                tenant_id=tenant_id,
                name=client_name,
                company_name=client_name,
                total_sops_analyzed=0,
                active_profile_json={}
            )
            db.add(profile)
            db.flush()
            is_new_profile = True
            
        detected.client_profile_id = profile.id
        
        # Calculate new version number first
        latest_version = db.query(ProfileVersion).filter(
            ProfileVersion.profile_id == profile.id
        ).order_by(ProfileVersion.version_number.desc()).first()
        new_version_num = (latest_version.version_number + 1) if latest_version else 1
        
        # 4. Merge analysis into profile
        merged_json = merge_analysis_into_profile(profile.active_profile_json or {}, analysis)
        merged_json["profile_version"] = f"{new_version_num}.0"
        profile.active_profile_json = merged_json
        profile.total_sops_analyzed += 1
        
        # 5. Generate profile markdown
        if HAS_ROOT_NLP:
            profile_md = nlp_pipeline.generate_profile_md(merged_json)
        else:
            profile_md = "# Profile\nGenerated without root nlp_pipeline."
            
        profile.active_profile_md = profile_md
        
        # 6. Create ProfileVersion
        prof_version = ProfileVersion(
            profile_id=profile.id,
            version_number=new_version_num,
            rules_json=merged_json,
            profile_md=profile_md,
            source_sop_id=sop_id,
            source_version_id=sop_version_id,
            change_reason="Initial creation" if is_new_profile else "SOP Analysis update",
            detected_parameters_snapshot=analysis
        )
        db.add(prof_version)
        db.flush()
        
        profile.current_version_id = prof_version.id
        
        # 7. Append ProfileHistoryEvent
        event_type = "PROFILE_CREATED" if is_new_profile else "SOP_ANALYZED"
        history = ProfileHistoryEvent(
            client_profile_id=profile.id,
            profile_version_id=prof_version.id,
            event_type=event_type,
            event_summary=f"Profile updated via SOP {source_filename}",
            after_snapshot=merged_json,
            source_sop_id=sop_id,
            created_by=str(user_id) if user_id else "system"
        )
        db.add(history)
        
        # 8. Store detected profile suggestions
        style_suggs = analysis.get("style_suggestions", [])
        for sugg in style_suggs:
            rule_text = sugg.get("suggestion") or sugg.get("suggested_rule") or ""
            if not rule_text:
                continue
            
            # Check if this suggestion already exists for this profile and SOP
            existing_sugg = db.query(ProfileSuggestion).filter(
                ProfileSuggestion.profile_id == profile.id,
                ProfileSuggestion.sop_id == sop_id,
                ProfileSuggestion.suggested_rule == rule_text
            ).first()
            
            if not existing_sugg:
                new_sugg = ProfileSuggestion(
                    tenant_id=tenant_id,
                    profile_id=profile.id,
                    sop_id=sop_id,
                    suggestion_type=sugg.get("area") or sugg.get("suggestion_type") or "style",
                    suggested_rule=rule_text,
                    evidence_json=sugg,
                    confidence=0.8,
                    status="pending"
                )
                db.add(new_sugg)
        
        db.commit()
    except Exception as e:
        logger.exception("Failed to analyze and store SOP profile")
        db.rollback()


def _deduplicate_list(lst: list) -> list:
    seen = []
    for item in lst:
        if item not in seen:
            seen.append(item)
    return seen

def merge_analysis_into_profile(existing_profile: Dict[str, Any], new_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Properly merges new analysis into existing profile JSON structure.
    """
    merged = dict(existing_profile)
    
    # Extract client_profile if it exists, otherwise fallback to the root new_analysis dict
    new_prof = new_analysis.get("client_profile") or {}
    if not new_prof and "preferred_style" in new_analysis:
        new_prof = new_analysis

    # 1. Basic Metadata
    merged["client_name"] = new_prof.get("client_name") or existing_profile.get("client_name") or new_analysis.get("client_name", "Client")
    
    # 2. Lists of domains, departments, sop types
    for key in ["detected_domains", "detected_departments", "detected_sop_types"]:
        merged[key] = _deduplicate_list(existing_profile.get(key, []) + new_prof.get(key, []))

    # 3. preferred_style (formality, tone, directive_wording, writing_complexity, primary_format)
    ex_style = existing_profile.get("preferred_style", {}) or {}
    n_style = new_prof.get("preferred_style", {}) or {}
    merged_style = dict(ex_style)
    for k in ["formality", "tone", "directive_wording", "writing_complexity", "primary_format"]:
        val = n_style.get(k) or ex_style.get(k)
        if val:
            merged_style[k] = val
    merged["preferred_style"] = merged_style

    # 4. modal_language
    ex_modal = existing_profile.get("modal_language", {}) or {}
    n_modal = new_prof.get("modal_language", {}) or {}
    merged_modal = dict(ex_modal)
    for k, v in n_modal.items():
        if isinstance(v, list) and isinstance(ex_modal.get(k), list):
            merged_modal[k] = _deduplicate_list(ex_modal[k] + v)
        else:
            merged_modal[k] = v
    merged["modal_language"] = merged_modal

    # 5. common_sections
    merged["common_sections"] = _deduplicate_list(existing_profile.get("common_sections", []) + new_prof.get("common_sections", []))

    # 6. terminology
    ex_term = existing_profile.get("terminology", {}) or {}
    n_term = new_prof.get("terminology", {}) or {}
    if not n_term:
        n_term = new_analysis.get("terminology", {}) or {}
    
    merged_term = {}
    for sub_key in ["acronyms", "controlled_terms", "domain_terms"]:
        ex_list = ex_term.get(sub_key, []) or []
        n_list = n_term.get(sub_key, []) or []
        merged_term[sub_key] = _deduplicate_list(ex_list + n_list)
    merged["terminology"] = merged_term

    # 7. compliance_elements
    ex_comp = existing_profile.get("compliance_elements", {}) or {}
    n_comp = new_analysis.get("compliance_elements", {}) or {}
    
    merged_comp = {}
    all_comp_keys = set(list(ex_comp.keys()) + list(n_comp.keys()))
    for c_key in all_comp_keys:
        ex_val = ex_comp.get(c_key, [])
        n_val = n_comp.get(c_key, [])
        if isinstance(ex_val, list) and isinstance(n_val, list):
            merged_comp[c_key] = _deduplicate_list(ex_val + n_val)
        else:
            merged_comp[c_key] = n_val or ex_val
    merged["compliance_elements"] = merged_comp

    # 8. workflow_patterns
    ex_wf = existing_profile.get("workflow_patterns", {}) or {}
    n_wf = new_prof.get("workflow_patterns", {}) or {}
    merged_wf = dict(ex_wf)
    for k, v in n_wf.items():
        ex_v = ex_wf.get(k, {}) or {}
        merged_wf[k] = {
            "detected": v.get("detected") or ex_v.get("detected", False),
            "stages": _deduplicate_list(ex_v.get("stages", []) + v.get("stages", []))
        }
    merged["workflow_patterns"] = merged_wf

    # 9. rewrite_rules
    merged["rewrite_rules"] = _deduplicate_list(existing_profile.get("rewrite_rules", []) + new_prof.get("rewrite_rules", []))
    
    # 10. writing_style (raw)
    merged["writing_style"] = new_analysis.get("writing_style", {}) or new_prof.get("writing_style", {})

    return merged


def get_profile_context_for_llm(db: Session, client_profile_id: UUID) -> Dict[str, Any]:
    """
    Returns a compact dict from active_profile_json for LLM context.
    """
    profile = db.query(ClientProfile).filter(ClientProfile.id == client_profile_id).first()
    if not profile or not profile.active_profile_json:
        return {}
    
    return {
        "client_name": profile.company_name or profile.name,
        "terminology": profile.active_profile_json.get("terminology", []),
        "writing_style": profile.active_profile_json.get("writing_style", {})
    }
