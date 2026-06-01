from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from uuid import UUID
from typing import List, Any, Dict
from pydantic import BaseModel
import logging

from .database import get_db
from .models import ClientProfile, ProfileVersion, SOPDetectedParameters, ProfileHistoryEvent, ProfileSuggestion, AISuggestion, SOP
from .schemas import ProfileSuggestionResponse, AcceptRejectSuggestionRequest

logger = logging.getLogger(__name__)

router = APIRouter()

# Schema definitions for Phase 1 (read-only)
class ProfileVersionResponse(BaseModel):
    id: UUID
    version_number: int
    change_reason: str | None = None
    created_at: Any
    
    class Config:
        from_attributes = True

class ClientProfileResponse(BaseModel):
    id: UUID
    name: str
    company_name: str | None = None
    total_sops_analyzed: int
    active_profile_md: str | None = None
    active_profile_json: Any = None
    current_version_id: UUID | None = None
    created_at: Any = None
    updated_at: Any = None

    class Config:
        from_attributes = True

class SOPDetectedParametersResponse(BaseModel):
    id: UUID
    sop_id: UUID | None = None
    sop_version_id: UUID | None = None
    source_filename: str | None = None
    created_at: Any
    
    class Config:
        from_attributes = True


class ProfileFromAcceptedStyleRequest(BaseModel):
    suggestion_id: UUID | None = None
    source_action: str | None = None
    source_sop_id: UUID | None = None
    change_reason: str | None = None


class ProfileManualVersionRequest(BaseModel):
    profile_md: str
    rules_json: Dict[str, Any] | None = None
    change_reason: str | None = None


class ProfileActivateVersionRequest(BaseModel):
    change_reason: str | None = None


def save_profile_version_from_accepted_suggestion(
    db: Session,
    *,
    profile: ClientProfile,
    suggestion: AISuggestion,
    change_reason: str | None = None,
) -> dict[str, Any]:
    active_json = dict(profile.active_profile_json or {})
    rewrite_rules = list(active_json.get("rewrite_rules") or [])
    learned_rule = (
        f"Accepted {suggestion.action} style from SOP action {suggestion.id}: "
        "preserve formal German pharmaceutical SOP register, controlled modal language, "
        "traceable responsibilities, and section-scoped rewrite behavior."
    )
    if learned_rule not in rewrite_rules:
        rewrite_rules.append(learned_rule)
    active_json["rewrite_rules"] = rewrite_rules
    active_json["profile_version"] = "1.0"
    active_json["changed_parameters"] = ["tone", "style", "terminology", "structure_pattern"]
    active_json["last_learned_from"] = {
        "source_action": suggestion.action,
        "source_sop_id": str(suggestion.sop_id) if suggestion.sop_id else None,
        "source_sop_version_id": str(suggestion.accepted_version_id or suggestion.sop_version_id)
        if (suggestion.accepted_version_id or suggestion.sop_version_id)
        else None,
        "suggestion_id": str(suggestion.id),
    }

    profile_md = profile.active_profile_md or f"# {profile.name}\n"
    profile_md = (
        profile_md.rstrip()
        + f"\n\n## Learned Style Update\n\n"
        + "- Source action: "
        + suggestion.action
        + "\n- Change: accepted rewrite/improve style reinforces formal company wording, modal-control language, terminology alignment, and structured responsibilities.\n"
    )
    profile.active_profile_json = active_json
    profile.active_profile_md = profile_md
    profile.current_version_id = None

    suggestion_meta = dict(suggestion.metadata_json or {})
    suggestion_meta["profile_auto_updated"] = True
    suggestion_meta["profile_auto_update_result"] = {
        "profile_updated": True,
        "source_action": suggestion.action,
        "changed_parameters": ["tone", "style", "terminology", "structure_pattern"],
    }
    suggestion.metadata_json = suggestion_meta
    flag_modified(suggestion, "metadata_json")

    sop = db.query(SOP).filter(SOP.id == suggestion.sop_id).first() if suggestion.sop_id else None
    return {
        "profile_updated": True,
        "source_action": suggestion.action,
        "source_sop_id": sop.sop_number if sop else str(suggestion.sop_id),
        "source_sop_uuid": str(suggestion.sop_id) if suggestion.sop_id else None,
        "source_sop_version_id": str(suggestion.accepted_version_id or suggestion.sop_version_id)
        if (suggestion.accepted_version_id or suggestion.sop_version_id)
        else None,
        "changed_parameters": ["tone", "style", "terminology", "structure_pattern"],
    }

@router.get("/api/client-profiles", response_model=List[ClientProfileResponse])
def list_client_profiles(db: Session = Depends(get_db)):
    """List all client profiles, newest first."""
    profiles = (
        db.query(ClientProfile)
        .order_by(ClientProfile.created_at.desc())
        .all()
    )
    return profiles


@router.get("/api/client-profiles/by-sop/{sop_id}", response_model=ClientProfileResponse)
def get_client_profile_by_sop(sop_id: UUID, db: Session = Depends(get_db)):
    """Return the ClientProfile for a specific SOP (looked up via SOPDetectedParameters)."""
    param = (
        db.query(SOPDetectedParameters)
        .filter(
            SOPDetectedParameters.sop_id == sop_id,
            SOPDetectedParameters.client_profile_id.isnot(None),
        )
        .order_by(SOPDetectedParameters.created_at.desc())
        .first()
    )
    if not param or not param.client_profile_id:
        raise HTTPException(status_code=404, detail="No profile found for this SOP")
    profile = db.query(ClientProfile).filter(ClientProfile.id == param.client_profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="ClientProfile not found")
    return profile


@router.get("/api/client-profiles/by-name/{profile_name}", response_model=ClientProfileResponse)
def get_client_profile_by_name(profile_name: str, db: Session = Depends(get_db)):
    """Get a profile by stable display name."""
    profile = db.query(ClientProfile).filter(ClientProfile.name == profile_name).first()
    if not profile:
        raise HTTPException(status_code=404, detail="ClientProfile not found")
    return profile


@router.get("/api/client-profiles/{profile_id}", response_model=ClientProfileResponse)
def get_client_profile(profile_id: UUID, db: Session = Depends(get_db)):
    """Get a specific client profile."""
    profile = db.query(ClientProfile).filter(ClientProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="ClientProfile not found")
    return profile


@router.get("/api/client-profiles/{profile_id}/versions", response_model=List[ProfileVersionResponse])
def list_profile_versions(profile_id: UUID, db: Session = Depends(get_db)):
    """List all versions for a profile (versioning disabled — returns empty list)."""
    return []


@router.get("/api/client-profiles/{profile_id}/versions/{version_id}")
def get_profile_version(profile_id: UUID, version_id: UUID, db: Session = Depends(get_db)):
    """Get a specific version (versioning is disabled)."""
    raise HTTPException(status_code=404, detail="ProfileVersion not found (versioning is disabled)")


@router.get("/api/client-profiles/{profile_id}/profile.md")
def get_profile_markdown(profile_id: UUID, db: Session = Depends(get_db)):
    """Get the active profile.md markdown content directly."""
    profile = db.query(ClientProfile).filter(ClientProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="ClientProfile not found")
    if not profile.active_profile_md:
        return {"markdown": "# Profile\nNo active profile content yet."}
    return {"markdown": profile.active_profile_md}


@router.delete("/api/client-profiles/{profile_id}/profile.md")
def delete_profile_markdown(profile_id: UUID, db: Session = Depends(get_db)):
    """
    Clear (delete) the active profile.md and profile JSON for this profile.
    The profile row itself is kept — only the generated markdown content is wiped
    so it can be regenerated by re-uploading or re-analysing the SOP.
    """
    profile = db.query(ClientProfile).filter(ClientProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="ClientProfile not found")

    had_content = bool(profile.active_profile_md)
    profile.active_profile_md = None
    profile.active_profile_json = None
    profile.current_version_id = None

    # Log the deletion event
    try:
        event = ProfileHistoryEvent(
            client_profile_id=profile.id,
            event_type="profile_md_deleted",
            event_summary=f"profile.md deleted by user for profile '{profile.name}'",
            metadata_json={"had_content": had_content},
        )
        db.add(event)
    except Exception:
        pass  # non-critical

    db.commit()
    db.refresh(profile)
    logger.info("[profile] deleted profile.md for profile_id=%s name=%s", profile_id, profile.name)
    return {
        "deleted": True,
        "profile_id": str(profile_id),
        "profile_name": profile.name,
        "had_content": had_content,
        "message": "profile.md has been cleared. Re-upload the SOP to regenerate it.",
    }


@router.delete("/api/client-profiles/{profile_id}")
def delete_client_profile(profile_id: UUID, db: Session = Depends(get_db)):
    """
    Completely delete the ClientProfile row and all cascade-related entries.
    """
    profile = db.query(ClientProfile).filter(ClientProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="ClientProfile not found")

    db.delete(profile)
    db.commit()
    logger.info("[profile] completely deleted ClientProfile profile_id=%s name=%s", profile_id, profile.name)
    return {
        "deleted": True,
        "profile_id": str(profile_id),
        "message": f"Client profile '{profile.name}' has been completely deleted.",
    }


@router.post("/api/client-profiles/{profile_id}/versions/{version_id}/activate")
def activate_profile_version(
    profile_id: UUID,
    version_id: UUID,
    payload: ProfileActivateVersionRequest | None = None,
    db: Session = Depends(get_db),
):
    """Make an existing profile version active so editor actions use it immediately."""
    raise HTTPException(status_code=404, detail="ProfileVersion not found (versioning is disabled)")


@router.post("/api/client-profiles/{profile_id}/versions/manual")
def create_manual_profile_version(
    profile_id: UUID,
    payload: ProfileManualVersionRequest,
    db: Session = Depends(get_db),
):
    """Save manually edited profile markdown/json as a new version and make it active."""
    profile = db.query(ClientProfile).filter(ClientProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="ClientProfile not found")

    base_json = dict(profile.active_profile_json or {})
    merged_json = dict(base_json)
    if isinstance(payload.rules_json, dict):
        merged_json.update(payload.rules_json)
    merged_json["profile_version"] = "1.0"
    merged_json["manual_edit"] = True

    profile.active_profile_md = payload.profile_md
    profile.active_profile_json = merged_json
    profile.current_version_id = None

    db.commit()
    return {
        "profile_updated": True,
        "profile_id": str(profile.id),
        "message": "Manual profile edit saved successfully",
    }


@router.post("/api/client-profiles/{profile_id}/versions/from-accepted-style")
def create_profile_version_from_accepted_style(
    profile_id: UUID,
    payload: ProfileFromAcceptedStyleRequest,
    db: Session = Depends(get_db),
):
    """Create a new profile version from an explicitly accepted rewrite/improve suggestion."""
    profile = db.query(ClientProfile).filter(ClientProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="ClientProfile not found")

    query = db.query(AISuggestion).filter(
        AISuggestion.profile_id == profile_id,
        AISuggestion.status == "accepted",
    )
    if payload.suggestion_id:
        query = query.filter(AISuggestion.id == payload.suggestion_id)
    if payload.source_action:
        query = query.filter(AISuggestion.action == payload.source_action)
    if payload.source_sop_id:
        query = query.filter(AISuggestion.sop_id == payload.source_sop_id)
    suggestion = query.order_by(AISuggestion.accepted_at.desc(), AISuggestion.created_at.desc()).first()
    if not suggestion:
        raise HTTPException(status_code=404, detail="No accepted AI suggestion found for this profile")

    result = save_profile_version_from_accepted_suggestion(
        db,
        profile=profile,
        suggestion=suggestion,
        change_reason=payload.change_reason,
    )
    db.commit()
    return result


@router.get("/api/client-profiles/{profile_id}/suggestions", response_model=List[ProfileSuggestionResponse])
def list_profile_suggestions(profile_id: UUID, db: Session = Depends(get_db)):
    """List all suggestions for a client profile."""
    profile = db.query(ClientProfile).filter(ClientProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="ClientProfile not found")
    suggestions = db.query(ProfileSuggestion).filter(ProfileSuggestion.profile_id == profile_id).all()
    return suggestions


@router.post("/api/client-profiles/{profile_id}/suggestions/{suggestion_id}/action")
def accept_reject_suggestion(
    profile_id: UUID,
    suggestion_id: UUID,
    payload: AcceptRejectSuggestionRequest,
    db: Session = Depends(get_db)
):
    """Accept or reject a pending profile suggestion."""
    profile = db.query(ClientProfile).filter(ClientProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="ClientProfile not found")
        
    sugg = db.query(ProfileSuggestion).filter(
        ProfileSuggestion.id == suggestion_id,
        ProfileSuggestion.profile_id == profile_id
    ).first()
    if not sugg:
        raise HTTPException(status_code=404, detail="ProfileSuggestion not found")
        
    if sugg.status != "pending":
        raise HTTPException(status_code=400, detail=f"Suggestion is already {sugg.status}")
        
    status = payload.status.lower()
    if status not in ["accepted", "rejected"]:
        raise HTTPException(status_code=400, detail="Status must be 'accepted' or 'rejected'")
        
    sugg.status = status
    sugg.rejection_reason = payload.rejection_reason
    
    if status == "accepted":
        # 1. Update ClientProfile active_profile_json
        active_json = dict(profile.active_profile_json or {})
        rules = list(active_json.get("rewrite_rules", []))
        if sugg.suggested_rule not in rules:
            rules.append(sugg.suggested_rule)
        active_json["rewrite_rules"] = rules
        profile.active_profile_json = active_json
        
        # 2. Update active_profile_md
        try:
            import nlp_pipeline
            profile_md = nlp_pipeline.generate_profile_md(active_json)
        except Exception:
            profile_md = profile.active_profile_md or "# Profile"
        profile.active_profile_md = profile_md
        profile.current_version_id = None
        
    db.commit()
    return {"status": "success", "suggestion_status": status}
