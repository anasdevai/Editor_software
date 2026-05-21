from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
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
    current_version_id: UUID | None = None
    
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
    latest_version = db.query(ProfileVersion).filter(
        ProfileVersion.profile_id == profile.id
    ).order_by(ProfileVersion.version_number.desc()).first()
    previous_version = latest_version.version_number if latest_version else 0
    new_version_num = previous_version + 1

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
    active_json["profile_version"] = new_version_num
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
        + f"\n\n## Learned Style Update v{new_version_num}\n\n"
        + "- Source action: "
        + suggestion.action
        + "\n- Change: accepted rewrite/improve style reinforces formal company wording, modal-control language, terminology alignment, and structured responsibilities.\n"
    )
    profile.active_profile_json = active_json
    profile.active_profile_md = profile_md

    new_version = ProfileVersion(
        profile_id=profile.id,
        version_number=new_version_num,
        rules_json=active_json,
        profile_md=profile_md,
        change_reason=change_reason or f"Saved accepted {suggestion.action} style as profile version",
        source_sop_id=suggestion.sop_id,
        source_version_id=suggestion.accepted_version_id or suggestion.sop_version_id,
        detected_parameters_snapshot={
            "source_action": suggestion.action,
            "suggestion_id": str(suggestion.id),
            "changed_parameters": ["tone", "style", "terminology", "structure_pattern"],
            "suggestion_metadata": suggestion.metadata_json,
        },
    )
    db.add(new_version)
    db.flush()
    profile.current_version_id = new_version.id

    history = ProfileHistoryEvent(
        client_profile_id=profile.id,
        profile_version_id=new_version.id,
        source_sop_id=suggestion.sop_id,
        event_type="accepted_style_saved",
        event_summary=f"Accepted {suggestion.action} style saved as profile version {new_version_num}",
        diff_json={
            "previous_profile_version": previous_version,
            "new_profile_version": new_version_num,
            "changed_parameters": ["tone", "style", "terminology", "structure_pattern"],
        },
        after_snapshot=active_json,
    )
    db.add(history)

    suggestion_meta = dict(suggestion.metadata_json or {})
    suggestion_meta["profile_auto_updated"] = True
    suggestion_meta["profile_auto_update_result"] = {
        "profile_updated": True,
        "previous_profile_version": previous_version,
        "new_profile_version": new_version_num,
        "source_action": suggestion.action,
        "changed_parameters": ["tone", "style", "terminology", "structure_pattern"],
    }
    suggestion.metadata_json = suggestion_meta

    sop = db.query(SOP).filter(SOP.id == suggestion.sop_id).first() if suggestion.sop_id else None
    return {
        "profile_updated": True,
        "previous_profile_version": previous_version,
        "new_profile_version": new_version_num,
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
    """List all client profiles."""
    profiles = db.query(ClientProfile).all()
    return profiles


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
    """List all versions for a profile."""
    versions = db.query(ProfileVersion).filter(ProfileVersion.profile_id == profile_id).order_by(ProfileVersion.version_number.desc()).all()
    return versions

@router.get("/api/client-profiles/{profile_id}/versions/{version_id}")
def get_profile_version(profile_id: UUID, version_id: UUID, db: Session = Depends(get_db)):
    """Get a specific version (including raw JSON rules)."""
    version = db.query(ProfileVersion).filter(ProfileVersion.id == version_id, ProfileVersion.profile_id == profile_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="ProfileVersion not found")
    return {
        "id": version.id,
        "version_number": version.version_number,
        "rules_json": version.rules_json,
        "profile_md": version.profile_md,
        "change_reason": version.change_reason,
        "created_at": version.created_at
    }

@router.get("/api/client-profiles/{profile_id}/profile.md")
def get_profile_markdown(profile_id: UUID, db: Session = Depends(get_db)):
    """Get the active profile markdown directly."""
    profile = db.query(ClientProfile).filter(ClientProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="ClientProfile not found")
    if not profile.active_profile_md:
        return {"markdown": "# Profile\nNo active profile content yet."}
    return {"markdown": profile.active_profile_md}


@router.post("/api/client-profiles/{profile_id}/versions/{version_id}/activate")
def activate_profile_version(
    profile_id: UUID,
    version_id: UUID,
    payload: ProfileActivateVersionRequest | None = None,
    db: Session = Depends(get_db),
):
    """Make an existing profile version active so editor actions use it immediately."""
    profile = db.query(ClientProfile).filter(ClientProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="ClientProfile not found")
    version = db.query(ProfileVersion).filter(
        ProfileVersion.id == version_id,
        ProfileVersion.profile_id == profile_id,
    ).first()
    if not version:
        raise HTTPException(status_code=404, detail="ProfileVersion not found")

    profile.current_version_id = version.id
    profile.active_profile_md = version.profile_md
    profile.active_profile_json = version.rules_json

    history = ProfileHistoryEvent(
        client_profile_id=profile.id,
        profile_version_id=version.id,
        source_sop_id=version.source_sop_id,
        event_type="version_activated",
        event_summary=f"Activated profile version {version.version_number}",
        diff_json={
            "activated_version": version.version_number,
            "version_id": str(version.id),
        },
        after_snapshot=version.rules_json,
        created_by="manual_ui_activation",
    )
    db.add(history)
    db.commit()
    return {
        "profile_id": str(profile.id),
        "active_version_id": str(version.id),
        "active_version_number": version.version_number,
        "message": f"Profile version {version.version_number} activated",
    }


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

    latest_version = db.query(ProfileVersion).filter(
        ProfileVersion.profile_id == profile.id
    ).order_by(ProfileVersion.version_number.desc()).first()
    previous_version = latest_version.version_number if latest_version else 0
    new_version_num = previous_version + 1

    base_json = dict(profile.active_profile_json or {})
    merged_json = dict(base_json)
    if isinstance(payload.rules_json, dict):
        merged_json.update(payload.rules_json)
    merged_json["profile_version"] = new_version_num
    merged_json["manual_edit"] = True

    profile.active_profile_md = payload.profile_md
    profile.active_profile_json = merged_json

    new_version = ProfileVersion(
        profile_id=profile.id,
        version_number=new_version_num,
        rules_json=merged_json,
        profile_md=payload.profile_md,
        change_reason=payload.change_reason or "Manual profile edit saved from UI",
        source_sop_id=latest_version.source_sop_id if latest_version else None,
        source_version_id=latest_version.source_version_id if latest_version else None,
        detected_parameters_snapshot={
            "manual_edit": True,
            "source": "profile_workspace_ui",
        },
    )
    db.add(new_version)
    db.flush()
    profile.current_version_id = new_version.id

    history = ProfileHistoryEvent(
        client_profile_id=profile.id,
        profile_version_id=new_version.id,
        source_sop_id=new_version.source_sop_id,
        event_type="manual_profile_edit",
        event_summary=f"Manual profile edit saved as version {new_version_num}",
        diff_json={
            "previous_profile_version": previous_version,
            "new_profile_version": new_version_num,
            "change_reason": payload.change_reason or "Manual profile edit saved from UI",
        },
        after_snapshot=merged_json,
        created_by="profile_workspace_ui",
    )
    db.add(history)
    db.commit()
    return {
        "profile_updated": True,
        "profile_id": str(profile.id),
        "previous_profile_version": previous_version,
        "new_profile_version": new_version_num,
        "current_version_id": str(new_version.id),
        "message": f"Manual profile edit saved as version {new_version_num}",
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
        # Calculate new version number first
        latest_version = db.query(ProfileVersion).filter(
            ProfileVersion.profile_id == profile.id
        ).order_by(ProfileVersion.version_number.desc()).first()
        new_version_num = (latest_version.version_number + 1) if latest_version else 1

        # 1. Update ClientProfile active_profile_json
        active_json = dict(profile.active_profile_json or {})
        rules = list(active_json.get("rewrite_rules", []))
        if sugg.suggested_rule not in rules:
            rules.append(sugg.suggested_rule)
        active_json["rewrite_rules"] = rules
        active_json["profile_version"] = f"{new_version_num}.0"
        profile.active_profile_json = active_json
        
        # 2. Update active_profile_md
        try:
            import nlp_pipeline
            profile_md = nlp_pipeline.generate_profile_md(active_json)
        except Exception:
            profile_md = profile.active_profile_md or "# Profile"
        profile.active_profile_md = profile_md
        
        # 3. Create a new ProfileVersion
        new_version = ProfileVersion(
            profile_id=profile.id,
            version_number=new_version_num,
            rules_json=active_json,
            profile_md=profile_md,
            source_sop_id=sugg.sop_id,
            change_reason=f"Accepted suggestion: {sugg.suggested_rule[:60]}",
            detected_parameters_snapshot=sugg.evidence_json
        )
        db.add(new_version)
        db.flush()
        
        profile.current_version_id = new_version.id
        
        # 4. Create ProfileHistoryEvent
        history = ProfileHistoryEvent(
            client_profile_id=profile.id,
            profile_version_id=new_version.id,
            event_type="SUGGESTION_ACCEPTED",
            event_summary=f"Suggestion accepted: {sugg.suggested_rule[:100]}",
            after_snapshot=active_json,
            source_sop_id=sugg.sop_id
        )
        db.add(history)
        
    elif status == "rejected":
        # Create ProfileHistoryEvent for rejection
        history = ProfileHistoryEvent(
            client_profile_id=profile.id,
            profile_version_id=profile.current_version_id,
            event_type="SUGGESTION_REJECTED",
            event_summary=f"Suggestion rejected: {sugg.suggested_rule[:100]}. Reason: {payload.rejection_reason or 'None'}",
            after_snapshot=profile.active_profile_json,
            source_sop_id=sugg.sop_id
        )
        db.add(history)
        
    db.commit()
    return {"status": "success", "suggestion_status": status}
