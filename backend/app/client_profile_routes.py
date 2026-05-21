from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID
from typing import List, Any, Dict
from pydantic import BaseModel
import logging

from .database import get_db
from .models import ClientProfile, ProfileVersion, SOPDetectedParameters, ProfileHistoryEvent, ProfileSuggestion
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

@router.get("/api/client-profiles", response_model=List[ClientProfileResponse])
def list_client_profiles(db: Session = Depends(get_db)):
    """List all client profiles."""
    profiles = db.query(ClientProfile).all()
    return profiles

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
