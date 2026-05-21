from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
import uuid

from .database import get_db
from .models import ClientProfile, ProfileVersion, ProfileSuggestion, ProfileAuditLog, User
from .schemas import (
    ClientProfileCreate, ClientProfileResponse, 
    ProfileSuggestionResponse, AcceptRejectSuggestionRequest,
    ProfileVersionResponse, ProfileDetectionOutput
)
from .services.profile_analysis import analyze_sop_text_profile, analyze_sop_traceable

router = APIRouter(prefix="/api/profiles", tags=["profiles"])

# Mock tenant_id for now as per existing pattern in other routes
MOCK_TENANT_ID = "00000000-0000-0000-0000-000000000000"


def _safe_confidence(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

@router.post("", response_model=ClientProfileResponse)
def create_profile(payload: ClientProfileCreate, db: Session = Depends(get_db)):
    """
    Create a new client profile.
    """
    new_profile = ClientProfile(
        tenant_id=MOCK_TENANT_ID,
        name=payload.name,
        description=payload.description,
        domain=payload.domain
    )
    db.add(new_profile)
    db.commit()
    db.refresh(new_profile)
    
    # Create initial empty version
    initial_version = ProfileVersion(
        profile_id=new_profile.id,
        version_number=1,
        rules_json={
            "terminology": [],
            "preferred_wording": [],
            "forbidden_wording": [],
            "writing_style": {}
        }
    )
    db.add(initial_version)
    db.commit()
    db.refresh(initial_version)
    
    new_profile.current_version_id = initial_version.id
    db.commit()
    db.refresh(new_profile)
    
    return new_profile

@router.get("", response_model=List[ClientProfileResponse])
def list_profiles(db: Session = Depends(get_db)):
    """
    List all available client profiles.
    """
    return db.query(ClientProfile).all()

@router.post("/detect", response_model=ProfileDetectionOutput)
def detect_profile_from_text(payload: dict):
    """
    Detect an SOP profile from raw text without persisting suggestions.
    Useful for KL Assistant / chatbot context and smoke testing.
    """
    text = str(payload.get("text") or payload.get("content") or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required")
    try:
        result = analyze_sop_text_profile(
            text,
            use_llm=bool(payload.get("use_llm", False)),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Profile detection failed: {exc}")

    suggestions = []
    for suggestion in result.get("profile_suggestions") or []:
        suggestions.append(
            {
                "suggestion_type": suggestion.get("suggestion_type", "general"),
                "suggested_rule": suggestion.get("suggested_rule", ""),
                "evidence": [
                    {
                        "text": suggestion.get("evidence_from_document", ""),
                        **(suggestion.get("evidence_metadata") or {}),
                    }
                ]
                if suggestion.get("evidence_from_document")
                else [],
                "confidence": _safe_confidence(suggestion.get("confidence_score")),
            }
        )
    return {
        "summary": result.get("summary", ""),
        "detected_domain": result.get("detected_domain", ""),
        "suggestions": suggestions,
        "overall_confidence_score": result.get("overall_confidence_score", 0.0),
    }

@router.get("/{profile_id}", response_model=ClientProfileResponse)
def get_profile(profile_id: UUID, db: Session = Depends(get_db)):
    """
    Get a specific client profile by ID.
    """
    profile = db.query(ClientProfile).filter(ClientProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile

@router.get("/{profile_id}/versions", response_model=List[ProfileVersionResponse])
def list_profile_versions(profile_id: UUID, db: Session = Depends(get_db)):
    """
    List all versions for a profile.
    """
    return db.query(ProfileVersion).filter(ProfileVersion.profile_id == profile_id).all()

@router.get("/suggestions/pending", response_model=List[ProfileSuggestionResponse])
def list_pending_suggestions(db: Session = Depends(get_db), profile_id: Optional[UUID] = None):
    """
    List all pending profile suggestions.
    """
    query = db.query(ProfileSuggestion).filter(ProfileSuggestion.status == "pending")
    if profile_id:
        query = query.filter(ProfileSuggestion.profile_id == profile_id)
    return query.all()

@router.post("/suggestions/{suggestion_id}/review")
def review_suggestion(
    suggestion_id: UUID, 
    payload: AcceptRejectSuggestionRequest, 
    db: Session = Depends(get_db)
):
    """
    Accept or reject a profile suggestion.
    If accepted, it updates the current version of the associated profile.
    """
    suggestion = db.query(ProfileSuggestion).filter(ProfileSuggestion.id == suggestion_id).first()
    if not suggestion:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    
    suggestion.status = payload.status
    suggestion.rejection_reason = payload.rejection_reason
    
    if payload.status == "accepted" and suggestion.profile_id:
        profile = db.query(ClientProfile).filter(ClientProfile.id == suggestion.profile_id).first()
        if profile and profile.current_version_id:
            version = db.query(ProfileVersion).filter(ProfileVersion.id == profile.current_version_id).first()
            if version and not version.is_locked:
                # Update rules_json
                rules = dict(version.rules_json or {})
                s_type = suggestion.suggestion_type.lower()
                
                if s_type not in rules:
                    rules[s_type] = []
                
                if isinstance(rules[s_type], list):
                    if suggestion.suggested_rule not in rules[s_type]:
                        rules[s_type].append(suggestion.suggested_rule)
                
                rules["profile_version"] = f"{version.version_number}.0"
                version.rules_json = rules
                db.add(version)
                
                profile.active_profile_json = rules
                try:
                    import nlp_pipeline
                    profile.active_profile_md = nlp_pipeline.generate_profile_md(rules)
                except Exception:
                    pass
                db.add(profile)
    
    # Audit Log
    log = ProfileAuditLog(
        tenant_id=MOCK_TENANT_ID,
        action=f"suggestion_{payload.status}",
        profile_id=suggestion.profile_id,
        sop_id=suggestion.sop_id,
        details_json={
            "suggestion_id": str(suggestion_id),
            "rejection_reason": payload.rejection_reason
        }
    )
    db.add(log)
    
    db.commit()
    return {"message": f"Suggestion {payload.status}"}


@router.post("/analyze")
async def analyze_profile(
    profile_id: UUID,
    sop_id: Optional[UUID] = None,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Analyze an SOP and generate profile suggestions.
    """
    # 1. Run Analysis with Traceability
    try:
        # Use file.file which is a SpooledTemporaryFile
        analysis_result = analyze_sop_traceable(file.file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
    # 2. Persist Suggestions as Pending
    suggestions_created_count = 0
    if "profile_suggestions" in analysis_result:
        for sug in analysis_result["profile_suggestions"]:
            # Combine LLM evidence snippet with our traceable metadata
            evidence_data = {
                "snippet": sug.get("evidence_from_document", ""),
                "metadata": sug.get("evidence_metadata", {})
            }
            
            new_sug = ProfileSuggestion(
                tenant_id=MOCK_TENANT_ID,
                profile_id=profile_id,
                sop_id=sop_id,
                suggestion_type=sug.get("suggestion_type", "general"),
                suggested_rule=sug.get("suggested_rule", ""),
                evidence_json=evidence_data,
                confidence=_safe_confidence(sug.get("confidence_score")),
                status="pending"
            )
            db.add(new_sug)
            suggestions_created_count += 1
            
    db.commit()
    
    # Audit log the analysis run
    log = ProfileAuditLog(
        tenant_id=MOCK_TENANT_ID,
        action="profile_analysis_run",
        profile_id=profile_id,
        sop_id=sop_id,
        details_json={
            "filename": file.filename,
            "suggestions_generated": suggestions_created_count
        }
    )
    db.add(log)
    db.commit()
    
    return {
        "summary": analysis_result.get("summary", ""),
        "suggestions_count": suggestions_created_count,
        "detected_domain": analysis_result.get("detected_domain", ""),
        "analysis_raw": analysis_result
    }
