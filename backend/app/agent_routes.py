from importlib import metadata
import os
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .database import get_db
from .services.agent_orchestrator import (
    DEEPAGENTS_AVAILABLE,
    DeepAgentExecutionError,
    ENABLE_REAL_DEEP_AGENTS,
    _agent_mode,
    compare_sops,
    create_sop_draft,
    generate_sop_preview,
    learn_template,
    run_sop_action,
)


agent_router = APIRouter(prefix="/api/agents", tags=["Agent Orchestrator"])


class AgentHealthResponse(BaseModel):
    ok: bool = True
    deepagents_available: bool
    deepagents_enabled: bool
    deepagents_version: str
    mode: str
    subagents: list[str]
    mandatory: bool = True
    configured_context_tokens: int
    minimum_recommended_context_tokens: int
    context_ready: bool
    warning: str | None = None


class SopActionAgentRequest(BaseModel):
    action: Literal[
        "rewrite",
        "improve",
        "summarize",
        "explain",
        "gap_check",
        "compliance",
        "rewrite_with_profile_style",
        "improve_with_profile_style",
    ]
    user_request: str = ""
    target_text: str = Field(..., min_length=1)
    active_sop_id: str | None = None
    source_sop_ids: list[str] = Field(default_factory=list)
    profile_id: str | None = None
    style_source_profile_id: str | None = None


class CompareSopsRequest(BaseModel):
    source_sop_ids: list[str] = Field(..., min_length=2)
    query: str = ""


class LearnTemplateRequest(BaseModel):
    source_sop_ids: list[str] = Field(..., min_length=2)
    client_name: str = "Client"
    template_name: str | None = None


class GenerateSopPreviewRequest(BaseModel):
    source_sop_ids: list[str] = Field(default_factory=list)
    template_id: str | None = None
    target_title: str = Field(..., min_length=3)
    target_department: str = ""
    target_sop_type: str = ""
    language: Literal["en", "de"] = "en"
    requirements: str = ""
    client_name: str = "Client"


class CreateSopDraftRequest(BaseModel):
    preview: dict[str, Any]
    title: str = Field(..., min_length=3)
    department: str = ""
    client_name: str = "Client"


@agent_router.get("/health", response_model=AgentHealthResponse)
def agent_health():
    configured_context = int(os.getenv("ACTION_MODEL_CONTEXT_TOKENS", "4096") or "4096")
    minimum_context = 8192
    return {
        "ok": True,
        "deepagents_available": DEEPAGENTS_AVAILABLE,
        "deepagents_enabled": ENABLE_REAL_DEEP_AGENTS,
        "deepagents_version": metadata.version("deepagents"),
        "mode": _agent_mode(),
        "subagents": [
            "rag_evidence_agent",
            "profile_template_agent",
            "rewrite_improve_agent",
            "compliance_agent",
            "sop_generation_agent",
        ],
        "mandatory": True,
        "configured_context_tokens": configured_context,
        "minimum_recommended_context_tokens": minimum_context,
        "context_ready": configured_context >= minimum_context,
        "warning": None if configured_context >= minimum_context else (
            "DeepAgents is mandatory and usually needs at least 8192 context tokens. "
            "Increase LM Studio model context and ACTION_MODEL_CONTEXT_TOKENS before production use."
        ),
    }


@agent_router.post("/sop-action")
def sop_action(payload: SopActionAgentRequest, db: Session = Depends(get_db)):
    try:
        return run_sop_action(
            db=db,
            action=payload.action,
            user_request=payload.user_request,
            target_text=payload.target_text,
            active_sop_id=payload.active_sop_id,
            source_sop_ids=payload.source_sop_ids,
            profile_id=payload.style_source_profile_id or payload.profile_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DeepAgentExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Mandatory DeepAgents SOP action failed: {type(exc).__name__}: {exc}",
        ) from exc


@agent_router.post("/compare-sops")
def compare_sops_endpoint(payload: CompareSopsRequest, db: Session = Depends(get_db)):
    try:
        return compare_sops(db, payload.source_sop_ids, query=payload.query)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@agent_router.post("/learn-template")
def learn_template_endpoint(payload: LearnTemplateRequest, db: Session = Depends(get_db)):
    try:
        result = learn_template(
            db,
            payload.source_sop_ids,
            client_name=payload.client_name,
            template_name=payload.template_name,
            persist=True,
        )
        db.commit()
        return result
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@agent_router.post("/generate-sop-preview")
def generate_sop_preview_endpoint(payload: GenerateSopPreviewRequest, db: Session = Depends(get_db)):
    try:
        result = generate_sop_preview(
            db=db,
            source_sop_ids=payload.source_sop_ids,
            target_title=payload.target_title,
            target_department=payload.target_department,
            target_sop_type=payload.target_sop_type,
            language=payload.language,
            requirements=payload.requirements,
            client_name=payload.client_name,
            template_id=payload.template_id,
        )
        db.commit()
        return result
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@agent_router.post("/create-sop-draft")
def create_sop_draft_endpoint(payload: CreateSopDraftRequest, db: Session = Depends(get_db)):
    try:
        result = create_sop_draft(
            db=db,
            preview=payload.preview,
            title=payload.title,
            department=payload.department,
            client_name=payload.client_name,
        )
        db.commit()
        return result
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
