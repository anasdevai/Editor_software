from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Optional, List
from uuid import UUID
from datetime import datetime

# ==========================================
# EDITOR COMPATIBILITY LAYER — REQUEST BODIES
# ==========================================

class CreateDocumentRequest(BaseModel):
    title: Optional[str] = None
    profile: str = "sop"
    doc_json: Optional[Any] = None
    metadata_json: Optional[Any] = None

class UpdateDocumentRequest(BaseModel):
    title: Optional[str] = None
    doc_json: Any
    metadata_json: Optional[Any] = None

class CreateVersionRequest(BaseModel):
    doc_json: Any
    change_summary: Optional[str] = None
    change_justification: Optional[str] = None # Enforced in later logic
    metadata_json: Optional[Any] = None
    suggestion_id: Optional[UUID] = None

class LinkRequest(BaseModel):
    source_id: UUID
    target_id: UUID
    link_type: str # 'sop-deviation', 'deviation-capa', etc.
    link_reason: Optional[str] = None
    rationale_text: Optional[str] = None

class UpdateVersionStatusRequest(BaseModel):
    status: str
    metadata_json: Optional[Any] = None


# ==========================================
# EDITOR COMPATIBILITY LAYER — RESPONSES
# These use old editor field names (doc_json, doc_id, status)
# Mapping: doc_json = content_json, status = external_status, doc_id = sop_id
# ==========================================

class EditorVersionResponse(BaseModel):
    """
    Old editor version response shape.
    doc_json   <- sop_versions.content_json
    doc_id     <- sop_versions.sop_id
    status     <- sop_versions.external_status
    """
    id: UUID
    doc_id: UUID                    # maps from sop_versions.sop_id
    version_number: str
    status: str                     # maps from sop_versions.external_status
    doc_json: Optional[Any]         # maps from sop_versions.content_json
    metadata_json: Optional[Any]
    effective_date: Optional[datetime] = None
    review_date: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    profile_error: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class EditorDocResponse(BaseModel):
    """
    Old editor top-level doc response shape.
    doc_json   <- sop_versions.content_json (from current version)
    status     <- sop_versions.external_status
    """
    id: UUID                        # sops.id
    title: Optional[str] = None
    doc_type: str = "sop"
    doc_json: Optional[Any]         # maps from current version.content_json
    metadata_json: Optional[Any]    # maps from current version.metadata_json
    current_version_id: Optional[UUID]
    version_number: Optional[str]   # from current version
    status: Optional[str]           # maps from current version.external_status
    created_at: datetime
    updated_at: datetime
    profile_error: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# Older alias kept for any internal usage (not used in new routes)
class VersionResponse(BaseModel):
    id: UUID
    document_id: UUID
    version_number: str
    doc_json: Any
    change_summary: Optional[str] = None
    status: str
    metadata_json: Optional[Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# DOMAIN MODELS — NATIVE SCHEMA FIELD NAMES
# content_json, external_status, sop_id etc.
# ==========================================

class SOPVersionResponse(BaseModel):
    id: UUID
    sop_id: UUID
    external_version_id: Optional[str] = None
    version_number: str
    external_status: Optional[str] = None
    effective_date: Optional[datetime] = None
    review_date: Optional[datetime] = None
    content_json: Optional[Any] = None
    metadata_json: Optional[Any] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SOPResponse(BaseModel):
    id: UUID
    tenant_id: UUID
    external_id: Optional[str] = None
    sop_number: str
    title: str
    department: Optional[str] = None
    client_id: Optional[str] = None
    client_name: Optional[str] = None
    category: Optional[str] = None
    document_family: Optional[str] = None
    source_system: Optional[str] = None
    is_active: bool
    current_version_id: Optional[UUID] = None
    # Embedded current version for GET /api/sops/{id}
    current_version: Optional[SOPVersionResponse] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ClientWorkspaceSOPResponse(BaseModel):
    id: UUID
    sop_number: str
    title: str
    category: Optional[str] = None
    document_family: Optional[str] = None
    department: Optional[str] = None
    current_version_id: Optional[UUID] = None
    current_version_number: Optional[str] = None
    version_count: int = 0
    status: Optional[str] = None
    updated_at: datetime


class ClientWorkspaceResponse(BaseModel):
    client_id: Optional[str] = None
    client_name: str
    sop_count: int = 0
    category_count: int = 0
    document_family_count: int = 0
    version_count: int = 0
    sops: List[ClientWorkspaceSOPResponse] = Field(default_factory=list)


class DeviationCreateUpdate(BaseModel):
    title: str
    deviation_number: str
    category: Optional[str] = None
    site: Optional[str] = None
    product_line: Optional[str] = None
    external_status: Optional[str] = None
    description_text: Optional[str] = None
    root_cause_text: Optional[str] = None
    impact_level: Optional[str] = None
    source_system: Optional[str] = None
    event_date: Optional[datetime] = None

class CapaCreateUpdate(BaseModel):
    title: str
    capa_number: str
    external_status: Optional[str] = None
    action_type: Optional[str] = None
    action_text: Optional[str] = None
    effectiveness_text: Optional[str] = None
    owner_name: Optional[str] = None
    due_date: Optional[datetime] = None
    effectiveness_status: Optional[str] = None
    source_system: Optional[str] = None

class AuditFindingCreateUpdate(BaseModel):
    audit_number: Optional[str] = None
    finding_number: Optional[str] = None
    authority: Optional[str] = None
    site: Optional[str] = None
    audit_date: Optional[datetime] = None
    question_text: Optional[str] = None
    finding_text: Optional[str] = None
    response_text: Optional[str] = None
    acceptance_status: Optional[str] = None
    source_system: Optional[str] = None

class DecisionCreateUpdate(BaseModel):
    title: str
    decision_number: Optional[str] = None
    decision_type: Optional[str] = None
    decision_statement: str
    rationale_text: Optional[str] = None
    risk_assessment_text: Optional[str] = None
    alternatives_text: Optional[str] = None
    final_conclusion: Optional[str] = None
    decision_date: Optional[datetime] = None
    decided_by_role: Optional[str] = None
    source_system: Optional[str] = None

class DatasetImportRequest(BaseModel):
    entities: List[dict] # Generic list of entities to import


class DeviationResponse(BaseModel):
    id: UUID
    tenant_id: UUID
    external_id: Optional[str] = None
    deviation_number: str
    title: str
    category: Optional[str] = None
    site: Optional[str] = None
    product_line: Optional[str] = None
    external_status: Optional[str] = None
    description_text: Optional[str] = None
    root_cause_text: Optional[str] = None
    impact_level: Optional[str] = None
    source_system: Optional[str] = None
    event_date: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CapaResponse(BaseModel):
    id: UUID
    tenant_id: UUID
    external_id: Optional[str] = None
    capa_number: str
    title: str
    external_status: Optional[str] = None
    action_type: Optional[str] = None
    action_text: Optional[str] = None
    effectiveness_text: Optional[str] = None
    owner_name: Optional[str] = None
    due_date: Optional[datetime] = None
    effectiveness_status: Optional[str] = None
    source_system: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AuditFindingResponse(BaseModel):
    id: UUID
    tenant_id: UUID
    external_id: Optional[str] = None
    audit_number: Optional[str] = None
    finding_number: Optional[str] = None
    authority: Optional[str] = None
    site: Optional[str] = None
    audit_date: Optional[datetime] = None
    question_text: Optional[str] = None
    finding_text: Optional[str] = None
    response_text: Optional[str] = None
    acceptance_status: Optional[str] = None
    source_system: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DecisionResponse(BaseModel):
    id: UUID
    tenant_id: UUID
    external_id: Optional[str] = None
    decision_number: Optional[str] = None
    decision_type: Optional[str] = None
    title: str
    decision_statement: str
    rationale_text: Optional[str] = None
    risk_assessment_text: Optional[str] = None
    alternatives_text: Optional[str] = None
    final_conclusion: Optional[str] = None
    decision_date: Optional[datetime] = None
    decided_by_role: Optional[str] = None
    source_system: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# CONTEXT / RELATED RESPONSES
# ==========================================

class DeviationContextResponse(BaseModel):
    deviation: DeviationResponse
    related_sops: List[SOPResponse] = []
    related_capas: List[CapaResponse] = []
    related_audits: List[AuditFindingResponse] = []
    related_decisions: List[DecisionResponse] = []

    model_config = ConfigDict(from_attributes=True)


class SopRelatedResponse(BaseModel):
    sop: SOPResponse
    related_sops: List[SOPResponse] = []
    related_deviations: List[DeviationResponse] = []
    related_capas: List[CapaResponse] = []
    related_audit_findings: List[AuditFindingResponse] = []
    related_decisions: List[DecisionResponse] = []

    model_config = ConfigDict(from_attributes=True)

# ==========================================
# AI ACTION SCHEMAS
# ==========================================

class AIActionRequest(BaseModel):
    action: str  # 'gap_check', 'rewrite', 'improve'
    text: str
    sop_title: Optional[str] = None
    section_name: Optional[str] = None
    section_type: Optional[str] = None
    # section_only | full_document — drives prompt scope (partial vs whole SOP)
    edit_scope: Optional[str] = None
    # When the client already holds a validated structured result (e.g. re-apply), skip LLM.
    client_structured_json: Optional[dict] = None
    # UUID of the SOP row for precise DB + ProfileDetection context (editor bubble / KL assistant).
    sop_entity_id: Optional[str] = None
    # Who invoked the action: ``editor_bubble`` | ``kl_assistant`` (for logs only).
    triggered_by: Optional[str] = None
    # Original user instruction/prompt so explicit style references can reach the LLM layer.
    instruction: Optional[str] = None
    # When true, accepting the suggestion should also save learned style back into the linked company profile.
    learn_to_profile: Optional[bool] = False

class AIActionResponse(BaseModel):
    action: str
    original_text: str
    suggested_text: str
    explanation: Optional[str] = None
    structured_data: Optional[dict] = None


class SemanticReindexRequest(BaseModel):
    entity_type: Optional[str] = None
    entity_id: Optional[UUID] = None
    version_id: Optional[UUID] = None
    full_reindex: bool = False


class LinkSuggestionResponse(BaseModel):
    id: UUID
    source_entity_type: str
    source_entity_id: UUID
    target_entity_type: str
    target_entity_id: UUID
    suggested_link_type: str
    score: float
    reason: Optional[str] = None
    status: str
    created_at: datetime
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class SemanticStatusResponse(BaseModel):
    entity_type: str
    entity_id: UUID
    latest_job_id: Optional[UUID] = None
    latest_job_status: Optional[str] = None
    latest_job_error: Optional[str] = None
    latest_job_finished_at: Optional[datetime] = None
    active_pipeline_job_id: Optional[UUID] = None
    chunking_status: Optional[str] = None
    embeddings_status: Optional[str] = None
    qdrant_status: Optional[str] = None
    nlp_status: Optional[str] = None
    semantic_linking_status: Optional[str] = None
    pending_suggestions: int = 0
    accepted_suggestions: int = 0
    rejected_suggestions: int = 0

# ==========================================
# QA-COMPLIANT PROFILE DETECTION SCHEMAS
# ==========================================

class EvidenceItem(BaseModel):
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    paragraph_index: Optional[int] = None
    traceability_id: Optional[str] = None

class ProfileSuggestionBase(BaseModel):
    suggestion_type: str
    suggested_rule: str
    evidence: Optional[List[EvidenceItem]] = None
    confidence: Optional[float] = None

class ProfileDetectionOutput(BaseModel):
    summary: str
    detected_domain: Optional[str] = None
    suggestions: List[ProfileSuggestionBase]
    overall_confidence_score: float

class ClientProfileCreate(BaseModel):
    name: str
    description: Optional[str] = None
    domain: Optional[str] = None

class ProfileVersionResponse(BaseModel):
    id: UUID
    profile_id: UUID
    version_number: int
    rules_json: Any
    is_locked: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ClientProfileResponse(BaseModel):
    id: UUID
    tenant_id: UUID
    name: str
    description: Optional[str] = None
    domain: Optional[str] = None
    current_version_id: Optional[UUID] = None
    active_profile_json: Optional[Any] = None
    active_profile_md: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ProfileSuggestionResponse(BaseModel):
    id: UUID
    profile_id: Optional[UUID] = None
    sop_id: Optional[UUID] = None
    suggestion_type: str
    suggested_rule: str
    evidence_json: Optional[Any] = None
    confidence: Optional[float] = None
    status: str
    rejection_reason: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class AcceptRejectSuggestionRequest(BaseModel):
    status: str # 'accepted' or 'rejected'
    rejection_reason: Optional[str] = None

