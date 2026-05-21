from sqlalchemy import Column, String, Text, Integer, TIMESTAMP, ForeignKey, func, Boolean, Float, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from .database import Base
from sqlalchemy.orm import relationship
import uuid

# Re-use existing tables logic but completely swap out the models

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    role = Column(String(20), default="user", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)

    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    title = Column(String(500), nullable=True)
    collection_name = Column(String(255), nullable=False, default="docs_sops")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)

    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True)
    retrieval_metadata = Column(JSON, nullable=True)
    metadata_snapshot = Column(JSON, nullable=True)
    audit_log_snapshot = Column(JSON, nullable=True)
    action_metadata = Column(JSON, nullable=True)
    category_filter = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("ChatSession", back_populates="messages")


class AIActionLog(Base):
    __tablename__ = "ai_action_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    action = Column(String(50), nullable=False, index=True)
    sop_title = Column(String(255), nullable=True)
    section_name = Column(String(255), nullable=True)
    section_type = Column(String(100), nullable=True)
    original_text = Column(Text, nullable=False)
    suggested_text = Column(Text, nullable=False)
    explanation = Column(Text, nullable=True)
    structured_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class EmbeddingJob(Base):
    __tablename__ = "embedding_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type = Column(String(50), nullable=False, index=True)
    entity_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    version_id = Column(UUID(as_uuid=True), nullable=True)
    job_type = Column(String(50), nullable=False, default="entity_reindex")
    # Overall job: pending | processing | completed | failed | cancelled
    status = Column(String(30), nullable=False, default="pending", index=True)
    error_message = Column(Text, nullable=True)
    # Content fingerprint at enqueue time (SOP version body); used to drop stale work.
    enqueued_content_hash = Column(String(64), nullable=True, index=True)
    # Per-stage lifecycle for SOP pipeline (pending|processing|completed|failed|skipped|cancelled)
    chunking_status = Column(String(30), nullable=False, default="pending")
    embeddings_status = Column(String(30), nullable=False, default="pending")
    qdrant_status = Column(String(30), nullable=False, default="pending")
    nlp_status = Column(String(30), nullable=False, default="pending")
    semantic_linking_status = Column(String(30), nullable=False, default="pending")
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    started_at = Column(TIMESTAMP, nullable=True)
    finished_at = Column(TIMESTAMP, nullable=True)


class AILinkSuggestion(Base):
    __tablename__ = "ai_link_suggestions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_entity_type = Column(String(50), nullable=False, index=True)
    source_entity_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    target_entity_type = Column(String(50), nullable=False, index=True)
    target_entity_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    suggested_link_type = Column(String(50), nullable=False, index=True)
    score = Column(Float, nullable=False)
    reason = Column(Text, nullable=True)
    status = Column(String(30), nullable=False, default="pending", index=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    approved_by = Column(String(255), nullable=True)
    approved_at = Column(TIMESTAMP, nullable=True)

class SOP(Base):
    __tablename__ = "sops"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    external_id = Column(String(255), nullable=True)
    sop_number = Column(String(100), nullable=False, unique=True)
    title = Column(String(255), nullable=False)
    department = Column(String(100), nullable=True)
    source_system = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    
    # We add this purely for UI Editor compatibility mapping. The domain schema doesn't strictly need it to find versions, but it helps the editor mock workflow.
    current_version_id = Column(UUID(as_uuid=True), nullable=True)
    # Latest accepted background pipeline job for this SOP (chunk/embed/qdrant/nlp).
    active_pipeline_job_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False)

    versions = relationship("SOPVersion", back_populates="sop", cascade="all, delete-orphan")
    profile_detections = relationship(
        "ProfileDetection", back_populates="sop", cascade="all, delete-orphan"
    )


class SOPVersion(Base):
    __tablename__ = "sop_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sop_id = Column(UUID(as_uuid=True), ForeignKey("sops.id", ondelete="CASCADE"), nullable=False)
    external_version_id = Column(String(100), nullable=True)
    version_number = Column(String(50), nullable=False)
    external_status = Column(String(50), nullable=True)
    effective_date = Column(TIMESTAMP, nullable=True)
    review_date = Column(TIMESTAMP, nullable=True)
    content_json = Column(JSONB, nullable=False)
    metadata_json = Column(JSONB, nullable=True)
    superseded_by_version_id = Column(UUID(as_uuid=True), ForeignKey("sop_versions.id"), nullable=True)

    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False)

    sop = relationship("SOP", back_populates="versions")
    profile_detections = relationship(
        "ProfileDetection",
        back_populates="sop_version_row",
        cascade="all, delete-orphan",
    )


class ProfileDetection(Base):
    """
    NLP / style snapshot for a current SOP version (editor AI actions).
    One logical active profile per (sop_id, sop_version_id); history kept via is_active.
    """

    __tablename__ = "profile_detections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sop_id = Column(UUID(as_uuid=True), ForeignKey("sops.id", ondelete="CASCADE"), nullable=False, index=True)
    sop_version_id = Column(
        UUID(as_uuid=True), ForeignKey("sop_versions.id", ondelete="CASCADE"), nullable=True, index=True
    )
    sop_version = Column(String(50), nullable=True)
    profile_type = Column(String(80), nullable=False, default="nlp_action_profile")
    source_hash = Column(String(64), nullable=False, index=True)
    language = Column(String(32), nullable=True)
    tone = Column(String(120), nullable=True)
    formality = Column(String(80), nullable=True)
    avg_sentence_words = Column(Float, nullable=True)
    readability_score = Column(Float, nullable=True)
    structure_json = Column(JSONB, nullable=True)
    parameters_json = Column(JSONB, nullable=True)
    detected_entities_json = Column(JSONB, nullable=True)
    nlp_analysis_json = Column(JSONB, nullable=True)
    prompt_block = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False)

    sop = relationship("SOP", back_populates="profile_detections")
    sop_version_row = relationship("SOPVersion", back_populates="profile_detections")


class Deviation(Base):
    __tablename__ = "deviations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    external_id = Column(String(255), nullable=True)
    deviation_number = Column(String(100), nullable=False)
    title = Column(String(255), nullable=False)
    category = Column(String(100), nullable=True)
    site = Column(String(100), nullable=True)
    product_line = Column(String(100), nullable=True)
    external_status = Column(String(50), nullable=True)
    description_text = Column(Text, nullable=True)
    root_cause_text = Column(Text, nullable=True)
    impact_level = Column(String(50), nullable=True)
    source_system = Column(String(100), nullable=True)
    event_date = Column(TIMESTAMP, nullable=True)

    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False)

class Capa(Base):
    __tablename__ = "capas"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    external_id = Column(String(255), nullable=True)
    capa_number = Column(String(100), nullable=False)
    title = Column(String(255), nullable=False)
    external_status = Column(String(50), nullable=True)
    action_type = Column(String(50), nullable=True)
    action_text = Column(Text, nullable=True)
    effectiveness_text = Column(Text, nullable=True)
    owner_name = Column(String(255), nullable=True)
    due_date = Column(TIMESTAMP, nullable=True)
    effectiveness_status = Column(String(50), nullable=True)
    source_system = Column(String(100), nullable=True)

    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False)

class AuditFinding(Base):
    __tablename__ = "audit_findings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    external_id = Column(String(255), nullable=True)
    audit_number = Column(String(100), nullable=True)
    finding_number = Column(String(100), nullable=True)
    authority = Column(String(100), nullable=True)
    site = Column(String(100), nullable=True)
    audit_date = Column(TIMESTAMP, nullable=True)
    question_text = Column(Text, nullable=True)
    finding_text = Column(Text, nullable=True)
    response_text = Column(Text, nullable=True)
    acceptance_status = Column(String(50), nullable=True)
    source_system = Column(String(100), nullable=True)

    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False)

class Decision(Base):
    __tablename__ = "decisions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    external_id = Column(String(255), nullable=True)
    decision_number = Column(String(100), nullable=True)
    decision_type = Column(String(100), nullable=True)
    title = Column(String(255), nullable=False)
    decision_statement = Column(Text, nullable=False)
    rationale_text = Column(Text, nullable=True)
    risk_assessment_text = Column(Text, nullable=True)
    alternatives_text = Column(Text, nullable=True)
    final_conclusion = Column(Text, nullable=True)
    decision_date = Column(TIMESTAMP, nullable=True)
    decided_by_role = Column(String(100), nullable=True)
    source_system = Column(String(100), nullable=True)

    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False)

# Link Tables
class SopDeviationLink(Base):
    __tablename__ = "sop_deviation_links"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    sop_id = Column(UUID(as_uuid=True), ForeignKey("sops.id"), nullable=False)
    deviation_id = Column(UUID(as_uuid=True), ForeignKey("deviations.id"), nullable=False)
    link_reason = Column(String(100), nullable=True)
    rationale_text = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

class DeviationCapaLink(Base):
    __tablename__ = "deviation_capa_links"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    deviation_id = Column(UUID(as_uuid=True), ForeignKey("deviations.id"), nullable=False)
    capa_id = Column(UUID(as_uuid=True), ForeignKey("capas.id"), nullable=False)
    link_reason = Column(String(100), nullable=True)
    rationale_text = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

class CapaAuditLink(Base):
    __tablename__ = "capa_audit_links"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    capa_id = Column(UUID(as_uuid=True), ForeignKey("capas.id"), nullable=False)
    audit_finding_id = Column(UUID(as_uuid=True), ForeignKey("audit_findings.id"), nullable=False)
    link_reason = Column(String(100), nullable=True)
    rationale_text = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

class AuditDecisionLink(Base):
    __tablename__ = "audit_decision_links"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    audit_finding_id = Column(UUID(as_uuid=True), ForeignKey("audit_findings.id"), nullable=False)
    decision_id = Column(UUID(as_uuid=True), ForeignKey("decisions.id"), nullable=False)
    link_reason = Column(String(100), nullable=True)
    rationale_text = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

class DecisionSopLink(Base):
    __tablename__ = "decision_sop_links"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    decision_id = Column(UUID(as_uuid=True), ForeignKey("decisions.id"), nullable=False)
    sop_id = Column(UUID(as_uuid=True), ForeignKey("sops.id"), nullable=False)
    sop_version_id = Column(UUID(as_uuid=True), ForeignKey("sop_versions.id"), nullable=True)
    link_reason = Column(String(100), nullable=True)
    rationale_text = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

# Supporting Tables
class SourceReference(Base):
    __tablename__ = "source_references"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(UUID(as_uuid=True), nullable=False)
    reference_type = Column(String(50), nullable=False)
    reference_label = Column(String(255), nullable=True)
    reference_value = Column(String(255), nullable=False)
    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(UUID(as_uuid=True), nullable=False)
    entity_version_id = Column(UUID(as_uuid=True), nullable=True)
    chunk_type = Column(String(50), nullable=True)
    block_id = Column(String(100), nullable=True)
    chunk_text = Column(Text, nullable=False)
    chunk_order = Column(Integer, nullable=False)
    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

class LifecycleConfig(Base):
    __tablename__ = "lifecycle_configs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    entity_type = Column(String(50), nullable=False) # e.g., 'sop', 'deviation'
    config_json = Column(JSONB, nullable=False)
    
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False)

# ==========================================
# QA-COMPLIANT PROFILE DETECTION MODELS
# ==========================================

class ClientProfile(Base):
    __tablename__ = "client_profiles"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    name = Column(String(255), nullable=False)
    company_name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    domain = Column(String(100), nullable=True)
    current_version_id = Column(UUID(as_uuid=True), nullable=True)
    total_sops_analyzed = Column(Integer, default=0, nullable=False)
    active_profile_md = Column(Text, nullable=True)
    active_profile_json = Column(JSONB, nullable=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False)

    versions = relationship("ProfileVersion", back_populates="profile", cascade="all, delete-orphan")

class ProfileVersion(Base):
    __tablename__ = "profile_versions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("client_profiles.id", ondelete="CASCADE"), nullable=False)
    version_number = Column(Integer, nullable=False, default=1)
    rules_json = Column(JSONB, nullable=False) # Stores the actual style/terminology rules
    profile_md = Column(Text, nullable=True)
    change_reason = Column(String(255), nullable=True)
    source_sop_id = Column(UUID(as_uuid=True), nullable=True)
    source_version_id = Column(UUID(as_uuid=True), nullable=True)
    detected_parameters_snapshot = Column(JSONB, nullable=True)
    is_locked = Column(Boolean, default=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

    profile = relationship("ClientProfile", back_populates="versions")

class ProfileSuggestion(Base):
    __tablename__ = "profile_suggestions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("client_profiles.id", ondelete="SET NULL"), nullable=True)
    sop_id = Column(UUID(as_uuid=True), ForeignKey("sops.id", ondelete="SET NULL"), nullable=True)
    suggestion_type = Column(String(100), nullable=False) # terminology, preferred_wording, modal_verb, etc.
    suggested_rule = Column(Text, nullable=False)
    evidence_json = Column(JSONB, nullable=True) # Traceable snippets
    confidence = Column(Float, nullable=True)
    status = Column(String(30), default="pending", index=True) # pending, accepted, rejected
    reviewed_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    rejection_reason = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

class SOPDetectedParameters(Base):
    __tablename__ = "sop_detected_parameters"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sop_id = Column(UUID(as_uuid=True), ForeignKey("sops.id", ondelete="CASCADE"), nullable=False)
    sop_version_id = Column(UUID(as_uuid=True), ForeignKey("sop_versions.id", ondelete="CASCADE"), nullable=False)
    client_profile_id = Column(UUID(as_uuid=True), ForeignKey("client_profiles.id", ondelete="SET NULL"), nullable=True)
    client_name = Column(String, nullable=True)
    source_filename = Column(String, nullable=True)
    
    analysis_json = Column(JSONB, nullable=True)
    document_information = Column(JSONB, nullable=True)
    writing_style = Column(JSONB, nullable=True)
    roles_raci = Column(JSONB, nullable=True)
    workflows = Column(JSONB, nullable=True)
    compliance_elements = Column(JSONB, nullable=True)
    risks_gaps = Column(JSONB, nullable=True)
    terminology = Column(JSONB, nullable=True)
    structure_patterns = Column(JSONB, nullable=True)
    style_suggestions = Column(JSONB, nullable=True)
    readiness_check = Column(JSONB, nullable=True)
    
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)


class ProfileHistoryEvent(Base):
    __tablename__ = "profile_history_events"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_profile_id = Column(UUID(as_uuid=True), ForeignKey("client_profiles.id", ondelete="CASCADE"), nullable=False)
    profile_version_id = Column(UUID(as_uuid=True), ForeignKey("profile_versions.id", ondelete="SET NULL"), nullable=True)
    source_sop_id = Column(UUID(as_uuid=True), nullable=True)
    
    event_type = Column(String(50), nullable=False) # e.g., "initial_creation", "sop_analyzed", "manual_override"
    event_summary = Column(Text, nullable=False)
    diff_json = Column(JSONB, nullable=True)
    after_snapshot = Column(JSONB, nullable=True)
    
    created_by = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)


class ProfileAuditLog(Base):
    __tablename__ = "profile_audit_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(100), nullable=False)
    sop_id = Column(UUID(as_uuid=True), nullable=True)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("client_profiles.id", ondelete="SET NULL"), nullable=True)
    profile_version_id = Column(UUID(as_uuid=True), ForeignKey("profile_versions.id", ondelete="SET NULL"), nullable=True)
    details_json = Column(JSONB, nullable=True)
    timestamp = Column(TIMESTAMP, server_default=func.now(), nullable=False)

