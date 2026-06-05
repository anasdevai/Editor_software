import io
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, UploadFile, File
from typing import List, Any
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy import func, or_, asc
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from qdrant_client.http import models as qmodels
import hashlib
from .database import get_db, SessionLocal
from .models import (
    SOP, SOPVersion, Deviation, Capa, AuditFinding, Decision, ClientProfile,
    SopDeviationLink, DeviationCapaLink, CapaAuditLink, AuditDecisionLink, DecisionSopLink,
    AILinkSuggestion,
    AISuggestion,
    EmbeddingJob,
    SOPDetectedParameters,
    ProfileDetection,
)
from .schemas import (
    # Editor compat request bodies
    CreateDocumentRequest,
    UpdateDocumentRequest,
    CreateVersionRequest,
    UpdateVersionStatusRequest,
    # Editor compat response shapes
    EditorDocResponse,
    EditorVersionResponse,
    # Native domain response shapes
    SOPResponse,
    SOPVersionResponse,
    DeviationResponse,
    CapaResponse,
    AuditFindingResponse,
    DecisionResponse,
    DeviationContextResponse,
    SopRelatedResponse,
    DeviationCreateUpdate,
    CapaCreateUpdate,
    AuditFindingCreateUpdate,
    DecisionCreateUpdate,
    DatasetImportRequest,
    LinkRequest,
    SemanticReindexRequest,
    LinkSuggestionResponse,
    SemanticStatusResponse,
)
from .services.semantic_pipeline import (
    SemanticPipelineService,
    ENTITY_TYPES,
    DEFAULT_COLLECTION,
    LINK_RULES,
    AUTO_ACCEPT_DELTA,
    _get_embedder,
    _get_qdrant,
    _split_long_text,
)
from uuid import UUID
import uuid
import os
import re
import threading
import logging
from datetime import datetime
from .services.sop_metadata_extractor import strip_invalid_control_chars
from .services.semantic_jobs import schedule_semantic_reindex
from .services.webhook_service import entities_for_link
from .utils.tiptap_text import extract_plain_text_from_tiptap
from .services.sop_profile_storage_service import analyze_and_store_sop_profile, cleanup_profile_for_deleted_sop
from .client_profile_routes import save_profile_version_from_accepted_suggestion

# ==========================================
# CONSTANTS
# ==========================================

# Fixed tenant for dev/seed environment
FIXED_TENANT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
logger = logging.getLogger(__name__)


# ==========================================
# HELPERS
# ==========================================

def check_mock_mode():
    """Guard: only allow mutation routes when MOCK_EDITOR_MODE=true."""
    # Default to enabled in local/dev so editor save/version actions work
    # unless explicitly disabled by environment configuration.
    if os.getenv("MOCK_EDITOR_MODE", "true").lower() != "true":
        raise HTTPException(
            status_code=403,
            detail="System is in Read-Only mode. Document mutation is disabled."
        )


def _is_tiptap_empty(doc_json: dict | None) -> bool:
    """
    Return True if a TipTap JSON document has no meaningful text content.

    A document is empty when:
    - It is None or not a dict
    - It has no 'content' list, or the list is empty
    - Every text leaf in the tree is whitespace-only
    - The only nodes are blank paragraphs (paragraph with no 'content' children)

    This mirrors the frontend isEditorContentEmpty() in src/utils/editorUtils.js.
    """
    if not doc_json or not isinstance(doc_json, dict):
        return True

    nodes = doc_json.get("content", [])
    if not nodes:
        return True

    def extract_text(node: dict) -> str:
        if node.get("type") == "text":
            return node.get("text", "").strip()
        return " ".join(
            filter(None, [extract_text(c) for c in node.get("content", [])])
        ).strip()

    # Check for any non-whitespace text in the entire tree
    all_text = extract_text(doc_json).strip()
    if all_text:
        return False

    # Also accept non-text meaningful nodes (image, table, codeBlock, etc.)
    meaningful_types = {"image", "horizontalRule", "codeBlock", "table"}
    for node in nodes:
        if node.get("type") in meaningful_types:
            return False

    return True


def _tenant_scoped_query(db: Session, model):
    """
    Default to inclusive query in local/dev so records inserted directly
    (with varying tenant_id values) and records imported through dataset flow
    are both visible to frontend APIs.

    Set STRICT_TENANT_SCOPING=true to enforce fixed-tenant-only behavior.
    """
    strict_tenant = os.getenv("STRICT_TENANT_SCOPING", "false").lower() == "true"
    scoped = db.query(model).filter(model.tenant_id == FIXED_TENANT_ID)
    if strict_tenant:
        return scoped
    return db.query(model)


def _resolve_sop_lookup(db: Session, sop_ref: str, *, include_inactive: bool = False):
    """
    Resolve SOP by UUID or SOP number while respecting tenant fallback logic.
    """
    base_query = _tenant_scoped_query(db, SOP)
    try:
        id_val = uuid.UUID(sop_ref)
        q = base_query.filter(SOP.id == id_val)
        if not include_inactive:
            q = q.filter(SOP.is_active == True)  # noqa: E712
        return q.first()
    except ValueError:
        q = base_query.filter(SOP.sop_number == sop_ref)
        if not include_inactive:
            q = q.filter(SOP.is_active == True)  # noqa: E712
        return q.first()


def _truncate_field(value: str | None, max_len: int) -> str:
    if not value:
        return ""
    s = strip_invalid_control_chars(value).strip()
    return s[:max_len] if len(s) > max_len else s


def _sanitize_json_like(value: Any) -> Any:
    """Recursively sanitize strings for JSONB-safe persistence."""
    if isinstance(value, str):
        return strip_invalid_control_chars(value)
    if isinstance(value, list):
        return [_sanitize_json_like(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_json_like(v) for k, v in value.items()}
    return value


def _compute_next_version_number(db: Session, sop_internal_id: uuid.UUID) -> str:
    """Next integer version string for an SOP (matches create_version logic)."""
    all_versions = db.query(SOPVersion).filter(SOPVersion.sop_id == sop_internal_id).all()
    max_v = 0
    for v in all_versions:
        try:
            val = int(str(v.version_number).split(".")[0])
            if val > max_v:
                max_v = val
        except Exception:
            continue
    return str(max_v + 1)


def _create_new_version_for_existing_sop(
    db: Session,
    existing: SOP,
    payload: CreateDocumentRequest,
    resolved_title: str,
    department: str,
    sop_number: str,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Add a new sop_version row, point current_version_id to it, refresh SOP header.
    Used when importing/creating a document with an SOP number that already exists.
    """
    payload_meta_json = _sanitize_json_like(payload.metadata_json)
    doc_json = _sanitize_json_like(payload.doc_json) if payload.doc_json is not None else {"type": "doc", "content": []}
    if _is_tiptap_empty(doc_json):
        raise HTTPException(
            status_code=422,
            detail="Cannot create a new version with empty content.",
        )

    dept_final = _truncate_field(department, 100) or "Quality"
    title_final = _truncate_field(resolved_title, 255) or "Untitled SOP"

    normalized_meta = _normalize_sop_metadata(
        sop_number=sop_number,
        title=title_final,
        department=dept_final,
        raw_meta=payload_meta_json,
    )
    resolved_external_status = _resolve_external_status_from_payload(payload_meta_json, fallback="draft")
    logger.info(
        "[sop-status] create new version for existing sop_number=%s resolved_external_status=%s payload_sopStatus=%s payload_status=%s",
        sop_number,
        resolved_external_status,
        (payload_meta_json or {}).get("sopStatus") if isinstance(payload_meta_json, dict) else None,
        (payload_meta_json or {}).get("status") if isinstance(payload_meta_json, dict) else None,
    )

    next_version = _compute_next_version_number(db, existing.id)
    new_version = SOPVersion(
        sop_id=existing.id,
        version_number=next_version,
        content_json=doc_json,
        metadata_json=normalized_meta,
        effective_date=_metadata_date(normalized_meta.get("sopMetadata", {}).get("effectiveDate")),
        review_date=_metadata_date(normalized_meta.get("sopMetadata", {}).get("reviewDate")),
        external_status=resolved_external_status,
    )
    db.add(new_version)
    try:
        db.flush()
        existing.current_version_id = new_version.id
        existing.title = title_final
        existing.department = dept_final
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"Failed to create new SOP version due to a constraint conflict: {str(exc.orig)[:300]}",
        ) from None
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail=f"Failed to create new SOP version: {str(exc)[:500]}",
        ) from None
    db.refresh(existing)
    db.refresh(new_version)

    _upsert_import_context_entities(db, existing, new_version, background_tasks)
    _schedule_semantic_job(background_tasks, "sop", existing.id, new_version.id)

    try:
        plain_text = _extract_plain_text_from_tiptap(new_version.content_json)
        analyze_and_store_sop_profile(
            db=db,
            sop_id=existing.id,
            sop_version_id=new_version.id,
            text=plain_text,
            client_name="Client",
            source_filename=existing.title
        )
    except Exception as e:
        logger.error(f"SOP profile analysis failed for {existing.id}: {e}")

    return _build_editor_doc_response(existing, new_version)


def _resolve_current_version(db: Session, sop: SOP) -> SOPVersion | None:
    """
    Return SOP current version with a safe fallback for imported records
    where current_version_id may be missing but version rows exist.
    """
    if sop.current_version_id:
        current = db.query(SOPVersion).filter(SOPVersion.id == sop.current_version_id).first()
        if current:
            return current

    latest = (
        db.query(SOPVersion)
        .filter(SOPVersion.sop_id == sop.id)
        .order_by(SOPVersion.created_at.desc())
        .first()
    )
    if latest:
        sop.current_version_id = latest.id
        db.commit()
        db.refresh(sop)
        return latest

    # Last-resort bootstrap: create an initial editor-compatible draft version
    # so legacy/imported SOP rows without versions can still be opened/edited.
    initial_version = SOPVersion(
        id=uuid.uuid4(),
        sop_id=sop.id,
        version_number="1",
        content_json={"type": "doc", "content": []},
        metadata_json=_normalize_sop_metadata(
            sop_number=sop.sop_number,
            title=sop.title,
            department=sop.department,
            raw_meta={},
        ),
        external_status="draft",
    )
    db.add(initial_version)
    sop.current_version_id = initial_version.id
    db.commit()
    db.refresh(sop)
    db.refresh(initial_version)
    return initial_version


def _schedule_semantic_job(
    background_tasks: BackgroundTasks,
    entity_type: str,
    entity_id: uuid.UUID,
    version_id: uuid.UUID | None = None,
    job_type: str = "entity_reindex",
):
    try:
        if entity_type not in ENTITY_TYPES:
            return
        print(f"[semantic-job] Scheduling {job_type} for {entity_type} {entity_id}", flush=True)
        schedule_semantic_reindex(entity_type, entity_id, version_id, job_type=job_type)
    except Exception as exc:
        print(f"[semantic-job] ERROR: Failed to schedule job for {entity_type} {entity_id}: {exc}", flush=True)


def _extract_plain_text_from_tiptap(doc_json: dict | None) -> str:
    return extract_plain_text_from_tiptap(doc_json)


def _parse_uuid_or_400(value: str, field_name: str = "id") -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name} format")


def _extract_entity_tokens(text: str, prefix: str) -> list[str]:
    # Deterministic ID extraction, e.g. DEV-001 / CAPA-42 / AUD-7 / DEC-5
    pattern = rf"\b{prefix}-[A-Z0-9][A-Z0-9\-]*\b"
    found = re.findall(pattern, text.upper())
    dedup: list[str] = []
    seen = set()
    for token in found:
        if token not in seen:
            seen.add(token)
            dedup.append(token)
    return dedup


def _stable_embedded_ref(prefix: str, sop_id: uuid.UUID, section_text: str) -> str:
    digest = hashlib.sha1(
        f"{sop_id}:{prefix}:{section_text.strip()}".encode("utf-8", errors="ignore")
    ).hexdigest()[:10].upper()
    return f"{prefix}-EMB-{digest}"


def _extract_embedded_sections(text: str, heading_pattern: str) -> list[str]:
    if not text:
        return []
    regex = re.compile(
        rf"(?ims)^\s*(?:\d+[\.\)]\s*)?(?:{heading_pattern})\s*[:\-]\s*(.+?)(?=^\s*(?:\d+[\.\)]\s*)?(?:deviation|capa|audit(?:\s+finding)?|decision|sop)\s*[:\-]|\Z)"
    )
    sections = []
    for match in regex.finditer(text):
        body = (match.group(1) or "").strip()
        if len(body) >= 80:
            sections.append(body[:3000])
    return sections


def _semantic_candidates_from_text(text: str, target_type: str, limit: int = 5) -> list[dict]:
    if not text.strip():
        return []
    try:
        embedder = _get_embedder()
        client = _get_qdrant()
    except Exception as exc:
        print(f"[import-linking] Semantic runtime unavailable: {exc}", flush=True)
        return []

    scores: dict[str, dict] = {}
    chunks = _split_long_text(text, size=900, overlap=150)[:8]
    if not chunks:
        chunks = [text[:900]]
    for idx, chunk in enumerate(chunks):
        try:
            vec = embedder.encode([chunk], normalize_embeddings=True)[0].tolist()
            filt = qmodels.Filter(
                must=[qmodels.FieldCondition(key="entity_type", match=qmodels.MatchValue(value=target_type))]
            )
            if hasattr(client, "search"):
                hits = client.search(
                    collection_name=DEFAULT_COLLECTION,
                    query_vector=vec,
                    query_filter=filt,
                    limit=limit * 2,
                )
            else:
                result = client.query_points(
                    collection_name=DEFAULT_COLLECTION,
                    query=vec,
                    query_filter=filt,
                    limit=limit * 2,
                    with_payload=True,
                    with_vectors=False,
                )
                hits = result.points
            for hit in hits:
                target_id = str(hit.payload.get("entity_id") or "")
                if not target_id:
                    continue
                score = float(hit.score)
                prev = scores.get(target_id)
                if prev is None or score > prev["score"]:
                    scores[target_id] = {"score": score, "chunk_index": idx, "source_chunk": chunk[:300]}
        except Exception as exc:
            print(f"[import-linking] Semantic query failed for {target_type}: {exc}", flush=True)
            continue
    ranked = sorted(scores.items(), key=lambda kv: kv[1]["score"], reverse=True)[:limit]
    return [{"target_id": tid, **payload} for tid, payload in ranked]


def _upsert_semantic_suggestion(db: Session, source_type: str, source_id: uuid.UUID, target_type: str, target_id: uuid.UUID, link_type: str, score: float, reason: str) -> AILinkSuggestion:
    target_exists = False
    if target_type == "deviation":
        target_exists = db.query(Deviation.id).filter(Deviation.id == target_id).first() is not None
    elif target_type == "capa":
        target_exists = db.query(Capa.id).filter(Capa.id == target_id).first() is not None
    elif target_type == "audit_finding":
        target_exists = db.query(AuditFinding.id).filter(AuditFinding.id == target_id).first() is not None
    elif target_type == "decision":
        target_exists = db.query(Decision.id).filter(Decision.id == target_id).first() is not None
    elif target_type == "sop":
        target_exists = db.query(SOP.id).filter(SOP.id == target_id).first() is not None
    if not target_exists:
        return None

    suggestion = db.query(AILinkSuggestion).filter(
        AILinkSuggestion.source_entity_type == source_type,
        AILinkSuggestion.source_entity_id == source_id,
        AILinkSuggestion.target_entity_type == target_type,
        AILinkSuggestion.target_entity_id == target_id,
        AILinkSuggestion.suggested_link_type == link_type,
    ).order_by(AILinkSuggestion.created_at.desc()).first()
    if suggestion is None:
        suggestion = AILinkSuggestion(
            source_entity_type=source_type,
            source_entity_id=source_id,
            target_entity_type=target_type,
            target_entity_id=target_id,
            suggested_link_type=link_type,
            score=score,
            reason=reason,
            status="pending",
        )
        db.add(suggestion)
        db.flush()
    elif suggestion.status == "pending":
        suggestion.score = max(float(suggestion.score or 0.0), score)
        suggestion.reason = reason
    return suggestion


def _upsert_import_context_entities(
    db: Session,
    sop: SOP,
    version: SOPVersion,
    background_tasks: BackgroundTasks | None = None,
) -> dict:
    t0 = datetime.utcnow()
    text = _extract_plain_text_from_tiptap(version.content_json)
    if not text:
        return {"deviations": 0, "capas": 0, "audits": 0, "decisions": 0, "links": 0, "semantic_suggestions": 0, "semantic_auto_accepted": 0}
    context_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    version_meta = version.metadata_json if isinstance(version.metadata_json, dict) else {}
    if version_meta.get("_import_context_hash") == context_hash:
        return {"deviations": 0, "capas": 0, "audits": 0, "decisions": 0, "links": 0, "semantic_suggestions": 0, "semantic_auto_accepted": 0}

    tenant_id = sop.tenant_id or FIXED_TENANT_ID
    created = {"deviations": 0, "capas": 0, "audits": 0, "decisions": 0, "links": 0, "semantic_suggestions": 0, "semantic_auto_accepted": 0}

    dev_tokens = _extract_entity_tokens(text, "DEV")
    capa_tokens = _extract_entity_tokens(text, "CAPA")
    aud_tokens = _extract_entity_tokens(text, "AUD")
    dec_tokens = _extract_entity_tokens(text, "DEC")
    sop_tokens = _extract_entity_tokens(text, "SOP")

    embedded_dev_sections = _extract_embedded_sections(text, r"deviation")
    embedded_capa_sections = _extract_embedded_sections(text, r"capa")
    embedded_audit_sections = _extract_embedded_sections(text, r"audit(?:\s+finding)?")
    embedded_decision_sections = _extract_embedded_sections(text, r"decision")

    deviations: list[Deviation] = []
    for token in dev_tokens:
        dev = db.query(Deviation).filter(Deviation.deviation_number == token).first()
        if not dev:
            dev = Deviation(id=uuid.uuid4(), tenant_id=tenant_id, deviation_number=token, title=f"Imported {token}", description_text=f"Deterministically extracted from SOP {sop.sop_number}", source_system="pdf_text_import")
            db.add(dev)
            created["deviations"] += 1
        deviations.append(dev)
    if not dev_tokens:
        for section in embedded_dev_sections[:2]:
            stable_number = _stable_embedded_ref("DEV", sop.id, section)
            dev = db.query(Deviation).filter(Deviation.deviation_number == stable_number).first()
            if not dev:
                dev = Deviation(id=uuid.uuid4(), tenant_id=tenant_id, deviation_number=stable_number, title=f"Embedded deviation from {sop.sop_number}", description_text=section, source_system="pdf_text_import")
                db.add(dev)
                created["deviations"] += 1
            elif (dev.description_text or "").strip() != section.strip():
                dev.description_text = section
            deviations.append(dev)

    capas: list[Capa] = []
    for token in capa_tokens:
        capa = db.query(Capa).filter(Capa.capa_number == token).first()
        if not capa:
            capa = Capa(id=uuid.uuid4(), tenant_id=tenant_id, capa_number=token, title=f"Imported {token}", action_text=f"Deterministically extracted from SOP {sop.sop_number}", source_system="pdf_text_import")
            db.add(capa)
            created["capas"] += 1
        capas.append(capa)
    if not capa_tokens:
        for section in embedded_capa_sections[:2]:
            stable_number = _stable_embedded_ref("CAPA", sop.id, section)
            capa = db.query(Capa).filter(Capa.capa_number == stable_number).first()
            if not capa:
                capa = Capa(id=uuid.uuid4(), tenant_id=tenant_id, capa_number=stable_number, title=f"Embedded CAPA from {sop.sop_number}", action_text=section, source_system="pdf_text_import")
                db.add(capa)
                created["capas"] += 1
            elif (capa.action_text or "").strip() != section.strip():
                capa.action_text = section
            capas.append(capa)

    audits: list[AuditFinding] = []
    for token in aud_tokens:
        audit = db.query(AuditFinding).filter(AuditFinding.finding_number == token).first()
        if not audit:
            audit = AuditFinding(id=uuid.uuid4(), tenant_id=tenant_id, finding_number=token, finding_text=f"Deterministically extracted from SOP {sop.sop_number}", source_system="pdf_text_import")
            db.add(audit)
            created["audits"] += 1
        audits.append(audit)
    if not aud_tokens:
        for section in embedded_audit_sections[:2]:
            stable_number = _stable_embedded_ref("AUD", sop.id, section)
            audit = db.query(AuditFinding).filter(AuditFinding.finding_number == stable_number).first()
            if not audit:
                audit = AuditFinding(id=uuid.uuid4(), tenant_id=tenant_id, finding_number=stable_number, finding_text=section, source_system="pdf_text_import")
                db.add(audit)
                created["audits"] += 1
            elif (audit.finding_text or "").strip() != section.strip():
                audit.finding_text = section
            audits.append(audit)

    decisions: list[Decision] = []
    for token in dec_tokens:
        dec = db.query(Decision).filter(Decision.decision_number == token).first()
        if not dec:
            dec = Decision(id=uuid.uuid4(), tenant_id=tenant_id, decision_number=token, title=f"Imported {token}", decision_statement=f"Deterministically extracted from SOP {sop.sop_number}", source_system="pdf_text_import")
            db.add(dec)
            created["decisions"] += 1
        decisions.append(dec)
    if not dec_tokens:
        for section in embedded_decision_sections[:2]:
            stable_number = _stable_embedded_ref("DEC", sop.id, section)
            dec = db.query(Decision).filter(Decision.decision_number == stable_number).first()
            if not dec:
                dec = Decision(id=uuid.uuid4(), tenant_id=tenant_id, decision_number=stable_number, title=f"Embedded decision from {sop.sop_number}", decision_statement=section[:3000], source_system="pdf_text_import")
                db.add(dec)
                created["decisions"] += 1
            elif (dec.decision_statement or "").strip() != section[:3000].strip():
                dec.decision_statement = section[:3000]
            decisions.append(dec)

    db.flush()
    t_structured = datetime.utcnow()

    for dev in deviations:
        if not db.query(SopDeviationLink).filter(SopDeviationLink.sop_id == sop.id, SopDeviationLink.deviation_id == dev.id).first():
            db.add(SopDeviationLink(tenant_id=tenant_id, sop_id=sop.id, deviation_id=dev.id, link_reason="import_deterministic"))
            created["links"] += 1
    for dev in deviations:
        for capa in capas:
            if not db.query(DeviationCapaLink).filter(DeviationCapaLink.deviation_id == dev.id, DeviationCapaLink.capa_id == capa.id).first():
                db.add(DeviationCapaLink(tenant_id=tenant_id, deviation_id=dev.id, capa_id=capa.id, link_reason="import_deterministic"))
                created["links"] += 1
    for capa in capas:
        for audit in audits:
            if not db.query(CapaAuditLink).filter(CapaAuditLink.capa_id == capa.id, CapaAuditLink.audit_finding_id == audit.id).first():
                db.add(CapaAuditLink(tenant_id=tenant_id, capa_id=capa.id, audit_finding_id=audit.id, link_reason="import_deterministic"))
                created["links"] += 1
    for audit in audits:
        for dec in decisions:
            if not db.query(AuditDecisionLink).filter(AuditDecisionLink.audit_finding_id == audit.id, AuditDecisionLink.decision_id == dec.id).first():
                db.add(AuditDecisionLink(tenant_id=tenant_id, audit_finding_id=audit.id, decision_id=dec.id, link_reason="import_deterministic"))
                created["links"] += 1
    for dec in decisions:
        if not db.query(DecisionSopLink).filter(DecisionSopLink.decision_id == dec.id, DecisionSopLink.sop_id == sop.id).first():
            db.add(DecisionSopLink(tenant_id=tenant_id, decision_id=dec.id, sop_id=sop.id, sop_version_id=version.id, link_reason="import_deterministic"))
            created["links"] += 1

    db.commit()
    t_links = datetime.utcnow()

    for dev in deviations:
        _schedule_semantic_job(background_tasks, "deviation", dev.id)
    for capa in capas:
        _schedule_semantic_job(background_tasks, "capa", capa.id)
    for audit in audits:
        _schedule_semantic_job(background_tasks, "audit_finding", audit.id)
    for dec in decisions:
        _schedule_semantic_job(background_tasks, "decision", dec.id)
    _schedule_semantic_job(background_tasks, "sop", sop.id, version.id, job_type="import_reindex")
    t_jobs = datetime.utcnow()

    # Deterministic sibling SOP linking when explicit SOP references exist and decisions are present.
    linked_decisions = [d.id for d in decisions]
    for token in sop_tokens:
        if token == (sop.sop_number or "").upper():
            continue
        sibling = db.query(SOP).filter(SOP.sop_number == token).first()
        if not sibling:
            continue
        for decision_id in linked_decisions:
            if not db.query(DecisionSopLink).filter(DecisionSopLink.decision_id == decision_id, DecisionSopLink.sop_id == sibling.id).first():
                db.add(DecisionSopLink(tenant_id=tenant_id, decision_id=decision_id, sop_id=sibling.id, sop_version_id=sibling.current_version_id, link_reason="import_deterministic_ref"))
                created["links"] += 1

    db.commit()
    version_meta["_import_context_hash"] = context_hash
    version.metadata_json = version_meta
    db.commit()
    t_end = datetime.utcnow()
    print(
        "[import-linking] "
        f"sop={sop.id} "
        f"extract+entity={int((t_structured - t0).total_seconds()*1000)}ms "
        f"deterministic_links={int((t_links - t_structured).total_seconds()*1000)}ms "
        f"job_enqueue={int((t_jobs - t_links).total_seconds()*1000)}ms "
        f"finalize={int((t_end - t_jobs).total_seconds()*1000)}ms "
        f"created={created}",
        flush=True,
    )
    return created


def _deduplicate_audit_trail(entries: list) -> list:
    """
    Return a deduplicated copy of an audit trail list.

    Deduplication key: (action, version, note).  On collision the *last*
    occurrence is kept so that the entry carrying the most-recent ``createdAt``
    timestamp wins.  Entries that are not dicts are silently dropped.

    The returned list preserves the original insertion order of the *first*
    appearance of each unique key (i.e. we do a stable, last-wins dedup by
    rebuilding the index in a single pass and then reordering).
    """
    if not entries:
        return []

    # Two-pass: collect canonical keys first so we can emit entries in their
    # original first-seen order while still keeping the *last* value for each key.
    seen: dict = {}  # key -> last entry with that key
    order: list = []  # tracks first-seen insertion order of each key
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = (
            str(entry.get("action", "")),
            str(entry.get("version", "")),
            str(entry.get("note", "")),
        )
        if key not in seen:
            order.append(key)
        seen[key] = entry  # always overwrite so last occurrence wins

    return [seen[k] for k in order]


def _normalize_sop_metadata(sop_number: str, title: str, department: str = None, raw_meta: dict = None) -> dict:
    """
    Ensures metadata is in the full 'thick shell' shape the frontend expects.
    MANDATORY fields for Editor compatibility:
    - sopStatus, variables, approvedBy, auditTrail, versionNote, sopMetadata, etc.
    """
    if not isinstance(raw_meta, dict):
        raw_meta = {}
    raw_meta = _sanitize_json_like(raw_meta)
    sop_number = strip_invalid_control_chars(sop_number or "")
    title = strip_invalid_control_chars(title or "")
    department = strip_invalid_control_chars(department or "")
    
    # 1. Base Structure
    normalized = {
        "sopStatus": raw_meta.get("sopStatus", "draft"),
        "variables": raw_meta.get("variables", {}),
        "approvedBy": raw_meta.get("approvedBy", ""),
        "auditTrail": _deduplicate_audit_trail(
            raw_meta.get("auditTrail") if isinstance(raw_meta.get("auditTrail"), list) else []
        ),
        "versionNote": raw_meta.get("versionNote", ""),
        "obsoleteReason": raw_meta.get("obsoleteReason", ""),
        "approvalSignature": raw_meta.get("approvalSignature", ""),
        "replacementDocumentId": raw_meta.get("replacementDocumentId", ""),
        "sopMetadata": {
            "title": title or "",
            "author": raw_meta.get("sopMetadata", {}).get("author", "System"),
            "reviewer": raw_meta.get("sopMetadata", {}).get("reviewer", ""),
            "riskLevel": raw_meta.get("sopMetadata", {}).get("riskLevel", "Low"),
            "department": department or "Quality",
            "documentId": sop_number or "",
            "docType": raw_meta.get("sopMetadata", {}).get("docType", "SOP"),
            "category": raw_meta.get("sopMetadata", {}).get("category", ""),
            "sopVersion": raw_meta.get("sopMetadata", {}).get("sopVersion", ""),
            "references": raw_meta.get("sopMetadata", {}).get("references", []),
            "reviewDate": raw_meta.get("sopMetadata", {}).get("reviewDate", ""),
            "effectiveDate": raw_meta.get("sopMetadata", {}).get("effectiveDate", ""),
            "regulatoryReferences": raw_meta.get("sopMetadata", {}).get("regulatoryReferences", [])
        }
    }
    
    # Merge nested sopMetadata: client may send documentId/title; incoming values win when non-empty
    input_sop_meta = raw_meta.get("sopMetadata", {})
    if isinstance(input_sop_meta, dict):
        for k, v in input_sop_meta.items():
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "" and k not in ("author", "reviewer"):
                continue
            if k in ("title", "documentId"):
                if v:
                    normalized["sopMetadata"][k] = v
            else:
                normalized["sopMetadata"][k] = v

    # Preserve internal processing markers so idempotency guards survive metadata updates.
    for key, value in raw_meta.items():
        if isinstance(key, str) and key.startswith("_"):
            normalized[key] = value

    return normalized


def _metadata_date(value: str):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _metadata_debug_sources(structured: dict) -> dict:
    return {
        "sop_id": "SOP ID label/generic SOP token" if structured.get("sop_id") else "not found",
        "title": "Title/Titel label or combined SOP ID + Titel line" if structured.get("title") else "not found",
        "version": "explicit version/revision label or change-history table" if structured.get("version") else "not found",
        "date": "effective date label or change-history date" if structured.get("date") else "not found",
        "department": "department label or SOP/context inference" if structured.get("department") else "not found",
        "category": "keyword/context inference" if structured.get("category") else "not found",
        "status": "Status label/value normalization from OCR text" if structured.get("status") else "not found",
    }


def _blocks_to_elements(blocks: list) -> list:
    elements = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type", "")).lower()
        if btype in {"section_heading", "heading", "title"}:
            text = str(block.get("text") or "").strip()
            if text:
                elements.append({"type": "text", "style": "heading", "content": text})
        elif btype in {"paragraph", "line", "note"}:
            text = str(block.get("text") or "").strip()
            if text:
                elements.append({"type": "text", "style": "paragraph", "content": text})
        elif btype in {"two_column_row", "key_value"}:
            left = str(block.get("left") or "").strip()
            right = str(block.get("right") or "").strip()
            text = f"{left}: {right}".strip(": ").strip()
            if text:
                elements.append({"type": "text", "style": "paragraph", "content": text})
        elif btype in {"bullet_list", "numbered_list", "list", "ordered_list"}:
            for item in block.get("items") or []:
                text = str(item).strip()
                if text:
                    elements.append({"type": "text", "style": "paragraph", "content": text})
        elif btype == "table":
            rows = block.get("rows") or []
            if rows:
                elements.append({
                    "type": "table",
                    "content": rows,
                    "header_rows": int(block.get("header_rows") or 0),
                })
    return elements


def _build_extract_response(text: str, blocks: list, structured: dict, elements: list | None = None, scanned_pdf: bool = False) -> dict:
    from .services.document_structure import blocks_to_structured_document
    from .services.sop_metadata_extractor import to_frontend_sop_metadata

    sop_ui = to_frontend_sop_metadata(structured)
    public_meta = {k: v for k, v in structured.items() if not str(k).startswith("_")}
    structured_document = blocks_to_structured_document(blocks, public_meta)
    resolved_elements = elements if elements is not None else _blocks_to_elements(blocks)
    response = {
        "text": (text or "").strip(),
        "blocks": blocks,
        "elements": resolved_elements,
        "scanned_pdf": bool(scanned_pdf),
        "structured_document": structured_document,
        "sop_metadata": public_meta,
        "sop_metadata_ui": sop_ui,
        "metadata_sources": _metadata_debug_sources(public_meta),
    }
    logger.info("[ocr-import] raw text first 1000 chars: %s", (text or "")[:1000])
    logger.info("[ocr-import] extracted metadata result: %s", public_meta)
    logger.info(
        "[ocr-import] extracted status detail: structured.status=%s sop_ui.sopStatus=%s sop_ui.status=%s",
        public_meta.get("status"),
        sop_ui.get("sopStatus"),
        sop_ui.get("status"),
    )
    logger.info("[ocr-import] metadata sources: %s", response["metadata_sources"])
    logger.info(
        "[ocr-import] final response sent to frontend: %s",
        {**response, "text": response["text"][:300], "blocks": f"{len(blocks)} blocks", "elements": f"{len(resolved_elements)} elements"},
    )
    return response


def _normalize_external_status(value: str) -> str:
    v = str(value or "").strip()
    if not v:
        return ""
    compact = re.sub(r"[\s-]+", "_", v.lower())
    alias_map = {
        "in_review": "under_review",
        "underreview": "under_review",
        "freigegeben": "effective",
        "entwurf": "draft",
        "prufung": "under_review",
        "pruefung": "under_review",
    }
    normalized = alias_map.get(compact, compact)
    return normalized if normalized in {"draft", "under_review", "effective", "obsolete", "approved", "accepted", "rejected", "changes_requested"} else ""


def _resolve_external_status_from_payload(raw_meta: dict | None, fallback: str = "draft") -> str:
    if not isinstance(raw_meta, dict):
        return fallback
    candidate = (
        raw_meta.get("sopStatus")
        or raw_meta.get("status")
        or (raw_meta.get("sopMetadata", {}) or {}).get("sopStatus")
        or (raw_meta.get("sopMetadata", {}) or {}).get("status")
    )
    return _normalize_external_status(candidate) or fallback


def _build_editor_doc_response(sop: SOP, version: SOPVersion) -> dict:
    """
    Compatibility adapter: maps SOP + SOPVersion onto old editor response shape.
    Normalizes metadata on-the-fly to ensure frontend consistency.
    """
    normalized_meta = _normalize_sop_metadata(
        sop_number=sop.sop_number,
        title=sop.title,
        department=sop.department,
        raw_meta=version.metadata_json
    )

    return {
        "id": str(sop.id),
        "title": sop.title,
        "doc_type": "sop",
        "doc_json": version.content_json,
        "metadata_json": normalized_meta,
        "current_version_id": str(sop.current_version_id) if sop.current_version_id else None,
        "version_number": version.version_number,
        "status": version.external_status or "draft",
        "created_at": sop.created_at,
        "updated_at": sop.updated_at,
    }


def _build_editor_version_response(version: SOPVersion) -> dict:
    """Compatibility adapter for single version response."""
    sop_title = version.sop.title if version.sop else "Untitled"
    sop_num = version.sop.sop_number if version.sop else ""
    sop_dept = version.sop.department if version.sop else "Quality"

    normalized_meta = _normalize_sop_metadata(
        sop_number=sop_num,
        title=sop_title,
        department=sop_dept,
        raw_meta=version.metadata_json
    )

    return {
        "id": str(version.id),
        "doc_id": str(version.sop_id),
        "version_number": version.version_number,
        "status": version.external_status or "draft",
        "doc_json": version.content_json,
        "metadata_json": normalized_meta,
        "effective_date": version.effective_date,
        "review_date": version.review_date,
        "created_at": version.created_at,
        "updated_at": version.updated_at,
    }


def _build_sop_dict(sop: SOP, include_current_version: bool = False, db: Session = None) -> dict:
    """
    Build native SOPResponse dict, optionally embedding the current_version object.
    Standardizes metadata to the 'thick shell' format.
    """
    result = {
        "id": sop.id,
        "tenant_id": sop.tenant_id,
        "external_id": sop.external_id,
        "sop_number": sop.sop_number,
        "title": sop.title,
        "department": sop.department,
        "source_system": sop.source_system,
        "is_active": sop.is_active,
        "current_version_id": sop.current_version_id,
        "current_version": None,
        "created_at": sop.created_at,
        "updated_at": sop.updated_at,
    }
    if include_current_version and db and sop.current_version_id:
        cv = db.query(SOPVersion).filter(SOPVersion.id == sop.current_version_id).first()
        if cv:
            normalized_meta = _normalize_sop_metadata(
                sop_number=sop.sop_number,
                title=sop.title,
                department=sop.department,
                raw_meta=cv.metadata_json
            )
            result["current_version"] = {
                "id": cv.id,
                "sop_id": cv.sop_id,
                "external_version_id": cv.external_version_id,
                "version_number": cv.version_number,
                "external_status": cv.external_status,
                "content_json": cv.content_json,
                "metadata_json": normalized_meta,
                "effective_date": cv.effective_date,
                "review_date": cv.review_date,
                "created_at": cv.created_at,
                "updated_at": cv.updated_at,
            }
    return result


# ==========================================
# ROUTER
# ==========================================

router = APIRouter()


@router.get("/api/health")
def health():
    from .services.pdf_extractor import check_ocr_setup

    ocr = check_ocr_setup()
    return {
        "status": "ok",
        "ocr": ocr,
        "ocr_ready": bool(ocr.get("tesseract_binary") and ocr.get("poppler_binaries")),
    }


@router.post("/api/extract-text")
async def extract_text_from_upload(file: UploadFile = File(...)):
    """
    Extract text from an uploaded SOP file.
    Returns legacy blocks plus reading-order elements for stronger PDF/OCR rendering.
    """
    from .services.document_structure import enrich_metadata_text, structure_blocks_from_text
    from .services.sequential_import import extract_sequential_upload
    from .services.sop_metadata_extractor import extract_sop_metadata_from_text

    name = (file.filename or "").lower()
    try:
        if name.endswith((".pdf", ".docx", ".txt", ".md", ".csv", ".json")):
            raw = await file.read()
            elements, blocks, text, scanned_pdf = extract_sequential_upload(raw, name)
            text = strip_invalid_control_chars(text)
            meta_text = enrich_metadata_text(text, blocks)
            structured = extract_sop_metadata_from_text(meta_text, blocks)
            return _build_extract_response(text, blocks, structured, elements=elements, scanned_pdf=scanned_pdf)
        else:
            # Best-effort UTF-8 for unknown extensions
            raw = await file.read()
            try:
                text = raw.decode("utf-8")
                text = strip_invalid_control_chars(text)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported or binary file; use .pdf, .docx, or .txt",
                ) from None
            blocks = structure_blocks_from_text(text)
            meta_text = enrich_metadata_text(text, blocks)
            structured = extract_sop_metadata_from_text(meta_text, blocks)
            return _build_extract_response(text, blocks, structured)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[extract-text] extraction failed for filename=%s", name)
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed ({type(e).__name__}): {e!s}",
        ) from e


# ==========================================
# OLD EDITOR COMPATIBILITY ROUTES
# All field mappings live here — NOT in the DB
# doc_json = content_json, status = external_status, doc_id = sop_id
# ==========================================

_DUP_SOP_MSG = (
    "SOP with this SOP ID already exists. Please create a new version or choose another SOP ID."
)


@router.post("/api/editor/docs", response_model=EditorDocResponse)
def create_document(
    payload: CreateDocumentRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _=Depends(check_mock_mode),
):
    """
    Create a new SOP + its first version.
    If metadata documentId/sop_number matches an existing SOP, add a new version instead of INSERT (no duplicate key).
    """
    payload_doc_json = _sanitize_json_like(payload.doc_json)
    payload_meta_json = _sanitize_json_like(payload.metadata_json)

    new_sop_id = uuid.uuid4()
    new_ver_id = uuid.uuid4()

    sop_number = payload_meta_json.get("sopMetadata", {}).get("documentId") if isinstance(payload_meta_json, dict) else None
    sop_number = (sop_number or "").strip()
    if not sop_number:
        sop_number = f"SOP-{uuid.uuid4().hex[:8].upper()}"

    dept_sm = None
    if isinstance(payload_meta_json, dict) and isinstance(payload_meta_json.get("sopMetadata"), dict):
        dept_sm = (payload_meta_json["sopMetadata"].get("department") or "").strip() or None
    department = dept_sm or "Quality"

    resolved_title = (payload.title or "").strip() if payload.title else ""
    if not resolved_title and isinstance(payload_meta_json, dict) and isinstance(payload_meta_json.get("sopMetadata"), dict):
        resolved_title = (payload_meta_json["sopMetadata"].get("title") or "").strip()
    if not resolved_title:
        resolved_title = "Untitled SOP"

    existing = (
        db.query(SOP)
        .filter(SOP.sop_number == sop_number, SOP.tenant_id == FIXED_TENANT_ID)
        .first()
    )
    if existing:
        has_params = db.query(SOPDetectedParameters).filter(SOPDetectedParameters.sop_id == existing.id).first()
        if has_params:
            logger.info("SOP %s already uploaded and analyzed, returning existing document without duplication.", sop_number)
            current_version = _resolve_current_version(db, existing)
            return _build_editor_doc_response(existing, current_version)
        else:
            return _create_new_version_for_existing_sop(
                db, existing, payload, resolved_title, department, sop_number, background_tasks
            )

    dept_final = _truncate_field(department, 100) or "Quality"
    title_final = _truncate_field(resolved_title, 255) or "Untitled SOP"

    sop = SOP(
        id=new_sop_id,
        tenant_id=FIXED_TENANT_ID,
        title=title_final,
        sop_number=sop_number,
        department=dept_final,
        is_active=True,
        current_version_id=new_ver_id,
    )

    normalized_meta = _normalize_sop_metadata(
        sop_number=sop_number,
        title=title_final,
        department=dept_final,
        raw_meta=payload_meta_json,
    )
    resolved_external_status = _resolve_external_status_from_payload(payload_meta_json, fallback="draft")
    logger.info(
        "[sop-status] create document sop_number=%s resolved_external_status=%s payload_sopStatus=%s payload_status=%s",
        sop_number,
        resolved_external_status,
        (payload_meta_json or {}).get("sopStatus") if isinstance(payload_meta_json, dict) else None,
        (payload_meta_json or {}).get("status") if isinstance(payload_meta_json, dict) else None,
    )

    initial_version = SOPVersion(
        id=new_ver_id,
        sop_id=new_sop_id,
        version_number=normalized_meta.get("sopMetadata", {}).get("sopVersion") or "1",
        content_json=payload_doc_json if payload_doc_json is not None else {"type": "doc", "content": []},
        metadata_json=normalized_meta,
        effective_date=_metadata_date(normalized_meta.get("sopMetadata", {}).get("effectiveDate")),
        review_date=_metadata_date(normalized_meta.get("sopMetadata", {}).get("reviewDate")),
        external_status=resolved_external_status,
    )

    db.add(sop)
    db.add(initial_version)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raced = (
            db.query(SOP)
            .filter(SOP.sop_number == sop_number, SOP.tenant_id == FIXED_TENANT_ID)
            .first()
        )
        if raced:
            return _create_new_version_for_existing_sop(
                db, raced, payload, resolved_title, department, sop_number, background_tasks
            )
        logger.exception("[create_document] integrity error for sop_number=%s", sop_number)
        raise HTTPException(
            status_code=409,
            detail=f"{_DUP_SOP_MSG} DB detail: {str(exc.orig)[:300]}",
        ) from None
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("[create_document] database error for sop_number=%s", sop_number)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to create document due to database validation error: {str(exc)[:500]}",
        ) from None
    except Exception as exc:
        db.rollback()
        logger.exception("[create_document] unexpected error for sop_number=%s", sop_number)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create document: {type(exc).__name__}: {str(exc)[:500]}",
        ) from None

    db.refresh(sop)
    db.refresh(initial_version)
    logger.info("[sop-editor] upload saved sop_id=%s version_id=%s", sop.id, initial_version.id)

    _upsert_import_context_entities(db, sop, initial_version, background_tasks)
    _schedule_semantic_job(background_tasks, "sop", sop.id, initial_version.id)

    try:
        plain_text = _extract_plain_text_from_tiptap(initial_version.content_json)
        analyze_and_store_sop_profile(
            db=db,
            sop_id=sop.id,
            sop_version_id=initial_version.id,
            text=plain_text,
            client_name="Client",
            source_filename=sop.title
        )
    except Exception as e:
        logger.error(f"SOP profile analysis failed for {sop.id}: {e}")

    return _build_editor_doc_response(sop, initial_version)


@router.get("/api/editor/docs/{doc_id}", response_model=EditorDocResponse)
def get_document(doc_id: str, db: Session = Depends(get_db)):
    """
    Fetch SOP + current version, return in old editor shape.
    Response uses doc_json (mapped from content_json) and status (mapped from external_status).
    """
    # Handle lookup by either UUID (id) or SOP Number
    sop = None
    try:
        id_val = uuid.UUID(doc_id)
        sop = db.query(SOP).filter(SOP.id == id_val, SOP.is_active == True).first()  # noqa: E712
    except ValueError:
        sop = db.query(SOP).filter(SOP.sop_number == doc_id, SOP.is_active == True).first()  # noqa: E712

    if not sop:
        raise HTTPException(status_code=404, detail="Document not found")

    current_version = _resolve_current_version(db, sop)
    if not current_version:
        raise HTTPException(status_code=404, detail="Current version not found in sop_versions")

    return _build_editor_doc_response(sop, current_version)


@router.put("/api/editor/docs/{doc_id}", response_model=EditorDocResponse)
def update_document(
    doc_id: str,
    payload: UpdateDocumentRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _=Depends(check_mock_mode),
):
    """
    Update the current version's content in-place.
    Stores incoming doc_json into content_json — no column renamed.
    Does NOT break version history (other versions untouched).
    """
    # Handle lookup by either UUID (id) or SOP Number
    sop = None
    try:
        id_val = uuid.UUID(doc_id)
        sop = db.query(SOP).filter(SOP.id == id_val, SOP.is_active == True).first()  # noqa: E712
    except ValueError:
        sop = db.query(SOP).filter(SOP.sop_number == doc_id, SOP.is_active == True).first()  # noqa: E712

    if not sop:
        raise HTTPException(status_code=404, detail="Document not found")

    current_version = _resolve_current_version(db, sop)
    if not current_version:
        raise HTTPException(status_code=404, detail="Current version not found")

    # Keep SOP header/title in sync with editor metadata or explicit payload title.
    payload_doc_json = _sanitize_json_like(payload.doc_json)
    payload_meta_json = _sanitize_json_like(payload.metadata_json)

    incoming_title = strip_invalid_control_chars(payload.title) if payload.title else payload.title
    if not incoming_title and isinstance(payload_meta_json, dict):
        incoming_title = payload_meta_json.get("sopMetadata", {}).get("title")
    if incoming_title:
        sop.title = incoming_title.strip() or sop.title
    incoming_meta = payload_meta_json.get("sopMetadata", {}) if isinstance(payload_meta_json, dict) else {}
    incoming_sop_number = (incoming_meta.get("documentId") or "").strip()
    if incoming_sop_number and incoming_sop_number != sop.sop_number:
        conflict = db.query(SOP).filter(SOP.sop_number == incoming_sop_number, SOP.id != sop.id).first()
        if conflict:
            raise HTTPException(status_code=409, detail=_DUP_SOP_MSG)
        sop.sop_number = incoming_sop_number
    incoming_department = (incoming_meta.get("department") or "").strip()
    if incoming_department:
        sop.department = _truncate_field(incoming_department, 100) or sop.department

    # doc_json from frontend → stored as content_json in DB
    current_version.content_json = payload_doc_json
    if payload_meta_json is not None:
        normalized_meta = _normalize_sop_metadata(
            sop_number=sop.sop_number,
            title=sop.title,
            department=sop.department,
            raw_meta=payload_meta_json,
        )
        current_version.metadata_json = normalized_meta
        current_version.effective_date = _metadata_date(normalized_meta.get("sopMetadata", {}).get("effectiveDate"))
        current_version.review_date = _metadata_date(normalized_meta.get("sopMetadata", {}).get("reviewDate"))

    db.commit()
    db.refresh(sop)
    db.refresh(current_version)
    logger.info("[sop-editor] document saved sop_id=%s version_id=%s", sop.id, current_version.id)

    _upsert_import_context_entities(db, sop, current_version, background_tasks)
    _schedule_semantic_job(background_tasks, "sop", sop.id, current_version.id)

    try:
        plain_text = _extract_plain_text_from_tiptap(current_version.content_json)
        analyze_and_store_sop_profile(
            db=db,
            sop_id=sop.id,
            sop_version_id=current_version.id,
            text=plain_text,
            client_name="Client",
            source_filename=sop.title
        )
    except Exception as e:
        logger.error(f"SOP profile analysis failed for {sop.id}: {e}")

    return _build_editor_doc_response(sop, current_version)


@router.delete("/api/editor/docs/{doc_id}")
def delete_document(
    doc_id: str,
    db: Session = Depends(get_db),
    _=Depends(check_mock_mode),
):
    """
    Permanently delete a SOP and all its versions/links.
    """
    # Handle lookup by either UUID (id) or SOP Number
    sop = None
    try:
        id_val = uuid.UUID(doc_id)
        sop = db.query(SOP).filter(SOP.id == id_val).first()
    except ValueError:
        sop = db.query(SOP).filter(SOP.sop_number == doc_id).first()

    if not sop:
        raise HTTPException(status_code=404, detail="Document not found")

    profile_cleanup = cleanup_profile_for_deleted_sop(db, sop.id)

    # Collect directly linked entities before removing link rows.
    linked_deviation_ids = [
        row[0]
        for row in db.query(SopDeviationLink.deviation_id)
        .filter(SopDeviationLink.sop_id == sop.id)
        .all()
    ]
    linked_decision_ids = [
        row[0]
        for row in db.query(DecisionSopLink.decision_id)
        .filter(DecisionSopLink.sop_id == sop.id)
        .all()
    ]

    # Remove SOP link edges first (FK constraints are not consistently CASCADE).
    db.query(SopDeviationLink).filter(SopDeviationLink.sop_id == sop.id).delete(synchronize_session=False)
    db.query(DecisionSopLink).filter(DecisionSopLink.sop_id == sop.id).delete(synchronize_session=False)

    # Versions are deleted by SQLAlchemy cascade (all, delete-orphan).
    db.delete(sop)
    db.flush()

    # Delete orphan deviations (only those no longer linked to any SOP).
    orphan_deviation_ids = [
        dev_id for dev_id in set(linked_deviation_ids)
        if not db.query(SopDeviationLink.id).filter(SopDeviationLink.deviation_id == dev_id).first()
    ]

    linked_capa_ids = []
    if orphan_deviation_ids:
        linked_capa_ids = [
            row[0]
            for row in db.query(DeviationCapaLink.capa_id)
            .filter(DeviationCapaLink.deviation_id.in_(orphan_deviation_ids))
            .all()
        ]
        db.query(DeviationCapaLink).filter(
            DeviationCapaLink.deviation_id.in_(orphan_deviation_ids)
        ).delete(synchronize_session=False)
        db.query(Deviation).filter(Deviation.id.in_(orphan_deviation_ids)).delete(synchronize_session=False)

    # Delete orphan CAPAs (no remaining Deviation->CAPA links).
    orphan_capa_ids = [
        capa_id for capa_id in set(linked_capa_ids)
        if not db.query(DeviationCapaLink.id).filter(DeviationCapaLink.capa_id == capa_id).first()
    ]

    linked_audit_ids = []
    if orphan_capa_ids:
        linked_audit_ids = [
            row[0]
            for row in db.query(CapaAuditLink.audit_finding_id)
            .filter(CapaAuditLink.capa_id.in_(orphan_capa_ids))
            .all()
        ]
        db.query(CapaAuditLink).filter(
            CapaAuditLink.capa_id.in_(orphan_capa_ids)
        ).delete(synchronize_session=False)
        db.query(Capa).filter(Capa.id.in_(orphan_capa_ids)).delete(synchronize_session=False)

    # Delete orphan Audits (no remaining CAPA->Audit links).
    orphan_audit_ids = [
        audit_id for audit_id in set(linked_audit_ids)
        if not db.query(CapaAuditLink.id).filter(CapaAuditLink.audit_finding_id == audit_id).first()
    ]

    audit_decision_ids = []
    if orphan_audit_ids:
        audit_decision_ids = [
            row[0]
            for row in db.query(AuditDecisionLink.decision_id)
            .filter(AuditDecisionLink.audit_finding_id.in_(orphan_audit_ids))
            .all()
        ]
        db.query(AuditDecisionLink).filter(
            AuditDecisionLink.audit_finding_id.in_(orphan_audit_ids)
        ).delete(synchronize_session=False)
        db.query(AuditFinding).filter(
            AuditFinding.id.in_(orphan_audit_ids)
        ).delete(synchronize_session=False)

    # Delete orphan Decisions (no remaining SOP/Audit links).
    decision_candidates = set(linked_decision_ids + audit_decision_ids)
    orphan_decision_ids = [
        decision_id for decision_id in decision_candidates
        if not db.query(DecisionSopLink.id).filter(DecisionSopLink.decision_id == decision_id).first()
        and not db.query(AuditDecisionLink.id).filter(AuditDecisionLink.decision_id == decision_id).first()
    ]

    if orphan_decision_ids:
        db.query(Decision).filter(Decision.id.in_(orphan_decision_ids)).delete(synchronize_session=False)

    db.commit()

    # Keep semantic index in sync with relational source-of-truth after hard deletes.
    SemanticPipelineService.purge_entity_artifacts("sop", sop.id)
    for dev_id in orphan_deviation_ids:
        SemanticPipelineService.purge_entity_artifacts("deviation", dev_id)
    for capa_id in orphan_capa_ids:
        SemanticPipelineService.purge_entity_artifacts("capa", capa_id)
    for audit_id in orphan_audit_ids:
        SemanticPipelineService.purge_entity_artifacts("audit_finding", audit_id)
    for decision_id in orphan_decision_ids:
        SemanticPipelineService.purge_entity_artifacts("decision", decision_id)

    return {"status": "deleted", "id": doc_id, "profile_cleanup": profile_cleanup}


@router.get("/api/editor/docs/{doc_id}/versions", response_model=List[EditorVersionResponse])
def list_versions(doc_id: str, db: Session = Depends(get_db)):
    """
    Return all versions for a SOP using old editor field names.
    doc_json   <- content_json
    doc_id     <- sop_id
    status     <- external_status
    """
    sop = db.query(SOP).filter(SOP.id == doc_id, SOP.is_active == True).first()  # noqa: E712
    if not sop:
        raise HTTPException(status_code=404, detail="Document not found")

    versions = (
        db.query(SOPVersion)
        .filter(SOPVersion.sop_id == doc_id)
        .order_by(SOPVersion.created_at.asc())
        .all()
    )
    return [_build_editor_version_response(v) for v in versions]


@router.post("/api/editor/docs/{doc_id}/versions", response_model=EditorVersionResponse)
def create_version(
    doc_id: str,
    payload: CreateVersionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _=Depends(check_mock_mode),
):
    """
    Create a new version row. Uses TRUE sequential integer calculation.
    """
    sop = db.query(SOP).filter(SOP.id == doc_id, SOP.is_active == True).first()  # noqa: E712
    if not sop:
        raise HTTPException(status_code=404, detail="Document not found")

    payload_doc_json = _sanitize_json_like(payload.doc_json)
    payload_meta_json = _sanitize_json_like(payload.metadata_json)

    if _is_tiptap_empty(payload_doc_json):
        raise HTTPException(
            status_code=422,
            detail="Cannot create a new version with empty content.",
        )

    # Calculate real next integer version
    all_versions = db.query(SOPVersion).filter(SOPVersion.sop_id == doc_id).all()
    max_v = 0
    for v in all_versions:
        try:
            val = int(float(v.version_number))
            if val > max_v: max_v = val
        except:
            try:
                val = int(v.version_number.split('.')[0])
                if val > max_v: max_v = val
            except:
                pass
    next_version = str(max_v + 1)

    version = SOPVersion(
        sop_id=sop.id,
        version_number=next_version,
        content_json=payload_doc_json,
        external_status="draft",
        metadata_json=payload_meta_json or {},
    )
    db.add(version)
    db.flush()
    
    # Point parent to new version
    sop.current_version_id = version.id
    
    # Store justification in metadata if provided
    if payload.change_justification:
        meta_dict = dict(version.metadata_json) if version.metadata_json else {}
        audit_trail = meta_dict.get("auditTrail", [])
        if not isinstance(audit_trail, list):
            audit_trail = []

        new_entry = {
            "action": "created_new_revision",
            "note": payload.change_justification,
            "version": next_version,
            "createdAt": datetime.utcnow().isoformat(),
            "actor": "System",
        }

        # Deduplicate before appending: drop any pre-existing entry that has the
        # same (action, version, note) triple so that re-runs of this endpoint
        # (e.g. from a client retry) never produce cumulative duplicate entries.
        audit_trail = _deduplicate_audit_trail([*audit_trail, new_entry])

        meta_dict["auditTrail"] = audit_trail
        meta_dict["change_justification"] = payload.change_justification
        version.metadata_json = meta_dict

    db.commit()
    db.refresh(version)
    db.refresh(sop)

    accepted_suggestion = None
    if payload.suggestion_id:
        accepted_suggestion = (
            db.query(AISuggestion)
            .filter(AISuggestion.id == payload.suggestion_id, AISuggestion.sop_id == sop.id)
            .first()
        )
    if accepted_suggestion is None and payload.change_justification:
        lowered_note = payload.change_justification.lower()
        if "accepted" in lowered_note and ("rewrite" in lowered_note or "improve" in lowered_note):
            accepted_suggestion = (
                db.query(AISuggestion)
                .filter(
                    AISuggestion.sop_id == sop.id,
                    AISuggestion.status == "pending",
                )
                .order_by(AISuggestion.created_at.desc())
                .first()
            )
            if accepted_suggestion is None:
                accepted_suggestion = (
                    db.query(AISuggestion)
                    .filter(
                        AISuggestion.sop_id.is_(None),
                        AISuggestion.status == "pending",
                    )
                    .order_by(AISuggestion.created_at.desc())
                    .first()
                )
    if accepted_suggestion is not None:
        if accepted_suggestion.sop_id is None:
            accepted_suggestion.sop_id = sop.id
        if accepted_suggestion.sop_version_id is None:
            accepted_suggestion.sop_version_id = version.id
        accepted_suggestion.status = "accepted"
        accepted_suggestion.accepted_version_id = version.id
        accepted_suggestion.accepted_at = datetime.utcnow()
        meta = dict(accepted_suggestion.metadata_json or {})
        meta["accepted_change_justification"] = payload.change_justification
        meta["accepted_sop_version"] = next_version
        accepted_suggestion.metadata_json = meta
        flag_modified(accepted_suggestion, "metadata_json")
        should_learn = bool((accepted_suggestion.metadata_json or {}).get("learn_to_profile"))
        if should_learn and accepted_suggestion.profile_id:
            profile = db.query(ClientProfile).filter(ClientProfile.id == accepted_suggestion.profile_id).first()
            if profile:
                profile_result = save_profile_version_from_accepted_suggestion(
                    db,
                    profile=profile,
                    suggestion=accepted_suggestion,
                    change_reason=f"Auto-learned from accepted {accepted_suggestion.action} suggestion",
                )
                meta["profile_auto_updated"] = True
                meta["profile_auto_update_result"] = profile_result
                accepted_suggestion.metadata_json = meta
                flag_modified(accepted_suggestion, "metadata_json")
        db.commit()
        db.refresh(accepted_suggestion)

    _schedule_semantic_job(background_tasks, "sop", sop.id, version.id)
    return _build_editor_version_response(version)


@router.get("/api/editor/docs/{doc_id}/versions/{version_id}", response_model=EditorVersionResponse)
def get_version(doc_id: str, version_id: str, db: Session = Depends(get_db)):
    """
    Fetch a specific version by doc_id (= sop_id) and version_id.
    Returns old editor field names.
    """
    sop = db.query(SOP).filter(SOP.id == doc_id, SOP.is_active == True).first()  # noqa: E712
    if not sop:
        raise HTTPException(status_code=404, detail="Document not found")

    version = (
        db.query(SOPVersion)
        .filter(SOPVersion.sop_id == doc_id, SOPVersion.id == version_id)
        .first()
    )
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    return _build_editor_version_response(version)


@router.post("/api/editor/docs/{doc_id}/duplicate", response_model=EditorDocResponse)
def duplicate_document(
    doc_id: str,
    payload: CreateDocumentRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _=Depends(check_mock_mode),
):
    """
    Duplicate an existing SOP. Reset to version 1.
    """
    source_sop = db.query(SOP).filter(SOP.id == doc_id, SOP.is_active == True).first()  # noqa: E712
    if not source_sop:
        raise HTTPException(status_code=404, detail="Source document not found")

    payload_doc_json = _sanitize_json_like(payload.doc_json)
    payload_meta_json = _sanitize_json_like(payload.metadata_json)
    payload_title = strip_invalid_control_chars(payload.title) if payload.title else payload.title

    if payload_doc_json is None:
        source_version = db.query(SOPVersion).filter(SOPVersion.id == source_sop.current_version_id).first()
        content = source_version.content_json if source_version else {"type": "doc", "content": []}
    else:
        content = payload_doc_json

    new_sop_id = uuid.uuid4()
    new_ver_id = uuid.uuid4()
    new_sop_num = f"SOP-{uuid.uuid4().hex[:8].upper()}"

    new_sop = SOP(
        id=new_sop_id,
        tenant_id=FIXED_TENANT_ID,
        title=payload_title or f"Copy of {source_sop.title}",
        sop_number=new_sop_num,
        department=source_sop.department,
        is_active=True,
        current_version_id=new_ver_id
    )
    db.add(new_sop)

    new_version = SOPVersion(
        id=new_ver_id,
        sop_id=new_sop_id,
        version_number="1",
        content_json=content,
        external_status="draft",
        metadata_json=payload_meta_json or {},
    )
    db.add(new_version)
    
    # CRITICAL: Link parent SOP's current_version_id back to this new version
    new_sop.current_version_id = new_ver_id
    
    db.commit()
    db.refresh(new_sop)
    db.refresh(new_version)
    _schedule_semantic_job(background_tasks, "sop", new_sop.id, new_version.id)
    return _build_editor_doc_response(new_sop, new_version)


@router.put("/api/editor/docs/{doc_id}/versions/{version_id}/status", response_model=EditorVersionResponse)
def update_version_status(
    doc_id: str,
    version_id: str,
    payload: UpdateVersionStatusRequest,
    db: Session = Depends(get_db),
    _=Depends(check_mock_mode),
):
    """
    Update sop_versions.external_status.
    Supports: draft, under_review, effective, obsolete.
    """
    VALID_STATUSES = {"draft", "under_review", "effective", "obsolete"}
    if payload.status not in VALID_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status '{payload.status}'. Must be one of: {', '.join(sorted(VALID_STATUSES))}"
        )

    sop = db.query(SOP).filter(SOP.id == doc_id, SOP.is_active == True).first()  # noqa: E712
    if not sop:
        raise HTTPException(status_code=404, detail="Document not found")

    version = (
        db.query(SOPVersion)
        .filter(SOPVersion.sop_id == doc_id, SOPVersion.id == version_id)
        .first()
    )
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")

    version.external_status = strip_invalid_control_chars(payload.status)
    if payload.metadata_json is not None:
        version.metadata_json = _sanitize_json_like(payload.metadata_json)

    db.commit()
    db.refresh(version)

    return _build_editor_version_response(version)


# ==========================================
# NEW SOP NATIVE ROUTES
# All field names match DB schema exactly: content_json, external_status, sop_id
# ==========================================

@router.get("/api/sops", response_model=List[SOPResponse])
def get_all_sops(db: Session = Depends(get_db)):
    """
    Return all SOPs for the fixed tenant.
    Each entry includes current_version embedded summary for convenience.
    """
    sops = _tenant_scoped_query(db, SOP).filter(SOP.is_active == True).all()  # noqa: E712
    logger.info("[api/sops] returning active SOPs count=%s", len(sops))
    if not sops:
        return []

    version_ids = [s.current_version_id for s in sops if s.current_version_id]
    version_map = {}
    if version_ids:
        for version in db.query(SOPVersion).filter(SOPVersion.id.in_(version_ids)).all():
            version_map[version.id] = version

    out = []
    for sop in sops:
        result = _build_sop_dict(sop, include_current_version=False, db=db)
        cv = version_map.get(sop.current_version_id) if sop.current_version_id else None
        if cv:
            normalized_meta = _normalize_sop_metadata(
                sop_number=sop.sop_number,
                title=sop.title,
                department=sop.department,
                raw_meta=cv.metadata_json,
            )
            result["current_version"] = {
                "id": cv.id,
                "sop_id": cv.sop_id,
                "external_version_id": cv.external_version_id,
                "version_number": cv.version_number,
                "external_status": cv.external_status,
                "content_json": cv.content_json,
                "metadata_json": normalized_meta,
                "effective_date": cv.effective_date,
                "review_date": cv.review_date,
                "created_at": cv.created_at,
                "updated_at": cv.updated_at,
            }
        out.append(result)
    return out


@router.get("/api/sops/{id}", response_model=SOPResponse)
def get_sop_by_id(id: str, db: Session = Depends(get_db)):
    """
    Return one SOP by id, with current_version embedded as a nested object.
    Uses native DB field names: content_json, external_status.
    """
    sop = _resolve_sop_lookup(db, id, include_inactive=False)

    if not sop:
        raise HTTPException(status_code=404, detail="SOP not found")

    return _build_sop_dict(sop, include_current_version=True, db=db)


@router.get("/api/sops/{id}/versions", response_model=list[SOPVersionResponse])
def get_sop_versions(id: str, db: Session = Depends(get_db)):
    """
    Return all sop_versions rows where sop_id = {id}.
    Native field names preserved.
    """
    sop = _resolve_sop_lookup(db, id, include_inactive=False)

    if not sop:
        raise HTTPException(status_code=404, detail="SOP not found")

    return (
        db.query(SOPVersion)
        .filter(SOPVersion.sop_id == sop.id)
        .order_by(SOPVersion.created_at.asc())
        .all()
    )


@router.get("/api/sops/{id}/related", response_model=SopRelatedResponse)
def get_sop_related_context(id: str, db: Session = Depends(get_db)):
    """
    Return full related context for the SOP traversing the full link chain:
    sop → deviations → CAPAs → audit_findings → decisions
    Also resolves decision → sop back-links.
    """
    # Handle lookup by either UUID (id) or SOP Number
    sop = _resolve_sop_lookup(db, id, include_inactive=False)

    if not sop:
        raise HTTPException(status_code=404, detail="SOP not found")

    # 1. Deviations linked to SOP
    dev_ids = {
        row[0] for row in db.query(SopDeviationLink.deviation_id).filter(SopDeviationLink.sop_id == sop.id).all()
    }

    # 2. Decisions directly linked to SOP
    decision_ids = {
        row[0] for row in db.query(DecisionSopLink.decision_id).filter(DecisionSopLink.sop_id == sop.id).all()
    }

    # 3. Traversal: expand from decisions (Decision → Audit → CAPA → Deviation)
    audit_ids = set()
    if decision_ids:
        audit_ids = {
            row[0]
            for row in db.query(AuditDecisionLink.audit_finding_id)
            .filter(AuditDecisionLink.decision_id.in_(list(decision_ids)))
            .all()
        }

    # 4. Traversal: expand from deviations (Deviation → CAPA → Audit → Decision)
    capa_ids = set()
    if dev_ids:
        capa_ids = {
            row[0]
            for row in db.query(DeviationCapaLink.capa_id)
            .filter(DeviationCapaLink.deviation_id.in_(list(dev_ids)))
            .all()
        }

    # 5. Connect CAPAs and Audits (Bidirectional)
    if capa_ids:
        audit_ids.update(
            row[0]
            for row in db.query(CapaAuditLink.audit_finding_id)
            .filter(CapaAuditLink.capa_id.in_(list(capa_ids)))
            .all()
        )
            
    if audit_ids:
        capa_ids.update(
            row[0]
            for row in db.query(CapaAuditLink.capa_id)
            .filter(CapaAuditLink.audit_finding_id.in_(list(audit_ids)))
            .all()
        )

    # 6. Re-expand from CAPAs to Deviations (Reverse)
    if capa_ids:
        dev_ids.update(
            row[0]
            for row in db.query(DeviationCapaLink.deviation_id)
            .filter(DeviationCapaLink.capa_id.in_(list(capa_ids)))
            .all()
        )

    # 7. Final expansion for Decisions from Audits
    if audit_ids:
        decision_ids.update(
            row[0]
            for row in db.query(AuditDecisionLink.decision_id)
            .filter(AuditDecisionLink.audit_finding_id.in_(list(audit_ids)))
            .all()
        )

    # 8. SOP-to-SOP chaining via shared decisions:
    # gather all SOPs connected to the expanded decision set
    related_sop_ids = set()
    if decision_ids:
        related_sop_ids.update(
            row[0]
            for row in db.query(DecisionSopLink.sop_id)
            .filter(DecisionSopLink.decision_id.in_(list(decision_ids)))
            .all()
            if row[0] != sop.id
        )

    # include incoming reverse links to this SOP as additional chaining evidence
    incoming_decision_ids = {
        row[0] for row in db.query(DecisionSopLink.decision_id).filter(DecisionSopLink.sop_id == sop.id).all()
    }
    if incoming_decision_ids:
        related_sop_ids.update(
            row[0]
            for row in db.query(DecisionSopLink.sop_id)
            .filter(DecisionSopLink.decision_id.in_(list(incoming_decision_ids)))
            .all()
            if row[0] != sop.id
        )

    related_sops_raw = _tenant_scoped_query(db, SOP).filter(SOP.id.in_(list(related_sop_ids))).all() if related_sop_ids else []
    related_sops = [_build_sop_dict(item, include_current_version=True, db=db) for item in related_sops_raw]

    related_deviations = db.query(Deviation).filter(Deviation.id.in_(list(dev_ids))).all() if dev_ids else []
    related_capas = db.query(Capa).filter(Capa.id.in_(list(capa_ids))).all() if capa_ids else []
    related_audit_findings = db.query(AuditFinding).filter(AuditFinding.id.in_(list(audit_ids))).all() if audit_ids else []
    related_decisions = db.query(Decision).filter(Decision.id.in_(list(decision_ids))).all() if decision_ids else []

    return {
        "sop": _build_sop_dict(sop, include_current_version=True, db=db),
        "related_sops": related_sops,
        "related_deviations": related_deviations,
        "related_capas": related_capas,
        "related_audit_findings": related_audit_findings,
        "related_decisions": related_decisions,
    }


# ==========================================
# DEVIATION ROUTES
# ==========================================

@router.get("/api/deviations/{id}", response_model=DeviationResponse)
def get_deviation_by_id(id: str, db: Session = Depends(get_db)):
    """Return a single Deviation record."""
    dev = db.query(Deviation).filter(Deviation.id == id, Deviation.tenant_id == FIXED_TENANT_ID).first()
    if not dev:
        raise HTTPException(status_code=404, detail="Deviation not found")
    return dev


@router.get("/api/deviations/{id}/context", response_model=DeviationContextResponse)
def get_deviation_context(id: str, db: Session = Depends(get_db)):
    """
    Return full chain context for a Deviation:
    deviation → SOP, CAPA, audit_finding, decisions
    """
    dev = db.query(Deviation).filter(Deviation.id == id, Deviation.tenant_id == FIXED_TENANT_ID).first()
    if not dev:
        raise HTTPException(status_code=404, detail="Deviation not found")

    # Linked SOPs
    sop_links = db.query(SopDeviationLink).filter(SopDeviationLink.deviation_id == dev.id).all()
    sop_ids = [l.sop_id for l in sop_links]
    related_sops_raw = db.query(SOP).filter(SOP.id.in_(sop_ids)).all() if sop_ids else []
    related_sops = [_build_sop_dict(s, include_current_version=True, db=db) for s in related_sops_raw]

    # Linked CAPAs
    capa_links = db.query(DeviationCapaLink).filter(DeviationCapaLink.deviation_id == dev.id).all()
    capa_ids = [l.capa_id for l in capa_links]
    related_capas = db.query(Capa).filter(Capa.id.in_(capa_ids)).all() if capa_ids else []

    # Linked Audit Findings (from CAPAs)
    audit_links = db.query(CapaAuditLink).filter(CapaAuditLink.capa_id.in_(capa_ids)).all() if capa_ids else []
    audit_ids = [l.audit_finding_id for l in audit_links]
    related_audits = db.query(AuditFinding).filter(AuditFinding.id.in_(audit_ids)).all() if audit_ids else []

    # Linked Decisions (from audit findings)
    decision_links = (
        db.query(AuditDecisionLink).filter(AuditDecisionLink.audit_finding_id.in_(audit_ids)).all()
        if audit_ids else []
    )
    decision_ids = [l.decision_id for l in decision_links]
    related_decisions = db.query(Decision).filter(Decision.id.in_(decision_ids)).all() if decision_ids else []

    return {
        "deviation": dev,
        "related_sops": related_sops,
        "related_capas": related_capas,
        "related_audits": related_audits,
        "related_decisions": related_decisions,
    }

@router.get("/api/deviations", response_model=List[DeviationResponse])
def get_all_deviations(db: Session = Depends(get_db)):
    """Return all Deviation records."""
    return _tenant_scoped_query(db, Deviation).all()

@router.post("/api/deviations", response_model=DeviationResponse)
def create_deviation(payload: DeviationCreateUpdate, db: Session = Depends(get_db), background_tasks: BackgroundTasks = None):
    """Create a new Deviation record."""
    dev = Deviation(
        tenant_id=FIXED_TENANT_ID,
        **payload.model_dump()
    )
    db.add(dev)
    db.commit()
    db.refresh(dev)
    if background_tasks:
        _schedule_semantic_job(background_tasks, "deviation", dev.id)
    return dev

@router.put("/api/deviations/{id}", response_model=DeviationResponse)
def update_deviation(id: str, payload: DeviationCreateUpdate, db: Session = Depends(get_db), background_tasks: BackgroundTasks = None):
    """Update an existing Deviation record."""
    dev = db.query(Deviation).filter(Deviation.id == id, Deviation.tenant_id == FIXED_TENANT_ID).first()
    if not dev:
        raise HTTPException(status_code=404, detail="Deviation not found")
    
    for key, value in payload.model_dump().items():
        setattr(dev, key, value)
    
    db.commit()
    db.refresh(dev)
    if background_tasks:
        _schedule_semantic_job(background_tasks, "deviation", dev.id)
    return dev

# ==========================================
# CAPA ROUTES
# ==========================================

@router.get("/api/capas", response_model=List[CapaResponse])
def get_all_capas(db: Session = Depends(get_db)):
    """Return all CAPA records."""
    return _tenant_scoped_query(db, Capa).all()

@router.post("/api/capas", response_model=CapaResponse)
def create_capa(payload: CapaCreateUpdate, db: Session = Depends(get_db), background_tasks: BackgroundTasks = None):
    """Create a new CAPA record."""
    capa = Capa(
        tenant_id=FIXED_TENANT_ID,
        **payload.model_dump()
    )
    db.add(capa)
    db.commit()
    db.refresh(capa)
    if background_tasks:
        _schedule_semantic_job(background_tasks, "capa", capa.id)
    return capa

@router.get("/api/capas/{id}", response_model=CapaResponse)
def get_capa(id: str, db: Session = Depends(get_db)):
    """Return a single CAPA record."""
    capa = db.query(Capa).filter(Capa.id == id, Capa.tenant_id == FIXED_TENANT_ID).first()
    if not capa:
        raise HTTPException(status_code=404, detail="CAPA not found")
    return capa

@router.put("/api/capas/{id}", response_model=CapaResponse)
def update_capa(id: str, payload: CapaCreateUpdate, db: Session = Depends(get_db), background_tasks: BackgroundTasks = None):
    """Update an existing CAPA record."""
    capa = db.query(Capa).filter(Capa.id == id, Capa.tenant_id == FIXED_TENANT_ID).first()
    if not capa:
        raise HTTPException(status_code=404, detail="CAPA not found")
    
    for key, value in payload.model_dump().items():
        setattr(capa, key, value)
    
    db.commit()
    db.refresh(capa)
    if background_tasks:
        _schedule_semantic_job(background_tasks, "capa", capa.id)
    return capa

# ==========================================
# AUDIT ROUTES
# ==========================================

@router.get("/api/audits", response_model=List[AuditFindingResponse])
def get_all_audits(db: Session = Depends(get_db)):
    """Return all Audit Finding records."""
    return _tenant_scoped_query(db, AuditFinding).all()

@router.post("/api/audits", response_model=AuditFindingResponse)
def create_audit(payload: AuditFindingCreateUpdate, db: Session = Depends(get_db), background_tasks: BackgroundTasks = None):
    """Create a new Audit Finding record."""
    audit = AuditFinding(
        tenant_id=FIXED_TENANT_ID,
        **payload.model_dump()
    )
    db.add(audit)
    db.commit()
    db.refresh(audit)
    if background_tasks:
        _schedule_semantic_job(background_tasks, "audit_finding", audit.id)
    return audit

@router.get("/api/audits/{id}", response_model=AuditFindingResponse)
def get_audit(id: str, db: Session = Depends(get_db)):
    """Return a single Audit Finding record."""
    parsed_id = _parse_uuid_or_400(id)
    audit = db.query(AuditFinding).filter(AuditFinding.id == parsed_id, AuditFinding.tenant_id == FIXED_TENANT_ID).first()
    if not audit:
        raise HTTPException(status_code=404, detail="Audit Finding not found")
    return audit

@router.put("/api/audits/{id}", response_model=AuditFindingResponse)
def update_audit(id: str, payload: AuditFindingCreateUpdate, db: Session = Depends(get_db), background_tasks: BackgroundTasks = None):
    """Update an existing Audit Finding record."""
    parsed_id = _parse_uuid_or_400(id)
    audit = db.query(AuditFinding).filter(AuditFinding.id == parsed_id, AuditFinding.tenant_id == FIXED_TENANT_ID).first()
    if not audit:
        raise HTTPException(status_code=404, detail="Audit Finding not found")
    
    for key, value in payload.model_dump().items():
        setattr(audit, key, value)
    
    db.commit()
    db.refresh(audit)
    if background_tasks:
        _schedule_semantic_job(background_tasks, "audit_finding", audit.id)
    return audit

# ==========================================
# DECISION ROUTES
# ==========================================

@router.get("/api/decisions", response_model=List[DecisionResponse])
def get_all_decisions(db: Session = Depends(get_db)):
    """Return all Decision records."""
    return _tenant_scoped_query(db, Decision).all()

@router.post("/api/decisions", response_model=DecisionResponse)
def create_decision(payload: DecisionCreateUpdate, db: Session = Depends(get_db), background_tasks: BackgroundTasks = None):
    """Create a new Decision record."""
    decision = Decision(
        tenant_id=FIXED_TENANT_ID,
        **payload.model_dump()
    )
    db.add(decision)
    db.commit()
    db.refresh(decision)
    if background_tasks:
        _schedule_semantic_job(background_tasks, "decision", decision.id)
    return decision

@router.get("/api/decisions/{id}", response_model=DecisionResponse)
def get_decision(id: str, db: Session = Depends(get_db)):
    """Return a single Decision record."""
    parsed_id = _parse_uuid_or_400(id)
    decision = db.query(Decision).filter(Decision.id == parsed_id, Decision.tenant_id == FIXED_TENANT_ID).first()
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    return decision

@router.put("/api/decisions/{id}", response_model=DecisionResponse)
def update_decision(id: str, payload: DecisionCreateUpdate, db: Session = Depends(get_db), background_tasks: BackgroundTasks = None):
    """Update an existing Decision record."""
    parsed_id = _parse_uuid_or_400(id)
    decision = db.query(Decision).filter(Decision.id == parsed_id, Decision.tenant_id == FIXED_TENANT_ID).first()
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    
    for key, value in payload.model_dump().items():
        setattr(decision, key, value)
    
    db.commit()
    db.refresh(decision)
    if background_tasks:
        _schedule_semantic_job(background_tasks, "decision", decision.id)
    return decision

# ==========================================
# DATASET IMPORT ROUTE
# ==========================================

@router.post("/api/import/dataset")
def import_dataset(payload: DatasetImportRequest, db: Session = Depends(get_db), background_tasks: BackgroundTasks = None):
    """
    Import a dataset of entities (SOPs, Deviations, CAPAs, Audit Findings, Decisions, and Links).
    Transactional and supports nested batches for bulk ingestion.
    """
    try:
        default_tenant = uuid.UUID("11111111-1111-1111-1111-111111111111")
        counts = {"sops": 0, "deviations": 0, "capas": 0, "audits": 0, "decisions": 0, "links": 0, "failed_links": 0}
        reindex_entities: set[tuple[str, uuid.UUID]] = set()

        all_links = []

        def find_existing(model, *, entity_id=None, external_id=None, lookup_field=None, lookup_value=None):
            if entity_id:
                existing = db.query(model).filter(model.id == entity_id).first()
                if existing:
                    return existing
            if external_id:
                existing = db.query(model).filter(model.external_id == external_id).first()
                if existing:
                    return existing
            if lookup_field and lookup_value:
                existing = db.query(model).filter(getattr(model, lookup_field) == lookup_value).first()
                if existing:
                    return existing
            return None

        def resolve_link_entity(model, entity_id=None, external_id=None):
            if entity_id:
                try:
                    parsed_id = uuid.UUID(str(entity_id))
                    existing = db.query(model).filter(model.id == parsed_id).first()
                    if existing:
                        return existing
                except Exception:
                    pass
            if external_id:
                return db.query(model).filter(model.external_id == external_id).first()
            return None

        for batch in payload.entities:
            # Normalize batch keys
            batch_sops = batch.get("sops", [])
            batch_deviations = batch.get("deviations", [])
            batch_capas = batch.get("capas", [])
            batch_audits = batch.get("audit_findings", []) or batch.get("audits", [])
            batch_decisions = batch.get("decisions", [])
            
            # 1. SOPs
            for s in batch_sops:
                sop_id = uuid.UUID(s["id"]) if s.get("id") else None
                sop = find_existing(
    AISuggestion,
    SOP,
                    entity_id=sop_id,
                    external_id=s.get("external_id"),
                    lookup_field="sop_number",
                    lookup_value=s.get("sop_number"),
)
                if not sop:
                    sop_id = sop_id or uuid.uuid4()
                    sop = SOP(
                        id=sop_id,
                        tenant_id=uuid.UUID(s.get("tenant_id")) if s.get("tenant_id") else default_tenant,
                        external_id=s.get("external_id"),
                        sop_number=s.get("sop_number", "SOP-NEW"),
                        title=s.get("title", "Untitled SOP"),
                        department=s.get("department", "Quality"),
                        source_system=s.get("source_system", "import"),
                        is_active=s.get("is_active", True)
                    )
                    db.add(sop)
                    counts["sops"] += 1
                else:
                    sop_id = sop.id
                    if s.get("external_id") and not sop.external_id:
                        sop.external_id = s.get("external_id")
                reindex_entities.add(("sop", sop_id))
                
                # Add Initial Version if provided
                if s.get("versions"):
                    for v in s.get("versions"):
                        v_id = uuid.UUID(v["id"]) if v.get("id") else uuid.uuid4()
                        existing_v = db.query(SOPVersion).filter(SOPVersion.id == v_id).first()
                        if not existing_v:
                            new_v = SOPVersion(
                                id=v_id,
                                sop_id=sop_id,
                                version_number=v.get("version_number", "1"),
                                external_status=v.get("external_status", "effective"),
                                content_json=v.get("content_json", {"type": "doc", "content": []}),
                                metadata_json=v.get("metadata_json", {}),
                                effective_date=v.get("effective_date"),
                                review_date=v.get("review_date")
                            )
                            db.add(new_v)
                            if v.get("is_current") or not sop.current_version_id:
                                sop.current_version_id = v_id
            
            # 2. Deviations
            for d in batch_deviations:
                d_id = uuid.UUID(d["id"]) if d.get("id") else None
                dev = find_existing(
                    Deviation,
                    entity_id=d_id,
                    external_id=d.get("external_id"),
                    lookup_field="deviation_number",
                    lookup_value=d.get("deviation_number"),
                )
                if not dev:
                    d_id = d_id or uuid.uuid4()
                    dev = Deviation(
                        id=d_id,
                        tenant_id=uuid.UUID(d.get("tenant_id")) if d.get("tenant_id") else default_tenant,
                        external_id=d.get("external_id"),
                        deviation_number=d.get("deviation_number", "DEV-NEW"),
                        title=d.get("title", "Untitled Deviation"),
                        category=d.get("category"),
                        site=d.get("site"),
                        product_line=d.get("product_line"),
                        external_status=d.get("external_status", "open"),
                        description_text=d.get("description_text"),
                        root_cause_text=d.get("root_cause_text"),
                        impact_level=d.get("impact_level"),
                        source_system=d.get("source_system", "import")
                    )
                    db.add(dev)
                    counts["deviations"] += 1
                elif d.get("external_id") and not dev.external_id:
                    dev.external_id = d.get("external_id")
                reindex_entities.add(("deviation", dev.id))

            # 3. CAPAs
            for c in batch_capas:
                c_id = uuid.UUID(c["id"]) if c.get("id") else None
                capa = find_existing(
                    Capa,
                    entity_id=c_id,
                    external_id=c.get("external_id"),
                    lookup_field="capa_number",
                    lookup_value=c.get("capa_number"),
                )
                if not capa:
                    c_id = c_id or uuid.uuid4()
                    capa = Capa(
                        id=c_id,
                        tenant_id=uuid.UUID(c.get("tenant_id")) if c.get("tenant_id") else default_tenant,
                        external_id=c.get("external_id"),
                        capa_number=c.get("capa_number", "CAPA-NEW"),
                        title=c.get("title", "Untitled CAPA"),
                        external_status=c.get("external_status", "open"),
                        action_type=c.get("action_type"),
                        action_text=c.get("action_text"),
                        owner_name=c.get("owner_name"),
                        source_system=c.get("source_system", "import")
                    )
                    db.add(capa)
                    counts["capas"] += 1
                elif c.get("external_id") and not capa.external_id:
                    capa.external_id = c.get("external_id")
                reindex_entities.add(("capa", capa.id))

            # 4. Audit Findings
            for a in batch_audits:
                a_id = uuid.UUID(a["id"]) if a.get("id") else None
                audit = find_existing(
                    AuditFinding,
                    entity_id=a_id,
                    external_id=a.get("external_id"),
                    lookup_field="finding_number",
                    lookup_value=a.get("finding_number") or a.get("audit_number"),
                )
                if not audit:
                    a_id = a_id or uuid.uuid4()
                    audit = AuditFinding(
                        id=a_id,
                        tenant_id=uuid.UUID(a.get("tenant_id")) if a.get("tenant_id") else default_tenant,
                        external_id=a.get("external_id"),
                        audit_number=a.get("audit_number"),
                        finding_number=a.get("finding_number"),
                        authority=a.get("authority"),
                        question_text=a.get("question_text"),
                        finding_text=a.get("finding_text"),
                        acceptance_status=a.get("acceptance_status", "pending"),
                        source_system=a.get("source_system", "import")
                    )
                    db.add(audit)
                    counts["audits"] += 1
                elif a.get("external_id") and not audit.external_id:
                    audit.external_id = a.get("external_id")
                reindex_entities.add(("audit_finding", audit.id))

            # 5. Decisions
            for dec in batch_decisions:
                dec_id = uuid.UUID(dec["id"]) if dec.get("id") else None
                decision = find_existing(
                    Decision,
                    entity_id=dec_id,
                    external_id=dec.get("external_id"),
                    lookup_field="decision_number",
                    lookup_value=dec.get("decision_number") or dec.get("title"),
                )
                if not decision:
                    dec_id = dec_id or uuid.uuid4()
                    decision = Decision(
                        id=dec_id,
                        tenant_id=uuid.UUID(dec.get("tenant_id")) if dec.get("tenant_id") else default_tenant,
                        external_id=dec.get("external_id"),
                        decision_number=dec.get("decision_number"),
                        title=dec.get("title", "Untitled Decision"),
                        decision_statement=dec.get("decision_statement", "No statement"),
                        source_system=dec.get("source_system", "import")
                    )
                    db.add(decision)
                    counts["decisions"] += 1
                elif dec.get("external_id") and not decision.external_id:
                    decision.external_id = dec.get("external_id")
                reindex_entities.add(("decision", decision.id))

            # Accumulate Links for after flush
            all_links.extend(batch.get("links", []))

        # Flush entity creations to Database to establish valid primary keys
        db.flush()

        # 6. Process Links (with Validation)
        for l in all_links:
            l_type = (l.get("link_type") or l.get("type", "")).lower()
            source_id = l.get("source_id")
            target_id = l.get("target_id")
            source_external_id = l.get("source_external_id")
            target_external_id = l.get("target_external_id")
            
            if l_type == "sop-deviation":
                source = resolve_link_entity(SOP, entity_id=source_id, external_id=source_external_id)
                target = resolve_link_entity(Deviation, entity_id=target_id, external_id=target_external_id)
                if source and target:
                    if not db.query(SopDeviationLink).filter(SopDeviationLink.sop_id == source.id, SopDeviationLink.deviation_id == target.id).first():
                        db.add(SopDeviationLink(id=uuid.uuid4(), tenant_id=default_tenant, sop_id=source.id, deviation_id=target.id))
                        counts["links"] += 1
                else:
                    counts["failed_links"] += 1
            elif l_type == "deviation-capa":
                source = resolve_link_entity(Deviation, entity_id=source_id, external_id=source_external_id)
                target = resolve_link_entity(Capa, entity_id=target_id, external_id=target_external_id)
                if source and target:
                    if not db.query(DeviationCapaLink).filter(DeviationCapaLink.deviation_id == source.id, DeviationCapaLink.capa_id == target.id).first():
                        db.add(DeviationCapaLink(id=uuid.uuid4(), tenant_id=default_tenant, deviation_id=source.id, capa_id=target.id))
                        counts["links"] += 1
                else:
                    counts["failed_links"] += 1
            elif l_type == "capa-audit":
                source = resolve_link_entity(Capa, entity_id=source_id, external_id=source_external_id)
                target = resolve_link_entity(AuditFinding, entity_id=target_id, external_id=target_external_id)
                if source and target:
                    if not db.query(CapaAuditLink).filter(CapaAuditLink.capa_id == source.id, CapaAuditLink.audit_finding_id == target.id).first():
                        db.add(CapaAuditLink(id=uuid.uuid4(), tenant_id=default_tenant, capa_id=source.id, audit_finding_id=target.id))
                        counts["links"] += 1
                else:
                    counts["failed_links"] += 1
            elif l_type == "audit-decision":
                source = resolve_link_entity(AuditFinding, entity_id=source_id, external_id=source_external_id)
                target = resolve_link_entity(Decision, entity_id=target_id, external_id=target_external_id)
                if source and target:
                    if not db.query(AuditDecisionLink).filter(AuditDecisionLink.audit_finding_id == source.id, AuditDecisionLink.decision_id == target.id).first():
                        db.add(AuditDecisionLink(id=uuid.uuid4(), tenant_id=default_tenant, audit_finding_id=source.id, decision_id=target.id))
                        counts["links"] += 1
                else:
                    counts["failed_links"] += 1
            elif l_type == "decision-sop":
                source = resolve_link_entity(Decision, entity_id=source_id, external_id=source_external_id)
                target = resolve_link_entity(SOP, entity_id=target_id, external_id=target_external_id)
                if source and target:
                    if not db.query(DecisionSopLink).filter(DecisionSopLink.decision_id == source.id, DecisionSopLink.sop_id == target.id).first():
                        db.add(DecisionSopLink(id=uuid.uuid4(), tenant_id=default_tenant, decision_id=source.id, sop_id=target.id))
                        counts["links"] += 1
                else:
                    counts["failed_links"] += 1
            else:
                counts["failed_links"] += 1

        db.commit()
        if background_tasks:
            for et, eid in reindex_entities:
                _schedule_semantic_job(background_tasks, et, eid, job_type="import_reindex")
        return {
            "message": "Import successful",
            "stats": counts
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")

# ==========================================
# SEARCH & STATS AGGREGATION ROUTES
# ==========================================

@router.get("/api/stats")
def get_knowledge_stats(db: Session = Depends(get_db)):
    """Return total counts of all entities in the tenant."""
    return {
        "sops": _tenant_scoped_query(db, SOP).count(),
        "deviations": _tenant_scoped_query(db, Deviation).count(),
        "capas": _tenant_scoped_query(db, Capa).count(),
        "audits": _tenant_scoped_query(db, AuditFinding).count(),
        "decisions": _tenant_scoped_query(db, Decision).count()
    }


@router.get("/api/search")
def search_knowledge(q: str, db: Session = Depends(get_db)):
    """
    Search across all knowledge entities (SOP, Deviation, Capa, Audit, Decision).
    Maps into structured cards for the UI renderer.
    """
    query = f"%{q}%"
    results = []

    # SOPs
    sops = _tenant_scoped_query(db, SOP).filter(
        or_(
            SOP.title.ilike(query),
            SOP.sop_number.ilike(query),
            SOP.department.ilike(query)
        )
    ).all()
    
    for sop in sops:
        has_content = bool(sop.current_version_id)
        results.append({
            "id": str(sop.id),
            "type": "sop",
            "typeLabel": "SOP",
            "metadata": f"{sop.sop_number or ''} · {sop.department or 'Allgemein'}",
            "matchPercent": 95,
            "title": sop.title or "Ohne Titel",
            "excerpt": "SOP Inhalt für KI Kontext indexiert..." if has_content else "Kein Inhalt verfügbar.",
            "badges": [
                {"label": "Aktiv" if sop.is_active else "Inaktiv", "color": "green" if sop.is_active else "gray"}
            ],
            "sourceIcon": "📄",
            "sourceColorClass": "source-sop"
        })

    # Deviations
    devs = _tenant_scoped_query(db, Deviation).filter(
        or_(
            Deviation.title.ilike(query),
            Deviation.deviation_number.ilike(query),
            Deviation.description_text.ilike(query)
        )
    ).all()

    for dev in devs:
        desc = dev.description_text or "Keine Beschreibung"
        results.append({
            "id": str(dev.id),
            "type": "deviation",
            "typeLabel": "Abweichung",
            "metadata": f"{dev.deviation_number or ''} · {dev.site or 'Allgemein'}",
            "matchPercent": 88,
            "title": dev.title or "Unbekannte Abweichung",
            "excerpt": (desc[:140] + '...') if len(desc) > 140 else desc,
            "badges": [
                {"label": dev.external_status or "Offen", "color": "green" if getattr(dev, 'external_status', '') == "closed" else "orange"},
                {"label": dev.impact_level or "Normal", "color": "red" if dev.impact_level == "high" else "gray"}
            ],
            "sourceIcon": "⚠",
            "sourceColorClass": "source-warning"
        })

    # CAPAs
    capas = _tenant_scoped_query(db, Capa).filter(
        or_(
            Capa.title.ilike(query),
            Capa.capa_number.ilike(query),
            Capa.action_text.ilike(query)
        )
    ).all()
    
    for capa in capas:
        desc = capa.action_text or "Keine Beschreibung"
        results.append({
            "id": str(capa.id),
            "type": "capa",
            "typeLabel": "CAPA",
            "metadata": f"{capa.capa_number or ''}",
            "matchPercent": 85,
            "title": capa.title or "Unbekannte CAPA",
            "excerpt": (desc[:140] + '...') if len(desc) > 140 else desc,
            "badges": [
                {"label": capa.external_status or "Offen", "color": "green" if getattr(capa, 'external_status', '') == "closed" else "orange"}
            ],
            "sourceIcon": "◆",
            "sourceColorClass": "source-warning"
        })

    # Audits
    audits = _tenant_scoped_query(db, AuditFinding).filter(
        or_(
            AuditFinding.finding_number.ilike(query),
            AuditFinding.finding_text.ilike(query),
            AuditFinding.question_text.ilike(query)
        )
    ).all()
    
    for aud in audits:
        desc = aud.finding_text or aud.question_text or "Keine Beschreibung"
        results.append({
            "id": str(aud.id),
            "type": "audit",
            "typeLabel": "Audit Finding",
            "metadata": f"{aud.finding_number or ''}",
            "matchPercent": 82,
            "title": aud.finding_number or "Unbekanntes Finding",
            "excerpt": (desc[:140] + '...') if len(desc) > 140 else desc,
            "badges": [
                {"label": aud.acceptance_status or 'Minor', "color": "blue"}
            ],
            "sourceIcon": "✓",
            "sourceColorClass": "source-audit"
        })

    # Decisions
    is_decision_query = q.lower() in ["decision", "decisions", "entscheidung", "entscheidungen"]
    decisions = _tenant_scoped_query(db, Decision).filter(
        or_(
            Decision.title.ilike(query),
            Decision.decision_number.ilike(query),
            Decision.decision_type.ilike(query),
            Decision.decision_statement.ilike(query),
            Decision.rationale_text.ilike(query),
            Decision.risk_assessment_text.ilike(query),
            Decision.final_conclusion.ilike(query),
            True if is_decision_query else False
        )
    ).all()
    
    for dec in decisions:
        desc = dec.decision_statement or dec.rationale_text or "Keine Beschreibung"
        results.append({
            "id": str(dec.id),
            "type": "decision",
            "typeLabel": "Decision",
            "metadata": f"{dec.decision_number or ''} · {dec.decision_type or 'Allgemein'}",
            "matchPercent": 80,
            "title": dec.title or dec.decision_number or "Unbekannte Entscheidung",
            "excerpt": (desc[:140] + '...') if len(desc) > 140 else desc,
            "badges": [],
            "sourceIcon": "❓",
            "sourceColorClass": "source-decision"
        })

    # Sort logic mimicking the frontend
    results.sort(key=lambda x: x["matchPercent"], reverse=True)
    
    return results

# ==========================================
# MANUAL LINKING ROUTES
# ==========================================

@router.post("/api/links")
def create_link(
    payload: LinkRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Create a manual link between two entities."""
    l_type = payload.link_type.lower()
    source_id = payload.source_id
    target_id = payload.target_id
    
    link_obj = None
    if l_type == "sop-deviation":
        link_obj = SopDeviationLink(id=uuid.uuid4(), tenant_id=FIXED_TENANT_ID, sop_id=source_id, deviation_id=target_id, rationale_text=payload.rationale_text)
    elif l_type == "deviation-capa":
        link_obj = DeviationCapaLink(id=uuid.uuid4(), tenant_id=FIXED_TENANT_ID, deviation_id=source_id, capa_id=target_id, rationale_text=payload.rationale_text)
    elif l_type == "capa-audit":
        link_obj = CapaAuditLink(id=uuid.uuid4(), tenant_id=FIXED_TENANT_ID, capa_id=source_id, audit_finding_id=target_id, rationale_text=payload.rationale_text)
    elif l_type == "audit-decision":
        link_obj = AuditDecisionLink(id=uuid.uuid4(), tenant_id=FIXED_TENANT_ID, audit_finding_id=source_id, decision_id=target_id, rationale_text=payload.rationale_text)
    elif l_type == "decision-sop":
        link_obj = DecisionSopLink(id=uuid.uuid4(), tenant_id=FIXED_TENANT_ID, decision_id=source_id, sop_id=target_id, rationale_text=payload.rationale_text)
    
    if not link_obj:
        raise HTTPException(status_code=400, detail=f"Unsupported link type: {l_type}")
        
    db.add(link_obj)
    db.commit()
    for entity_type, entity_id in entities_for_link(l_type, source_id, target_id):
        _schedule_semantic_job(background_tasks, entity_type, entity_id, job_type="link_reindex")
    return {"status": "success", "link_id": str(link_obj.id)}

@router.delete("/api/links/{link_type}/{link_id}")
def delete_link(link_type: str, link_id: UUID, db: Session = Depends(get_db)):
    """Delete a manual link."""
    l_type = link_type.lower()
    
    model_map = {
        "sop-deviation": SopDeviationLink,
        "deviation-capa": DeviationCapaLink,
        "capa-audit": CapaAuditLink,
        "audit-decision": AuditDecisionLink,
        "decision-sop": DecisionSopLink
    }
    
    if l_type not in model_map:
        raise HTTPException(status_code=400, detail=f"Unsupported link type: {l_type}")
        
    link = db.query(model_map[l_type]).filter(model_map[l_type].id == link_id).first()
    if not link:
        raise HTTPException(status_code=404, detail="Link not found")
        
    db.delete(link)
    db.commit()
    return {"status": "success"}

# ==========================================
# NORMALIZATION & BGE-M3 PREP (PLACEHOLDERS)
# ==========================================

@router.post("/api/normalization/unified-ingest")
def unified_ingest(payload: dict, db: Session = Depends(get_db)):
    """
    Placeholder for the unified normalization flow.
    Ensures all ingested content (Editor or Upload) follows one pipe.
    """
    return {"status": "stub", "message": "Normalization service ready for integration"}

@router.post("/api/semantic/reindex")
def semantic_reindex(
    payload: SemanticReindexRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Queue semantic reindex jobs (delta or full) for BGE-M3 + Qdrant indexing.
    """
    queued = []
    if payload.full_reindex:
        for sop in _tenant_scoped_query(db, SOP).all():
            SemanticPipelineService.enqueue_reindex("sop", sop.id, sop.current_version_id, "full_reindex")
            queued.append({"entity_type": "sop", "entity_id": str(sop.id)})
        for dev in _tenant_scoped_query(db, Deviation).all():
            SemanticPipelineService.enqueue_reindex("deviation", dev.id, None, "full_reindex")
            queued.append({"entity_type": "deviation", "entity_id": str(dev.id)})
        for capa in _tenant_scoped_query(db, Capa).all():
            SemanticPipelineService.enqueue_reindex("capa", capa.id, None, "full_reindex")
            queued.append({"entity_type": "capa", "entity_id": str(capa.id)})
        for audit in _tenant_scoped_query(db, AuditFinding).all():
            SemanticPipelineService.enqueue_reindex("audit_finding", audit.id, None, "full_reindex")
            queued.append({"entity_type": "audit_finding", "entity_id": str(audit.id)})
        for decision in _tenant_scoped_query(db, Decision).all():
            SemanticPipelineService.enqueue_reindex("decision", decision.id, None, "full_reindex")
            queued.append({"entity_type": "decision", "entity_id": str(decision.id)})

        def _drain_embedding_queue() -> None:
            while True:
                s = SessionLocal()
                try:
                    nxt = (
                        s.query(EmbeddingJob)
                        .filter(EmbeddingJob.status == "pending")
                        .order_by(asc(EmbeddingJob.created_at))
                        .first()
                    )
                    if not nxt:
                        return
                    jid = nxt.id
                finally:
                    s.close()
                try:
                    SemanticPipelineService.process_job(jid)
                except Exception as exc:
                    print(f"[semantic reindex] job {jid} failed: {exc}", flush=True)

        if queued:
            background_tasks.add_task(_drain_embedding_queue)
    else:
        if not payload.entity_type or not payload.entity_id:
            raise HTTPException(status_code=422, detail="entity_type and entity_id are required unless full_reindex=true.")
        normalized_type = payload.entity_type.strip().lower()
        if normalized_type not in ENTITY_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported entity_type '{payload.entity_type}'.")
        _schedule_semantic_job(background_tasks, normalized_type, payload.entity_id, payload.version_id, "manual_reindex")
        queued.append({"entity_type": normalized_type, "entity_id": str(payload.entity_id)})

    return {
        "status": "queued",
        "count": len(queued),
        "jobs": queued,
    }


@router.get("/api/semantic/suggestions", response_model=list[LinkSuggestionResponse])
def get_semantic_suggestions(
    entity_type: str = Query(...),
    entity_id: UUID = Query(...),
    db: Session = Depends(get_db),
):
    normalized_type = entity_type.strip().lower()
    if normalized_type not in ENTITY_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported entity_type '{entity_type}'.")
    return (
        db.query(AILinkSuggestion)
        .filter(
            AILinkSuggestion.source_entity_type == normalized_type,
            AILinkSuggestion.source_entity_id == entity_id,
        )
        .order_by(AILinkSuggestion.score.desc(), AILinkSuggestion.created_at.desc())
        .all()
    )


@router.post("/api/semantic/suggestions/{suggestion_id}/accept")
def accept_semantic_suggestion(suggestion_id: UUID, approved_by: str | None = None, db: Session = Depends(get_db)):
    suggestion = db.query(AILinkSuggestion).filter(AILinkSuggestion.id == suggestion_id).first()
    if not suggestion:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    SemanticPipelineService.accept_suggestion(db, suggestion, approved_by=approved_by)
    return {"status": "accepted", "id": str(suggestion_id)}


@router.post("/api/semantic/suggestions/{suggestion_id}/reject")
def reject_semantic_suggestion(suggestion_id: UUID, approved_by: str | None = None, db: Session = Depends(get_db)):
    suggestion = db.query(AILinkSuggestion).filter(AILinkSuggestion.id == suggestion_id).first()
    if not suggestion:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    SemanticPipelineService.reject_suggestion(db, suggestion, approved_by=approved_by)
    return {"status": "rejected", "id": str(suggestion_id)}


@router.get("/api/semantic/status", response_model=SemanticStatusResponse)
def get_semantic_status(entity_type: str = Query(...), entity_id: UUID = Query(...), db: Session = Depends(get_db)):
    normalized_type = entity_type.strip().lower()
    if normalized_type not in ENTITY_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported entity_type '{entity_type}'.")
    return SemanticPipelineService.get_entity_status(db, normalized_type, entity_id)


# ==========================================
# MAINTENANCE / RAG STATE
# ==========================================

@router.post("/api/semantic/maintenance/purge")
def purge_rag_state(
    recreate_collection: bool = Query(True),
    clear_embedding_jobs: bool = Query(True),
    clear_source_references: bool = Query(True),
    clear_link_suggestions: bool = Query(False),
):
    """
    [rag-maintenance] Wipe Qdrant + knowledge_chunks for a clean rebuild.

    Use this when the RAG index is stale or corrupted. SOP / business tables
    (sops, sop_versions, deviations, capas, audit_findings, decisions, links)
    are NEVER touched.
    """
    counts = SemanticPipelineService.purge_all_semantic_state(
        recreate_collection=recreate_collection,
        clear_embedding_jobs=clear_embedding_jobs,
        clear_source_references=clear_source_references,
        clear_link_suggestions=clear_link_suggestions,
    )
    return {"status": "purged", "counts": counts}


@router.post("/api/semantic/maintenance/rebuild")
def rebuild_rag_state(background_tasks: BackgroundTasks):
    """
    [rag-maintenance] Enqueue indexing jobs for every active entity, then drain the
    queue in the background so RAG retrieval rehydrates from scratch.
    """
    counts = SemanticPipelineService.queue_full_reindex()

    def _drain_embedding_queue():
        while True:
            s = SessionLocal()
            try:
                nxt = (
                    s.query(EmbeddingJob)
                    .filter(EmbeddingJob.status == "pending")
                    .order_by(asc(EmbeddingJob.created_at))
                    .first()
                )
                if not nxt:
                    return
                jid = nxt.id
            finally:
                s.close()
            try:
                SemanticPipelineService.process_job(jid)
            except Exception as exc:
                logger.warning("[rag-maintenance] job %s failed: %s", jid, exc)

    background_tasks.add_task(_drain_embedding_queue)
    return {"status": "rebuild_queued", "counts": counts}
