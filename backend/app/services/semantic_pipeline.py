import os
import uuid
import time
import hashlib
import logging
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer 
from sqlalchemy import and_, func, exists, not_, or_, select
from sqlalchemy.orm import Session

from ..database import SessionLocal
from .profile_detection_store import persist_profile_detection_for_sop_version
from ..models import (
    AILinkSuggestion,
    AuditDecisionLink,
    AuditFinding,
    Capa,
    CapaAuditLink,
    Decision,
    DecisionSopLink,
    Deviation,
    DeviationCapaLink,
    EmbeddingJob,
    KnowledgeChunk,
    SourceReference,
    SOP,
    SOPVersion,
    SopDeviationLink,
)
from retrieval.hybrid_retriever import invalidate_bm25_cache

logger = logging.getLogger(__name__)

BGE_M3_MODEL = "BAAI/bge-m3"
DEFAULT_COLLECTION = os.getenv("SEMANTIC_QDRANT_COLLECTION", "qa_semantic_chunks")
ENTITY_TYPES = {"sop", "deviation", "capa", "audit_finding", "decision"}
# Auto-accept only very strong suggestions; weaker (but valid) ones remain pending.
AUTO_ACCEPT_DELTA = float(os.getenv("SEMANTIC_AUTO_ACCEPT_DELTA", "0.05"))
LINK_RULES = {
    "sop": ("deviation", "sop-deviation", 0.63),
    "deviation": ("capa", "deviation-capa", 0.62),
    "capa": ("audit_finding", "capa-audit", 0.62),
    "audit_finding": ("decision", "audit-decision", 0.6),
    "decision": ("sop", "decision-sop", 0.64),
}

_embedder: Any | None = None
_qdrant: QdrantClient | None = None
_embedder_lock = threading.Lock()


def _resolve_hf_cache_dir() -> str:
    configured = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("EMBEDDING_HF_CACHE_DIR")
    if configured:
        cache_dir = Path(configured).expanduser().resolve()
    else:
        cache_dir = (Path(__file__).resolve().parents[3] / ".hf-cache").resolve()

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
    return str(cache_dir)


def _is_model_cached(cache_dir: str, model_name: str) -> bool:
    model_key = model_name.replace("/", "--")
    candidate_roots = [
        Path(cache_dir) / "hub" / f"models--{model_key}",
        Path(cache_dir) / f"models--{model_key}",
    ]
    for root in candidate_roots:
        snapshots_dir = root / "snapshots"
        if snapshots_dir.exists() and any(p.is_dir() for p in snapshots_dir.iterdir()):
            return True
    return False


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer
                cache_dir = _resolve_hf_cache_dir()
                local_only = _is_model_cached(cache_dir, BGE_M3_MODEL)
                if local_only:
                    print(f"[hf-cache] BGE-M3 loading from local cache: {cache_dir}", flush=True)
                else:
                    print(f"[hf-cache] BGE-M3 cache miss, attempting download into: {cache_dir}", flush=True)
                _embedder = SentenceTransformer(
                    BGE_M3_MODEL,
                    device=os.getenv("EMBEDDING_DEVICE", "cpu"),
                    cache_folder=cache_dir,
                    local_files_only=local_only,
                )
                print(f"[semantic-pipeline] BGE-M3 initialized once (cache={cache_dir}, local_only={local_only})")
    return _embedder


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        qdrant_url = os.getenv("QDRANT_URL")
        if not qdrant_url:
            raise RuntimeError("QDRANT_URL is not configured.")
        timeout_s = float(os.getenv("QDRANT_TIMEOUT_SECONDS", "30"))
        t0 = time.perf_counter()
        _qdrant = QdrantClient(
            url=qdrant_url,
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=timeout_s,
        )
        logger.info(
            "[startup-qdrant] client_initialized url_host=%s ms=%d",
            qdrant_url.split("//", 1)[-1].split("/", 1)[0],
            int((time.perf_counter() - t0) * 1000),
        )
    return _qdrant


def prewarm_runtime() -> None:
    """
    Worker startup hook: load heavy runtime once and probe embedding dimension.
    """
    embedder = _get_embedder()
    probe = embedder.encode(["warmup_probe"], normalize_embeddings=True)
    SemanticPipelineService._ensure_collection(len(probe[0]))
    _get_qdrant()


def _split_long_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def _extract_tiptap_sections(content_json: dict[str, Any] | None) -> list[tuple[str, str]]:
    if not isinstance(content_json, dict):
        return []
    current_section = "General"
    sections: dict[str, list[str]] = defaultdict(list)
    for node in content_json.get("content", []) or []:
        ntype = node.get("type")
        if ntype == "heading":
            texts = []
            for c in node.get("content", []) or []:
                if c.get("type") == "text" and c.get("text"):
                    texts.append(c["text"])
            heading = " ".join(texts).strip()
            if heading:
                current_section = heading
            continue
        texts = []
        for c in node.get("content", []) or []:
            if c.get("type") == "text" and c.get("text"):
                texts.append(c["text"])
        txt = " ".join(texts).strip()
        if txt:
            sections[current_section].append(txt)
    return [(name, "\n".join(lines).strip()) for name, lines in sections.items() if lines]


def _sections_content_fingerprint(sections: list[tuple[str, str]]) -> str:
    return hashlib.sha256(
        "\n\n".join([f"{name}\n{text}" for name, text in sections]).encode("utf-8", errors="ignore")
    ).hexdigest()

_entity_locks: dict[tuple[str, uuid.UUID], threading.Lock] = {}
_entity_locks_lock = threading.Lock()


def get_entity_lock(entity_type: str, entity_id: uuid.UUID) -> threading.Lock:
    key = (entity_type, entity_id)
    with _entity_locks_lock:
        if key not in _entity_locks:
            _entity_locks[key] = threading.Lock()
        return _entity_locks[key]


class SemanticPipelineService:
    @staticmethod
    def purge_all_semantic_state(
        recreate_collection: bool = True,
        clear_embedding_jobs: bool = True,
        clear_source_references: bool = True,
        clear_link_suggestions: bool = False,
    ) -> dict[str, int]:
        """
        [rag-maintenance] Wipe stale RAG state for a fresh rebuild.

        Removes (in this order):
          1. All Qdrant points in the active collection (drops + recreates collection)
          2. All knowledge_chunks rows
          3. embedding_jobs (optional)
          4. source_references (optional)
          5. ai_link_suggestions (optional)

        Returns a dict with counts of what was cleared.
        """
        counts = {
            "knowledge_chunks_deleted": 0,
            "embedding_jobs_deleted": 0,
            "source_references_deleted": 0,
            "ai_link_suggestions_deleted": 0,
            "sop_active_pipeline_job_id_reset": 0,
            "qdrant_collection_recreated": 0,
            "qdrant_points_deleted": 0,
        }

        try:
            client = _get_qdrant()
            try:
                stats = client.count(collection_name=DEFAULT_COLLECTION, exact=False)
                counts["qdrant_points_deleted"] = int(getattr(stats, "count", 0) or 0)
            except Exception:
                counts["qdrant_points_deleted"] = 0

            if recreate_collection:
                try:
                    if client.collection_exists(DEFAULT_COLLECTION):
                        client.delete_collection(DEFAULT_COLLECTION)
                        counts["qdrant_collection_recreated"] = 1
                        logger.info("[rag-maintenance] qdrant_collection_dropped name=%s", DEFAULT_COLLECTION)
                except Exception as exc:
                    logger.warning("[rag-maintenance] qdrant_drop_failed err=%s", exc)
            else:
                try:
                    client.delete(
                        collection_name=DEFAULT_COLLECTION,
                        points_selector=qmodels.FilterSelector(
                            filter=qmodels.Filter(must=[])
                        ),
                        wait=True,
                    )
                except Exception as exc:
                    logger.warning("[rag-maintenance] qdrant_purge_failed err=%s", exc)
        except Exception as exc:
            logger.warning("[rag-maintenance] qdrant_unavailable_skipping_vector_cleanup err=%s", exc)

        invalidate_bm25_cache(DEFAULT_COLLECTION)

        db = SessionLocal()
        try:
            counts["knowledge_chunks_deleted"] = int(
                db.query(KnowledgeChunk).delete(synchronize_session=False) or 0
            )
            if clear_source_references:
                counts["source_references_deleted"] = int(
                    db.query(SourceReference).delete(synchronize_session=False) or 0
                )
            if clear_embedding_jobs:
                counts["embedding_jobs_deleted"] = int(
                    db.query(EmbeddingJob).delete(synchronize_session=False) or 0
                )
            counts["sop_active_pipeline_job_id_reset"] = int(
                db.query(SOP).update(
                    {SOP.active_pipeline_job_id: None}, synchronize_session=False
                )
                or 0
            )
            if clear_link_suggestions:
                counts["ai_link_suggestions_deleted"] = int(
                    db.query(AILinkSuggestion).delete(synchronize_session=False) or 0
                )
            db.commit()
        finally:
            db.close()

        logger.info("[rag-maintenance] purge_done counts=%s", counts)
        return counts

    @staticmethod
    def queue_full_reindex() -> dict[str, int]:
        """
        [rag-maintenance] Enqueue fresh indexing jobs for every active entity so the
        embedding worker rebuilds Qdrant + knowledge_chunks from scratch.
        """
        queued = {"sop": 0, "deviation": 0, "capa": 0, "audit_finding": 0, "decision": 0}
        db = SessionLocal()
        try:
            for sop in db.query(SOP).filter(SOP.is_active == True).all():  # noqa: E712
                job_id = SemanticPipelineService.enqueue_reindex(
                    "sop", sop.id, sop.current_version_id, "manual_full_reindex"
                )
                if job_id:
                    queued["sop"] += 1
            for dev in db.query(Deviation).all():
                if SemanticPipelineService.enqueue_reindex("deviation", dev.id, None, "manual_full_reindex"):
                    queued["deviation"] += 1
            for capa in db.query(Capa).all():
                if SemanticPipelineService.enqueue_reindex("capa", capa.id, None, "manual_full_reindex"):
                    queued["capa"] += 1
            for af in db.query(AuditFinding).all():
                if SemanticPipelineService.enqueue_reindex("audit_finding", af.id, None, "manual_full_reindex"):
                    queued["audit_finding"] += 1
            for dec in db.query(Decision).all():
                if SemanticPipelineService.enqueue_reindex("decision", dec.id, None, "manual_full_reindex"):
                    queued["decision"] += 1
        finally:
            db.close()
        logger.info("[rag-maintenance] full_reindex_queued counts=%s", queued)
        return queued

    @staticmethod
    def purge_entity_artifacts(entity_type: str, entity_id: uuid.UUID) -> None:
        """
        Remove stale semantic artifacts for an entity from both Postgres chunks
        and Qdrant vectors. Safe to call after hard deletes.
        """
        db = SessionLocal()
        try:
            if entity_type == "sop":
                sop = db.query(SOP).filter(SOP.id == entity_id).first()
                if sop:
                    sop.active_pipeline_job_id = None
            db.query(KnowledgeChunk).filter(
                KnowledgeChunk.entity_type == entity_type,
                KnowledgeChunk.entity_id == entity_id,
            ).delete(synchronize_session=False)
            db.query(SourceReference).filter(
                SourceReference.entity_type == entity_type,
                SourceReference.entity_id == entity_id,
            ).delete(synchronize_session=False)
            db.commit()
        finally:
            db.close()

        try:
            client = _get_qdrant()
            client.delete(
                collection_name=DEFAULT_COLLECTION,
                wait=True,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="entity_id",
                                match=qmodels.MatchValue(value=str(entity_id)),
                            ),
                            qmodels.FieldCondition(
                                key="entity_type",
                                match=qmodels.MatchValue(value=entity_type),
                            ),
                        ]
                    )
                ),
            )
            invalidate_bm25_cache(DEFAULT_COLLECTION)
            try:
                from .rag_cache import invalidate_runtime_rag_cache

                invalidate_runtime_rag_cache()
            except Exception:
                pass
        except Exception as ex:
            print(f"[semantic-pipeline] purge warning for {entity_type} {entity_id}: {ex}", flush=True)

    @staticmethod
    def reconcile_stale_sop_chunks(db: Session) -> dict[str, int]:
        """
        [rag-retrieval] Delete knowledge_chunks that are not tied to an active SOP row
        with the current version id, purge Qdrant for affected entities, and queue
        reindex for still-active SOPs.

        Resilient: never raises if Qdrant is unreachable; just logs and skips the
        vector cleanup so backend startup continues.
        """
        reconcile_t0 = time.time()
        ok = exists(
            select(1)
            .select_from(SOP)
            .where(
                SOP.id == KnowledgeChunk.entity_id,
                SOP.is_active == True,  # noqa: E712
                or_(
                    SOP.current_version_id == KnowledgeChunk.entity_version_id,
                    and_(
                        SOP.current_version_id.is_(None),
                        KnowledgeChunk.entity_version_id.is_(None),
                    ),
                ),
            )
        ).correlate(KnowledgeChunk)

        stale_entity_ids = [
            row[0]
            for row in (
                db.query(KnowledgeChunk.entity_id)
                .filter(KnowledgeChunk.entity_type == "sop", not_(ok))
                .distinct()
                .all()
            )
            if row[0] is not None
        ]

        deleted = (
            db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.entity_type == "sop", not_(ok))
            .delete(synchronize_session=False)
        )
        db.commit()

        requeued = 0
        purged = 0
        try:
            client = _get_qdrant()
        except Exception as exc:
            print(f"[semantic-reconcile] Qdrant unavailable, skipping vector cleanup: {exc}", flush=True)
            client = None

        for eid in set(stale_entity_ids):
            sop = db.query(SOP).filter(SOP.id == eid).first()
            if sop and sop.is_active:
                if client:
                    try:
                        client.delete(
                            collection_name=DEFAULT_COLLECTION,
                            wait=True,
                            points_selector=qmodels.FilterSelector(
                                filter=qmodels.Filter(
                                    must=[
                                        qmodels.FieldCondition(
                                            key="entity_id",
                                            match=qmodels.MatchValue(value=str(eid)),
                                        ),
                                        qmodels.FieldCondition(
                                            key="entity_type",
                                            match=qmodels.MatchValue(value="sop"),
                                        ),
                                    ]
                                )
                            ),
                        )
                    except Exception as ex:
                        print(f"[semantic-reconcile] qdrant delete sop {eid}: {ex}", flush=True)
                SemanticPipelineService.enqueue_reindex(
                    "sop", eid, sop.current_version_id, "reconcile_stale_sop_chunks"
                )
                requeued += 1
            else:
                SemanticPipelineService.purge_entity_artifacts("sop", eid)
                purged += 1

        invalidate_bm25_cache(DEFAULT_COLLECTION)
        elapsed_ms = int((time.time() - reconcile_t0) * 1000)
        logger.info(
            "[rag-retrieval] reconcile_done deleted_chunks=%s distinct_stale_sops=%s queued=%s purged=%s ms=%s",
            deleted,
            len(set(stale_entity_ids)),
            requeued,
            purged,
            elapsed_ms,
        )
        return {
            "deleted_knowledge_chunks": int(deleted or 0),
            "reindex_queued": requeued,
            "purged": purged,
        }

    @staticmethod
    def _entity_exists(db: Session, entity_type: str, entity_id: uuid.UUID) -> bool:
        if entity_type == "sop":
            return db.query(SOP.id).filter(SOP.id == entity_id).first() is not None
        if entity_type == "deviation":
            return db.query(Deviation.id).filter(Deviation.id == entity_id).first() is not None
        if entity_type == "capa":
            return db.query(Capa.id).filter(Capa.id == entity_id).first() is not None
        if entity_type == "audit_finding":
            return db.query(AuditFinding.id).filter(AuditFinding.id == entity_id).first() is not None
        if entity_type == "decision":
            return db.query(Decision.id).filter(Decision.id == entity_id).first() is not None
        return False

    @staticmethod
    def _pipeline_stage_names() -> tuple[str, ...]:
        return (
            "chunking_status",
            "embeddings_status",
            "qdrant_status",
            "nlp_status",
            "semantic_linking_status",
        )

    @staticmethod
    def _cancel_embedding_job(db: Session, job: EmbeddingJob) -> None:
        job.status = "cancelled"
        job.finished_at = datetime.utcnow()
        for fn in SemanticPipelineService._pipeline_stage_names():
            cur = getattr(job, fn)
            if cur in ("pending", "processing"):
                setattr(job, fn, "cancelled")
        db.commit()
        logger.info("[pipeline] job cancelled id=%s entity=%s:%s", job.id, job.entity_type, job.entity_id)

    @staticmethod
    def _fail_embedding_job(db: Session, job: EmbeddingJob, message: str) -> None:
        job.status = "failed"
        job.finished_at = datetime.utcnow()
        job.error_message = (message or "")[:2000]
        for fn in SemanticPipelineService._pipeline_stage_names():
            cur = getattr(job, fn)
            if cur in ("pending", "processing"):
                setattr(job, fn, "failed")
        db.commit()

    @staticmethod
    def _sop_pipeline_superseded(db: Session, job: EmbeddingJob) -> bool:
        db.expire_all()
        sop = db.query(SOP).filter(SOP.id == job.entity_id).first()
        if not sop:
            return True
        aid = sop.active_pipeline_job_id
        return aid is not None and aid != job.id

    @staticmethod
    def _process_sop_pipeline(db: Session, job: EmbeddingJob) -> bool:
        job_id = job.id
        entity_id = job.entity_id
        version_id = job.version_id

        def superseded() -> bool:
            if SemanticPipelineService._sop_pipeline_superseded(db, job):
                SemanticPipelineService._cancel_embedding_job(db, job)
                return True
            return False

        logger.info("[sop-pipeline] chunking started job=%s sop=%s version=%s", job_id, entity_id, version_id)
        job.chunking_status = "processing"
        db.commit()

        sections, resolved_version = SemanticPipelineService._normalize_entity(db, "sop", entity_id, version_id)
        if not sections:
            logger.info("[sop-pipeline] no sections job=%s", job_id)
            for fn in SemanticPipelineService._pipeline_stage_names():
                setattr(job, fn, "skipped")
            job.status = "completed"
            job.finished_at = datetime.utcnow()
            db.commit()
            return False

        live_fp = _sections_content_fingerprint(sections)
        if job.enqueued_content_hash and job.enqueued_content_hash != live_fp:
            logger.info(
                "[sop-pipeline] content drift since enqueue job=%s — requeue latest",
                job_id,
            )
            SemanticPipelineService._cancel_embedding_job(db, job)
            SemanticPipelineService.enqueue_reindex(
                "sop", entity_id, resolved_version or version_id, job.job_type
            )
            return False

        chunk_scope = db.query(KnowledgeChunk).filter(
            KnowledgeChunk.entity_type == "sop",
            KnowledgeChunk.entity_id == entity_id,
        )
        if resolved_version:
            chunk_scope = chunk_scope.filter(KnowledgeChunk.entity_version_id == resolved_version)
        chunk_exists = chunk_scope.with_entities(KnowledgeChunk.id).first() is not None

        if resolved_version:
            version_row = db.query(SOPVersion).filter(SOPVersion.id == resolved_version).first()
            if version_row:
                meta = version_row.metadata_json if isinstance(version_row.metadata_json, dict) else {}
                if chunk_exists and meta.get("_semantic_hash") == live_fp:
                    logger.info("[sop-pipeline] unchanged skip job=%s", job_id)
                    job.chunking_status = "completed"
                    job.embeddings_status = "completed"
                    job.qdrant_status = "completed"
                    job.nlp_status = "skipped"
                    job.semantic_linking_status = "skipped"
                    job.status = "completed"
                    job.finished_at = datetime.utcnow()
                    db.commit()
                    return False

        db.query(KnowledgeChunk).filter(
            KnowledgeChunk.entity_type == "sop",
            KnowledgeChunk.entity_id == entity_id,
        ).delete(synchronize_session=False)
        db.query(SourceReference).filter(
            SourceReference.entity_type == "sop",
            SourceReference.entity_id == entity_id,
        ).delete(synchronize_session=False)
        db.commit()

        display = SemanticPipelineService._entity_rag_fields(db, "sop", entity_id)
        doc_type_norm = SemanticPipelineService._doc_type_for_entity("sop")
        ref = (display.get("ref_number") or "").strip() or str(entity_id)
        title = (display.get("title") or "").strip() or "Untitled"
        rag_meta = {
            "doc_type": doc_type_norm,
            "entity_type": "sop",
            "ref_number": ref,
            "source_id": str(entity_id),
            "title": title,
            "department": display.get("department") or "",
            "status": display.get("status") or "",
        }
        if display.get("sop_number"):
            rag_meta["sop_number"] = display["sop_number"]

        chunk_rows: list[tuple[str, str, int]] = []
        order = 0
        for section_name, section_text in sections:
            for text in _split_long_text(section_text):
                chunk_rows.append((section_name, text, order))
                order += 1

        for section_name, text, ord_ in chunk_rows:
            meta = {
                "entity_type": "sop",
                "entity_id": str(entity_id),
                "version_id": str(resolved_version) if resolved_version else None,
                "section_name": section_name,
                "chunk_index": ord_,
                "content_hash": live_fp,
                "rag_ready": False,
                **{k: v for k, v in rag_meta.items() if v is not None and v != ""},
            }
            db.add(
                KnowledgeChunk(
                    tenant_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                    entity_type="sop",
                    entity_id=entity_id,
                    entity_version_id=resolved_version,
                    chunk_type="semantic_section",
                    chunk_text=text,
                    chunk_order=ord_,
                    metadata_json=meta,
                )
            )
        db.commit()
        job.chunking_status = "completed"
        db.commit()
        logger.info("[sop-pipeline] chunking completed job=%s chunks=%s", job_id, len(chunk_rows))

        if superseded():
            return False

        logger.info("[sop-pipeline] embeddings started job=%s", job_id)
        job.embeddings_status = "processing"
        db.commit()

        rows = (
            db.query(KnowledgeChunk)
            .filter(
                KnowledgeChunk.entity_type == "sop",
                KnowledgeChunk.entity_id == entity_id,
                KnowledgeChunk.entity_version_id == resolved_version,
            )
            .order_by(KnowledgeChunk.chunk_order.asc())
            .all()
        )
        if not rows:
            SemanticPipelineService._fail_embedding_job(db, job, "Chunks missing after chunking stage")
            logger.error("[sop-pipeline] embeddings aborted — no rows job=%s", job_id)
            return False

        embedder = _get_embedder()
        example_vec = embedder.encode(["dimension_probe"], normalize_embeddings=True)[0]
        SemanticPipelineService._ensure_collection(len(example_vec))
        texts = [r.chunk_text for r in rows]
        embedding_vectors = embedder.encode(
            texts,
            normalize_embeddings=True,
            batch_size=min(16, max(1, len(texts))),
        )
        job.embeddings_status = "completed"
        db.commit()
        logger.info("[sop-pipeline] embeddings completed job=%s", job_id)

        if superseded():
            return False

        job.qdrant_status = "processing"
        db.commit()
        logger.info("[sop-pipeline] qdrant ingestion started job=%s", job_id)

        client = _get_qdrant()
        try:
            client.delete(
                collection_name=DEFAULT_COLLECTION,
                wait=True,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="entity_id",
                                match=qmodels.MatchValue(value=str(entity_id)),
                            ),
                            qmodels.FieldCondition(
                                key="entity_type",
                                match=qmodels.MatchValue(value="sop"),
                            ),
                        ]
                    )
                ),
            )
            invalidate_bm25_cache(DEFAULT_COLLECTION)
        except Exception as ex:
            logger.warning("[sop-pipeline] qdrant delete non-fatal job=%s err=%s", job_id, ex)

        points: list[qmodels.PointStruct] = []
        for row, emb_arr in zip(rows, embedding_vectors):
            ord_ = row.chunk_order
            md = row.metadata_json if isinstance(row.metadata_json, dict) else {}
            section_name = md.get("section_name", "General")
            text = row.chunk_text
            emb = emb_arr.tolist()
            qid = str(uuid.uuid4())
            pl = {
                "entity_type": "sop",
                "entity_id": str(entity_id),
                "version_id": str(resolved_version) if resolved_version else None,
                "section_name": section_name,
                "chunk_index": ord_,
                "embedding_model": BGE_M3_MODEL,
                "page_content": text,
                "chunk_text": text,
                "ref_number": ref,
                "title": title,
                "department": rag_meta.get("department", ""),
                "status": rag_meta.get("status", ""),
                "metadata": rag_meta,
                "rag_ready": True,
            }
            points.append(qmodels.PointStruct(id=qid, vector=emb, payload=pl))

        if points:
            client.upsert(collection_name=DEFAULT_COLLECTION, points=points, wait=True)
            invalidate_bm25_cache(DEFAULT_COLLECTION)

        for row in rows:
            md = dict(row.metadata_json) if isinstance(row.metadata_json, dict) else {}
            md["rag_ready"] = True
            md["embedding_model"] = BGE_M3_MODEL
            row.metadata_json = md

        if resolved_version:
            version_row = db.query(SOPVersion).filter(SOPVersion.id == resolved_version).first()
            if version_row:
                meta = version_row.metadata_json if isinstance(version_row.metadata_json, dict) else {}
                meta["_semantic_hash"] = live_fp
                version_row.metadata_json = meta

        db.commit()
        job.qdrant_status = "completed"
        db.commit()
        logger.info("[sop-pipeline] qdrant ingestion completed job=%s points=%s", job_id, len(points))

        if superseded():
            return False

        job.nlp_status = "processing"
        db.commit()
        logger.info("[sop-pipeline] nlp pipeline started job=%s", job_id)
        version_for_nlp = db.query(SOPVersion).filter(SOPVersion.id == resolved_version).first()
        if version_for_nlp:
            persist_profile_detection_for_sop_version(db, version_for_nlp)
        job.nlp_status = "completed"
        db.commit()
        logger.info("[sop-pipeline] nlp pipeline completed job=%s", job_id)

        if superseded():
            return False

        job.semantic_linking_status = "processing"
        db.commit()
        logger.info("[sop-pipeline] semantic linking started job=%s", job_id)
        SemanticPipelineService._generate_suggestions(db, "sop", entity_id)
        job.semantic_linking_status = "completed"
        db.commit()
        logger.info("[sop-pipeline] semantic linking completed job=%s", job_id)

        return True

    @staticmethod
    def enqueue_reindex(entity_type: str, entity_id: uuid.UUID, version_id: uuid.UUID | None = None, job_type: str = "entity_reindex") -> uuid.UUID | None:
        db = SessionLocal()
        try:
            if entity_type not in ENTITY_TYPES:
                return None

            if entity_type == "sop":
                sop = db.query(SOP).filter(SOP.id == entity_id).first()
                if not sop:
                    return None
                resolved_vid = version_id or sop.current_version_id
                version = None
                if resolved_vid:
                    version = (
                        db.query(SOPVersion)
                        .filter(SOPVersion.id == resolved_vid, SOPVersion.sop_id == entity_id)
                        .first()
                    )
                fingerprint: str | None = None
                if version:
                    secs, _ = SemanticPipelineService._normalize_entity(db, "sop", entity_id, resolved_vid)
                    if secs:
                        fingerprint = _sections_content_fingerprint(secs)

                reuse = (
                    db.query(EmbeddingJob)
                    .filter(
                        EmbeddingJob.entity_type == "sop",
                        EmbeddingJob.entity_id == entity_id,
                        EmbeddingJob.status.in_(("pending", "processing")),
                        EmbeddingJob.version_id == resolved_vid,
                        EmbeddingJob.enqueued_content_hash == fingerprint,
                    )
                    .order_by(EmbeddingJob.created_at.desc())
                    .first()
                )
                if reuse and fingerprint is not None:
                    sop.active_pipeline_job_id = reuse.id
                    db.commit()
                    logger.info("[pipeline] reuse job=%s sop=%s", reuse.id, entity_id)
                    return reuse.id

                stale_pending = (
                    db.query(EmbeddingJob)
                    .filter(
                        EmbeddingJob.entity_type == "sop",
                        EmbeddingJob.entity_id == entity_id,
                        EmbeddingJob.status == "pending",
                    )
                    .all()
                )
                for j in stale_pending:
                    j.status = "cancelled"
                    j.finished_at = datetime.utcnow()
                    for fn in SemanticPipelineService._pipeline_stage_names():
                        if getattr(j, fn) in ("pending", "processing"):
                            setattr(j, fn, "cancelled")

                job = EmbeddingJob(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    version_id=resolved_vid,
                    job_type=job_type,
                    status="pending",
                    enqueued_content_hash=fingerprint,
                )
                db.add(job)
                db.flush()
                sop.active_pipeline_job_id = job.id
                db.commit()
                db.refresh(job)
                logger.info("[pipeline] enqueued job=%s sop=%s version=%s", job.id, entity_id, resolved_vid)
                return job.id

            existing = (
                db.query(EmbeddingJob)
                .filter(
                    EmbeddingJob.entity_type == entity_type,
                    EmbeddingJob.entity_id == entity_id,
                    EmbeddingJob.version_id == version_id,
                    EmbeddingJob.status.in_(("pending", "processing")),
                )
                .order_by(EmbeddingJob.created_at.desc())
                .first()
            )
            if existing:
                return existing.id
            job = EmbeddingJob(
                entity_type=entity_type,
                entity_id=entity_id,
                version_id=version_id,
                job_type=job_type,
                status="pending",
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            return job.id
        finally:
            db.close()

    @staticmethod
    def process_job(job_id: uuid.UUID):
        db = SessionLocal()
        job = None
        try:
            job = db.query(EmbeddingJob).filter(EmbeddingJob.id == job_id).first()
            if not job:
                return
            if job.status not in ("pending", "processing"):
                return

            # Acquire lock for this specific entity to serialize concurrent indexing jobs on the same entity
            lock = get_entity_lock(job.entity_type, job.entity_id)
            with lock:
                db.rollback()
                job = db.query(EmbeddingJob).filter(EmbeddingJob.id == job_id).first()
                if not job or job.status not in ("pending", "processing"):
                    return

                # Check if this SOP job is superseded before starting
                if job.entity_type == "sop" and SemanticPipelineService._sop_pipeline_superseded(db, job):
                    SemanticPipelineService._cancel_embedding_job(db, job)
                    return

                if job.status == "pending":
                    job.status = "processing"
                    job.started_at = datetime.utcnow()
                    db.commit()

                logger.info("[semantic-job] started job=%s entity=%s:%s", job_id, job.entity_type, job.entity_id)

                if job.entity_type == "sop":
                    did_reindex = SemanticPipelineService._process_sop_pipeline(db, job)
                else:
                    did_reindex = SemanticPipelineService._index_entity(
                        db, job.entity_type, job.entity_id, job.version_id
                    )
                    for fn in SemanticPipelineService._pipeline_stage_names():
                        setattr(job, fn, "completed")
                    if did_reindex:
                        SemanticPipelineService._generate_suggestions(db, job.entity_type, job.entity_id)

                db.refresh(job)
                if job.status in ("cancelled", "failed"):
                    logger.info("[semantic-job] finished job=%s status=%s", job_id, job.status)
                    return

                job.status = "completed"
                job.finished_at = datetime.utcnow()
                job.error_message = None
                db.commit()
                try:
                    from .rag_cache import invalidate_runtime_rag_cache

                    invalidate_runtime_rag_cache()
                except Exception as cache_exc:
                    logger.warning("[semantic-job] rag cache invalidate failed: %s", cache_exc)
                logger.info("[semantic-job] completed job=%s did_reindex=%s", job_id, did_reindex)
        except Exception as exc:
            if job:
                try:
                    db.rollback()
                    job = db.query(EmbeddingJob).filter(EmbeddingJob.id == job_id).first()
                    if job:
                        SemanticPipelineService._fail_embedding_job(db, job, str(exc))
                except Exception as fail_exc:
                    logger.warning("Failed to mark job as failed: %s", fail_exc)
            logger.exception("[semantic-job] failed job=%s", job_id)
            raise
        finally:
            db.close()

    @staticmethod
    def _ensure_collection(dim: int):
        client = _get_qdrant()
        if not client.collection_exists(DEFAULT_COLLECTION):
            client.create_collection(
                collection_name=DEFAULT_COLLECTION,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            )
        # Cloud Qdrant can require payload indexes for filtered query performance/validity.
        for _field, _typ in (
            ("entity_type", qmodels.PayloadSchemaType.KEYWORD),
            ("entity_id", qmodels.PayloadSchemaType.KEYWORD),
        ):
            try:
                client.create_payload_index(
                    collection_name=DEFAULT_COLLECTION,
                    field_name=_field,
                    field_schema=_typ,
                )
            except Exception:
                # Index may already exist; keep indexing flow idempotent.
                pass

    @staticmethod
    def _normalize_entity(db: Session, entity_type: str, entity_id: uuid.UUID, version_id: uuid.UUID | None):
        if entity_type == "sop":
            sop = db.query(SOP).filter(SOP.id == entity_id).first()
            if not sop:
                return [], None
            version = None
            if version_id:
                version = db.query(SOPVersion).filter(SOPVersion.id == version_id, SOPVersion.sop_id == sop.id).first()
            if not version:
                version = db.query(SOPVersion).filter(SOPVersion.id == sop.current_version_id).first() if sop.current_version_id else None
            if not version:
                return [], None
            meta = version.metadata_json or {}
            sop_meta = meta.get("sopMetadata", {}) if isinstance(meta, dict) else {}
            sections = _extract_tiptap_sections(version.content_json if isinstance(version.content_json, dict) else {})
            if not sections:
                normalized = "\n".join(
                    [
                        f"title: {sop.title or ''}",
                        f"purpose: {sop_meta.get('purpose', '')}",
                        f"scope: {sop_meta.get('scope', '')}",
                        f"responsibilities: {sop_meta.get('responsibilities', '')}",
                        f"procedure: {sop_meta.get('procedure', '')}",
                        f"documentation: {sop_meta.get('documentation', '')}",
                    ]
                ).strip()
                sections = [("General", normalized)]
            return sections, version.id

        if entity_type == "deviation":
            row = db.query(Deviation).filter(Deviation.id == entity_id).first()
            if not row:
                return [], None
            text = "\n".join(
                [
                    f"title: {row.title or ''}",
                    f"description: {row.description_text or ''}",
                    f"root_cause: {row.root_cause_text or ''}",
                    f"category: {row.category or ''}",
                    f"impact_level: {row.impact_level or ''}",
                ]
            ).strip()
            return [("Deviation", text)], None

        if entity_type == "capa":
            row = db.query(Capa).filter(Capa.id == entity_id).first()
            if not row:
                return [], None
            text = "\n".join(
                [
                    f"title: {row.title or ''}",
                    f"action: {row.action_text or ''}",
                    f"effectiveness: {row.effectiveness_text or ''}",
                ]
            ).strip()
            return [("CAPA", text)], None

        if entity_type == "audit_finding":
            row = db.query(AuditFinding).filter(AuditFinding.id == entity_id).first()
            if not row:
                return [], None
            text = "\n".join(
                [
                    f"question: {row.question_text or ''}",
                    f"finding: {row.finding_text or ''}",
                    f"response: {row.response_text or ''}",
                ]
            ).strip()
            return [("Audit Finding", text)], None

        if entity_type == "decision":
            row = db.query(Decision).filter(Decision.id == entity_id).first()
            if not row:
                return [], None
            text = "\n".join(
                [
                    f"title: {row.title or ''}",
                    f"decision_statement: {row.decision_statement or ''}",
                    f"rationale: {row.rationale_text or ''}",
                    f"risk_assessment: {row.risk_assessment_text or ''}",
                    f"final_conclusion: {row.final_conclusion or ''}",
                ]
            ).strip()
            return [("Decision", text)], None
        return [], None

    @staticmethod
    def _doc_type_for_entity(entity_type: str) -> str:
        m = {
            "sop": "sop",
            "deviation": "deviation",
            "capa": "capa",
            "audit_finding": "audit",
            "decision": "decision",
        }
        return m.get(entity_type, entity_type or "")

    @staticmethod
    def _entity_rag_fields(db: Session, entity_type: str, entity_id: uuid.UUID) -> dict:
        if entity_type == "sop":
            sop = db.query(SOP).filter(SOP.id == entity_id).first()
            if not sop:
                return {}
            st = None
            if sop.current_version_id:
                st = (
                    db.query(SOPVersion)
                    .filter(SOPVersion.id == sop.current_version_id)
                    .first()
                )
            return {
                "ref_number": sop.sop_number or "",
                "title": sop.title or "",
                "sop_number": sop.sop_number or "",
                "department": sop.department or "",
                "status": (st.external_status if st else None) or "",
            }
        if entity_type == "deviation":
            row = db.query(Deviation).filter(Deviation.id == entity_id).first()
            if not row:
                return {}
            return {
                "ref_number": row.deviation_number or "",
                "title": row.title or "",
                "department": row.site or row.category or "",
                "status": row.external_status or "",
            }
        if entity_type == "capa":
            row = db.query(Capa).filter(Capa.id == entity_id).first()
            if not row:
                return {}
            return {
                "ref_number": row.capa_number or "",
                "title": row.title or "",
                "department": row.owner_name or "",
                "status": row.external_status or "",
            }
        if entity_type == "audit_finding":
            row = db.query(AuditFinding).filter(AuditFinding.id == entity_id).first()
            if not row:
                return {}
            ref = row.finding_number or row.audit_number or str(entity_id)[:8]
            return {
                "ref_number": str(ref) if ref else str(entity_id)[:8],
                "title": (row.finding_text or row.question_text or "Audit finding")[:255],
                "department": row.authority or row.site or "",
                "status": row.acceptance_status or "",
            }
        if entity_type == "decision":
            row = db.query(Decision).filter(Decision.id == entity_id).first()
            if not row:
                return {}
            return {
                "ref_number": (row.decision_number or str(entity_id)[:8]) or "",
                "title": row.title or "",
                "department": row.decision_type or "",
                "status": (row.decision_type or row.decided_by_role) or "",
            }
        return {}

    @staticmethod
    def _index_entity(
        db: Session,
        entity_type: str,
        entity_id: uuid.UUID,
        version_id: uuid.UUID | None = None,
    ) -> bool:
        t0 = time.perf_counter()
        if entity_type == "sop":
            logger.warning("_index_entity called for SOP — use staged pipeline (ignored)")
            return False
        sections, resolved_version = SemanticPipelineService._normalize_entity(db, entity_type, entity_id, version_id)
        if not sections:
            return False
        t_norm = time.perf_counter()

        content_fingerprint = hashlib.sha256(
            "\n\n".join([f"{name}\n{text}" for name, text in sections]).encode("utf-8", errors="ignore")
        ).hexdigest()
        chunk_scope = db.query(KnowledgeChunk).filter(
            KnowledgeChunk.entity_type == entity_type,
            KnowledgeChunk.entity_id == entity_id,
        )
        if resolved_version:
            chunk_scope = chunk_scope.filter(KnowledgeChunk.entity_version_id == resolved_version)
        chunk_exists = chunk_scope.with_entities(KnowledgeChunk.id).first() is not None

        # Non-SOP entities use chunk metadata hash for idempotent reindexing.
        if chunk_exists:
            latest_chunk = (
                chunk_scope.order_by(KnowledgeChunk.created_at.desc())
                .with_entities(KnowledgeChunk.metadata_json)
                .first()
            )
            latest_hash = None
            if latest_chunk and isinstance(latest_chunk[0], dict):
                latest_hash = latest_chunk[0].get("content_hash")
            if latest_hash == content_fingerprint:
                print(
                    f"[semantic-pipeline] Skip unchanged {entity_type} {entity_id} "
                    f"(normalize={int((t_norm - t0)*1000)}ms)",
                    flush=True,
                )
                return False

        delete_query = db.query(KnowledgeChunk).filter(
            KnowledgeChunk.entity_type == entity_type,
            KnowledgeChunk.entity_id == entity_id,
        )
        delete_query.delete(synchronize_session=False)
        db.query(SourceReference).filter(
            SourceReference.entity_type == entity_type,
            SourceReference.entity_id == entity_id,
        ).delete(synchronize_session=False)
        db.commit()
        t_delete = time.perf_counter()

        embedder = _get_embedder()
        example_vec = embedder.encode(["dimension_probe"], normalize_embeddings=True)[0]
        SemanticPipelineService._ensure_collection(len(example_vec))
        client = _get_qdrant()
        # Remove prior Qdrant points for this entity so orphan vectors cannot drift from knowledge_chunks
        try:
            client.delete(
                collection_name=DEFAULT_COLLECTION,
                wait=True,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="entity_id",
                                match=qmodels.MatchValue(value=str(entity_id)),
                            ),
                            qmodels.FieldCondition(
                                key="entity_type",
                                match=qmodels.MatchValue(value=entity_type),
                            ),
                        ]
                    )
                ),
            )
            invalidate_bm25_cache(DEFAULT_COLLECTION)
        except Exception as ex:
            print(f"[semantic-pipeline] Qdrant delete (entity scope) non-fatal: {ex}")

        display = SemanticPipelineService._entity_rag_fields(db, entity_type, entity_id)
        doc_type_norm = SemanticPipelineService._doc_type_for_entity(entity_type)
        ref = (display.get("ref_number") or "").strip() or str(entity_id)
        title = (display.get("title") or "").strip() or "Untitled"
        rag_meta = {
            "doc_type": doc_type_norm,
            "entity_type": entity_type,
            "ref_number": ref,
            "source_id": str(entity_id),
            "title": title,
            "department": display.get("department") or "",
            "status": display.get("status") or "",
        }
        if display.get("sop_number"):
            rag_meta["sop_number"] = display["sop_number"]

        points = []
        chunk_order = 0
        chunk_rows: list[tuple[str, str, int]] = []
        for section_name, section_text in sections:
            for text in _split_long_text(section_text):
                chunk_rows.append((section_name, text, chunk_order))
                chunk_order += 1

        embedding_vectors = embedder.encode(
            [row[1] for row in chunk_rows],
            normalize_embeddings=True,
            batch_size=min(16, max(1, len(chunk_rows))),
        )
        t_embed = time.perf_counter()

        for (section_name, text, order), emb_arr in zip(chunk_rows, embedding_vectors):
            emb = emb_arr.tolist()
            chunk = KnowledgeChunk(
                tenant_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                entity_type=entity_type,
                entity_id=entity_id,
                entity_version_id=resolved_version,
                chunk_type="semantic_section",
                chunk_text=text,
                chunk_order=order,
                metadata_json={
                    "entity_type": entity_type,
                    "entity_id": str(entity_id),
                    "version_id": str(resolved_version) if resolved_version else None,
                    "section_name": section_name,
                    "chunk_index": order,
                    "embedding_model": BGE_M3_MODEL,
                    "content_hash": content_fingerprint,
                    "rag_ready": True,
                    **{k: v for k, v in rag_meta.items() if v is not None and v != ""},
                },
            )
            db.add(chunk)
            qid = str(uuid.uuid4())
            pl = {
                "entity_type": entity_type,
                "entity_id": str(entity_id),
                "version_id": str(resolved_version) if resolved_version else None,
                "section_name": section_name,
                "chunk_index": order,
                "embedding_model": BGE_M3_MODEL,
                "page_content": text,
                "chunk_text": text,
                "ref_number": ref,
                "title": title,
                "department": rag_meta.get("department", ""),
                "status": rag_meta.get("status", ""),
                "metadata": rag_meta,
                "rag_ready": True,
            }
            points.append(
                qmodels.PointStruct(
                    id=qid,
                    vector=emb,
                    payload=pl,
                )
            )
        print(
            f"[semantic-pipeline] Created {len(points)} chunks for {entity_type} {entity_id} "
            f"(normalize={int((t_norm-t0)*1000)}ms, delete={int((t_delete-t_norm)*1000)}ms, embed={int((t_embed-t_delete)*1000)}ms)",
            flush=True,
        )
        db.commit()
        t_db = time.perf_counter()
        if points:
            print(f"[semantic-pipeline] Upserting {len(points)} points to Qdrant ({DEFAULT_COLLECTION})", flush=True)
            client.upsert(collection_name=DEFAULT_COLLECTION, points=points, wait=True)
            invalidate_bm25_cache(DEFAULT_COLLECTION)
            t_upsert = time.perf_counter()
            print(
                f"[semantic-pipeline] Qdrant upsert complete for {entity_type} {entity_id} "
                f"(db={int((t_db-t_embed)*1000)}ms, upsert={int((t_upsert-t_db)*1000)}ms, total={int((t_upsert-t0)*1000)}ms)",
                flush=True,
            )
        return True

    @staticmethod
    def _generate_suggestions(db: Session, entity_type: str, entity_id: uuid.UUID):
        t0 = time.perf_counter()
        if entity_type not in LINK_RULES:
            return
        target_type, link_type, threshold = LINK_RULES[entity_type]
        auto_accept_threshold = min(0.99, threshold + max(0.0, AUTO_ACCEPT_DELTA))
        source_chunks = db.query(KnowledgeChunk).filter(
            KnowledgeChunk.entity_type == entity_type,
            KnowledgeChunk.entity_id == entity_id,
        ).all()
        if not source_chunks:
            return

        embedder = _get_embedder()
        client = _get_qdrant()
        entity_scores: dict[str, float] = {}
        source_texts = [c.chunk_text for c in source_chunks[:6]]
        source_vectors = embedder.encode(
            source_texts,
            normalize_embeddings=True,
            batch_size=min(16, max(1, len(source_texts))),
        )
        for vec_arr in source_vectors:
            vec = vec_arr.tolist()
            filt = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="entity_type", match=qmodels.MatchValue(value=target_type)),
                ]
            )
            try:
                if hasattr(client, "search"):
                    hits = client.search(
                        collection_name=DEFAULT_COLLECTION,
                        query_vector=vec,
                        query_filter=filt,
                        limit=8,
                    )
                else:
                    result = client.query_points(
                        collection_name=DEFAULT_COLLECTION,
                        query=vec,
                        query_filter=filt,
                        limit=8,
                        with_payload=True,
                        with_vectors=False,
                    )
                    hits = result.points
            except Exception as ex:
                print(f"[semantic-pipeline] Suggestion search failed for {entity_type} {entity_id}: {ex}", flush=True)
                continue
            for hit in hits:
                target_id = str(hit.payload.get("entity_id"))
                if not target_id:
                    continue
                score = float(hit.score)
                entity_scores[target_id] = max(score, entity_scores.get(target_id, 0.0))

        top = sorted(entity_scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for target_id, score in top:
            if score < threshold:
                continue
            try:
                target_uuid = uuid.UUID(target_id)
                if not SemanticPipelineService._entity_exists(db, target_type, target_uuid):
                    continue
                if SemanticPipelineService._already_linked(db, link_type, entity_id, target_uuid):
                    continue
                suggestion = db.query(AILinkSuggestion).filter(
                    AILinkSuggestion.source_entity_type == entity_type,
                    AILinkSuggestion.source_entity_id == entity_id,
                    AILinkSuggestion.target_entity_type == target_type,
                    AILinkSuggestion.target_entity_id == target_uuid,
                    AILinkSuggestion.suggested_link_type == link_type,
                ).order_by(AILinkSuggestion.created_at.desc()).first()

                if suggestion is None:
                    suggestion = AILinkSuggestion(
                        source_entity_type=entity_type,
                        source_entity_id=entity_id,
                        target_entity_type=target_type,
                        target_entity_id=target_uuid,
                        suggested_link_type=link_type,
                        score=score,
                        reason=f"Semantic similarity ({BGE_M3_MODEL}) score {score:.3f} exceeded threshold {threshold:.2f}.",
                        status="pending",
                    )
                    db.add(suggestion)
                    db.flush()
                elif suggestion.status == "pending":
                    # Keep pending suggestion metadata fresh with the latest score.
                    suggestion.score = max(float(suggestion.score or 0.0), score)
                    suggestion.reason = (
                        f"Semantic similarity ({BGE_M3_MODEL}) score {float(suggestion.score):.3f} "
                        f"exceeded threshold {threshold:.2f}."
                    )

                # Auto-bridge high-confidence suggestions into real link tables.
                # Lower-confidence suggestions remain pending for manual review.
                if suggestion.status == "pending" and score >= auto_accept_threshold:
                    SemanticPipelineService.accept_suggestion(
                        db,
                        suggestion,
                        approved_by="semantic-auto-accept",
                    )
            except Exception as ex:
                db.rollback()
                print(f"[semantic-pipeline] Suggestion upsert failed for {entity_type} {entity_id}: {ex}", flush=True)
        elapsed = int((time.perf_counter() - t0) * 1000)
        print(
            f"[semantic-pipeline] Generated {len(top)} potential suggestions for {entity_type} {entity_id} "
            f"(search+linking={elapsed}ms)",
            flush=True,
        )
        db.commit()

    @staticmethod
    def _already_linked(db: Session, link_type: str, source_id: uuid.UUID, target_id: uuid.UUID) -> bool:
        if link_type == "sop-deviation":
            return db.query(SopDeviationLink).filter(SopDeviationLink.sop_id == source_id, SopDeviationLink.deviation_id == target_id).first() is not None
        if link_type == "deviation-capa":
            return db.query(DeviationCapaLink).filter(DeviationCapaLink.deviation_id == source_id, DeviationCapaLink.capa_id == target_id).first() is not None
        if link_type == "capa-audit":
            return db.query(CapaAuditLink).filter(CapaAuditLink.capa_id == source_id, CapaAuditLink.audit_finding_id == target_id).first() is not None
        if link_type == "audit-decision":
            return db.query(AuditDecisionLink).filter(AuditDecisionLink.audit_finding_id == source_id, AuditDecisionLink.decision_id == target_id).first() is not None
        if link_type == "decision-sop":
            return db.query(DecisionSopLink).filter(DecisionSopLink.decision_id == source_id, DecisionSopLink.sop_id == target_id).first() is not None
        return False

    @staticmethod
    def accept_suggestion(db: Session, suggestion: AILinkSuggestion, approved_by: str | None = None):
        if suggestion.status != "pending":
            return
        link_type = suggestion.suggested_link_type
        sid = suggestion.source_entity_id
        tid = suggestion.target_entity_id
        tenant_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
        if not SemanticPipelineService._already_linked(db, link_type, sid, tid):
            if link_type == "sop-deviation":
                db.add(SopDeviationLink(tenant_id=tenant_id, sop_id=sid, deviation_id=tid, link_reason="ai_suggestion", confidence_score=suggestion.score, rationale_text=suggestion.reason))
            elif link_type == "deviation-capa":
                db.add(DeviationCapaLink(tenant_id=tenant_id, deviation_id=sid, capa_id=tid, link_reason="ai_suggestion", confidence_score=suggestion.score, rationale_text=suggestion.reason))
            elif link_type == "capa-audit":
                db.add(CapaAuditLink(tenant_id=tenant_id, capa_id=sid, audit_finding_id=tid, link_reason="ai_suggestion", confidence_score=suggestion.score, rationale_text=suggestion.reason))
            elif link_type == "audit-decision":
                db.add(AuditDecisionLink(tenant_id=tenant_id, audit_finding_id=sid, decision_id=tid, link_reason="ai_suggestion", confidence_score=suggestion.score, rationale_text=suggestion.reason))
            elif link_type == "decision-sop":
                db.add(DecisionSopLink(tenant_id=tenant_id, decision_id=sid, sop_id=tid, link_reason="ai_suggestion", confidence_score=suggestion.score, rationale_text=suggestion.reason))
        suggestion.status = "accepted"
        suggestion.approved_by = approved_by
        suggestion.approved_at = datetime.utcnow()
        db.commit()
        print(f"[semantic-pipeline] AUTO-ACCEPTED suggestion {suggestion.id} ({link_type}) from {sid} to {tid}", flush=True)

    @staticmethod
    def reject_suggestion(db: Session, suggestion: AILinkSuggestion, approved_by: str | None = None):
        if suggestion.status != "pending":
            return
        suggestion.status = "rejected"
        suggestion.approved_by = approved_by
        suggestion.approved_at = datetime.utcnow()
        db.commit()

    @staticmethod
    def get_entity_status(db: Session, entity_type: str, entity_id: uuid.UUID) -> dict[str, Any]:
        latest_job = (
            db.query(EmbeddingJob)
            .filter(EmbeddingJob.entity_type == entity_type, EmbeddingJob.entity_id == entity_id)
            .order_by(EmbeddingJob.created_at.desc())
            .first()
        )
        counts = dict(
            db.query(AILinkSuggestion.status, func.count(AILinkSuggestion.id))
            .filter(AILinkSuggestion.source_entity_type == entity_type, AILinkSuggestion.source_entity_id == entity_id)
            .group_by(AILinkSuggestion.status)
            .all()
        )
        active_job_id = None
        if entity_type == "sop":
            sop = db.query(SOP).filter(SOP.id == entity_id).first()
            if sop:
                active_job_id = sop.active_pipeline_job_id
        out: dict[str, Any] = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "latest_job_id": latest_job.id if latest_job else None,
            "latest_job_status": latest_job.status if latest_job else None,
            "latest_job_error": latest_job.error_message if latest_job else None,
            "latest_job_finished_at": latest_job.finished_at if latest_job else None,
            "active_pipeline_job_id": active_job_id,
            "chunking_status": None,
            "embeddings_status": None,
            "qdrant_status": None,
            "nlp_status": None,
            "semantic_linking_status": None,
            "pending_suggestions": int(counts.get("pending", 0)),
            "accepted_suggestions": int(counts.get("accepted", 0)),
            "rejected_suggestions": int(counts.get("rejected", 0)),
        }
        if latest_job:
            out["chunking_status"] = latest_job.chunking_status
            out["embeddings_status"] = latest_job.embeddings_status
            out["qdrant_status"] = latest_job.qdrant_status
            out["nlp_status"] = latest_job.nlp_status
            out["semantic_linking_status"] = latest_job.semantic_linking_status
        return out
