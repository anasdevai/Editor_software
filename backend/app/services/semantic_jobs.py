"""
Shared async semantic reindex scheduler (CRUD, webhooks, imports, links).
"""
from __future__ import annotations

import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

from ..database import SessionLocal
from ..models import SOPVersion
from .semantic_pipeline import SemanticPipelineService

logger = logging.getLogger(__name__)

SEMANTIC_WORKER_THREADS = max(1, int(os.getenv("SEMANTIC_WORKER_THREADS", "2")))
_executor = ThreadPoolExecutor(max_workers=SEMANTIC_WORKER_THREADS, thread_name_prefix="semantic-job")


def schedule_semantic_reindex(
    entity_type: str,
    entity_id: uuid.UUID,
    version_id: uuid.UUID | None = None,
    job_type: str = "entity_reindex",
    *,
    skip_unchanged_import: bool = True,
) -> str | None:
    """
    Enqueue and process a semantic job in the background thread pool.
    Returns job_id when queued, else None.
    """
    if entity_type == "sop" and version_id and skip_unchanged_import:
        db = SessionLocal()
        try:
            version = (
                db.query(SOPVersion)
                .filter(SOPVersion.id == version_id, SOPVersion.sop_id == entity_id)
                .first()
            )
            if version and isinstance(version.metadata_json, dict):
                from ..utils.tiptap_text import extract_plain_text_from_tiptap

                plain_text = extract_plain_text_from_tiptap(version.content_json)
                if plain_text:
                    import hashlib

                    content_hash = hashlib.sha256(
                        plain_text.encode("utf-8", errors="ignore")
                    ).hexdigest()
                    import_hash = version.metadata_json.get("_import_context_hash")
                    if import_hash == content_hash:
                        logger.info(
                            "[semantic-job] skipped unchanged sop %s v=%s",
                            entity_id,
                            version_id,
                        )
                        return None
        finally:
            db.close()

    job_id = SemanticPipelineService.enqueue_reindex(
        entity_type=entity_type,
        entity_id=entity_id,
        version_id=version_id,
        job_type=job_type,
    )
    if not job_id:
        return None

    def _run() -> None:
        try:
            SemanticPipelineService.process_job(job_id)
        except Exception as exc:
            logger.exception("[semantic-job] job %s failed: %s", job_id, exc)

    _executor.submit(_run)
    logger.info(
        "[semantic-job] scheduled %s for %s:%s job=%s",
        job_type,
        entity_type,
        entity_id,
        job_id,
    )
    return str(job_id)


def schedule_entities(entities: list[tuple[str, uuid.UUID, uuid.UUID | None]], job_type: str = "webhook_sync") -> list[str]:
    """Queue reindex for multiple entities; returns job ids."""
    job_ids: list[str] = []
    for entity_type, entity_id, version_id in entities:
        jid = schedule_semantic_reindex(
            entity_type,
            entity_id,
            version_id,
            job_type=job_type,
            skip_unchanged_import=False,
        )
        if jid:
            job_ids.append(jid)
    return job_ids
