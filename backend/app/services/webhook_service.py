"""
Webhook-driven ingestion sync and entity lifecycle hooks.

Fetches entity snapshots from WEBHOOK_ENDPOINT_* URLs (existing .env) and
queues semantic reindex jobs so Qdrant stays aligned with the relational DB.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

import httpx

from ..database import SessionLocal
from ..models import AuditFinding, Capa, Decision, Deviation, SOP
from .semantic_jobs import schedule_entities, schedule_semantic_reindex
from .semantic_pipeline import ENTITY_TYPES, SemanticPipelineService
from .webhook_config import get_all_webhook_endpoints

logger = logging.getLogger(__name__)

_HEADERS = {"Accept": "application/json"}


def _parse_records(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("results", "data", "items", "sops"):
            val = data.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def _entity_id_from_record(record: dict) -> uuid.UUID | None:
    raw = record.get("id") or record.get("entity_id") or record.get("sop_id")
    if not raw:
        return None
    try:
        return uuid.UUID(str(raw))
    except ValueError:
        return None


def _version_id_from_record(record: dict) -> uuid.UUID | None:
    version = record.get("current_version") or {}
    raw = (
        record.get("version_id")
        or record.get("current_version_id")
        or (version.get("id") if isinstance(version, dict) else None)
    )
    if not raw:
        return None
    try:
        return uuid.UUID(str(raw))
    except ValueError:
        return None


async def fetch_webhook_records(cfg_url: str, timeout: float = 60.0) -> list[dict]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(cfg_url, headers=_HEADERS)
        resp.raise_for_status()
        return _parse_records(resp.json())


async def probe_webhook_endpoints() -> dict[str, Any]:
    """HTTP probe each configured webhook URL (read-only)."""
    results: list[dict[str, Any]] = []
    for cfg in get_all_webhook_endpoints():
        row: dict[str, Any] = {
            "entity_key": cfg.entity_key,
            "entity_type": cfg.entity_type,
            "url": cfg.url,
            "ok": False,
        }
        try:
            records = await fetch_webhook_records(cfg.url, timeout=30.0)
            row["ok"] = True
            row["record_count"] = len(records)
        except Exception as exc:
            row["error"] = str(exc)
        results.append(row)
    return {"endpoints": results, "all_ok": all(r.get("ok") for r in results)}


def _db_entity_exists(entity_type: str, entity_id: uuid.UUID) -> bool:
    db = SessionLocal()
    try:
        model_map = {
            "sop": SOP,
            "deviation": Deviation,
            "capa": Capa,
            "audit_finding": AuditFinding,
            "decision": Decision,
        }
        model = model_map.get(entity_type)
        if not model:
            return False
        return db.query(model.id).filter(model.id == entity_id).first() is not None
    finally:
        db.close()


async def sync_all_from_webhooks(*, process: bool = True) -> dict[str, Any]:
    """
    Pull all entity types from WEBHOOK_ENDPOINT_* URLs and queue semantic reindex
    for every record that exists in Postgres.
    """
    summary: dict[str, Any] = {"queued_jobs": [], "entities": {}, "skipped": 0}
    to_schedule: list[tuple[str, uuid.UUID, uuid.UUID | None]] = []

    for cfg in get_all_webhook_endpoints():
        entity_stats = {"fetched": 0, "queued": 0, "missing_in_db": 0}
        try:
            records = await fetch_webhook_records(cfg.url)
        except Exception as exc:
            entity_stats["error"] = str(exc)
            summary["entities"][cfg.entity_key] = entity_stats
            continue

        entity_stats["fetched"] = len(records)
        for rec in records:
            eid = _entity_id_from_record(rec)
            if not eid:
                summary["skipped"] += 1
                continue
            if not _db_entity_exists(cfg.entity_type, eid):
                entity_stats["missing_in_db"] += 1
                continue
            vid = _version_id_from_record(rec) if cfg.entity_type == "sop" else None
            to_schedule.append((cfg.entity_type, eid, vid))
            entity_stats["queued"] += 1

        summary["entities"][cfg.entity_key] = entity_stats

    if process and to_schedule:
        job_ids = schedule_entities(to_schedule, job_type="webhook_sync")
        summary["queued_jobs"] = job_ids
        summary["queued_count"] = len(job_ids)

    return summary


def handle_entity_event(
    event: str,
    entity_type: str,
    entity_id: uuid.UUID,
    version_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    """
    Runtime CRUD hook: create/update → reindex; delete → purge Qdrant + chunks.
    """
    normalized = entity_type.strip().lower()
    if normalized not in ENTITY_TYPES:
        raise ValueError(f"Unsupported entity_type: {entity_type}")

    ev = event.strip().lower()
    if ev == "deleted":
        SemanticPipelineService.purge_entity_artifacts(normalized, entity_id)
        from .rag_cache import invalidate_runtime_rag_cache

        invalidate_runtime_rag_cache()
        return {"status": "purged", "entity_type": normalized, "entity_id": str(entity_id)}

    if ev in ("created", "updated", "linked", "imported"):
        job_id = schedule_semantic_reindex(
            normalized,
            entity_id,
            version_id,
            job_type=f"webhook_{ev}",
            skip_unchanged_import=(ev != "updated"),
        )
        return {
            "status": "queued" if job_id else "skipped",
            "job_id": job_id,
            "entity_type": normalized,
            "entity_id": str(entity_id),
        }

    raise ValueError(f"Unsupported event: {event}")


def entities_for_link(link_type: str, source_id: uuid.UUID, target_id: uuid.UUID) -> list[tuple[str, uuid.UUID]]:
    """Map manual link types to entity pairs that should refresh in Qdrant."""
    lt = link_type.strip().lower()
    mapping: dict[str, list[tuple[str, uuid.UUID]]] = {
        "sop-deviation": [("sop", source_id), ("deviation", target_id)],
        "deviation-capa": [("deviation", source_id), ("capa", target_id)],
        "capa-audit": [("capa", source_id), ("audit_finding", target_id)],
        "audit-decision": [("audit_finding", source_id), ("decision", target_id)],
        "decision-sop": [("decision", source_id), ("sop", target_id)],
    }
    return mapping.get(lt, [])
