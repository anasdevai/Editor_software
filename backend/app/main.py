"""
Cybrain QS API entrypoint.

Startup is split into three phases with timing diagnostics:

  [startup-db]      → schema check + idempotent performance indexes
  [startup-qdrant]  → vector store connectivity probe (non-blocking)
  [startup-rag]     → SmartRAG / BGE-M3 prewarm and stale-chunk reconcile

Heavy initialisation runs inside a background daemon thread so the HTTP
server reaches "Application startup complete" within milliseconds and
endpoints like /api/health respond immediately.
"""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from .database import Base, engine, SessionLocal
from .routes import router
from .public_routes import public_router
from .ai_routes import ai_router, CHATBOT_USE_LOCAL_DB, _get_smart_rag_chain
from .auth_routes import router as auth_router
from .chat_history_routes import router as chat_history_router
from .profile_routes import router as profile_router
from .client_profile_routes import router as client_profile_router
from .webhook_routes import webhook_router
from .agent_routes import agent_router
from .services.semantic_pipeline import SemanticPipelineService
from .services.webhook_config import validate_webhook_configuration

logger = logging.getLogger("cybrain.startup")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(os.getenv("STARTUP_LOG_LEVEL", "INFO").upper())


PERFORMANCE_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS idx_sops_is_active_updated ON sops (is_active, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_sops_client ON sops (client_name, is_active)",
    "CREATE INDEX IF NOT EXISTS idx_sops_client_category ON sops (client_name, category, is_active)",
    "CREATE INDEX IF NOT EXISTS idx_sops_client_family ON sops (client_name, document_family, is_active)",
    "CREATE INDEX IF NOT EXISTS idx_sop_versions_sop_created ON sop_versions (sop_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_sop_versions_sop_version ON sop_versions (sop_id, version_number)",
    "CREATE INDEX IF NOT EXISTS idx_sop_deviation_links_sop ON sop_deviation_links (sop_id)",
    "CREATE INDEX IF NOT EXISTS idx_sop_deviation_links_dev ON sop_deviation_links (deviation_id)",
    "CREATE INDEX IF NOT EXISTS idx_deviation_capa_links_dev ON deviation_capa_links (deviation_id)",
    "CREATE INDEX IF NOT EXISTS idx_deviation_capa_links_capa ON deviation_capa_links (capa_id)",
    "CREATE INDEX IF NOT EXISTS idx_capa_audit_links_capa ON capa_audit_links (capa_id)",
    "CREATE INDEX IF NOT EXISTS idx_capa_audit_links_audit ON capa_audit_links (audit_finding_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_decision_links_audit ON audit_decision_links (audit_finding_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_decision_links_decision ON audit_decision_links (decision_id)",
    "CREATE INDEX IF NOT EXISTS idx_decision_sop_links_decision ON decision_sop_links (decision_id)",
    "CREATE INDEX IF NOT EXISTS idx_decision_sop_links_sop ON decision_sop_links (sop_id)",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_entity ON knowledge_chunks (entity_type, entity_id)",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_entity_version ON knowledge_chunks (entity_type, entity_id, entity_version_id)",
    "CREATE INDEX IF NOT EXISTS idx_ai_link_suggestions_source_status ON ai_link_suggestions (source_entity_type, source_entity_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_ai_link_suggestions_target_status ON ai_link_suggestions (target_entity_type, target_entity_id, status)",
    "CREATE INDEX IF NOT EXISTS ix_profile_detections_sop_version_active ON profile_detections (sop_id, sop_version_id, is_active)",
    "CREATE INDEX IF NOT EXISTS ix_profile_detections_source_hash ON profile_detections (source_hash)",
    "CREATE INDEX IF NOT EXISTS idx_sop_detected_parameters_sop ON sop_detected_parameters (sop_id)",
    "CREATE INDEX IF NOT EXISTS idx_profile_history_events_profile ON profile_history_events (client_profile_id)",
    "CREATE INDEX IF NOT EXISTS idx_profile_history_events_created ON profile_history_events (created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_sop_generation_templates_client ON sop_generation_templates (tenant_id, client_name, created_at DESC)",
)

SCHEMA_COMPAT_ALTER_STATEMENTS = (
    ("client_id", "VARCHAR(120)"),
    ("client_name", "VARCHAR(255)"),
    ("category", "VARCHAR(120)"),
    ("document_family", "VARCHAR(160)"),
)


def _bootstrap_database_schema() -> None:
    """[startup-db] Ensure ORM-managed tables exist and apply tuning indexes."""
    t0 = time.perf_counter()
    Base.metadata.create_all(bind=engine)
    schema_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("[startup-db] schema_create_all_ms=%d", schema_ms)

    t1 = time.perf_counter()
    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
            dialect = engine.dialect.name
            existing_sop_columns: set[str] = set()
            if dialect == "sqlite":
                existing_sop_columns = {row[1] for row in conn.execute(text("PRAGMA table_info(sops)")).fetchall()}
            for column_name, column_type in SCHEMA_COMPAT_ALTER_STATEMENTS:
                if dialect == "sqlite" and column_name in existing_sop_columns:
                    continue
                stmt = (
                    f"ALTER TABLE sops ADD COLUMN {column_name} {column_type}"
                    if dialect == "sqlite"
                    else f"ALTER TABLE sops ADD COLUMN IF NOT EXISTS {column_name} {column_type}"
                )
                try:
                    conn.execute(text(stmt))
                    existing_sop_columns.add(column_name)
                except Exception as alter_exc:
                    logger.debug("[startup-db] schema compat skipped stmt=%s err=%s", stmt, alter_exc)
            for stmt in PERFORMANCE_INDEX_STATEMENTS:
                conn.execute(text(stmt))
    except Exception as exc:  # pragma: no cover - surfaced in logs
        logger.error("[startup-db] index ensure failed: %s", exc)
        return
    index_ms = int((time.perf_counter() - t1) * 1000)
    logger.info(
        "[startup-db] indexes_ensured count=%d ms=%d total_ms=%d",
        len(PERFORMANCE_INDEX_STATEMENTS),
        index_ms,
        schema_ms + index_ms,
    )


def _bootstrap_qdrant_probe() -> None:
    """[startup-qdrant] Optional connectivity ping; never blocks application startup."""
    if CHATBOT_USE_LOCAL_DB:
        logger.info("[startup-qdrant] skipped (CHATBOT_USE_LOCAL_DB=true)")
        return
    if not os.getenv("QDRANT_URL"):
        logger.info("[startup-qdrant] skipped (QDRANT_URL not configured)")
        return
    t0 = time.perf_counter()
    try:
        from .services.semantic_pipeline import _get_qdrant  # local import to avoid heavy chain at boot
        client = _get_qdrant()
        info = client.get_collections()
        latency_ms = int((time.perf_counter() - t0) * 1000)
        try:
            names = [c.name for c in info.collections]
        except AttributeError:
            names = []
        logger.info(
            "[startup-qdrant] connect_ok latency_ms=%d collections=%s",
            latency_ms,
            names[:8],
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning("[startup-qdrant] connect_failed latency_ms=%d err=%s", latency_ms, exc)


def _bootstrap_rag_runtime() -> None:
    """[startup-rag] Warm BGE-M3 and reconcile stale SOP chunks once on boot."""
    if CHATBOT_USE_LOCAL_DB:
        logger.info("[startup-rag] skipped (CHATBOT_USE_LOCAL_DB=true)")
        return

    t0 = time.perf_counter()
    try:
        _get_smart_rag_chain()
        warm_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("[startup-rag] prewarm_ok ms=%d", warm_ms)
    except Exception as exc:
        warm_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning("[startup-rag] prewarm_skipped ms=%d err=%s", warm_ms, exc)

    if os.getenv("SEMANTIC_RECONCILE_ON_STARTUP", "true").strip().lower() != "true":
        logger.info("[startup-rag] reconcile_skipped (SEMANTIC_RECONCILE_ON_STARTUP=false)")
        return

    t1 = time.perf_counter()
    db = SessionLocal()
    try:
        result = SemanticPipelineService.reconcile_stale_sop_chunks(db)
        reconcile_ms = int((time.perf_counter() - t1) * 1000)
        logger.info("[startup-rag] reconcile_done ms=%d result=%s", reconcile_ms, result)
    except Exception as exc:
        reconcile_ms = int((time.perf_counter() - t1) * 1000)
        logger.warning("[startup-rag] reconcile_skipped ms=%d err=%s", reconcile_ms, exc)
    finally:
        db.close()

    wh = validate_webhook_configuration()
    logger.info(
        "[startup-webhook] complete=%s endpoints=%d api_base=%s",
        wh.get("complete"),
        len(wh.get("endpoints") or []),
        wh.get("api_base_url"),
    )


def _spawn_background_bootstrap() -> threading.Thread:
    def _runner() -> None:
        boot_t0 = time.perf_counter()
        logger.info("[startup] background_bootstrap starting")
        _bootstrap_qdrant_probe()
        _bootstrap_rag_runtime()
        total_ms = int((time.perf_counter() - boot_t0) * 1000)
        logger.info("[startup] background_bootstrap completed total_ms=%d", total_ms)

    thread = threading.Thread(target=_runner, name="cybrain-bootstrap", daemon=True)
    thread.start()
    return thread


@asynccontextmanager
async def lifespan(app: FastAPI):
    overall_t0 = time.perf_counter()
    logger.info("[startup] lifespan begin pid=%s", os.getpid())
    _bootstrap_database_schema()
    _spawn_background_bootstrap()
    ready_ms = int((time.perf_counter() - overall_t0) * 1000)
    logger.info("[startup] http_ready_ms=%d", ready_ms)
    try:
        yield
    finally:
        logger.info("[shutdown] lifespan end")


app = FastAPI(
    title="Cybrain QS API",
    description="SOP Editor + Stage 1 Public Chatbot Data Provisioning API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173" , 
    "http://65.21.244.158",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(public_router)
app.include_router(ai_router)
app.include_router(chat_history_router)
app.include_router(auth_router)
app.include_router(profile_router)
app.include_router(client_profile_router)
app.include_router(webhook_router)
app.include_router(agent_router)


@app.middleware("http")
async def api_latency_middleware(request, call_next):
    """[api-latency] Tag every request with timing for slow-query investigations."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    response.headers["X-Process-Time-ms"] = str(elapsed_ms)
    if elapsed_ms >= int(os.getenv("API_SLOW_LOG_MS", "1500")):
        logger.warning(
            "[slow-query] %s %s status=%s ms=%d",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
    return response


@app.get("/", tags=["Root"])
def root():
    return {
        "status": "ok",
        "message": "Cybrain QS API is running",
        "version": "1.0.0",
        "docs": "/api/docs",
    }


@app.get("/api/ready", tags=["Root"])
def readiness():
    """Lightweight readiness probe — does not depend on the RAG runtime."""
    return {"status": "ready", "pid": os.getpid()}
