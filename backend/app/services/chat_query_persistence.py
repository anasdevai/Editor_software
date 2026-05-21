"""Persist /api/ai/query exchanges into existing chat_sessions / chat_messages tables."""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import ChatMessage, ChatSession

logger = logging.getLogger(__name__)

MAX_TITLE = 500
MAX_ASSISTANT_CONTEXT_JSON = 18_000


def _parse_session_uuid(raw: object) -> UUID | None:
    if raw is None or raw == "":
        return None
    try:
        return UUID(str(raw).strip())
    except (ValueError, AttributeError, TypeError):
        return None


def _parse_user_uuid(user_id: str | None) -> UUID | None:
    if user_id is None:
        return None
    try:
        return UUID(str(user_id).strip())
    except (ValueError, AttributeError, TypeError):
        return None


def _sop_snapshot(assistant_context: dict | None) -> dict | None:
    if not isinstance(assistant_context, dict):
        return None
    cur = assistant_context.get("current_sop")
    if not isinstance(cur, dict):
        return None
    snap: dict = {}
    id_val = cur.get("id") or cur.get("documentId")
    if id_val is not None and str(id_val).strip():
        snap["sop_id"] = str(id_val).strip()
    ver = cur.get("current_version_id") or cur.get("version_id") or cur.get("versionId")
    if ver is not None and str(ver).strip():
        snap["sop_version_id"] = str(ver).strip()
    sn = cur.get("sop_number") or cur.get("sopNumber")
    if sn:
        snap["sop_number"] = str(sn)
    title = cur.get("title")
    if title:
        snap["title"] = str(title)[:300]
    return snap or None


def _compact_assistant_context(ctx: dict | None) -> dict | None:
    """Shrink assistant_context for JSON columns (retain structure, trim heavy strings)."""
    if not isinstance(ctx, dict) or not ctx:
        return None
    d = copy.deepcopy(ctx)
    excerpt = d.get("editor_excerpt")
    if isinstance(excerpt, str) and len(excerpt) > 2500:
        d["editor_excerpt"] = excerpt[:2500] + "…"
    raw = json.dumps(d, default=str)
    if len(raw) <= MAX_ASSISTANT_CONTEXT_JSON:
        return d
    d.pop("editor_excerpt", None)
    d["linked_context"] = {}
    d["opened_tabs"] = []
    raw2 = json.dumps(d, default=str)
    if len(raw2) > MAX_ASSISTANT_CONTEXT_JSON:
        return {"route": d.get("route"), "current_document_id": d.get("current_document_id"), "_truncated": True}
    return d


def _build_retrieval_metadata(response: dict, llm_provider: str, llm_model: str) -> dict:
    stats = response.get("retrieval_stats")
    if not isinstance(stats, dict):
        stats = {}
    return {
        "retrieval_stats": stats,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
    }


def _build_metadata_snapshot(
    assistant_context: dict | None,
    *,
    surface: str | None,
    route: str | None,
    category: str | None,
) -> dict | None:
    snap: dict = {}
    sop = _sop_snapshot(assistant_context)
    if sop:
        snap.update(sop)
    if surface:
        snap["surface"] = str(surface)[:120]
    if route:
        snap["route"] = str(route)[:500]
    if category:
        snap["category"] = str(category)[:100]
    compact = _compact_assistant_context(assistant_context)
    if compact:
        snap["assistant_context"] = compact
    return snap or None


def persist_chat_query_exchange(
    *,
    user_id: str | None,
    client_session_id: str | None,
    collection_name: str,
    category: str | None,
    question: str,
    response: dict,
    assistant_context: dict | None,
    llm_provider: str,
    llm_model: str,
    surface: str | None = None,
    route: str | None = None,
) -> dict:
    """
    Insert one user row and one assistant row. Swallows all errors (logs only).

    ``user_id`` may be None for anonymous clients (``chat_sessions.user_id`` NULL).

    Returns optional keys to merge into the API JSON: session_id, message_id
    (assistant message id).
    """
    uid = _parse_user_uuid(user_id)
    if user_id is not None and uid is None:
        logger.warning("[chat-history-error] persistence skipped: invalid user_id=%r", user_id)
        return {}

    db: Session = SessionLocal()
    try:
        coll = (collection_name or "").strip() or "docs_sops"
        title_hint = (question or "").strip().replace("\n", " ")[:MAX_TITLE] or "Chat"
        cat_filter = (str(category).strip()[:100] if category else None) or None

        sid = _parse_session_uuid(client_session_id)
        session_row = None
        reused = False
        if sid is not None:
            q = (
                db.query(ChatSession)
                .filter(
                    ChatSession.id == sid,
                    ChatSession.is_active == True,  # noqa: E712
                )
            )
            if uid is not None:
                q = q.filter(ChatSession.user_id == uid)
            else:
                q = q.filter(ChatSession.user_id.is_(None))
            session_row = q.first()
            if session_row is not None:
                reused = True

        if session_row is None:
            session_row = ChatSession(
                user_id=uid,
                title=title_hint,
                collection_name=coll,
                is_active=True,
            )
            if sid is not None:
                session_row.id = sid
            db.add(session_row)
            db.flush()
            logger.info(
                "[chat-history-session-create] user_id=%s session_id=%s title=%s collection=%s",
                uid or "anon",
                session_row.id,
                title_hint[:80],
                coll,
            )
        else:
            logger.info(
                "[chat-history-session-reuse] user_id=%s session_id=%s reused=%s",
                uid or "anon",
                session_row.id,
                reused,
            )

        if not session_row.title or not str(session_row.title).strip():
            session_row.title = title_hint

        meta_snap = _build_metadata_snapshot(
            assistant_context,
            surface=surface,
            route=route,
            category=category,
        )
        retrieval_meta = _build_retrieval_metadata(response, llm_provider, llm_model)
        answer = str(response.get("answer") or "")
        try:
            from chatbot.rag.rag_chain import sanitize_user_facing_answer

            answer = sanitize_user_facing_answer(answer)
        except Exception:
            pass
        citations = response.get("citations")
        trace_meta = {
            "surface": (surface or "")[:120] or None,
            "route": (route or "")[:500] or None,
        }

        pair_t0 = datetime.now(timezone.utc)
        pair_t1 = pair_t0 + timedelta(milliseconds=1)
        user_msg = ChatMessage(
            session_id=session_row.id,
            role="user",
            content=(question or "").strip() or "(empty)",
            citations=None,
            retrieval_metadata=None,
            metadata_snapshot=meta_snap,
            action_metadata=trace_meta,
            category_filter=cat_filter,
            created_at=pair_t0,
        )
        asst_msg = ChatMessage(
            session_id=session_row.id,
            role="assistant",
            content=answer if answer.strip() else "(empty)",
            citations=citations if citations is not None else None,
            retrieval_metadata=retrieval_meta,
            metadata_snapshot=meta_snap,
            action_metadata=trace_meta,
            category_filter=cat_filter,
            created_at=pair_t1,
        )
        db.add(user_msg)
        db.add(asst_msg)
        db.flush()
        logger.info(
            "[chat-history-user-save] session_id=%s message_id=%s role=user qlen=%s",
            session_row.id,
            user_msg.id,
            len((question or "").strip()),
        )
        logger.info(
            "[chat-history-assistant-save] session_id=%s message_id=%s role=assistant alen=%s citations=%s",
            session_row.id,
            asst_msg.id,
            len(answer),
            len(citations) if isinstance(citations, list) else 0,
        )
        db.commit()
        db.refresh(asst_msg)

        return {
            "session_id": str(session_row.id),
            "message_id": str(asst_msg.id),
        }
    except Exception as exc:
        db.rollback()
        logger.warning("[chat-history-error] chat query persistence failed (non-fatal): %s", exc, exc_info=True)
        return {}
    finally:
        db.close()
