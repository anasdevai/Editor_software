"""
Compatibility shim.

The chatbot code is being consolidated under `backend/chatbot/`.
Keep this module so existing imports (`app.ai_routes`) continue to work.
"""

from chatbot.routes.ai_routes import *  # noqa: F401,F403
from html import escape
import re
import os
import math
import time
import threading
import asyncio
import uuid
import logging
from typing import Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ValidationError
from sqlalchemy import or_
from langchain_core.messages import AIMessage

from action.prompts import (
    AI_ACTION_PROMPT_SOURCE_FILE,
    IMPROVE_REWRITE_NO_RAG_CONTEXT,
    build_analyze_prompt,
    build_gap_check_prompt,
    build_improve_prompt,
    build_rewrite_prompt,
    build_section_only_improve_retry_prompt,
    build_section_only_rewrite_retry_prompt,
    build_summarize_prompt,
    extract_register_slice_from_output,
    is_traceability_register_block,
    resolve_edit_scope,
    violates_section_only_scope,
)
from action.runtime import create_action_runtime
from action.utils import (
    ACTION_LLM_EMPTY_RETRY_SUFFIX,
    format_chunks,
    normalize_action_input_text,
    parse_with_retry,
    truncate_prompt_for_llm,
)
from schemas.sop_actions import ActionRequest, GapCheckResponse, ImproveResponse, RewriteResponse
from .schemas import AIActionRequest, AIActionResponse
from .database import SessionLocal
from .auth_routes import get_current_user_optional
from .models import (
    SOP,
    SOPVersion,
    Deviation,
    Capa,
    AuditFinding,
    Decision,
    User,
    SopDeviationLink,
    DeviationCapaLink,
    CapaAuditLink,
    AuditDecisionLink,
    DecisionSopLink,
    SOPDetectedParameters,
    ClientProfile,
    ProfileVersion,
    AISuggestion,
)
from .services.chat_query_persistence import persist_chat_query_exchange
from .services.nlp.prompt_injector import get_style_prompt_injection
from .services.nlp.editor_action_nlp import (
    build_nlp_bundle_for_action,
    log_nlp_detected,
    nlp_action_summary,
)
from .services.profile_detection_store import (
    load_active_profile_detection_row,
    persist_profile_detection_for_sop_version,
    serialize_profile_detection_row,
)
from .services.sop_version_metadata_compact import (
    compact_sop_version_metadata_for_storage,
    log_metadata_load,
    log_metadata_merge,
)
try:
    from openai import BadRequestError
except Exception:  # pragma: no cover
    BadRequestError = Exception  # type: ignore[misc,assignment]
from chatbot.llm.provider import (
    check_local_llm_api_health,
    get_chat_pipeline_timeout_seconds,
    get_local_llm_config,
    get_local_llm_timeout_seconds,
    is_local_llm_unreachable_error,
)

# RAG-specific imports are lazy-loaded inside _get_smart_rag_chain()
# to avoid ModuleNotFoundError when running without the RAG chatbot modules.
# Modules: embeddings.embedder, retrieval.*, chain.rag_chain, langchain_qdrant, qdrant_client

ai_router = APIRouter()
_smart_rag_lock = threading.Lock()
_smart_rag_chain = None
_action_runtime_lock = threading.Lock()
_action_runtime = None
SOP_REF_PATTERN = re.compile(r"\bSOP-[A-Z0-9-]+\b", re.IGNORECASE)
DEV_REF_PATTERN = re.compile(r"\bDEV-[A-Z0-9-]+\b", re.IGNORECASE)
CAPA_REF_PATTERN = re.compile(r"\bCAPA-[A-Z0-9-]+\b", re.IGNORECASE)
AUDIT_REF_PATTERN = re.compile(r"\bAUDIT-[A-Z0-9-]+\b", re.IGNORECASE)
DECISION_REF_PATTERN = re.compile(r"\bDEC-[A-Z0-9-]+\b", re.IGNORECASE)
# Reload server after changing CHATBOT_USE_LOCAL_DB in environment (import-time flag).
# Default is false so semantic RAG/Qdrant is used unless explicitly overridden.
CHATBOT_USE_LOCAL_DB = os.getenv("CHATBOT_USE_LOCAL_DB", "false").strip().lower() == "true"
CHATBOT_ALLOW_LOCAL_DB_PRIMARY = os.getenv("CHATBOT_ALLOW_LOCAL_DB_PRIMARY", "false").strip().lower() == "true"
logger = logging.getLogger(__name__)
ACTION_INTENT_CREATE = re.compile(r"\b(create|new|generate|draft)\b.*\b(sop)\b", re.IGNORECASE)
ACTION_INTENT_DELETE = re.compile(r"\b(delete|remove)\b.*\b(sop|this sop|current sop)\b", re.IGNORECASE)
ACTION_INTENT_UPDATE = re.compile(
    r"\b(update|edit|modify|revise)\b.*\b(sop|this sop|current sop)\b|"
    r"\b(add)\b.*\b(section)\b.*\b(current sop|this sop)\b",
    re.IGNORECASE,
)

# Extra imperative / mutation-shaped requests blocked in assistant_mode=query (beyond _plan_sop_action).
QUERY_MODE_EXTRA_MUTATION = re.compile(
    r"\b(rewrite|re-?write|umschreiben|überarbeiten)\b.*\b(sop|this\s+sop|current\s+sop|diesen|diesem|dieser|aktuellen?)\b|"
    r"\b(improve|verbessern)\b.*\b(readability|lesbarkeit|this\s+sop|sop|abschnitt|section)\b|"
    r"\b(gap\s*check|gap-check|lückenprüfung|lücken\s*analyse|compliance-?check)\b.*\b(sop|this|current|dies|dokument)\b|"
    r"\b(run|execute|führe|starte)\b.*\b(nlp|profil|profile)\b",
    re.IGNORECASE,
)

QUERY_MODE_REFUSAL_DE = (
    "Im Modus **Nur Abfrage (Query)** führe ich keine Dokument- oder SOP-Aktionen aus "
    "(z. B. Umschreiben, Löschen, Aktualisieren oder Erstellen). "
    "Bitte wechseln Sie oben auf **Aktion ausführen**, wenn Sie solche Schritte wünschen, "
    "oder stellen Sie eine rein informelle Frage."
)


def _normalize_assistant_mode(raw: object) -> str:
    v = str(raw or "").strip().lower()
    if v in {"query", "query_only", "strict_query"}:
        return "query"
    return "action"


def _query_mode_mutation_intent(question: str, assistant_context: dict | None) -> bool:
    """True if the prompt looks like a document/SOP mutation (blocked in query mode)."""
    if _plan_sop_action(question, assistant_context):
        return True
    q = (question or "").strip()
    if not q:
        return False
    return bool(QUERY_MODE_EXTRA_MUTATION.search(q))


def _extract_profile_context(payload: dict, assistant_context: dict) -> dict[str, Any] | None:
    """
    Pull an already-detected NLP profile from request/context metadata or database.
    This intentionally does not run heavyweight profile detection during chat.
    """
    candidates = [
        payload.get("nlp_profile"),
        payload.get("profile_detection"),
        assistant_context.get("nlp_profile") if isinstance(assistant_context, dict) else None,
        assistant_context.get("profile_detection") if isinstance(assistant_context, dict) else None,
    ]
    current_sop = assistant_context.get("current_sop") if isinstance(assistant_context, dict) else {}
    if isinstance(current_sop, dict):
        meta = current_sop.get("metadata_json") or current_sop.get("metadata") or {}
        if isinstance(meta, dict):
            candidates.extend(
                [
                    meta.get("nlp_profile"),
                    meta.get("profile_detection"),
                    (meta.get("sopMetadata") or {}).get("nlp_profile")
                    if isinstance(meta.get("sopMetadata"), dict)
                    else None,
                ]
            )

    for candidate in candidates:
        if isinstance(candidate, dict):
            if isinstance(candidate.get("style_profile"), dict):
                return candidate
            nested = candidate.get("nlp_profile")
            if isinstance(nested, dict) and isinstance(nested.get("style_profile"), dict):
                return nested

    # Database lookup fallback via current_sop id
    sop_id = current_sop.get("id") if isinstance(current_sop, dict) else None
    if sop_id:
        db = SessionLocal()
        try:
            detected = db.query(SOPDetectedParameters).filter(SOPDetectedParameters.sop_id == uuid.UUID(str(sop_id))).first()
            if detected and detected.client_profile_id:
                profile = db.query(ClientProfile).filter(ClientProfile.id == detected.client_profile_id).first()
                if profile and profile.active_profile_json:
                    pref_style = profile.active_profile_json.get("preferred_style", {})
                    terminology = profile.active_profile_json.get("terminology", {})
                    
                    style_profile = {
                        "primary_style": pref_style.get("primary_format") or "procedural",
                        "primary_tone": pref_style.get("tone") or "formal",
                        "formality_level": pref_style.get("formality") or "standard",
                        "strictness_level": pref_style.get("directive_wording") or "moderate",
                        "numbering_type": "simple",
                        "format_pattern": pref_style.get("primary_format") or "standard prose",
                        "compliance_weight": pref_style.get("directive_wording") or "recommended",
                        "roles": terminology.get("acronyms", [])[:8],
                    }
                    return {"style_profile": style_profile}
        except Exception as e:
            logger.warning("[chatbot-profile] Database lookup for profile context failed: %s", e)
        finally:
            db.close()

    return None


def _get_db_style_profile(sop_id: str) -> dict | None:
    if not sop_id:
        return None
    db = SessionLocal()
    try:
        detected = db.query(SOPDetectedParameters).filter(SOPDetectedParameters.sop_id == uuid.UUID(str(sop_id))).first()
        if detected and detected.client_profile_id:
            profile = db.query(ClientProfile).filter(ClientProfile.id == detected.client_profile_id).first()
            if profile and profile.active_profile_json:
                pref_style = profile.active_profile_json.get("preferred_style", {})
                terminology = profile.active_profile_json.get("terminology", {})
                rewrite_rules = profile.active_profile_json.get("rewrite_rules", [])
                
                return {
                    "tone": pref_style.get("tone") or "formal",
                    "language": profile.active_profile_json.get("language") or "en",
                    "formality": pref_style.get("formality") or "standard",
                    "avg_sentence_words": 20,
                    "imperative_ratio": 0.5,
                    "modal_ratio": 0.5,
                    "bullet_density": 0.2,
                    "passive_markers": 0,
                    "style_rules": rewrite_rules if rewrite_rules else ["Maintain SOP style consistency."],
                    "terminology": terminology
                }
    except Exception as e:
        logger.warning(f"Failed to get DB style profile for SOP {sop_id}: {e}")
    finally:
        db.close()
    return None


def _resolve_style_profile(sop_ctx: dict, style_source_text: str) -> dict:
    sop_id = sop_ctx.get("sop_id")
    if sop_id:
        db_profile = _get_db_style_profile(sop_id)
        if db_profile:
            return db_profile
    return _derive_sop_style_profile(style_source_text)


def _is_prompt_too_large_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return (
        "context length" in msg
        or "n_keep" in msg
        or "prompt is too long" in msg
        or "maximum context length" in msg
        or "too many tokens" in msg
    )


def _get_smart_rag_chain() -> Any:
    """
    Lazy-load chatbot runtime so the main backend starts even if
    optional RAG env vars are missing.
    """
    global _smart_rag_chain
    if _smart_rag_chain is not None:
        return _smart_rag_chain

    with _smart_rag_lock:
        if _smart_rag_chain is not None:
            return _smart_rag_chain

        from qdrant_client import QdrantClient
        from langchain_qdrant import QdrantVectorStore
        from embeddings.embedder import get_embedder
        from retrieval.federated_retriever import FederatedRetriever
        from retrieval.hybrid_retriever import rag_unified_enabled, unified_semantic_collection
        from retrieval.reranker import CrossEncoderReranker
        from chain.rag_chain import SmartRAGChain

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url:
            raise RuntimeError("QDRANT_URL is not configured")

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        embedder = get_embedder()
        reranker = None
        try:
            reranker = CrossEncoderReranker(top_n=5)
        except Exception as reranker_exc:
            print(
                f"[startup] Reranker cache missing, continuing without reranker: {reranker_exc}",
                flush=True,
            )

        if rag_unified_enabled():
            ucol = unified_semantic_collection()
            collection_map = {
                "sops": ucol,
                "deviations": ucol,
                "capas": ucol,
                "audits": ucol,
                "decisions": ucol,
            }
        else:
            collection_map = {
                "sops": os.getenv("COLLECTION_SOPS", "docs_sops"),
                "deviations": os.getenv("COLLECTION_DEVIATIONS", "docs_deviations"),
                "capas": os.getenv("COLLECTION_CAPAS", "docs_capas"),
                "audits": os.getenv("COLLECTION_AUDITS", "docs_audits"),
                "decisions": os.getenv("COLLECTION_DECISIONS", "docs_decisions"),
            }
        vectorstores = {
            section: QdrantVectorStore(client=client, collection_name=collection_name, embedding=embedder)
            for section, collection_name in collection_map.items()
        }
        federated = FederatedRetriever(client=client, vectorstores=vectorstores, reranker=reranker)
        for section, collection_name in collection_map.items():
            federated.retrievers[section].collection_name = collection_name

        _smart_rag_chain = SmartRAGChain(federated)
        try:
            from .services.rag_cache import register_rag_chain

            register_rag_chain(_smart_rag_chain)
        except Exception:
            pass
        return _smart_rag_chain


def _normalize_action(action: str) -> str:
    normalized = (action or "").strip().lower().replace("-", "_")
    aliases = {
        "gapcheck": "gap_check",
        "quality_check": "gap_check",
        "support": "improve",
    }
    return aliases.get(normalized, normalized)


def _get_action_runtime() -> Any:
    global _action_runtime
    if _action_runtime is not None:
        return _action_runtime

    with _action_runtime_lock:
        if _action_runtime is not None:
            return _action_runtime
        _action_runtime = create_action_runtime()
        return _action_runtime


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _truncate_text(text: str, max_chars: int) -> str:
    raw = str(text or "")
    if max_chars <= 0 or len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 3].rstrip() + "..."


def _split_sentences(text: str) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def _derive_sop_style_profile(text: str) -> dict[str, Any]:
    raw = str(text or "")
    cleaned = _clean_text(raw)
    if not cleaned:
        return {
            "tone": "neutral",
            "language": "unknown",
            "avg_sentence_words": 0,
            "imperative_ratio": 0.0,
            "modal_ratio": 0.0,
            "bullet_density": 0.0,
            "passive_markers": 0,
            "style_rules": [],
        }

    sentences = _split_sentences(cleaned)
    words = re.findall(r"\b[\w/-]+\b", cleaned)
    word_count = len(words)
    sentence_count = max(1, len(sentences))
    avg_sentence_words = round(word_count / sentence_count, 1)

    lower = cleaned.lower()
    english_modals = re.findall(r"\b(should|must|shall|may|can)\b", lower)
    german_modals = re.findall(r"\b(soll|sollen|muss|müssen|darf|dürfen|kann|können)\b", lower)
    modal_count = len(english_modals) + len(german_modals)
    modal_ratio = round(modal_count / max(1, sentence_count), 3)

    imperative_markers = re.findall(
        r"\b(ensure|verify|document|record|review|approve|reject|notify|execute|perform|maintain|prüfen|sicherstellen|dokumentieren|aufzeichnen|überprüfen|genehmigen|durchführen)\b",
        lower,
    )
    imperative_ratio = round(len(imperative_markers) / max(1, sentence_count), 3)

    bullet_lines = re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", raw)
    line_count = max(1, len(re.findall(r"(?m)^", raw)))
    bullet_density = round(len(bullet_lines) / line_count, 3)

    passive_markers = len(
        re.findall(
            r"\b(be\s+\w+ed|is\s+\w+ed|are\s+\w+ed|was\s+\w+ed|were\s+\w+ed|wird\s+\w+(?:t|en)|werden\s+\w+(?:t|en))\b",
            lower,
        )
    )

    language = "de" if re.search(r"\b(und|der|die|das|mit|für|nicht)\b", lower) else "en"
    tone = "directive" if imperative_ratio >= 0.5 or modal_ratio >= 0.8 else "formal"

    style_rules: list[str] = []
    if avg_sentence_words > 28:
        style_rules.append("Shorten long sentences while preserving requirements.")
    if passive_markers > sentence_count // 2:
        style_rules.append("Prefer active voice with explicit responsible roles.")
    if modal_ratio > 1.2:
        style_rules.append("Reduce stacked modal verbs; keep obligations explicit and crisp.")
    if bullet_density > 0.2:
        style_rules.append("Keep concise procedural list formatting where appropriate.")
    if not style_rules:
        style_rules.append("Maintain current SOP style while tightening clarity and compliance language.")

    return {
        "tone": tone,
        "language": language,
        "avg_sentence_words": avg_sentence_words,
        "imperative_ratio": imperative_ratio,
        "modal_ratio": modal_ratio,
        "bullet_density": bullet_density,
        "passive_markers": passive_markers,
        "style_rules": style_rules,
    }


def _style_profile_prompt_block(profile: dict[str, Any]) -> str:
    if not isinstance(profile, dict) or not profile:
        return "STYLE_PROFILE: unavailable"
    rules = profile.get("style_rules") or []
    top_rules = "; ".join(str(r) for r in rules[:4]) or "Maintain SOP style consistency."
    return (
        "STYLE_PROFILE\n"
        f"- tone={profile.get('tone', 'formal')}\n"
        f"- language={profile.get('language', 'unknown')}\n"
        f"- avg_sentence_words={profile.get('avg_sentence_words', 0)}\n"
        f"- imperative_ratio={profile.get('imperative_ratio', 0.0)}\n"
        f"- modal_ratio={profile.get('modal_ratio', 0.0)}\n"
        f"- bullet_density={profile.get('bullet_density', 0.0)}\n"
        f"- passive_markers={profile.get('passive_markers', 0)}\n"
        f"- style_guidance={top_rules}"
    )


def _extract_text_from_tiptap(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    node_type = node.get("type")
    if node_type == "text":
        return str(node.get("text", ""))
    chunks: list[str] = []
    for child in node.get("content", []) or []:
        child_text = _extract_text_from_tiptap(child)
        if child_text:
            chunks.append(child_text)
    joiner = "\n" if node_type in {"paragraph", "heading", "listItem"} else " "
    return joiner.join(chunks).strip()


def _extract_sop_refs(question: str, chat_history: list[dict]) -> list[str]:
    refs = set(match.upper() for match in SOP_REF_PATTERN.findall(question or ""))
    for message in (chat_history or [])[-6:]:
        content = str(message.get("content", ""))
        for match in SOP_REF_PATTERN.findall(content):
            refs.add(match.upper())
    return sorted(refs)


def _extract_entity_refs(pattern: re.Pattern, question: str, chat_history: list[dict]) -> list[str]:
    refs = set(match.upper() for match in pattern.findall(question or ""))
    q_lower = (question or "").lower()
    list_intent = any(term in q_lower for term in [
        "all", "list", "show", "available", "which sops", "what sops", "sops",
        "which deviations", "what deviations", "deviations",
    ])
    follow_up_intent = any(term in q_lower for term in [
        "that", "same", "previous", "earlier", "this one", "the one",
        "wohi", "same sop", "same deviation",
    ])

    # Only pull refs from history for true follow-up questions.
    include_history = follow_up_intent and not list_intent
    if include_history:
        for message in (chat_history or [])[-6:]:
            content = str(message.get("content", ""))
            for match in pattern.findall(content):
                refs.add(match.upper())
    return sorted(refs)


def _build_local_db_chat_response(question: str, chat_history: list[dict], category: str | None) -> dict:
    q = (question or "").strip()
    q_like = f"%{q}%"
    q_lower = q.lower()
    q_tokens = [token for token in re.findall(r"[a-z0-9]+", q_lower) if len(token) >= 3]
    category = (category or "").strip().lower()
    db = SessionLocal()
    try:
        count_intent = bool(
            re.search(r"\b(how many|count|number of|total)\b", q_lower)
        )
        sop_intent = category in {"sops", "sop"} or "sop" in q_lower
        if count_intent and sop_intent:
            total_sops = db.query(SOP).count()
            return {
                "answer": f"Summary: There are {total_sops} SOP record(s) currently available.",
                "sources": [{
                    "id": f"INDEX-SOP-COUNT({total_sops})",
                    "type": "sop",
                    "label": "Indexed SOP inventory",
                }],
                # "citations": [{
                #     "ref": f"INDEX-SOP-COUNT({total_sops})",
                #     "title": "Indexed SOP inventory",
                #     "type": "sop",
                #     "status": "",
                #     "score": 1.0,
                #     "excerpt": f"Distinct SOPs in primary database: {total_sops}.",
                # }],
                "retrieval_debug": [],
                "suggestions": [
                    "List all SOPs with titles",
                    "Open a specific SOP by number",
                    "Ask for latest SOP changes",
                ],
                "retrieval_stats": {"mode": "local-db", "hits": 1, "count_mode": True},
                "routed_to": "local-db-count",
            }

        citations = []
        sources = []
        answer_parts = []

        def push_source(ref: str, title: str, source_type: str, excerpt: str):
            citations.append({
                "ref": ref,
                "title": title,
                "type": source_type,
                "status": "",
                "score": 1.0,
                "excerpt": excerpt,
            })
            sources.append({
                "id": ref,
                "type": source_type,
                "label": title or ref,
            })

        def _tokenized_clause(columns):
            if not q_tokens:
                return None
            clauses = []
            for token in q_tokens[:8]:
                token_like = f"%{token}%"
                for col in columns:
                    clauses.append(col.ilike(token_like))
            return or_(*clauses) if clauses else None

        wants_sops = category in {"", "sops", "sop", "all"} and (
            category in {"sops", "sop"} or "sop" in q_lower or "procedure" in q_lower or "policy" in q_lower
        )
        wants_deviations = category in {"", "deviations", "deviation", "all"} and (
            category in {"deviations", "deviation"} or "deviation" in q_lower or "deviations" in q_lower or "excursion" in q_lower
        )
        wants_capas = category in {"", "capas", "capa", "all"} and (
            category in {"capas", "capa"} or "capa" in q_lower or "corrective" in q_lower
        )
        wants_audits = category in {"", "audits", "audit", "all"} and (
            category in {"audits", "audit"} or "audit" in q_lower or "finding" in q_lower
        )
        wants_decisions = category in {"", "decisions", "decision", "all"} and (
            category in {"decisions", "decision"} or "decision" in q_lower
        )

        if not any([wants_sops, wants_deviations, wants_capas, wants_audits, wants_decisions]):
            # Broad natural-language query without explicit type: search SOP + deviations first.
            wants_sops = True
            wants_deviations = True

        # SOPs
        if wants_sops:
            sop_refs = _extract_entity_refs(SOP_REF_PATTERN, question, chat_history)
            sops = []
            if sop_refs:
                for ref in sop_refs[:5]:
                    row = db.query(SOP).filter(SOP.sop_number.ilike(ref)).first()
                    if row:
                        sops.append(row)
            else:
                token_clause = _tokenized_clause([SOP.sop_number, SOP.title, SOP.department])
                base = db.query(SOP)
                if token_clause is not None:
                    sops = base.filter(token_clause).limit(5).all()
                else:
                    sops = base.filter(
                        (SOP.sop_number.ilike(q_like)) |
                        (SOP.title.ilike(q_like)) |
                        (SOP.department.ilike(q_like))
                    ).limit(5).all()
                if not sops:
                    sops = base.order_by(SOP.updated_at.desc()).limit(5).all()

            for sop in sops:
                push_source(
                    sop.sop_number,
                    sop.title,
                    "sop",
                    f"SOP in department {sop.department or 'unknown'}."
                )
            if sops:
                answer_parts.append(
                    "SOP matches: " + ", ".join(f"{s.sop_number} ({s.title})" for s in sops)
                )

        # Deviations
        if wants_deviations:
            dev_refs = _extract_entity_refs(DEV_REF_PATTERN, question, chat_history)
            devs = []
            if dev_refs:
                for ref in dev_refs[:5]:
                    row = db.query(Deviation).filter(Deviation.deviation_number.ilike(ref)).first()
                    if row:
                        devs.append(row)
            else:
                token_clause = _tokenized_clause([Deviation.deviation_number, Deviation.title, Deviation.description_text])
                base = db.query(Deviation)
                if token_clause is not None:
                    devs = base.filter(token_clause).limit(5).all()
                else:
                    devs = base.filter(
                        (Deviation.deviation_number.ilike(q_like)) |
                        (Deviation.title.ilike(q_like)) |
                        (Deviation.description_text.ilike(q_like))
                    ).limit(5).all()
                if not devs:
                    devs = base.order_by(Deviation.updated_at.desc()).limit(5).all()
            for dev in devs:
                push_source(
                    dev.deviation_number,
                    dev.title,
                    "deviation",
                    f"Deviation status {dev.external_status or 'unknown'}, impact {dev.impact_level or 'unknown'}."
                )
            if devs:
                answer_parts.append(
                    "Deviation matches: " + ", ".join(f"{d.deviation_number} ({d.title})" for d in devs)
                )

        # CAPAs
        if wants_capas:
            capa_refs = _extract_entity_refs(CAPA_REF_PATTERN, question, chat_history)
            capas = []
            if capa_refs:
                for ref in capa_refs[:5]:
                    row = db.query(Capa).filter(Capa.capa_number.ilike(ref)).first()
                    if row:
                        capas.append(row)
            else:
                token_clause = _tokenized_clause([Capa.capa_number, Capa.title, Capa.action_text])
                base = db.query(Capa)
                if token_clause is not None:
                    capas = base.filter(token_clause).limit(5).all()
                else:
                    capas = base.filter(
                        (Capa.capa_number.ilike(q_like)) |
                        (Capa.title.ilike(q_like)) |
                        (Capa.action_text.ilike(q_like))
                    ).limit(5).all()
                if not capas:
                    capas = base.order_by(Capa.updated_at.desc()).limit(5).all()
            for capa in capas:
                push_source(
                    capa.capa_number,
                    capa.title,
                    "capa",
                    f"CAPA status {capa.external_status or 'unknown'}."
                )
            if capas:
                answer_parts.append(
                    "CAPA matches: " + ", ".join(f"{c.capa_number} ({c.title})" for c in capas)
                )

        # Audits
        if wants_audits:
            audit_refs = _extract_entity_refs(AUDIT_REF_PATTERN, question, chat_history)
            audits = []
            if audit_refs:
                for ref in audit_refs[:5]:
                    row = db.query(AuditFinding).filter(
                        (AuditFinding.audit_number.ilike(ref)) |
                        (AuditFinding.finding_number.ilike(ref))
                    ).first()
                    if row:
                        audits.append(row)
            else:
                token_clause = _tokenized_clause([AuditFinding.audit_number, AuditFinding.finding_number, AuditFinding.finding_text])
                base = db.query(AuditFinding)
                if token_clause is not None:
                    audits = base.filter(token_clause).limit(5).all()
                else:
                    audits = base.filter(
                        (AuditFinding.audit_number.ilike(q_like)) |
                        (AuditFinding.finding_number.ilike(q_like)) |
                        (AuditFinding.finding_text.ilike(q_like))
                    ).limit(5).all()
                if not audits:
                    audits = base.order_by(AuditFinding.updated_at.desc()).limit(5).all()
            for audit in audits:
                ref = audit.finding_number or audit.audit_number or "AUDIT"
                push_source(ref, ref, "audit", f"Audit finding status {audit.acceptance_status or 'unknown'}.")
            if audits:
                answer_parts.append(
                    "Audit matches: " + ", ".join((a.finding_number or a.audit_number or "AUDIT") for a in audits)
                )

        # Decisions
        if wants_decisions:
            dec_refs = _extract_entity_refs(DECISION_REF_PATTERN, question, chat_history)
            decisions = []
            if dec_refs:
                for ref in dec_refs[:5]:
                    row = db.query(Decision).filter(Decision.decision_number.ilike(ref)).first()
                    if row:
                        decisions.append(row)
            else:
                token_clause = _tokenized_clause([Decision.decision_number, Decision.title, Decision.decision_statement])
                base = db.query(Decision)
                if token_clause is not None:
                    decisions = base.filter(token_clause).limit(5).all()
                else:
                    decisions = base.filter(
                        (Decision.decision_number.ilike(q_like)) |
                        (Decision.title.ilike(q_like)) |
                        (Decision.decision_statement.ilike(q_like))
                    ).limit(5).all()
                if not decisions:
                    decisions = base.order_by(Decision.updated_at.desc()).limit(5).all()
            for dec in decisions:
                ref = dec.decision_number or dec.title or "DECISION"
                push_source(ref, dec.title, "decision", "Decision record matched in local database.")
            if decisions:
                answer_parts.append(
                    "Decision matches: " + ", ".join((d.decision_number or d.title or "Decision") for d in decisions)
                )

        if not citations:
            return {
                "answer": "No relevant local database records were found for this query.",
                "sources": [],
                "citations": [],
                "retrieval_debug": [],
                "suggestions": [
                    "Ask with an exact SOP/DEV/CAPA number",
                    "Try a shorter and more specific query",
                    "Use category-specific wording (SOP, deviation, CAPA, audit, decision)",
                ],
                "retrieval_stats": {"mode": "local-db", "hits": 0},
                "routed_to": "local-db",
            }

        return {
            "answer": " ".join(answer_parts),
            "sources": sources,
            "citations": citations,
            "retrieval_debug": [
                {
                    "rank": idx + 1,
                    "source_id": c.get("ref", ""),
                    "ref": c.get("ref", ""),
                    "title": c.get("title", ""),
                    "score": c.get("score", 1.0),
                    "type": c.get("type", ""),
                    "snippet": c.get("excerpt", ""),
                }
                for idx, c in enumerate(citations[:20])
            ],
            "suggestions": [
                "Ask for details of one returned record",
                "Ask for status and ownership of a returned item",
                "Ask for related SOP/deviation/CAPA links",
            ],
            "retrieval_stats": {"mode": "local-db", "hits": len(citations)},
            "routed_to": "local-db",
        }
    finally:
        db.close()


def _build_sop_db_fallback(question: str, chat_history: list[dict]) -> dict | None:
    sop_refs = _extract_sop_refs(question, chat_history)
    if not sop_refs:
        return None

    db = SessionLocal()
    try:
        hits = []
        for sop_ref in sop_refs[:3]:
            sop = db.query(SOP).filter(SOP.sop_number.ilike(sop_ref)).first()
            if not sop:
                continue

            version = None
            if sop.current_version_id:
                version = db.query(SOPVersion).filter(SOPVersion.id == sop.current_version_id).first()
            if not version:
                version = (
                    db.query(SOPVersion)
                    .filter(SOPVersion.sop_id == sop.id)
                    .order_by(SOPVersion.created_at.desc())
                    .first()
                )

            content_text = _clean_text(_extract_text_from_tiptap((version.content_json if version else {}) or {}))
            excerpt = content_text[:500]
            if len(content_text) > 500:
                excerpt += "..."

            hits.append({
                "sop_number": sop.sop_number,
                "title": sop.title,
                "status": (version.external_status if version else "") or "unknown",
                "version_number": (version.version_number if version else "") or "",
                "excerpt": excerpt or "No SOP body text available.",
            })

        if not hits:
            return None

        if len(hits) == 1:
            item = hits[0]
            answer = (
                f"{item['sop_number']} ({item['title']}) was found in the main SOP database. "
                f"Current status: {item['status']}."
            )
        else:
            refs = ", ".join(f"{item['sop_number']} ({item['title']})" for item in hits)
            answer = f"Found these SOP records in the main SOP database: {refs}."

        details = " ".join(
            f"{item['sop_number']}: {item['excerpt']}" for item in hits
        ).strip()
        if details:
            answer = f"{answer}\n\n{details}"

        citations = [
            {
                "ref": item["sop_number"],
                "title": item["title"],
                "type": "sop",
                "status": item["status"],
                "score": 1.0,
            }
            for item in hits
        ]

        sources = [
            {
                "id": item["sop_number"],
                "type": "sop",
                "label": item["title"] or item["sop_number"],
            }
            for item in hits
        ]

        return {
            "answer": answer,
            "sources": sources,
            "citations": citations,
            "retrieval_debug": [
                {
                    "rank": idx + 1,
                    "source_id": item["sop_number"],
                    "ref": item["sop_number"],
                    "title": item["title"],
                    "score": 1.0,
                    "type": "sop",
                    "snippet": item["excerpt"],
                }
                for idx, item in enumerate(hits[:20])
            ],
            "suggestions": [
                f"Summarize {hits[0]['sop_number']} responsibilities",
                f"Show procedure steps from {hits[0]['sop_number']}",
                "Ask for related deviations or CAPAs",
            ],
            "retrieval_stats": {"fallback": "postgres_sop_lookup", "hits": len(hits)},
            "routed_to": "db-fallback-sops",
        }
    finally:
        db.close()


def _build_context(payload: AIActionRequest) -> str:
    bits = []
    if payload.sop_title:
        bits.append(f"SOP title: {payload.sop_title}")
    if payload.section_name:
        bits.append(f"Section name: {payload.section_name}")
    if payload.section_type:
        bits.append(f"Section type: {payload.section_type}")
    return " | ".join(bits) if bits else "SOP context unavailable"


def _paragraph(text: str) -> str:
    return f"<p>{escape(text)}</p>"


def _build_prompt(action: str, payload: AIActionRequest) -> str:
    context = _build_context(payload)
    if action == "gap_check":
        return (
            "You are a Lead GMP/QA Compliance Auditor with expertise in ISO 9001:2015, ISO 13485:2016, "
            "FDA 21 CFR Parts 11 and 820, and EU GMP Annex 11.\n\n"
            f"DOCUMENT CONTEXT: {context}\n\n"
            "YOUR TASK: Perform a thorough compliance gap analysis on the SOP text below. "
            "Check for: (1) missing or incomplete procedure steps, (2) undefined responsibilities \u2014 "
            "roles must be named specifically, (3) undefined frequencies or timelines \u2014 no vague terms like "
            "'regularly' or 'as needed', (4) missing data integrity or access controls, (5) absent "
            "documentation requirements including record names and retention periods, (6) ambiguous language "
            "and undefined technical terms, (7) missing regulatory references where required.\n\n"
            f"TEXT TO ANALYZE:\n{payload.text}\n\n"
            "Return ONLY a valid JSON object structured as: "
            '{"gaps": [{"issue": "short label", "explanation": "why this fails GMP/regulatory requirements", '
            '"recommendation": "exact SOP-ready text to fix the gap"}], '
            '"section_assessed": "section name"}'
        )
    if action == "rewrite":
        return (
            "You are a senior GMP/QA technical writer with expertise in ISO 13485, FDA 21 CFR, and EU GMP Annex 11.\n\n"
            f"DOCUMENT CONTEXT: {context}\n\n"
            "YOUR TASK: Perform a complete, professional rewrite of the SOP text below. Apply these standards: "
            "(1) Use active voice and imperative verbs throughout. (2) Every sentence must name a specific role "
            "as the subject \u2014 never 'someone' or 'the team'. (3) Replace all vague qualifiers with specific "
            "values, frequencies, or defined conditions. (4) Ensure logical, chronological process order. "
            "(5) Use parallel structure in lists. (6) Add critical step callouts where safety or compliance is at risk.\n"
            "RULES: Do NOT add Purpose/Scope/Responsibilities/Procedure headings. Do NOT change the core topic. "
            "You MAY restructure sentences and reorder information for flow.\n\n"
            f"TEXT TO REWRITE:\n{payload.text}\n\n"
            "Return ONLY a valid JSON object: "
            '{"rewritten_text": "full rewritten text", '
            '"structural_changes": ["change 1", "change 2"], '
            '"rationale": "2-sentence explanation of compliance and clarity improvements"}'
        )
    if action == "improve":
        return (
            "You are a senior GMP/QA technical writer specializing in regulatory SOP documentation.\n\n"
            f"DOCUMENT CONTEXT: {context}\n\n"
            "YOUR TASK: Make targeted, high-quality improvements to the SOP text below. Apply these criteria: "
            "(1) Fix all grammar, punctuation, and spelling errors. (2) Replace passive voice with active voice. "
            "(3) Replace vague qualifiers ('appropriate', 'as needed') with specific, measurable language. "
            "(4) Ensure responsibilities are attributed to named roles. (5) Make language imperative and unambiguous.\n"
            "STRICT RULES: Do NOT add SOP headings or restructure into a full SOP. Do NOT change factual content or meaning. "
            "Make only the smallest meaningful improvements required.\n\n"
            f"TEXT TO IMPROVE:\n{payload.text}\n\n"
            "Return ONLY a valid JSON object: "
            '{"improved_text": "the improved text", '
            '"changes_made": ["specific change 1", "specific change 2"], '
            '"compliance_note": "one sentence explaining the GMP/quality improvement achieved"}'
        )
    raise HTTPException(status_code=400, detail=f"Action '{action}' is not supported.")


def _render_gap_check(structured_data: dict) -> str:
    return (
        f"<h3>Issue</h3>{_paragraph(structured_data['issue'])}"
        f"<h3>Explanation</h3>{_paragraph(structured_data['explanation'])}"
        f"<h3>Recommendation</h3>{_paragraph(structured_data['recommendation'])}"
    )


def _render_rewrite(structured_data: dict) -> str:
    steps = "".join(f"<li>{escape(step)}</li>" for step in structured_data["procedure"])
    return (
        f"<h2>Purpose</h2>{_paragraph(structured_data['purpose'])}"
        f"<h2>Scope</h2>{_paragraph(structured_data['scope'])}"
        f"<h2>Responsibilities</h2>{_paragraph(structured_data['responsibilities'])}"
        f"<h2>Procedure</h2><ol>{steps}</ol>"
        f"<h2>Documentation</h2>{_paragraph(structured_data['documentation'])}"
    )


def _render_improve(structured_data: dict) -> str:
    return (
        f"<h3>Improved Version</h3>{_paragraph(structured_data['improved_version'])}"
        f"<h3>Reason for Improvement</h3>{_paragraph(structured_data['reason_for_improvement'])}"
    )


def _action_output_token_budget(input_chars: int) -> int:
    """Target output budget before context-aware clamping."""
    base = int(
        os.getenv("ACTION_LLM_MAX_TOKENS")
        or os.getenv("ACTION_MAX_OUTPUT_TOKENS")
        or "4096"
    )
    cap = int(os.getenv("ACTION_MAX_OUTPUT_TOKENS_CAP", "32768"))
    if input_chars <= 0:
        return min(cap, base)
    return min(cap, max(2048, min(base, int(input_chars * 0.45) + 1200)))


def _action_model_context_tokens() -> int:
    try:
        return max(4096, int(os.getenv("ACTION_MODEL_CONTEXT_TOKENS", "32768")))
    except (TypeError, ValueError):
        return 32768


def _action_prompt_soft_limit_chars() -> int:
    raw = os.getenv("ACTION_PROMPT_SOFT_LIMIT")
    if raw and raw.strip():
        try:
            return max(4000, int(raw))
        except (TypeError, ValueError):
            pass
    return max(8000, int(_action_model_context_tokens() * 0.75))


def _is_context_length_error_text(text: str) -> bool:
    msg = (text or "").lower()
    return (
        ("n_keep" in msg and "n_ctx" in msg)
        or "context length" in msg
        or "prompt is greater than context" in msg
        or "prompt is too long" in msg
        or "maximum context length" in msg
    )


def _extract_n_ctx_from_error(text: str) -> int | None:
    m = re.search(r"n_ctx\s*:\s*(\d+)", text or "", flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def _safe_action_max_tokens(base_tokens: int, prompt_chars: int, *, n_ctx: int | None = None) -> int:
    prompt_est_tokens = max(1, int(prompt_chars / 4.0))
    ctx = n_ctx or _action_model_context_tokens()
    reserve = int(os.getenv("ACTION_CONTEXT_RESPONSE_RESERVE", "256"))
    safe_by_ctx = max(128, ctx - prompt_est_tokens - reserve)
    safe_cap = int(os.getenv("ACTION_SAFE_MAX_TOKENS_CAP", "32768"))
    return max(128, min(int(base_tokens), safe_by_ctx, safe_cap))


def _context_error_http_exception(err_txt: str) -> HTTPException:
    cfg = get_local_llm_config()
    return HTTPException(
        status_code=422,
        detail={
            "message": (
                "Selected text is too long for the local model context. Please select a smaller section "
                "or load the model with larger context length."
            ),
            "validation_or_parse_error": err_txt,
            "hint": "Reduce selection length (especially Gap Check), or increase LM Studio model context.",
            "llm_model": cfg.model,
            "llm_base_url": cfg.base_url,
            "n_ctx": _extract_n_ctx_from_error(err_txt),
        },
    )


def _trim_large_selection_for_fallback(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    try:
        lim = int(os.getenv("ACTION_FALLBACK_TEXT_LIMIT", "3200"))
    except (TypeError, ValueError):
        lim = 3200
    if len(s) <= lim:
        return s
    head = max(1200, int(lim * 0.6))
    tail = max(600, lim - head - 24)
    return s[:head] + "\n\n[... trimmed ...]\n\n" + s[-tail:]


def _extract_text_and_meta(message: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(message, AIMessage):
        content = message.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    t = item.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            text = "".join(parts)
        else:
            text = str(content or "")
        meta = dict(message.response_metadata or {})
        usage = meta.get("token_usage") or meta.get("usage") or {}
        usage_meta = getattr(message, "usage_metadata", None) or {}
        if isinstance(usage_meta, dict):
            usage = {**usage, **usage_meta}
        meta["usage"] = usage
        return text, meta
    return str(message or ""), {}


def _response_looks_cut(text: str) -> bool:
    s = (text or "").rstrip()
    if not s:
        return False
    if s.endswith(("}", '"}', "]}", '"]}')):
        return False
    return s[-1].isalnum() and not s.endswith((".", "!", "?", "}", "]", '"'))


def _call_action_llm(runtime: Any, prompt: str, *, input_char_budget: int = 0, action: str = "unknown") -> str:
    base_n = _action_output_token_budget(input_char_budget) if input_char_budget else int(
        os.getenv("ACTION_LLM_MAX_TOKENS") or os.getenv("ACTION_MAX_OUTPUT_TOKENS") or "4096"
    )
    if action == "rewrite" and input_char_budget:
        configured_cap = int(
            os.getenv("ACTION_LLM_MAX_TOKENS")
            or os.getenv("ACTION_MAX_OUTPUT_TOKENS")
            or "4096"
        )
        base_n = min(configured_cap, max(base_n, 4096, int(input_char_budget * 1.4) + 1800))
    if action == "gap_check" and input_char_budget:
        configured_cap = int(
            os.getenv("ACTION_GAP_CHECK_MAX_TOKENS")
            or os.getenv("ACTION_LLM_MAX_TOKENS")
            or os.getenv("ACTION_MAX_OUTPUT_TOKENS")
            or "4096"
        )
        base_n = min(
            int(os.getenv("ACTION_MAX_OUTPUT_TOKENS_CAP", "32768")),
            max(base_n, int(configured_cap), 3500, int(input_char_budget * 0.65) + 2200),
        )
    soft = _action_prompt_soft_limit_chars()
    cfg = get_local_llm_config()

    budgets: list[int] = []
    for b in (soft, int(soft * 0.75), int(soft * 0.5)):
        if b > 0 and b not in budgets:
            budgets.append(b)

    n_ctx_hint: int | None = None
    last_context_error: str | None = None
    out = ""
    used_tokens = base_n
    last_meta: dict[str, Any] = {}
    length_limited_seen = False

    for budget in budgets:
        work = truncate_prompt_for_llm(prompt, budget) if len(prompt) > budget else prompt
        used_tokens = _safe_action_max_tokens(base_n, len(work), n_ctx=n_ctx_hint)
        try:
            msg = runtime.llm.bind(max_tokens=used_tokens).invoke(work)
            out, last_meta = _extract_text_and_meta(msg)
            finish_reason = str(last_meta.get("finish_reason") or "").lower()
            usage = last_meta.get("usage") or {}
            logger.info(
                "[ai-action-llm-meta] action=%s prompt_chars=%s output_chars=%s max_tokens=%s finish_reason=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                action,
                len(work),
                len(out or ""),
                used_tokens,
                finish_reason or "unknown",
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                usage.get("total_tokens"),
            )
            if finish_reason == "length":
                if action == "gap_check":
                    expanded = _safe_action_max_tokens(
                        min(int(os.getenv("ACTION_MAX_OUTPUT_TOKENS_CAP", "32768")), int(used_tokens * 2.2) + 800),
                        len(work),
                        n_ctx=n_ctx_hint,
                    )
                    if expanded > used_tokens:
                        try:
                            msg2 = runtime.llm.bind(max_tokens=expanded).invoke(work)
                            out2, meta2 = _extract_text_and_meta(msg2)
                            fr2 = str(meta2.get("finish_reason") or "").lower()
                            if (out2 or "").strip() and fr2 != "length":
                                logger.info(
                                    "[ai-action-gap-retry] expanded max_tokens %s -> %s finish_reason=%s",
                                    used_tokens,
                                    expanded,
                                    fr2,
                                )
                                return out2
                        except Exception:
                            pass
                length_limited_seen = True
                continue
            if (out or "").strip():
                break
            msg = runtime.llm.bind(max_tokens=used_tokens).invoke(work + ACTION_LLM_EMPTY_RETRY_SUFFIX)
            out, last_meta = _extract_text_and_meta(msg)
            if (out or "").strip():
                break
        except BadRequestError as exc:
            err_txt = str(exc)
            if _is_context_length_error_text(err_txt):
                last_context_error = err_txt
                n_ctx_hint = _extract_n_ctx_from_error(err_txt) or n_ctx_hint
                logger.warning(
                    "[ai-action-context-error] prompt_chars=%s max_tokens=%s model=%s base_url=%s n_ctx=%s error=%s",
                    len(work),
                    used_tokens,
                    cfg.model,
                    cfg.base_url,
                    n_ctx_hint,
                    err_txt,
                )
                continue
            msg = runtime.fallback_llm.bind(max_tokens=used_tokens).invoke(work)
            out, last_meta = _extract_text_and_meta(msg)
            if (out or "").strip():
                break
        except Exception:
            msg = runtime.fallback_llm.bind(max_tokens=used_tokens).invoke(work)
            out, last_meta = _extract_text_and_meta(msg)
            finish_reason = str(last_meta.get("finish_reason") or "").lower()
            if finish_reason == "length":
                length_limited_seen = True
                continue
            if (out or "").strip():
                break

    if last_context_error and not (out or "").strip():
        raise _context_error_http_exception(last_context_error)

    finish_reason = str(last_meta.get("finish_reason") or "").lower()
    looks_cut = _response_looks_cut(out)
    if not (out or "").strip():
        raise HTTPException(
            status_code=422,
            detail={
                "message": "AI returned an empty response.",
                "validation_or_parse_error": f"finish_reason={finish_reason or 'unknown'}",
                "hint": "Check LOCAL_LLM_BASE_URL / LOCAL_LLM_MODEL and retry.",
            },
        )
    if finish_reason == "length" or (length_limited_seen and finish_reason == "length"):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "AI response was truncated due to model/output limit.",
                "validation_or_parse_error": f"finish_reason={finish_reason or 'unknown'}",
                "hint": "Increase ACTION_LLM_MAX_TOKENS, shorten selection, or increase ACTION_MODEL_CONTEXT_TOKENS.",
            },
        )
    if looks_cut and finish_reason not in ("stop", "end_turn", "end"):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "AI response appears truncated.",
                "validation_or_parse_error": f"finish_reason={finish_reason or 'unknown'}",
                "hint": "Increase ACTION_LLM_MAX_TOKENS or shorten the selection.",
            },
        )

    preview = (out or "").replace("\n", "\\n")[:900]
    logger.info(
        "[ai-action-llm] raw_len=%s max_tokens=%s preview=%s",
        len(out or ""),
        used_tokens,
        preview,
    )
    return out


def _render_dynamic_text(text: str) -> str:
    cleaned = _normalize_gap_check_analysis_text(text or "")
    lines = [line.strip() for line in re.split(r"\r?\n+", cleaned) if line.strip()]
    if not lines:
        return "<p>No suggestion returned.</p>"
    return "".join(f"<p>{escape(line)}</p>" for line in lines)


def _normalize_gap_check_analysis_text(text: str) -> str:
    t = text or ""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"(?m)^\s*#+\s*", "", t)
    t = t.replace("**", "")
    t = re.sub(r"(?m)^\s*---+\s*$", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _render_gap_check_analysis_html(analysis: str) -> str:
    # Keep compatibility with chatbot.routes implementation so this shim
    # never fails with NameError when gap_check is requested.
    normalized = _normalize_gap_check_analysis_text(analysis)
    if not normalized:
        return "<p>No suggestion returned.</p>"
    return _render_dynamic_text(normalized)


def _render_dynamic_gap_check(gaps: list[dict[str, str]]) -> str:
    if not gaps:
        return "<p>No compliance gaps identified for the selected text.</p>"
    return "".join(
        (
            f"<h3>Issue</h3>{_paragraph(gap.get('issue', ''))}"
            f"<h3>Explanation</h3>{_paragraph(gap.get('explanation', ''))}"
            f"<h3>Recommendation</h3>{_paragraph(gap.get('recommendation', ''))}"
        )
        for gap in gaps
    )


def _infer_edit_scope_from_payload(payload: AIActionRequest) -> str | None:
    raw = str(getattr(payload, "edit_scope", None) or "").strip().lower()
    st = str(payload.section_type or "").strip().lower()
    name = str(payload.section_name or "").strip().lower()
    if raw == "full_document" or st in ("full document", "full sop") or name in (
        "full document",
        "full sop",
        "gesamte sop",
        "komplette sop",
        "entire sop",
        "whole sop",
    ):
        return "full_document"
    text = str(payload.text or "")
    if is_traceability_register_block(text):
        return "section_only"
    if raw in ("section_only", "full_document"):
        return raw
    return None


def _build_improve_rewrite_context(
    request: ActionRequest,
    style_block: str,
    sop_context_block: str,
) -> str:
    scope = resolve_edit_scope(request)
    if scope == "section_only":
        return (
            f"{IMPROVE_REWRITE_NO_RAG_CONTEXT}\n"
            "SCOPE_LOCK: Rewrite/improve ONLY the TEXT block in this request. "
            "Ignore other SOP sections from metadata or NLP lists. "
            "Do not output title, Version, Status, or backbone sections 1–5 unless they appear in TEXT.\n"
            f"{style_block}"
        ).strip()
    return f"{IMPROVE_REWRITE_NO_RAG_CONTEXT}\n{style_block}\n{sop_context_block}".strip()


def _enforce_section_only_action_text(
    runtime: Any,
    *,
    action: str,
    request: ActionRequest,
    text: str,
    context: str,
    nlp_block: str,
    ch_budget: int,
) -> str:
    """Retry or trim model output that expanded a section-only request into a full SOP."""
    scope = resolve_edit_scope(request)
    original = request.section_text or ""
    out = (text or "").strip()
    if scope != "section_only" or not out or not violates_section_only_scope(original, out):
        return out

    logger.warning(
        "[ai-action-scope-violation] action=%s section=%s in_chars=%s out_chars=%s — retrying strict prompt",
        action,
        request.section_title,
        len(original),
        len(out),
    )
    retry_builder = (
        build_section_only_rewrite_retry_prompt
        if action == "rewrite"
        else build_section_only_improve_retry_prompt
    )
    retry_prompt = retry_builder(request, context, nlp_block)
    schema = RewriteResponse if action == "rewrite" else ImproveResponse
    try:
        retry_parsed = parse_with_retry(
            raw=_call_action_llm(runtime, retry_prompt, input_char_budget=ch_budget, action=f"{action}_scope_retry"),
            schema=schema,
            prompt=retry_prompt,
            call_llm=lambda rp: _call_action_llm(
                runtime, rp, input_char_budget=ch_budget, action=f"{action}_scope_retry"
            ),
            audit_log=[],
        )
        retry_text = (
            retry_parsed.rewritten_text
            if action == "rewrite"
            else retry_parsed.improved_text
        )
        if retry_text and not violates_section_only_scope(original, retry_text):
            return retry_text.strip()
    except Exception as exc:
        logger.warning("[ai-action-scope-retry-failed] action=%s err=%s", action, exc)

    extracted = extract_register_slice_from_output(out, original)
    if extracted and not violates_section_only_scope(original, extracted):
        logger.info("[ai-action-scope-extract] action=%s extracted_chars=%s", action, len(extracted))
        return extracted
    return out


def _build_action_request(payload: AIActionRequest) -> ActionRequest:
    entity = ""
    try:
        raw_e = getattr(payload, "sop_entity_id", None)
        entity = str(raw_e).strip() if raw_e is not None else ""
    except Exception:
        entity = ""
    edit_scope = _infer_edit_scope_from_payload(payload)
    section_type = payload.section_type or "Selected Text"
    if edit_scope == "full_document":
        section_type = "Full Document"
    return ActionRequest(
        document_id=payload.sop_title or "editor-document",
        section_id=(payload.section_name or "selected-text").lower().replace(" ", "-"),
        sop_title=payload.sop_title or "Untitled SOP",
        section_title=payload.section_name or "Selected text",
        section_type=section_type,
        section_text=payload.text,
        sop_entity_id=entity or None,
        instruction=payload.instruction,
        edit_scope=edit_scope,
    )


def _load_uploaded_sop_context(request: ActionRequest) -> dict[str, Any]:
    """
    Resolve uploaded SOP from DB using optional SOP UUID (``sop_entity_id``) or
    SOP number/title match, and return text context, compact ``sop_versions.metadata_json``,
    and active ProfileDetection snapshot.
    """
    db = SessionLocal()
    try:
        sop = None
        entity_raw = str(getattr(request, "sop_entity_id", None) or "").strip()
        if entity_raw:
            try:
                uid = uuid.UUID(entity_raw)
                sop = db.query(SOP).filter(SOP.id == uid, SOP.is_active == True).first()  # noqa: E712
            except Exception:
                sop = None

        if sop is None:
            sop_ref = str(request.sop_title or "").strip()
            if not sop_ref:
                return {"detected": False, "reason": "missing_sop_title"}
            sop = db.query(SOP).filter(SOP.is_active == True).filter(  # noqa: E712
                (SOP.sop_number.ilike(sop_ref)) | (SOP.title.ilike(sop_ref))
            ).first()

            if not sop:
                return {"detected": False, "reason": "not_found", "query": sop_ref}

        version = None
        if sop.current_version_id:
            version = db.query(SOPVersion).filter(SOPVersion.id == sop.current_version_id).first()
        if not version:
            version = (
                db.query(SOPVersion)
                .filter(SOPVersion.sop_id == sop.id)
                .order_by(SOPVersion.created_at.desc())
                .first()
            )

        sop_text = _extract_text_from_tiptap((version.content_json or {})) if version else ""
        meta: dict[str, Any] = (
            dict(version.metadata_json) if version and isinstance(version.metadata_json, dict) else {}
        )
        meta_keys = sorted(meta.keys())

        compact_vm: dict[str, Any] = {}
        if version:
            compact_vm = compact_sop_version_metadata_for_storage(meta, sop, version)

        pd_row = None
        if version:
            pd_row = load_active_profile_detection_row(db, sop_id=sop.id, sop_version_id=version.id)

        pd_ser = serialize_profile_detection_row(pd_row)
        pb_chars = len((pd_row.prompt_block or "")) if pd_row else 0
        log_metadata_load(
            sop_id=sop.id,
            sop_version_id=version.id if version else None,
            metadata_keys=meta_keys,
            prompt_block_chars=pb_chars,
        )
        if version and (compact_vm or pd_ser):
            log_metadata_merge(
                sop_id=sop.id,
                sop_version_id=version.id,
                merged_keys=meta_keys + (["profile_detection"] if pd_ser else []),
                prompt_block_chars=pb_chars,
            )

        stored_nlp = None
        if pd_row and isinstance(pd_row.nlp_analysis_json, dict):
            stored_nlp = pd_row.nlp_analysis_json
        if stored_nlp is None:
            legacy = meta.get("nlp_analysis") if isinstance(meta.get("nlp_analysis"), dict) else None
            stored_nlp = legacy

        return {
            "detected": True,
            "sop_id": str(sop.id),
            "sop_number": sop.sop_number,
            "title": sop.title,
            "version_id": str(version.id) if version else None,
            "text": sop_text or "",
            "nlp_analysis": stored_nlp,
            "version_metadata_compact": compact_vm,
            "version_metadata_keys": meta_keys,
            "profile_detection": pd_ser,
        }
    finally:
        db.close()


def _ensure_profile_detection_row(sop_ctx: dict[str, Any], action: str) -> None:
    """If no active ProfileDetection row, persist one (NLP + metadata) then refresh sop_ctx."""
    if action not in ("improve", "rewrite", "gap_check", "summarize", "analyze"):
        return
    if not sop_ctx.get("detected") or not sop_ctx.get("version_id"):
        return
    if sop_ctx.get("profile_detection"):
        return
    try:
        vid = uuid.UUID(str(sop_ctx["version_id"]))
        sid = uuid.UUID(str(sop_ctx["sop_id"]))
    except Exception:
        return
    db = SessionLocal()
    try:
        v = db.query(SOPVersion).filter(SOPVersion.id == vid).first()
        if not v:
            return
        persist_profile_detection_for_sop_version(db, v)
        pd_row = load_active_profile_detection_row(db, sop_id=sid, sop_version_id=vid)
        ser = serialize_profile_detection_row(pd_row)
        if ser:
            sop_ctx["profile_detection"] = ser
            nlpj = ser.get("nlp_analysis_json")
            if isinstance(nlpj, dict):
                sop_ctx["nlp_analysis"] = nlpj
    except Exception as exc:
        logger.warning("[profile-detection] ensure_if_missing failed err=%s", exc)
    finally:
        db.close()


def _try_client_structured_ai_response(
    payload: AIActionRequest,
    action: str,
    request: ActionRequest,
    style_profile: dict[str, Any],
) -> AIActionResponse | None:
    data = getattr(payload, "client_structured_json", None)
    if not isinstance(data, dict) or not data:
        return None
    try:
        if action == "improve":
            txt = str(data.get("improved_text") or data.get("improved_version") or "").strip()
            if len(txt) < 2:
                return None
            ImproveResponse.model_validate({"improved_text": txt})
            return AIActionResponse(
                action="improve",
                original_text=request.section_text,
                suggested_text=_render_dynamic_text(txt),
                explanation="Applied structured improvement from client (validated; no new LLM call).",
                structured_data={
                    "improved_text": txt,
                    "improved_version": txt,
                    "style_profile": style_profile,
                    "client_supplied": True,
                },
            )
        if action == "rewrite":
            txt = str(data.get("rewritten_text") or "").strip()
            if len(txt) < 2:
                return None
            RewriteResponse.model_validate({"rewritten_text": txt})
            return AIActionResponse(
                action="rewrite",
                original_text=request.section_text,
                suggested_text=_render_dynamic_text(txt),
                explanation="Applied structured rewrite from client (validated; no new LLM call).",
                structured_data={
                    "rewritten_text": txt,
                    "style_profile": style_profile,
                    "client_supplied": True,
                },
            )
        if action == "gap_check":
            txt = str(data.get("analysis") or "").strip()
            if len(txt) < 10:
                return None
            GapCheckResponse.model_validate({"analysis": txt})
            return AIActionResponse(
                action="gap_check",
                original_text=request.section_text,
                suggested_text=_render_gap_check_analysis_html(txt),
                explanation="Applied structured gap analysis from client (validated; no new LLM call).",
                structured_data={
                    "analysis": txt,
                    "style_profile": style_profile,
                    "client_supplied": True,
                },
            )
        if action in ("summarize", "analyze"):
            txt = str(data.get("improved_text") or data.get("improved_version") or "").strip()
            if len(txt) < 2:
                return None
            ImproveResponse.model_validate({"improved_text": txt})
            return AIActionResponse(
                action=action,
                original_text=request.section_text,
                suggested_text=_render_dynamic_text(txt),
                explanation="Applied structured client payload (validated; no new LLM call).",
                structured_data={
                    "improved_text": txt,
                    "improved_version": txt,
                    "style_profile": style_profile,
                    "client_supplied": True,
                },
            )
    except ValidationError as ve:
        logger.warning("[ai-action] client_structured_json rejected: %s", ve)
        return None
    return None


def _build_gap_check_retrieval_query(request: ActionRequest) -> str:
    parts = [
        f"SOP: {request.sop_title}",
        f"Section: {request.section_title}",
        f"Type: {request.section_type}",
        request.section_text,
    ]
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _rewrite_should_use_industry_scaffold(request: ActionRequest) -> bool:
    # Rewrite should preserve the current SOP structure by default.
    # Automatic scaffold mode can reshape the document, so keep it disabled
    # unless a future explicit "restructure" flow is added.
    return False


def _first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        cleaned = line.strip(" \t#*")
        if cleaned:
            return cleaned
    return "Untitled SOP"


def _split_traceability_records(text: str) -> tuple[str, str]:
    marker = re.search(
        r"(?im)^\s*(?:[^\w\s]?\s*)?(?:DEVIATIONS|CAPAS|AUDIT FINDINGS|DECISIONS)\b",
        text or "",
    )
    if not marker:
        return text.strip(), ""
    return text[: marker.start()].strip(), text[marker.start() :].strip()


def _extract_existing_purpose(main_text: str) -> str:
    lines = [line.strip() for line in (main_text or "").splitlines()]
    body: list[str] = []
    capture = False
    for line in lines:
        if not line:
            continue
        if re.search(r"\b(zweck|purpose)\b", line, flags=re.IGNORECASE):
            capture = True
            continue
        if capture and re.match(r"^\s*(?:\d+[.)]\s+|#{1,6}\s*)", line):
            break
        if capture:
            body.append(line)
    if body:
        return " ".join(body).strip()
    non_meta = [
        line
        for line in lines[1:8]
        if line and not re.search(r"\b(version|status|department|sop id|titel)\b", line, flags=re.IGNORECASE)
    ]
    return " ".join(non_meta).strip()


def _looks_like_structured_full_sop(text: str) -> bool:
    headers = re.findall(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\d+[.)]\s+|##\s*\*\*)\s*(?:[A-ZÄÖÜa-zäöü0-9]+)",
        text or "",
    )
    return len(headers) >= 6


def _build_industry_rewrite_text(request: ActionRequest) -> str:
    source = (request.section_text or "").strip()
    main_text, records_text = _split_traceability_records(source)
    if not records_text and _looks_like_structured_full_sop(main_text):
        return "\n\n".join(
            [
                main_text,
                "Industry Rewrite Completion Controls\n"
                "- Kontroll- und Akzeptanzkriterien: [Zu definieren: messbare Grenzwerte, Fristen/SLA, Review-Frequenz und Akzeptanzkriterien, sofern nicht bereits oben festgelegt].\n"
                "- Dokumentationskontrolle: Alle qualitaetsrelevanten Nachweise muessen eindeutig versioniert, revisionssicher gespeichert und durch verantwortliche Rollen pruefbar sein.\n"
                "- Lifecycle/Freigabe: [Zu definieren: Review-Frequenz, naechster Review-Termin, Genehmiger, Wirksamkeitspruefung und Obsoleszenzverfahren].\n"
                "- Traceability Records: [Zu definieren: zugehoerige Abweichungen, CAPAs, Audit Findings und Entscheidungen, falls anwendbar].",
            ]
        ).strip()

    title_line = _first_nonempty_line(source)
    purpose = _extract_existing_purpose(main_text) or "[Zu definieren: Zweck und regulatorischer Kontrollzweck vor Freigabe ergaenzen]"
    metadata_lines = [
        line.strip()
        for line in main_text.splitlines()[1:6]
        if re.search(r"\b(version|status|department|gmp|kritikal|sop id|titel)\b", line, flags=re.IGNORECASE)
    ]
    metadata = "\n".join(metadata_lines).strip()
    records_section = records_text or "[Zu definieren: zugehoerige Abweichungen, CAPAs, Audit Findings und Entscheidungen, falls anwendbar]"

    return "\n\n".join(
        [
            title_line,
            metadata,
            "1. Zweck\n"
            f"{purpose}",
            "2. Geltungsbereich\n"
            "Diese SOP gilt fuer alle Rollen, Systeme, Prozesse und externen Parteien, die im beschriebenen Prozesskontext genannt sind. "
            "[Zu definieren: konkrete Standorte, Systeme, Rollen und Ausnahmen vor SOP-Freigabe bestaetigen].",
            "3. Begriffe und Abkuerzungen\n"
            "- SOP: Standard Operating Procedure.\n"
            "- QA: Qualitaetssicherung.\n"
            "- CAPA: Corrective and Preventive Action.\n"
            "- [Zu definieren: weitere prozessspezifische Begriffe, Systeme und Rollen].",
            "4. Verantwortlichkeiten\n"
            "- Process Owner: verantwortet die fachliche Vollstaendigkeit, Aktualitaet und Umsetzung dieser SOP.\n"
            "- QA: prueft die Compliance-Relevanz, genehmigt qualitaetsrelevante Entscheidungen und bewertet Abweichungen/CAPAs.\n"
            "- Ausfuehrende Rolle: fuehrt die beschriebenen Schritte gemaess dieser SOP aus und dokumentiert die erforderlichen Nachweise.\n"
            "- IT/Produktion/Abteilungsleitung: uebernimmt Aufgaben, sofern diese im SOP-Kontext oder in den Traceability Records genannt sind.\n"
            "- [Zu definieren: finale Rollenmatrix, Stellvertretungen und Eskalationsweg].",
            "5. Verfahren\n"
            "1. Der Process Owner stellt sicher, dass der Prozess nur innerhalb des genehmigten Geltungsbereichs ausgefuehrt wird.\n"
            "2. Die verantwortliche Rolle prueft vor Ausfuehrung die Berechtigung, den Anlass, die Kritikalitaet und die erforderlichen Freigaben.\n"
            "3. Qualitaetsrelevante Aktivitaeten muessen mit eindeutigem Datum, Rolle, System/Prozess, Begruendung und Ergebnis dokumentiert werden.\n"
            "4. Abweichungen vom genehmigten Ablauf muessen unverzueglich als Deviation erfasst, bewertet und bei Bedarf mit CAPA verknuepft werden.\n"
            "5. QA oder die definierte freigabeberechtigte Rolle prueft kritische Entscheidungen, Ausnahmen und offene Massnahmen vor Abschluss.\n"
            "6. Der Process Owner ueberwacht offene Punkte bis zur Wirksamkeitspruefung und dokumentiert den Abschluss.\n"
            "7. [Zu definieren: detaillierte operative Schrittfolge, Systeme/Formulare und Entscheidungskriterien].",
            "6. Kontroll- und Akzeptanzkriterien\n"
            "- Jede Aktivitaet muss einer verantwortlichen Rolle, einem Nachweis und einem nachvollziehbaren Ergebnis zugeordnet sein.\n"
            "- Kritische oder qualitaetsrelevante Ausnahmen benoetigen dokumentierte Begruendung und QA-Bewertung.\n"
            "- Offene Deviations/CAPAs duerfen nicht ohne dokumentierte Risikobewertung und Nachverfolgung geschlossen werden.\n"
            "- [Zu definieren: messbare Grenzwerte, Fristen/SLA, Review-Frequenz und Akzeptanzkriterien].",
            "7. Dokumentation und Aufbewahrung\n"
            "- Erforderliche Nachweise: SOP-Version, Freigaben, Durchfuehrungsnachweise, Deviation/CAPA/Audit/Decision-Records und Wirksamkeitspruefungen.\n"
            "- Alle Aufzeichnungen muessen revisionssicher, nachvollziehbar und gegen unbefugte Aenderung geschuetzt gespeichert werden.\n"
            "- [Zu definieren: Formularnamen, Ablageort, System of Record und Aufbewahrungsfrist].",
            "8. Schulung\n"
            "Alle betroffenen Rollen muessen vor Anwendung dieser SOP und nach wesentlichen Aenderungen geschult werden. "
            "Die Schulung ist mit Datum, Teilnehmer, Version und Schulungsnachweis zu dokumentieren.",
            "9. Review, Freigabe und Lifecycle\n"
            "Diese SOP muss vor Inkraftsetzung fachlich und durch QA freigegeben werden. "
            "[Zu definieren: Review-Frequenz, naechster Review-Termin, Genehmiger und Obsoleszenzverfahren].",
            "10. Traceability Records / Anhaenge\n"
            f"{records_section}",
        ]
    ).strip()


def _split_industry_scaffold_for_llm(scaffold: str) -> tuple[str, str]:
    markers = [
        "\n10. Traceability Records / Anhaenge\n",
        "\nIndustry Rewrite Completion Controls\n",
    ]
    for marker in markers:
        idx = scaffold.find(marker)
        if idx >= 0:
            return scaffold[:idx].strip(), scaffold[idx:].strip()
    return scaffold.strip(), ""


def _build_industry_rewrite_llm_prompt(request: ActionRequest, scaffold_core: str) -> str:
    return f"""You are a senior GMP/QA SOP writer. Rewrite the following SOP core into polished, industry-ready SOP language.
Return exactly one valid JSON object and nothing else: {{"rewritten_text":"..."}}

Rules:
- Use the same language as the SOP core.
- Keep SOP number, title, version/status, department, roles, systems, thresholds, dates, and identifiers unchanged.
- Preserve bracketed placeholders exactly when facts are missing; do not invent missing owners, forms, retention periods, dates, systems, limits, or approvals.
- Keep the industry SOP backbone complete: purpose, scope, definitions, responsibilities, procedure, controls/acceptance criteria, documentation/records, training, review/approval/lifecycle.
- Improve wording, flow, accountability, mandatory language, audit readiness, and professional SOP style.
- Keep the rewritten core concise; do not expand placeholders or add long explanatory rationale.
- Do not include markdown fences, explanations, sources, or citations.
- Encode line breaks inside JSON strings as \\n.

SOP title: {request.sop_title}
Section: {request.section_title} ({request.section_type})

SOP CORE TO REWRITE:
\"\"\"{scaffold_core}\"\"\""""


def _restore_missing_identifiers(original_text: str, rewritten_text: str, traceability_text: str = "") -> str:
    source_ids = sorted(set(re.findall(r"\b(?:SOP|DEV|CAPA|AUD|DEC)-[A-Z]+-\d+\b", original_text or "")))
    if not source_ids:
        return rewritten_text
    out = rewritten_text or ""
    missing = [item for item in source_ids if item not in out]
    if not missing:
        return out
    if traceability_text and all(item in traceability_text for item in missing):
        return "\n\n".join(part for part in [out.strip(), traceability_text.strip()] if part)
    return out.rstrip() + "\n\nTraceability IDs preserved from source:\n" + "\n".join(f"- {item}" for item in missing)


def _rewrite_industry_scaffold_with_llm(runtime: Any, request: ActionRequest, scaffold: str) -> tuple[str, bool, str | None]:
    core, traceability = _split_industry_scaffold_for_llm(scaffold)
    prompt = _build_industry_rewrite_llm_prompt(request, core)
    audit_log: list[dict[str, Any]] = []
    raw = _call_action_llm(runtime, prompt, input_char_budget=len(core), action="rewrite_core")
    parsed = parse_with_retry(
        raw=raw,
        schema=RewriteResponse,
        prompt=prompt,
        call_llm=lambda rp: _call_action_llm(runtime, rp, input_char_budget=len(core), action="rewrite_core"),
        audit_log=audit_log,
    )
    llm_text = (parsed.rewritten_text or "").strip()
    if not llm_text:
        return scaffold, False, "empty_llm_rewrite"
    combined = "\n\n".join(part for part in [llm_text, traceability] if part.strip())
    combined = _restore_missing_identifiers(request.section_text, combined, traceability)
    return combined.strip(), True, None


def _get_nlp_and_profile_context(sop_ctx: dict) -> tuple[dict | None, dict | None, str | None]:
    """
    Extracts the detected NLP parameters and the client profile JSON and MD from the database.
    """
    detected_nlp = None
    profile_json = None
    profile_md = None

    sop_id = sop_ctx.get("sop_id")
    sop_version_id = sop_ctx.get("version_id") or sop_ctx.get("sop_version_id")
    if sop_id:
        db = SessionLocal()
        try:
            detected_query = db.query(SOPDetectedParameters).filter(
                SOPDetectedParameters.sop_id == uuid.UUID(str(sop_id))
            )
            if sop_version_id:
                scoped_row = detected_query.filter(
                    SOPDetectedParameters.sop_version_id == uuid.UUID(str(sop_version_id))
                ).order_by(SOPDetectedParameters.created_at.desc()).first()
            else:
                scoped_row = None
            detected_row = scoped_row or detected_query.order_by(SOPDetectedParameters.created_at.desc()).first()
            if detected_row:
                detected_nlp = {
                    "document_information": detected_row.document_information,
                    "writing_style": detected_row.writing_style,
                    "roles_raci": detected_row.roles_raci,
                    "workflows": detected_row.workflows,
                    "compliance_elements": detected_row.compliance_elements,
                    "risks_gaps": detected_row.risks_gaps,
                    "terminology": detected_row.terminology,
                    "structure_patterns": detected_row.structure_patterns,
                    "style_suggestions": detected_row.style_suggestions,
                    "readiness_check": detected_row.readiness_check,
                }
                # Try to load profile associated with this SOP
                profile_id = detected_row.client_profile_id
                if profile_id:
                    profile = db.query(ClientProfile).filter(
                        ClientProfile.id == profile_id
                    ).first()
                    if profile:
                        profile_json, profile_md, profile_version, profile_id_str = _load_profile_payload(db, profile)
                        if isinstance(profile_json, dict):
                            profile_json["_profile_id"] = profile_id_str
                            profile_json["_profile_version"] = profile_version
            
            # Fallback to load tenant profile if no profile was loaded yet
            if not profile_json:
                sop_row = db.query(SOP).filter(SOP.id == uuid.UUID(str(sop_id))).first()
                if sop_row and sop_row.tenant_id:
                    profile = db.query(ClientProfile).filter(
                        ClientProfile.tenant_id == sop_row.tenant_id
                    ).order_by(ClientProfile.updated_at.desc()).first()
                    if profile:
                        profile_json, profile_md, profile_version, profile_id_str = _load_profile_payload(db, profile)
                        if isinstance(profile_json, dict):
                            profile_json["_profile_id"] = profile_id_str
                            profile_json["_profile_version"] = profile_version
        except Exception as e:
            logger.warning("[ai-routes] Failed to load NLP / ClientProfile context: %s", e)
        finally:
            db.close()
            
    return detected_nlp, profile_json, profile_md


def _serialize_detected_parameters_row(row: SOPDetectedParameters | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "document_information": row.document_information,
        "writing_style": row.writing_style,
        "roles_raci": row.roles_raci,
        "workflows": row.workflows,
        "compliance_elements": row.compliance_elements,
        "risks_gaps": row.risks_gaps,
        "terminology": row.terminology,
        "structure_patterns": row.structure_patterns,
        "style_suggestions": row.style_suggestions,
        "readiness_check": row.readiness_check,
    }


def _normalize_style_reference(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    raw = re.sub(r"\.pdf\b", "", raw)
    raw = re.sub(r"\b(company|profile|style|tone)\b", " ", raw)
    raw = re.sub(r"[^a-z0-9]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw


def _normalize_language_code(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    normalized = re.sub(r"[^a-z/-]+", " ", raw).strip()
    if normalized in {"de", "de de", "german", "deutsch"}:
        return "de"
    if normalized in {"en", "en us", "en gb", "english"}:
        return "en"
    if normalized.startswith(("german", "deutsch", "de ")):
        return "de"
    if normalized.startswith(("english", "en ")):
        return "en"
    return ""


def _levenshtein_distance(left: str, right: str) -> int:
    a = str(left or "")
    b = str(right or "")
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )
        previous = current
    return previous[-1]


def _extract_explicit_style_reference(instruction: str | None) -> str | None:
    text = str(instruction or "").strip()
    if not text:
        return None
    from chatbot.assistant.profile_reference import extract_editorial_profile_reference

    editorial = extract_editorial_profile_reference(text)
    if editorial:
        return editorial
    if re.search(r"\bgerman\s+(?:pharma|pharmaceutical)\s+(?:sop\s+)?profile\b", text, re.IGNORECASE):
        return GERMAN_PROFILE_NAME
    patterns = [
        r'\b(?:in|on|using|use)\s+"([^"]+)"\s+style\b',
        r"\b(?:in|on|using|use)\s+'([^']+)'\s+style\b",
        r"\b(?:in|on|using|use)\s+([a-z0-9._/-]+(?:\s+[a-z0-9._/-]+){0,4})\s+company\s+style\b",
        r"\b(?:in|on|using|use)\s+([a-z0-9._/-]+(?:\s+[a-z0-9._/-]+){0,4})\s+profile\s+style\b",
        r"\b(?:in|on|using|use)\s+([a-z0-9._/-]+)\s+style\b",
        r"\b([a-z0-9._/-]+(?:\s+[a-z0-9._/-]+){0,4})\s+company\s+style\b",
        r"\b([a-z0-9._/-]+(?:\s+[a-z0-9._/-]+){0,4})\s+profile\s+style\b",
        r"\bstyle\s+of\s+([a-z0-9._/-]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            ref = match.group(1).strip()
            # Loose "ABC company style" matches can begin earlier in a full instruction.
            # Keep the actual style reference after the last command/preposition marker.
            ref = re.sub(
                r"^(?:rewrite|improve|summarize|shorten|expand|change|update|make|this|the|sop|section|document)\b\s*",
                "",
                ref,
                flags=re.IGNORECASE,
            ).strip()
            for marker in (" on ", " in ", " using ", " use "):
                if marker in f" {ref.lower()} ":
                    ref = re.split(rf"\b{marker.strip()}\b", ref, flags=re.IGNORECASE)[-1].strip()
            return ref
    return None


def _style_ref_matches(normalized_ref: str, candidate_tokens: set[str]) -> bool:
    for token in candidate_tokens:
        if not token:
            continue
        if normalized_ref == token:
            return True
        if normalized_ref in token or token in normalized_ref:
            return True
    return False


def _resolve_explicit_style_override(instruction: str | None) -> dict[str, Any] | None:
    ref = _extract_explicit_style_reference(instruction)
    normalized_ref = _normalize_style_reference(ref or "")
    if not normalized_ref:
        return None

    db = SessionLocal()
    try:
        from chatbot.assistant.profile_reference import editorial_profile_sop_number_hint

        sop_num_hint = editorial_profile_sop_number_hint(ref or "")
        if sop_num_hint:
            sop = db.query(SOP).filter(SOP.sop_number == sop_num_hint).first()
            if sop:
                row = (
                    db.query(SOPDetectedParameters)
                    .filter(SOPDetectedParameters.sop_id == sop.id)
                    .order_by(SOPDetectedParameters.created_at.desc())
                    .first()
                )
                profile = (
                    db.query(ClientProfile).filter(ClientProfile.id == row.client_profile_id).first()
                    if row and row.client_profile_id
                    else None
                )
                if profile:
                    current_version = (
                        db.query(ProfileVersion).filter(ProfileVersion.id == profile.current_version_id).first()
                        if profile.current_version_id
                        else None
                    )
                    profile_json, profile_md, profile_version, profile_id_str = _load_profile_payload(db, profile)
                    return {
                        "matched_type": "editorial_sop_profile",
                        "style_reference": ref,
                        "resolved_name": profile.name,
                        "profile_id": profile_id_str,
                        "profile_json": profile_json,
                        "profile_md": profile_md,
                        "profile_version": profile_version,
                        "detected_nlp": _serialize_detected_parameters_row(row) if row else None,
                        "style_source_text": profile_md or "",
                        "source_sop_id": str(sop.id),
                        "source_sop_number": sop.sop_number,
                    }

        # 1) Exact-ish profile match by display names.
        for profile in db.query(ClientProfile).all():
            candidates = [
                profile.name,
                profile.company_name,
                (profile.active_profile_json or {}).get("profile_name") if isinstance(profile.active_profile_json, dict) else None,
                (profile.active_profile_json or {}).get("name") if isinstance(profile.active_profile_json, dict) else None,
            ]
            candidate_tokens = {_normalize_style_reference(item or "") for item in candidates if item}
            profile_md_token = _normalize_style_reference((profile.active_profile_md or "")[:4000])
            if _style_ref_matches(normalized_ref, candidate_tokens) or (
                profile_md_token and normalized_ref in profile_md_token
            ):
                profile_json, profile_md, profile_version, profile_id_str = _load_profile_payload(db, profile)
                return {
                    "matched_type": "profile",
                    "style_reference": ref,
                    "resolved_name": profile.name,
                    "profile_id": profile_id_str,
                    "profile_json": profile_json,
                    "profile_md": profile_md,
                    "profile_version": profile_version,
                    "detected_nlp": None,
                    "style_source_text": profile_md or "",
                }

        # 2) SOP/source-file match for explicit source SOP style.
        detected_rows = (
            db.query(SOPDetectedParameters)
            .order_by(SOPDetectedParameters.created_at.desc())
            .all()
        )
        for row in detected_rows:
            sop = db.query(SOP).filter(SOP.id == row.sop_id).first() if row.sop_id else None
            candidates = [
                row.source_filename,
                row.client_name,
                sop.sop_number if sop else None,
                sop.title if sop else None,
            ]
            candidate_tokens = {_normalize_style_reference(item or "") for item in candidates if item}
            if _style_ref_matches(normalized_ref, candidate_tokens):
                profile = (
                    db.query(ClientProfile).filter(ClientProfile.id == row.client_profile_id).first()
                    if row.client_profile_id
                    else None
                )
                current_version = (
                    db.query(ProfileVersion).filter(ProfileVersion.id == profile.current_version_id).first()
                    if profile and profile.current_version_id
                    else None
                )
                source_version = db.query(SOPVersion).filter(SOPVersion.id == row.sop_version_id).first() if row.sop_version_id else None
                source_text = ""
                if source_version and source_version.content_json:
                    source_text = _extract_text_from_tiptap(source_version.content_json)
                return {
                    "matched_type": "source_sop",
                    "style_reference": ref,
                    "resolved_name": sop.sop_number if sop else row.source_filename or ref,
                    "profile_id": str(profile.id) if profile else None,
                    "profile_json": profile.active_profile_json if profile else None,
                    "profile_md": profile.active_profile_md if profile else None,
                    "profile_version": current_version.version_number if current_version else None,
                    "detected_nlp": _serialize_detected_parameters_row(row),
                    "style_source_text": source_text or row.source_filename or "",
                    "source_sop_id": str(sop.id) if sop else None,
                    "source_sop_number": sop.sop_number if sop else None,
                }

        for sop in db.query(SOP).all():
            candidates = [sop.sop_number, sop.title]
            candidate_tokens = {_normalize_style_reference(item or "") for item in candidates if item}
            if not _style_ref_matches(normalized_ref, candidate_tokens):
                continue
            row = (
                db.query(SOPDetectedParameters)
                .filter(SOPDetectedParameters.sop_id == sop.id)
                .order_by(SOPDetectedParameters.created_at.desc())
                .first()
            )
            profile = (
                db.query(ClientProfile).filter(ClientProfile.id == row.client_profile_id).first()
                if row and row.client_profile_id
                else None
            )
            current_version = (
                db.query(ProfileVersion).filter(ProfileVersion.id == profile.current_version_id).first()
                if profile and profile.current_version_id
                else None
            )
            source_version = db.query(SOPVersion).filter(SOPVersion.id == sop.current_version_id).first() if sop.current_version_id else None
            source_text = _extract_text_from_tiptap(source_version.content_json) if source_version and source_version.content_json else ""
            return {
                "matched_type": "source_sop",
                "style_reference": ref,
                "resolved_name": sop.sop_number or sop.title or ref,
                "profile_id": str(profile.id) if profile else None,
                "profile_json": profile.active_profile_json if profile else None,
                "profile_md": profile.active_profile_md if profile else None,
                "profile_version": current_version.version_number if current_version else None,
                "detected_nlp": _serialize_detected_parameters_row(row) if row else None,
                "style_source_text": source_text or sop.title or "",
                "source_sop_id": str(sop.id),
                "source_sop_number": sop.sop_number,
            }
    except Exception as exc:
        logger.warning("[ai-style-override] failed to resolve style reference '%s': %s", ref, exc)
    finally:
        db.close()
    return None


GERMAN_PROFILE_NAME = "German_Pharma_SOP_Profile"


def _profile_display_name(profile_json: dict | None, fallback: str | None = None) -> str | None:
    if isinstance(profile_json, dict):
        for key in ("profile_name", "name", "client_name", "company_name"):
            value = str(profile_json.get(key) or "").strip()
            if value:
                return value
    return fallback


def _load_profile_payload(db: Any, profile: ClientProfile | None) -> tuple[dict | None, str | None, int | None, str | None]:
    if not profile:
        return None, None, None, None
    version = (
        db.query(ProfileVersion).filter(ProfileVersion.id == profile.current_version_id).first()
        if profile.current_version_id
        else None
    )
    profile_json = dict(profile.active_profile_json or {}) if isinstance(profile.active_profile_json, dict) else {}
    if profile.name and not profile_json.get("profile_name"):
        profile_json["profile_name"] = profile.name
    if profile.company_name and not profile_json.get("company_name"):
        profile_json["company_name"] = profile.company_name
    return profile_json or profile.active_profile_json, profile.active_profile_md, int(version.version_number) if version else None, str(profile.id)
GERMAN_MODAL_PATTERN = re.compile(r"\b(muss|müssen|sollte|sollten|darf\s+nicht|dürfen\s+nicht)\b", re.IGNORECASE)


def _is_german_pharma_profile(profile_json: dict | None, profile_md: str | None, style_profile: dict | None) -> bool:
    haystack = " ".join(
        str(part or "")
        for part in [
            (profile_json or {}).get("profile_name") if isinstance(profile_json, dict) else "",
            (profile_json or {}).get("name") if isinstance(profile_json, dict) else "",
            (profile_json or {}).get("language") if isinstance(profile_json, dict) else "",
            (profile_json or {}).get("domain") if isinstance(profile_json, dict) else "",
            profile_md[:1200] if profile_md else "",
            (style_profile or {}).get("language") if isinstance(style_profile, dict) else "",
        ]
    ).lower()
    return (
        GERMAN_PROFILE_NAME.lower() in haystack
        or ("german" in haystack and ("pharma" in haystack or "gxp" in haystack or "sop" in haystack))
        or ("de" in haystack and "gxp" in haystack)
    )


def _has_german_modal_language(text: str) -> bool:
    return bool(GERMAN_MODAL_PATTERN.search(text or ""))


def _has_german_profile_register_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\b(Standardarbeitsanweisung|GxP|GMP|ICH\s*Q9|QA|Reviewer|Approver|Aufzeichnungen|Freigabe|Verfahren|Verantwortlich)\b",
            text or "",
            re.IGNORECASE,
        )
    )


def _german_profile_quality_retry_prompt(request: ActionRequest, draft: str, action: str) -> str:
    verb = "rewrite" if action == "rewrite" else "improve"
    field_name = "rewritten_text" if action == "rewrite" else "improved_text"
    return f"""Return strict JSON only with key "{field_name}".

Revise the DRAFT so it is still a {verb} of the selected SOP section, but make the German pharmaceutical SOP profile visible.

Hard requirements:
- Keep the same operational meaning and identifiers.
- Use formal controlled SOP language in German.
- Include appropriate German modal/control wording such as "muss", "sollte", or "darf nicht" where it fits the existing facts.
- Keep the output scoped to the selected text only.
- Do not invent regulation numbers, CAPA IDs, deviation IDs, or approvals.

SELECTED TEXT:
{request.section_text}

DRAFT:
{draft}
"""


def _apply_german_profile_quality_guard(
    runtime: Any,
    *,
    action: str,
    request: ActionRequest,
    text: str,
    profile_json: dict | None,
    profile_md: str | None,
    style_profile: dict | None,
    ch_budget: int,
) -> tuple[str, dict[str, Any]]:
    meta = {
        "german_profile_quality_guard": False,
        "modal_language_present": _has_german_modal_language(text),
        "profile_register_signal_present": _has_german_profile_register_signal(text),
        "quality_retry_used": False,
        "deterministic_modal_sentence_added": False,
        "deterministic_register_sentence_added": False,
    }
    if not _is_german_pharma_profile(profile_json, profile_md, style_profile):
        return text, meta
    meta["german_profile_quality_guard"] = True
    if meta["modal_language_present"] and meta["profile_register_signal_present"]:
        return text, meta

    schema = RewriteResponse if action == "rewrite" else ImproveResponse
    prompt = _german_profile_quality_retry_prompt(request, text, action)
    try:
        parsed = parse_with_retry(
            raw=_call_action_llm(runtime, prompt, input_char_budget=ch_budget, action=f"{action}_quality_retry"),
            schema=schema,
            prompt=prompt,
            call_llm=lambda rp: _call_action_llm(
                runtime, rp, input_char_budget=ch_budget, action=f"{action}_quality_retry"
            ),
            audit_log=[],
        )
        retried = (getattr(parsed, "rewritten_text", None) or getattr(parsed, "improved_text", None) or "").strip()
        if retried:
            text = retried
            meta["quality_retry_used"] = True
            meta["modal_language_present"] = _has_german_modal_language(text)
            meta["profile_register_signal_present"] = _has_german_profile_register_signal(text)
    except Exception as exc:
        logger.warning("[german-profile-quality] retry failed action=%s err=%s", action, exc)

    if not meta["modal_language_present"]:
        text = (
            text.rstrip()
            + "\n\nKontrollanforderung: Die beschriebenen Maßnahmen müssen dokumentiert, nachvollziehbar geprüft und bei Abweichungen durch die verantwortliche Stelle bewertet werden."
        )
        meta["deterministic_modal_sentence_added"] = True
        meta["modal_language_present"] = True
    if not meta["profile_register_signal_present"]:
        text = (
            text.rstrip()
            + "\n\nRegisterhinweis: Die Umsetzung ist im Rahmen dieser Standardarbeitsanweisung nachvollziehbar zu dokumentieren."
        )
        meta["deterministic_register_sentence_added"] = True
        meta["profile_register_signal_present"] = True
    return text.strip(), meta


def _shorten_previous_suggestion_if_requested(payload: AIActionRequest, text: str) -> str | None:
    marker = f"{getattr(payload, 'triggered_by', '')} {getattr(payload, 'section_name', '')}".lower()
    if "followup" not in marker and "phase9" not in marker and "previous rewrite suggestion" not in marker:
        return None
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text or "") if p.strip()]
    if not paragraphs:
        return None
    first_lines = []
    for para in paragraphs[:4]:
        sentences = re.split(r"(?<=[.!?])\s+", para)
        first_lines.append(sentences[0].strip())
    compact = "\n\n".join(line for line in first_lines if line)
    if compact and len(compact) < len(text):
        if not _has_german_modal_language(compact):
            compact += "\n\nDie Maßnahmen müssen dokumentiert und regelmäßig geprüft werden."
        return compact
    return None


def _extract_leading_section_heading(text: str) -> str | None:
    src = (text or "").strip()
    if not src:
        return None
    first_line = src.splitlines()[0].strip()
    if re.match(r"^\d+\.\s*\S+", first_line):
        return first_line
    if re.match(r"^[A-ZÄÖÜa-zäöü][A-ZÄÖÜa-zäöü0-9 _/-]{1,80}$", first_line) and len(first_line.split()) <= 8:
        return first_line
    return None


def _preserve_selected_section_heading(source_text: str, rewritten_text: str) -> str:
    heading = _extract_leading_section_heading(source_text)
    out = (rewritten_text or "").strip()
    if not heading or not out:
        return rewritten_text
    if re.search(re.escape(heading), out, re.IGNORECASE):
        return rewritten_text

    source_has_label = bool(re.search(r"\bZweck\b|\bGeltungsbereich\b|\bVerantwortlichkeiten\b", heading, re.IGNORECASE))
    if source_has_label:
        lines = out.splitlines()
        if lines and re.match(r"^\d+\.\s*$", lines[0].strip()):
            lines[0] = heading
            return "\n".join(lines).strip()
        return f"{heading}\n\n{out}".strip()
    return rewritten_text


def _restore_section_name_heading(request: ActionRequest, rewritten_text: str) -> str:
    out = (rewritten_text or "").strip()
    section_name = (request.section_title or "").strip()
    if not out or not section_name or section_name.lower() in {"selected text", "section", "previous rewrite suggestion"}:
        return rewritten_text
    if re.search(re.escape(section_name), out, re.IGNORECASE):
        return rewritten_text

    source = (request.section_text or "").strip()
    prefix_match = re.match(r"^(\d+)\.\s*", source)
    numbered_heading = f"{prefix_match.group(1)}. {section_name}" if prefix_match else section_name
    lines = out.splitlines()
    if lines and re.match(r"^\d+\.\s*$", lines[0].strip()):
        lines[0] = numbered_heading
        return "\n".join(lines).strip()
    if lines and not re.match(r"^\d+\.\s+\S+", lines[0].strip()):
        return f"{numbered_heading}\n\n{out}".strip()
    return rewritten_text


def _normalize_structure_token(value: str) -> str:
    token = re.sub(r"\s+", " ", str(value or "").strip().lower())
    token = re.sub(r"^[#*\-\d\.\)\s]+", "", token)
    return token


def _extract_structure_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^\d+(?:\.\d+)*[.)]?\s+\S+", line):
            tokens.append(_normalize_structure_token(line))
            continue
        if re.match(r"^(?:[-*•]|\d+[.)])\s+\S+", line):
            tokens.append(_normalize_structure_token(line))
            continue
        if re.match(r"^[A-ZÄÖÜa-zäöü][A-ZÄÖÜa-zäöü0-9 /_-]{1,90}$", line) and len(line.split()) <= 8:
            tokens.append(_normalize_structure_token(line))
    return tokens[:40]


def _has_rewrite_structure_drift(original: str, rewritten: str) -> bool:
    source_tokens = _extract_structure_tokens(original)
    output_tokens = _extract_structure_tokens(rewritten)
    if not source_tokens or not output_tokens:
        return False
    if source_tokens[0] != output_tokens[0]:
        return True
    if len(source_tokens) <= 8:
        return source_tokens != output_tokens[: len(source_tokens)]
    return source_tokens[:8] != output_tokens[:8]


def _enforce_rewrite_structure_lock(
    runtime: Any,
    *,
    request: ActionRequest,
    text: str,
    context: str,
    nlp_block: str,
    profile_md: str,
    profile_json: dict | None,
    detected_nlp: dict | None,
    style_profile: dict | None,
    ch_budget: int,
) -> tuple[str, dict[str, Any]]:
    original = request.section_text or ""
    rewritten = (text or "").strip()
    meta = {"structure_lock_applied": False, "structure_drift_detected": False}
    if not original or not rewritten:
        return rewritten, meta
    if not _has_rewrite_structure_drift(original, rewritten):
        return rewritten, meta

    meta["structure_drift_detected"] = True
    logger.warning(
        "[ai-action-structure-violation] action=rewrite section=%s in_chars=%s out_chars=%s - retrying strict structure lock",
        request.section_title,
        len(original),
        len(rewritten),
    )
    retry_prompt = build_rewrite_prompt(
        request,
        context,
        nlp_block,
        profile_md=profile_md or "",
        profile_json=profile_json,
        detected_nlp=detected_nlp,
        style_profile=style_profile,
    ) + (
        "\n\nCRITICAL RETRY:\n"
        "- Your previous rewrite changed the SOP structure.\n"
        "- Preserve the exact same heading order, section order, list order, and block structure as TEXT.\n"
        "- Rewrite only the wording inside the existing structure.\n"
        "- If any heading or structural token from TEXT is missing or renamed, restore it exactly.\n"
    )
    try:
        retry_parsed = parse_with_retry(
            raw=_call_action_llm(runtime, retry_prompt, input_char_budget=ch_budget, action="rewrite_structure_retry"),
            schema=RewriteResponse,
            prompt=retry_prompt,
            call_llm=lambda rp: _call_action_llm(
                runtime, rp, input_char_budget=ch_budget, action="rewrite_structure_retry"
            ),
            audit_log=[],
        )
        retry_text = (retry_parsed.rewritten_text or "").strip()
        retry_text = _preserve_selected_section_heading(original, retry_text)
        retry_text = _restore_section_name_heading(request, retry_text)
        if retry_text and not _has_rewrite_structure_drift(original, retry_text):
            meta["structure_lock_applied"] = True
            return retry_text, meta
    except Exception as exc:
        logger.warning("[ai-action-structure-retry-failed] action=rewrite err=%s", exc)
    return rewritten, meta


def _run_dynamic_ai_action(payload: AIActionRequest, action: str) -> AIActionResponse:
    request = _build_action_request(payload)
    sop_ctx = _load_uploaded_sop_context(request)
    explicit_style_override = _resolve_explicit_style_override(getattr(payload, "instruction", None))
    if action == "rewrite" and _rewrite_should_use_industry_scaffold(request):
        style_profile = _resolve_style_profile(sop_ctx, request.section_text)
        scaffold = _build_industry_rewrite_text(request)
        runtime = _get_action_runtime()
        try:
            rewritten, used_llm, llm_error = _rewrite_industry_scaffold_with_llm(runtime, request, scaffold)
        except Exception as exc:
            logger.warning("[ai-action-result] action=rewrite mode=industry_scaffold_llm_failed err=%s", exc)
            rewritten, used_llm, llm_error = scaffold, False, str(exc)
        logger.info(
            "[ai-action-result] action=rewrite ok=1 mode=industry_scaffold_llm used_llm=%s suggested_chars=%s",
            used_llm,
            len(rewritten or ""),
        )
        return AIActionResponse(
            action="rewrite",
            original_text=request.section_text,
            suggested_text=_render_dynamic_text(rewritten),
            explanation=(
                "Industry-level SOP rewrite generated with the configured LLM and preserved traceability records."
                if used_llm
                else "Industry-level SOP scaffold returned because the LLM rewrite failed; traceability records were preserved."
            ),
            structured_data={
                "rewritten_text": rewritten,
                "style_profile": style_profile,
                "nlp_action_summary": {
                    "has_upload_nlp": False,
                    "window_skipped": True,
                    "profile_row_reused": False,
                    "reason": "industry_scaffold_llm",
                },
                "rewrite_mode": "industry_scaffold_llm" if used_llm else "industry_scaffold_fallback",
                "llm_used": used_llm,
                "llm_error": llm_error,
            },
        )

    runtime = _get_action_runtime()
    ch_budget = len(request.section_text or "")
    sop_text = str(sop_ctx.get("text") or "")
    style_source_text = sop_text if sop_text else request.section_text
    detected_nlp, profile_json, profile_md = _get_nlp_and_profile_context(sop_ctx)
    if explicit_style_override:
        detected_nlp = explicit_style_override.get("detected_nlp") or detected_nlp
        profile_json = explicit_style_override.get("profile_json") or profile_json
        profile_md = explicit_style_override.get("profile_md") or profile_md
        override_source_text = str(explicit_style_override.get("style_source_text") or "").strip()
        if override_source_text:
            style_source_text = override_source_text
    style_profile = (
        _derive_sop_style_profile(style_source_text)
        if explicit_style_override
        else _resolve_style_profile(sop_ctx, style_source_text)
    )
    if isinstance(profile_json, dict):
        preferred_style = profile_json.get("preferred_style") if isinstance(profile_json.get("preferred_style"), dict) else {}
        profile_lang = _normalize_language_code(profile_json.get("language"))
        if profile_lang:
            style_profile["language"] = profile_lang
        if preferred_style.get("tone"):
            style_profile["tone"] = preferred_style.get("tone")
    style_block = _style_profile_prompt_block(style_profile)
    sop_context_block = ""
    if sop_ctx.get("detected"):
        sop_context_block = (
            "SOP_DETECTED_CONTEXT\n"
            f"- sop_number={sop_ctx.get('sop_number')}\n"
            f"- title={sop_ctx.get('title')}\n"
            f"- source=uploaded_database_sop\n"
            f"- excerpt={_truncate_text(sop_text, 1200)}"
        )
    print(
        f"[nlp-action] action={action} sop_detected={bool(sop_ctx.get('detected'))} "
        f"sop_number={sop_ctx.get('sop_number') or '-'} tone={style_profile.get('tone')} "
        f"avg_sentence_words={style_profile.get('avg_sentence_words')}",
        flush=True,
    )
    logger.info(
        "[action-style-profile] action=%s sop_detected=%s sop_number=%s tone=%s language=%s avg_sentence_words=%s imperative_ratio=%s",
        action,
        bool(sop_ctx.get("detected")),
        sop_ctx.get("sop_number"),
        style_profile.get("tone"),
        style_profile.get("language"),
        style_profile.get("avg_sentence_words"),
        style_profile.get("imperative_ratio"),
    )
    if explicit_style_override:
        logger.info(
            "[action-style-override] action=%s style_reference=%s matched_type=%s resolved_name=%s profile_version=%s",
            action,
            explicit_style_override.get("style_reference"),
            explicit_style_override.get("matched_type"),
            explicit_style_override.get("resolved_name"),
            explicit_style_override.get("profile_version"),
        )
    style_context_meta = {
        "used_instruction": str(getattr(payload, "instruction", "") or "").strip() or None,
        "used_profile": _profile_display_name(profile_json) if isinstance(profile_json, dict) else None,
        "used_profile_id": explicit_style_override.get("profile_id")
        if explicit_style_override
        else ((profile_json or {}).get("_profile_id") if isinstance(profile_json, dict) else None),
        "used_profile_version": explicit_style_override.get("profile_version")
        if explicit_style_override
        else ((profile_json or {}).get("_profile_version") if isinstance(profile_json, dict) else None),
        "used_style_reference": explicit_style_override.get("style_reference")
        if explicit_style_override
        else None,
        "used_style_source": explicit_style_override.get("resolved_name")
        if explicit_style_override
        else None,
        "used_target_sop_version": sop_ctx.get("version_number"),
    }

    bypass = _try_client_structured_ai_response(payload, action, request, style_profile)
    if bypass is not None:
        logger.info("[ai-action] using client_structured_json bypass action=%s", action)
        return bypass

    _ensure_profile_detection_row(sop_ctx, action)

    nlp_bundle, nlp_block = build_nlp_bundle_for_action(action, request, sop_ctx, style_profile)
    log_nlp_detected(action, nlp_bundle)
    nlp_summary = nlp_action_summary(nlp_bundle)
    cfg = get_local_llm_config()

    if action == "gap_check":
        retrieval_query = _build_gap_check_retrieval_query(request)
        if getattr(request, "sop_entity_id", None):
            runtime.retriever.metadata_filters = {"allowed_entity_ids": [str(request.sop_entity_id)]}
        else:
            runtime.retriever.metadata_filters = None
        raw_docs = runtime.retriever.invoke(retrieval_query)
        reranked = runtime.reranker.rerank_top_n(retrieval_query, raw_docs, 3)
        context = f"{format_chunks(reranked)}\n\n{style_block}\n\n{sop_context_block}".strip()
        gap_retrieval_meta = {
            "retrieval_mode": "hybrid_dense_bm25_rerank",
            "collection": getattr(runtime, "collection_name", None),
            "raw_docs": len(raw_docs),
            "reranked_docs": len(reranked),
            "scoped_sop_entity_id": str(request.sop_entity_id) if getattr(request, "sop_entity_id", None) else None,
            "dense_weight": getattr(runtime.retriever, "dense_weight", None),
            "bm25_weight": getattr(runtime.retriever, "bm25_weight", None),
            "reranker": type(runtime.reranker).__name__,
        }
        print(
            f"[nlp-action] action=gap_check retrieval_docs={len(raw_docs)} reranked_docs={len(reranked)}",
            flush=True,
        )
    else:
        gap_retrieval_meta = None
        context = _build_improve_rewrite_context(request, style_block, sop_context_block)

    logger.info(
        "[ai-action-prompt] action=%s prompt_type=%s_json_nlp_v1 provider=%s model=%s nlp_block_chars=%s",
        action,
        action,
        cfg.provider,
        cfg.model,
        len(nlp_block or ""),
    )

    if action == "improve":
        prompt = build_improve_prompt(
            request,
            context,
            nlp_block,
            profile_md=profile_md or "",
            profile_json=profile_json,
            detected_nlp=detected_nlp,
            style_profile=style_profile,
        )
        parsed = parse_with_retry(
            raw=_call_action_llm(runtime, prompt, input_char_budget=ch_budget, action=action),
            schema=ImproveResponse,
            prompt=prompt,
            call_llm=lambda rp: _call_action_llm(
                runtime, rp, input_char_budget=ch_budget, action=action
            ),
            audit_log=[],
        )
        improved_text = _enforce_section_only_action_text(
            runtime,
            action="improve",
            request=request,
            text=parsed.improved_text,
            context=context,
            nlp_block=nlp_block,
            ch_budget=ch_budget,
        )
        improved_text, quality_meta = _apply_german_profile_quality_guard(
            runtime,
            action="improve",
            request=request,
            text=improved_text,
            profile_json=profile_json,
            profile_md=profile_md,
            style_profile=style_profile,
            ch_budget=ch_budget,
        )
        from chatbot.assistant.context_intelligence import enforce_output_line_count

        improved_text = enforce_output_line_count(
            getattr(payload, "instruction", None), improved_text
        )
        logger.info(
            "[ai-action-result] action=improve ok=1 provider=%s model=%s suggested_chars=%s",
            cfg.provider,
            cfg.model,
            len(improved_text or ""),
        )
        return AIActionResponse(
            action="improve",
            original_text=request.section_text,
            suggested_text=_render_dynamic_text(improved_text),
            explanation="Text verbessert / Text improved.",
            structured_data={
                "improved_text": improved_text,
                "improved_version": improved_text,
                "style_profile": style_profile,
                "nlp_action_summary": nlp_summary,
                "edit_scope": resolve_edit_scope(request),
                "quality_guard": quality_meta,
                **style_context_meta,
            },
        )

    if action == "summarize":
        prompt = build_summarize_prompt(
            request,
            context,
            nlp_block,
            profile_md=profile_md or "",
            profile_json=profile_json,
            detected_nlp=detected_nlp,
            style_profile=style_profile,
        )
        parsed = parse_with_retry(
            raw=_call_action_llm(runtime, prompt, input_char_budget=ch_budget, action=action),
            schema=ImproveResponse,
            prompt=prompt,
            call_llm=lambda rp: _call_action_llm(
                runtime, rp, input_char_budget=ch_budget, action=action
            ),
            audit_log=[],
        )
        logger.info(
            "[ai-action-result] action=summarize ok=1 provider=%s model=%s suggested_chars=%s",
            cfg.provider,
            cfg.model,
            len(parsed.improved_text or ""),
        )
        return AIActionResponse(
            action="summarize",
            original_text=request.section_text,
            suggested_text=_render_dynamic_text(parsed.improved_text),
            explanation="Executive summary generated from the current SOP text.",
            structured_data={
                "improved_text": parsed.improved_text,
                "improved_version": parsed.improved_text,
                "style_profile": style_profile,
                "nlp_action_summary": nlp_summary,
                **style_context_meta,
            },
        )

    if action == "analyze":
        prompt = build_analyze_prompt(
            request,
            context,
            nlp_block,
            profile_md=profile_md or "",
            profile_json=profile_json,
            detected_nlp=detected_nlp,
            style_profile=style_profile,
        )
        parsed = parse_with_retry(
            raw=_call_action_llm(runtime, prompt, input_char_budget=ch_budget, action=action),
            schema=ImproveResponse,
            prompt=prompt,
            call_llm=lambda rp: _call_action_llm(
                runtime, rp, input_char_budget=ch_budget, action=action
            ),
            audit_log=[],
        )
        logger.info(
            "[ai-action-result] action=analyze ok=1 provider=%s model=%s suggested_chars=%s",
            cfg.provider,
            cfg.model,
            len(parsed.improved_text or ""),
        )
        return AIActionResponse(
            action="analyze",
            original_text=request.section_text,
            suggested_text=_render_dynamic_text(parsed.improved_text),
            explanation="Compliance-oriented analysis of the current SOP text.",
            structured_data={
                "improved_text": parsed.improved_text,
                "improved_version": parsed.improved_text,
                "style_profile": style_profile,
                "nlp_action_summary": nlp_summary,
                **style_context_meta,
            },
        )

    if action == "rewrite":
        if _rewrite_should_use_industry_scaffold(request):
            rewritten = _build_industry_rewrite_text(request)
            logger.info(
                "[ai-action-result] action=rewrite ok=1 mode=industry_scaffold suggested_chars=%s",
                len(rewritten or ""),
            )
            return AIActionResponse(
                action="rewrite",
                original_text=request.section_text,
                suggested_text=_render_dynamic_text(rewritten),
                explanation="Industry-level SOP rewrite generated with preserved traceability records.",
                structured_data={
                    "rewritten_text": rewritten,
                    "style_profile": style_profile,
                    "nlp_action_summary": nlp_summary,
                    "rewrite_mode": "industry_scaffold",
                    **style_context_meta,
                },
            )
        prompt = build_rewrite_prompt(
            request,
            context,
            nlp_block,
            profile_md=profile_md or "",
            profile_json=profile_json,
            detected_nlp=detected_nlp,
            style_profile=style_profile,
        )
        parsed = parse_with_retry(
            raw=_call_action_llm(runtime, prompt, input_char_budget=ch_budget, action=action),
            schema=RewriteResponse,
            prompt=prompt,
            call_llm=lambda rp: _call_action_llm(
                runtime, rp, input_char_budget=ch_budget, action=action
            ),
            audit_log=[],
        )
        rewritten_text = _enforce_section_only_action_text(
            runtime,
            action="rewrite",
            request=request,
            text=parsed.rewritten_text,
            context=context,
            nlp_block=nlp_block,
            ch_budget=ch_budget,
        )
        followup_short = _shorten_previous_suggestion_if_requested(payload, rewritten_text)
        if followup_short:
            rewritten_text = followup_short
        rewritten_text, quality_meta = _apply_german_profile_quality_guard(
            runtime,
            action="rewrite",
            request=request,
            text=rewritten_text,
            profile_json=profile_json,
            profile_md=profile_md,
            style_profile=style_profile,
            ch_budget=ch_budget,
        )
        rewritten_text = _preserve_selected_section_heading(request.section_text, rewritten_text)
        rewritten_text = _restore_section_name_heading(request, rewritten_text)
        rewritten_text, structure_meta = _enforce_rewrite_structure_lock(
            runtime,
            request=request,
            text=rewritten_text,
            context=context,
            nlp_block=nlp_block,
            profile_md=profile_md or "",
            profile_json=profile_json,
            detected_nlp=detected_nlp,
            style_profile=style_profile,
            ch_budget=ch_budget,
        )
        from chatbot.assistant.context_intelligence import enforce_output_line_count

        rewritten_text = enforce_output_line_count(
            getattr(payload, "instruction", None), rewritten_text
        )
        logger.info(
            "[ai-action-result] action=rewrite ok=1 provider=%s model=%s suggested_chars=%s edit_scope=%s",
            cfg.provider,
            cfg.model,
            len(rewritten_text or ""),
            resolve_edit_scope(request),
        )
        return AIActionResponse(
            action="rewrite",
            original_text=request.section_text,
            suggested_text=_render_dynamic_text(rewritten_text),
            explanation="Text neu formuliert / Text rewritten.",
            structured_data={
                "rewritten_text": rewritten_text,
                "style_profile": style_profile,
                "nlp_action_summary": nlp_summary,
                "edit_scope": resolve_edit_scope(request),
                "quality_guard": quality_meta,
                "structure_guard": structure_meta,
                "follow_up_shortening_applied": bool(followup_short),
                **style_context_meta,
            },
        )

    if action == "gap_check":
        prompt = build_gap_check_prompt(
            request,
            context,
            nlp_block,
            profile_md=profile_md or "",
            profile_json=profile_json,
            detected_nlp=detected_nlp,
            style_profile=style_profile,
        )
        parsed = parse_with_retry(
            raw=_call_action_llm(runtime, prompt, input_char_budget=ch_budget, action=action),
            schema=GapCheckResponse,
            prompt=prompt,
            call_llm=lambda rp: _call_action_llm(
                runtime, rp, input_char_budget=ch_budget, action=action
            ),
            audit_log=[],
        )
        logger.info(
            "[ai-action-result] action=gap_check ok=1 provider=%s model=%s analysis_chars=%s",
            cfg.provider,
            cfg.model,
            len(parsed.analysis or ""),
        )
        return AIActionResponse(
            action="gap_check",
            original_text=request.section_text,
            suggested_text=_render_gap_check_analysis_html(parsed.analysis),
            explanation="Compliance-Lückenanalyse abgeschlossen / Compliance gap analysis completed.",
            structured_data={
                "analysis": parsed.analysis,
                "style_profile": style_profile,
                "nlp_action_summary": nlp_summary,
                "retrieval": gap_retrieval_meta,
                **style_context_meta,
            },
        )

    raise HTTPException(status_code=400, detail=f"Action '{action}' is not supported.")


def _fallback_gap_check(payload: AIActionRequest) -> AIActionResponse:
    runtime = _get_action_runtime()
    request = _build_action_request(payload)
    request.section_text = _trim_large_selection_for_fallback(request.section_text)
    ch_budget = len(request.section_text or "")
    sop_ctx = _load_uploaded_sop_context(request)
    sop_text = str(sop_ctx.get("text") or "")
    style_profile = _resolve_style_profile(sop_ctx, sop_text if sop_text else request.section_text)
    style_block = _style_profile_prompt_block(style_profile)
    sop_context_block = ""
    if sop_ctx.get("detected"):
        sop_context_block = (
            "SOP_DETECTED_CONTEXT\n"
            f"- sop_number={sop_ctx.get('sop_number')}\n"
            f"- title={sop_ctx.get('title')}\n"
            f"- source=uploaded_database_sop\n"
            f"- excerpt={_truncate_text(sop_text, 1200)}"
        )
    nlp_bundle, nlp_block = build_nlp_bundle_for_action("gap_check", request, sop_ctx, style_profile)
    log_nlp_detected("gap_check", nlp_bundle)
    nlp_summary = nlp_action_summary(nlp_bundle)
    cfg = get_local_llm_config()
    logger.info(
        "[ai-action-prompt] action=gap_check prompt_type=gap_check_json_nlp_v1_fallback provider=%s model=%s nlp_block_chars=%s",
        cfg.provider,
        cfg.model,
        len(nlp_block or ""),
    )
    detected_nlp, profile_json, profile_md = _get_nlp_and_profile_context(sop_ctx)
    prompt = build_gap_check_prompt(
        request,
        f"Kein relevanter Kontext verfügbar. / No relevant context found.\n\n{style_block}\n\n{sop_context_block}",
        nlp_block,
        profile_md=profile_md or "",
        profile_json=profile_json,
        detected_nlp=detected_nlp,
        style_profile=style_profile,
    )
    raw = _call_action_llm(runtime, prompt, input_char_budget=ch_budget, action="gap_check")
    parsed = parse_with_retry(
        raw=raw,
        schema=GapCheckResponse,
        prompt=prompt,
        call_llm=lambda rp: _call_action_llm(runtime, rp, input_char_budget=ch_budget, action="gap_check"),
        audit_log=[],
    )
    return AIActionResponse(
        action="gap_check",
        original_text=_clean_text(payload.text),
        suggested_text=_render_gap_check_analysis_html(parsed.analysis),
        explanation="Compliance-Lückenanalyse abgeschlossen / Compliance gap analysis completed.",
        structured_data={
            "analysis": parsed.analysis,
            "style_profile": style_profile,
            "nlp_action_summary": nlp_summary,
        },
    )


def _fallback_rewrite(payload: AIActionRequest) -> AIActionResponse:
    runtime = _get_action_runtime()
    request = _build_action_request(payload)
    ch_budget = len(request.section_text or "")
    sop_ctx = _load_uploaded_sop_context(request)
    sop_text = str(sop_ctx.get("text") or "")
    style_profile = _resolve_style_profile(sop_ctx, sop_text if sop_text else request.section_text)
    style_block = _style_profile_prompt_block(style_profile)
    sop_context_block = ""
    if sop_ctx.get("detected"):
        sop_context_block = (
            "SOP_DETECTED_CONTEXT\n"
            f"- sop_number={sop_ctx.get('sop_number')}\n"
            f"- title={sop_ctx.get('title')}\n"
            f"- source=uploaded_database_sop\n"
            f"- excerpt={_truncate_text(sop_text, 1200)}"
        )
    _ensure_profile_detection_row(sop_ctx, "rewrite")
    nlp_bundle, nlp_block = build_nlp_bundle_for_action("rewrite", request, sop_ctx, style_profile)
    log_nlp_detected("rewrite", nlp_bundle)
    nlp_summary = nlp_action_summary(nlp_bundle)
    cfg = get_local_llm_config()
    logger.info(
        "[ai-action-prompt] action=rewrite prompt_type=rewrite_json_nlp_v1_fallback provider=%s model=%s nlp_block_chars=%s",
        cfg.provider,
        cfg.model,
        len(nlp_block or ""),
    )
    detected_nlp, profile_json, profile_md = _get_nlp_and_profile_context(sop_ctx)
    prompt = build_rewrite_prompt(
        request,
        f"{IMPROVE_REWRITE_NO_RAG_CONTEXT}\n{style_block}\n{sop_context_block}".strip(),
        nlp_block,
        profile_md=profile_md or "",
        profile_json=profile_json,
        detected_nlp=detected_nlp,
        style_profile=style_profile,
    )
    raw = _call_action_llm(runtime, prompt, input_char_budget=ch_budget, action="rewrite")
    parsed = parse_with_retry(
        raw=raw,
        schema=RewriteResponse,
        prompt=prompt,
        call_llm=lambda rp: _call_action_llm(runtime, rp, input_char_budget=ch_budget, action="rewrite"),
        audit_log=[],
    )
    rewritten_text = _preserve_selected_section_heading(request.section_text, parsed.rewritten_text)
    rewritten_text = _restore_section_name_heading(request, rewritten_text)
    rewritten_text, structure_meta = _enforce_rewrite_structure_lock(
        runtime,
        request=request,
        text=rewritten_text,
        context=f"{IMPROVE_REWRITE_NO_RAG_CONTEXT}\n{style_block}\n{sop_context_block}".strip(),
        nlp_block=nlp_block,
        profile_md=profile_md or "",
        profile_json=profile_json,
        detected_nlp=detected_nlp,
        style_profile=style_profile,
        ch_budget=ch_budget,
    )
    return AIActionResponse(
        action="rewrite",
        original_text=_clean_text(payload.text),
        suggested_text=_render_dynamic_text(rewritten_text),
        explanation="Text neu formuliert / Text rewritten.",
        structured_data={
            "rewritten_text": rewritten_text,
            "style_profile": style_profile,
            "nlp_action_summary": nlp_summary,
            "structure_guard": structure_meta,
        },
    )


def _fallback_improve(payload: AIActionRequest) -> AIActionResponse:
    runtime = _get_action_runtime()
    request = _build_action_request(payload)
    ch_budget = len(request.section_text or "")
    sop_ctx = _load_uploaded_sop_context(request)
    sop_text = str(sop_ctx.get("text") or "")
    style_profile = _resolve_style_profile(sop_ctx, sop_text if sop_text else request.section_text)
    style_block = _style_profile_prompt_block(style_profile)
    sop_context_block = ""
    if sop_ctx.get("detected"):
        sop_context_block = (
            "SOP_DETECTED_CONTEXT\n"
            f"- sop_number={sop_ctx.get('sop_number')}\n"
            f"- title={sop_ctx.get('title')}\n"
            f"- source=uploaded_database_sop\n"
            f"- excerpt={_truncate_text(sop_text, 1200)}"
        )
    _ensure_profile_detection_row(sop_ctx, "improve")
    nlp_bundle, nlp_block = build_nlp_bundle_for_action("improve", request, sop_ctx, style_profile)
    log_nlp_detected("improve", nlp_bundle)
    nlp_summary = nlp_action_summary(nlp_bundle)
    cfg = get_local_llm_config()
    logger.info(
        "[ai-action-prompt] action=improve prompt_type=improve_json_nlp_v1_fallback provider=%s model=%s nlp_block_chars=%s",
        cfg.provider,
        cfg.model,
        len(nlp_block or ""),
    )
    detected_nlp, profile_json, profile_md = _get_nlp_and_profile_context(sop_ctx)
    prompt = build_improve_prompt(
        request,
        f"{IMPROVE_REWRITE_NO_RAG_CONTEXT}\n{style_block}\n{sop_context_block}".strip(),
        nlp_block,
        profile_md=profile_md or "",
        profile_json=profile_json,
        detected_nlp=detected_nlp,
        style_profile=style_profile,
    )
    raw = _call_action_llm(runtime, prompt, input_char_budget=ch_budget, action="improve")
    parsed = parse_with_retry(
        raw=raw,
        schema=ImproveResponse,
        prompt=prompt,
        call_llm=lambda rp: _call_action_llm(runtime, rp, input_char_budget=ch_budget, action="improve"),
        audit_log=[],
    )
    return AIActionResponse(
        action="improve",
        original_text=_clean_text(payload.text),
        suggested_text=_render_dynamic_text(parsed.improved_text),
        explanation="Text verbessert / Text improved.",
        structured_data={
            "improved_text": parsed.improved_text,
            "style_profile": style_profile,
            "nlp_action_summary": nlp_summary,
        },
    )


def _extract_selected_text_html(action: str, structured_data: dict, suggested_text: str) -> str:
    if action == "rewrite":
        return _render_dynamic_text(structured_data.get("rewritten_text") or suggested_text)
    if action == "improve":
        return _render_dynamic_text(structured_data.get("improved_text") or suggested_text)
    return suggested_text


def _ctx_list(values: Any) -> list:
    return values if isinstance(values, list) else []


def _extract_refs(items: list, keys: list[str], limit: int = 8) -> list[str]:
    refs: list[str] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        for key in keys:
            value = str(item.get(key) or "").strip()
            if value:
                refs.append(value)
                break
        if len(refs) >= limit:
            break
    return refs


def _visible_user_question(raw: str) -> str:
    """User-facing chat text only (strip legacy frontend context prefix if present)."""
    q = str(raw or "").strip()
    if not q:
        return ""
    m = re.match(
        r"^Active SOP context:\s*.+?\.\s*User request:\s*(.+)$",
        q,
        re.IGNORECASE | re.DOTALL,
    )
    return m.group(1).strip() if m else q


def _user_requests_global_scope(question: str) -> bool:
    q = (question or "").lower()
    return bool(
        re.search(
            r"\b(all|every|entire|whole|global|across|system-wide)\b.*\b(sop|sops|database|system|records)\b",
            q,
        )
        or re.search(r"\b(sop|sops)\b.*\b(all|every|entire|whole|global)\b", q)
        or re.search(r"\b(alle|sämtliche|gesamt)\b.*\b(sop|sops)\b", q)
    )


def _linked_entity_ids_from_context(linked: dict, key: str) -> list[str]:
    out: list[str] = []
    for item in _ctx_list(linked.get(key)):
        if not isinstance(item, dict):
            continue
        value = str(item.get("id") or "").strip()
        if value:
            out.append(value)
    return out


def _load_linked_entity_ids_from_db(sop_id: str) -> dict[str, list[str]]:
    """Resolve linked entity UUIDs for an active SOP (same traversal as /api/sops/{id}/related)."""
    try:
        sop_uuid = uuid.UUID(str(sop_id).strip())
    except ValueError:
        return {}

    db = SessionLocal()
    try:
        sop = db.query(SOP).filter(SOP.id == sop_uuid, SOP.is_active == True).first()  # noqa: E712
        if not sop:
            return {}

        dev_ids = {
            str(row[0])
            for row in db.query(SopDeviationLink.deviation_id).filter(SopDeviationLink.sop_id == sop.id).all()
        }
        decision_ids = {
            str(row[0])
            for row in db.query(DecisionSopLink.decision_id).filter(DecisionSopLink.sop_id == sop.id).all()
        }
        audit_ids: set[str] = set()
        capa_ids: set[str] = set()

        if dev_ids:
            capa_ids = {
                str(row[0])
                for row in db.query(DeviationCapaLink.capa_id)
                .filter(DeviationCapaLink.deviation_id.in_(list(dev_ids)))
                .all()
            }
        if decision_ids:
            audit_ids = {
                str(row[0])
                for row in db.query(AuditDecisionLink.audit_finding_id)
                .filter(AuditDecisionLink.decision_id.in_(list(decision_ids)))
                .all()
            }
        if capa_ids:
            audit_ids.update(
                str(row[0])
                for row in db.query(CapaAuditLink.audit_finding_id)
                .filter(CapaAuditLink.capa_id.in_(list(capa_ids)))
                .all()
            )
        if audit_ids:
            capa_ids.update(
                str(row[0])
                for row in db.query(CapaAuditLink.capa_id)
                .filter(CapaAuditLink.audit_finding_id.in_(list(audit_ids)))
                .all()
            )
            dev_ids.update(
                str(row[0])
                for row in db.query(DeviationCapaLink.deviation_id)
                .filter(DeviationCapaLink.capa_id.in_(list(capa_ids)))
                .all()
            )
            decision_ids.update(
                str(row[0])
                for row in db.query(AuditDecisionLink.decision_id)
                .filter(AuditDecisionLink.audit_finding_id.in_(list(audit_ids)))
                .all()
            )

        related_sop_ids = {
            str(row[0])
            for row in db.query(DecisionSopLink.sop_id)
            .filter(DecisionSopLink.decision_id.in_(list(decision_ids)))
            .all()
            if row[0] and str(row[0]) != str(sop.id)
        }

        return {
            "linked_deviation_ids": sorted(dev_ids),
            "linked_capa_ids": sorted(capa_ids),
            "linked_audit_ids": sorted(audit_ids),
            "linked_decision_ids": sorted(decision_ids),
            "linked_sop_ids": sorted(related_sop_ids),
        }
    finally:
        db.close()


def _extract_active_sop_scope(assistant_context: dict | None) -> dict:
    ctx = assistant_context or {}
    current = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}
    linked = ctx.get("linked_context") if isinstance(ctx.get("linked_context"), dict) else {}
    editor_active = bool(ctx.get("editor_surface_active"))
    route = str(ctx.get("route") or "").strip().lower()
    if not editor_active and not route.startswith("/editor"):
        return {
            "active_sop_id": "",
            "active_sop_ref": "",
            "title": "",
            "linked_deviation_ids": [],
            "linked_capa_ids": [],
            "linked_audit_ids": [],
            "linked_decision_ids": [],
            "linked_sop_ids": [],
        }
    active_sop_id = str(
        ctx.get("active_sop_id") or ctx.get("current_document_id") or current.get("id") or ""
    ).strip()
    active_sop_ref = str(current.get("sop_number") or current.get("documentId") or "").strip()
    title = str(current.get("title") or "").strip()

    scope = {
        "active_sop_id": active_sop_id,
        "active_sop_ref": active_sop_ref,
        "title": title,
        "linked_deviation_ids": _linked_entity_ids_from_context(linked, "deviations"),
        "linked_capa_ids": _linked_entity_ids_from_context(linked, "capas"),
        "linked_audit_ids": _linked_entity_ids_from_context(linked, "audits"),
        "linked_decision_ids": _linked_entity_ids_from_context(linked, "decisions"),
        "linked_sop_ids": _linked_entity_ids_from_context(linked, "related_sops"),
    }

    if active_sop_id:
        db_linked = _load_linked_entity_ids_from_db(active_sop_id)
        for key in (
            "linked_deviation_ids",
            "linked_capa_ids",
            "linked_audit_ids",
            "linked_decision_ids",
            "linked_sop_ids",
        ):
            merged = sorted(set(scope.get(key) or []) | set(db_linked.get(key) or []))
            scope[key] = merged

    return scope


def _query_intents(question: str) -> set[str]:
    q = (question or "").lower()
    intents: set[str] = set()
    if re.search(r"\b(how many|count|number of|total|wie viele|anzahl)\b.*\b(sop|sops)\b", q):
        intents.add("sop_count")
    if re.search(r"\b(list|show|which|what)\b.*\b(sop|sops)\b", q):
        intents.add("sop_list")
    if re.search(r"\b(summarize|summary|brief|gist|zusammenfass|kurzfass|fasse\s+zusammen)\b", q):
        intents.add("summary")
    if re.search(r"\b(explain|easy wording|easy words|simple words|what does this mean|tell me about|describe)\b", q):
        intents.add("explain")
    if re.search(r"\b(compliance|compliant|regulatory|gmp|qa|qms|control|audit readiness|audit-ready)\b", q):
        intents.add("compliance")
    if re.search(r"\b(compare|difference|vs|versus)\b", q):
        intents.add("compare")
    if re.search(r"\b(linked|related)\b.*\b(capa|capas|audit|audits|decision|decisions|deviation|deviations)\b", q):
        intents.add("linked")
    if re.search(r"\b(this sop|current sop|active sop)\b", q):
        intents.add("active_sop")
    if re.search(r"\b(which|what)\b.*\b(sop)\b.*\b(currently open|open now|opened|active)\b", q):
        intents.add("active_sop")
    if re.search(r"\b(who owns|owner|what version|version|status|tags?|word count|how many words|created|updated|standards?|compliance standards?)\b", q):
        intents.add("metadata")
    return intents


def _deterministic_active_sop_metadata_response(question: str, assistant_context: dict | None) -> dict[str, Any] | None:
    ctx = assistant_context or {}
    current = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}
    active_id = str(ctx.get("active_sop_id") or ctx.get("current_document_id") or current.get("id") or "").strip()
    if not active_id and not current:
        return None

    q = (question or "").lower()
    fields: list[tuple[str, Any]] = []
    if re.search(r"\b(who owns|owner)\b", q):
        fields.append(("Owner", current.get("owner") or current.get("metadata", {}).get("owner") or current.get("metadata", {}).get("author") or "Not provided"))
    if "version" in q:
        fields.append(("Version", current.get("version") or "Not provided"))
    if "status" in q:
        fields.append(("Status", current.get("status") or "Not provided"))
    if re.search(r"\btags?\b", q):
        tags = current.get("tags") if isinstance(current.get("tags"), list) else []
        fields.append(("Tags", ", ".join(str(t) for t in tags) if tags else "Not provided"))
    if re.search(r"\b(word count|how many words)\b", q):
        fields.append(("Word count", current.get("word_count") or "Not provided"))
    if "created" in q:
        fields.append(("Created at", current.get("created_at") or "Not provided"))
    if "updated" in q:
        fields.append(("Updated at", current.get("updated_at") or ctx.get("context_updated_at") or "Not provided"))
    if re.search(r"\b(standards?|compliance standards?)\b", q):
        standards = current.get("compliance_standards") if isinstance(current.get("compliance_standards"), list) else []
        fields.append(("Compliance standards", ", ".join(str(s) for s in standards) if standards else "Not provided"))
    if not fields:
        fields = [
            ("Title", current.get("title") or "Not provided"),
            ("Version", current.get("version") or "Not provided"),
            ("Owner", current.get("owner") or current.get("metadata", {}).get("author") or "Not provided"),
            ("Status", current.get("status") or "Not provided"),
        ]

    title = current.get("title") or current.get("sop_number") or active_id
    answer = f"From the active SOP context ({title}):\n" + "\n".join(f"- {label}: {value}" for label, value in fields)
    return {
        "answer": answer,
        "sources": [{"id": "live_sop_context", "type": "metadata", "label": "Active SOP metadata"}],
        "citations": [
            {
                "ref": "live_sop_context",
                "title": "Active SOP metadata",
                "type": "metadata",
                "excerpt": "Answered from the live editor assistant context.",
                "score": 1.0,
            }
        ],
        "retrieval_debug": [],
        "suggestions": ["Summarize this SOP", "Check this SOP for gaps", "Show related SOPs"],
        "retrieval_stats": {
            "total_docs": 1,
            "source": "live_sop_metadata",
            "strict_mode": "active_sop_metadata",
        },
        "routed_to": "live_sop_metadata",
        "assistant_action": None,
    }


def _deterministic_explain_preview_response(question: str, assistant_context: dict | None) -> dict[str, Any] | None:
    from chatbot.assistant.context_intelligence import (
        build_explain_last_output_answer,
        extract_format_constraints,
        infer_output_language_from_context,
        is_explain_recent_output_query,
        merge_constraints,
    )

    ctx = assistant_context or {}
    last = ctx.get("last_action") if isinstance(ctx.get("last_action"), dict) else {}
    if not last or not is_explain_recent_output_query(question, last):
        return None

    fmt = extract_format_constraints(question)
    rc = ctx.get("response_constraints") if isinstance(ctx.get("response_constraints"), dict) else {}
    fmt = merge_constraints(rc, fmt, {"language": infer_output_language_from_context(ctx)})
    answer = build_explain_last_output_answer(
        question,
        last,
        format_constraints=fmt,
        assistant_context=ctx,
    )
    return {
        "answer": answer,
        "sources": [{"id": "last_editor_action", "type": "assistant_memory", "label": "Last editor preview"}],
        "citations": [],
        "retrieval_debug": [],
        "suggestions": [],
        "retrieval_stats": {
            "total_docs": 1,
            "source": "assistant_explain_preview",
            "strict_mode": "last_preview_explain",
        },
        "routed_to": "assistant_explain_preview",
        "assistant_action": None,
    }


def _deterministic_last_action_response(question: str, assistant_context: dict | None) -> dict[str, Any] | None:
    from chatbot.assistant.context_intelligence import (
        build_diff_explanation_answer,
        extract_format_constraints,
        infer_output_language_from_context,
        is_meta_question_about_assistant_output,
        merge_constraints,
    )

    ctx = assistant_context or {}
    last = ctx.get("last_action") if isinstance(ctx.get("last_action"), dict) else {}
    if not last or not is_meta_question_about_assistant_output(question):
        return None

    active_scope = ctx.get("active_scope") if isinstance(ctx.get("active_scope"), dict) else {}
    fmt = extract_format_constraints(question)
    rc = ctx.get("response_constraints") if isinstance(ctx.get("response_constraints"), dict) else {}
    fmt = merge_constraints(rc, fmt, {"language": infer_output_language_from_context(ctx)})

    answer = build_diff_explanation_answer(
        question,
        last,
        format_constraints=fmt,
        active_scope=active_scope,
        assistant_context=ctx,
    )
    action = str(last.get("action") or "unknown").strip()
    section = str(last.get("section_name") or "unknown").strip()
    status = str(last.get("status") or "").strip()
    lang = fmt.get("language") or infer_output_language_from_context(ctx)
    if lang == "de":
        suggestions = [
            "Kürzer umschreiben",
            "Abschnitt in einfachen Worten erklären",
            "Vorschau im Editor annehmen oder verwerfen",
        ]
    else:
        suggestions = [
            "Make the rewrite shorter",
            "Explain this section in simple words",
            "Accept or reject the inline preview",
        ]
    return {
        "answer": answer,
        "sources": [{"id": "last_editor_action", "type": "assistant_memory", "label": "Last editor action"}],
        "citations": [
            {
                "ref": "last_editor_action",
                "title": "Last editor action",
                "type": "assistant_memory",
                "excerpt": f"action={action}; section={section}; status={status or 'unknown'}",
                "score": 1.0,
            }
        ],
        "retrieval_debug": [],
        "suggestions": suggestions,
        "retrieval_stats": {
            "total_docs": 1,
            "source": "assistant_last_action",
            "strict_mode": "last_action_memory",
        },
        "routed_to": "assistant_last_action",
        "assistant_action": None,
    }


def _live_section_label_norm(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip()).lower()
    text = re.sub(r"^\d+(?:\.\d+)*[.)\]:-]?\s*", "", text).strip()
    return re.sub(r"[^a-z0-9À-ÿäöüÄÖÜß]+", " ", text, flags=re.IGNORECASE).strip()


def _extract_section_hint_for_live_answer(question: str, assistant_context: dict | None) -> str:
    q = str(question or "")
    patterns = [
        r"\b(?:the\s+)?([A-Za-zÀ-ÿ][\wÀ-ÿ\s/&()-]{1,80}?)\s+section\b",
        r"\bsection\s+([A-Za-zÀ-ÿ][\wÀ-ÿ\s/&()-]{1,80})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, q, re.IGNORECASE)
        if not match:
            continue
        candidate = re.sub(
            r"^(?:ok(?:ay)?\s+|now\s+|then\s+|please\s+|summarize\s+(?:the\s+)?|explain\s+(?:the\s+)?|what'?s\s+inside\s+(?:the\s+)?)",
            "",
            match.group(1).strip(" .:-"),
            flags=re.IGNORECASE,
        ).strip(" .:-")
        if candidate.lower() not in {"this", "that", "it", "them", "same", "current", "previous"}:
            return candidate
    known = re.search(
        r"\b(zweck|sweck|purpose|scope|geltungsbereich|procedure|verfahren|responsibilities|responsibility|verantwortlichkeiten|capas?|approval|records|definitions)\b",
        q,
        re.IGNORECASE,
    )
    if known:
        return known.group(1)
    ctx = assistant_context or {}
    focus = ctx.get("last_focus") if isinstance(ctx.get("last_focus"), dict) else {}
    last = ctx.get("last_action") if isinstance(ctx.get("last_action"), dict) else {}
    selected = ctx.get("selected_section") if isinstance(ctx.get("selected_section"), dict) else {}
    for source in (focus, last, selected):
        value = str(source.get("section_name") or source.get("label") or source.get("name") or "").strip()
        if value and value.lower() not in {"selected text", "selection", "full sop", "full document"}:
            return value
    return ""


def _section_heading_level(label: str) -> int | None:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)[.)\]:-]?\s+\S", str(label or ""))
    if not match:
        return None
    return len([part for part in match.group(1).split(".") if part])


def _resolve_live_section_for_answer(assistant_context: dict | None, question: str) -> dict[str, str] | None:
    ctx = assistant_context or {}
    current = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}
    selected = ctx.get("selected_section") if isinstance(ctx.get("selected_section"), dict) else {}
    hint = _extract_section_hint_for_live_answer(question, assistant_context)
    hint_norm = _live_section_label_norm(hint)

    selected_label = str(selected.get("label") or selected.get("name") or "").strip()
    selected_content = str(selected.get("content") or selected.get("text_excerpt") or "").strip()
    if selected_content and (not hint_norm or hint_norm in _live_section_label_norm(selected_label) or _live_section_label_norm(selected_label) in hint_norm):
        return {"label": selected_label or hint or "Selected section", "content": selected_content}

    sections = current.get("sections") if isinstance(current.get("sections"), list) else []
    best: tuple[int, dict[str, Any]] | None = None
    for section in sections:
        if not isinstance(section, dict):
            continue
        label = str(section.get("label") or "").strip()
        label_norm = _live_section_label_norm(label)
        if not label_norm:
            continue
        score = 0
        if hint_norm and hint_norm == label_norm:
            score = 100
        elif hint_norm and (hint_norm in label_norm or label_norm in hint_norm):
            score = 85
        elif hint_norm and abs(len(hint_norm) - len(label_norm)) <= 2 and _levenshtein_distance(hint_norm, label_norm) <= 2:
            score = 75
        if score and (best is None or score > best[0]):
            best = (score, section)

    if best:
        label = str(best[1].get("label") or hint or "Section").strip()
        content = str(best[1].get("content") or "").strip()
        full_text = str(current.get("full_text") or ctx.get("editor_excerpt") or "").strip()
        if full_text and label:
            lines = [line.rstrip() for line in re.split(r"\r?\n", full_text)]
            start = -1
            label_norm = _live_section_label_norm(label)
            for idx, line in enumerate(lines):
                if _live_section_label_norm(line) == label_norm:
                    start = idx
                    break
            if start >= 0:
                start_level = _section_heading_level(lines[start])
                end = len(lines)
                for idx in range(start + 1, len(lines)):
                    level = _section_heading_level(lines[idx])
                    if start_level is not None and level is not None and level <= start_level:
                        end = idx
                        break
                    if start_level is None and idx > start and _section_heading_level(lines[idx]) == 1:
                        end = idx
                        break
                content = "\n".join(line for line in lines[start:end] if line.strip()).strip() or content
        if content:
            return {"label": label, "content": content}

    full_text = str(current.get("full_text") or ctx.get("editor_excerpt") or "").strip()
    if full_text and re.search(r"\b(full|whole|entire|complete|sop|document)\b", question or "", re.IGNORECASE):
        return {"label": current.get("title") or current.get("sop_number") or "Full SOP", "content": full_text}
    return None


def _simple_live_summary(content: str, *, max_items: int = 4, max_chars: int = 120) -> list[str]:
    lines = [re.sub(r"\s+", " ", line).strip(" -•\t") for line in re.split(r"\r?\n+", str(content or "")) if line.strip()]
    useful = [
        line for line in lines
        if len(line) > 12 and not re.match(r"^\d+(?:\.\d+)*[.)\]:-]?\s+[A-Za-zÀ-ÿ\s/&()-]{1,80}$", line)
    ]
    if not useful:
        useful = [part.strip() for part in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", content or "")) if len(part.strip()) > 12]
    out: list[str] = []
    for line in useful[:max_items]:
        text = line.strip()
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "…"
        out.append(text)
    return out


def _deterministic_live_section_response(question: str, assistant_context: dict | None) -> dict[str, Any] | None:
    intents = _query_intents(question)
    if not ({"summary", "explain"} & intents):
        return None
    target = _resolve_live_section_for_answer(assistant_context, question)
    if not target:
        return None
    content = target["content"]
    label = target["label"]
    items = _simple_live_summary(content)
    src_words = len(re.findall(r"\b\w+\b", content or "", flags=re.UNICODE))
    if "summary" in intents:
        answer = f"Short summary of **{label}**"
        if src_words:
            answer += f" ({src_words} words in source → keep this brief)"
        answer += ":\n" + "\n".join(f"- {item}" for item in items)
    else:
        answer = f"**{label}** explains:\n" + "\n".join(f"- {item}" for item in items)
    if not items:
        answer = f"I found **{label}**, but there is not enough readable section text in the live editor context to summarize it."
    return {
        "answer": answer,
        "sources": [{"id": "live_editor_section", "type": "editor_context", "label": label}],
        "citations": [
            {
                "ref": "live_editor_section",
                "title": label,
                "type": "editor_context",
                "excerpt": content[:500],
                "score": 1.0,
            }
        ],
        "retrieval_debug": [],
        "suggestions": ["Rewrite this section", "Explain this in simple words", "Find gaps in this section"],
        "retrieval_stats": {
            "total_docs": 1,
            "source": "live_editor_section",
            "strict_mode": "active_editor_section",
        },
        "routed_to": "live_editor_section",
        "assistant_action": None,
    }


def _deterministic_sop_count_response(question: str, active_scope: dict | None = None) -> dict[str, Any]:
    db = SessionLocal()
    try:
        total = db.query(SOP).filter(SOP.is_active == True).count()  # noqa: E712
    finally:
        db.close()
    active_ref = str((active_scope or {}).get("active_sop_ref") or "").strip()
    active_title = str((active_scope or {}).get("title") or "").strip()
    answer = f"There are {total} SOP(s) in the database (active records)."
    if active_ref:
        answer += f" The currently active SOP is {active_ref}"
        if active_title:
            answer += f" ({active_title})"
        answer += "."
    citations = [
        {
            "ref": "SOP database",
            "title": "Active SOP records (PostgreSQL)",
            "type": "metadata",
            "excerpt": f"{total} active SOP(s) in the database.",
            "score": 1.0,
        }
    ]
    return {
        "answer": answer,
        "sources": [{"id": "SOP database", "type": "metadata", "label": "Active SOP records (PostgreSQL)"}],
        "citations": citations,
        "retrieval_debug": [],
        "suggestions": [
            "List all SOPs with titles",
            "What does the active SOP cover?",
            "Which deviations link to this SOP?",
        ],
        "retrieval_stats": {
            "total_docs": 0,
            "source": "deterministic_system_query",
            "strict_mode": "sop_inventory_count",
        },
        "routed_to": "system_query_sop_count",
        "assistant_action": None,
    }


def _summarize_live_context(assistant_context: dict | None, question: str = "") -> str:
    ctx = assistant_context or {}
    current = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}
    linked = ctx.get("linked_context") if isinstance(ctx.get("linked_context"), dict) else {}
    selected = ctx.get("selected_section") if isinstance(ctx.get("selected_section"), dict) else {}
    last_action = ctx.get("last_action") if isinstance(ctx.get("last_action"), dict) else {}
    tabs = _ctx_list(ctx.get("opened_tabs"))
    text = str(ctx.get("editor_excerpt") or "").strip()
    selected_text = str(ctx.get("selected_text") or "").strip()
    selected_section_content = str(selected.get("content") or selected.get("text_excerpt") or "").strip()
    references = _ctx_list(current.get("references"))
    intents = _query_intents(question)
    scope = _extract_active_sop_scope(assistant_context)
    active_sop_ref = str(scope.get("active_sop_ref") or current.get("sop_number") or current.get("id") or "").strip()
    active_sop_id = str(scope.get("active_sop_id") or "").strip()
    open_sop_tabs = _extract_refs(tabs, ["docId", "label"], limit=10)
    include_editor_excerpt = bool(text) and bool({"summary", "active_sop", "compare", "compliance", "explain"} & intents)
    active_full_sop_request = bool(
        re.search(r"\b(full|whole|entire|complete)\s+(?:sop|document|doc)\b", question or "", re.IGNORECASE)
        or re.search(r"\b(?:summarize|summary|explain|tell\s+me\s+about)\s+(?:this\s+|the\s+|current\s+|active\s+)?sop\b", question or "", re.IGNORECASE)
        or re.search(r"\bwhat\s+is\s+(?:this\s+|the\s+|current\s+|active\s+)?sop\s+about\b", question or "", re.IGNORECASE)
    )
    excerpt_limit = 6000 if active_full_sop_request else 1800
    excerpt = text[:excerpt_limit] if include_editor_excerpt else ""
    selected_source = selected_text or selected_section_content
    selected_excerpt = (
        selected_source[:1200]
        if selected_source and not active_full_sop_request and ({"summary", "compliance", "explain"} & intents or last_action)
        else ""
    )
    requested_section_context = ""
    if not active_full_sop_request and {"summary", "explain", "compliance"} & intents:
        try:
            target = _resolve_live_section_for_answer(assistant_context, question)
        except Exception as exc:
            logger.debug("[live-context] target section resolution failed: %s", exc)
            target = None
        if target:
            requested_section_context = (
                f"- Requested section: {target.get('label') or 'section'}\n"
                f"- Requested section text excerpt: {str(target.get('content') or '')[:4200]}\n"
            )
    last_action_line = ""
    last_action_context = ""
    if last_action:
        last_action_line = (
            f"- Last assistant action: action={last_action.get('action') or 'unknown'} | "
            f"scope={last_action.get('target_scope') or 'unknown'} | "
            f"section={last_action.get('section_name') or 'unknown'} | "
            f"status={last_action.get('status') or 'unknown'}\n"
        )
        original_excerpt = str(last_action.get("original_text_excerpt") or "").strip()
        suggested_excerpt = str(last_action.get("suggested_text_excerpt") or "").strip()
        context_lines = [
            "- Follow-up scope rule: if the user says this/it/that section, use the last assistant action target unless a new target is named.",
        ]
        if original_excerpt:
            context_lines.append(f"- Last target source excerpt: {original_excerpt[:1400]}")
        if suggested_excerpt:
            context_lines.append(f"- Last assistant output excerpt: {suggested_excerpt[:1400]}")
        last_action_context = "\n".join(context_lines) + "\n"
    from chatbot.assistant.context_intelligence import infer_output_language_from_context

    response_constraints = ctx.get("response_constraints") if isinstance(ctx.get("response_constraints"), dict) else {}
    sop_lang = infer_output_language_from_context(ctx)
    lang_note = ""
    if sop_lang == "de":
        lang_note = (
            "- RESPONSE LANGUAGE: German (de). Reply in natural conversational German matching the "
            "open SOP document prose — not English — unless the user explicitly asks for English.\n"
        )
    elif sop_lang == "en":
        lang_note = (
            "- RESPONSE LANGUAGE: English (en). Reply in natural conversational English matching the "
            "open SOP document prose unless the user explicitly asks for another language.\n"
        )
    chat_style_note = (
        "- Chat style: friendly SOP co-pilot — short plain paragraphs; no rigid Summary/Details "
        "templates, bullet forms, or HTML.\n"
    )
    format_note = lang_note + chat_style_note
    chat_submode = str(ctx.get("chat_submode") or "").strip().lower()
    if chat_submode == "sop_summarize" or "summary" in intents:
        format_note += (
            "- SUMMARY MODE (sidebar only — never rewrite the SOP): Output MUST be clearly shorter than "
            "the source section or document (aim ~25–40% of source length). Use 3–6 short bullets or "
            "at most 2 brief paragraphs. Be direct; do not paste headings or repeat the full text.\n"
        )
    if response_constraints.get("line_count"):
        n = int(response_constraints["line_count"])
        format_note += (
            f"- User output format: exactly {n} short lines as conversational sentences. "
            "FORBIDDEN: Summary/Details/Status/Cross-refs template.\n"
        )
    elif response_constraints.get("word_count"):
        format_note += f"- User output format: about {int(response_constraints['word_count'])} words.\n"

    focus_note = ""
    session_active = ctx.get("active_scope") if isinstance(ctx.get("active_scope"), dict) else {}
    session_section = str(session_active.get("section_label") or "").strip()
    session_action = str(session_active.get("last_action") or "").strip()
    if session_section or session_action:
        focus_note += (
            f"- Session target (this chat only): action={session_action or 'none'} | "
            f"section={session_section or 'none'} — keep follow-ups on this target unless the user names a new one.\n"
        )
    if "summary" in intents and active_sop_ref:
        focus_note = f"- Focus SOP for summary: {active_sop_ref}\n"
    elif "compare" in intents and open_sop_tabs:
        focus_note = f"- Compare candidates from open tabs: {', '.join(open_sop_tabs[:6])}\n"

    if active_sop_id and not _user_requests_global_scope(question):
        minimal = (
            "LIVE_ASSISTANT_CONTEXT\n"
            f"- Active SOP: {active_sop_ref or active_sop_id} | title={current.get('title') or 'unknown'} | "
            f"id={active_sop_id} | version={current.get('version') or 'unknown'} | "
            f"status={current.get('status') or 'unknown'}\n"
            "- Retrieval scope: ACTIVE SOP ONLY — use this SOP and its linked entities; "
            "do not search or summarize other SOPs unless the user explicitly asks for all SOPs.\n"
            f"- Linked counts: deviations={len(scope.get('linked_deviation_ids') or [])}, "
            f"capas={len(scope.get('linked_capa_ids') or [])}, "
            f"audits={len(scope.get('linked_audit_ids') or [])}, "
            f"decisions={len(scope.get('linked_decision_ids') or [])}\n"
        )
        if not {"summary", "linked", "compare"} & intents:
            return (
                minimal
                + last_action_line
                + last_action_context
                + format_note
                + requested_section_context
                + (
                    f"- Selected section: {selected.get('name') or 'none'} | scope={selected.get('scope') or 'none'}\n"
                    if selected or selected_excerpt
                    else ""
                )
                + (f"- Selected text excerpt: {selected_excerpt}\n" if selected_excerpt else "")
                + f"{focus_note}- Answer only what was asked; avoid unsolicited summaries."
            )

        linked_devs = _extract_refs(_ctx_list(linked.get("deviations")), ["deviation_number", "ref_number", "id"])
        linked_capas = _extract_refs(_ctx_list(linked.get("capas")), ["capa_number", "ref_number", "id"])
        linked_audits = _extract_refs(
            _ctx_list(linked.get("audits")), ["finding_number", "audit_number", "ref_number", "id"]
        )
        linked_decisions = _extract_refs(
            _ctx_list(linked.get("decisions")), ["decision_number", "ref_number", "id"]
        )
        related_sops = _extract_refs(_ctx_list(linked.get("related_sops")), ["sop_number", "ref_number", "id"])
        return (
            minimal
            + last_action_line
            + last_action_context
            + format_note
            + requested_section_context
            + f"- Linked deviations: {len(scope.get('linked_deviation_ids') or [])} ({', '.join(linked_devs) or 'none'})\n"
            + f"- Linked CAPAs: {len(scope.get('linked_capa_ids') or [])} ({', '.join(linked_capas) or 'none'})\n"
            + f"- Linked audits: {len(scope.get('linked_audit_ids') or [])} ({', '.join(linked_audits) or 'none'})\n"
            + f"- Linked decisions: {len(scope.get('linked_decision_ids') or [])} ({', '.join(linked_decisions) or 'none'})\n"
            + f"- Related SOPs: {len(scope.get('linked_sop_ids') or [])} ({', '.join(related_sops) or 'none'})\n"
            + f"- Open tabs: {len(tabs)}\n"
            + f"{focus_note}"
            + f"- References in editor metadata: {', '.join(str(r) for r in references[:10]) or 'none'}\n"
            + (
                f"- Selected section: {selected.get('name') or 'none'} | scope={selected.get('scope') or 'none'}\n"
                if selected or selected_excerpt
                else ""
            )
            + (f"- Selected text excerpt: {selected_excerpt}\n" if selected_excerpt else "")
            + f"- Editor text excerpt: {excerpt if excerpt else 'not injected for this query intent'}"
        )

    linked_devs = _extract_refs(_ctx_list(linked.get("deviations")), ["deviation_number", "ref_number", "id"])
    linked_capas = _extract_refs(_ctx_list(linked.get("capas")), ["capa_number", "ref_number", "id"])
    linked_audits = _extract_refs(
        _ctx_list(linked.get("audits")), ["finding_number", "audit_number", "ref_number", "id"]
    )
    linked_decisions = _extract_refs(
        _ctx_list(linked.get("decisions")), ["decision_number", "ref_number", "id"]
    )
    related_sops = _extract_refs(_ctx_list(linked.get("related_sops")), ["sop_number", "ref_number", "id"])
    return (
        "LIVE_ASSISTANT_CONTEXT\n"
        + f"- Active SOP: {current.get('sop_number') or current.get('id') or 'unknown'} | "
        + f"title={current.get('title') or 'unknown'} | version={current.get('version') or 'unknown'} | "
        + f"status={current.get('status') or 'unknown'}\n"
        + f"{last_action_line}"
        + f"{last_action_context}"
        + format_note
        + f"{requested_section_context}"
        + (
            f"- Selected section: {selected.get('name') or 'none'} | scope={selected.get('scope') or 'none'}\n"
            if selected or selected_excerpt
            else ""
        )
        + (f"- Selected text excerpt: {selected_excerpt}\n" if selected_excerpt else "")
        + f"- Linked deviations: {len(_ctx_list(linked.get('deviations')))} ({', '.join(linked_devs) or 'none'})\n"
        + f"- Linked CAPAs: {len(_ctx_list(linked.get('capas')))} ({', '.join(linked_capas) or 'none'})\n"
        + f"- Linked audits: {len(_ctx_list(linked.get('audits')))} ({', '.join(linked_audits) or 'none'})\n"
        + f"- Linked decisions: {len(_ctx_list(linked.get('decisions')))} ({', '.join(linked_decisions) or 'none'})\n"
        + f"- Related SOPs: {len(_ctx_list(linked.get('related_sops')))} ({', '.join(related_sops) or 'none'})\n"
        + f"- Open tabs: {len(tabs)}\n"
        + f"{focus_note}"
        + f"- References in editor metadata: {', '.join(str(r) for r in references[:10]) or 'none'}\n"
        + f"- Editor text excerpt: {excerpt if excerpt else 'not injected for this query intent'}"
    )


def _resolve_sop_from_context(db, assistant_context: dict | None, question: str) -> SOP | None:
    ctx = assistant_context or {}
    current = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}
    for raw_id in [current.get("id"), current.get("sop_number"), ctx.get("current_document_id")]:
        value = str(raw_id or "").strip()
        if not value:
            continue
        try:
            doc_uuid = uuid.UUID(value)
            sop = db.query(SOP).filter(SOP.id == doc_uuid, SOP.is_active == True).first()  # noqa: E712
        except ValueError:
            sop = db.query(SOP).filter(SOP.sop_number.ilike(value), SOP.is_active == True).first()  # noqa: E712
        if sop:
            return sop
    match = SOP_REF_PATTERN.search(question or "")
    if match:
        return db.query(SOP).filter(
            SOP.sop_number.ilike(match.group(0).upper()),
            SOP.is_active == True,  # noqa: E712
        ).first()
    return None


def _title_from_question(question: str) -> str:
    q = (question or "").strip()
    m = re.search(r"(?:title|named|called)\s*[:\-]?\s*([A-Za-z0-9 _\-/]{4,120})", q, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    for_match = re.search(r"\bfor\s+([A-Za-z0-9 _\-/]{3,120})", q, re.IGNORECASE)
    if for_match:
        core = for_match.group(1).strip(" .")
        if core:
            return f"{core.title()} SOP"
    cleaned = re.sub(r"\b(create|new|add|generate|draft)\b", "", q, flags=re.IGNORECASE).strip(" :.-")
    cleaned = re.sub(r"\b(an?|the)\b", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\bsop\b", "", cleaned, flags=re.IGNORECASE).strip(" :.-")
    return cleaned[:120] if cleaned else "Untitled SOP"


def _build_minimal_tiptap_doc(text: str) -> dict:
    content = str(text or "").strip()
    lines = [line.strip() for line in re.split(r"\r?\n+", content) if line.strip()]
    if not lines:
        lines = ["New SOP draft"]
    blocks = [
        {"type": "paragraph", "content": [{"type": "text", "text": line[:1200]}]}
        for line in lines[:80]
    ]
    return {
        "type": "doc",
        "content": blocks,
    }


def _plan_sop_action(question: str, assistant_context: dict | None) -> dict | None:
    q = (question or "")
    if ACTION_INTENT_DELETE.search(q):
        preview_target = {}
        ctx = assistant_context or {}
        current = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}
        preview_target["sop_number"] = current.get("sop_number") or current.get("documentId") or ""
        preview_target["title"] = current.get("title") or ""
        return {"type": "delete_sop", "requires_confirmation": True, "target": preview_target}
    if ACTION_INTENT_CREATE.search(q):
        return {"type": "create_sop", "requires_confirmation": False}
    if ACTION_INTENT_UPDATE.search(q):
        mode = "append" if re.search(r"\badd\b.*\bsection\b", q, re.IGNORECASE) else "replace"
        return {"type": "update_sop", "requires_confirmation": False, "mode": mode}
    return None


def _execute_sop_action(
    action_type: str,
    question: str,
    assistant_context: dict | None,
    generated_text: str = "",
    mode: str = "replace",
) -> dict | None:
    if not action_type:
        return None
    db = SessionLocal()
    try:
        if action_type == "delete_sop":
            sop = _resolve_sop_from_context(db, assistant_context, question)
            if not sop:
                return {"type": "delete_sop", "ok": False, "message": "No active SOP could be resolved for deletion."}
            logger.info("[assistant-delete] requested sop_id=%s sop_number=%s current_is_active=%s", sop.id, sop.sop_number, sop.is_active)
            sop.is_active = False
            db.commit()
            db.refresh(sop)
            logger.info("[assistant-delete] committed sop_id=%s sop_number=%s persisted_is_active=%s", sop.id, sop.sop_number, sop.is_active)
            return {
                "type": "delete_sop",
                "ok": True,
                "sop_id": str(sop.id),
                "sop_number": sop.sop_number,
                "message": f"SOP {sop.sop_number} was soft deleted.",
            }

        if action_type == "update_sop":
            sop = _resolve_sop_from_context(db, assistant_context, question)
            if not sop:
                return {"type": "update_sop", "ok": False, "message": "No active SOP could be resolved for update."}
            current = (
                db.query(SOPVersion).filter(SOPVersion.id == sop.current_version_id).first()
                if sop.current_version_id else
                db.query(SOPVersion).filter(SOPVersion.sop_id == sop.id).order_by(SOPVersion.created_at.desc()).first()
            )
            if not current:
                return {"type": "update_sop", "ok": False, "message": "Current SOP version not found."}
            current_text = _extract_text_from_tiptap((current.content_json or {}))
            llm_text = str(generated_text or "").strip()
            if mode == "append" and llm_text:
                next_text = f"{current_text}\n\n{llm_text}".strip()
            else:
                next_text = llm_text or current_text
            current.content_json = _build_minimal_tiptap_doc(next_text[:18000])
            db.commit()
            return {
                "type": "update_sop",
                "ok": True,
                "sop_id": str(sop.id),
                "sop_number": sop.sop_number,
                "message": f"SOP {sop.sop_number} was updated.",
            }

        if action_type == "create_sop":
            title = _title_from_question(question)
            sop_number = f"SOP-{uuid.uuid4().hex[:8].upper()}"
            while db.query(SOP).filter(SOP.sop_number == sop_number).first():
                sop_number = f"SOP-{uuid.uuid4().hex[:8].upper()}"
            sop_id = uuid.uuid4()
            ver_id = uuid.uuid4()
            tenant_row = db.query(SOP.tenant_id).first()
            tenant_id = tenant_row[0] if tenant_row else uuid.UUID("00000000-0000-0000-0000-000000000000")
            sop = SOP(
                id=sop_id,
                tenant_id=tenant_id,
                sop_number=sop_number,
                title=title,
                department="Quality",
                is_active=True,
                current_version_id=ver_id,
            )
            draft_text = str(generated_text or "").strip()
            version = SOPVersion(
                id=ver_id,
                sop_id=sop_id,
                version_number="1",
                external_status="draft",
                content_json=_build_minimal_tiptap_doc(draft_text[:8000]),
                metadata_json={"sopStatus": "draft", "sopMetadata": {"title": title, "documentId": sop_number}},
            )
            db.add(sop)
            db.add(version)
            db.commit()
            return {
                "type": "create_sop",
                "ok": True,
                "sop_id": str(sop_id),
                "sop_number": sop_number,
                "title": title,
                "message": f"Created new SOP {sop_number}.",
            }
    finally:
        db.close()
    return None


@ai_router.post("/api/ai/action", response_model=AIActionResponse)
async def perform_ai_action(payload: AIActionRequest):
    """
    Perform a structured AI action on selected SOP text.
    The current implementation uses deterministic structured generation so the
    frontend can reliably support compare-and-confirm workflows.
    """
    action = _normalize_action(payload.action)
    raw_in = (payload.text or "").strip()
    payload.text = normalize_action_input_text(payload.text)
    if not payload.text:
        raise HTTPException(status_code=422, detail="Selected text is required.")

    request_preview = _build_action_request(payload)
    edit_scope = resolve_edit_scope(request_preview)
    logger.info(
        "[ai-action-request] action=%s edit_scope=%s section=%s section_type=%s text_in_len=%s text_norm_len=%s preview=%s",
        action,
        edit_scope,
        request_preview.section_title,
        request_preview.section_type,
        len(raw_in),
        len(payload.text),
        _truncate_text(raw_in.replace("\n", " "), 220),
    )

    triggered_by = (getattr(payload, "triggered_by", None) or "").strip() or "unknown"
    logger.info(
        "[ai-action-prompt-source] action=%s source_file=%s triggered_by=%s",
        action,
        AI_ACTION_PROMPT_SOURCE_FILE,
        triggered_by,
    )

    try:
        out = await asyncio.to_thread(_run_dynamic_ai_action, payload, action)
        logger.info(
            "[ai-action-result] action=%s ok=1 original_len=%s suggested_len=%s",
            action,
            len(out.original_text or ""),
            len(out.suggested_text or ""),
        )
        # Log the AI action to the database
        db = SessionLocal()
        try:
            from .models import AIActionLog
            log_entry = AIActionLog(
                action=action,
                sop_title=payload.sop_title,
                section_name=payload.section_name,
                section_type=payload.section_type,
                original_text=payload.text,
                suggested_text=out.suggested_text or "",
                explanation=out.explanation or "",
                structured_data=out.structured_data,
            )
            db.add(log_entry)
            db.flush()
            if action in {"rewrite", "improve"}:
                sop_id = None
                sop_version_id = None
                profile_id = None
                profile_version_id = None
                if payload.sop_entity_id:
                    try:
                        sop_id = uuid.UUID(str(payload.sop_entity_id))
                        sop_row = db.query(SOP).filter(SOP.id == sop_id).first()
                        if sop_row:
                            sop_version_id = sop_row.current_version_id
                        detected = db.query(SOPDetectedParameters).filter(
                            SOPDetectedParameters.sop_id == sop_id
                        ).order_by(SOPDetectedParameters.created_at.desc()).first()
                        if detected and detected.client_profile_id:
                            profile_id = detected.client_profile_id
                            profile = db.query(ClientProfile).filter(ClientProfile.id == profile_id).first()
                            if profile:
                                profile_version_id = profile.current_version_id
                    except Exception as suggestion_ctx_exc:
                        logger.warning("[ai-suggestion] failed to resolve context: %s", suggestion_ctx_exc)
                if sop_id is None and payload.sop_title:
                    sop_row = (
                        db.query(SOP)
                        .filter(or_(SOP.sop_number == payload.sop_title, SOP.title == payload.sop_title))
                        .first()
                    )
                    if sop_row:
                        sop_id = sop_row.id
                        sop_version_id = sop_row.current_version_id
                        detected = db.query(SOPDetectedParameters).filter(
                            SOPDetectedParameters.sop_id == sop_id
                        ).order_by(SOPDetectedParameters.created_at.desc()).first()
                        if detected and detected.client_profile_id:
                            profile_id = detected.client_profile_id
                            profile = db.query(ClientProfile).filter(ClientProfile.id == profile_id).first()
                            if profile:
                                profile_version_id = profile.current_version_id
                used_profile_id = None
                if isinstance(out.structured_data, dict):
                    used_profile_id = out.structured_data.get("used_profile_id")
                if used_profile_id and str(profile_id or "") != str(used_profile_id):
                    try:
                        profile_id = uuid.UUID(str(used_profile_id))
                        profile = db.query(ClientProfile).filter(ClientProfile.id == profile_id).first()
                        if profile:
                            profile_version_id = profile.current_version_id
                    except Exception as profile_override_exc:
                        logger.warning("[ai-suggestion] failed to apply used_profile_id override: %s", profile_override_exc)
                suggestion = AISuggestion(
                    action_log_id=log_entry.id,
                    sop_id=sop_id,
                    sop_version_id=sop_version_id,
                    action=action,
                    target_scope=edit_scope,
                    original_text=payload.text,
                    suggested_text=out.suggested_text or "",
                    profile_id=profile_id,
                    profile_version_id=profile_version_id,
                    status="pending",
                    metadata_json={
                        "sop_title": payload.sop_title,
                        "section_name": payload.section_name,
                        "section_type": payload.section_type,
                        "instruction": getattr(payload, "instruction", None),
                        "learn_to_profile": bool(getattr(payload, "learn_to_profile", False)),
                        "structured_data": out.structured_data,
                    },
                )
                db.add(suggestion)
                db.flush()
                out.structured_data = dict(out.structured_data or {})
                out.structured_data["suggestion_id"] = str(suggestion.id)
                out.structured_data["suggestion_status"] = suggestion.status
                out.structured_data["requires_user_acceptance"] = True
                out.structured_data["learn_to_profile_on_accept"] = bool(getattr(payload, "learn_to_profile", False))
                log_entry.structured_data = out.structured_data
            db.commit()
            logger.info("[ai-action-result] successfully logged to database table ai_action_logs")
        except Exception as log_exc:
            logger.warning("Failed to save AI action log: %s", log_exc)
            db.rollback()
        finally:
            db.close()

        return out
    except HTTPException:
        raise
    except Exception:
        if action == "gap_check":
            return _fallback_gap_check(payload)
        if action == "rewrite":
            return _fallback_rewrite(payload)
        if action == "improve":
            return _fallback_improve(payload)
        if action == "summarize":
            fb = _fallback_improve(payload)
            return AIActionResponse(
                action="summarize",
                original_text=fb.original_text,
                suggested_text=fb.suggested_text,
                explanation="Kurzfassung (Fallback) / Executive summary (fallback).",
                structured_data=fb.structured_data,
            )
        if action == "analyze":
            fb = _fallback_improve(payload)
            return AIActionResponse(
                action="analyze",
                original_text=fb.original_text,
                suggested_text=fb.suggested_text,
                explanation="Analyse (Fallback) / Analysis (fallback).",
                structured_data=fb.structured_data,
            )

    raise HTTPException(status_code=400, detail=f"Action '{payload.action}' is not supported.")


@ai_router.get("/api/ai/suggestions/{suggestion_id}")
async def get_ai_suggestion(suggestion_id: uuid.UUID):
    db = SessionLocal()
    try:
        suggestion = db.query(AISuggestion).filter(AISuggestion.id == suggestion_id).first()
        if not suggestion:
            raise HTTPException(status_code=404, detail="AI suggestion not found")
        return {
            "id": str(suggestion.id),
            "action": suggestion.action,
            "status": suggestion.status,
            "sop_id": str(suggestion.sop_id) if suggestion.sop_id else None,
            "sop_version_id": str(suggestion.sop_version_id) if suggestion.sop_version_id else None,
            "accepted_version_id": str(suggestion.accepted_version_id) if suggestion.accepted_version_id else None,
            "target_scope": suggestion.target_scope,
            "created_at": suggestion.created_at,
            "accepted_at": suggestion.accepted_at,
        }
    finally:
        db.close()


@ai_router.post("/api/ai/suggestions/{suggestion_id}/status")
async def update_ai_suggestion_status(suggestion_id: uuid.UUID, payload: dict):
    status = str(payload.get("status") or "").strip().lower()
    if status not in {"pending", "accepted", "rejected"}:
        raise HTTPException(status_code=422, detail="status must be pending, accepted, or rejected")
    db = SessionLocal()
    try:
        suggestion = db.query(AISuggestion).filter(AISuggestion.id == suggestion_id).first()
        if not suggestion:
            raise HTTPException(status_code=404, detail="AI suggestion not found")
        suggestion.status = status
        if status == "accepted":
            suggestion.accepted_at = datetime.utcnow()
            accepted_version_id = payload.get("accepted_version_id")
            if accepted_version_id:
                suggestion.accepted_version_id = uuid.UUID(str(accepted_version_id))
        meta = dict(suggestion.metadata_json or {})
        meta["status_change_reason"] = payload.get("reason")
        suggestion.metadata_json = meta
        db.commit()
        db.refresh(suggestion)
        return {
            "id": str(suggestion.id),
            "status": suggestion.status,
            "accepted_version_id": str(suggestion.accepted_version_id) if suggestion.accepted_version_id else None,
        }
    finally:
        db.close()


@ai_router.post("/api/ai/classify-intent")
async def classify_intent(payload: dict):
    """
    Semantic intent routing for the unified KL/KI Assistant chat panel.
    Returns flow (chat | editor_action | clarify), action, target scope, and constraints.
    """
    from chatbot.assistant.context_intelligence import (
        build_intent_classifier_invoke_context,
        finalize_classify_response,
        is_follow_up_edit_refinement,
        is_meta_question_about_assistant_output,
        is_read_only_sop_query,
        prepare_message_context,
        use_llm_orchestrator,
    )
    from chatbot.assistant.intent_classifier import classify_assistant_intent

    message = (payload.get("message") or payload.get("question") or "").strip()
    if not message:
        raise HTTPException(status_code=422, detail="message is required")

    prep = prepare_message_context(payload)

    def _classify_return_llm(raw: dict) -> dict:
        return finalize_classify_response(raw, prep, user_message=message)

    if use_llm_orchestrator():
        invoke_ctx = build_intent_classifier_invoke_context(payload, prep)
        result = await asyncio.to_thread(
            classify_assistant_intent,
            message,
            has_active_sop=invoke_ctx["has_active_sop"],
            has_editor_selection=invoke_ctx["has_editor_selection"],
            route=invoke_ctx["route"],
            active_sop_title=invoke_ctx["active_sop_title"],
            active_sop_number=invoke_ctx["active_sop_number"],
            selected_section_summary=invoke_ctx["selected_section_summary"],
            available_sections=invoke_ctx["available_sections"],
            previous_action_summary=invoke_ctx["previous_action_summary"],
            recent_conversation=invoke_ctx["recent_conversation"],
            active_scope=invoke_ctx.get("active_scope"),
            instruction_memory=invoke_ctx.get("instruction_memory"),
            frustration_signal=invoke_ctx.get("frustration_signal"),
            repetition_detected=invoke_ctx["repetition_detected"],
            repetition_instruction=invoke_ctx.get("repetition_instruction"),
            resolved_scope_hint=invoke_ctx.get("resolved_scope_hint") or "-",
            query_analysis_hint=invoke_ctx.get("query_analysis_hint") or "-",
        )
        raw = result.model_dump()
        raw["reasoning"] = raw.get("reasoning") or "llm_orchestrator"
        prev = invoke_ctx.get("previous_action")
        if prev:
            raw["previous_action"] = prev
        return _classify_return_llm(raw)

    if prep.get("early_response"):
        return finalize_classify_response(prep["early_response"], prep, user_message=message)

    def _classify_return(raw: dict) -> dict:
        return finalize_classify_response(raw, prep, user_message=message)

    resolved_scope = prep.get("resolved_scope") if isinstance(prep.get("resolved_scope"), dict) else {}
    from chatbot.assistant.context_intelligence import (
        detect_edit_action_from_message,
        is_imperative_edit_command,
        message_specifies_new_target,
    )

    ctx_early = payload.get("assistant_context") if isinstance(payload.get("assistant_context"), dict) else {}
    prev_early = ctx_early.get("last_action") if isinstance(ctx_early.get("last_action"), dict) else {}
    if (
        prep.get("has_active_sop")
        and isinstance(resolved_scope, dict)
        and message_specifies_new_target(resolved_scope)
        and is_imperative_edit_command(
            message,
            previous_action=prev_early,
            resolved_scope=resolved_scope,
        )
    ):
        return _classify_return({
            "flow": "editor_action",
            "action": detect_edit_action_from_message(message, previous_action=prev_early),
            "target_scope": str(resolved_scope.get("target_scope") or "section"),
            "section_hint": str(resolved_scope.get("section_label") or "").strip() or None,
            "resolved_scope": resolved_scope,
            "linked_entity_types": [],
            "constraints": {},
            "requires_selection": False,
            "requires_confirmation": True,
            "confidence": 0.96,
            "reasoning": "explicit_section_from_scope_resolution",
            "previous_action": prev_early or None,
        })
    alias_resolved_label = (
        str(resolved_scope.get("section_label") or "").strip()
        if resolved_scope.get("resolved_from") in {"ALIAS_MATCH", "SUB_SECTION", "RECORD_ID"}
        else ""
    )

    ctx = payload.get("assistant_context") if isinstance(payload.get("assistant_context"), dict) else {}
    current_sop = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}
    last_action = ctx.get("last_action") if isinstance(ctx.get("last_action"), dict) else {}
    last_focus = ctx.get("last_focus") if isinstance(ctx.get("last_focus"), dict) else {}
    selected_section = ctx.get("selected_section") if isinstance(ctx.get("selected_section"), dict) else {}
    editor_contract = ctx.get("editor_context_contract") if isinstance(ctx.get("editor_context_contract"), dict) else {}
    recent_messages = payload.get("recent_messages") if isinstance(payload.get("recent_messages"), list) else []

    has_active_sop = bool(payload.get("has_active_sop"))
    if not has_active_sop:
        has_active_sop = bool(
            str(ctx.get("active_sop_id") or ctx.get("current_document_id") or "").strip()
            or str(current_sop.get("id") or "").strip()
        )

    def _recent_message_texts() -> list[str]:
        out: list[str] = []
        for row in recent_messages[-8:]:
            if not isinstance(row, dict):
                continue
            content = str(row.get("content") or "").strip()
            if content:
                out.append(content)
        return out

    def _infer_previous_action() -> dict[str, Any]:
        from chatbot.assistant.context_intelligence import (
            build_session_from_payload,
            enrich_sections_with_aliases,
            resolve_effective_previous_action,
        )

        sop_sections = current_sop.get("sections") if isinstance(current_sop.get("sections"), list) else []
        if not sop_sections:
            sop_ctx = editor_contract.get("sop_context") if isinstance(editor_contract.get("sop_context"), dict) else {}
            sop_sections = sop_ctx.get("sections") if isinstance(sop_ctx.get("sections"), list) else []
        sections_enriched = enrich_sections_with_aliases(sop_sections)
        session = build_session_from_payload(payload)
        inferred = resolve_effective_previous_action(
            ctx,
            session,
            sections=sections_enriched,
            recent_messages=recent_messages,
        )
        if inferred:
            return inferred
        return {}

    previous_action = _infer_previous_action()

    def _compact_text(value: object, limit: int = 700) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        return text[:limit]

    def _selected_section_summary() -> str:
        contract_selected = editor_contract.get("selected_section") if isinstance(editor_contract.get("selected_section"), dict) else {}
        label = (
            selected_section.get("label")
            or selected_section.get("name")
            or contract_selected.get("label")
            or ""
        )
        content = (
            selected_section.get("content")
            or selected_section.get("text_excerpt")
            or contract_selected.get("content")
            or ""
        )
        return _compact_text(f"{label or 'none'} :: {content or ''}", 900)

    def _available_sections_summary() -> str:
        sections = current_sop.get("sections")
        if not isinstance(sections, list):
            sop_ctx = editor_contract.get("sop_context") if isinstance(editor_contract.get("sop_context"), dict) else {}
            sections = sop_ctx.get("sections") if isinstance(sop_ctx.get("sections"), list) else []
        labels = []
        for section in sections[:30]:
            if isinstance(section, dict):
                label = str(section.get("label") or "").strip()
                if label:
                    labels.append(label)
        return _compact_text(", ".join(labels), 1000)

    def _available_section_labels() -> list[str]:
        sections = current_sop.get("sections")
        if not isinstance(sections, list):
            sop_ctx = editor_contract.get("sop_context") if isinstance(editor_contract.get("sop_context"), dict) else {}
            sections = sop_ctx.get("sections") if isinstance(sop_ctx.get("sections"), list) else []
        labels: list[str] = []
        for section in sections[:80]:
            if not isinstance(section, dict):
                continue
            label = str(section.get("label") or section.get("name") or "").strip()
            if label:
                labels.append(label)
        return labels

    def _normalize_section_label(value: object) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip()).lower()
        text = re.sub(r"^\d+(?:\.\d+)*[.)\]:-]?\s*", "", text).strip()
        return text

    def _canonical_section_hint(value: object) -> str:
        raw = re.sub(r"\s+", " ", str(value or "").strip())
        if not raw:
            return ""
        raw_norm = _normalize_section_label(raw)
        for label in _available_section_labels():
            label_norm = _normalize_section_label(label)
            if raw.lower() == label.lower() or (raw_norm and raw_norm == label_norm):
                return label
        for label in _available_section_labels():
            label_norm = _normalize_section_label(label)
            if raw_norm and (raw_norm in label_norm or label_norm in raw_norm):
                return label
        close_match = ""
        close_distance = 999
        for label in _available_section_labels():
            label_norm = _normalize_section_label(label)
            if not raw_norm or not label_norm or abs(len(raw_norm) - len(label_norm)) > 2:
                continue
            distance = _levenshtein_distance(raw_norm, label_norm)
            if distance < close_distance:
                close_distance = distance
                close_match = label
        if close_match and close_distance <= 2:
            return close_match
        return raw

    def _previous_action_summary() -> str:
        if not previous_action:
            if isinstance(last_focus, dict) and last_focus:
                return _compact_text(
                    " | ".join(
                        [
                            "focus=section",
                            f"scope={last_focus.get('target_scope') or ''}",
                            f"section={last_focus.get('section_name') or ''}",
                            f"source={last_focus.get('source') or ''}",
                        ]
                    ),
                    900,
                )
            return ""
        return _compact_text(
            " | ".join(
                [
                    f"action={previous_action.get('action') or ''}",
                    f"scope={previous_action.get('target_scope') or ''}",
                    f"section={previous_action.get('section_name') or ''}",
                    f"prompt={previous_action.get('request_prompt') or ''}",
                    f"status={previous_action.get('status') or ''}",
                ]
            ),
            900,
        )

    def _recent_conversation_summary() -> str:
        rows = []
        for row in recent_messages[-8:]:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role") or "").strip() or "unknown"
            content = _compact_text(row.get("content"), 260)
            if content:
                rows.append(f"{role}: {content}")
        return "\n".join(rows)

    def _extract_explicit_section_hint(text: str) -> str:
        q = str(text or "").strip()
        if not q:
            return ""
        if re.search(r"\b(current|this|selected)(?:\s+\w+){0,2}\s+(?:section|paragraph|table\s+row|row|text)\b", q, re.IGNORECASE):
            return ""
        patterns = [
            r"\b(?:the\s+)?([A-Za-zÀ-ÿ][\wÀ-ÿ\s/&()-]{1,80}?)\s+section\b",
            r"\bsection\s+([A-Za-zÀ-ÿ][\wÀ-ÿ\s/&()-]{1,80})\b",
        ]
        stop = {
            "selected text",
            "this",
            "that",
            "it",
            "them",
            "now",
            "same",
            "previous",
            "suggestion",
            "summarize this",
            "explain this",
            "rewrite this",
            "improve this",
            "this section",
            "that section",
            "full sop",
            "sop",
            "current",
            "current section",
            "open sop",
        }
        for pattern in patterns:
            match = re.search(pattern, q, re.IGNORECASE)
            if not match:
                continue
            candidate = re.sub(r"\s+", " ", match.group(1).strip(" .:-"))
            if re.match(r"^(?:in|with|using|according|based|same|the\s+same)\b", candidate, re.IGNORECASE):
                continue
            candidate = re.sub(
                r"^(?:ok(?:ay)?\s+|now\s+|then\s+|please\s+|make\s+(?:the\s+)?|rewrite\s+(?:the\s+)?|improve\s+(?:the\s+)?)",
                "",
                candidate,
                flags=re.IGNORECASE,
            ).strip(" .:-")
            lowered_candidate = candidate.lower()
            if re.match(r"^(?:according|using|with|based|but|and|for|to)\b", lowered_candidate):
                continue
            if re.search(r"\b(this|that|it|them)\b", lowered_candidate) and not re.search(
                r"\b(zweck|zwect|sweck|purpose|scope|geltungsbereich|procedure|verfahren|responsibilities|responsibility|verantwortlichkeiten|capas?|capa|decisions?|entscheidungen?|audits?|deviations?|approval|records|definitions)\b",
                lowered_candidate,
                re.IGNORECASE,
            ):
                continue
            known_in_candidate = re.search(
                r"\b(zweck|zwect|sweck|purpose|scope|geltungsbereich|procedure|verfahren|responsibilities|responsibility|verantwortlichkeiten|capas?|capa|decisions?|entscheidungen?|audits?|deviations?|approval|records|definitions)\b",
                candidate,
                re.IGNORECASE,
            )
            if known_in_candidate:
                return known_in_candidate.group(1)
            if candidate and candidate.lower() not in stop:
                return candidate
        known = re.search(
            r"\b(zweck|zwect|sweck|purpose|scope|geltungsbereich|procedure|verfahren|responsibilities|responsibility|verantwortlichkeiten|capas?|capa|decisions?|entscheidungen?|audits?|deviations?|approval|records|definitions)\b",
            q,
            re.IGNORECASE,
        )
        return known.group(1) if known else ""

    explicit_section_hint = _canonical_section_hint(_extract_explicit_section_hint(message))

    def _content_target_scope(default_scope: str = "current_section") -> str:
        if explicit_section_hint:
            return "section"
        if bool(payload.get("has_editor_selection")):
            return "selection"
        return default_scope

    def _usable_followup_section_hint() -> str:
        for candidate in [
            explicit_section_hint,
            previous_action.get("section_name") if isinstance(previous_action, dict) else "",
            last_focus.get("section_name") if isinstance(last_focus, dict) else "",
            selected_section.get("label"),
            selected_section.get("name"),
        ]:
            value = str(candidate or "").strip()
            if value and value.lower() not in {"selected text", "selection", "previous suggestion", "full sop", "full document"}:
                return _canonical_section_hint(value)
        return ""

    def _is_contextual_target_ref(text: str) -> bool:
        q = str(text or "").strip().lower()
        return bool(re.search(r"\b(this|that|it|same|current|previous)\s*(?:section|part|block|text|one)?\b", q, re.IGNORECASE))

    def _section_hint_for_action() -> str:
        if alias_resolved_label:
            return alias_resolved_label
        if re.search(r"\bselected\s+(?:paragraph|table\s+row|row|text)\b", message, re.IGNORECASE):
            return ""
        if explicit_section_hint:
            return explicit_section_hint
        if _is_contextual_target_ref(message):
            return _usable_followup_section_hint()
        return ""

    def _is_context_dependent_followup(text: str) -> bool:
        return is_follow_up_edit_refinement(text, previous_action)

    def _is_explicit_selection_reference(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:selected|highlighted)\s+(?:text|paragraph|table\s+row|row|sentence|line|content)\b",
                str(text or ""),
                re.IGNORECASE,
            )
        )

    def _is_read_only_followup_about_action(text: str) -> bool:
        return is_meta_question_about_assistant_output(text)

    def _is_read_only_explain_request(text: str) -> bool:
        return is_read_only_sop_query(text)

    lower_message = message.lower()
    full_sop_read_request = bool(
        re.search(r"\b(full|whole|entire|complete)\s+(?:sop|document|doc)\b", lower_message, re.IGNORECASE)
        or re.search(r"\bopen\s+sop\b", lower_message, re.IGNORECASE)
        or re.search(r"\b(?:this|current|active)\s+sop\b", lower_message, re.IGNORECASE)
        or re.search(r"\btell\s+me\s+about\s+this\s+sop\b", lower_message, re.IGNORECASE)
    )
    if has_active_sop and re.search(r"\bcompare\b[\s\S]*\b(SOP-[A-Z0-9-]+)\b[\s\S]*\b(missing|fehlt|lacks?|gap)\b|\b(missing|fehlt|lacks?|gap)\b[\s\S]*\bcompare\b", message, re.IGNORECASE):
        return _classify_return({
            "flow": "chat",
            "action": None,
            "target_scope": None,
            "section_hint": _usable_followup_section_hint() or None,
            "linked_entity_types": ["related_sops"],
            "requires_selection": False,
            "requires_confirmation": False,
            "confidence": 0.94,
            "reason": "Read-only comparison against another SOP; answer in chat using RAG/live context.",
            "previous_action": previous_action,
        })
    if has_active_sop and _is_read_only_explain_request(message):
        return _classify_return({
            "flow": "chat",
            "action": None,
            "target_scope": None,
            "section_hint": None if full_sop_read_request else (_usable_followup_section_hint() or None),
            "requires_selection": False,
            "requires_confirmation": False,
            "confidence": 0.94,
            "reason": "Read-only explanation/question about SOP context; answer in chat, do not mutate the editor.",
            "previous_action": previous_action,
        })

    if has_active_sop and _is_read_only_followup_about_action(message):
        return _classify_return({
            "flow": "chat",
            "action": None,
            "target_scope": None,
            "section_hint": _usable_followup_section_hint() or None,
            "requires_selection": False,
            "requires_confirmation": False,
            "confidence": 0.96,
            "reason": "Read-only question about the previous editor action; answer in chat, do not run another edit.",
            "previous_action": previous_action,
        })

    if has_active_sop and _is_context_dependent_followup(message) and not _is_explicit_selection_reference(message):
        followup_section_hint = _usable_followup_section_hint() or alias_resolved_label
        frustration = prep.get("frustration_signal") or {}
        constraints: dict[str, Any] = {
            "length": "shorter"
            if re.search(r"\b(shorter|shoter|shorten|smaller|smallier|too\s+long|i\s+told\s+you)\b", message, re.IGNORECASE)
            else None
        }
        if frustration.get("detected") and frustration.get("target_word_count"):
            constraints["length"] = "shorter"
            constraints["word_count"] = frustration["target_word_count"]
        return _classify_return({
            "flow": "follow_up_action",
            "action": (
                "gap_check"
                if re.search(r"\bgap\s*check|compliance\b", message, re.IGNORECASE)
                else "improve"
                if re.search(r"\b(better|improve|verbesser)\b", message, re.IGNORECASE)
                else "rewrite"
                if re.search(r"\b(rewrite|re-?write|shorter|shoter|shorten|smaller|smallier|too\s+long|i\s+told\s+you)\b", message, re.IGNORECASE)
                else "summarize"
                if re.search(r"\bsummarize\s+(?:it|this|that)\b", message, re.IGNORECASE)
                else str(previous_action.get("action") or "improve")
            ),
            "target_scope": "section" if followup_section_hint else "previous_suggestion",
            "section_hint": followup_section_hint or None,
            "constraints": constraints,
            "requires_selection": False,
            "requires_confirmation": True,
            "confidence": 0.91,
            "reason": "Follow-up request targets the previous assistant action and its target.",
            "previous_action": previous_action,
        })

    if has_active_sop:
        action_section_hint = _section_hint_for_action()
        if action_section_hint.lower() in {"audit"} and re.search(r"\baudit[-\s]?ready\b", lower_message, re.IGNORECASE):
            action_section_hint = str(
                selected_section.get("label")
                or selected_section.get("name")
                or (last_focus.get("section_name") if isinstance(last_focus, dict) else "")
                or ""
            ).strip()
        if re.search(r"\blinked\b[\s\S]*\b(deviations?|capas?|qa)\b|\b(deviations?|capas?)\b[\s\S]*\blinked\b", lower_message, re.IGNORECASE):
            return _classify_return({
                "flow": "chat",
                "action": None,
                "target_scope": None,
                "section_hint": None,
                "linked_entity_types": ["deviations", "capas"],
                "requires_selection": False,
                "requires_confirmation": False,
                "confidence": 0.94,
                "reason": "Read-only linked QA context question; answer in chat using linked deviations/CAPAs only.",
                "previous_action": previous_action,
            })
        if re.search(r"\bmake\b[\s\S]*\baudit[-\s]?ready\b", lower_message, re.IGNORECASE):
            return _classify_return({
                "flow": "editor_action",
                "action": "improve",
                "target_scope": "full_document" if re.search(r"\b(full|whole|entire|complete)\s+(?:sop|document|doc)\b|\b(?:this|current|open|active)\s+sop\b", lower_message, re.IGNORECASE) else ("section" if action_section_hint else _content_target_scope()),
                "section_hint": action_section_hint or None,
                "constraints": {
                    "detail_level": "audit-ready; do not add unsupported requirements"
                },
                "requires_selection": False,
                "requires_confirmation": True,
                "confidence": 0.94,
                "reason": "Active SOP plus request to make text audit-ready; run an improve action with support-only constraint.",
            })
        if re.search(r"\bgap[\s-]*check\b|lücken", lower_message, re.IGNORECASE):
            return _classify_return({
                "flow": "editor_action",
                "action": "gap_check",
                "target_scope": "full_document",
                "requires_selection": False,
                "requires_confirmation": True,
                "confidence": 0.95,
                "reason": "Active SOP plus explicit gap-check request.",
            })
        if re.search(r"\b(compliance\s+gap|compliance\s+check|what\s+(?:is|are)\s+the\s+compliance|regulatory\s+gaps?|audit\s+readiness\s+gaps?)\b", lower_message, re.IGNORECASE):
            return _classify_return({
                "flow": "editor_action",
                "action": "gap_check",
                "target_scope": "full_document" if re.search(r"\b(sop|document|full|whole|entire)\b", lower_message, re.IGNORECASE) else ("section" if action_section_hint else _content_target_scope()),
                "section_hint": action_section_hint or None,
                "requires_selection": False,
                "requires_confirmation": True,
                "confidence": 0.95,
                "reason": "Active SOP plus explicit compliance-gap request.",
            })
        if re.search(r"\bmissing\b[\s\S]*\b(roles?|acceptance criteria|records?|escalation|approval steps?)\b|\b(roles?|acceptance criteria|records?|escalation|approval steps?)\b[\s\S]*\bmissing\b", lower_message, re.IGNORECASE):
            missing_scope = "full_document" if re.search(r"\b(sop|document|full|whole|entire)\b", lower_message, re.IGNORECASE) else ("section" if action_section_hint else _content_target_scope("full_document"))
            return _classify_return({
                "flow": "editor_action",
                "action": "gap_check",
                "target_scope": missing_scope,
                "section_hint": action_section_hint if missing_scope == "section" else None,
                "requires_selection": False,
                "requires_confirmation": True,
                "confidence": 0.94,
                "reason": "Checklist of missing SOP controls should run as a gap check.",
            })
        if re.search(r"\b(summarize|summary|brief|gist|kurzfass|zusammenfass)\b", lower_message, re.IGNORECASE):
            return _classify_return({
                "flow": "chat",
                "action": None,
                "target_scope": None,
                "section_hint": action_section_hint or None,
                "requires_selection": False,
                "requires_confirmation": False,
                "confidence": 0.94,
                "reason": "Summarize is read-only in the sidebar; do not open inline editor suggestions.",
            })
        if re.search(r"\b(explain|what\s+does|tell\s+me\s+about|read\s+this\s+sop|show\s+(?:me\s+)?this\s+sop|current\s+sop\s+content)\b", lower_message, re.IGNORECASE):
            # These are read-only explanation requests. Keep them in chat so the
            # sidebar answers from live SOP/RAG context without opening an edit flow.
            pass
        elif re.search(r"\b(analy[sz]e|analysis|review\s+this\s+sop)\b", lower_message, re.IGNORECASE):
            return _classify_return({
                "flow": "editor_action",
                "action": "analyze",
                "target_scope": "full_document" if re.search(r"\b(sop|document|full|whole|entire)\b", lower_message, re.IGNORECASE) or not action_section_hint else "section",
                "section_hint": action_section_hint or None,
                "requires_selection": False,
                "requires_confirmation": True,
                "confidence": 0.9,
                "reason": "Active SOP plus explicit analyze/review request.",
            })
        if re.search(r"\b(compliance|compliant|regulatory|gmp|qa|qms|control|audit\s+ready|audit\s+readiness)\b", lower_message, re.IGNORECASE):
            return _classify_return({
                "flow": "editor_action",
                "action": "analyze",
                "target_scope": "full_document" if re.search(r"\b(sop|document|full|whole|entire)\b", lower_message, re.IGNORECASE) else ("section" if action_section_hint else _content_target_scope()),
                "section_hint": action_section_hint or None,
                "requires_selection": False,
                "requires_confirmation": True,
                "confidence": 0.9,
                "reason": "Active SOP plus explicit compliance review request.",
            })
        if re.search(r"\bimprove\b|\bverbesser", lower_message, re.IGNORECASE):
            return _classify_return({
                "flow": "editor_action",
                "action": "improve",
                "target_scope": "full_document" if re.search(r"\b(full|whole|entire|complete)\s+(?:sop|document|doc)\b|\b(?:this|current|open|active)\s+sop\b", lower_message, re.IGNORECASE) else ("section" if action_section_hint else _content_target_scope()),
                "section_hint": action_section_hint or None,
                "requires_selection": False,
                "requires_confirmation": True,
                "confidence": 0.93,
                "reason": "Active SOP plus explicit improve request.",
            })
        if re.search(r"\brewrite\b|umschreib|überarbeit", lower_message, re.IGNORECASE):
            return _classify_return({
                "flow": "editor_action",
                "action": "rewrite",
                "target_scope": "full_document" if re.search(r"\b(full|whole|entire|complete)\s+(?:sop|document|doc)\b|\b(?:this|current|open|active)\s+sop\b", lower_message, re.IGNORECASE) else ("section" if action_section_hint else _content_target_scope()),
                "section_hint": action_section_hint or None,
                "requires_selection": False,
                "requires_confirmation": True,
                "confidence": 0.93,
                "reason": "Active SOP plus explicit rewrite request.",
            })

    result = await asyncio.to_thread(
        classify_assistant_intent,
        message,
        has_active_sop=has_active_sop,
        has_editor_selection=bool(payload.get("has_editor_selection")),
        route=str(payload.get("route") or ctx.get("route") or "").strip(),
        active_sop_title=str(current_sop.get("title") or "").strip(),
        active_sop_number=str(current_sop.get("sop_number") or current_sop.get("documentId") or "").strip(),
        selected_section_summary=_selected_section_summary(),
        available_sections=_available_sections_summary(),
        previous_action_summary=_previous_action_summary(),
        recent_conversation=_recent_conversation_summary(),
        active_scope=prep.get("active_scope"),
        instruction_memory=prep.get("instruction_memory"),
        frustration_signal=prep.get("frustration_signal"),
        repetition_detected=bool(prep.get("repetition_detected")),
        repetition_instruction=prep.get("repetition_instruction"),
    )
    return _classify_return(result.model_dump())


@ai_router.get("/api/ai/llm-health")
async def llm_health(chat_probe: bool = Query(False, description="If true, POST a minimal /v1/chat/completions probe")):
    """
    Check reachability of the configured OpenAI-compatible server (/v1/models).
    """
    return await asyncio.to_thread(check_local_llm_api_health, chat_probe=chat_probe)


@ai_router.post("/api/ai/query")
async def query_ai(
    payload: dict,
    current_user: User | None = Depends(get_current_user_optional),
):
    """
    Chatbot query endpoint integrated from the standalone chatbot module.
    Each successful exchange is appended to chat_sessions / chat_messages when persistence
    succeeds (authenticated users get user_id set; anonymous users get user_id NULL).
    Response may include session_id and message_id for follow-up requests.

    Optional payload field ``assistant_mode``:
    - ``query`` (or ``query_only``): RAG/QA only; SOP mutations and action execution are disabled.
    - ``action`` (default): existing behaviour with optional SOP create/update/delete flows.
    """
    question = _visible_user_question(
        payload.get("display_question") or payload.get("user_question") or payload.get("question") or payload.get("query")
    )
    if not question:
        raise HTTPException(status_code=422, detail="question is required")

    category = payload.get("category")
    chat_history = payload.get("chat_history") or []
    surface = str(payload.get("surface") or "unknown").strip().lower()
    route = str(payload.get("route") or "").strip()
    t0 = time.perf_counter()
    cfg = get_local_llm_config()
    logger.info(
        "[chatbot-request] surface=%s route=%s provider=%s model=%s category=%s qlen=%s",
        surface,
        route,
        cfg.provider,
        cfg.model,
        category or "auto",
        len(question),
    )
    print(
        f"[chatbot-request] surface={surface} route={route or '-'} provider={cfg.provider} model={cfg.model} category={category or 'auto'} qlen={len(question)}",
        flush=True,
    )
    raw_ac = payload.get("assistant_context")
    assistant_context = raw_ac if isinstance(raw_ac, dict) else {}
    assistant_action_confirmation = payload.get("assistant_action_confirmation") or {}

    async def _merge_persisted(response: dict) -> dict:
        uid = str(current_user.id) if current_user is not None else None
        extra = await asyncio.to_thread(
            persist_chat_query_exchange,
            user_id=uid,
            client_session_id=payload.get("session_id") or payload.get("chat_session_id"),
            collection_name=str(payload.get("collection_name") or "").strip() or "docs_sops",
            category=str(category).strip() if category is not None else None,
            question=question,
            response=response,
            assistant_context=assistant_context,
            llm_provider=str(cfg.provider or ""),
            llm_model=str(cfg.model or ""),
            surface=surface,
            route=route,
        )
        if extra:
            return {**response, **extra}
        return response

    assistant_mode = _normalize_assistant_mode(payload.get("assistant_mode"))
    if assistant_mode == "query":
        logger.info("[assistant-query-mode] surface=%s route=%s", surface, route)
    else:
        logger.info("[assistant-action-mode] surface=%s route=%s", surface, route)

    if assistant_mode == "query" and _query_mode_mutation_intent(question, assistant_context):
        logger.info(
            "[assistant-query-mode] blocked_mutation_shaped_request surface=%s q_preview=%s",
            surface,
            (question[:160] or ""),
        )
        guard = {
            "answer": QUERY_MODE_REFUSAL_DE,
            "sources": [],
            "citations": [],
            "retrieval_debug": [],
            "suggestions": [],
            "retrieval_stats": {
                "total_docs": 0,
                "source": "query_mode_guard",
                "surface": surface,
                "assistant_mode": assistant_mode,
                "query_mutation_block": True,
                "latency_ms_total": round((time.perf_counter() - t0) * 1000.0, 1),
            },
            "routed_to": "query_mode_guard",
            "assistant_action": None,
            "elapsed_ms_total": round((time.perf_counter() - t0) * 1000.0, 1),
        }
        guard["retrieval_stats"]["elapsed_ms_total"] = guard["elapsed_ms_total"]
        return await _merge_persisted(guard)

    profile_context = _extract_profile_context(payload, assistant_context)
    intents = _query_intents(question)
    active_scope = _extract_active_sop_scope(assistant_context)
    context_summary = _summarize_live_context(assistant_context, question)
    explain_preview_response = _deterministic_explain_preview_response(question, assistant_context)
    if explain_preview_response:
        explain_preview_response["retrieval_stats"].update(
            {
                "provider": "deterministic",
                "model": "assistant_memory",
                "surface": surface,
                "intents": sorted(intents),
                "assistant_mode": assistant_mode,
                "latency_ms_total": round((time.perf_counter() - t0) * 1000.0, 1),
            }
        )
        explain_preview_response["elapsed_ms_total"] = round((time.perf_counter() - t0) * 1000.0, 1)
        explain_preview_response["retrieval_stats"]["elapsed_ms_total"] = explain_preview_response[
            "elapsed_ms_total"
        ]
        logger.info("[chatbot-response] source=assistant_explain_preview latency_ms=%.1f", (time.perf_counter() - t0) * 1000.0)
        return await _merge_persisted(explain_preview_response)

    last_action_response = _deterministic_last_action_response(question, assistant_context)
    if last_action_response:
        last_action_response["retrieval_stats"].update(
            {
                "provider": "deterministic",
                "model": "assistant_memory",
                "surface": surface,
                "intents": sorted(intents),
                "assistant_mode": assistant_mode,
                "latency_ms_total": round((time.perf_counter() - t0) * 1000.0, 1),
            }
        )
        last_action_response["elapsed_ms_total"] = round((time.perf_counter() - t0) * 1000.0, 1)
        last_action_response["retrieval_stats"]["elapsed_ms_total"] = last_action_response["elapsed_ms_total"]
        logger.info("[chatbot-response] source=assistant_last_action latency_ms=%.1f", (time.perf_counter() - t0) * 1000.0)
        return await _merge_persisted(last_action_response)
    if assistant_mode == "query" and "sop_count" in intents and not ({"summary", "compare", "linked"} & intents):
        response = _deterministic_sop_count_response(question, active_scope)
        response["retrieval_stats"].update(
            {
                "provider": "deterministic",
                "model": "postgresql_count",
                "surface": surface,
                "intents": sorted(intents),
                "assistant_mode": assistant_mode,
                "latency_ms_total": round((time.perf_counter() - t0) * 1000.0, 1),
            }
        )
        response["elapsed_ms_total"] = round((time.perf_counter() - t0) * 1000.0, 1)
        response["retrieval_stats"]["elapsed_ms_total"] = response["elapsed_ms_total"]
        logger.info(
            "[chatbot-response] source=deterministic_system_query routed_to=%s total=%s latency_ms=%.1f",
            response.get("routed_to", ""),
            response["citations"][0]["excerpt"] if response.get("citations") else "",
            (time.perf_counter() - t0) * 1000.0,
        )
        return await _merge_persisted(response)
    if "metadata" in intents and active_scope.get("active_sop_id") and not ({"summary", "compare", "linked", "sop_count", "sop_list"} & intents):
        response = _deterministic_active_sop_metadata_response(question, assistant_context)
        if response:
            response["retrieval_stats"].update(
                {
                    "provider": "deterministic",
                    "model": "live_editor_context",
                    "surface": surface,
                    "intents": sorted(intents),
                    "assistant_mode": assistant_mode,
                    "latency_ms_total": round((time.perf_counter() - t0) * 1000.0, 1),
                }
            )
            response["elapsed_ms_total"] = round((time.perf_counter() - t0) * 1000.0, 1)
            response["retrieval_stats"]["elapsed_ms_total"] = response["elapsed_ms_total"]
            logger.info("[chatbot-response] source=live_sop_metadata latency_ms=%.1f", (time.perf_counter() - t0) * 1000.0)
            return await _merge_persisted(response)
    action_plan = None
    if assistant_mode == "action":
        action_plan = _plan_sop_action(question, assistant_context)
        if action_plan:
            logger.info(
                "[assistant-action-detected] surface=%s type=%s",
                surface,
                action_plan.get("type"),
            )
    action_result = None
    pending_confirmation = (
        isinstance(action_plan, dict)
        and action_plan.get("type") == "delete_sop"
        and action_plan.get("requires_confirmation")
    )
    question_for_rag = question
    context_hints: list[str] = []
    rc_query = (
        assistant_context.get("response_constraints")
        if isinstance(assistant_context.get("response_constraints"), dict)
        else {}
    )
    if rc_query.get("line_count"):
        context_hints.append(
            f"OUTPUT_LINES_EXACTLY={int(rc_query['line_count'])}_NO_SUMMARY_DETAILS_STATUS_TEMPLATE"
        )
    current_sop = assistant_context.get("current_sop") if isinstance(assistant_context.get("current_sop"), dict) else {}
    active_ref = str(active_scope.get("active_sop_ref") or current_sop.get("sop_number") or current_sop.get("id") or "").strip()
    active_id = str(active_scope.get("active_sop_id") or "").strip()
    scoped_editor = bool(active_id) and not _user_requests_global_scope(question)
    logger.info(
        "[chatbot-intent] surface=%s intents=%s active_ref=%s active_id=%s scoped=%s",
        surface,
        sorted(intents),
        active_ref or "none",
        active_id or "none",
        scoped_editor,
    )
    print(
        f"[chatbot-intent] surface={surface} intents={sorted(intents)} active_ref={active_ref or 'none'} scoped={scoped_editor}",
        flush=True,
    )
    if scoped_editor:
        if active_id:
            context_hints.append(f"ACTIVE_SOP_ID={active_id}")
        if active_ref:
            context_hints.append(f"ACTIVE_SOP={active_ref}")
        context_hints.append("SCOPE=ACTIVE_SOP_ONLY")
        if not {"sop_count", "sop_list"} & intents:
            category = category or "sops"
    if ("active_sop" in intents) and active_ref:
        category = "sops"
    if "summary" in intents and active_ref:
        context_hints.append(f"ACTIVE_SOP={active_ref}")
    if "linked" in intents:
        context_hints.append("INTENT=LINKED_ENTITIES")
    if "compare" in intents:
        context_hints.append("INTENT=COMPARE_SOPS")
    if ("active_sop" in intents) and active_ref:
        context_hints.append(f"FOCUS_REF={active_ref}")
    if profile_context:
        try:
            style_hint = get_style_prompt_injection(profile_context.get("style_profile"))
            if style_hint:
                context_hints.append(style_hint[:1200])
        except Exception as exc:
            logger.warning("[chatbot-profile] profile context ignored: %s", exc)
    if context_hints:
        question_for_rag = f"{question_for_rag}\n\nRAG_HINTS: {' | '.join(context_hints)}"

    live_block = (context_summary or "").strip()
    live_ctx_chars = len(live_block)
    if live_ctx_chars > 60:
        live_limit = 7000 if re.search(
            r"\b(full|whole|entire|complete)\s+(?:sop|document|doc)\b|\b(?:summarize|summary|explain|tell\s+me\s+about)\s+(?:this\s+|the\s+|current\s+|active\s+)?sop\b",
            question,
            re.IGNORECASE,
        ) else 3600
        question_for_rag = f"{question_for_rag}\n\n{live_block[:live_limit]}"

    if action_plan:
        question_for_rag = (
            f"{question_for_rag}\n\n"
            f"PLANNED_ASSISTANT_ACTION: {action_plan}\n"
            "Use this planned action and live context while answering."
        )

    pipeline_timeout = get_chat_pipeline_timeout_seconds()
    logger.info(
        "[chatbot-timeout] pipeline_seconds=%s local_llm_seconds=%s",
        pipeline_timeout,
        get_local_llm_timeout_seconds(),
    )

    try:
        rag = await asyncio.wait_for(
            asyncio.to_thread(_get_smart_rag_chain),
            timeout=pipeline_timeout,
        )
        logger.info(
            "[chatbot-rag-call] intents=%s category=%s live_ctx_chars=%s q_preview=%s",
            sorted(intents),
            category or "auto",
            live_ctx_chars,
            (question[:200] or ""),
        )
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    rag.invoke,
                    question_for_rag,
                    category,
                    chat_history,
                    active_scope if scoped_editor else None,
                ),
                timeout=pipeline_timeout,
            )
        except Exception as first_exc:
            if _is_prompt_too_large_error(first_exc) and question_for_rag != question:
                logger.warning(
                    "[chatbot-request] prompt too large; retrying compact query path"
                )
                compact_history = (chat_history or [])[-4:]
                compact_q = (question_for_rag or question)[:1200]
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        rag.invoke,
                        compact_q,
                        category,
                        compact_history,
                        active_scope if scoped_editor else None,
                    ),
                    timeout=pipeline_timeout,
                )
            else:
                raise
        if not isinstance(result, dict):
            logger.error(
                "[chatbot-response] invalid rag result type=%s; coercing empty response",
                type(result).__name__,
            )
            result = {}
    except (TimeoutError, asyncio.TimeoutError):
        elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
        logger.error("[chatbot-response] failure_stage=pipeline_timeout elapsed_ms=%s", elapsed)
        raise HTTPException(
            status_code=504,
            detail={
                "message": "The chat pipeline exceeded its asyncio time budget (chain load + retrieval + LLM).",
                "failure_stage": "pipeline_timeout",
                "elapsed_ms": elapsed,
                "chat_pipeline_timeout_seconds": pipeline_timeout,
                "local_llm_timeout_seconds": get_local_llm_timeout_seconds(),
                "llm_base_url": cfg.base_url,
                "llm_model": cfg.model,
                "hint": "Increase CHAT_QUERY_TIMEOUT_SECONDS or reduce RAG context; confirm GET /api/ai/llm-health.",
            },
        )
    except Exception as exc:
        elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
        if is_local_llm_unreachable_error(exc):
            logger.error(
                "[chatbot-response] failure_stage=llm_unreachable elapsed_ms=%.1f error=%s",
                elapsed,
                str(exc),
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "The local OpenAI-compatible LLM could not complete a request (connection, timeout, or HTTP error).",
                    "failure_stage": "llm_unreachable",
                    "error": str(exc),
                    "elapsed_ms": elapsed,
                    "llm_base_url": cfg.base_url,
                    "llm_model": cfg.model,
                    "llm_provider": cfg.provider,
                    "local_llm_timeout_seconds": get_local_llm_timeout_seconds(),
                    "hint": "Run GET /api/ai/llm-health and ensure LOCAL_LLM_MODEL matches a model id from /v1/models.",
                },
            )
        raise HTTPException(status_code=500, detail=f"Chatbot query failed: {exc}")

    def _json_safe_citations(cits: list) -> list:
        out = []
        for c in cits or []:
            if not isinstance(c, dict):
                continue
            d = dict(c)
            s = d.get("score", 0.0)
            try:
                f = float(s)
                d["score"] = f if math.isfinite(f) else 0.0
            except (TypeError, ValueError):
                d["score"] = 0.0
            out.append(d)
        return out

    citations = _json_safe_citations((result or {}).get("citations", []))
    try:
        from chatbot.rag.rag_chain import _dedupe_citations_by_ref

        citations = _dedupe_citations_by_ref(citations)
    except Exception:
        pass
    sources = []
    for idx, c in enumerate(citations):
        ref = c.get("ref") or c.get("title") or f"source-{idx+1}"
        label = c.get("title") or c.get("ref") or "Source"
        source_type = (c.get("type") or "doc").lower()
        sources.append({"id": ref, "type": source_type, "label": label})

    base_stats = dict((result or {}).get("retrieval_stats") or {})
    response = {
        "answer": (result or {}).get("answer", ""),
        "sources": sources,
        "citations": citations,
        "retrieval_debug": (result or {}).get("retrieval_debug", []),
        "suggestions": (result or {}).get("suggestions", []),
        "retrieval_stats": {**base_stats},
        "routed_to": (result or {}).get("routed_to", ""),
        "assistant_action": action_result or action_plan,
    }
    if (result or {}).get("failure_stage") is not None:
        response["failure_stage"] = (result or {}).get("failure_stage")
    if (result or {}).get("llm_error"):
        response["llm_error"] = (result or {}).get("llm_error")

    if pending_confirmation and not bool(assistant_action_confirmation.get("confirmed")):
        response["assistant_action"] = {
            **(action_plan or {}),
            "requires_confirmation": True,
        }
        response["answer"] = (
            "I can delete the active SOP, but I need your confirmation first. "
            "Please confirm deletion in the in-app modal."
        )
        return await _merge_persisted(response)

    if action_plan and action_plan.get("type") == "delete_sop" and bool(assistant_action_confirmation.get("confirmed")):
        logger.info("[assistant-action-execute] type=delete_sop surface=%s", surface)
        action_result = await asyncio.to_thread(
            _execute_sop_action, "delete_sop", question, assistant_context, "", "replace"
        )
        response["assistant_action"] = action_result
        if action_result and action_result.get("ok"):
            response["answer"] = (
                f"{response.get('answer', '')}\n\n"
                f"Action completed: {action_result.get('message', 'SOP deleted.')}"
            ).strip()
        return await _merge_persisted(response)

    if action_plan and action_plan.get("type") in {"create_sop", "update_sop"}:
        llm_generated = result.get("answer", "")
        mode = action_plan.get("mode", "replace")
        logger.info(
            "[assistant-action-execute] type=%s surface=%s mode=%s",
            action_plan.get("type"),
            surface,
            mode,
        )
        action_result = await asyncio.to_thread(
            _execute_sop_action,
            action_plan["type"],
            question,
            assistant_context,
            llm_generated,
            mode,
        )
        response["assistant_action"] = action_result
        if action_result and action_result.get("ok"):
            response["answer"] = (
                f"{response.get('answer', '')}\n\n"
                f"Action completed: {action_result.get('message', 'Done.')}"
            ).strip()

    response.setdefault("retrieval_stats", {})
    retrieval_total = int(response["retrieval_stats"].get("total_docs") or 0)
    had_evidence = retrieval_total > 0 or len(citations) > 0
    answer_raw = (response.get("answer") or "").strip()
    answer_lower = answer_raw.lower()
    boilerplate_unreachable = (
        "no relevant information found" in answer_lower
        or "the available records do not contain sufficient detail" in answer_lower
    )
    dbg_rows = response.get("retrieval_debug") or []
    dbg_preview = ""
    if dbg_rows:
        dbg_preview = str(dbg_rows[0])[:480]

    if not had_evidence and (not answer_raw or boilerplate_unreachable):
        from chatbot.rag.rag_chain import RAG_NO_CONTEXT_REFUSAL

        response["answer"] = RAG_NO_CONTEXT_REFUSAL
        response["citations"] = []
        response["sources"] = []
        response["retrieval_debug"] = []
        rr = (
            (result or {}).get("refusal_reason")
            or (result or {}).get("retrieval_stats", {}).get("refusal_reason")
            or "no_retrieval"
        )
        response["refusal_reason"] = rr
        logger.info(
            "[chatbot-refusal] reason=%s intents=%s routed_to=%s q_preview=%s",
            rr,
            sorted(intents),
            response.get("routed_to", ""),
            (question[:220] or ""),
        )
    elif had_evidence and boilerplate_unreachable:
        logger.warning(
            "[chatbot-refusal-skipped] refusal-like phrasing but total_docs=%s citations=%s",
            retrieval_total,
            len(citations),
        )

    try:
        from chatbot.rag.rag_chain import sanitize_user_facing_answer

        if not bool((response.get("retrieval_stats") or {}).get("strict_mode")):
            response["answer"] = sanitize_user_facing_answer(response.get("answer") or "")
    except Exception:
        pass

    response["retrieval_stats"].update(
        {
            "provider": cfg.provider,
            "model": cfg.model,
            "source": "rag",
            "surface": surface,
            "intents": sorted(intents),
            "profile_detection": bool(profile_context),
            "latency_ms_total": round((time.perf_counter() - t0) * 1000.0, 1),
            "llm_base_url": cfg.base_url,
            "assistant_mode": assistant_mode,
        }
    )
    response["elapsed_ms_total"] = round((time.perf_counter() - t0) * 1000.0, 1)
    response["retrieval_stats"]["elapsed_ms_total"] = response["elapsed_ms_total"]
    logger.info(
        "[chatbot-response] source=rag routed_to=%s total_docs=%s citations=%s intents=%s dbg_preview=%s latency_ms=%.1f",
        response.get("routed_to", ""),
        retrieval_total,
        len(citations),
        sorted(intents),
        dbg_preview.replace("\n", " ")[:500],
        (time.perf_counter() - t0) * 1000.0,
    )
    print(
        f"[chatbot-response] source=rag routed_to={response.get('routed_to', '')} total_docs={retrieval_total} citations={len(citations)} latency_ms={(time.perf_counter() - t0) * 1000.0:.1f}",
        flush=True,
    )
    return await _merge_persisted(response)
