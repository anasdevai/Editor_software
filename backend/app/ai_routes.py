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
    from chatbot.actions.prompts import resolve_edit_scope

    if resolve_edit_scope(request) != "full_document":
        return False
    # Actions tab / API clients that set edit_scope explicitly expect the canonical
    # build_rewrite_prompt FULL_DOCUMENT path (preserves TEXT order from title onward).
    explicit = getattr(request, "edit_scope", None)
    if explicit == "full_document":
        return False
    text = request.section_text or ""
    section_type = (request.section_type or "").strip().lower()
    if section_type == "full document" and len(text) >= 1800:
        return True
    record_ids = re.findall(r"\b(?:DEV|CAPA|AUD|DEC)-[A-Z]+-\d+\b", text)
    section_headers = re.findall(r"(?m)^\s*(?:#{1,6}\s*)?(?:\d+[.)]\s+|##\s*\*\*)", text)
    return len(text) >= 2500 and (len(record_ids) >= 3 or len(section_headers) >= 4)


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
    if sop_id:
        db = SessionLocal()
        try:
            detected_row = db.query(SOPDetectedParameters).filter(
                SOPDetectedParameters.sop_id == uuid.UUID(str(sop_id))
            ).first()
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
                        profile_json = profile.active_profile_json
                        profile_md = profile.active_profile_md
            
            # Fallback to load tenant profile if no profile was loaded yet
            if not profile_json:
                sop_row = db.query(SOP).filter(SOP.id == uuid.UUID(str(sop_id))).first()
                if sop_row and sop_row.tenant_id:
                    profile = db.query(ClientProfile).filter(
                        ClientProfile.tenant_id == sop_row.tenant_id
                    ).first()
                    if profile:
                        profile_json = profile.active_profile_json
                        profile_md = profile.active_profile_md
        except Exception as e:
            logger.warning("[ai-routes] Failed to load NLP / ClientProfile context: %s", e)
        finally:
            db.close()
            
    return detected_nlp, profile_json, profile_md


def _run_dynamic_ai_action(payload: AIActionRequest, action: str) -> AIActionResponse:
    request = _build_action_request(payload)
    sop_ctx = _load_uploaded_sop_context(request)
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
    style_profile = _resolve_style_profile(sop_ctx, style_source_text)
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
        raw_docs = runtime.retriever.invoke(retrieval_query)
        reranked = runtime.reranker.rerank_top_n(retrieval_query, raw_docs, 3)
        context = f"{format_chunks(reranked)}\n\n{style_block}\n\n{sop_context_block}".strip()
        print(
            f"[nlp-action] action=gap_check retrieval_docs={len(raw_docs)} reranked_docs={len(reranked)}",
            flush=True,
        )
    else:
        context = _build_improve_rewrite_context(request, style_block, sop_context_block)

    logger.info(
        "[ai-action-prompt] action=%s prompt_type=%s_json_nlp_v1 provider=%s model=%s nlp_block_chars=%s",
        action,
        action,
        cfg.provider,
        cfg.model,
        len(nlp_block or ""),
    )

    detected_nlp, profile_json, profile_md = _get_nlp_and_profile_context(sop_ctx)

    if action == "improve":
        prompt = build_improve_prompt(
            request,
            context,
            nlp_block,
            profile_md=profile_md or "",
            profile_json=profile_json,
            detected_nlp=detected_nlp,
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
            },
        )

    if action == "summarize":
        prompt = build_summarize_prompt(request, context, nlp_block)
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
            },
        )

    if action == "analyze":
        prompt = build_analyze_prompt(request, context, nlp_block)
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
                },
            )
        prompt = build_rewrite_prompt(
            request,
            context,
            nlp_block,
            profile_md=profile_md or "",
            profile_json=profile_json,
            detected_nlp=detected_nlp,
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
    )
    raw = _call_action_llm(runtime, prompt, input_char_budget=ch_budget, action="rewrite")
    parsed = parse_with_retry(
        raw=raw,
        schema=RewriteResponse,
        prompt=prompt,
        call_llm=lambda rp: _call_action_llm(runtime, rp, input_char_budget=ch_budget, action="rewrite"),
        audit_log=[],
    )
    return AIActionResponse(
        action="rewrite",
        original_text=_clean_text(payload.text),
        suggested_text=_render_dynamic_text(parsed.rewritten_text),
        explanation="Text neu formuliert / Text rewritten.",
        structured_data={
            "rewritten_text": parsed.rewritten_text,
            "style_profile": style_profile,
            "nlp_action_summary": nlp_summary,
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
    if re.search(r"\b(summarize|summary|brief|gist)\b", q):
        intents.add("summary")
    if re.search(r"\b(compare|difference|vs|versus)\b", q):
        intents.add("compare")
    if re.search(r"\b(linked|related)\b.*\b(capa|capas|audit|audits|decision|decisions|deviation|deviations)\b", q):
        intents.add("linked")
    if re.search(r"\b(this sop|current sop|active sop)\b", q):
        intents.add("active_sop")
    if re.search(r"\b(which|what)\b.*\b(sop)\b.*\b(currently open|open now|opened|active)\b", q):
        intents.add("active_sop")
    return intents


def _summarize_live_context(assistant_context: dict | None, question: str = "") -> str:
    ctx = assistant_context or {}
    current = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}
    linked = ctx.get("linked_context") if isinstance(ctx.get("linked_context"), dict) else {}
    tabs = _ctx_list(ctx.get("opened_tabs"))
    text = str(ctx.get("editor_excerpt") or "").strip()
    references = _ctx_list(current.get("references"))
    intents = _query_intents(question)
    scope = _extract_active_sop_scope(assistant_context)
    active_sop_ref = str(scope.get("active_sop_ref") or current.get("sop_number") or current.get("id") or "").strip()
    active_sop_id = str(scope.get("active_sop_id") or "").strip()
    open_sop_tabs = _extract_refs(tabs, ["docId", "label"], limit=10)
    include_editor_excerpt = bool(text) and bool({"summary", "active_sop", "compare"} & intents)
    excerpt = text[:1200] if include_editor_excerpt else ""
    focus_note = ""
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
            return minimal + f"{focus_note}- Answer only what was asked; avoid unsolicited summaries."

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
            + f"- Linked deviations: {len(scope.get('linked_deviation_ids') or [])} ({', '.join(linked_devs) or 'none'})\n"
            + f"- Linked CAPAs: {len(scope.get('linked_capa_ids') or [])} ({', '.join(linked_capas) or 'none'})\n"
            + f"- Linked audits: {len(scope.get('linked_audit_ids') or [])} ({', '.join(linked_audits) or 'none'})\n"
            + f"- Linked decisions: {len(scope.get('linked_decision_ids') or [])} ({', '.join(linked_decisions) or 'none'})\n"
            + f"- Related SOPs: {len(scope.get('linked_sop_ids') or [])} ({', '.join(related_sops) or 'none'})\n"
            + f"- Open tabs: {len(tabs)}\n"
            + f"{focus_note}"
            + f"- References in editor metadata: {', '.join(str(r) for r in references[:10]) or 'none'}\n"
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
        f"- Active SOP: {current.get('sop_number') or current.get('id') or 'unknown'} | "
        f"title={current.get('title') or 'unknown'} | version={current.get('version') or 'unknown'} | "
        f"status={current.get('status') or 'unknown'}\n"
        f"- Linked deviations: {len(_ctx_list(linked.get('deviations')))} ({', '.join(linked_devs) or 'none'})\n"
        f"- Linked CAPAs: {len(_ctx_list(linked.get('capas')))} ({', '.join(linked_capas) or 'none'})\n"
        f"- Linked audits: {len(_ctx_list(linked.get('audits')))} ({', '.join(linked_audits) or 'none'})\n"
        f"- Linked decisions: {len(_ctx_list(linked.get('decisions')))} ({', '.join(linked_decisions) or 'none'})\n"
        f"- Related SOPs: {len(_ctx_list(linked.get('related_sops')))} ({', '.join(related_sops) or 'none'})\n"
        f"- Open tabs: {len(tabs)}\n"
        f"{focus_note}"
        f"- References in editor metadata: {', '.join(str(r) for r in references[:10]) or 'none'}\n"
        f"- Editor text excerpt: {excerpt if excerpt else 'not injected for this query intent'}"
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


@ai_router.post("/api/ai/classify-intent")
async def classify_intent(payload: dict):
    """
    Semantic intent routing for the unified KL/KI Assistant chat panel.
    Returns flow (chat | editor_action | clarify), action, target scope, and constraints.
    """
    from chatbot.assistant.intent_classifier import classify_assistant_intent

    message = (payload.get("message") or payload.get("question") or "").strip()
    if not message:
        raise HTTPException(status_code=422, detail="message is required")

    ctx = payload.get("assistant_context") if isinstance(payload.get("assistant_context"), dict) else {}
    current_sop = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}

    has_active_sop = bool(payload.get("has_active_sop"))
    if not has_active_sop:
        has_active_sop = bool(
            str(ctx.get("active_sop_id") or ctx.get("current_document_id") or "").strip()
            or str(current_sop.get("id") or "").strip()
        )

    result = await asyncio.to_thread(
        classify_assistant_intent,
        message,
        has_active_sop=has_active_sop,
        has_editor_selection=bool(payload.get("has_editor_selection")),
        route=str(payload.get("route") or ctx.get("route") or "").strip(),
        active_sop_title=str(current_sop.get("title") or "").strip(),
        active_sop_number=str(current_sop.get("sop_number") or current_sop.get("documentId") or "").strip(),
    )
    return result.model_dump()


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
        question_for_rag = f"{question_for_rag}\n\n{live_block[:3200]}"

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
