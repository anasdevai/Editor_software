from html import escape
import re
import os
import math
import time
import logging
import threading
import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import or_
from langchain_core.messages import AIMessage

from action.prompts import (
    IMPROVE_REWRITE_NO_RAG_CONTEXT,
    build_gap_check_prompt,
    build_improve_prompt,
    build_rewrite_prompt,
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
from pydantic import ValidationError

from app.schemas import AIActionRequest, AIActionResponse
from app.database import SessionLocal
from app.models import SOP, SOPVersion, Deviation, Capa, AuditFinding, Decision
try:
    from openai import BadRequestError
except Exception:  # pragma: no cover - import fallback for older envs
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
                f"[startup] Chatbot reranker cache missing, continuing without reranker: {reranker_exc}",
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


def _split_sentences(text: str) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


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
    # Conservative char/token for unknown tokenization.
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
    # Conservative estimate for unknown local tokenization.
    prompt_est_tokens = max(1, int(prompt_chars / 4.0))
    ctx = n_ctx or _action_model_context_tokens()
    reserve = int(os.getenv("ACTION_CONTEXT_RESPONSE_RESERVE", "256"))
    safe_by_ctx = max(128, ctx - prompt_est_tokens - reserve)
    safe_cap = int(os.getenv("ACTION_SAFE_MAX_TOKENS_CAP", "32768"))
    return max(128, min(int(base_tokens), safe_by_ctx, safe_cap))


def _context_error_http_exception(err_txt: str) -> HTTPException:
    cfg = get_local_llm_config()
    n_ctx = _extract_n_ctx_from_error(err_txt)
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
            "n_ctx": n_ctx,
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
    return s[-1].isalnum() and not s.endswith((".", "!", "?", "}", "]", '"'))


def _call_action_llm(runtime: Any, prompt: str, *, input_char_budget: int = 0, action: str = "unknown") -> str:
    base_n = _action_output_token_budget(input_char_budget) if input_char_budget else int(
        os.getenv("ACTION_LLM_MAX_TOKENS") or os.getenv("ACTION_MAX_OUTPUT_TOKENS") or "4096"
    )
    soft = _action_prompt_soft_limit_chars()

    cfg = get_local_llm_config()
    budgets: list[int] = []
    for b in (soft, int(soft * 0.75), int(soft * 0.5)):
        if b > 0 and b not in budgets:
            budgets.append(b)

    def _invoke_once(p: str, max_tokens: int) -> tuple[str, dict[str, Any]]:
        msg = runtime.llm.bind(max_tokens=max_tokens).invoke(p)
        return _extract_text_and_meta(msg)

    def _invoke_fallback_once(p: str, max_tokens: int) -> tuple[str, dict[str, Any]]:
        msg = runtime.fallback_llm.bind(max_tokens=max_tokens).invoke(p)
        return _extract_text_and_meta(msg)

    last_context_error: str | None = None
    n_ctx_hint: int | None = None
    out = ""
    used_budget = len(prompt)
    used_tokens = base_n
    last_meta: dict[str, Any] = {}
    length_limited_seen = False

    for budget in budgets:
        work = truncate_prompt_for_llm(prompt, budget) if len(prompt) > budget else prompt
        used_budget = len(work)
        used_tokens = _safe_action_max_tokens(base_n, used_budget, n_ctx=n_ctx_hint)
        try:
            out, last_meta = _invoke_once(work, used_tokens)
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
                length_limited_seen = True
                raise ValueError("llm_finish_reason_length")
            if (out or "").strip():
                break
            out, last_meta = _invoke_once(work + ACTION_LLM_EMPTY_RETRY_SUFFIX, used_tokens)
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
            try:
                out, last_meta = _invoke_fallback_once(work, used_tokens)
                if (out or "").strip():
                    break
            except BadRequestError as fb_exc:
                fb_txt = str(fb_exc)
                if _is_context_length_error_text(fb_txt):
                    last_context_error = fb_txt
                    n_ctx_hint = _extract_n_ctx_from_error(fb_txt) or n_ctx_hint
                    logger.warning(
                        "[ai-action-context-error] prompt_chars=%s max_tokens=%s model=%s base_url=%s n_ctx=%s error=%s",
                        len(work),
                        used_tokens,
                        cfg.model,
                        cfg.base_url,
                        n_ctx_hint,
                        fb_txt,
                    )
                    continue
                raise
        except Exception:
            out, last_meta = _invoke_fallback_once(work, used_tokens)
            finish_reason = str(last_meta.get("finish_reason") or "").lower()
            if finish_reason == "length":
                length_limited_seen = True
                continue
            if (out or "").strip():
                break

    if last_context_error and not (out or "").strip():
        raise _context_error_http_exception(last_context_error)

    finish_reason = str(last_meta.get("finish_reason") or "").lower()
    if finish_reason == "length" or length_limited_seen or _response_looks_cut(out):
        logger.warning(
            "[ai-action-llm-truncated] prompt_chars=%s output_chars=%s max_tokens=%s model=%s base_url=%s finish_reason=%s",
            used_budget,
            len(out or ""),
            used_tokens,
            cfg.model,
            cfg.base_url,
            finish_reason or "unknown",
        )
        raise HTTPException(
            status_code=422,
            detail={
                "message": "AI response was truncated due to model/output limit.",
                "validation_or_parse_error": f"finish_reason={finish_reason or 'unknown'}",
                "hint": "Increase ACTION_LLM_MAX_TOKENS, shorten selection, or increase ACTION_MODEL_CONTEXT_TOKENS.",
            },
        )

    preview = (out or "").replace("\n", "\\n").replace("\r", "\\r")[:900]
    logger.info(
        "[ai-action-llm-raw] raw_len=%s max_tokens=%s preview=%s",
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
    """
    Gap Check sometimes returns markdown-ish formatting (###, **, ---).
    Normalize the raw analysis so we can render clean HTML.
    """
    t = text or ""
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Remove common markdown tokens from headings/bold.
    t = re.sub(r"(?m)^\s*#+\s*", "", t)  # e.g. "### Heading" -> "Heading"
    t = t.replace("**", "")
    t = re.sub(r"(?m)^\s*---+\s*$", "", t)

    # Collapse excessive blank lines.
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _render_gap_check_analysis_html(analysis: str) -> str:
    """
    Render Gap Check analysis into clean, consistently structured HTML.
    """
    normalized = _normalize_gap_check_analysis_text(analysis)
    if not normalized:
        return "<p>No suggestion returned.</p>"

    canonical_headings = [
        "Summary",
        "Identified Gaps",
        "Risk/Impact",
        "Recommended Fixes",
        "Suggested SOP Text",
    ]

    heading_re = re.compile(
        r"^\s*(Summary|Identified Gaps|Risk/Impact|Recommended Fixes|Suggested SOP Text)\s*:?\s*$",
        re.IGNORECASE,
    )

    lines = [ln.rstrip() for ln in normalized.split("\n")]
    sections: dict[str, list[str]] = {}
    order: list[str] = []
    current: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current:
                sections.setdefault(current, []).append("")
            continue

        m = heading_re.match(line)
        if m:
            matched = m.group(1)
            # Preserve canonical heading casing.
            current = next((h for h in canonical_headings if h.lower() == matched.lower()), matched)
            if current not in sections:
                sections[current] = []
                order.append(current)
            continue

        if not current:
            current = "Summary"
            if current not in sections:
                sections[current] = []
                order.append(current)

        sections.setdefault(current, []).append(line)

    def render_paragraphs(block_lines: list[str]) -> str:
        paragraphs: list[list[str]] = []
        buf: list[str] = []
        for ln in block_lines:
            if ln == "":
                if buf:
                    paragraphs.append(buf)
                    buf = []
                continue
            buf.append(ln)
        if buf:
            paragraphs.append(buf)

        if not paragraphs:
            return ""

        rendered_paras = []
        for para in paragraphs:
            text = "\n".join(para)
            rendered_paras.append(f"<p>{escape(text).replace(chr(10), '<br />')}</p>")
        return "".join(rendered_paras)

    num_re = re.compile(r"^\s*(\d+)[\)\.]\s+(.*)$")
    bullet_re = re.compile(r"^\s*([-*•])\s+(.*)$")

    def render_body(block_lines: list[str]) -> str:
        html_parts: list[str] = []
        i = 0

        while i < len(block_lines):
            ln = block_lines[i]
            if ln == "":
                i += 1
                continue

            m_num = num_re.match(ln)
            m_bul = bullet_re.match(ln)
            if m_num or m_bul:
                is_ordered = bool(m_num)
                tag = "ol" if is_ordered else "ul"
                items: list[list[str]] = []
                current_item: list[str] = []

                def flush_item():
                    nonlocal current_item
                    if current_item:
                        items.append(current_item)
                        current_item = []

                while i < len(block_lines):
                    cur = block_lines[i]
                    if cur == "":
                        flush_item()
                        i += 1
                        break

                    m2_num = num_re.match(cur)
                    m2_bul = bullet_re.match(cur)

                    if is_ordered and m2_num:
                        flush_item()
                        current_item = [m2_num.group(2).strip()]
                        i += 1
                        continue
                    if (not is_ordered) and m2_bul:
                        flush_item()
                        current_item = [m2_bul.group(2).strip()]
                        i += 1
                        continue

                    # If the line looks like a *different* list type, stop the current list.
                    if is_ordered and m2_bul:
                        break
                    if (not is_ordered) and m2_num:
                        break

                    # Otherwise treat as continuation.
                    if current_item:
                        current_item.append(cur.strip())
                    i += 1

                flush_item()

                li_html = []
                for item_lines in items:
                    if not any(x.strip() for x in item_lines):
                        continue
                    text = "\n".join(item_lines)
                    li_html.append(f"<li>{escape(text).replace(chr(10), '<br />')}</li>")
                html_parts.append(f"<{tag}>" + "".join(li_html) + f"</{tag}>")
                continue

            # Paragraph lines until next blank line.
            para_lines: list[str] = []
            while i < len(block_lines) and block_lines[i] != "":
                para_lines.append(block_lines[i])
                i += 1
            html_parts.append(
                f"<p>{escape('\n'.join(para_lines)).replace(chr(10), '<br />')}</p>"
            )

        return "".join(html_parts) if html_parts else render_paragraphs(block_lines)

    rendered: list[str] = []
    for heading in canonical_headings:
        if heading not in sections:
            continue
        block = sections[heading]
        if not any((ln or "").strip() for ln in block):
            continue

        rendered.append(f"<h3>{escape(heading)}</h3>")
        rendered.append(render_body(block))

    if not rendered:
        return render_paragraphs(lines)

    return "".join(rendered)


@ai_router.post("/api/ai/action", response_model=AIActionResponse)
async def perform_ai_action(payload: AIActionRequest):
    """
    Canonical implementation is ``app.ai_routes.perform_ai_action`` (mounted router).
    Prompt builders live in ``chatbot/actions/prompts.py`` — this shim keeps standalone
    chatbot imports from diverging.
    """
    from app.ai_routes import perform_ai_action as _app_perform_ai_action

    return await _app_perform_ai_action(payload)


@ai_router.post("/api/ai/classify-intent")
async def classify_intent(payload: dict):
    from chatbot.assistant.intent_classifier import classify_assistant_intent

    message = (payload.get("message") or payload.get("question") or "").strip()
    if not message:
        raise HTTPException(status_code=422, detail="message is required")

    ctx = payload.get("assistant_context") if isinstance(payload.get("assistant_context"), dict) else {}
    current_sop = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}
    has_active_sop = bool(payload.get("has_active_sop")) or bool(
        str(ctx.get("active_sop_id") or ctx.get("current_document_id") or current_sop.get("id") or "").strip()
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
async def llm_health(chat_probe: bool = Query(False)):
    return await asyncio.to_thread(check_local_llm_api_health, chat_probe=chat_probe)


@ai_router.post("/api/ai/query")
async def query_ai(payload: dict):
    """
    Chatbot query endpoint integrated from the standalone chatbot module.
    """
    question = (payload.get("question") or payload.get("query") or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="question is required")

    category = payload.get("category")
    chat_history = payload.get("chat_history") or []
    assistant_context = payload.get("assistant_context") or {}
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
    q = question.lower()
    intents: set[str] = set()
    if re.search(r"\b(this sop|current sop|active sop)\b", q) or re.search(
        r"\b(which|what)\b.*\b(sop)\b.*\b(currently open|open now|opened|active)\b", q
    ):
        intents.add("active_sop")
    if re.search(r"\b(linked|related)\b.*\b(capa|capas|audit|audits|deviation|deviations)\b", q):
        intents.add("linked_entities")
    current = assistant_context.get("current_sop") if isinstance(assistant_context.get("current_sop"), dict) else {}
    active_ref = str(current.get("sop_number") or current.get("id") or "").strip()
    logger.info("[chatbot-intent] surface=%s intents=%s active_ref=%s", surface, sorted(intents), active_ref or "none")

    question_for_rag = question
    context_hints: list[str] = []
    if "active_sop" in intents and active_ref:
        if not category:
            category = "sops"
        context_hints.append(f"ACTIVE_SOP={active_ref}")
        context_hints.append(f"FOCUS_REF={active_ref}")
    if "linked_entities" in intents:
        context_hints.append("INTENT=LINKED_ENTITIES")

    try:
        from app import ai_routes as _app_ai_routes

        context_summary = _app_ai_routes._summarize_live_context(assistant_context, question)
    except Exception:
        context_summary = ""

    if context_hints:
        question_for_rag = f"{question_for_rag}\n\nRAG_HINTS: {' | '.join(context_hints)}"

    live_block = (context_summary or "").strip()
    live_ctx_chars = len(live_block)
    if live_ctx_chars > 60:
        question_for_rag = f"{question_for_rag}\n\n{live_block[:3200]}"

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
                ),
                timeout=pipeline_timeout,
            )
        except Exception as first_exc:
            if _is_prompt_too_large_error(first_exc) and question_for_rag != question:
                logger.warning(
                    "[chatbot-request] prompt too large; retrying compact query path"
                )
                compact_history = (chat_history or [])[-4:]
                compact_question = (question_for_rag or question)[:1200]
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        rag.invoke,
                        compact_question,
                        category,
                        compact_history,
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
        raise HTTPException(
            status_code=500,
            detail=f"Chatbot query failed: {exc}",
        )

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
    }
    if (result or {}).get("failure_stage") is not None:
        response["failure_stage"] = (result or {}).get("failure_stage")
    if (result or {}).get("llm_error"):
        response["llm_error"] = (result or {}).get("llm_error")

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
    dbg_preview = str(dbg_rows[0])[:480] if dbg_rows else ""

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
            "latency_ms_total": round((time.perf_counter() - t0) * 1000.0, 1),
            "llm_base_url": cfg.base_url,
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
    return response
