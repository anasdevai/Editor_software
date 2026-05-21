"""
chain/rag_chain.py

Two chain classes:
  - HybridRAGChain     : original single-collection chain (backward compat.)
  - SmartRAGChain      : routes query to relevant collections only, returns
                         clean prose answer + citations + dynamic suggestions.
"""

import time
import re
import json
import math
import logging
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.context_builder import build_context
from retrieval.federated_retriever import FederatedRetriever
from retrieval.hybrid_retriever import rag_unified_enabled
from retrieval.query_router import route_query, describe_route
from retrieval.llm_router import LLMRouter
from chatbot.llm.provider import (
    classify_llm_exception,
    create_chat_llm,
    get_local_llm_config,
)
import os
from dotenv import load_dotenv
from app.database import SessionLocal
from app.models import SOP, Deviation, Capa, AuditFinding, Decision

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


MAX_QUERY_CHARS = int(os.getenv("RAG_MAX_QUERY_CHARS", "3000"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "8000"))
MAX_HISTORY_MESSAGE_CHARS = int(os.getenv("RAG_MAX_HISTORY_MESSAGE_CHARS", "600"))
MAX_HISTORY_MESSAGES = int(os.getenv("RAG_MAX_HISTORY_MESSAGES", "6"))
RAG_DEBUG_RETRIEVAL = os.getenv("RAG_DEBUG_RETRIEVAL", "false").strip().lower() == "true"
RAG_DEBUG_MAX_CHUNKS = int(os.getenv("RAG_DEBUG_MAX_CHUNKS", "8"))
logger = logging.getLogger(__name__)
RAG_STRICT_INVENTORY_MODE = os.getenv("RAG_STRICT_INVENTORY_MODE", "false").strip().lower() == "true"
# SOP inventory count/list bypasses RAG when false; default true = retrieval-only answers.
RAG_DISABLE_SOP_INVENTORY = os.getenv("RAG_DISABLE_SOP_INVENTORY", "true").strip().lower() == "true"

RAG_NO_CONTEXT_REFUSAL = (
    "No relevant information found in the current documents/context."
)
RAG_OUT_OF_SCOPE_REFUSAL = (
    "I can only answer questions related to uploaded documents, SOPs, or retrieved system context."
)


def _console_safe(value) -> str:
    """Keep debug console output from failing on Windows non-UTF-8 code pages."""
    return str(value).encode("ascii", errors="replace").decode("ascii")


def _safe_print(message: str) -> None:
    print(_console_safe(message), flush=True)


def _json_safe_float(v, default: float = 0.0) -> float:
    """Finite floats only; JSON cannot encode inf, -inf, or nan."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    return x


def _sanitize_citation_list(cits: List[dict]) -> List[dict]:
    out: List[dict] = []
    for c in cits or []:
        if not isinstance(c, dict):
            continue
        d = dict(c)
        d["score"] = round(_json_safe_float(d.get("score", 0.0)), 4)
        out.append(d)
    return out


def _db_active_sop_count() -> int:
    db = SessionLocal()
    try:
        return int(db.query(SOP).filter(SOP.is_active == True).count())  # noqa: E712
    finally:
        db.close()


def _user_query_expects_sop_context(query: str) -> bool:
    """True when the question is primarily about SOPs / procedures (not generic chat)."""
    q = (query or "").lower()
    if not q.strip():
        return False
    if re.search(r"\b(sop|sops|standard operating procedures?|work instruction|wi-|procedure document)\b", q):
        return True
    if re.search(r"\b(procedure|procedures)\b", q) and re.search(
        r"\b(compliance|qms|quality|controlled document|document control)\b", q
    ):
        return True
    return False


def _db_active_sop_rows(limit: int) -> List[Tuple[str, str, str]]:
    """(sop_number, title, status_label) for active SOPs from PostgreSQL."""
    db = SessionLocal()
    try:
        rows_db: List[Tuple[str, str, str]] = []
        for sop in (
            db.query(SOP)
            .filter(SOP.is_active == True)  # noqa: E712
            .order_by(SOP.sop_number)
            .limit(max(1, limit))
            .all()
        ):
            rows_db.append((str(sop.sop_number or ""), str(sop.title or "Untitled"), "active"))
        return rows_db
    finally:
        db.close()


def _extract_active_sop_from_prompt(prompt: str) -> tuple[str, str]:
    """Parse LIVE_ASSISTANT_CONTEXT / ACTIVE_SOP hints injected by the API."""
    raw = str(prompt or "")
    m = re.search(
        r"Active SOP:\s*([^\n|]+?)\s*\|\s*title=([^\n|]+)",
        raw,
        re.IGNORECASE,
    )
    if m:
        ref = m.group(1).strip()
        title = m.group(2).strip()
        if ref.lower() not in ("unknown", "none", ""):
            return ref, title if title.lower() != "unknown" else ""
    m = re.search(r"ACTIVE_SOP=([^\s\n|]+)", raw, re.IGNORECASE)
    if m:
        return m.group(1).strip(), ""
    return "", ""


def _extract_active_sop_id_from_prompt(prompt: str) -> str:
    raw = str(prompt or "")
    m = re.search(r"ACTIVE_SOP_ID=([^\s\n|]+)", raw, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"\|\s*id=([0-9a-fA-F-]{36})\s*\|", raw)
    if m:
        return m.group(1).strip()
    return ""


def _prompt_requests_active_sop_scope(prompt: str) -> bool:
    return bool(re.search(r"\bSCOPE=ACTIVE_SOP_ONLY\b", str(prompt or ""), re.IGNORECASE))


def _strip_sources_footer_from_answer(text: str) -> str:
    """Remove trailing 'Sources:' / '📎 Sources:' blocks so UI + citations are single source."""
    s = (text or "").strip()
    if not s:
        return ""
    pattern = re.compile(
        r"(?:\n\n|\n|^)(?:📎\s*)?sources?\s*:.*$",
        re.IGNORECASE | re.DOTALL,
    )
    return pattern.sub("", s).rstrip()


def _strip_sources_lines_from_answer(text: str) -> str:
    """Drop standalone 'Sources:' / '📎 Sources:' lines (LLM duplicates app citation UI)."""
    if not (text or "").strip():
        return ""
    out: List[str] = []
    for line in (text or "").splitlines():
        if re.match(r"^\s*(?:📎\s*)?sources?\s*:\s*", line, re.IGNORECASE):
            continue
        out.append(line)
    return "\n".join(out).rstrip()


# Internal markers that must never appear in user-facing assistant text.
_INTERNAL_ANSWER_LINE_PREFIXES = (
    "- active sop:",
    "- retrieval scope:",
    "- linked deviations:",
    "- linked capas:",
    "- linked audits:",
    "- linked decisions:",
    "- related sops:",
    "- open tabs:",
    "- references in editor",
    "- editor text excerpt:",
    "- focus sop for",
    "- compare candidates",
    "- answer only what was asked",
    "live_assistant_context",
    "rag_hints:",
    "planned_assistant_action:",
)


def sanitize_user_facing_answer(text: str) -> str:
    """
    Strip retrieval/system annotations from the final answer shown in chat UIs.
    Citations like [SOP-IT-001] are kept; LIVE_ASSISTANT_CONTEXT and debug tags are removed.
    """
    s = (text or "").strip()
    if not s:
        return ""

    # Drop machine blocks if the parser missed them.
    for marker in ("---SUGGESTIONS---", "---CITATIONS---"):
        if marker.lower() in s.lower():
            s = re.split(re.escape(marker), s, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    # Remove multiline injected context dumps.
    s = re.sub(
        r"LIVE_ASSISTANT_CONTEXT\b[\s\S]*?(?=\n\n(?:Summary|Details|Status|Cross-refs|The |Die |Der |In )|\Z)",
        "",
        s,
        flags=re.IGNORECASE,
    )

    # Inline / bracket system markers (keep document refs like [SOP-IT-001]).
    bracketed = (
        r"LIVE_ASSISTANT_CONTEXT",
        r"LIVE_ASSISTANT",
        r"editor_context",
        r"RETRIEVED\s+CONTEXT",
    )
    for name in bracketed:
        s = re.sub(rf"\[\s*{name}\s*\]", "", s, flags=re.IGNORECASE)
        s = re.sub(rf"\b{name}\b", "", s, flags=re.IGNORECASE)

    inline_patterns = (
        r"\[REASONING\]",
        r"\[CONFIDENCE\]",
        r"\[ANSWER\]",
        r"\bSCOPE=ACTIVE_SOP_ONLY\b",
        r"\bACTIVE_SOP_ID=[0-9a-fA-F-]{8,}\b",
        r"\bRAG_HINTS:\s*[^\n\]]*",
        r"\bPLANNED_ASSISTANT_ACTION:\s*[^\n\]]*",
        r"\[\s*Retrieval scope:\s*[^\]]+\]",
        r"Retrieval scope:\s*ACTIVE SOP ONLY[^\n]*",
    )
    for pat in inline_patterns:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)

    # Remove internal context bullet lines.
    cleaned_lines: List[str] = []
    for line in s.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if not stripped:
            cleaned_lines.append("")
            continue
        if any(low.startswith(p) for p in _INTERNAL_ANSWER_LINE_PREFIXES):
            continue
        if re.match(r"^\[?(LIVE_ASSISTANT|RAG_HINTS|ACTIVE_SOP|SCOPE=)", stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    s = "\n".join(cleaned_lines)

    # Orphan brackets / punctuation left after stripping.
    s = re.sub(r"\[\s*\]", "", s)
    s = re.sub(r"\s+\[\s*(?=[,.;:!?\s]|$)", "", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = s.strip()

    s = _strip_sources_lines_from_answer(s)
    s = _strip_sources_footer_from_answer(s)
    return s.strip()


def _dedupe_citations_by_ref(cits: List[dict]) -> List[dict]:
    seen: set[str] = set()
    out: List[dict] = []
    for c in cits or []:
        if not isinstance(c, dict):
            continue
        ref = str(c.get("ref") or "").strip().lower()
        key = ref or str(id(c))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _format_sources_footer(cits: List[dict], max_refs: int = 14) -> str:
    refs: List[str] = []
    seen: set[str] = set()
    for c in cits or []:
        if not isinstance(c, dict):
            continue
        r = str(c.get("ref") or "").strip()
        if not r or r.startswith("INDEX-") or r.startswith("#"):
            continue
        rl = r.lower()
        if rl in seen:
            continue
        seen.add(rl)
        refs.append(r)
        if len(refs) >= max_refs:
            break
    if not refs:
        return ""
    return f"Sources: {', '.join(refs)}"


def _truncate_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _canonical_entity_id_key(raw: str) -> str:
    """Match Qdrant entity_id to Postgres UUID strings across formats."""
    s = str(raw or "").strip()
    if not s:
        return ""
    try:
        return str(uuid.UUID(s))
    except (ValueError, AttributeError, TypeError):
        return s


def _strip_injected_context_blocks(query: str) -> str:
    """Use core user wording for keyword routing when the prompt carries UI context."""
    q = (query or "").strip()
    for marker in (
        "\n\nRAG_HINTS:",
        "\n\nLIVE_ASSISTANT_CONTEXT",
        "\n\nPLANNED_ASSISTANT_ACTION:",
    ):
        if marker in q:
            q = q.split(marker, 1)[0].strip()
    return q


def _normalize_router_metadata_filters(filters: dict) -> dict:
    """
    LLM router uses sop_number / deviation_number keys; Qdrant payloads use ref_number
    at the top level for all entity types.
    """
    if not isinstance(filters, dict):
        return {}
    out = dict(filters)
    id_aliases = (
        ("sop_number", "ref_number"),
        ("deviation_number", "ref_number"),
        ("capa_number", "ref_number"),
        ("finding_number", "ref_number"),
        ("audit_number", "ref_number"),
        ("decision_number", "ref_number"),
    )
    for alt_key, canonical in id_aliases:
        alt_val = out.get(alt_key)
        if alt_val and not out.get(canonical):
            out[canonical] = alt_val
    return out


def _debug_chunk_summary(doc: Document, idx: int) -> str:
    meta = doc.metadata or {}
    ref = str(meta.get("ref_number") or meta.get("source_id") or f"chunk-{idx}")
    title = str(meta.get("title") or "")
    section = str(meta.get("_section") or meta.get("entity_type") or "unknown")
    score = _json_safe_float(meta.get("rerank_score") if meta.get("rerank_score") is not None else meta.get("hybrid_score", 0.0))
    snippet = _truncate_text((doc.page_content or "").replace("\n", " ").strip(), 220)
    return (
        f"[rag-debug] chunk#{idx} section={section} ref={ref} "
        f"title=\"{_console_safe(title)}\" score={score:.4f} text=\"{_console_safe(snippet)}\""
    )


def _build_retrieval_debug_rows(docs: List[Document], limit: int = 20) -> List[dict]:
    rows: List[dict] = []
    for i, doc in enumerate(docs[:max(0, limit)], 1):
        meta = doc.metadata or {}
        rows.append(
            {
                "rank": i,
                "section": str(meta.get("_section") or meta.get("entity_type") or ""),
                "source_id": str(meta.get("source_id") or meta.get("entity_id") or ""),
                "ref": str(meta.get("ref_number") or ""),
                "title": str(meta.get("title") or ""),
                "score": round(_json_safe_float(meta.get("rerank_score") if meta.get("rerank_score") is not None else meta.get("hybrid_score", 0.0)), 4),
                "status": str(meta.get("status") or ""),
                "snippet": _truncate_text((doc.page_content or "").replace("\n", " ").strip(), 280),
            }
        )
    return rows


# ─────────────────────────────────────────────
# Shared LLM
# ─────────────────────────────────────────────
def get_llm(temperature: float = 0.2):
    max_tokens = int(os.getenv("RAG_MAX_OUTPUT_TOKENS", "4096"))
    return create_chat_llm(
        temperature=temperature,
        max_output_tokens=max_tokens,
        max_retries=1,
    )


def get_fallback_llm(temperature: float = 0.2):
    max_tokens = int(os.getenv("RAG_MAX_OUTPUT_TOKENS", "4096"))
    return create_chat_llm(
        temperature=temperature,
        max_output_tokens=max_tokens,
        max_retries=0,
    )


# ─────────────────────────────────────────────
# ORIGINAL SINGLE-COLLECTION CHAIN (unchanged)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are SOPSearch AI - a compliance assistant for SOPs and regulatory processes.
Answer from context only. Be concise. If not found say: "Information not available in the knowledge base."
Do NOT fabricate document numbers or dates.
Write every answer in the same language as the user's question. If they explicitly request a language
(e.g. "reply in English", "answer in German"), follow that instruction instead.
"""
USER_PROMPT = "## Context\n{context}\n\n## Question\n{question}\n\nAnswer:"


class HybridRAGChain:
    def __init__(self, retriever: HybridRetriever, reranker: CrossEncoderReranker):
        self.retriever = retriever
        self.reranker  = reranker
        self.llm = get_llm()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT), ("human", USER_PROMPT),
        ])

    def invoke(self, query: str, category_filter: str = None) -> dict:
        self.retriever.category_filter = category_filter
        raw  = self.retriever.invoke(query)
        rnk  = self.reranker.rerank(query, raw)
        ctx, cits = build_context(rnk)
        ans = (self.prompt | self.llm | StrOutputParser()).invoke({"context": ctx, "question": query})
        return {"answer": ans, "citations": cits, "num_docs_retrieved": len(raw), "num_docs_reranked": len(rnk)}


# ─────────────────────────────────────────────────────────────────
# SMART RAG CHAIN — routes to relevant collection(s) only
# ─────────────────────────────────────────────────────────────────

SMART_SYSTEM = """\
You are a precise, bilingual QMS/IT Compliance AI Assistant integrated with a
production Hybrid RAG system.

When LIVE_ASSISTANT_CONTEXT names an Active SOP or SCOPE=ACTIVE_SOP_ONLY is present,
restrict every answer to that SOP and its linked entities only — never other SOPs.
Use LIVE_ASSISTANT_CONTEXT / ACTIVE_SOP / ACTIVE_SOP_ID hints when the question is about
"this SOP", "current", or linked deviations, CAPAs, audits, or decisions.

You have access to a structured Qdrant vector database with the following SEPARATE
collections. You MUST search the correct collection based on the user's intent:

================================================================
COLLECTION MAP
================================================================

Collection: "sops"
  → Contains : Standard Operating Procedures (SOPs)
  → Fields   : sop_number, title, department, sop_content,
                version_number, effective_date, review_date, status
  → Trigger keywords: "SOP", "procedure", "standard", "policy",
    "how to", "zugriffsmanagement", "patch", "firewall", "notfall",
    "KI-Systeme", "governance"

Collection: "deviations"
  → Contains : Deviation records and incidents
  → Fields   : deviation_number, title, description_text,
                root_cause_text, impact_level, external_status, event_date
  → Trigger keywords: "deviation", "incident", "issue", "problem",
    "DEV-", "breach", "excursion", "fehler", "abweichung", "kritisch"

Collection: "sop_versions"
  → Contains : Specific version content of SOPs
  → Fields   : version_number, content_json, effective_date,
                review_date, external_version_id, external_status
  → Trigger keywords: "version", "current version", "v4", "effective",
    "latest revision", "content of", "what does SOP say"

Collection: "capas"
  Contains: Corrective and Preventive Actions
  Fields: capa_number, title, action_text, external_status, effectiveness
  Triggers: "CAPA", "corrective action", "preventive action"

Collection: "audits"
  Contains: Audit findings
  Fields: finding / audit identifiers, finding_text, acceptance_status
  Triggers: "audit", "finding", "inspection", "AUDIT-"

Collection: "decisions"
  Contains: Decisions, rationales, conclusions
  Fields: decision_number, title, decision_statement, rationale_text, final_conclusion
  Triggers: "decision", "rationale", "conclusion", "approval", "DEC-"

================================================================
RULES YOU MUST ALWAYS FOLLOW
================================================================

RULE 1 — COLLECTION ROUTING
Before answering, explicitly identify which collection(s) to search.
Never merge data from deviations into SOPs or vice versa unless the user
explicitly asks for a cross-reference.

RULE 2 — EXACT POINT MATCHING
When the user mentions a specific identifier (e.g., "SOP-IT-001",
"DEV-IT-401", "DEV-2026-103"), you MUST filter on that exact field value.
Do not rely on semantic similarity alone.
Use metadata filter: { "sop_number": "SOP-IT-001" }
                  or { "deviation_number": "DEV-IT-401" }

RULE 3 — CHAIN OF THOUGHT
Before generating your final answer, you MUST perform and show a brief
reasoning block tagged as [REASONING]. In this block:
  (a) identify what the user is asking
  (b) decide which collection to search
  (c) identify any exact identifiers to filter on
  (d) plan your answer structure
Then produce your [ANSWER].

RULE 4 — CITATIONS
Every factual claim in your answer MUST be linked to its source record
using this format: [SOP-IT-001], [DEV-IT-401], [SOP-QA-010 v4.0]
Never state a fact without a citation tag.
If you cannot cite it, do not state it.

RULE 5 — CONVERSATION MEMORY
You have access to the full conversation history. When the user says
"that deviation", "the one we just discussed", "same SOP", "previous answer"
— you MUST resolve the reference from earlier in the conversation history.
Never ask the user to repeat what they already told you.

RULE 6 — IMPACT LEVEL AWARENESS
When discussing deviations, always surface the impact_level in your answer.
Priority order: Critical > Major > Moderate > Minor
Flag Critical and Major deviations explicitly with a ⚠️ marker.

RULE 7 — RESPONSE LANGUAGE (MANDATORY)
The knowledge base contains German and English documents. Retrieval may surface either language.
Your [ANSWER], any refusal sentence, and ---SUGGESTIONS--- must use the language of the user's latest
message in USER QUESTION — not the language of retrieved snippets or document titles.
1. User asks in English → respond in English.
2. User asks in German → respond in German.
3. Explicit language instruction ("reply in English", "answer in German", etc.) overrides detection.
4. Mixed-language input → use the language of the latest clear user instruction.
5. Do not force all answers into German or English when the user used another language.
6. Translate or summarize retrieved content into the user's language as needed; keep record IDs
   (e.g. [SOP-IT-001], [DEV-IT-401]) unchanged.
For search: if the user asks in English about a German SOP title (or vice versa), interpret intent
correctly and search both languages; still answer in the user's language.

RULE 8 — STATUS AWARENESS
Always report the current status of records:
  - Deviations  : open | under_investigation | closed
  - SOP versions: effective | draft | obsolete
Never present a closed deviation or obsolete SOP version as currently active.

RULE 9 — CROSS-REFERENCE DETECTION
If the user asks about a deviation, check if a related SOP exists that
governs that area.
Example: DEV-IT-101 → SOP-IT-001 (OT access management)
Proactively surface this link as: [RELATED SOP: SOP-IT-001]

RULE 10 — REFUSAL RULE (RETRIEVAL-GROUNDED ONLY)
You are NOT a general-purpose chatbot. Answer ONLY using facts present in
RETRIEVED CONTEXT below. Do not use outside knowledge, training data, or guesses.
When RETRIEVED CONTEXT includes records that bear on the question, answer from
that context with citations. Refuse when snippets are empty, off-topic, or when
the user asks general knowledge unrelated to QMS documents (e.g. "what is an LLM?").
If you must refuse, reply with exactly one sentence in the user's language (see RULE 7), e.g. English:
"I can only answer questions related to uploaded documents, SOPs, or retrieved system context."
or German:
"Ich kann nur Fragen zu hochgeladenen Dokumenten, SOPs oder abgerufenem Systemkontext beantworten."
Never hallucinate fields, dates, or root causes that are null or missing in the data.
"""

SMART_USER = """\
{history_focus}

────────────────────────────────────────
CONVERSATION HISTORY:
(Carried in the message list before this user turn; use it for follow-ups.)

────────────────────────────────────────
RETRIEVED CONTEXT:
{context}

────────────────────────────────────────
USER QUESTION:
{question}

────────────────────────────────────────
INSTRUCTIONS FOR THIS RESPONSE:

STEP 1 — [REASONING]  (required; always show this block first)
Answer each point briefly before [ANSWER]:
  • What is the user asking? (one sentence)
  • Which collection(s) does the retrieved context correspond to, and why?
  • Any exact ID in the question or history? Which field/record?
  • Any reference to earlier messages to resolve?
  • Impact level / status for the records involved (if applicable)?
  • Any cross-collection links to surface?

STEP 2 — [ANSWER]
  • RESPONSE LANGUAGE (RULE 7): Write [ANSWER] in the same language as USER QUESTION above.
    Do not answer in German because context is German, or in English because context is English.
    Honor explicit language instructions from the user.
  • Answer directly and completely.
  • Cite every fact with bracket notation, e.g.
    [SOP-IT-001], [DEV-IT-401], [CAPA-22], [AUDIT-7], [DEC-15]
  • For deviations with impact_level Critical or Major, start that bullet or
    sentence with the warning marker (⚠️).
  • If a related SOP governs the topic, add a line:
    [RELATED SOP: SOP-XX-XXX — title]
  • If version or effective date appears in the context, you may include it
    in the citation line, e.g. [SOP-QA-010 v4.0 | effective: YYYY-MM-DD]
  • If SCOPE=ACTIVE_SOP_ONLY: never cite or discuss records outside the active SOP scope.

  For non-trivial answers, use this structure (plain text, no markdown tables):
  Summary: one short paragraph
  Details: bullet lines, each with citations
  Status: current status / impact when known from context
  Cross-refs: related SOPs, deviations, CAPAs, audits, or decisions if grounded in context

  Do not use markdown headings (no #), bold, tables, or code fences.
  Never echo internal labels in [ANSWER] (e.g. LIVE_ASSISTANT_CONTEXT, RAG_HINTS,
  ACTIVE_SOP_ID, SCOPE=ACTIVE_SOP_ONLY, editor_context, or retrieval-scope notes).
  Stay within 400 words unless the user explicitly asks for full detail.
  End the [ANSWER] section with a line:
  Sources: list every cited record ID in brackets, comma-separated
  (You may prefix that line with 📎 for example: "📎 Sources: [SOP-IT-001], [DEV-IT-401]")

STEP 3 — [CONFIDENCE]
  One line, e.g.:
  [CONFIDENCE] HIGH — exact record aligned with an identifier in context;
  or MEDIUM — semantic match, recommend verification;
  or LOW — partial context only; still summarize what is present and cite it.

────────────────────────────────────────
FORMAT RULES
  Do not use vague phrasing like "the document mentions" when you can name
  [SOP-…] or [DEV-…] from context.
  Do not present null or missing fields as if they were populated.
  Do not use markdown headings, bold markers, tables, or code fences.

After [CONFIDENCE], you MUST append the following machine-readable blocks
exactly (the application parses them). List each cited source once in
---CITATIONS---; then three to four follow-up questions in JSON, in the same
language as USER QUESTION (RULE 7).

---CITATIONS---
[[REF_ID|Document Title|Type|One sentence excerpt]]
[[REF_ID|Document Title|Type|One sentence excerpt]]

---SUGGESTIONS---
["Follow-up using record IDs from context", "Second follow-up", "Third follow-up"]
"""


def _editor_context_documents_from_prompt(full_prompt: str) -> List[Document]:
    """Turn injected LIVE_ASSISTANT_CONTEXT into retrievable context (not a hardcoded answer)."""
    raw = str(full_prompt or "")
    marker = "LIVE_ASSISTANT_CONTEXT"
    if marker not in raw:
        return []
    block = raw.split(marker, 1)[1]
    for end_marker in ("\n\nRAG_HINTS:", "\n\nPLANNED_ASSISTANT_ACTION:"):
        if end_marker in block:
            block = block.split(end_marker, 1)[0]
    text = f"{marker}\n{block.strip()}"
    if len(text.strip()) < 24:
        return []
    active_ref, active_title = _extract_active_sop_from_prompt(raw)
    return [
        Document(
            page_content=text[:4000],
            metadata={
                "ref_number": active_ref or "EDITOR_CONTEXT",
                "title": active_title or "Active editor context",
                "entity_type": "editor_context",
                "source_id": "live_assistant_context",
                "_section": "sops",
            },
        )
    ]


def _build_unified_context(docs: List[Document], prefix_label: str) -> Tuple[str, List[dict]]:
    """Build a numbered context string from retrieved docs, regardless of collection."""
    if not docs:
        return "", []

    parts, raw_cits = [], []
    total = 0
    MAX = MAX_CONTEXT_CHARS

    for i, doc in enumerate(docs):
        text = doc.page_content.strip()
        if not text or total + len(text) > MAX:
            break

        meta     = doc.metadata
        ref      = meta.get("ref_number", "")
        title    = meta.get("title", "")
        doc_type = meta.get("doc_type", prefix_label)
        status   = meta.get("status", "")

        header_parts = [f"[{i}]", doc_type.upper()]
        if ref:    header_parts.append(ref)
        if title:  header_parts.append(f'"{title}"')
        if status: header_parts.append(f"({status})")
        header = " ".join(header_parts)

        parts.append(f"{header}\n{text}")
        raw_cits.append({
            "ref":    ref or f"#{i}",
            "title":  title,
            "type":   doc_type,
            "status": status,
            "score":  round(_json_safe_float(meta.get("rerank_score") if meta.get("rerank_score") is not None else meta.get("hybrid_score", 0.0)), 4),
        })
        total += len(text)

    return "\n\n---\n\n".join(parts), raw_cits


def _unique_by_source(docs: List[Document], limit: int, max_per_source: int = 3) -> List[Document]:
    """
    Keep top documents while allowing multiple chunks per source_id/ref.
    This prevents one document from dominating context while ensuring we get
    more than just the header/title page of a document.
    """
    out: List[Document] = []
    counts = {}  # {key: count}
    for doc in docs:
        meta = doc.metadata or {}
        key = meta.get("source_id") or meta.get("ref_number") or meta.get("title")
        if not key:
            key = id(doc)
            
        current_count = counts.get(key, 0)
        if current_count >= max_per_source:
            continue
            
        counts[key] = current_count + 1
        out.append(doc)
        
        if len(out) >= limit:
            break
    return out


def _parse_answer_citations_suggestions(raw: str) -> Tuple[str, List[dict], List[str], str, str]:
    """
    Parse the LLM output into:
      answer     : clean prose text from [ANSWER] block
      citations  : list of dicts extracted from [[REF|TITLE|TYPE|EXCERPT]] tags
      suggestions: list of strings from the ---SUGGESTIONS--- block
      reasoning  : text from [REASONING] block
      confidence : text from [CONFIDENCE] block
    """
    answer      = ""
    citations   = []
    suggestions = []
    reasoning   = ""
    confidence  = ""

    # 1. Extract ---SUGGESTIONS---
    sug_match = re.search(r'---SUGGESTIONS---\s*(\[.*?\])', raw, re.DOTALL | re.IGNORECASE)
    if sug_match:
        try:    suggestions = json.loads(sug_match.group(1))
        except: suggestions = []
        raw = raw[:sug_match.start()].strip()

    # 2. Extract Citations using Tag Format: [[ref|title|type|excerpt]]
    cit_marker = "---CITATIONS---"
    if cit_marker in raw:
        parts = raw.split(cit_marker)
        raw_content = parts[0].strip()
        cit_text = parts[1].strip()
        
        # Match [[ ... | ... | ... | ... ]]
        matches = re.findall(r'\[\[(.*?)\|(.*?)\|(.*?)\|(.*?)\]\]', cit_text)
        for ref, title, doc_type, excerpt in matches:
            citations.append({
                "ref":     ref.strip(),
                "title":   title.strip(),
                "type":    doc_type.strip(),
                "excerpt": excerpt.strip()
            })
    else:
        raw_content = raw.strip()

    # 3. Extract [REASONING], [ANSWER], [CONFIDENCE] blocks
    # Looking for blocks started by bracketed headers
    reason_match = re.search(r'\[REASONING\](.*?)(?=\[ANSWER\]|\[CONFIDENCE\]|$)', raw_content, re.DOTALL | re.IGNORECASE)
    if reason_match:
        reasoning = reason_match.group(1).strip()
    
    answer_match = re.search(r'\[ANSWER\](.*?)(?=\[CONFIDENCE\]|\[REASONING\]|$)', raw_content, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Fallback if no specific block found, use everything but reasoning/confidence
        answer = raw_content

    conf_match = re.search(r'\[CONFIDENCE\](.*?)$', raw_content, re.DOTALL | re.IGNORECASE)
    if conf_match:
        confidence = conf_match.group(1).strip()

    if not (answer or "").strip() and raw_content:
        stripped = re.sub(
            r"\[REASONING\][\s\S]*?(?=\[ANSWER\]|\[CONFIDENCE\]|$)",
            "",
            raw_content,
            flags=re.IGNORECASE,
        ).strip()
        stripped = re.sub(r"\[ANSWER\]\s*", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"\[CONFIDENCE\][\s\S]*$", "", stripped, flags=re.IGNORECASE).strip()
        if stripped:
            answer = stripped[:8000]

    # Clamp suggestions
    suggestions = [s for s in suggestions if isinstance(s, str)][:4]

    answer = sanitize_user_facing_answer(
        _strip_sources_footer_from_answer(_strip_sources_lines_from_answer(answer))
    )

    return answer, citations, suggestions, reasoning, confidence


from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import HumanMessagePromptTemplate, MessagesPlaceholder


def _looks_like_sop_generation_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    sop_terms = r"\b(sop|standard operating procedure|procedure document|work instruction)\b"
    intent_terms = (
        r"\b(generate|create|draft|write|prepare|convert|format|structure|"
        r"make this|turn this|build)\b"
    )
    # Generation mode should only trigger on explicit create/draft intent.
    # Multiline/context-heavy assistant prompts can otherwise cause false positives.
    return bool(re.search(sop_terms, q)) and bool(re.search(intent_terms, q))


def _build_sop_generation_prompt(raw_input: str) -> str:
    return f"""You are a senior SOP technical writer for regulated environments.

TASK
Transform the raw user input into a complete, production-ready SOP document.

RAW INPUT
{raw_input}

OUTPUT REQUIREMENTS
1) Output ONLY the SOP body in clean plain text (no markdown headings, no code fences).
2) Use the same language as the user's RAW INPUT. If they explicitly request a language, use that language.
   Use professional, concise, domain-appropriate wording.
3) Build a logical, complete hierarchy with numbered sections and subsections.
4) Numbering style must be consistent (e.g., 1.0, 1.1, 1.2 ... 2.0 ...).
5) Include these core sections when relevant:
   - Title
   - Purpose
   - Scope
   - Responsibilities
   - Procedure
6) Add additional sections when context requires them, for example:
   - Definitions
   - Safety / Precautions
   - Compliance / Regulatory References
   - Records / Documentation
   - Deviations / Exceptions
   - Revision History
7) Do NOT force irrelevant sections. If a section is not relevant, omit it.
8) In Procedure, provide clear ordered steps and substeps with role ownership where possible.
9) Resolve fragmented/raw notes into polished paragraphs and structured bullets.
10) Keep terminology and tone consistent throughout.

QUALITY BAR
- The SOP must be ready to paste directly into an editor with minimal/no formatting edits.
- Avoid placeholders like TBD unless absolutely necessary.
"""


def _classify_sop_inventory_query(query: str) -> Optional[Literal["count", "list"]]:
    """
    Detects SOP inventory questions so we can return a deterministic count/list
    without LLM drift. "count" = how many; "list" = enumerate SOPs.
    """
    q = (query or "").lower()
    if not re.search(
        r"\b(sop|sops|standard operating procedures?)\b",
        q,
        re.IGNORECASE,
    ):
        return None
    has_list_intent = bool(
        re.search(
            r"\b(list all|list every|show all|show me all|get all|name all|enumerate|all sops|every sop)\b",
            q,
        )
    ) or bool(re.search(r"\b(list|show)\b.+\b(sop|sops)\b", q)) or bool(
        re.search(r"\b(which|what) sops\b", q)
    )
    has_count_intent = any(
        p in q
        for p in (
            "how many",
            "how much",
            "number of",
            "total",
            "count ",
            " count",
            "quantity",
            "sop count",
        )
    )
    if re.search(r"\bkitne\b", q):
        has_count_intent = True
    if re.search(r"\b(how many sops|count sops|sop count|number of sops|total sops)\b", q):
        has_count_intent = True
    if re.search(
        r"\b(do we have|have we|is there|are there)\b", q
    ) and re.search(r"\b(sop|sops)\b", q):
        has_count_intent = True

    if has_list_intent and not has_count_intent:
        return "list"
    if has_count_intent and not has_list_intent:
        return "count"
    if has_list_intent and has_count_intent:
        if re.search(r"\bhow many\b", q) or re.search(
            r"\b(number|count|total) of\b", q
        ):
            return "count"
        return "list"
    if re.search(
        r"\b(available|exist|in the (system|index|database))\b", q
    ) and re.search(r"\bwhich\b.*\b(sop|sops)\b", q):
        return "list"
    if re.search(r"\b(available|exist|inventory)\b", q) and re.search(
        r"\b(how many|count|number)\b", q
    ):
        return "count"
    if re.search(r"\b(how many|count|number of|total)\b", q) and re.search(
        r"\b(relevant|related|applicable)\b.*\b(sop|sops)\b", q
    ):
        return "count"
    if re.search(r"\b(how many|count|number of|total)\b", q) and re.search(
        r"\b(sop|sops)\b.*\b(relevant|related|applicable)\b", q
    ):
        return "count"
    return None


def _looks_cross_domain_query(query: str) -> bool:
    q = (query or "").lower()
    return bool(
        re.search(
            r"\b(deviation|deviations|dev-|capa|capas|audit|audits|finding|findings|decision|decisions|linked|related)\b",
            q,
        )
    )


def _strict_sop_inventory_response(
    docs: List[Document],
    query: str,
    retriever: HybridRetriever | None = None,
    mode: Literal["count", "list"] = "list",
    full_prompt: str = "",
) -> dict:
    """Deterministic SOP count/list: PostgreSQL active count + indexed distinct SOPs."""
    prompt_for_active = (full_prompt or query or "").strip()
    active_ref, active_title = _extract_active_sop_from_prompt(prompt_for_active)

    inventory_docs: List[Document] = list(docs or [])
    allowed_ids: set[str] = set()
    raw_ids = None
    if retriever is not None:
        mf = getattr(retriever, "metadata_filters", {}) or {}
        raw_ids = mf.get("allowed_entity_ids") if isinstance(mf, dict) else None
        if isinstance(raw_ids, list):
            allowed_ids = {_canonical_entity_id_key(str(v)) for v in raw_ids if str(v).strip()}
    if isinstance(raw_ids, list) and len(raw_ids) == 0:
        inventory_docs = []
    elif retriever is not None:
        try:
            corpus_docs, _ = retriever._get_bm25_corpus()
            if corpus_docs:
                if allowed_ids:
                    inventory_docs = [
                        d
                        for d in corpus_docs
                        if _canonical_entity_id_key(
                            str((d.metadata or {}).get("entity_id", ""))
                        )
                        in allowed_ids
                    ]
                elif raw_ids is None:
                    inventory_docs = corpus_docs
        except Exception:
            pass

    rows: List[Tuple[str, str, str]] = []
    seen: set = set()
    for doc in inventory_docs:
        meta = doc.metadata or {}
        et = str(meta.get("entity_type", "")).lower()
        if rag_unified_enabled() and et and et != "sop":
            continue
        ref = (
            (meta.get("ref_number") or meta.get("sop_number") or meta.get("source_id"))
            or ""
        )
        if not ref and meta.get("entity_id"):
            ref = f"id:{str(meta.get('entity_id'))[:8]}"
        title = meta.get("title") or "Untitled SOP"
        status = meta.get("status") or "Unknown"
        page_content = (doc.page_content or "").strip()

        if (not ref or ref.startswith("id:")) and page_content:
            first_line = page_content.splitlines()[0].strip()
            if " - " in first_line:
                maybe_ref, maybe_title = first_line.split(" - ", 1)
                if maybe_ref.strip() and not maybe_ref.strip().lower().startswith(
                    "id:"
                ):
                    ref = maybe_ref.strip()
                if maybe_title.strip() and title == "Untitled SOP":
                    title = maybe_title.strip()

        eid = str(meta.get("entity_id") or "").lower()
        dedupe_key = f"{eid}|{(ref or '').lower()}" if eid else f"r|{(ref or title).lower()}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        display_ref = ref or title
        rows.append((display_ref, title, status))

    rows = sorted(rows, key=lambda x: (x[0] or "").lower())
    indexed_distinct = len(rows)
    db_total = _db_active_sop_count()
    list_cap = int(os.getenv("SOP_INVENTORY_LIST_MAX", "50"))

    if mode == "list" and not rows and db_total > 0:
        rows = _db_active_sop_rows(list_cap * 2)[:list_cap]
        rows = sorted(rows, key=lambda x: (x[0] or "").lower())

    def _inventory_citations() -> List[dict]:
        cits: List[dict] = [
            {
                "ref": "SOP database",
                "title": "Active SOP records (PostgreSQL)",
                "type": "metadata",
                "excerpt": f"{db_total} active SOP(s) in the database.",
                "score": 1.0,
            },
        ]
        if indexed_distinct != db_total:
            cits.append(
                {
                    "ref": "SOP index",
                    "title": "Indexed SOP knowledge",
                    "type": "metadata",
                    "excerpt": f"{indexed_distinct} distinct SOP(s) in the vector/BM25 index.",
                    "score": 0.9,
                }
            )
        for ref, title, status in rows[: min(12, list_cap)]:
            cits.append(
                {
                    "ref": ref,
                    "title": title,
                    "type": "SOP",
                    "excerpt": f"Status: {status}",
                    "score": 0.5,
                }
            )
        if active_ref:
            cits.insert(
                1,
                {
                    "ref": active_ref,
                    "title": active_title or "Active SOP (editor)",
                    "type": "SOP",
                    "excerpt": "Currently open in the SOP editor.",
                    "score": 0.95,
                },
            )
        return _dedupe_citations_by_ref(_sanitize_citation_list(cits))

    if mode == "count":
        count_citations: List[dict] = []
        if db_total == 0:
            count_answer = NO_SOP_CONTEXT_FALLBACK
        else:
            lines = []
            if indexed_distinct != db_total:
                lines.append(
                    f"Database has {db_total} SOP(s); indexed knowledge contains {indexed_distinct} SOP(s)."
                )
            else:
                lines.append(
                    f"There are {db_total} SOP(s) in the database (active records)."
                )
            if active_ref:
                tpart = f", {active_title}" if active_title else ""
                lines.append(f"The currently active SOP is {active_ref}{tpart}.")
            count_citations = _inventory_citations()
            foot = _format_sources_footer(count_citations)
            if foot:
                lines.append("")
                lines.append(foot)
            count_answer = "\n".join(lines)
        return {
            "answer": count_answer,
            "citations": count_citations if db_total or indexed_distinct else [],
            "suggestions": [
                "List all SOPs with titles",
                "What does the active SOP cover?",
                "Which deviations link to this SOP?",
            ],
            "retrieval_stats": {},
            "routed_to": "SOPs (strict count)",
            "cached": False,
            "metadata_snapshot": [],
            "audit_log_snapshot": [],
            "action_metadata": {
                "query": query,
                "routing": ["sops"],
                "latency_ms": 0.0,
                "timestamp": time.time(),
                "model": "deterministic",
                "strict_mode": "sop_inventory_count",
            },
        }

    if db_total == 0:
        return {
            "answer": NO_SOP_CONTEXT_FALLBACK,
            "citations": [],
            "suggestions": [
                "How many SOPs are in the database?",
                "Summarize the active SOP",
                "What deviations link to this SOP?",
            ],
            "retrieval_stats": {},
            "routed_to": "SOPs",
            "cached": False,
            "metadata_snapshot": [],
            "audit_log_snapshot": [],
            "action_metadata": {
                "query": query,
                "routing": ["sops"],
                "latency_ms": 0.0,
                "timestamp": time.time(),
                "model": "deterministic",
                "strict_mode": "sop_inventory_list_empty",
            },
        }

    key_points = "\n".join(
        [f"- {ref}: {title} [{status}]" for ref, title, status in rows[:list_cap]]
    )
    if len(rows) > list_cap:
        key_points += (
            f"\n- … and {len(rows) - list_cap} more (truncated; increase SOP_INVENTORY_LIST_MAX)."
        )
    citations = _inventory_citations()

    answer_lines: List[str] = []
    if indexed_distinct != db_total:
        answer_lines.append(
            f"Database has {db_total} SOP(s); indexed knowledge contains {indexed_distinct} SOP(s)."
        )
    else:
        answer_lines.append(
            f"There are {db_total} SOP(s) in the database (active records)."
        )
    answer_lines.append("")
    answer_lines.append("SOP list:")
    answer_lines.append(
        key_points if key_points else "(No SOP rows in the index; list above is from the database.)"
    )
    if active_ref:
        tpart = f", {active_title}" if active_title else ""
        answer_lines.append("")
        answer_lines.append(f"Current SOP in the editor: {active_ref}{tpart}.")
    foot = _format_sources_footer(citations)
    if foot:
        answer_lines.append("")
        answer_lines.append(foot)

    suggestions = [
        "How many SOPs are in the database?",
        "Summarize the active SOP",
        "What deviations link to this SOP?",
    ]

    return {
        "answer": "\n".join(answer_lines),
        "citations": citations,
        "suggestions": suggestions,
        "retrieval_stats": {},
        "routed_to": "SOPs",
        "cached": False,
        "metadata_snapshot": [],
        "audit_log_snapshot": [],
        "action_metadata": {
            "query": query,
            "routing": ["sops"],
            "latency_ms": 0.0,
            "timestamp": time.time(),
            "model": get_local_llm_config().model,
            "strict_mode": "sop_inventory",
        },
    }


class SmartRAGChain:
    """
    Intelligent RAG chain that:
      1. Routes the query to the relevant collection(s) only.
      2. Does Hybrid Search (Dense + BM25) + Cross-Encoder reranking.
      3. Injects chat history for multi-turn memory + CoT reasoning.
      4. Returns: clean prose answer | citations | dynamic suggestions.
    """

    def __init__(self, federated_retriever: FederatedRetriever):
        self.federated = federated_retriever
        self.llm = get_llm()
        self.router = LLMRouter(llm=self.llm)
        self._active_ids_cache: dict[str, tuple[datetime, list[str]]] = {}
        self._last_scoped_sop_id: str = ""
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SMART_SYSTEM),
            MessagesPlaceholder(variable_name="chat_history_messages"),
            HumanMessagePromptTemplate.from_template(SMART_USER),
        ])

    def _get_active_entity_ids(self, section: str) -> list[str]:
        cache_ttl = int(os.getenv("RAG_ACTIVE_IDS_CACHE_SECONDS", "30"))
        now = datetime.utcnow()
        cached = self._active_ids_cache.get(section)
        if cached and (now - cached[0]) < timedelta(seconds=cache_ttl):
            return cached[1]

        db = SessionLocal()
        try:
            if section == "sops":
                ids = [str(row[0]) for row in db.query(SOP.id).filter(SOP.is_active == True).all()]  # noqa: E712
            elif section == "deviations":
                ids = [str(row[0]) for row in db.query(Deviation.id).all()]
            elif section == "capas":
                ids = [str(row[0]) for row in db.query(Capa.id).all()]
            elif section == "audits":
                ids = [str(row[0]) for row in db.query(AuditFinding.id).all()]
            elif section == "decisions":
                ids = [str(row[0]) for row in db.query(Decision.id).all()]
            else:
                ids = []
        finally:
            db.close()

        self._active_ids_cache[section] = (now, ids)
        return ids

    def _clear_scope_cache_if_switched(self, active_sop_id: str) -> None:
        key = _canonical_entity_id_key(active_sop_id)
        if key and key != self._last_scoped_sop_id:
            self._active_ids_cache.clear()
            self._last_scoped_sop_id = key

    def _resolve_allowed_entity_ids(
        self,
        section: str,
        active_scope: dict | None,
        full_prompt: str = "",
    ) -> list[str] | None:
        """
        None  → corpus-wide active entities (_get_active_entity_ids).
        list  → restrict retrieval to these entity UUIDs (may be empty).
        """
        scope = active_scope if isinstance(active_scope, dict) else {}
        scoped_id = str(scope.get("active_sop_id") or "").strip()
        if not scoped_id and _prompt_requests_active_sop_scope(full_prompt):
            scoped_id = _extract_active_sop_id_from_prompt(full_prompt)
        if not scoped_id:
            return None

        sid = _canonical_entity_id_key(scoped_id)

        def _norm_list(key: str) -> list[str]:
            raw = scope.get(key) or []
            if not isinstance(raw, list):
                return []
            return [_canonical_entity_id_key(str(x)) for x in raw if str(x).strip()]

        if section == "sops":
            ids = [sid] + [x for x in _norm_list("linked_sop_ids") if x and x != sid]
        elif section == "deviations":
            ids = _norm_list("linked_deviation_ids")
        elif section == "capas":
            ids = _norm_list("linked_capa_ids")
        elif section == "audits":
            ids = _norm_list("linked_audit_ids")
        elif section == "decisions":
            ids = _norm_list("linked_decision_ids")
        else:
            ids = []

        return list(dict.fromkeys(x for x in ids if x))

    def _extract_metadata_filters(self, query: str) -> dict:
        """
        Extracts department or specific document reference filters from the query.
        Example: 'IT/sops' -> {'department': 'IT'}
        Example: 'SOP-IT-001' -> {'ref_number': 'SOP-IT-001'}
        """
        filters = {}
        q = query.upper()
        
        # 1. Department pattern (e.g. IT/sops, HR documents)
        dept_match = re.search(r'\b(IT|HR|FINANCE|QUALITY|COMPLIANCE|SECURITY|OPS|LEGAL)\b', q)
        if dept_match:
            filters["department"] = dept_match.group(1)
            
        # 2. Document ID pattern (e.g. SOP-xxx, DEV-xxx)
        id_match = re.search(r'\b(SOP|DEV|CAPA|AUDIT|DEC)-[A-Z0-9-]+\b', q)
        if id_match:
            filters["ref_number"] = id_match.group(0)
            
        return filters

    def _find_active_doc_id(self, chat_history: List[Dict]) -> str:
        """Scan last 2-3 messages in history for any document IDs (SOP, DEV, etc)."""
        if not chat_history:
            return ""
        
        # Scan in reverse, looking for document ID patterns
        pattern = re.compile(r'\b(SOP|DEV|CAPA|AUDIT|DEC)-[A-Z0-9-]+\b', re.IGNORECASE)
        for msg in reversed(chat_history[-4:]):
            content = msg.get("content", "")
            match = pattern.search(content)
            if match:
                return match.group(0).upper()
        return ""

    def _retrieve_ranked_for_section(
        self,
        section: str,
        query_for_routing: str,
        metadata_filters: dict,
        active_doc_id: str,
        num_target_sections: int,
        active_scope: dict | None = None,
        full_prompt: str = "",
    ) -> tuple[str, List[Document], int]:
        """Hybrid retrieve + rerank for one router section (thread-safe per section retriever)."""
        retriever = self.federated.retrievers.get(section)
        if not retriever:
            return section, [], 0
        try:
            section_filters = dict(metadata_filters or {})
            scoped_ids = self._resolve_allowed_entity_ids(section, active_scope, full_prompt)
            if scoped_ids is not None:
                section_filters["allowed_entity_ids"] = scoped_ids
            else:
                section_filters["allowed_entity_ids"] = self._get_active_entity_ids(section)
            retriever.metadata_filters = section_filters
            if rag_unified_enabled():
                retriever.category_filter = section
            else:
                retriever.category_filter = None
            docs = retriever.invoke(query_for_routing)
            allowed_raw = section_filters.get("allowed_entity_ids")
            if isinstance(allowed_raw, list) and len(allowed_raw) == 0:
                docs = []
            else:
                allowed_ids = {
                    _canonical_entity_id_key(str(x))
                    for x in (allowed_raw or [])
                    if str(x).strip()
                }
                if allowed_ids:
                    docs = [
                        d
                        for d in docs
                        if _canonical_entity_id_key(
                            str((d.metadata or {}).get("entity_id") or (d.metadata or {}).get("source_id") or "")
                        )
                        in allowed_ids
                    ]
            top_n = 20 if num_target_sections == 1 else 10
            ranked = self.federated.reranker.rerank_top_n(query_for_routing, docs, top_n)
            max_chunks = 6 if active_doc_id else 4
            unique_limit = 15 if num_target_sections == 1 else 8
            ranked = _unique_by_source(ranked, unique_limit, max_per_source=max_chunks)
            for d in ranked:
                d.metadata["_section"] = section
            return section, ranked, len(ranked)
        except Exception as e:
            logger.warning("[rag-retrieval] section=%s failed: %s", section, e)
            return section, [], 0

    def _generate_structured_sop(self, user_input: str) -> str:
        prompt = _build_sop_generation_prompt(_truncate_text(user_input, MAX_QUERY_CHARS))
        parser = StrOutputParser()
        try:
            return (self.llm | parser).invoke(prompt).strip()
        except Exception:
            fallback_llm = get_fallback_llm()
            return (fallback_llm | parser).invoke(prompt).strip()

    def invoke(
        self,
        query: str,
        category: str = None,
        chat_history: List[Dict] = None,
        active_scope: dict | None = None,
    ) -> dict:
        t0 = time.time()
        query_for_routing = _strip_injected_context_blocks(query)
        full_ctx_query = (query or "").strip()
        scoped_sop_id = ""
        if isinstance(active_scope, dict):
            scoped_sop_id = str(active_scope.get("active_sop_id") or "").strip()
        if not scoped_sop_id and _prompt_requests_active_sop_scope(full_ctx_query):
            scoped_sop_id = _extract_active_sop_id_from_prompt(full_ctx_query)
        if scoped_sop_id:
            self._clear_scope_cache_if_switched(scoped_sop_id)
        llm_cfg = get_local_llm_config()
        logger.info(
            "[rag-invoke] provider=%s model=%s retrieval_query_preview=%s",
            llm_cfg.provider,
            llm_cfg.model,
            _console_safe(_truncate_text(query_for_routing or query, 240)),
        )
        cat_norm = (category or "").strip().lower()
        route_data = self.router.route(query_for_routing)
        sop_inventory_mode: Optional[Literal["count", "list"]] = None
        editor_scoped = bool(scoped_sop_id) or _prompt_requests_active_sop_scope(full_ctx_query)
        if not RAG_DISABLE_SOP_INVENTORY and ((not cat_norm) or cat_norm == "sops") and not editor_scoped:
            sop_inventory_mode = _classify_sop_inventory_query(query_for_routing)

        # ── Step 0: Extract Metadata Filters & Active Doc ID ──
        metadata_filters = self._extract_metadata_filters(query_for_routing)
        active_doc_id = self._find_active_doc_id(chat_history) if chat_history else ""
        if editor_scoped and scoped_sop_id:
            active_ref_hint, _ = _extract_active_sop_from_prompt(full_ctx_query)
            if active_ref_hint and active_ref_hint.upper().startswith("SOP"):
                metadata_filters["ref_number"] = active_ref_hint.upper()
        elif active_doc_id and not sop_inventory_mode:
            _safe_print(f"  [context] identified active doc from history: {active_doc_id}")
            is_sop_query = any(
                k in (query_for_routing or "").lower() for k in ["sop", "procedure", "standard"]
            )
            if active_doc_id.startswith("SOP") and is_sop_query:
                metadata_filters["ref_number"] = active_doc_id
        if sop_inventory_mode == "count" and not re.search(
            r"\bSOP-[A-Z0-9-]+\b", query_for_routing or "", re.IGNORECASE
        ):
            metadata_filters.pop("ref_number", None)

        logger.info(
            "[rag-routing] filters=%s sop_inventory_mode=%s",
            metadata_filters,
            sop_inventory_mode,
        )

        # ── Step 1: Route query using LLM Router (Prompt 3) ──
        forced_category = category and category.strip().lower() in {
            "sops",
            "deviations",
            "capas",
            "audits",
            "decisions",
        }
        if forced_category and not _looks_cross_domain_query(query_for_routing):
            target_sections = [category.strip().lower()]
            route_data = {"collections": target_sections, "exact_filters": dict(metadata_filters)}
        elif sop_inventory_mode:
            target_sections = ["sops"]
            route_data = {"collections": ["sops"], "exact_filters": dict(metadata_filters)}
        else:
            target_sections = route_data.get("collections", [])
            metadata_filters.update(route_data.get("exact_filters", {}))

        metadata_filters = _normalize_router_metadata_filters(metadata_filters)

        if not target_sections:
            fb = route_query(query_for_routing)
            target_sections = fb or ["sops", "deviations", "capas", "audits", "decisions"]
            logger.warning(
                "[rag-routing] empty collections after router; keyword_fallback=%s",
                target_sections,
            )

        routed_label = describe_route(target_sections)
        logger.info(
            "[rag-routing] query='%s' sections=%s filters=%s",
            (query_for_routing or "")[:120],
            target_sections,
            metadata_filters,
        )
        _safe_print(f"[rag-routing] sections={target_sections} filters={metadata_filters}")

        if sop_inventory_mode and not RAG_DISABLE_SOP_INVENTORY:
            sop_retriever = self.federated.retrievers.get("sops")
            if sop_retriever:
                section_filters = dict(metadata_filters or {})
                scoped_ids = self._resolve_allowed_entity_ids("sops", active_scope, full_ctx_query)
                section_filters["allowed_entity_ids"] = (
                    scoped_ids if scoped_ids is not None else self._get_active_entity_ids("sops")
                )
                sop_retriever.metadata_filters = section_filters
                if rag_unified_enabled():
                    sop_retriever.category_filter = "sops"
                strict_resp = _strict_sop_inventory_response(
                    [],
                    query_for_routing,
                    sop_retriever,
                    mode=sop_inventory_mode,
                    full_prompt=full_ctx_query,
                )
                strict_resp["retrieval_stats"] = {
                    "searched": ["sops"],
                    "per_section": {"sops": 0},
                    "total_docs": 0,
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                    "strict_mode": True,
                }
                return strict_resp
            strict_resp = _strict_sop_inventory_response(
                [],
                query_for_routing,
                None,
                mode=sop_inventory_mode,
                full_prompt=full_ctx_query,
            )
            strict_resp["retrieval_stats"] = {
                "searched": ["sops"],
                "per_section": {"sops": 0},
                "total_docs": 0,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "strict_mode": True,
            }
            return strict_resp

        # ── Step 2: Hybrid search on targeted collections only ──
        all_docs: List[Document] = []
        per_section_counts: Dict[str, int] = {s: 0 for s in target_sections}
        section_ranked: Dict[str, List[Document]] = {s: [] for s in target_sections}

        t_r0 = time.time()
        pool_sec = float(os.getenv("RAG_PARALLEL_RETRIEVAL_SECONDS", "55"))
        n_sec = len(target_sections)
        logger.info(
            "[rag-retrieval-start] sections=%s parallel_budget_s=%.1f",
            target_sections,
            pool_sec,
        )
        mf_copy = dict(metadata_filters or {})
        if n_sec <= 0:
            pass
        elif n_sec == 1:
            sec = target_sections[0]
            _, ranked, cnt = self._retrieve_ranked_for_section(
                sec, query_for_routing, mf_copy, active_doc_id, n_sec, active_scope, full_ctx_query
            )
            section_ranked[sec] = ranked
            per_section_counts[sec] = cnt
        else:
            with ThreadPoolExecutor(max_workers=min(8, n_sec)) as executor:
                futures = {
                    executor.submit(
                        self._retrieve_ranked_for_section,
                        section,
                        query_for_routing,
                        mf_copy,
                        active_doc_id,
                        n_sec,
                        active_scope,
                        full_ctx_query,
                    ): section
                    for section in target_sections
                }
                pending = set(futures.keys())
                deadline = time.time() + pool_sec
                while pending:
                    wait_s = min(5.0, max(0.05, deadline - time.time()))
                    if wait_s <= 0:
                        break
                    done, pending = wait(pending, timeout=wait_s, return_when=FIRST_COMPLETED)
                    for fut in done:
                        try:
                            section, ranked, cnt = fut.result()
                            section_ranked[section] = ranked
                            per_section_counts[section] = cnt
                        except Exception as ex:
                            sec = futures.get(fut, "?")
                            logger.warning("[rag-retrieval] future failed section=%s: %s", sec, ex)
                for fut in pending:
                    fut.cancel()
                if pending:
                    logger.warning(
                        "[rag-retrieval] parallel budget exhausted; incomplete_sections=%s",
                        [futures[f] for f in pending],
                    )

        for section in target_sections:
            ranked = section_ranked.get(section) or []
            all_docs.extend(ranked)
            if ranked:
                top_preview = [
                    {
                        "ref": str((d.metadata or {}).get("ref_number") or (d.metadata or {}).get("source_id") or ""),
                        "score": round(_json_safe_float((d.metadata or {}).get("rerank_score") if (d.metadata or {}).get("rerank_score") is not None else (d.metadata or {}).get("hybrid_score", 0.0)), 4),
                        "section": str((d.metadata or {}).get("_section") or section),
                    }
                    for d in ranked[:5]
                ]
                logger.info("[rag-retrieval] section=%s top_chunks=%s", section, top_preview)
                _safe_print(f"[rag-retrieval] section={section} top_chunks={top_preview}")
            if RAG_DEBUG_RETRIEVAL and ranked:
                for i, doc in enumerate(ranked[:RAG_DEBUG_MAX_CHUNKS], 1):
                    _safe_print(_debug_chunk_summary(doc, i))

        t_r1 = time.time()
        retrieval_phase_ms = round((t_r1 - t_r0) * 1000.0, 1)
        logger.info(
            "[rag-retrieval-done] sections=%s total_chunks=%s retrieval_ms=%.1f",
            target_sections,
            len(all_docs),
            retrieval_phase_ms,
        )

        editor_ctx_docs = _editor_context_documents_from_prompt(full_ctx_query)
        if editor_ctx_docs:
            all_docs = editor_ctx_docs + list(all_docs or [])
            per_section_counts["editor_context"] = len(editor_ctx_docs)

        if not all_docs:
            logger.info(
                "[rag-refusal] reason=zero_chunks sections=%s filters=%s provider=%s model=%s",
                target_sections,
                metadata_filters,
                llm_cfg.provider,
                llm_cfg.model,
            )
            return {
                "answer": RAG_NO_CONTEXT_REFUSAL,
                "citations": [],
                "suggestions": [
                    "Ask about a specific SOP number",
                    "Search for related deviations",
                    "Check CAPA status",
                ],
                "retrieval_stats": {
                    "searched": target_sections,
                    "total_docs": 0,
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                    "retrieval_phase_ms": retrieval_phase_ms,
                    "llm_provider": llm_cfg.provider,
                    "llm_model": llm_cfg.model,
                    "llm_base_url": llm_cfg.base_url,
                    "failure_stage": "no_retrieval",
                    "elapsed_ms": round((time.time() - t0) * 1000, 1),
                    "refusal_reason": "zero_chunks_after_filters",
                },
                "routed_to": routed_label,
                "refusal_reason": "zero_chunks_after_filters",
                "failure_stage": "no_retrieval",
            }

        if sop_inventory_mode and not RAG_DISABLE_SOP_INVENTORY:
            strict_resp = _strict_sop_inventory_response(
                all_docs,
                query_for_routing,
                self.federated.retrievers.get("sops"),
                mode=sop_inventory_mode,
                full_prompt=full_ctx_query,
            )
            strict_resp["retrieval_stats"] = {
                "searched": target_sections,
                "per_section": per_section_counts,
                "total_docs": len(all_docs),
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "strict_mode": True,
            }
            return strict_resp

        # ── Step 3: Build unified context ──
        context_str, raw_cits = _build_unified_context(all_docs, "document")

        # ── Step 3b: Format chat history for CoT continuity ──
        chat_history_messages = []
        if chat_history:
            for msg in chat_history:
                role = msg.get("role")
                content = msg.get("content", "").strip()
                if role == "assistant":
                    content = _truncate_text(content, MAX_HISTORY_MESSAGE_CHARS)
                    chat_history_messages.append(AIMessage(content=content))
                else:
                    content = _truncate_text(content, MAX_HISTORY_MESSAGE_CHARS)
                    chat_history_messages.append(HumanMessage(content=content))

            if len(chat_history_messages) > MAX_HISTORY_MESSAGES:
                chat_history_messages = chat_history_messages[-MAX_HISTORY_MESSAGES:]

        # ── Step 4: LLM generation ──
        query = _truncate_text(query, MAX_QUERY_CHARS)
        context_str = _truncate_text(context_str, MAX_CONTEXT_CHARS)
        logger.info(
            "[rag-prompt] sections=%s context_chars=%s context_preview=%s",
            target_sections,
            len(context_str),
            _console_safe(context_str[:600].replace("\n", " ")),
        )
        _safe_print(
            f"[rag-prompt] sections={target_sections} context_chars={len(context_str)} preview={context_str[:350].replace(chr(10), ' ')}"
        )
        llm_source = "primary"
        t_llm0 = time.time()
        raw_answer: str | None = None
        llm_exc: Exception | None = None
        history_focus = (
            f"HISTORY FOCUS: Priority should be given to {active_doc_id} as it was discussed recently."
            if active_doc_id
            else ""
        )
        logger.info(
            "[rag-llm-start] model=%s context_chars=%s history_msgs=%s",
            llm_cfg.model,
            len(context_str),
            len(chat_history_messages),
        )
        try:
            raw_answer = (self.prompt | self.llm | StrOutputParser()).invoke({
                "context": context_str,
                "question": query,
                "chat_history_messages": chat_history_messages,
                "history_focus": history_focus,
            })
        except Exception as e:
            llm_exc = e
            err = str(e).lower()
            if "503" in err or "unavailable" in err or "high demand" in err:
                try:
                    fallback_llm = get_fallback_llm()
                    llm_source = "fallback"
                    raw_answer = (self.prompt | fallback_llm | StrOutputParser()).invoke({
                        "context": context_str,
                        "question": query,
                        "chat_history_messages": chat_history_messages,
                        "history_focus": history_focus,
                    })
                    llm_exc = None
                except Exception as e2:
                    llm_exc = e2
            else:
                pass

        llm_phase_ms = round((time.time() - t_llm0) * 1000.0, 1)

        if raw_answer is None and llm_exc is not None:
            stage = classify_llm_exception(llm_exc)
            logger.error(
                "[rag-llm-failed] stage=%s llm_ms=%.1f err=%s",
                stage,
                llm_phase_ms,
                llm_exc,
            )
            excerpt_cits = _sanitize_citation_list(list(raw_cits))
            pieces: List[str] = []
            for c in excerpt_cits[:10]:
                if not isinstance(c, dict):
                    continue
                ref = str(c.get("ref") or "").strip()
                ex = str(c.get("excerpt") or "").strip()
                title = str(c.get("title") or "").strip()
                body = ex or title
                if body:
                    pieces.append(f"{ref}: {body}" if ref else body)
            answer_fb = RAG_NO_CONTEXT_REFUSAL
            latency_ms = round((time.time() - t0) * 1000.0, 1)
            return {
                "answer": answer_fb,
                "reasoning": "",
                "confidence": "LOW",
                "citations": excerpt_cits,
                "retrieval_debug": _build_retrieval_debug_rows(all_docs),
                "suggestions": [
                    "Retry after confirming the local model is loaded in LM Studio",
                    "Check that LOCAL_LLM_MODEL matches a model id from GET /v1/models",
                    "Try a shorter question or reduce RAG context size",
                ],
                "retrieval_stats": {
                    "searched": target_sections,
                    "per_section": per_section_counts,
                    "total_docs": len(all_docs),
                    "latency_ms": latency_ms,
                    "retrieval_phase_ms": retrieval_phase_ms,
                    "llm_phase_ms": llm_phase_ms,
                    "llm_provider": llm_cfg.provider,
                    "llm_model": llm_cfg.model,
                    "llm_base_url": llm_cfg.base_url,
                    "llm_source": llm_source,
                    "failure_stage": stage,
                    "elapsed_ms": latency_ms,
                },
                "routed_to": routed_label,
                "cached": False,
                "metadata_snapshot": [],
                "audit_log_snapshot": [],
                "llm_error": str(llm_exc),
                "failure_stage": stage,
                "action_metadata": {
                    "query": query,
                    "routing": target_sections,
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                    "model": llm_cfg.model,
                    "llm_provider": llm_cfg.provider,
                    "llm_source": llm_source,
                    "failure_stage": stage,
                },
            }

        logger.info("[rag-llm-done] llm_ms=%.1f source=%s", llm_phase_ms, llm_source)

        # ── Step 5: Parse answer, citations, suggestions, reasoning, confidence ──
        answer, llm_citations, suggestions, reasoning, confidence = _parse_answer_citations_suggestions(
            raw_answer or ""
        )

        # Merge LLM-parsed citations with raw retrieval metadata for richer response
        final_citations = []
        used_refs = set()
        for lc in llm_citations:
            ref = lc.get("ref", "")
            # Try to enrich from raw_cits
            match = next((r for r in raw_cits if ref in r.get("ref", "") or (r.get("title") and r["title"] in lc.get("title", ""))), None)
            entry = {
                "ref":     ref,
                "title":   lc.get("title", match.get("title","") if match else ""),
                "type":    lc.get("type", match.get("type","") if match else ""),
                "excerpt": lc.get("excerpt", ""),
                "status":  match.get("status","") if match else "",
                "score":   _json_safe_float(
                    (match.get("score", 0.0) if match else 0.0)
                ),
            }
            if ref not in used_refs:
                final_citations.append(entry)
                used_refs.add(ref)

        # Fall back to raw citations if LLM did not produce any
        if not final_citations:
            final_citations = raw_cits
        final_citations = _sanitize_citation_list(final_citations)
        final_citations = _dedupe_citations_by_ref(final_citations)
        if len(final_citations) > 12:
            final_citations = sorted(
                final_citations,
                key=lambda c: _json_safe_float((c or {}).get("score", 0.0)),
                reverse=True,
            )[:12]
        if RAG_DEBUG_RETRIEVAL:
            cited_refs = [str(c.get("ref", "")).strip() for c in final_citations if isinstance(c, dict)]
            _safe_print(
                f"[rag-debug] final_citations={len(cited_refs)} refs={cited_refs[:RAG_DEBUG_MAX_CHUNKS]}"
            )

        if not (answer or "").strip() and final_citations:
            pieces: List[str] = []
            for c in final_citations[:8]:
                if not isinstance(c, dict):
                    continue
                ref = str(c.get("ref") or "").strip()
                ex = str(c.get("excerpt") or "").strip()
                title = str(c.get("title") or "").strip()
                body = ex or title
                if body:
                    pieces.append(f"{ref}: {body}" if ref else body)
            if pieces:
                answer = "\n\n".join(pieces)
                logger.warning(
                    "[rag-answer] empty LLM [ANSWER]; using excerpt-aligned fallback lines=%s",
                    len(pieces),
                )

        # ── Step 6: Assemble full Audit Vault snapshots ──

        metadata_snapshot = []
        audit_log_snapshot = []
        
        seen_docs = set()
        for doc in all_docs:
            source_id = doc.metadata.get("source_id")
            if source_id not in seen_docs:
                metadata_snapshot.append(doc.metadata.get("full_metadata", doc.metadata))
                audit_log_snapshot.extend(doc.metadata.get("audit_trail", []))
                seen_docs.add(source_id)

        latency_ms = round((time.time() - t0) * 1000, 1)
        logger.info(
            "[rag-answer] model=%s provider=%s llm_source=%s routed_to=%s chunks=%s citations=%s latency_ms=%s retrieval_ms=%s llm_ms=%s",
            llm_cfg.model,
            llm_cfg.provider,
            llm_source,
            routed_label,
            len(all_docs),
            len(final_citations),
            latency_ms,
            retrieval_phase_ms,
            llm_phase_ms,
        )

        return {
            "answer":      answer,
            "reasoning":   reasoning,
            "confidence":  confidence,
            "citations":   final_citations,
            "retrieval_debug": _build_retrieval_debug_rows(all_docs),
            "suggestions": suggestions,
            "retrieval_stats": {
                "searched":     target_sections,
                "per_section":  per_section_counts,
                "total_docs":   len(all_docs),
                "latency_ms":   latency_ms,
                "elapsed_ms":   latency_ms,
                "retrieval_phase_ms": retrieval_phase_ms,
                "llm_phase_ms":       llm_phase_ms,
                "llm_provider": llm_cfg.provider,
                "llm_model":    llm_cfg.model,
                "llm_base_url": llm_cfg.base_url,
                "llm_source":   llm_source,
                "failure_stage": None,
            },
            "routed_to":   routed_label,
            "cached":      False,
            # Audit Vault Fields
            "metadata_snapshot":  metadata_snapshot,
            "audit_log_snapshot": audit_log_snapshot,
            "action_metadata": {
                "query": query,
                "routing": target_sections,
                "latency_ms": latency_ms,
                "timestamp": time.time(),
                "model": llm_cfg.model,
                "llm_provider": llm_cfg.provider,
                "llm_source": llm_source,
            }
        }


# Keep FederatedRAGChain as alias for backward compat
FederatedRAGChain = SmartRAGChain
