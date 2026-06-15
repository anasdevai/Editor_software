"""Controlled SOP agent orchestration.

DeepAgents is a required orchestration layer for agentic SOP synthesis. The
deterministic database/RAG/profile/editor-targeting tools remain the authority
for sensitive state and ranges; DeepAgents plans, delegates, and drafts.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from ..models import ClientProfile, KnowledgeChunk, SOP, SOPDetectedParameters, SOPGenerationTemplate, SOPVersion
from ..utils.tiptap_text import extract_plain_text_from_tiptap
from chatbot.llm.provider import create_chat_llm, get_local_llm_config
from .semantic_jobs import schedule_semantic_reindex
from .sop_profile_storage_service import analyze_and_store_sop_profile

from deepagents import create_deep_agent

DEEPAGENTS_AVAILABLE = True


DEFAULT_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")
ENABLE_REAL_DEEP_AGENTS = True
ACTION_SET = {
    "rewrite",
    "improve",
    "summarize",
    "explain",
    "gap_check",
    "compliance",
    "rewrite_with_profile_style",
    "improve_with_profile_style",
}


class DeepAgentExecutionError(RuntimeError):
    """Raised when the mandatory DeepAgents orchestration path cannot complete."""


@dataclass
class LoadedSOP:
    sop: SOP
    version: SOPVersion | None
    text: str
    headings: list[str]
    profile: ClientProfile | None
    detected: SOPDetectedParameters | None
    chunks: list[KnowledgeChunk]


def _agent_mode() -> str:
    return "deep_agent"


def _deep_agent_enabled() -> bool:
    return True


def _message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    return str(content or "").strip()


def _deep_agent_text(result: Any) -> str:
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            return _message_content(messages[-1])
        for key in ("output", "content", "text"):
            if result.get(key):
                return _message_content(result[key])
    return _message_content(result)


def _build_sop_deep_agent(
    *,
    db: Session,
    target_text: str,
    user_request: str,
    action: str,
    active_sop_id: str | None,
    source_sop_ids: list[str] | None,
    profile: ClientProfile | None,
    evidence: dict[str, Any],
    style_profile_only: bool = False,
) -> Any:
    source_ids = source_sop_ids or ([active_sop_id] if active_sop_id else [])
    is_full_sop_rewrite = bool(
        action in {"rewrite", "improve"}
        and re.search(r"\b(full|whole|entire|complete)\s+(?:sop|document|doc)\b|\brewrite\s+(?:the\s+)?(?:current|open|active)?\s*sop\b", user_request or "", re.I)
    )

    def inspect_resolved_target() -> dict[str, Any]:
        """Inspect the already-resolved editor target text. Never choose editor positions."""
        return {
            "action": action,
            "user_request": user_request,
            "target_text": target_text[:8000],
            "target_word_count": len((target_text or "").split()),
            "full_sop_rewrite_requested": is_full_sop_rewrite,
            "targeting_rule": "Backend resolver already chose this range. Use only this text for rewrite/improve facts.",
            "full_sop_output_rule": (
                "When full_sop_rewrite_requested is true, return a complete SOP-shaped replacement: title/metadata if present, "
                "numbered headings, rewritten section bodies, tables/registers/appendices preserved, and missing backbone sections "
                "with bracketed placeholders. Do not return a report or summary."
            ),
        }

    def retrieve_relevant_sop_chunks(query: str, limit: int = 6) -> dict[str, Any]:
        """Retrieve relevant indexed SOP chunks only, filtered to selected source SOP IDs."""
        return retrieve_rag_evidence(db, query or user_request or target_text, source_ids, limit=max(1, min(10, int(limit or 6))))

    def inspect_style_profile() -> dict[str, Any]:
        """Return the active or style-source profile summary. Use as style only when requested."""
        profile_json = _profile_json(profile)
        return {
            "profile_id": str(profile.id) if profile else None,
            "profile_name": profile.name if profile else None,
            "profile_md": (profile.active_profile_md or "")[:8000] if profile else "",
            "profile_md_used": bool(profile and (profile.active_profile_md or "").strip()),
            "profile_parameter_keys": sorted(profile_json.keys()) if isinstance(profile_json, dict) else [],
            "preferred_style": profile_json.get("preferred_style") or {},
            "terminology": profile_json.get("terminology") or {},
            "modal_language": profile_json.get("modal_language") or {},
            "rewrite_rules": profile_json.get("rewrite_rules") or [],
            "structure_patterns": profile_json.get("structure_patterns") or {},
            "workflow_patterns": profile_json.get("workflow_patterns") or {},
            "rewrite_improve_parameters": profile_json.get("rewrite_improve_parameters") or {},
            "style_profile_only": style_profile_only,
            "style_only_guardrail": "When style_profile_only is true, use tone/format/terminology preferences only. Do not import facts.",
        }

    system_prompt = """You are the real LangChain Deep Agent for a regulated SOP assistant.

Use tools before answering:
- inspect_resolved_target gives the only editor text you may rewrite/improve.
- retrieve_relevant_sop_chunks gives RAG evidence only when the action needs evidence.
- inspect_style_profile gives style/profile guidance.

Critical guardrails:
- Never decide editor positions; the backend resolver already did that.
- Rewrite/improve may change wording only inside the resolved target.
- Cross-profile style is style-only. Never copy facts, IDs, dates, roles, standards, controls, or requirements from the profile.
- profile.md is mandatory style/parameter context when available. Apply its terminology, modal verbs, structure patterns, workflow patterns, rewrite_rules, and rewrite_improve_parameters.
- For normal rewrite, use the active SOP profile.md/profile JSON. For cross-profile rewrite, use the selected profile.md as STYLE ONLY and use facts only from inspect_resolved_target.target_text.
- For "rewrite the full SOP", the result must look like a complete SOP document, not a chat answer/report: title/metadata if present, all headings and bodies in order, tables/registers preserved, missing backbone sections with bracketed placeholders.
- Summaries/explanations/gap checks must distinguish evidence from assumptions.
- If evidence is weak, say evidence is weak instead of inventing facts.
- If the user asks for a shorter result or exact line count, obey it.

Return only the final user-facing result text. Do not include markdown code fences."""

    subagents = [
        {
            "name": "rag-evidence-agent",
            "description": "Retrieves and summarizes relevant SOP chunks with citations for gap, compliance, explain, and generation tasks.",
            "system_prompt": "Use retrieve_relevant_sop_chunks. Return concise evidence, citations, and weak-evidence warnings.",
            "tools": [retrieve_relevant_sop_chunks],
        },
        {
            "name": "rewrite-improve-agent",
            "description": "Rewrites or improves the already-resolved SOP target while preserving facts and respecting style/length constraints.",
            "system_prompt": "Use inspect_resolved_target and inspect_style_profile. Preserve facts and obey length/style constraints.",
            "tools": [inspect_resolved_target, inspect_style_profile],
        },
        {
            "name": "compliance-agent",
            "description": "Performs compliance gap review on the resolved SOP target using relevant RAG evidence.",
            "system_prompt": "Use inspect_resolved_target and retrieve_relevant_sop_chunks. Separate confirmed gaps, assumptions, and recommended wording.",
            "tools": [inspect_resolved_target, retrieve_relevant_sop_chunks],
        },
    ]

    model = create_chat_llm(temperature=0.1, max_output_tokens=1800, max_retries=0, use_cache=True)
    return create_deep_agent(
        model=model,
        tools=[inspect_resolved_target, retrieve_relevant_sop_chunks, inspect_style_profile],
        system_prompt=system_prompt,
        subagents=subagents,
        name="sop-action-agent",
    )


def _invoke_sop_deep_agent(
    *,
    db: Session,
    action: str,
    user_request: str,
    target_text: str,
    active_sop_id: str | None,
    source_sop_ids: list[str] | None,
    profile: ClientProfile | None,
    evidence: dict[str, Any],
    constraints: dict[str, Any],
    style_profile_only: bool = False,
) -> dict[str, Any]:
    agent = _build_sop_deep_agent(
        db=db,
        target_text=target_text,
        user_request=user_request,
        action=action,
        active_sop_id=active_sop_id,
        source_sop_ids=source_sop_ids,
        profile=profile,
        style_profile_only=style_profile_only,
        evidence=evidence,
    )
    cfg = get_local_llm_config()
    prompt = {
        "action": action,
        "user_request": user_request,
        "constraints": constraints,
        "target_text": target_text[:8000],
        "profile_context": {
            "profile_id": str(profile.id) if profile else None,
            "profile_name": profile.name if profile else None,
            "profile_md_used": bool(profile and (profile.active_profile_md or "").strip()),
            "profile_md_excerpt": (profile.active_profile_md or "")[:4000] if profile else "",
            "profile_parameters": {
                "preferred_style": (_profile_json(profile).get("preferred_style") or {}) if profile else {},
                "terminology": (_profile_json(profile).get("terminology") or {}) if profile else {},
                "modal_language": (_profile_json(profile).get("modal_language") or {}) if profile else {},
                "rewrite_rules": (_profile_json(profile).get("rewrite_rules") or []) if profile else [],
                "structure_patterns": (_profile_json(profile).get("structure_patterns") or {}) if profile else {},
                "workflow_patterns": (_profile_json(profile).get("workflow_patterns") or {}) if profile else {},
                "rewrite_improve_parameters": (_profile_json(profile).get("rewrite_improve_parameters") or {}) if profile else {},
            },
            "cross_profile_style_only": style_profile_only,
            "fact_guardrail": "Use facts only from target_text/current SOP. Profile facts are forbidden unless present in target_text.",
        },
        "rag_evidence_snapshot": {
            "chunk_count": len(evidence.get("chunks") or []),
            "weak_evidence": bool(evidence.get("weak_evidence")),
            "citations": evidence.get("citations") or [],
        },
    }
    result = agent.invoke({"messages": [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]})
    text = _deep_agent_text(result)
    if not text:
        raise DeepAgentExecutionError(f"DeepAgents returned an empty response for action={action} model={cfg.model}")
    return {"used": True, "text": text, "model": cfg.model, "raw_type": type(result).__name__}


def _compact_sidebar_context(assistant_context: dict[str, Any] | None) -> dict[str, Any]:
    ctx = assistant_context if isinstance(assistant_context, dict) else {}
    current_sop = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}
    selected_section = ctx.get("selected_section") if isinstance(ctx.get("selected_section"), dict) else {}
    last_action = ctx.get("last_action") if isinstance(ctx.get("last_action"), dict) else {}
    active_scope = ctx.get("active_scope") if isinstance(ctx.get("active_scope"), dict) else {}
    profile = ctx.get("nlp_profile") if isinstance(ctx.get("nlp_profile"), dict) else {}
    sections = current_sop.get("sections") if isinstance(current_sop.get("sections"), list) else []
    return {
        "active_sop": {
            "id": current_sop.get("id") or ctx.get("active_sop_id") or ctx.get("current_document_id"),
            "sop_number": current_sop.get("sop_number") or current_sop.get("documentId"),
            "title": current_sop.get("title"),
            "status": current_sop.get("status") or (current_sop.get("metadata") or {}).get("status") if isinstance(current_sop.get("metadata"), dict) else None,
            "section_count": len(sections),
            "headings": [
                str(s.get("label") or s.get("name") or "").strip()
                for s in sections[:30]
                if isinstance(s, dict) and str(s.get("label") or s.get("name") or "").strip()
            ],
        },
        "selected_section": {
            "label": selected_section.get("label") or selected_section.get("name"),
            "content": str(selected_section.get("content") or "")[:5000],
        },
        "active_scope": active_scope,
        "last_action": last_action,
        "profile_summary": {
            "domain": profile.get("domain") or profile.get("detected_domain"),
            "department": profile.get("department") or profile.get("detected_department"),
            "style": profile.get("preferred_style") or profile.get("style_profile"),
            "terminology": profile.get("terminology"),
        },
        "live_excerpt": str(
            current_sop.get("full_text")
            or current_sop.get("plain_text")
            or ctx.get("editor_excerpt")
            or selected_section.get("content")
            or ""
        )[:9000],
    }


SIDEBAR_ORCHESTRATOR_REFUSAL_EN = (
    "I can only answer questions related to the current SOP, uploaded SOPs, SOP metadata, "
    "linked QMS records, or retrieved RAG context."
)
SIDEBAR_ORCHESTRATOR_REFUSAL_DE = (
    "Ich kann nur Fragen zum aktuellen SOP, hochgeladenen SOPs, SOP-Metadaten, "
    "verknuepften QMS-Datensaetzen oder abgerufenem RAG-Kontext beantworten."
)


_SIDEBAR_SOP_ALLOWED_RE = re.compile(
    r"\b("
    r"sop|sops|standard operating procedure|document|dokument|procedure|verfahren|"
    r"section|abschnitt|paragraph|heading|purpose|scope|zweck|geltungsbereich|"
    r"rewrite|re-?write|improve|summari[sz]e|explain|gap\s*check|compliance|"
    r"gmp|qms|qa|quality|audit|audits|capa|capas|deviation|deviations|decision|decisions|"
    r"risk|control|approval|revision|version|metadata|profile|template|traceability|"
    r"rag|retrieved context|retrieval|source|citation|chunk|current|selected|open"
    r")\b",
    re.IGNORECASE,
)

_SIDEBAR_GENERAL_KNOWLEDGE_RE = re.compile(
    r"\b("
    r"what\s+is\s+(?:an?\s+)?(?:llm|ai|rag|chatgpt|python|javascript|react|fastapi|sql|database)|"
    r"define\s+(?:llm|ai|rag|chatgpt|python|javascript|react|fastapi|sql|database)|"
    r"who\s+is\s+[A-Za-z][\w-]*|"
    r"tell\s+me\s+about\s+[A-Za-z][\w-]*|"
    r"weather|sports|news|joke|recipe|movie|capital\s+of|translate\s+this\s+sentence"
    r")\b",
    re.IGNORECASE,
)

_SOP_REF_RE = re.compile(r"\bSOP-[A-Z0-9-]+\b", re.IGNORECASE)
_QMS_REF_RE = re.compile(r"\b(?:DEV|CAPA|AUDIT|DEC)-[A-Z0-9-]+\b", re.IGNORECASE)


def _is_german_sidebar_question(text: str) -> bool:
    return bool(
        re.search(
            r"\b(ich|du|sie|der|die|das|was|wie|warum|bitte|aktuell|hochgeladenen|verknuepft|verknüpft)\b",
            text or "",
            re.IGNORECASE,
        )
    )


def _sidebar_orchestrator_scope_decision(
    question: str,
    assistant_context: dict[str, Any] | None,
    intents: list[str] | None,
    action_plan: dict[str, Any] | None,
) -> tuple[bool, str]:
    """First orchestrator gate: sidebar answers must stay inside SOP/QMS/RAG/editor context."""
    q = re.sub(r"\s+", " ", str(question or "").strip())
    if not q:
        return False, "empty_question"

    if isinstance(action_plan, dict) and action_plan:
        return True, "planned_sop_action"

    ctx = assistant_context if isinstance(assistant_context, dict) else {}
    compact = _compact_sidebar_context(ctx)
    active = compact.get("active_sop") if isinstance(compact.get("active_sop"), dict) else {}
    selected = compact.get("selected_section") if isinstance(compact.get("selected_section"), dict) else {}
    has_current_sop = bool(active.get("id") or active.get("sop_number") or active.get("title"))
    has_selection = bool(selected.get("label") or selected.get("content"))
    intent_set = {str(x) for x in (intents or [])}

    if _SOP_REF_RE.search(q) or _QMS_REF_RE.search(q):
        return True, "record_reference"
    if intent_set & {"sop_count", "sop_list", "summary", "explain", "compliance", "compare", "linked", "active_sop", "metadata"}:
        return True, "classified_sop_intent"
    if _SIDEBAR_SOP_ALLOWED_RE.search(q):
        if _SIDEBAR_GENERAL_KNOWLEDGE_RE.search(q) and not re.search(
            r"\b(current|open|active|this|selected|uploaded|sop|document|chunk|source|citation|retrieved|context|pipeline)\b",
            q,
            re.IGNORECASE,
        ):
            return False, "general_knowledge_keyword"
        return True, "sop_keyword"
    if has_current_sop and re.search(r"\b(this|that|it|current|selected|open|here|above|same)\b", q, re.IGNORECASE):
        return True, "contextual_followup"
    if has_selection and re.search(r"\b(explain|summari[sz]e|rewrite|improve|shorter|longer|formal|better|fix)\b", q, re.IGNORECASE):
        return True, "selection_followup"
    if _SIDEBAR_GENERAL_KNOWLEDGE_RE.search(q):
        return False, "general_knowledge"
    return False, "no_sop_or_rag_signal"


def _sidebar_orchestrator_refusal(
    question: str,
    *,
    surface: str,
    assistant_mode: str,
    category: str | None,
    reason: str,
) -> dict[str, Any]:
    answer = SIDEBAR_ORCHESTRATOR_REFUSAL_DE if _is_german_sidebar_question(question) else SIDEBAR_ORCHESTRATOR_REFUSAL_EN
    return {
        "answer": answer,
        "sources": [],
        "citations": [],
        "retrieval_debug": [],
        "suggestions": [
            "Ask about the current SOP",
            "Summarize the selected section",
            "Run a gap check on this SOP",
        ],
        "retrieval_stats": {
            "total_docs": 0,
            "source": "sidebar_orchestrator_scope_guard",
            "surface": surface,
            "assistant_mode": assistant_mode,
            "category": category,
            "agent_mode": "deep_agent",
            "strict_mode": "sop_context_only",
            "refusal_reason": reason,
        },
        "routed_to": "sidebar_orchestrator_scope_guard",
        "assistant_action": None,
        "refusal_reason": reason,
        "orchestrator_metadata": {
            "agent_mode": "deep_agent",
            "subagent_used": None,
            "guardrail": "sop_context_only",
            "blocked_before_llm": True,
            "rag": {"used": False, "chunk_count": 0, "citations": []},
            "run_editor_action": False,
            "run_query": False,
            "requires_approval": False,
        },
    }


def _format_sidebar_answer(text: str) -> str:
    """Normalize DeepAgent prose into readable sidebar text."""
    s = str(text or "").strip()
    if not s:
        return ""

    try:
        from chatbot.rag.rag_chain import sanitize_user_facing_answer

        s = sanitize_user_facing_answer(s)
    except Exception:
        pass

    # Convert inline markdown labels like "... text. **Scope:** More text" into
    # separate sidebar sections. This fixes compact LLM output without relying on UI rendering.
    s = re.sub(r"\s*\*\*([^*\n:]{2,80}):\*\*\s*", r"\n\n\1:\n", s)
    s = re.sub(r"\s*\*\*([^*\n]{2,80})\*\*\s*:\s*", r"\n\n\1:\n", s)
    s = re.sub(r"\s*\*\*([^*\n]{2,80})\*\*\s*", r"\n\n\1\n", s)

    # If a known label still appears inline after a sentence, split it onto its own line.
    labels = (
        "Overall Purpose",
        "Purpose",
        "Scope & Responsibilities",
        "Scope",
        "Responsibilities",
        "Core Workflows",
        "Vendor Oversight",
        "Training Compliance",
        "Summary",
        "Details",
        "Status",
        "Sources",
        "Gaps",
        "Recommendations",
        "Suggested SOP Text",
    )
    label_alt = "|".join(re.escape(label) for label in labels)
    s = re.sub(rf"(?<!^)(?<!\n)\s+({label_alt}):\s*", r"\n\n\1:\n", s)

    # Make dense workflow lines easier to scan.
    s = re.sub(r"(?m)^(Vendor Oversight|Training Compliance|Recommendations|Gaps):\s*(.+)", r"\1:\n- \2", s)
    s = re.sub(r"\n-\s+", "\n- ", s)

    lines: list[str] = []
    for raw in s.splitlines():
        line = raw.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9 /&()-]{1,80}:", line):
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(line)
            continue
        lines.append(line)

    s = "\n".join(lines)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def _build_sidebar_deep_agent(
    *,
    db: Session,
    question: str,
    assistant_context: dict[str, Any] | None,
    source_sop_ids: list[str] | None = None,
) -> Any:
    compact_context = _compact_sidebar_context(assistant_context)
    active_id = str((compact_context.get("active_sop") or {}).get("id") or "").strip()
    source_ids = source_sop_ids or ([active_id] if active_id else [])

    def inspect_sidebar_context() -> dict[str, Any]:
        """Inspect active SOP, selected section, active scope, metadata, and profile context."""
        return compact_context

    def retrieve_sidebar_rag_chunks(query: str, limit: int = 6) -> dict[str, Any]:
        """Retrieve relevant SOP chunks for the sidebar question. Uses active SOP filters when available."""
        return retrieve_rag_evidence(db, query or question, source_ids, limit=max(1, min(10, int(limit or 6))))

    def inspect_sidebar_target() -> dict[str, Any]:
        """Return the current selected/active target context. Do not choose editor positions."""
        selected = compact_context.get("selected_section") or {}
        active_scope = compact_context.get("active_scope") or {}
        return {
            "selected_section_label": selected.get("label"),
            "selected_section_text": selected.get("content"),
            "active_scope": active_scope,
            "targeting_rule": "Frontend/editor resolver owns exact ranges. Sidebar agent may explain or draft, not write positions.",
        }

    def resolve_editor_target(target_query: str, action: str = "rewrite") -> dict[str, Any]:
        """
        Dynamically resolve which section, table, or paragraph in the active editor
        the user is referring to based on a target description or query.
        For example: resolve_editor_target("procedure section") or resolve_editor_target("document history table").
        Returns the target's ID, type, label, parent section, confidence, and whether it requires clarification.
        """
        from .target_resolver_agent import resolve_sop_target_with_deep_agent
        ctx = assistant_context if isinstance(assistant_context, dict) else {}
        current_sop = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}
        editor_contract = ctx.get("editor_context_contract") if isinstance(ctx.get("editor_context_contract"), dict) else {}
        
        # Extract indexes with safety fallbacks
        sections = current_sop.get("sections")
        if not isinstance(sections, list):
            sop_ctx = editor_contract.get("sop_context") if isinstance(editor_contract.get("sop_context"), dict) else {}
            sections = sop_ctx.get("sections") if isinstance(sop_ctx.get("sections"), list) else []
            
        tables = ctx.get("tables") or editor_contract.get("tables") or []
        if not isinstance(tables, list):
            tables = []
            
        paragraphs = ctx.get("paragraphs") or editor_contract.get("paragraphs") or []
        if not isinstance(paragraphs, list):
            paragraphs = []
            
        document_tree = ctx.get("document_tree") or editor_contract.get("document_tree") or []
        if not isinstance(document_tree, list):
            document_tree = []
            
        selection = ctx.get("selected_section") or ctx.get("selection") or {}
        if not isinstance(selection, dict):
            selection = {}
            
        active_scope = ctx.get("active_scope") or {}
        if not isinstance(active_scope, dict):
            active_scope = {}
            
        sop_metadata = current_sop.get("metadata") or {}
        if not isinstance(sop_metadata, dict):
            sop_metadata = {}
            
        document_excerpt = ctx.get("editor_excerpt") or ""
        full_text = current_sop.get("full_text") or ""
        
        return resolve_sop_target_with_deep_agent(
            user_query=target_query,
            action=action,
            sections=sections,
            tables=tables,
            selection=selection,
            active_scope=active_scope,
            sop_metadata=sop_metadata,
            document_excerpt=document_excerpt,
            paragraphs=paragraphs,
            document_tree=document_tree,
            document_schema="",
            full_text=full_text,
        )

    system_prompt = """You are the real LangChain Deep Agent for the SOP sidebar chat.

You are allowed to answer sidebar questions and prepare safe action guidance. Use tools first:
- inspect_sidebar_context for active SOP metadata, selected section, profile, and live editor excerpt.
- inspect_sidebar_target for selected/active section context.
- retrieve_sidebar_rag_chunks for evidence-backed RAG, compliance, gap, and cross-SOP questions.
- resolve_editor_target to dynamically find the correct section or table when the user specifies one in their query.

Rules:
- Do not invent SOP facts. If evidence/context is weak, say so.
- For RAG/compliance/gap answers, cite source chunk labels when available.
- For rewrite/improve requests from sidebar, explain the resolved target/action and provide draft text only; do not claim DB/editor write.
- Preserve current SOP facts. Any profile is style-only unless the user asks to compare/generate from multiple SOPs.
- Keep responses concise and usable in the sidebar.
- Format sidebar answers with short paragraphs and clear section labels on their own lines.
- Do not put section labels inline inside paragraphs. Bad: "text. Overall Purpose: more text". Good: "text.\n\nOverall Purpose:\nmore text".

Return final sidebar answer only. No markdown code fences."""

    subagents = [
        {
            "name": "sidebar-rag-agent",
            "description": "Handles sidebar RAG, evidence, compliance, and gap questions with citations.",
            "system_prompt": "Use retrieve_sidebar_rag_chunks and inspect_sidebar_context. Return grounded evidence and weak-evidence warnings.",
            "tools": [retrieve_sidebar_rag_chunks, inspect_sidebar_context],
        },
        {
            "name": "sidebar-target-agent",
            "description": "Handles questions about selected section, active scope, rewrite/improve targeting, and follow-up context.",
            "system_prompt": "Use inspect_sidebar_target, inspect_sidebar_context, and resolve_editor_target to find the exact target section or table ID when the user refers to one. Never choose editor positions; describe target safely.",
            "tools": [inspect_sidebar_target, inspect_sidebar_context, resolve_editor_target],
        },
        {
            "name": "sidebar-profile-agent",
            "description": "Handles profile/style questions and cross-profile style-only rewrite guidance.",
            "system_prompt": "Use inspect_sidebar_context. Treat external profiles as style-only and preserve current SOP facts.",
            "tools": [inspect_sidebar_context],
        },
    ]

    model = create_chat_llm(temperature=0.1, max_output_tokens=1800, max_retries=0, use_cache=True)
    return create_deep_agent(
        model=model,
        tools=[inspect_sidebar_context, retrieve_sidebar_rag_chunks, inspect_sidebar_target, resolve_editor_target],
        system_prompt=system_prompt,
        subagents=subagents,
        name="sop-sidebar-agent",
    )


def run_sidebar_deep_agent(
    *,
    db: Session,
    question: str,
    assistant_context: dict[str, Any] | None,
    surface: str = "",
    assistant_mode: str = "action",
    category: str | None = None,
    chat_history: list[dict[str, Any]] | None = None,
    intents: list[str] | None = None,
    action_plan: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run the mandatory DeepAgents sidebar path unless a DB mutation action owns the request."""
    if isinstance(action_plan, dict) and action_plan.get("type") in {"delete_sop", "create_sop", "update_sop"}:
        return None
    in_scope, scope_reason = _sidebar_orchestrator_scope_decision(
        question,
        assistant_context,
        intents,
        action_plan,
    )
    if not in_scope:
        return _sidebar_orchestrator_refusal(
            question,
            surface=surface,
            assistant_mode=assistant_mode,
            category=category,
            reason=scope_reason,
        )
    agent = _build_sidebar_deep_agent(db=db, question=question, assistant_context=assistant_context)
    cfg = get_local_llm_config()
    context = _compact_sidebar_context(assistant_context)
    prompt = {
        "question": question,
        "surface": surface,
        "assistant_mode": assistant_mode,
        "category": category,
        "intents": intents or [],
        "action_plan": action_plan or {},
        "chat_history_tail": (chat_history or [])[-6:],
        "context_snapshot": context,
    }
    result = agent.invoke({"messages": [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]})
    answer = _format_sidebar_answer(_deep_agent_text(result))
    if not answer:
        raise DeepAgentExecutionError(f"DeepAgents returned an empty sidebar response model={cfg.model}")
    active = context.get("active_sop") or {}
    citations: list[dict[str, Any]] = []
    if active.get("sop_number") or active.get("title"):
        citations.append(
            {
                "ref": active.get("sop_number") or active.get("id") or "active_sop",
                "title": active.get("title") or active.get("sop_number") or "Active SOP",
                "type": "live_sop_context",
                "excerpt": "Active SOP context and selected section supplied by sidebar/editor.",
                "score": 1.0,
            }
        )
    sources = [
        {
            "id": c.get("ref") or f"source-{idx+1}",
            "type": c.get("type") or "live_sop_context",
            "label": c.get("title") or c.get("ref") or "Active SOP context",
        }
        for idx, c in enumerate(citations)
    ]
    return {
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "retrieval_debug": [],
        "suggestions": [],
        "retrieval_stats": {
            "total_docs": len(citations),
            "source": "sidebar_deep_agent",
            "surface": surface,
            "assistant_mode": assistant_mode,
            "category": category,
            "provider": "local_openai",
            "model": cfg.model,
            "agent_mode": "deep_agent",
            "deepagents_available": DEEPAGENTS_AVAILABLE,
            "subagents": ["sidebar-rag-agent", "sidebar-target-agent", "sidebar-profile-agent"],
        },
        "routed_to": "sidebar_deep_agent",
        "assistant_action": action_plan,
        "orchestrator_metadata": {
            "agent_mode": "deep_agent",
            "subagent_used": "sop-sidebar-agent",
            "rag": {"used": True, "chunk_count": None, "citations": citations},
            "run_editor_action": False,
            "run_query": True,
            "requires_approval": False,
        },
    }


def _extract_action_constraints(user_request: str, source_text: str = "") -> dict[str, Any]:
    request = str(user_request or "")
    constraints: dict[str, Any] = {}
    line_patterns = [
        r"\b(?:in|into|within|using|with|as|to)\s+(\d{1,3})\s*(?:lines?|zeilen?|line\s+limit)\b",
        r"\b(\d{1,3})\s*[-\s]*(?:line|lines|zeilen)(?:\s+limit)?\b",
        r"\b(?:rewrite|improve|explain|describe|summarize)\w*[^.?\n]{0,100}?\b(\d{1,3})\s*(?:lines?|zeilen?)\b",
    ]
    for pattern in line_patterns:
        match = re.search(pattern, request, re.IGNORECASE)
        if match:
            constraints["line_count"] = max(1, min(120, int(match.group(1))))
            constraints["format"] = "plain_lines"
            break
    word_match = re.search(r"\b(\d{2,5})\s*(?:words?|wörter|woerter)\b", request, re.IGNORECASE)
    if word_match:
        constraints["word_count"] = max(10, min(10_000, int(word_match.group(1))))
    if re.search(r"\b(?:shorter|shoter|shorten|concise|brief|compress|kürzer|kuerzer|zu\s+lang)\b", request, re.I):
        constraints["length"] = "shorter"
    elif re.search(r"\b(?:longer|expand|more\s+detail|elaborate|ausführlicher|ausfuehrlicher)\b", request, re.I):
        constraints["length"] = "longer"
    if re.search(r"\b(?:bullet|bullets|numbered\s+list|auflistung|aufzählung|aufzaehlung)\b", request, re.I):
        constraints["format"] = constraints.get("format") or "bullets"
    if re.search(r"\b(?:formal|formeller|official|controlled|audit[-\s]?ready)\b", request, re.I):
        constraints["tone"] = "formal"
    if source_text:
        constraints["source_word_count"] = len(source_text.split())
    return constraints


def _enforce_line_count(text: str, count: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if count <= 0 or not cleaned:
        return str(text or "").strip()
    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if p.strip()]
    if not pieces:
        pieces = [cleaned]
    if len(pieces) < count:
        words = cleaned.split()
        if len(words) >= count:
            chunk_size = max(1, (len(words) + count - 1) // count)
            pieces = [" ".join(words[i : i + chunk_size]).strip() for i in range(0, len(words), chunk_size)]
        pieces = [p for p in pieces if p]
    lines = pieces[:count]
    while len(lines) < count and lines:
        lines.append(lines[-1])
    return "\n".join(lines[:count]).strip()


def _enforce_shorter_text(text: str, source_text: str) -> str:
    source_words = [w for w in str(source_text or "").split() if w]
    if not source_words:
        return str(text or "").strip()
    max_words = max(12, int(len(source_words) * 0.75))
    words = [w for w in str(text or "").split() if w]
    if len(words) <= max_words:
        return str(text or "").strip()
    return " ".join(words[:max_words]).rstrip(" ,;:") + "."


def _build_action_metadata(
    *,
    action: str,
    user_request: str,
    target_text: str,
    active_sop_id: str | None,
    source_sop_ids: list[str] | None,
    profile_id: str | None,
    evidence: dict[str, Any],
    subagent_used: str,
    requires_approval: bool,
) -> dict[str, Any]:
    constraints = _extract_action_constraints(user_request, target_text)
    rag_used = action in {"gap_check", "compliance", "summarize", "explain"} or bool(source_sop_ids)
    return {
        "primary_action": action,
        "target_scope": "external_resolved_target",
        "target_hint": None,
        "resolved_section": None,
        "resolved_range": None,
        "length_constraint": {
            "type": "lines" if constraints.get("line_count") else constraints.get("length"),
            "value": constraints.get("line_count") or constraints.get("word_count"),
        },
        "constraints": constraints,
        "style_constraint": {
            "profile_id": profile_id,
            "style_only": bool(profile_id and action in {"rewrite", "improve"}),
        },
        "rag": {
            "used": rag_used,
            "chunk_count": len(evidence.get("chunks") or []),
            "citations": evidence.get("citations") or [],
            "weak_evidence": bool(evidence.get("weak_evidence")),
        },
        "agent_mode": _agent_mode(),
        "deepagents_available": DEEPAGENTS_AVAILABLE,
        "subagent_used": subagent_used,
        "llm_model": None,
        "source_sop_ids": source_sop_ids or ([active_sop_id] if active_sop_id else []),
        "requires_approval": requires_approval,
        "request_marker": f"SOP_AGENT_{uuid.uuid4().hex[:12].upper()}",
    }


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _split_sentences(text: str, limit: int = 5) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [_clean_text(p) for p in pieces if _clean_text(p)][:limit]


def _extract_heading_lines(text: str) -> list[str]:
    headings: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^\d+(?:\.\d+)*\s+.{3,100}$", line):
            headings.append(line)
        elif len(line) <= 90 and re.search(r"\b(purpose|scope|procedure|responsib|definitions|records|training|zweck|geltungsbereich|verfahren|verantwort)\b", line, re.I):
            headings.append(line)
    return headings[:40]


def _extract_headings_from_doc(doc_json: dict[str, Any] | None) -> list[str]:
    headings: list[str] = []

    def text_of(node: Any) -> str:
        if isinstance(node, str):
            return node
        if isinstance(node, list):
            return " ".join(text_of(item) for item in node).strip()
        if not isinstance(node, dict):
            return ""
        if node.get("type") == "text" and isinstance(node.get("text"), str):
            return node["text"]
        return " ".join(text_of(item) for item in node.get("content", []) or []).strip()

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "heading":
            heading = _clean_text(text_of(node))
            if heading:
                headings.append(heading[:160])
        for child in node.get("content", []) or []:
            walk(child)
        for child in node.get("sections", []) or []:
            title = _clean_text(child.get("title")) if isinstance(child, dict) else ""
            if title:
                headings.append(title[:160])
            walk(child)

    walk(doc_json or {})
    return list(dict.fromkeys(headings))[:40]


def _profile_json(profile: ClientProfile | None) -> dict[str, Any]:
    return profile.active_profile_json if profile and isinstance(profile.active_profile_json, dict) else {}


def _tenant_id_for_loaded(loaded: list[LoadedSOP]) -> uuid.UUID:
    for item in loaded:
        if item.sop and item.sop.tenant_id:
            return item.sop.tenant_id
    return DEFAULT_TENANT_ID


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, dict):
        return list(value.values())
    if value in (None, ""):
        return []
    return [value]


def _detected_json(row: SOPDetectedParameters | None) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "document_information": row.document_information or {},
        "writing_style": row.writing_style or {},
        "roles_raci": row.roles_raci or {},
        "workflows": row.workflows or {},
        "compliance_elements": row.compliance_elements or {},
        "risks_gaps": row.risks_gaps or {},
        "terminology": row.terminology or {},
        "structure_patterns": row.structure_patterns or {},
        "style_suggestions": row.style_suggestions or {},
        "readiness_check": row.readiness_check or {},
    }


def _source_summary(item: LoadedSOP) -> dict[str, Any]:
    detected = _detected_json(item.detected)
    profile = _profile_json(item.profile)
    return {
        "sop_id": str(item.sop.id),
        "sop_number": item.sop.sop_number,
        "title": item.sop.title,
        "department": item.sop.department,
        "version_id": str(item.version.id) if item.version else None,
        "profile_id": str(item.profile.id) if item.profile else None,
        "headings": item.headings,
        "detected": detected,
        "profile": profile,
        "word_count": len((item.text or "").split()),
        "chunk_count": len(item.chunks),
    }


def _normalize_heading(value: str) -> str:
    value = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", value or "")
    value = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return value


_HEADING_ARTIFACT_RE = re.compile(
    r"\b(?:isbn|issn|doi|pat|page|seite|www\.|http|copyright|figure|table of contents)\b",
    re.I,
)


def _clean_template_heading(value: Any) -> str:
    raw = _clean_text(value)
    if not raw:
        return ""
    raw = raw.strip(" -–—:;,.")
    if len(raw) < 3 or len(raw) > 140:
        return ""
    if _HEADING_ARTIFACT_RE.search(raw):
        return ""
    if re.fullmatch(r"[\W_()]+", raw):
        return ""
    if re.fullmatch(r"\(?[A-Z]{1,6}\)?\)?", raw) and len(raw) <= 8:
        return ""
    if re.search(r"\bISBN\s*\d+", raw, re.I):
        return ""
    if raw.count("(") != raw.count(")") and len(raw) < 20:
        return ""
    return raw


def _heading_confidence(heading: str, source_count: int = 1) -> float:
    text = _clean_template_heading(heading)
    if not text:
        return 0.0
    score = 0.25
    norm = _normalize_heading(text)
    if re.match(r"^\d+(?:\.\d+)*\s+\S+", text):
        score += 0.2
    if re.search(r"\b(purpose|scope|responsib|procedure|records|training|approval|review|risk|control|zweck|geltungsbereich|verfahren|verantwort|dokumentation)\b", norm, re.I):
        score += 0.25
    if source_count > 1:
        score += 0.2
    if len(text.split()) <= 8:
        score += 0.1
    return min(score, 1.0)


def _load_sop(db: Session, sop_id: str) -> LoadedSOP:
    sid = uuid.UUID(str(sop_id))
    sop = db.query(SOP).filter(SOP.id == sid, SOP.is_active == True).first()  # noqa: E712
    if not sop:
        raise ValueError(f"SOP not found: {sop_id}")

    version = (
        db.query(SOPVersion).filter(SOPVersion.id == sop.current_version_id).first()
        if sop.current_version_id
        else db.query(SOPVersion).filter(SOPVersion.sop_id == sop.id).order_by(SOPVersion.created_at.desc()).first()
    )
    content_json = version.content_json if version else {}
    text = extract_plain_text_from_tiptap(content_json) if version else ""
    headings = _extract_headings_from_doc(content_json) or _extract_heading_lines(text)
    detected = (
        db.query(SOPDetectedParameters)
        .filter(SOPDetectedParameters.sop_id == sop.id)
        .order_by(SOPDetectedParameters.created_at.desc())
        .first()
    )
    profile = None
    if detected and detected.client_profile_id:
        profile = db.query(ClientProfile).filter(ClientProfile.id == detected.client_profile_id).first()
    chunks = (
        db.query(KnowledgeChunk)
        .filter(KnowledgeChunk.entity_type == "sop", KnowledgeChunk.entity_id == sop.id)
        .order_by(KnowledgeChunk.chunk_order.asc())
        .limit(8)
        .all()
    )
    return LoadedSOP(sop=sop, version=version, text=text, headings=headings, profile=profile, detected=detected, chunks=chunks)


def _common_list(values: list[Any], limit: int = 12) -> list[str]:
    counter: Counter[str] = Counter()
    for value in values:
        if isinstance(value, str):
            item = value.strip()
            if item:
                counter[item] += 1
        elif isinstance(value, dict):
            item = str(value.get("standard") or value.get("value") or value.get("term") or "").strip()
            if item:
                counter[item] += 1
    return [item for item, _count in counter.most_common(limit)]


def _collect_roles(profile_jsons: list[dict[str, Any]], detected_jsons: list[dict[str, Any]], limit: int = 16) -> list[str]:
    def role_ok(value: Any) -> str:
        text = _clean_text(value)
        if not text:
            return ""
        norm = _normalize_heading(text)
        if not norm or norm in {"roles", "raci summary", "detected role count", "missing expected roles", "needs review role count", "expected roles for context"}:
            return ""
        if re.search(r"\b(count|summary|expected|missing|review)\b", norm) and len(norm.split()) > 1:
            return ""
        if len(text) > 80:
            return ""
        return text

    values: list[Any] = []
    for pj in profile_jsons:
        values.extend(role_ok(v) for v in _as_list(pj.get("roles")))
        for key in ("roles_raci", "raci", "responsibilities"):
            val = pj.get(key)
            if isinstance(val, dict):
                values.extend(role_ok(v) for v in val.keys())
            else:
                values.extend(role_ok(v) for v in _as_list(val))
    for dj in detected_jsons:
        roles_raci = dj.get("roles_raci") or {}
        if isinstance(roles_raci, dict):
            for role, spec in roles_raci.items():
                if not isinstance(spec, dict) or spec.get("detected") or spec.get("confidence") or spec.get("actions"):
                    values.append(role_ok(role))
        values.extend(role_ok(v) for v in _as_list((dj.get("document_information") or {}).get("roles")))
    return _common_list([v for v in values if v], limit=limit)


def _learn_style_profile(profile_jsons: list[dict[str, Any]], detected_jsons: list[dict[str, Any]]) -> dict[str, Any]:
    counters: dict[str, Counter[str]] = {
        "tone": Counter(),
        "formality": Counter(),
        "primary_format": Counter(),
        "directive_wording": Counter(),
        "writing_style": Counter(),
    }
    modal_counter: Counter[str] = Counter()
    table_counter: Counter[str] = Counter()

    for pj in profile_jsons:
        for source in [pj.get("preferred_style") or {}, pj.get("style_profile") or {}, pj.get("writing_style") or {}]:
            if isinstance(source, dict):
                for key in ("tone", "formality", "primary_format", "directive_wording", "primary_style"):
                    value = _clean_text(source.get(key))
                    if value:
                        mapped = "writing_style" if key == "primary_style" else key
                        counters.setdefault(mapped, Counter())[value] += 1
        modal = pj.get("modal_language") or {}
        if isinstance(modal, dict):
            for group in ("mandatory", "recommended", "prohibited"):
                for term in _as_list(modal.get(group)):
                    if _clean_text(term) and not re.fullmatch(r"\d+(?:\.\d+)?", _clean_text(term)):
                        modal_counter[_clean_text(term)] += 1
    for dj in detected_jsons:
        style = dj.get("writing_style") or {}
        if isinstance(style, dict):
            for key, value in style.items():
                if isinstance(value, str) and _clean_text(value):
                    counters["writing_style"][_clean_text(value)] += 1
        structure = dj.get("structure_patterns") or {}
        if isinstance(structure, dict):
            for table_name in _as_list(structure.get("tables") or structure.get("table_patterns")):
                if _clean_text(table_name):
                    table_counter[_clean_text(table_name)] += 1

    def top(counter: Counter[str]) -> str | None:
        return counter.most_common(1)[0][0] if counter else None

    return {
        "tone": top(counters["tone"]) or top(counters["writing_style"]) or "controlled",
        "formality": top(counters["formality"]) or "formal",
        "primary_format": top(counters["primary_format"]) or "structured_sop",
        "directive_wording": top(counters["directive_wording"]) or "role_based_must_shall",
        "modal_terms": [item for item, _ in modal_counter.most_common(12)],
        "table_usage": [item for item, _ in table_counter.most_common(8)],
        "section_rhythm": "numbered_sections_with_controlled_records",
        "compliance_wording": "audit_ready_evidence_and_approval_language",
    }


def _build_minimal_tiptap_doc(text: str) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        heading_match = re.match(r"^(\d+(?:\.\d+)*)\s+(.{3,120})$", line)
        if heading_match:
            level = min(3, heading_match.group(1).count(".") + 1)
            nodes.append({"type": "heading", "attrs": {"level": level}, "content": [{"type": "text", "text": line[:240]}]})
        else:
            nodes.append({"type": "paragraph", "content": [{"type": "text", "text": line[:1800]}]})
    return {"type": "doc", "content": nodes or [{"type": "paragraph", "content": [{"type": "text", "text": "New SOP draft"}]}]}


def _table_node(rows: list[list[Any]]) -> dict[str, Any]:
    return {
        "type": "table",
        "rows": [[_clean_text(cell) for cell in row] for row in rows if isinstance(row, list)],
        "header_rows": 1,
    }


def _generation_table_metadata(doc_json: dict[str, Any] | None, tables: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    labels = [_clean_text(t.get("label")) for t in (tables or []) if isinstance(t, dict) and _clean_text(t.get("label"))]
    count = 0

    def walk(node: Any) -> None:
        nonlocal count
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "table":
            count += 1
        for child in node.get("content", []) or []:
            walk(child)

    walk(doc_json or {})
    label_text = " ".join(labels).lower()
    return {
        "table_count": count,
        "table_labels": labels,
        "has_approval_table": "approval" in label_text,
        "has_traceability_table": "traceability" in label_text or "source" in label_text,
    }


def _default_generation_tables(
    *,
    source_traceability: list[dict[str, Any]],
    learned_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    roles = _as_list(learned_profile.get("roles"))[:6] or ["Process Owner", "Quality Assurance", "Department Manager"]
    standards = _as_list(learned_profile.get("compliance_standards"))[:6]
    source_rows = [["Source SOP", "Title", "Reusable Elements"]]
    for source in source_traceability[:8]:
        if not isinstance(source, dict):
            continue
        source_rows.append([
            source.get("source_sop_number") or source.get("sop_number") or "Source SOP",
            source.get("title") or "",
            ", ".join(str(x) for x in _as_list(source.get("sample_sections") or source.get("headings"))[:4]),
        ])
    if len(source_rows) == 1:
        source_rows.append(["Selected SOPs", "", "Structure, style, terminology"])
    return [
        {
            "label": "Approval Table",
            "rows": [
                ["Role", "Name", "Signature", "Date"],
                ["Author", "To be assigned", "", ""],
                ["Reviewer", "Quality Assurance", "", ""],
                ["Approver", "Quality Manager", "", ""],
            ],
        },
        {
            "label": "Revision History",
            "rows": [["Version", "Change Summary", "Author", "Date"], ["0.1", "Initial generated draft", "Agent Orchestrator", ""]],
        },
        {
            "label": "Roles / RACI Table",
            "rows": [["Role", "Responsibility", "Accountability"]] + [[str(role), "Execute assigned SOP activities", "Maintain objective evidence"] for role in roles],
        },
        {
            "label": "Records / Evidence Table",
            "rows": [
                ["Record", "Owner", "Retention / Location"],
                ["Investigation record", "Process Owner", "Approved quality repository"],
                ["CAPA evidence", "Action Owner", "Approved quality repository"],
                ["Training record", "Department Manager", "Training system"],
            ],
        },
        {
            "label": "Risk / Control Matrix",
            "rows": [
                ["Risk", "Control", "Evidence"],
                ["Incomplete investigation", "QA review before closure", "Approved investigation report"],
                ["Overdue action", "Escalation and due-date monitoring", "Action tracker"],
                ["Ineffective CAPA", "Effectiveness verification", "Effectiveness check record"],
            ],
        },
        {
            "label": "Source Traceability Table",
            "rows": source_rows,
        },
        {
            "label": "Compliance Pattern Table",
            "rows": [["Standard / Pattern", "Usage in Draft"]] + [[str(std), "Considered as reusable compliance language"] for std in standards[:6]],
        } if standards else {
            "label": "Compliance Pattern Table",
            "rows": [["Standard / Pattern", "Usage in Draft"], ["Good Documentation Practice", "Applied to records and evidence language"]],
        },
    ]


def _build_tiptap_doc_from_sections(
    title: str,
    tables: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    fallback_text: str,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    if title:
        nodes.append({"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": title[:240]}]})
    for table in tables or []:
        rows = table.get("rows") if isinstance(table, dict) else None
        if isinstance(rows, list) and rows:
            label = _clean_text(table.get("label"))
            if label:
                nodes.append({"type": "heading", "attrs": {"level": 2}, "content": [{"type": "text", "text": label[:160]}]})
            nodes.append(_table_node(rows))
    for idx, section in enumerate(sections or [], start=1):
        if not isinstance(section, dict):
            continue
        heading = _clean_text(section.get("heading") or f"Section {idx}")
        if heading:
            level = 2 if re.match(r"^\d+\.\d+", heading) else 1
            nodes.append({"type": "heading", "attrs": {"level": min(level, 3)}, "content": [{"type": "text", "text": heading[:240]}]})
        for para in _as_list(section.get("paragraphs")):
            text = _clean_text(para)
            if text:
                nodes.append({"type": "paragraph", "content": [{"type": "text", "text": text[:1800]}]})
        for bullet in _as_list(section.get("bullets")):
            text = _clean_text(bullet)
            if text:
                nodes.append({"type": "paragraph", "content": [{"type": "text", "text": f"- {text}"[:1800]}]})
        for table in _as_list(section.get("tables")):
            if isinstance(table, dict) and isinstance(table.get("rows"), list):
                label = _clean_text(table.get("label"))
                if label:
                    nodes.append({"type": "paragraph", "content": [{"type": "text", "text": label[:240]}]})
                nodes.append(_table_node(table["rows"]))
    return {"type": "doc", "content": nodes} if nodes else _build_minimal_tiptap_doc(fallback_text)


def _sections_to_text(title: str, tables: list[dict[str, Any]], sections: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    if title:
        lines.append(title)
    for table in tables or []:
        label = _clean_text(table.get("label") if isinstance(table, dict) else "")
        rows = table.get("rows") if isinstance(table, dict) else None
        if label:
            lines.extend(["", label])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, list):
                    lines.append(" | ".join(_clean_text(cell) for cell in row))
    for section in sections or []:
        if not isinstance(section, dict):
            continue
        heading = _clean_text(section.get("heading"))
        if heading:
            lines.extend(["", heading])
        for para in _as_list(section.get("paragraphs")):
            text = _clean_text(para)
            if text:
                lines.append(text)
        for bullet in _as_list(section.get("bullets")):
            text = _clean_text(bullet)
            if text:
                lines.append(f"- {text}")
        for table in _as_list(section.get("tables")):
            if isinstance(table, dict) and isinstance(table.get("rows"), list):
                label = _clean_text(table.get("label"))
                if label:
                    lines.append(label)
                for row in table["rows"]:
                    if isinstance(row, list):
                        lines.append(" | ".join(_clean_text(cell) for cell in row))
    return "\n".join(lines).strip()


def _sections_from_generated_text(text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^\d+(?:\.\d+)?\s+\S+", line):
            if current:
                sections.append(current)
            current = {"heading": line, "paragraphs": [], "bullets": []}
            continue
        if current is None:
            continue
        if line.startswith("- "):
            current.setdefault("bullets", []).append(line[2:].strip())
        elif " | " in line and len(line.split("|")) >= 2:
            continue
        else:
            current.setdefault("paragraphs", []).append(line)
    if current:
        sections.append(current)
    return sections


def compare_sops(db: Session, source_sop_ids: list[str], query: str = "") -> dict[str, Any]:
    if len(source_sop_ids) < 2:
        raise ValueError("Select at least two SOPs to compare.")

    loaded = [_load_sop(db, sid) for sid in source_sop_ids]
    source_summaries = [_source_summary(item) for item in loaded]
    heading_sets = [
        {_normalize_heading(_clean_template_heading(h)) for h in item.headings if _normalize_heading(_clean_template_heading(h))}
        for item in loaded
    ]
    common_heading_keys = set.intersection(*heading_sets) if heading_sets else set()
    all_heading_keys = set.union(*heading_sets) if heading_sets else set()

    display_by_key: dict[str, str] = {}
    for item in loaded:
        for heading in item.headings:
            cleaned = _clean_template_heading(heading)
            key = _normalize_heading(cleaned)
            if key and key not in display_by_key:
                display_by_key[key] = re.sub(r"^\d+(?:\.\d+)*\s+", "", cleaned).strip()

    missing_by_sop = []
    for item, heading_set in zip(loaded, heading_sets, strict=False):
        missing = sorted(all_heading_keys - heading_set)
        missing_by_sop.append({
            "sop_id": str(item.sop.id),
            "sop_number": item.sop.sop_number,
            "missing_headings": [display_by_key.get(key, key) for key in missing[:20]],
        })

    profile_jsons = [_profile_json(item.profile) for item in loaded]
    detected_jsons = [_detected_json(item.detected) for item in loaded]
    standards = _common_list([
        item
        for pj in profile_jsons
        for item in _as_list((pj.get("compliance_elements") or {}).get("standards_detected"))
    ] + [
        item
        for dj in detected_jsons
        for item in _as_list((dj.get("compliance_elements") or {}).get("standards_detected"))
    ])
    terminology = _common_list([
        item
        for pj in profile_jsons
        for item in _as_list((pj.get("terminology") or {}).get("acronyms"))
        + _as_list((pj.get("terminology") or {}).get("controlled_terms"))
        + _as_list((pj.get("terminology") or {}).get("domain_terms"))
    ] + [
        item
        for dj in detected_jsons
        for item in _as_list((dj.get("terminology") or {}).get("acronyms"))
        + _as_list((dj.get("terminology") or {}).get("controlled_terms"))
        + _as_list((dj.get("terminology") or {}).get("domain_terms"))
    ])
    roles = _common_list([
        role
        for dj in detected_jsons
        for role, spec in (dj.get("roles_raci") or {}).items()
        if isinstance(spec, dict) and (spec.get("detected") or spec.get("confidence"))
    ])
    risks = _common_list([
        item
        for dj in detected_jsons
        for item in _as_list((dj.get("risks_gaps") or {}).get("risks"))
        + _as_list((dj.get("risks_gaps") or {}).get("gaps"))
    ])
    evidence = retrieve_rag_evidence(db, query or " ".join(display_by_key.values()), source_sop_ids, limit=8)

    return {
        "comparison_id": str(uuid.uuid4()),
        "source_count": len(loaded),
        "sources": [
            {
                "sop_id": s["sop_id"],
                "sop_number": s["sop_number"],
                "title": s["title"],
                "department": s["department"],
                "profile_id": s["profile_id"],
                "word_count": s["word_count"],
            }
            for s in source_summaries
        ],
        "common_headings": [display_by_key.get(key, key) for key in sorted(common_heading_keys)],
        "different_or_unique_headings": [display_by_key.get(key, key) for key in sorted(all_heading_keys - common_heading_keys)],
        "missing_by_sop": missing_by_sop,
        "shared_terms": terminology,
        "shared_roles": roles,
        "shared_standards": standards,
        "shared_risks_or_gaps": risks,
        "rag_evidence": evidence,
        "source_traceability": [
            {
                "source_sop_id": str(item.sop.id),
                "source_sop_number": item.sop.sop_number,
                "title": item.sop.title,
                "headings": [_clean_template_heading(h) for h in item.headings if _clean_template_heading(h)][:12],
                "raw_headings": item.headings[:20],
                "profile_id": str(item.profile.id) if item.profile else None,
            }
            for item in loaded
        ],
    }


def learn_template(
    db: Session,
    source_sop_ids: list[str],
    client_name: str = "Client",
    template_name: str | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    if len(source_sop_ids) < 2:
        raise ValueError("Select at least two SOPs to learn a reusable template.")

    loaded = [_load_sop(db, sid) for sid in source_sop_ids]
    profile_jsons = [_profile_json(item.profile) for item in loaded]
    headings = [h for item in loaded for h in item.headings]
    raw_headings = [_clean_text(h) for h in headings if _clean_text(h)]
    heading_display: dict[str, str] = {}
    heading_sources: dict[str, set[str]] = {}
    heading_counts: Counter[str] = Counter()
    for item in loaded:
        seen_in_sop: set[str] = set()
        for raw_heading in item.headings:
            cleaned = _clean_template_heading(re.sub(r"^\d+(?:\.\d+)*\s+", "", raw_heading).strip())
            key = _normalize_heading(cleaned)
            if not key:
                continue
            heading_display.setdefault(key, cleaned)
            seen_in_sop.add(key)
        for key in seen_in_sop:
            heading_counts[key] += 1
            heading_sources.setdefault(key, set()).add(str(item.sop.id))

    domains = _common_list([d for pj in profile_jsons for d in (pj.get("detected_domains") or [])])
    departments = _common_list([d for pj in profile_jsons for d in (pj.get("detected_departments") or [])])
    sop_types = _common_list([d for pj in profile_jsons for d in (pj.get("detected_sop_types") or [])])
    standards = _common_list([
        item
        for pj in profile_jsons
        for item in _as_list((pj.get("compliance_elements") or {}).get("standards_detected"))
    ])
    terminology = _common_list([
        item
        for pj in profile_jsons
        for item in _as_list((pj.get("terminology") or {}).get("acronyms"))
        + _as_list((pj.get("terminology") or {}).get("controlled_terms"))
        + _as_list((pj.get("terminology") or {}).get("domain_terms"))
    ])

    detected_jsons = [_detected_json(item.detected) for item in loaded]
    style_profile = _learn_style_profile(profile_jsons, detected_jsons)
    roles = _collect_roles(profile_jsons, detected_jsons) or ["Process Owner", "Quality Assurance", "Department Manager"]
    template_outline = [
        {
            "heading": heading_display.get(key, key),
            "required": count >= max(2, len(loaded) // 2),
            "source_count": count,
            "confidence": _heading_confidence(heading_display.get(key, key), count),
            "source_sop_ids": sorted(heading_sources.get(key, set())),
        }
        for key, count in heading_counts.most_common(18)
        if _heading_confidence(heading_display.get(key, key), count) >= 0.45
    ]
    if not template_outline:
        template_outline = [
            {"heading": heading, "required": True, "source_count": len(loaded), "confidence": 0.75, "source_sop_ids": [str(item.sop.id) for item in loaded]}
            for heading in ["Purpose", "Scope", "Responsibilities", "Procedure", "Records"]
        ]

    comparison = compare_sops(db, source_sop_ids, query=f"{client_name} template learning")
    comparison_summary = {
        **comparison,
        "common_headings": template_outline[:10],
        "domains": domains,
        "departments": departments,
        "sop_types": sop_types,
        "standards": standards,
        "terminology": terminology[:15],
    }

    learned_profile = {
        "client_name": client_name,
        "agent_mode": _agent_mode(),
        "detected_domains": domains,
        "detected_departments": departments,
        "detected_sop_types": sop_types,
        "preferred_style": {
            "tone": style_profile.get("tone"),
            "formality": style_profile.get("formality"),
            "primary_format": style_profile.get("primary_format"),
            "directive_wording": style_profile.get("directive_wording"),
        },
        "style_profile": style_profile,
        "roles": roles,
        "modal_terms": style_profile.get("modal_terms") or [],
        "terminology": terminology,
        "compliance_standards": standards,
        "template_outline": template_outline,
        "raw_headings": raw_headings[:80],
        "clean_template_outline": template_outline,
    }

    source_traceability = [
        {
            "source_sop_id": str(item.sop.id),
            "source_sop_number": item.sop.sop_number,
            "title": item.sop.title,
            "profile_id": str(item.profile.id) if item.profile else None,
            "sample_sections": [_clean_template_heading(h) for h in item.headings if _clean_template_heading(h)][:8],
            "raw_headings": item.headings[:20],
        }
        for item in loaded
    ]
    template_id = None
    if persist:
        template = SOPGenerationTemplate(
            tenant_id=_tenant_id_for_loaded(loaded),
            client_name=client_name or "Client",
            name=template_name or f"{client_name or 'Client'} SOP Template",
            source_sop_ids=[str(item.sop.id) for item in loaded],
            learned_structure_json={"template_outline": template_outline, "raw_headings": raw_headings[:120], "clean_template_outline": template_outline},
            style_profile_json=learned_profile,
            terminology_json={"terms": terminology},
            compliance_patterns_json={"standards": standards},
            source_traceability_json=source_traceability,
            comparison_summary_json=comparison_summary,
            generated_draft_sop_ids=[],
        )
        db.add(template)
        db.flush()
        template_id = str(template.id)

    return {
        "template_id": template_id,
        "learned_profile": learned_profile,
        "template_outline": template_outline,
        "raw_headings": raw_headings[:120],
        "clean_template_outline": template_outline,
        "comparison_summary": comparison_summary,
        "source_traceability": source_traceability,
    }


def retrieve_rag_evidence(db: Session, query: str, source_sop_ids: list[str] | None = None, limit: int = 8) -> dict[str, Any]:
    q_terms = [t.lower() for t in re.findall(r"[A-Za-zÄÖÜäöüß0-9]{3,}", query or "")]
    sop_uuid_filter = []
    for sid in source_sop_ids or []:
        try:
            sop_uuid_filter.append(uuid.UUID(str(sid)))
        except Exception:
            continue

    query_rows = db.query(KnowledgeChunk).filter(KnowledgeChunk.entity_type == "sop")
    if sop_uuid_filter:
        query_rows = query_rows.filter(KnowledgeChunk.entity_id.in_(sop_uuid_filter))
    rows = query_rows.order_by(KnowledgeChunk.created_at.desc()).limit(200).all()

    ranked = []
    for row in rows:
        text = row.chunk_text or ""
        score = sum(1 for term in q_terms if term in text.lower())
        if score or not q_terms:
            ranked.append((score, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    chunks = [
        {
            "chunk_id": str(row.id),
            "entity_id": str(row.entity_id),
            "entity_version_id": str(row.entity_version_id) if row.entity_version_id else None,
            "label": (row.metadata_json or {}).get("title") or (row.metadata_json or {}).get("source_label") or f"SOP chunk {row.chunk_order}",
            "text": row.chunk_text[:1200],
            "score": score,
        }
        for score, row in ranked[:limit]
    ]
    return {
        "answer_basis": "relevant_chunks_only",
        "chunks": chunks,
        "citations": [{"chunk_id": c["chunk_id"], "label": c["label"], "entity_id": c["entity_id"]} for c in chunks],
        "weak_evidence": len(chunks) == 0,
    }


def _outline_headings_for_generation(outline: list[dict[str, Any]], limit: int = 8) -> list[str]:
    skip = {
        "document body",
        "table of content",
        "table of contents",
        "name signature",
        "author",
        "reviewer",
        "approver",
    }
    headings: list[str] = []
    for row in outline or []:
        heading = _clean_text(row.get("heading") if isinstance(row, dict) else row)
        heading = re.sub(r"^\d+(?:\.\d+)*\s+", "", heading).strip()
        key = _normalize_heading(heading)
        if not key or key in skip or len(key) < 3:
            continue
        if any(key == _normalize_heading(existing) for existing in headings):
            continue
        headings.append(heading)
        if len(headings) >= limit:
            break
    return headings


def _generated_section_trace(headings: list[str], learned: dict[str, Any], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    source_refs = [
        {
            "source_sop_id": str(s.get("source_sop_id") or s.get("sop_id") or ""),
            "source_sop_number": str(s.get("source_sop_number") or s.get("sop_number") or ""),
            "title": str(s.get("title") or ""),
        }
        for s in learned.get("source_traceability") or []
        if isinstance(s, dict)
    ]
    source_numbers = [s["source_sop_number"] for s in source_refs if s.get("source_sop_number")]
    evidence_labels = [c["label"] for c in (evidence.get("chunks") or [])[:3]]
    return [
        {
            "generated_section": heading,
            "source_sops": source_numbers,
            "source_refs": source_refs,
            "source_sop_id": source_refs[0]["source_sop_id"] if source_refs else "",
            "source_sop_number": source_refs[0]["source_sop_number"] if source_refs else "",
            "evidence_labels": evidence_labels,
        }
        for heading in headings
    ]


def _build_full_generation_text(
    *,
    target_title: str,
    target_department: str,
    target_sop_type: str,
    requirements: str,
    client_name: str,
    learned: dict[str, Any],
    evidence: dict[str, Any],
) -> tuple[str, list[str]]:
    profile = learned.get("learned_profile") if isinstance(learned.get("learned_profile"), dict) else {}
    outline = learned.get("template_outline") if isinstance(learned.get("template_outline"), list) else []
    source_labels = ", ".join(
        str(s.get("source_sop_number") or s.get("sop_number") or "")
        for s in learned.get("source_traceability") or []
        if isinstance(s, dict)
    )
    standards = _as_list(profile.get("compliance_standards"))[:6]
    terms = _as_list(profile.get("terminology"))[:10]
    learned_headings = _outline_headings_for_generation(outline, limit=8)
    is_deviation_capa = bool(re.search(r"\b(deviation|capa|corrective|preventive|non[-\s]?conformance|investigation)\b", f"{target_title} {target_sop_type} {requirements}", re.I))

    references = [
        "Applicable Quality Management System procedures",
        "Good Documentation Practice requirements",
        "Training and qualification requirements",
        "Change control and supplier qualification procedures, where applicable",
    ]
    if standards:
        references.append("Applicable standards and regulations detected from source SOPs: " + ", ".join(str(x) for x in standards))

    definitions = [
        ("Deviation", "An unplanned departure from an approved procedure, specification, instruction, or expected process state."),
        ("CAPA", "Corrective and Preventive Action used to eliminate the cause of an existing or potential quality issue."),
        ("Root Cause", "The most probable underlying reason why a deviation, failure, or quality event occurred."),
        ("Effectiveness Check", "Documented verification that implemented actions resolved the issue and reduced recurrence risk."),
    ] if is_deviation_capa else [
        ("Controlled Document", "An approved document managed under version, review, approval, and archival controls."),
        ("Record", "Evidence generated during execution of this SOP and retained according to the applicable retention period."),
    ]

    procedure_steps = [
        ("Event identification and immediate containment", [
            "Any employee who identifies a quality event shall notify the responsible department and Quality Assurance without undue delay.",
            "The process owner shall assess whether immediate containment is required to protect patient safety, data integrity, product quality, or compliance.",
            "Containment actions shall be documented with date, owner, rationale, and objective evidence.",
        ]),
        ("Deviation registration and classification", [
            "Quality Assurance shall ensure that each deviation is recorded in the approved tracking system or controlled register.",
            "The deviation record shall include a unique identifier, title, date of discovery, affected process, impacted material or record, initial risk assessment, and responsible owner.",
            "Classification shall consider quality impact, regulatory impact, recurrence, detectability, and whether escalation to management is required.",
        ]),
        ("Investigation planning", [
            "The responsible owner shall define the investigation scope, required subject matter experts, expected records, and target completion date.",
            "The investigation plan shall be proportionate to the risk and complexity of the event.",
            "Where supplier, training, documentation, or process-control weaknesses are suspected, the related SOPs and records shall be reviewed.",
        ]),
        ("Root cause analysis", [
            "The investigation shall identify the most probable root cause using a justified method such as 5-Why, fishbone analysis, process mapping, or documented expert assessment.",
            "The record shall distinguish confirmed root cause, contributing factors, and assumptions that could not be verified.",
            "Evidence used for the conclusion shall be traceable to attachments, batch records, training records, logs, audit observations, or system data.",
        ]),
        ("CAPA definition and approval", [
            "Corrective actions shall address the immediate issue and preventive actions shall reduce the likelihood of recurrence.",
            "Each CAPA shall define the action owner, due date, required evidence, acceptance criteria, and dependency on other quality processes.",
            "Quality Assurance shall review and approve CAPA actions before implementation when the action affects a GxP process, controlled document, supplier, system, or training requirement.",
        ]),
        ("Implementation and evidence collection", [
            "Action owners shall complete assigned actions by the approved due date or document an approved extension with justification.",
            "Evidence shall be complete, legible, attributable, contemporaneous, original or verified copy, and accurate.",
            "If implementation changes an approved process or document, the applicable change control or document revision process shall be followed.",
        ]),
        ("Effectiveness verification", [
            "Quality Assurance shall define whether an effectiveness check is required based on risk, recurrence, and quality impact.",
            "Effectiveness checks may include trend review, record sampling, audit verification, training confirmation, or process performance review.",
            "A CAPA shall not be closed until required effectiveness evidence has been reviewed and accepted, or a justified rationale for no effectiveness check is approved.",
        ]),
        ("Closure and quality review", [
            "Quality Assurance shall verify that the deviation investigation, CAPA actions, attachments, approvals, and required cross-references are complete before closure.",
            "Closure shall include a concise conclusion, final impact assessment, residual risk statement, and confirmation that records are retained.",
            "Significant or recurring issues shall be escalated to management review, quality council, or the applicable governance forum.",
        ]),
    ] if is_deviation_capa else [
        ("Process execution", [
            "The responsible owner shall execute the process according to approved instructions and document required evidence.",
            "Quality Assurance shall review critical records where required by the applicable quality process.",
        ]),
        ("Review and approval", [
            "Outputs shall be reviewed for completeness, accuracy, traceability, and compliance with applicable requirements.",
            "Approval shall be completed by authorized personnel before the output is implemented or released.",
        ]),
        ("Record retention", [
            "Records shall be retained in the approved repository and protected from unauthorized change or loss.",
            "Obsolete or superseded records shall be managed according to document control requirements.",
        ]),
    ]

    lines: list[str] = [
        target_title,
        f"SOP Number: DRAFT-{uuid.uuid4().hex[:8].upper()}",
        "Version: 0.1",
        "Status: Draft",
        f"Department: {target_department or 'Quality Assurance'}",
        f"Generated style source: {client_name} template learned from {source_labels or 'selected SOPs'}",
        "",
        "Approval Table",
        "Role | Name | Signature | Date",
        "Author | To be assigned | |",
        "Reviewer | Quality Assurance | |",
        "Approver | Quality Manager | |",
        "",
        "1 Purpose",
        f"The purpose of this SOP is to define the controlled process for {target_title}. The procedure establishes how events are identified, assessed, investigated, corrected, documented, reviewed, and closed in a manner consistent with the learned {client_name} SOP structure and controlled-document style.",
        "This SOP is intended to ensure that quality decisions are traceable, responsibilities are clear, required records are retained, and actions are completed with documented evidence.",
        "",
        "2 Scope",
        f"This SOP applies to {target_department or 'the responsible department'} and all personnel involved in {target_sop_type or target_title.lower()} activities.",
        "The scope includes event notification, initial risk assessment, investigation, root cause analysis, CAPA planning, action implementation, effectiveness verification, closure, trending, and archival of associated records.",
        "This SOP applies to internal personnel, contractors, consultants, and service suppliers when their work may affect GxP activities, quality records, or controlled processes.",
        "",
        "3 References",
    ]
    lines.extend(f"- {ref}" for ref in references)
    if terms:
        lines.append("- Controlled terminology reflected from source SOPs: " + ", ".join(str(t) for t in terms[:8]))
    lines.extend(["", "4 Abbreviations and Definitions"])
    for term, definition in definitions:
        lines.append(f"- {term}: {definition}")

    lines.extend([
        "",
        "5 Responsibilities",
        "- Employee or initiator: Identifies and reports the event promptly, supports containment, and provides factual information and records.",
        "- Process owner: Assesses operational impact, leads the investigation, proposes corrective and preventive actions, and ensures timely implementation.",
        "- Quality Assurance: Reviews classification, approves investigation conclusions, challenges root cause rationale, approves CAPA actions, verifies closure readiness, and ensures record traceability.",
        "- Subject Matter Expert: Provides technical assessment, evaluates process or supplier impact, and supports evidence review.",
        "- Department manager: Ensures resources are available, overdue actions are escalated, and recurring issues are reviewed.",
        "",
        "6 Procedure",
    ])
    for index, (heading, bullets) in enumerate(procedure_steps, start=1):
        lines.append(f"6.{index} {heading}")
        lines.extend(f"- {bullet}" for bullet in bullets)

    lines.extend([
        "",
        "7 Training and Intended Users",
        "Personnel who initiate, investigate, approve, implement, or verify records under this SOP shall be trained before performing assigned responsibilities.",
        "Training shall be documented in the approved training system or training matrix. Re-training shall be performed when this SOP is revised, when recurring errors are identified, or when role responsibilities change.",
        "",
        "8 Records and Attachments",
        "The following records may be generated by this SOP and shall be retained according to applicable retention requirements:",
        "- Deviation or event record",
        "- Initial impact and risk assessment",
        "- Investigation plan and investigation report",
        "- Root cause analysis worksheet",
        "- CAPA plan and implementation evidence",
        "- Effectiveness check record",
        "- Closure approval record",
        "- Supporting attachments, logs, training records, supplier records, audit observations, or change-control references",
        "",
        "9 Timelines and Escalation",
        "Target completion dates shall be assigned according to risk, complexity, and applicable quality requirements. Overdue investigations, CAPA actions, or effectiveness checks shall be escalated to the department manager and Quality Assurance.",
        "High-risk, critical, recurring, or potentially reportable events shall be escalated immediately to Quality Assurance and senior management for decision and documented follow-up.",
        "",
        "10 Quality Review and Trending",
        "Quality Assurance shall periodically review deviation and CAPA records for recurrence, overdue actions, weak root cause rationale, ineffective actions, and repeated documentation issues.",
        "Trends shall be evaluated for potential process improvement, supplier follow-up, training updates, document revision, audit focus, or management review.",
        "",
        "11 Document Control",
        "This SOP shall be reviewed, approved, distributed, revised, archived, and retired according to the applicable controlled document process.",
        "Only the current approved version shall be used for operational activities. Printed or exported copies are uncontrolled unless specifically identified and managed as controlled copies.",
        "",
        "12 Source Template Reuse and Traceability",
        f"This draft reuses the {client_name} learned SOP template and style patterns from: {source_labels or 'selected source SOPs'}.",
    ])
    if learned_headings:
        lines.append("Template sections considered during generation: " + ", ".join(learned_headings) + ".")
    if evidence.get("chunks"):
        lines.append("Relevant indexed evidence was retrieved and used for source traceability.")
    else:
        lines.append("No indexed chunks were available at generation time; stored profiles, detected headings, and source SOP structure were used.")
    if requirements:
        lines.extend(["", "13 Additional Generation Requirements", requirements.strip()])

    text = "\n".join(lines)
    generated_headings = [line for line in lines if re.match(r"^\d+(?:\.\d+)?\s+\S+", line)]
    return text, generated_headings


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()
    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])
    repaired_candidates = []
    for candidate in candidates:
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        repaired = repaired.replace("\ufeff", "").strip()
        repaired_candidates.append(repaired)
    for candidate in candidates + repaired_candidates:
        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else None
        except Exception:
            continue
    return None


def _normalize_llm_tables(data: dict[str, Any], learned: dict[str, Any]) -> list[dict[str, Any]]:
    raw_tables = data.get("tables") if isinstance(data.get("tables"), list) else []
    normalized: list[dict[str, Any]] = []
    for table in raw_tables:
        if not isinstance(table, dict):
            continue
        rows = table.get("rows")
        if not isinstance(rows, list) or not rows:
            continue
        clean_rows = []
        for row in rows:
            if isinstance(row, list):
                clean_rows.append([_clean_text(cell) for cell in row])
        if clean_rows:
            normalized.append({"label": _clean_text(table.get("label")) or "Generated Table", "rows": clean_rows})
    defaults = _default_generation_tables(
        source_traceability=learned.get("source_traceability") or [],
        learned_profile=learned.get("learned_profile") or {},
    )
    labels = {_normalize_heading(t.get("label", "")) for t in normalized}
    for table in defaults:
        key = _normalize_heading(table.get("label", ""))
        if key and key not in labels:
            normalized.append(table)
    return normalized


def _normalize_llm_sections(data: dict[str, Any]) -> list[dict[str, Any]]:
    sections = data.get("sections")
    if not isinstance(sections, list):
        return []
    normalized = []
    for idx, section in enumerate(sections, start=1):
        if isinstance(section, str):
            normalized.append({"heading": f"{idx} Section", "paragraphs": [section], "bullets": []})
            continue
        if not isinstance(section, dict):
            continue
        heading = _clean_text(section.get("heading") or section.get("title") or f"{idx} Section")
        paragraphs = [_clean_text(p) for p in _as_list(section.get("paragraphs") or section.get("content")) if _clean_text(p)]
        bullets = [_clean_text(p) for p in _as_list(section.get("bullets") or section.get("steps")) if _clean_text(p)]
        tables = section.get("tables") if isinstance(section.get("tables"), list) else []
        if heading or paragraphs or bullets or tables:
            normalized.append({"heading": heading, "paragraphs": paragraphs, "bullets": bullets, "tables": tables})
    return normalized


def _draft_with_local_llm(
    *,
    loaded_sources: list[LoadedSOP],
    learned: dict[str, Any],
    target_title: str,
    target_department: str,
    target_sop_type: str,
    requirements: str,
    client_name: str,
    fallback_title: str,
) -> dict[str, Any] | None:
    try:
        from chatbot.llm.provider import create_openai_client, get_local_llm_config
    except Exception:
        return None

    source_cards = []
    for item in loaded_sources[:4]:
        source_cards.append(
            {
                "sop_number": item.sop.sop_number,
                "title": item.sop.title,
                "headings": item.headings[:18],
                "sample": item.text[:1800],
            }
        )
    profile = learned.get("learned_profile") if isinstance(learned.get("learned_profile"), dict) else {}
    outline = learned.get("template_outline") if isinstance(learned.get("template_outline"), list) else []
    cfg = get_local_llm_config()
    request_marker = f"SOP_GEN_LOCAL_LLM_{uuid.uuid4().hex[:12].upper()}"
    prompt = f"""
You are generating a full GMP-style SOP for a consultant demo.
Request marker for LM Studio logs: {request_marker}

Use the local learned client template and source SOP style. Do not create a short summary.
Create a complete SOP draft with:
- a new SOP title appropriate to the theme and source style
- approval/signature table
- document history table
- purpose, scope, references, abbreviations/definitions
- responsibilities
- detailed workflow/procedure steps
- records/attachments table
- training and document-control sections
- source traceability section

Return ONLY valid JSON using this schema:
{{
  "title": "new SOP title",
  "document_metadata": {{"department": "Quality", "status": "Draft", "version": "0.1"}},
  "tables": [{{"label": "Approval Table", "rows": [["Role","Name","Signature","Date"], ["Author","To be assigned","",""]]}}],
  "style_summary": "brief explanation of reused client style",
  "source_traceability": [{{"source_sop_number": "SOP-...", "reused_elements": ["structure", "terminology"]}}],
  "warnings": [],
  "sections": [
    {{"heading": "1 Purpose", "paragraphs": ["..."], "bullets": []}},
    {{"heading": "6 Procedure", "paragraphs": [], "bullets": ["..."], "tables": []}}
  ]
}}

Important guardrails:
- Use content theme: {target_title} / {target_sop_type}
- Department: {target_department}
- Client/style: {client_name}
- User requirements: {requirements}
- Preserve the Client SOP tone: controlled, audit-ready, procedural, clear responsibilities.
- Use source SOPs for structure/style only unless facts are generic quality-system facts.
- The SOP must be at least 900 words.
- Include multiple table-style blocks in JSON tables, not pipe text.

Learned template outline:
{json.dumps(outline[:20], ensure_ascii=False)}

Learned profile:
{json.dumps(profile, ensure_ascii=False)[:5000]}

Source SOP cards:
{json.dumps(source_cards, ensure_ascii=False)[:9000]}
"""
    try:
        client = create_openai_client()
        response = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": "You draft complete regulated SOPs as strict JSON. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=5000,
        )
        content = response.choices[0].message.content if response.choices else ""
    except Exception:
        return None
    data = _extract_json_object(content or "")
    if not data:
        return None
    sections = _normalize_llm_sections(data)
    if not sections:
        return None
    title = _clean_text(data.get("title")) or fallback_title
    tables = _normalize_llm_tables(data, learned)
    generated_text = _sections_to_text(title, tables, sections)
    if len(generated_text.split()) < 220:
        return None
    return {
        "title": title,
        "tables": tables,
        "sections": sections,
        "generated_text": generated_text,
        "generated_doc_json": _build_tiptap_doc_from_sections(title, tables, sections, generated_text),
        "document_metadata": data.get("document_metadata") if isinstance(data.get("document_metadata"), dict) else {},
        "style_summary": _clean_text(data.get("style_summary")),
        "source_traceability": data.get("source_traceability") if isinstance(data.get("source_traceability"), list) else [],
        "warnings": data.get("warnings") if isinstance(data.get("warnings"), list) else [],
        "llm_provider": cfg.provider,
        "llm_model": cfg.model,
        "request_marker": request_marker,
    }


def generate_sop_preview(
    db: Session,
    source_sop_ids: list[str],
    target_title: str,
    target_department: str = "",
    target_sop_type: str = "",
    language: str = "en",
    requirements: str = "",
    client_name: str = "Client",
    template_id: str | None = None,
) -> dict[str, Any]:
    template = None
    if template_id:
        try:
            template = db.query(SOPGenerationTemplate).filter(SOPGenerationTemplate.id == uuid.UUID(str(template_id))).first()
        except Exception:
            template = None
        if not template:
            raise ValueError(f"SOP generation template not found: {template_id}")
        source_sop_ids = [str(item) for item in (template.source_sop_ids or [])]
        learned = {
            "template_id": str(template.id),
            "learned_profile": template.style_profile_json or {},
            "template_outline": (template.learned_structure_json or {}).get("template_outline") or [],
            "comparison_summary": template.comparison_summary_json or {},
            "source_traceability": template.source_traceability_json or [],
        }
        client_name = template.client_name or client_name
    else:
        learned = learn_template(db, source_sop_ids, client_name=client_name, persist=True)
    evidence = retrieve_rag_evidence(db, f"{target_title} {target_sop_type} {requirements}", source_sop_ids, limit=8)
    lang_note = "German" if str(language).lower().startswith("de") else "English"
    loaded_sources = []
    for sid in source_sop_ids[:4]:
        try:
            loaded_sources.append(_load_sop(db, sid))
        except Exception:
            continue
    llm_draft = _draft_with_local_llm(
        loaded_sources=loaded_sources,
        learned=learned,
        target_title=target_title,
        target_department=target_department,
        target_sop_type=target_sop_type,
        requirements=requirements,
        client_name=client_name,
        fallback_title=target_title,
    )
    generated_text, generated_headings = _build_full_generation_text(
        target_title=target_title,
        target_department=target_department,
        target_sop_type=target_sop_type,
        requirements=requirements,
        client_name=client_name,
        learned=learned,
        evidence=evidence,
    )
    fallback_tables = _default_generation_tables(
        source_traceability=learned.get("source_traceability") or [],
        learned_profile=learned.get("learned_profile") or {},
    )
    fallback_sections = _sections_from_generated_text(generated_text)
    generated_doc_json = _build_tiptap_doc_from_sections(target_title, fallback_tables, fallback_sections, generated_text)
    generation_tables = fallback_tables
    generation_engine = "deterministic_orchestrator"
    fallback_reason = "local_llm_unavailable_or_invalid_json"
    llm_metadata: dict[str, Any] = {"used": False}
    if llm_draft:
        target_title = llm_draft.get("title") or target_title
        generated_text = llm_draft["generated_text"]
        generated_doc_json = llm_draft["generated_doc_json"]
        generation_tables = llm_draft.get("tables") or generation_tables
        generated_headings = [
            _clean_text(section.get("heading"))
            for section in llm_draft.get("sections", [])
            if isinstance(section, dict) and _clean_text(section.get("heading"))
        ]
        generation_engine = "local_llm"
        fallback_reason = None
        llm_metadata = {
            "used": True,
            "provider": llm_draft.get("llm_provider"),
            "model": llm_draft.get("llm_model"),
            "request_marker": llm_draft.get("request_marker"),
            "style_summary": llm_draft.get("style_summary"),
            "document_metadata": llm_draft.get("document_metadata") or {},
        }
    warnings = []
    if evidence["weak_evidence"]:
        warnings.append("No indexed RAG chunks were found for the selected SOPs; generation used stored SOP profiles and editor text only.")
    if len(source_sop_ids) < 3:
        warnings.append("For stronger client style learning, use three or more source SOPs.")
    if not llm_draft:
        warnings.append("Local LLM draft was unavailable or incomplete; deterministic full-SOP generator was used.")
    else:
        warnings.extend(str(w) for w in _as_list(llm_draft.get("warnings")) if _clean_text(w))

    section_trace = _generated_section_trace(generated_headings, learned, evidence)
    table_metadata = _generation_table_metadata(generated_doc_json, generation_tables)

    return {
        "preview_id": str(uuid.uuid4()),
        "template_id": learned.get("template_id"),
        "agent_mode": _agent_mode(),
        "generation_engine": generation_engine,
        "fallback_reason": fallback_reason,
        "llm_model": llm_metadata.get("model"),
        "local_llm": llm_metadata,
        "target": {
            "title": target_title,
            "department": target_department,
            "sop_type": target_sop_type,
            "language": lang_note,
        },
        "learned_profile": learned["learned_profile"],
        "template_outline": learned["template_outline"],
        "comparison_summary": learned["comparison_summary"],
        "source_traceability": section_trace,
        "rag_evidence": evidence,
        "generated_text": generated_text,
        "generated_doc_json": generated_doc_json,
        "tables": generation_tables,
        "style_summary": llm_metadata.get("style_summary") or (learned.get("learned_profile") or {}).get("style_profile") or {},
        "table_validation": table_metadata,
        **table_metadata,
        "warnings": warnings,
    }


def run_sop_action(
    db: Session,
    action: str,
    user_request: str,
    target_text: str,
    active_sop_id: str | None = None,
    source_sop_ids: list[str] | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    action_norm = (action or "").strip().lower()
    if action_norm not in ACTION_SET:
        raise ValueError(f"Unsupported SOP agent action: {action}")
    if action_norm == "rewrite_with_profile_style":
        action_norm = "rewrite"
    elif action_norm == "improve_with_profile_style":
        action_norm = "improve"
    evidence = retrieve_rag_evidence(db, user_request or target_text, source_sop_ids or ([active_sop_id] if active_sop_id else []), limit=6)
    constraints = _extract_action_constraints(user_request, target_text)
    profile = None
    if profile_id:
        profile_uuid = uuid.UUID(str(profile_id))
        profile = db.query(ClientProfile).filter(ClientProfile.id == profile_uuid).first()
        if not profile:
            raise ValueError(f"Profile not found: {profile_id}")
    elif active_sop_id:
        loaded = _load_sop(db, active_sop_id)
        profile = loaded.profile

    deep_agent_result = _invoke_sop_deep_agent(
        db=db,
        action=action_norm,
        user_request=user_request,
        target_text=target_text,
        active_sop_id=active_sop_id,
        source_sop_ids=source_sop_ids,
        profile=profile,
        evidence=evidence,
        constraints=constraints,
        style_profile_only=bool(profile_id),
    )
    suggested = str(deep_agent_result["text"]).strip()
    text = target_text.strip()
    style_only_guardrail = bool(profile_id and action_norm in {"rewrite", "improve"})
    if constraints.get("length") == "shorter" and action_norm in {"rewrite", "improve", "summarize"}:
        suggested = _enforce_shorter_text(suggested, text)
    if constraints.get("line_count"):
        suggested = _enforce_line_count(suggested, int(constraints["line_count"]))
    subagent_used = {
        "rewrite": "rewrite_improve_agent",
        "improve": "rewrite_improve_agent",
        "summarize": "rag_evidence_agent",
        "explain": "rag_evidence_agent",
        "gap_check": "compliance_agent",
        "compliance": "compliance_agent",
    }.get(action_norm, "sop_orchestrator")
    requires_approval = action_norm in {"rewrite", "improve", "gap_check", "compliance"}
    metadata = _build_action_metadata(
        action=action_norm,
        user_request=user_request,
        target_text=target_text,
        active_sop_id=active_sop_id,
        source_sop_ids=source_sop_ids,
        profile_id=profile_id,
        evidence=evidence,
        subagent_used=subagent_used,
        requires_approval=requires_approval,
    )
    metadata["agent_mode"] = "deep_agent"
    metadata["deep_agent"] = deep_agent_result
    metadata["llm_model"] = deep_agent_result.get("model")
    metadata["profile_md_used"] = bool(profile and (profile.active_profile_md or "").strip())
    metadata["profile_parameter_keys"] = sorted((_profile_json(profile) or {}).keys()) if profile else []
    metadata["cross_profile_style_only"] = style_only_guardrail

    return {
        "action": action_norm,
        "primary_action": action_norm,
        "agent_mode": metadata["agent_mode"],
        "deepagents_available": DEEPAGENTS_AVAILABLE,
        "subagent_used": subagent_used,
        "original_text": target_text,
        "suggested_text": suggested,
        "explanation": "Agent orchestration used deterministic target text, profile context, and relevant RAG evidence. Apply only through the existing approval flow.",
        "rag_evidence": evidence,
        "metadata": metadata,
        "orchestrator_metadata": metadata,
        "profile_used": {"id": str(profile.id), "name": profile.name} if profile else None,
        "guardrails": {
            "editor_targeting_external": True,
            "db_write_performed": False,
            "requires_user_approval": requires_approval,
            "style_profile_only": style_only_guardrail,
            "content_source": "current_editor_target_text",
            "style_source": "profile_id" if style_only_guardrail else "active_sop_profile",
            "do_not_import_profile_facts": style_only_guardrail,
        },
        "requires_approval": requires_approval,
    }


def create_sop_draft(
    db: Session,
    preview: dict[str, Any],
    title: str,
    department: str = "",
    client_name: str = "Client",
) -> dict[str, Any]:
    generated_doc = preview.get("generated_doc_json") if isinstance(preview, dict) else None
    generated_text = preview.get("generated_text") if isinstance(preview, dict) else ""
    content_json = generated_doc if isinstance(generated_doc, dict) else _build_minimal_tiptap_doc(generated_text)
    text_for_profile = extract_plain_text_from_tiptap(content_json)
    table_metadata = _generation_table_metadata(content_json, preview.get("tables") if isinstance(preview, dict) else [])

    tenant_row = db.query(SOP.tenant_id).first()
    tenant_id = tenant_row[0] if tenant_row else DEFAULT_TENANT_ID
    sop_id = uuid.uuid4()
    version_id = uuid.uuid4()
    sop_number = f"SOP-GEN-{uuid.uuid4().hex[:8].upper()}"
    while db.query(SOP).filter(SOP.sop_number == sop_number).first():
        sop_number = f"SOP-GEN-{uuid.uuid4().hex[:8].upper()}"

    sop = SOP(
        id=sop_id,
        tenant_id=tenant_id,
        sop_number=sop_number,
        title=title or (preview.get("target") or {}).get("title") or "Generated SOP",
        department=department or (preview.get("target") or {}).get("department") or "Quality",
        source_system="agent_orchestrator",
        is_active=True,
        current_version_id=version_id,
    )
    version = SOPVersion(
        id=version_id,
        sop_id=sop_id,
        version_number="1",
        external_status="draft",
        content_json=content_json,
        metadata_json={
            "sopStatus": "draft",
            "sopMetadata": {
                "title": sop.title,
                "documentId": sop_number,
                "department": sop.department,
                "generatedBy": "agent_orchestrator",
            },
            "agent_generation": {
                "preview_id": preview.get("preview_id"),
                "template_id": preview.get("template_id"),
                "learned_profile": preview.get("learned_profile"),
                "source_traceability": preview.get("source_traceability"),
                "comparison_summary": preview.get("comparison_summary"),
            },
        },
    )
    db.add(sop)
    db.add(version)
    db.flush()

    template_id = preview.get("template_id") if isinstance(preview, dict) else None
    if template_id:
        try:
            template = db.query(SOPGenerationTemplate).filter(SOPGenerationTemplate.id == uuid.UUID(str(template_id))).first()
        except Exception:
            template = None
        if template:
            existing = [str(item) for item in (template.generated_draft_sop_ids or [])]
            if str(sop.id) not in existing:
                template.generated_draft_sop_ids = existing + [str(sop.id)]

    profile_generated = False
    try:
        analyze_and_store_sop_profile(
            db=db,
            sop_id=sop.id,
            sop_version_id=version.id,
            text=text_for_profile,
            client_name=client_name,
            source_filename=sop.title,
        )
        profile_generated = True
    except Exception:
        db.rollback()
        raise

    schedule_semantic_reindex("sop", sop.id, version.id, job_type="agent_generated_sop")
    return {
        "ok": True,
        "sop_id": str(sop.id),
        "sop_number": sop.sop_number,
        "title": sop.title,
        "version_id": str(version.id),
        "current_version_id": str(version.id),
        "profile_generated": profile_generated,
        "rag_indexed": True,
        "source_traceability_count": len(preview.get("source_traceability") or []) if isinstance(preview, dict) else 0,
        "table_count": table_metadata["table_count"],
        "table_validation": table_metadata,
        "open_editor_target": {"sop_id": str(sop.id), "version_id": str(version.id)},
        "message": f"Created draft SOP {sop.sop_number}.",
    }
