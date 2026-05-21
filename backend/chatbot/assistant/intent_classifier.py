"""
LLM-based intent classification for the unified KL/KI Assistant chat panel.

Classifies whether a user message should be answered via RAG chat, routed to an
editor action, or clarified — without relying on fixed keyword lists.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Literal, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from chatbot.llm.provider import create_chat_llm

logger = logging.getLogger(__name__)

FlowType = Literal["chat", "editor_action", "clarify"]
ActionType = Literal[
    "rewrite",
    "improve",
    "gap_check",
    "summarize",
    "analyze",
    "compare",
    "read",
]
TargetScope = Literal["selection", "section", "full_document", "linked_context"]
LengthHint = Literal["shorter", "longer", "unchanged"]


class IntentConstraints(BaseModel):
    tone: Optional[str] = None
    word_count: Optional[int] = Field(default=None, ge=1, le=10000)
    line_count: Optional[int] = Field(default=None, ge=1, le=500)
    length: Optional[LengthHint] = None
    language: Optional[str] = None
    detail_level: Optional[str] = None
    format: Optional[str] = None


class AssistantIntentResult(BaseModel):
    flow: FlowType
    action: Optional[ActionType] = None
    target_scope: Optional[TargetScope] = None
    section_hint: Optional[str] = None
    linked_entity_types: list[str] = Field(default_factory=list)
    constraints: IntentConstraints = Field(default_factory=IntentConstraints)
    clarification_question: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: Optional[str] = None


CLASSIFIER_PROMPT = """\
You are the intent router for a pharmaceutical QA SOP assistant (English and German).

Given the user message and context, return ONLY valid JSON (no markdown) matching this schema:

{{
  "flow": "chat" | "editor_action" | "clarify",
  "action": null | "rewrite" | "improve" | "gap_check" | "summarize" | "analyze" | "compare" | "read",
  "target_scope": null | "selection" | "section" | "full_document" | "linked_context",
  "section_hint": null | string,
  "linked_entity_types": [],
  "constraints": {{
    "tone": null | string,
    "word_count": null | integer,
    "line_count": null | integer,
    "length": null | "shorter" | "longer" | "unchanged",
    "language": null | "en" | "de" | string,
    "detail_level": null | string,
    "format": null | string
  }},
  "clarification_question": null | string,
  "confidence": 0.0-1.0,
  "reasoning": string
}}

ROUTING RULES (semantic — infer meaning; users may paraphrase in English or German):

A) flow = "chat" — default for questions and explanations:
   - Questions (including with "?"), "what/why/how/welche/was/wie/warum", requests to explain, list, compare records, or summarize/explain **information** without changing the open SOP body.
   - Explaining or listing linked CAPAs, deviations, audits, decisions, or SOP meaning **without** asking to rewrite/improve/shorten the document text.
   - If the user only wants QA/record context as an answer, use flow="chat". You may still set linked_entity_types to steer retrieval.

B) flow = "editor_action" — user wants the **open SOP text** changed or generated in-place:
   - Rewrite, rephrase, shorten, expand, formalize, bulletize, translate output **into the document**, tighten wording, compliance polish, gap/risk review **on the prose**, section-level or full-document edits.
   - Summarize **into** the document (replace or condense a section) is editor_action with action "summarize" and the right target_scope.
   - Requires has_active_sop=true for document edits. If has_active_sop=false but they clearly want document work → flow="clarify" (ask to open an SOP).

C) flow = "clarify" — one short follow-up when necessary:
   - Cannot tell if they want a **chat answer** vs **editing the SOP**, or cannot tell **which target** (selection vs section vs whole document) and it matters.
   - Do **not** clarify if a safe default exists: e.g. has_editor_selection=true and the message refers to "this" / "the selected text" → target_scope "selection"; a clearly named section (Purpose, Scope, …) → target_scope "section" with section_hint.

TARGET_SCOPE (editor_action):
- "selection": non-empty editor selection exists AND the user refers to the selection, this paragraph, highlighted text, or similar. If has_editor_selection=true and the target is vague ("this", "here"), prefer "selection".
- "section": a named heading/section in the SOP; set section_hint to the best heading label (Purpose, Scope, Procedure, Responsibilities, Zweck, Geltungsbereich, Verfahren, …). The app will expand to the **full section body**, not the heading line alone.
- "full_document": entire SOP / whole document / "die ganze SOP".
- "linked_context": user wants the action focused on **blocks that reference** linked CAPAs, deviations, audits, or decisions (set linked_entity_types). Use for gap_check on registers; for pure Q&A about links use "chat" instead.

ACTION (editor_action only):
- rewrite: restructure/rephrase wording (not necessarily shorter).
- improve: clarity, compliance tone, professionalism, fix awkward phrasing.
- gap_check: risks, gaps, missing controls, Lückenanalyse on the chosen target.
- summarize: produce a shorter version; set word_count and/or line_count when the user asks (e.g. "100 words", "two lines", "vier Zeilen").
- analyze: deep structure/compliance analysis (often whole document).
- compare: version comparison.
- read: show/confirm current document content only.

CONSTRAINTS (extract when mentioned):
- length: "shorter" | "longer" | "unchanged" for concise vs expand.
- word_count: explicit word limits (e.g. 100, 200).
- line_count: explicit line limits (e.g. 2, 4, "two lines", "vier Zeilen").
- tone, language (en/de), format, detail_level as appropriate.

CONTEXT:
- has_active_sop: {has_active_sop}
- has_editor_selection: {has_editor_selection}
- route: {route}
- active_sop_title: {active_sop_title}
- active_sop_number: {active_sop_number}

When target_scope is section, the editor applies the action to the full section body under that heading.

USER MESSAGE:
{user_message}
"""

_VALID_ACTIONS = {
    "rewrite",
    "improve",
    "gap_check",
    "summarize",
    "analyze",
    "compare",
    "read",
}
_VALID_SCOPES = {"selection", "section", "full_document", "linked_context"}
_VALID_LINKED = {"capas", "deviations", "audits", "decisions", "related_sops"}
_VALID_FLOWS = {"chat", "editor_action", "clarify"}


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty classifier response")
    if "```" in raw:
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
        else:
            raw = raw.split("```", 1)[1].split("```", 1)[0].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)


def _normalize_payload(
    data: dict[str, Any],
    *,
    has_active_sop: bool,
    has_editor_selection: bool = False,
) -> AssistantIntentResult:
    flow = str(data.get("flow") or "chat").strip().lower()
    if flow not in _VALID_FLOWS:
        flow = "chat"

    action_raw = data.get("action")
    action = None
    if isinstance(action_raw, str) and action_raw.strip().lower() in _VALID_ACTIONS:
        action = action_raw.strip().lower()  # type: ignore[assignment]

    scope_raw = data.get("target_scope")
    target_scope = None
    if isinstance(scope_raw, str) and scope_raw.strip().lower() in _VALID_SCOPES:
        target_scope = scope_raw.strip().lower()  # type: ignore[assignment]

    section_hint = data.get("section_hint")
    if isinstance(section_hint, str):
        section_hint = section_hint.strip() or None
    else:
        section_hint = None

    linked: list[str] = []
    raw_linked = data.get("linked_entity_types")
    if isinstance(raw_linked, list):
        for item in raw_linked:
            if isinstance(item, str):
                key = item.strip().lower()
                if key in _VALID_LINKED:
                    linked.append(key)

    constraints_raw = data.get("constraints")
    constraints = IntentConstraints()
    if isinstance(constraints_raw, dict):
        try:
            constraints = IntentConstraints.model_validate(constraints_raw)
        except ValidationError:
            constraints = IntentConstraints()

    clarification = data.get("clarification_question")
    if isinstance(clarification, str):
        clarification = clarification.strip() or None
    else:
        clarification = None

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    reasoning = data.get("reasoning")
    if isinstance(reasoning, str):
        reasoning = reasoning.strip() or None
    else:
        reasoning = None

    # Enforce consistency
    if flow == "chat":
        action = None
        target_scope = None
        clarification = None
    elif flow == "clarify":
        action = None
        if not clarification:
            clarification = (
                "Soll ich die Anfrage als Antwort im Chat beantworten oder als Aktion im geöffneten SOP-Editor ausführen?"
                if re.search(r"\b(der|die|das|und|nicht|für|bitte)\b", str(data.get("_user_message") or ""), re.I)
                else "Should I answer in chat or run this on the open SOP in the editor?"
            )
    elif flow == "editor_action":
        clarification = None
        if not has_active_sop:
            flow = "clarify"  # type: ignore[assignment]
            action = None
            clarification = (
                "Bitte öffnen Sie zuerst eine SOP im Editor, damit ich diese Aktion ausführen kann."
            )
        elif not action:
            flow = "chat"  # type: ignore[assignment]
        else:
            # Default target: explicit non-empty selection + vague "this/here" → selection
            content_actions = {"rewrite", "improve", "summarize", "gap_check"}
            if (
                action in content_actions
                and not target_scope
                and not section_hint
                and has_editor_selection
            ):
                target_scope = "selection"  # type: ignore[assignment]
            elif (
                action in content_actions
                and not target_scope
                and not section_hint
                and not has_editor_selection
                and confidence < 0.55
            ):
                flow = "clarify"  # type: ignore[assignment]
                action = None
                clarification = (
                    "Meinen Sie die gesamte SOP, einen bestimmten Abschnitt (z. B. Zweck, Geltungsbereich) "
                    "oder nur die aktuelle Auswahl im Editor?"
                    if re.search(r"\b(der|die|das|und|nicht|für|bitte|mir|hier)\b", str(data.get("_user_message") or ""), re.I)
                    else "Should I apply this to the full SOP, a specific section (e.g. Purpose, Scope), or only the current selection?"
                )

    return AssistantIntentResult(
        flow=flow,  # type: ignore[arg-type]
        action=action,
        target_scope=target_scope,
        section_hint=section_hint,
        linked_entity_types=linked,
        constraints=constraints,
        clarification_question=clarification,
        confidence=confidence,
        reasoning=reasoning,
    )


def _heuristic_fallback(message: str, *, has_active_sop: bool) -> AssistantIntentResult:
    """When the LLM is unavailable, prefer safe chat routing (no keyword-based editor routing)."""
    _ = (message, has_active_sop)
    return AssistantIntentResult(
        flow="chat",
        confidence=0.2,
        reasoning="classifier_unavailable_default_chat",
    )


def classify_assistant_intent(
    message: str,
    *,
    has_active_sop: bool = False,
    has_editor_selection: bool = False,
    route: str = "",
    active_sop_title: str = "",
    active_sop_number: str = "",
) -> AssistantIntentResult:
    """Classify user intent for the KL/KI Assistant using a small LLM call."""
    user_message = (message or "").strip()
    if not user_message:
        return AssistantIntentResult(flow="chat", confidence=1.0, reasoning="empty")

    max_chars = int(os.getenv("ASSISTANT_INTENT_MAX_CHARS", "4000"))
    if max_chars > 0 and len(user_message) > max_chars:
        user_message = user_message[: max_chars - 1].rstrip() + "…"

    llm = create_chat_llm(
        temperature=0.0,
        max_output_tokens=int(os.getenv("ASSISTANT_INTENT_MAX_TOKENS", "512")),
        max_retries=1,
    )
    prompt = ChatPromptTemplate.from_template(CLASSIFIER_PROMPT)
    chain = prompt | llm | StrOutputParser()

    try:
        raw = chain.invoke(
            {
                "user_message": user_message,
                "has_active_sop": str(bool(has_active_sop)).lower(),
                "has_editor_selection": str(bool(has_editor_selection)).lower(),
                "route": route or "-",
                "active_sop_title": active_sop_title or "-",
                "active_sop_number": active_sop_number or "-",
            }
        )
        data = _extract_json_object(raw)
        data["_user_message"] = user_message
        result = _normalize_payload(
            data,
            has_active_sop=has_active_sop,
            has_editor_selection=has_editor_selection,
        )
        logger.info(
            "[assistant-intent] flow=%s action=%s scope=%s confidence=%.2f",
            result.flow,
            result.action,
            result.target_scope,
            result.confidence,
        )
        return result
    except Exception as exc:
        logger.warning("[assistant-intent] LLM classification failed: %s", exc)
        return _heuristic_fallback(user_message, has_active_sop=has_active_sop)
