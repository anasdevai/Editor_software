"""
Context intelligence for the agentic SOP sidebar: session scope, frustration,
cross-language aliases, full-document overrides, chained refinements, repetition.
"""

from __future__ import annotations

import re
from typing import Any

# --- TASK 3: cross-language section aliases ---

ALIAS_MAP: dict[str, list[str]] = {
    "purpose": ["Zweck", "Ziel", "1. Purpose", "objective"],
    "scope": ["Geltungsbereich", "Anwendungsbereich", "applicability"],
    "responsibilities": ["Verantwortlichkeiten", "roles", "who is responsible"],
    "definitions": ["Definitionen", "glossary", "terms", "begriffe"],
    "deviations": ["Abweichungen", "DEV", "nonconformance"],
    "capas": ["Korrekturmaßnahmen", "corrective actions", "CAPA", "CAPAs"],
    "audits": ["Auditbericht", "audit", "findings", "Audit Findings"],
    "references": ["Referenzen", "related documents", "see also"],
    "zweck": ["Purpose", "Objective", "Ziel", "1. Zweck"],
    "geltungsbereich": ["Scope", "applicability", "Geltungsbereich"],
    "abweichungen": ["Deviations", "DEV", "Abweichungen"],
    "korrekturmaßnahmen": ["CAPAs", "CAPA", "corrective actions", "Korrekturmaßnahmen"],
}

FULL_DOC_SIGNALS = [
    "the whole sop",
    "full sop",
    "entire sop",
    "the full doc",
    "summarize the full sop",
    "tell me about this sop",
    "what is this sop about",
    "overview of this sop",
    "what does this sop cover",
    "give me a summary of this sop",
    "das ganze sop",
    "gesamte sop",
    "überblick",
    "uberblick",
    "whole sop",
    "complete sop",
]

SHORTER_SIGNALS = [
    "shorter", "too long", "make it shorter", "more concise", "brief",
    "compress", "i told you", "still too long", "not shorter",
    "kürzer", "kuerzer", "zu lang", "mach es kürzer", "kürzer bitte", "shorten",
    "shoter", "smaller", "smallier",
]

LONGER_SIGNALS = [
    "too short", "more detail", "expand", "add more", "elaborate",
    "zu kurz", "mehr details", "ausführlicher",
]

WRONG_SIGNALS = [
    "not what i wanted", "that's wrong", "you misunderstood",
    "not like that", "redo", "try again", "das ist falsch", "nochmal",
]

CHAINED_CAPABILITIES = {
    "shorten", "improve", "rewrite", "expand",
    "shorten_section", "improve_section", "rewrite_section", "expand_section",
    "summarize",
}

TRACEABILITY_KIND_ALIASES: dict[str, list[str]] = {
    "capa": ["capa", "capas", "korrekturmaßnahmen", "korrekturmassnahmen", "corrective actions", "CAPA", "CAPAs"],
    "deviation": ["deviation", "deviations", "abweichung", "abweichungen", "DEV", "devs"],
    "audit": ["audit", "audits", "audit findings", "auditbericht", "findings"],
    "decision": ["decision", "decisions", "entscheidung", "entscheidungen", "DEC"],
}

USER_TRACEABILITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "capa": re.compile(r"\bcapas?\b", re.I),
    "deviation": re.compile(r"\b(?:deviations?|abweichungen?|devs?)\b", re.I),
    "audit": re.compile(r"\b(?:audits?|audit\s+findings?|auditbericht)\b", re.I),
    "decision": re.compile(r"\b(?:decisions?|entscheidungen?)\b", re.I),
}


def _traceability_kind_from_label(label: str) -> str | None:
    low = str(label or "").lower()
    if re.search(r"\bcapas?\b|korrektur", low):
        return "capa"
    if re.search(r"\b(?:deviations?|abweichungen?)\b", low):
        return "deviation"
    if re.search(r"\b(?:audits?|audit\s+findings?)\b", low):
        return "audit"
    if re.search(r"\b(?:decisions?|entscheidungen?)\b", low):
        return "decision"
    return None


def _traceability_aliases_for_label(label: str) -> list[str]:
    kind = _traceability_kind_from_label(label)
    if not kind:
        return []
    return list(TRACEABILITY_KIND_ALIASES.get(kind, []))


def default_active_scope() -> dict[str, Any]:
    return {
        "section_id": None,
        "section_label": None,
        "last_action": None,
        "last_result": None,
        "last_result_length": 0,
    }


def build_session_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Hydrate session state from client payload (TASK 1)."""
    ctx = payload.get("assistant_context") if isinstance(payload.get("assistant_context"), dict) else {}
    stored = ctx.get("active_scope") if isinstance(ctx.get("active_scope"), dict) else {}
    last_action = ctx.get("last_action") if isinstance(ctx.get("last_action"), dict) else {}

    active_scope = default_active_scope()
    if stored:
        active_scope.update(
            {
                "section_id": stored.get("section_id"),
                "section_label": stored.get("section_label"),
                "last_action": stored.get("last_action"),
                "last_result": stored.get("last_result"),
                "last_result_length": int(stored.get("last_result_length") or 0),
            }
        )
    elif last_action:
        active_scope.update(
            {
                "section_id": last_action.get("section_id") or last_action.get("sop_id"),
                "section_label": last_action.get("section_name") or last_action.get("section_label"),
                "last_action": last_action.get("action"),
                "last_result": last_action.get("suggested_text_excerpt") or last_action.get("last_result"),
                "last_result_length": _word_count(
                    last_action.get("suggested_text_excerpt") or last_action.get("last_result") or ""
                ),
            }
        )

    instruction_memory = ctx.get("instruction_memory")
    if not isinstance(instruction_memory, list):
        instruction_memory = []

    conversation_history: list[dict[str, Any]] = []
    if isinstance(ctx.get("conversation_history"), list):
        conversation_history = list(ctx["conversation_history"][-24:])
    recent = payload.get("recent_messages")
    if isinstance(recent, list):
        for row in recent[-8:]:
            if isinstance(row, dict) and row.get("content"):
                conversation_history.append(
                    {
                        "role": row.get("role") or "user",
                        "content": str(row.get("content") or "")[:2400],
                    }
                )

    return {
        "active_scope": active_scope,
        "instruction_memory": instruction_memory[-12:],
        "conversation_history": conversation_history[-24:],
        "scope_before_full_doc": active_scope.copy(),
    }


def enrich_sections_with_aliases(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """TASK 3 — attach label_aliases to each parsed section."""
    enriched: list[dict[str, Any]] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        row = dict(section)
        label = str(row.get("label") or row.get("name") or "").strip()
        normalized = re.sub(r"^\d+(?:\.\d+)*[.)\]:-]?\s*", "", label.lower()).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        existing = row.get("label_aliases") if isinstance(row.get("label_aliases"), list) else []
        aliases = list(ALIAS_MAP.get(normalized, []))
        if normalized in ALIAS_MAP:
            aliases.extend([normalized, normalized.title()])
        aliases.extend(existing)
        aliases.extend(_traceability_aliases_for_label(label))
        aliases.append(label)
        row["label_aliases"] = list(dict.fromkeys(a for a in aliases if a))
        enriched.append(row)
    return enriched


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", str(text or ""), flags=re.UNICODE))


def detect_frustration(user_message: str, session: dict[str, Any]) -> dict[str, Any]:
    """TASK 2 — frustration / refinement signals."""
    msg = str(user_message or "").lower()
    active = session.get("active_scope") if isinstance(session.get("active_scope"), dict) else {}
    last_len = int(active.get("last_result_length") or 0)
    last_result = active.get("last_result")

    def _has_any(signals: list[str]) -> bool:
        return any(sig in msg for sig in signals)

    ftype = None
    if _has_any(SHORTER_SIGNALS):
        ftype = "TOO_LONG"
    elif _has_any(LONGER_SIGNALS):
        ftype = "TOO_SHORT"
    elif _has_any(WRONG_SIGNALS):
        ftype = "WRONG_APPROACH"

    target_word_count = None
    if ftype == "TOO_LONG" and last_len > 0:
        target_word_count = max(20, int(last_len * 0.45))
    elif ftype == "TOO_SHORT" and last_len > 0:
        target_word_count = max(last_len + 10, int(last_len * 1.5))

    return {
        "detected": ftype is not None,
        "type": ftype,
        "target_word_count": target_word_count,
        "source_content": last_result if last_result else None,
    }


def detect_repetition(user_message: str, session: dict[str, Any]) -> dict[str, Any]:
    """TASK 6 — repeated near-identical user requests."""
    history = session.get("conversation_history") if isinstance(session.get("conversation_history"), list) else []
    last3 = [
        str(row.get("content") or "").lower().strip()
        for row in history
        if isinstance(row, dict) and str(row.get("role") or "").lower() == "user"
    ][-3:]
    current = str(user_message or "").lower().strip()

    def semantic_overlap(a: str, b: str) -> float:
        words_a = set(re.findall(r"\b\w+\b", a, flags=re.UNICODE))
        words_b = set(re.findall(r"\b\w+\b", b, flags=re.UNICODE))
        if not words_a or not words_b:
            return 0.0
        shared = len(words_a & words_b)
        return shared / max(len(words_a), len(words_b))

    is_repetition = any(semantic_overlap(current, prior) > 0.6 for prior in last3 if prior)
    instruction = None
    if is_repetition:
        instruction = (
            "User repeated this request. Prior output was unsatisfactory. "
            "Do NOT produce the same output again. Change approach completely: "
            "if you used bullets use prose; if prose use a table or numbered list; "
            "lead with the most critical point first; change abstraction level."
        )
    return {
        "repetition_detected": is_repetition,
        "repetition_instruction": instruction,
    }


def resolve_scope_from_message(
    user_message: str,
    sections: list[dict[str, Any]],
    *,
    has_editor_selection: bool = False,
) -> dict[str, Any] | None:
    """
    TASK 3 + TASK 4 — resolve target level before LLM classification.
    Returns dict with level, section_id, section_label, resolved_from, line_number, record_id.
    """
    msg = str(user_message or "")
    lower = msg.lower()

    # TASK 4 — full document (chat / summarize overview)
    if any(sig in lower for sig in FULL_DOC_SIGNALS):
        return {
            "level": "full",
            "section_id": None,
            "section_label": "Full Document",
            "resolved_from": "FULL_DOC",
            "target_scope": "full_document",
        }

    # Specific line
    line_match = re.search(r"\bline\s+(\d{1,4})\b", lower, re.I)
    if line_match:
        return {
            "level": "line",
            "section_id": f"line_{line_match.group(1)}",
            "section_label": f"Line {line_match.group(1)}",
            "resolved_from": "LINE",
            "target_scope": "selection",
            "line_number": int(line_match.group(1)),
        }

    # Specific record (CAPA-IT-011, DEV-IT-025, …)
    record_match = re.search(r"\b((?:DEV|CAPA|AUD|DEC)-[A-Z0-9]+-\d+)\b", msg, re.I)
    if record_match and re.search(r"\b(?:rewrite|improve|summarize|explain|gap)\b", lower, re.I):
        return {
            "level": "record",
            "section_id": record_match.group(1).upper(),
            "section_label": record_match.group(1).upper(),
            "resolved_from": "RECORD_ID",
            "target_scope": "selection",
            "record_id": record_match.group(1).upper(),
        }

    # Numbered sub-heading (1.2.3 …)
    sub_match = re.search(
        r"\b(?:section\s+)?(\d+(?:\.\d+)+)(?:[.)\]:-]|\s+)",
        msg,
        re.I,
    )
    if sub_match:
        return {
            "level": "sub_section",
            "section_id": sub_match.group(1),
            "section_label": sub_match.group(1),
            "resolved_from": "SUB_SECTION",
            "target_scope": "section",
        }

    enriched = enrich_sections_with_aliases(sections)

    # TASK 3 — alias match on label + label_aliases
    for section in enriched:
        label = str(section.get("label") or "").strip()
        if not label:
            continue
        all_labels = [label, *(section.get("label_aliases") or [])]
        for candidate in all_labels:
            cand = str(candidate or "").strip()
            if len(cand) < 3:
                continue
            if cand.lower() in lower or re.search(rf"\b{re.escape(cand.lower())}\b", lower, re.I):
                return {
                    "level": "section",
                    "section_id": section.get("id") or label,
                    "section_label": label,
                    "resolved_from": "ALIAS_MATCH",
                    "target_scope": "section",
                }

    # Traceability blocks (CAPAs, Deviations, …) — "rewrite the capa section"
    for section in enriched:
        label = str(section.get("label") or "").strip()
        if not label:
            continue
        kind = _traceability_kind_from_label(label)
        if not kind:
            continue
        pattern = USER_TRACEABILITY_PATTERNS.get(kind)
        if pattern and pattern.search(msg):
            return {
                "level": "section",
                "section_id": section.get("id") or label,
                "section_label": label,
                "resolved_from": "ALIAS_MATCH",
                "target_scope": "section",
            }

    if has_editor_selection and re.search(r"\b(?:selected|highlighted|this\s+paragraph|this\s+text)\b", lower, re.I):
        return {
            "level": "selection",
            "section_id": None,
            "section_label": "Selected text",
            "resolved_from": "SELECTION",
            "target_scope": "selection",
        }

    return None


def build_source_content_override(
    capability: str | None,
    session: dict[str, Any],
    *,
    resolved_section_id: str | None = None,
) -> dict[str, Any]:
    """TASK 5 — chained refinements use last LLM output, not original SOP."""
    active = session.get("active_scope") if isinstance(session.get("active_scope"), dict) else {}
    cap = str(capability or "").strip().lower().replace("-", "_")
    section_id = str(active.get("section_id") or "").strip()
    resolved_id = str(resolved_section_id or "").strip()
    last_result = active.get("last_result")

    same_section = (
        not resolved_id
        or not section_id
        or resolved_id == section_id
        or str(active.get("section_label") or "").lower() in resolved_id.lower()
    )

    if cap in CHAINED_CAPABILITIES and last_result and same_section:
        wc = int(active.get("last_result_length") or _word_count(str(last_result)))
        return {
            "enabled": True,
            "content": last_result,
            "word_count": wc,
            "note": "Operate on this content, not the original SOP section.",
        }
    return {"enabled": False}


def prepare_message_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Run pre-LLM intelligence (tasks 2, 4, 6 + scope resolution)."""
    message = str(payload.get("message") or payload.get("question") or "").strip()
    session = build_session_from_payload(payload)
    ctx = payload.get("assistant_context") if isinstance(payload.get("assistant_context"), dict) else {}
    current_sop = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}
    sections = current_sop.get("sections") if isinstance(current_sop.get("sections"), list) else []
    sections = enrich_sections_with_aliases(sections)

    frustration = detect_frustration(message, session)
    repetition = detect_repetition(message, session)
    resolved_scope = resolve_scope_from_message(
        message,
        sections,
        has_editor_selection=bool(payload.get("has_editor_selection")),
    )

    early_response = None
    if resolved_scope and resolved_scope.get("level") == "full":
        early_response = {
            "flow": "chat",
            "action": None,
            "target_scope": "full_document",
            "section_hint": None,
            "linked_entity_types": [],
            "constraints": {},
            "clarification_question": None,
            "confidence": 0.95,
            "reasoning": "full_doc_signal_override",
            "resolved_scope": resolved_scope,
            "frustration_signal": frustration,
            "repetition_detected": repetition.get("repetition_detected"),
            "repetition_instruction": repetition.get("repetition_instruction"),
            "preserve_active_scope": True,
        }

    return {
        "session": session,
        "sections": sections,
        "frustration_signal": frustration,
        "repetition_detected": repetition.get("repetition_detected"),
        "repetition_instruction": repetition.get("repetition_instruction"),
        "resolved_scope": resolved_scope,
        "early_response": early_response,
        "active_scope": session["active_scope"],
        "instruction_memory": session["instruction_memory"],
    }


def _constraints_from_frustration(frustration: dict[str, Any], existing: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(existing or {})
    if not frustration.get("detected"):
        return out
    ftype = frustration.get("type")
    if ftype == "TOO_LONG":
        out["length"] = "shorter"
        twc = frustration.get("target_word_count")
        if twc:
            out["word_count"] = twc
    elif ftype == "TOO_SHORT":
        out["length"] = "longer"
        twc = frustration.get("target_word_count")
        if twc:
            out["word_count"] = twc
    return out


def apply_prep_to_classification(result: dict[str, Any], prep: dict[str, Any]) -> dict[str, Any]:
    """Merge scope/frustration/repetition into classifier output."""
    out = dict(result)
    resolved = prep.get("resolved_scope")
    frustration = prep.get("frustration_signal") or {}

    if resolved and resolved.get("level") == "full":
        out["flow"] = "chat"
        out["action"] = None
        out["target_scope"] = "full_document"
        out["section_hint"] = None
    elif resolved and out.get("flow") == "editor_action":
        out["target_scope"] = resolved.get("target_scope") or out.get("target_scope")
        if resolved.get("section_label") and resolved.get("level") in {"section", "sub_section", "record"}:
            out["section_hint"] = resolved.get("section_label")
        if resolved.get("level") == "line":
            out["target_scope"] = "selection"
            out["line_number"] = resolved.get("line_number")
        if resolved.get("level") == "record":
            out["target_scope"] = "selection"
            out["record_id"] = resolved.get("record_id")

    if frustration.get("detected") and out.get("flow") == "editor_action":
        out["constraints"] = _constraints_from_frustration(frustration, out.get("constraints"))
        if frustration.get("type") == "TOO_LONG" and not out.get("action"):
            out["action"] = "rewrite"
        if frustration.get("type") == "TOO_LONG" and out.get("action") in {None, "improve"}:
            prev = str((prep.get("session") or {}).get("active_scope", {}).get("last_action") or "")
            out["action"] = "rewrite" if "rewrite" in prev else (out.get("action") or "rewrite")
        if not out.get("target_scope"):
            prev_label = (prep.get("session") or {}).get("active_scope", {}).get("section_label")
            if prev_label:
                out["target_scope"] = "section"
                out["section_hint"] = prev_label

    if prep.get("repetition_detected"):
        out["repetition_detected"] = True
        out["repetition_instruction"] = prep.get("repetition_instruction")

    out["frustration_signal"] = frustration
    out["resolved_scope"] = resolved
    out["source_content_override"] = build_source_content_override(
        out.get("action"),
        prep.get("session") or {},
        resolved_section_id=str((resolved or {}).get("section_id") or ""),
    )
    return out


def persist_session_after_response(
    session: dict[str, Any],
    response: dict[str, Any],
    *,
    user_message: str = "",
    preserve_scope: bool = False,
) -> dict[str, Any]:
    """TASK 1 — update session after each turn."""
    active = dict(session.get("active_scope") or default_active_scope())
    scope_before = session.get("scope_before_full_doc") or active.copy()

    if preserve_scope and response.get("flow") == "chat" and response.get("target_scope") == "full_document":
        active = dict(scope_before)
    else:
        updated = response.get("updated_active_scope")
        if isinstance(updated, dict):
            active.update(updated)
        elif response.get("flow") == "editor_action":
            active["last_action"] = response.get("action")
            label = response.get("section_hint") or active.get("section_label")
            if label:
                active["section_label"] = label
                active["section_id"] = label
            result_text = ""
            actions = response.get("actions")
            if isinstance(actions, list) and actions:
                result_text = str(actions[0].get("result") or "")
            if not result_text:
                result_text = str(response.get("last_result") or "")
            if result_text:
                active["last_result"] = result_text
                active["last_result_length"] = _word_count(result_text)

    instruction_memory = list(session.get("instruction_memory") or [])
    mem = response.get("instruction_memory")
    if isinstance(mem, list) and mem:
        instruction_memory = mem[-12:]

    conversation_history = list(session.get("conversation_history") or [])
    if user_message:
        conversation_history.append({"role": "user", "content": user_message[:2400]})
    assistant_summary = response.get("assistant_message") or response.get("reasoning") or ""
    if assistant_summary:
        conversation_history.append(
            {
                "role": "assistant",
                "content": str(assistant_summary)[:2400],
                "scope_used": active.get("section_id"),
                "action_used": response.get("action"),
                "output_word_count": active.get("last_result_length"),
            }
        )

    return {
        "active_scope": active,
        "instruction_memory": instruction_memory[-12:],
        "conversation_history": conversation_history[-24:],
    }


def finalize_classify_response(
    raw: dict[str, Any],
    prep: dict[str, Any],
    *,
    user_message: str = "",
) -> dict[str, Any]:
    """Apply prep, build updated_active_scope, persist session snapshot for client."""
    if prep.get("early_response"):
        out = dict(prep["early_response"])
    else:
        out = apply_prep_to_classification(dict(raw), prep)

    session = prep.get("session") or {}
    preserve = bool(out.get("preserve_active_scope"))

    updated_scope = dict(session.get("active_scope") or default_active_scope())
    if out.get("flow") == "editor_action":
        resolved = out.get("resolved_scope") or {}
        updated_scope["section_label"] = (
            out.get("section_hint")
            or resolved.get("section_label")
            or updated_scope.get("section_label")
        )
        updated_scope["section_id"] = (
            resolved.get("section_id")
            or updated_scope.get("section_id")
            or updated_scope.get("section_label")
        )
        updated_scope["last_action"] = out.get("action")
    elif preserve:
        updated_scope = dict(session.get("scope_before_full_doc") or updated_scope)

    out["updated_active_scope"] = {
        "section_id": updated_scope.get("section_id"),
        "section_label": updated_scope.get("section_label"),
        "last_action": updated_scope.get("last_action"),
        "last_result": updated_scope.get("last_result"),
        "last_result_length": int(updated_scope.get("last_result_length") or 0),
    }
    out["active_scope"] = updated_scope
    out["instruction_memory"] = list(session.get("instruction_memory") or [])

    persisted = persist_session_after_response(
        session,
        out,
        user_message=user_message,
        preserve_scope=preserve,
    )
    out["session_snapshot"] = persisted
    out.pop("preserve_active_scope", None)
    return out
