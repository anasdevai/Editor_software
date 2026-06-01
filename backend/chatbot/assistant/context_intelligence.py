"""
Context intelligence for the agentic SOP sidebar: session scope, frustration,
cross-language aliases, full-document overrides, chained refinements, repetition.
"""

from __future__ import annotations

import html as html_module
import json
import os
import re
from typing import Any


def use_llm_orchestrator() -> bool:
    """When true, intent routing is decided by the classifier LLM; rules only enrich/guard."""
    return os.getenv("ASSISTANT_LLM_ORCHESTRATOR", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

_DIFF_STOPWORDS = frozenset({
    "that", "this", "with", "from", "have", "were", "been", "will", "shall",
    "must", "should", "would", "could", "into", "than", "then", "when", "where",
    "which", "their", "there", "these", "those", "such", "only", "also", "more",
    "most", "some", "same", "other", "about", "after", "before", "between",
    "dass", "dies", "diese", "dieser", "damit", "sowie", "oder", "aber", "wenn",
    "wird", "werden", "wurde", "wurden", "kann", "können", "muss", "müssen",
    "soll", "sollen", "darf", "dürfen", "nach", "über", "unter", "durch", "beim",
    "beim", "eine", "einer", "eines", "einem", "einen",
})

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

# Read-only full-document chat (summarize / explain) — not used for rewrite/improve scope.
READ_ONLY_FULL_DOC_SIGNALS = [
    "summarize the full sop",
    "summarize this sop",
    "tell me about this sop",
    "what is this sop about",
    "overview of this sop",
    "what does this sop cover",
    "give me a summary of this sop",
    "überblick",
    "uberblick",
    "zusammenfass",
    "kurzfass",
]

_FULL_DOCUMENT_SCOPE_RE = re.compile(
    r"\b(?:"
    r"(?:full|whole|entire|complete|gesamte?|ganze?|komplette?)\s+(?:sop|document|doc)"
    r"|(?:this|the|current|open|active)\s+sop"
    r"|das\s+ganze\s+sop|gesamte\s+sop|ganze\s+sop|komplettes\s+sop"
    r"|(?:diese|die|der)\s+(?:gesamte?|komplette?|aktuelle?)\s+sop"
    r"|standard\s+operating\s+procedures?"
    r")\b",
    re.IGNORECASE,
)

_EDIT_VERB_RE = re.compile(
    r"\b(?:rewrite|re-?write|improve|revise|verbesser\w*|umschreib|überarbeit|ueberarbeit)\b",
    re.IGNORECASE,
)

_GAP_CHECK_COMMAND_RE = re.compile(
    r"\b(?:"
    r"gap[\s-]*checks?|gap\s+analysis|what\s+(?:is|are)\s+the\s+gaps?|"
    r"gaps?\s+in|gap\s+checks?\s+in|"
    r"(?:find|identify|check|analyse|analyze|review|run|perform|führe|finde|prüfe)\s+"
    r"(?:the\s+)?(?:gap\s*checks?|gaps?|compliance\s+gaps?|lücken)|"
    r"compliance\s+(?:gap|check|review|audit)|qa\s+review|identify\s+risks?|"
    r"missing\s+controls|audit[\s-]?ready\s+check|"
    r"l(?:ü|ue|u)cken[\s-]?(?:analyse|pr(?:ü|ue)fung|check)?|"
    r"(?:finde|zeige|identifiziere|pr(?:ü|ue)fe)\s+(?:the\s+)?(?:die\s+)?(?:gap\s*checks?|gaps?|lücken)|"
    r"welche\s+(?:gap\s*checks?|gaps?|lücken)|risiken\s+und\s+l(?:ü|ue|u)cken"
    r")\b",
    re.IGNORECASE,
)

_SUMMARIZE_VERB_RE = re.compile(
    r"\b(?:summarize|summary|zusammenfass\w*|kurzfass\w*|fasse|verkürz\w*|verkuerz\w*|kürze\w*|kuerze\w*)\b|"
    r"\bfasse\b[^.?\n]{0,120}\bzusammen\b",
    re.IGNORECASE,
)

_SECTION_TARGET_RE = re.compile(
    r"\b(?:"
    r"(?:this|that|the|den|die|das|diesen|diesem|dieser|aktuelle[nr]?)\s+"
    r"(?:section|abschnitt|paragraph|heading|überschrift|ueberschrift)"
    r"|(?:abschnitt|section)\s+[\"']?[\wäöüÄÖÜß0-9./&()-]{2,80}"
    r"|(?:im\s+)?abschnitt\s+[\"']?[\wäöüÄÖÜß0-9./&()-]{2,80}"
    r"|(?:section|abschnitt)\s+\d+(?:\.\d+)*"
    r")\b",
    re.IGNORECASE,
)


def is_gap_check_command(message: str) -> bool:
    """True when the user requests a gap / compliance audit on SOP content."""
    q = str(message or "").strip()
    if not q:
        return False
    if _GAP_CHECK_COMMAND_RE.search(q):
        return True
    if re.search(
        r"\b(?:find|identify|check|analyse|analyze|review)\b[\s\S]{0,80}\b(?:gap\s*checks?|gaps?|lücken)\b",
        q,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:gap\s*checks?|gaps?|lücken)\b[\s\S]{0,80}\b(?:section|abschnitt|capas?|deviations?|sop)\b",
        q,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:l(?:ü|ue|u)cken|risiken|compliance)\b.*\b(?:sop|dokument|document|diese(?:r|m|s)?|aktuelle?)\b",
        q,
        re.I,
    ):
        return True
    if re.search(
        r"\b(?:sop|dokument|document|diese(?:r|m|s)?|aktuelle?)\b.*\b(?:l(?:ü|ue|u)cken|risiken)\b",
        q,
        re.I,
    ):
        return True
    return False


def is_summarize_chat_query(message: str) -> bool:
    """All summarize requests are answered in the sidebar chat (never inline in the editor)."""
    return bool(_SUMMARIZE_VERB_RE.search(str(message or "")))


def is_summarize_into_document(
    message: str,
    *,
    resolved_scope: dict[str, Any] | None = None,
) -> bool:
    """Summarize is chat-only; never replace text in the editor."""
    return False


def message_targets_full_document(message: str) -> bool:
    """User refers to the whole open SOP (edit or read)."""
    msg = str(message or "")
    if _FULL_DOCUMENT_SCOPE_RE.search(msg):
        return True
    if _EDIT_VERB_RE.search(msg) and re.search(
        r"\b(?:standard\s+operating\s+procedures?|operating\s+procedure)\b",
        msg,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:revise|rewrite|improve|umschreib|verbesser\w*|überarbeit|ueberarbeit)\s+"
        r"(?:the\s+)?(?:standard\s+operating\s+)?(?:procedure|sop)\b",
        msg,
        re.IGNORECASE,
    ):
        return True
    return False

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
    "capa": ["capa", "capas", "caps", "korrekturmaßnahmen", "korrekturmassnahmen", "corrective actions", "CAPA", "CAPAs"],
    "deviation": ["deviation", "deviations", "abweichung", "abweichungen", "DEV", "devs"],
    "audit": ["audit", "audits", "audit findings", "auditbericht", "findings"],
    "decision": ["decision", "decisions", "entscheidung", "entscheidungen", "DEC"],
}

USER_TRACEABILITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "capa": re.compile(r"\b(?:capas?|caps)\b", re.I),
    "deviation": re.compile(r"\b(?:deviations?|abweichungen?|devs?)\b", re.I),
    "audit": re.compile(r"\b(?:audits?|audit\s+findings?|auditbericht)\b", re.I),
    "decision": re.compile(r"\b(?:decisions?|entscheidungen?)\b", re.I),
}


def _traceability_kind_from_label(label: str) -> str | None:
    low = str(label or "").lower()
    if re.search(r"\b(?:capas?|caps)\b|korrektur", low):
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


def _leading_section_number(text: str) -> str:
    match = re.match(r"^(\d+(?:\.\d+)*)", str(text or "").strip())
    return match.group(1) if match else ""


def _section_root_label(text: str) -> str:
    root = re.sub(r"^\d+(?:\.\d+)*[.)\]:-]?\s*", "", str(text or "").strip(), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", root).lower().strip()


def _semantic_roots_for_token(token: str) -> set[str]:
    root = _section_root_label(token)
    roots: set[str] = set()
    if root:
        roots.add(root)
    for key, aliases in ALIAS_MAP.items():
        pool = {key.lower(), *[str(a).lower() for a in aliases]}
        if root in pool or any(root and (root in item or item in root) for item in pool):
            roots.add(key.lower())
            for alias in aliases:
                alias_root = _section_root_label(str(alias))
                if alias_root:
                    roots.add(alias_root)
    return {item for item in roots if item and len(item) >= 3}


def _semantic_section_tokens_match(token: str, label: str) -> bool:
    left = _semantic_roots_for_token(token)
    right = _semantic_roots_for_token(label)
    return bool(left & right)


def user_requests_entire_section_body(message: str) -> bool:
    """User wants the full section content under a heading, not a word/heading/selection."""
    msg = str(message or "")
    return bool(
        re.search(
            r"\b(?:entire|whole|full|complete|gesamte?|komplette?|ganze?)\s+(?:section|abschnitt)\b",
            msg,
            re.IGNORECASE,
        )
        or re.search(r"\b(?:gesamten?|kompletten?)\s+abschnitt\b", msg, re.IGNORECASE)
    )


def extract_section_name_token_from_message(message: str) -> str:
    """Pull section name from phrases like 'entire section of Zweck' or 'rewrite the Zweck section'."""
    msg = str(message or "").strip()
    if not msg:
        return ""
    patterns = [
        r"\b(?:entire|whole|full|complete|gesamte?|komplette?|ganze?)\s+(?:section|abschnitt)\s+"
        r"(?:of|von|für|fur|for)?\s*[\"']?([^\"'.?!\n]{2,80})",
        r"\b(?:rewrite|re-?write|improve|revise|umschreib\w*|verbesser\w*)\s+"
        r"(?:the\s+)?(?:entire|whole|full|gesamte?|komplette?)?\s*(?:section|abschnitt)\s+"
        r"(?:of|von|für|fur|for)?\s*[\"']?([^\"'.?!\n]{2,80})",
        r"\b(?:section|abschnitt)\s+(?:of|von|für|fur|for)\s+[\"']?([^\"'.?!\n]{2,80})",
        r"\b(?:the\s+)?([A-Za-zÀ-ÿ][\wäöüÄÖÜß0-9./&()\- ]{1,60}?)\s+(?:section|abschnitt)\b",
        r"\b(?:im\s+)?(?:abschnitt|section)\s+[\"']?([\wäöüÄÖÜß0-9./&()\-]{2,80}?)(?:[\"']|\s*$|[.!?])",
    ]
    stop = {
        "the",
        "this",
        "that",
        "entire",
        "whole",
        "full",
        "complete",
        "gesamte",
        "komplette",
        "rewrite",
        "improve",
        "revise",
    }
    for pattern in patterns:
        match = re.search(pattern, msg, re.IGNORECASE)
        if not match:
            continue
        token = re.sub(r"\s+", " ", match.group(1).strip(" .:-\"'"))
        if token and token.lower() not in stop and len(token) >= 2:
            return token
    known = re.search(
        r"\b(zweck|zwect|sweck|purpose|scope|geltungsbereich|procedure|verfahren|"
        r"responsibilities|verantwortlichkeiten|capas?|caps|deviations?|abweichungen?|"
        r"decisions?|entscheidungen?|audits?|definitions?)\b",
        msg,
        re.IGNORECASE,
    )
    return known.group(1) if known else ""


def enforce_gap_check_editor_route(
    out: dict[str, Any],
    user_message: str,
    prep: dict[str, Any],
) -> dict[str, Any]:
    """
    Gap/compliance analysis on open SOP text → editor_action gap_check (hybrid RAG runs inside /api/ai/action).
    Do not answer only via sidebar RAG listing of CAPA/deviation records.
    """
    if not prep.get("has_active_sop") or not is_gap_check_command(user_message):
        return out
    merged = dict(out)
    msg = str(user_message or "")
    sections = prep.get("sections") if isinstance(prep.get("sections"), list) else []

    merged["flow"] = "editor_action"
    merged["action"] = "gap_check"
    merged["requires_confirmation"] = True
    merged["requires_selection"] = False
    merged["reasoning"] = merged.get("reasoning") or "gap_check_on_open_sop_section"

    linked: list[str] = []
    if re.search(r"\b(?:capas?|caps|korrektur)\b", msg, re.I):
        linked.append("capas")
    if re.search(r"\b(?:deviations?|abweichungen?)\b", msg, re.I):
        linked.append("deviations")
    if re.search(r"\b(?:audits?|audit\s+findings?)\b", msg, re.I):
        linked.append("audits")
    if re.search(r"\b(?:decisions?|entscheidungen?)\b", msg, re.I):
        linked.append("decisions")
    if linked:
        merged["linked_entity_types"] = linked

    token = extract_section_name_token_from_message(msg)
    picked = pick_section_by_user_hint(token, sections) if token else None
    if not picked and linked:
        want_kind = {"capas": "capa", "deviations": "deviation", "audits": "audit", "decisions": "decision"}
        for entity in linked:
            kind = want_kind.get(entity)
            if not kind:
                continue
            for section in enrich_sections_with_aliases(sections):
                label = str(section.get("label") or "")
                if _traceability_kind_from_label(label) == kind:
                    picked = section
                    break
            if picked:
                break
    if picked:
        label = str(picked.get("label") or "").strip()
        merged["target_scope"] = "section"
        merged["section_hint"] = label
        merged["constraints"] = merge_constraints(
            merged.get("constraints") if isinstance(merged.get("constraints"), dict) else {},
            {"detail_level": "full_section_body"},
        )
    elif re.search(r"\b(?:this|the|current|open|active|diese)\s+sop\b", msg, re.I) or message_targets_full_document(msg):
        merged["target_scope"] = "full_document"
        merged["section_hint"] = None
    elif merged.get("target_scope") not in {"section", "full_document", "linked_context", "selection"}:
        merged["target_scope"] = "full_document"

    merged = enforce_full_section_body_target(merged, msg, sections)
    return merge_editorial_profile_metadata(merged, msg, prep)


def merge_editorial_profile_metadata(
    out: dict[str, Any],
    user_message: str,
    prep: dict[str, Any],
) -> dict[str, Any]:
    """Attach editorial-profile fields when the user names a profile.md source for rewrite/improve."""
    from chatbot.assistant.profile_reference import extract_editorial_profile_reference

    if not prep.get("has_active_sop"):
        return out
    editorial_ref = extract_editorial_profile_reference(str(user_message or ""))
    if not editorial_ref:
        return out
    merged = dict(out)
    merged["editorial_profile_reference"] = editorial_ref
    merged["content_source"] = "open_sop"
    merged["reasoning"] = merged.get("reasoning") or "editorial_profile_on_open_sop_content"
    extra: dict[str, Any] = {"format": "editorial_profile_on_open_sop"}
    if re.search(r"\b(?:full|entire|whole)\s+(?:section|abschnitt)\b", str(user_message or ""), re.I):
        extra["detail_level"] = "full_section_body"
    merged["constraints"] = merge_constraints(
        merged.get("constraints") if isinstance(merged.get("constraints"), dict) else {},
        extra,
    )
    return merged


def enforce_editorial_profile_editor_route(
    out: dict[str, Any],
    user_message: str,
    prep: dict[str, Any],
) -> dict[str, Any]:
    """
    Rewrite/improve using another profile's parameters → editor_action on open SOP text.
    """
    from chatbot.assistant.profile_reference import extract_editorial_profile_reference

    msg = str(user_message or "")
    editorial_ref = extract_editorial_profile_reference(msg)
    if not editorial_ref or not prep.get("has_active_sop"):
        return out
    if not is_imperative_edit_command(msg):
        return out
    if is_gap_check_command(msg):
        return out

    merged = merge_editorial_profile_metadata(dict(out), msg, prep)
    merged["flow"] = "editor_action"
    merged["requires_confirmation"] = True
    merged["requires_selection"] = False
    merged["reasoning"] = merged.get("reasoning") or "editorial_profile_on_open_sop_content"
    if not merged.get("action") or str(merged.get("action")).lower() not in INLINE_EDITOR_ACTIONS:
        merged["action"] = detect_edit_action_from_message(msg)

    sections = prep.get("sections") if isinstance(prep.get("sections"), list) else []
    token = extract_section_name_token_from_message(msg)
    picked = pick_section_by_user_hint(token, sections) if token else None
    if picked:
        label = str(picked.get("label") or "").strip()
        merged["target_scope"] = "section"
        merged["section_hint"] = label
    elif message_targets_full_document(msg):
        merged["target_scope"] = "full_document"
        merged["section_hint"] = None
    elif re.search(r"\b(?:full|entire|whole)\s+(?:section|abschnitt)\b", msg, re.I):
        merged["target_scope"] = "section"
        merged["constraints"] = merge_constraints(
            merged.get("constraints") if isinstance(merged.get("constraints"), dict) else {},
            {"detail_level": "full_section_body"},
        )
    return merged


def enforce_full_section_body_target(
    out: dict[str, Any],
    user_message: str,
    sections: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Align orchestrator output with user intent: named / entire section → full body under heading.
    Overrides mistaken target_scope=selection when the user named a section.
    """
    merged = dict(out)
    if str(merged.get("flow") or "").lower() not in {"editor_action", "follow_up_action"}:
        return merged
    msg = str(user_message or "")
    names_section = bool(
        user_requests_entire_section_body(msg)
        or re.search(
            r"\b(?:section|abschnitt)\s+(?:of|von|für|fur|for)\b",
            msg,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:rewrite|re-?write|improve|revise|umschreib|verbesser)\w*[\s\S]{0,100}?"
            r"\b(?:section|abschnitt)\b",
            msg,
            re.IGNORECASE,
        )
        or merged.get("section_hint")
    )
    if not names_section:
        return merged

    token = extract_section_name_token_from_message(msg) or str(merged.get("section_hint") or "").strip()
    picked = pick_section_by_user_hint(token, sections) if token else None
    if picked:
        label = str(picked.get("label") or picked.get("name") or "").strip()
        merged["target_scope"] = "section"
        merged["section_hint"] = label
        merged["requires_selection"] = False
    elif user_requests_entire_section_body(msg) or names_section:
        merged["target_scope"] = "section"
        merged["requires_selection"] = False

    if merged.get("target_scope") == "section":
        merged["constraints"] = merge_constraints(
            merged.get("constraints") if isinstance(merged.get("constraints"), dict) else {},
            {"detail_level": "full_section_body"},
        )
    return merged


def pick_section_by_user_hint(token: str, sections: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Map user wording (e.g. 1. Purpose) to the canonical section row (e.g. 1. Zweck)."""
    hint = str(token or "").strip().strip("\"'")
    if not hint:
        return None
    want_num = _leading_section_number(hint)
    for section in enrich_sections_with_aliases(sections):
        label = str(section.get("label") or section.get("name") or "").strip()
        if not label:
            continue
        have_num = _leading_section_number(label)
        if want_num and have_num and want_num != have_num:
            continue
        if _semantic_section_tokens_match(hint, label):
            return section
        for alias in section.get("label_aliases") or []:
            if _semantic_section_tokens_match(hint, str(alias)):
                return section
    return None


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

    quoted = re.search(r'["\']([^"\']{2,80})["\']', msg)
    if quoted:
        picked = pick_section_by_user_hint(quoted.group(1), sections)
        if picked:
            label = str(picked.get("label") or picked.get("name") or "").strip()
            return {
                "level": "section",
                "section_id": picked.get("id") or label,
                "section_label": label,
                "resolved_from": "ALIAS_MATCH",
                "target_scope": "section",
            }

    entire_section_of = re.search(
        r"\b(?:entire|whole|full|complete|gesamte?|komplette?|ganze?)\s+(?:section|abschnitt)\s+"
        r"(?:of|von|für|fur|for)?\s*[\"']?([\wäöüÄÖÜß0-9./&()\- ]{2,80}?)(?:[\"']|\s*$|[.!?])",
        msg,
        re.IGNORECASE,
    )
    if entire_section_of:
        hint = entire_section_of.group(1).strip(" .:-\"'")
        picked = pick_section_by_user_hint(hint, sections)
        if picked:
            label = str(picked.get("label") or picked.get("name") or "").strip()
            return {
                "level": "section",
                "section_id": picked.get("id") or label,
                "section_label": label,
                "resolved_from": "ENTIRE_SECTION",
                "target_scope": "section",
                "prefer_full_section_body": True,
            }

    named_numbered = re.search(
        r"\b(?:rewrite|improve|revise|gap|umschreib|verbesser|summarize|zusammenfass|fasse)\b"
        r"[\s\S]{0,50}?"
        r'(?:the\s+)?["\']?(\d+(?:\.\d+)*[.)\]:-]?\s*[\wäöüÄÖÜß][\wäöüÄÖÜß\s/&()-]{1,60}?)'
        r'["\']?\s*(?:section|abschnitt)?\b',
        msg,
        re.IGNORECASE,
    )
    if named_numbered:
        picked = pick_section_by_user_hint(named_numbered.group(1), sections)
        if picked:
            label = str(picked.get("label") or picked.get("name") or "").strip()
            return {
                "level": "section",
                "section_id": picked.get("id") or label,
                "section_label": label,
                "resolved_from": "ALIAS_MATCH",
                "target_scope": "section",
            }

    # TASK 4 — full document scope (rewrite/improve entire SOP or read-only overview)
    if message_targets_full_document(msg):
        return {
            "level": "full",
            "section_id": None,
            "section_label": "Full Document",
            "resolved_from": "FULL_DOC",
            "target_scope": "full_document",
        }

    # Specific line (EN + DE)
    line_match = re.search(
        r"\b(?:line|zeile)\s+(\d{1,4})\b|\bin\s+zeile\s+(\d{1,4})\b",
        lower,
        re.I,
    )
    if line_match:
        line_no = int(line_match.group(1) or line_match.group(2))
        return {
            "level": "line",
            "section_id": f"line_{line_no}",
            "section_label": f"Line {line_no}",
            "resolved_from": "LINE",
            "target_scope": "selection",
            "line_number": line_no,
        }

    # German/English: "im Abschnitt Zweck", "section Purpose"
    abschnitt_match = re.search(
        r"\b(?:im\s+)?(?:abschnitt|section)\s+[\"']?([\wäöüÄÖÜß0-9./&()\-]{2,80}?)(?:[\"']|\s+in\s+\d|\s*$|[.!?])",
        msg,
        re.I,
    )
    if abschnitt_match:
        hint = abschnitt_match.group(1).strip(" .:-\"'")
        if hint and len(hint) >= 2:
            for section in enrich_sections_with_aliases(sections):
                label = str(section.get("label") or "").strip()
                if not label:
                    continue
                all_labels = [label, *(section.get("label_aliases") or [])]
                for candidate in all_labels:
                    cand = str(candidate or "").strip()
                    if len(cand) < 2:
                        continue
                    if hint.lower() in cand.lower() or cand.lower() in hint.lower():
                        return {
                            "level": "section",
                            "section_id": section.get("id") or label,
                            "section_label": label,
                            "resolved_from": "ALIAS_MATCH",
                            "target_scope": "section",
                        }
            return {
                "level": "section",
                "section_id": hint,
                "section_label": hint,
                "resolved_from": "SECTION_PHRASE",
                "target_scope": "section",
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

    # Traceability intent in the message (CAPAs/caps, deviations, …) — before loose alias substring match.
    for kind, pattern in USER_TRACEABILITY_PATTERNS.items():
        if not pattern.search(msg):
            continue
        for section in enriched:
            label = str(section.get("label") or "").strip()
            if not label or _traceability_kind_from_label(label) != kind:
                continue
            return {
                "level": "section",
                "section_id": section.get("id") or label,
                "section_label": label,
                "resolved_from": "ALIAS_MATCH",
                "target_scope": "section",
                "traceability_kind": kind,
            }
        display = {
            "capa": "CAPAs",
            "deviation": "DEVIATIONS",
            "audit": "AUDIT",
            "decision": "DECISIONS",
        }.get(kind, kind.upper())
        return {
            "level": "section",
            "section_id": kind,
            "section_label": display,
            "resolved_from": "ALIAS_MATCH",
            "target_scope": "section",
            "traceability_kind": kind,
        }

    def _token_in_message(token: str, haystack: str) -> bool:
        t = str(token or "").strip().lower()
        if len(t) < 3:
            return False
        if len(t) <= 4:
            return bool(re.search(rf"\b{re.escape(t)}\b", haystack, re.I))
        return t in haystack or bool(re.search(rf"\b{re.escape(t)}\b", haystack, re.I))

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
            if _token_in_message(cand, lower):
                return {
                    "level": "section",
                    "section_id": section.get("id") or label,
                    "section_label": label,
                    "resolved_from": "ALIAS_MATCH",
                    "target_scope": "section",
                }

    if is_contextual_section_reference(msg):
        return None

    if has_editor_selection and re.search(r"\b(?:selected|highlighted|this\s+paragraph|this\s+text)\b", lower, re.I):
        return {
            "level": "selection",
            "section_id": None,
            "section_label": "Selected text",
            "resolved_from": "SELECTION",
            "target_scope": "selection",
        }

    return None


def is_contextual_section_reference(message: str) -> bool:
    """User refers to the last-discussed section (not a new named heading)."""
    q = str(message or "").strip()
    if not q:
        return False
    return bool(
        re.search(
            r"\b(?:this|that|the|same|current|dieser|diesen|diesem|den|die|das)\s+section\b",
            q,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:rewrite|improve|gap\s*check|umschreib|verbesser)\b[\s\S]{0,60}\b(?:this|that|the)\s+section\b",
            q,
            re.IGNORECASE,
        )
        or re.search(
            r"^(?:ok(?:ay)?\s+)?(?:now\s+)?(?:rewrite|improve|gap)\b[\s\S]{0,40}\bsection\b",
            q,
            re.IGNORECASE,
        )
    )


def message_specifies_new_target(resolved: dict[str, Any] | None) -> bool:
    """True when this turn explicitly names a new scope (not a session continuation)."""
    if not isinstance(resolved, dict):
        return False
    origin = str(resolved.get("resolved_from") or "").strip().upper()
    if origin == "SESSION_MEMORY":
        return False
    return origin in {
        "ALIAS_MATCH",
        "SECTION_PHRASE",
        "SUB_SECTION",
        "RECORD_ID",
        "LINE",
        "FULL_DOC",
        "SELECTION",
    }


def _has_edit_summarize_or_gap_intent(message: str) -> bool:
    q = str(message or "")
    return bool(
        is_gap_check_command(q)
        or is_summarize_chat_query(q)
        or _EDIT_VERB_RE.search(q)
        or re.search(r"\bgap\b", q, re.IGNORECASE)
    )


def is_session_target_continuation(
    message: str,
    *,
    resolved: dict[str, Any] | None = None,
    previous_action: dict[str, Any] | None = None,
    session_active: dict[str, Any] | None = None,
) -> bool:
    """
    User continues the same section/SOP target from the current chat session
    (e.g. "rewrite the capas section" → "rewrite in 6 lines") without re-naming the heading.
    """
    prev = previous_action if isinstance(previous_action, dict) else {}
    active = session_active if isinstance(session_active, dict) else {}
    prior_action = str(prev.get("action") or active.get("last_action") or "").strip().lower()
    prior_section = str(prev.get("section_name") or active.get("section_label") or "").strip()
    prior_scope = str(prev.get("target_scope") or active.get("target_scope") or "").strip().lower()

    if not prior_action:
        return False
    if message_specifies_new_target(resolved):
        return False
    if is_meta_question_about_assistant_output(message):
        return False
    # Avoid is_read_only_sop_query here — it calls is_imperative_edit_command, which calls this helper (recursion).
    q = str(message or "").strip()
    q_low = q.lower()
    if re.search(
        r"^(?:explain|describe|tell\s+me|what\s+(?:is|are)|summarize|summary|zusammenfass|fasse)\b",
        q_low,
        re.IGNORECASE,
    ):
        return False

    if prior_action and re.search(
        r"^(?:rewrite|improve|gap(?:\s*check)?|summarize|verbesser|umschreib|lücken|luecken)\s+(?:it|them|that|this)\s*\.?$",
        q,
        re.IGNORECASE,
    ):
        return True

    if (prior_action or prior_section) and is_contextual_section_reference(q):
        if re.search(r"\b(?:rewrite|improve|gap|umschreib|verbesser)\b", q, re.IGNORECASE):
            return True

    if not _has_edit_summarize_or_gap_intent(q):
        return False

    if extract_format_constraints(q):
        return True
    if prior_section or prior_scope == "full_document":
        if re.search(
            r"\b(?:in|into|within|auf)\s+\d{1,3}\s*(?:lines?|zeilen?)\b|\b\d{1,3}\s*(?:lines?|zeilen?)\b",
            q,
            re.IGNORECASE,
        ):
            return True
        if re.search(
            r"^(?:rewrite|improve|gap|summarize|zusammenfass|fasse|lücken|luecken)\b",
            q,
            re.IGNORECASE,
        ):
            return True
    return is_follow_up_edit_refinement(q, prev, _from_session_continuation=True)


def merge_resolved_scope_with_session(
    resolved: dict[str, Any] | None,
    *,
    message: str,
    session: dict[str, Any],
    previous_action: dict[str, Any],
    sections: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Inherit section/full target from session memory when the user omits it on the next turn."""
    if message_specifies_new_target(resolved):
        return resolved
    active = session.get("active_scope") if isinstance(session.get("active_scope"), dict) else {}
    prev = previous_action if isinstance(previous_action, dict) else {}
    if not is_session_target_continuation(
        message,
        resolved=resolved,
        previous_action=prev,
        session_active=active,
    ):
        return resolved

    scope = str(prev.get("target_scope") or active.get("target_scope") or "section").strip().lower()
    label = str(prev.get("section_name") or active.get("section_label") or "").strip()
    if not label:
        prompt = str(prev.get("request_prompt") or active.get("last_request_prompt") or "").strip()
        if prompt:
            resolved_prompt = resolve_scope_from_message(prompt, sections)
            if isinstance(resolved_prompt, dict) and resolved_prompt.get("section_label"):
                label = str(resolved_prompt["section_label"]).strip()
    if scope == "full_document" or label.lower() in {"full document", "full sop"}:
        return {
            "level": "full",
            "section_id": None,
            "section_label": "Full Document",
            "resolved_from": "SESSION_MEMORY",
            "target_scope": "full_document",
        }

    if not label:
        return resolved

    section_id = str(active.get("section_id") or prev.get("section_id") or label).strip()
    for section in sections:
        if str(section.get("label") or "").strip() == label:
            section_id = str(section.get("id") or label).strip()
            break

    line_no = prev.get("line_number") or active.get("line_number")
    if line_no is not None:
        try:
            line_no = int(line_no)
        except (TypeError, ValueError):
            line_no = None
    if line_no and line_no > 0:
        return {
            "level": "line",
            "section_id": f"line_{line_no}",
            "section_label": f"Line {line_no}",
            "resolved_from": "SESSION_MEMORY",
            "target_scope": "selection",
            "line_number": line_no,
        }

    return {
        "level": "section",
        "section_id": section_id or label,
        "section_label": label,
        "resolved_from": "SESSION_MEMORY",
        "target_scope": "section",
    }


def resolve_effective_previous_action(
    ctx: dict[str, Any],
    session: dict[str, Any],
    *,
    sections: list[dict[str, Any]],
    recent_messages: list | None = None,
) -> dict[str, Any]:
    """
    Merge last_action, session active_scope, and recent user turns so follow-ups like
    "improve it" still target the last rewritten section.
    """
    prev = ctx.get("last_action") if isinstance(ctx.get("last_action"), dict) else {}
    if str(prev.get("action") or "").strip():
        out = dict(prev)
        if not str(out.get("section_name") or "").strip():
            prompt = str(out.get("request_prompt") or "").strip()
            if prompt:
                resolved = resolve_scope_from_message(prompt, sections)
                if isinstance(resolved, dict) and resolved.get("section_label"):
                    out["section_name"] = str(resolved["section_label"]).strip()
                    out["target_scope"] = str(resolved.get("target_scope") or out.get("target_scope") or "section")
        return out

    active = session.get("active_scope") if isinstance(session.get("active_scope"), dict) else {}
    act = str(active.get("last_action") or "").strip().lower()
    label = str(active.get("section_label") or "").strip()
    if act:
        return {
            "action": act,
            "target_scope": str(active.get("target_scope") or "section").strip(),
            "section_name": label,
            "section_id": str(active.get("section_id") or label).strip(),
            "request_prompt": "",
            "status": "session_active_scope",
        }

    for row in reversed(list(recent_messages or [])[-12:]):
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "user").strip().lower()
        if role not in {"user", ""}:
            continue
        text = str(row.get("content") or "").strip()
        if not text or not _has_edit_summarize_or_gap_intent(text):
            continue
        resolved = resolve_scope_from_message(text, sections)
        return {
            "action": detect_edit_action_from_message(text),
            "target_scope": str((resolved or {}).get("target_scope") or "section"),
            "section_name": str((resolved or {}).get("section_label") or ""),
            "section_id": str((resolved or {}).get("section_id") or ""),
            "request_prompt": text[:400],
            "status": "inferred_from_recent",
        }
    return {}


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


def extract_format_constraints(message: str) -> dict[str, Any]:
    """Parse explicit output shape from the user message (lines, words)."""
    q = str(message or "")
    out: dict[str, Any] = {}
    line_patterns = [
        r"\b(?:in|into|within|auf|using|with|as|to)\s+(\d{1,3})\s*(?:lines?|zeilen?|line\s+limit)\b",
        r"\btell\s+me\s+in\s+(\d{1,3})\s*lines?\b",
        r"\b(\d{1,3})\s*[-\s]*(?:line|zeilen)(?:\s+limit)?\b",
        r"\b(?:rewrite|improve|explain|describe|fasse|zusammenfass)\w*[^.?\n]{0,100}?\b(\d{1,3})\s*(?:lines?|zeilen?)\b",
    ]
    line_match = None
    for pattern in line_patterns:
        line_match = re.search(pattern, q, re.IGNORECASE)
        if line_match:
            break
    if line_match:
        out["line_count"] = max(1, min(120, int(line_match.group(1))))
        out["format"] = "plain_lines"
    word_match = re.search(r"\b(\d{2,5})\s*(?:words?|wörter|woerter)\b", q, re.IGNORECASE)
    if word_match:
        out["word_count"] = max(10, min(10_000, int(word_match.group(1))))
    if re.search(r"\b(?:bullet|bullets|numbered\s+list|aufzählung)\b", q, re.IGNORECASE):
        out["format"] = out.get("format") or "bullets"
    return out


def merge_constraints(*parts: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        if isinstance(part, dict):
            merged.update({k: v for k, v in part.items() if v is not None})
    return merged


def strip_html_to_plain(text: str) -> str:
    """Plain text for chat explanations (no tags / entities)."""
    s = str(text or "")
    if not s.strip():
        return ""
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"</p\s*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<li[^>]*>", "\n• ", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    s = html_module.unescape(s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return re.sub(r" +", " ", s).strip()


def infer_output_language_from_context(ctx: dict[str, Any] | None) -> str:
    """Match rewrite/chat output to the open SOP document language (de/en)."""
    from chatbot.actions.prompts import _detect_text_language, _extract_detected_language

    ctx = ctx if isinstance(ctx, dict) else {}
    current = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}
    samples: list[str] = []

    for candidate in (
        ctx.get("nlp_profile"),
        ctx.get("profile_detection"),
        current.get("metadata_json"),
        current.get("metadata"),
    ):
        if isinstance(candidate, dict):
            nested = candidate.get("nlp_profile") if isinstance(candidate.get("nlp_profile"), dict) else candidate
            detected = (
                nested.get("detected_nlp")
                if isinstance(nested.get("detected_nlp"), dict)
                else nested
            )
            lang = _extract_detected_language(detected if isinstance(detected, dict) else nested)
            if lang:
                return lang

    samples.append(str(current.get("full_text") or ""))
    samples.append(str(ctx.get("editor_excerpt") or ""))
    last = ctx.get("last_action") if isinstance(ctx.get("last_action"), dict) else {}
    samples.append(str(last.get("original_text_excerpt") or ""))
    for sec in current.get("sections") or []:
        if isinstance(sec, dict):
            samples.append(str(sec.get("content") or sec.get("text") or ""))
    combined = strip_html_to_plain("\n".join(samples))[:14_000]
    return _detect_text_language(combined)


def _action_label(action: str, lang: str) -> str:
    a = str(action or "edit").strip().lower()
    if lang == "de":
        return {
            "rewrite": "Umschreibung",
            "improve": "Verbesserung",
            "gap_check": "Lückenprüfung",
            "summarize": "Zusammenfassung",
        }.get(a, "Bearbeitung")
    return {
        "rewrite": "rewrite",
        "improve": "improvement",
        "gap_check": "gap check",
        "summarize": "summary",
    }.get(a, "edit")


_REWRITE_AGAIN_RE = re.compile(
    r"\b(?:rewrite|re-?write|improve|verbesser)\s+(?:again|once\s+more|redo|erneut|nochmal|noch\s+einmal)\b"
    r"|\b(?:again|redo|once\s+more|nochmal|erneut)\b[\s\S]{0,40}\b(?:rewrite|re-?write|improve)\b",
    re.IGNORECASE,
)


def is_rewrite_again_followup(message: str, previous_action: dict[str, Any] | None) -> bool:
    """Repeat the last edit on the same target (uses session last_result as source)."""
    if not previous_action or not str(previous_action.get("action") or "").strip():
        return False
    if is_meta_question_about_assistant_output(message) or is_read_only_sop_query(
        message, previous_action=previous_action, resolved_scope=None
    ):
        return False
    if _REWRITE_AGAIN_RE.search(str(message or "")):
        return True
    if is_imperative_edit_command(message):
        return False
    return bool(
        re.search(
            r"\b(?:again|redo|once\s+more|another\s+time|repeat|noch\s+einmal|nochmal|erneut)\b",
            str(message or ""),
            re.IGNORECASE,
        )
    )


def _meaningful_new_terms(original: str, suggested: str, *, limit: int = 4) -> list[str]:
    orig_words = {
        w.lower()
        for w in re.findall(r"\b[\wäöüÄÖÜß]{4,}\b", str(original or ""), flags=re.UNICODE)
    }
    sug_words = re.findall(r"\b[\wäöüÄÖÜß]{4,}\b", str(suggested or ""), flags=re.UNICODE)
    picked: list[str] = []
    for w in sug_words:
        low = w.lower()
        if low in _DIFF_STOPWORDS or low in orig_words:
            continue
        if w not in picked:
            picked.append(w)
        if len(picked) >= limit:
            break
    return picked


def _simple_diff_summary(
    original: str,
    suggested: str,
    *,
    max_points: int = 5,
    lang: str = "en",
) -> list[str]:
    """Lightweight diff bullets in conversational language."""
    orig_plain = strip_html_to_plain(original)
    sug_plain = strip_html_to_plain(suggested)
    orig_lines = [ln.strip() for ln in orig_plain.splitlines() if ln.strip()]
    sug_lines = [ln.strip() for ln in sug_plain.splitlines() if ln.strip()]
    points: list[str] = []
    de = lang == "de"
    if len(sug_lines) < len(orig_lines):
        points.append(
            "Der Text wurde gestrafft (weniger Zeilen, klarere Formulierungen)."
            if de
            else "The text was tightened (fewer lines, clearer wording)."
        )
    elif len(sug_lines) > len(orig_lines):
        points.append(
            "Es wurden mehr Details bzw. Aufzählungen ergänzt."
            if de
            else "More detail or bullet structure was added."
        )
    if len(sug_plain) > int(len(orig_plain) * 1.08):
        points.append(
            "Formulierungen wurden präziser und compliance-orientierter."
            if de
            else "Wording was made more precise and compliance-oriented."
        )
    elif len(sug_plain) < int(len(orig_plain) * 0.92) and orig_plain:
        points.append(
            "Der Inhalt wurde gekürzt, der Abschnittsumfang bleibt gleich."
            if de
            else "Content was shortened while keeping the same section scope."
        )
    new_terms = _meaningful_new_terms(orig_plain, sug_plain)
    if new_terms:
        joined = ", ".join(new_terms)
        points.append(
            f"Neue Begriffe u. a.: {joined}."
            if de
            else f"Notable new terms include: {joined}."
        )
    if not points:
        points.append(
            "Sätze wurden für Klarheit und audit-tauglichen Ton umgestellt — ohne den Abschnittsumfang zu wechseln."
            if de
            else "Sentences were restructured for clarity and audit-ready tone without changing the section scope."
        )
    return points[:max_points]


def build_diff_explanation_answer(
    user_message: str,
    previous_action: dict[str, Any] | None,
    *,
    format_constraints: dict[str, Any] | None = None,
    active_scope: dict[str, Any] | None = None,
    assistant_context: dict[str, Any] | None = None,
) -> str:
    """Explain what changed in the last rewrite/improve — natural chat, same language as SOP."""
    last = previous_action if isinstance(previous_action, dict) else {}
    active = active_scope if isinstance(active_scope, dict) else {}
    ctx = assistant_context if isinstance(assistant_context, dict) else {}
    lang = (
        str((format_constraints or {}).get("language") or "").strip().lower()
        or infer_output_language_from_context(ctx)
        or infer_output_language_from_context({"last_action": last, "current_sop": {}})
        or "en"
    )
    action_raw = str(last.get("action") or active.get("last_action") or "edit").strip()
    action_label = _action_label(action_raw, lang)
    section = str(last.get("section_name") or active.get("section_label") or "").strip()
    if not section:
        section = "dem letzten Zielabschnitt" if lang == "de" else "the last target section"
    original = strip_html_to_plain(
        str(last.get("original_text_excerpt") or active.get("last_original") or "")
    )
    suggested = strip_html_to_plain(
        str(last.get("suggested_text_excerpt") or active.get("last_result") or "")
    )
    fc = format_constraints if isinstance(format_constraints, dict) else {}
    line_cap = int(fc.get("line_count") or 0)
    brief = line_cap and line_cap <= 4
    max_points = 1 if brief else 3

    bullets = _simple_diff_summary(original, suggested, max_points=max_points, lang=lang)

    if lang == "de":
        opener = (
            f"Beim Abschnitt „{section}“ liegt eine {action_label} als Vorschau im Editor bereit — "
            "du kannst sie dort mit Annehmen oder Verwerfen prüfen."
        )
        if not original and not suggested:
            opener = (
                f"Die letzte {action_label} für „{section}“ ist im Editor als Vorschau hinterlegt; "
                "ein Textvergleich ist noch nicht vollständig gespeichert."
            )
    else:
        opener = (
            f"For “{section}”, the last {action_label} is ready as an inline preview in the editor — "
            "you can accept or reject it there."
        )
        if not original and not suggested:
            opener = (
                f"The last {action_label} on “{section}” is in the editor preview; "
                "before/after text was not fully stored yet for a detailed diff."
            )

    if original and suggested:
        ow, sw = len(original.split()), len(suggested.split())
        if lang == "de":
            bullets.append(f"Länge: ca. {ow} → {sw} Wörter in der Vorschau.")
        else:
            bullets.append(f"Length: about {ow} → {sw} words in the preview.")

    if brief:
        detail = bullets[0] if bullets else ""
        return f"{opener} {detail}".strip()

    if lang == "de":
        body = " ".join(bullets)
        return f"{opener}\n\n{body}".strip()

    body = " ".join(bullets)
    return f"{opener}\n\n{body}".strip()


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


_EDIT_NOUN = r"(?:rewrites?|re-?writes?|improvements?|improv(?:ed|ement)?|suggestions?|edits?)"

_META_OUTPUT_RE = re.compile(
    r"(?:"
    r"\b(?:explain|describe|tell\s+me|show\s+me|what|which|why|how)\b[\s\S]{0,160}\b(?:chang(?:es)?|difference|diff|"
    + _EDIT_NOUN
    + r"|modif(?:y|ied)?|updat(?:e|ed)?|made|done)\b"
    r"|"
    r"\bwhat\s+(?:is\s+)?(?:the\s+)?(?:difference|diff|chang(?:es)?)\b[\s\S]{0,80}\b(?:in|from|with)\b[\s\S]{0,40}\b"
    + _EDIT_NOUN
    + r"\b"
    r"|"
    r"\bwhat\s+(?:did\s+you|have\s+you|were\s+you)\s+(?:chang|rewrit|improv|do|make)\w*\b"
    r"|"
    r"\b(?:chang(?:es)?|difference|diff)\s+(?:in|from|during|with)\s+(?:the\s+)?(?:last|previous|your)?\s*"
    + _EDIT_NOUN
    + r"\b"
    r"|"
    r"\bwhy\s+(?:did\s+you|have\s+you)\s+(?:chang|rewrit|improv)\w*\b"
    r"|"
    r"\btell\s+me\s+(?:what|about)\s+(?:the\s+)?(?:difference|diff|chang(?:es)?)\b"
    r")",
    re.IGNORECASE,
)


def _edit_word_is_noun_reference(message: str) -> bool:
    """True when rewrite/improve appears only as a noun (the rewrite), not a command."""
    q = str(message or "")
    if not re.search(
        r"\b(?:the|this|that|your|last|previous)\s+(?:rewrite|re-?write|improvement|suggestion|edit)\b",
        q,
        re.IGNORECASE,
    ):
        return False
    return not is_imperative_edit_command(q)


def is_meta_question_about_assistant_output(message: str) -> bool:
    """User asks about the assistant's prior edit — chat only, not a new rewrite."""
    return bool(_META_OUTPUT_RE.search(str(message or "")))


def is_explain_recent_output_query(message: str, previous_action: dict[str, Any] | None) -> bool:
    """Explain the last preview/solution in N lines — not a full-SOP RAG summary."""
    last = previous_action if isinstance(previous_action, dict) else {}
    preview = str(last.get("suggested_text_excerpt") or "").strip()
    if not preview:
        return False
    q = str(message or "").strip()
    if not q or is_meta_question_about_assistant_output(q) or is_imperative_edit_command(q):
        return False
    if not re.search(
        r"\b(?:explain|describe|clarify|summarize|outline|erkläre|erklaere|beschreib|fasse\s+zusammen)\b",
        q,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"\b(?:this|the|it|that|your|last|previous)\s+"
        r"(?:solution|rewrite|suggestion|preview|change|result|output|version|text)\b",
        q,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"\bexplain\s+(?:this|the|it|that)\b", q, re.IGNORECASE):
        return True
    return bool(extract_format_constraints(q).get("line_count"))


def _split_into_lines(text: str, *, max_lines: int) -> list[str]:
    """Split plain text into at most max_lines non-empty lines."""
    plain = strip_html_to_plain(text)
    if not plain:
        return []
    lines = [ln.strip() for ln in re.split(r"\r?\n+", plain) if ln.strip()]
    if len(lines) >= max_lines:
        return lines[:max_lines]
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\s*;\s*", plain)
        if len(s.strip()) > 12
    ]
    if not sentences:
        sentences = [plain]
    merged: list[str] = []
    for sent in sentences:
        if len(merged) >= max_lines:
            break
        merged.append(sent)
    return merged[:max_lines]


def build_explain_last_output_answer(
    user_message: str,
    previous_action: dict[str, Any] | None,
    *,
    format_constraints: dict[str, Any] | None = None,
    assistant_context: dict[str, Any] | None = None,
) -> str:
    """Explain the last editor preview in exactly N conversational lines."""
    last = previous_action if isinstance(previous_action, dict) else {}
    ctx = assistant_context if isinstance(assistant_context, dict) else {}
    fc = format_constraints if isinstance(format_constraints, dict) else {}
    lang = (
        str(fc.get("language") or "").strip().lower()
        or infer_output_language_from_context(ctx)
        or "en"
    )
    line_cap = int(fc.get("line_count") or extract_format_constraints(user_message).get("line_count") or 5)
    line_cap = max(1, min(20, line_cap))
    section = str(last.get("section_name") or "").strip() or (
        "dem letzten Abschnitt" if lang == "de" else "the last section"
    )
    suggested = strip_html_to_plain(str(last.get("suggested_text_excerpt") or ""))
    lines = _split_into_lines(suggested, max_lines=line_cap)
    if lang == "de":
        if not lines:
            return (
                f"Die letzte Vorschau für „{section}“ ist im Editor — "
                "ein erklärender Text war noch nicht gespeichert."
            )
        intro = f"Kurz zur letzten Vorschau für „{section}“ ({line_cap} Zeilen):"
    else:
        if not lines:
            return (
                f"The latest preview for “{section}” is in the editor — "
                "no stored text was available to explain yet."
            )
        intro = f"About the latest preview for “{section}” ({line_cap} lines):"
    return intro + "\n" + "\n".join(lines[:line_cap])


def enforce_output_line_count(instruction: str | None, text: str) -> str:
    """Hard-cap editor action output to the user's requested line count."""
    fc = extract_format_constraints(instruction or "")
    n = int(fc.get("line_count") or 0)
    if not n or not str(text or "").strip():
        return text
    lines = _split_into_lines(text, max_lines=n)
    if not lines:
        return text
    return "\n".join(lines)


def is_imperative_edit_command(
    message: str,
    *,
    previous_action: dict[str, Any] | None = None,
    session_active: dict[str, Any] | None = None,
    resolved_scope: dict[str, Any] | None = None,
    _checking_continuation: bool = False,
) -> bool:
    """True when the user commands a new edit (not merely mentioning 'rewrite' as a noun)."""
    q = str(message or "").strip()
    if not q or is_meta_question_about_assistant_output(q):
        return False
    if not _checking_continuation and is_session_target_continuation(
        q,
        resolved=resolved_scope,
        previous_action=previous_action,
        session_active=session_active,
    ):
        return False
    # Chat overview ("summarize this SOP in bullets") — not an in-document edit.
    if _SUMMARIZE_VERB_RE.search(q) and not is_summarize_into_document(q):
        if not _EDIT_VERB_RE.search(q) and not re.search(
            r"\b(?:gap|lücken|luecken)\b",
            q,
            re.IGNORECASE,
        ):
            return False
    if is_gap_check_command(q):
        return True
    if is_summarize_into_document(q):
        return True
    if re.search(
        r"(?:^|\b)(?:ok(?:ay)?\s+)?(?:now\s+)?(?:please\s+)?"
        r"(?:rewrite|re-?write|improve|revise|verbesser|umschreib|überarbeit|gap\s*check|lücken|luecken|"
        r"summarize|zusammenfass|fasse)\s+"
        r"(?:the|this|that|den|die|das|section|abschnitt|[\"'])",
        q,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:rewrite|re-?write|improve|revise|verbesser|umschreib)\s+"
        r"(?:the\s+)?(?:deviations?|capas?|decisions?|audits?|zweck|purpose|scope|section|\d|standard\s+operating\s+procedure|procedure|sop)",
        q,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:rewrite|improve|verbesser\w*|umschreib)\s+(?:[\"']|den\s+|die\s+|das\s+)?\d",
        q,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"^(?:rewrite|improve|revise|verbesser\w*|umschreib|gap\s*check|summarize|zusammenfass|fasse)\s+",
        q,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:rewrite|re-?write|improve|revise|verbesser\w*|umschreib|überarbeit)\s+"
        r"(?:(?:the|this|current|open|active|den|die|das)\s+)?"
        r"(?:full|whole|entire|complete|gesamte?|ganze?|komplette?)\s+(?:sop|document|doc)\b",
        q,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:rewrite|improve|revise|verbesser\w*|umschreib)\s+(?:this|the|current|open|active)\s+sop\b",
        q,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:revise|rewrite|improve|umschreib|verbesser\w*|überarbeit|ueberarbeit)\s+"
        r"(?:the\s+)?(?:standard\s+operating\s+)?procedure\b",
        q,
        re.IGNORECASE,
    ):
        return True
    return False


def is_read_only_sop_query(
    message: str,
    *,
    previous_action: dict[str, Any] | None = None,
    resolved_scope: dict[str, Any] | None = None,
    _checking_continuation: bool = False,
) -> bool:
    """Summarize / explain / ask about open SOP — no editor mutation."""
    q = str(message or "").strip()
    if not q or is_imperative_edit_command(
        q,
        previous_action=previous_action,
        resolved_scope=resolved_scope,
        _checking_continuation=True,
    ):
        return False
    if is_gap_check_command(q):
        return False
    if is_summarize_into_document(q, resolved_scope=resolved_scope):
        return False
    if is_explain_recent_output_query(q, previous_action):
        return False
    if is_meta_question_about_assistant_output(q):
        return True
    if re.search(
        r"^(?:ok(?:ay)?\s+)?(?:now\s+)?(?:explain|describe|tell\s+me|what\s+(?:is|are|was|were|did)|summarize|summary|zusammenfass|fasse)\b",
        q,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"\btell\s+me\s+(?:what|about)\s+(?:the\s+)?(?:difference|diff|chang(?:es)?)\b", q, re.IGNORECASE):
        return True
    if re.search(
        r"\b(explain|describe|tell\s+me\s+about|what\s+is\s+this\s+sop|summarize|summary|zusammenfass|"
        r"fasse\s+(?:den|die|das)?\s*abschnitt|kurzfass|overview|überblick|uberblick)\b",
        q,
        re.IGNORECASE,
    ):
        return True
    return False


def detect_edit_action_from_message(message: str, *, previous_action: dict | None = None) -> str:
    q = str(message or "")
    if is_gap_check_command(q):
        return "gap_check"
    if re.search(r"\b(?:improve|verbesser|better|audit[-\s]?ready)\b", q, re.IGNORECASE):
        return "improve"
    if re.search(r"\b(?:revise|revision)\b", q, re.IGNORECASE):
        return "rewrite"
    if re.search(r"\b(?:rewrite|re-?write|umschreib|überarbeit|ueberarbeit)\b", q, re.IGNORECASE):
        return "rewrite"
    if _SUMMARIZE_VERB_RE.search(q) or re.search(r"\bzusammen\b", q, re.IGNORECASE):
        return "summarize"
    if re.search(r"\b(?:analyze|analyse)\b", q, re.IGNORECASE):
        return "analyze"
    if re.search(r"\bcompliance\s+review\b", q, re.IGNORECASE):
        return "gap_check"
    prev = str((previous_action or {}).get("action") or "").strip().lower()
    if prev in {"rewrite", "improve", "gap_check", "summarize", "analyze"}:
        return prev
    return "improve"


def is_follow_up_edit_refinement(
    message: str,
    previous_action: dict | None,
    *,
    _from_session_continuation: bool = False,
) -> bool:
    """Continuation of a prior edit (shorter, rewrite it) — not meta questions or new section commands."""
    if not previous_action or not str(previous_action.get("action") or "").strip():
        return False
    q = str(message or "").strip().lower()
    if not q or is_meta_question_about_assistant_output(q) or is_read_only_sop_query(
        q,
        previous_action=previous_action,
        resolved_scope=None,
        _checking_continuation=_from_session_continuation,
    ):
        return False
    if is_imperative_edit_command(
        message,
        previous_action=previous_action,
        _checking_continuation=True,
    ):
        return False
    if is_contextual_section_reference(message) and re.search(
        r"\b(?:rewrite|improve|gap)\b", message, re.IGNORECASE
    ):
        return True
    if re.search(
        r"\b(?:in|into|within|auf)\s+\d{1,3}\s*(?:lines?|zeilen?)\b",
        q,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:previous|suggestion|i\s+told\s+you|too\s+long|still\s+too\s+long|shorter|shorten|"
        r"make\s+it\s+shorter|summarize\s+it|improve\s+that|same\s+style)\b",
        q,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(?:make|rewrite|improve|summarize|shorten)\s+(?:it|them|that|this\s+version)\b",
        q,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"^(?:ok(?:ay)?\s+)?(?:now\s+)?(?:rewrite|improve|shorten)\s+(?:it|that|this)\b", q, re.IGNORECASE):
        return True
    # "okay now improve X" — but not "okay now tell me ... the rewrite"
    if (
        re.search(r"^(?:ok(?:ay)?\s+)?now\s+", q, re.IGNORECASE)
        and re.search(r"\b(?:rewrite|improve|verbesser|gap)\s+(?:the|this|that|it|section)\b", q, re.IGNORECASE)
        and not _edit_word_is_noun_reference(message)
    ):
        return True
    return False


def _section_hint_from_scope(resolved: dict[str, Any] | None, previous_action: dict | None) -> str:
    if isinstance(resolved, dict) and resolved.get("section_label"):
        return str(resolved["section_label"]).strip()
    if isinstance(previous_action, dict):
        return str(previous_action.get("section_name") or "").strip()
    return ""


def analyze_turn_pipeline(
    message: str,
    *,
    session: dict[str, Any],
    sections: list[dict[str, Any]],
    has_active_sop: bool,
    has_editor_selection: bool,
    previous_action: dict[str, Any] | None = None,
    resolved_scope: dict[str, Any] | None = None,
    frustration: dict[str, Any] | None = None,
    repetition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Dynamic turn analysis (Python-only):
      1) Parse user query intent
      2) Choose RAG vs open-SOP chat vs editor action
      3) Resolve target on current SOP sections
    Returns { query_analysis, early_response? }.
    """
    msg = str(message or "").strip()
    frustration = frustration if isinstance(frustration, dict) else {}
    repetition = repetition if isinstance(repetition, dict) else {}
    previous_action = previous_action if isinstance(previous_action, dict) else {}
    resolved = resolved_scope if isinstance(resolved_scope, dict) else None

    use_rag = not has_active_sop
    use_open_sop = has_active_sop
    use_editor_action = False
    primary_intent = "sop_query"
    early: dict[str, Any] | None = None
    format_constraints = extract_format_constraints(msg)
    session_active = session.get("active_scope") if isinstance(session.get("active_scope"), dict) else {}
    llm_orchestrator = use_llm_orchestrator()

    if llm_orchestrator:
        primary_intent = "llm_orchestrated"
    elif is_meta_question_about_assistant_output(msg):
        primary_intent = "sop_query"
        use_editor_action = False
        early = {
            "flow": "chat",
            "action": None,
            "target_scope": None,
            "section_hint": _section_hint_from_scope(resolved, previous_action) or None,
            "linked_entity_types": [],
            "constraints": format_constraints,
            "clarification_question": None,
            "confidence": 0.97,
            "reasoning": "meta_question_about_previous_assistant_output",
            "chat_submode": "explain_last_edit_diff",
            "resolved_scope": resolved,
            "frustration_signal": frustration,
            "repetition_detected": repetition.get("repetition_detected"),
            "repetition_instruction": repetition.get("repetition_instruction"),
            "previous_action": previous_action or None,
        }
    elif is_explain_recent_output_query(msg, previous_action):
        primary_intent = "sop_query"
        use_editor_action = False
        early = {
            "flow": "chat",
            "action": None,
            "target_scope": None,
            "section_hint": _section_hint_from_scope(resolved, previous_action) or None,
            "linked_entity_types": [],
            "constraints": format_constraints,
            "clarification_question": None,
            "confidence": 0.96,
            "reasoning": "explain_last_editor_preview",
            "chat_submode": "explain_last_output",
            "resolved_scope": resolved,
            "frustration_signal": frustration,
            "repetition_detected": repetition.get("repetition_detected"),
            "repetition_instruction": repetition.get("repetition_instruction"),
            "previous_action": previous_action or None,
        }
    elif has_active_sop and is_summarize_chat_query(msg) and not is_meta_question_about_assistant_output(msg):
        primary_intent = "sop_query"
        if not resolved:
            resolved = resolve_scope_from_message(msg, sections, has_editor_selection=has_editor_selection)
        section_hint = _section_hint_from_scope(resolved, previous_action)
        target_scope = "full_document" if (resolved and resolved.get("level") == "full") or message_targets_full_document(msg) else (
            "section" if section_hint or (resolved and resolved.get("target_scope") == "section") else None
        )
        early = {
            "flow": "chat",
            "action": None,
            "target_scope": target_scope,
            "section_hint": section_hint or None,
            "linked_entity_types": [],
            "constraints": merge_constraints(
                format_constraints,
                {"length": "shorter", "format": "concise_sidebar_summary"},
            ),
            "clarification_question": None,
            "confidence": 0.95,
            "reasoning": "summarize_in_sidebar_chat",
            "chat_submode": "sop_summarize",
            "resolved_scope": resolved,
            "frustration_signal": frustration,
            "repetition_detected": repetition.get("repetition_detected"),
            "repetition_instruction": repetition.get("repetition_instruction"),
            "preserve_active_scope": True,
        }
    elif has_active_sop and is_gap_check_command(msg) and not is_meta_question_about_assistant_output(msg):
        primary_intent = "action"
        use_editor_action = True
        if not resolved:
            resolved = resolve_scope_from_message(msg, sections, has_editor_selection=has_editor_selection)
        section_hint = _section_hint_from_scope(resolved, previous_action)
        targets_full = (
            (resolved and resolved.get("level") == "full")
            or message_targets_full_document(msg)
            or bool(re.search(r"\b(?:diese|dieser|diesem|this|the|current|open|aktuelle?)\s+sop\b", msg, re.I))
        )
        target_scope = "full_document" if targets_full else (
            "section" if section_hint or (resolved and resolved.get("target_scope") == "section") else "selection"
        )
        early = {
            "flow": "editor_action",
            "action": "gap_check",
            "target_scope": target_scope,
            "section_hint": section_hint or None,
            "line_number": resolved.get("line_number") if resolved else None,
            "record_id": resolved.get("record_id") if resolved else None,
            "constraints": format_constraints,
            "requires_selection": target_scope == "selection",
            "requires_confirmation": True,
            "confidence": 0.94,
            "reasoning": "gap_check_on_open_sop",
            "resolved_scope": resolved,
            "frustration_signal": frustration,
            "repetition_detected": repetition.get("repetition_detected"),
            "repetition_instruction": repetition.get("repetition_instruction"),
            "previous_action": previous_action or None,
        }
    elif (
        has_active_sop
        and resolved
        and resolved.get("level") == "full"
        and is_imperative_edit_command(msg)
    ):
        primary_intent = "action"
        use_editor_action = True
        early = {
            "flow": "editor_action",
            "action": detect_edit_action_from_message(msg),
            "target_scope": "full_document",
            "section_hint": None,
            "linked_entity_types": [],
            "constraints": merge_constraints(format_constraints, _constraints_from_frustration(frustration, {})),
            "requires_selection": False,
            "requires_confirmation": True,
            "confidence": 0.94,
            "reasoning": "full_document_edit_command",
            "resolved_scope": resolved,
            "frustration_signal": frustration,
            "repetition_detected": repetition.get("repetition_detected"),
            "repetition_instruction": repetition.get("repetition_instruction"),
            "previous_action": previous_action or None,
        }
    elif (
        resolved
        and resolved.get("level") == "full"
        and not is_imperative_edit_command(
            msg,
            previous_action=previous_action,
            session_active=session_active,
            resolved_scope=resolved,
        )
    ):
        primary_intent = "sop_query"
        early = {
            "flow": "chat",
            "action": None,
            "target_scope": "full_document",
            "section_hint": None,
            "linked_entity_types": [],
            "constraints": format_constraints,
            "clarification_question": None,
            "confidence": 0.95,
            "reasoning": "full_doc_read_only_overview",
            "chat_submode": "sop_explain",
            "resolved_scope": resolved,
            "frustration_signal": frustration,
            "repetition_detected": repetition.get("repetition_detected"),
            "repetition_instruction": repetition.get("repetition_instruction"),
            "preserve_active_scope": True,
        }
    elif is_read_only_sop_query(msg, previous_action=previous_action, resolved_scope=resolved) and has_active_sop:
        primary_intent = "sop_query"
        early = {
            "flow": "chat",
            "action": None,
            "target_scope": "full_document" if resolved and resolved.get("level") == "full" else None,
            "section_hint": _section_hint_from_scope(resolved, previous_action) or None,
            "linked_entity_types": [],
            "constraints": format_constraints,
            "chat_submode": "sop_explain",
            "clarification_question": None,
            "confidence": 0.94,
            "reasoning": "read_only_sop_query",
            "resolved_scope": resolved,
            "frustration_signal": frustration,
            "repetition_detected": repetition.get("repetition_detected"),
            "repetition_instruction": repetition.get("repetition_instruction"),
        }
    elif has_active_sop and is_rewrite_again_followup(msg, previous_action):
        primary_intent = "followup"
        use_editor_action = True
        section_hint = _section_hint_from_scope(resolved, previous_action)
        prev_action = str(previous_action.get("action") or "rewrite").strip().lower()
        early = {
            "flow": "follow_up_action",
            "action": prev_action if prev_action in INLINE_EDITOR_ACTIONS else "rewrite",
            "target_scope": "section" if section_hint else "previous_suggestion",
            "section_hint": section_hint or None,
            "constraints": merge_constraints(format_constraints, _constraints_from_frustration(frustration, {})),
            "requires_selection": False,
            "requires_confirmation": True,
            "confidence": 0.93,
            "reasoning": "rewrite_again_same_target",
            "resolved_scope": resolved,
            "frustration_signal": frustration,
            "repetition_detected": repetition.get("repetition_detected"),
            "repetition_instruction": repetition.get("repetition_instruction"),
            "previous_action": previous_action or None,
        }
    elif (
        has_active_sop
        and is_session_target_continuation(
            msg,
            resolved=resolved,
            previous_action=previous_action,
            session_active=session_active,
        )
    ):
        primary_intent = "followup"
        use_editor_action = True
        section_hint = _section_hint_from_scope(resolved, previous_action)
        target_scope = "full_document" if resolved and resolved.get("level") == "full" else (
            "section" if section_hint or (resolved and resolved.get("target_scope") == "section") else "previous_suggestion"
        )
        if resolved and resolved.get("level") == "line":
            target_scope = "selection"
        early = {
            "flow": "follow_up_action",
            "action": detect_edit_action_from_message(msg, previous_action=previous_action),
            "target_scope": target_scope,
            "section_hint": section_hint or None,
            "line_number": resolved.get("line_number") if resolved else None,
            "record_id": resolved.get("record_id") if resolved else None,
            "constraints": merge_constraints(format_constraints, _constraints_from_frustration(frustration, {})),
            "requires_selection": target_scope == "selection",
            "requires_confirmation": True,
            "confidence": 0.94,
            "reasoning": "session_target_continuation",
            "resolved_scope": resolved,
            "frustration_signal": frustration,
            "repetition_detected": repetition.get("repetition_detected"),
            "repetition_instruction": repetition.get("repetition_instruction"),
            "previous_action": previous_action or None,
        }
    elif is_imperative_edit_command(
        msg,
        previous_action=previous_action,
        session_active=session_active,
        resolved_scope=resolved,
    ) and has_active_sop:
        primary_intent = "action"
        use_editor_action = True
        if not resolved:
            resolved = resolve_scope_from_message(msg, sections, has_editor_selection=has_editor_selection)
        section_hint = _section_hint_from_scope(resolved, previous_action)
        target_scope = "full_document" if resolved and resolved.get("level") == "full" else (
            "section" if section_hint or (resolved and resolved.get("target_scope") == "section") else "selection"
        )
        if resolved and resolved.get("level") == "line":
            target_scope = "selection"
        early = {
            "flow": "editor_action",
            "action": detect_edit_action_from_message(msg),
            "target_scope": target_scope,
            "section_hint": section_hint or None,
            "line_number": resolved.get("line_number") if resolved else None,
            "record_id": resolved.get("record_id") if resolved else None,
            "constraints": merge_constraints(format_constraints, _constraints_from_frustration(frustration, {})),
            "requires_selection": target_scope == "selection",
            "requires_confirmation": True,
            "confidence": 0.93,
            "reasoning": "imperative_edit_on_open_sop",
            "previous_action": previous_action or None,
            "resolved_scope": resolved,
            "frustration_signal": frustration,
            "repetition_detected": repetition.get("repetition_detected"),
            "repetition_instruction": repetition.get("repetition_instruction"),
        }
    elif (
        has_active_sop
        and (frustration.get("detected") or is_follow_up_edit_refinement(msg, previous_action))
        and not is_read_only_sop_query(msg, previous_action=previous_action, resolved_scope=resolved)
        and not is_imperative_edit_command(
            msg,
            previous_action=previous_action,
            session_active=session_active,
            resolved_scope=resolved,
        )
    ):
        primary_intent = "followup"
        use_editor_action = True
        section_hint = _section_hint_from_scope(resolved, previous_action)
        constraints: dict[str, Any] = {}
        if frustration.get("detected"):
            constraints = _constraints_from_frustration(frustration, {})
        early = {
            "flow": "follow_up_action",
            "action": detect_edit_action_from_message(msg, previous_action=previous_action),
            "target_scope": "section" if section_hint else "previous_suggestion",
            "section_hint": section_hint or None,
            "constraints": merge_constraints(format_constraints, constraints),
            "requires_selection": False,
            "requires_confirmation": True,
            "confidence": 0.92,
            "reasoning": "follow_up_edit_refinement",
            "resolved_scope": resolved,
            "frustration_signal": frustration,
            "repetition_detected": repetition.get("repetition_detected"),
            "repetition_instruction": repetition.get("repetition_instruction"),
            "previous_action": previous_action or None,
        }
    elif not has_active_sop and re.search(
        r"\b(?:iso|gmp|fda|regulation|standard|best\s+practice|compliance\s+requirement)\b",
        msg,
        re.IGNORECASE,
    ):
        primary_intent = "rag"
        use_rag = True
        use_open_sop = False

    query_analysis = {
        "primary_intent": primary_intent,
        "use_rag": use_rag,
        "use_open_sop": use_open_sop,
        "use_editor_action": use_editor_action,
        "resolved_scope": resolved,
    }
    return {"query_analysis": query_analysis, "early_response": early}


def prepare_message_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Run pre-LLM intelligence (tasks 2, 4, 6 + scope resolution)."""
    message = str(payload.get("message") or payload.get("question") or "").strip()
    session = build_session_from_payload(payload)
    ctx = payload.get("assistant_context") if isinstance(payload.get("assistant_context"), dict) else {}
    current_sop = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}
    sections = current_sop.get("sections") if isinstance(current_sop.get("sections"), list) else []
    sections = enrich_sections_with_aliases(sections)
    recent_messages = payload.get("recent_messages") if isinstance(payload.get("recent_messages"), list) else []
    previous_action = resolve_effective_previous_action(
        ctx,
        session,
        sections=sections,
        recent_messages=recent_messages,
    )

    frustration = detect_frustration(message, session)
    repetition = detect_repetition(message, session)
    resolved_scope = resolve_scope_from_message(
        message,
        sections,
        has_editor_selection=bool(payload.get("has_editor_selection")),
    )
    resolved_scope = merge_resolved_scope_with_session(
        resolved_scope,
        message=message,
        session=session,
        previous_action=previous_action if isinstance(previous_action, dict) else {},
        sections=sections,
    )

    sop_lang = infer_output_language_from_context(ctx)

    turn = analyze_turn_pipeline(
        message,
        session=session,
        sections=sections,
        has_active_sop=bool(payload.get("has_active_sop")),
        has_editor_selection=bool(payload.get("has_editor_selection")),
        previous_action=previous_action,
        resolved_scope=resolved_scope,
        frustration=frustration,
        repetition=repetition,
    )
    early_response = turn.get("early_response")
    if use_llm_orchestrator():
        early_response = None
    if isinstance(early_response, dict) and sop_lang:
        early_response["constraints"] = merge_constraints(
            early_response.get("constraints") if isinstance(early_response.get("constraints"), dict) else {},
            {"language": sop_lang},
        )
    query_analysis = turn.get("query_analysis")

    return {
        "session": session,
        "sections": sections,
        "frustration_signal": frustration,
        "repetition_detected": repetition.get("repetition_detected"),
        "repetition_instruction": repetition.get("repetition_instruction"),
        "resolved_scope": resolved_scope,
        "early_response": early_response,
        "query_analysis": query_analysis,
        "active_scope": session["active_scope"],
        "instruction_memory": session["instruction_memory"],
        "has_active_sop": bool(payload.get("has_active_sop")),
        "sop_output_language": sop_lang,
        "assistant_context": ctx,
    }


def apply_prep_to_classification(
    result: dict[str, Any],
    prep: dict[str, Any],
    *,
    user_message: str = "",
) -> dict[str, Any]:
    """Merge scope/frustration/repetition into classifier output."""
    out = dict(result)
    if use_llm_orchestrator():
        prep = {**prep, "_user_message": user_message}
        out = apply_llm_scope_enrichment(out, prep)
        out = apply_orchestrator_guardrails(out, prep, user_message=user_message)
        return out

    resolved = prep.get("resolved_scope")
    frustration = prep.get("frustration_signal") or {}

    if resolved and resolved.get("level") == "full":
        action = str(out.get("action") or "").strip().lower()
        if out.get("flow") in {"editor_action", "follow_up_action"} and action in INLINE_EDITOR_ACTIONS | BRIDGE_EDITOR_ACTIONS:
            out["target_scope"] = "full_document"
            out["section_hint"] = None
            out["requires_selection"] = False
        else:
            out["flow"] = "chat"
            out["action"] = None
            out["target_scope"] = "full_document"
            out["section_hint"] = None
    elif resolved and out.get("flow") in {"editor_action", "follow_up_action"}:
        out["target_scope"] = resolved.get("target_scope") or out.get("target_scope")
        if resolved.get("section_label") and resolved.get("level") in {"section", "sub_section", "record"}:
            out["section_hint"] = resolved.get("section_label")
        elif resolved.get("resolved_from") == "SESSION_MEMORY" and resolved.get("section_label"):
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
    if prep.get("sop_output_language"):
        out["constraints"] = merge_constraints(
            out.get("constraints") if isinstance(out.get("constraints"), dict) else {},
            {"language": prep["sop_output_language"]},
        )
    return merge_editorial_profile_metadata(out, user_message, prep)


def apply_llm_scope_enrichment(out: dict[str, Any], prep: dict[str, Any]) -> dict[str, Any]:
    """Merge deterministic scope hints into LLM output without overriding flow/action."""
    merged = dict(out)
    resolved = prep.get("resolved_scope") if isinstance(prep.get("resolved_scope"), dict) else {}
    frustration = prep.get("frustration_signal") or {}

    if resolved and merged.get("flow") in {"editor_action", "follow_up_action"}:
        level = str(resolved.get("level") or "").lower()
        if level == "full":
            merged["target_scope"] = "full_document"
            merged["section_hint"] = None
            merged["requires_selection"] = False
        elif level in {"section", "sub_section", "record"} and resolved.get("section_label"):
            if not merged.get("section_hint"):
                merged["section_hint"] = resolved.get("section_label")
            if not merged.get("target_scope") or merged.get("target_scope") == "selection":
                merged["target_scope"] = resolved.get("target_scope") or "section"
        elif level == "line":
            merged["target_scope"] = "selection"
            merged["line_number"] = resolved.get("line_number")
        if resolved.get("record_id"):
            merged["record_id"] = resolved.get("record_id")

    if frustration.get("detected") and merged.get("flow") in {"editor_action", "follow_up_action"}:
        merged["constraints"] = _constraints_from_frustration(
            frustration,
            merged.get("constraints") if isinstance(merged.get("constraints"), dict) else {},
        )

    if prep.get("repetition_detected"):
        merged["repetition_detected"] = True
        merged["repetition_instruction"] = prep.get("repetition_instruction")

    merged["frustration_signal"] = frustration
    merged["resolved_scope"] = resolved
    merged["source_content_override"] = build_source_content_override(
        merged.get("action"),
        prep.get("session") or {},
        resolved_section_id=str((resolved or {}).get("section_id") or ""),
    )
    if prep.get("sop_output_language"):
        merged["constraints"] = merge_constraints(
            merged.get("constraints") if isinstance(merged.get("constraints"), dict) else {},
            {"language": prep["sop_output_language"]},
        )
    sections = prep.get("sections") if isinstance(prep.get("sections"), list) else []
    return enforce_full_section_body_target(merged, str(prep.get("_user_message") or ""), sections)


def apply_orchestrator_guardrails(
    out: dict[str, Any],
    prep: dict[str, Any],
    *,
    user_message: str = "",
) -> dict[str, Any]:
    """Minimal post-LLM policy: summarize→chat, meta/explain shortcuts, session follow-ups."""
    guarded = dict(out)
    msg = str(user_message or "").strip()
    ctx = prep.get("assistant_context") if isinstance(prep.get("assistant_context"), dict) else {}
    previous_action = ctx.get("last_action") if isinstance(ctx.get("last_action"), dict) else {}
    session = prep.get("session") or {}
    prev_from_session = resolve_effective_previous_action(
        ctx,
        session if isinstance(session, dict) else {},
        sections=prep.get("sections") if isinstance(prep.get("sections"), list) else [],
        recent_messages=[],
    )
    previous_action = previous_action or prev_from_session or {}

    if prep.get("has_active_sop") and is_gap_check_command(msg):
        return enforce_gap_check_editor_route(guarded, msg, prep)

    if prep.get("has_active_sop"):
        guarded = enforce_editorial_profile_editor_route(guarded, msg, prep)

    if is_meta_question_about_assistant_output(msg):
        guarded["flow"] = "chat"
        guarded["action"] = None
        guarded["chat_submode"] = "explain_last_edit_diff"
        guarded["reasoning"] = "orchestrator_guardrail_meta_about_edit"
    elif is_explain_recent_output_query(msg, previous_action):
        guarded["flow"] = "chat"
        guarded["action"] = None
        guarded["chat_submode"] = "explain_last_output"
        guarded["reasoning"] = "orchestrator_guardrail_explain_preview"
    elif (
        str(guarded.get("action") or "").lower() == "summarize"
        or is_summarize_chat_query(msg)
    ):
        guarded["flow"] = "chat"
        guarded["action"] = None
        guarded["chat_submode"] = "sop_summarize"
        guarded["preserve_active_scope"] = True
        guarded["reasoning"] = "orchestrator_guardrail_summarize_sidebar_only"

    flow = str(guarded.get("flow") or "").lower()
    if (
        flow == "editor_action"
        and previous_action
        and is_follow_up_edit_refinement(msg, previous_action)
        and not is_imperative_edit_command(msg, previous_action=previous_action)
    ):
        guarded["flow"] = "follow_up_action"
        guarded["previous_action"] = previous_action
        if not guarded.get("section_hint") and previous_action.get("section_name"):
            guarded["section_hint"] = previous_action.get("section_name")
        if not guarded.get("target_scope"):
            guarded["target_scope"] = previous_action.get("target_scope") or "previous_suggestion"

    if flow in {"editor_action", "follow_up_action"}:
        guarded["requires_confirmation"] = True
        if guarded.get("target_scope") == "selection":
            guarded["requires_selection"] = True
        else:
            guarded["requires_selection"] = False

    if not prep.get("has_active_sop") and flow in {"editor_action", "follow_up_action"}:
        guarded["flow"] = "clarify"
        guarded["action"] = None
        guarded["clarification_question"] = (
            "Bitte öffnen Sie zuerst eine SOP im Editor, damit ich diese Aktion ausführen kann."
        )

    sections = prep.get("sections") if isinstance(prep.get("sections"), list) else []
    return enforce_full_section_body_target(guarded, msg, sections)


def build_intent_classifier_invoke_context(
    payload: dict[str, Any],
    prep: dict[str, Any],
) -> dict[str, Any]:
    """Build kwargs for classify_assistant_intent from API payload + prepare_message_context."""
    ctx = payload.get("assistant_context") if isinstance(payload.get("assistant_context"), dict) else {}
    current_sop = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}
    editor_contract = ctx.get("editor_context_contract") if isinstance(ctx.get("editor_context_contract"), dict) else {}
    selected_section = ctx.get("selected_section") if isinstance(ctx.get("selected_section"), dict) else {}
    recent_messages = payload.get("recent_messages") if isinstance(payload.get("recent_messages"), list) else []
    last_focus = ctx.get("last_focus") if isinstance(ctx.get("last_focus"), dict) else {}
    previous_action = resolve_effective_previous_action(
        ctx,
        prep.get("session") or {},
        sections=prep.get("sections") if isinstance(prep.get("sections"), list) else [],
        recent_messages=recent_messages,
    )

    def _compact(value: object, limit: int = 700) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        return text[:limit]

    contract_selected = editor_contract.get("selected_section") if isinstance(editor_contract.get("selected_section"), dict) else {}
    selected_summary = _compact(
        f"{selected_section.get('label') or selected_section.get('name') or contract_selected.get('label') or 'none'}"
        f" :: {selected_section.get('content') or contract_selected.get('content') or ''}",
        900,
    )

    sections = current_sop.get("sections")
    if not isinstance(sections, list):
        sop_ctx = editor_contract.get("sop_context") if isinstance(editor_contract.get("sop_context"), dict) else {}
        sections = sop_ctx.get("sections") if isinstance(sop_ctx.get("sections"), list) else []
    labels = [
        str(s.get("label") or "").strip()
        for s in (sections or [])[:40]
        if isinstance(s, dict) and str(s.get("label") or "").strip()
    ]

    prev_summary = ""
    if previous_action:
        prev_summary = _compact(
            " | ".join(
                [
                    f"action={previous_action.get('action') or ''}",
                    f"scope={previous_action.get('target_scope') or ''}",
                    f"section={previous_action.get('section_name') or ''}",
                    f"prompt={previous_action.get('request_prompt') or ''}",
                ]
            ),
            900,
        )
    elif last_focus:
        prev_summary = _compact(
            f"focus | scope={last_focus.get('target_scope')} | section={last_focus.get('section_name')}",
            900,
        )

    conv_rows = []
    for row in recent_messages[-8:]:
        if isinstance(row, dict) and row.get("content"):
            conv_rows.append(f"{row.get('role') or 'user'}: {_compact(row.get('content'), 260)}")

    has_active_sop = bool(prep.get("has_active_sop"))
    if not has_active_sop:
        has_active_sop = bool(
            str(ctx.get("active_sop_id") or ctx.get("current_document_id") or "").strip()
            or str(current_sop.get("id") or "").strip()
        )

    resolved = prep.get("resolved_scope") if isinstance(prep.get("resolved_scope"), dict) else {}
    query_analysis = prep.get("query_analysis") if isinstance(prep.get("query_analysis"), dict) else {}

    return {
        "has_active_sop": has_active_sop,
        "has_editor_selection": bool(payload.get("has_editor_selection")),
        "route": str(payload.get("route") or ctx.get("route") or "").strip(),
        "active_sop_title": str(current_sop.get("title") or "").strip(),
        "active_sop_number": str(current_sop.get("sop_number") or current_sop.get("documentId") or "").strip(),
        "selected_section_summary": selected_summary,
        "available_sections": _compact(", ".join(labels), 1200),
        "previous_action_summary": prev_summary,
        "recent_conversation": "\n".join(conv_rows),
        "active_scope": prep.get("active_scope"),
        "instruction_memory": prep.get("instruction_memory"),
        "frustration_signal": prep.get("frustration_signal"),
        "repetition_detected": bool(prep.get("repetition_detected")),
        "repetition_instruction": prep.get("repetition_instruction"),
        "resolved_scope_hint": json.dumps(resolved, ensure_ascii=False)[:1500] if resolved else "-",
        "query_analysis_hint": json.dumps(query_analysis, ensure_ascii=False)[:800] if query_analysis else "-",
        "previous_action": previous_action,
    }


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
        elif response.get("flow") in {"editor_action", "follow_up_action"}:
            active["last_action"] = response.get("action")
            label = response.get("section_hint") or active.get("section_label")
            if label:
                active["section_label"] = label
                active["section_id"] = label
            prev = response.get("previous_action") if isinstance(response.get("previous_action"), dict) else {}
            if prev.get("original_text_excerpt"):
                active["last_original"] = prev.get("original_text_excerpt")
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
        instruction_memory.append(
            {
                "role": "user",
                "content": user_message[:400],
                "action": response.get("action"),
                "section": active.get("section_label"),
                "target_scope": response.get("target_scope") or active.get("target_scope"),
            }
        )
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


INLINE_EDITOR_ACTIONS = frozenset({"rewrite", "improve", "gap_check"})
BRIDGE_EDITOR_ACTIONS = frozenset({"analyze", "compare", "read"})


def derive_sidebar_intent(out: dict[str, Any], *, has_active_sop: bool) -> str:
    """High-level sidebar intent for the client (no client-side keyword rules)."""
    flow = str(out.get("flow") or "").lower()
    action = str(out.get("action") or "").strip().lower()
    if flow == "clarify":
        return "clarify"
    if flow == "follow_up_action":
        return "followup"
    if flow == "editor_action" or action in INLINE_EDITOR_ACTIONS or action in BRIDGE_EDITOR_ACTIONS:
        return "action"
    if not has_active_sop:
        return "rag"
    return "sop_query"


def build_target_resolution(out: dict[str, Any]) -> dict[str, Any]:
    """Explicit editor target hints — client only maps positions in TipTap."""
    resolved = out.get("resolved_scope") if isinstance(out.get("resolved_scope"), dict) else {}
    section_hint = str(out.get("section_hint") or resolved.get("section_label") or "").strip() or None
    target_scope = str(out.get("target_scope") or resolved.get("target_scope") or "selection").strip()
    line_number = out.get("line_number") if out.get("line_number") is not None else resolved.get("line_number")
    record_id = str(out.get("record_id") or resolved.get("record_id") or "").strip() or None
    level = str(resolved.get("level") or "").lower()
    constraints = out.get("constraints") if isinstance(out.get("constraints"), dict) else {}
    prefer_full_section = target_scope != "full_document" and (
        level in {"section", "sub_section"}
        or bool(section_hint and target_scope == "section")
        or "full_section_body" in str(constraints.get("detail_level") or "").lower()
        or bool(resolved.get("prefer_full_section_body"))
    )
    return {
        "target_scope": target_scope,
        "section_hint": section_hint,
        "line_number": line_number,
        "record_id": record_id,
        "prefer_full_section": prefer_full_section,
        "resolved_from": resolved.get("resolved_from"),
    }


def build_enriched_instruction(
    user_message: str,
    out: dict[str, Any],
    *,
    prep: dict[str, Any] | None = None,
) -> str:
    """Instruction text for /api/ai/action — built server-side only."""
    base = str(user_message or "").strip()
    if not base:
        return ""
    hints: list[str] = []
    ctx = (prep or {}).get("assistant_context") if isinstance((prep or {}).get("assistant_context"), dict) else {}
    current_sop = ctx.get("current_sop") if isinstance(ctx.get("current_sop"), dict) else {}
    from chatbot.assistant.profile_reference import build_editorial_profile_hints

    from chatbot.assistant.profile_reference import extract_editorial_profile_reference

    editorial_ref = out.get("editorial_profile_reference") or extract_editorial_profile_reference(user_message)
    hints.extend(
        build_editorial_profile_hints(
            user_message,
            open_sop_number=str(current_sop.get("sop_number") or current_sop.get("documentId") or "").strip(),
            open_sop_title=str(current_sop.get("title") or "").strip(),
        )
    )
    c = out.get("constraints") if isinstance(out.get("constraints"), dict) else {}

    lang = str(c.get("language") or "").strip().lower()
    if lang == "de":
        hints.append(
            "Ausgabesprache: Deutsch — wie im geöffneten SOP-Fließtext und Profil, "
            "auch wenn die Chat-Anfrage auf Englisch ist."
        )
    elif lang == "en":
        hints.append(
            "Output language: English — match the open SOP body text and profile, "
            "even if the user's chat message is in another language."
        )
    else:
        hints.append(
            "Output language: keep the same language as the target SOP text/profile "
            "unless the user explicitly requests another language."
        )
    if not editorial_ref:
        hints.append(
            "Apply the active client profile.md and stored NLP parameters from the database "
            "(rewrite_rules, terminology, workflow patterns, rewrite_improve_parameters) — already injected server-side. "
            "Use the profile to reshape procedures, controls, and sensitive operational wording in the target TEXT, "
            "not only tone or sentence rhythm. Preserve all record IDs, dates, and system identifiers."
        )
    fmt = extract_format_constraints(user_message)
    if fmt.get("line_count") and not c.get("line_count"):
        c = {**c, "line_count": fmt["line_count"]}
    if fmt.get("word_count") and not c.get("word_count"):
        c = {**c, "word_count": fmt["word_count"]}
    if c.get("tone"):
        hints.append(f"Tone: {c['tone']}")
    if c.get("word_count"):
        hints.append(f"Target length: about {c['word_count']} words")
    if c.get("line_count"):
        n = int(c["line_count"])
        hints.append(
            f"MANDATORY OUTPUT SHAPE: exactly {n} lines (newline-separated plain sentences). "
            f"Not more than {n} lines. This overrides default SOP length rules (70–130% of source)."
        )
    if c.get("length") == "shorter":
        hints.append("Make the result shorter than the source.")
    if c.get("length") == "longer":
        hints.append("Expand the result with more detail.")
    if c.get("language"):
        hints.append(f"Output language: {c['language']}")
    if c.get("detail_level"):
        hints.append(f"Detail level: {c['detail_level']}")
    if "full_section_body" in str(c.get("detail_level") or "").lower():
        hints.append(
            "MANDATORY SCOPE: Rewrite/improve ALL body paragraphs under the target section heading "
            "until the next section heading — the complete section body. "
            "Do NOT change only the heading line, a single word, or a small editor highlight."
        )
    if c.get("format") == "concise_sidebar_summary":
        hints.append(
            "SUMMARY (sidebar chat only — do not modify the SOP): Reply with a short summary strictly "
            "shorter than the source (about 25–40% of source length). Use 3–6 tight bullets or at most "
            "2 brief paragraphs. No full section copy-paste."
        )
    elif c.get("format"):
        hints.append(f"Format: {c['format']}")

    if str(out.get("action") or "").lower() == "gap_check":
        hints.append(
            "GAP CHECK: Analyze compliance/risk gaps in the target SOP section text. "
            "Use hybrid RAG context plus the section body. Output structured findings for Accept/Reject in the editor — "
            "not a sidebar-only list of CAPA database records."
        )
    if out.get("section_hint"):
        hints.append(f"Target section: {out['section_hint']}")
        hints.append(
            "Apply to the complete section body under that heading (all paragraphs until the next section), "
            "not the heading line alone."
        )
    prev = out.get("previous_action") if isinstance(out.get("previous_action"), dict) else {}
    if prev.get("action"):
        hints.append(f"Previous assistant action: {prev['action']}")
    resolved_scope_obj = out.get("resolved_scope") if isinstance(out.get("resolved_scope"), dict) else {}
    explicit_new_target = str(resolved_scope_obj.get("resolved_from") or "") in {
        "ALIAS_MATCH",
        "SECTION_PHRASE",
        "SUB_SECTION",
        "RECORD_ID",
    }
    if prev.get("section_name") and not out.get("section_hint") and not explicit_new_target:
        hints.append(f"Continue working on the same target section: {prev['section_name']}")
    if prev.get("target_scope"):
        hints.append(f"Previous target scope: {prev['target_scope']}")
    if prev.get("request_prompt"):
        hints.append(f"Previous instruction: {str(prev['request_prompt'])[:400]}")
    if str(resolved_scope_obj.get("resolved_from") or "") == "SESSION_MEMORY":
        hints.append(
            "Session memory: continue on the same target section/SOP as the previous turn in this chat "
            "(user did not name a new heading)."
        )
    if out.get("target_scope") == "full_document":
        hints.append(
            "Scope: entire open SOP document. Use the full section_text supplied for this action "
            "(all headings, sections, and traceability blocks). Read the complete document before rewriting."
        )
    if out.get("target_scope") == "selection":
        hints.append("Apply only to the current editor selection.")
    if out.get("line_number"):
        hints.append(f"Target line number: {out['line_number']}")
    if out.get("record_id"):
        hints.append(f"Target record entry: {out['record_id']}")
    override = out.get("source_content_override")
    if isinstance(override, dict) and override.get("enabled"):
        hints.append("Operate on the previous assistant output for this section, not the original SOP source.")
    if out.get("repetition_instruction"):
        hints.append(str(out["repetition_instruction"]))
    frustration = out.get("frustration_signal") if isinstance(out.get("frustration_signal"), dict) else {}
    if frustration.get("detected"):
        twc = frustration.get("target_word_count")
        if twc:
            hints.append(f"Target word count: about {twc} words")
        hints.append("User is refining the previous result — keep the same section target.")
    linked = out.get("linked_entity_types")
    if out.get("target_scope") == "linked_context" and isinstance(linked, list) and linked:
        hints.append(f"Focus on linked records: {', '.join(str(x) for x in linked)}")

    if not hints:
        return base
    return f"{base}\n\n[Assistant constraints]\n" + "\n".join(hints)


def attach_sidebar_routing(
    out: dict[str, Any],
    *,
    has_active_sop: bool,
    user_message: str = "",
    prep: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach client routing flags so the sidebar does not re-derive intent in JS."""
    if str(out.get("action") or "").strip().lower() == "summarize" or is_summarize_chat_query(user_message):
        out["flow"] = "chat"
        out["action"] = None
        out["chat_submode"] = out.get("chat_submode") or "sop_summarize"
        c = out.get("constraints") if isinstance(out.get("constraints"), dict) else {}
        out["constraints"] = merge_constraints(c, {"length": "shorter", "format": "concise_sidebar_summary"})
    flow = str(out.get("flow") or "").lower()
    action = str(out.get("action") or "").strip().lower()
    sidebar_intent = derive_sidebar_intent(out, has_active_sop=has_active_sop)

    run_editor = bool(
        has_active_sop
        and sidebar_intent in {"action", "followup"}
        and (
            flow in {"editor_action", "follow_up_action"}
            or action in INLINE_EDITOR_ACTIONS
            or action in BRIDGE_EDITOR_ACTIONS
        )
    )
    run_query = bool(
        flow == "chat"
        or sidebar_intent in {"rag", "sop_query"}
        or (sidebar_intent == "clarify" and not out.get("clarification_question"))
    )
    if run_editor:
        run_query = False
    if sidebar_intent == "clarify" and out.get("clarification_question"):
        run_query = False

    out["sidebar_intent"] = sidebar_intent
    out["run_editor_action"] = run_editor
    out["run_query"] = run_query
    out["enriched_instruction"] = build_enriched_instruction(user_message, out, prep=prep)
    out["target_resolution"] = build_target_resolution(out)
    return out


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
        out = apply_prep_to_classification(dict(raw), prep, user_message=user_message)

    session = prep.get("session") or {}
    preserve = bool(out.get("preserve_active_scope"))

    updated_scope = dict(session.get("active_scope") or default_active_scope())
    if out.get("flow") in {"editor_action", "follow_up_action"}:
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
        updated_scope["target_scope"] = out.get("target_scope") or resolved.get("target_scope")
    elif preserve:
        updated_scope = dict(session.get("scope_before_full_doc") or updated_scope)
        if out.get("chat_submode") == "sop_summarize":
            resolved = out.get("resolved_scope") if isinstance(out.get("resolved_scope"), dict) else {}
            label = str(out.get("section_hint") or resolved.get("section_label") or "").strip()
            if label:
                updated_scope["section_label"] = label
                updated_scope["section_id"] = str(resolved.get("section_id") or label).strip()
            updated_scope["last_action"] = "summarize"
            scope = str(out.get("target_scope") or resolved.get("target_scope") or "section").strip()
            if scope:
                updated_scope["target_scope"] = scope

    if out.get("chat_submode") in {"explain_last_edit_diff", "explain_last_output"}:
        prev = out.get("previous_action") if isinstance(out.get("previous_action"), dict) else {}
        fc = out.get("constraints") if isinstance(out.get("constraints"), dict) else {}
        if prep.get("sop_output_language"):
            fc = merge_constraints(fc, {"language": prep["sop_output_language"]})
        ctx = prep.get("assistant_context") if isinstance(prep.get("assistant_context"), dict) else {}
        if out.get("chat_submode") == "explain_last_output":
            out["assistant_message"] = build_explain_last_output_answer(
                user_message,
                prev,
                format_constraints=fc,
                assistant_context=ctx,
            )
        else:
            out["assistant_message"] = build_diff_explanation_answer(
                user_message,
                prev,
                format_constraints=fc,
                active_scope=updated_scope,
                assistant_context=ctx,
            )

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
    out = enforce_editorial_profile_editor_route(out, user_message, prep)
    attach_sidebar_routing(
        out,
        has_active_sop=bool(prep.get("has_active_sop")),
        user_message=user_message,
        prep=prep,
    )
    if prep.get("query_analysis"):
        out["query_analysis"] = prep["query_analysis"]
    return out
