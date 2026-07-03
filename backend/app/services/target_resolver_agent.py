"""DeepAgents-based semantic target resolver for the live SOP editor.

The DeepAgent understands the user query, but it must choose from live editor
candidate IDs only. The frontend remains the final authority for TipTap ranges.
"""

from __future__ import annotations

import json
import re
from typing import Any

from deepagents import create_deep_agent

from chatbot.llm.provider import create_chat_llm, get_local_llm_config


ALLOWED_TARGET_TYPES = {
    "full_document",
    "section",
    "table",
    "table_section",
    "paragraph",
    "selection",
}


class TargetResolverAgentError(RuntimeError):
    """Raised when the DeepAgent target resolver cannot produce usable JSON."""


def _message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or item.get("content") or item)
            if isinstance(item, dict)
            else str(item)
            for item in content
        ).strip()
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


def _extract_json_object(raw: object) -> dict[str, Any]:
    text = _message_content(raw)
    if not text:
        raise TargetResolverAgentError("DeepAgent returned empty content.")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S).strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise TargetResolverAgentError(f"DeepAgent did not return JSON: {text[:240]}")
    try:
        value = json.loads(match.group(0))
    except Exception as exc:
        raise TargetResolverAgentError(f"DeepAgent JSON parse failed: {exc}") from exc
    if not isinstance(value, dict):
        raise TargetResolverAgentError("DeepAgent JSON was not an object.")
    return value


def _compact_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, item in enumerate(items[:limit]):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        # IDs are issued by the live TipTap document. Never manufacture an ID
        # in the backend because the editor could not validate or apply it.
        if not item_id:
            continue
        label = str(
            item.get("label")
            or item.get("caption")
            or item.get("sectionName")
            or item.get("title")
            or ""
        ).strip()
        out.append(
            {
                "index": index,
                "id": item_id,
                "label": label,
                "target_type": item.get("target_type") or item.get("type") or "section",
                "from": item.get("from"),
                "to": item.get("to"),
                "owning_section": item.get("owning_section") or item.get("owningSection") or "",
                "row_count": item.get("rowCount") or item.get("row_count") or 0,
                "column_count": item.get("columnCount") or item.get("column_count") or 0,
                "text_excerpt": str(item.get("text") or item.get("text_excerpt") or "")[:900],
            }
        )
    return out


def _normalize_result(
    parsed: dict[str, Any],
    *,
    model_name: str,
    valid_targets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_type = str(parsed.get("target_type") or "").strip().lower()
    if target_type not in ALLOWED_TARGET_TYPES:
        target_type = "section"
    target_id = str(parsed.get("target_id") or "").strip() or None
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0))))
    except Exception:
        confidence = 0.0
    candidates = parsed.get("candidate_targets") if isinstance(parsed.get("candidate_targets"), list) else []
    normalized_candidates = []
    for item in candidates[:8]:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("id") or item.get("target_id") or "").strip()
        label = str(item.get("label") or item.get("target_label") or "").strip()
        if not label and candidate_id in valid_targets:
            label = str(valid_targets[candidate_id].get("label") or candidate_id)
        if not label:
            continue
        candidate_type = str(item.get("target_type") or item.get("type") or target_type).strip().lower()
        if candidate_id and candidate_id not in valid_targets:
            continue
        normalized_candidates.append(
            {
                "id": candidate_id or None,
                "target_id": candidate_id or None,
                "label": label,
                "target_type": candidate_type,
                "reason": str(item.get("reason") or "DeepAgent candidate").strip(),
            }
        )

    if target_id and target_id not in valid_targets:
        return {
            "target_type": target_type,
            "target_id": None,
            "target_label": None,
            "owning_section": None,
            "confidence": min(confidence, 0.35),
            "requires_clarification": True,
            "candidate_targets": normalized_candidates,
            "reasoning_summary": (
                f"DeepAgent returned target_id '{target_id}', but that ID is not present in the live editor snapshot."
            )[:800],
            "source": "deep_agent_target_resolver",
            "agent_mode": "deep_agent",
            "llm_model": model_name,
        }

    if target_id and target_id in valid_targets:
        target = valid_targets[target_id]
        target_type = str(target.get("target_type") or target.get("type") or target_type).strip().lower()
        parsed["target_label"] = parsed.get("target_label") or target.get("label")
        parsed["owning_section"] = parsed.get("owning_section") or target.get("owning_section")
        parsed["requires_clarification"] = False
        normalized_candidates = []
        confidence = max(confidence, 0.72)

    if not target_id and parsed.get("target_label"):
        wanted_label = str(parsed.get("target_label") or "").strip().lower()
        wanted_type = target_type
        exact = [
            item
            for item in valid_targets.values()
            if str(item.get("label") or "").strip().lower() == wanted_label
            and str(item.get("target_type") or item.get("type") or "").strip().lower() == wanted_type
        ]
        if len(exact) == 1:
            target_id = str(exact[0].get("id") or "").strip() or None
            confidence = max(confidence, 0.68)

    return {
        "target_type": target_type,
        "target_id": target_id,
        "target_label": str(parsed.get("target_label") or "").strip() or None,
        "owning_section": str(parsed.get("owning_section") or parsed.get("owningSection") or "").strip() or None,
        "confidence": confidence,
        "requires_clarification": bool(parsed.get("requires_clarification")) or (confidence < 0.45 and not target_id and not parsed.get("target_label")),
        "candidate_targets": normalized_candidates,
        "reasoning_summary": str(parsed.get("reasoning_summary") or "DeepAgent target analysis completed.")[:800],
        "source": "deep_agent_target_resolver",
        "agent_mode": "deep_agent",
        "llm_model": model_name,
    }


def resolve_sop_target_with_deep_agent(
    *,
    user_query: str,
    action: str,
    sections: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    selection: dict[str, Any],
    active_scope: dict[str, Any],
    sop_metadata: dict[str, Any],
    document_excerpt: str,
    paragraphs: list[dict[str, Any]] | None = None,
    document_tree: list[dict[str, Any]] | None = None,
    document_schema: str = "",
    full_text: str = "",
) -> dict[str, Any]:
    """Use a LangChain DeepAgent to choose the semantic SOP target."""

    compact_sections = _compact_items(sections, 120)
    compact_tables = _compact_items(tables, 80)
    compact_paragraphs = _compact_items(paragraphs or [], 160)
    compact_tree = _compact_items(document_tree or [], 220)
    selection_payload = selection if isinstance(selection, dict) else {}
    active_scope_payload = active_scope if isinstance(active_scope, dict) else {}
    metadata_payload = sop_metadata if isinstance(sop_metadata, dict) else {}
    valid_targets: dict[str, dict[str, Any]] = {}
    for item in [*compact_sections, *compact_tables, *compact_paragraphs, *compact_tree]:
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            continue
        existing = valid_targets.get(item_id)
        if existing:
            existing_signature = (
                str(existing.get("target_type") or ""),
                existing.get("from"),
                existing.get("to"),
            )
            incoming_signature = (
                str(item.get("target_type") or ""),
                item.get("from"),
                item.get("to"),
            )
            if existing_signature != incoming_signature:
                raise TargetResolverAgentError(
                    f"Live editor target ID collision for '{item_id}'. Refresh the editor target map."
                )
            continue
        valid_targets[item_id] = item
    if selection_payload and selection_payload.get("empty") is not True:
        valid_targets.setdefault(
            "selection",
            {
                "id": "selection",
                "label": "Selected text",
                "target_type": "selection",
                "from": selection_payload.get("from"),
                "to": selection_payload.get("to"),
                "owning_section": "",
                "text_excerpt": str(selection_payload.get("text") or "")[:900],
            },
        )
    valid_targets.setdefault(
        "doc_root",
        {
            "id": "doc_root",
            "label": "Current SOP",
            "target_type": "full_document",
            "from": 0,
            "to": None,
            "owning_section": "",
            "text_excerpt": str(document_excerpt or "")[:900],
        },
    )

    def list_editor_sections(query: str = "", limit: int = 30) -> dict[str, Any]:
        """Return live editor section candidates with labels/ranges/excerpts."""
        q = str(query or "").strip().lower()
        items = compact_sections
        if q:
            q_tokens = {t for t in re.findall(r"[a-zA-Z0-9ÄÖÜäöüß]+", q) if len(t) >= 2}
            scored = []
            for section in items:
                haystack = f"{section.get('label', '')} {section.get('text_excerpt', '')}".lower()
                score = sum(1 for token in q_tokens if token in haystack)
                scored.append((score, section))
            items = [section for score, section in sorted(scored, key=lambda x: x[0], reverse=True) if score > 0] or items
        return {"sections": items[: max(1, min(int(limit or 30), 80))]}

    def list_editor_tables(query: str = "", limit: int = 30) -> dict[str, Any]:
        """Return live editor table candidates with captions/owning sections/ranges/excerpts."""
        q = str(query or "").strip().lower()
        items = compact_tables
        if q:
            q_tokens = {t for t in re.findall(r"[a-zA-Z0-9ÄÖÜäöüß]+", q) if len(t) >= 2}
            scored = []
            for table in items:
                haystack = f"{table.get('label', '')} {table.get('owning_section', '')} {table.get('text_excerpt', '')}".lower()
                score = sum(1 for token in q_tokens if token in haystack)
                scored.append((score, table))
            items = [table for score, table in sorted(scored, key=lambda x: x[0], reverse=True) if score > 0] or items
        return {"tables": items[: max(1, min(int(limit or 30), 80))]}

    def list_editor_paragraphs(query: str = "", limit: int = 30) -> dict[str, Any]:
        """Return live editor paragraph candidates for local span requests."""
        q = str(query or "").strip().lower()
        items = compact_paragraphs
        if q:
            q_tokens = {t for t in re.findall(r"[a-zA-Z0-9Ã„Ã–ÃœÃ¤Ã¶Ã¼ÃŸ]+", q) if len(t) >= 2}
            scored = []
            for paragraph in items:
                haystack = f"{paragraph.get('label', '')} {paragraph.get('text_excerpt', '')}".lower()
                score = sum(1 for token in q_tokens if token in haystack)
                scored.append((score, paragraph))
            items = [paragraph for score, paragraph in sorted(scored, key=lambda x: x[0], reverse=True) if score > 0] or items
        return {"paragraphs": items[: max(1, min(int(limit or 30), 80))]}

    def inspect_document_outline() -> dict[str, Any]:
        """Return compact live document outline IDs/types/labels."""
        return {
            "outline": [
                {
                    "id": item.get("id"),
                    "target_type": item.get("target_type"),
                    "label": item.get("label"),
                    "owning_section": item.get("owning_section"),
                    "from": item.get("from"),
                    "to": item.get("to"),
                }
                for item in compact_tree[:160]
            ],
            "counts": {
                "sections": len(compact_sections),
                "tables": len(compact_tables),
                "paragraphs": len(compact_paragraphs),
            },
        }

    def inspect_candidate(target_id: str) -> dict[str, Any]:
        """Return one live editor candidate by exact ID."""
        candidate = valid_targets.get(str(target_id or "").strip())
        return {"candidate": candidate, "found": bool(candidate)}

    def get_block_content(block_id: str) -> dict[str, Any]:
        """Fetch full available content for a live editor block by exact ID."""
        candidate = valid_targets.get(str(block_id or "").strip())
        return {"block": candidate, "found": bool(candidate)}

    def search_by_meaning(query: str = "", limit: int = 5) -> dict[str, Any]:
        """Find live editor targets whose label/content overlaps a phrase or concept."""
        q_tokens = {t for t in re.findall(r"[a-zA-Z0-9]+", str(query or "").lower()) if len(t) >= 2}
        scored = []
        for candidate in valid_targets.values():
            haystack = f"{candidate.get('label', '')} {candidate.get('owning_section', '')} {candidate.get('text_excerpt', '')}".lower()
            score = sum(1 for token in q_tokens if token in haystack)
            if score > 0:
                scored.append((score, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        return {
            "matches": [
                {
                    "id": item.get("id"),
                    "target_type": item.get("target_type"),
                    "label": item.get("label"),
                    "owning_section": item.get("owning_section"),
                    "text_excerpt": item.get("text_excerpt"),
                    "relevance_reason": f"{score} query tokens matched label/content",
                }
                for score, item in scored[: max(1, min(int(limit or 5), 10))]
            ]
        }

    def inspect_selection() -> dict[str, Any]:
        """Return the current live editor selection and active scope."""
        return {
            "selection": selection_payload,
            "active_scope": active_scope_payload,
            "selection_rule": "Use selection only when the user says selected/highlighted/this selection/this paragraph.",
        }

    def inspect_document_excerpt() -> dict[str, Any]:
        """Return SOP metadata and bounded document text for semantic understanding."""
        return {
            "sop_metadata": metadata_payload,
            "document_schema": str(document_schema or "")[:12000],
            "document_excerpt": str(document_excerpt or "")[:9000],
            "full_text_excerpt": str(full_text or "")[:20000],
        }

    system_prompt = """You are the DeepAgents Target Resolver for a regulated SOP rich-text editor.

Your only job is to understand WHERE the user wants the action applied.
You must use the tools to inspect the live editor sections/tables/selection before deciding.
Do not rewrite or improve content.

Critical behavior:
- You must choose target_id from tool results. Never invent target IDs.
- If no valid candidate ID is available, set requires_clarification=true.
- Multilingual / Cross-Lingual matching: The user query may be in a different language (e.g., German) than the SOP's sections and headings (e.g., English), or vice-versa. Translate the target concept dynamically to match the live candidate labels (for example, "Geltungsbereich" maps to "Scope", "Änderungsverlauf" or "Dokumentenhistorie" maps to "Document History", etc.).
- If the user asks for a named table, call list_editor_tables and choose target_type="table" with target_id from the table candidate.
- "improve the document history table" must target the Document History table, never Scope.
- If the user asks for a named section, call list_editor_sections and choose target_type="section" with target_id from the section candidate.
- "current SOP style" is a style constraint, not a full-document target.
- Use full_document only for explicit full/whole/entire/complete SOP/document requests.
- Use selection only if the user explicitly refers to selected/highlighted text or this selection.
- Prefer exact semantic meaning over keyword order.
- If two or more real live editor targets are equally plausible, return requires_clarification=true with candidate_targets.
- Never invent a section/table that is not present in the tool results.
- If schema/outline is not enough, call search_by_meaning, then inspect_candidate or get_block_content before deciding.

Return strict JSON only with:
{
  "target_type": "full_document|section|table|table_section|paragraph|selection",
  "target_id": "exact candidate id or null",
  "target_label": "exact live label or null",
  "owning_section": "exact parent section or null",
  "confidence": 0.0,
  "requires_clarification": false,
  "candidate_targets": [{"id": "...", "label": "...", "target_type": "...", "reason": "..."}],
  "reasoning_summary": "short explanation"
}"""

    subagents = [
        {
            "name": "table-target-agent",
            "description": "Resolves user requests that mention tables, matrices, histories, registers, or tabular sections.",
            "system_prompt": "Use list_editor_tables first. Return an exact table target_id. Never choose a section unless the user says table section.",
            "tools": [list_editor_tables, inspect_candidate, inspect_document_excerpt],
        },
        {
            "name": "section-target-agent",
            "description": "Resolves user requests that mention headings, sections, subsections, or named SOP parts.",
            "system_prompt": "Use list_editor_sections first. Return an exact section target_id. Prefer exact child heading over parent heading.",
            "tools": [list_editor_sections, inspect_candidate, inspect_selection, inspect_document_excerpt],
        },
        {
            "name": "local-span-agent",
            "description": "Resolves selected paragraphs, current rows/cells, and local paragraph/sentence requests.",
            "system_prompt": "Use inspect_selection first, then list_editor_paragraphs. Return a valid selection or paragraph target_id only.",
            "tools": [inspect_selection, list_editor_paragraphs, inspect_candidate],
        },
    ]

    model = create_chat_llm(temperature=0, max_output_tokens=900, max_retries=0, use_cache=True)
    agent = create_deep_agent(
        model=model,
        tools=[
            list_editor_sections,
            list_editor_tables,
            list_editor_paragraphs,
            search_by_meaning,
            get_block_content,
            inspect_selection,
            inspect_document_outline,
            inspect_candidate,
            inspect_document_excerpt,
        ],
        system_prompt=system_prompt,
        subagents=subagents,
        name="sop-target-resolver-agent",
    )
    prompt = {
        "user_query": user_query,
        "action": action,
        "must_call_tools": ["inspect_selection", "inspect_document_outline"],
        "available_counts": {"sections": len(compact_sections), "tables": len(compact_tables), "paragraphs": len(compact_paragraphs)},
        "document_schema": str(document_schema or "")[:12000],
    }
    result = agent.invoke({"messages": [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]})
    parsed = _extract_json_object(_deep_agent_text(result))
    return _normalize_result(parsed, model_name=get_local_llm_config().model, valid_targets=valid_targets)
