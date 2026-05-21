from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from .nlp.pipeline import analyze_sop_text
from .pdf_extractor import extract_traceable_text

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an AI Client Profile Detection Engine for SOP documents.
Return raw JSON only, with these keys:
summary, detected_domain, overall_confidence_score, profile_suggestions.
Each profile_suggestions item must include suggestion_type, suggested_rule,
evidence_from_document, confidence_score.
Use null or an empty list when a field is not present.
"""


def _extract_json(content: str) -> dict[str, Any]:
    content = (content or "").strip()
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if json_match:
        content = json_match.group(1).strip()
    start_idx = content.find("{")
    end_idx = content.rfind("}")
    if start_idx != -1 and end_idx != -1:
        content = content[start_idx : end_idx + 1]
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _find_best_match(snippet: str, chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
    snippet_clean = re.sub(r"\s+", " ", (snippet or "").lower().strip())
    if not snippet_clean:
        return None

    for chunk in chunks:
        chunk_clean = re.sub(r"\s+", " ", str(chunk.get("text", "")).lower().strip())
        if snippet_clean in chunk_clean or (chunk_clean and chunk_clean in snippet_clean):
            return chunk

    words = snippet_clean.split()
    if len(words) > 5:
        target = " ".join(words[:5])
        for chunk in chunks:
            if target in str(chunk.get("text", "")).lower():
                return chunk
    return None


def _fallback_summary(profile: dict[str, Any]) -> str:
    domain = profile.get("domain") or "Quality_Management"
    style = (profile.get("style_profile") or {}).get("primary_style", "FREE_PROSE")
    tone = (profile.get("style_profile") or {}).get("primary_tone", "technical_descriptive")
    sections = profile.get("sections") or []
    risks = profile.get("risks") or []
    compliance = profile.get("compliance") or []
    return (
        f"The document appears to belong to the {domain} domain. "
        f"Its primary writing style is {style} with a {tone} tone. "
        f"Detected sections include {', '.join(sections[:5]) if sections else 'no clear canonical section headers'}. "
        f"Compliance references include {', '.join(compliance[:5]) if compliance else 'none explicitly detected'}. "
        f"Risk markers include {', '.join(risks[:5]) if risks else 'none explicitly detected'}."
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _suggestions_from_profile(profile: dict[str, Any]) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    style = profile.get("style_profile") or {}
    if style.get("primary_style"):
        suggestions.append(
            {
                "suggestion_type": "writing_style",
                "suggested_rule": f"Preserve {style['primary_style']} structure in generated or revised SOP content.",
                "evidence_from_document": ", ".join(profile.get("sections") or [])[:300],
                "confidence_score": 0.78,
            }
        )
    if profile.get("roles"):
        suggestions.append(
            {
                "suggestion_type": "roles",
                "suggested_rule": "Prefer detected role terminology: " + ", ".join(profile["roles"][:8]),
                "evidence_from_document": ", ".join(profile["roles"][:8]),
                "confidence_score": 0.72,
            }
        )
    if profile.get("compliance"):
        suggestions.append(
            {
                "suggestion_type": "compliance",
                "suggested_rule": "Retain references to " + ", ".join(profile["compliance"][:8]),
                "evidence_from_document": ", ".join(profile["compliance"][:8]),
                "confidence_score": 0.74,
            }
        )
    return suggestions


def _llm_profile_suggestions(text: str, base_profile: dict[str, Any]) -> dict[str, Any]:
    if os.getenv("PROFILE_DETECTION_USE_LLM", "true").strip().lower() not in {"1", "true", "yes"}:
        return {}

    try:
        from chatbot.llm.provider import create_openai_client, get_local_llm_config

        cfg = get_local_llm_config()
        model = os.getenv("LOCAL_LLM_PROFILE_MODEL") or cfg.model
        client = create_openai_client()
        prompt = (
            SYSTEM_PROMPT
            + "\n\nDeterministic profile snapshot:\n"
            + json.dumps(
                {
                    "domain": base_profile.get("domain"),
                    "style_profile": base_profile.get("style_profile"),
                    "roles": base_profile.get("roles"),
                    "workflow": base_profile.get("workflow"),
                    "compliance": base_profile.get("compliance"),
                    "risks": base_profile.get("risks"),
                    "gaps": base_profile.get("gaps"),
                },
                ensure_ascii=True,
            )
            + "\n\nSOP text:\n"
            + text[:14000]
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        raw = ((response.choices[0].message.content if response.choices else "") or "").strip()
        return _extract_json(raw)
    except Exception as exc:
        logger.warning("[profile-analysis] optional LLM profile suggestions skipped: %s", exc)
        return {}


def analyze_sop_text_profile(text: str, *, use_llm: bool = True) -> dict[str, Any]:
    """Analyze raw SOP text and return a profile-detection payload."""
    profile = analyze_sop_text(text or "")
    llm_result = _llm_profile_suggestions(text or "", profile) if use_llm else {}
    suggestions = llm_result.get("profile_suggestions")
    if not isinstance(suggestions, list):
        suggestions = _suggestions_from_profile(profile)

    return {
        "summary": llm_result.get("summary") or _fallback_summary(profile),
        "detected_domain": llm_result.get("detected_domain") or profile.get("domain"),
        "overall_confidence_score": _safe_float(llm_result.get("overall_confidence_score"), 0.72),
        "profile_suggestions": suggestions,
        "nlp_profile": profile,
    }


def analyze_sop_traceable(file_obj) -> dict[str, Any]:
    """Extract traceable text from an uploaded SOP and analyze it for profile suggestions."""
    traceable_chunks = extract_traceable_text(file_obj)
    full_text = ""
    for chunk in traceable_chunks:
        full_text += f"\n\n{chunk.get('text', '')}"
        if len(full_text) > 15000:
            break

    result = analyze_sop_text_profile(full_text, use_llm=True)
    for suggestion in result.get("profile_suggestions") or []:
        evidence_snippet = suggestion.get("evidence_from_document", "")
        match = _find_best_match(evidence_snippet, traceable_chunks)
        if match:
            suggestion["evidence_metadata"] = {
                "page": match.get("page"),
                "section": match.get("section"),
                "paragraph_index": match.get("paragraph_index"),
                "traceability_id": match.get("traceability_id"),
            }
    return result
