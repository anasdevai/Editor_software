"""Orchestrates lightweight NLP before LLM actions (no external LLM calls)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _derive_rewrite_improve_params(nlp: dict[str, Any]) -> dict[str, Any]:
    sp = nlp.get("style_profile") or {}
    return {
        "preserve_numbering": sp.get("numbering_type", "simple"),
        "match_tone": sp.get("primary_tone", ""),
        "match_style": sp.get("primary_style", ""),
        "preserve_roles": (nlp.get("roles") or [])[:12],
        "address_structural_gaps": (nlp.get("gaps") or [])[:8],
        "keep_compliance_refs": (nlp.get("compliance") or [])[:15],
        "workflow_awareness": bool(nlp.get("workflow")),
    }


def run_nlp_layer(text: str, *, max_chars: int = 60_000, source: str = "unknown") -> dict[str, Any]:
    """
    Run deterministic NLP on plain text. Returns a dict suitable for JSON storage
    and for LLM prompt injection (via llm_action_context / llm_service_utils).
    """
    from app.services.nlp.pipeline import analyze_sop_text

    raw = (text or "")[:max_chars].strip()
    if len(raw) < 25:
        return {"skipped": True, "reason": "text_too_short", "source": source}

    try:
        out = analyze_sop_text(raw)
        out["rewrite_improve_parameters"] = _derive_rewrite_improve_params(out)
        out["nlp_source"] = source
        return out
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[nlp-layer] analyze_sop_text failed source=%s err=%s", source, exc)
        return {"skipped": True, "reason": str(type(exc).__name__), "source": source}
