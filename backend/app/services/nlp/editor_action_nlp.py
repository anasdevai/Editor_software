"""NLP bundle construction for `/api/ai/action` (improve / rewrite / gap_check)."""

from __future__ import annotations

import logging
from typing import Any

from schemas.sop_actions import ActionRequest

from app.services.nlp.layer import run_nlp_layer
from app.services.nlp.llm_action_context import build_nlp_structure_parameters_inner_body

logger = logging.getLogger(__name__)


def _trunc(s: str, n: int) -> str:
    raw = str(s or "")
    if n <= 0 or len(raw) <= n:
        return raw
    return raw[: n - 3].rstrip() + "..."


def _profile_nlp_usable(profile: dict[str, Any] | None) -> bool:
    if not isinstance(profile, dict):
        return False
    data = profile.get("nlp_analysis_json")
    if not isinstance(data, dict) or data.get("skipped"):
        return False
    return True


def build_nlp_bundle_for_action(
    action: str,
    request: ActionRequest,
    sop_ctx: dict[str, Any],
    style_profile: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """
    Build NLP bundle + inner body for `NLP_STRUCTURE_AND_PARAMETERS`.
    Improve/Rewrite: reuse active ProfileDetection (no window NLP) when usable.
    Gap check: always run window NLP on excerpt+selection; upload NLP from ProfileDetection when present.
    """
    profile = sop_ctx.get("profile_detection") if isinstance(sop_ctx.get("profile_detection"), dict) else None
    from chatbot.actions.prompts import resolve_edit_scope

    scope = resolve_edit_scope(request)
    sop_text = str(sop_ctx.get("text") or "")
    selection = request.section_text or ""
    parts: list[str] = []
    if scope == "section_only":
        parts.append(f"---TARGET_SECTION:{request.section_title or 'selection'}---")
        parts.append(selection)
        if sop_text.strip():
            parts.append("---SOP_CONTEXT_EXCERPT (style reference only; do not rewrite)---")
            parts.append(_trunc(sop_text, 8_000))
    else:
        if sop_text.strip():
            parts.append(_trunc(sop_text, 50_000))
        parts.append("---FULL_DOCUMENT_TEXT---")
        parts.append(selection)
    combined = "\n\n".join(parts).strip()

    reuse_improve_rewrite = bool(
        profile and action in ("improve", "rewrite", "summarize", "analyze") and _profile_nlp_usable(profile)
    )

    if reuse_improve_rewrite and profile is not None:
        bundle: dict[str, Any] = {
            "upload_nlp": profile["nlp_analysis_json"],
            "action_window_nlp": None,
        }
        saved_pb = str(profile.get("prompt_block") or "").strip() or None
        logger.info(
            "[nlp-action] profile_detection_reuse action=%s sop_id=%s sop_version_id=%s",
            action,
            sop_ctx.get("sop_id"),
            sop_ctx.get("version_id"),
        )
        inner = build_nlp_structure_parameters_inner_body(
            action=action,
            sop_ctx=sop_ctx,
            style_profile=style_profile,
            nlp_bundle=bundle,
            request=request,
            saved_prompt_block=saved_pb,
            log_context=True,
        )
        return bundle, inner

    if action == "gap_check":
        upload = None
        if _profile_nlp_usable(profile):
            upload = profile["nlp_analysis_json"]  # type: ignore[index]
        if upload is None:
            upload = sop_ctx.get("nlp_analysis") if isinstance(sop_ctx.get("nlp_analysis"), dict) else None
        window = run_nlp_layer(combined, source=f"ai_action:{action}")
        bundle = {"upload_nlp": upload, "action_window_nlp": window}
        inner = build_nlp_structure_parameters_inner_body(
            action=action,
            sop_ctx=sop_ctx,
            style_profile=style_profile,
            nlp_bundle=bundle,
            request=request,
            saved_prompt_block=None,
            log_context=True,
        )
        return bundle, inner

    # improve / rewrite without stored profile: run window NLP; upload from legacy ctx or None
    stored = sop_ctx.get("nlp_analysis") if isinstance(sop_ctx.get("nlp_analysis"), dict) else None
    window = run_nlp_layer(combined, source=f"ai_action:{action}")
    bundle = {"upload_nlp": stored, "action_window_nlp": window}
    inner = build_nlp_structure_parameters_inner_body(
        action=action,
        sop_ctx=sop_ctx,
        style_profile=style_profile,
        nlp_bundle=bundle,
        request=request,
        saved_prompt_block=None,
        log_context=True,
    )
    return bundle, inner


def nlp_action_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    up = bundle.get("upload_nlp") if isinstance(bundle.get("upload_nlp"), dict) else None
    win = bundle.get("action_window_nlp") if isinstance(bundle.get("action_window_nlp"), dict) else None
    return {
        "has_upload_nlp": bool(up and not up.get("skipped")),
        "upload_domain": (up or {}).get("domain"),
        "window_skipped": bool((win or {}).get("skipped")) if win is not None else True,
        "window_domain": (win or {}).get("domain"),
        "window_sections_n": len((win or {}).get("sections") or []),
        "profile_row_reused": win is None and bool(up and not up.get("skipped")),
    }


def log_nlp_detected(action: str, bundle: dict[str, Any]) -> None:
    win = bundle.get("action_window_nlp") if isinstance(bundle.get("action_window_nlp"), dict) else {}
    if win is None or not win:
        up = bundle.get("upload_nlp") if isinstance(bundle.get("upload_nlp"), dict) else {}
        logger.info(
            "[nlp-detected] action=%s window=skipped profile_upload_domain=%s sections=%s",
            action,
            up.get("domain"),
            len(up.get("sections") or []),
        )
        return
    if win.get("skipped"):
        logger.info("[nlp-detected] action=%s window=skipped reason=%s", action, win.get("reason"))
        return
    logger.info(
        "[nlp-detected] action=%s domain=%s title_preview=%s sections=%s roles=%s compliance_n=%s risks_n=%s gaps=%s",
        action,
        win.get("domain"),
        (str(win.get("title") or ""))[:100],
        len(win.get("sections") or []),
        len(win.get("roles") or []),
        len(win.get("compliance") or []),
        len(win.get("risks") or []),
        win.get("gaps"),
    )
