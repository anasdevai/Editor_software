"""Build compact NLP + style blocks for editor LLM actions (improve / rewrite / gap_check)."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_MAX_STORED_PROMPT_TAIL = 4000


def _compact_list(val: Any, n: int = 12) -> str:
    if not isinstance(val, list):
        return str(val)[:240]
    items = [str(x)[:140] for x in val[:n]]
    tail = "; ..." if len(val) > n else ""
    return "; ".join(items) + tail


def _append_nlp_section(lines: list[str], label: str, d: dict[str, Any] | None) -> None:
    if not d or not isinstance(d, dict) or d.get("skipped"):
        return
    lines.append(f"[{label}]")
    lines.append(f"- sop_type_domain={d.get('domain', '')}")
    dt = str(d.get("title") or "")
    lines.append(f"- detected_title={dt[:220]}")
    lines.append(f"- sections={_compact_list(d.get('sections') or [])}")
    sp = d.get("style_profile") or {}
    lines.append(
        f"- doc_style={sp.get('primary_style', '')}|tone={sp.get('primary_tone', '')}"
        f"|formality={sp.get('formality_level', '')}|numbering={sp.get('numbering_type', '')}"
    )
    lines.append(f"- roles={_compact_list(d.get('roles') or [])}")
    lines.append(f"- workflow_steps={_compact_list(d.get('workflow') or [], 10)}")
    lines.append(f"- compliance_refs={_compact_list(d.get('compliance') or [], 14)}")
    lines.append(f"- risks_keywords={_compact_list(d.get('risks') or [], 18)}")
    lines.append(f"- structural_gaps={_compact_list(d.get('gaps') or [], 8)}")
    rip = d.get("rewrite_improve_parameters")
    if isinstance(rip, dict) and rip:
        try:
            blob = json.dumps(rip, ensure_ascii=False)[:900]
        except Exception:
            blob = str(rip)[:900]
        lines.append(f"- rewrite_improve_parameters={blob}")


def build_action_llm_context_block(
    nlp_bundle: dict[str, Any] | None,
    style_profile: dict[str, Any] | None,
    task: str,
    *,
    max_chars: int = 12_000,
) -> str:
    """
    Human-readable block for the LLM. `nlp_bundle` may contain:
    - upload_nlp: full-document analysis from SOP save/upload
    - action_window_nlp: analysis of SOP excerpt + selected text window
    """
    if not nlp_bundle:
        return ""

    lines: list[str] = [
        "NLP_DETECTED_PARAMETERS",
        f"- llm_task={task}",
    ]
    upload = nlp_bundle.get("upload_nlp")
    window = nlp_bundle.get("action_window_nlp")
    if isinstance(upload, dict):
        _append_nlp_section(lines, "FROM_UPLOADED_SOP_NLP", upload)
    if isinstance(window, dict):
        _append_nlp_section(lines, "FROM_ACTION_TEXT_WINDOW", window)

    if isinstance(style_profile, dict) and style_profile:
        lines.append("[EDITOR_STYLE_PROFILE]")
        lines.append(
            f"- tone={style_profile.get('tone')}|language={style_profile.get('language')}"
            f"|avg_sentence_words={style_profile.get('avg_sentence_words')}"
        )
        rules = style_profile.get("style_rules") or []
        if rules:
            lines.append(f"- style_rules={'; '.join(str(r) for r in rules[:5])}")

    out = "\n".join(lines).strip()
    if len(out) > max_chars:
        out = out[: max_chars - 3].rstrip() + "..."
        logger.info("[nlp-prompt-block] truncated to max_chars=%s", max_chars)
    return out


def _scalar_line(label: str, value: Any) -> str | None:
    if value is None or value == "":
        return None
    return f"- {label}: {_line_value(value)}"


def _line_value(value: Any, max_len: int = 220) -> str:
    s = str(value).replace("\r", " ").replace("\n", " ").strip()
    return s if len(s) <= max_len else s[: max_len - 1].rstrip() + "…"


def build_nlp_structure_parameters_inner_body(
    *,
    action: str,
    sop_ctx: dict[str, Any],
    style_profile: dict[str, Any] | None,
    nlp_bundle: dict[str, Any] | None,
    request: Any | None = None,
    saved_prompt_block: str | None = None,
    max_total: int = 10_000,
    log_context: bool = True,
) -> str:
    """
    Inner body for `NLP_STRUCTURE_AND_PARAMETERS` (prompts.py wraps with that heading).
    Order: SOP/version scalars → compact metadata key/values → NLP detections → optional stored profile prompt.
    """
    lines: list[str] = []
    vm = sop_ctx.get("version_metadata_compact") if isinstance(sop_ctx.get("version_metadata_compact"), dict) else {}
    scalars = vm.get("scalars") if isinstance(vm.get("scalars"), dict) else {}

    sop_num = scalars.get("sop_number") or sop_ctx.get("sop_number") or ""
    sop_title = (getattr(request, "sop_title", None) if request else None) or scalars.get("sop_title") or sop_ctx.get("title") or ""
    ver_lbl = scalars.get("version_label") or (sop_ctx.get("version_id") and f"id:{sop_ctx.get('version_id')}") or ""

    if request is not None and action in ("improve", "rewrite", "summarize", "analyze"):
        try:
            from chatbot.actions.prompts import resolve_edit_scope

            scope = resolve_edit_scope(request)
            ln = _scalar_line("Edit Scope", scope)
            if ln:
                lines.append(ln)
            if scope == "section_only":
                target = getattr(request, "section_title", None) or "Selected text"
                ln = _scalar_line("Target Section (rewrite/improve only this block)", target)
                if ln:
                    lines.append(ln)
        except Exception:
            pass

    for part in (
        _scalar_line("SOP Number", sop_num),
        _scalar_line("SOP Title", sop_title),
        _scalar_line("Version", ver_lbl),
        _scalar_line("Department", scalars.get("department")),
        _scalar_line("Status", scalars.get("lifecycle_status")),
        _scalar_line("Document Type", scalars.get("doc_type")),
        _scalar_line("Category", scalars.get("category")),
        _scalar_line("Risk Level", scalars.get("risk_level")),
    ):
        if part:
            lines.append(part)

    sp = style_profile if isinstance(style_profile, dict) else {}
    for part in (
        _scalar_line("Language", sp.get("language")),
        _scalar_line("Tone", sp.get("tone")),
        _scalar_line("Formality", sp.get("formality")),
        _scalar_line("Avg Sentence Words", sp.get("avg_sentence_words")),
    ):
        if part:
            lines.append(part)

    upload = None
    if isinstance(nlp_bundle, dict):
        upload = nlp_bundle.get("upload_nlp") if isinstance(nlp_bundle.get("upload_nlp"), dict) else None
    if isinstance(upload, dict) and not upload.get("skipped"):
        usp = upload.get("style_profile") or {}
        if usp.get("primary_tone") and not any("- Tone:" in x for x in lines):
            ln = _scalar_line("NLP Tone", usp.get("primary_tone"))
            if ln:
                lines.append(ln)
        if usp.get("formality_level") and not any("Formality:" in x for x in lines):
            ln = _scalar_line("NLP Formality", usp.get("formality_level"))
            if ln:
                lines.append(ln)
        lang = usp.get("language") or {}
        if isinstance(lang, dict):
            lc = lang.get("lang_code") or lang.get("primary_language")
            if lc and not any("- Language:" in x for x in lines):
                ln = _scalar_line("NLP Language", lc)
                if ln:
                    lines.append(ln)
        if upload.get("domain"):
            ln = _scalar_line("SOP Type / Domain", upload.get("domain"))
            if ln:
                lines.append(ln)

    kv_lines = vm.get("kv_lines") if isinstance(vm.get("kv_lines"), list) else []
    if kv_lines:
        lines.append("- SOP Version Metadata:")
        for item in kv_lines[:36]:
            if isinstance(item, dict):
                k = str(item.get("key", ""))[:80]
                v = _line_value(item.get("value", ""), 200)
                if k:
                    lines.append(f"  {k}: {v}")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                lines.append(f"  {_line_value(item[0], 80)}: {_line_value(item[1], 200)}")

    nlp_tail = ""
    if (
        action in ("improve", "rewrite", "summarize", "analyze")
        and (saved_prompt_block or "").strip()
    ):
        tail = (saved_prompt_block or "").strip()
        if len(tail) > _MAX_STORED_PROMPT_TAIL:
            tail = tail[: _MAX_STORED_PROMPT_TAIL - 3].rstrip() + "..."
        nlp_tail = f"\n--- STORED_PROFILE_NLP ---\n{tail}"

    if not nlp_tail.strip() and isinstance(nlp_bundle, dict):
        nlp_tail = "\n" + (build_action_llm_context_block(nlp_bundle, style_profile, action, max_chars=max_total) or "")

    body = "\n".join([ln for ln in lines if ln]).strip() + nlp_tail
    body = body.strip()
    if len(body) > max_total:
        body = body[: max_total - 3].rstrip() + "..."
    if log_context:
        try:
            logger.info(
                "[ai-action-nlp-context] action=%s sop_id=%s sop_version_id=%s metadata_keys=%s prompt_block_chars=%s",
                action,
                sop_ctx.get("sop_id"),
                sop_ctx.get("version_id"),
                (sop_ctx.get("version_metadata_keys") or [])[:40],
                len(body),
            )
        except Exception:
            pass
    return body
