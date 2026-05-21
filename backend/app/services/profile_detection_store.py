"""Persist and load SOP-linked NLP profile rows (`profile_detections`)."""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.models import ProfileDetection, SOP, SOPVersion
from app.services.nlp.layer import run_nlp_layer
from types import SimpleNamespace

from app.services.nlp.llm_action_context import (
    build_action_llm_context_block,
    build_nlp_structure_parameters_inner_body,
)
from app.services.sop_version_metadata_compact import compact_sop_version_metadata_for_storage

logger = logging.getLogger(__name__)

PROFILE_TYPE_DEFAULT = "nlp_action_profile"


def _extract_plain_text_from_tiptap(doc_json: dict | None) -> str:
    """Mirror of app.routes._extract_plain_text_from_tiptap (avoid import cycle)."""
    if not isinstance(doc_json, dict):
        return ""
    out: list[str] = []

    def walk(node: dict):
        if not isinstance(node, dict):
            return
        if node.get("type") == "text" and node.get("text"):
            out.append(str(node.get("text")))
        for child in node.get("content", []) or []:
            walk(child)

    walk(doc_json)
    return " ".join(out).strip()


def compute_source_hash(plain_text: str) -> str:
    return hashlib.sha256((plain_text or "").encode("utf-8", errors="ignore")).hexdigest()


def _avg_sentence_words(plain: str) -> float | None:
    cleaned = re.sub(r"\s+", " ", (plain or "").strip())
    if not cleaned:
        return None
    sentences = [s for s in re.split(r"[.!?]+", cleaned) if len(s.strip()) > 5]
    words = re.findall(r"\b\w+\b", cleaned)
    if not sentences:
        return round(len(words), 1)
    return round(len(words) / max(len(sentences), 1), 1)


def _style_dict_for_prompt_block(nlp: dict[str, Any], plain: str) -> dict[str, Any]:
    if not isinstance(nlp, dict) or nlp.get("skipped"):
        return {}
    sp = nlp.get("style_profile") or {}
    lang = sp.get("language") or {}
    lc = None
    if isinstance(lang, dict):
        lc = lang.get("lang_code") or lang.get("primary_language")
    aw = _avg_sentence_words(plain) or 0.0
    return {
        "tone": sp.get("primary_tone") or "neutral",
        "language": (lc or "unknown"),
        "avg_sentence_words": aw,
        "formality": sp.get("formality_level"),
        "imperative_ratio": 0.0,
        "modal_ratio": 0.0,
        "bullet_density": 0.0,
        "passive_markers": 0,
        "style_rules": [f"Match SOP style {sp.get('primary_style', '')}"],
    }


def _merge_nlp_with_version_metadata(nlp: dict[str, Any], compact_vm: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(nlp, dict):
        return {"sop_version_metadata": compact_vm}
    out = dict(nlp)
    out["sop_version_metadata"] = compact_vm
    return out


def _split_nlp_for_columns(
    nlp: dict[str, Any],
    plain: str,
    *,
    sop: SOP | None,
    version: SOPVersion | None,
    compact_vm: dict[str, Any],
) -> dict[str, Any]:
    """Map NLP dict + plain text + compact version metadata into ProfileDetection column payloads."""
    meta_raw = version.metadata_json if version and isinstance(version.metadata_json, dict) else {}

    if not isinstance(nlp, dict) or nlp.get("skipped"):
        nlp_merged = _merge_nlp_with_version_metadata(nlp if isinstance(nlp, dict) else {"skipped": True}, compact_vm)
        return {
            "language": None,
            "tone": None,
            "formality": None,
            "avg_sentence_words": _avg_sentence_words(plain),
            "readability_score": None,
            "structure_json": {},
            "parameters_json": {"sop_version_metadata": compact_vm},
            "detected_entities_json": {},
            "nlp_analysis_json": nlp_merged,
            "prompt_block": _build_prompt_block_for_row(
                nlp if isinstance(nlp, dict) else {"skipped": True},
                plain,
                compact_vm,
                meta_raw,
                sop,
                version,
            ),
        }

    sp = nlp.get("style_profile") or {}
    lang_info = sp.get("language") or {}
    lang_code = None
    if isinstance(lang_info, dict):
        lang_code = lang_info.get("lang_code") or lang_info.get("primary_language")

    structure_json = {
        "sections": nlp.get("sections") or [],
        "domain": nlp.get("domain"),
        "gaps": nlp.get("gaps") or [],
        "workflow": (nlp.get("workflow") or [])[:20],
    }
    parameters_json = {
        "rewrite_improve_parameters": nlp.get("rewrite_improve_parameters") or {},
        "sop_version_metadata": compact_vm,
    }
    detected_entities_json = {
        "roles": nlp.get("roles") or [],
        "compliance": nlp.get("compliance") or [],
        "risks": nlp.get("risks") or [],
    }
    nlp_merged = _merge_nlp_with_version_metadata(nlp, compact_vm)
    prompt_block = _build_prompt_block_for_row(nlp, plain, compact_vm, meta_raw, sop, version)

    return {
        "language": lang_code,
        "tone": sp.get("primary_tone"),
        "formality": sp.get("formality_level"),
        "avg_sentence_words": _avg_sentence_words(plain),
        "readability_score": None,
        "structure_json": structure_json,
        "parameters_json": parameters_json,
        "detected_entities_json": detected_entities_json,
        "nlp_analysis_json": nlp_merged,
        "prompt_block": prompt_block,
    }


def _build_prompt_block_for_row(
    nlp: dict[str, Any],
    plain: str,
    compact_vm: dict[str, Any],
    meta_raw: dict[str, Any],
    sop: SOP | None,
    version: SOPVersion | None,
) -> str | None:
    """Full stored prompt_block including metadata + NLP bundle (upload-only)."""
    style_for_block = _style_dict_for_prompt_block(nlp, plain)
    bundle = {"upload_nlp": nlp, "action_window_nlp": None}
    sop_ctx = {
        "sop_number": getattr(sop, "sop_number", "") if sop else "",
        "title": getattr(sop, "title", "") if sop else "",
        "version_id": str(version.id) if version else "",
        "version_metadata_compact": compact_vm,
        "version_metadata_keys": sorted(meta_raw.keys())[:40],
    }
    req = SimpleNamespace(sop_title=getattr(sop, "title", "") if sop else "")

    inner = build_nlp_structure_parameters_inner_body(
        action=PROFILE_TYPE_DEFAULT,
        sop_ctx=sop_ctx,
        style_profile=style_for_block,
        nlp_bundle=bundle,
        request=req,
        saved_prompt_block=None,
        max_total=9000,
        log_context=False,
    )
    base = build_action_llm_context_block(bundle, style_for_block, PROFILE_TYPE_DEFAULT, max_chars=6000)
    combined = (inner + "\n\n" + base).strip() if base else inner
    return combined or None


def persist_profile_detection_for_sop_version(db: Session, version: SOPVersion) -> None:
    """
    Run NLP on version content, then upsert `profile_detections`:
    at most one active row per (sop, sop_version); deactivate previous actives when text (hash) changes.
    """
    try:
        plain = _extract_plain_text_from_tiptap(version.content_json)
        if len(plain.strip()) < 40:
            logger.info("[profile-detection] skipped short text version_id=%s len=%s", version.id, len(plain))
            return

        source_hash = compute_source_hash(plain)
        sop_id = version.sop_id
        ver_id = version.id

        sop = db.query(SOP).filter(SOP.id == sop_id).first()
        meta_raw = version.metadata_json if isinstance(version.metadata_json, dict) else {}
        compact_vm = compact_sop_version_metadata_for_storage(meta_raw, sop, version)

        existing_active = (
            db.query(ProfileDetection)
            .filter(
                ProfileDetection.sop_id == sop_id,
                ProfileDetection.sop_version_id == ver_id,
                ProfileDetection.is_active == True,  # noqa: E712
                ProfileDetection.profile_type == PROFILE_TYPE_DEFAULT,
                ProfileDetection.source_hash == source_hash,
            )
            .first()
        )
        if existing_active:
            logger.info(
                "[profile-detection] unchanged hash sop_id=%s version_id=%s hash=%s…",
                sop_id,
                ver_id,
                source_hash[:12],
            )
            return

        db.query(ProfileDetection).filter(
            ProfileDetection.sop_id == sop_id,
            ProfileDetection.sop_version_id == ver_id,
            ProfileDetection.is_active == True,  # noqa: E712
            ProfileDetection.profile_type == PROFILE_TYPE_DEFAULT,
        ).update({"is_active": False})

        nlp = run_nlp_layer(plain, source="sop_upload")
        cols = _split_nlp_for_columns(nlp, plain, sop=sop, version=version, compact_vm=compact_vm)

        row = ProfileDetection(
            id=uuid.uuid4(),
            sop_id=sop_id,
            sop_version_id=ver_id,
            sop_version=(version.version_number or None),
            profile_type=PROFILE_TYPE_DEFAULT,
            source_hash=source_hash,
            language=cols["language"],
            tone=cols["tone"],
            formality=cols["formality"],
            avg_sentence_words=cols["avg_sentence_words"],
            readability_score=cols["readability_score"],
            structure_json=cols["structure_json"],
            parameters_json=cols["parameters_json"],
            detected_entities_json=cols["detected_entities_json"],
            nlp_analysis_json=cols["nlp_analysis_json"],
            prompt_block=cols["prompt_block"],
            is_active=True,
        )
        db.add(row)
        db.commit()
        logger.info(
            "[profile-detection] persisted sop_id=%s version_id=%s version_label=%s hash=%s… active=1",
            sop_id,
            ver_id,
            row.sop_version,
            source_hash[:12],
        )
    except Exception as exc:
        db.rollback()
        logger.warning("[profile-detection] failed version_id=%s err=%s", getattr(version, "id", None), exc)


def load_active_profile_detection_row(
    db: Session,
    *,
    sop_id: uuid.UUID,
    sop_version_id: uuid.UUID | None,
) -> ProfileDetection | None:
    if sop_version_id is None:
        return None
    return (
        db.query(ProfileDetection)
        .filter(
            ProfileDetection.sop_id == sop_id,
            ProfileDetection.sop_version_id == sop_version_id,
            ProfileDetection.is_active == True,  # noqa: E712
            ProfileDetection.profile_type == PROFILE_TYPE_DEFAULT,
        )
        .order_by(ProfileDetection.created_at.desc())
        .first()
    )


def serialize_profile_detection_row(row: ProfileDetection | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "nlp_analysis_json": row.nlp_analysis_json if isinstance(row.nlp_analysis_json, dict) else {},
        "prompt_block": row.prompt_block,
        "parameters_json": row.parameters_json if isinstance(row.parameters_json, dict) else {},
    }


def load_active_profile_detection_json(
    db: Session,
    *,
    sop_id: uuid.UUID,
    sop_version_id: uuid.UUID | None,
) -> dict[str, Any] | None:
    """Return the active row's `nlp_analysis_json` for the given SOP + version, or None."""
    row = load_active_profile_detection_row(db, sop_id=sop_id, sop_version_id=sop_version_id)
    if not row:
        return None
    data = row.nlp_analysis_json
    return data if isinstance(data, dict) else None
