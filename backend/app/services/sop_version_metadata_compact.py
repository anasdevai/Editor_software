"""Compact `sop_versions.metadata_json` for editor AI prompts and ProfileDetection storage."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_RAW_SNIPPET = 1800
MAX_KV_PAIRS = 40
MAX_VALUE_LEN = 200
MAX_NESTED_JSON = 400


def _trunc(val: Any, n: int = MAX_VALUE_LEN) -> str:
    s = str(val).replace("\r", " ").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def compact_sop_version_metadata_for_storage(
    metadata_json: dict | None,
    sop: Any,
    version: Any,
) -> dict[str, Any]:
    """
    Return a small JSON-serializable dict for `parameters_json.sop_version_metadata`
    and `nlp_analysis_json.sop_version_metadata`.
    """
    meta = metadata_json if isinstance(metadata_json, dict) else {}
    sm = meta.get("sopMetadata") if isinstance(meta.get("sopMetadata"), dict) else {}

    scalars: dict[str, Any] = {
        "sop_number": getattr(sop, "sop_number", None) or sm.get("documentId") or "",
        "sop_title": getattr(sop, "title", None) or sm.get("title") or "",
        "version_label": getattr(version, "version_number", None) or sm.get("sopVersion") or "",
        "department": sm.get("department") or getattr(sop, "department", None) or "",
        "lifecycle_status": getattr(version, "external_status", None)
        or meta.get("sopStatus")
        or meta.get("status")
        or "",
        "doc_type": sm.get("docType") or "",
        "category": sm.get("category") or "",
        "risk_level": sm.get("riskLevel") or "",
        "effective_date": sm.get("effectiveDate") or "",
        "review_date": sm.get("reviewDate") or "",
    }

    kv_lines: list[dict[str, str]] = []
    for key in (
        "sopStatus",
        "status",
        "versionNote",
        "obsoleteReason",
        "approvalSignature",
        "replacementDocumentId",
    ):
        if key in meta and meta.get(key) not in (None, "", []):
            kv_lines.append({"key": key, "value": _trunc(meta.get(key), 180)})

    for key in sorted(sm.keys()):
        if key in (
            "title",
            "documentId",
            "department",
            "sopVersion",
            "docType",
            "category",
            "riskLevel",
            "reviewDate",
            "effectiveDate",
        ):
            continue
        v = sm.get(key)
        if v in (None, "", []):
            continue
        kv_lines.append({"key": f"sopMetadata.{key}", "value": _trunc(v, MAX_VALUE_LEN)})
        if len(kv_lines) >= MAX_KV_PAIRS:
            break

    raw_trim = ""
    try:
        raw_trim = json.dumps(meta, ensure_ascii=False, default=str)[:MAX_RAW_SNIPPET]
    except Exception:
        raw_trim = _trunc(str(meta), MAX_RAW_SNIPPET)

    return {
        "scalars": scalars,
        "kv_lines": kv_lines[:MAX_KV_PAIRS],
        "metadata_top_level_keys": sorted(meta.keys())[:40],
        "metadata_raw_trim": raw_trim,
    }


def log_metadata_load(
    *,
    sop_id: Any,
    sop_version_id: Any,
    metadata_keys: list[str],
    prompt_block_chars: int | None,
) -> None:
    logger.info(
        "[profile-detection-metadata-load] sop_id=%s sop_version_id=%s metadata_keys=%s prompt_block_chars=%s",
        sop_id,
        sop_version_id,
        metadata_keys[:50],
        prompt_block_chars if prompt_block_chars is not None else "-",
    )


def log_metadata_merge(
    *,
    sop_id: Any,
    sop_version_id: Any,
    merged_keys: list[str],
    prompt_block_chars: int,
) -> None:
    logger.info(
        "[profile-detection-metadata-merge] sop_id=%s sop_version_id=%s merged_keys=%s prompt_block_chars=%s",
        sop_id,
        sop_version_id,
        merged_keys[:50],
        prompt_block_chars,
    )
