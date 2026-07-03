"""Verify chat can use open SOP context and route SOP actions correctly.

This avoids live LLM/API dependencies by exercising the backend's deterministic
context, metadata, and intent-routing helpers directly.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.ai_routes import (  # noqa: E402
    _deterministic_active_sop_metadata_response,
    _deterministic_live_section_response,
    _query_intents,
    _summarize_live_context,
    classify_intent,
)


ACTIVE_SOP_ID = "11111111-1111-1111-1111-111111111111"
SELECTED_SENTINEL = "SELECTED_FIREWALL_RULE_APPROVAL_SENTINEL"
METADATA_SENTINEL = "ISO-9001-QMS-SENTINEL"
FULL_TEXT_SENTINEL = "FULL_SOP_CONTEXT_SENTINEL"


CONTEXT: dict[str, Any] = {
    "active_sop_id": ACTIVE_SOP_ID,
    "current_document_id": ACTIVE_SOP_ID,
    "editor_surface_active": True,
    "current_sop": {
        "id": ACTIVE_SOP_ID,
        "sop_number": "SOP-IT-002",
        "title": "Network Security / Firewall SOP",
        "version": "2.3",
        "owner": "IT Security",
        "status": "Effective",
        "tags": ["firewall", "network", "GMP"],
        "word_count": 93,
        "compliance_standards": [METADATA_SENTINEL, "GxP"],
        "references": ["REF-FW-001", "REF-LOG-002"],
        "metadata": {
            "author": "IT Security",
            "department": "IT",
            "owner": "IT Security",
        },
        "sections": [
            {
                "id": "sec_1",
                "label": "1. Zweck",
                "content": "Diese SOP beschreibt den Schutz des Produktionsnetzwerks durch Firewall-Regeln.",
            },
            {
                "id": "sec_2",
                "label": "2. Scope",
                "content": "Applies to production firewall changes and network segmentation.",
            },
            {
                "id": "sec_3",
                "label": "3. Procedure",
                "content": (
                    "Firewall rule changes must be requested, approved, logged, implemented, "
                    "and reviewed with evidence."
                ),
            },
            {
                "id": "sec_4",
                "label": "4. CAPAs",
                "content": "CAPA records must link firewall deviations to remediation evidence.",
            },
        ],
        "full_text": (
            "1. Zweck\n"
            "Diese SOP beschreibt den Schutz des Produktionsnetzwerks durch Firewall-Regeln.\n\n"
            "2. Scope\n"
            "Applies to production firewall changes and network segmentation.\n\n"
            "3. Procedure\n"
            "Firewall rule changes must be requested, approved, logged, implemented, and reviewed with evidence.\n"
            f"{FULL_TEXT_SENTINEL}\n\n"
            "4. CAPAs\n"
            "CAPA records must link firewall deviations to remediation evidence."
        ),
    },
    "selected_text": (
        f"{SELECTED_SENTINEL}: selected Procedure text says firewall rule changes "
        "require approval, logging, implementation evidence, and review."
    ),
    "selected_range": {"from": 120, "to": 260, "empty": False},
    "selected_section": {
        "id": "sec_3",
        "name": "3. Procedure",
        "label": "3. Procedure",
        "type": "section",
        "scope": "selection",
        "content": (
            f"{SELECTED_SENTINEL}: firewall rule changes require approval, logging, "
            "implementation evidence, and review."
        ),
        "text_excerpt": (
            f"{SELECTED_SENTINEL}: firewall rule changes require approval, logging, "
            "implementation evidence, and review."
        ),
    },
    "linked_context": {
        "deviations": [{"id": "DEV-IT-011", "deviation_number": "DEV-IT-011", "title": "Unapproved firewall rule"}],
        "capas": [{"id": "CAPA-IT-011", "capa_number": "CAPA-IT-011", "title": "Firewall approval control"}],
        "audits": [],
        "decisions": [],
        "related_sops": [],
    },
    "editor_excerpt": (
        "1. Zweck\nDiese SOP beschreibt den Schutz des Produktionsnetzwerks durch Firewall-Regeln.\n"
        "3. Procedure\nFirewall rule changes must be requested, approved, logged, implemented, and reviewed."
    ),
    "context_updated_at": "2026-06-04T13:00:00Z",
}


def check(name: str, condition: bool, detail: Any = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed: {detail}")
    print(f"PASS: {name}")


async def main() -> int:
    os.environ["ASSISTANT_LLM_ORCHESTRATOR"] = "false"

    summary_ctx = _summarize_live_context(CONTEXT, "summarize this SOP")
    check("Chat accesses SOP context", "SOP-IT-002" in summary_ctx and "ACTIVE SOP ONLY" in summary_ctx, summary_ctx)
    selected_ctx = _summarize_live_context(CONTEXT, "is this selected section compliant?")
    check("Chat accesses selected section", SELECTED_SENTINEL in selected_ctx and "Selected" in selected_ctx, selected_ctx)

    metadata_response = _deterministic_active_sop_metadata_response(
        "what version, owner, status and compliance standards are in this SOP metadata?",
        CONTEXT,
    )
    check("Chat accesses metadata", bool(metadata_response) and METADATA_SENTINEL in metadata_response["answer"], metadata_response)

    summary_response = _deterministic_live_section_response("summarize the Procedure section", CONTEXT)
    check(
        "SOP summary query works",
        bool(summary_response)
        and summary_response["retrieval_stats"]["source"] == "live_editor_section"
        and SELECTED_SENTINEL in summary_response["citations"][0]["excerpt"],
        summary_response,
    )

    rewrite = await classify_intent(
        {
            "message": "rewrite the Procedure section",
            "has_active_sop": True,
            "has_editor_selection": True,
            "assistant_context": CONTEXT,
        }
    )
    check(
        "Rewrite query works",
        rewrite.get("flow") in {"editor_action", "follow_up_action"}
        and rewrite.get("action") == "rewrite"
        and rewrite.get("target_scope") == "section",
        rewrite,
    )

    gap = await classify_intent(
        {
            "message": "run a gap check on this SOP",
            "has_active_sop": True,
            "has_editor_selection": True,
            "assistant_context": CONTEXT,
        }
    )
    check(
        "Gap check query works",
        gap.get("flow") == "editor_action"
        and gap.get("action") == "gap_check"
        and gap.get("target_scope") == "full_document",
        gap,
    )

    compliance_intents = _query_intents("is this SOP audit-ready and compliant with GMP controls?")
    compliance_ctx = _summarize_live_context(CONTEXT, "is this SOP audit-ready and compliant with GMP controls?")
    check(
        "Compliance query works",
        "compliance" in compliance_intents
        and "ACTIVE SOP ONLY" in compliance_ctx
        and ("Selected text excerpt" in compliance_ctx or "Requested section" in compliance_ctx),
        {"intents": sorted(compliance_intents), "context": compliance_ctx},
    )

    check(
        "Chat responses are context-aware",
        "Procedure" in summary_response["answer"]
        and "firewall" in summary_response["answer"].lower()
        and metadata_response["retrieval_stats"]["strict_mode"] == "active_sop_metadata",
        {
            "summary": summary_response["answer"],
            "metadata": metadata_response["answer"],
        },
    )

    print("DETAIL: rewrite classification", json.dumps(rewrite, ensure_ascii=False, sort_keys=True))
    print("DETAIL: gap classification", json.dumps(gap, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
