#!/usr/bin/env python3
"""Validate generated adaptive SOP profile.md and client_profile outputs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERIFY_OUT = ROOT / "verify_out"

REQUIRED_MD_SECTIONS = [
    "## Detected Document Context",
    "## Adaptive Dynamic Extraction",
    "## Writing Style & Tone Detection",
    "## Modal Verbs & Language Controls",
    "## Roles & RACI Mapping",
    "## Workflow Patterns",
    "## Compliance Elements & Regulatory Signals",
    "## Risks & Gap Analysis",
    "## Terminology & Structure Extraction",
    "## Rewrite Rules",
]

REQUIRED_PROFILE_KEYS = [
    "client_name",
    "detected_domains",
    "detected_sop_types",
    "preferred_style",
    "terminology",
    "workflow_patterns",
    "roles_raci",
    "adaptive_extraction",
    "adaptive_document_context",
    "rewrite_rules",
]


def main() -> None:
    rows = []
    for analysis_path in sorted(VERIFY_OUT.glob("SOP-*_adaptive_analysis.json")):
        stem = analysis_path.name.replace("_adaptive_analysis.json", "")
        profile_md_path = VERIFY_OUT / f"{stem}_adaptive_profile.md"
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        client_profile = analysis.get("client_profile", {})
        profile_md = profile_md_path.read_text(encoding="utf-8", errors="ignore") if profile_md_path.exists() else ""

        missing_keys = [key for key in REQUIRED_PROFILE_KEYS if key not in client_profile]
        missing_sections = [section for section in REQUIRED_MD_SECTIONS if section not in profile_md]
        doc = analysis.get("document_information", {})
        adaptive = analysis.get("adaptive_extraction", {})
        rows.append({
            "file": stem,
            "profile_md_exists": profile_md_path.exists(),
            "profile_md_chars": len(profile_md),
            "missing_profile_keys": missing_keys,
            "missing_md_sections": missing_sections,
            "domain": doc.get("domain", {}).get("value"),
            "domain_source": doc.get("domain", {}).get("source"),
            "sop_type": doc.get("sop_type", {}).get("value"),
            "sop_type_source": doc.get("sop_type", {}).get("source"),
            "adaptive_context_terms": client_profile.get("adaptive_document_context", {}).get("terms", [])[:8],
            "dynamic_roles": len(adaptive.get("dynamic_roles", [])),
            "dynamic_workflows": len(adaptive.get("dynamic_workflows", [])),
            "rewrite_rules": len(client_profile.get("rewrite_rules", [])),
            "status": "PASS" if not missing_keys and not missing_sections and len(profile_md) > 1500 else "CHECK",
        })

    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
