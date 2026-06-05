#!/usr/bin/env python3
"""Analyze requested Desktop SOPs one by one with the adaptive NLP pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nlp_pipeline import analyze_sop_industry_level


FILES = [
    Path(r"C:\Users\AbdulRauf(AIEngineer\Desktop\SOP-QA-007.txt"),
    Path(r"C:\Users\AbdulRauf(AIEngineer\Desktop\SOP-IT-001.txt"),
    Path(r"C:\Users\AbdulRauf(AIEngineer\Desktop\SOP-IT-002.txt"),
    Path(r"C:\Users\AbdulRauf(AIEngineer\Desktop\SOP-IT-003.txt"),
    Path(r"C:\Users\AbdulRauf(AIEngineer\Desktop\SOP-IT-004.txt"),
]


def main() -> None:
    out_dir = ROOT / "verify_out"
    out_dir.mkdir(exist_ok=True)
    summaries = []

    for path in FILES:
        text = path.read_text(encoding="utf-8", errors="ignore")
        result = analyze_sop_industry_level(text, client_name=path.stem, include_profile_md=True)

        analysis_path = out_dir / f"{path.stem}_adaptive_analysis.json"
        profile_path = out_dir / f"{path.stem}_adaptive_profile.md"
        analysis_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        profile_path.write_text(result.get("profile_md", ""), encoding="utf-8")

        doc = result.get("document_information", {})
        adaptive = result.get("adaptive_extraction", {})
        fingerprint = adaptive.get("domain_fingerprint", {})
        summaries.append({
            "file": path.name,
            "chars": len(text),
            "language": result.get("language", {}).get("primary_language"),
            "domain": doc.get("domain", {}).get("value"),
            "domain_confidence": doc.get("domain", {}).get("confidence"),
            "domain_source": doc.get("domain", {}).get("source"),
            "sop_type": doc.get("sop_type", {}).get("value"),
            "sop_type_confidence": doc.get("sop_type", {}).get("confidence"),
            "sop_type_source": doc.get("sop_type", {}).get("source"),
            "department": doc.get("department", {}).get("value"),
            "sections": result.get("structure_patterns", {}).get("section_count"),
            "section_titles": [
                s.get("title")
                for s in result.get("structure_patterns", {}).get("sections", [])[:12]
            ],
            "dynamic_roles": [
                r.get("role")
                for r in adaptive.get("dynamic_roles", [])[:8]
            ],
            "roles_count": result.get("roles_raci", {}).get("detected_role_count"),
            "dynamic_workflows": [
                {
                    "name": w.get("name"),
                    "status": w.get("status"),
                    "confidence": w.get("confidence"),
                    "stages": w.get("stage_count"),
                }
                for w in adaptive.get("dynamic_workflows", [])[:6]
            ],
            "dominant_terms": [
                x.get("term")
                for x in fingerprint.get("dominant_terms", [])[:8]
            ],
            "profile": str(profile_path.relative_to(ROOT)),
        })

    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
