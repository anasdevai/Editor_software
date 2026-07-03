#!/usr/bin/env python3
"""Check whether generated SOP analyses contain compliance evidence."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERIFY_OUT = ROOT / "verify_out"


def has_compliance_evidence(compliance: dict) -> bool:
    evidence_fields = [
        "standards_detected",
        "traceability_ids",
        "control_terms",
        "recordkeeping_terms",
        "approval_terms",
        "review_terms",
        "implicit_compliance_signals",
    ]
    return any(bool(compliance.get(field)) for field in evidence_fields)


def main() -> None:
    rows = []
    for path in sorted(VERIFY_OUT.glob("SOP-*_adaptive_analysis.json")):
        analysis = json.loads(path.read_text(encoding="utf-8"))
        compliance = analysis.get("compliance_elements", {})
        rows.append({
            "file": path.name.replace("_adaptive_analysis.json", ""),
            "status": "PASS" if has_compliance_evidence(compliance) else "FAIL",
            "standards": len(compliance.get("standards_detected", [])),
            "trace_ids": len(compliance.get("traceability_ids", [])),
            "controls": len(compliance.get("control_terms", [])),
            "records": len(compliance.get("recordkeeping_terms", [])),
            "approvals": len(compliance.get("approval_terms", [])),
            "reviews": len(compliance.get("review_terms", [])),
            "implicit": compliance.get("implicit_compliance_signals", []),
        })

    print(json.dumps({
        "compliance_elements_detected": "PASS" if all(row["status"] == "PASS" for row in rows) else "PARTIAL",
        "passed": sum(1 for row in rows if row["status"] == "PASS"),
        "total": len(rows),
        "rows": rows,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
