"""
Verify cross-profile rewrite: editorial profile style on open SOP content without swapping documents.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from dotenv import load_dotenv

load_dotenv(ROOT / "backend" / ".env")
import os

os.environ.setdefault(
    "DATABASE_URL_LOCAL", "postgresql://postgres:Admin123@127.0.0.1:5432/editor_db"
)

from app.ai_routes import _resolve_explicit_style_override
from app.database import SessionLocal
from app.models import ClientProfile, SOP, SOPDetectedParameters, SOPVersion
from chatbot.assistant.profile_reference import (
    build_editorial_profile_hints,
    extract_editorial_profile_reference,
)

API_BASE = os.getenv("PHASE_API_BASE", "http://127.0.0.1:8001").rstrip("/")
OPEN_SOP = "SOP-IT-002"
EDITORIAL_QUERY = "Emergency access"
USER_MSG = f"rewrite the Purpose section using {EDITORIAL_QUERY} sop profile"

errors: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        errors.append(label)


def main() -> None:
    print("=" * 72)
    print("  VERIFY: Editorial profile rewrite on open SOP (content preserved)")
    print("=" * 72)

    ref = extract_editorial_profile_reference(USER_MSG)
    check("Extract editorial profile reference", ref == EDITORIAL_QUERY, ref or "none")

    hints = build_editorial_profile_hints(
        USER_MSG, open_sop_number=OPEN_SOP, open_sop_title="Network Security"
    )
    check("Hints include CONTENT SOURCE lock", any("CONTENT SOURCE" in h for h in hints))
    check(
        "Hints forbid copying other SOP body",
        any("another sop document" in h.lower() for h in hints),
    )
    enriched = USER_MSG + "\n\n[Assistant constraints]\n" + "\n".join(hints)
    override = _resolve_explicit_style_override(enriched)
    check("Style override resolves in DB", override is not None)

    sop_open_id = None
    db = SessionLocal()
    try:
        sop_open = db.query(SOP).filter(SOP.sop_number == OPEN_SOP).first()
        if sop_open:
            sop_open_id = str(sop_open.id)
        sop_emergency = db.query(SOP).filter(SOP.sop_number == "SOP-IT-003").first()
        check("Open SOP exists", sop_open is not None, OPEN_SOP)
        check("Emergency SOP exists", sop_emergency is not None, "SOP-IT-003")

        row_open = (
            db.query(SOPDetectedParameters)
            .filter(SOPDetectedParameters.sop_id == sop_open.id)
            .order_by(SOPDetectedParameters.created_at.desc())
            .first()
            if sop_open
            else None
        )
        row_em = (
            db.query(SOPDetectedParameters)
            .filter(SOPDetectedParameters.sop_id == sop_emergency.id)
            .order_by(SOPDetectedParameters.created_at.desc())
            .first()
            if sop_emergency
            else None
        )
        prof_open = (
            db.query(ClientProfile).filter(ClientProfile.id == row_open.client_profile_id).first()
            if row_open
            else None
        )
        prof_em = (
            db.query(ClientProfile).filter(ClientProfile.id == row_em.client_profile_id).first()
            if row_em
            else None
        )
        print(f"\n  Open SOP profile:      {prof_open.name if prof_open else '?'}")
        print(f"  Emergency SOP profile: {prof_em.name if prof_em else '?'}")
        if override:
            print(f"  Override resolved to:  {override.get('resolved_name')}")

        if override and prof_open and prof_em:
            oid = str(override.get("profile_id") or "")
            check(
                "Override profile != open SOP default profile",
                oid != str(prof_open.id),
                f"open={prof_open.name}",
            )
            check(
                "Override profile matches Emergency Access family",
                oid == str(prof_em.id)
                or EDITORIAL_QUERY.lower() in str(override.get("resolved_name") or "").lower()
                or "emergency" in str(override.get("resolved_name") or "").lower()
                or "break" in str(override.get("resolved_name") or "").lower(),
                override.get("resolved_name"),
            )

        version = (
            db.query(SOPVersion).filter(SOPVersion.id == sop_open.current_version_id).first()
            if sop_open
            else None
        )
        sample_text = (
            "1. Purpose\n"
            "This procedure defines firewall rules for OT/IT separation on SOP-IT-002. "
            "Owner: Network Operations. Review cycle: annual."
        )
        if version and version.content_json:
            from app.utils.tiptap_text import extract_plain_text_from_tiptap

            full = extract_plain_text_from_tiptap(version.content_json) or ""
            m = re.search(r"(?is)(?:1\.\s*)?(?:purpose|zweck)[^\n]*\n(.*?)(?=\n\s*(?:2\.|##\s*\d))", full)
            if m and len(m.group(1).strip()) > 40:
                sample_text = ("1. Purpose\n" + m.group(1).strip())[:800]
    finally:
        db.close()

    print("\n--- Classify-intent API ---")
    classify_payload = {
        "message": USER_MSG,
        "has_active_sop": True,
        "has_editor_selection": False,
        "assistant_context": {
            "active_sop_id": sop_open_id or "x",
            "current_sop": {
                "sop_number": OPEN_SOP,
                "title": "Network Security",
                "sections": [{"label": "1. Purpose"}],
            },
            "last_action": {},
        },
    }
    req = urllib.request.Request(
        f"{API_BASE}/api/ai/classify-intent",
        data=json.dumps(classify_payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        classified = json.loads(resp.read().decode("utf-8", errors="replace"))
    check("Classify: editor_action", classified.get("flow") == "editor_action", classified.get("flow"))
    check("Classify: rewrite action", classified.get("action") == "rewrite", classified.get("action"))
    check(
        "Classify: editorial_profile_reference",
        classified.get("editorial_profile_reference") == EDITORIAL_QUERY,
        classified.get("editorial_profile_reference"),
    )
    check("Classify: content_source open_sop", classified.get("content_source") == "open_sop")
    ei = str(classified.get("enriched_instruction") or "")
    check("enriched_instruction has override phrase", 'Apply using "Emergency access" profile style' in ei)
    check("enriched_instruction locks open SOP facts", "SOP-IT-002" in ei or "Network Security" in ei)

    print("\n--- Live /api/ai/action (optional) ---")
    try:
        payload = {
            "action": "rewrite",
            "text": sample_text,
            "instruction": classified.get("enriched_instruction") or enriched,
            "sop_title": OPEN_SOP,
            "section_name": "1. Purpose",
            "section_type": "Purpose",
            "edit_scope": "section_only",
            "sop_entity_id": sop_open_id,
            "triggered_by": "verify_editorial_profile",
        }
        req = urllib.request.Request(
            f"{API_BASE}/api/ai/action",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            action_resp = json.loads(resp.read().decode("utf-8", errors="replace"))
        out = (
            (action_resp.get("structured_data") or {}).get("rewritten_text")
            or action_resp.get("suggested_text")
            or ""
        )
        plain = re.sub(r"<[^>]+>", " ", out)
        check("Action returned output", len(plain.strip()) > 20, f"chars={len(plain)}")
        check(
            "Output still references open SOP context",
            "SOP-IT-002" in plain or "OT/IT" in plain or "firewall" in plain.lower(),
            plain[:120].replace("\n", " "),
        )
        meta = (action_resp.get("structured_data") or {}).get("nlp_action_summary") or {}
        print(f"  [INFO] nlp_action_summary keys: {list(meta.keys())[:8]}")
        if meta.get("explicit_style_reference") or meta.get("style_override_name"):
            print(f"  [INFO] style_override={meta.get('style_override_name') or meta.get('explicit_style_reference')}")
    except Exception as exc:
        print(f"  [WARN] Live action skipped or failed: {exc}")

    print("\n" + "=" * 72)
    if errors:
        print(f"  RESULT: {len(errors)} failure(s): {', '.join(errors)}")
        sys.exit(1)
    print("  RESULT: All verification checks passed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
