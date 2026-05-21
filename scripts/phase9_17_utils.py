from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv

load_dotenv(BACKEND_DIR / ".env", override=True)
os.environ["DATABASE_URL_LOCAL"] = "postgresql://postgres:Admin123@127.0.0.1:5432/editor_db"

from app.database import SessionLocal
from app.models import (
    AIActionLog,
    AISuggestion,
    ChatMessage,
    ChatSession,
    ClientProfile,
    EmbeddingJob,
    KnowledgeChunk,
    ProfileHistoryEvent,
    ProfileVersion,
    SOP,
    SOPDetectedParameters,
    SOPVersion,
    SourceReference,
)
from app.utils.tiptap_text import extract_plain_text_from_tiptap


API_BASE = os.getenv("PHASE_API_BASE", os.getenv("PHASE8_API_BASE", "http://127.0.0.1:8001")).rstrip("/")
PROFILE_NAME = "German_Pharma_SOP_Profile"
TARGET_SOP = "SOP-IT-002"
SEP = "=" * 72


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def check(label: str, condition: bool, detail: str = "") -> bool:
    suffix = f" - {detail}" if detail else ""
    (ok if condition else fail)(f"{label}{suffix}")
    return condition


def api_json(method: str, path: str, payload: dict | None = None, timeout: int = 120) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw else {}


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def plain_html(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value or "")


def latest_sop_bundle(db=None):
    own = db is None
    db = db or SessionLocal()
    try:
        sop = db.query(SOP).filter(SOP.sop_number == TARGET_SOP).first()
        if not sop:
            raise RuntimeError(f"{TARGET_SOP} not found")
        version = db.query(SOPVersion).filter(SOPVersion.id == sop.current_version_id).first()
        if not version:
            version = db.query(SOPVersion).filter(SOPVersion.sop_id == sop.id).order_by(SOPVersion.created_at.desc()).first()
        if not version:
            raise RuntimeError(f"{TARGET_SOP} has no SOP versions")
        text = extract_plain_text_from_tiptap(version.content_json or {})
        return sop, version, text
    finally:
        if own:
            db.close()


def latest_profile(db=None):
    own = db is None
    db = db or SessionLocal()
    try:
        profile = db.query(ClientProfile).filter(ClientProfile.name == PROFILE_NAME).first()
        version = None
        if profile and profile.current_version_id:
            version = db.query(ProfileVersion).filter(ProfileVersion.id == profile.current_version_id).first()
        return profile, version
    finally:
        if own:
            db.close()


def latest_action_log(action: str = "rewrite", db=None):
    own = db is None
    db = db or SessionLocal()
    try:
        return (
            db.query(AIActionLog)
            .filter(AIActionLog.action == action, AIActionLog.sop_title == TARGET_SOP)
            .order_by(AIActionLog.created_at.desc())
            .first()
        )
    finally:
        if own:
            db.close()


def classify(message: str, selected_text: str = "", route: str = "/editor") -> dict:
    db = SessionLocal()
    try:
        sop, version, _ = latest_sop_bundle(db)
        payload = {
            "message": message,
            "route": route,
            "has_active_sop": True,
            "has_editor_selection": bool(selected_text),
            "assistant_context": {
                "active_sop_id": str(sop.id),
                "current_document_id": str(sop.id),
                "current_version_id": str(version.id),
                "selected_text": selected_text,
                "current_sop": {
                    "id": str(sop.id),
                    "sop_number": sop.sop_number,
                    "title": sop.title,
                    "version": version.version_number,
                },
            },
        }
    finally:
        db.close()
    return api_json("POST", "/api/ai/classify-intent", payload, timeout=90)


def german_quality_signals(text: str, structured: dict | None = None) -> dict[str, bool]:
    structured = structured or {}
    plain = plain_html(text)
    return {
        "modal_language_present": bool(re.search(r"\b(muss|müssen|sollte|sollten|darf nicht|dürfen nicht)\b", plain, re.I)),
        "control_language_present": bool(re.search(r"\b(Prüfung|Kontrolle|Freigabe|Nachweis|Dokumentation|Verifizierung|Überprüfung)\b", plain, re.I)),
        "formal_german_register": bool(re.search(r"\b(dieses verfahren|sicherstellung|gemäß|ziel ist|dokumentation)\b", plain, re.I)),
        "source_ids_preserved": "DEV-IT-011" in plain or "SOP-IT-002" in plain,
        "profile_metadata_formal": (structured.get("style_profile") or {}).get("tone") == "formal",
    }


def print_result(errors: list[str]) -> None:
    section("RESULT")
    if errors:
        fail("FAILED")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    ok("PASSED")
    sys.exit(0)
