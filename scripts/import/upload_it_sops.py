"""
Upload SOP-IT-002 and SOP-IT-003 into the database
===================================================
Uses the FastAPI TestClient to call /api/extract-text and /api/editor/docs
so the SOPs are fully ingested with NLP parameters, profile links, etc.

Run from project root:
    .venv\\Scripts\\python.exe scripts\\upload_it_sops.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR  = PROJECT_ROOT / "backend"
SOP_DIR      = PROJECT_ROOT / "SOP"

for p in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env", override=True)
os.environ["DATABASE_URL_LOCAL"] = "postgresql://postgres:Admin123@127.0.0.1:5432/editor_db"

from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal
from app.models import SOP, SOPVersion, SOPDetectedParameters

client = TestClient(app)

TARGET_FILES = [
    ("SOP-IT-002.txt", "SOP-IT-002", "Netzwerksicherheit & Firewall (OT/IT-Trennung)"),
    ("SOP-IT-003.txt", "SOP-IT-003", "Notfallzugriff (Break-Glass-Verfahren)"),
]

SEP = "=" * 60

def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def check_existing(db, sop_number: str) -> bool:
    existing = db.query(SOP).filter(SOP.sop_number == sop_number).first()
    if existing:
        print(f"  [SKIP] {sop_number} already exists in DB (id={existing.id})")
        return True
    return False


def upload_txt_sop(filename: str, sop_number: str, title_hint: str) -> dict:
    txt_path = SOP_DIR / filename
    result = {
        "file": filename,
        "sop_number": sop_number,
        "status": "fail",
        "sop_id": None,
        "issues": [],
    }

    if not txt_path.exists():
        result["issues"].append(f"{filename} not found at {txt_path}")
        return result

    raw_text = txt_path.read_text(encoding="utf-8", errors="replace")

    # Build a minimal TipTap doc from the raw text lines
    paragraphs = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped:
            paragraphs.append({
                "type": "paragraph",
                "content": [{"type": "text", "text": stripped}]
            })
    doc_json = {"type": "doc", "content": paragraphs}

    metadata = {
        "sopMetadata": {
            "documentId": sop_number,
            "title": title_hint,
            "version": "1.0",
            "status": "Effective",
            "department": "IT",
            "effectiveDate": "2024-01-01",
            "reviewDate": "2025-01-01",
        }
    }

    print(f"  Creating doc {sop_number} / {title_hint} ...")
    payload = {
        "title": title_hint,
        "doc_json": doc_json,
        "metadata_json": metadata,
    }
    res = client.post("/api/editor/docs", json=payload)
    if res.status_code not in (200, 201):
        result["issues"].append(f"create doc failed ({res.status_code}): {res.text[:300]}")
        return result

    doc_data = res.json()
    result["status"] = "success"
    result["sop_id"] = doc_data.get("id")
    print(f"  [OK] Created SOP id={result['sop_id']} sop_number={sop_number}")

    # Update sop_number field directly if the API used a generic one
    db = SessionLocal()
    try:
        sop_row = db.query(SOP).filter(SOP.id == result["sop_id"]).first()
        if sop_row and sop_row.sop_number != sop_number:
            old = sop_row.sop_number
            sop_row.sop_number = sop_number
            db.commit()
            print(f"  [FIX] sop_number corrected: {old!r} -> {sop_number!r}")
    except Exception as exc:
        db.rollback()
        result["issues"].append(f"sop_number fix failed: {exc}")
    finally:
        db.close()

    return result


def trigger_nlp_analysis(sop_id: str, sop_number: str) -> None:
    """
    POST to /api/sops/{sop_id}/analyze-profile to generate SOPDetectedParameters.
    Falls back silently if endpoint not available.
    """
    try:
        res = client.post(f"/api/sops/{sop_id}/analyze-profile", json={})
        if res.status_code in (200, 201, 202):
            print(f"  [NLP] Profile analysis triggered for {sop_number}")
        else:
            print(f"  [NLP] analyze-profile returned {res.status_code} (may be normal): {res.text[:120]}")
    except Exception as exc:
        print(f"  [NLP] analyze-profile call failed (skipping): {exc}")


def main() -> None:
    section("Upload IT SOPs (SOP-IT-002, SOP-IT-003)")

    db = SessionLocal()
    results = []
    try:
        for filename, sop_number, title_hint in TARGET_FILES:
            section(f"Processing {sop_number}")
            if check_existing(db, sop_number):
                results.append({"sop_number": sop_number, "status": "already_exists"})
                continue
            db.close()
            db = SessionLocal()

            result = upload_txt_sop(filename, sop_number, title_hint)
            results.append(result)

            if result["status"] == "success" and result["sop_id"]:
                time.sleep(0.5)
                trigger_nlp_analysis(result["sop_id"], sop_number)

    finally:
        db.close()

    section("RESULTS")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    section("DB CHECK")
    db2 = SessionLocal()
    try:
        for _, sop_number, _ in TARGET_FILES:
            sop = db2.query(SOP).filter(SOP.sop_number == sop_number).first()
            if sop:
                nlp_count = db2.query(SOPDetectedParameters).filter(
                    SOPDetectedParameters.sop_id == sop.id
                ).count()
                print(f"  {sop_number}: id={sop.id} | nlp_rows={nlp_count}")
            else:
                print(f"  {sop_number}: NOT FOUND in DB")
    finally:
        db2.close()

    print(f"\n{SEP}")
    print("  DONE – now run: fix_phase8_german_profile_link.py")
    print(SEP)


if __name__ == "__main__":
    main()
