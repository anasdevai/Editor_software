from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv

load_dotenv(BACKEND_DIR / ".env", override=True)
os.environ["DATABASE_URL_LOCAL"] = "postgresql://postgres:Admin123@127.0.0.1:5432/editor_db"

from app.database import SessionLocal
from app.models import SOP, SOPVersion
from app.services.sop_profile_storage_service import analyze_and_store_sop_profile
from app.utils.tiptap_text import extract_plain_text_from_tiptap


def main() -> None:
    db = SessionLocal()
    try:
        sops = (
            db.query(SOP)
            .filter(SOP.title.in_(["german_sop.pdf", "german_sop2.pdf", "german_sop3.pdf"]))
            .order_by(SOP.title)
            .all()
        )
        print(f"reanalyzing={len(sops)}")
        for sop in sops:
            version = db.query(SOPVersion).filter(SOPVersion.id == sop.current_version_id).first()
            if not version:
                print(f"skip {sop.sop_number}: no current version")
                continue
            text = extract_plain_text_from_tiptap(version.content_json or {})
            print(f"{sop.sop_number} text_chars={len(text)}")
            analyze_and_store_sop_profile(
                db,
                sop.id,
                version.id,
                text,
                client_name="GermanPharmaClient",
                source_filename=sop.title or sop.sop_number,
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
