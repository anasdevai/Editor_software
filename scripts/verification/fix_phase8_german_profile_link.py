"""
Fix Phase 8 profile linkage
===========================

Links internal SOP NLP/action profile rows to the learned German profile so
rewrite/improve/gap-check actions load German_Pharma_SOP_Profile instead of the
generic Client profile.

Run from project root:
    .venv\\Scripts\\python.exe scripts\\fix_phase8_german_profile_link.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv

load_dotenv(BACKEND_DIR / ".env", override=True)
os.environ["DATABASE_URL_LOCAL"] = "postgresql://postgres:Admin123@127.0.0.1:5432/editor_db"

from app.database import SessionLocal
from app.models import ClientProfile, SOP, SOPDetectedParameters


PROFILE_NAME = "German_Pharma_SOP_Profile"
TARGET_SOPS = ["SOP-IT-002", "SOP-IT-003"]


def main() -> None:
    db = SessionLocal()
    try:
        profile = db.query(ClientProfile).filter(ClientProfile.name == PROFILE_NAME).first()
        if not profile:
            raise RuntimeError(f"{PROFILE_NAME} not found. Run Phase 5 first.")

        print(f"German profile: {profile.name} id={profile.id} current_version_id={profile.current_version_id}")
        total_updated = 0

        for sop_number in TARGET_SOPS:
            sop = db.query(SOP).filter(SOP.sop_number == sop_number).first()
            if not sop:
                print(f"[WARN] {sop_number} not found")
                continue

            rows = (
                db.query(SOPDetectedParameters)
                .filter(SOPDetectedParameters.sop_id == sop.id)
                .order_by(SOPDetectedParameters.created_at.desc())
                .all()
            )
            print(f"{sop_number}: {len(rows)} NLP row(s)")

            for row in rows:
                before = row.client_profile_id
                if before != profile.id:
                    row.client_profile_id = profile.id
                    row.client_name = profile.name
                    total_updated += 1
                    print(f"  updated nlp_id={row.id} {before} -> {profile.id}")
                else:
                    print(f"  already linked nlp_id={row.id}")

        db.commit()
        print(f"\nUpdated rows: {total_updated}")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
