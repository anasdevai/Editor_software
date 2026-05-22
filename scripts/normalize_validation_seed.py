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

from sqlalchemy.orm.attributes import flag_modified

from app.database import SessionLocal
from app.models import ClientProfile, ProfileHistoryEvent, ProfileVersion, SOP


def main() -> None:
    db = SessionLocal()
    try:
        for title, number in [
            ("german_sop.pdf", "GERMAN-SOP-PHASE3"),
            ("german_sop2.pdf", "GERMAN-SOP2-PHASE3"),
            ("german_sop3.pdf", "GERMAN-SOP3-PHASE3"),
        ]:
            sop = db.query(SOP).filter(SOP.title == title).first()
            if sop:
                sop.sop_number = number

        profile = db.query(ClientProfile).filter(ClientProfile.name == "German_Pharma_SOP_Profile").first()
        if profile:
            versions = (
                db.query(ProfileVersion)
                .filter(ProfileVersion.profile_id == profile.id)
                .order_by(ProfileVersion.version_number.desc(), ProfileVersion.created_at.desc())
                .all()
            )
            keep = versions[0] if versions else None
            if keep:
                for old in versions[1:]:
                    (
                        db.query(ProfileHistoryEvent)
                        .filter(ProfileHistoryEvent.profile_version_id == old.id)
                        .update({ProfileHistoryEvent.profile_version_id: keep.id})
                    )
                    db.delete(old)
                keep.version_number = 1
                profile.current_version_id = keep.id
                active_json = dict(profile.active_profile_json or {})
                active_json["profile_version"] = 1
                profile.active_profile_json = active_json
                flag_modified(profile, "active_profile_json")
                rules_json = dict(keep.rules_json or {})
                rules_json["profile_version"] = 1
                keep.rules_json = rules_json
                flag_modified(keep, "rules_json")

        db.commit()
        print("validation seed normalized")
    finally:
        db.close()


if __name__ == "__main__":
    main()
