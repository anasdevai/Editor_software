import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["DATABASE_URL_LOCAL"] = "postgresql://postgres:Admin123@127.0.0.1:5432/editor_db"

from app.database import SessionLocal  # noqa: E402
from app.models import ClientProfile, SOP, SOPDetectedParameters  # noqa: E402
from app.ai_routes import _normalize_style_reference, _extract_explicit_style_reference  # noqa: E402

instruction = "Rewrite SOP-IT-002 in german_sop2 style"
print("EXTRACTED:", _extract_explicit_style_reference(instruction))
print("NORMALIZED:", _normalize_style_reference(_extract_explicit_style_reference(instruction) or ""))

db = SessionLocal()
try:
    print("\nPROFILES")
    for profile in db.query(ClientProfile).all():
        fields = [
            profile.name,
            profile.company_name,
            (profile.active_profile_json or {}).get("profile_name") if isinstance(profile.active_profile_json, dict) else None,
            (profile.active_profile_json or {}).get("name") if isinstance(profile.active_profile_json, dict) else None,
        ]
        print([f"{value} -> {_normalize_style_reference(value or '')}" for value in fields if value])

    print("\nSOP DETECTED ROWS")
    rows = db.query(SOPDetectedParameters).order_by(SOPDetectedParameters.created_at.desc()).all()
    for row in rows[:12]:
        sop = db.query(SOP).filter(SOP.id == row.sop_id).first() if row.sop_id else None
        fields = [
            row.source_filename,
            row.client_name,
            sop.sop_number if sop else None,
            sop.title if sop else None,
        ]
        print([f"{value} -> {_normalize_style_reference(value or '')}" for value in fields if value])

    print("\nSOP TABLE")
    for sop in db.query(SOP).all():
        if "german" in _normalize_style_reference(sop.sop_number or "") or "german" in _normalize_style_reference(sop.title or ""):
            print(
                [
                    f"{sop.sop_number} -> {_normalize_style_reference(sop.sop_number or '')}",
                    f"{sop.title} -> {_normalize_style_reference(sop.title or '')}",
                ]
            )
finally:
    db.close()
