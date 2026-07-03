import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from phase9_17_utils import *

db = SessionLocal()
try:
    suggestion = db.query(AISuggestion).order_by(AISuggestion.created_at.desc()).first()
    print(
        {
            "id": str(suggestion.id) if suggestion else None,
            "status": suggestion.status if suggestion else None,
            "profile_id": str(suggestion.profile_id) if suggestion and suggestion.profile_id else None,
            "profile_version_id": str(suggestion.profile_version_id) if suggestion and suggestion.profile_version_id else None,
            "metadata_json": suggestion.metadata_json if suggestion else None,
        }
    )
finally:
    db.close()
