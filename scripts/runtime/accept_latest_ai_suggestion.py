from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv

load_dotenv(BACKEND_DIR / ".env", override=True)
os.environ["DATABASE_URL_LOCAL"] = "postgresql://postgres:Admin123@127.0.0.1:5432/editor_db"

from app.database import SessionLocal
from app.models import AISuggestion


def main() -> None:
    db = SessionLocal()
    try:
        suggestion = (
            db.query(AISuggestion)
            .filter(AISuggestion.action.in_(["rewrite", "improve"]))
            .order_by(AISuggestion.created_at.desc())
            .first()
        )
        if not suggestion:
            print("no suggestion found")
            return
        suggestion.status = "accepted"
        suggestion.accepted_at = datetime.utcnow()
        db.commit()
        print(f"accepted {suggestion.id}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
