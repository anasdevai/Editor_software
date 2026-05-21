import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    from app.database import SessionLocal
    from app.models import SOP

    db = SessionLocal()
    try:
        for s in db.query(SOP).all():
            print(f"{s.id} | {s.sop_number} | {s.title} | active={s.is_active}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
