"""[startup-db] DB schema inspection helper."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    from app.database import engine
    from sqlalchemy import text

    with engine.begin() as conn:
        ver = conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one_or_none()
        print(f"alembic_version: {ver}")

        rows = conn.execute(
            text(
                """
                SELECT column_name FROM information_schema.columns
                 WHERE table_name = 'sops'
                 ORDER BY ordinal_position
                """
            )
        ).fetchall()
        print("sops columns:", [r[0] for r in rows])

        rows = conn.execute(
            text(
                """
                SELECT column_name FROM information_schema.columns
                 WHERE table_name = 'embedding_jobs'
                 ORDER BY ordinal_position
                """
            )
        ).fetchall()
        print("embedding_jobs columns:", [r[0] for r in rows])

        rows = conn.execute(text("SELECT COUNT(*) FROM sops")).scalar_one()
        print(f"sops count: {rows}")
        rows = conn.execute(text("SELECT COUNT(*) FROM knowledge_chunks")).scalar_one()
        print(f"knowledge_chunks count: {rows}")
        rows = conn.execute(text("SELECT COUNT(*) FROM embedding_jobs")).scalar_one()
        print(f"embedding_jobs count: {rows}")


if __name__ == "__main__":
    main()
