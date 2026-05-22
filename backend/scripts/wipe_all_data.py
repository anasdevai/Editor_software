import sys

sys.path.insert(0, r"backend")

from sqlalchemy import text

from app.database import engine
from app.services.semantic_pipeline import DEFAULT_COLLECTION, _get_qdrant


def wipe_postgres() -> dict:
    keep = {"alembic_version"}
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
        ).fetchall()
        tables = [r[0] for r in rows if r[0] not in keep]
        if tables:
            quoted = ", ".join([f'"{t}"' for t in tables])
            conn.execute(text(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE"))
    return {"truncated_tables": len(tables), "tables": tables}


def verify_postgres() -> dict:
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
        ).fetchall()
        tables = [r[0] for r in rows if r[0] != "alembic_version"]
        counts = {}
        for t in tables:
            counts[t] = conn.execute(text(f'SELECT COUNT(*) FROM "{t}"')).scalar_one()
    return counts


def verify_qdrant() -> dict:
    client = _get_qdrant()
    if not client.collection_exists(DEFAULT_COLLECTION):
        return {"collection_exists": False, "count": 0}
    stats = client.count(collection_name=DEFAULT_COLLECTION, exact=True)
    return {"collection_exists": True, "count": int(getattr(stats, "count", 0) or 0)}


if __name__ == "__main__":
    pg = wipe_postgres()
    pg_counts = verify_postgres()
    qv = verify_qdrant()
    print({"postgres_wipe": pg, "postgres_counts_after": pg_counts, "qdrant_after": qv})
