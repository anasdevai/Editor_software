"""
[startup-db] Repair schema drift between alembic_version=0005 and the actual DB.

The 0005 migration was stamped but never applied, so the columns it adds
(sops.active_pipeline_job_id and the per-stage status columns on embedding_jobs)
are missing. This script applies them idempotently using IF NOT EXISTS guards.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


SQL_STATEMENTS = (
    """
    ALTER TABLE embedding_jobs
      ADD COLUMN IF NOT EXISTS enqueued_content_hash VARCHAR(64)
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_embedding_jobs_enqueued_content_hash
      ON embedding_jobs (enqueued_content_hash)
    """,
    """
    ALTER TABLE embedding_jobs
      ADD COLUMN IF NOT EXISTS chunking_status VARCHAR(30) NOT NULL DEFAULT 'pending'
    """,
    """
    ALTER TABLE embedding_jobs
      ADD COLUMN IF NOT EXISTS embeddings_status VARCHAR(30) NOT NULL DEFAULT 'pending'
    """,
    """
    ALTER TABLE embedding_jobs
      ADD COLUMN IF NOT EXISTS qdrant_status VARCHAR(30) NOT NULL DEFAULT 'pending'
    """,
    """
    ALTER TABLE embedding_jobs
      ADD COLUMN IF NOT EXISTS nlp_status VARCHAR(30) NOT NULL DEFAULT 'pending'
    """,
    """
    ALTER TABLE embedding_jobs
      ADD COLUMN IF NOT EXISTS semantic_linking_status VARCHAR(30) NOT NULL DEFAULT 'pending'
    """,
    """
    UPDATE embedding_jobs SET
      chunking_status = CASE
        WHEN status = 'completed' THEN 'completed'
        WHEN status = 'failed' THEN 'failed'
        WHEN status = 'cancelled' THEN 'cancelled'
        ELSE 'pending' END,
      embeddings_status = CASE
        WHEN status = 'completed' THEN 'completed'
        WHEN status = 'failed' THEN 'failed'
        WHEN status = 'cancelled' THEN 'cancelled'
        ELSE 'pending' END,
      qdrant_status = CASE
        WHEN status = 'completed' THEN 'completed'
        WHEN status = 'failed' THEN 'failed'
        WHEN status = 'cancelled' THEN 'cancelled'
        ELSE 'pending' END,
      nlp_status = CASE
        WHEN status = 'completed' THEN 'completed'
        WHEN status = 'failed' THEN 'failed'
        WHEN status = 'cancelled' THEN 'cancelled'
        WHEN entity_type = 'sop' THEN 'pending'
        ELSE 'skipped' END,
      semantic_linking_status = CASE
        WHEN status = 'completed' THEN 'completed'
        WHEN status = 'failed' THEN 'failed'
        WHEN status = 'cancelled' THEN 'cancelled'
        ELSE 'pending' END
    """,
    """
    UPDATE embedding_jobs SET status = 'processing' WHERE status = 'running'
    """,
    """
    ALTER TABLE sops
      ADD COLUMN IF NOT EXISTS active_pipeline_job_id UUID
    """,
    """
    CREATE INDEX IF NOT EXISTS ix_sops_active_pipeline_job_id
      ON sops (active_pipeline_job_id)
    """,
    """
    UPDATE alembic_version SET version_num = '0005'
    """,
)


def main() -> None:
    from app.database import engine
    from sqlalchemy import text

    with engine.begin() as conn:
        for stmt in SQL_STATEMENTS:
            conn.execute(text(stmt))
        ver = conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        print(f"[startup-db] schema_repair_done alembic_version={ver}")


if __name__ == "__main__":
    main()
