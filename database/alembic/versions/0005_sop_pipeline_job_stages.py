"""SOP pipeline job stages and active job pointer on sops.

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-13

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    tables = insp.get_table_names()

    if "embedding_jobs" in tables:
        op.add_column(
            "embedding_jobs",
            sa.Column("enqueued_content_hash", sa.String(length=64), nullable=True),
        )
        op.create_index(
            "ix_embedding_jobs_enqueued_content_hash",
            "embedding_jobs",
            ["enqueued_content_hash"],
            unique=False,
        )
        for col, default in (
            ("chunking_status", "pending"),
            ("embeddings_status", "pending"),
            ("qdrant_status", "pending"),
            ("nlp_status", "pending"),
            ("semantic_linking_status", "pending"),
        ):
            op.add_column(
                "embedding_jobs",
                sa.Column(col, sa.String(length=30), nullable=False, server_default=default),
            )
        op.execute(
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
            """
        )
        op.execute(
            "UPDATE embedding_jobs SET status = 'processing' WHERE status = 'running'"
        )

    if "sops" in tables:
        op.add_column(
            "sops",
            sa.Column("active_pipeline_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        )
        op.create_index(
            "ix_sops_active_pipeline_job_id",
            "sops",
            ["active_pipeline_job_id"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    tables = insp.get_table_names()

    if "sops" in tables:
        op.drop_index("ix_sops_active_pipeline_job_id", table_name="sops")
        op.drop_column("sops", "active_pipeline_job_id")

    if "embedding_jobs" in tables:
        op.drop_index("ix_embedding_jobs_enqueued_content_hash", table_name="embedding_jobs")
        for col in (
            "semantic_linking_status",
            "nlp_status",
            "qdrant_status",
            "embeddings_status",
            "chunking_status",
            "enqueued_content_hash",
        ):
            op.drop_column("embedding_jobs", col)
        op.execute(
            "UPDATE embedding_jobs SET status = 'running' WHERE status = 'processing'"
        )
