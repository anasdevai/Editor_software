"""Allow chat_sessions without a user (anonymous / unauthenticated chat).

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-12

"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint("chat_sessions_user_id_fkey", "chat_sessions", type_="foreignkey")
    op.alter_column(
        "chat_sessions",
        "user_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
    )
    op.create_foreign_key(
        "chat_sessions_user_id_fkey",
        "chat_sessions",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.execute(
        "DELETE FROM chat_messages WHERE session_id IN (SELECT id FROM chat_sessions WHERE user_id IS NULL)"
    )
    op.execute("DELETE FROM chat_sessions WHERE user_id IS NULL")
    op.drop_constraint("chat_sessions_user_id_fkey", "chat_sessions", type_="foreignkey")
    op.alter_column(
        "chat_sessions",
        "user_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )
    op.create_foreign_key(
        "chat_sessions_user_id_fkey",
        "chat_sessions",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
