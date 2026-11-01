# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Add persisted clinical session attachments.

Images and JSON files attached in the Metadata tab were stored only as name and
size stubs. Their content is now persisted per session and version.

Revision ID: 202611010002
Revises: 202611010001
Create Date: 2026-11-01
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "202611010002"
down_revision: Union[str, Sequence[str], None] = "202611010001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

###############################################################################
def upgrade() -> None:
    op.create_table(
        "clinical_session_attachments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("version_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("file_name", sa.String(), nullable=False),
        sa.Column("media_type", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind IN ('image', 'json', 'document')",
            name="ck_clinical_session_attachments_kind",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"], ["clinical_sessions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["version_id"], ["clinical_session_versions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "session_id",
            "version_id",
            "sha256",
            name="uq_clinical_session_attachments_session_version_sha256",
        ),
    )
    op.create_index(
        "ix_clinical_session_attachments_session_id",
        "clinical_session_attachments",
        ["session_id"],
    )

###############################################################################
def downgrade() -> None:
    op.drop_index(
        "ix_clinical_session_attachments_session_id",
        table_name="clinical_session_attachments",
    )
    op.drop_table("clinical_session_attachments")
