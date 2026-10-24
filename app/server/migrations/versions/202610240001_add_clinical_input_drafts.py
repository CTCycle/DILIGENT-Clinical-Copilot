# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Add the single resumable DILI Agent input draft.

Revision ID: 202610240001
Revises: 202609170001
Create Date: 2026-10-24
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "202610240001"
down_revision: Union[str, Sequence[str], None] = "202609170001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

###############################################################################
def upgrade() -> None:
    op.create_table(
        "clinical_input_drafts",
        sa.Column("id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("draft_key", sa.String(length=36), nullable=True),
        sa.Column("revision", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("patient_name", sa.String(length=200), nullable=True),
        sa.Column("visit_date", sa.Date(), nullable=True),
        sa.Column("clinical_input", sa.Text(), nullable=True),
        sa.Column("use_rag", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_clinical_input_drafts")),
    )

###############################################################################
def downgrade() -> None:
    op.drop_table("clinical_input_drafts")
