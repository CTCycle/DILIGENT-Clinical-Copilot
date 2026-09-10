"""Add FDA DILIrank 2.0 structured knowledge records.

Revision ID: 202609100001
Revises: 202608200003
Create Date: 2026-09-10
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "202609100001"
down_revision: Union[str, Sequence[str], None] = "202608200003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

###############################################################################
def upgrade() -> None:
    op.create_table(
        "dilirank_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_id", sa.Integer(), nullable=True),
        sa.Column("ltkb_id", sa.String(), nullable=False),
        sa.Column("compound_name", sa.Text(), nullable=False),
        sa.Column("compound_name_norm", sa.String(), nullable=False),
        sa.Column("severity_class", sa.Integer(), nullable=True),
        sa.Column("label_section", sa.Text(), nullable=True),
        sa.Column("dili_concern", sa.String(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("source_last_modified", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["drug_id"],
            ["drugs.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ltkb_id", name="uq_dilirank_records_ltkb_id"),
    )
    op.create_index(
        "ix_dilirank_records_drug_id", "dilirank_records", ["drug_id"], unique=False
    )
    op.create_index(
        "ix_dilirank_records_compound_name_norm",
        "dilirank_records",
        ["compound_name_norm"],
        unique=False,
    )
    op.create_index(
        "ix_dilirank_records_dili_concern",
        "dilirank_records",
        ["dili_concern"],
        unique=False,
    )

###############################################################################
def downgrade() -> None:
    op.drop_index("ix_dilirank_records_dili_concern", table_name="dilirank_records")
    op.drop_index(
        "ix_dilirank_records_compound_name_norm", table_name="dilirank_records"
    )
    op.drop_index("ix_dilirank_records_drug_id", table_name="dilirank_records")
    op.drop_table("dilirank_records")
