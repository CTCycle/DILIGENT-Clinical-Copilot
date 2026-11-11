# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Add reusable versioned agents and link runs to the revision that produced them.

Revision ID: 202611110001
Revises: 202611010003
Create Date: 2026-11-11
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "202611110001"
down_revision: Union[str, Sequence[str], None] = "202611010003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SESSIONS_FK = "fk_clinical_sessions_agent_revision_id_agent_revisions"
_RUNS_FK = "fk_clinical_session_revision_runs_agent_revision_id_agent_revisions"


###############################################################################
def upgrade() -> None:
    op.create_table(
        "agent_definitions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column(
            "description", sa.Text(), server_default=sa.text("''"), nullable=False
        ),
        sa.Column("current_revision", sa.Integer(), nullable=False),
        sa.Column(
            "is_default", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("default_kind", sa.String(length=16), nullable=True),
        sa.Column(
            "is_archived", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "kind IN ('dili', 'revision')", name="ck_agent_definitions_kind"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_definitions")),
        sa.UniqueConstraint(
            "default_kind", name=op.f("uq_agent_definitions_default_kind")
        ),
    )
    op.create_index("ix_agent_definitions_kind", "agent_definitions", ["kind"])

    op.create_table(
        "agent_revisions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("agent_id", sa.String(length=36), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("configuration_json", sa.JSON(), nullable=False),
        sa.Column("configuration_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agent_definitions.id"],
            name=op.f("fk_agent_revisions_agent_id_agent_definitions"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_revisions")),
        sa.UniqueConstraint(
            "agent_id",
            "revision_number",
            name=op.f("uq_agent_revisions_agent_revision"),
        ),
    )
    op.create_index("ix_agent_revisions_agent_id", "agent_revisions", ["agent_id"])

    with op.batch_alter_table("clinical_sessions", schema=None) as batch_op:
        batch_op.add_column(sa.Column("agent_revision_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            _SESSIONS_FK, "agent_revisions", ["agent_revision_id"], ["id"]
        )
        batch_op.create_index(
            "ix_clinical_sessions_agent_revision_id", ["agent_revision_id"]
        )

    with op.batch_alter_table(
        "clinical_session_revision_runs", schema=None
    ) as batch_op:
        batch_op.add_column(sa.Column("agent_revision_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            _RUNS_FK, "agent_revisions", ["agent_revision_id"], ["id"]
        )
        batch_op.create_index(
            "ix_clinical_session_revision_runs_agent_revision_id",
            ["agent_revision_id"],
        )


###############################################################################
def downgrade() -> None:
    with op.batch_alter_table(
        "clinical_session_revision_runs", schema=None
    ) as batch_op:
        batch_op.drop_index("ix_clinical_session_revision_runs_agent_revision_id")
        batch_op.drop_constraint(_RUNS_FK, type_="foreignkey")
        batch_op.drop_column("agent_revision_id")

    with op.batch_alter_table("clinical_sessions", schema=None) as batch_op:
        batch_op.drop_index("ix_clinical_sessions_agent_revision_id")
        batch_op.drop_constraint(_SESSIONS_FK, type_="foreignkey")
        batch_op.drop_column("agent_revision_id")

    op.drop_index("ix_agent_revisions_agent_id", table_name="agent_revisions")
    op.drop_table("agent_revisions")
    op.drop_index("ix_agent_definitions_kind", table_name="agent_definitions")
    op.drop_table("agent_definitions")
