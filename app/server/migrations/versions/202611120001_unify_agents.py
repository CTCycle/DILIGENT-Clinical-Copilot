# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Unify agents: one entity with three capabilities and explicit lifecycle flags.

Existing kind-specific agents are discarded (they are reseeded as one default
agent at initialization); run links are cleared because every run keeps its
own frozen runtime snapshot. Timelines gain an agent revision link.

Revision ID: 202611120001
Revises: 202611110001
Create Date: 2026-11-12
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "202611120001"
down_revision: Union[str, Sequence[str], None] = "202611110001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TIMELINES_FK = "fk_clinical_session_timelines_agent_revision_id_agent_revisions"
_DEFAULT_INDEX = "uq_agent_definitions_single_default"


###############################################################################
def upgrade() -> None:
    op.execute("UPDATE clinical_sessions SET agent_revision_id = NULL")
    op.execute("UPDATE clinical_session_revision_runs SET agent_revision_id = NULL")
    op.execute("DELETE FROM agent_revisions")
    op.execute("DELETE FROM agent_definitions")

    with op.batch_alter_table("agent_definitions", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_agent_definitions_kind", type_="check")
        batch_op.drop_constraint("uq_agent_definitions_default_kind", type_="unique")
        batch_op.drop_index("ix_agent_definitions_kind")
        batch_op.drop_column("kind")
        batch_op.drop_column("default_kind")
        batch_op.add_column(
            sa.Column(
                "is_active",
                sa.Boolean(),
                server_default=sa.text("true"),
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column(
                "is_locked",
                sa.Boolean(),
                server_default=sa.text("false"),
                nullable=False,
            )
        )
    op.create_index(
        _DEFAULT_INDEX,
        "agent_definitions",
        ["is_default"],
        unique=True,
        sqlite_where=sa.text("is_default"),
        postgresql_where=sa.text("is_default"),
    )

    with op.batch_alter_table("clinical_session_timelines") as batch_op:
        batch_op.add_column(sa.Column("agent_revision_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            _TIMELINES_FK, "agent_revisions", ["agent_revision_id"], ["id"]
        )
        batch_op.create_index(
            "ix_clinical_session_timelines_agent_revision_id", ["agent_revision_id"]
        )


###############################################################################
def downgrade() -> None:
    with op.batch_alter_table("clinical_session_timelines") as batch_op:
        batch_op.drop_index("ix_clinical_session_timelines_agent_revision_id")
        batch_op.drop_constraint(_TIMELINES_FK, type_="foreignkey")
        batch_op.drop_column("agent_revision_id")

    op.drop_index(_DEFAULT_INDEX, table_name="agent_definitions")
    op.execute("UPDATE clinical_sessions SET agent_revision_id = NULL")
    op.execute("UPDATE clinical_session_revision_runs SET agent_revision_id = NULL")
    op.execute("DELETE FROM agent_revisions")
    op.execute("DELETE FROM agent_definitions")
    with op.batch_alter_table("agent_definitions", recreate="always") as batch_op:
        batch_op.drop_column("is_active")
        batch_op.drop_column("is_locked")
        batch_op.add_column(
            sa.Column("kind", sa.String(length=16), nullable=False)
        )
        batch_op.add_column(
            sa.Column("default_kind", sa.String(length=16), nullable=True)
        )
        batch_op.create_check_constraint(
            "ck_agent_definitions_kind", "kind IN ('dili', 'revision')"
        )
        batch_op.create_unique_constraint(
            "uq_agent_definitions_default_kind", ["default_kind"]
        )
        batch_op.create_index("ix_agent_definitions_kind", ["kind"])
