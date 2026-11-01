# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Persist the clinical session lifecycle and fold drafts into sessions.

A clinical session row now exists from the moment it is saved as a draft or
launched, with one authoritative status: draft, queued, running, completed,
cancelled or failed. Legacy ``successful`` sessions become ``completed``. The
single legacy ``clinical_input_drafts`` row, when it holds content, is copied
into a ``draft`` session before that table is removed.

Revision ID: 202611010003
Revises: 202611010002
Create Date: 2026-11-01
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "202611010003"
down_revision: Union[str, Sequence[str], None] = "202611010002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_STATUS_CONSTRAINT = "ck_clinical_sessions_lifecycle_status"
_STATUS_SQL = (
    "session_status IN ('draft', 'queued', 'running', 'completed', "
    "'cancelled', 'failed')"
)

###############################################################################
def upgrade() -> None:
    with op.batch_alter_table("clinical_sessions", schema=None) as batch_op:
        batch_op.add_column(sa.Column("clinical_input", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("job_id", sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column("run_request_json", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("progress_stage", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("progress_percent", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("progress_message", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("error_json", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "state_version",
                sa.Integer(),
                server_default=sa.text("0"),
                nullable=False,
            )
        )
        batch_op.add_column(sa.Column("created_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("updated_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("queued_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("started_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("finished_at", sa.DateTime(), nullable=True))

    op.execute(
        sa.text(
            """
            UPDATE clinical_sessions
            SET session_status = CASE
                    WHEN lower(coalesce(session_status, '')) = 'failed' THEN 'failed'
                    ELSE 'completed'
                END,
                created_at = coalesce(session_timestamp, CURRENT_TIMESTAMP),
                updated_at = coalesce(session_timestamp, CURRENT_TIMESTAMP),
                finished_at = session_timestamp
            """
        )
    )

    # Every session owns its clinician-entered input; rebuild it for sessions
    # that stored only the split sections.
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT id, anamnesis, drugs_text, laboratory_analysis "
            "FROM clinical_sessions"
        )
    ).all()
    for session_id, *sections in rows:
        clinical_input = "\n\n".join(part for part in sections if part)
        if clinical_input:
            bind.execute(
                sa.text(
                    "UPDATE clinical_sessions SET clinical_input = :clinical_input "
                    "WHERE id = :session_id"
                ),
                {"clinical_input": clinical_input, "session_id": session_id},
            )

    # Preserve the one legacy working draft as a first-class draft session.
    op.execute(
        sa.text(
            """
            INSERT INTO clinical_sessions (
                patient_name, visit_date, clinical_input, session_status,
                session_kind, state_version, created_at, updated_at
            )
            SELECT patient_name, visit_date, clinical_input, 'draft',
                   'standard_assessment', 0,
                   coalesce(updated_at, CURRENT_TIMESTAMP),
                   coalesce(updated_at, CURRENT_TIMESTAMP)
            FROM clinical_input_drafts
            WHERE coalesce(patient_name, '') <> ''
               OR visit_date IS NOT NULL
               OR coalesce(clinical_input, '') <> ''
            """
        )
    )
    op.drop_table("clinical_input_drafts")

    with op.batch_alter_table("clinical_sessions", schema=None) as batch_op:
        batch_op.alter_column(
            "session_status",
            existing_type=sa.String(),
            nullable=False,
            server_default=sa.text("'draft'"),
        )
        batch_op.create_check_constraint(_STATUS_CONSTRAINT, _STATUS_SQL)
        batch_op.create_index(
            "ix_clinical_sessions_status_queued_at",
            ["session_status", "queued_at"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_sessions_updated_at_id",
            ["updated_at", "id"],
            unique=False,
        )

    with op.batch_alter_table("clinical_session_timelines", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("source_text_hash", sa.String(length=64), nullable=True)
        )

###############################################################################
def downgrade() -> None:
    with op.batch_alter_table("clinical_session_timelines", schema=None) as batch_op:
        batch_op.drop_column("source_text_hash")

    op.create_table(
        "clinical_input_drafts",
        sa.Column("id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("draft_key", sa.String(length=36), nullable=True),
        sa.Column("revision", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("patient_name", sa.String(length=200), nullable=True),
        sa.Column("visit_date", sa.Date(), nullable=True),
        sa.Column("clinical_input", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_clinical_input_drafts")),
    )
    with op.batch_alter_table("clinical_sessions", schema=None) as batch_op:
        batch_op.drop_constraint(_STATUS_CONSTRAINT, type_="check")
    # The previous schema has no unfinished sessions.
    op.execute(
        sa.text(
            "DELETE FROM clinical_sessions "
            "WHERE session_status IN ('draft', 'queued', 'running')"
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE clinical_sessions
            SET session_status = CASE
                WHEN session_status = 'completed' THEN 'successful'
                ELSE 'failed'
            END
            """
        )
    )

    with op.batch_alter_table("clinical_sessions", schema=None) as batch_op:
        batch_op.drop_index("ix_clinical_sessions_updated_at_id")
        batch_op.drop_index("ix_clinical_sessions_status_queued_at")
        batch_op.alter_column(
            "session_status",
            existing_type=sa.String(),
            nullable=True,
            server_default=None,
        )
        for column in (
            "finished_at",
            "started_at",
            "queued_at",
            "updated_at",
            "created_at",
            "state_version",
            "error_json",
            "progress_message",
            "progress_percent",
            "progress_stage",
            "run_request_json",
            "job_id",
            "clinical_input",
        ):
            batch_op.drop_column(column)
