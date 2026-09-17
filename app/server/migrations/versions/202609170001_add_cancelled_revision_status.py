"""Persist cancellation as a terminal revision-version status.

Revision ID: 202609170001
Revises: 202609100001
Create Date: 2026-09-17
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "202609170001"
down_revision: Union[str, Sequence[str], None] = "202609100001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Alembic's batch operation applies the metadata naming convention to this
# explicit name, producing the table-qualified name used by the baseline.
_VERSION_STATUS_CONSTRAINT = "ck_clinical_session_versions_version_status"
_VERSION_STATUS_SQL = (
    "version_status IN ("
    "'current', 'superseded', 'draft_revision', 'pending_qa', 'cancelled', "
    "'qa_failed', 'requires_human_review', 'llm_qa_passed', "
    "'human_approved', 'human_rejected'"
    ")"
)
_LEGACY_VERSION_STATUS_SQL = (
    "version_status IN ("
    "'current', 'superseded', 'draft_revision', 'pending_qa', "
    "'qa_failed', 'requires_human_review', 'llm_qa_passed', "
    "'human_approved', 'human_rejected'"
    ")"
)

###############################################################################
def upgrade() -> None:
    with op.batch_alter_table("clinical_session_versions", schema=None) as batch_op:
        batch_op.drop_constraint(_VERSION_STATUS_CONSTRAINT, type_="check")
        batch_op.create_check_constraint(
            _VERSION_STATUS_CONSTRAINT,
            _VERSION_STATUS_SQL,
        )

    op.execute(
        sa.text(
            """
            UPDATE clinical_session_revision_runs
            SET completed_at = COALESCE(completed_at, CURRENT_TIMESTAMP)
            WHERE status = 'cancelled'
              AND completed_at IS NULL
            """
        )
    )

    op.execute(
        sa.text(
            """
            UPDATE clinical_session_versions
            SET version_status = 'cancelled',
                llm_qa_status = 'not_run',
                session_id = NULL,
                completed_at = COALESCE(
                    completed_at,
                    (
                        SELECT revision_run.completed_at
                        FROM clinical_session_revision_runs AS revision_run
                        WHERE revision_run.pipeline_run_id = clinical_session_versions.pipeline_run_id
                    ),
                    CURRENT_TIMESTAMP
                )
            WHERE version_status IN ('draft_revision', 'pending_qa')
              AND EXISTS (
                  SELECT 1
                  FROM clinical_session_revision_runs AS revision_run
                  WHERE revision_run.pipeline_run_id = clinical_session_versions.pipeline_run_id
                    AND revision_run.status = 'cancelled'
              )
            """
        )
    )

    op.execute(
        sa.text(
            """
            UPDATE clinical_session_revision_steps
            SET status = 'cancelled',
                error_json = COALESCE(
                    error_json,
                    '{"message": "Revision was cancelled."}'
                ),
                completed_at = COALESCE(completed_at, CURRENT_TIMESTAMP)
            WHERE pipeline_run_id IN (
                SELECT pipeline_run_id
                FROM clinical_session_revision_runs
                WHERE status = 'cancelled'
            )
              AND status NOT IN ('completed', 'failed', 'cancelled')
            """
        )
    )

###############################################################################
def downgrade() -> None:
    op.execute(
        sa.text(
            """
            UPDATE clinical_session_versions
            SET version_status = 'draft_revision',
                llm_qa_status = 'pending',
                completed_at = NULL
            WHERE version_status = 'cancelled'
            """
        )
    )

    with op.batch_alter_table("clinical_session_versions", schema=None) as batch_op:
        batch_op.drop_constraint(_VERSION_STATUS_CONSTRAINT, type_="check")
        batch_op.create_check_constraint(
            _VERSION_STATUS_CONSTRAINT,
            _LEGACY_VERSION_STATUS_SQL,
        )
