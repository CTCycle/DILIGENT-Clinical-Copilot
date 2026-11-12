# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Agent config schema 3: custom instructions and report template.

The assessment instruction fields were replaced by a single custom instruction
and an optional report template. Stored agents are discarded and reseeded as
the default agent at initialization (no legacy compatibility); run links are
cleared because every run keeps its own frozen runtime snapshot.

Revision ID: 202611120002
Revises: 202611120001
Create Date: 2026-11-12
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "202611120002"
down_revision: Union[str, Sequence[str], None] = "202611120001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


###############################################################################
def _discard_agents() -> None:
    op.execute("UPDATE clinical_sessions SET agent_revision_id = NULL")
    op.execute("UPDATE clinical_session_revision_runs SET agent_revision_id = NULL")
    op.execute("UPDATE clinical_session_timelines SET agent_revision_id = NULL")
    op.execute("DELETE FROM agent_revisions")
    op.execute("DELETE FROM agent_definitions")


###############################################################################
def upgrade() -> None:
    _discard_agents()


###############################################################################
def downgrade() -> None:
    _discard_agents()
