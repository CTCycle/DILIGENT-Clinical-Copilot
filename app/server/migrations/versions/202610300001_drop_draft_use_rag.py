# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Drop the RAG flag from the DILI Agent input draft.

Evidence sources are now chosen per session in the pre-analysis wizard and are
no longer part of the resumable clinical input draft.

Revision ID: 202610300001
Revises: 202610290001
Create Date: 2026-10-30
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "202610300001"
down_revision: Union[str, Sequence[str], None] = "202610290001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

###############################################################################
def upgrade() -> None:
    with op.batch_alter_table("clinical_input_drafts") as batch_op:
        batch_op.drop_column("use_rag")

###############################################################################
def downgrade() -> None:
    with op.batch_alter_table("clinical_input_drafts") as batch_op:
        batch_op.add_column(
            sa.Column(
                "use_rag",
                sa.Boolean(),
                server_default=sa.text("false"),
                nullable=False,
            )
        )
