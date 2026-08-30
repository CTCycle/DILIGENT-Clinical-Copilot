"""Remove the obsolete application configuration schema marker.

Revision ID: 202608200002
Revises: 202608200001
Create Date: 2026-08-20
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "202608200002"
down_revision: Union[str, Sequence[str], None] = "202608200001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


###############################################################################
def upgrade() -> None:
    with op.batch_alter_table("application_configuration") as batch_op:
        batch_op.drop_column("schema_version")


###############################################################################
def downgrade() -> None:
    with op.batch_alter_table("application_configuration") as batch_op:
        batch_op.add_column(
            sa.Column(
                "schema_version",
                sa.Integer(),
                server_default=sa.text("1"),
                nullable=False,
            )
        )
