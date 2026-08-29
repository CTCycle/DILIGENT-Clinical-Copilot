"""Require explicit assignments for every model configuration role.

Revision ID: 202608200003
Revises: 202608200002
Create Date: 2026-08-29
"""

from __future__ import annotations

import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "202608200003"
down_revision: Union[str, Sequence[str], None] = "202608200002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

###############################################################################
def _normalized_role(payload: dict[str, object], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None

###############################################################################
def upgrade() -> None:
    configuration = sa.table(
        "application_configuration",
        sa.column("id", sa.Integer()),
        sa.column("payload", sa.JSON()),
    )
    connection = op.get_bind()
    row = connection.execute(
        sa.select(configuration.c.payload).where(configuration.c.id == 1)
    ).first()
    if row is None:
        return

    raw_payload = row[0]
    if isinstance(raw_payload, str):
        try:
            raw_payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Application configuration payload is not valid JSON."
            ) from exc
    if not isinstance(raw_payload, dict):
        raise RuntimeError("Application configuration payload must be an object.")

    payload = dict(raw_payload)
    missing_sources: list[str] = []
    if _normalized_role(payload, "revision_model") is None:
        source = _normalized_role(payload, "clinical_model")
        if source is None:
            missing_sources.append("clinical_model")
        else:
            payload["revision_model"] = source
    if _normalized_role(payload, "timeline_model") is None:
        source = _normalized_role(payload, "text_extraction_model")
        if source is None:
            missing_sources.append("text_extraction_model")
        else:
            payload["timeline_model"] = source
    if missing_sources:
        raise RuntimeError(
            "Cannot populate required model roles; missing source assignment(s): "
            + ", ".join(missing_sources)
        )

    connection.execute(
        configuration.update()
        .where(configuration.c.id == 1)
        .values(payload=payload)
    )

###############################################################################
def downgrade() -> None:
    # Role normalization is intentionally one-way. Downgrading schema does not
    # remove persisted assignments that are valid in both revisions.
    pass
