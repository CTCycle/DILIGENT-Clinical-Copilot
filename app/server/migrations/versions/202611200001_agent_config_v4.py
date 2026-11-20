# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Agent config schema 4: fixed review pipeline and scripted timeline.

Stored agent revisions are rewritten in place (not discarded):

- Report review instructions become one ``custom_instructions`` field built
  from the former review objectives and editing style.
- Review stage model overrides, execution limits and the tool allow-list are
  removed; the pipeline owns them.
- Timeline instructions are removed; timeline extraction is scripted.
- The capability-level reasoning level is folded into every model reference
  that inherited it, so each model carries its own effective level, and the
  unused per-agent Ollama seed is dropped (the seed is a global setting).

Each rewritten revision gets a recomputed configuration hash so execution
integrity checks keep passing. The transform is self-contained on purpose:
it must not depend on the evolving domain models.

Revision ID: 202611200001
Revises: 202611120002
Create Date: 2026-11-20
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "202611200001"
down_revision: Union[str, Sequence[str], None] = "202611120002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_INSTRUCTION_MAX_LENGTH = 4000


###############################################################################
def _fold_reasoning(capability: dict[str, Any], refs: list[Any]) -> None:
    level = capability.pop("reasoning_level", None)
    capability.pop("ollama_seed", None)
    if not level or level == "off":
        return
    for ref in refs:
        if isinstance(ref, dict) and not ref.get("reasoning_level"):
            ref["reasoning_level"] = level


###############################################################################
def upgrade_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the schema 4 shape of one schema 3 agent configuration."""

    assessment = config["assessment"]
    models = assessment["models"]
    _fold_reasoning(
        assessment,
        [
            models.get("extraction"),
            models.get("clinical"),
            *(models.get("extraction_overrides") or {}).values(),
        ],
    )

    review = config["revision"]
    instructions = review.get("instructions") or {}
    merged = "\n\n".join(
        text.strip()
        for text in (
            str(instructions.get("review_objectives") or ""),
            str(instructions.get("editing_style") or ""),
        )
        if text.strip()
    )
    review["instructions"] = {
        "custom_instructions": merged[:_INSTRUCTION_MAX_LENGTH]
    }
    planning = review["models"].get("planning")
    review["models"] = {"planning": planning}
    review.pop("limits", None)
    _fold_reasoning(review, [planning])

    timeline = config["timeline"]
    timeline.pop("instructions", None)
    _fold_reasoning(timeline, [timeline.get("model")])

    config["schema_version"] = 4
    return config


###############################################################################
def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


###############################################################################
def upgrade() -> None:
    connection = op.get_bind()
    rows = connection.execute(
        sa.text(
            "SELECT id, configuration_json FROM agent_revisions "
            "WHERE schema_version = 3"
        )
    ).fetchall()
    for row_id, raw in rows:
        payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
        config = upgrade_config(payload)
        connection.execute(
            sa.text(
                "UPDATE agent_revisions SET configuration_json = :config, "
                "configuration_hash = :hash, schema_version = 4 WHERE id = :id"
            ).bindparams(sa.bindparam("config", type_=sa.JSON)),
            {
                "config": config,
                "hash": config_hash(config),
                "id": row_id,
            },
        )


###############################################################################
def downgrade() -> None:
    op.execute("UPDATE clinical_sessions SET agent_revision_id = NULL")
    op.execute("UPDATE clinical_session_revision_runs SET agent_revision_id = NULL")
    op.execute("UPDATE clinical_session_timelines SET agent_revision_id = NULL")
    op.execute("DELETE FROM agent_revisions")
    op.execute("DELETE FROM agent_definitions")
