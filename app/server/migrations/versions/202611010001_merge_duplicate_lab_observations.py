# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Merge duplicate laboratory observations.

The same marker, date, value and unit restated in several sections was stored
once per source label. Duplicates are collapsed in the stored lab timeline and
in the structured observation rows; differing sources are kept as ``merged``.

Revision ID: 202611010001
Revises: 202610300001
Create Date: 2026-11-01
"""

from __future__ import annotations

import json
from typing import Any, Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "202611010001"
down_revision: Union[str, Sequence[str], None] = "202610300001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

###############################################################################
def _entry_key(entry: dict[str, Any]) -> tuple[str, str, str, str]:
    value = entry.get("value")
    value_token = f"{value:.6f}" if isinstance(value, (int, float)) else str(
        entry.get("value_text") or ""
    )
    return (
        str(entry.get("marker_name") or "").strip().upper(),
        str(entry.get("sample_date") or "").strip().lower(),
        value_token.strip().lower(),
        str(entry.get("unit") or "").strip().lower(),
    )

###############################################################################
def _merge_timeline(entries: list[Any]) -> list[Any]:
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    ordered: list[Any] = []
    for entry in entries:
        if not isinstance(entry, dict):
            ordered.append(entry)
            continue
        key = _entry_key(entry)
        kept = merged.get(key)
        if kept is None:
            merged[key] = entry
            ordered.append(entry)
        elif kept.get("source") != entry.get("source"):
            kept["source"] = "merged"
            for field in ("evidence", "upper_limit_normal", "upper_limit_text"):
                if kept.get(field) is None and entry.get(field) is not None:
                    kept[field] = entry[field]
    return ordered

###############################################################################
def upgrade() -> None:
    bind = op.get_bind()
    results = bind.execute(
        sa.text("SELECT id, payload_json FROM clinical_session_results")
    ).fetchall()
    for result_id, payload_json in results:
        try:
            payload = json.loads(payload_json)
        except (TypeError, ValueError):
            continue
        timeline = payload.get("lab_timeline") if isinstance(payload, dict) else None
        if not isinstance(timeline, list):
            continue
        merged = _merge_timeline(timeline)
        if len(merged) != len(timeline) or merged != timeline:
            payload["lab_timeline"] = merged
            bind.execute(
                sa.text(
                    "UPDATE clinical_session_results SET payload_json = :payload "
                    "WHERE id = :id"
                ),
                {"payload": json.dumps(payload, ensure_ascii=False), "id": result_id},
            )
    bind.execute(
        sa.text(
            "DELETE FROM clinical_lab_observations WHERE id NOT IN ("
            "SELECT MIN(id) FROM clinical_lab_observations "
            "GROUP BY session_id, marker_code, observation_at, value_text, "
            "upper_limit_numeric, unit)"
        )
    )

###############################################################################
def downgrade() -> None:
    # Duplicate observations are redundant representations and are not restored.
    pass
