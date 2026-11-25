# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Add reusable, versioned agents (configuration schema 4).

Creates the agent tables and links sessions, revision runs and timelines to
the agent revision that produced them.

A v3.6.0 database stores its execution settings (provider, role models,
reasoning level, retrieval keys) in the ``application_configuration``
singleton. They are converted here into one default agent and then removed
from the singleton, which keeps only shared settings. A database without a
complete stored configuration gets its default agent at initialization.

The transform is self-contained on purpose: it must not depend on the
evolving domain models.

Revision ID: 202611200001
Revises: 202611010003
Create Date: 2026-11-20
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from typing import Any, Sequence, Union, cast
from uuid import uuid4

from alembic import op
import sqlalchemy as sa

revision: str = "202611200001"
down_revision: Union[str, Sequence[str], None] = "202611010003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_LINKED_TABLES = (
    "clinical_sessions",
    "clinical_session_revision_runs",
    "clinical_session_timelines",
)

_EXECUTION_KEYS = (
    "use_cloud_models",
    "use_cloud_services",
    "cloud_provider",
    "llm_provider",
    "cloud_model",
    "text_extraction_model",
    "clinical_model",
    "revision_model",
    "timeline_model",
    "reasoning_level",
)
_ROLE_KEYS = (
    "text_extraction_model",
    "clinical_model",
    "revision_model",
    "timeline_model",
)
_PROVIDERS = frozenset(
    {"openai", "gemini", "deepseek", "anthropic", "opencode_zen", "opencode_go"}
)
_REASONING_LEVELS = frozenset({"low", "medium", "high"})
_DEFAULT_RETRIEVAL: dict[str, Any] = {
    "evidence_sources": ["livertox"],
    "use_hybrid_search": True,
    "use_reranking": True,
    "reranker_model": "lightweight-balanced-v1",
    "retrieval_candidate_count": 40,
    "retrieval_selected_count": 6,
    "hybrid_vector_weight": 0.65,
    "hybrid_text_weight": 0.35,
}


###############################################################################
def _retrieval(rag: object) -> dict[str, Any]:
    """Stored agent-owned retrieval keys, or the defaults when any is invalid."""

    policy = dict(_DEFAULT_RETRIEVAL)
    if not isinstance(rag, dict):
        return policy
    stored = cast(dict[str, Any], rag)
    for key, default in _DEFAULT_RETRIEVAL.items():
        if key == "evidence_sources" or key not in stored:
            continue
        value: Any = stored[key]
        if isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, int):
            valid = (
                isinstance(value, int) and not isinstance(value, bool) and value >= 1
            )
        elif isinstance(default, float):
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and 0.0 <= value <= 1.0
            )
            value = float(value) if valid else value
        else:
            valid = isinstance(value, str) and 0 < len(value.strip()) <= 255
        if not valid:
            return dict(_DEFAULT_RETRIEVAL)
        policy[key] = value
    if policy["retrieval_candidate_count"] < policy["retrieval_selected_count"]:
        return dict(_DEFAULT_RETRIEVAL)
    return policy


###############################################################################
def agent_config_from_settings(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return the schema 4 agent configuration for one v3.6.0 settings payload."""

    models = {key: str(payload.get(key) or "").strip() for key in _ROLE_KEYS}
    if not all(models.values()) or any(len(model) > 255 for model in models.values()):
        return None

    use_cloud = payload.get("use_cloud_models", payload.get("use_cloud_services"))
    if isinstance(use_cloud, str):
        use_cloud = use_cloud.strip().lower() in {"1", "true", "yes", "on"}
    provider = "ollama"
    if use_cloud:
        provider = (
            str(payload.get("cloud_provider") or payload.get("llm_provider") or "")
            .strip()
            .lower()
        )
        if provider not in _PROVIDERS:
            return None

    level = str(payload.get("reasoning_level") or "").strip().lower()
    reasoning_level = level if level in _REASONING_LEVELS else None

    def ref(key: str) -> dict[str, Any]:
        return {
            "provider": provider,
            "model": models[key],
            "reasoning_level": reasoning_level,
        }

    return {
        "schema_version": 4,
        "assessment": {
            "enabled": True,
            "instructions": {"custom_instructions": ""},
            "report_template": None,
            "models": {
                "extraction": ref("text_extraction_model"),
                "clinical": ref("clinical_model"),
                "extraction_overrides": {},
            },
            "retrieval": _retrieval(payload.get("rag_settings")),
        },
        "revision": {
            "enabled": True,
            "instructions": {"custom_instructions": ""},
            "models": {"planning": ref("revision_model")},
        },
        "timeline": {"enabled": True, "model": ref("timeline_model")},
    }


###############################################################################
def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


###############################################################################
def shared_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """The singleton payload without execution settings or agent-owned keys."""

    shared = {
        key: value
        for key, value in payload.items()
        if key not in _EXECUTION_KEYS and key != "database"
    }
    rag = shared.get("rag_settings")
    if isinstance(rag, dict):
        shared["rag_settings"] = {
            key: value
            for key, value in cast(dict[str, Any], rag).items()
            if key not in _DEFAULT_RETRIEVAL
        }
    return shared


###############################################################################
def _create_tables() -> None:
    op.create_table(
        "agent_definitions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column(
            "description", sa.Text(), server_default=sa.text("''"), nullable=False
        ),
        sa.Column("current_revision", sa.Integer(), nullable=False),
        sa.Column(
            "is_default", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False
        ),
        sa.Column(
            "is_locked", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "is_archived", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_definitions")),
    )
    op.create_index(
        "uq_agent_definitions_single_default",
        "agent_definitions",
        ["is_default"],
        unique=True,
        sqlite_where=sa.text("is_default"),
        postgresql_where=sa.text("is_default"),
    )
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
            "agent_id", "revision_number", name="uq_agent_revisions_agent_revision"
        ),
    )
    op.create_index("ix_agent_revisions_agent_id", "agent_revisions", ["agent_id"])

    for table in _LINKED_TABLES:
        with op.batch_alter_table(table) as batch_op:
            batch_op.add_column(
                sa.Column("agent_revision_id", sa.Integer(), nullable=True)
            )
            batch_op.create_foreign_key(
                op.f(f"fk_{table}_agent_revision_id_agent_revisions"),
                "agent_revisions",
                ["agent_revision_id"],
                ["id"],
            )
            batch_op.create_index(
                f"ix_{table}_agent_revision_id", ["agent_revision_id"]
            )


###############################################################################
def _convert_stored_settings() -> None:
    connection = op.get_bind()
    row = connection.execute(
        sa.text("SELECT payload FROM application_configuration WHERE id = 1")
    ).first()
    if row is None:
        return
    raw: Any = row[0]
    loaded: Any = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(loaded, dict):
        return
    payload = cast(dict[str, Any], loaded)

    config = agent_config_from_settings(payload)
    if config is not None:
        now = datetime.now(UTC).replace(tzinfo=None)
        agent_id = uuid4().hex
        connection.execute(
            sa.text(
                "INSERT INTO agent_definitions (id, name, description, "
                "current_revision, is_default, is_active, is_locked, is_archived, "
                "created_at, updated_at) VALUES (:id, :name, :description, 1, "
                ":yes, :yes, :no, :no, :now, :now)"
            ),
            {
                "id": agent_id,
                "name": "Default agent",
                "description": "Created from the v3.6.0 model settings.",
                "yes": True,
                "no": False,
                "now": now,
            },
        )
        connection.execute(
            sa.text(
                "INSERT INTO agent_revisions (agent_id, revision_number, "
                "schema_version, configuration_json, configuration_hash, created_at) "
                "VALUES (:agent_id, 1, 4, :config, :hash, :now)"
            ).bindparams(sa.bindparam("config", type_=sa.JSON)),
            {
                "agent_id": agent_id,
                "config": config,
                "hash": config_hash(config),
                "now": now,
            },
        )

    connection.execute(
        sa.text(
            "UPDATE application_configuration SET payload = :payload WHERE id = 1"
        ).bindparams(sa.bindparam("payload", type_=sa.JSON)),
        {"payload": shared_settings(payload)},
    )


###############################################################################
def upgrade() -> None:
    _create_tables()
    _convert_stored_settings()


###############################################################################
def downgrade() -> None:
    for table in reversed(_LINKED_TABLES):
        with op.batch_alter_table(table) as batch_op:
            batch_op.drop_index(f"ix_{table}_agent_revision_id")
            batch_op.drop_constraint(
                op.f(f"fk_{table}_agent_revision_id_agent_revisions"),
                type_="foreignkey",
            )
            batch_op.drop_column("agent_revision_id")
    op.drop_index("ix_agent_revisions_agent_id", table_name="agent_revisions")
    op.drop_table("agent_revisions")
    op.drop_index("uq_agent_definitions_single_default", table_name="agent_definitions")
    op.drop_table("agent_definitions")
