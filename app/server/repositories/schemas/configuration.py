# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from repositories.schemas.base import Base, UtcDateTime

DRUGS_ID_FK = "drugs.id"
CLINICAL_SESSIONS_ID_FK = "clinical_sessions.id"
ACTIVE_SQLITE_WHERE = "is_active = 1"
ACTIVE_POSTGRESQL_WHERE = "is_active = true"


###############################################################################
class ReferenceCatalogEntry(Base):
    __tablename__ = "reference_catalog_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    manifest: Mapped[str] = mapped_column(String, nullable=False)
    manifest_version: Mapped[int] = mapped_column(Integer, nullable=False)
    domain: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    key: Mapped[str] = mapped_column(String, nullable=False)
    locale: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'und'")
    )
    value: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_value: Mapped[str] = mapped_column(String, nullable=False)
    priority: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("100")
    )
    match_mode: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'token'")
    )
    case_sensitive: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
    )
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        UniqueConstraint(
            "manifest",
            "domain",
            "category",
            "key",
            "locale",
            "normalized_value",
            name="uq_reference_catalog_entries_identity",
        ),
        Index("ix_reference_catalog_entries_manifest", "manifest"),
        Index(
            "ix_reference_catalog_entries_lookup", "domain", "category", "key", "locale"
        ),
        Index("ix_reference_catalog_entries_active", "active"),
    )


###############################################################################
class ApplicationConfiguration(Base):
    __tablename__ = "application_configuration"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    revision: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=text("CURRENT_TIMESTAMP"),
    )


###############################################################################
class ProviderModelCatalogCache(Base):
    __tablename__ = "provider_model_catalog_cache"

    provider_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    configuration_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)
    models: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_attempt_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    last_attempt_status: Mapped[str] = mapped_column(String(32), nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=text("CURRENT_TIMESTAMP"),
    )


###############################################################################
class ReferenceCatalogManifest(Base):
    __tablename__ = "reference_catalog_manifests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    manifest: Mapped[str] = mapped_column(String, nullable=False)
    installed_version: Mapped[int] = mapped_column(Integer, nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String, nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    entry_count: Mapped[int] = mapped_column(Integer, nullable=False)
    installed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )

    __table_args__ = (
        UniqueConstraint("manifest", name="uq_reference_catalog_manifests_manifest"),
        Index("ix_reference_catalog_manifests_hash", "manifest_hash"),
    )


###############################################################################
def _utc_now() -> datetime:
    return datetime.now(UTC)


###############################################################################
class AgentDefinition(Base):
    __tablename__ = "agent_definitions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str] = mapped_column(
        Text, nullable=False, default="", server_default=text("''")
    )
    current_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    is_default: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("true")
    )
    is_locked: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    is_archived: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), nullable=False, default=_utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), nullable=False, default=_utc_now
    )

    __table_args__ = (
        # At most one default agent on every backend.
        Index(
            "uq_agent_definitions_single_default",
            "is_default",
            unique=True,
            sqlite_where=text("is_default"),
            postgresql_where=text("is_default"),
        ),
    )


###############################################################################
class AgentRevision(Base):
    __tablename__ = "agent_revisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_definitions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    configuration_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    configuration_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), nullable=False, default=_utc_now
    )

    __table_args__ = (
        UniqueConstraint(
            "agent_id", "revision_number", name="uq_agent_revisions_agent_revision"
        ),
        Index("ix_agent_revisions_agent_id", "agent_id"),
    )
