from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from repositories.schemas.base import Base

DRUGS_ID_FK = "drugs.id"
CLINICAL_SESSIONS_ID_FK = "clinical_sessions.id"
ACTIVE_SQLITE_WHERE = "is_active = 1"
ACTIVE_POSTGRESQL_WHERE = "is_active = true"

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
    schema_version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )
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
