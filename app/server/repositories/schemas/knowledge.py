from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from repositories.schemas.base import Base

DRUGS_ID_FK = "drugs.id"
CLINICAL_SESSIONS_ID_FK = "clinical_sessions.id"
ACTIVE_SQLITE_WHERE = "is_active = 1"
ACTIVE_POSTGRESQL_WHERE = "is_active = true"


###############################################################################
class Drug(Base):
    __tablename__ = "drugs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_name_norm: Mapped[str] = mapped_column(String, nullable=False)
    livertox_nbk_id: Mapped[str | None] = mapped_column(String, nullable=True)
    rxnav_last_update: Mapped[str | None] = mapped_column(String, nullable=True)

    rxnorm_codes: Mapped[list["DrugRxnormCode"]] = relationship(
        "DrugRxnormCode",
        back_populates="drug",
    )
    aliases: Mapped[list["DrugAlias"]] = relationship(
        "DrugAlias",
        back_populates="drug",
    )
    monographs: Mapped[list["LiverToxMonograph"]] = relationship(
        "LiverToxMonograph",
        back_populates="drug",
    )
    kb_match_cache_entries: Mapped[list["KbMatchCache"]] = relationship(
        "KbMatchCache",
        back_populates="drug",
    )

    __table_args__ = (
        UniqueConstraint("canonical_name_norm", name="uq_drugs_canonical_name_norm"),
        Index("ix_drugs_livertox_nbk_id", "livertox_nbk_id"),
    )


###############################################################################
class DrugRxnormCode(Base):
    __tablename__ = "drug_rxnorm_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(DRUGS_ID_FK), nullable=False
    )
    rxcui: Mapped[str] = mapped_column(String, nullable=False)

    drug: Mapped["Drug"] = relationship("Drug", back_populates="rxnorm_codes")

    __table_args__ = (
        UniqueConstraint("rxcui", name="uq_drug_rxnorm_codes_rxcui"),
        UniqueConstraint("drug_id", "rxcui", name="uq_drug_rxnorm_codes_identity"),
        Index("ix_drug_rxnorm_codes_drug_id", "drug_id"),
    )


###############################################################################
class DrugAlias(Base):
    __tablename__ = "drug_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(DRUGS_ID_FK), nullable=False
    )
    alias: Mapped[str] = mapped_column(Text, nullable=False)
    alias_norm: Mapped[str] = mapped_column(String, nullable=False)
    alias_kind: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    term_type: Mapped[str | None] = mapped_column(String, nullable=True)

    drug: Mapped["Drug"] = relationship("Drug", back_populates="aliases")

    __table_args__ = (
        UniqueConstraint(
            "drug_id",
            "alias_norm",
            "alias_kind",
            "source",
            name="uq_drug_aliases_identity",
        ),
        Index("ix_drug_aliases_alias_norm_source", "alias_norm", "source"),
        Index("ix_drug_aliases_drug_id", "drug_id"),
    )


###############################################################################
class LiverToxMonograph(Base):
    __tablename__ = "livertox_monographs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(DRUGS_ID_FK), nullable=False
    )
    monograph_key: Mapped[str] = mapped_column(String, nullable=False)
    drug_name_norm: Mapped[str] = mapped_column(String, nullable=False)
    nbk_id: Mapped[str | None] = mapped_column(String, nullable=True)
    excerpt: Mapped[str | None] = mapped_column(Text)
    likelihood_score: Mapped[str | None] = mapped_column(String)
    last_update: Mapped[str | None] = mapped_column(String)
    reference_count: Mapped[int | None] = mapped_column(Integer)
    year_approved: Mapped[int | None] = mapped_column(Integer)
    agent_classification: Mapped[str | None] = mapped_column(String)
    primary_classification: Mapped[str | None] = mapped_column(String)
    secondary_classification: Mapped[str | None] = mapped_column(String)
    include_in_livertox: Mapped[bool | None] = mapped_column(Boolean)
    source_url: Mapped[str | None] = mapped_column(String)
    source_last_modified: Mapped[str | None] = mapped_column(String)

    drug: Mapped["Drug"] = relationship("Drug", back_populates="monographs")

    __table_args__ = (
        UniqueConstraint("monograph_key", name="uq_livertox_monographs_monograph_key"),
        Index("ix_livertox_monographs_drug_id", "drug_id"),
        Index("ix_livertox_monographs_nbk_id", "nbk_id"),
        Index("ix_livertox_monographs_drug_name_norm", "drug_name_norm"),
    )


###############################################################################
class KbMatchCache(Base):
    __tablename__ = "kb_match_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    raw_drug_name: Mapped[str] = mapped_column(Text, nullable=False)
    raw_drug_name_norm: Mapped[str] = mapped_column(String, nullable=False)
    normalized_drug_key: Mapped[str] = mapped_column(String, nullable=False)
    drug_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey(DRUGS_ID_FK), nullable=True
    )
    rxnorm_rxcui: Mapped[str | None] = mapped_column(String, nullable=True)
    livertox_monograph_key: Mapped[str | None] = mapped_column(String, nullable=True)
    livertox_nbk_id: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    deterministic_evidence_version: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    invalidation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
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

    drug: Mapped["Drug | None"] = relationship(
        "Drug", back_populates="kb_match_cache_entries"
    )

    __table_args__ = (
        CheckConstraint(
            "source IN ('rxnav', 'livertox', 'rag')",
            name="ck_kb_match_cache_source",
        ),
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="ck_kb_match_cache_confidence",
        ),
        UniqueConstraint(
            "normalized_drug_key",
            "source",
            name="uq_kb_match_cache_key_source",
        ),
        Index("ix_kb_match_cache_raw_drug_name_norm", "raw_drug_name_norm"),
        Index("ix_kb_match_cache_normalized_source", "normalized_drug_key", "source"),
        Index("ix_kb_match_cache_drug_id", "drug_id"),
        Index("ix_kb_match_cache_valid", "invalidated_at"),
    )


###############################################################################
class DrugIdentifier(Base):
    __tablename__ = "drug_identifiers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(DRUGS_ID_FK, ondelete="CASCADE"),
        nullable=False,
    )
    identifier_system: Mapped[str] = mapped_column(String, nullable=False)
    identifier_value: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )

    __table_args__ = (
        UniqueConstraint(
            "identifier_system",
            "identifier_value",
            name="uq_drug_identifiers_system_value",
        ),
        Index("ix_drug_identifiers_drug_id", "drug_id"),
    )
