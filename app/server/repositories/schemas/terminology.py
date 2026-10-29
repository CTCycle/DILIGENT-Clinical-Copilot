# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
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
from sqlalchemy.orm import Mapped, mapped_column

from repositories.schemas.base import Base, JsonPayload

DRUGS_ID_FK = "drugs.id"
IMPORT_RUNS_ID_FK = "terminology_import_runs.id"

###############################################################################
class TerminologyImportRun(Base):
    """Provenance and import report for one terminology source refresh."""

    __tablename__ = "terminology_import_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    source_version: Mapped[str | None] = mapped_column(String, nullable=True)
    license: Mapped[str | None] = mapped_column(Text, nullable=True)
    mode: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    stats_json: Mapped[dict[str, Any] | None] = mapped_column(JsonPayload, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint("mode IN ('full', 'incremental')", name="mode"),
        CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'cancelled', 'skipped')",
            name="status",
        ),
        Index(
            "ix_terminology_import_runs_source_status",
            "source",
            "status",
            "finished_at",
        ),
    )

###############################################################################
class DrugConceptRelation(Base):
    """Typed relation between concepts; related concepts are never merged."""

    __tablename__ = "drug_concept_relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(DRUGS_ID_FK, ondelete="CASCADE"), nullable=False
    )
    related_drug_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(DRUGS_ID_FK, ondelete="CASCADE"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(String, nullable=False)
    position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    import_run_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey(IMPORT_RUNS_ID_FK, ondelete="SET NULL"), nullable=True
    )

    __table_args__ = (
        CheckConstraint(
            "relation_type IN ('form_of', 'has_component')", name="relation_type"
        ),
        UniqueConstraint(
            "drug_id",
            "related_drug_id",
            "relation_type",
            "source",
            name="uq_drug_concept_relations_identity",
        ),
        Index("ix_drug_concept_relations_drug_id", "drug_id"),
        Index("ix_drug_concept_relations_related_drug_id", "related_drug_id"),
    )

###############################################################################
class DrugAtcCode(Base):
    __tablename__ = "drug_atc_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(DRUGS_ID_FK, ondelete="CASCADE"), nullable=False
    )
    atc_code: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    import_run_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey(IMPORT_RUNS_ID_FK, ondelete="SET NULL"), nullable=True
    )

    __table_args__ = (
        UniqueConstraint(
            "drug_id", "atc_code", "source", name="uq_drug_atc_codes_identity"
        ),
        Index("ix_drug_atc_codes_atc_code", "atc_code"),
    )

###############################################################################
class MedicinalProduct(Base):
    """Commercial product, kept separate from its active-ingredient concepts."""

    __tablename__ = "medicinal_products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    name_norm: Mapped[str] = mapped_column(String, nullable=False)
    brand_name_norm: Mapped[str | None] = mapped_column(String, nullable=True)
    country: Mapped[str | None] = mapped_column(String, nullable=True)
    dose_form: Mapped[str | None] = mapped_column(String, nullable=True)
    route: Mapped[str | None] = mapped_column(String, nullable=True)
    composition_status: Mapped[str] = mapped_column(String, nullable=False)
    drug_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey(DRUGS_ID_FK, ondelete="SET NULL"), nullable=True
    )
    source: Mapped[str] = mapped_column(String, nullable=False)
    source_product_id: Mapped[str] = mapped_column(String, nullable=False)
    import_run_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey(IMPORT_RUNS_ID_FK, ondelete="SET NULL"), nullable=True
    )

    __table_args__ = (
        CheckConstraint(
            "composition_status IN ('verified', 'unverified')",
            name="composition_status",
        ),
        UniqueConstraint(
            "source", "source_product_id", name="uq_medicinal_products_source_record"
        ),
        Index("ix_medicinal_products_name_norm", "name_norm"),
        Index("ix_medicinal_products_brand_name_norm", "brand_name_norm"),
        Index("ix_medicinal_products_drug_id", "drug_id"),
    )

###############################################################################
class MedicinalProductIngredient(Base):
    __tablename__ = "medicinal_product_ingredients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("medicinal_products.id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    drug_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey(DRUGS_ID_FK, ondelete="SET NULL"), nullable=True
    )
    ingredient_name: Mapped[str] = mapped_column(Text, nullable=False)
    strength_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    strength_unit: Mapped[str | None] = mapped_column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "product_id",
            "position",
            name="uq_medicinal_product_ingredients_position",
        ),
        Index("ix_medicinal_product_ingredients_drug_id", "drug_id"),
    )

###############################################################################
class TerminologyConflict(Base):
    """Cross-source disagreement recorded instead of silently overwriting."""

    __tablename__ = "terminology_conflicts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    import_run_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey(IMPORT_RUNS_ID_FK, ondelete="SET NULL"), nullable=True
    )
    conflict_type: Mapped[str] = mapped_column(String, nullable=False)
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    existing_drug_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey(DRUGS_ID_FK, ondelete="SET NULL"), nullable=True
    )
    proposed_drug_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey(DRUGS_ID_FK, ondelete="SET NULL"), nullable=True
    )
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("'open'")
    )
    detected_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "conflict_type IN ('identifier_mismatch', 'name_identity_mismatch', "
            "'composition_mismatch')",
            name="conflict_type",
        ),
        CheckConstraint(
            "status IN ('open', 'resolved', 'dismissed')", name="status"
        ),
        Index("ix_terminology_conflicts_status", "status"),
        Index("ix_terminology_conflicts_source", "source"),
    )

###############################################################################
class DrugLiverToxLink(Base):
    """Explicit, derived link from a terminology concept to a LiverTox record."""

    __tablename__ = "drug_livertox_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(DRUGS_ID_FK, ondelete="CASCADE"), nullable=False
    )
    monograph_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("livertox_monographs.id", ondelete="CASCADE"),
        nullable=False,
    )
    link_type: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )

    __table_args__ = (
        CheckConstraint(
            "link_type IN ('monograph_owner', 'verified_alias', 'component_set', "
            "'form_of_parent', 'form_of_child')",
            name="link_type",
        ),
        UniqueConstraint(
            "drug_id", "monograph_id", name="uq_drug_livertox_links_identity"
        ),
        Index("ix_drug_livertox_links_monograph_id", "monograph_id"),
    )
