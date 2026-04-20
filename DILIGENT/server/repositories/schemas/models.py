from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


DRUGS_ID_FK = "drugs.id"
CLINICAL_SESSIONS_ID_FK = "clinical_sessions.id"
PATIENTS_ID_FK = "patients.id"
ACTIVE_SQLITE_WHERE = "is_active = 1"
ACTIVE_POSTGRESQL_WHERE = "is_active = true"


###############################################################################
class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str | None] = mapped_column(String)
    visit_date: Mapped[date | None] = mapped_column(Date)
    anamnesis: Mapped[str | None] = mapped_column(Text)
    drugs: Mapped[str | None] = mapped_column(Text)
    laboratory_analysis: Mapped[str | None] = mapped_column(Text)
    image_blob: Mapped[bytes | None] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    sessions: Mapped[list["ClinicalSession"]] = relationship(
        "ClinicalSession",
        back_populates="patient",
    )

    __table_args__ = (
        Index("ix_patients_name", "name"),
        Index("ix_patients_visit_date", "visit_date"),
    )


###############################################################################
class ClinicalSession(Base):
    __tablename__ = "clinical_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(PATIENTS_ID_FK),
        nullable=False,
    )
    session_timestamp: Mapped[datetime | None] = mapped_column(DateTime)
    hepatic_pattern: Mapped[str | None] = mapped_column(String)
    text_extraction_model: Mapped[str | None] = mapped_column(String)
    clinical_model: Mapped[str | None] = mapped_column(String)
    total_duration: Mapped[float | None] = mapped_column(Float)
    session_status: Mapped[str | None] = mapped_column(String, nullable=True)

    patient: Mapped["Patient"] = relationship(
        "Patient",
        back_populates="sessions",
    )
    sections: Mapped[list["ClinicalSessionSection"]] = relationship(
        "ClinicalSessionSection",
        back_populates="session",
    )
    labs: Mapped[list["ClinicalSessionLab"]] = relationship(
        "ClinicalSessionLab",
        back_populates="session",
    )
    drugs: Mapped[list["ClinicalSessionDrug"]] = relationship(
        "ClinicalSessionDrug",
        back_populates="session",
    )
    result_payload: Mapped["ClinicalSessionResult | None"] = relationship(
        "ClinicalSessionResult",
        back_populates="session",
        uselist=False,
    )

    __table_args__ = (
        Index("ix_clinical_sessions_patient_id", "patient_id"),
        Index("ix_clinical_sessions_timestamp", "session_timestamp"),
        Index("ix_clinical_sessions_status", "session_status"),
    )


###############################################################################
class ClinicalSessionResult(Base):
    __tablename__ = "clinical_session_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(CLINICAL_SESSIONS_ID_FK),
        nullable=False,
    )
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    session: Mapped["ClinicalSession"] = relationship(
        "ClinicalSession",
        back_populates="result_payload",
    )

    __table_args__ = (
        UniqueConstraint("session_id", name="uq_clinical_session_results_session_id"),
        Index("ix_clinical_session_results_session_id", "session_id"),
    )


###############################################################################
class Drug(Base):
    __tablename__ = "drugs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_name_norm: Mapped[str] = mapped_column(String, nullable=False)
    rxnorm_rxcui: Mapped[str | None] = mapped_column(String, nullable=True)
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
    monograph: Mapped["LiverToxMonograph | None"] = relationship(
        "LiverToxMonograph",
        back_populates="drug",
        uselist=False,
    )
    dili_annotations: Mapped[list["DrugDiliAnnotation"]] = relationship(
        "DrugDiliAnnotation",
        back_populates="drug",
    )
    label_documents: Mapped[list["DrugLabelDocument"]] = relationship(
        "DrugLabelDocument",
        back_populates="drug",
    )
    session_drugs: Mapped[list["ClinicalSessionDrug"]] = relationship(
        "ClinicalSessionDrug",
        back_populates="drug",
    )

    __table_args__ = (
        UniqueConstraint("canonical_name_norm", name="uq_drugs_canonical_name_norm"),
        UniqueConstraint("rxnorm_rxcui", name="uq_drugs_rxnorm_rxcui"),
        UniqueConstraint("livertox_nbk_id", name="uq_drugs_livertox_nbk_id"),
        Index("ix_drugs_rxnorm_rxcui", "rxnorm_rxcui"),
        Index("ix_drugs_livertox_nbk_id", "livertox_nbk_id"),
    )


###############################################################################
class DrugRxnormCode(Base):
    __tablename__ = "drug_rxnorm_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int] = mapped_column(Integer, ForeignKey(DRUGS_ID_FK), nullable=False)
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
    drug_id: Mapped[int] = mapped_column(Integer, ForeignKey(DRUGS_ID_FK), nullable=False)
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
        Integer,
        ForeignKey(DRUGS_ID_FK),
        nullable=False,
        unique=True,
    )
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

    drug: Mapped["Drug"] = relationship("Drug", back_populates="monograph")

    __table_args__ = (
        UniqueConstraint("drug_id", name="uq_livertox_monographs_drug_id"),
        Index("ix_livertox_monographs_drug_id", "drug_id"),
    )


###############################################################################
class DrugDiliAnnotation(Base):
    __tablename__ = "drug_dili_annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int | None] = mapped_column(Integer, ForeignKey(DRUGS_ID_FK), nullable=True)
    source_dataset: Mapped[str] = mapped_column(String, nullable=False)
    source_record_id: Mapped[str] = mapped_column(String, nullable=False)
    source_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_name_norm: Mapped[str | None] = mapped_column(String, nullable=True)
    classification: Mapped[str | None] = mapped_column(String, nullable=True)
    severity_class: Mapped[str | None] = mapped_column(String, nullable=True)
    concern_class: Mapped[str | None] = mapped_column(String, nullable=True)
    label_section: Mapped[str | None] = mapped_column(String, nullable=True)
    routes: Mapped[str | None] = mapped_column(Text, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str | None] = mapped_column(String, nullable=True)
    source_last_modified: Mapped[str | None] = mapped_column(String, nullable=True)

    drug: Mapped["Drug | None"] = relationship("Drug", back_populates="dili_annotations")

    __table_args__ = (
        UniqueConstraint(
            "source_dataset",
            "source_record_id",
            name="uq_drug_dili_annotations_source_identity",
        ),
        Index("ix_drug_dili_annotations_drug_id", "drug_id"),
        Index(
            "ix_drug_dili_annotations_source_name_norm",
            "source_dataset",
            "source_name_norm",
        ),
    )


###############################################################################
class DrugLabelDocument(Base):
    __tablename__ = "drug_label_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drug_id: Mapped[int] = mapped_column(Integer, ForeignKey(DRUGS_ID_FK), nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    set_id: Mapped[str] = mapped_column(String, nullable=False)
    spl_version: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    labeler: Mapped[str | None] = mapped_column(Text, nullable=True)
    effective_date: Mapped[str | None] = mapped_column(String, nullable=True)
    source_url: Mapped[str | None] = mapped_column(String, nullable=True)
    source_last_modified: Mapped[str | None] = mapped_column(String, nullable=True)

    drug: Mapped["Drug"] = relationship("Drug", back_populates="label_documents")
    sections: Mapped[list["DrugLabelSection"]] = relationship(
        "DrugLabelSection",
        back_populates="document",
    )

    __table_args__ = (
        UniqueConstraint(
            "source",
            "set_id",
            "spl_version",
            name="uq_drug_label_documents_source_identity",
        ),
        Index("ix_drug_label_documents_drug_id", "drug_id"),
        Index("ix_drug_label_documents_source_set_id", "source", "set_id"),
    )


###############################################################################
class DrugLabelSection(Base):
    __tablename__ = "drug_label_sections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("drug_label_documents.id"),
        nullable=False,
    )
    section_key: Mapped[str] = mapped_column(String, nullable=False)
    section_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    contains_hepatic_keywords: Mapped[bool] = mapped_column(Boolean, nullable=False)
    display_order: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    document: Mapped["DrugLabelDocument"] = relationship(
        "DrugLabelDocument",
        back_populates="sections",
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "section_key",
            name="uq_drug_label_sections_document_key",
        ),
        Index("ix_drug_label_sections_document_id", "document_id"),
        Index(
            "ix_drug_label_sections_key_contains_hepatic",
            "section_key",
            "contains_hepatic_keywords",
        ),
    )


###############################################################################
class ClinicalSessionSection(Base):
    __tablename__ = "clinical_session_sections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(CLINICAL_SESSIONS_ID_FK),
        nullable=False,
    )
    section_kind: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    session: Mapped["ClinicalSession"] = relationship(
        "ClinicalSession",
        back_populates="sections",
    )

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "section_kind",
            name="uq_clinical_session_sections_identity",
        ),
        Index("ix_clinical_session_sections_session_id", "session_id"),
    )


###############################################################################
class ClinicalSessionLab(Base):
    __tablename__ = "clinical_session_labs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(CLINICAL_SESSIONS_ID_FK),
        nullable=False,
    )
    lab_code: Mapped[str] = mapped_column(String, nullable=False)
    value_raw: Mapped[str | None] = mapped_column(String)
    upper_limit_raw: Mapped[str | None] = mapped_column(String)

    session: Mapped["ClinicalSession"] = relationship(
        "ClinicalSession",
        back_populates="labs",
    )

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "lab_code",
            name="uq_clinical_session_labs_identity",
        ),
        Index("ix_clinical_session_labs_session_id", "session_id"),
    )


###############################################################################
class ClinicalSessionDrug(Base):
    __tablename__ = "clinical_session_drugs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(CLINICAL_SESSIONS_ID_FK),
        nullable=False,
    )
    raw_drug_name: Mapped[str] = mapped_column(Text, nullable=False)
    raw_drug_name_norm: Mapped[str] = mapped_column(String, nullable=False)
    drug_id: Mapped[int | None] = mapped_column(Integer, ForeignKey(DRUGS_ID_FK), nullable=True)
    match_confidence: Mapped[float | None] = mapped_column(Float)
    match_reason: Mapped[str | None] = mapped_column(String)
    match_notes: Mapped[str | None] = mapped_column(Text)

    session: Mapped["ClinicalSession"] = relationship(
        "ClinicalSession",
        back_populates="drugs",
    )
    drug: Mapped["Drug | None"] = relationship("Drug", back_populates="session_drugs")

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "raw_drug_name_norm",
            name="uq_clinical_session_drugs_identity",
        ),
        Index("ix_clinical_session_drugs_session_id", "session_id"),
        Index("ix_clinical_session_drugs_drug_id", "drug_id"),
        Index("ix_clinical_session_drugs_raw_drug_name_norm", "raw_drug_name_norm"),
    )


###############################################################################
class ModelSelection(Base):
    __tablename__ = "model_selections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role_type: Mapped[str] = mapped_column(
        Enum(
            "clinical",
            "text_extraction",
            "cloud",
            name="model_role_type",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
        ),
        nullable=False,
    )
    provider: Mapped[str | None] = mapped_column(String, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(
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
        UniqueConstraint("role_type", name="uq_model_selections_role_type"),
        Index("ix_model_selections_role_type", "role_type"),
        Index(
            "uq_model_selections_active_role_type",
            "role_type",
            unique=True,
            sqlite_where=text(ACTIVE_SQLITE_WHERE),
            postgresql_where=text(ACTIVE_POSTGRESQL_WHERE),
        ),
    )


###############################################################################
class RuntimeSetting(Base):
    __tablename__ = "runtime_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    setting_key: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    setting_value: Mapped[str | None] = mapped_column(String, nullable=True)
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
        UniqueConstraint("setting_key", name="uq_runtime_settings_setting_key"),
        Index("ix_runtime_settings_setting_key", "setting_key"),
    )


###############################################################################
class AccessKeyEncryptionMaterial(Base):
    __tablename__ = "access_key_encryption_materials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key_purpose: Mapped[str] = mapped_column(String, nullable=False)
    key_version: Mapped[int] = mapped_column(Integer, nullable=False)
    key_material: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
    )
    seeded_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deactivated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
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
        CheckConstraint(
            "key_purpose IN ('provider_access_keys')",
            name="ck_access_key_encryption_materials_key_purpose",
        ),
        UniqueConstraint(
            "key_purpose",
            "key_version",
            name="uq_access_key_encryption_materials_purpose_version",
        ),
        Index(
            "uq_access_key_encryption_materials_active_purpose",
            "key_purpose",
            unique=True,
            sqlite_where=text("is_active = 1"),
            postgresql_where=text("is_active = true"),
        ),
        Index(
            "ix_access_key_encryption_materials_purpose_version",
            "key_purpose",
            "key_version",
        ),
    )


###############################################################################
class AccessKey(Base):
    __tablename__ = "access_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    encrypted_value: Mapped[str] = mapped_column(Text, nullable=False)
    encryption_key_version: Mapped[int] = mapped_column(Integer, nullable=False)
    fingerprint: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
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
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "provider IN ('openai', 'gemini')",
            name="ck_access_keys_provider",
        ),
        Index("ix_access_keys_provider", "provider"),
        Index(
            "uq_access_keys_active_provider",
            "provider",
            unique=True,
            sqlite_where=text("is_active = 1"),
            postgresql_where=text("is_active = true"),
        ),
    )


###############################################################################
class ResearchAccessKey(Base):
    __tablename__ = "research_access_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(
        String,
        nullable=False,
        server_default=text("'tavily'"),
    )
    encrypted_value: Mapped[str] = mapped_column(Text, nullable=False)
    encryption_key_version: Mapped[int] = mapped_column(Integer, nullable=False)
    fingerprint: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("false"),
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
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "provider = 'tavily'",
            name="ck_research_access_keys_provider",
        ),
        Index("ix_research_access_keys_provider", "provider"),
        Index(
            "uq_research_access_keys_active_provider",
            "provider",
            unique=True,
            sqlite_where=text("is_active = 1"),
            postgresql_where=text("is_active = true"),
        ),
    )
