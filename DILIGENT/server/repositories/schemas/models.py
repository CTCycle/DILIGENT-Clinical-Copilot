from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()
DRUGS_ID_FK = "drugs.id"
CLINICAL_SESSIONS_ID_FK = "clinical_sessions.id"


###############################################################################
class ClinicalSession(Base):
    __tablename__ = "clinical_sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_name = Column(String)
    session_timestamp = Column(DateTime)
    hepatic_pattern = Column(String)
    parsing_model = Column(String)
    clinical_model = Column(String)
    total_duration = Column(Float)
    sections = relationship("ClinicalSessionSection", back_populates="session")
    labs = relationship("ClinicalSessionLab", back_populates="session")
    drugs = relationship("ClinicalSessionDrug", back_populates="session")
    result_payload = relationship(
        "ClinicalSessionResult",
        back_populates="session",
        uselist=False,
    )
    __table_args__ = (Index("ix_clinical_sessions_timestamp", "session_timestamp"),)


###############################################################################
class ClinicalSessionResult(Base):
    __tablename__ = "clinical_session_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey(CLINICAL_SESSIONS_ID_FK), nullable=False)
    payload_json = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    session = relationship("ClinicalSession", back_populates="result_payload")
    __table_args__ = (
        UniqueConstraint("session_id", name="uq_clinical_session_results_session_id"),
        Index("ix_clinical_session_results_session_id", "session_id"),
    )


###############################################################################
class Drug(Base):
    __tablename__ = "drugs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    canonical_name = Column(Text, nullable=False)
    canonical_name_norm = Column(String, nullable=False)
    rxnorm_rxcui = Column(String, nullable=True)
    livertox_nbk_id = Column(String, nullable=True)
    rxnorm_codes = relationship("DrugRxnormCode", back_populates="drug")
    aliases = relationship("DrugAlias", back_populates="drug")
    monograph = relationship("LiverToxMonograph", back_populates="drug", uselist=False)
    session_drugs = relationship("ClinicalSessionDrug", back_populates="drug")
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
    id = Column(Integer, primary_key=True, autoincrement=True)
    drug_id = Column(Integer, ForeignKey(DRUGS_ID_FK), nullable=False)
    rxcui = Column(String, nullable=False)
    drug = relationship("Drug", back_populates="rxnorm_codes")
    __table_args__ = (
        UniqueConstraint("rxcui", name="uq_drug_rxnorm_codes_rxcui"),
        UniqueConstraint("drug_id", "rxcui", name="uq_drug_rxnorm_codes_identity"),
        Index("ix_drug_rxnorm_codes_drug_id", "drug_id"),
    )


###############################################################################
class DrugAlias(Base):
    __tablename__ = "drug_aliases"
    id = Column(Integer, primary_key=True, autoincrement=True)
    drug_id = Column(Integer, ForeignKey(DRUGS_ID_FK), nullable=False)
    alias = Column(Text, nullable=False)
    alias_norm = Column(String, nullable=False)
    alias_kind = Column(String, nullable=False)
    source = Column(String, nullable=False)
    term_type = Column(String, nullable=True)
    drug = relationship("Drug", back_populates="aliases")
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
    id = Column(Integer, primary_key=True, autoincrement=True)
    drug_id = Column(Integer, ForeignKey(DRUGS_ID_FK), nullable=False, unique=True)
    excerpt = Column(Text)
    likelihood_score = Column(String)
    last_update = Column(String)
    reference_count = Column(Integer)
    year_approved = Column(Integer)
    agent_classification = Column(String)
    primary_classification = Column(String)
    secondary_classification = Column(String)
    include_in_livertox = Column(Boolean)
    source_url = Column(String)
    source_last_modified = Column(String)
    drug = relationship("Drug", back_populates="monograph")
    __table_args__ = (
        UniqueConstraint("drug_id", name="uq_livertox_monographs_drug_id"),
        Index("ix_livertox_monographs_drug_id", "drug_id"),
    )


###############################################################################
class ClinicalSessionSection(Base):
    __tablename__ = "clinical_session_sections"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey(CLINICAL_SESSIONS_ID_FK), nullable=False)
    section_kind = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    session = relationship("ClinicalSession", back_populates="sections")
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
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey(CLINICAL_SESSIONS_ID_FK), nullable=False)
    lab_code = Column(String, nullable=False)
    value_raw = Column(String)
    upper_limit_raw = Column(String)
    session = relationship("ClinicalSession", back_populates="labs")
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
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey(CLINICAL_SESSIONS_ID_FK), nullable=False)
    raw_drug_name = Column(Text, nullable=False)
    raw_drug_name_norm = Column(String, nullable=False)
    drug_id = Column(Integer, ForeignKey(DRUGS_ID_FK), nullable=True)
    match_confidence = Column(Float)
    match_reason = Column(String)
    match_notes = Column(Text)
    session = relationship("ClinicalSession", back_populates="drugs")
    drug = relationship("Drug", back_populates="session_drugs")
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
    id = Column(Integer, primary_key=True, autoincrement=True)
    role_type = Column(
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
    provider = Column(String, nullable=True)
    model_name = Column(String, nullable=True)
    is_active = Column(Boolean, nullable=False, server_default=text("true"))
    created_at = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(
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
            sqlite_where=text("is_active = 1"),
            postgresql_where=text("is_active = true"),
        ),
    )


###############################################################################
class AccessKey(Base):
    __tablename__ = "access_keys"
    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String, nullable=False)
    encrypted_value = Column(Text, nullable=False)
    fingerprint = Column(String, nullable=False)
    is_active = Column(Boolean, nullable=False, server_default=text("false"))
    created_at = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=text("CURRENT_TIMESTAMP"),
    )
    last_used_at = Column(DateTime, nullable=True)
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
    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String, nullable=False, server_default=text("'tavily'"))
    encrypted_value = Column(Text, nullable=False)
    fingerprint = Column(String, nullable=False)
    is_active = Column(Boolean, nullable=False, server_default=text("false"))
    created_at = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=text("CURRENT_TIMESTAMP"),
    )
    last_used_at = Column(DateTime, nullable=True)
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
