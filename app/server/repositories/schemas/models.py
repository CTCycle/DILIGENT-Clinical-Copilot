from __future__ import annotations

from datetime import date, datetime
from typing import Any

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
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from repositories.schemas.base import Base

###############################################################################
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
        ForeignKey(PATIENTS_ID_FK, ondelete="CASCADE"),
        nullable=False,
    )
    session_timestamp: Mapped[datetime | None] = mapped_column(DateTime)
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )
    original_session_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey(CLINICAL_SESSIONS_ID_FK, ondelete="CASCADE"),
        nullable=True,
    )
    hepatic_pattern: Mapped[str | None] = mapped_column(String)
    text_extraction_model: Mapped[str | None] = mapped_column(String)
    clinical_model: Mapped[str | None] = mapped_column(String)
    total_duration: Mapped[float | None] = mapped_column(Float)
    session_status: Mapped[str | None] = mapped_column(String, nullable=True)
    session_kind: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

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
    timelines: Mapped[list["ClinicalSessionTimeline"]] = relationship(
        "ClinicalSessionTimeline",
        back_populates="session",
    )
    manual_edits: Mapped[list["ClinicalSessionManualEdit"]] = relationship(
        "ClinicalSessionManualEdit",
        back_populates="session",
        foreign_keys="ClinicalSessionManualEdit.session_id",
    )
    parent_session: Mapped["ClinicalSession | None"] = relationship(
        "ClinicalSession",
        remote_side=[id],
    )

    __table_args__ = (
        Index("ix_clinical_sessions_patient_id", "patient_id"),
        Index("ix_clinical_sessions_original_session_id", "original_session_id"),
        Index("ix_clinical_sessions_timestamp", "session_timestamp"),
        Index("ix_clinical_sessions_status", "session_status"),
    )

###############################################################################
class ClinicalSessionResult(Base):
    __tablename__ = "clinical_session_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(CLINICAL_SESSIONS_ID_FK, ondelete="CASCADE"),
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
class ClinicalSessionTimeline(Base):
    __tablename__ = "clinical_session_timelines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(CLINICAL_SESSIONS_ID_FK, ondelete="CASCADE"),
        nullable=False,
    )
    generated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    generation_status: Mapped[str] = mapped_column(String, nullable=False)
    generation_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_model: Mapped[str | None] = mapped_column(String, nullable=True)
    source_kind: Mapped[str | None] = mapped_column(String, nullable=True)
    model_provider: Mapped[str | None] = mapped_column(String, nullable=True)
    timeline_payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    session: Mapped["ClinicalSession"] = relationship(
        "ClinicalSession",
        back_populates="timelines",
    )

    __table_args__ = (
        CheckConstraint(
            "generation_status IN ('llm_generated', 'fallback')",
            name="ck_clinical_session_timelines_generation_status",
        ),
        CheckConstraint(
            "source_kind IS NULL OR source_kind IN ('local', 'cloud')",
            name="ck_clinical_session_timelines_source_kind",
        ),
        Index("ix_clinical_session_timelines_session_id", "session_id"),
        Index("ix_clinical_session_timelines_generated_at", "generated_at"),
    )

###############################################################################
class ClinicalSessionManualEdit(Base):
    __tablename__ = "clinical_session_manual_edits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(CLINICAL_SESSIONS_ID_FK, ondelete="CASCADE"),
        nullable=False,
    )
    current_version_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("clinical_session_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    edited_by: Mapped[str | None] = mapped_column(String, nullable=True)
    actor_id: Mapped[str | None] = mapped_column(String, nullable=True)
    actor_display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    actor_source: Mapped[str] = mapped_column(String, nullable=False)
    actor_confidence: Mapped[str] = mapped_column(String, nullable=False)
    edited_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    previous_text_hash: Mapped[str] = mapped_column(String, nullable=False)
    new_text_hash: Mapped[str] = mapped_column(String, nullable=False)
    edited_fields_json: Mapped[str] = mapped_column(Text, nullable=False)
    reviewer_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    session: Mapped["ClinicalSession"] = relationship(
        "ClinicalSession",
        back_populates="manual_edits",
        foreign_keys=[session_id],
    )

    __table_args__ = (
        CheckConstraint(
            "actor_source IN ('authenticated_user', 'local_profile', 'manual_entry', 'system', 'unknown')",
            name="ck_clinical_session_manual_edits_actor_source",
        ),
        CheckConstraint(
            "actor_confidence IN ('verified', 'unverified', 'system')",
            name="ck_clinical_session_manual_edits_actor_confidence",
        ),
        Index("ix_clinical_session_manual_edits_session_id", "session_id"),
        Index(
            "ix_clinical_session_manual_edits_current_version_id",
            "current_version_id",
        ),
        Index("ix_clinical_session_manual_edits_edited_at", "edited_at"),
    )

###############################################################################
class ClinicalSessionVersion(Base):
    __tablename__ = "clinical_session_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey(CLINICAL_SESSIONS_ID_FK, ondelete="CASCADE"),
        nullable=True,
    )
    root_session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(CLINICAL_SESSIONS_ID_FK),
        nullable=False,
    )
    source_version_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("clinical_session_versions.id"),
        nullable=True,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    version_status: Mapped[str] = mapped_column(String, nullable=False)
    revision_kind: Mapped[str] = mapped_column(String, nullable=False)
    llm_qa_status: Mapped[str] = mapped_column(String, nullable=False)
    clinical_review_status: Mapped[str] = mapped_column(String, nullable=False)
    pipeline_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    model_configuration_json: Mapped[str | None] = mapped_column(Text, nullable=True)
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
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    session: Mapped["ClinicalSession | None"] = relationship(
        "ClinicalSession",
        foreign_keys=[session_id],
    )
    root_session: Mapped["ClinicalSession"] = relationship(
        "ClinicalSession",
        foreign_keys=[root_session_id],
    )
    source_version: Mapped["ClinicalSessionVersion | None"] = relationship(
        "ClinicalSessionVersion",
        remote_side=[id],
        foreign_keys=[source_version_id],
    )

    __table_args__ = (
        CheckConstraint(
            "version_status IN ('current', 'superseded', 'draft_revision', 'pending_qa', 'qa_failed', 'requires_human_review', 'llm_qa_passed', 'human_approved', 'human_rejected')",
            name="ck_clinical_session_versions_version_status",
        ),
        CheckConstraint(
            "revision_kind IN ('original', 'manual_edit', 'llm_assisted_revision')",
            name="ck_clinical_session_versions_revision_kind",
        ),
        CheckConstraint(
            "llm_qa_status IN ('not_run', 'pending', 'passed', 'passed_with_warnings', 'failed', 'requires_human_review')",
            name="ck_clinical_session_versions_llm_qa_status",
        ),
        CheckConstraint(
            "clinical_review_status IN ('not_reviewed', 'under_review', 'approved_by_human', 'rejected_by_human')",
            name="ck_clinical_session_versions_clinical_review_status",
        ),
        UniqueConstraint(
            "session_id",
            "version_number",
            name="uq_clinical_session_versions_session_version_number",
        ),
        UniqueConstraint(
            "root_session_id",
            "version_number",
            name="uq_clinical_session_versions_root_version_number",
        ),
        Index("ix_clinical_session_versions_root_session_id", "root_session_id"),
        Index("ix_clinical_session_versions_source_version_id", "source_version_id"),
        Index("ix_clinical_session_versions_session_id", "session_id"),
        Index("ix_clinical_session_versions_pipeline_run_id", "pipeline_run_id"),
        Index("ix_clinical_session_versions_status", "version_status"),
    )

###############################################################################
class ClinicalSessionRevisionRun(Base):
    __tablename__ = "clinical_session_revision_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pipeline_run_id: Mapped[str] = mapped_column(String, nullable=False)
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(CLINICAL_SESSIONS_ID_FK),
        nullable=False,
    )
    root_session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(CLINICAL_SESSIONS_ID_FK),
        nullable=False,
    )
    source_version_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("clinical_session_versions.id"),
        nullable=False,
    )
    target_revision_version_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("clinical_session_versions.id"),
        nullable=True,
    )
    revision_mode: Mapped[str] = mapped_column(String, nullable=False)
    revision_kind: Mapped[str] = mapped_column(String, nullable=False)
    configuration_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    initiated_by: Mapped[str | None] = mapped_column(String, nullable=True)
    actor_id: Mapped[str | None] = mapped_column(String, nullable=True)
    actor_display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    actor_source: Mapped[str] = mapped_column(String, nullable=False)
    actor_confidence: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_usage_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String, nullable=True)
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
            "revision_kind IN ('original', 'manual_edit', 'llm_assisted_revision')",
            name="ck_clinical_session_revision_runs_revision_kind",
        ),
        CheckConstraint(
            "actor_source IN ('authenticated_user', 'local_profile', 'manual_entry', 'system', 'unknown')",
            name="ck_clinical_session_revision_runs_actor_source",
        ),
        CheckConstraint(
            "actor_confidence IN ('verified', 'unverified', 'system')",
            name="ck_clinical_session_revision_runs_actor_confidence",
        ),
        UniqueConstraint(
            "pipeline_run_id",
            name="uq_clinical_session_revision_runs_pipeline_run_id",
        ),
        Index("ix_clinical_session_revision_runs_session_id", "session_id"),
        Index("ix_clinical_session_revision_runs_root_session_id", "root_session_id"),
        Index(
            "ix_clinical_session_revision_runs_source_version_id",
            "source_version_id",
        ),
        Index(
            "ix_clinical_session_revision_runs_target_revision_version_id",
            "target_revision_version_id",
        ),
        Index("ix_clinical_session_revision_runs_status", "status"),
        Index("ix_clinical_session_revision_runs_started_at", "started_at"),
    )

###############################################################################
class ClinicalSessionRevisionReview(Base):
    __tablename__ = "clinical_session_revision_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    revision_version_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("clinical_session_versions.id"),
        nullable=False,
    )
    session_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey(CLINICAL_SESSIONS_ID_FK),
        nullable=True,
    )
    clinical_review_status: Mapped[str] = mapped_column(String, nullable=False)
    reviewer_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String, nullable=True)
    actor_id: Mapped[str | None] = mapped_column(String, nullable=True)
    actor_display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    actor_source: Mapped[str] = mapped_column(String, nullable=False)
    actor_confidence: Mapped[str] = mapped_column(String, nullable=False)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
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
        CheckConstraint(
            "clinical_review_status IN ('under_review', 'approved_by_human', 'rejected_by_human')",
            name="ck_clinical_session_revision_reviews_clinical_review_status",
        ),
        CheckConstraint(
            "actor_source IN ('authenticated_user', 'local_profile', 'manual_entry', 'system', 'unknown')",
            name="ck_clinical_session_revision_reviews_actor_source",
        ),
        CheckConstraint(
            "actor_confidence IN ('verified', 'unverified', 'system')",
            name="ck_clinical_session_revision_reviews_actor_confidence",
        ),
        Index(
            "ix_clinical_session_revision_reviews_revision_version_id",
            "revision_version_id",
        ),
        Index("ix_clinical_session_revision_reviews_session_id", "session_id"),
        Index("ix_clinical_session_revision_reviews_reviewed_at", "reviewed_at"),
    )

###############################################################################
class ClinicalSessionRevisionStep(Base):
    __tablename__ = "clinical_session_revision_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    revision_run_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("clinical_session_revision_runs.id", ondelete="CASCADE"),
        nullable=True,
    )
    pipeline_run_id: Mapped[str] = mapped_column(String, nullable=False)
    step_name: Mapped[str] = mapped_column(String, nullable=False)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    step_count: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    input_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    output_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    input_summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    schema_name: Mapped[str | None] = mapped_column(String, nullable=True)
    schema_version: Mapped[str | None] = mapped_column(String, nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String, nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String, nullable=True)
    model_provider: Mapped[str | None] = mapped_column(String, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    token_usage_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    error_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
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
            "pipeline_run_id",
            "step_name",
            "attempt_number",
            name="uq_clinical_session_revision_steps_run_step_attempt",
        ),
        Index("ix_clinical_session_revision_steps_pipeline_run_id", "pipeline_run_id"),
        Index("ix_clinical_session_revision_steps_step_name", "step_name"),
        Index("ix_clinical_session_revision_steps_status", "status"),
    )

###############################################################################
class ClinicalSessionRevisionArtifact(Base):
    __tablename__ = "clinical_session_revision_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    revision_run_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("clinical_session_revision_runs.id", ondelete="CASCADE"),
        nullable=True,
    )
    revision_version_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("clinical_session_versions.id"),
        nullable=False,
    )
    pipeline_run_id: Mapped[str] = mapped_column(String, nullable=False)
    artifact_kind: Mapped[str] = mapped_column(String, nullable=False)
    artifact_key: Mapped[str] = mapped_column(String, nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String, nullable=True)
    entity_name: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str | None] = mapped_column(String, nullable=True)
    schema_version: Mapped[str | None] = mapped_column(String, nullable=True)
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
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
            "artifact_kind IN ('structured_case_entity', 'llm_qa_output', 'report_comparison', 'pipeline_artifact')",
            name="ck_clinical_session_revision_artifacts_artifact_kind",
        ),
        Index(
            "ix_clinical_session_revision_artifacts_revision_version_id",
            "revision_version_id",
        ),
        Index(
            "ix_clinical_session_revision_artifacts_pipeline_run_id",
            "pipeline_run_id",
        ),
        Index(
            "ix_clinical_session_revision_artifacts_artifact_kind",
            "artifact_kind",
        ),
        Index(
            "ix_clinical_session_revision_artifacts_entity_type",
            "entity_type",
        ),
    )

###############################################################################
class ClinicalSessionRevisionEntity(Base):
    __tablename__ = "clinical_session_revision_entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    revision_version_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("clinical_session_versions.id"),
        nullable=False,
    )
    source_version_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("clinical_session_versions.id"),
        nullable=True,
    )
    pipeline_run_id: Mapped[str] = mapped_column(String, nullable=False)
    step_name: Mapped[str] = mapped_column(String, nullable=False)
    entity_type: Mapped[str] = mapped_column(String, nullable=False)
    entity_revision_status: Mapped[str] = mapped_column(String, nullable=False)
    source_section: Mapped[str | None] = mapped_column(String, nullable=True)
    original_entity_id: Mapped[str | None] = mapped_column(String, nullable=True)
    original_name: Mapped[str | None] = mapped_column(String, nullable=True)
    revised_name: Mapped[str | None] = mapped_column(String, nullable=True)
    normalized_name: Mapped[str | None] = mapped_column(String, nullable=True)
    requires_human_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    human_review_status: Mapped[str | None] = mapped_column(String, nullable=True)
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    schema_name: Mapped[str | None] = mapped_column(String, nullable=True)
    schema_version: Mapped[str | None] = mapped_column(String, nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String, nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String, nullable=True)
    model_provider: Mapped[str | None] = mapped_column(String, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    input_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    output_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "entity_type IN ('drug', 'disease', 'lab_timeline_entry', 'livertox_match', 'dili_assessment')",
            name="ck_clinical_session_revision_entities_entity_type",
        ),
        Index("ix_revision_entities_revision_version_id", "revision_version_id"),
        Index("ix_revision_entities_source_version_id", "source_version_id"),
        Index("ix_revision_entities_pipeline_run_id", "pipeline_run_id"),
        Index("ix_revision_entities_entity_type", "entity_type"),
        Index("ix_revision_entities_status", "entity_revision_status"),
        Index("ix_revision_entities_normalized_name", "normalized_name"),
        Index(
            "ix_revision_entities_requires_human_review",
            "requires_human_review",
        ),
    )

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
    session_drugs: Mapped[list["ClinicalSessionDrug"]] = relationship(
        "ClinicalSessionDrug",
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
    drug_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey(DRUGS_ID_FK), nullable=True
    )
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
class ReferenceCatalogSeedRun(Base):
    __tablename__ = "reference_catalog_seed_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    manifest: Mapped[str] = mapped_column(String, nullable=False)
    manifest_version: Mapped[int] = mapped_column(Integer, nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String, nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    seeded_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    entry_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "manifest",
            "manifest_hash",
            "status",
            name="uq_reference_catalog_seed_runs_manifest_hash_status",
        ),
        Index("ix_reference_catalog_seed_runs_manifest", "manifest"),
        Index("ix_reference_catalog_seed_runs_status", "status"),
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
            "provider IN ('openai', 'gemini', 'deepseek', 'anthropic', 'opencode', 'brave')",
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
class ClinicalLabObservation(Base):
    __tablename__ = "clinical_lab_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(CLINICAL_SESSIONS_ID_FK, ondelete="CASCADE"),
        nullable=False,
    )
    marker_code: Mapped[str] = mapped_column(String, nullable=False)
    observation_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    value_numeric: Mapped[float | None] = mapped_column(Float, nullable=True)
    value_text: Mapped[str | None] = mapped_column(String, nullable=True)
    unit: Mapped[str | None] = mapped_column(String, nullable=True)
    upper_limit_numeric: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_ordinal: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        Index(
            "ix_clinical_lab_observations_session_marker",
            "session_id",
            "marker_code",
        ),
        Index("ix_clinical_lab_observations_observation_at", "observation_at"),
    )


###############################################################################
class ClinicalDrugMention(Base):
    __tablename__ = "clinical_drug_mentions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(CLINICAL_SESSIONS_ID_FK, ondelete="CASCADE"),
        nullable=False,
    )
    mention_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_name: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_name: Mapped[str] = mapped_column(String, nullable=False)
    drug_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey(DRUGS_ID_FK, ondelete="SET NULL"),
        nullable=True,
    )
    match_status: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    match_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    evidence_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "mention_ordinal",
            name="uq_clinical_drug_mentions_session_ordinal",
        ),
        Index("ix_clinical_drug_mentions_session_id", "session_id"),
        Index("ix_clinical_drug_mentions_normalized_name", "normalized_name"),
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


###############################################################################
