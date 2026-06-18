from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from domain.clinical.entities import (
    DeterministicDiseaseExtractionResult,
    DeterministicDrugExtractionResult,
    PatientDrugs,
)
from domain.clinical.extras import CandidateSelectionResult

###############################################################################
class RevisedDrugPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(..., min_length=1, max_length=200)
    role: str | None = Field(default=None, max_length=120)
    confidence: str | int | float | None = None
    dosage: str | None = Field(default=None, max_length=200)
    administration_mode: str | None = Field(default=None, max_length=120)
    route: str | None = Field(default=None, max_length=120)
    administration_pattern: str | None = Field(default=None, max_length=200)
    daytime_administration: list[int | float] = Field(default_factory=list)
    suspension_status: bool | None = None
    suspension_date: str | None = Field(default=None, max_length=120)
    therapy_start_status: bool | None = None
    therapy_start_date: str | None = Field(default=None, max_length=120)
    source: Literal["therapy", "anamnesis"] | None = None
    temporal_classification: (
        Literal["temporal_known", "temporal_uncertain"] | None
    ) = None
    historical_flag: bool | None = None

    # -------------------------------------------------------------------------
    @field_validator(
        "name",
        "role",
        "dosage",
        "administration_mode",
        "route",
        "administration_pattern",
        "suspension_date",
        "therapy_start_date",
        mode="before",
    )
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

###############################################################################
class RevisedDiseasePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(..., min_length=1, max_length=200)
    occurrence_time: str | None = Field(default=None, max_length=120)
    timeline: str | None = Field(default=None, max_length=200)
    severity: str | None = Field(default=None, max_length=120)
    diagnosis_status: str | None = Field(default=None, max_length=120)
    symptoms: str | None = Field(default=None, max_length=500)
    clinical_context: str | None = Field(default=None, max_length=500)
    chronic: bool | None = None
    hepatic_related: bool | None = None
    evidence: str | None = Field(default=None, max_length=500)

    # -------------------------------------------------------------------------
    @field_validator(
        "name",
        "occurrence_time",
        "timeline",
        "severity",
        "diagnosis_status",
        "symptoms",
        "clinical_context",
        "evidence",
        mode="before",
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

###############################################################################
class RevisedLabPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    marker_name: str = Field(..., min_length=1, max_length=40)
    value: int | float | None = None
    value_text: str | None = Field(default=None, max_length=100)
    unit: str | None = Field(default=None, max_length=50)
    upper_limit_normal: int | float | None = None
    upper_limit_text: str | None = Field(default=None, max_length=100)
    sample_date: str | None = Field(default=None, max_length=40)
    relative_time: str | None = Field(default=None, max_length=120)
    evidence: str | None = Field(default=None, max_length=500)
    source: Literal["laboratory_analysis", "anamnesis", "merged"] = "anamnesis"

    # -------------------------------------------------------------------------
    @field_validator(
        "marker_name",
        "value_text",
        "unit",
        "upper_limit_text",
        "sample_date",
        "relative_time",
        "evidence",
        mode="before",
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

###############################################################################
class RevisionLiverToxDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    decision_id: str = Field(..., min_length=1, max_length=120)
    drug_name: str = Field(..., min_length=1, max_length=200)
    normalized_drug_name: str | None = Field(default=None, max_length=200)
    decision: str = Field(..., min_length=1, max_length=120)
    decision_reason: str = Field(..., min_length=1, max_length=500)
    match_status: str | None = Field(default=None, max_length=120)
    match_confidence: int | float | None = None
    requires_human_review: bool = False
    reviewer_challenged: bool = False
    source: str = Field(..., min_length=1, max_length=120)
    previous_match_found: bool = False
    previous_match_confidence: int | float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    # -------------------------------------------------------------------------
    @field_validator(
        "decision_id",
        "drug_name",
        "normalized_drug_name",
        "decision",
        "decision_reason",
        "match_status",
        "source",
        mode="before",
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

###############################################################################
class RevisedDiliAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    drug_id: str | None = Field(default=None, max_length=120)
    revised_drug_entry_id: str = Field(..., min_length=1, max_length=120)
    revision_version_id: int
    source_version_id: int
    assessment_version: str = Field(..., min_length=1, max_length=20)
    drug_name: str = Field(..., min_length=1, max_length=200)
    causality_assessment: str = Field(..., min_length=1, max_length=120)
    confidence: str = Field(..., min_length=1, max_length=50)
    evidence_for: list[str] = Field(default_factory=list)
    evidence_against: list[str] = Field(default_factory=list)
    lab_support: list[str] = Field(default_factory=list)
    temporal_support: list[str] = Field(default_factory=list)
    alternative_causes: list[str] = Field(default_factory=list)
    livertox_support: list[str] = Field(default_factory=list)
    changes_from_previous_version: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    previous_assessment_present: bool = False
    provenance: dict[str, Any] = Field(default_factory=dict)

    # -------------------------------------------------------------------------
    @field_validator(
        "drug_id",
        "revised_drug_entry_id",
        "assessment_version",
        "drug_name",
        "causality_assessment",
        "confidence",
        mode="before",
    )
    @classmethod
    def strip_scalar_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

###############################################################################
class RevisionFinalReportPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_text: str
    report_present: bool
    report_character_count: int
    source_excerpt_present: bool
    reviewer_instruction_summary: str | None = None
    comparison_outcome: str | None = None
    changed_focus_areas: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

###############################################################################
class RevisionQaValidationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["passed", "passed_with_warnings", "failed", "requires_human_review"]
    version_status: Literal[
        "llm_qa_passed",
        "qa_failed",
        "requires_human_review",
    ]
    addressed_items: list[str] = Field(default_factory=list)
    unaddressed_items: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    manual_review_required: bool = False
    finding_count: int = 0

###############################################################################
@dataclass(frozen=True)
class RevisionConsultationInputs:
    analysis_drugs: PatientDrugs
    snapshot_context: str | None
    consultation_context: str
    context_metadata: dict[str, Any]

###############################################################################
@dataclass(frozen=True)
class RevisionConsultationExecution:
    inputs: RevisionConsultationInputs
    clinical_session: Any
    final_report: str
    payload: dict[str, Any]

###############################################################################
@dataclass(frozen=True)
class RevisionCandidateSelectionResolution:
    analysis_drugs: PatientDrugs
    candidate_selection: CandidateSelectionResult
    entity_pipeline: dict[str, Any]

###############################################################################
@dataclass(frozen=True)
class RevisionExtractionResolution:
    therapy_deterministic: DeterministicDrugExtractionResult | Any
    anamnesis_deterministic: DeterministicDrugExtractionResult | Any
    disease_deterministic: DeterministicDiseaseExtractionResult | Any
    therapy_drugs: PatientDrugs
    anamnesis_drugs: PatientDrugs
    extraction_bundle: dict[str, Any]

###############################################################################
@dataclass(frozen=True)
class RevisionFinalizationOutputs:
    final_report: str
    generated_report: str
    report_metadata: Any
    faithfulness_audit: Any
    report_comparison_payload: dict[str, Any]
    payload: dict[str, Any]
