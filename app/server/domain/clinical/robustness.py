from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SpanKind = Literal["raw", "clean"]
DocumentBlockType = Literal["clinical_content", "administrative", "bibliography"]
FactNodeFamily = Literal[
    "drug_exposure",
    "lab_event",
    "clinical_event",
    "causality_statement",
    "dili_pattern_statement",
    "recommendation_statement",
]
FactOrigin = Literal["source_verbatim", "derived"]
AuditOutcome = Literal[
    "faithful",
    "mostly_faithful_with_minor_issues",
    "partially_faithful_with_major_issues",
    "not_faithful",
    "comparison_not_possible",
]
GateSeverity = Literal["blocking", "non_blocking"]


class SourceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    span_id: str
    kind: SpanKind = "raw"
    page: int | None = None
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    text: str = Field(default="", max_length=5000)


class SpanMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_start: int = Field(..., ge=0)
    raw_end: int = Field(..., ge=0)
    clean_start: int = Field(..., ge=0)
    clean_end: int = Field(..., ge=0)


class NormalizedDocumentBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str
    block_type: DocumentBlockType
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_spans: list[SourceSpan] = Field(default_factory=list)


class NormalizedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    raw_text: str
    clean_text: str
    span_mappings: list[SpanMapping] = Field(default_factory=list)
    blocks: list[NormalizedDocumentBlock] = Field(default_factory=list)


class ExtractedSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_spans: list[SourceSpan] = Field(default_factory=list)
    missing: bool = False
    issues: list[str] = Field(default_factory=list)


class ContaminationFlags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    therapy_contaminated_by_bibliography_or_admin: bool = False
    assessment_contaminated_by_non_clinical_content: bool = False
    labs_embedded_without_dedicated_lab_section: bool = False


class TimedDrugMention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drug: str
    timing_type: str
    timing_value: str | None = None
    status: str = "uncertain"
    source_span: SourceSpan | None = None


class ExtractionArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    sections: dict[str, ExtractedSection] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    contamination_flags: ContaminationFlags = Field(default_factory=ContaminationFlags)
    timed_drugs: list[TimedDrugMention] = Field(default_factory=list)
    extraction_issues: list[dict[str, Any]] = Field(default_factory=list)


class FactGraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    family: FactNodeFamily
    value: dict[str, Any]
    source_spans: list[SourceSpan] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    origin: FactOrigin
    supports: list[str] = Field(default_factory=list)


class FactGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    nodes: list[FactGraphNode] = Field(default_factory=list)


class FactGraphValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    hard_issues: list[dict[str, Any]] = Field(default_factory=list)
    soft_issues: list[dict[str, Any]] = Field(default_factory=list)


class ReportMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    report_mode: str
    patient_name_source: Literal["ui_metadata"] = "ui_metadata"
    report_date_source: Literal["ui_metadata"] = "ui_metadata"
    claim_references: dict[str, list[str]] = Field(default_factory=dict)


class FaithfulnessAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    outcome: AuditOutcome
    manual_review_required: bool = False
    blocking_issues: list[dict[str, Any]] = Field(default_factory=list)
    non_blocking_issues: list[dict[str, Any]] = Field(default_factory=list)
    gate_decisions: list[dict[str, Any]] = Field(default_factory=list)
    discrepancy_report: str = ""


class RunBundleIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    run_id: str
    session_id: int | None = None
    storage: Literal["database_session_result_payload"] = "database_session_result_payload"
    artifacts: dict[str, str] = Field(default_factory=dict)


class ClinicalInputPreflightIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: GateSeverity
    code: str
    message: str
    field: str | None = None


class ClinicalInputPreflightResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool
    blocking_issues: list[ClinicalInputPreflightIssue] = Field(default_factory=list)
    non_blocking_issues: list[ClinicalInputPreflightIssue] = Field(default_factory=list)
    runtime_settings: dict[str, Any] = Field(default_factory=dict)
    extraction_quality: dict[str, Any] = Field(default_factory=dict)

