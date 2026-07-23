from __future__ import annotations

import re
from datetime import date as DateValue
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SessionStatus = Literal["successful", "failed"]
DateFilterMode = Literal["before", "after", "exact"]
InspectionUpdateTarget = Literal["rxnav", "livertox", "rag"]
InspectionUpdateJobType = Literal[
    "rxnav_update",
    "livertox_update",
    "rag_update",
]
InspectionJobPhase = Literal[
    "configuration_accepted",
    "update_started",
    "source_data_loading",
    "processing_extraction",
    "persistence_indexing",
    "finalization",
    "completed",
    "cancelled",
    "failed",
]

CONTROL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
MAX_SEARCH_LENGTH = 256


###############################################################################
class SessionCatalogItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: int
    patient_name: str | None = None
    session_timestamp: datetime | None = None
    version: int = 1
    status: SessionStatus
    total_duration: float | None = None
    has_report: bool = False
    has_timeline: bool = False
    can_generate_timeline: bool = False


###############################################################################
class SessionCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[SessionCatalogItem] = Field(default_factory=list)
    total: int
    offset: int
    limit: int


###############################################################################
class SessionDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: int
    patient_name: str | None = None
    visit_date: DateValue | None = None
    session_timestamp: datetime | None = None
    version: int = 1
    status: SessionStatus
    text_extraction_model: str | None = None
    clinical_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    sections: dict[str, str] = Field(default_factory=dict)
    session_text: str = ""
    source_clinical_text: str = ""
    result_payload: dict[str, Any] = Field(default_factory=dict)
    report: str | None = None
    official_report_text: str | None = None
    manual_edit_history: list["ManualReportEditAudit"] = Field(default_factory=list)


###############################################################################
class SessionUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_text: str | None = Field(default=None, max_length=100000)
    report_text: str | None = Field(default=None, max_length=200000)
    edited_fields: list[str] = Field(default_factory=lambda: ["report_text"])
    reviewer_note: str | None = Field(default=None, max_length=2000)
    edited_by: str | None = Field(default=None, max_length=200)
    metadata: dict[str, Any] | None = None


###############################################################################
class SessionRevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selected_text: str | None = Field(default=None, max_length=100000)
    revision_instruction: str | None = Field(default=None, max_length=4000)
    model_overrides: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    max_tasks: int = Field(default=8, ge=1, le=8)
    ###############################################################################
    max_tool_iterations: int = Field(default=24, ge=1, le=24)
    allowed_tools: list[str] | None = Field(default=None, max_length=12)
    revision_goal: Literal[
        "full_report_revision", "selected_text_revision", "metadata_review"
    ] = "full_report_revision"
    dry_run: bool = False


###############################################################################
class ManualReportEditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    report_text: str = Field(..., min_length=1, max_length=200000)
    edited_fields: list[str] = Field(default_factory=lambda: ["report_text"])
    reviewer_note: str | None = Field(default=None, max_length=2000)
    edited_by: str | None = Field(default=None, max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)


###############################################################################
class ManualReportEditAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: int
    current_version_id: int
    edited_by: str | None = None
    actor_id: str | None = None
    actor_display_name: str | None = None
    actor_source: Literal[
        "authenticated_user", "local_profile", "manual_entry", "system", "unknown"
    ]
    actor_confidence: Literal["verified", "unverified", "system"]
    edited_at: datetime
    previous_text_hash: str
    new_text_hash: str
    edited_fields: list[str] = Field(default_factory=list)
    reviewer_note: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


###############################################################################
class ManualReportEditResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session: SessionDetailResponse
    audit: ManualReportEditAudit


###############################################################################
class SessionVersionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version_id: int
    session_id: int | None = None
    root_session_id: int
    source_version_id: int | None = None
    revision_version_id: int
    version_number: int
    version_status: Literal[
        "current",
        "superseded",
        "draft_revision",
        "pending_qa",
        "qa_failed",
        "requires_human_review",
        "llm_qa_passed",
        "human_approved",
        "human_rejected",
    ]
    revision_kind: Literal["original", "manual_edit", "llm_assisted_revision"]
    llm_qa_status: Literal[
        "not_run",
        "pending",
        "passed",
        "passed_with_warnings",
        "failed",
        "requires_human_review",
    ]
    clinical_review_status: Literal[
        "not_reviewed",
        "under_review",
        "approved_by_human",
        "rejected_by_human",
    ]
    pipeline_run_id: str | None = None
    model_configuration: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


###############################################################################
class SessionVersionListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[SessionVersionSummary] = Field(default_factory=list)


###############################################################################
class SessionVersionDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: SessionVersionSummary
    session: SessionDetailResponse | None = None


###############################################################################
class RevisionEntityDiff(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity_type: str
    normalized_name: str | None = None
    source_section: str | None = None
    change_type: Literal[
        "added",
        "removed",
        "corrected",
        "replaced",
        "unresolved",
        "unchanged",
    ]
    summary: str
    requires_human_review: bool = False
    left_entity: dict[str, Any] | None = None
    right_entity: dict[str, Any] | None = None


###############################################################################
class ReportTextDiff(BaseModel):
    model_config = ConfigDict(extra="forbid")
    changed: bool
    left_character_count: int
    right_character_count: int
    left_line_count: int
    right_line_count: int
    similarity_ratio: float
    diff_lines: list[str] = Field(default_factory=list)


###############################################################################
class RevisionQaSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left_llm_qa_status: str
    right_llm_qa_status: str
    left_clinical_review_status: str
    right_clinical_review_status: str
    left_version_status: str
    right_version_status: str
    left_warning_count: int = 0
    right_warning_count: int = 0
    left_blocking_issue_count: int = 0
    right_blocking_issue_count: int = 0
    left_finding_count: int = 0
    right_finding_count: int = 0
    manual_review_required: bool = False


###############################################################################
class SessionVersionComparisonResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left_version: SessionVersionSummary
    right_version: SessionVersionSummary
    added_entities: list[RevisionEntityDiff] = Field(default_factory=list)
    removed_entities: list[RevisionEntityDiff] = Field(default_factory=list)
    corrected_entities: list[RevisionEntityDiff] = Field(default_factory=list)
    replaced_entities: list[RevisionEntityDiff] = Field(default_factory=list)
    unresolved_entities: list[RevisionEntityDiff] = Field(default_factory=list)
    unchanged_entities: list[RevisionEntityDiff] = Field(default_factory=list)
    report_text_diff: ReportTextDiff
    qa_summary: RevisionQaSummary


###############################################################################
class RevisionPipelineRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipeline_run_id: str
    session_id: int
    root_session_id: int
    source_version_id: int
    target_revision_version_id: int | None = None
    revision_mode: str
    revision_kind: Literal["original", "manual_edit", "llm_assisted_revision"]
    configuration: dict[str, Any] = Field(default_factory=dict)
    reviewer_note: str | None = None
    initiated_by: str | None = None
    actor_id: str | None = None
    actor_display_name: str | None = None
    actor_source: Literal[
        "authenticated_user", "local_profile", "manual_entry", "system", "unknown"
    ]
    actor_confidence: Literal["verified", "unverified", "system"]
    started_at: datetime
    completed_at: datetime | None = None
    status: str
    error: dict[str, Any] | None = None
    token_usage: dict[str, Any] | None = None
    latency_ms: int | None = None
    cost_estimate: float | None = None
    trace_id: str | None = None
    created_at: datetime
    updated_at: datetime


###############################################################################
class RevisionPipelineStepResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipeline_run_id: str
    step_name: str
    step_index: int
    step_count: int
    attempt_number: int
    status: str
    input_hash: str | None = None
    output_hash: str | None = None
    input_summary: dict[str, Any] | None = None
    output_summary: dict[str, Any] | None = None
    output_payload: dict[str, Any] | None = None
    schema_name: str | None = None
    schema_version: str | None = None
    prompt_version: str | None = None
    parser_version: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    token_usage: dict[str, Any] | None = None
    latency_ms: int | None = None
    retry_count: int
    error: dict[str, Any] | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    superseded_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


###############################################################################
class RevisionPipelineStepListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[RevisionPipelineStepResponse] = Field(default_factory=list)


###############################################################################
class ReviewerInstructionProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_intent: str | None = None
    main_goal: str | None = None
    instruction_summary: str
    target_sections: list[
        Literal[
            "anamnesis",
            "therapy",
            "labs",
            "livertox_matching",
            "dili_assessment",
            "final_report",
            "qa",
            "unknown",
        ]
    ] = Field(default_factory=list)
    target_entities: list[
        Literal[
            "drugs",
            "diseases",
            "labs",
            "report_wording",
            "source_evidence",
            "matching_errors",
            "causality_reasoning",
            "missing_data",
            "ambiguity_resolution",
            "other",
        ]
    ] = Field(default_factory=list)
    mentioned_drugs: list[str] = Field(default_factory=list)
    mentioned_diseases: list[str] = Field(default_factory=list)
    mentioned_lab_values: list[str] = Field(default_factory=list)
    mentioned_dates: list[str] = Field(default_factory=list)
    extra_data: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    reviewer_assumptions: list[str] = Field(default_factory=list)
    safety_or_quality_concerns: list[str] = Field(default_factory=list)
    prompt_injection_flags: list[str] = Field(default_factory=list)
    pipeline_routing_decision: dict[str, list[str]] = Field(default_factory=dict)


###############################################################################
class ReviewerInstructionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instruction_id: str
    raw_instruction_text: str
    normalized_instruction_summary: str
    routed_pipeline_steps: list[str]
    affected_entities: list[str] = Field(default_factory=list)
    applied: bool
    ignored: bool
    reason_if_ignored: str | None = None
    prompt_injection_detected: bool = False
    prompt_injection_flags: list[str] = Field(default_factory=list)
    evidence_addressed: list[str] = Field(default_factory=list)
    qa_validation_result: str | None = None


###############################################################################
RevisionIssueCategory = Literal[
    "missing_context",
    "mismatched_context",
    "hallucination_risk",
    "ambiguity",
    "unsupported_claim",
    "chronology_gap",
    "tool_needed",
    "other",
]
RevisionIssueSeverity = Literal["low", "medium", "high", "critical"]
RevisionIssueEvidenceStatus = Literal[
    "supported_by_source",
    "missing_from_source",
    "conflicts_with_source",
    "report_only",
    "unclear",
]


###############################################################################
class RevisionToolIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    reason: str
    target: str | None = None
    proposed_inputs: dict[str, Any] = Field(default_factory=dict)


###############################################################################
class RevisionIssueFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: RevisionIssueCategory
    severity: RevisionIssueSeverity
    affected_report_area: str
    evidence_status: RevisionIssueEvidenceStatus
    source_evidence: str | None = None
    missing_evidence_statement: str | None = None
    rationale: str
    recommended_next_action: str
    tool_intents: list[RevisionToolIntent] = Field(default_factory=list)


###############################################################################
class RevisionIssueScanResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    issues: list[RevisionIssueFinding] = Field(default_factory=list)
    tool_intents: list[RevisionToolIntent] = Field(default_factory=list)
    limits: list[str] = Field(default_factory=list)


###############################################################################
class RevisionArtifactResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision_version_id: int
    pipeline_run_id: str
    artifact_kind: Literal[
        "structured_case_entity",
        "llm_qa_output",
        "report_comparison",
        "pipeline_artifact",
    ]
    artifact_key: str | None = None
    entity_type: str | None = None
    entity_name: str | None = None
    status: str | None = None
    schema_version: str | None = None
    payload: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


###############################################################################
class RevisionArtifactListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[RevisionArtifactResponse] = Field(default_factory=list)


###############################################################################
class RevisionEntityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision_version_id: int
    source_version_id: int | None = None
    pipeline_run_id: str
    step_name: str
    entity_type: Literal[
        "drug",
        "disease",
        "lab_timeline_entry",
        "livertox_match",
        "dili_assessment",
    ]
    entity_revision_status: str
    source_section: str | None = None
    original_entity_id: str | None = None
    original_name: str | None = None
    revised_name: str | None = None
    normalized_name: str | None = None
    requires_human_review: bool
    human_review_status: str | None = None
    payload: dict[str, Any] | None = None
    schema_name: str | None = None
    schema_version: str | None = None
    prompt_version: str | None = None
    parser_version: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    created_at: datetime
    superseded_at: datetime | None = None


###############################################################################
class RevisionEntityListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[RevisionEntityResponse] = Field(default_factory=list)


###############################################################################
class RevisionClinicalReviewActionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision_version_id: int
    session_id: int | None = None
    clinical_review_status: Literal[
        "under_review",
        "approved_by_human",
        "rejected_by_human",
    ]
    reviewer_note: str | None = None
    reviewed_by: str | None = None
    actor_id: str | None = None
    actor_display_name: str | None = None
    actor_source: Literal[
        "authenticated_user", "local_profile", "manual_entry", "system", "unknown"
    ]
    actor_confidence: Literal["verified", "unverified", "system"]
    metadata: dict[str, Any] = Field(default_factory=dict)
    reviewed_at: datetime
    created_at: datetime
    updated_at: datetime


###############################################################################
class RevisionClinicalReviewActionListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[RevisionClinicalReviewActionResponse] = Field(default_factory=list)


###############################################################################
class RevisionClinicalReviewUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clinical_review_status: Literal[
        "under_review",
        "approved_by_human",
        "rejected_by_human",
    ]
    reviewer_note: str | None = Field(default=None, max_length=2000)
    reviewed_by: str | None = Field(default=None, max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)


###############################################################################
class RevisionClinicalReviewUpdateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: SessionVersionSummary
    review_action: RevisionClinicalReviewActionResponse


###############################################################################
class RxNavCatalogItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_id: int
    drug_name: str
    last_update: str | None = None


###############################################################################
class RxNavCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[RxNavCatalogItem] = Field(default_factory=list)
    total: int
    offset: int
    limit: int


###############################################################################
class RxNavCatalogUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_name: str = Field(..., min_length=1, max_length=200)

    # -------------------------------------------------------------------------
    @field_validator("drug_name", mode="before")
    @classmethod
    def normalize_drug_name(cls, value: Any) -> str:
        normalized = CONTROL_CHARACTERS_RE.sub(" ", str(value or "")).strip()
        if not normalized:
            raise ValueError("Drug name is required.")
        return normalized


###############################################################################
class DrugAliasEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alias: str
    alias_kind: str


###############################################################################
class DrugAliasGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    aliases: list[DrugAliasEntry] = Field(default_factory=list)


###############################################################################
class DrugAliasesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_id: int
    drug_name: str
    groups: list[DrugAliasGroup] = Field(default_factory=list)


###############################################################################
class LiverToxCatalogItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_id: int
    drug_name: str
    last_update: str | None = None


###############################################################################
class LiverToxCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[LiverToxCatalogItem] = Field(default_factory=list)
    total: int
    offset: int
    limit: int


###############################################################################
class LiverToxExcerptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drug_id: int
    drug_name: str
    excerpt: str
    last_update: str | None = None


###############################################################################
class DeleteEntityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deleted: bool


###############################################################################
class SessionListFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search: str | None = Field(default=None, max_length=MAX_SEARCH_LENGTH)
    status: SessionStatus | None = None
    date_mode: DateFilterMode | None = None
    date: DateValue | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=100)

    # -------------------------------------------------------------------------
    @field_validator("search", mode="before")
    @classmethod
    def normalize_search(cls, value: Any) -> str | None:
        if value is None:
            return None
        normalized = CONTROL_CHARACTERS_RE.sub(" ", str(value)).strip()
        return normalized or None


###############################################################################
class CatalogListFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search: str | None = Field(default=None, max_length=MAX_SEARCH_LENGTH)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=100)

    # -------------------------------------------------------------------------
    @field_validator("search", mode="before")
    @classmethod
    def normalize_search(cls, value: Any) -> str | None:
        if value is None:
            return None
        normalized = CONTROL_CHARACTERS_RE.sub(" ", str(value)).strip()
        return normalized or None


###############################################################################
class InspectionUpdateConfigResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: InspectionUpdateTarget
    defaults: dict[str, Any] = Field(default_factory=dict)
    allowed_fields: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    read_only: bool = False


###############################################################################
class InspectionRxNavOverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rxnav_request_timeout: float | None = Field(default=None, ge=1.0, le=120.0)
    rxnav_max_concurrency: int | None = Field(default=None, ge=1, le=64)


###############################################################################
class InspectionLiverToxOverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    livertox_monograph_max_workers: int | None = Field(default=None, ge=1, le=32)
    livertox_archive: str | None = Field(default=None, max_length=255)
    redownload: bool | None = None

    # -------------------------------------------------------------------------
    @field_validator("livertox_archive")
    @classmethod
    def validate_livertox_archive(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if "/" in normalized or "\\" in normalized:
            raise ValueError("livertox_archive must be a file name only")
        return normalized


###############################################################################
class InspectionRagUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documents_path: str | None = Field(default=None, max_length=1024)


###############################################################################
class RagDocumentListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    file_name: str
    extension: str
    file_size: int
    last_modified: str
    supported_for_ingestion: bool
    vector_model: str | None = None


###############################################################################
class RagDocumentListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[RagDocumentListItem] = Field(default_factory=list)
    total: int
    offset: int = 0
    limit: int = 0


###############################################################################
class LanceVectorStoreSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_documents_path: str
    vector_db_path: str
    collection_name: str
    collection_exists: bool
    embedding_count: int
    distinct_document_count: int
    embedding_dimension: int | None = None
    index_ready: bool
    configured_metric: str | None = None
    configured_index_type: str | None = None
    embedding_model: str = "ibm-granite/granite-embedding-97m-multilingual-r2"
    embedding_revision: str = ""
    index_status: str = "reindex_required"
    embedding_fingerprint: str | None = None
    built_at: str | None = None


###############################################################################
class RagUpdateJobSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    documents: int = 0
    chunks: int = 0
    backend: str = "local"


###############################################################################
class ReferenceCatalogRuntimeObservationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    category: str
    term: str
    replacement: str | None = None
    source: str
    encounter_count: int
    is_active: bool


###############################################################################
class ReferenceCatalogRuntimeObservationUpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    term: str
    replacement: str | None = None
    source: str = "runtime"
    is_active: bool = True


###############################################################################
class RevisionAgentTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    priority: Literal["low", "medium", "high", "critical"]
    objective: str
    affected_sections: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    stop_criteria: str


###############################################################################
class RevisionAgentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instruction_profile: str
    evident_issues: list[str] = Field(default_factory=list)
    tasks: list[RevisionAgentTask] = Field(default_factory=list)
    expected_final_output_type: Literal["revised_report", "review_only"] = (
        "revised_report"
    )


###############################################################################
class RevisionAgentToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(min_length=1, max_length=1000)
    task_complete: bool = False


###############################################################################
class RevisionReportPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    replacement: str = Field(max_length=20000)
    expected_text: str = Field(max_length=20000)
    evidence_references: list[str] = Field(default_factory=list)


###############################################################################
class RevisionDraftResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revised_report_text: str = Field(min_length=1, max_length=200000)
    patches: list[RevisionReportPatch] = Field(default_factory=list)
    changed_sections: list[str] = Field(default_factory=list)
    unchanged_sections: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    human_review_requirements: list[str] = Field(default_factory=list)
    entity_change_proposals: list[dict[str, Any]] = Field(default_factory=list)


###############################################################################
class RevisionAgentQaResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    supported_claim_count: int = Field(default=0, ge=0)
    manual_review_required: bool = True


###############################################################################
class RevisionAgentFinalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pipeline_run_id: str
    revision_version_id: int
    revised_session_id: int | None = None
    revision_status: str
    task_count: int = 0
    tool_call_count: int = 0
    blocking_issue_count: int = 0
    manual_review_required: bool = True
