from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EvidenceStatus = Literal["excluded", "not_excluded", "unknown", "missing_data"]


###############################################################################
class ClinicalEvidenceQuote(BaseModel):
    claim: str
    quote: str | None = None
    source_section: str | None = None
    event_date: str | None = None
    source_kind: Literal["patient_record", "livertox", "rag", "calculated", "missing"]
    confidence: Literal["low", "moderate", "high"] = "moderate"


###############################################################################
class ClinicalDataCompleteness(BaseModel):
    complete_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    manual_review_required: bool = True
    reasons: list[str] = Field(default_factory=list)


###############################################################################
class DiliCaseQualification(BaseModel):
    status: Literal[
        "meets_typical_detection_criteria",
        "below_typical_detection_criteria",
        "insufficient_data",
    ]
    qualifying_criteria: list[str] = Field(default_factory=list)
    pending_confirmation: list[str] = Field(default_factory=list)
    baseline_date: str | None = None
    baseline_abnormal: bool | None = None
    baseline_multiples: dict[str, float | None] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)
    evidence: list[ClinicalEvidenceQuote] = Field(default_factory=list)


###############################################################################
class DiliTimelineEvent(BaseModel):
    event_type: str
    event_date: str | None = None
    drug_name: str | None = None
    marker: str | None = None
    value: float | None = None
    uln: float | None = None
    evidence: ClinicalEvidenceQuote | None = None


###############################################################################
class DiliTimeline(BaseModel):
    events: list[DiliTimelineEvent] = Field(default_factory=list)
    first_abnormal_liver_test_date: str | None = None
    first_symptom_date: str | None = None
    jaundice_or_bilirubin_rise_date: str | None = None
    peak_dates: dict[str, str | None] = Field(default_factory=dict)
    dechallenge_status: Literal[
        "no_follow_up",
        "improving_after_stop",
        "worsening_after_stop",
        "stable_abnormality",
        "resolved_to_baseline",
        "chronic_or_persistent",
        "insufficient_interval",
    ] = "no_follow_up"
    recovery_date: str | None = None
    last_abnormal_date: str | None = None
    missing_fields: list[str] = Field(default_factory=list)


###############################################################################
class DiliInjuryPattern(BaseModel):
    assessment_point: Literal["first_qualifying", "peak"]
    alt: float | None = None
    alt_uln: float | None = None
    alp: float | None = None
    alp_uln: float | None = None
    r_ratio: float | None = None
    pattern: Literal["hepatocellular", "cholestatic", "mixed", "indeterminate"]
    pattern_source: Literal["calculated", "explicit_text", "unavailable", "conflicting"]
    sample_date: str | None = None
    explicit_pattern: str | None = None
    evidence: list[ClinicalEvidenceQuote] = Field(default_factory=list)


###############################################################################
class DiliPhenotypeAssessment(BaseModel):
    candidates: list[str] = Field(default_factory=list)
    primary_candidate: str | None = None
    deterministic_basis: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    requires_review: bool = True


###############################################################################
class DiliCompetingCause(BaseModel):
    cause: str
    status: EvidenceStatus
    evidence: list[ClinicalEvidenceQuote] = Field(default_factory=list)
    rationale: str


###############################################################################
class DiliDifferentialAssessment(BaseModel):
    causes: list[DiliCompetingCause] = Field(default_factory=list)
    all_major_causes_excluded: bool = False
    unresolved_causes: list[str] = Field(default_factory=list)


###############################################################################
class DiliHysLawAssessment(BaseModel):
    status: Literal["meets_criteria", "possible", "not_met", "not_assessable"]
    aminotransferase_threshold_met: bool | None = None
    bilirubin_threshold_met: bool | None = None
    cholestasis_excluded: bool | None = None
    alternative_causes_excluded: bool | None = None
    exposure_timing_compatible: bool | None = None
    same_episode: bool | None = None
    baseline_aminotransferase_multiple: float | None = None
    baseline_bilirubin_multiple: float | None = None
    initial_cholestasis_present: bool | None = None
    compatible_exposures: list[str] = Field(default_factory=list)
    signal_context: Literal["individual_patient_risk_flag", "clinical_trial_signal"] = (
        "individual_patient_risk_flag"
    )
    evidence: list[ClinicalEvidenceQuote] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)


###############################################################################
class DiliSeverityAssessment(BaseModel):
    grade: Literal[
        "1_mild",
        "2_moderate",
        "3_moderate_severe",
        "4_severe",
        "5_fatal_or_transplant",
        "unassessable",
    ]
    symptom_flag: Literal["S", "A", "unknown"] = "unknown"
    evidence: list[ClinicalEvidenceQuote] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)


###############################################################################
class DiliRucamComponent(BaseModel):
    component: str
    score: int | None = None
    status: Literal["scored", "not_assessable", "excluded"]
    evidence_quote: str | None = None
    evidence_date: str | None = None
    rationale: str | None = None


###############################################################################
class DiliRucamAssessment(BaseModel):
    drug_name: str
    total_score: int | None = None
    category: str
    calculation_method: Literal[
        "source_reported",
        "structured_rucam",
        "not_calculated",
    ] = "not_calculated"
    score_source: str | None = None
    estimated: bool = False
    components: list[DiliRucamComponent] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    is_supportive_not_dispositive: bool = True


###############################################################################
class StructuredCausalityAssessment(BaseModel):
    drug_name: str
    category: Literal["supportive", "limited", "argues_against", "unassessable"]
    temporal_compatibility: str
    dechallenge_rechallenge: str
    drug_signature_concordance: str
    known_hepatotoxic_potential: str
    competing_cause_exclusion: str
    drug_identity_quality: str
    source_evidence_quality: str
    rationale: list[str] = Field(default_factory=list)


###############################################################################
class DiliAcceptanceQuestion(BaseModel):
    question: str
    answer: str
    supporting_evidence: list[ClinicalEvidenceQuote] = Field(default_factory=list)
    missing_data_statement: str | None = None


###############################################################################
class DrugIdentityResolution(BaseModel):
    raw_mention: str
    source_section: str | None = None
    evidence_quote: str | None = None
    normalized_name: str | None = None
    rxnav_candidates: list[Any] = Field(default_factory=list)
    livertox_candidates: list[Any] = Field(default_factory=list)
    accepted_identity: str | None = None
    identity_confidence: float | None = None
    identity_reason: str | None = None
    rejected_candidates: list[Any] = Field(default_factory=list)
    combination_components: list[str] = Field(default_factory=list)
    is_current_exposure: bool = False
    is_historical_exposure: bool = False
    is_negated: bool = False


###############################################################################
class DrugExposureAssessment(BaseModel):
    drug_name: str
    identity: DrugIdentityResolution
    start_date: str | None = None
    dose_changes: list[DiliTimelineEvent] = Field(default_factory=list)
    stop_date: str | None = None
    rechallenge_date: str | None = None
    rechallenge_status: Literal["positive", "present_unclear", "absent", "unknown"] = (
        "unknown"
    )
    livertox_likelihood: str | None = None
    direct_toxin_or_dose_dependent: bool = False
    causality: StructuredCausalityAssessment | None = None
    rucam: DiliRucamAssessment | None = None


###############################################################################
class DiliEvidenceBundle(BaseModel):
    completeness: ClinicalDataCompleteness
    case_qualification: DiliCaseQualification
    timeline: DiliTimeline
    patterns: list[DiliInjuryPattern] = Field(default_factory=list)
    phenotype: DiliPhenotypeAssessment
    differential: DiliDifferentialAssessment
    exposures: list[DrugExposureAssessment] = Field(default_factory=list)
    hys_law: DiliHysLawAssessment
    severity: DiliSeverityAssessment
    evidence: list[ClinicalEvidenceQuote] = Field(default_factory=list)
    acceptance_questions: list[DiliAcceptanceQuestion] = Field(default_factory=list)
    source_hierarchy: list[str] = Field(
        default_factory=lambda: ["AASLD", "LiverTox", "FDA", "DILIN expert method/RUCAM"]
    )
    manual_review_required: bool = True
