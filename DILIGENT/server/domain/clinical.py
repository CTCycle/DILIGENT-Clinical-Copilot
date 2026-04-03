from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Comparator = Literal["<=", "<", ">=", ">"]
CONTROL_CHARACTERS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
SAFE_PROVIDER_RE = re.compile(r"^[a-z][a-z0-9-]{1,31}$")
SAFE_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,199}$")
MAX_LAB_TEXT_LENGTH = 20000


###############################################################################
class PatientData(BaseModel):
    """
    Input schema for submitting structured clinical data.
    - Accepts manual clinical sections captured in the GUI.
    - Normalizes whitespace and keeps compat with legacy single-text payloads.

    """
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        None,
        min_length=1,
        max_length=200,
        description="Name of the patient (optional).",
        examples=["Marco Rossi"],
    )
    visit_date: date | None = Field(
        None,
        description="Date of the patient evaluation.",
        examples=[{"day": 15, "month": 1, "year": 2024}],
    )
    anamnesis: str | None = Field(
        None,
        max_length=20000,
        description="Patient anamnesis, including exam findings when provided.",
    )
    drugs: str | None = Field(
        None,
        max_length=20000,
        description="Medication list and dosage notes.",
    )
    laboratory_analysis: str | None = Field(
        None,
        max_length=MAX_LAB_TEXT_LENGTH,
        description="Free-text laboratory section, potentially with dated values and ULN notes.",
    )
    has_hepatic_diseases: bool = Field(
        default=False,
        description="Indicates whether the patient has a history of hepatic diseases.",
    )
    use_rag: bool = Field(
        default=False,
        description="Enables retrieval augmented generation during analysis.",
    )
    use_web_search: bool = Field(
        default=False,
        description="Enables Tavily-backed web evidence retrieval per analyzed drug.",
    )

    # -------------------------------------------------------------------------
    @field_validator(
        "name", "anamnesis", "drugs", "laboratory_analysis", mode="before"
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return cls._strip_optional_text(value)

    # -------------------------------------------------------------------------
    @staticmethod
    def _strip_optional_text(value: Any) -> str | None:
        if value is None:
            return None
        without_controls = CONTROL_CHARACTERS_RE.sub(" ", str(value))
        stripped = without_controls.strip()
        return stripped or None

    # -------------------------------------------------------------------------
    @field_validator("visit_date", mode="before")
    @classmethod
    def coerce_visit_date(cls, value: Any) -> date | None:
        if value is None:
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, Mapping):
            return cls._parse_date_mapping(value)
        if isinstance(value, str):
            return cls._parse_date_string(value)
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _parse_date_mapping(value: Mapping[Any, Any]) -> date | None:
        try:
            day = int(str(value.get("day", "")).strip())
            month = int(str(value.get("month", "")).strip())
            year = int(str(value.get("year", "")).strip())
        except (TypeError, ValueError):
            return None
        try:
            return date(year, month, day)
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _parse_date_string(value: str) -> date | None:
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed_datetime = datetime.fromisoformat(stripped)
        except ValueError:
            try:
                return date.fromisoformat(stripped)
            except ValueError:
                return None
        else:
            return parsed_datetime.date()

    # -------------------------------------------------------------------------
    @field_validator("visit_date")
    @classmethod
    def validate_visit_date(cls, value: date | None) -> date | None:
        if value is None:
            return None
        today = date.today()
        if value > today:
            return today
        return value

    @model_validator(mode="after")
    def require_sections(self) -> "PatientData":
        return self

    # -------------------------------------------------------------------------
    def compose_structured_text(self) -> str | None:
        sections: list[str] = []
        anamnesis_body = self._compose_anamnesis_body()
        if anamnesis_body:
            sections.append(f"# ANAMNESIS\n{anamnesis_body}")
        if self.drugs:
            sections.append(f"# DRUGS\n{self.drugs}")
        if self.laboratory_analysis:
            sections.append(f"# LABORATORY ANALYSIS\n{self.laboratory_analysis}")
        if not sections:
            return None
        return "\n\n".join(section.strip() for section in sections if section.strip())

    # -------------------------------------------------------------------------
    def _compose_anamnesis_body(self) -> str | None:
        lines: list[str] = []
        if self.anamnesis:
            lines.append(self.anamnesis)
        if not lines:
            return None
        body = "\n".join(line for line in lines if line.strip())
        return body or None


###############################################################################
class ClinicalSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, max_length=200)
    visit_date: date | dict[str, int] | str | None = None
    anamnesis: str | None = Field(default=None, max_length=20000)
    has_hepatic_diseases: bool = False
    use_rag: bool = False
    use_web_search: bool = False
    drugs: str | None = Field(default=None, max_length=20000)
    laboratory_analysis: str | None = Field(default=None, max_length=MAX_LAB_TEXT_LENGTH)
    use_cloud_services: bool | None = None
    llm_provider: str | None = Field(default=None, max_length=32)
    cloud_model: str | None = Field(default=None, max_length=200)
    parsing_model: str | None = Field(default=None, max_length=200)
    clinical_model: str | None = Field(default=None, max_length=200)
    ollama_temperature: float | None = None
    cloud_temperature: float | None = None
    ollama_reasoning: bool | None = None

    # -------------------------------------------------------------------------
    @field_validator(
        "name",
        "anamnesis",
        "drugs",
        "laboratory_analysis",
        "llm_provider",
        "cloud_model",
        "parsing_model",
        "clinical_model",
        mode="before",
    )
    @classmethod
    def strip_text_fields(cls, value: Any) -> str | None:
        if value is None:
            return None
        without_controls = CONTROL_CHARACTERS_RE.sub(" ", str(value))
        stripped = without_controls.strip()
        return stripped or None

    # -------------------------------------------------------------------------
    @field_validator("llm_provider")
    @classmethod
    def validate_provider_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not SAFE_PROVIDER_RE.fullmatch(normalized):
            raise ValueError("Invalid provider value")
        return normalized

    # -------------------------------------------------------------------------
    @field_validator("cloud_model", "parsing_model", "clinical_model")
    @classmethod
    def validate_model_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if not SAFE_MODEL_NAME_RE.fullmatch(normalized):
            raise ValueError("Invalid model name")
        return normalized

    # -------------------------------------------------------------------------
    @field_validator("ollama_temperature")
    @classmethod
    def validate_ollama_temperature(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError("ollama_temperature must be finite")
        return round(max(0.0, min(2.0, float(value))), 2)

    # -------------------------------------------------------------------------
    @field_validator("cloud_temperature")
    @classmethod
    def validate_cloud_temperature(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError("cloud_temperature must be finite")
        return round(max(0.0, min(2.0, float(value))), 2)


###############################################################################
class PatientOutputReport(BaseModel):
    report: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Multiline text output with the final report.",
        examples=["This is a sample note."],
    )

    @field_validator("report", mode="before")
    @classmethod
    def strip_report(cls, v: str) -> str:
        if v is None:
            return v
        return str(v).strip()


###############################################################################
class DrugEntry(BaseModel):
    """A single drug prescription extracted from text."""

    name: str = Field(..., description="Drug name as found in the source text.")
    dosage: str | None = Field(None, description="Dosage or concentration details.")
    administration_mode: str | None = Field(
        None, description="Pharmaceutical form or administration mode (e.g., cpr, sir)."
    )
    route: str | None = Field(
        None,
        description="Administration route when explicitly reported (e.g., oral, iv).",
    )
    administration_pattern: str | None = Field(
        None,
        description="Normalized raw administration schedule text when available.",
    )
    daytime_administration: list[float] = Field(
        default_factory=list,
        description="Administration schedule across the day (four slots).",
    )
    suspension_status: bool | None = Field(
        None, description="True if the drug is suspended, False if explicitly active."
    )
    suspension_date: str | None = Field(
        None, description="Suspension date in the original format, if captured."
    )
    therapy_start_status: bool | None = Field(
        None,
        description=(
            "True if the therapy start was explicitly mentioned, False if reported as not started."
        ),
    )
    therapy_start_date: str | None = Field(
        None, description="Therapy start date in the original format, if captured."
    )
    source: Literal["therapy", "anamnesis"] | None = Field(
        default=None,
        description=(
            "Source of the drug entry: 'therapy' for drugs from the medication list, "
            "'anamnesis' for drugs mentioned in clinical history."
        ),
    )
    temporal_classification: Literal["temporal_known", "temporal_uncertain"] | None = (
        Field(
            default=None,
            description="Whether temporal metadata was detected for this drug entry.",
        )
    )
    historical_flag: bool | None = Field(
        default=None,
        description="True for historical/anamnesis mentions, False for active therapy entries.",
    )

    @field_validator("daytime_administration", mode="before")
    @classmethod
    def validate_schedule(cls, value: Any) -> list[float]:
        if value is None:
            return []

        if not isinstance(value, list) or not value:
            return []

        cleaned: list[float] = []
        for slot in value:
            if slot is None:
                continue
            try:
                cleaned.append(float(slot))
            except (TypeError, ValueError):
                continue
        if not cleaned:
            return []
        if len(cleaned) >= 4:
            return cleaned[:4]
        cleaned.extend([0.0] * (4 - len(cleaned)))
        return cleaned


###############################################################################
class PipelineIssue(BaseModel):
    severity: Literal["warning", "error"]
    code: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=500)
    field: str | None = Field(default=None, max_length=100)
    line_index: int | None = Field(default=None, ge=0)
    raw_line: str | None = Field(default=None, max_length=5000)


###############################################################################
class ClinicalPipelineValidationError(Exception):
    def __init__(
        self,
        issues: list[PipelineIssue],
        message: str | None = None,
    ) -> None:
        self.issues = issues
        first_line = message or (
            issues[0].message if issues else "Clinical pipeline validation failed."
        )
        super().__init__(first_line)


# -----------------------------------------------------------------------------
class PatientDrugs(BaseModel):
    """Container for parsed drug entries."""
    entries: list[DrugEntry] = Field(default_factory=list)


###############################################################################
class DiseaseContextEntry(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    occurrence_time: str | None = Field(default=None, max_length=120)
    chronic: bool | None = Field(default=None)
    hepatic_related: bool | None = Field(default=None)
    evidence: str | None = Field(default=None, max_length=500)

    @field_validator("name", "occurrence_time", "evidence", mode="before")
    @classmethod
    def strip_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None


###############################################################################
class PatientDiseaseContext(BaseModel):
    entries: list[DiseaseContextEntry] = Field(default_factory=list)


###############################################################################
class ClinicalLabEntry(BaseModel):
    marker_name: str = Field(..., min_length=1, max_length=40)
    value: float | None = Field(default=None)
    value_text: str | None = Field(default=None, max_length=100)
    unit: str | None = Field(default=None, max_length=50)
    upper_limit_normal: float | None = Field(default=None)
    upper_limit_text: str | None = Field(default=None, max_length=100)
    sample_date: str | None = Field(default=None, max_length=40)
    relative_time: str | None = Field(default=None, max_length=120)
    evidence: str | None = Field(default=None, max_length=500)
    source: Literal["laboratory_analysis", "anamnesis", "merged"] = Field(default="anamnesis")

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
    def strip_optional_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None


###############################################################################
class PatientLabTimeline(BaseModel):
    entries: list[ClinicalLabEntry] = Field(default_factory=list)


###############################################################################
class LiverInjuryOnsetContext(BaseModel):
    onset_date: str | None = Field(default=None, max_length=40)
    onset_basis: Literal[
        "first_symptom",
        "first_abnormal_lab",
        "visit_proxy",
        "unknown",
    ] = Field(default="unknown")
    evidence: str | None = Field(default=None, max_length=500)

    @field_validator("onset_date", "evidence", mode="before")
    @classmethod
    def strip_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None


###############################################################################
class RucamComponentAssessment(BaseModel):
    component_key: str = Field(..., min_length=1, max_length=60)
    label: str = Field(..., min_length=1, max_length=200)
    score: int = Field(default=0)
    status: Literal["scored", "not_assessable", "excluded"] = Field(default="scored")
    evidence: str | None = Field(default=None, max_length=1000)
    rationale: str | None = Field(default=None, max_length=2000)

    @field_validator("component_key", "label", "evidence", "rationale", mode="before")
    @classmethod
    def strip_component_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None


###############################################################################
class DrugRucamAssessment(BaseModel):
    drug_name: str = Field(..., min_length=1, max_length=200)
    injury_type_for_rucam: Literal[
        "hepatocellular",
        "cholestatic",
        "mixed",
        "indeterminate",
    ] = Field(default="indeterminate")
    total_score: int = Field(default=0)
    causality_category: Literal[
        "excluded",
        "unlikely",
        "possible",
        "probable",
        "highly probable",
    ] = Field(default="excluded")
    confidence: Literal["low", "moderate", "high"] = Field(default="low")
    estimated: bool = Field(default=True)
    components: list[RucamComponentAssessment] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    summary: str | None = Field(default=None, max_length=2000)

    @field_validator("drug_name", "summary", mode="before")
    @classmethod
    def strip_rucam_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    @field_validator("limitations", mode="before")
    @classmethod
    def normalize_limitations(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned


###############################################################################
class PatientRucamAssessmentBundle(BaseModel):
    entries: list[DrugRucamAssessment] = Field(default_factory=list)


###############################################################################
class LiverToxMatchInfo(BaseModel):
    nbk_id: str = Field(..., min_length=1, max_length=50)
    matched_name: str = Field(..., min_length=1, max_length=200)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., min_length=1, max_length=50)
    notes: list[str] = Field(default_factory=list)


###############################################################################
class LiverToxBatchMatchItem(BaseModel):
    drug_name: str = Field(..., min_length=1, max_length=200)
    match_name: str | None = Field(default=None, max_length=200)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str | None = Field(default=None, max_length=500)

    @field_validator("drug_name", mode="before")
    @classmethod
    def strip_drug_name(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("drug_name cannot be null")
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("drug_name cannot be empty")
        return cleaned


###############################################################################
class LiverToxBatchMatchSuggestion(BaseModel):
    matches: list[LiverToxBatchMatchItem] = Field(default_factory=list)


###############################################################################
class LiverToxMatchSuggestion(BaseModel):
    match_name: str | None = Field(default=None, max_length=200)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str | None = Field(default=None, max_length=500)


###############################################################################
class HepatotoxicityPatternScore(BaseModel):
    alt_multiple: float | None = Field(
        None,
        description="ALT value divided by its upper reference limit.",
    )
    alp_multiple: float | None = Field(
        None,
        description="ALP value divided by its upper reference limit.",
    )
    r_score: float | None = Field(
        None,
        description="R ratio computed as (ALT multiple) / (ALP multiple).",
    )
    classification: Literal[
        "hepatocellular",
        "cholestatic",
        "mixed",
        "indeterminate",
    ] = Field(
        "indeterminate",
        description="DILI pattern classification derived from the R ratio.",
    )


###############################################################################
class HepatotoxicityPatternAssessment(BaseModel):
    score: HepatotoxicityPatternScore
    status: Literal["ok", "undetermined_due_to_missing_labs"] = "ok"
    issues: list[PipelineIssue] = Field(default_factory=list)


###############################################################################
class DrugToxicityFindings(BaseModel):
    pattern: list[str] = Field(default_factory=list)
    adverse_reactions: list[str] = Field(default_factory=list)

    @field_validator("pattern", "adverse_reactions")
    @classmethod
    def normalize_unique(cls, value: list[str]) -> list[str]:
        unique: dict[str, str] = {}
        for item in value:
            if not item:
                continue
            normalized = item.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key not in unique:
                unique[key] = normalized
        return list(unique.values())


###############################################################################
class DrugHepatotoxicityAnalysis(BaseModel):
    drug_name: str = Field(..., min_length=1, max_length=200)
    source_text: str | None = Field(
        None,
        description="Excerpt from LiverTox or related knowledge base used for the analysis.",
    )
    analysis: DrugToxicityFindings | None = Field(
        None, description="Structured LLM findings for the drug."
    )
    error: str | None = Field(
        None,
        description="Optional error message if the analysis could not be completed.",
    )
    livertox_match: LiverToxMatchInfo | None = Field(
        None,
        description="Metadata about the LiverTox monograph matched for this drug.",
    )

    @model_validator(mode="after")
    def require_result_or_error(self) -> "DrugHepatotoxicityAnalysis":
        if self.analysis is None and not self.error:
            raise ValueError("Either analysis or error must be provided for each drug.")
        return self


###############################################################################
class PatientDrugToxicityBundle(BaseModel):
    entries: list[DrugHepatotoxicityAnalysis] = Field(default_factory=list)


###############################################################################
class DrugSuspensionContext(BaseModel):
    suspended: bool = Field(False)
    suspension_date: date | None = Field(default=None)
    excluded: bool = Field(False)
    note: str | None = Field(default=None)
    interval_days: int | None = Field(
        default=None,
        description=(
            "Difference in days between the clinical visit and suspension date (visit - suspension)."
        ),
    )
    start_reported: bool = Field(
        False,
        description="Indicates whether a therapy start event was detected during parsing.",
    )
    start_date: date | None = Field(
        default=None,
        description="Therapy start date parsed from the clinical notes, if available.",
    )
    start_interval_days: int | None = Field(
        default=None,
        description=(
            "Difference in days between the clinical visit and therapy start (visit - start)."
        ),
    )
    start_note: str | None = Field(
        default=None,
        description="Human-readable summary of the therapy start timing.",
    )


###############################################################################
def create_drug_suspension_context() -> DrugSuspensionContext:
    return DrugSuspensionContext(
        suspended=False,
        suspension_date=None,
        excluded=False,
        note=None,
        interval_days=None,
        start_reported=False,
        start_date=None,
        start_interval_days=None,
        start_note=None,
    )


###############################################################################
class DrugClinicalAssessment(BaseModel):
    drug_name: str = Field(..., min_length=1, max_length=200)
    canonical_name: str | None = Field(default=None, max_length=200)
    origins: list[str] = Field(default_factory=list)
    extraction_metadata: list[dict[str, Any]] = Field(default_factory=list)
    match_status: str | None = Field(default=None, max_length=32)
    matched_livertox_row: dict[str, Any] | None = Field(default=None)
    extracted_excerpts: list[str] = Field(default_factory=list)
    missing_livertox: bool = Field(default=False)
    ambiguous_match: bool = Field(default=False)
    match_candidates: list[str] = Field(default_factory=list)
    suspension: DrugSuspensionContext = Field(
        default_factory=create_drug_suspension_context,
    )
    rucam: DrugRucamAssessment | None = Field(default=None)
    paragraph: str | None = Field(default=None)

    @field_validator("paragraph", mode="before")
    @classmethod
    def strip_paragraph(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None


###############################################################################
class PatientDrugClinicalReport(BaseModel):
    entries: list[DrugClinicalAssessment] = Field(default_factory=list)
    final_report: str | None = Field(default=None)

    @field_validator("final_report", mode="before")
    @classmethod
    def strip_final_report(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None
