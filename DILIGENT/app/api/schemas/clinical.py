from __future__ import annotations

from collections.abc import Mapping
import re
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Comparator = Literal["<=", "<", ">=", ">"]


###############################################################################
class PatientData(BaseModel):
    """
    Input schema for submitting structured clinical data.
    - Accepts manual clinical sections captured in the GUI.
    - Normalizes whitespace and keeps compat with legacy single-text payloads.

    """

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
    alt: str | None = Field(
        None,
        description="ALT laboratory value.",
        examples=["189", "189 U/L"],
    )
    alt_max: str | None = Field(
        None,
        description="Reference maximum for ALT.",
        examples=["47", "47 U/L"],
    )
    alp: str | None = Field(
        None,
        description="ALP laboratory value.",
        examples=["140", "140 U/L"],
    )
    alp_max: str | None = Field(
        None,
        description="Reference maximum for ALP.",
        examples=["150", "150 U/L"],
    )
    has_hepatic_diseases: bool = Field(
        default=False,
        description="Indicates whether the patient has a history of hepatic diseases.",
    )
    use_rag: bool = Field(
        default=False,
        description="Enables retrieval augmented generation during analysis.",
    )

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    @field_validator(
        "anamnesis", "drugs", "alt", "alt_max", "alp", "alp_max", mode="before"
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

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
        if isinstance(value, str):
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
        return None

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
        if any((self.anamnesis, self.drugs)):
            return self
        raise ValueError("Provide at least one clinical section before submitting.")

    @staticmethod
    def coerce_marker(value: str | None) -> tuple[float | None, str | None]:
        if value is None:
            return None, None
        stripped = str(value).strip()
        if not stripped:
            return None, None
        normalized = stripped.replace(",", ".")
        try:
            return float(normalized), None
        except ValueError:
            return None, stripped

    def manual_hepatic_markers(self) -> dict[str, Any]:
        markers: dict[str, Any] = {}
        alt_value, alt_text = self.coerce_marker(self.alt)
        alt_cutoff, alt_cutoff_text = self.coerce_marker(self.alt_max)
        if any((alt_value is not None, alt_text, alt_cutoff, alt_cutoff_text)):
            entry: dict[str, Any] = {
                "value": alt_value,
                "value_text": alt_text,
                "unit": None,
                "date": None,
            }
            if alt_cutoff is not None or alt_cutoff_text is not None:
                entry["cutoff"] = alt_cutoff
                entry["cutoff_text"] = alt_cutoff_text
            markers["ALAT"] = entry
        alp_value, alp_text = self.coerce_marker(self.alp)
        alp_cutoff, alp_cutoff_text = self.coerce_marker(self.alp_max)
        if any((alp_value is not None, alp_text, alp_cutoff, alp_cutoff_text)):
            entry = {
                "value": alp_value,
                "value_text": alp_text,
                "unit": None,
                "date": None,
            }
            if alp_cutoff is not None or alp_cutoff_text is not None:
                entry["cutoff"] = alp_cutoff
                entry["cutoff_text"] = alp_cutoff_text
            markers["ALP"] = entry
        return markers

    def compose_structured_text(self) -> str | None:
        sections: list[str] = []
        anamnesis_lines: list[str] = []
        if self.anamnesis:
            anamnesis_lines.append(self.anamnesis)
        marker_lines: list[str] = []
        if self.alt or self.alt_max:
            alt_tokens: list[str] = []
            if self.alt:
                alt_tokens.append(str(self.alt))
            if self.alt_max:
                alt_tokens.append(f"(max {self.alt_max})")
            marker_lines.append(f"ALAT: {' '.join(alt_tokens).strip()}")
        if self.alp or self.alp_max:
            alp_tokens: list[str] = []
            if self.alp:
                alp_tokens.append(str(self.alp))
            if self.alp_max:
                alp_tokens.append(f"(max {self.alp_max})")
            marker_lines.append(f"ALP: {' '.join(alp_tokens).strip()}")
        if marker_lines:
            cleaned_markers = [line for line in marker_lines if line.strip()]
            if cleaned_markers:
                if anamnesis_lines:
                    anamnesis_lines.append("")
                anamnesis_lines.append("Key laboratory markers:")
                anamnesis_lines.extend(cleaned_markers)
        if anamnesis_lines:
            anamnesis_body = "\n".join(line for line in anamnesis_lines if line.strip())
            if anamnesis_body:
                sections.append(f"# ANAMNESIS\n{anamnesis_body}")
        if self.drugs:
            sections.append(f"# DRUGS\n{self.drugs}")
        if not sections:
            return None
        return "\n\n".join(section.strip() for section in sections if section.strip())


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
class BloodTest(BaseModel):
    """A single blood test result extracted from text."""

    name: str = Field(
        ..., description="Test name exactly as found (minimally normalized)."
    )
    value: float | None = Field(
        None, description="Numeric value if applicable (dot-decimal)."
    )
    value_text: str | None = Field(
        None, description="Raw textual value when not numeric (e.g., '1:80')."
    )
    unit: str | None = Field(None, description="Unit as found, if present.")
    cutoff: float | None = Field(None, description="Cutoff/upper limit if provided.")
    cutoff_unit: str | None = Field(
        None, description="Cutoff unit if specified; often same as unit."
    )
    note: str | None = Field(
        None, description="Parenthetical note not related to cutoff."
    )
    context_date: str | None = Field(
        None,
        description="ISO YYYY-MM-DD if parsed, else original date string for this batch.",
    )

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        v = re.sub(r"\s+", " ", v.strip())
        return v.rstrip(",:;.- ")


# -----------------------------------------------------------------------------
class PatientBloodTests(BaseModel):
    """Container for parsed blood test entries."""

    entries: list[BloodTest] = Field(default_factory=list)


###############################################################################
class DrugEntry(BaseModel):
    """A single drug prescription extracted from text."""

    name: str = Field(..., description="Drug name as found in the source text.")
    dosage: str | None = Field(None, description="Dosage or concentration details.")
    administration_mode: str | None = Field(
        None, description="Pharmaceutical form or administration mode (e.g., cpr, sir)."
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

    @field_validator("daytime_administration")
    @classmethod
    def validate_schedule(cls, value: list[float]) -> list[float]:
        if not value:
            return []

        cleaned = [float(slot) for slot in value if slot is not None]
        if len(cleaned) >= 4:
            return cleaned[:4]

        return []


# -----------------------------------------------------------------------------
class PatientDrugs(BaseModel):
    """Container for parsed drug entries."""

    entries: list[DrugEntry] = Field(default_factory=list)


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
class DrugClinicalAssessment(BaseModel):
    drug_name: str = Field(..., min_length=1, max_length=200)
    matched_livertox_row: dict[str, Any] | None = Field(default=None)
    extracted_excerpts: list[str] = Field(default_factory=list)
    suspension: DrugSuspensionContext = Field(
        default_factory=DrugSuspensionContext,
    )
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
