from __future__ import annotations

import re
from typing import Literal, Any

from pydantic import BaseModel, Field, field_validator, model_validator

from Pharmagent.app.api.schemas.placeholders import EXAMPLE_INPUT_DATA

Comparator = Literal["<=", "<", ">=", ">"]


###############################################################################
class PatientData(BaseModel):
    """
    Input schema for submitting structured clinical data.
    - Accepts manual clinical sections or falls back to file-based loading.
    - Normalizes whitespace and keeps compat with legacy single-text payloads.
    """

    name: str | None = Field(
        None,
        min_length=1,
        max_length=200,
        description="Name of the patient (optional).",
        examples=["Marco Rossi"],
    )

    info: str | None = Field(
        None,
        min_length=1,
        max_length=20000,
        description="Legacy multiline text input with patient's info."        
    )

    anamnesis: str | None = Field(
        None,
        max_length=20000,
        description="Patient anamnesis notes.",
    )
    drugs: str | None = Field(
        None,
        max_length=20000,
        description="Medication list and dosage notes.",
    )
    exams: str | None = Field(
        None,
        max_length=20000,
        description="Blood and instrumental exam notes.",
    )
    alt: str | None = Field(
        None,
        description="ALT/ALAT laboratory value.",
        examples=["189", "189 U/L"],
    )
    alp: str | None = Field(
        None,
        description="ALP laboratory value.",
        examples=["140", "140 U/L"],
    )
    flags: list[str] = Field(
        default_factory=list, description="Additional boolean options from the UI."
    )
    from_files: bool = Field(
        False,
        description=(
            "If true, ignore manual sections and load all .txt files from default path."
        ),
        examples=[False],
    )

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    @field_validator("info", "anamnesis", "drugs", "exams", "alt", "alp", mode="before")
    @classmethod
    def _strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    @model_validator(mode="after")
    def _require_sections_or_files(self) -> "PatientData":
        if self.from_files:
            return self
        if any((self.info, self.anamnesis, self.drugs, self.exams)):
            return self
        raise ValueError("Either provide clinical sections or set 'from_files' to true.")

    @staticmethod
    def _coerce_marker(value: str | None) -> tuple[float | None, str | None]:
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
        alt_value, alt_text = self._coerce_marker(self.alt)
        if alt_value is not None or alt_text is not None:
            markers["ALAT"] = {
                "value": alt_value,
                "value_text": alt_text,
                "unit": None,
                "date": None,
            }
        alp_value, alp_text = self._coerce_marker(self.alp)
        if alp_value is not None or alp_text is not None:
            markers["ALP"] = {
                "value": alp_value,
                "value_text": alp_text,
                "unit": None,
                "date": None,
            }
        return markers

    def compose_structured_text(self) -> str | None:
        if self.info:
            return self.info
        sections: list[str] = []
        if self.anamnesis:
            sections.append(f"# ANAMNESIS\n{self.anamnesis}")
        blood_lines: list[str] = []
        if self.alt:
            blood_lines.append(f"ALAT: {self.alt}")
        if self.alp:
            blood_lines.append(f"ALP: {self.alp}")
        blood_body = "\n".join(line for line in blood_lines if line)
        if blood_body or self.exams:
            parts = [part for part in (blood_body, self.exams) if part]
            sections.append(f"# BLOOD TESTS\n{'\n'.join(parts)}")
        if self.exams:
            sections.append(f"# ADDITIONAL TESTS\n{self.exams}")
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
    def _strip_report(cls, v: str) -> str:
        if v is None:
            return v
        return str(v).strip()


###############################################################################
class PatientDiseases(BaseModel):
    diseases: list[str] = Field(default_factory=list)
    hepatic_diseases: list[str] = Field(default_factory=list)

    @field_validator("diseases", "hepatic_diseases")
    def strip_and_nonempty(cls, v) -> list[str]:
        # Clean up each string, skip empty/None
        return [str(item).strip() for item in v if item and str(item).strip()]

    @field_validator("diseases", "hepatic_diseases")
    def must_be_unique(cls, v):
        if len(v) != len(set(map(str.lower, v))):
            raise ValueError("List must contain unique items (case-insensitive)")
        return v

    @model_validator(mode="after")
    def hepatic_subset_of_diseases(self):
        disease_set = set(s.strip().lower() for s in self.diseases)
        hepatic_set = set(s.strip().lower() for s in self.hepatic_diseases)
        missing = hepatic_set - disease_set
        if missing:
            raise ValueError(
                f"hepatic_diseases contains items not present in diseases: {sorted(missing)}"
            )

        return self


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
    def _normalize_name(cls, v: str) -> str:
        v = re.sub(r"\s+", " ", v.strip())
        return v.rstrip(",:;.- ")


# -----------------------------------------------------------------------------
class PatientBloodTests(BaseModel):
    """Container with original text and all parsed test entries."""

    source_text: str = Field(..., description="Original text used for parsing.")

    entries: list[BloodTest] = Field(default_factory=list)
