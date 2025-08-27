import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator, field_validator

from Pharmagent.app.api.schemas.placeholders import EXAMPLE_INPUT_DATA

Comparator = Literal["<=", "<", ">=", ">"]


###############################################################################
class PatientData(BaseModel):
    """
    Input schema for submitting raw text.
    - Strips whitespace, requires at least 1 char after stripping.
    - Caps size to prevent abuse and oversized payloads.
    """
    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=200,
        strip_whitespace=True,
        description="Name of the patient (optional).",
        examples=["Marco Rossi"])

    info: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        strip_whitespace=True,
        description="Multiline text input with patient's info.",
        examples=[EXAMPLE_INPUT_DATA])


###############################################################################
class PatientOutputReport(BaseModel):    
    report: str = Field(
        ...,
        min_length=1,
        max_length=200,
        strip_whitespace=True,        
        description="Multiline text output with the final report.",
        examples=["This is a sample note."])
    

###############################################################################
class PatientDiseases(BaseModel):
    diseases: List[str] = Field(default_factory=list)
    hepatic_diseases: List[str] = Field(default_factory=list)

    @field_validator("diseases", "hepatic_diseases")
    def strip_and_nonempty(cls, v):
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
                f"hepatic_diseases contains items not present in diseases: {sorted(missing)}")
        
        return self
    

###############################################################################
class BloodTest(BaseModel):
    """A single blood test result extracted from text."""
    name: str = Field(..., description="Test name exactly as found (minimally normalized).")
    value: float | None = Field(
        None, description="Numeric value if applicable (dot-decimal).")
    value_text: str | None = Field(
        None, description="Raw textual value when not numeric (e.g., '1:80').")
    unit: str | None = Field(None, description="Unit as found, if present.")
    cutoff: float | None = Field(None, description="Cutoff/upper limit if provided.")
    cutoff_unit: str | None = Field(None, description="Cutoff unit if specified; often same as unit.")
    note: str | None = Field(None, description="Parenthetical note not related to cutoff.")
    context_date: str | None = Field(
        None, description="ISO YYYY-MM-DD if parsed, else original date string for this batch.")

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, v: str) -> str:
        v = re.sub(r"\s+", " ", v.strip())
        return v.rstrip(",:;.- ")

#-----------------------------------------------------------------------------  
class PatientBloodTests(BaseModel):
    """Container with original text and all parsed test entries."""

    source_text: str = Field(
        ..., 
        description="Original text used for parsing.")
    
    entries: list[BloodTest] = Field(
        default_factory=list)

   






