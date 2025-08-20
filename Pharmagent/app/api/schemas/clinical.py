import json
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, model_validator, field_validator

from Pharmagent.app.api.schemas.placeholders import EXAMPLE_INPUT_DATA

#------------------------------------------------------------------------------
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
        examples=["Marco Rossi"]
    )
    info: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        strip_whitespace=True,
        description="Multiline text input with patient's info.",
        examples=[EXAMPLE_INPUT_DATA]
    )


#------------------------------------------------------------------------------
class PatientOutputReport(BaseModel):    
    report: str = Field(
        ...,
        min_length=1,
        max_length=200,
        strip_whitespace=True,        
        description="Multiline text output with the final report.",
        examples=["This is a sample note."])
    
#------------------------------------------------------------------------------
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
                f"hepatic_diseases contains items not present in diseases: {sorted(missing)}"
            )
        return self
    
#------------------------------------------------------------------------------  
class Monography(BaseModel):
    title: str
    url: str
    last_update: Optional[str] = Field(
        None, 
        description="Date string, e.g. 'July 27, 2017'")
    likelihood_score: Optional[str] = None
    drug_classes: list[str] = []
    sections: dict[str, str] = {}