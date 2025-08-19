from typing import Any, Dict, Iterable, List, Literal, Optional
from typing_extensions import Annotated
from pydantic import BaseModel, Field

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
class Diseases(BaseModel):
    diseases: Annotated[
        List[Annotated[str, Field(min_length=1, max_length=200)]],
        Field(description="Canonical disease/condition name.")
    ]
    
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