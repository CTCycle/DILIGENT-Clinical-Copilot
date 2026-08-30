from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from domain.clinical.entities import PipelineIssue, RagDocumentReference


###############################################################################
class LocalDiseaseContextEntry(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    evidence: str | None = Field(default=None, max_length=500)
    chronic: bool | None = Field(default=None)
    hepatic_related: bool | None = Field(default=None)


###############################################################################
class LocalPatientDiseaseContext(BaseModel):
    entries: list[LocalDiseaseContextEntry] = Field(default_factory=list)


###############################################################################
class LocalLabEntryDraft(BaseModel):
    marker_name: str = Field(..., min_length=1, max_length=40)
    value_text: str | float | int | None = Field(default=None)
    unit: str | None = Field(default=None, max_length=50)
    sample_date: str | None = Field(default=None, max_length=120)
    evidence: str | None = Field(default=None, max_length=500)


###############################################################################
class LocalOnsetContextDraft(BaseModel):
    onset_date: str | None = Field(default=None, max_length=120)
    onset_basis: str | None = Field(default=None, max_length=200)
    evidence: str | None = Field(default=None, max_length=500)


###############################################################################
class LocalLabExtractionPayload(BaseModel):
    entries: list[LocalLabEntryDraft] = Field(default_factory=list)
    onset_context: LocalOnsetContextDraft | None = Field(default=None)


###############################################################################
class LocalDrugEntryDraft(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    dosage: str | None = Field(default=None, max_length=120)
    administration_mode: str | None = Field(default=None, max_length=80)
    route: str | None = Field(default=None, max_length=40)
    administration_pattern: str | None = Field(default=None, max_length=80)
    therapy_start_status: bool | None = Field(default=None)
    therapy_start_date: str | None = Field(default=None, max_length=40)
    suspension_status: bool | None = Field(default=None)
    suspension_date: str | None = Field(default=None, max_length=40)
    evidence: str | None = Field(default=None, max_length=500)
    current_status: str | None = Field(default=None, max_length=40)


###############################################################################
class LocalPatientDrugs(BaseModel):
    entries: list[LocalDrugEntryDraft] = Field(default_factory=list)


###############################################################################
class HepaticPatternResolutionInput(BaseModel):
    explicit_pattern: str | None = None
    calculated_pattern: str | None = None
    r_score: float | None = None


###############################################################################
class HepaticPatternResolutionResult(BaseModel):
    explicit_value: str | None = None
    calculated_value: str | None = None
    final_value: str = "indeterminate"
    source: Literal["provided", "calculated", "undetermined"] = "undetermined"
    conflict: bool = False
    r_score: float | None = None
    warnings: list[PipelineIssue] = Field(default_factory=list)


###############################################################################
@dataclass(frozen=True)
class RagRetrievalBundle:
    context_text: str | None
    references: tuple[RagDocumentReference, ...]
