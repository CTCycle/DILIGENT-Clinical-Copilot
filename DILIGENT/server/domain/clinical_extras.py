from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from DILIGENT.server.domain.clinical import (
    ClinicalLabEntry,
    LiverInjuryOnsetContext,
    PatientDrugs,
)


@dataclass(frozen=True)
class CandidateSelectionResult:
    relevant: list[dict[str, str]]
    excluded: list[dict[str, str]]
    unresolved: list[dict[str, str]]
    ordered_analysis_drugs: PatientDrugs


@dataclass(slots=True)
class HepatoxPreparedInputs:
    resolved_drugs: dict[str, dict[str, Any]]
    pattern_prompt: str
    clinical_context: str


class LabExtractionPayload(BaseModel):
    entries: list[ClinicalLabEntry] = Field(default_factory=list)
    onset_context: LiverInjuryOnsetContext | None = Field(default=None)
