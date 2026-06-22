from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ExtractionStrategy = Literal["deterministic", "llm", "hybrid"]

###############################################################################
class LlmClinicalSectionTextDraft(BaseModel):
    anamnesis: str = ""
    therapy: str = ""
    lab_analysis: str = ""

###############################################################################
class ExtractionStrategyDecision(BaseModel):
    section: str
    strategy: ExtractionStrategy
    confidence: float = Field(..., ge=0.0, le=1.0)
    therapy_structure_score: float | None = Field(default=None, ge=0.0, le=1.0)
    laboratory_structure_score: float | None = Field(default=None, ge=0.0, le=1.0)
    unresolved_line_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_span_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguity_count: int = Field(default=0, ge=0)
    reasons: list[str] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)
