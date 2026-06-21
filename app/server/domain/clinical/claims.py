from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


###############################################################################
class ClinicalClaim(BaseModel):
    claim: str = Field(..., min_length=1, max_length=1000)
    source: Literal["source_text", "livertox", "rucam", "derived", "unknown"]
    evidence_quote: str | None = Field(default=None, max_length=1000)
    confidence: Literal["high", "moderate", "low"]
    requires_review: bool = Field(default=False)

    # -------------------------------------------------------------------------
    @field_validator("claim", "evidence_quote", mode="before")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None


###############################################################################
class DrugClinicalNarrative(BaseModel):
    drug_name: str = Field(..., min_length=1, max_length=200)
    summary: str = Field(default="", max_length=2000)
    claims: list[ClinicalClaim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
