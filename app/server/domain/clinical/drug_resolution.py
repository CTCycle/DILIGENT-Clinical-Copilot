from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


RxNavValidationStatus = Literal[
    "exact_rxcui",
    "alias_to_rxcui",
    "ingredient_to_rxcui",
    "brand_to_rxcui",
    "no_rxnav_match",
    "ambiguous_rxnav",
    "not_applicable_livertox_direct",
]

DrugDecisionStatus = Literal[
    "accepted_exact_livertox",
    "accepted_rxnav_validated",
    "accepted_livertox_without_rxnav",
    "ambiguous_requires_review",
    "missing_rxnav",
    "missing_livertox",
    "rejected_false_positive",
]

###############################################################################
@dataclass(frozen=True)
class DrugIdentityCandidate:
    source_label: str
    canonical_candidate: str
    normalized_candidate: str
    kind: str
    confidence: float
    notes: tuple[str, ...] = field(default_factory=tuple)

###############################################################################
@dataclass(slots=True)
class NormalizedDrugMention:
    extracted_name: str
    canonical_name: str
    normalized_name: str
    source: Literal["therapy", "anamnesis", "unknown"]
    raw_mentions: list[str]
    origins: list[str]
    extraction_metadata: list[dict[str, Any]]
    regimen_group_id: str | None = None
    is_regimen_parent: bool = False
    regimen_components: list[str] = field(default_factory=list)

###############################################################################
class RxNavResolutionCandidate(BaseModel):
    rxcui: str | None = None
    name: str
    normalized_name: str
    term_type: str | None = None
    source: Literal["rxnorm", "manual", "livertox", "catalog"]
    alias_kind: str | None = None
    confidence: float | None = None
    reason: str
    accepted: bool = False
    rejected_reason: str | None = None

###############################################################################
class LiverToxResolutionCandidate(BaseModel):
    nbk_id: str | None = None
    drug_name: str
    normalized_name: str
    monograph_key: str | None = None
    has_excerpt: bool
    confidence: float | None = None
    reason: str
    accepted: bool = False
    rejected_reason: str | None = None

###############################################################################
class DrugResolutionDecision(BaseModel):
    extracted_name: str
    normalized_extracted_name: str
    source: Literal["therapy", "anamnesis", "unknown"] = "unknown"
    regimen_group_id: str | None = None
    is_regimen_parent: bool = False
    regimen_components: list[str] = Field(default_factory=list)
    rxnav_candidates: list[RxNavResolutionCandidate] = Field(default_factory=list)
    accepted_rxnav_rxcui: str | None = None
    rxnav_validation_status: RxNavValidationStatus = "no_rxnav_match"
    livertox_candidates: list[LiverToxResolutionCandidate] = Field(default_factory=list)
    accepted_livertox_nbk_id: str | None = None
    accepted_livertox_name: str | None = None
    accepted_livertox_match_has_excerpt: bool = False
    decision_status: DrugDecisionStatus
    confidence: float | None = None
    reasons: list[str] = Field(default_factory=list)
    requires_human_review: bool

###############################################################################
class DrugIdentityProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_mention: str = Field(..., min_length=1, max_length=200)
    proposed_canonical_name: str | None = Field(default=None, max_length=200)
    alternate_names: list[str] = Field(default_factory=list, max_length=12)
    ingredients: list[str] = Field(default_factory=list, max_length=8)
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., min_length=1, max_length=500)

    # -------------------------------------------------------------------------
    @field_validator(
        "original_mention",
        "proposed_canonical_name",
        "rationale",
        mode="before",
    )
    @classmethod
    def strip_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    # -------------------------------------------------------------------------
    @field_validator("alternate_names", "ingredients", mode="before")
    @classmethod
    def normalize_ingredients(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned

###############################################################################
class DrugIdentityProposalBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[DrugIdentityProposal] = Field(default_factory=list, max_length=50)
