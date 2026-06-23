from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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
