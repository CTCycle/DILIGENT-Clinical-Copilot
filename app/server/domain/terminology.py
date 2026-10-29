# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ResolutionStatus = Literal["verified", "probable", "ambiguous", "unresolved"]
ResolutionMethod = Literal[
    "exact_alias",
    "confirmed_local",
    "language_variant",
    "orthographic_variant",
    "combination",
    "fuzzy",
    "external_rxnav",
    "product_unverified",
    "none",
]

###############################################################################
class TerminologyIngredient(BaseModel):
    """One active ingredient of a resolved concept (salt form kept separately)."""

    model_config = ConfigDict(extra="forbid")

    drug_id: int
    name: str
    concept_type: str
    parent_drug_id: int | None = None
    parent_name: str | None = None

###############################################################################
class TerminologyCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drug_id: int
    name: str
    concept_type: str
    matched_alias: str | None = None
    alias_kind: str | None = None
    source: str | None = None
    country: str | None = None
    edit_distance: int | None = None
    # Raw string similarity in [0, 1]; a ranking signal, not a probability.
    similarity: float | None = None

###############################################################################
class TerminologyResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original: str
    normalized: str
    strengths: list[str] = Field(default_factory=list)
    qualifiers: list[str] = Field(default_factory=list)
    status: ResolutionStatus
    accepted: bool
    method: ResolutionMethod
    drug_id: int | None = None
    preferred_name: str | None = None
    concept_type: str | None = None
    ingredients: list[TerminologyIngredient] = Field(default_factory=list)
    precise_forms: list[TerminologyIngredient] = Field(default_factory=list)
    matched_alias: str | None = None
    alias_kind: str | None = None
    source: str | None = None
    source_version: str | None = None
    country: str | None = None
    strength_supported: bool | None = None
    candidates: list[TerminologyCandidate] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    external_calls: int = 0
    terminology_revision: int | None = None

###############################################################################
class TerminologyResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=200)
    allow_external: bool = True

###############################################################################
class TerminologyAliasDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alias: str = Field(..., min_length=1, max_length=200)
    drug_id: int = Field(..., ge=1)

###############################################################################
class TerminologyAliasDecisionResponse(BaseModel):
    alias: str
    drug_id: int
    validation_status: Literal["confirmed", "rejected"]

###############################################################################
class TerminologySourceSummary(BaseModel):
    source: str
    enabled: bool
    license: str
    attribution: str
    last_status: str | None = None
    last_version: str | None = None
    last_finished_at: datetime | None = None
    last_completed_version: str | None = None
    last_completed_at: datetime | None = None
    aliases: int = 0
    products: int = 0
    open_conflicts: int = 0
    last_error: str | None = None

###############################################################################
class TerminologySummaryResponse(BaseModel):
    revision: int | None
    concepts: int
    aliases: int
    resolvable_aliases: int
    provisional_aliases: int
    products: int
    verified_products: int
    livertox_links: int
    linked_monographs: int
    total_monographs: int
    open_conflicts: int
    index: dict[str, Any] = Field(default_factory=dict)
    sources: list[TerminologySourceSummary] = Field(default_factory=list)

###############################################################################
class TerminologyImportRunItem(BaseModel):
    id: int
    source: str
    source_version: str | None
    mode: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    stats: dict[str, Any] | None = None
    error: str | None = None

###############################################################################
class TerminologyImportRunsResponse(BaseModel):
    items: list[TerminologyImportRunItem]

###############################################################################
class TerminologyConflictItem(BaseModel):
    id: int
    source: str
    conflict_type: str
    subject: str
    existing_drug: str | None
    proposed_drug: str | None
    detail: str | None
    status: str
    detected_at: datetime

###############################################################################
class TerminologyConflictsResponse(BaseModel):
    items: list[TerminologyConflictItem]
    total: int
    offset: int
    limit: int
