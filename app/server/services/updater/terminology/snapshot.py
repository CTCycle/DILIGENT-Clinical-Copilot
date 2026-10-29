# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Source-agnostic staged terminology records produced by source adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

ConceptType = Literal["ingredient", "precise_ingredient", "multi_ingredient"]
IDENTITY_SYSTEMS = ("unii", "chembl", "drugcentral", "inchikey")
ProgressFn = Callable[[float, str], None]
StopFn = Callable[[], bool]

###############################################################################
@dataclass
class StagedConcept:
    """One source concept; ``key`` is unique only within its snapshot."""

    key: str
    name: str
    concept_type: ConceptType = "ingredient"
    rxcui: str | None = None
    identifiers: dict[str, str] = field(default_factory=dict)
    preferred_inn: str | None = None
    atc_codes: tuple[str, ...] = ()
    component_keys: tuple[str, ...] = ()
    parent_key: str | None = None

###############################################################################
@dataclass
class StagedAlias:
    concept_key: str
    alias: str
    alias_kind: str
    term_type: str | None = None
    language: str | None = None
    country: str | None = None
    source_record_id: str | None = None

###############################################################################
@dataclass
class StagedNamedAlias:
    """Alias whose target concept is found by exact verified lookup names."""

    alias: str
    alias_kind: str
    target_names: tuple[str, ...]
    term_type: str | None = None
    language: str | None = None
    country: str | None = None
    source_record_id: str | None = None

###############################################################################
@dataclass
class StagedProductIngredient:
    """Product component; resolved by concept key or by exact lookup names."""

    ingredient_name: str
    concept_key: str | None = None
    lookup_names: tuple[str, ...] = ()
    strength_value: float | None = None
    strength_unit: str | None = None

###############################################################################
@dataclass
class StagedProduct:
    product_id: str
    name: str
    brand_name: str | None
    ingredients: list[StagedProductIngredient]
    country: str | None = None
    dose_form: str | None = None
    route: str | None = None
    atc_code: str | None = None
    language: str | None = None

###############################################################################
@dataclass
class TerminologySnapshot:
    source: str
    version: str | None
    license: str
    concepts: dict[str, StagedConcept] = field(default_factory=dict)
    aliases: list[StagedAlias] = field(default_factory=list)
    products: list[StagedProduct] = field(default_factory=list)
    named_aliases: list[StagedNamedAlias] = field(default_factory=list)
    brand_families: bool = False
    stats: dict[str, int] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    def add_alias(self, concept_key: str, alias: str | None, alias_kind: str, **extra: str | None) -> None:
        text = (alias or "").strip()
        if text and concept_key in self.concepts:
            self.aliases.append(StagedAlias(concept_key, text, alias_kind, **extra))

###############################################################################
class TerminologySourceUnavailable(RuntimeError):
    """Raised when a source cannot be reached or its download is incomplete."""

###############################################################################
class TerminologySource(Protocol):
    key: str
    license: str
    attribution: str
    precedence: int

    def probe_version(self) -> str | None:
        """Return the upstream release identifier, or None when unknown."""
        ...

    def fetch(self, progress: ProgressFn, should_stop: StopFn) -> TerminologySnapshot:
        """Download and stage one complete snapshot; never writes the database."""
        ...
