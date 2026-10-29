# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Layered drug terminology resolution.

Stages: deterministic normalization, exact local aliases, curated language
variants and combination composition, bounded approximate matching with a
strict acceptance gate, and an optional budgeted RxNav fallback. Fuzzy results
are never promoted to aliases; only verified RxNav relations are persisted.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from rapidfuzz.distance import OSA

from common.utils.bounded_cache import BoundedCache
from common.utils.logger import logger
from domain.terminology import (
    ResolutionMethod,
    TerminologyCandidate,
    TerminologyIngredient,
    TerminologyResolution,
)
from repositories.terminology_lookup_repository import (
    RUNTIME_SOURCE,
    AliasHit,
    ConceptInfo,
    TerminologyLookupRepository,
)
from services.terminology.external import RxNavBudgetExceeded, RxNavConcept, RxNavRuntimeClient
from services.terminology.index import TerminologyIndexCache
from services.terminology.normalization import (
    NormalizedDrugInput,
    language_variants,
    normalize_drug_input,
)

BLOCKED_ALIAS_KIND = "error_prone_abbreviation"

MAX_CANDIDATES = 5

###############################################################################
@dataclass(frozen=True)
class ResolverSettings:
    fuzzy_min_query_length: int = 5
    fuzzy_shortlist_size: int = 50
    fuzzy_short_name_length: int = 8
    fuzzy_short_max_distance: int = 1
    fuzzy_long_max_distance: int = 2
    fuzzy_min_margin: int = 1
    resolution_cache_limit: int = 5000
    external_fallback_enabled: bool = True
    external_max_requests_per_name: int = 4
    external_negative_cache_days: int = 7

###############################################################################
@dataclass
class _Match:
    hits: list[AliasHit]
    method: ResolutionMethod
    key: str
    distance: int | None = None

###############################################################################
class TerminologyResolver:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        lookup: TerminologyLookupRepository,
        index_cache: TerminologyIndexCache,
        *,
        settings: ResolverSettings,
        external: RxNavRuntimeClient | None = None,
    ) -> None:
        self.lookup = lookup
        self.index_cache = index_cache
        self.settings = settings
        self.external = external
        self._cache: BoundedCache[tuple[str, int | None, bool], TerminologyResolution] = BoundedCache(
            settings.resolution_cache_limit
        )
        self._cache_lock = threading.Lock()

    # -------------------------------------------------------------------------
    def resolve(self, text: str, *, allow_external: bool = True) -> TerminologyResolution:
        normalized = normalize_drug_input(text)
        revision = self.lookup.revision()
        # The raw key keeps strengths and qualifiers, which shape the result.
        cache_key = (normalized.raw_key or normalized.full_key or normalized.key, revision, allow_external)
        with self._cache_lock:
            cached = self._cache.get(cache_key, None)
        if isinstance(cached, TerminologyResolution):
            return cached.model_copy(update={"original": text})
        result = self._resolve(normalized, revision, allow_external)
        result.terminology_revision = revision
        if result.external_calls == 0 or result.status != "unresolved":
            with self._cache_lock:
                self._cache.put(cache_key, result)
        return result

    # -------------------------------------------------------------------------
    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    # -------------------------------------------------------------------------
    def _resolve(
        self,
        normalized: NormalizedDrugInput,
        revision: int | None,
        allow_external: bool,
    ) -> TerminologyResolution:
        if not (normalized.raw_key or normalized.key or normalized.full_key):
            return self._empty(normalized, "unresolved", ["empty_after_normalization"])
        match = self._exact(normalized)
        if match is None and normalized.component_keys:
            combination = self._combination(normalized)
            if combination is not None:
                return combination
        if match is None:
            blocked = self.lookup.blocked_hits([normalized.raw_key, normalized.full_key, normalized.key])
            if blocked:
                result = self._empty(normalized, "ambiguous", ["error_prone_abbreviation"])
                result.candidates = self._candidates(
                    self._moieties(blocked), self.lookup.concepts({hit.drug_id for hit in blocked}), None
                )
                return result
        if match is None:
            match = self._variants(normalized.key)
        if match is None:
            match = self._orthographic(normalized.key, revision)
        if match is not None:
            return self._from_hits(normalized, match)
        unverified = self.lookup.unverified_products([normalized.full_key, normalized.key])
        if unverified:
            product = unverified[0]
            result = self._empty(normalized, "probable", ["product_composition_unverified"])
            result.method = "product_unverified"
            result.matched_alias = product.name
            result.country = product.country
            return result
        fuzzy = self._fuzzy(normalized, revision)
        if fuzzy is not None and fuzzy.accepted:
            return fuzzy
        calls = 0
        if allow_external and self._external_enabled():
            external, calls = self._external(normalized, revision)
            if external is not None:
                return external
        result = fuzzy if fuzzy is not None else self._empty(normalized, "unresolved", ["no_terminology_match"])
        result.external_calls += calls
        return result

    # -------------------------------------------------------------------------
    def _external_enabled(self) -> bool:
        return self.external is not None and self.settings.external_fallback_enabled

    # -------------------------------------------------------------------------
    def _exact(self, normalized: NormalizedDrugInput) -> _Match | None:
        keys = [
            key
            for key in dict.fromkeys([normalized.raw_key, normalized.full_key, normalized.key])
            if key
        ]
        hits = self.lookup.exact_hits(keys)
        for key in keys:
            selected = [hit for hit in hits if hit.alias_norm == key]
            if selected:
                return _Match(selected, self._exact_method(selected), key)
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _exact_method(hits: list[AliasHit]) -> ResolutionMethod:
        if all(hit.validation_status == "confirmed" for hit in hits):
            return "confirmed_local"
        return "exact_alias"

    # -------------------------------------------------------------------------
    def _variants(self, key: str) -> _Match | None:
        variants = language_variants(key) if key else []
        if not variants:
            return None
        hits = self.lookup.exact_hits(variants)
        for variant in variants:
            selected = [hit for hit in hits if hit.alias_norm == variant]
            if selected:
                return _Match(selected, "language_variant", variant)
        return None

    # -------------------------------------------------------------------------
    def _orthographic(self, key: str, revision: int | None) -> _Match | None:
        if len(key) < self.settings.fuzzy_min_query_length:
            return None
        index = self.index_cache.get(revision, self.lookup.fuzzy_vocabulary)
        for candidate in [key, *language_variants(key)]:
            aliases = [alias for alias in index.skeleton_matches(candidate) if alias != key]
            if aliases:
                hits = self.lookup.exact_hits(aliases)
                if hits:
                    return _Match(hits, "orthographic_variant", candidate)
        return None

    # -------------------------------------------------------------------------
    def _combination(self, normalized: NormalizedDrugInput) -> TerminologyResolution | None:
        component_ids: list[int] = []
        for component in normalized.component_keys:
            match = self._exact_key(component) or self._variants(component)
            if match is None:
                return None
            moieties = self._moieties(match.hits)
            if len(moieties) != 1:
                return None
            component_ids.append(next(iter(moieties)))
        multi = self.lookup.multi_concept_for(component_ids)
        if multi is None:
            result = self._empty(normalized, "probable", ["combination_concept_not_found"])
            concepts = self.lookup.concepts(component_ids)
            result.method = "combination"
            result.ingredients = [self._ingredient(concepts, drug_id) for drug_id in component_ids]
            return result
        concepts = self.lookup.concepts([multi])
        return self._build(
            normalized,
            drug_id=multi,
            concepts=concepts,
            method="combination",
            status="verified",
            accepted=True,
            hits=[],
            matched_alias=" + ".join(normalized.component_keys),
        )

    # -------------------------------------------------------------------------
    def _exact_key(self, key: str) -> _Match | None:
        hits = self.lookup.exact_hits([key])
        return _Match(hits, self._exact_method(hits), key) if hits else None

    # -------------------------------------------------------------------------
    def _moieties(self, hits: Iterable[AliasHit]) -> dict[int, list[AliasHit]]:
        hits = list(hits)
        concepts = self.lookup.concepts({hit.drug_id for hit in hits})
        grouped: dict[int, list[AliasHit]] = defaultdict(list)
        for hit in hits:
            info = concepts.get(hit.drug_id)
            parent = info.parent_id if info and info.parent_id else hit.drug_id
            grouped[parent].append(hit)
        return grouped

    # -------------------------------------------------------------------------
    def _from_hits(self, normalized: NormalizedDrugInput, match: _Match) -> TerminologyResolution:
        grouped = self._moieties(match.hits)
        concepts = self.lookup.concepts(
            {hit.drug_id for hit in match.hits} | set(grouped)
        )
        if len(grouped) == 1:
            moiety, hits = next(iter(grouped.items()))
            drug_ids = {hit.drug_id for hit in hits}
            chosen = next(iter(drug_ids)) if len(drug_ids) == 1 else moiety
            best = self._best_hit([hit for hit in hits if hit.drug_id == chosen] or hits)
            return self._build(
                normalized,
                drug_id=chosen,
                concepts=concepts,
                method=match.method,
                status="verified",
                accepted=True,
                hits=hits,
                matched_alias=best.alias,
                precise_ids=sorted(drug_ids - {chosen}),
                distance=match.distance,
            )
        result = self._empty(normalized, "ambiguous", ["alias_maps_to_multiple_concepts"])
        result.method = match.method
        result.candidates = self._candidates(grouped, concepts, match.distance)
        return result

    # -------------------------------------------------------------------------
    @staticmethod
    def _best_hit(hits: list[AliasHit]) -> AliasHit:
        order = {"confirmed": 0, "verified": 1}
        return sorted(hits, key=lambda hit: (order.get(hit.validation_status, 2), hit.source, hit.alias))[0]

    # -------------------------------------------------------------------------
    def _candidates(
        self,
        grouped: dict[int, list[AliasHit]],
        concepts: dict[int, ConceptInfo],
        distance: int | None,
    ) -> list[TerminologyCandidate]:
        candidates: list[TerminologyCandidate] = []
        for moiety, hits in grouped.items():
            best = self._best_hit(hits)
            drug_id = best.drug_id if len({hit.drug_id for hit in hits}) == 1 else moiety
            info = concepts.get(drug_id)
            if info is None:
                continue
            candidates.append(
                TerminologyCandidate(
                    drug_id=drug_id,
                    name=info.name,
                    concept_type=info.concept_type,
                    matched_alias=best.alias,
                    alias_kind=best.alias_kind,
                    source=best.source,
                    country=best.country,
                    edit_distance=distance,
                )
            )
        return sorted(candidates, key=lambda item: (item.edit_distance or 0, item.name))[:MAX_CANDIDATES]

    # -------------------------------------------------------------------------
    def _bound(self, length: int) -> int:
        if length < self.settings.fuzzy_short_name_length:
            return self.settings.fuzzy_short_max_distance
        return self.settings.fuzzy_long_max_distance

    # -------------------------------------------------------------------------
    def _fuzzy(self, normalized: NormalizedDrugInput, revision: int | None) -> TerminologyResolution | None:
        query = normalized.key
        if len(query) < self.settings.fuzzy_min_query_length:
            return None
        index = self.index_cache.get(revision, self.lookup.fuzzy_vocabulary)
        queries = [query, *language_variants(query)[:4]]
        bound = self._bound(len(query))
        distances: dict[str, int] = {}
        for candidate_query in queries:
            for alias in index.shortlist(
                candidate_query,
                limit=self.settings.fuzzy_shortlist_size,
                max_length_delta=bound + 1,
            ):
                distance = OSA.distance(candidate_query, alias)
                if alias not in distances or distance < distances[alias]:
                    distances[alias] = distance
        if not distances:
            return None
        nearby = {alias: distance for alias, distance in distances.items() if distance <= bound + self.settings.fuzzy_min_margin}
        if not nearby:
            return None
        hits = self.lookup.exact_hits(nearby)
        grouped = self._moieties(hits)
        if not grouped:
            return None
        best_by_moiety: dict[int, int] = {}
        for moiety, moiety_hits in grouped.items():
            best_by_moiety[moiety] = min(nearby[hit.alias_norm] for hit in moiety_hits)
        ranked = sorted(best_by_moiety.items(), key=lambda item: item[1])
        best_moiety, best_distance = ranked[0]
        concepts = self.lookup.concepts(set(grouped) | {hit.drug_id for hit in hits})
        if best_distance > bound:
            result = self._empty(normalized, "unresolved", ["no_candidate_within_edit_bound"])
            result.method = "fuzzy"
            result.candidates = self._fuzzy_candidates(grouped, concepts, nearby, query)
            return result
        runner_up = ranked[1][1] if len(ranked) > 1 else None
        if runner_up is not None and (
            runner_up - best_distance < self.settings.fuzzy_min_margin or runner_up <= bound
        ):
            # A second concept close to the input is a look-alike/sound-alike
            # risk (hydroxyzine/hydralazine); never auto-correct between them.
            reason = (
                "near_equidistant_candidates"
                if runner_up - best_distance < self.settings.fuzzy_min_margin
                else "lookalike_within_edit_bound"
            )
            result = self._empty(normalized, "ambiguous", [reason])
            result.method = "fuzzy"
            result.candidates = self._fuzzy_candidates(grouped, concepts, nearby, query)
            return result
        hits_for_best = [hit for hit in grouped[best_moiety] if nearby[hit.alias_norm] == best_distance]
        if not any(hit.alias_norm[:1] == query[:1] for hit in hits_for_best):
            # Errors in the first letter are rare in genuine misspellings and
            # common in look-alike confusions; never auto-correct them.
            result = self._empty(normalized, "ambiguous", ["first_letter_differs"])
            result.method = "fuzzy"
            result.candidates = self._fuzzy_candidates(grouped, concepts, nearby, query)
            return result
        drug_ids = {hit.drug_id for hit in hits_for_best}
        chosen = next(iter(drug_ids)) if len(drug_ids) == 1 else best_moiety
        best_hit = self._best_hit(hits_for_best)
        result = self._build(
            normalized,
            drug_id=chosen,
            concepts=concepts,
            method="fuzzy",
            status="probable",
            accepted=True,
            hits=hits_for_best,
            matched_alias=best_hit.alias,
            precise_ids=sorted(drug_ids - {chosen}),
            distance=best_distance,
        )
        result.review_reasons.append("spelling_corrected_requires_review")
        result.candidates = self._fuzzy_candidates(grouped, concepts, nearby, query)
        return result

    # -------------------------------------------------------------------------
    def _fuzzy_candidates(
        self,
        grouped: dict[int, list[AliasHit]],
        concepts: dict[int, ConceptInfo],
        distances: dict[str, int],
        query: str,
    ) -> list[TerminologyCandidate]:
        candidates: list[TerminologyCandidate] = []
        for moiety, hits in grouped.items():
            best = min(hits, key=lambda hit: (distances[hit.alias_norm], hit.alias))
            drug_id = best.drug_id
            info = concepts.get(drug_id) or concepts.get(moiety)
            if info is None:
                continue
            candidates.append(
                TerminologyCandidate(
                    drug_id=info.drug_id,
                    name=info.name,
                    concept_type=info.concept_type,
                    matched_alias=best.alias,
                    alias_kind=best.alias_kind,
                    source=best.source,
                    country=best.country,
                    edit_distance=distances[best.alias_norm],
                    similarity=round(OSA.normalized_similarity(query, best.alias_norm), 3),
                )
            )
        return sorted(candidates, key=lambda item: (item.edit_distance or 0, item.name))[:MAX_CANDIDATES]

    # -------------------------------------------------------------------------
    def _external(
        self,
        normalized: NormalizedDrugInput,
        revision: int | None,
    ) -> tuple[TerminologyResolution | None, int]:
        """Return the verified external resolution (or None) and calls made."""
        key = normalized.full_key or normalized.key
        cached = self.lookup.cached_outcome(key, revision)
        if cached is not None and cached.get("resolution_status") == "unresolved":
            return None, 0
        assert self.external is not None
        try:
            with self.external.session(self.settings.external_max_requests_per_name) as session:
                rxcuis = session.find_rxcuis(normalized.original.strip())
                ingredients = session.related_ingredients(rxcuis[0]) if rxcuis else []
                if rxcuis and not ingredients:
                    ingredients = session.history_ingredients(rxcuis[0])
                calls = session.calls
                failed = session.failed
        except RxNavBudgetExceeded:
            return None, self.settings.external_max_requests_per_name
        if failed:
            return None, calls
        drug_id = self._external_target(ingredients)
        if drug_id is None:
            self.lookup.store_outcome(
                key=key,
                raw_name=normalized.original,
                status="unresolved",
                method="external_rxnav",
                revision=revision,
                ttl_days=self.settings.external_negative_cache_days,
                evidence={"rxcuis": rxcuis, "related": [item.__dict__ for item in ingredients]},
            )
            return None, calls
        self.lookup.record_alias(
            drug_id=drug_id,
            alias=normalized.original.strip(),
            alias_norm=key,
            alias_kind="synonym",
            source=RUNTIME_SOURCE,
            validation_status="verified",
            source_record_id=rxcuis[0],
        )
        self.index_cache.invalidate()
        logger.info("Persisted verified RxNav runtime alias '%s' -> drug %s", key, drug_id)
        concepts = self.lookup.concepts([drug_id])
        result = self._build(
            normalized,
            drug_id=drug_id,
            concepts=concepts,
            method="external_rxnav",
            status="verified",
            accepted=True,
            hits=[],
            matched_alias=normalized.original.strip(),
        )
        result.source = RUNTIME_SOURCE
        result.external_calls = calls
        return result, calls

    # -------------------------------------------------------------------------
    def _external_target(self, ingredients: list[RxNavConcept]) -> int | None:
        """Map RxNav ingredient concepts to exactly one local concept."""
        multis = [item for item in ingredients if item.tty == "MIN"]
        singles = [item for item in ingredients if item.tty == "IN"]
        local = self.lookup.drugs_for_rxcuis(item.rxcui for item in [*multis, *singles])
        if len(multis) == 1 and multis[0].rxcui in local:
            return local[multis[0].rxcui]
        if multis or not singles or any(item.rxcui not in local for item in singles):
            return None
        members = {local[item.rxcui] for item in singles}
        if len(members) == 1:
            return next(iter(members))
        return self.lookup.multi_concept_for(members)

    # -------------------------------------------------------------------------
    @staticmethod
    def _ingredient(concepts: dict[int, ConceptInfo], drug_id: int) -> TerminologyIngredient:
        info = concepts.get(drug_id)
        if info is None:
            return TerminologyIngredient(drug_id=drug_id, name=str(drug_id), concept_type="ingredient")
        parent = concepts.get(info.parent_id) if info.parent_id else None
        return TerminologyIngredient(
            drug_id=drug_id,
            name=info.name,
            concept_type=info.concept_type,
            parent_drug_id=parent.drug_id if parent else None,
            parent_name=parent.name if parent else None,
        )

    # -------------------------------------------------------------------------
    def _build(
        self,
        normalized: NormalizedDrugInput,
        *,
        drug_id: int,
        concepts: dict[int, ConceptInfo],
        method: ResolutionMethod,
        status: str,
        accepted: bool,
        hits: list[AliasHit],
        matched_alias: str | None,
        precise_ids: list[int] | None = None,
        distance: int | None = None,
    ) -> TerminologyResolution:
        missing = {drug_id, *(precise_ids or [])} - set(concepts)
        if missing:
            concepts = {**concepts, **self.lookup.concepts(missing)}
        info = concepts[drug_id]
        if info.component_ids:
            ingredients = [self._ingredient(concepts, component) for component in info.component_ids]
        else:
            ingredients = [self._ingredient(concepts, drug_id)]
        best = self._best_hit(hits) if hits else None
        strength_supported: bool | None = None
        if normalized.strengths and len(ingredients) == 1:
            related = {drug_id, *(precise_ids or [])}
            strength = normalized.strengths[0]
            strength_supported = self.lookup.strength_supported(related, strength.value, strength.unit)
        reasons: list[str] = []
        if distance:
            reasons.append(f"edit_distance_{distance}")
        return TerminologyResolution(
            original=normalized.original,
            normalized=normalized.key,
            strengths=[strength.label() for strength in normalized.strengths],
            qualifiers=list(normalized.qualifiers),
            status=status,  # type: ignore[arg-type]
            accepted=accepted,
            method=method,
            drug_id=drug_id,
            preferred_name=info.name,
            concept_type=info.concept_type,
            ingredients=ingredients,
            precise_forms=[self._ingredient(concepts, item) for item in precise_ids or []],
            matched_alias=matched_alias,
            alias_kind=best.alias_kind if best else None,
            source=best.source if best else None,
            source_version=best.source_version if best else None,
            country=best.country if best else None,
            strength_supported=strength_supported,
            review_reasons=reasons,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _empty(normalized: NormalizedDrugInput, status: str, reasons: list[str]) -> TerminologyResolution:
        return TerminologyResolution(
            original=normalized.original,
            normalized=normalized.key,
            strengths=[strength.label() for strength in normalized.strengths],
            qualifiers=list(normalized.qualifiers),
            status=status,  # type: ignore[arg-type]
            accepted=False,
            method="none",
            review_reasons=reasons,
        )
