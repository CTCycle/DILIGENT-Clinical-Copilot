from __future__ import annotations

import re
from typing import Any

from common.utils.text_utils import coerce_text
from domain.clinical.drug_resolution import (
    NormalizedDrugMention,
    RxNavResolutionCandidate,
)
from services.catalogs.runtime import get_reference_catalog_snapshot
from services.text.normalization import normalize_drug_query_name

###############################################################################
class RxNavCandidateResolver:

    _DOSAGE_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")

    # -------------------------------------------------------------------------
    def __init__(self, matcher: Any | None) -> None:
        self.matcher = matcher
        snapshot = get_reference_catalog_snapshot()
        self.broad_categories = set(
            snapshot.values("drug_matching", "broad_drug_categories")
        )
        self.form_suffix_tokens = set(
            snapshot.values("drug_matching", "rxnav_form_suffix_tokens")
        )
        self.allowed_term_types = {
            value.upper()
            for value in snapshot.values(
                "drug_matching",
                "rxnav_allowed_term_types",
            )
        }

    # -------------------------------------------------------------------------
    def build_candidates(
        self, mention: NormalizedDrugMention
    ) -> list[RxNavResolutionCandidate]:
        if mention.normalized_name in self.broad_categories:
            return [
                RxNavResolutionCandidate(
                    rxcui=None,
                    name=mention.canonical_name,
                    normalized_name=mention.normalized_name,
                    term_type=None,
                    source="catalog",
                    alias_kind="broad_category",
                    confidence=0.0,
                    reason="broad_category_rejected",
                    rejected_reason="broad category is not a concrete drug",
                )
            ]
        lookup = getattr(self.matcher, "lookup", None)
        index = getattr(lookup, "catalog_global_index", {}) if lookup else {}
        if not index:
            return []
        exact = index.get(mention.normalized_name)
        if exact is not None:
            return [self._candidate_from_payload(exact, "exact_catalog_match")]
        matches: list[
            tuple[
                RxNavResolutionCandidate,
                tuple[dict[str, Any], bool, str],
                list[str],
                bool,
            ]
        ] = []
        query_tokens = mention.normalized_name.split()
        for key, payload in index.items():
            key_tokens = key.split()
            if not key_tokens:
                continue
            is_query_prefix = key_tokens == query_tokens[: len(key_tokens)]
            is_catalog_formulation = (
                len(query_tokens) < len(key_tokens)
                and query_tokens == key_tokens[: len(query_tokens)]
            )
            if not is_query_prefix and not is_catalog_formulation:
                continue
            suffix_tokens = (
                query_tokens[len(key_tokens) :]
                if is_query_prefix
                else key_tokens[len(query_tokens) :]
            )
            if not self.allow_catalog_prefix_match(suffix_tokens, payload):
                continue
            reason = (
                "formulation_prefix_catalog_match"
                if is_catalog_formulation
                else "guarded_prefix_catalog_match"
            )
            matches.append(
                (
                    self._candidate_from_payload(payload, reason),
                    payload,
                    suffix_tokens,
                    is_catalog_formulation,
                )
            )
        return self._rank_formulation_matches(mention, matches)[:4]

    # -------------------------------------------------------------------------
    def _rank_formulation_matches(
        self,
        mention: NormalizedDrugMention,
        matches: list[
            tuple[
                RxNavResolutionCandidate,
                tuple[dict[str, Any], bool, str],
                list[str],
                bool,
            ]
        ],
    ) -> list[RxNavResolutionCandidate]:
        """Deduplicate catalog aliases and use explicit exposure details safely.

        A bare ingredient combination can be a prefix of several RxNav product
        labels.  A strength and route captured from the therapy entry are
        deterministic formulation evidence; without that evidence, retain all
        viable candidates so the clinical policy keeps the match ambiguous.
        """
        if not matches:
            return []

        deduplicated: dict[str, tuple[RxNavResolutionCandidate, Any, list[str], bool]] = {}
        for candidate, payload, suffix_tokens, is_catalog_formulation in matches:
            identity = candidate.rxcui or candidate.normalized_name
            current = deduplicated.get(identity)
            if current is None or (
                is_catalog_formulation and not current[3]
            ):
                deduplicated[identity] = (
                    candidate,
                    payload,
                    suffix_tokens,
                    is_catalog_formulation,
                )

        unique_matches = list(deduplicated.values())
        formulation_matches = [item for item in unique_matches if item[3]]
        if not formulation_matches:
            return [item[0] for item in unique_matches]

        scored = [
            (
                item,
                self._formulation_evidence_score(
                    mention,
                    item[1],
                    item[2],
                ),
            )
            for item in formulation_matches
        ]
        dosage_numbers = self._mention_dosage_numbers(mention)
        if dosage_numbers:
            best_evidence = max(
                (score[0], score[1]) for _item, score in scored
            )
            strongest = [
                item
                for item, score in scored
                if (score[0], score[1]) == best_evidence
            ]
            strongest_rxcuis = {
                item[0].rxcui for item in strongest if item[0].rxcui
            }
            if best_evidence[0] > 0 and len(strongest_rxcuis) == 1:
                selected_rxcui = next(iter(strongest_rxcuis))
                for item, score in scored:
                    candidate = item[0]
                    if candidate.rxcui == selected_rxcui:
                        candidate.reason = (
                            "strength_matched_formulation_catalog_match"
                        )
                        candidate.confidence = 0.88
                        continue
                    candidate.rejected_reason = (
                        "not selected by formulation strength and route evidence"
                    )

        return [item[0] for item in unique_matches]

    # -------------------------------------------------------------------------
    def _formulation_evidence_score(
        self,
        mention: NormalizedDrugMention,
        payload: tuple[dict[str, Any], bool, str],
        suffix_tokens: list[str],
    ) -> tuple[int, int, int]:
        entry, _matched_is_synonym, _matched_value = payload
        dosage_numbers = self._mention_dosage_numbers(mention)
        catalog_numbers = self._catalog_numbers(entry)
        matched_strengths = sum(
            1 for number in dosage_numbers if number in catalog_numbers
        )
        route_tokens = {
            token
            for metadata in mention.extraction_metadata
            if isinstance(metadata, dict)
            for field_name in ("route", "administration_mode")
            for token in str(metadata.get(field_name) or "")
            .casefold()
            .replace("/", " ")
            .split()
        }
        route_match = int(bool(route_tokens & set(suffix_tokens)))
        return matched_strengths, route_match, -len(suffix_tokens)

    # -------------------------------------------------------------------------
    def _mention_dosage_numbers(self, mention: NormalizedDrugMention) -> set[str]:
        numbers: set[str] = set()
        for metadata in mention.extraction_metadata:
            if not isinstance(metadata, dict):
                continue
            dosage = str(metadata.get("dosage") or "")
            numbers.update(self._DOSAGE_NUMBER_RE.findall(dosage))
        return numbers

    # -------------------------------------------------------------------------
    def _catalog_numbers(self, entry: dict[str, Any]) -> set[str]:
        values: list[str] = []
        for field_name in (
            "raw_name",
            "name",
            "synonyms",
            "fallback_aliases",
            "brand_names",
        ):
            value = entry.get(field_name)
            if isinstance(value, list):
                values.extend(str(item) for item in value)
            elif value is not None:
                values.append(str(value))
        return set(
            number
            for value in values
            for number in self._DOSAGE_NUMBER_RE.findall(value)
        )

    # -------------------------------------------------------------------------
    def allow_catalog_prefix_match(
        self,
        suffix_tokens: list[str],
        payload: tuple[dict[str, Any], bool, str],
    ) -> bool:
        entry, _matched_is_synonym, _matched_value = payload
        if not suffix_tokens:
            return False
        if any(token not in self.form_suffix_tokens for token in suffix_tokens):
            return False
        if not coerce_text(entry.get("rxcui")):
            return False
        term_type = coerce_text(entry.get("term_type"))
        return term_type is None or term_type.upper() in self.allowed_term_types

    # -------------------------------------------------------------------------
    def _candidate_from_payload(
        self,
        payload: tuple[dict[str, Any], bool, str],
        reason: str,
    ) -> RxNavResolutionCandidate:
        entry, matched_is_synonym, matched_value = payload
        name = (
            coerce_text(matched_value)
            or coerce_text(entry.get("name"))
            or coerce_text(entry.get("raw_name"))
            or ""
        )
        alias_kind = "alias" if matched_is_synonym else "name"
        if matched_is_synonym or reason == "exact_catalog_match":
            alias_kind = self._classify_alias(entry, matched_value)
        rxcui = coerce_text(entry.get("rxcui"))
        return RxNavResolutionCandidate(
            rxcui=rxcui,
            name=name,
            normalized_name=normalize_drug_query_name(name),
            term_type=coerce_text(entry.get("term_type")),
            source="catalog",
            alias_kind=alias_kind,
            confidence=0.95 if reason == "exact_catalog_match" else 0.78,
            reason=reason,
            rejected_reason=None if rxcui else "catalog row has no RXCUI",
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _classify_alias(entry: dict[str, Any], matched_value: str) -> str:
        normalized = normalize_drug_query_name(matched_value)
        brand_values = entry.get("brand_names", []) or []
        if isinstance(brand_values, str):
            brand_values = [brand_values]
        if normalized and any(
            normalize_drug_query_name(str(value)) == normalized
            for value in brand_values
        ):
            return "brand"
        return "ingredient"
