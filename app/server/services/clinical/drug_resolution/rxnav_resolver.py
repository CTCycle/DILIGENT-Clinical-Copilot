from __future__ import annotations

from typing import Any

from common.utils.text_utils import coerce_text
from domain.clinical.drug_resolution import RxNavResolutionCandidate
from services.clinical.drug_resolution.normalizer import NormalizedDrugMention
from services.text.normalization import normalize_drug_query_name


class RxNavCandidateResolver:
    BROAD_CATEGORIES = {"vitamins", "minerals", "trace elements"}
    FORM_SUFFIX_TOKENS = {
        "capsule",
        "delayed",
        "dose",
        "extended",
        "gel",
        "injection",
        "mg",
        "ml",
        "oral",
        "prefilled",
        "release",
        "route",
        "schedule",
        "suspension",
        "syringe",
        "tablet",
    }
    ALLOWED_TERM_TYPES = {"IN", "MIN", "PIN", "SCD", "SBD", "GPCK", "BPCK"}

    def __init__(self, matcher: Any | None) -> None:
        self.matcher = matcher

    def build_candidates(
        self, mention: NormalizedDrugMention
    ) -> list[RxNavResolutionCandidate]:
        if mention.normalized_name in self.BROAD_CATEGORIES:
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
        candidates: list[RxNavResolutionCandidate] = []
        query_tokens = mention.normalized_name.split()
        for key, payload in index.items():
            key_tokens = key.split()
            if not key_tokens or key_tokens != query_tokens[: len(key_tokens)]:
                continue
            suffix_tokens = query_tokens[len(key_tokens) :]
            if not self.allow_catalog_prefix_match(suffix_tokens, payload):
                continue
            candidates.append(self._candidate_from_payload(payload, "guarded_prefix_catalog_match"))
        return candidates[:4]

    def allow_catalog_prefix_match(
        self,
        suffix_tokens: list[str],
        payload: tuple[dict[str, Any], bool, str],
    ) -> bool:
        entry, _matched_is_synonym, _matched_value = payload
        if not suffix_tokens:
            return False
        if any(token not in self.FORM_SUFFIX_TOKENS for token in suffix_tokens):
            return False
        if not coerce_text(entry.get("rxcui")):
            return False
        term_type = coerce_text(entry.get("term_type"))
        return term_type is None or term_type.upper() in self.ALLOWED_TERM_TYPES

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

    @staticmethod
    def _classify_alias(entry: dict[str, Any], matched_value: str) -> str:
        normalized = normalize_drug_query_name(matched_value)
        brand_values = entry.get("brand_names", []) or []
        if isinstance(brand_values, str):
            brand_values = [brand_values]
        if normalized and any(
            normalize_drug_query_name(str(value)) == normalized for value in brand_values
        ):
            return "brand"
        return "ingredient"
