from __future__ import annotations

import re
from typing import Any

from common.utils.text_utils import coerce_text
from domain.clinical.drug_resolution import DrugIdentityCandidate
from services.catalogs.runtime import get_reference_catalog_snapshot
from services.text.normalization import (
    canonicalize_drug_query,
    normalize_drug_query_name,
)

###############################################################################
class DrugIdentityResolver:
    """Resolve extracted labels to local, source-backed drug identity candidates."""

    # -------------------------------------------------------------------------
    def __init__(self, matcher: Any | None = None) -> None:
        self.matcher = matcher
        snapshot = get_reference_catalog_snapshot()
        self.broad_catalog_matches = set(
            snapshot.values("drug_matching", "broad_drug_categories")
        )
        self.reducible_suffixes = set(
            snapshot.values("drug_matching", "safe_query_reduction_suffixes")
        )

    # -------------------------------------------------------------------------
    def resolve(self, source_label: str) -> list[DrugIdentityCandidate]:
        label = (source_label or "").strip()
        if not label:
            return []
        candidates: list[DrugIdentityCandidate] = []
        self._add_candidate(
            candidates,
            source_label=label,
            value=canonicalize_drug_query(label) or label,
            kind="extracted_label",
            confidence=0.5,
            notes=("raw_extracted_label",),
        )
        for component in self.split_components(label):
            if component != label:
                self._add_candidate(
                    candidates,
                    source_label=label,
                    value=canonicalize_drug_query(component) or component,
                    kind="regimen_component",
                    confidence=0.72,
                    notes=("component_from_extracted_label",),
                )
                if self.matcher is not None:
                    self._add_matcher_candidates(candidates, component, label)
        if self.matcher is not None:
            self._add_matcher_candidates(candidates, label)
        return self._dedupe(candidates)

    # -------------------------------------------------------------------------
    def _add_matcher_candidates(
        self,
        candidates: list[DrugIdentityCandidate],
        label: str,
        source_label: str | None = None,
    ) -> None:
        candidate_source_label = source_label or label
        normalized_label = normalize_drug_query_name(label)
        lookup = getattr(self.matcher, "lookup", None)
        if lookup is None:
            return
        for value, kind, confidence, note in self._catalog_values(
            lookup, normalized_label
        ):
            self._add_candidate(
                candidates,
                source_label=candidate_source_label,
                value=value,
                kind=kind,
                confidence=confidence,
                notes=(note,),
            )
            for (
                livertox_value,
                livertox_kind,
                livertox_confidence,
                livertox_note,
            ) in self._livertox_values(lookup, value):
                if livertox_value.casefold() in self.broad_catalog_matches:
                    continue
                self._add_candidate(
                    candidates,
                    source_label=candidate_source_label,
                    value=livertox_value,
                    kind=livertox_kind,
                    confidence=max(confidence, livertox_confidence),
                    notes=(note, livertox_note),
                )
        for value, kind, confidence, note in self._livertox_values(lookup, label):
            self._add_candidate(
                candidates,
                source_label=candidate_source_label,
                value=value,
                kind=kind,
                confidence=confidence,
                notes=(note,),
            )

    # -------------------------------------------------------------------------
    def _catalog_values(
        self,
        lookup: Any,
        normalized_label: str,
    ) -> list[tuple[str, str, float, str]]:
        matches = self._find_catalog_matches(lookup, normalized_label)
        values: list[tuple[str, str, float, str]] = []
        for entry, matched_is_synonym, matched_value in matches:
            if matched_value:
                kind = "catalog_ingredient" if matched_is_synonym else "catalog_brand"
                values.append((matched_value, kind, 0.9, "catalog_alias_match"))
            for field_name, kind in (
                ("name", "catalog_ingredient"),
                ("raw_name", "catalog_ingredient"),
            ):
                value = coerce_text(entry.get(field_name))
                if self._is_real_catalog_value(value):
                    values.append((str(value), kind, 0.82, f"catalog_{field_name}"))
            for brand in entry.get("brand_names", []) or []:
                if self._is_real_catalog_value(brand):
                    values.append((str(brand), "catalog_brand", 0.78, "catalog_brand"))
            for alias in entry.get("fallback_aliases", []) or []:
                if self._is_real_catalog_value(alias):
                    values.append(
                        (str(alias), "catalog_ingredient", 0.76, "catalog_fallback")
                    )
        return values

    # -------------------------------------------------------------------------
    @staticmethod
    def _is_real_catalog_value(value: Any) -> bool:
        text = coerce_text(value)
        return bool(text and text.casefold() != "nan")

    # -------------------------------------------------------------------------
    def _find_catalog_matches(
        self,
        lookup: Any,
        normalized_label: str,
    ) -> list[tuple[dict[str, Any], bool, str]]:
        if not normalized_label:
            return []
        index = getattr(lookup, "catalog_global_index", {}) or {}
        direct = index.get(normalized_label)
        if direct is not None:
            return [direct]
        query_tokens = normalized_label.split()
        matches: list[tuple[int, tuple[dict[str, Any], bool, str]]] = []
        for key, payload in index.items():
            key_tokens = key.split()
            if not key_tokens:
                continue
            if key_tokens == query_tokens[: len(key_tokens)]:
                matches.append((len(key_tokens), payload))
        matches.sort(key=lambda item: item[0], reverse=True)
        return [payload for _score, payload in matches[:4]]

    # -------------------------------------------------------------------------
    def _livertox_values(
        self,
        lookup: Any,
        label: str,
    ) -> list[tuple[str, str, float, str]]:
        values: list[tuple[str, str, float, str]] = []
        for query in self._query_reductions(label):
            if self.matcher is not None:
                match = self.matcher.match_drug_names([query])[0]
                if match.status == "matched" and match.matched_name:
                    values.append(
                        (
                            match.matched_name,
                            "livertox_alias",
                            max(float(match.confidence or 0.0), 0.86),
                            f"livertox_match_from='{query}'",
                        )
                    )
            canonical = canonicalize_drug_query(query)
            normalized = normalize_drug_query_name(canonical)
            if not normalized:
                continue
            for record, _confidence, notes in lookup.match_normalized_all(normalized):
                kind = "livertox_alias" if notes else "livertox_primary"
                values.append((record.drug_name, kind, 0.95, "livertox_exact"))
            for (
                record,
                _confidence,
                _notes,
            ) in lookup.match_authoritative_spelling_candidates(normalized):
                values.append(
                    (
                        record.drug_name,
                        "livertox_primary",
                        0.84,
                        f"livertox_spelling_from='{normalized}'",
                    )
                )
        canonical = canonicalize_drug_query(label)
        tokens = canonical.split()
        while len(tokens) > 1:
            tokens = tokens[:-1]
            normalized_prefix = normalize_drug_query_name(" ".join(tokens))
            primary_keys = set(
                lookup.require_data().primary_index.get(normalized_prefix, [])
            )
            exact_primary = [
                record
                for record, _confidence, _notes in lookup.match_normalized_all(
                    normalized_prefix
                )
                if record.stable_key in primary_keys
                and record.normalized_name == normalized_prefix
            ]
            if len(exact_primary) == 1:
                values.append(
                    (
                        exact_primary[0].drug_name,
                        "livertox_primary",
                        0.88,
                        f"livertox_primary_prefix_from='{canonical}'",
                    )
                )
                break
        return values

    # -------------------------------------------------------------------------
    def _query_reductions(self, label: str) -> list[str]:
        canonical = canonicalize_drug_query(label)
        values = [canonical] if canonical else []
        tokens = canonical.split()
        while len(tokens) > 1 and tokens[-1] in self.reducible_suffixes:
            tokens = tokens[:-1]
            values.append(" ".join(tokens))
        return [value for value in dict.fromkeys(values) if value]

    # -------------------------------------------------------------------------
    def _add_candidate(
        self,
        candidates: list[DrugIdentityCandidate],
        *,
        source_label: str,
        value: str,
        kind: str,
        confidence: float,
        notes: tuple[str, ...],
    ) -> None:
        canonical = canonicalize_drug_query(value) or value.strip().lower()
        normalized = normalize_drug_query_name(canonical)
        if not normalized:
            return
        candidates.append(
            DrugIdentityCandidate(
                source_label=source_label,
                canonical_candidate=canonical,
                normalized_candidate=normalized,
                kind=kind,
                confidence=confidence,
                notes=notes,
            )
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _dedupe(
        candidates: list[DrugIdentityCandidate],
    ) -> list[DrugIdentityCandidate]:
        best: dict[str, DrugIdentityCandidate] = {}
        for candidate in candidates:
            existing = best.get(candidate.normalized_candidate)
            if existing is None or candidate.confidence > existing.confidence:
                best[candidate.normalized_candidate] = candidate
        return sorted(
            best.values(),
            key=lambda item: (-item.confidence, item.kind, item.normalized_candidate),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def split_components(value: str) -> list[str]:
        text = (value or "").strip()
        if not text:
            return []
        numeric_slash = "__NUMERIC_SLASH__"
        protected = text.replace("IU/ml", "IU per ml").replace("IU/ML", "IU per ML")
        protected = protected.replace("mg/ml", "mg per ml").replace(
            "MG/ML", "MG per ML"
        )
        protected = re.sub(r"(?<=\d)/(?=\d)", numeric_slash, protected)
        protected = re.sub(r"(?<=[A-Za-z]{4})-(?=[A-Za-z]{4})", " + ", protected)
        if "/" in protected:
            protected = protected.replace("/", " / ")
        for separator in (" + ", " plus ", " and "):
            protected = protected.replace(separator, " + ")
        raw_parts = [part.strip(" \t,;:.") for part in protected.split(" + ")]
        if len(raw_parts) == 1:
            raw_parts = [part.strip(" \t,;:.") for part in protected.split(" / ")]
        parts = [
            part.replace(numeric_slash, "/")
            for part in raw_parts
            if DrugIdentityResolver._is_name_component(part)
        ]
        return parts or [text]

    # -------------------------------------------------------------------------
    @staticmethod
    def _is_name_component(value: str) -> bool:
        normalized = normalize_drug_query_name(value)
        if not normalized:
            return False
        tokens = normalized.split()
        if not tokens:
            return False
        return any(
            any(char.isalpha() for char in token) and len(token) > 1 for token in tokens
        )
