from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextNormalizationSnapshot:
    matching_stopwords: frozenset[str]
    clinical_generic_terms: frozenset[str]
    formulation_stopwords: frozenset[str]
    manufacturer_tokens: frozenset[str]
    manufacturer_suffixes: tuple[str, ...]
    rxnav_salt_stopwords: frozenset[str]
    rxnav_form_stopwords: frozenset[str]
    rxnav_unit_stopwords: frozenset[str]
    rxnav_name_stopwords: frozenset[str]
    trailing_temporal_tokens: frozenset[str]
    query_aliases: dict[str, str]
    drug_non_mentions: frozenset[str]
    drug_duration_words: frozenset[str]
    drug_weekday_words: frozenset[str]
    lab_marker_aliases: dict[str, str]
    brand_combo_preferences: dict[str, str]
    knowledge_source_references: dict[str, str]

    @property
    def rxnav_synonym_stopwords(self) -> frozenset[str]:
        return self.matching_stopwords | self.clinical_generic_terms
