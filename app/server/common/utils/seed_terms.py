from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any


class SeedTermCatalog(tuple):
    __slots__ = ()

    def __new__(
        cls, *, keywords: set[str], stopwords: set[str], groups: dict[str, set[str]]
    ):
        return super().__new__(cls, (keywords, stopwords, groups))

    @property
    def keywords(self) -> set[str]:
        return self[0]

    @property
    def stopwords(self) -> set[str]:
        return self[1]

    @property
    def groups(self) -> dict[str, set[str]]:
        return self[2]


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().strip()


def load_seed_term_catalog() -> SeedTermCatalog:
    path = (
        Path(__file__).resolve().parents[3]
        / "resources"
        / "catalogs"
        / "text_normalization.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    keywords: set[str] = set()
    stopwords: set[str] = set()
    groups: dict[str, set[str]] = {}
    for entry in entries:
        category = str(entry.get("category") or "")
        values = entry.get("values") or []
        normalized_values = {_normalize(str(v)) for v in values if str(v).strip()}
        if not normalized_values:
            continue
        groups.setdefault(category, set()).update(normalized_values)
        if "stopword" in category:
            stopwords.update(normalized_values)
        else:
            keywords.update(normalized_values)
    return SeedTermCatalog(keywords=keywords, stopwords=stopwords, groups=groups)


def detect_seed_matches(text: str, catalog: SeedTermCatalog) -> dict[str, Any]:
    normalized = f" {_normalize(text)} "
    matched_keywords = sorted(
        {term for term in catalog.keywords if f" {term} " in normalized}
    )
    matched_stopwords = sorted(
        {term for term in catalog.stopwords if f" {term} " in normalized}
    )
    matched_terms = sorted(set(matched_keywords) | set(matched_stopwords))
    matched_groups: dict[str, list[str]] = {}
    matched_counts: dict[str, int] = {}
    for group, terms in catalog.groups.items():
        group_matches = sorted({term for term in terms if f" {term} " in normalized})
        if group_matches:
            matched_groups[group] = group_matches
            matched_counts[group] = len(group_matches)
    return {
        "matched_keywords": matched_keywords,
        "matched_stopwords": matched_stopwords,
        "matched_terms": matched_terms,
        "matched_term_groups": matched_groups,
        "matched_term_counts": matched_counts,
    }
