from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any

from services.catalogs.runtime import get_reference_catalog_snapshot

_STATUS_LABEL_RE = re.compile(
    r"^(?:last|first|final|ultima|ultimo|ricevut[oaie]?|"
    r"interrott[oaie]?|iniziat[oaie]?|sospes[oaie]?|termine)"
    r"(?:\s+(?:dose|doses|administration|somministrazione|somministrazioni|"
    r"received|given|il|dal|al))?$",
    re.IGNORECASE,
)
_LEADING_NARRATIVE_RE = re.compile(
    r"^(?:and|or|but|then|on|after|before|no)\b",
    re.IGNORECASE,
)
_NARRATIVE_SUBJECT_RE = re.compile(
    r"^(?:the|this|that|a|an|patient|paziente|subject|soggetto|"
    r"person|persona|case|caso)\b",
    re.IGNORECASE,
)
_NARRATIVE_PREDICATE_RE = re.compile(
    r"\b(?:reports?|denies?|states?|describes?|presents?|experienced?|"
    r"developed?|received?|takes?|took|has|have|had|is|was|were|shows?|"
    r"indicates?|underwent|used?|uses?|notes?|noted|riferisce|nega|"
    r"descrive|presenta|sviluppa|ricev(?:e|uto)|assume|assunto|"
    r"mostra|indica)\b",
    re.IGNORECASE,
)

###############################################################################
def normalize_parser_filter_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()

###############################################################################
def is_sentence_like_non_drug_name(value: str | None) -> bool:
    """Reject multi-token narrative clauses returned as entity names."""
    normalized = normalize_parser_filter_key(str(value or ""))
    if len(normalized.split()) < 4:
        return False
    return bool(
        _NARRATIVE_SUBJECT_RE.match(normalized)
        and _NARRATIVE_PREDICATE_RE.search(normalized)
    )

###############################################################################
def is_obvious_non_drug_name(value: str | None) -> bool:
    """Reject status labels and prose fragments before clinical resolution."""
    raw = str(value or "").strip()
    normalized = normalize_parser_filter_key(raw)
    if not normalized:
        return True
    if re.search(r"[.;:]\s", raw):
        return True
    if _STATUS_LABEL_RE.fullmatch(normalized):
        return True
    if _LEADING_NARRATIVE_RE.match(normalized):
        return True
    if is_sentence_like_non_drug_name(normalized):
        return True
    snapshot = get_reference_catalog_snapshot()
    catalog_names = {
        normalize_parser_filter_key(item)
        for item in (
            snapshot.values("clinical_extraction", "drug_non_name_exact")
            + snapshot.values("text_normalization", "drug_non_mentions")
        )
        if item
    }
    return normalized in catalog_names

###############################################################################
@lru_cache(maxsize=1)
def get_parser_validation_data() -> dict[str, Any]:
    snapshot = get_reference_catalog_snapshot()
    non_drug_exact_names = set(
        snapshot.values("clinical_extraction", "drug_non_name_exact", key="default")
    )
    non_drug_exact_names.update(
        snapshot.values("text_normalization", "drug_non_mentions")
    )
    return {
        "NON_DRUG_EXACT_NAMES": non_drug_exact_names,
        "NON_DRUG_PREFIXES": tuple(
            snapshot.values(
                "clinical_extraction", "drug_non_name_prefixes", key="default"
            )
        ),
        "NON_DRUG_CONTAINS": tuple(
            list(
                snapshot.values(
                    "clinical_extraction", "drug_non_name_contains", key="default"
                )
            )
            + ["obesita"]
        ),
        "WEEKDAY_TOKENS": set(
            snapshot.values("clinical_extraction", "weekday_terms", key="default")
        ),
        "NON_THERAPY_LINE_PREFIXES": tuple(
            snapshot.values("clinical_extraction", "drug_line_prefixes")
        ),
    }
