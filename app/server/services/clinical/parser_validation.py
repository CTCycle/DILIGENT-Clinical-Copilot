from __future__ import annotations

from services.catalogs.runtime import get_reference_catalog_snapshot

_snapshot = get_reference_catalog_snapshot()
NON_DRUG_EXACT_NAMES = set(
    _snapshot.values("clinical_extraction", "drug_non_name_exact", key="default")
)
NON_DRUG_EXACT_NAMES.update(_snapshot.values("text_normalization", "drug_non_mentions"))
NON_DRUG_PREFIXES = tuple(
    _snapshot.values("clinical_extraction", "drug_non_name_prefixes", key="default")
)
NON_DRUG_CONTAINS = tuple(
    _snapshot.values("clinical_extraction", "drug_non_name_contains", key="default")
)
WEEKDAY_TOKENS = set(
    _snapshot.values("clinical_extraction", "weekday_terms", key="default")
)
NON_THERAPY_LINE_PREFIXES = tuple(
    _snapshot.values("clinical_extraction", "drug_line_prefixes")
)
