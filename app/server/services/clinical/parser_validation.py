from __future__ import annotations

from functools import lru_cache
from typing import Any

from services.catalogs.runtime import get_reference_catalog_snapshot


###############################################################################
@lru_cache(maxsize=1)
def _get_parser_validation_data() -> dict[str, Any]:
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


###############################################################################
def __getattr__(name: str) -> Any:
    data = _get_parser_validation_data()
    if name in data:
        return data[name]
    msg = f"module 'parser_validation' has no attribute '{name}'"
    raise AttributeError(msg)
