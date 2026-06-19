from __future__ import annotations

from services.catalogs.runtime import get_reference_catalog_snapshot

_snapshot = get_reference_catalog_snapshot()
DEFAULT_NON_DRUG_EXACT_NAMES = {
    "in riserva",
    "paziente femmina",
    "dopo",
    "il lunedi",
    "ulteriore ciclo originariamente previsto il",
}
DEFAULT_NON_DRUG_PREFIXES = (
    "farmaci non assunti",
    "non assume",
    "nessuna terapia",
)
DEFAULT_NON_DRUG_CONTAINS = (
    "originariamente previsto",
    "peso della paziente",
)
DEFAULT_WEEKDAY_TOKENS = {
    "lunedi",
    "lunedì",
    "martedi",
    "martedì",
    "mercoledi",
    "mercoledì",
    "giovedi",
    "giovedì",
    "venerdi",
    "venerdì",
    "sabato",
    "domenica",
}
DEFAULT_NON_THERAPY_LINE_PREFIXES = (
    "farmaci non assunti",
    "non assunti",
)
NON_DRUG_EXACT_NAMES = set(
    _snapshot.values("clinical_extraction", "drug_non_name_exact", key="default")
)
NON_DRUG_EXACT_NAMES.update(_snapshot.values("text_normalization", "drug_non_mentions"))
NON_DRUG_EXACT_NAMES.update(DEFAULT_NON_DRUG_EXACT_NAMES)
NON_DRUG_PREFIXES = tuple(
    [
        *_snapshot.values(
            "clinical_extraction", "drug_non_name_prefixes", key="default"
        ),
        *DEFAULT_NON_DRUG_PREFIXES,
    ]
)
NON_DRUG_CONTAINS = tuple(
    [
        *_snapshot.values(
            "clinical_extraction", "drug_non_name_contains", key="default"
        ),
        *DEFAULT_NON_DRUG_CONTAINS,
    ]
)
WEEKDAY_TOKENS = set(
    _snapshot.values("clinical_extraction", "weekday_terms", key="default")
)
WEEKDAY_TOKENS.update(DEFAULT_WEEKDAY_TOKENS)
NON_THERAPY_LINE_PREFIXES = tuple(
    [
        *_snapshot.values("clinical_extraction", "drug_line_prefixes"),
        *DEFAULT_NON_THERAPY_LINE_PREFIXES,
    ]
)
