from __future__ import annotations

import re
from functools import lru_cache

from services.catalogs.runtime import get_reference_catalog_snapshot

SUPPORTED_REPORT_LANGUAGES: tuple[str, ...] = ("en", "it", "de", "fr", "es")
TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
MISSING_VISIT_LABEL_BY_LANGUAGE: dict[str, str] = {
    "en": "Not provided",
    "it": "Non disponibile",
    "de": "Nicht angegeben",
    "fr": "Non renseignée",
    "es": "No proporcionada",
}

LANGUAGE_HINTS: dict[str, set[str]] = {}
LANGUAGE_FUNCTION_HINTS: dict[str, set[str]] = {}
LANGUAGE_PHRASE_HINTS: dict[str, tuple[str, ...]] = {}
LANGUAGE_DIACRITIC_HINTS: dict[str, set[str]] = {}


@lru_cache(maxsize=1)
def _catalog_language_hints() -> dict[str, set[str]]:
    snapshot = get_reference_catalog_snapshot()
    result: dict[str, set[str]] = {}
    for lang in SUPPORTED_REPORT_LANGUAGES:
        values = snapshot.values("language_detection", "language_hints", key=lang)
        if values:
            result[lang] = {value.casefold() for value in values if value.strip()}
    return result


@lru_cache(maxsize=1)
def _catalog_phrase_hints() -> dict[str, tuple[str, ...]]:
    snapshot = get_reference_catalog_snapshot()
    result: dict[str, tuple[str, ...]] = {}
    for lang in SUPPORTED_REPORT_LANGUAGES:
        values = snapshot.values(
            "language_detection",
            "report_language_phrase_markers",
            key=lang,
        )
        if values:
            result[lang] = tuple(value.casefold() for value in values if value.strip())
    return result


@lru_cache(maxsize=1)
def _catalog_function_hints() -> dict[str, set[str]]:
    snapshot = get_reference_catalog_snapshot()
    result: dict[str, set[str]] = {}
    for lang in SUPPORTED_REPORT_LANGUAGES:
        values = snapshot.values(
            "language_detection",
            "clinical_language_scoring_terms",
            key=lang,
        )
        if values:
            result[lang] = {value.casefold() for value in values if value.strip()}
    return result


@lru_cache(maxsize=1)
def _catalog_diacritic_hints() -> dict[str, set[str]]:
    snapshot = get_reference_catalog_snapshot()
    result: dict[str, set[str]] = {lang: set() for lang in SUPPORTED_REPORT_LANGUAGES}
    for lang in SUPPORTED_REPORT_LANGUAGES:
        values = snapshot.values(
            "language_detection",
            "diacritic_detection_terms",
            key=lang,
        )
        if values:
            result[lang] = {value for value in values if value.strip()}
    return result


def get_language_hints() -> dict[str, set[str]]:
    return _catalog_language_hints()


def get_language_phrase_hints() -> dict[str, tuple[str, ...]]:
    return _catalog_phrase_hints()


def get_language_function_hints() -> dict[str, set[str]]:
    return _catalog_function_hints()


def get_language_diacritic_hints() -> dict[str, set[str]]:
    return _catalog_diacritic_hints()


VALIDATION_MESSAGE_BUNDLES: dict[str, dict[str, str]] = {
    "en": {
        "missing_anamnesis": "Provide the anamnesis.",
        "missing_visit_date": "Provide the visit date.",
        "missing_timed_drug": (
            "Provide at least one drug with start, stop, or other timing information."
        ),
        "insufficient_labs": (
            "Provide laboratory data sufficient to determine hepatotoxicity pattern, "
            "ideally dated ALT or AST, ALP, and bilirubin."
        ),
    },
    "it": {
        "missing_anamnesis": "Fornire l’anamnesi.",
        "missing_visit_date": "Fornire la data della visita.",
        "missing_timed_drug": (
            "Fornire almeno un farmaco con data di inizio, sospensione o altra "
            "informazione temporale."
        ),
        "insufficient_labs": (
            "Fornire dati laboratoristici sufficienti per determinare il pattern di "
            "epatotossicità, idealmente ALT o AST datati, ALP e bilirubina."
        ),
    },
    "de": {
        "missing_anamnesis": "Bitte Anamnese angeben.",
        "missing_visit_date": "Bitte Besuchsdatum angeben.",
        "missing_timed_drug": (
            "Bitte mindestens ein Arzneimittel mit Start-, Stopp- oder anderen "
            "Zeitangaben angeben."
        ),
        "insufficient_labs": (
            "Bitte ausreichend Laborwerte zur Bestimmung des Hepatotoxizitätsmusters "
            "angeben, idealerweise datiertes ALT oder AST, ALP und Bilirubin."
        ),
    },
    "fr": {
        "missing_anamnesis": "Veuillez fournir l’anamnèse.",
        "missing_visit_date": "Veuillez fournir la date de visite.",
        "missing_timed_drug": (
            "Veuillez fournir au moins un médicament avec une date de début, d’arrêt "
            "ou une autre information temporelle."
        ),
        "insufficient_labs": (
            "Veuillez fournir des données biologiques suffisantes pour déterminer le "
            "profil d’hépatotoxicité, idéalement ALT ou AST datés, PAL et bilirubine."
        ),
    },
    "es": {
        "missing_anamnesis": "Proporcione la anamnesis.",
        "missing_visit_date": "Proporcione la fecha de la visita.",
        "missing_timed_drug": (
            "Proporcione al menos un fármaco con fecha de inicio, suspensión u otra "
            "información temporal."
        ),
        "insufficient_labs": (
            "Proporcione datos de laboratorio suficientes para determinar el patrón de "
            "hepatotoxicidad, idealmente ALT o AST con fecha, FA y bilirrubina."
        ),
    },
}


def resolve_supported_language_code(language: str | None) -> str:
    normalized = (language or "").strip().casefold()
    if not normalized:
        return "en"
    for code in SUPPORTED_REPORT_LANGUAGES:
        if normalized.startswith(code):
            return code
    return "en"
