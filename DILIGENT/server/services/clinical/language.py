from __future__ import annotations

from dataclasses import dataclass
import re

from DILIGENT.server.domain.clinical import PatientData


@dataclass(frozen=True)
class LanguageDetectionResult:
    detected_input_language: str
    report_language: str
    confidence: str


SUPPORTED_REPORT_LANGUAGES: tuple[str, ...] = ("en", "it", "de", "fr", "es")
TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
SECTION_WEIGHTS: dict[str, int] = {
    "anamnesis": 3,
    "laboratory_analysis": 2,
    "drugs": 2,
}

LANGUAGE_HINTS: dict[str, set[str]] = {
    "en": {
        "patient", "anamnesis", "drug", "drugs", "visit", "bilirubin",
        "jaundice", "liver", "history", "therapy", "suspended",
    },
    "it": {
        "paziente", "anamnesi", "farmaco", "farmaci", "visita", "bilirubina",
        "ittero", "fegato", "terapia", "sospeso", "sospensione",
        "epatotossicita", "epatotossicità",
    },
    "de": {
        "patient", "anamnese", "arzneimittel", "medikament", "medikamente",
        "besuch", "bilirubin", "ikterus", "leber", "therapie", "abgesetzt",
    },
    "fr": {
        "patient", "anamnese", "anamnesis", "médicament", "médicaments",
        "visite", "bilirubine", "ictère", "foie", "thérapie", "arrêt",
    },
    "es": {
        "paciente", "anamnesis", "anamnesis", "fármaco", "fármacos",
        "medicamento", "medicamentos", "visita", "bilirrubina", "ictericia",
        "hígado", "terapia", "suspendido",
    },
}


def _tokenize(value: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(value)]


def _score_text_by_language(payload: PatientData) -> dict[str, float]:
    scores: dict[str, float] = {code: 0.0 for code in SUPPORTED_REPORT_LANGUAGES}
    sections = {
        "anamnesis": payload.anamnesis or "",
        "laboratory_analysis": payload.laboratory_analysis or "",
        "drugs": payload.drugs or "",
    }
    for section_name, section_text in sections.items():
        weight = float(SECTION_WEIGHTS.get(section_name, 1))
        for token in _tokenize(section_text):
            for lang_code, hints in LANGUAGE_HINTS.items():
                if token in hints:
                    scores[lang_code] += weight
    return scores


def detect_clinical_language(payload: PatientData) -> LanguageDetectionResult:
    scored_sections = [payload.anamnesis, payload.laboratory_analysis, payload.drugs]
    weighted_text = " ".join(section.strip() for section in scored_sections if section and section.strip())
    if not weighted_text:
        return LanguageDetectionResult(
            detected_input_language="en",
            report_language="en",
            confidence="low",
        )
    scores = _score_text_by_language(payload)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_language, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - second_score

    if best_score < 2.0:
        return LanguageDetectionResult(
            detected_input_language="en",
            report_language="en",
            confidence="low",
        )
    if best_score >= 8.0 and margin >= 3.0:
        confidence = "high"
    elif best_score >= 4.0 and margin >= 1.0:
        confidence = "moderate"
    else:
        confidence = "low"
    return LanguageDetectionResult(
        detected_input_language=best_language,
        report_language=best_language if best_language in SUPPORTED_REPORT_LANGUAGES else "en",
        confidence=confidence,
    )
