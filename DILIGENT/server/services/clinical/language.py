from __future__ import annotations

from dataclasses import dataclass

from DILIGENT.server.domain.clinical import PatientData


@dataclass(frozen=True)
class LanguageDetectionResult:
    detected_input_language: str
    report_language: str
    confidence: str


ITALIAN_HINTS = {
    "paziente",
    "anamnesi",
    "farmaco",
    "farmaci",
    "visita",
    "bilirubina",
    "ittero",
    "sospeso",
    "sospensione",
    "epatotossicita",
    "epatotossicità",
    "dolore",
    "fegato",
}


def detect_clinical_language(payload: PatientData) -> LanguageDetectionResult:
    scored_sections = [
        payload.anamnesis or "",
        payload.laboratory_analysis or "",
        payload.drugs or "",
    ]
    weighted_text = " ".join(section.strip() for section in scored_sections if section.strip())
    if not weighted_text:
        return LanguageDetectionResult(
            detected_input_language="en",
            report_language="en",
            confidence="low",
        )
    tokenized = [token.strip(".,;:()[]{}!?\"'").lower() for token in weighted_text.split()]
    italian_hits = sum(1 for token in tokenized if token in ITALIAN_HINTS)
    if italian_hits >= 2:
        return LanguageDetectionResult(
            detected_input_language="it",
            report_language="it",
            confidence="high",
        )
    return LanguageDetectionResult(
        detected_input_language="en",
        report_language="en",
        confidence="moderate",
    )

