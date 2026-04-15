from __future__ import annotations

from DILIGENT.server.api.session import NarrativeBuilder
from DILIGENT.server.domain.clinical import HepatotoxicityPatternScore
from DILIGENT.server.services.clinical.language import detect_clinical_language
from DILIGENT.server.services.clinical.validation import build_validation_bundle
from DILIGENT.server.domain.clinical import PatientData


def test_italian_input_yields_italian_language_detection_and_messages() -> None:
    payload = PatientData(
        anamnesis="Paziente con ittero e dolore addominale.",
        drugs="Paracetamolo sospeso dal 2025-01-10",
        laboratory_analysis="Bilirubina totale 3.1 mg/dL, ALT 220 U/L",
    )
    detected = detect_clinical_language(payload)
    assert detected.report_language == "it"
    bundle = build_validation_bundle(detected.report_language)
    assert bundle.missing_anamnesis.startswith("Fornire")


def test_english_input_yields_english_output_bundle() -> None:
    payload = PatientData(
        anamnesis="Patient with jaundice and dark urine.",
        drugs="Acetaminophen started on 2025-01-01",
        laboratory_analysis="ALT 220 U/L, ALP 140 U/L, bilirubin 2.0 mg/dL",
    )
    detected = detect_clinical_language(payload)
    assert detected.report_language == "en"
    bundle = build_validation_bundle(detected.report_language)
    assert bundle.missing_visit_date.startswith("Provide")


def test_mixed_input_prefers_dominant_section() -> None:
    payload = PatientData(
        anamnesis="Paziente con anamnesi positiva per steatosi epatica e ittero.",
        drugs="Acetaminophen started on 2025-01-01",
        laboratory_analysis="ALT 220 U/L ALP 140 U/L",
    )
    detected = detect_clinical_language(payload)
    assert detected.report_language == "it"


def test_narrative_builder_does_not_force_english_for_italian() -> None:
    report = NarrativeBuilder.build_patient_narrative(
        patient_label="Mario Rossi",
        visit_label="2025-01-12",
        anamnesis="Paziente con ittero.",
        drugs_text="Paracetamolo sospeso",
        pattern_score=HepatotoxicityPatternScore(classification="mixed", r_score=2.5),
        pattern_strings={"r_score": "2.5"},
        detected_drugs=["Paracetamolo"],
        anamnesis_detected_drugs=["Ibuprofene"],
        report_language="it",
        issues=[],
        final_report="Report clinico finale.",
    )
    assert "# Sintesi Visita Clinica" in report
    assert "## Report Clinico" in report

