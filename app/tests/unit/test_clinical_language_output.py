from __future__ import annotations

from domain.clinical import (
    DrugClinicalAssessment,
    HepatotoxicityPatternScore,
    PatientData,
)
from services.clinical.hepatox_core import HepatoxConsultation
from services.clinical.language import detect_clinical_language
from services.clinical.validation import build_validation_bundle
from services.session.session_shared import NarrativeBuilder


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


def test_italian_clinician_report_wrappers_do_not_use_english_labels() -> None:
    consultation = HepatoxConsultation.__new__(HepatoxConsultation)
    rendered = consultation.render_matched_drug_section(
        DrugClinicalAssessment(
            drug_name="Paracetamolo",
            match_status="matched",
            matched_livertox_row={"likelihood_score": "A"},
            evidence_quality="alta",
            paragraph="Valutazione clinica del farmaco.",
        ),
        report_language="it",
    )

    assert "**Report**" in rendered
    assert "**Fonte bibliografica**" in rendered
    assert "**Corrispondenza evidenza**" in rendered
    assert "Record locale associato" in rendered
    assert "Evidence match" not in rendered
    assert "Matched local record" not in rendered
    assert "Bibliography source" not in rendered

