# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import pytest
from common.utils.languages import DETECTABLE_REPORT_LANGUAGES, SUPPORTED_REPORT_LANGUAGES, resolve_supported_language_code
from domain.clinical import DrugClinicalAssessment, DrugRucamAssessment, HepatotoxicityPatternScore, PatientData, PipelineIssue
from domain.clinical.entities import ClinicalLabEntry, PatientLabTimeline, RagDocumentReference
from services.clinical.language import detect_clinical_language
from services.clinical.hepatox_scoring import is_materially_in_report_language
from services.clinical.report_finalizer import ReportFinalizer
from services.clinical.report_language import phrase, report_heading, resolve_report_language, rucam_summary_text
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

def test_narrative_builder_combines_pipeline_and_rucam_warnings() -> None:
    report = NarrativeBuilder.build_patient_narrative(
        patient_label="Patient",
        visit_label="2026-08-12",
        anamnesis="Clinical context.",
        drugs_text="Levothyroxine daily.",
        pattern_score=None,
        pattern_strings={},
        detected_drugs=["levothyroxine"],
        anamnesis_detected_drugs=[],
        report_language="en",
        issues=[
            PipelineIssue(
                severity="warning",
                code="rxnav_alias_not_validated",
                field="matched_drugs",
                message="RxNav alias was not validated for levothyroxine.",
            )
        ],
        rucam_assessments=[
            DrugRucamAssessment(
                drug_name="levothyroxine",
                total_score=-3,
                causality_category="excluded",
                confidence="low",
            )
        ],
        final_report="Clinical report.",
    )

    assert report.count("## Warnings") == 1
    assert "## RUCAM Evidence" in report
    assert "criterion-level RUCAM evidence captured" in report
    assert "## Estimated RUCAM" not in report
    assert "score -3" not in report
    assert "RxNav alias was not validated" in report
    assert "Structured RUCAM classifies levothyroxine as excluded" in report

def test_italian_clinician_report_wrappers_do_not_use_english_labels() -> None:
    rendered = ReportFinalizer().render_matched_drug_section(
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
    assert "**Commento clinico**" in rendered
    assert "Corrispondenza con evidenza locale" in rendered
    assert "record associato" in rendered
    assert "Claim review" not in rendered
    assert "Corrispondenza evidenza" not in rendered
    assert "Evidence match" not in rendered
    assert "Matched local record" not in rendered
    assert "Bibliography source" not in rendered
    assert rendered.index("**Commento clinico**") < rendered.index("**RUCAM**")
    assert rendered.index("**RUCAM**") < rendered.index("**Report**")

def test_report_heading_labels_exist_in_selected_language() -> None:
    assert report_heading("report_section_summary", "it")
    assert report_heading("report_section_per_drug", "it")
    assert phrase("bibliography_source", "it")
    assert phrase("case_summary", "en")
    assert phrase("laboratory_history", "en")
    assert phrase("not_calculated_insufficient_data", "en")

def test_missing_data_labels_are_stable() -> None:
    assert phrase("not_available", "en") == "not available"
    assert phrase("none", "en") == "None"
    assessment = DrugRucamAssessment(
        drug_name="Drug A",
        total_score=None,
        causality_category="not assessable",
        calculation_method="not_calculated",
    )
    assert "RUCAM" in rucam_summary_text(assessment, "en")

def test_deterministic_laboratory_section_rendering() -> None:
    finalizer = ReportFinalizer()
    section = finalizer.render_laboratory_section(
        PatientLabTimeline(
            entries=[
                ClinicalLabEntry(
                    marker_name="ALT",
                    value=210,
                    unit="U/L",
                    source="laboratory_analysis",
                )
            ]
        ),
        "en",
    )
    assert "## Laboratory history" in section
    assert "ALT: 210.0 U/L" in section

def test_deterministic_bibliography_section_rendering() -> None:
    finalizer = ReportFinalizer()
    section = finalizer.render_bibliography_section(
        [
            {
                "matched_livertox_name": "Amoxicillin",
                "match_strategy": "cache",
                "rxnav_validated": True,
            }
        ],
        "en",
    )
    assert "## Bibliography" in section
    assert "Amoxicillin" in section

def test_rag_bibliography_section_deduplicates_and_merges_ranges() -> None:
    finalizer = ReportFinalizer()
    section = finalizer.build_rag_bibliography_section(
        [
            DrugClinicalAssessment(
                drug_name="Drug A",
                rag_references=[
                    RagDocumentReference(
                        file_name="alpha.pdf", page_start=2, page_end=2
                    ),
                    RagDocumentReference(
                        file_name="alpha.pdf", page_start=3, page_end=3
                    ),
                    RagDocumentReference(
                        file_name="alpha.pdf", page_start=4, page_end=4
                    ),
                    RagDocumentReference(
                        file_name="alpha.pdf", page_start=4, page_end=4
                    ),
                    RagDocumentReference(
                        file_name="alpha.pdf", page_start=7, page_end=7
                    ),
                ],
            )
        ],
        report_language="en",
    )

    assert section is not None
    assert "## Bibliography" in section
    assert "- alpha.pdf, pp. 2-4, 7" in section

def test_rag_bibliography_uses_location_not_available_when_missing() -> None:
    finalizer = ReportFinalizer()
    section = finalizer.build_rag_bibliography_section(
        [
            DrugClinicalAssessment(
                drug_name="Drug A",
                rag_references=[
                    RagDocumentReference(file_name="alpha.pdf"),
                ],
            )
        ],
        report_language="en",
    )

    assert section is not None
    assert "- alpha.pdf, location not available" in section

def test_rag_bibliography_omits_raw_retrieved_text() -> None:
    finalizer = ReportFinalizer()
    entry = DrugClinicalAssessment(
        drug_name="Drug A",
        extracted_excerpts=[
            "Raw retrieved text that must stay out of the bibliography."
        ],
        rag_references=[
            RagDocumentReference(file_name="alpha.pdf", page_start=9, page_end=9),
        ],
    )

    section = finalizer.build_rag_bibliography_section([entry], report_language="en")

    assert section is not None
    assert "Raw retrieved text" not in section
    assert "- alpha.pdf, p. 9" in section

def test_rag_bibliography_absent_when_no_references_exist() -> None:
    finalizer = ReportFinalizer()
    section = finalizer.build_rag_bibliography_section(
        [DrugClinicalAssessment(drug_name="Drug A")],
        report_language="en",
    )

    assert section is None

def test_required_phrase_keys_exist_for_supported_languages() -> None:
    keys = {
        "rucam_source_reported",
        "rucam_structured_score",
        "rucam_not_calculated",
        "rucam_insufficient_data",
        "rucam_score_source",
        "rucam_causality_category",
        "rucam_limitations",
        "livertox_missing",
        "livertox_ambiguous",
        "unresolved_mentions",
        "evidence_quality",
        "report_section_summary",
        "report_section_per_drug",
    }
    for lang in SUPPORTED_REPORT_LANGUAGES:
        for key in keys:
            if key == "rucam_structured_score":
                assert phrase(key, lang, score=6, category="probable")
            else:
                assert phrase(key, lang)

def test_rucam_summary_text_returns_localized_or_safe_text() -> None:
    assessment = DrugRucamAssessment(
        drug_name="A",
        total_score=6,
        causality_category="probable",
        calculation_method="structured_rucam",
        estimated=False,
    )
    for lang in ("it", "es", "fr", "de", "pt"):
        text = rucam_summary_text(assessment, lang)
        assert "6" in text

def test_unsupported_language_code_resolves_to_english() -> None:
    assert resolve_report_language("xx") == "en"

def test_portuguese_is_renderable_but_not_deterministically_detectable() -> None:
    assert resolve_supported_language_code("pt-BR") == "pt"
    assert "pt" in SUPPORTED_REPORT_LANGUAGES
    assert "pt" not in DETECTABLE_REPORT_LANGUAGES

def test_missing_phrase_key_raises_deterministic_error() -> None:
    with pytest.raises(KeyError):
        phrase("missing_key", "en")

def test_material_report_language_detection_executes_for_non_english() -> None:
    assert is_materially_in_report_language(
        "La valutazione clinica richiede monitoraggio epatico.", "it"
    )
