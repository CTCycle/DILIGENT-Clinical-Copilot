from __future__ import annotations

from domain.clinical.entities import (
    ClinicalLabEntry,
    DrugRucamAssessment,
    DrugClinicalAssessment,
    PatientLabTimeline,
    RagDocumentReference,
)
from services.clinical.hepatox_core import HepatoxConsultation
from services.clinical.report_language import phrase, report_heading, rucam_summary_text
from services.clinical.report_finalizer import ReportFinalizer

###############################################################################
def test_report_heading_labels_exist_in_selected_language() -> None:
    assert report_heading("report_section_summary", "it")
    assert report_heading("report_section_per_drug", "it")
    assert phrase("bibliography_source", "it")
    assert phrase("case_summary", "en")
    assert phrase("laboratory_history", "en")
    assert phrase("not_calculated_insufficient_data", "en")

###############################################################################
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

###############################################################################
def test_deterministic_laboratory_section_rendering() -> None:
    consultation = HepatoxConsultation.__new__(HepatoxConsultation)
    section = consultation.render_laboratory_section(
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

###############################################################################
def test_deterministic_bibliography_section_rendering() -> None:
    consultation = HepatoxConsultation.__new__(HepatoxConsultation)
    section = consultation.render_bibliography_section(
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

###############################################################################
def test_rag_bibliography_section_deduplicates_and_merges_ranges() -> None:
    finalizer = ReportFinalizer(object())
    section = finalizer.build_rag_bibliography_section(
        [
            DrugClinicalAssessment(
                drug_name="Drug A",
                rag_references=[
                    RagDocumentReference(file_name="alpha.pdf", page_start=2, page_end=2),
                    RagDocumentReference(file_name="alpha.pdf", page_start=3, page_end=3),
                    RagDocumentReference(file_name="alpha.pdf", page_start=4, page_end=4),
                    RagDocumentReference(file_name="alpha.pdf", page_start=4, page_end=4),
                    RagDocumentReference(file_name="alpha.pdf", page_start=7, page_end=7),
                ],
            )
        ],
        report_language="en",
    )

    assert section is not None
    assert "## Bibliography" in section
    assert "- alpha.pdf, pp. 2-4, 7" in section

###############################################################################
def test_rag_bibliography_uses_page_not_available_when_missing() -> None:
    finalizer = ReportFinalizer(object())
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
    assert "- alpha.pdf, page not available" in section

###############################################################################
def test_rag_bibliography_omits_raw_retrieved_text() -> None:
    finalizer = ReportFinalizer(object())
    entry = DrugClinicalAssessment(
        drug_name="Drug A",
        extracted_excerpts=["Raw retrieved text that must stay out of the bibliography."],
        rag_references=[
            RagDocumentReference(file_name="alpha.pdf", page_start=9, page_end=9),
        ],
    )

    section = finalizer.build_rag_bibliography_section([entry], report_language="en")

    assert section is not None
    assert "Raw retrieved text" not in section
    assert "- alpha.pdf, p. 9" in section

###############################################################################
def test_rag_bibliography_absent_when_no_references_exist() -> None:
    finalizer = ReportFinalizer(object())
    section = finalizer.build_rag_bibliography_section(
        [DrugClinicalAssessment(drug_name="Drug A")],
        report_language="en",
    )

    assert section is None
