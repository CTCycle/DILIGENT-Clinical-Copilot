from __future__ import annotations

from domain.clinical.entities import (
    ClinicalLabEntry,
    DrugRucamAssessment,
    PatientLabTimeline,
)
from services.clinical.hepatox_core import HepatoxConsultation
from services.clinical.report_language import phrase, report_heading, rucam_summary_text


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
