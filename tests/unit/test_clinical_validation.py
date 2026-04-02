from __future__ import annotations

from datetime import date

import pytest

from DILIGENT.server.domain.clinical import (
    ClinicalLabEntry,
    ClinicalPipelineValidationError,
    DrugEntry,
    PatientData,
    PatientDrugs,
    PatientLabTimeline,
)
from DILIGENT.server.services.clinical.hepatox import HepatotoxicityPatternAnalyzer
from DILIGENT.server.services.clinical.validation import (
    build_validation_bundle,
    ensure_required_sections,
    ensure_timed_therapy_drug,
)


def test_missing_anamnesis_raises_localized_error() -> None:
    payload = PatientData(visit_date=date(2025, 1, 1), drugs="Drug A")
    bundle = build_validation_bundle("en")
    with pytest.raises(ClinicalPipelineValidationError) as exc_info:
        ensure_required_sections(payload, bundle=bundle)
    assert any(issue.code == "missing_anamnesis" for issue in exc_info.value.issues)


def test_missing_visit_date_raises_localized_error() -> None:
    payload = PatientData(anamnesis="History", drugs="Drug A")
    bundle = build_validation_bundle("en")
    with pytest.raises(ClinicalPipelineValidationError) as exc_info:
        ensure_required_sections(payload, bundle=bundle)
    assert any(issue.code == "missing_visit_date" for issue in exc_info.value.issues)


def test_missing_timed_drug_raises_error() -> None:
    drugs = PatientDrugs(entries=[DrugEntry(name="Drug A", source="therapy")])
    bundle = build_validation_bundle("en")
    with pytest.raises(ClinicalPipelineValidationError) as exc_info:
        ensure_timed_therapy_drug(drugs, bundle=bundle)
    assert any(issue.code == "missing_timed_drug" for issue in exc_info.value.issues)


def test_insufficient_pattern_labs_raise_blocker() -> None:
    analyzer = HepatotoxicityPatternAnalyzer()
    with pytest.raises(ClinicalPipelineValidationError) as exc_info:
        analyzer.assess_payload(PatientLabTimeline(entries=[]))
    assert any(issue.code == "missing_hepatotoxicity_inputs" for issue in exc_info.value.issues)


def test_non_critical_missing_data_does_not_block() -> None:
    payload = PatientData(
        visit_date=date(2025, 1, 1),
        anamnesis="Jaundice after therapy.",
        drugs="Drug A",
    )
    bundle = build_validation_bundle("en")
    ensure_required_sections(payload, bundle=bundle)
    ensure_timed_therapy_drug(
        PatientDrugs(
            entries=[
                DrugEntry(
                    name="Drug A",
                    source="therapy",
                    therapy_start_status=True,
                    therapy_start_date="2024-12-10",
                )
            ]
        ),
        bundle=bundle,
    )
    assessment = HepatotoxicityPatternAnalyzer().assess_payload(
        PatientLabTimeline(
            entries=[
                ClinicalLabEntry(
                    marker_name="ALT",
                    value=200,
                    upper_limit_normal=40,
                    sample_date="2025-01-01",
                    source="laboratory_analysis",
                ),
                ClinicalLabEntry(
                    marker_name="ALP",
                    value=120,
                    upper_limit_normal=120,
                    sample_date="2025-01-01",
                    source="laboratory_analysis",
                ),
            ]
        )
    )
    assert assessment.status == "ok"

