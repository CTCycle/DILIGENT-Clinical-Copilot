from __future__ import annotations

from datetime import date

import pytest
from domain.clinical import (
    ClinicalLabEntry,
    ClinicalPipelineValidationError,
    DrugEntry,
    PatientData,
    PatientDrugs,
    PatientLabTimeline,
)
from services.clinical.pattern_analyzer import HepatotoxicityPatternAnalyzer
from services.clinical.dili_pattern import DiliPatternEngine
from services.clinical.validation import (
    build_validation_bundle,
    ensure_required_sections,
    ensure_timed_therapy_drug,
)


###############################################################################
def test_missing_anamnesis_raises_localized_error() -> None:
    payload = PatientData(visit_date=date(2025, 1, 1), drugs="Drug A")
    bundle = build_validation_bundle("en")
    with pytest.raises(ClinicalPipelineValidationError) as exc_info:
        ensure_required_sections(payload, bundle=bundle)
    assert any(issue.code == "missing_anamnesis" for issue in exc_info.value.issues)


###############################################################################
def test_missing_visit_date_raises_localized_error() -> None:
    payload = PatientData(anamnesis="History", drugs="Drug A")
    bundle = build_validation_bundle("en")
    with pytest.raises(ClinicalPipelineValidationError) as exc_info:
        ensure_required_sections(payload, bundle=bundle)
    assert any(issue.code == "missing_visit_date" for issue in exc_info.value.issues)


###############################################################################
def test_missing_timed_drug_raises_error() -> None:
    drugs = PatientDrugs(entries=[DrugEntry(name="Drug A", source="therapy")])
    bundle = build_validation_bundle("en")
    with pytest.raises(ClinicalPipelineValidationError) as exc_info:
        ensure_timed_therapy_drug(drugs, bundle=bundle)
    assert any(issue.code == "missing_timed_drug" for issue in exc_info.value.issues)


###############################################################################
def test_drug_schedule_counts_as_timing_information() -> None:
    drugs = PatientDrugs(
        entries=[
            DrugEntry(
                name="Levetiracetam",
                source="therapy",
                administration_pattern="1-0-0-1",
                daytime_administration=[1, 0, 0, 1],
            )
        ]
    )
    bundle = build_validation_bundle("en")
    ensure_timed_therapy_drug(drugs, bundle=bundle)


###############################################################################
def test_insufficient_pattern_labs_raise_blocker() -> None:
    analyzer = HepatotoxicityPatternAnalyzer()
    assessment = analyzer.assess_payload(PatientLabTimeline(entries=[]))
    assert assessment.status == "undetermined_due_to_missing_labs"
    assert assessment.score.classification == "indeterminate"
    assert any(
        issue.code == "missing_hepatotoxicity_inputs" for issue in assessment.issues
    )


###############################################################################
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


###############################################################################
def test_case_a_first_abnormal_pair_is_presentation_anchor_and_peak_is_retained() -> None:
    entries: list[ClinicalLabEntry] = []
    for sample_date, alt, alp in (
        ("2026-01-04", 22, 82),
        ("2026-01-28", 800, 160),
        ("2026-02-04", 420, 145),
        ("2026-02-18", 95, 118),
    ):
        entries.extend(
            [
                ClinicalLabEntry(
                    marker_name="ALT",
                    value=alt,
                    upper_limit_normal=40,
                    sample_date=sample_date,
                    source="laboratory_analysis",
                ),
                ClinicalLabEntry(
                    marker_name="ALP",
                    value=alp,
                    upper_limit_normal=120,
                    sample_date=sample_date,
                    source="laboratory_analysis",
                ),
            ]
        )

    assessment = HepatotoxicityPatternAnalyzer().assess_payload(
        PatientLabTimeline(entries=entries)
    )
    assert assessment.score.r_score == pytest.approx(15.0)
    assert assessment.score.classification == "hepatocellular"

    structured_patterns = DiliPatternEngine().assess(
        PatientLabTimeline(entries=entries)
    )
    assert structured_patterns[0].assessment_point == "first_qualifying"
    assert structured_patterns[0].sample_date == "2026-01-28"
    assert structured_patterns[0].r_ratio == pytest.approx(15.0)
    assert structured_patterns[0].pattern == "hepatocellular"
    assert structured_patterns[1].assessment_point == "peak"


###############################################################################
def test_primary_injury_anchor_uses_first_abnormal_pair_with_varying_ulns() -> None:
    timeline = PatientLabTimeline(
        entries=[
            ClinicalLabEntry(
                marker_name="ALT",
                value=180,
                upper_limit_normal=30,
                sample_date="2026-04-01",
                source="laboratory_analysis",
            ),
            ClinicalLabEntry(
                marker_name="ALP",
                value=240,
                upper_limit_normal=120,
                sample_date="2026-04-01",
                source="laboratory_analysis",
            ),
            ClinicalLabEntry(
                marker_name="ALT",
                value=280,
                upper_limit_normal=70,
                sample_date="2026-04-08",
                source="laboratory_analysis",
            ),
            ClinicalLabEntry(
                marker_name="ALP",
                value=120,
                upper_limit_normal=120,
                sample_date="2026-04-08",
                source="laboratory_analysis",
            ),
        ]
    )

    score = HepatotoxicityPatternAnalyzer().assess_payload(timeline).score
    assert score.alt_multiple == pytest.approx(6.0)
    assert score.r_score == pytest.approx(3.0)
    assert score.classification == "mixed"


###############################################################################
def test_r_ratio_requires_alt_not_ast() -> None:
    timeline = PatientLabTimeline(
        entries=[
            ClinicalLabEntry(
                marker_name="AST",
                value=400,
                upper_limit_normal=40,
                sample_date="2026-05-01",
                source="laboratory_analysis",
            ),
            ClinicalLabEntry(
                marker_name="ALP",
                value=120,
                upper_limit_normal=120,
                sample_date="2026-05-01",
                source="laboratory_analysis",
            ),
        ]
    )
    assessment = HepatotoxicityPatternAnalyzer().assess_payload(timeline)
    assert assessment.status == "undetermined_due_to_missing_labs"
    assert assessment.score.r_score is None


###############################################################################
def test_r_ratio_does_not_invent_missing_uln() -> None:
    timeline = PatientLabTimeline(
        entries=[
            ClinicalLabEntry(
                marker_name="ALT",
                value=400,
                sample_date="2026-05-01",
                source="laboratory_analysis",
            ),
            ClinicalLabEntry(
                marker_name="ALP",
                value=120,
                sample_date="2026-05-01",
                source="laboratory_analysis",
            ),
        ]
    )
    assessment = HepatotoxicityPatternAnalyzer().assess_payload(timeline)
    assert assessment.status == "undetermined_due_to_missing_labs"
    assert assessment.score.r_score is None
