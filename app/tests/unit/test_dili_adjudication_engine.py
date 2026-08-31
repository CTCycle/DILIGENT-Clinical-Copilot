from __future__ import annotations

from domain.clinical.entities import (
    ClinicalLabEntry,
    DrugEntry,
    DrugRucamAssessment,
    PatientData,
    PatientDrugs,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
    RucamComponentAssessment,
)
from services.clinical.dili_evidence import DiliEvidenceBuilder
from services.clinical.dili_pattern import DiliPatternEngine


###############################################################################
def test_r_ratio_boundary_values_follow_livertox_definitions() -> None:
    assert DiliPatternEngine.classify(5.0) == "hepatocellular"
    assert DiliPatternEngine.classify(2.0) == "cholestatic"
    assert DiliPatternEngine.classify(3.0) == "mixed"


###############################################################################
def test_undated_multi_timepoint_labs_use_peak_multiples() -> None:
    patterns = DiliPatternEngine().assess(
        PatientLabTimeline(
            entries=[
                ClinicalLabEntry(
                    marker_name="ALT", value=32, source="laboratory_analysis"
                ),
                ClinicalLabEntry(
                    marker_name="AST", value=28, source="laboratory_analysis"
                ),
                ClinicalLabEntry(
                    marker_name="ALP", value=92, source="laboratory_analysis"
                ),
                ClinicalLabEntry(
                    marker_name="ALT", value=860, source="laboratory_analysis"
                ),
                ClinicalLabEntry(
                    marker_name="AST", value=610, source="laboratory_analysis"
                ),
                ClinicalLabEntry(
                    marker_name="ALP", value=210, source="laboratory_analysis"
                ),
            ]
        )
    )

    assert patterns[0].pattern == "hepatocellular"
    assert patterns[0].r_ratio == 12.285714285714286


###############################################################################
def test_structured_dossier_preserves_missing_competing_causes() -> None:
    bundle = DiliEvidenceBuilder().build(
        payload=PatientData(
            anamnesis="HAV negative. Autoimmune hepatitis excluded.",
            drugs="Drug A started 2026-01-01 and stopped 2026-01-20.",
            laboratory_analysis="ALT and bilirubin elevated.",
        ),
        drugs=PatientDrugs(
            entries=[
                DrugEntry(
                    name="Drug A",
                    source="therapy",
                    therapy_start_date="2026-01-01",
                    suspension_date="2026-01-20",
                    evidence="Drug A started 2026-01-01 and stopped 2026-01-20.",
                )
            ]
        ),
        labs=PatientLabTimeline(
            entries=[
                ClinicalLabEntry(
                    marker_name="ALT",
                    value=240,
                    upper_limit_normal=40,
                    sample_date="2026-01-15",
                    source="laboratory_analysis",
                ),
                ClinicalLabEntry(
                    marker_name="ALP",
                    value=120,
                    upper_limit_normal=120,
                    sample_date="2026-01-15",
                    source="laboratory_analysis",
                ),
                ClinicalLabEntry(
                    marker_name="BILIRUBIN",
                    value=3,
                    upper_limit_normal=1,
                    sample_date="2026-01-16",
                    source="laboratory_analysis",
                ),
            ]
        ),
        resolved_drugs={
            "drug a": {
                "decision_status": "accepted_exact_livertox",
                "accepted_livertox_name": "Drug A",
                "match_confidence": 1.0,
                "matched_livertox_row": {"likelihood_score": "A"},
            }
        },
        rucam_bundle=PatientRucamAssessmentBundle(
            entries=[
                DrugRucamAssessment(
                    drug_name="Drug A",
                    total_score=6,
                    causality_category="probable",
                    components=[
                        RucamComponentAssessment(
                            component_key="time_to_onset",
                            label="Time to onset",
                            score=2,
                            evidence="2026-01-01 to 2026-01-15",
                        )
                    ],
                )
            ]
        ),
    )

    assert bundle.patterns[0].pattern == "hepatocellular"
    assert bundle.hys_law.status == "possible"
    assert bundle.hys_law.same_episode is True
    assert bundle.exposures[0].livertox_likelihood == "A"
    assert bundle.exposures[0].rucam.components[0].evidence_quote
    assert "ebv_cmv_hsv" in bundle.differential.unresolved_causes
    assert len(bundle.acceptance_questions) == 12
    assert any(
        question.question.startswith("Does the episode satisfy Hy's Law")
        for question in bundle.acceptance_questions
    )
    assert bundle.manual_review_required is True


###############################################################################
def test_generated_narrative_safety_gate_blocks_unsupported_certainty() -> None:
    bundle = DiliEvidenceBuilder().build(
        payload=PatientData(
            anamnesis="Jaundice and fatigue.",
            drugs="Drug A started 2026-01-01 and stopped 2026-01-20.",
            laboratory_analysis="ALT 240 U/L, ALP 120 U/L, bilirubin 3 mg/dL.",
        ),
        drugs=PatientDrugs(
            entries=[DrugEntry(name="Drug A", therapy_start_date="2026-01-01")]
        ),
        labs=PatientLabTimeline(
            entries=[
                ClinicalLabEntry(
                    marker_name="ALT",
                    value=240,
                    upper_limit_normal=40,
                    sample_date="2026-01-15",
                    source="laboratory_analysis",
                ),
                ClinicalLabEntry(
                    marker_name="ALP",
                    value=120,
                    upper_limit_normal=120,
                    sample_date="2026-01-15",
                    source="laboratory_analysis",
                ),
                ClinicalLabEntry(
                    marker_name="BILIRUBIN",
                    value=3,
                    upper_limit_normal=1,
                    sample_date="2026-01-16",
                    source="laboratory_analysis",
                ),
            ]
        ),
        resolved_drugs=None,
        rucam_bundle=PatientRucamAssessmentBundle(entries=[]),
    )

    issues = DiliEvidenceBuilder.audit_generated_narrative(
        clinical_narrative=(
            "Viral hepatitis was ruled out and there are no competing causes. "
            "This is a confident diagnosis with lifelong avoidance required. "
            "The Hy's Law pattern is confirmed."
        ),
        bundle=bundle,
    )

    assert {issue["code"] for issue in issues} == {
        "clinical_narrative_contradicts_competing_causes",
        "clinical_narrative_overstates_hys_law",
        "clinical_narrative_overstates_causality",
    }

    safe_issues = DiliEvidenceBuilder.audit_generated_narrative(
        clinical_narrative=(
            "A definitive diagnosis cannot be made with absolute certainty; "
            "manual review remains required."
        ),
        bundle=bundle,
    )

    assert "clinical_narrative_overstates_causality" not in {
        issue["code"] for issue in safe_issues
    }


###############################################################################
def test_generated_narrative_safety_gate_blocks_rechallenge_permission() -> None:
    bundle = DiliEvidenceBuilder().build(
        payload=PatientData(anamnesis="Jaundice."),
        drugs=PatientDrugs(entries=[DrugEntry(name="Drug A")]),
        labs=PatientLabTimeline(entries=[]),
        resolved_drugs=None,
        rucam_bundle=PatientRucamAssessmentBundle(entries=[]),
    )

    unsafe_issues = DiliEvidenceBuilder.audit_generated_narrative(
        clinical_narrative=(
            "No rechallenge occurred, but a cautious trial of temporary interruption "
            "then rechallenge under observation may be considered."
        ),
        bundle=bundle,
    )
    assert "clinical_narrative_recommends_rechallenge" in {
        issue["code"] for issue in unsafe_issues
    }

    safe_issues = DiliEvidenceBuilder.audit_generated_narrative(
        clinical_narrative=(
            "No rechallenge occurred. Rechallenge is not recommended; clinical review "
            "is required."
        ),
        bundle=bundle,
    )
    assert "clinical_narrative_recommends_rechallenge" not in {
        issue["code"] for issue in safe_issues
    }


###############################################################################
def test_timeline_preserves_explicit_symptom_and_jaundice_dates() -> None:
    bundle = DiliEvidenceBuilder().build(
        payload=PatientData(
            anamnesis=(
                "Fatigue and pruritus began on 2026-02-05. "
                "Jaundice developed on 2026-02-06."
            )
        ),
        drugs=PatientDrugs(entries=[]),
        labs=PatientLabTimeline(
            entries=[
                ClinicalLabEntry(
                    marker_name="BILIRUBIN",
                    value=3,
                    sample_date="2026-02-06",
                    source="laboratory_analysis",
                )
            ]
        ),
        resolved_drugs=None,
        rucam_bundle=PatientRucamAssessmentBundle(entries=[]),
    )

    assert bundle.timeline.first_symptom_date == "2026-02-05"
    assert bundle.timeline.jaundice_or_bilirubin_rise_date == "2026-02-06"
    assert "first_symptom_date" not in bundle.timeline.missing_fields
    assert "jaundice_or_bilirubin_timing" not in bundle.timeline.missing_fields


###############################################################################
def test_causality_does_not_upgrade_stable_current_drug_from_global_dechallenge() -> (
    None
):
    bundle = DiliEvidenceBuilder().build(
        payload=PatientData(
            anamnesis="Mandatory alternative-cause workup is pending.",
            drugs="Rosuvastatin stable for approximately two years and continued without dose change.",
            laboratory_analysis="ALT 240 U/L and ALP 100 U/L.",
        ),
        drugs=PatientDrugs(
            entries=[
                DrugEntry(
                    name="Rosuvastatin",
                    source="therapy",
                    current_status="current",
                    evidence=(
                        "Rosuvastatin stable for approximately two years and continued "
                        "without dose change."
                    ),
                )
            ]
        ),
        labs=PatientLabTimeline(
            entries=[
                ClinicalLabEntry(
                    marker_name="ALT",
                    value=240,
                    upper_limit_normal=40,
                    sample_date="2026-02-10",
                    source="laboratory_analysis",
                ),
                ClinicalLabEntry(
                    marker_name="ALP",
                    value=100,
                    upper_limit_normal=100,
                    sample_date="2026-02-10",
                    source="laboratory_analysis",
                ),
            ]
        ),
        resolved_drugs={
            "rosuvastatin": {
                "decision_status": "accepted_exact_livertox",
                "accepted_livertox_name": "Rosuvastatin",
                "matched_livertox_row": {"likelihood_score": "A"},
            }
        },
        rucam_bundle=PatientRucamAssessmentBundle(
            entries=[
                DrugRucamAssessment(
                    drug_name="Rosuvastatin",
                    total_score=0,
                    causality_category="excluded",
                )
            ]
        ),
    )

    exposure = bundle.exposures[0]
    assert exposure.causality is not None
    assert exposure.causality.category == "unlikely"
    assert exposure.causality.temporal_compatibility == "incompatible"
    assert exposure.causality.dechallenge_rechallenge.startswith("not_assessable;")
    assert exposure.rucam is not None
    assert exposure.rucam.category == "excluded"


###############################################################################
def test_report_has_required_fda_style_sections() -> None:
    bundle = DiliEvidenceBuilder().build(
        payload=PatientData(drugs="Drug A"),
        drugs=PatientDrugs(entries=[DrugEntry(name="Drug A")]),
        labs=PatientLabTimeline(entries=[]),
        resolved_drugs=None,
        rucam_bundle=PatientRucamAssessmentBundle(entries=[]),
    )
    report = DiliEvidenceBuilder.render(bundle)
    for section_number in range(1, 15):
        assert f"## {section_number}." in report
    assert "Manual hepatology review required" in report


###############################################################################
def test_user_summary_groups_missing_data_without_raw_field_keys() -> None:
    bundle = DiliEvidenceBuilder().build(
        payload=PatientData(drugs="Aldactone\nCefepime"),
        drugs=PatientDrugs(
            entries=[
                DrugEntry(name="Aldactone"),
                DrugEntry(name="Cefepime", therapy_start_date="2025-02-19"),
            ]
        ),
        labs=PatientLabTimeline(entries=[]),
        resolved_drugs=None,
        rucam_bundle=PatientRucamAssessmentBundle(entries=[]),
    )

    summary = DiliEvidenceBuilder.render_user_summary(bundle)
    dossier = DiliEvidenceBuilder.render(bundle)

    assert "## DILI adjudication summary" in summary
    assert "### Clinically relevant missing data" in summary
    assert "Exposure timing" in summary
    assert "Aldactone: start date not documented" in summary
    assert "Cefepime: stop date not documented" in summary
    assert "Aldactone:drug_start_date" not in summary
    assert "Aldactone:drug_start_date" not in dossier
    assert "paired ALT and ALP values with ULN are unavailable" in summary
    assert DiliEvidenceBuilder._format_missing_field("Aldactone:drugstartdate") == (
        "Exposure timing",
        "Aldactone: start date not documented",
    )


###############################################################################
def test_dechallenge_tolerates_missing_pre_stop_labs() -> None:
    bundle = DiliEvidenceBuilder().build(
        payload=PatientData(
            anamnesis="Jaundice started 2026-06-15. Viral hepatitis negative.",
            drugs=(
                "Amoxicillin-clavulanate started 2026-06-01 and stopped 2026-06-14."
            ),
            laboratory_analysis="ALT improved after stopping therapy.",
        ),
        drugs=PatientDrugs(
            entries=[
                DrugEntry(
                    name="Amoxicillin-clavulanate",
                    source="therapy",
                    therapy_start_date="2026-06-01",
                    suspension_date="2026-06-14",
                    evidence=(
                        "Amoxicillin-clavulanate started 2026-06-01 and "
                        "stopped 2026-06-14."
                    ),
                )
            ]
        ),
        labs=PatientLabTimeline(
            entries=[
                ClinicalLabEntry(
                    marker_name="ALT",
                    value=620,
                    upper_limit_normal=40,
                    sample_date="2026-06-16",
                    source="laboratory_analysis",
                ),
                ClinicalLabEntry(
                    marker_name="ALT",
                    value=240,
                    upper_limit_normal=40,
                    sample_date="2026-06-22",
                    source="laboratory_analysis",
                ),
            ]
        ),
        resolved_drugs=None,
        rucam_bundle=PatientRucamAssessmentBundle(entries=[]),
    )

    assert bundle.timeline.dechallenge_status == "improving_after_stop"
