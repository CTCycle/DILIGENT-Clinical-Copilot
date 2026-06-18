from __future__ import annotations

from services.inspection.service import (
    DataInspectionService,
    ReviewerInstructionProfile,
)

###############################################################################
def test_revised_dili_assessment_tracks_previous_version_changes() -> None:
    assessments = DataInspectionService.build_revised_dili_assessments(
        rucam_assessments=[
            {
                "drug_name": "Drug A",
                "total_score": 7,
                "causality_category": "probable",
            }
        ],
        matched_drugs=[
            {
                "matched_drug_name": "Drug A",
                "match_status": "matched",
                "match_confidence": 0.99,
            }
        ],
        source_rucam_assessments=[
            {
                "drug_name": "Drug A",
                "total_score": 4,
                "causality_category": "possible",
            }
        ],
        revision_version_id=2,
        source_version_id=1,
        instruction_profile=None,
    )

    assert len(assessments) == 1
    assert assessments[0]["confidence"] == "moderate"
    assert "Causality changed from possible to probable." in assessments[0]["changes_from_previous_version"]
    assert "Total score changed from 4 to 7." in assessments[0]["changes_from_previous_version"]
    assert assessments[0]["previous_assessment_present"] is True
    assert assessments[0]["requires_human_review"] is False

###############################################################################
def test_revised_dili_assessment_requires_human_review_without_match_and_when_challenged() -> None:
    profile = ReviewerInstructionProfile(
        instruction_summary="Reassess the causality reasoning carefully.",
        target_sections=["dili_assessment"],
        target_entities=["causality_reasoning"],
        pipeline_routing_decision={},
    )

    assessments = DataInspectionService.build_revised_dili_assessments(
        rucam_assessments=[
            {
                "drug_name": "Drug B",
                "total_score": 2,
                "causality_category": "unlikely",
            }
        ],
        matched_drugs=[],
        source_rucam_assessments=[],
        revision_version_id=3,
        source_version_id=1,
        instruction_profile=profile,
    )

    assert len(assessments) == 1
    assert assessments[0]["confidence"] == "low"
    assert assessments[0]["requires_human_review"] is True
    assert "No reliable LiverTox match is available for this revised drug." in assessments[0]["unresolved_questions"]
    assert "Reviewer explicitly requested reassessment of causality reasoning." in assessments[0]["unresolved_questions"]

###############################################################################
def test_revised_dili_assessment_notes_retained_previous_assessment_when_unchanged() -> None:
    assessments = DataInspectionService.build_revised_dili_assessments(
        rucam_assessments=[
            {
                "drug_name": "Drug C",
                "total_score": 5,
                "causality_category": "possible",
            }
        ],
        matched_drugs=[
            {
                "matched_drug_name": "Drug C",
                "match_status": "matched_with_excerpt",
                "match_confidence": 0.97,
            }
        ],
        source_rucam_assessments=[
            {
                "drug_name": "Drug C",
                "total_score": 5,
                "causality_category": "possible",
            }
        ],
        revision_version_id=4,
        source_version_id=1,
        instruction_profile=None,
    )

    assert assessments[0]["changes_from_previous_version"] == [
        "Previous source-version assessment was reviewed and retained."
    ]
