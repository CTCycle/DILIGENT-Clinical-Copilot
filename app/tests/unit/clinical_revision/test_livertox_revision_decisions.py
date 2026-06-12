from __future__ import annotations

from services.inspection.service import (
    DataInspectionService,
    ReviewerInstructionProfile,
)


###############################################################################
def test_livertox_revision_decisions_reuse_high_confidence_previous_match() -> None:
    decisions = DataInspectionService.build_revision_livertox_decisions(
        matched_drugs=[
            {
                "matched_drug_name": "Drug A",
                "match_status": "matched",
                "match_confidence": 0.40,
            }
        ],
        source_matched_drugs=[
            {
                "matched_drug_name": "Drug A",
                "match_status": "matched_with_excerpt",
                "match_confidence": 0.99,
            }
        ],
        instruction_profile=None,
    )

    assert len(decisions) == 1
    assert decisions[0]["decision"] == "reused_high_confidence_previous_match"
    assert decisions[0]["source"] == "previous_version"
    assert decisions[0]["requires_human_review"] is False


###############################################################################
def test_livertox_revision_decisions_force_refresh_when_reviewer_challenges_matching() -> None:
    profile = ReviewerInstructionProfile(
        instruction_summary="Recheck whether the match is wrong.",
        target_sections=["livertox_matching"],
        target_entities=["matching_errors"],
        pipeline_routing_decision={},
    )

    decisions = DataInspectionService.build_revision_livertox_decisions(
        matched_drugs=[
            {
                "matched_drug_name": "Drug B",
                "match_status": "matched",
                "match_confidence": 0.99,
            }
        ],
        source_matched_drugs=[
            {
                "matched_drug_name": "Drug B",
                "match_status": "matched_with_excerpt",
                "match_confidence": 0.99,
            }
        ],
        instruction_profile=profile,
    )

    assert decisions[0]["decision"] == "llm_assisted_resolved_match"
    assert decisions[0]["reviewer_challenged"] is True
    assert decisions[0]["source"] == "llm_fallback"


###############################################################################
def test_livertox_revision_decisions_require_human_review_for_missing_match() -> None:
    decisions = DataInspectionService.build_revision_livertox_decisions(
        matched_drugs=[
            {
                "matched_drug_name": "Drug C",
                "match_status": "missing_match",
                "match_confidence": None,
            }
        ],
        source_matched_drugs=[],
        instruction_profile=None,
    )

    assert decisions[0]["decision"] == "no_reliable_match_found"
    assert decisions[0]["requires_human_review"] is True
    assert decisions[0]["source"] == "none"
