from __future__ import annotations

import pytest
from pydantic import ValidationError

from domain.clinical.revision import (
    RevisedDiliAssessment,
    RevisedDiseasePayload,
    RevisedDrugPayload,
    RevisedLabPayload,
    RevisionLiverToxDecision,
)

###############################################################################
def test_revised_drug_payload_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RevisedDrugPayload.model_validate(
            {
                "name": "drug-a",
                "role": "suspect",
                "unexpected_field": "should-fail",
            }
        )

###############################################################################
def test_revised_lab_payload_rejects_string_numeric_coercion() -> None:
    with pytest.raises(ValidationError):
        RevisedLabPayload.model_validate(
            {
                "marker_name": "ALT",
                "value": "150",
                "source": "laboratory_analysis",
            }
        )

###############################################################################
def test_revision_livertox_decision_requires_structured_shape() -> None:
    payload = RevisionLiverToxDecision.model_validate(
        {
            "decision_id": "livertox:0",
            "drug_name": "drug-a",
            "normalized_drug_name": "drug-a",
            "decision": "reused_high_confidence_previous_match",
            "decision_reason": "High-confidence previous source-version match remains valid.",
            "match_status": "matched",
            "match_confidence": 0.99,
            "requires_human_review": False,
            "reviewer_challenged": False,
            "source": "previous_version",
            "previous_match_found": True,
            "previous_match_confidence": 0.99,
            "payload": {"matched_drug_name": "drug-a"},
            "provenance": {"source_version_match": {"matched_drug_name": "drug-a"}},
        }
    )

    assert payload.decision == "reused_high_confidence_previous_match"
    assert payload.match_confidence == 0.99

###############################################################################
def test_revised_dili_assessment_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RevisedDiliAssessment.model_validate(
            {
                "revised_drug_entry_id": "revised-drug:0",
                "revision_version_id": 2,
                "source_version_id": 1,
                "assessment_version": "1",
                "drug_name": "drug-a",
                "causality_assessment": "probable",
                "confidence": "high",
                "evidence_for": [],
                "evidence_against": [],
                "lab_support": [],
                "temporal_support": [],
                "alternative_causes": [],
                "livertox_support": [],
                "changes_from_previous_version": [],
                "unresolved_questions": [],
                "requires_human_review": False,
                "previous_assessment_present": False,
                "provenance": {},
                "unknown": "should-fail",
            }
        )

###############################################################################
def test_revised_disease_payload_accepts_expected_fields() -> None:
    payload = RevisedDiseasePayload.model_validate(
        {
            "name": "autoimmune hepatitis",
            "diagnosis_status": "confirmed",
            "hepatic_related": True,
            "evidence": "Biopsy and serology",
        }
    )

    assert payload.name == "autoimmune hepatitis"
    assert payload.hepatic_related is True
