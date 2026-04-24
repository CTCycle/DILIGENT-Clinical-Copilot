from __future__ import annotations

from types import SimpleNamespace

from DILIGENT.server.domain.clinical import PatientRucamAssessmentBundle
from DILIGENT.server.services.clinical.match_quality import classify_match_evidence
from DILIGENT.server.services.session.session_service import ClinicalSessionService


def test_classify_match_evidence_marks_related_fallback() -> None:
    quality = classify_match_evidence(
        match_status="matched_with_excerpt",
        match_reason="exact_canonical",
        match_confidence=1.0,
        match_notes=["fallback_excerpt_from_related_monograph"],
        missing_livertox=False,
        ambiguous_match=False,
    )

    assert quality["evidence_quality"] == "fallback_related_monograph"
    assert quality["evidence_warnings"] == [
        "Evidence excerpt was borrowed from a related LiverTox monograph."
    ]


def test_matched_drug_payload_exposes_evidence_quality() -> None:
    prepared_inputs = SimpleNamespace(
        resolved_drugs={
            "ondansetron": {
                "matched_livertox_row": {"drug_name": "5-HT3 Receptor Antagonists"},
                "match_confidence": 0.92,
                "match_reason": "exact_alias",
                "match_notes": ["matched_record_missing_excerpt"],
                "match_status": "matched_no_excerpt",
                "missing_livertox": True,
                "ambiguous_match": False,
            }
        }
    )

    payload = ClinicalSessionService.build_matched_drugs_payload(
        detected_drugs=["Ondansetron"],
        prepared_inputs=prepared_inputs,
        rucam_bundle=PatientRucamAssessmentBundle(entries=[]),
    )

    assert payload[0]["evidence_quality"] == "weak_alias_or_class_match"
    assert "Drug match is not a direct canonical match." in payload[0]["evidence_warnings"]
    assert "Matched local drug record has no LiverTox excerpt." in payload[0]["evidence_warnings"]


def test_matched_drug_payload_marks_missing_evidence_when_knowledge_base_unavailable() -> None:
    payload = ClinicalSessionService.build_matched_drugs_payload(
        detected_drugs=["Amoxicillin"],
        prepared_inputs=None,
        rucam_bundle=PatientRucamAssessmentBundle(entries=[]),
    )

    assert payload[0]["match_status"] == "missing_match"
    assert payload[0]["match_reason"] == "knowledge_base_unavailable"
    assert payload[0]["evidence_quality"] == "missing_match"
    assert payload[0]["evidence_warnings"] == ["No local RxNav/LiverTox match was found."]


def test_knowledge_base_unavailable_issue_is_structured_for_persistence() -> None:
    issues = []

    ClinicalSessionService.append_knowledge_base_unavailable_issue(issues)

    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert issues[0].code == "knowledge_base_unavailable"
    assert issues[0].field == "knowledge_base"
    assert "RxNav/LiverTox knowledge base is unavailable or empty" in issues[0].message
