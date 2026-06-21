from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError
from repositories.schemas.models import Base
from repositories.serialization.data import DataSerializer
from services.inspection.service import DataInspectionService
from services.runtime.jobs import JobManager
from sqlalchemy import create_engine

###############################################################################
def build_service() -> tuple[DataInspectionService, DataSerializer]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    serializer = DataSerializer(engine=engine)
    service = DataInspectionService(serializer=serializer, jobs=JobManager())
    return service, serializer

###############################################################################
def seed_session(serializer: DataSerializer) -> int:
    session_id = serializer.save_clinical_session(
        {
            "patient_name": "Repository Session",
            "session_timestamp": datetime(2025, 1, 2, 10, 15),
            "version": 1,
            "anamnesis": "Stable source clinical narrative",
            "drugs": "amoxicillin",
            "session_result_payload": {
                "original_session_text": "Stable source clinical narrative",
                "report": "Initial report draft",
            },
        }
    )
    if session_id is None:
        raise AssertionError("Session seed failed")
    return session_id

###############################################################################
def test_legacy_update_session_route_now_performs_safe_manual_report_edit() -> None:
    service, serializer = build_service()
    session_id = seed_session(serializer)

    updated = service.update_session(
        session_id,
        report_text="Revised report content via current route",
        metadata={"reviewer": "Legacy Reviewer"},
    )

    assert updated is not None
    assert updated["official_report_text"] == "Revised report content via current route"
    assert updated["source_clinical_text"] == "Stable source clinical narrative"
    assert updated["version"] == 1
    assert updated["metadata"]["reviewer"] == "Legacy Reviewer"
    assert len(updated["manual_edit_history"]) == 1
    assert updated["manual_edit_history"][0]["edited_fields"] == ["report_text"]

###############################################################################
def test_metadata_only_update_does_not_create_manual_edit_audit() -> None:
    service, serializer = build_service()
    session_id = seed_session(serializer)

    updated = service.update_session(
        session_id,
        report_text=None,
        metadata={"reviewer": "Metadata Only"},
    )

    assert updated is not None
    assert updated["metadata"]["reviewer"] == "Metadata Only"
    assert updated["official_report_text"] == "Initial report draft"
    assert updated["source_clinical_text"] == "Stable source clinical narrative"
    assert updated["manual_edit_history"] == []

###############################################################################
def test_get_session_detail_reconstructs_source_text_from_persisted_sections() -> None:
    service, serializer = build_service()
    session_id = serializer.save_clinical_session(
        {
            "patient_name": "Section Fallback Patient",
            "session_timestamp": datetime(2025, 1, 2, 10, 15),
            "version": 1,
            "anamnesis": "Section-only anamnesis",
            "drugs": "Section-only drugs",
            "laboratory_analysis": "Section-only labs",
            "session_result_payload": {
                "report": "Persisted report without original source text",
            },
        }
    )
    assert session_id is not None

    detail = service.get_session_detail(session_id)

    assert detail is not None
    assert detail["session_text"] == (
        "Anamnesis:\nSection-only anamnesis\n\n"
        "Drugs:\nSection-only drugs\n\n"
        "Laboratory Analysis:\nSection-only labs"
    )
    assert detail["source_clinical_text"] == detail["session_text"]

###############################################################################
def test_revision_review_actions_are_persisted_and_update_version_state() -> None:
    service, serializer = build_service()
    session_id = seed_session(serializer)
    source_version = serializer.get_version_record_for_session(session_id)
    assert source_version is not None

    shell = serializer.create_revision_version_shell(
        session_id,
        reviewer_note="Initial revision request",
        configuration={"metadata": {"reviewer": "Reviewer D"}},
        pipeline_run_id="review-run-001",
        initiated_by="Reviewer D",
    )
    assert shell is not None

    revised_session_id = serializer.save_clinical_session(
        {
            "patient_name": "Repository Session",
            "session_timestamp": datetime(2025, 1, 3, 10, 15),
            "version": 2,
            "original_session_id": session_id,
            "anamnesis": "Stable source clinical narrative",
            "drugs": "amoxicillin",
            "session_result_payload": {
                "original_session_text": "Stable source clinical narrative",
                "report": "Revised report draft",
                "revision": {"revision_kind": "llm_assisted_revision"},
            },
        }
    )
    assert revised_session_id is not None

    finalized = serializer.finalize_revision_version(
        pipeline_run_id="review-run-001",
        persisted_session_id=revised_session_id,
        version_status="llm_qa_passed",
        llm_qa_status="passed",
        clinical_review_status="not_reviewed",
    )
    assert finalized is not None
    assert finalized["clinical_review_status"] == "not_reviewed"

    update = service.update_revision_clinical_review(
        session_id,
        version_id=int(shell["version_id"]),
        clinical_review_status="approved_by_human",
        reviewer_note="Approved after manual chronology check.",
        reviewed_by="Reviewer D",
        metadata={"decision_source": "clinical_review"},
    )
    assert update is not None
    assert update["version"]["version_status"] == "human_approved"
    assert update["version"]["clinical_review_status"] == "approved_by_human"
    assert update["review_action"]["clinical_review_status"] == "approved_by_human"
    assert update["review_action"]["reviewed_by"] == "Reviewer D"

    history = service.list_revision_reviews(
        revision_version_id=int(shell["version_id"]),
    )
    assert len(history) == 1
    assert history[0]["reviewer_note"] == "Approved after manual chronology check."
    assert history[0]["metadata"]["decision_source"] == "clinical_review"

###############################################################################
def test_compare_session_versions_returns_backend_diff_payload() -> None:
    service, serializer = build_service()
    root_session_id = serializer.save_clinical_session(
        {
            "patient_name": "Comparison Patient",
            "session_timestamp": datetime(2025, 1, 2, 10, 15),
            "version": 1,
            "anamnesis": "Initial source narrative",
            "drugs": "amoxicillin",
            "session_result_payload": {
                "original_session_text": "Initial source narrative",
                "report": "Initial report draft\nDrug A suspected.",
                "structured_case": {
                    "therapy_drugs": [{"name": "drug-a", "role": "suspect"}],
                    "anamnesis_drugs": [{"name": "historical-drug", "role": "historical"}],
                    "anamnesis_diseases": [{"name": "hepatitis"}],
                },
                "lab_timeline": [{"marker_name": "ALT", "value": 55}],
                "matched_drugs": [{"matched_drug_name": "drug-a", "match_status": "matched"}],
                "rucam_assessments": [{"drug_name": "drug-a", "total_score": 4}],
                "revision": {
                    "qa_validation": {
                        "status": "passed",
                        "version_status": "llm_qa_passed",
                        "warnings": [],
                        "blocking_issues": [],
                        "finding_count": 0,
                        "manual_review_required": False,
                    }
                },
            },
        }
    )
    assert root_session_id is not None

    revised_session_id = serializer.save_clinical_session(
        {
            "patient_name": "Comparison Patient",
            "session_timestamp": datetime(2025, 1, 3, 10, 15),
            "version": 2,
            "original_session_id": root_session_id,
            "anamnesis": "Revised source narrative",
            "drugs": "amoxicillin, acetaminophen",
            "session_result_payload": {
                "original_session_text": "Revised source narrative",
                "report": "Revised report draft\nDrug A and Drug B suspected.",
                "structured_case": {
                    "therapy_drugs": [
                        {"name": "drug-a", "role": "suspect", "confidence": "high"},
                        {"name": "drug-b", "role": "suspect"},
                    ],
                    "anamnesis_drugs": [{"name": "historical-drug", "role": "historical"}],
                    "anamnesis_diseases": [{"name": "hepatitis"}],
                },
                "lab_timeline": [{"marker_name": "ALT", "value": 150}],
                "matched_drugs": [{"matched_drug_name": "drug-b", "match_status": "matched"}],
                "rucam_assessments": [{"drug_name": "drug-b", "total_score": 7}],
                "revision": {
                    "qa_validation": {
                        "status": "passed_with_warnings",
                        "version_status": "llm_qa_passed",
                        "warnings": ["Manual chronology follow-up recommended."],
                        "blocking_issues": [],
                        "finding_count": 1,
                        "manual_review_required": False,
                    }
                },
            },
        }
    )
    assert revised_session_id is not None

    root_version = serializer.get_version_record_for_session(root_session_id)
    revised_version = serializer.get_version_record_for_session(revised_session_id)
    assert root_version is not None
    assert revised_version is not None

    comparison = service.compare_session_versions(
        root_session_id,
        left_version_id=int(root_version["version_id"]),
        right_version_id=int(revised_version["version_id"]),
    )

    assert comparison is not None
    assert comparison["left_version"]["version_number"] == 1
    assert comparison["right_version"]["version_number"] == 2
    assert comparison["report_text_diff"]["changed"] is True
    assert comparison["report_text_diff"]["similarity_ratio"] < 1.0
    assert any(
        item["normalized_name"] == "drug-b" for item in comparison["added_entities"]
    )
    assert any(
        str(item["normalized_name"]).casefold() == "alt"
        for item in comparison["corrected_entities"]
    )
    assert comparison["qa_summary"]["left_llm_qa_status"] == "not_run"
    assert comparison["qa_summary"]["right_llm_qa_status"] == "not_run"
    assert comparison["qa_summary"]["right_warning_count"] == 1

###############################################################################
def test_compare_session_versions_derives_entities_when_revision_entities_are_missing() -> None:
    service, serializer = build_service()
    root_session_id = serializer.save_clinical_session(
        {
            "patient_name": "Derived Comparison Patient",
            "session_timestamp": datetime(2025, 1, 2, 10, 15),
            "version": 1,
            "anamnesis": "Baseline source narrative",
            "drugs": "drug-a",
            "session_result_payload": {
                "original_session_text": "Baseline source narrative",
                "report": "Baseline report",
                "structured_case": {
                    "therapy_drugs": [{"name": "drug-a", "role": "suspect"}],
                    "anamnesis_diseases": [{"name": "hepatitis"}],
                },
                "lab_timeline": [{"marker_name": "ALT", "value": 55}],
                "matched_drugs": [{"matched_drug_name": "drug-a", "match_status": "matched"}],
                "rucam_assessments": [{"drug_name": "drug-a", "total_score": 4}],
            },
        }
    )
    assert root_session_id is not None

    revised_session_id = serializer.save_clinical_session(
        {
            "patient_name": "Derived Comparison Patient",
            "session_timestamp": datetime(2025, 1, 3, 10, 15),
            "version": 2,
            "original_session_id": root_session_id,
            "anamnesis": "Revised source narrative",
            "drugs": "drug-a",
            "session_result_payload": {
                "original_session_text": "Revised source narrative",
                "report": "Revised report",
                "structured_case": {
                    "therapy_drugs": [{"name": "drug-a", "role": "suspect"}],
                    "anamnesis_diseases": [{"name": "hepatitis"}],
                },
                "lab_timeline": [{"marker_name": "ALT", "value": 150}],
                "matched_drugs": [{"matched_drug_name": "drug-a", "match_status": "matched_with_excerpt"}],
                "rucam_assessments": [{"drug_name": "drug-a", "total_score": 7}],
            },
        }
    )
    assert revised_session_id is not None

    root_version = serializer.get_version_record_for_session(root_session_id)
    revised_version = serializer.get_version_record_for_session(revised_session_id)
    assert root_version is not None
    assert revised_version is not None

    comparison = service.compare_session_versions(
        root_session_id,
        left_version_id=int(root_version["version_id"]),
        right_version_id=int(revised_version["version_id"]),
    )

    assert comparison is not None
    assert any(
        str(item["normalized_name"]).casefold() == "alt"
        for item in comparison["corrected_entities"]
    )
    assert any(
        str(item["normalized_name"]).casefold() == "drug-a"
        and item["entity_type"] == "dili_assessment"
        for item in comparison["corrected_entities"]
    )
    assert any(
        str(item["normalized_name"]).casefold() == "drug-a"
        and item["entity_type"] == "livertox_match"
        for item in comparison["corrected_entities"]
    )

###############################################################################
def test_persist_revision_entities_records_schema_names_per_entity_type() -> None:
    _, serializer = build_service()
    session_id = seed_session(serializer)
    source_version = serializer.get_version_record_for_session(session_id)
    assert source_version is not None

    shell = serializer.create_revision_version_shell(
        session_id,
        reviewer_note="Schema validation test",
        configuration={},
        pipeline_run_id="schema-run-001",
        initiated_by="Reviewer S",
    )
    assert shell is not None

    entities = serializer.persist_revision_entities(
        pipeline_run_id="schema-run-001",
        revision_version_id=int(shell["version_id"]),
        source_version_id=int(source_version["version_id"]),
        result_payload={
            "structured_case": {
                "therapy_drugs": [{"name": "drug-a", "role": "suspect"}],
                "anamnesis_drugs": [{"name": "past-drug", "role": "historical"}],
                "anamnesis_diseases": [{"name": "hepatitis"}],
            },
            "lab_timeline": [
                {"marker_name": "ALT", "value": 150, "source": "laboratory_analysis"}
            ],
            "revision": {
                "livertox_revision_decisions": [
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
                ],
                "revised_dili_assessments": [
                    {
                        "revised_drug_entry_id": "revised-drug:0",
                        "revision_version_id": int(shell["version_id"]),
                        "source_version_id": int(source_version["version_id"]),
                        "assessment_version": "1",
                        "drug_name": "drug-a",
                        "causality_assessment": "probable",
                        "confidence": "high",
                        "evidence_for": [],
                        "evidence_against": [],
                        "lab_support": [],
                        "temporal_support": [],
                        "alternative_causes": [],
                        "livertox_support": ["drug-a"],
                        "changes_from_previous_version": [],
                        "unresolved_questions": [],
                        "requires_human_review": False,
                        "previous_assessment_present": True,
                        "provenance": {},
                    }
                ],
            },
        },
    )

    schema_names = {(item["entity_type"], item["schema_name"]) for item in entities}
    assert ("drug", "revised_drug_entry") in schema_names
    assert ("disease", "revised_disease_entry") in schema_names
    assert ("lab_timeline_entry", "revised_lab_entry") in schema_names
    assert ("livertox_match", "revision_livertox_decision") in schema_names
    assert ("dili_assessment", "revised_dili_assessment") in schema_names

###############################################################################
def test_persist_revision_entities_rejects_invalid_revision_payloads() -> None:
    _, serializer = build_service()
    session_id = seed_session(serializer)
    source_version = serializer.get_version_record_for_session(session_id)
    assert source_version is not None

    shell = serializer.create_revision_version_shell(
        session_id,
        reviewer_note="Invalid payload test",
        configuration={},
        pipeline_run_id="schema-run-002",
        initiated_by="Reviewer S",
    )
    assert shell is not None

    with pytest.raises(ValidationError):
        serializer.persist_revision_entities(
            pipeline_run_id="schema-run-002",
            revision_version_id=int(shell["version_id"]),
            source_version_id=int(source_version["version_id"]),
            result_payload={
                "structured_case": {
                    "therapy_drugs": [
                        {
                            "name": "drug-a",
                            "role": "suspect",
                            "unexpected_field": "should-fail",
                        }
                    ]
                }
            },
        )

    assert (
        serializer.list_revision_entities_for_version(
            revision_version_id=int(shell["version_id"])
        )
        == []
    )
