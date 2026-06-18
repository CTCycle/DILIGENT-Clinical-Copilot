from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from fastapi.testclient import TestClient

import app as server_app_module
from api import data_inspection as data_inspection_api

###############################################################################
def _get_route_service(route_path_fragment: str) -> Any:
    for route in data_inspection_api.router.routes:
        if route_path_fragment in getattr(route, "path", ""):
            endpoint_owner = getattr(route.endpoint, "__self__", None)
            if endpoint_owner is not None:
                return endpoint_owner.service
    raise AssertionError(f"Route not found for fragment {route_path_fragment}")

###############################################################################
def _wait_for_terminal_job_status(
    client: TestClient,
    job_id: str,
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = client.get(f"/api/inspection/sessions/revision/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for terminal revision job state.")

###############################################################################
def _seed_source_session(service: Any) -> int:
    session_id = service.serializer.save_clinical_session(
        {
            "patient_name": "API Integration Revision Patient",
            "session_timestamp": datetime(2025, 1, 7, 9, 0),
            "version": 1,
            "anamnesis": "Patient developed jaundice after exposure to Drug C.",
            "drugs": "Drug C",
            "final_report": "Initial official report",
            "session_result_payload": {
                "original_session_text": "Source clinical text for API integration revision test.",
                "report": "Initial official report",
                "structured_case": {
                    "therapy_drugs": [
                        {
                            "name": "Drug C",
                            "role": "suspect",
                            "confidence": 0.99,
                            "source": "therapy",
                        }
                    ],
                    "anamnesis_drugs": [],
                    "anamnesis_diseases": [
                        {
                            "name": "Jaundice",
                            "timeline": "after exposure",
                            "evidence": "Observed in the source note",
                        }
                    ],
                },
                "lab_timeline": [
                    {
                        "marker_name": "ALT",
                        "value": 320.0,
                        "unit": "U/L",
                        "upper_limit_normal": 40.0,
                        "sample_date": "2025-01-05",
                        "source": "merged",
                    }
                ],
                "matched_drugs": [
                    {
                        "raw_drug_name": "Drug C",
                        "matched_drug_name": "Drug C",
                        "match_status": "matched_with_excerpt",
                        "match_confidence": 0.98,
                    }
                ],
                "rucam_assessments": [
                    {
                        "drug_name": "Drug C",
                        "total_score": 6,
                        "causality_category": "probable",
                    }
                ],
            },
        }
    )
    if session_id is None:
        raise AssertionError("Failed to seed source session")
    return session_id

###############################################################################
def test_revision_api_routes_preserve_manual_edits_and_persist_revision_lineage(
    monkeypatch,
) -> None:
    service = _get_route_service("/sessions/{session_id}/revision/jobs")
    service.jobs.jobs.clear()
    service.jobs.threads.clear()
    session_id = _seed_source_session(service)

    def fake_run_revision_job(
        *,
        job_id: str | None,
        pipeline_run_id: str,
        source_version_id: int,
        target_revision_version_id: int,
        session_detail: dict[str, Any],
        root_session_id: int,
        version: int,
        selected_text: str | None,
        revision_instruction: str | None,
        model_overrides: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        del job_id, revision_instruction, model_overrides
        serializer = service.serializer
        source_text = str(session_detail.get("source_clinical_text") or "")
        revised_report = "Revised official report with clarified chronology."

        started_step = serializer.start_revision_step(
            pipeline_run_id=pipeline_run_id,
            step_name="generate_revision",
            step_index=1,
            step_count=2,
            input_summary={
                "selected_text_present": bool(str(selected_text or "").strip()),
                "source_version_id": source_version_id,
            },
            schema_name="revision_output",
            schema_version="1",
            prompt_version="revision-api-integration",
            model_provider="stub",
            model_name="stub-revision-model",
            started_at=datetime.now(UTC),
        )
        serializer.complete_revision_step(
            pipeline_run_id=pipeline_run_id,
            step_name="generate_revision",
            attempt_number=int(started_step["attempt_number"]),
            output_summary={
                "report_present": True,
                "revised_drug_count": 1,
                "qa_status": "passed_with_warnings",
            },
            output_payload={"report": revised_report},
            token_usage={"input_tokens": 11, "output_tokens": 17},
            latency_ms=12,
            completed_at=datetime.now(UTC),
        )

        result_payload = {
            "report": revised_report,
            "report_comparison": {
                "outcome": "changed",
                "manual_review": "yes",
            },
            "pipeline_artifacts": {
                "faithfulness_audit": {
                    "status": "passed",
                    "warnings": ["Chronology updated from selected excerpt."],
                }
            },
            "structured_case": {
                "therapy_drugs": [
                    {
                        "name": "Drug C",
                        "role": "suspect",
                        "confidence": 0.99,
                        "source": "therapy",
                    }
                ],
                "anamnesis_drugs": [
                    {
                        "name": "Acetaminophen",
                        "role": "co-medication",
                        "confidence": 0.71,
                        "source": "anamnesis",
                    }
                ],
                "anamnesis_diseases": [
                    {
                        "name": "Jaundice",
                        "timeline": "after exposure",
                        "evidence": "Observed in the source note",
                    }
                ],
            },
            "lab_timeline": [
                {
                    "marker_name": "ALT",
                    "value": 325.0,
                    "unit": "U/L",
                    "upper_limit_normal": 40.0,
                    "sample_date": "2025-01-05",
                    "source": "merged",
                }
            ],
            "matched_drugs": [
                {
                    "raw_drug_name": "Drug C",
                    "matched_drug_name": "Drug C",
                    "match_status": "matched_with_excerpt",
                    "match_confidence": 0.99,
                    "requires_human_review": False,
                }
            ],
            "rucam_assessments": [
                {
                    "drug_name": "Drug C",
                    "total_score": 7,
                    "causality_category": "probable",
                    "requires_human_review": False,
                }
            ],
            "revision": {
                "instruction_profile": {
                    "instruction_summary": "Clarify chronology and lab wording only.",
                    "target_sections": ["final_report", "labs"],
                    "target_entities": ["report_wording", "labs"],
                    "prompt_injection_flags": [],
                    "pipeline_routing_decision": {
                        "selected_sections": ["final_report", "labs"]
                    },
                },
                "instruction_trace": {
                    "instruction_id": "trace-api-integration",
                    "raw_instruction_text": "Clarify chronology and lab wording only.",
                    "normalized_instruction_summary": "Clarify chronology and lab wording only.",
                    "routed_pipeline_steps": ["generate_revision", "qa_validate_revision"],
                    "affected_entities": ["report_wording", "labs"],
                    "applied": True,
                    "ignored": False,
                    "prompt_injection_detected": False,
                    "prompt_injection_flags": [],
                    "evidence_addressed": ["selected_text"],
                    "qa_validation_result": "passed_with_warnings",
                },
                "final_report_rebuild": {
                    "report_present": True,
                    "warnings": ["Report comparison still requests manual review."],
                },
                "qa_validation": {
                    "status": "passed_with_warnings",
                    "version_status": "llm_qa_passed",
                    "addressed_items": ["section:final_report", "section:labs"],
                    "unaddressed_items": [],
                    "warnings": ["Report comparison still requests manual review."],
                    "blocking_issues": [],
                    "manual_review_required": False,
                    "finding_count": 1,
                },
                "entity_pipeline": {
                    "status": "completed",
                    "analysis_drug_names": ["Drug C"],
                    "relevant_drug_names": ["Drug C"],
                },
                "entity_snapshot_context": "Focused revision context for chronology and lab wording.",
                "consultation_execution": {
                    "analysis_drug_names": ["Drug C"],
                    "context_metadata": {"selected_sections": ["labs", "final_report"]},
                },
                "finalization_execution": {
                    "report_present": True,
                    "qa_status": "passed_with_warnings",
                },
                "livertox_revision_decisions": [
                    {
                        "decision_id": "livertox-drug-c",
                        "drug_name": "Drug C",
                        "decision": "reused_previous_match",
                        "decision_reason": "Prior high-confidence match remained valid.",
                        "match_status": "matched_with_excerpt",
                        "match_confidence": 0.99,
                        "requires_human_review": False,
                        "reviewer_challenged": False,
                        "source": "previous_version",
                        "previous_match_found": True,
                        "previous_match_confidence": 0.99,
                        "payload": {"matched_drug_name": "Drug C"},
                        "provenance": {"strategy": "reuse"},
                    }
                ],
                "revised_dili_assessments": [
                    {
                        "revised_drug_entry_id": "therapy_drugs:0",
                        "revision_version_id": target_revision_version_id,
                        "source_version_id": source_version_id,
                        "assessment_version": "1",
                        "drug_name": "Drug C",
                        "causality_assessment": "probable",
                        "confidence": "high",
                        "evidence_for": ["Temporal association retained."],
                        "evidence_against": [],
                        "lab_support": ["ALT elevation persisted."],
                        "temporal_support": ["Symptoms followed exposure."],
                        "alternative_causes": [],
                        "livertox_support": ["Prior matched excerpt reused."],
                        "changes_from_previous_version": ["Clarified chronology wording."],
                        "unresolved_questions": [],
                        "requires_human_review": False,
                        "previous_assessment_present": True,
                        "provenance": {"runner": "api-integration-fake"},
                    }
                ],
            },
        }

        persisted_session_id = serializer.save_clinical_session(
            {
                "patient_name": session_detail.get("patient_name"),
                "session_timestamp": datetime(2025, 1, 7, 10, 0),
                "version": version,
                "original_session_id": root_session_id,
                "anamnesis": session_detail.get("source_clinical_text") or source_text,
                "drugs": "Drug C",
                "final_report": revised_report,
                "session_result_payload": {
                    "original_session_text": session_detail.get("source_clinical_text")
                    or source_text,
                    **result_payload,
                },
            }
        )
        if persisted_session_id is None:
            raise AssertionError("Failed to persist finalized revision session")

        serializer.persist_revision_artifacts(
            pipeline_run_id=pipeline_run_id,
            revision_version_id=target_revision_version_id,
            result_payload=result_payload,
        )
        serializer.persist_revision_entities(
            pipeline_run_id=pipeline_run_id,
            revision_version_id=target_revision_version_id,
            source_version_id=source_version_id,
            result_payload=result_payload,
        )
        serializer.finalize_revision_version(
            pipeline_run_id=pipeline_run_id,
            persisted_session_id=persisted_session_id,
            model_configuration={
                "clinical_model": "stub-clinical-model",
                "text_extraction_model": "stub-extraction-model",
            },
            version_status="requires_human_review",
            llm_qa_status="passed_with_warnings",
            clinical_review_status="not_reviewed",
        )
        serializer.create_or_update_revision_run(
            pipeline_run_id=pipeline_run_id,
            session_id=session_id,
            root_session_id=root_session_id,
            source_version_id=source_version_id,
            target_revision_version_id=target_revision_version_id,
            revision_mode="instruction_guided",
            revision_kind="llm_assisted_revision",
            configuration={
                "selected_text": selected_text,
                "selected_text_present": bool(str(selected_text or "").strip()),
                "revision_instruction": "Clarify chronology and lab wording only.",
                "model_overrides": {},
                "metadata": metadata,
            },
            reviewer_note=str(metadata.get("revision_note") or "").strip() or None,
            status="completed",
            initiated_by=str(metadata.get("reviewer") or "").strip() or None,
            actor_source="manual_entry",
            actor_confidence="unverified",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            trace_id=pipeline_run_id,
            latency_ms=12,
        )
        return {
            "pipeline_run_id": pipeline_run_id,
            "target_revision_version_id": target_revision_version_id,
            "persisted_session_id": persisted_session_id,
        }

    monkeypatch.setattr(service, "run_revision_job", fake_run_revision_job)

    with TestClient(server_app_module.app, raise_server_exceptions=False) as client:
        detail_response = client.get(f"/api/inspection/sessions/{session_id}")
        assert detail_response.status_code == 200
        detail_before = detail_response.json()
        assert detail_before["version"] == 1
        original_source_text = detail_before["source_clinical_text"]

        manual_edit_response = client.put(
            f"/api/inspection/sessions/{session_id}/report",
            json={
                "report_text": "Manual report correction from API integration test",
                "edited_fields": ["report_text"],
                "reviewer_note": "Corrected a wording issue.",
                "edited_by": "API Reviewer",
                "metadata": {"source": "api-integration"},
            },
        )
        assert manual_edit_response.status_code == 200
        manual_payload = manual_edit_response.json()
        assert manual_payload["session"]["official_report_text"] == (
            "Manual report correction from API integration test"
        )
        assert manual_payload["session"]["version"] == 1
        assert manual_payload["session"]["source_clinical_text"] == original_source_text
        assert manual_payload["audit"]["reviewer_note"] == "Corrected a wording issue."

        edits_response = client.get(
            f"/api/inspection/sessions/{session_id}/manual-edits"
        )
        assert edits_response.status_code == 200
        edits_payload = edits_response.json()
        assert len(edits_payload) >= 1
        assert edits_payload[0]["edited_fields"] == ["report_text"]

        revision_start = client.post(
            f"/api/inspection/sessions/{session_id}/revision/jobs",
            json={
                "selected_text": "Clarify chronology excerpt",
                "revision_instruction": "Clarify chronology and lab wording only.",
                "model_overrides": {},
                "metadata": {
                    "reviewer": "API Reviewer",
                    "revision_note": "Integration-driven revision run",
                },
            },
        )
        assert revision_start.status_code == 202
        start_payload = revision_start.json()
        job_id = start_payload["job_id"]

        job_payload = _wait_for_terminal_job_status(client, job_id)
        assert job_payload["status"] == "completed"
        result_payload = job_payload["result"]
        pipeline_run_id = result_payload["pipeline_run_id"]
        target_revision_version_id = int(result_payload["target_revision_version_id"])

        run_response = client.get(
            f"/api/inspection/sessions/revision/pipeline-runs/{pipeline_run_id}"
        )
        assert run_response.status_code == 200
        run_payload = run_response.json()
        assert run_payload["status"] == "completed"
        assert run_payload["target_revision_version_id"] == target_revision_version_id

        steps_response = client.get(
            f"/api/inspection/sessions/revision/pipeline-runs/{pipeline_run_id}/steps"
        )
        assert steps_response.status_code == 200
        steps_payload = steps_response.json()["items"]
        assert len(steps_payload) == 1
        assert steps_payload[0]["status"] == "completed"
        assert steps_payload[0]["step_name"] == "generate_revision"

        versions_response = client.get(
            f"/api/inspection/sessions/{session_id}/versions"
        )
        assert versions_response.status_code == 200
        versions_payload = versions_response.json()["items"]
        revision_version = next(
            item
            for item in versions_payload
            if int(item["version_id"]) == target_revision_version_id
        )
        assert revision_version["revision_kind"] == "llm_assisted_revision"
        assert revision_version["source_version_id"] is not None
        assert revision_version["version_status"] == "requires_human_review"
        assert revision_version["llm_qa_status"] == "passed_with_warnings"
        assert revision_version["clinical_review_status"] == "not_reviewed"

        artifacts_response = client.get(
            f"/api/inspection/sessions/{session_id}/versions/{target_revision_version_id}/artifacts"
        )
        assert artifacts_response.status_code == 200
        artifacts_payload = artifacts_response.json()["items"]
        assert any(
            item["artifact_key"] == "revision_qa_validation"
            for item in artifacts_payload
        )

        entities_response = client.get(
            f"/api/inspection/sessions/{session_id}/versions/{target_revision_version_id}/entities"
        )
        assert entities_response.status_code == 200
        entities_payload = entities_response.json()["items"]
        assert any(item["entity_type"] == "drug" for item in entities_payload)
        assert any(item["entity_type"] == "dili_assessment" for item in entities_payload)

        compare_response = client.get(
            f"/api/inspection/sessions/{session_id}/versions/1/compare/{target_revision_version_id}"
        )
        assert compare_response.status_code == 200
        compare_payload = compare_response.json()
        assert compare_payload["report_text_diff"]["changed"] is True
        assert compare_payload["qa_summary"]["right_llm_qa_status"] == "passed_with_warnings"

        review_update = client.put(
            f"/api/inspection/sessions/{session_id}/versions/{target_revision_version_id}/clinical-review",
            json={
                "clinical_review_status": "approved_by_human",
                "reviewer_note": "Approved after review.",
                "reviewed_by": "Clinical Reviewer",
                "metadata": {"source": "api-integration"},
            },
        )
        assert review_update.status_code == 200
        review_update_payload = review_update.json()
        assert (
            review_update_payload["version"]["clinical_review_status"]
            == "approved_by_human"
        )
        assert review_update_payload["review_action"]["reviewer_note"] == (
            "Approved after review."
        )

        reviews_response = client.get(
            f"/api/inspection/sessions/{session_id}/versions/{target_revision_version_id}/reviews"
        )
        assert reviews_response.status_code == 200
        reviews_payload = reviews_response.json()["items"]
        assert len(reviews_payload) == 1
        assert reviews_payload[0]["clinical_review_status"] == "approved_by_human"
