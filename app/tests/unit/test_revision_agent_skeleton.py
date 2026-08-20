from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from common.prompts.revision_agent import editor_prompt
from domain.inspection import (
    RevisionAgentToolCall,
    RevisionIssueScanResult,
    SessionRevisionRequest,
)
from pydantic import ValidationError
from repositories.schemas.base import Base
from repository_fixtures import build_repository_graph
from services.inspection.revision_agent import (
    RevisionAgentRunner,
    build_revision_agent_user_prompt,
)
from services.inspection.revision_scaffold import SessionRevisionConflictError
from services.inspection.service import DataInspectionService
from services.runtime.jobs import JobManager
from sqlalchemy import create_engine


###############################################################################
def build_file_serializer(tmp_path: Path) -> Any:
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'revision.db'}", future=True
    )
    Base.metadata.create_all(engine)
    return build_repository_graph(engine=engine)

###############################################################################
def save_revision_source_session(serializer: Any) -> int:
    session_id = serializer.clinical_session_repository.save_clinical_session(
        {
            "patient_name": "Revision Patient",
            "session_timestamp": datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
            "session_status": "successful",
            "anamnesis": "Patient has jaundice after antibiotic exposure.",
            "drugs": "Amoxicillin started 2026-01-01.",
            "laboratory_analysis": "ALT 400 U/L on 2026-01-10.",
            "final_report": "Possible DILI from amoxicillin.",
            "detected_drugs": ["Amoxicillin"],
            "session_result_payload": {
                "original_session_text": "Patient has jaundice after antibiotic exposure.",
                "report": "Possible DILI from amoxicillin.",
                "pipeline_artifacts": {
                    "structured_dili_report": "Structured dossier text."
                },
            },
        }
    )
    assert session_id is not None
    return int(session_id)

###############################################################################
def build_service(serializer: Any, jobs: JobManager) -> DataInspectionService:
    graph = build_repository_graph(
        engine=serializer.context.engine, session_factory=serializer.context.session_factory
    )
    return DataInspectionService(
        clinical_session_repository=graph.clinical_session_repository,
        drug_catalog_repository=graph.drug_catalog_repository,
        knowledge_repository=graph.knowledge_repository,
        session_timeline_repository=graph.session_timeline_repository,
        session_revision_repository=graph.session_revision_repository,
        jobs=jobs,
    )

###############################################################################
def build_runner(serializer: Any, **kwargs: Any) -> RevisionAgentRunner:
    graph = build_repository_graph(
        engine=serializer.context.engine, session_factory=serializer.context.session_factory
    )
    return RevisionAgentRunner(
        clinical_session_repository=graph.clinical_session_repository,
        session_revision_repository=graph.session_revision_repository,
        knowledge_repository=graph.knowledge_repository,
        **kwargs,
    )

###############################################################################
def fake_issue_scan_call(**kwargs: Any) -> dict[str, Any]:
    schema_name = kwargs["schema"].__name__
    if schema_name == "RevisionAgentPlan":
        return {
            "instruction_profile": "Review unsupported claims.",
            "evident_issues": ["report-only causality"],
            "tasks": [
                {
                    "task_id": "review-report",
                    "priority": "medium",
                    "objective": "Review report evidence.",
                    "affected_sections": ["final_report"],
                    "required_tools": [],
                    "stop_criteria": "Report reviewed.",
                }
            ],
            "expected_final_output_type": "revised_report",
        }
    if schema_name == "RevisionAgentToolCall":
        return {
            "tool_name": "read_session_context",
            "arguments": {},
            "rationale": "Read evidence.",
            "task_complete": True,
        }
    if schema_name == "RevisionDraftResult":
        return {
            "revised_report_text": "Possible DILI from amoxicillin.",
            "patches": [],
            "changed_sections": [],
            "unchanged_sections": ["final_report"],
            "unresolved_issues": ["Dechallenge is not documented."],
            "human_review_requirements": ["Clinical review required."],
            "entity_change_proposals": [],
        }
    if schema_name == "RevisionAgentQaResult":
        return {
            "blocking_issues": [],
            "warnings": ["No report text change proposed."],
            "supported_claim_count": 0,
            "manual_review_required": True,
        }
    raise AssertionError(f"Unexpected schema: {schema_name}")

###############################################################################
def test_revision_issue_scan_schema_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        RevisionIssueScanResult.model_validate(
            {
                "summary": "Invalid category",
                "issues": [
                    {
                        "category": "wrong",
                        "severity": "medium",
                        "affected_report_area": "report",
                        "evidence_status": "unclear",
                        "rationale": "bad category",
                        "recommended_next_action": "fix",
                    }
                ],
            }
        )

###############################################################################
def test_revision_agent_tool_call_accepts_provider_rationale_within_budget() -> None:
    decision = RevisionAgentToolCall.model_validate(
        {
            "tool_name": "read_session_context",
            "arguments": {},
            "rationale": "evidence " * 250,
            "task_complete": True,
        }
    )

    assert len(decision.rationale) > 1000

###############################################################################
def test_revision_prompt_merges_session_report_and_user_instruction() -> None:
    prompt = build_revision_agent_user_prompt(
        session={
            "session_id": 10,
            "patient_name": "Prompt Patient",
            "source_clinical_text": "Original clinical text",
            "sections": {"anamnesis": "Section anamnesis"},
            "official_report_text": "Generated clinical report",
            "result_payload": {"pipeline_artifacts": {"fact": "value"}},
        },
        request=SessionRevisionRequest(
            selected_text="Selected report sentence",
            revision_instruction="Check hallucinations around dechallenge.",
            metadata={"reviewer": "unit-test"},
        ),
    )

    assert "Original clinical text" in prompt
    assert "Generated clinical report" in prompt
    assert "Selected report sentence" in prompt
    assert "Check hallucinations around dechallenge." in prompt
    assert "may steer focus but is not clinical evidence" in prompt
    assert "No tools are available" in prompt

###############################################################################
def test_revision_editor_prompt_requires_exact_source_patches() -> None:
    prompt = editor_prompt(
        {"review_target": {"official_report": {"text": "Canonical report"}}},
        [],
    )

    assert "zero-based Python slice offsets" in prompt
    assert "expected_text must equal the exact source substring" in prompt
    assert "return patches as an empty list" in prompt

###############################################################################
def test_revision_job_persists_issue_scan_step_and_artifact(tmp_path: Path) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    jobs = JobManager()
    service = build_service(serializer, jobs)
    service.revision_agent_runner = build_runner(
        serializer,
        structured_call=fake_issue_scan_call,
    )

    started = service.start_revision_job(
        session_id,
        SessionRevisionRequest(
            revision_instruction="Focus on unsupported claims.",
            model_overrides={"clinical_model": "fake-revision-model"},
        ),
    )
    assert started["status"] in {"running", "completed"}
    assert started["job_type"] == service.REVISION_JOB_TYPE
    assert started["result"]["pipeline_run_id"]

    for _ in range(50):
        status = service.get_revision_job_status(started["job_id"])
        if status and status["status"] == "completed":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("Revision job did not complete")

    pipeline_run_id = started["result"]["pipeline_run_id"]
    run = service.get_revision_run(pipeline_run_id)
    assert run is not None
    assert run["status"] == "completed"
    assert run["revision_mode"] == "agentic_revision"

    steps = service.list_revision_steps(pipeline_run_id)
    assert len(steps) >= 1
    assert steps[0]["step_name"].startswith("revision_agent_task_")
    assert steps[0]["output_payload"]["observations"] == []

    revision_version_id = int(started["result"]["revision_version_id"])
    artifacts = service.list_revision_artifacts(
        session_id,
        version_id=revision_version_id,
    )
    assert len(artifacts) >= 4
    assert {item["artifact_key"] for item in artifacts} >= {
        "revision_agent_context",
        "revision_agent_plan",
        "revision_agent_draft_report",
        "revision_agent_qa",
    }

###############################################################################
def test_revision_agent_recovers_from_invalid_tool_arguments(tmp_path: Path) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    tool_decisions = 0

    def structured_call(**kwargs: Any) -> dict[str, Any]:
        nonlocal tool_decisions
        if kwargs["schema"].__name__ != "RevisionAgentToolCall":
            return fake_issue_scan_call(**kwargs)
        tool_decisions += 1
        if tool_decisions == 1:
            return {
                "tool_name": "get_livertox_excerpt",
                "arguments": {},
                "rationale": "Inspect the suspected drug evidence.",
                "task_complete": False,
            }
        return {
            "tool_name": "read_session_context",
            "arguments": {},
            "rationale": "Continue after correcting invalid tool input.",
            "task_complete": True,
        }

    service = build_service(serializer, JobManager())
    service.revision_agent_runner = build_runner(
        serializer,
        structured_call=structured_call,
    )

    started = service.start_revision_job(session_id, SessionRevisionRequest())
    for _ in range(50):
        status = service.get_revision_job_status(started["job_id"])
        if status and status["status"] == "completed":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("Revision job did not recover from invalid tool input")

    steps = service.list_revision_steps(started["result"]["pipeline_run_id"])
    observation = steps[0]["output_payload"]["observations"][0]["observation"]
    assert observation == {
        "error": "Tool ids must be positive integers.",
        "invalid_tool_input": True,
    }

###############################################################################
def test_revision_uses_latest_manual_edit_version(tmp_path: Path) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)

    serializer.session_revision_repository.update_current_report_text_with_manual_audit(
        session_id,
        report_text="Manually corrected DILI report.",
        edited_fields=["report_text"],
        reviewer_note="Corrected wording.",
        edited_by="Unit test",
        metadata={},
    )

    version = serializer.session_revision_repository.get_version_record_for_session(session_id)

    assert version is not None
    assert version["version_number"] == 2
    assert version["revision_kind"] == "manual_edit"

    service = build_service(serializer, JobManager())
    service.revision_agent_runner = build_runner(
        serializer,
        structured_call=fake_issue_scan_call,
    )

    started = service.start_revision_job(session_id, SessionRevisionRequest())

    assert started["job_type"] == service.REVISION_JOB_TYPE

###############################################################################
def test_manual_edit_skips_orphaned_revision_version_numbers(tmp_path: Path) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)

    serializer.session_revision_repository.create_revision_version_shell(
        session_id,
        reviewer_note="First failed revision.",
        configuration={"model": "deepseek-v4-flash"},
        pipeline_run_id="failed-revision-1",
    )
    serializer.session_revision_repository.create_revision_version_shell(
        session_id,
        reviewer_note="Second failed revision.",
        configuration={"model": "deepseek-v4-flash"},
        pipeline_run_id="failed-revision-2",
    )

    serializer.session_revision_repository.update_current_report_text_with_manual_audit(
        session_id,
        report_text="Manually corrected after failed revisions.",
        edited_fields=["report_text"],
        reviewer_note="Corrected wording.",
        edited_by="Unit test",
        metadata={},
    )

    version = serializer.session_revision_repository.get_version_record_for_session(session_id)
    assert version is not None
    assert version["version_number"] == 4
    assert version["revision_kind"] == "manual_edit"

###############################################################################
class SlowRevisionRunner:

    # -------------------------------------------------------------------------
    def run_agentic(self, **_kwargs: Any) -> dict[str, Any]:
        time.sleep(0.4)
        return {}

###############################################################################
class FailingRevisionRunner:

    # -------------------------------------------------------------------------
    def run_agentic(self, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Synthetic revision failure")

###############################################################################
def test_failed_revision_marks_persisted_run_failed(tmp_path: Path) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    service = build_service(serializer, JobManager())
    service.revision_agent_runner = FailingRevisionRunner()

    started = service.start_revision_job(session_id, SessionRevisionRequest())
    pipeline_run_id = started["result"]["pipeline_run_id"]
    for _ in range(20):
        status = service.get_revision_job_status(started["job_id"])
        if status and status["status"] == "failed":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("Revision job did not fail")

    run = service.get_revision_run(pipeline_run_id)
    assert run is not None
    assert run["status"] == "failed"
    assert run["error"] == {"message": "Revision processing failed. Retry the revision if needed."}

###############################################################################
def test_session_delete_cleans_revision_shell_and_run(tmp_path: Path) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    source_version = serializer.session_revision_repository.get_version_record_for_session(session_id)
    assert source_version is not None
    pipeline_run_id = "synthetic-delete-run"
    shell = serializer.session_revision_repository.create_revision_version_shell(
        session_id,
        reviewer_note="Synthetic cleanup validation.",
        configuration={},
        pipeline_run_id=pipeline_run_id,
    )
    assert shell is not None
    serializer.session_revision_repository.create_or_update_revision_run(
        pipeline_run_id=pipeline_run_id,
        session_id=session_id,
        root_session_id=session_id,
        source_version_id=int(source_version["version_id"]),
        target_revision_version_id=int(shell["revision_version_id"]),
        revision_mode="agentic_revision",
        revision_kind="llm_assisted_revision",
        configuration={},
        reviewer_note="Synthetic cleanup validation.",
        status="failed",
    )

    inspection_service = build_service(serializer, JobManager())
    assert inspection_service.delete_session(session_id) is True
    assert serializer.clinical_session_repository.get_session_detail(session_id) is None
    assert serializer.session_revision_repository.get_revision_run(pipeline_run_id) is None

###############################################################################
def test_incomplete_revision_shell_cannot_be_clinically_reviewed(tmp_path: Path) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    shell = serializer.session_revision_repository.create_revision_version_shell(
        session_id,
        reviewer_note=None,
        configuration={},
        pipeline_run_id="incomplete-review-run",
    )
    assert shell is not None

    with pytest.raises(ValueError, match="Only completed revision versions"):
        serializer.session_revision_repository.record_revision_review_action(
            revision_version_id=int(shell["revision_version_id"]),
            clinical_review_status="approved_by_human",
            reviewer_note=None,
            reviewed_by="QA",
        )

###############################################################################
def test_revision_job_rejects_same_root_concurrent_start(tmp_path: Path) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    service = build_service(serializer, JobManager())
    service.revision_agent_runner = SlowRevisionRunner()

    started = service.start_revision_job(session_id, SessionRevisionRequest())
    assert started["job_type"] == service.REVISION_JOB_TYPE

    with pytest.raises(SessionRevisionConflictError):
        service.start_revision_job(session_id, SessionRevisionRequest())

    for _ in range(20):
        status = service.get_revision_job_status(started["job_id"])
        if status and status["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.05)
