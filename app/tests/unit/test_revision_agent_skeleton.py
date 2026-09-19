from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import services.inspection.revision_agent as revision_agent_module
from common.prompts.revision_agent import editor_prompt, planner_prompt
from domain.inspection import (
    RevisionAgentPlan,
    RevisionAgentQaResult,
    RevisionAgentToolCall,
    RevisionDraftResult,
    RevisionIssueScanResult,
    SessionRevisionRequest,
)
from pydantic import ValidationError
from repositories.schemas.base import Base
from repository_fixtures import build_repository_graph
from services.inspection.revision_agent import (
    REVISION_PROVIDER_MAX_RETRIES,
    RevisionAgentRunner,
    RevisionAgentRuntime,
    _requested_append_sentence,
    build_revision_agent_user_prompt,
    revision_error_payload,
)
from services.inspection.revision_context import build_revision_context
from services.inspection.revision_scaffold import SessionRevisionConflictError
from services.inspection.service import DataInspectionService
from services.llm.cloud import LLMError
from services.llm.generation_policy import GenerationPurpose
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
        engine=serializer.context.engine,
        session_factory=serializer.context.session_factory,
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
        engine=serializer.context.engine,
        session_factory=serializer.context.session_factory,
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
    if schema_name == "RevisionIssueScanResult":
        return {"summary": "No issues detected."}
    raise AssertionError(f"Unexpected schema: {schema_name}")

###############################################################################
def fake_mismatched_patch_call(**kwargs: Any) -> dict[str, Any]:
    if kwargs["schema"].__name__ != "RevisionDraftResult":
        return fake_issue_scan_call(**kwargs)
    source = "Possible DILI from amoxicillin."
    replacement = "Clinician review required: Possible DILI from amoxicillin."
    return {
        "revised_report_text": "Model text that is not the patch result.",
        "patches": [
            {
                "start": 0,
                "end": len(source),
                "replacement": replacement,
                "expected_text": source,
                "evidence_references": ["dili_evidence_bundle"],
            }
        ],
        "changed_sections": ["final_report"],
        "unchanged_sections": [],
        "unresolved_issues": [],
        "human_review_requirements": ["Clinical review required."],
        "entity_change_proposals": [],
    }

###############################################################################
def fake_unsafe_revision_call(**kwargs: Any) -> dict[str, Any]:
    if kwargs["schema"].__name__ != "RevisionDraftResult":
        return fake_issue_scan_call(**kwargs)
    source = "Possible DILI from amoxicillin."
    replacement = "A cautious rechallenge under observation may be considered."
    return {
        "revised_report_text": replacement,
        "patches": [
            {
                "start": 0,
                "end": len(source),
                "replacement": replacement,
                "expected_text": source,
                "evidence_references": ["dili_evidence_bundle"],
            }
        ],
        "changed_sections": ["final_report"],
        "unchanged_sections": [],
        "unresolved_issues": [],
        "human_review_requirements": [],
        "entity_change_proposals": [],
    }

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
def test_revision_request_rejects_active_model_overrides() -> None:
    with pytest.raises(ValidationError):
        SessionRevisionRequest.model_validate(
            {"model_overrides": {"clinical_model": "x"}}
        )

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
    assert "may steer review focus but is not clinical evidence" in prompt
    assert "No tools are available" in prompt

###############################################################################
def test_revision_editor_prompt_requires_exact_source_patches() -> None:
    prompt = editor_prompt(
        {"review_target": {"official_report": {"text": "Canonical report"}}},
        [],
    )

    assert "zero-based Python slice offsets" in prompt
    assert "expected_text` must equal the exact source substring character-for-character" in prompt
    assert "return an empty `patches` list" in prompt

###############################################################################
def test_revision_prompts_serialize_context_as_strict_json() -> None:
    prompt = planner_prompt(
        {"enabled": True, "nested": {"items": [1, "two"]}},
        ["read_session_context"],
    )

    assert '"enabled":true' in prompt
    assert "'enabled'" not in prompt
    assert "True" not in prompt

###############################################################################
def test_revision_plan_accepts_25_evident_issues_but_keeps_eight_task_bound() -> None:
    plan = RevisionAgentPlan.model_validate(
        {
            "instruction_profile": "Preserve all distinct evidence issues.",
            "evident_issues": [f"issue-{index}" for index in range(25)],
            "tasks": [],
            "expected_final_output_type": "revised_report",
        }
    )

    assert len(plan.evident_issues) == 25
    with pytest.raises(ValidationError):
        RevisionAgentPlan.model_validate(
            {
                "instruction_profile": "Too many tasks.",
                "evident_issues": [],
                "tasks": [
                    {
                        "task_id": f"task-{index}",
                        "priority": "low",
                        "objective": "Review.",
                        "stop_criteria": "Reviewed.",
                    }
                    for index in range(9)
                ],
                "expected_final_output_type": "revised_report",
            }
        )

###############################################################################
def test_revision_error_payload_persists_sanitized_provider_status() -> None:
    error = LLMError(
        "Cloud provider returned HTTP 530",
        error_code="upstream_error",
        retryable=True,
        provider="opencode_go",
        model="deepseek-v4-flash",
        operation="structured_output",
        status_code=530,
        request_id="req-revision-530",
        provider_detail="api_key=secret-value",
    )

    payload = revision_error_payload(error)

    assert payload["error_code"] == "upstream_error"
    assert payload["retryable"] is True
    assert payload["status_code"] == 530
    assert payload["provider"] == "opencode_go"
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["request_id"] == "req-revision-530"
    assert "secret-value" not in str(payload)
    assert "provider_detail" not in payload

###############################################################################
def test_revision_context_exposes_exact_section_offsets_and_truncation_state() -> None:
    report = "# Summary\nSupported.\n\n## Global Conclusion\nReview required.\n\n# Follow-up\nPending."
    context = build_revision_context(
        session={"session_id": 10, "report": report},
        manual_edits=[],
        lineage=[],
        selected_text=None,
        instruction=None,
        input_budget=100000,
    )

    canonical = context["review_target"]["official_report"]
    conclusion = next(
        section
        for section in canonical["sections"]
        if section["heading"] == "Global Conclusion"
    )
    assert canonical["editability"] == {
        "available": True,
        "editable": True,
        "reason": "available",
        "source_length": len(report),
    }
    assert report[conclusion["start"] : conclusion["end"]] == conclusion["text"]
    assert report[conclusion["start"] : conclusion["end"]].startswith(
        "## Global Conclusion"
    )

    truncated_report = "x" * 20001
    truncated_context = build_revision_context(
        session={"session_id": 11, "report": truncated_report},
        manual_edits=[],
        lineage=[],
        selected_text=None,
        instruction=None,
        input_budget=100000,
    )
    truncated = truncated_context["review_target"]["official_report"]
    assert truncated["editability"]["reason"] == "truncated"
    assert truncated["editability"]["editable"] is False
    assert truncated["source_length"] == len(truncated_report)

###############################################################################
def test_revision_runner_does_not_auto_append_when_canonical_context_is_truncated() -> None:
    request = SessionRevisionRequest(
        revision_instruction="Append exactly this sentence to the revised report: Human review is required."
    )
    draft = RevisionDraftResult(
        revised_report_text="Canonical report.",
        patches=[],
    )

    normalized, applied = RevisionAgentRunner._normalize_draft(
        source_report="Canonical report.",
        draft=draft,
        request=request,
        canonical_report_editability={
            "available": True,
            "editable": False,
            "reason": "truncated",
        },
    )

    assert applied == "Canonical report."
    assert normalized.patches == []
    assert any("not fully available" in issue for issue in normalized.unresolved_issues)

###############################################################################
def test_requested_append_sentence_is_extracted_from_explicit_instruction() -> None:
    instruction = (
        "Append exactly this sentence to the revised report: "
        "Reviewer instruction check: human clinical review is required before reuse. "
        "Preserve all existing clinical facts and sections."
    )

    assert _requested_append_sentence(instruction) == (
        "Reviewer instruction check: human clinical review is required before reuse."
    )
    assert _requested_append_sentence("Focus on unsupported claims.") is None

###############################################################################
def test_revision_context_preserves_long_canonical_report() -> None:
    report = "Canonical report text. " * 700
    context = build_revision_context(
        session={
            "session_id": 10,
            "report": report,
            "result_payload": {"report": report},
        },
        manual_edits=[],
        lineage=[],
        selected_text=None,
        instruction=None,
        input_budget=100000,
    )

    canonical = context["review_target"]["official_report"]
    assert canonical["text"] == report
    assert canonical["truncated"] is False
    assert canonical["sha256"]

###############################################################################
def test_revision_agent_assigns_stage_specific_generation_purposes(
    tmp_path: Path,
) -> None:
    serializer = build_file_serializer(tmp_path)
    calls: list[GenerationPurpose] = []

    def structured_call(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["purpose"])
        return fake_issue_scan_call(**kwargs)

    runner = build_runner(serializer, structured_call=structured_call)
    runtime = RevisionAgentRuntime(provider="ollama", model="revision-model")
    for schema, purpose in (
        (RevisionAgentPlan, GenerationPurpose.REVISION_PLANNING),
        (RevisionAgentToolCall, GenerationPurpose.REVISION_TOOL_SELECTION),
        (RevisionDraftResult, GenerationPurpose.REVISION_EDITING),
        (RevisionAgentQaResult, GenerationPurpose.REVISION_QA),
    ):
        runner._call_schema(runtime, "{}", schema, purpose=purpose)

    runner._run_structured_scan(runtime=runtime, user_prompt="{}")

    assert calls == [
        GenerationPurpose.REVISION_PLANNING,
        GenerationPurpose.REVISION_TOOL_SELECTION,
        GenerationPurpose.REVISION_EDITING,
        GenerationPurpose.REVISION_QA,
        GenerationPurpose.REVISION_SCAN,
    ]

###############################################################################
def test_revision_issue_scan_allows_bounded_provider_repair_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    provider_factory: dict[str, object] = {}

    ###############################################################################
    class FakeProvider:

        # -------------------------------------------------------------------------
        async def llm_structured_call(self, **kwargs: object) -> RevisionIssueScanResult:
            captured.update(kwargs)
            return RevisionIssueScanResult(summary="No issues detected.")

    monkeypatch.setattr(
        revision_agent_module,
        "select_llm_provider",
        lambda **kwargs: (provider_factory.update(kwargs) or FakeProvider()),
    )
    runner = object.__new__(RevisionAgentRunner)
    runner.structured_call = None

    result = runner._run_structured_scan(
        runtime=RevisionAgentRuntime(
            provider="opencode_go",
            model="deepseek-v4-flash",
        ),
        user_prompt="{}",
    )

    assert result.summary == "No issues detected."
    assert captured["max_repair_attempts"] == 3
    assert captured["purpose"] is GenerationPurpose.REVISION_SCAN
    assert REVISION_PROVIDER_MAX_RETRIES == 2
    assert provider_factory["max_retries"] == 2

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
    task_step = next(
        step for step in steps if step["step_name"].startswith("revision_agent_task_")
    )
    assert task_step["output_payload"]["observations"] == []
    result = service.get_revision_job_status(started["job_id"])["result"]
    assert result["revision_status"] == "requires_human_review"
    assert result["revised_session_id"] is None
    assert result["repair_attempt"] == 1
    source_session = serializer.clinical_session_repository.get_session_detail(session_id)
    assert source_session is not None
    assert source_session["report"] == "Possible DILI from amoxicillin."

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
def test_revision_repairs_noop_draft_once_before_accepting_validated_patch(
    tmp_path: Path,
) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    draft_calls = 0

    def structured_call(**kwargs: Any) -> dict[str, Any]:
        nonlocal draft_calls
        if kwargs["schema"].__name__ != "RevisionDraftResult":
            return fake_issue_scan_call(**kwargs)
        draft_calls += 1
        if draft_calls == 1:
            return fake_issue_scan_call(**kwargs)
        source = "Possible DILI from amoxicillin."
        replacement = "Clinician review required: Possible DILI from amoxicillin."
        return {
            "revised_report_text": "",
            "patches": [
                {
                    "start": 0,
                    "end": len(source),
                    "replacement": replacement,
                    "expected_text": source,
                    "evidence_references": ["dili_evidence_bundle"],
                }
            ],
            "changed_sections": ["final_report"],
            "unchanged_sections": [],
            "unresolved_issues": [],
            "human_review_requirements": ["Clinical review required."],
            "entity_change_proposals": [],
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
        raise AssertionError("Revision repair job did not complete")

    result = status["result"]
    assert result["revision_status"] == "llm_qa_passed"
    assert result["repair_attempt"] == 1
    assert result["revised_session_id"] is not None
    artifacts = service.list_revision_artifacts(
        session_id,
        version_id=int(result["revision_version_id"]),
    )
    assert {item["artifact_key"] for item in artifacts} >= {
        "revision_agent_draft_report_repair",
        "revision_agent_qa_repair",
    }
    steps = service.list_revision_steps(started["result"]["pipeline_run_id"])
    editor_steps = [step for step in steps if step["step_name"] == "revision_agent_editor"]
    qa_steps = [step for step in steps if step["step_name"] == "revision_agent_qa"]
    assert [step["attempt_number"] for step in editor_steps] == [1, 2]
    assert [step["attempt_number"] for step in qa_steps] == [1, 2]
    source_session = serializer.clinical_session_repository.get_session_detail(session_id)
    revised_session = serializer.clinical_session_repository.get_session_detail(
        int(result["revised_session_id"])
    )
    assert source_session is not None
    assert revised_session is not None
    assert source_session["report"] == "Possible DILI from amoxicillin."
    assert revised_session["report"] == (
        "Clinician review required: Possible DILI from amoxicillin."
    )

###############################################################################
def test_revision_persists_deterministic_patch_when_model_text_differs(
    tmp_path: Path,
) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    service = build_service(serializer, JobManager())
    service.revision_agent_runner = build_runner(
        serializer,
        structured_call=fake_mismatched_patch_call,
    )

    started = service.start_revision_job(session_id, SessionRevisionRequest())
    for _ in range(50):
        status = service.get_revision_job_status(started["job_id"])
        if status and status["status"] == "completed":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("Revision job did not complete")

    result = status["result"]
    assert result["revision_status"] == "llm_qa_passed"
    assert result["revised_session_id"] is not None
    artifacts = service.list_revision_artifacts(
        session_id,
        version_id=int(result["revision_version_id"]),
    )
    draft = next(
        item
        for item in artifacts
        if item["artifact_key"] == "revision_agent_draft_report"
    )
    assert draft["payload"]["revised_report_text"] == (
        "Clinician review required: Possible DILI from amoxicillin."
    )
    assert any(
        "deterministic patch output" in issue
        for issue in draft["payload"]["unresolved_issues"]
    )
    revised_session = serializer.clinical_session_repository.get_session_detail(
        int(result["revised_session_id"])
    )
    assert revised_session is not None
    assert revised_session["report"] == draft["payload"]["revised_report_text"]
    assert revised_session["sections"]["drugs"] == "Amoxicillin started 2026-01-01."

###############################################################################
def test_accepted_revision_reload_and_subsequent_revision_preserve_lineage(
    tmp_path: Path,
) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    original_version = serializer.session_revision_repository.get_version_record_for_session(
        session_id
    )
    assert original_version is not None

    first_source = "Possible DILI from amoxicillin."
    first_replacement = "Clinician review required: Possible DILI from amoxicillin."
    second_replacement = "Clinician review required before reuse: Possible DILI from amoxicillin."

    def patch_call(source: str, replacement: str) -> Any:
        def structured_call(**kwargs: Any) -> dict[str, Any]:
            if kwargs["schema"].__name__ != "RevisionDraftResult":
                return fake_issue_scan_call(**kwargs)
            return {
                "revised_report_text": "",
                "patches": [
                    {
                        "start": 0,
                        "end": len(source),
                        "replacement": replacement,
                        "expected_text": source,
                        "evidence_references": ["dili_evidence_bundle"],
                    }
                ],
                "changed_sections": ["final_report"],
                "unchanged_sections": [],
                "unresolved_issues": [],
                "human_review_requirements": ["Clinical review required."],
                "entity_change_proposals": [],
            }

        return structured_call

    first_service = build_service(serializer, JobManager())
    first_service.revision_agent_runner = build_runner(
        serializer,
        structured_call=patch_call(first_source, first_replacement),
    )
    first_started = first_service.start_revision_job(session_id, SessionRevisionRequest())
    for _ in range(50):
        first_status = first_service.get_revision_job_status(first_started["job_id"])
        if first_status and first_status["status"] == "completed":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("First accepted revision did not complete")

    first_result = first_status["result"]
    assert first_result["revision_status"] == "llm_qa_passed"
    accepted_session_id = int(first_result["revised_session_id"])
    accepted_version = serializer.session_revision_repository.get_version_record_for_session(
        accepted_session_id
    )
    assert accepted_version is not None
    assert accepted_version["session_id"] == accepted_session_id
    first_run = first_service.get_revision_run(first_result["pipeline_run_id"])
    assert first_run is not None
    assert first_run["source_version_id"] == original_version["version_id"]

    reloaded_service = build_service(serializer, JobManager())
    reloaded_session = reloaded_service.get_session_detail(accepted_session_id)
    assert reloaded_session is not None
    assert reloaded_session["report"] == first_replacement
    reloaded_service.revision_agent_runner = build_runner(
        serializer,
        structured_call=patch_call(first_replacement, second_replacement),
    )
    second_started = reloaded_service.start_revision_job(
        accepted_session_id,
        SessionRevisionRequest(
            revision_instruction="Clarify the report conclusion before reuse."
        ),
    )
    for _ in range(50):
        second_status = reloaded_service.get_revision_job_status(second_started["job_id"])
        if second_status and second_status["status"] == "completed":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("Subsequent accepted revision did not complete")

    second_result = second_status["result"]
    assert second_result["revision_status"] == "llm_qa_passed"
    second_run = reloaded_service.get_revision_run(second_result["pipeline_run_id"])
    assert second_run is not None
    assert second_run["source_version_id"] == accepted_version["version_id"]
    second_session = reloaded_service.get_session_detail(
        int(second_result["revised_session_id"])
    )
    assert second_session is not None
    assert second_session["report"] == second_replacement
    original_session = reloaded_service.get_session_detail(session_id)
    assert original_session is not None
    assert original_session["report"] == first_source

###############################################################################
def test_revision_safety_gate_keeps_rechallenge_draft_out_of_sessions(
    tmp_path: Path,
) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    service = build_service(serializer, JobManager())
    service.revision_agent_runner = build_runner(
        serializer,
        structured_call=fake_unsafe_revision_call,
    )

    started = service.start_revision_job(session_id, SessionRevisionRequest())
    for _ in range(50):
        status = service.get_revision_job_status(started["job_id"])
        if status and status["status"] == "completed":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("Revision safety-gated job did not complete")

    result = status["result"]
    assert result["revision_status"] == "qa_failed"
    assert result["revised_session_id"] is None
    artifacts = service.list_revision_artifacts(
        session_id,
        version_id=int(result["revision_version_id"]),
    )
    qa = next(item for item in artifacts if item["artifact_key"] == "revision_agent_qa")
    assert qa["status"] == "qa_failed"
    assert any(
        "rechallenge" in issue.lower() for issue in qa["payload"]["blocking_issues"]
    )
    version_detail = serializer.session_revision_repository.get_session_version_detail(
        session_id,
        version_id=int(result["revision_version_id"]),
    )
    assert version_detail is not None
    assert version_detail["version"]["session_id"] is None
    assert version_detail["version"]["version_status"] == "qa_failed"

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
    task_step = next(
        step for step in steps if step["step_name"].startswith("revision_agent_task_")
    )
    observation = task_step["output_payload"]["observations"][0]["observation"]
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

    version = serializer.session_revision_repository.get_version_record_for_session(
        session_id
    )

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
    for _ in range(50):
        status = service.get_revision_job_status(started["job_id"])
        if status and status["status"] == "completed":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("Revision job did not complete")

    artifacts = service.list_revision_artifacts(
        session_id,
        version_id=int(started["result"]["revision_version_id"]),
    )
    context_artifact = next(
        item for item in artifacts if item["artifact_key"] == "revision_agent_context"
    )
    assert len(context_artifact["payload"]["audit"]["manual_edits"]) == 1
    assert (
        context_artifact["payload"]["audit"]["manual_edits"][0]["current_version_id"]
        == version["version_id"]
    )
    assert context_artifact["payload"]["audit"]["version_lineage"]

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

    version = serializer.session_revision_repository.get_version_record_for_session(
        session_id
    )
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


def test_running_revision_persists_job_recovery_metadata(tmp_path: Path) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    service = build_service(serializer, JobManager())
    service.revision_agent_runner = SlowRevisionRunner()

    started = service.start_revision_job(session_id, SessionRevisionRequest())
    version_id = int(started["result"]["revision_version_id"])
    version_detail = serializer.session_revision_repository.get_session_version_detail(
        session_id,
        version_id=version_id,
    )
    assert version_detail is not None
    configuration = version_detail["version"]["model_configuration"]
    assert configuration["job_id"] == started["job_id"]
    assert configuration["pipeline_run_id"] == started["result"]["pipeline_run_id"]

    for _ in range(20):
        status = service.get_revision_job_status(started["job_id"])
        if status and status["status"] != "running":
            break
        time.sleep(0.05)


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
    assert run["error"] == {
        "message": "Revision processing failed. Retry the revision if needed."
    }

###############################################################################
def test_revision_provider_failure_persists_sanitized_retry_metadata(
    tmp_path: Path,
) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)

    def failing_provider_call(**_: Any) -> dict[str, Any]:
        raise LLMError(
            "Cloud provider returned HTTP 503",
            error_code="upstream_error",
            retryable=True,
            provider="opencode_go",
            model="deepseek-v4-flash",
            operation="structured_output",
            status_code=503,
            request_id="req-revision-503",
            provider_detail="authorization=Bearer secret-value",
        )

    service = build_service(serializer, JobManager())
    service.revision_agent_runner = build_runner(
        serializer,
        structured_call=failing_provider_call,
    )

    started = service.start_revision_job(session_id, SessionRevisionRequest())
    for _ in range(50):
        status = service.get_revision_job_status(started["job_id"])
        if status and status["status"] == "failed":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("Provider failure job did not fail")

    run = service.get_revision_run(started["result"]["pipeline_run_id"])
    assert run is not None
    assert run["error"]["error_code"] == "upstream_error"
    assert run["error"]["retryable"] is True
    assert run["error"]["status_code"] == 503
    assert run["error"]["provider"] == "opencode_go"
    assert run["error"]["model"] == "deepseek-v4-flash"
    assert "secret-value" not in str(run["error"])
    assert "provider_detail" not in run["error"]
    steps = service.list_revision_steps(started["result"]["pipeline_run_id"])
    planner_step = next(
        step for step in steps if step["step_name"] == "revision_agent_planner"
    )
    assert planner_step["error"]["status_code"] == 503
    assert planner_step["error"]["retryable"] is True

###############################################################################
def test_cancelled_revision_finalizes_version_and_steps(tmp_path: Path) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    source_version = (
        serializer.session_revision_repository.get_version_record_for_session(
            session_id
        )
    )
    assert source_version is not None
    pipeline_run_id = "cancelled-revision-run"
    shell = serializer.session_revision_repository.create_revision_version_shell(
        session_id,
        reviewer_note="Synthetic cancellation validation.",
        configuration={"job_id": "cancelled-job"},
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
        configuration={"job_id": "cancelled-job"},
        reviewer_note="Synthetic cancellation validation.",
        status="running",
    )
    serializer.session_revision_repository.start_revision_step(
        pipeline_run_id=pipeline_run_id,
        step_name="revision_agent_task_1",
        step_index=1,
        step_count=1,
    )

    serializer.session_revision_repository.cancel_revision_run(
        pipeline_run_id=pipeline_run_id
    )

    run = serializer.session_revision_repository.get_revision_run(pipeline_run_id)
    assert run is not None
    assert run["status"] == "cancelled"
    version_detail = serializer.session_revision_repository.get_session_version_detail(
        session_id,
        version_id=int(shell["revision_version_id"]),
    )
    assert version_detail is not None
    assert version_detail["version"]["version_status"] == "cancelled"
    assert version_detail["version"]["llm_qa_status"] == "not_run"
    assert version_detail["version"]["session_id"] is None
    assert version_detail["version"]["completed_at"] is not None
    steps = serializer.session_revision_repository.list_revision_steps(pipeline_run_id)
    assert [step["status"] for step in steps] == ["cancelled"]

    serializer.session_revision_repository.complete_revision_step(
        pipeline_run_id=pipeline_run_id,
        step_name="revision_agent_task_1",
        attempt_number=1,
        output_summary={"late": True},
    )
    serializer.session_revision_repository.fail_revision_step(
        pipeline_run_id=pipeline_run_id,
        step_name="revision_agent_task_1",
        attempt_number=1,
        error={"message": "Late worker failure."},
    )
    serializer.session_revision_repository.fail_revision_run(
        pipeline_run_id=pipeline_run_id,
        error={"message": "Late worker failure."},
    )
    serializer.session_revision_repository.create_or_update_revision_run(
        pipeline_run_id=pipeline_run_id,
        session_id=session_id,
        root_session_id=session_id,
        source_version_id=int(source_version["version_id"]),
        target_revision_version_id=int(shell["revision_version_id"]),
        revision_mode="agentic_revision",
        revision_kind="llm_assisted_revision",
        configuration={"job_id": "cancelled-job", "late_worker": True},
        reviewer_note="Synthetic cancellation validation.",
        status="completed",
    )

    serializer.session_revision_repository.cancel_revision_run(
        pipeline_run_id=pipeline_run_id
    )
    run_after_retry = serializer.session_revision_repository.get_revision_run(
        pipeline_run_id
    )
    assert run_after_retry is not None
    assert run_after_retry["status"] == "cancelled"
    assert [
        step["status"]
        for step in serializer.session_revision_repository.list_revision_steps(
            pipeline_run_id
        )
    ] == ["cancelled"]

###############################################################################
def test_finalized_revision_wins_over_late_cancellation(tmp_path: Path) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    source_version = (
        serializer.session_revision_repository.get_version_record_for_session(
            session_id
        )
    )
    assert source_version is not None
    pipeline_run_id = "finalized-before-cancel-run"
    shell = serializer.session_revision_repository.create_revision_version_shell(
        session_id,
        reviewer_note="Synthetic finalization race validation.",
        configuration={"job_id": "finalized-before-cancel-job"},
        pipeline_run_id=pipeline_run_id,
        source_version_id=int(source_version["version_id"]),
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
        configuration={"job_id": "finalized-before-cancel-job"},
        reviewer_note="Synthetic finalization race validation.",
        status="running",
    )

    serializer.session_revision_repository.finalize_revision_version(
        pipeline_run_id=pipeline_run_id,
        persisted_session_id=session_id,
        version_status="llm_qa_passed",
        llm_qa_status="passed",
        clinical_review_status="not_reviewed",
    )
    serializer.session_revision_repository.cancel_revision_run(
        pipeline_run_id=pipeline_run_id
    )

    run = serializer.session_revision_repository.get_revision_run(pipeline_run_id)
    assert run is not None
    assert run["status"] == "completed"
    version_detail = serializer.session_revision_repository.get_session_version_detail(
        session_id,
        version_id=int(shell["revision_version_id"]),
    )
    assert version_detail is not None
    assert version_detail["version"]["version_status"] == "llm_qa_passed"
    assert version_detail["version"]["llm_qa_status"] == "passed"
    assert version_detail["version"]["session_id"] == session_id

###############################################################################
def test_session_delete_cleans_revision_shell_and_run(tmp_path: Path) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    source_version = (
        serializer.session_revision_repository.get_version_record_for_session(
            session_id
        )
    )
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
    assert (
        serializer.session_revision_repository.get_revision_run(pipeline_run_id) is None
    )

###############################################################################
def test_incomplete_revision_shell_cannot_be_clinically_reviewed(
    tmp_path: Path,
) -> None:
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

###############################################################################
def test_revision_shell_uses_explicit_source_version(tmp_path: Path) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    original = serializer.session_revision_repository.get_version_record_for_session(session_id)
    assert original is not None
    serializer.session_revision_repository.update_current_report_text_with_manual_audit(
        session_id,
        report_text="Manually corrected DILI report.",
        edited_fields=["report_text"],
        reviewer_note="Create a newer source version.",
        edited_by="Unit test",
        metadata={},
    )
    latest = serializer.session_revision_repository.get_version_record_for_session(session_id)
    assert latest is not None
    assert latest["version_id"] != original["version_id"]
    shell = serializer.session_revision_repository.create_revision_version_shell(
        session_id,
        reviewer_note=None,
        configuration={},
        pipeline_run_id="explicit-source-version",
        source_version_id=int(original["version_id"]),
    )
    assert shell is not None
    assert shell["source_version_id"] == original["version_id"]
