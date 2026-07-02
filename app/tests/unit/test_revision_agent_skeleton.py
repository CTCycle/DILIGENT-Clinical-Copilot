from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine

from domain.inspection import RevisionIssueScanResult, SessionRevisionRequest
from repositories.schemas.models import Base
from repositories.serialization.data import DataSerializer
from services.inspection.revision_agent import (
    REVISION_AGENT_STEP_NAME,
    RevisionAgentRunner,
    build_revision_agent_user_prompt,
)
from services.inspection.revision_scaffold import SessionRevisionConflictError
from services.inspection.service import DataInspectionService
from services.runtime.jobs import JobManager

###############################################################################
def build_file_serializer(tmp_path: Path) -> DataSerializer:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'revision.db'}", future=True)
    Base.metadata.create_all(engine)
    return DataSerializer(engine=engine)

###############################################################################
def save_revision_source_session(serializer: DataSerializer) -> int:
    session_id = serializer.save_clinical_session(
        {
            "patient_name": "Revision Patient",
            "session_timestamp": datetime(2026, 1, 15, 10, 0),
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
def fake_issue_scan_call(**_kwargs: Any) -> dict[str, Any]:
    return {
        "summary": "One unsupported report claim requires review.",
        "issues": [
            {
                "category": "unsupported_claim",
                "severity": "medium",
                "affected_report_area": "causality summary",
                "evidence_status": "report_only",
                "source_evidence": None,
                "missing_evidence_statement": "The report states causality without source support.",
                "rationale": "The source text does not provide dechallenge information.",
                "recommended_next_action": "Review chronology and dechallenge evidence.",
                "tool_intents": [
                    {
                        "tool_name": "timeline_review",
                        "reason": "Clarify exposure and lab chronology.",
                        "target": "therapy timeline",
                        "proposed_inputs": {"drug": "Amoxicillin"},
                    }
                ],
            }
        ],
        "tool_intents": [],
        "limits": ["No tools were available during this scan."],
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
def test_revision_job_persists_issue_scan_step_and_artifact(tmp_path: Path) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    jobs = JobManager()
    service = DataInspectionService(serializer=serializer, jobs=jobs)
    service.revision_agent_runner = RevisionAgentRunner(
        serializer=serializer,
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
    assert run["revision_mode"] == "agent_issue_scan"

    steps = service.list_revision_steps(pipeline_run_id)
    assert len(steps) == 1
    assert steps[0]["step_name"] == REVISION_AGENT_STEP_NAME
    assert steps[0]["output_payload"]["issues"][0]["category"] == "unsupported_claim"

    revision_version_id = int(started["result"]["revision_version_id"])
    artifacts = service.list_revision_artifacts(
        session_id,
        version_id=revision_version_id,
    )
    assert len(artifacts) == 1
    assert artifacts[0]["artifact_key"] == "revision_agent_issue_scan"
    assert artifacts[0]["payload"]["issues"][0]["tool_intents"][0]["tool_name"] == (
        "timeline_review"
    )

###############################################################################
class SlowRevisionRunner:

    # -------------------------------------------------------------------------
    def run_issue_scan(self, **_kwargs: Any) -> dict[str, Any]:
        time.sleep(0.4)
        return {}

###############################################################################
def test_revision_job_rejects_same_root_concurrent_start(tmp_path: Path) -> None:
    serializer = build_file_serializer(tmp_path)
    session_id = save_revision_source_session(serializer)
    service = DataInspectionService(serializer=serializer, jobs=JobManager())
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
