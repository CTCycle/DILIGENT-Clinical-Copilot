from __future__ import annotations

from datetime import datetime

import configurations.llm_configs as llm_configs_module
from repositories.schemas.models import Base
from repositories.serialization.data import DataSerializer
from services.inspection.service import DataInspectionService
from services.runtime.jobs import JobManager
from sqlalchemy import create_engine

import services.inspection.service as inspection_service_module


def build_service() -> tuple[DataInspectionService, DataSerializer]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    serializer = DataSerializer(engine=engine)
    return DataInspectionService(serializer=serializer, jobs=JobManager()), serializer


def seed_session(serializer: DataSerializer) -> int:
    session_id = serializer.save_clinical_session(
        {
            "patient_name": "Revision Step Patient",
            "session_timestamp": datetime(2025, 1, 4, 9, 0),
            "version": 1,
            "anamnesis": "Source text for revision",
            "drugs": "drug-c",
            "session_result_payload": {
                "original_session_text": "Source text for revision",
                "report": "Initial report",
            },
        }
    )
    if session_id is None:
        raise AssertionError("Session seed failed")
    return session_id


def test_run_revision_job_persists_step_lifecycle(monkeypatch) -> None:
    service, serializer = build_service()
    session_id = seed_session(serializer)
    session_detail = service.get_session_detail(session_id)
    assert session_detail is not None
    source_version = serializer.get_version_record_for_session(session_id)
    assert source_version is not None

    pipeline_run_id = "pipe-steps-001"
    target_shell = serializer.create_revision_version_shell(
        session_id,
        reviewer_note="Investigate chronology",
        configuration={
            "selected_text": "ALT trend",
            "selected_text_present": True,
            "revision_instruction": "Focus on chronology.",
            "model_overrides": {},
            "metadata": {"reviewer": "Reviewer C"},
        },
        pipeline_run_id=pipeline_run_id,
        initiated_by="Reviewer C",
    )
    assert target_shell is not None
    serializer.create_or_update_revision_run(
        pipeline_run_id=pipeline_run_id,
        session_id=session_id,
        root_session_id=session_id,
        source_version_id=int(source_version["version_id"]),
        target_revision_version_id=int(target_shell["version_id"]),
        revision_mode="instruction_guided",
        revision_kind="llm_assisted_revision",
        configuration={"metadata": {"reviewer": "Reviewer C"}},
        reviewer_note="Investigate chronology",
        status="running",
        initiated_by="Reviewer C",
        actor_source="manual_entry",
        actor_confidence="unverified",
    )

    captured_runtime: dict[str, object] = {}
    captured_entrypoint: dict[str, object] = {}

    class FakeSnapshot:
        clinical_model = "baseline-clinical"
        text_extraction_model = "baseline-extraction"
        use_cloud_models = False
        cloud_provider = None
        cloud_model = None
        ollama_temperature = None
        cloud_temperature = None
        ollama_reasoning = None
        rag_settings = {}
        updated_at = None

    class FakeModelConfigSerializer:
        save_calls = 0

        def load_snapshot(self) -> FakeSnapshot:
            return FakeSnapshot()

        def save_snapshot(self, **kwargs) -> None:
            type(self).save_calls += 1
            return None

    class FakeClinicalService:
        def apply_persisted_runtime_configuration(self) -> None:
            return None

        async def preprocess_unified_input(self, request):
            return request, {"history_of_present_illness": "Extracted HPI"}

        def build_patient_payload(self, preprocessed_request):
            return {
                "name": preprocessed_request.name,
                "clinical_input": preprocessed_request.clinical_input,
            }

        async def process_single_patient(
            self,
            *args,
            **kwargs,
        ):
            raise AssertionError("Revision runner should not call process_single_patient")

        async def process_revision_patient(
            self,
            patient_payload,
            *,
            section_extraction,
            session_version,
            original_session_id,
            session_metadata,
            original_session_text,
            revision_focus_context,
            progress_callback,
            stop_check,
        ):
            captured_entrypoint["method"] = "process_revision_patient"
            captured_runtime["clinical_model"] = llm_configs_module.LLMRuntimeConfig.get_clinical_model()
            captured_runtime["text_extraction_model"] = llm_configs_module.LLMRuntimeConfig.get_text_extraction_model()
            captured_runtime["use_cloud_models"] = llm_configs_module.LLMRuntimeConfig.is_cloud_enabled()
            captured_runtime["cloud_provider"] = llm_configs_module.LLMRuntimeConfig.get_llm_provider()
            captured_runtime["cloud_model"] = llm_configs_module.LLMRuntimeConfig.get_cloud_model()
            persisted_session_id = serializer.save_clinical_session(
                {
                    "patient_name": patient_payload["name"],
                    "session_timestamp": datetime(2025, 1, 5, 9, 0),
                    "version": session_version,
                    "original_session_id": original_session_id,
                    "anamnesis": original_session_text,
                    "drugs": "drug-c",
                    "session_result_payload": {
                        "original_session_text": original_session_text,
                        "report": "Revised report",
                        "matched_drugs": ["drug-c"],
                        "rucam_assessments": [],
                        "section_extraction": section_extraction,
                    },
                }
            )
            if persisted_session_id is None:
                raise AssertionError("Failed to persist revised session")
            return {
                "session_id": persisted_session_id,
                "report": "Revised report",
                "matched_drugs": ["drug-c"],
                "rucam_assessments": [],
                "section_extraction": section_extraction,
                "structured_case": {
                    "therapy_drugs": [{"name": "drug-c", "role": "suspect"}],
                    "anamnesis_drugs": [{"name": "past-drug", "role": "historical"}],
                    "anamnesis_diseases": [{"name": "autoimmune hepatitis"}],
                },
                "manual_review_required": False,
                "blocking_issues": [],
                "report_comparison": {
                    "outcome": "aligned",
                    "manual_review": "no",
                },
                "pipeline_artifacts": {
                    "faithfulness_audit": {
                        "manual_review_required": False,
                        "blocking_issues": [],
                    },
                    "fact_graph_validation": {
                        "is_valid": True,
                    },
                },
                "revision": {
                    "execution_mode": "revision",
                },
            }

    monkeypatch.setattr(
        inspection_service_module,
        "build_clinical_session_service",
        lambda jobs: FakeClinicalService(),
    )
    monkeypatch.setattr(
        llm_configs_module,
        "ModelConfigSerializer",
        FakeModelConfigSerializer,
    )

    result = service.run_revision_job(
        job_id=None,
        pipeline_run_id=pipeline_run_id,
        source_version_id=int(source_version["version_id"]),
        target_revision_version_id=int(target_shell["version_id"]),
        session_detail=session_detail,
        root_session_id=session_id,
        version=2,
        selected_text="ALT trend",
        revision_instruction="Focus on chronology.",
        model_overrides={
            "clinical_model": "override-clinical",
            "text_extraction_model": "override-parser",
            "use_cloud_services": True,
            "provider": "openai",
            "cloud_model": "gpt-4.1-mini",
        },
        metadata={"reviewer": "Reviewer C", "revision_note": "Investigate chronology"},
    )

    assert isinstance(result.get("session_id"), int)
    assert captured_entrypoint == {"method": "process_revision_patient"}
    assert captured_runtime == {
        "clinical_model": "override-clinical",
        "text_extraction_model": "override-parser",
        "use_cloud_models": True,
        "cloud_provider": "openai",
        "cloud_model": "gpt-4.1-mini",
    }
    assert FakeModelConfigSerializer.save_calls == 0
    assert llm_configs_module.LLMRuntimeConfig.get_clinical_model() == "baseline-clinical"
    assert llm_configs_module.LLMRuntimeConfig.get_text_extraction_model() == "baseline-extraction"
    assert llm_configs_module.LLMRuntimeConfig.is_cloud_enabled() is False
    steps = service.list_revision_steps(pipeline_run_id)
    assert [step["step_name"] for step in steps] == [
        "prepare_runtime",
        "preprocess_input",
        "generate_revision",
        "persist_revision",
    ]
    assert all(step["status"] == "completed" for step in steps)
    assert all(step["attempt_number"] == 1 for step in steps)
    assert all(step["completed_at"] is not None for step in steps)

    run = service.get_revision_run(pipeline_run_id)
    assert run is not None
    assert run["status"] == "completed"

    version_detail = serializer.get_session_version_detail(
        session_id,
        version_id=int(target_shell["version_id"]),
    )
    assert version_detail is not None
    artifacts = serializer.list_revision_artifacts_for_version(
        revision_version_id=int(target_shell["version_id"]),
    )
    assert [artifact["artifact_kind"] for artifact in artifacts] == [
        "llm_qa_output",
        "pipeline_artifact",
        "report_comparison",
        "structured_case_entity",
        "structured_case_entity",
        "structured_case_entity",
    ]
    assert artifacts[0]["status"] == "passed"
    assert artifacts[1]["artifact_key"] == "fact_graph_validation"
    assert artifacts[2]["artifact_key"] == "report_comparison"
    assert (
        version_detail["session"]["result_payload"]["revision"]["execution_mode"]
        == "revision"
    )
    assert version_detail["version"]["session_id"] == result["session_id"]
    assert version_detail["version"]["version_status"] == "llm_qa_passed"
    assert version_detail["version"]["llm_qa_status"] == "passed"
