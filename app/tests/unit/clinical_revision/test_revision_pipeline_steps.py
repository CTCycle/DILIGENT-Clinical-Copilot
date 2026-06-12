from __future__ import annotations

from datetime import datetime

import configurations.llm_configs as llm_configs_module
from domain.clinical import ClinicalSessionRequest
from repositories.schemas.models import Base
from repositories.serialization.data import DataSerializer
from services.inspection.service import DataInspectionService
from services.runtime.jobs import JobManager
from sqlalchemy import create_engine

import services.inspection.service as inspection_service_module
import services.inspection.revision_runner as revision_runner_module


###############################################################################
def build_service() -> tuple[DataInspectionService, DataSerializer]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    serializer = DataSerializer(engine=engine)
    return DataInspectionService(serializer=serializer, jobs=JobManager()), serializer


###############################################################################
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
                "matched_drugs": [
                    {
                        "raw_drug_name": "drug-c",
                        "matched_drug_name": "drug-c",
                        "match_status": "matched_with_excerpt",
                        "match_confidence": 0.99,
                    }
                ],
                "rucam_assessments": [
                    {
                        "drug_name": "drug-c",
                        "total_score": 5,
                        "causality_category": "possible",
                    }
                ],
            },
        }
    )
    if session_id is None:
        raise AssertionError("Session seed failed")
    return session_id


###############################################################################
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
    captured_revision_context: dict[str, object] = {}

    ###############################################################################
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

    ###############################################################################
    class FakeModelConfigSerializer:
        save_calls = 0

        # -------------------------------------------------------------------------
        def load_snapshot(self) -> FakeSnapshot:
            return FakeSnapshot()

        # -------------------------------------------------------------------------
        def save_snapshot(self, **kwargs) -> None:
            type(self).save_calls += 1
            return None

    ###############################################################################
    class FakeClinicalService:

        # -------------------------------------------------------------------------
        def apply_persisted_runtime_configuration(self) -> None:
            return None

        # -------------------------------------------------------------------------
        async def preprocess_unified_input(self, request):
            return request, {"history_of_present_illness": "Extracted HPI"}

        # -------------------------------------------------------------------------
        async def prepare_revision_source_request(self, *, session_detail, use_rag):
            request = ClinicalSessionRequest(
                name=session_detail.get("patient_name"),
                visit_date=session_detail.get("visit_date"),
                clinical_input=session_detail.get("session_text"),
                use_rag=use_rag,
            )
            preprocessed_request, section_extraction = await self.preprocess_unified_input(
                request
            )
            return preprocessed_request, section_extraction, "reparsed_source_text"

        # -------------------------------------------------------------------------
        def build_patient_payload(self, preprocessed_request):
            return {
                "name": preprocessed_request.name,
                "clinical_input": preprocessed_request.clinical_input,
            }

        # -------------------------------------------------------------------------
        async def process_single_patient(
            self,
            *args,
            **kwargs,
        ):
            raise AssertionError("Revision runner should not call process_single_patient")

        # -------------------------------------------------------------------------
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
            captured_revision_context["focus_context"] = revision_focus_context
            captured_revision_context["instruction_profile"] = session_metadata.get("instruction_profile")
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
                        "matched_drugs": [
                            {
                                "raw_drug_name": "drug-c",
                                "matched_drug_name": "drug-c",
                                "match_status": "matched_with_excerpt",
                                "match_confidence": 0.98,
                            }
                        ],
                        "rucam_assessments": [
                            {
                                "drug_name": "drug-c",
                                "total_score": 7,
                                "causality_category": "probable",
                            }
                        ],
                        "section_extraction": section_extraction,
                    },
                }
            )
            if persisted_session_id is None:
                raise AssertionError("Failed to persist revised session")
            return {
                "session_id": persisted_session_id,
                "report": "Revised report",
                "matched_drugs": [
                    {
                        "raw_drug_name": "drug-c",
                        "matched_drug_name": "drug-c",
                        "match_status": "matched_with_excerpt",
                        "match_confidence": 0.98,
                    }
                ],
                "rucam_assessments": [
                    {
                        "drug_name": "drug-c",
                        "total_score": 7,
                        "causality_category": "probable",
                    }
                ],
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
                    "consultation_context_metadata": {
                        "source_version_id": int(source_version["version_id"]),
                        "revision_version_id": int(target_shell["version_id"]),
                        "pipeline_run_id": pipeline_run_id,
                        "has_previous_report": True,
                        "previous_assessment_count": 1,
                        "previous_assessment_summaries": ["drug-c: possible"],
                    },
                    "consultation_execution": {
                        "analysis_entrypoint": "run_revision_analysis",
                        "drug_analysis_entrypoint": "request_revision_drug_analysis",
                        "report_finalization_entrypoint": "finalize_revision_patient_report",
                        "conclusion_entrypoint": "generate_revision_conclusion",
                        "synthesis_mode": "revision_comparison_aware",
                        "analysis_drug_names": ["drug-c"],
                        "snapshot_context_present": True,
                        "consultation_context_length": 146,
                        "source_version_id": int(source_version["version_id"]),
                        "revision_version_id": int(target_shell["version_id"]),
                        "pipeline_run_id": pipeline_run_id,
                        "has_previous_report_context": True,
                        "previous_assessment_count": 1,
                    },
                    "finalization_execution": {
                        "final_report_present": True,
                        "generated_report_present": True,
                        "manual_review_required": False,
                        "blocking_issue_count": 0,
                        "comparison_outcome": "aligned",
                        "consultation_model": "override-clinical",
                    },
                    "entity_pipeline": {
                        "resolve_revision_extraction": {
                            "status": "resolved",
                            "therapy_deterministic_names": [],
                            "anamnesis_deterministic_names": [],
                            "disease_deterministic_names": [],
                            "therapy_structured_names": ["drug-c"],
                            "anamnesis_structured_names": ["past-drug"],
                            "therapy_unresolved_lines": [],
                            "anamnesis_unresolved_lines": [],
                            "anamnesis_regimen_lines": [],
                            "disease_unresolved_lines": [],
                        },
                        "validate_anamnesis_drugs": {
                            "status": "supplemented",
                            "deterministic_detected_names": ["past-drug"],
                            "revised_detected_names": ["past-drug", "drug-c"],
                            "overlapping_names": ["past-drug"],
                            "deterministic_only_names": [],
                            "revised_only_names": ["drug-c"],
                            "unresolved_lines": [],
                        },
                        "extract_missing_anamnesis_drugs": {
                            "status": "supplemented",
                            "supplemental_drug_names": ["drug-c"],
                            "supplemental_entries": [
                                {"name": "drug-c", "source": "anamnesis"}
                            ],
                        },
                        "revise_labs_timeline": {
                            "status": "revised",
                            "lab_entry_count": 0,
                            "source_counts": {},
                            "marker_names": [],
                            "onset_context_present": False,
                            "pattern_classification": "hepatocellular",
                        },
                        "reconcile_revision_candidates": {
                            "status": "reconciled",
                            "analysis_drug_names": ["drug-c"],
                            "relevant_drug_names": ["drug-c"],
                            "excluded_drug_names": [],
                            "unresolved_drug_names": [],
                        },
                        "merge_revision_snapshot": {
                            "status": "merged",
                            "therapy_drug_names": ["drug-c"],
                            "anamnesis_drug_names": ["past-drug"],
                            "disease_names": ["autoimmune hepatitis"],
                            "lab_marker_names": [],
                            "analysis_drug_names": ["drug-c"],
                            "relevant_drug_names": ["drug-c"],
                            "excluded_drug_names": [],
                            "unresolved_drug_names": [],
                            "rucam_assessment_count": 1,
                        },
                    },
                    "entity_snapshot_context": (
                        "Revision anamnesis additions:\ndrug-c\n\n"
                        "Revision supplemental anamnesis drugs:\ndrug-c\n\n"
                        "Revision analysis drugs:\ndrug-c\n\n"
                        "Revision disease context:\nautoimmune hepatitis"
                    ),
                    "consultation_drug_names": ["drug-c"],
                },
            }

    monkeypatch.setattr(
        revision_runner_module,
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
    assert captured_revision_context["focus_context"] == (
        "Reviewer-selected source excerpt:\nALT trend\n\n"
        "Reviewer instruction summary:\nFocus on chronology.\n\n"
        "Target sections:\nunknown\n\n"
        "Target entities:\nother"
    )
    assert isinstance(captured_revision_context["instruction_profile"], dict)
    assert FakeModelConfigSerializer.save_calls == 0
    assert llm_configs_module.LLMRuntimeConfig.get_clinical_model() == "baseline-clinical"
    assert llm_configs_module.LLMRuntimeConfig.get_text_extraction_model() == "baseline-extraction"
    assert llm_configs_module.LLMRuntimeConfig.is_cloud_enabled() is False
    steps = service.list_revision_steps(pipeline_run_id)
    assert [step["step_name"] for step in steps] == [
        "load_source_version",
        "analyze_reviewer_instructions",
        "prepare_runtime",
        "preprocess_input",
        "generate_revision",
        "resolve_revision_extraction",
        "validate_anamnesis_drugs",
        "extract_missing_anamnesis_drugs",
        "revise_labs_timeline",
        "reconcile_revision_candidates",
        "merge_revision_snapshot",
        "resolve_livertox_matches",
        "rerun_dili_assessments",
        "rebuild_final_report",
        "qa_validate_revision",
        "persist_revision",
        "finalize_revision_version",
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
    entities = serializer.list_revision_entities_for_version(
        revision_version_id=int(target_shell["version_id"]),
    )
    assert [artifact["artifact_kind"] for artifact in artifacts] == [
        "llm_qa_output",
        "llm_qa_output",
        "pipeline_artifact",
        "pipeline_artifact",
        "pipeline_artifact",
        "pipeline_artifact",
        "pipeline_artifact",
        "pipeline_artifact",
        "pipeline_artifact",
        "pipeline_artifact",
        "report_comparison",
        "structured_case_entity",
        "structured_case_entity",
        "structured_case_entity",
    ]
    assert artifacts[0]["status"] == "passed"
    assert artifacts[1]["artifact_key"] == "revision_qa_validation"
    assert artifacts[1]["status"] == "passed"
    assert artifacts[2]["artifact_key"] == "fact_graph_validation"
    assert artifacts[3]["artifact_key"] == "reviewer_instruction_profile"
    assert artifacts[4]["artifact_key"] == "reviewer_instruction_trace"
    assert artifacts[5]["artifact_key"] == "final_report_rebuild"
    assert artifacts[6]["artifact_key"] == "revision_entity_pipeline"
    assert artifacts[7]["artifact_key"] == "revision_entity_snapshot_context"
    assert artifacts[8]["artifact_key"] == "revision_consultation_execution"
    assert artifacts[9]["artifact_key"] == "revision_finalization_execution"
    assert artifacts[10]["artifact_key"] == "report_comparison"
    assert (
        version_detail["session"]["result_payload"]["revision"]["execution_mode"]
        == "revision"
    )
    assert (
        version_detail["session"]["result_payload"]["revision"]["instruction_profile"][
            "instruction_summary"
        ]
        == "Focus on chronology."
    )
    assert (
        version_detail["session"]["result_payload"]["revision"]["instruction_trace"][
            "prompt_injection_detected"
        ]
        is False
    )
    assert (
        version_detail["session"]["result_payload"]["revision"]["qa_validation"][
            "status"
        ]
        == "passed"
    )
    analyze_step = next(
        step for step in steps if step["step_name"] == "analyze_reviewer_instructions"
    )
    assert analyze_step["output_summary"]["prompt_injection_detected"] is False
    assert analyze_step["output_summary"]["prompt_injection_flag_count"] == 0
    assert (
        version_detail["session"]["result_payload"]["revision"][
            "livertox_revision_decisions"
        ][0]["decision"]
        == "reused_high_confidence_previous_match"
    )
    assert (
        version_detail["session"]["result_payload"]["revision"][
            "livertox_revision_decisions"
        ][0]["source"]
        == "previous_version"
    )
    assert (
        version_detail["session"]["result_payload"]["revision"][
            "revised_dili_assessments"
        ][0]["previous_assessment_present"]
        is True
    )
    assert (
        version_detail["session"]["result_payload"]["revision"][
            "revised_dili_assessments"
        ][0]["changes_from_previous_version"]
        == [
            "Causality changed from possible to probable.",
            "Total score changed from 5 to 7.",
        ]
    )
    assert (
        version_detail["session"]["result_payload"]["revision"]["entity_pipeline"][
            "resolve_revision_extraction"
        ]["status"]
        == "resolved"
    )
    assert (
        version_detail["session"]["result_payload"]["revision"][
            "entity_snapshot_context"
        ]
        == "Revision anamnesis additions:\ndrug-c\n\n"
        "Revision supplemental anamnesis drugs:\ndrug-c\n\n"
        "Revision analysis drugs:\ndrug-c\n\n"
        "Revision disease context:\nautoimmune hepatitis"
    )
    assert (
        version_detail["session"]["result_payload"]["revision"][
            "consultation_drug_names"
        ]
        == ["drug-c"]
    )
    assert (
        version_detail["session"]["result_payload"]["revision"][
            "consultation_context_metadata"
        ]["pipeline_run_id"]
        == pipeline_run_id
    )
    assert (
        version_detail["session"]["result_payload"]["revision"][
            "consultation_execution"
        ]["analysis_entrypoint"]
        == "run_revision_analysis"
    )
    assert (
        version_detail["session"]["result_payload"]["revision"][
            "consultation_execution"
        ]["conclusion_entrypoint"]
        == "generate_revision_conclusion"
    )
    assert (
        version_detail["session"]["result_payload"]["revision"][
            "consultation_execution"
        ]["analysis_drug_names"]
        == ["drug-c"]
    )
    assert (
        version_detail["session"]["result_payload"]["revision"][
            "finalization_execution"
        ]["comparison_outcome"]
        == "aligned"
    )
    assert [entity["entity_type"] for entity in entities] == [
        "dili_assessment",
        "disease",
        "drug",
        "drug",
        "livertox_match",
    ]
    assert entities[0]["source_section"] == "therapy"
    assert entities[1]["source_section"] == "anamnesis"
    assert entities[2]["source_section"] == "anamnesis"
    assert entities[3]["source_section"] == "therapy"
    assert entities[4]["step_name"] == "resolve_livertox_matches"
    assert version_detail["version"]["session_id"] == result["session_id"]
    assert version_detail["version"]["version_status"] == "llm_qa_passed"
    assert version_detail["version"]["llm_qa_status"] == "passed"


###############################################################################
def test_run_revision_job_reuses_persisted_source_sections(monkeypatch) -> None:
    service, serializer = build_service()
    session_id = serializer.save_clinical_session(
        {
            "patient_name": "Persisted Source Patient",
            "session_timestamp": datetime(2025, 1, 6, 9, 0),
            "version": 1,
            "anamnesis": "Source text for revision",
            "drugs": "Current therapy text",
            "session_result_payload": {
                "original_session_text": (
                    "Anamnesis:\nHistorical exposure\n\n"
                    "Therapy:\nCurrent therapy text\n\n"
                    "Laboratory analysis:\nALT 120 U/L"
                ),
                "report": "Initial report",
                "section_extraction": {
                    "source_text": (
                        "Anamnesis:\nHistorical exposure\n\n"
                        "Therapy:\nCurrent therapy text\n\n"
                        "Laboratory analysis:\nALT 120 U/L"
                    ),
                    "anamnesis": "Historical exposure",
                    "drugs": "Current therapy text",
                    "laboratory_analysis": "ALT 120 U/L",
                    "line_ranges": {},
                    "confidence": 1.0,
                    "metadata": {"parser": "persisted-test"},
                },
            },
        }
    )
    assert session_id is not None
    session_detail = service.get_session_detail(session_id)
    assert session_detail is not None
    source_version = serializer.get_version_record_for_session(session_id)
    assert source_version is not None

    pipeline_run_id = "pipe-steps-persisted-001"
    target_shell = serializer.create_revision_version_shell(
        session_id,
        reviewer_note="Reuse persisted source sections",
        configuration={
            "selected_text": None,
            "selected_text_present": False,
            "revision_instruction": "Check chronology only.",
            "model_overrides": {},
            "metadata": {"reviewer": "Reviewer D"},
        },
        pipeline_run_id=pipeline_run_id,
        initiated_by="Reviewer D",
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
        configuration={"metadata": {"reviewer": "Reviewer D"}},
        reviewer_note="Reuse persisted source sections",
        status="running",
        initiated_by="Reviewer D",
        actor_source="manual_entry",
        actor_confidence="unverified",
    )

    captured_section_extraction: dict[str, object] = {}

    ###############################################################################
    class FakeClinicalService:

        # -------------------------------------------------------------------------
        def apply_persisted_runtime_configuration(self) -> None:
            return None

        # -------------------------------------------------------------------------
        async def preprocess_unified_input(self, request):
            raise AssertionError(
                "Revision runner should reuse persisted source sections before reparsing"
            )

        # -------------------------------------------------------------------------
        async def prepare_revision_source_request(self, *, session_detail, use_rag):
            return await inspection_service_module.build_clinical_session_service(
                JobManager()
            ).prepare_revision_source_request(
                session_detail=session_detail,
                use_rag=use_rag,
            )

        # -------------------------------------------------------------------------
        def build_patient_payload(self, preprocessed_request):
            return {
                "name": preprocessed_request.name,
                "clinical_input": preprocessed_request.clinical_input,
                "anamnesis": preprocessed_request.anamnesis,
                "drugs": preprocessed_request.drugs,
                "laboratory_analysis": preprocessed_request.laboratory_analysis,
            }

        # -------------------------------------------------------------------------
        async def process_single_patient(self, *args, **kwargs):
            raise AssertionError("Revision runner should not call process_single_patient")

        # -------------------------------------------------------------------------
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
            section_extraction_payload = (
                section_extraction.model_dump()
                if hasattr(section_extraction, "model_dump")
                else section_extraction
            )
            captured_section_extraction["value"] = section_extraction_payload
            persisted_session_id = serializer.save_clinical_session(
                {
                    "patient_name": patient_payload["name"],
                    "session_timestamp": datetime(2025, 1, 7, 9, 0),
                    "version": session_version,
                    "original_session_id": original_session_id,
                    "anamnesis": original_session_text,
                    "drugs": "Current therapy text",
                    "session_result_payload": {
                        "original_session_text": original_session_text,
                        "report": "Revised report",
                        "matched_drugs": [],
                        "rucam_assessments": [],
                        "section_extraction": section_extraction_payload,
                    },
                }
            )
            assert persisted_session_id is not None
            return {
                "session_id": persisted_session_id,
                "report": "Revised report",
                "matched_drugs": [],
                "rucam_assessments": [],
                "section_extraction": section_extraction_payload,
                "structured_case": {
                    "therapy_drugs": [{"name": "current-therapy", "role": "suspect"}],
                    "anamnesis_drugs": [{"name": "historical-exposure", "role": "historical"}],
                    "anamnesis_diseases": [],
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
                    }
                },
                "revision": {
                    "execution_mode": "revision",
                    "consultation_context_metadata": {
                        "source_version_id": int(source_version["version_id"]),
                        "revision_version_id": int(target_shell["version_id"]),
                        "pipeline_run_id": pipeline_run_id,
                        "has_previous_report": True,
                        "previous_assessment_count": 0,
                        "previous_assessment_summaries": [],
                    },
                    "consultation_execution": {
                        "analysis_entrypoint": "run_revision_analysis",
                        "drug_analysis_entrypoint": "request_revision_drug_analysis",
                        "report_finalization_entrypoint": "finalize_revision_patient_report",
                        "conclusion_entrypoint": "generate_revision_conclusion",
                        "synthesis_mode": "revision_comparison_aware",
                        "analysis_drug_names": [],
                        "snapshot_context_present": False,
                        "consultation_context_length": 0,
                        "source_version_id": int(source_version["version_id"]),
                        "revision_version_id": int(target_shell["version_id"]),
                        "pipeline_run_id": pipeline_run_id,
                        "has_previous_report_context": True,
                        "previous_assessment_count": 0,
                    },
                    "finalization_execution": {
                        "final_report_present": True,
                        "generated_report_present": False,
                        "manual_review_required": False,
                        "blocking_issue_count": 0,
                        "comparison_outcome": "aligned",
                        "consultation_model": None,
                    },
                },
            }

    original_builder = revision_runner_module.build_clinical_session_service

    def build_fake_service(_jobs):
        fake_service = FakeClinicalService()
        real_service = original_builder(_jobs)
        fake_service.prepare_revision_source_request = (  # type: ignore[method-assign]
            real_service.prepare_revision_source_request
        )
        return fake_service

    monkeypatch.setattr(
        revision_runner_module,
        "build_clinical_session_service",
        build_fake_service,
    )

    result = service.run_revision_job(
        job_id=None,
        pipeline_run_id=pipeline_run_id,
        source_version_id=int(source_version["version_id"]),
        target_revision_version_id=int(target_shell["version_id"]),
        session_detail=session_detail,
        root_session_id=session_id,
        version=2,
        selected_text=None,
        revision_instruction="Check chronology only.",
        model_overrides={},
        metadata={"reviewer": "Reviewer D"},
    )

    assert isinstance(result.get("session_id"), int)
    assert isinstance(captured_section_extraction["value"], dict)
    assert captured_section_extraction["value"]["anamnesis"] == "Historical exposure"
    assert captured_section_extraction["value"]["drugs"] == "Current therapy text"
    assert captured_section_extraction["value"]["laboratory_analysis"] == "ALT 120 U/L"
    preprocess_step = next(
        step
        for step in service.list_revision_steps(pipeline_run_id)
        if step["step_name"] == "preprocess_input"
    )
    assert preprocess_step["output_summary"]["source_mode"] == "persisted_section_extraction"
    assert preprocess_step["output_summary"]["reparsed_source_text"] is False
