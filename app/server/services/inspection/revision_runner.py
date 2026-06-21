from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from functools import partial
from typing import Any

from configurations.llm_configs import LLMRuntimeConfig
from domain.inspection import (
    ReviewerInstructionProfile,
    ReviewerInstructionTrace,
)
from services.clinical.revision.qa import build_revision_qa_validation_payload
from services.clinical.revision.report_builder import (
    build_revision_final_report_payload,
)
from services.inspection.revision_runner_support import (
    REVISION_STEP_SEQUENCE,
    derive_revision_qa_outcome,
    derive_revision_run_actor_source,
    ensure_revision_not_cancelled,
    get_revision_entity_pipeline,
    report_revision_progress,
    summarize_revision_entity_stage_payload,
)
from services.session.factory import build_clinical_session_service

###############################################################################
def build_revision_job_scope_key(root_session_id: int) -> str:
    return f"revision:{int(root_session_id)}"

###############################################################################
class InspectionRevisionRunnerMixin:

    # -------------------------------------------------------------------------
    def _start_revision_background_job(
        self,
        *,
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
        job_id = self.jobs.start_job(
            job_type=self.REVISION_JOB_TYPE,
            runner=self.run_revision_job,
            scope_key=build_revision_job_scope_key(root_session_id),
            kwargs={
                "job_id": None,
                "pipeline_run_id": pipeline_run_id,
                "source_version_id": int(source_version_id),
                "target_revision_version_id": int(target_revision_version_id),
                "session_detail": session_detail,
                "root_session_id": root_session_id,
                "version": version,
                "selected_text": selected_text,
                "revision_instruction": revision_instruction,
                "model_overrides": model_overrides,
                "metadata": metadata,
            },
        )
        status_payload = self.jobs.get_job_status(job_id)
        if status_payload is None:
            raise RuntimeError("Failed to initialize revision job")
        self.patch_job_result(
            job_id=job_id,
            patch={
                "pipeline_run_id": pipeline_run_id,
                "target_revision_version_id": int(target_revision_version_id),
            },
        )
        return status_payload

    # -------------------------------------------------------------------------
    def _record_revision_step_start(
        self,
        *,
        pipeline_run_id: str,
        step_name: str,
        input_summary: dict[str, Any] | None = None,
        input_payload: Any = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        step_index = next(
            (
                index
                for index, (candidate_name, _) in enumerate(
                    REVISION_STEP_SEQUENCE, start=1
                )
                if candidate_name == step_name
            ),
            None,
        )
        if step_index is None:
            raise ValueError(f"Unknown revision step: {step_name}")
        return self.serializer.start_revision_step(
            pipeline_run_id=pipeline_run_id,
            step_name=step_name,
            step_index=step_index,
            step_count=len(REVISION_STEP_SEQUENCE),
            input_summary=input_summary,
            input_payload=input_payload,
            model_name=model_name,
        )

    # -------------------------------------------------------------------------
    def _record_revision_step_success(
        self,
        *,
        pipeline_run_id: str,
        step_name: str,
        attempt_number: int,
        started_at: datetime,
        output_summary: dict[str, Any] | None = None,
        output_payload: dict[str, Any] | None = None,
    ) -> None:
        latency_ms = int((datetime.now(UTC) - started_at).total_seconds() * 1000)
        self.serializer.complete_revision_step(
            pipeline_run_id=pipeline_run_id,
            step_name=step_name,
            attempt_number=attempt_number,
            output_summary=output_summary,
            output_payload=output_payload,
            latency_ms=latency_ms,
        )

    # -------------------------------------------------------------------------
    def _record_revision_step_failure(
        self,
        *,
        pipeline_run_id: str,
        step_name: str,
        attempt_number: int,
        started_at: datetime,
        exc: Exception,
    ) -> None:
        latency_ms = int((datetime.now(UTC) - started_at).total_seconds() * 1000)
        self.serializer.fail_revision_step(
            pipeline_run_id=pipeline_run_id,
            step_name=step_name,
            attempt_number=attempt_number,
            error={"message": str(exc)[:500]},
            latency_ms=latency_ms,
        )

    # -------------------------------------------------------------------------
    def start_revision_job(
        self,
        session_id: int,
        *,
        selected_text: str | None,
        revision_instruction: str | None,
        model_overrides: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        detail = self.get_session_detail(session_id)
        if detail is None:
            raise ValueError("Session not found")
        source_text = str(detail.get("session_text") or "").strip()
        if not source_text:
            raise ValueError("Session text is empty")
        root_session_id = int(detail.get("original_session_id") or session_id)
        if self.jobs.is_job_running(
            self.REVISION_JOB_TYPE,
            scope_key=build_revision_job_scope_key(root_session_id),
        ):
            raise ValueError("Session revision is already running for this session")
        version = self.serializer.get_next_session_version(root_session_id)
        source_version = self.serializer.get_version_record_for_session(session_id)
        if source_version is None:
            raise ValueError("Session version history is unavailable")
        run_configuration = self._build_revision_run_configuration(
            selected_text=selected_text,
            revision_instruction=revision_instruction,
            model_overrides=model_overrides,
            metadata=metadata,
        )
        revision_mode = (
            "instruction_guided"
            if str(revision_instruction or "").strip()
            else "default"
        )
        pipeline_run_id = uuid.uuid4().hex
        target_shell = self.serializer.create_revision_version_shell(
            session_id,
            reviewer_note=str(metadata.get("revision_note") or "").strip() or None,
            configuration=run_configuration,
            pipeline_run_id=pipeline_run_id,
            initiated_by=str(metadata.get("reviewer") or "").strip() or None,
        )
        if target_shell is None:
            raise RuntimeError("Failed to create revision version shell")
        self.serializer.create_or_update_revision_run(
            pipeline_run_id=pipeline_run_id,
            session_id=int(session_id),
            root_session_id=root_session_id,
            source_version_id=int(source_version["version_id"]),
            target_revision_version_id=int(target_shell["version_id"]),
            revision_mode=revision_mode,
            revision_kind="llm_assisted_revision",
            configuration=run_configuration,
            reviewer_note=str(metadata.get("revision_note") or "").strip() or None,
            status="running",
            initiated_by=str(metadata.get("reviewer") or "").strip() or None,
            actor_source=derive_revision_run_actor_source(metadata),
            actor_confidence="unverified",
            started_at=datetime.now(UTC),
            trace_id=pipeline_run_id,
        )
        return self._start_revision_background_job(
            pipeline_run_id=pipeline_run_id,
            source_version_id=int(source_version["version_id"]),
            target_revision_version_id=int(target_shell["version_id"]),
            session_detail=detail,
            root_session_id=root_session_id,
            version=version,
            selected_text=selected_text,
            revision_instruction=revision_instruction,
            model_overrides=model_overrides,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    def retry_revision_job(self, pipeline_run_id: str) -> dict[str, Any]:
        run = self.serializer.get_revision_run(pipeline_run_id)
        if run is None:
            raise ValueError("Revision pipeline run not found")
        root_session_id = int(run["root_session_id"])
        if self.jobs.is_job_running(
            self.REVISION_JOB_TYPE,
            scope_key=build_revision_job_scope_key(root_session_id),
        ):
            raise ValueError("Session revision is already running for this session")
        if str(run.get("status") or "").casefold() == "running":
            raise ValueError("Session revision is already running")
        target_revision_version_id = run.get("target_revision_version_id")
        if not isinstance(target_revision_version_id, int):
            raise ValueError(
                "Revision pipeline run is missing its target version shell"
            )
        target_detail = self.serializer.get_session_version_detail(
            int(run["session_id"]),
            version_id=target_revision_version_id,
        )
        if target_detail is None:
            raise ValueError("Target revision version not found")
        target_version = target_detail["version"]
        if target_version["session_id"] is not None:
            raise ValueError("Completed revision runs cannot be retried")
        if target_version["version_status"] != "draft_revision":
            raise ValueError("Only draft revision shells can be retried")
        source_detail = self.serializer.get_session_version_detail(
            int(run["session_id"]),
            version_id=int(run["source_version_id"]),
        )
        if source_detail is None or source_detail.get("session") is None:
            raise ValueError("Source version session could not be loaded")
        configuration = run.get("configuration")
        if not isinstance(configuration, dict):
            configuration = {}
        metadata = configuration.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        selected_text_value = configuration.get("selected_text")
        selected_text = (
            str(selected_text_value).strip()
            if isinstance(selected_text_value, str)
            else None
        ) or None
        revision_instruction_value = configuration.get("revision_instruction")
        revision_instruction = (
            str(revision_instruction_value).strip()
            if isinstance(revision_instruction_value, str)
            else None
        ) or None
        model_overrides = configuration.get("model_overrides")
        if not isinstance(model_overrides, dict):
            model_overrides = {}
        revision_mode = "instruction_guided" if revision_instruction else "default"
        self.serializer.create_or_update_revision_run(
            pipeline_run_id=pipeline_run_id,
            session_id=int(run["session_id"]),
            root_session_id=int(run["root_session_id"]),
            source_version_id=int(run["source_version_id"]),
            target_revision_version_id=target_revision_version_id,
            revision_mode=revision_mode,
            revision_kind="llm_assisted_revision",
            configuration=configuration,
            reviewer_note=str(metadata.get("revision_note") or "").strip() or None,
            status="running",
            initiated_by=str(metadata.get("reviewer") or "").strip() or None,
            actor_source=derive_revision_run_actor_source(metadata),
            actor_confidence="unverified",
            started_at=datetime.now(UTC),
            completed_at=None,
            error=None,
            trace_id=pipeline_run_id,
            latency_ms=None,
        )
        return self._start_revision_background_job(
            pipeline_run_id=pipeline_run_id,
            source_version_id=int(run["source_version_id"]),
            target_revision_version_id=target_revision_version_id,
            session_detail=source_detail["session"],
            root_session_id=root_session_id,
            version=int(target_version["version_number"]),
            selected_text=selected_text,
            revision_instruction=revision_instruction,
            model_overrides=model_overrides,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    def run_revision_job(
        self,
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
        run_started_at = datetime.now(UTC)
        source_text = str(session_detail.get("session_text") or "").strip()
        if not source_text:
            raise ValueError("Session text is empty")
        source_result_payload = (
            session_detail.get("result_payload")
            if isinstance(session_detail.get("result_payload"), dict)
            else {}
        )
        source_deterministic_extraction = (
            source_result_payload.get("deterministic_extraction")
            if isinstance(source_result_payload.get("deterministic_extraction"), dict)
            else None
        )
        source_structured_case = (
            source_result_payload.get("structured_case")
            if isinstance(source_result_payload.get("structured_case"), dict)
            else None
        )
        source_lab_timeline = (
            source_result_payload.get("lab_timeline")
            if isinstance(source_result_payload.get("lab_timeline"), list)
            else None
        )
        source_onset_context = (
            source_result_payload.get("onset_context")
            if isinstance(source_result_payload.get("onset_context"), dict)
            else None
        )
        source_matched_drugs = (
            source_result_payload.get("matched_drugs")
            if isinstance(source_result_payload.get("matched_drugs"), list)
            else None
        )
        source_rucam_assessments = (
            source_result_payload.get("rucam_assessments")
            if isinstance(source_result_payload.get("rucam_assessments"), list)
            else None
        )
        selected_focus_text = str(selected_text or "").strip() or None
        focus_instruction = str(revision_instruction or "").strip() or None
        instruction_profile: ReviewerInstructionProfile | None = None
        instruction_trace: ReviewerInstructionTrace | None = None
        clinical_service = build_clinical_session_service(self.jobs)
        effective_overrides = {
            key: value
            for key, value in (model_overrides or {}).items()
            if value is not None
        }
        runtime_overrides = self._build_revision_runtime_overrides(
            effective_overrides=effective_overrides
        )
        revision_mode = "instruction_guided" if focus_instruction else "default"
        run_configuration = self._build_revision_run_configuration(
            selected_text=selected_focus_text,
            revision_instruction=focus_instruction,
            model_overrides=effective_overrides,
            metadata=metadata,
        )
        actor_source = derive_revision_run_actor_source(metadata)
        progress_callback = partial(report_revision_progress, self.jobs, job_id)
        stop_check = partial(ensure_revision_not_cancelled, self.jobs, job_id)

        try:
            load_step_started_at = datetime.now(UTC)
            load_step = self._record_revision_step_start(
                pipeline_run_id=pipeline_run_id,
                step_name="load_source_version",
                input_summary={
                    "session_id": int(session_detail["session_id"]),
                    "source_version_id": int(source_version_id),
                    "target_revision_version_id": int(target_revision_version_id),
                },
                input_payload={
                    "session_id": int(session_detail["session_id"]),
                    "source_version_id": int(source_version_id),
                    "target_revision_version_id": int(target_revision_version_id),
                },
            )
            self._record_revision_step_success(
                pipeline_run_id=pipeline_run_id,
                step_name="load_source_version",
                attempt_number=int(load_step["attempt_number"]),
                started_at=load_step_started_at,
                output_summary={
                    "source_text_length": len(source_text),
                    "has_official_report_text": bool(
                        str(session_detail.get("official_report_text") or "").strip()
                    ),
                    "has_section_extraction": bool(session_detail.get("sections")),
                    "has_deterministic_extraction": bool(
                        source_deterministic_extraction
                    ),
                    "has_structured_case": bool(source_structured_case),
                    "has_lab_timeline": bool(source_lab_timeline),
                    "has_onset_context": bool(source_onset_context),
                    "has_source_matches": bool(source_matched_drugs),
                    "has_source_assessments": bool(source_rucam_assessments),
                },
                output_payload={
                    "root_session_id": int(root_session_id),
                    "source_version_id": int(source_version_id),
                    "target_revision_version_id": int(target_revision_version_id),
                    "selected_text_present": bool(selected_focus_text),
                    "revision_instruction_present": bool(focus_instruction),
                    "source_deterministic_extraction_present": bool(
                        source_deterministic_extraction
                    ),
                    "source_structured_case_present": bool(source_structured_case),
                    "source_lab_timeline_present": bool(source_lab_timeline),
                    "source_onset_context_present": bool(source_onset_context),
                    "source_matched_drugs_present": bool(source_matched_drugs),
                    "source_rucam_assessments_present": bool(source_rucam_assessments),
                },
            )
            analysis_step_started_at = datetime.now(UTC)
            analysis_step = self._record_revision_step_start(
                pipeline_run_id=pipeline_run_id,
                step_name="analyze_reviewer_instructions",
                input_summary={
                    "revision_instruction_present": bool(focus_instruction),
                    "selected_text_present": bool(selected_focus_text),
                },
                input_payload={
                    "selected_text": selected_focus_text,
                    "revision_instruction": focus_instruction,
                },
            )
            if focus_instruction:
                instruction_profile, instruction_trace = (
                    self.analyze_reviewer_instructions(
                        raw_instruction_text=focus_instruction,
                        selected_text=selected_focus_text,
                    )
                )
                self._record_revision_step_success(
                    pipeline_run_id=pipeline_run_id,
                    step_name="analyze_reviewer_instructions",
                    attempt_number=int(analysis_step["attempt_number"]),
                    started_at=analysis_step_started_at,
                    output_summary={
                        "instruction_summary_length": len(
                            instruction_profile.instruction_summary
                        ),
                        "target_section_count": len(
                            instruction_profile.target_sections
                        ),
                        "target_entity_count": len(instruction_profile.target_entities),
                        "prompt_injection_detected": bool(
                            instruction_trace.prompt_injection_detected
                        ),
                        "prompt_injection_flag_count": len(
                            instruction_trace.prompt_injection_flags
                        ),
                    },
                    output_payload={
                        "instruction_profile": instruction_profile.model_dump(),
                        "instruction_trace": instruction_trace.model_dump(),
                    },
                )
            else:
                self._record_revision_step_success(
                    pipeline_run_id=pipeline_run_id,
                    step_name="analyze_reviewer_instructions",
                    attempt_number=int(analysis_step["attempt_number"]),
                    started_at=analysis_step_started_at,
                    output_summary={
                        "skipped": True,
                        "reason": "No reviewer instruction provided.",
                    },
                    output_payload={"skipped": True},
                )
            revision_focus_context = self.build_revision_instruction_context(
                selected_text=selected_focus_text,
                instruction_profile=instruction_profile,
            )
            prepare_step_started_at = datetime.now(UTC)
            prepare_step = self._record_revision_step_start(
                pipeline_run_id=pipeline_run_id,
                step_name="prepare_runtime",
                input_summary={
                    "has_model_overrides": bool(effective_overrides),
                    "override_keys": sorted(str(key) for key in effective_overrides),
                },
                input_payload=run_configuration,
                model_name=str(
                    effective_overrides.get("clinical_model")
                    or session_detail.get("clinical_model")
                    or ""
                ).strip()
                or None,
            )
            with LLMRuntimeConfig.override_for_run(runtime_overrides):
                clinical_service.apply_persisted_runtime_configuration()
                self._record_revision_step_success(
                    pipeline_run_id=pipeline_run_id,
                    step_name="prepare_runtime",
                    attempt_number=int(prepare_step["attempt_number"]),
                    started_at=prepare_step_started_at,
                    output_summary={
                        "runtime_configuration_applied": True,
                        "runtime_override_active": bool(runtime_overrides),
                    },
                )
                revision_use_rag = bool(metadata.get("use_rag"))
                preprocess_started_at = datetime.now(UTC)
                preprocess_step = self._record_revision_step_start(
                    pipeline_run_id=pipeline_run_id,
                    step_name="preprocess_input",
                    input_summary={
                        "session_id": int(session_detail["session_id"]),
                        "source_text_length": len(source_text),
                        "selected_text_length": len(selected_focus_text or ""),
                        "use_rag": revision_use_rag,
                        "has_persisted_section_extraction": bool(
                            isinstance(session_detail.get("result_payload"), dict)
                            and isinstance(
                                session_detail["result_payload"].get(
                                    "section_extraction"
                                ),
                                dict,
                            )
                        ),
                    },
                    input_payload={
                        "source_text": source_text,
                        "selected_text": selected_focus_text,
                        "revision_instruction": focus_instruction,
                    },
                )
                try:
                    (
                        preprocessed_request,
                        section_extraction,
                        preprocess_source_mode,
                    ) = asyncio.run(
                        clinical_service.prepare_revision_source_request(
                            session_detail=session_detail,
                            use_rag=revision_use_rag,
                        )
                    )
                except Exception as exc:
                    self._record_revision_step_failure(
                        pipeline_run_id=pipeline_run_id,
                        step_name="preprocess_input",
                        attempt_number=int(preprocess_step["attempt_number"]),
                        started_at=preprocess_started_at,
                        exc=exc,
                    )
                    raise
                self._record_revision_step_success(
                    pipeline_run_id=pipeline_run_id,
                    step_name="preprocess_input",
                    attempt_number=int(preprocess_step["attempt_number"]),
                    started_at=preprocess_started_at,
                    output_summary={
                        "section_extraction_available": bool(section_extraction),
                        "source_mode": preprocess_source_mode,
                        "reparsed_source_text": (
                            preprocess_source_mode == "reparsed_source_text"
                        ),
                        "patient_name_present": bool(
                            str(getattr(preprocessed_request, "name", "") or "").strip()
                        ),
                    },
                    output_payload={
                        "source_mode": preprocess_source_mode,
                        "section_extraction_available": bool(section_extraction),
                    },
                )
                patient_payload = clinical_service.build_patient_payload(
                    preprocessed_request
                )
                generation_started_at = datetime.now(UTC)
                generation_step = self._record_revision_step_start(
                    pipeline_run_id=pipeline_run_id,
                    step_name="generate_revision",
                    input_summary={
                        "patient_name_present": bool(
                            str(patient_payload.get("name") or "").strip()
                        ),
                        "section_extraction_available": bool(section_extraction),
                        "revision_focus_context_present": bool(
                            str(revision_focus_context or "").strip()
                        ),
                    },
                    input_payload={
                        "patient_payload_keys": sorted(
                            str(key) for key in patient_payload.keys()
                        ),
                        "revision_focus_context": revision_focus_context,
                    },
                    model_name=str(
                        effective_overrides.get("clinical_model")
                        or session_detail.get("clinical_model")
                        or ""
                    ).strip()
                    or None,
                )
                try:
                    result_payload = asyncio.run(
                        clinical_service.process_revision_patient(
                            patient_payload,
                            section_extraction=section_extraction,
                            session_version=version,
                            original_session_id=root_session_id,
                            session_metadata={
                                **metadata,
                                "instruction_profile": (
                                    instruction_profile.model_dump()
                                    if instruction_profile is not None
                                    else None
                                ),
                                "instruction_trace": (
                                    instruction_trace.model_dump()
                                    if instruction_trace is not None
                                    else None
                                ),
                                "model_overrides": effective_overrides,
                                "pipeline_run_id": pipeline_run_id,
                                "source_version_id": int(source_version_id),
                                "target_revision_version_id": int(
                                    target_revision_version_id
                                ),
                            },
                            original_session_text=source_text,
                            revision_focus_context=revision_focus_context,
                            progress_callback=progress_callback,
                            stop_check=stop_check,
                        )
                    )
                except Exception as exc:
                    self._record_revision_step_failure(
                        pipeline_run_id=pipeline_run_id,
                        step_name="generate_revision",
                        attempt_number=int(generation_step["attempt_number"]),
                        started_at=generation_started_at,
                        exc=exc,
                    )
                    raise
                self._record_revision_step_success(
                    pipeline_run_id=pipeline_run_id,
                    step_name="generate_revision",
                    attempt_number=int(generation_step["attempt_number"]),
                    started_at=generation_started_at,
                    output_summary={
                        "result_payload_keys": sorted(
                            str(key) for key in result_payload.keys()
                        ),
                        "revision_present": isinstance(
                            result_payload.get("revision"), dict
                        ),
                        "pipeline_artifacts_present": isinstance(
                            result_payload.get("pipeline_artifacts"), dict
                        ),
                    },
                )
                entity_pipeline = get_revision_entity_pipeline(result_payload)
                for step_name in (
                    "resolve_revision_extraction",
                    "validate_anamnesis_drugs",
                    "extract_missing_anamnesis_drugs",
                    "revise_labs_timeline",
                    "reconcile_revision_candidates",
                    "merge_revision_snapshot",
                ):
                    step_started_at = datetime.now(UTC)
                    step = self._record_revision_step_start(
                        pipeline_run_id=pipeline_run_id,
                        step_name=step_name,
                        input_summary={"source": "revision_result_payload"},
                    )
                    step_payload = entity_pipeline.get(step_name, {})
                    self._record_revision_step_success(
                        pipeline_run_id=pipeline_run_id,
                        step_name=step_name,
                        attempt_number=int(step["attempt_number"]),
                        started_at=step_started_at,
                        output_summary=summarize_revision_entity_stage_payload(
                            step_name,
                            step_payload,
                        ),
                        output_payload=step_payload,
                    )
                derived_steps: tuple[tuple[str, dict[str, Any]], ...] = (
                    (
                        "resolve_livertox_matches",
                        {
                            "matched_drug_count": len(
                                result_payload.get("matched_drugs") or []
                            ),
                        },
                    ),
                    (
                        "rerun_dili_assessments",
                        {
                            "assessment_count": len(
                                result_payload.get("rucam_assessments") or []
                            ),
                        },
                    ),
                    (
                        "rebuild_final_report",
                        {
                            "report_present": bool(
                                str(result_payload.get("report") or "").strip()
                            ),
                        },
                    ),
                    (
                        "qa_validate_revision",
                        {
                            "manual_review_required": bool(
                                result_payload.get("manual_review_required")
                            ),
                            "blocking_issue_count": len(
                                result_payload.get("blocking_issues") or []
                            ),
                        },
                    ),
                    (
                        "persist_revision",
                        {
                            "session_id": result_payload.get("session_id"),
                            "artifact_sources_present": bool(
                                result_payload.get("pipeline_artifacts")
                            ),
                        },
                    ),
                    (
                        "finalize_revision_version",
                        {
                            "session_id": result_payload.get("session_id"),
                            "target_revision_version_id": int(
                                target_revision_version_id
                            ),
                        },
                    ),
                )
                for step_name, output_summary in derived_steps:
                    step_started_at = datetime.now(UTC)
                    step = self._record_revision_step_start(
                        pipeline_run_id=pipeline_run_id,
                        step_name=step_name,
                        input_summary={"source": "revision_result_payload"},
                    )
                    self._record_revision_step_success(
                        pipeline_run_id=pipeline_run_id,
                        step_name=step_name,
                        attempt_number=int(step["attempt_number"]),
                        started_at=step_started_at,
                        output_summary=output_summary,
                    )
                persisted_session_id = int(result_payload.get("session_id") or 0)
                if persisted_session_id <= 0:
                    raise ValueError("Revision result did not include a persisted session id")
                revision_payload = result_payload.get("revision")
                if not isinstance(revision_payload, dict):
                    revision_payload = {}
                    result_payload["revision"] = revision_payload
                if (
                    instruction_profile is not None
                    and not isinstance(revision_payload.get("instruction_profile"), dict)
                ):
                    revision_payload["instruction_profile"] = (
                        instruction_profile.model_dump()
                    )
                if (
                    instruction_trace is not None
                    and not isinstance(revision_payload.get("instruction_trace"), dict)
                ):
                    revision_payload["instruction_trace"] = instruction_trace.model_dump()
                revision_payload["livertox_revision_decisions"] = (
                    self.build_revision_livertox_decisions(
                        matched_drugs=result_payload.get("matched_drugs") or [],
                        source_matched_drugs=source_matched_drugs,
                        instruction_profile=instruction_profile,
                    )
                )
                revision_payload["revised_dili_assessments"] = (
                    self.build_revised_dili_assessments(
                        rucam_assessments=result_payload.get("rucam_assessments")
                        or [],
                        matched_drugs=result_payload.get("matched_drugs") or [],
                        source_rucam_assessments=source_rucam_assessments,
                        revision_version_id=int(target_revision_version_id),
                        source_version_id=int(source_version_id),
                        instruction_profile=instruction_profile,
                    )
                )
                final_report_payload = build_revision_final_report_payload(
                    result_payload=result_payload,
                    selected_text=selected_focus_text,
                    instruction_profile=instruction_profile,
                )
                revision_payload["final_report_rebuild"] = (
                    final_report_payload.model_dump()
                )
                qa_validation_payload = build_revision_qa_validation_payload(
                    result_payload=result_payload,
                    instruction_profile=instruction_profile,
                    final_report_payload=final_report_payload,
                )
                revision_payload["qa_validation"] = (
                    qa_validation_payload.model_dump()
                )
                version_status, llm_qa_status = derive_revision_qa_outcome(
                    result_payload
                )
                self.serializer.upsert_session_result_payload(
                    persisted_session_id,
                    result_payload,
                )
                self.serializer.persist_revision_artifacts(
                    pipeline_run_id=pipeline_run_id,
                    revision_version_id=int(target_revision_version_id),
                    result_payload=result_payload,
                )
                self.serializer.persist_revision_entities(
                    pipeline_run_id=pipeline_run_id,
                    revision_version_id=int(target_revision_version_id),
                    source_version_id=int(source_version_id),
                    result_payload=result_payload,
                )
                self.serializer.finalize_revision_version(
                    pipeline_run_id=pipeline_run_id,
                    persisted_session_id=persisted_session_id,
                    model_configuration=run_configuration,
                    version_status=version_status,
                    llm_qa_status=llm_qa_status,
                    clinical_review_status="not_reviewed",
                )
                self.serializer.create_or_update_revision_run(
                    pipeline_run_id=pipeline_run_id,
                    session_id=int(session_detail["session_id"]),
                    root_session_id=int(root_session_id),
                    source_version_id=int(source_version_id),
                    target_revision_version_id=int(target_revision_version_id),
                    revision_mode=revision_mode,
                    revision_kind="llm_assisted_revision",
                    configuration=run_configuration,
                    reviewer_note=str(metadata.get("revision_note") or "").strip()
                    or None,
                    status="completed",
                    initiated_by=str(metadata.get("reviewer") or "").strip() or None,
                    actor_source=actor_source,
                    actor_confidence="unverified",
                    completed_at=datetime.now(UTC),
                    error=None,
                    trace_id=pipeline_run_id,
                    latency_ms=int(
                        (datetime.now(UTC) - run_started_at).total_seconds() * 1000
                    ),
                )
            return result_payload
        except Exception as exc:
            self.serializer.fail_revision_run(
                pipeline_run_id=pipeline_run_id,
                error={"message": str(exc)[:500]},
            )
            raise
