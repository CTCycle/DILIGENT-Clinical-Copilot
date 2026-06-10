from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from configurations.llm_configs import LLMRuntimeConfig
from domain.inspection import (
    ReviewerInstructionProfile,
    ReviewerInstructionTrace,
)
from services.session.factory import build_clinical_session_service


class InspectionRevisionRunnerMixin:
    REVISION_STEP_SEQUENCE: list[tuple[str, str]] = [
        ("load_source_version", "Loading selected source version"),
        ("analyze_reviewer_instructions", "Analyzing reviewer instructions"),
        ("prepare_runtime", "Preparing revision runtime"),
        ("preprocess_input", "Preprocessing source clinical text"),
        ("generate_revision", "Generating revised clinical session"),
        ("resolve_revision_extraction", "Resolving revision extraction bundle"),
        ("validate_anamnesis_drugs", "Validating revised anamnesis drugs"),
        (
            "extract_missing_anamnesis_drugs",
            "Extracting missing anamnesis drug candidates",
        ),
        ("revise_labs_timeline", "Revising structured laboratory timeline"),
        ("reconcile_revision_candidates", "Reconciling revision candidate selection"),
        ("merge_revision_snapshot", "Merging revision entity snapshot"),
        ("resolve_livertox_matches", "Resolving revision LiverTox matches"),
        ("rerun_dili_assessments", "Rebuilding revision DILI assessments"),
        ("rebuild_final_report", "Rebuilding revision final report"),
        ("qa_validate_revision", "Validating rebuilt revision output"),
        ("persist_revision", "Persisting revision artifacts"),
        ("finalize_revision_version", "Finalizing revision version state"),
    ]

    @staticmethod
    def _revision_run_actor_source(metadata: dict[str, Any]) -> str:
        return (
            "manual_entry"
            if str((metadata or {}).get("reviewer") or "").strip()
            else "unknown"
        )

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
                    self.REVISION_STEP_SEQUENCE, start=1
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
            step_count=len(self.REVISION_STEP_SEQUENCE),
            input_summary=input_summary,
            input_payload=input_payload,
            model_name=model_name,
        )

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

    @staticmethod
    def _derive_revision_qa_outcome(
        result_payload: dict[str, Any],
    ) -> tuple[str, str]:
        revision_payload = result_payload.get("revision")
        if isinstance(revision_payload, dict):
            qa_validation = revision_payload.get("qa_validation")
            if isinstance(qa_validation, dict):
                version_status = str(qa_validation.get("version_status") or "").strip()
                qa_status = str(qa_validation.get("status") or "").strip()
                if version_status in {
                    "llm_qa_passed",
                    "qa_failed",
                    "requires_human_review",
                } and qa_status in {
                    "passed",
                    "passed_with_warnings",
                    "failed",
                    "requires_human_review",
                }:
                    return version_status, qa_status
        blocking_issues = result_payload.get("blocking_issues")
        if isinstance(blocking_issues, list) and blocking_issues:
            return "qa_failed", "failed"
        if bool(result_payload.get("manual_review_required")):
            return "requires_human_review", "requires_human_review"
        pipeline_artifacts = result_payload.get("pipeline_artifacts")
        if isinstance(pipeline_artifacts, dict) and isinstance(
            pipeline_artifacts.get("faithfulness_audit"),
            dict,
        ):
            return "llm_qa_passed", "passed"
        return "requires_human_review", "not_run"

    @staticmethod
    def _get_revision_entity_pipeline(
        result_payload: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        revision_payload = result_payload.get("revision")
        if not isinstance(revision_payload, dict):
            return {}
        entity_pipeline = revision_payload.get("entity_pipeline")
        if not isinstance(entity_pipeline, dict):
            return {}
        return {
            str(step_name): payload
            for step_name, payload in entity_pipeline.items()
            if isinstance(step_name, str) and isinstance(payload, dict)
        }

    @staticmethod
    def _summarize_revision_entity_stage_payload(
        step_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if step_name == "validate_anamnesis_drugs":
            return {
                "status": payload.get("status"),
                "deterministic_detected_count": len(
                    payload.get("deterministic_detected_names") or []
                ),
                "revised_detected_count": len(
                    payload.get("revised_detected_names") or []
                ),
                "supplemental_detected_count": len(
                    payload.get("revised_only_names") or []
                ),
            }
        if step_name == "resolve_revision_extraction":
            return {
                "status": payload.get("status"),
                "therapy_source": payload.get("therapy_source"),
                "anamnesis_source": payload.get("anamnesis_source"),
                "disease_source": payload.get("disease_source"),
                "therapy_structured_count": len(
                    payload.get("therapy_structured_names") or []
                ),
                "anamnesis_structured_count": len(
                    payload.get("anamnesis_structured_names") or []
                ),
                "disease_deterministic_count": len(
                    payload.get("disease_deterministic_names") or []
                ),
            }
        if step_name == "extract_missing_anamnesis_drugs":
            return {
                "status": payload.get("status"),
                "supplemental_drug_count": len(
                    payload.get("supplemental_drug_names") or []
                ),
            }
        if step_name == "revise_labs_timeline":
            return {
                "status": payload.get("status"),
                "lab_entry_count": int(payload.get("lab_entry_count") or 0),
                "marker_count": len(payload.get("marker_names") or []),
            }
        if step_name == "reconcile_revision_candidates":
            return {
                "status": payload.get("status"),
                "analysis_drug_count": len(payload.get("analysis_drug_names") or []),
                "relevant_drug_count": len(payload.get("relevant_drug_names") or []),
                "unresolved_drug_count": len(
                    payload.get("unresolved_drug_names") or []
                ),
            }
        if step_name == "merge_revision_snapshot":
            return {
                "status": payload.get("status"),
                "therapy_drug_count": len(payload.get("therapy_drug_names") or []),
                "anamnesis_drug_count": len(payload.get("anamnesis_drug_names") or []),
                "analysis_drug_count": len(payload.get("analysis_drug_names") or []),
                "rucam_assessment_count": int(
                    payload.get("rucam_assessment_count") or 0
                ),
            }
        return {"status": payload.get("status")}

    def start_revision_job(
        self,
        session_id: int,
        *,
        selected_text: str | None,
        revision_instruction: str | None,
        model_overrides: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if self.jobs.is_job_running(self.REVISION_JOB_TYPE):
            raise ValueError("Session revision is already running")
        detail = self.get_session_detail(session_id)
        if detail is None:
            raise ValueError("Session not found")
        root_session_id = int(detail.get("original_session_id") or session_id)
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
            actor_source=self._revision_run_actor_source(metadata),
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

    def retry_revision_job(self, pipeline_run_id: str) -> dict[str, Any]:
        if self.jobs.is_job_running(self.REVISION_JOB_TYPE):
            raise ValueError("Session revision is already running")
        run = self.serializer.get_revision_run(pipeline_run_id)
        if run is None:
            raise ValueError("Revision pipeline run not found")
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
            actor_source=self._revision_run_actor_source(metadata),
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
            root_session_id=int(run["root_session_id"]),
            version=int(target_version["version_number"]),
            selected_text=selected_text,
            revision_instruction=revision_instruction,
            model_overrides=model_overrides,
            metadata=metadata,
        )

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
        actor_source = self._revision_run_actor_source(metadata)

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
                # The rest of the pipeline continues with patient_payload...
                result_payload = asyncio.run(
                    clinical_service.process_revision_patient(
                        patient_payload,
                        section_extraction=section_extraction,
                        revision_focus_context=revision_focus_context,
                    )
                )
            return result_payload
        except Exception:
            self.serializer.fail_revision_run(
                pipeline_run_id=pipeline_run_id,
            )
            raise
