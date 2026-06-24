from __future__ import annotations

import re
import uuid
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from common.constants import DOCUMENT_SUPPORTED_EXTENSIONS
from common.paths import VECTOR_DB_PATH
from common.utils.logger import logger
from common.utils.text_utils import unique_preserve_order
from configurations.startup import get_server_settings
from domain.inspection import (
    InspectionJobPhase,
    ReviewerInstructionProfile,
    ReviewerInstructionTrace,
)
from domain.patient_timeline import PatientTimeline
from repositories.serialization.data import DataSerializer
from repositories.serialization.document_serializer import DocumentSerializer
from repositories.vectors import LanceVectorDatabase
from services.retrieval.settings import build_effective_rag_settings
from services.clinical.timeline import PatientTimelineExtractor
from services.inspection.normalization import (
    extract_lab_marker as extract_lab_marker_value,
)
from services.clinical.revision.helpers import (
    build_revision_section_validation as build_revision_section_validation_value,
)
from services.clinical.revision.helpers import (
    extract_revision_drug_names as extract_revision_drug_names_value,
)
from services.inspection.normalization import (
    first_iso_date as first_iso_date_value,
)
from services.inspection.normalization import (
    normalize_text as normalize_text_value,
)
from services.inspection.timeline import (
    build_fallback_timeline as build_fallback_timeline_value,
)
from services.inspection.update_jobs import DataInspectionUpdateJobRunner
from services.inspection.timeline import (
    generate_session_timeline as generate_session_timeline_value,
)
from services.inspection.timeline import (
    get_session_timeline as get_session_timeline_value,
)
from services.inspection.timeline import (
    get_session_timeline_by_id as get_session_timeline_by_id_value,
)
from services.inspection.timeline import (
    list_session_timelines as list_session_timelines_value,
)
from services.inspection.update_config import InspectionUpdateConfigMixin
from services.inspection.revision_diff import InspectionRevisionDiffMixin
from services.inspection.revision_decisions import InspectionRevisionDecisionsMixin
from services.inspection.revision_runner import InspectionRevisionRunnerMixin
from services.inspection.revision_runner_support import (
    REVISION_STEP_SEQUENCE as REVISION_STEP_SEQUENCE_VALUE,
)
from services.runtime.jobs import JobManager
from services.text.normalization import normalize_drug_query_name

PhaseStep = tuple[InspectionJobPhase, int, int, str]
UpdateTarget = Literal["rxnav", "livertox", "rag"]

###############################################################################
class DataInspectionService(
    InspectionUpdateConfigMixin,
    InspectionRevisionDiffMixin,
    InspectionRevisionDecisionsMixin,
    InspectionRevisionRunnerMixin,
):
    RXNAV_JOB_TYPE = "rxnav_update"
    LIVERTOX_JOB_TYPE = "livertox_update"
    RAG_JOB_TYPE = "rag_update"
    REVISION_JOB_TYPE = "session_revision"
    RAG_MANIFEST_FILE_NAME = "rag_index_manifest.json"
    REVISION_STEP_SEQUENCE = REVISION_STEP_SEQUENCE_VALUE
    UPDATE_PHASES: dict[UpdateTarget, list[PhaseStep]] = {
        "rxnav": [
            ("configuration_accepted", 1, 7, "Configuration accepted"),
            ("update_started", 2, 7, "Update started"),
            ("source_data_loading", 3, 7, "Downloading source catalog data"),
            ("processing_extraction", 4, 7, "Loading aliases and synonyms"),
            ("persistence_indexing", 5, 7, "Persisting catalog updates"),
            ("finalization", 6, 7, "Finalizing RxNav update"),
            ("completed", 7, 7, "RxNav update completed"),
        ],
        "livertox": [
            ("configuration_accepted", 1, 7, "Configuration accepted"),
            ("update_started", 2, 7, "Update started"),
            ("source_data_loading", 3, 7, "Loading archive and source metadata"),
            ("processing_extraction", 4, 7, "Extracting and processing monographs"),
            ("persistence_indexing", 5, 7, "Persisting extracted LiverTox data"),
            ("finalization", 6, 7, "Finalizing LiverTox update"),
            ("completed", 7, 7, "LiverTox update completed"),
        ],
        "rag": [
            ("configuration_accepted", 1, 7, "Configuration accepted"),
            ("update_started", 2, 7, "Update started"),
            ("source_data_loading", 3, 7, "Loading RAG source documents"),
            ("processing_extraction", 4, 7, "Chunking and embedding documents"),
            ("persistence_indexing", 5, 7, "Persisting embeddings and index state"),
            ("finalization", 6, 7, "Finalizing vector store update"),
            ("completed", 7, 7, "RAG embeddings update completed"),
        ],
    }

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        serializer: DataSerializer | None = None,
        timeline_extractor: PatientTimelineExtractor | None = None,
        jobs: JobManager,
    ) -> None:
        self.serializer = serializer or DataSerializer()
        self.timeline_extractor = timeline_extractor or PatientTimelineExtractor()
        self.jobs = jobs
        self.timeline_generation_lock = Lock()
        self.timeline_generation_inflight: set[int] = set()
        self.timeline_generation_cooldown_until: dict[int, float] = {}
        self.update_job_runner = DataInspectionUpdateJobRunner(
            serializer=self.serializer,
            jobs=self.jobs,
            report_phase_by_target=self._report_phase_by_target_for_runner,
            report_job_progress=self._report_job_progress_for_runner,
            write_rag_manifest=self._write_rag_manifest_for_runner,
        )
        self.reconcile_process_local_revision_runs()

    # -------------------------------------------------------------------------
    def reconcile_process_local_revision_runs(self) -> None:
        try:
            running_runs = self.serializer.list_revision_runs_by_status("running")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Unable to reconcile process-local revision runs: error_type=%s",
                type(exc).__name__,
            )
            return
        for run in running_runs:
            pipeline_run_id = str(run.get("pipeline_run_id") or "").strip()
            if not pipeline_run_id:
                continue
            self.serializer.fail_revision_run(
                pipeline_run_id=pipeline_run_id,
                error={
                    "code": "revision_job_process_lost",
                    "message": (
                        "Revision job state is process-local and was lost during "
                        "backend restart. Retry the draft revision if needed."
                    ),
                },
            )

    # -------------------------------------------------------------------------
    def list_sessions(
        self,
        *,
        search: str | None,
        status_filter: str | None,
        date_mode: str | None,
        filter_date: date | None,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        items, total = self.serializer.list_sessions(
            search=search,
            status_filter=status_filter,
            date_mode=date_mode,
            filter_date=filter_date,
            offset=offset,
            limit=limit,
        )
        return {
            "items": items,
            "total": total,
            "offset": max(int(offset), 0),
            "limit": max(int(limit), 1),
        }

    # -------------------------------------------------------------------------
    def get_session_detail(self, session_id: int) -> dict[str, Any] | None:
        return self.serializer.get_session_detail(session_id)

    # -------------------------------------------------------------------------
    def list_session_versions(self, session_id: int) -> list[dict[str, Any]]:
        return self.serializer.list_session_versions(session_id)

    # -------------------------------------------------------------------------
    def get_session_version_detail(
        self,
        session_id: int,
        *,
        version_id: int,
    ) -> dict[str, Any] | None:
        return self.serializer.get_session_version_detail(
            session_id,
            version_id=version_id,
        )

    # -------------------------------------------------------------------------
    def list_manual_report_edits(self, session_id: int) -> list[dict[str, Any]]:
        return self.serializer.list_manual_report_edits(session_id)

    # -------------------------------------------------------------------------
    def compare_session_versions(
        self,
        session_id: int,
        *,
        left_version_id: int,
        right_version_id: int,
    ) -> dict[str, Any] | None:
        left_detail = self.get_session_version_detail(
            session_id,
            version_id=left_version_id,
        )
        right_detail = self.get_session_version_detail(
            session_id,
            version_id=right_version_id,
        )
        if left_detail is None or right_detail is None:
            return None

        left_version = left_detail.get("version") or {}
        right_version = right_detail.get("version") or {}
        if int(left_version.get("root_session_id") or 0) != int(
            right_version.get("root_session_id") or 0
        ):
            raise ValueError("Versions do not belong to the same session lineage.")

        left_entities = self._resolve_version_comparison_entities(
            version_id=left_version_id,
            detail=left_detail,
        )
        right_entities = self._resolve_version_comparison_entities(
            version_id=right_version_id,
            detail=right_detail,
        )
        entity_diff = self._build_version_entity_diff(
            left_entities=left_entities,
            right_entities=right_entities,
        )
        return {
            "left_version": left_version,
            "right_version": right_version,
            **entity_diff,
            "report_text_diff": self._build_report_text_diff(
                left_text=self._extract_version_report_text(left_detail),
                right_text=self._extract_version_report_text(right_detail),
            ),
            "qa_summary": self._build_revision_qa_summary(
                left_detail=left_detail,
                right_detail=right_detail,
            ),
        }

    # -------------------------------------------------------------------------
    def update_session(
        self,
        session_id: int,
        *,
        report_text: str | None = None,
        edited_fields: list[str] | None = None,
        reviewer_note: str | None = None,
        edited_by: str | None = None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        resolved_report_text = str(report_text or "").strip() or None
        if resolved_report_text is not None:
            updated = self.serializer.update_current_report_text_with_manual_audit(
                session_id,
                report_text=resolved_report_text,
                edited_fields=edited_fields,
                reviewer_note=reviewer_note,
                edited_by=edited_by,
                metadata=metadata,
            )
            return updated["session"] if isinstance(updated, dict) else None
        return self.serializer.update_session_metadata(
            session_id,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    def manual_edit_report(
        self,
        session_id: int,
        *,
        report_text: str,
        edited_fields: list[str] | None,
        reviewer_note: str | None,
        edited_by: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return self.serializer.update_current_report_text_with_manual_audit(
            session_id,
            report_text=report_text,
            edited_fields=edited_fields,
            reviewer_note=reviewer_note,
            edited_by=edited_by,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    def build_revision_audit(
        self,
        *,
        source_detail: dict[str, Any],
        result_payload: dict[str, Any],
        selected_text: str | None,
        revision_instruction: str | None,
        effective_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        source_payload_value = source_detail.get("result_payload")
        source_payload: dict[str, Any]
        if isinstance(source_payload_value, dict):
            source_payload = source_payload_value
        else:
            source_payload = {}
        original_detected = self.extract_revision_drug_names(source_payload)
        revised_detected = self.extract_revision_drug_names(result_payload)
        original_keys = {
            normalize_drug_query_name(name) for name in original_detected if name
        }
        revised_keys = {
            normalize_drug_query_name(name) for name in revised_detected if name
        }
        new_drug_keys = sorted(key for key in revised_keys - original_keys if key)
        removed_drug_keys = sorted(key for key in original_keys - revised_keys if key)
        section_extraction = result_payload.get("section_extraction")
        source_sections_value = source_detail.get("sections")
        source_sections: dict[str, Any]
        if isinstance(source_sections_value, dict):
            source_sections = source_sections_value
        else:
            source_sections = {}
        extracted_sections: dict[str, Any]
        if isinstance(section_extraction, dict):
            extracted_sections = section_extraction
        else:
            extracted_sections = {}
        section_validation = self.build_revision_section_validation(
            source_sections=source_sections,
            extracted_sections=extracted_sections,
            selected_text=selected_text,
        )
        parser_cross_validation = {
            "rerun_completed": True,
            "source_scope": "selected_text" if selected_text else "full_session",
            "selected_text_length": len(selected_text or ""),
            "section_extraction_available": isinstance(section_extraction, dict),
            "sections": section_validation["sections"],
            "missing_sections_after_revision": section_validation[
                "missing_sections_after_revision"
            ],
            "changed_sections_after_revision": section_validation[
                "changed_sections_after_revision"
            ],
        }
        matched_drugs = result_payload.get("matched_drugs")
        rucam_assessments = result_payload.get("rucam_assessments")
        return {
            "source_session_id": source_detail.get("session_id"),
            "source_version": source_detail.get("version"),
            "focused_selection": bool(selected_text),
            "revision_instruction": revision_instruction,
            "model_overrides": effective_overrides,
            "parser_cross_validation": parser_cross_validation,
            "original_detected_drugs": original_detected,
            "revised_detected_drugs": revised_detected,
            "newly_identified_drugs": new_drug_keys,
            "previously_identified_drugs_missing_after_revision": removed_drug_keys,
            "drug_analysis_rerun": isinstance(rucam_assessments, list),
            "livertox_retrieval_rerun": isinstance(matched_drugs, list),
            "conclusion_action": (
                "generated_new_conclusion_for_new_drugs"
                if new_drug_keys
                else "improved_existing_conclusion"
            ),
        }

    # -------------------------------------------------------------------------
    def build_revision_section_validation(
        self,
        *,
        source_sections: dict[str, Any],
        extracted_sections: dict[str, Any],
        selected_text: str | None,
    ) -> dict[str, Any]:
        return build_revision_section_validation_value(
            source_sections=source_sections,
            extracted_sections=extracted_sections,
            selected_text=selected_text,
        )

    # -------------------------------------------------------------------------
    def extract_revision_drug_names(self, payload: dict[str, Any]) -> list[str]:
        return extract_revision_drug_names_value(payload)

    # -------------------------------------------------------------------------
    def get_revision_run(self, pipeline_run_id: str) -> dict[str, Any] | None:
        return self.serializer.get_revision_run(pipeline_run_id)

    # -------------------------------------------------------------------------
    def list_revision_steps(self, pipeline_run_id: str) -> list[dict[str, Any]]:
        return self.serializer.list_revision_steps(pipeline_run_id)

    # -------------------------------------------------------------------------
    def list_revision_artifacts(
        self,
        *,
        revision_version_id: int,
    ) -> list[dict[str, Any]]:
        return self.serializer.list_revision_artifacts_for_version(
            revision_version_id=revision_version_id
        )

    # -------------------------------------------------------------------------
    def list_revision_entities(
        self,
        *,
        revision_version_id: int,
    ) -> list[dict[str, Any]]:
        return self.serializer.list_revision_entities_for_version(
            revision_version_id=revision_version_id
        )

    # -------------------------------------------------------------------------
    def list_revision_reviews(
        self,
        *,
        revision_version_id: int,
    ) -> list[dict[str, Any]]:
        return self.serializer.list_revision_reviews_for_version(
            revision_version_id=revision_version_id
        )

    # -------------------------------------------------------------------------
    def update_revision_clinical_review(
        self,
        session_id: int,
        *,
        version_id: int,
        clinical_review_status: str,
        reviewer_note: str | None,
        reviewed_by: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        detail = self.get_session_version_detail(session_id, version_id=version_id)
        if detail is None:
            return None
        review_action = self.serializer.record_revision_review_action(
            revision_version_id=version_id,
            clinical_review_status=clinical_review_status,
            reviewer_note=reviewer_note,
            reviewed_by=reviewed_by,
            metadata=metadata or {},
        )
        if review_action is None:
            return None
        refreshed = self.get_session_version_detail(session_id, version_id=version_id)
        if refreshed is None:
            return None
        return {
            "version": refreshed["version"],
            "review_action": review_action,
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_revision_run_configuration(
        *,
        selected_text: str | None,
        revision_instruction: str | None,
        model_overrides: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        selected_focus_text = str(selected_text or "").strip() or None
        focus_instruction = str(revision_instruction or "").strip() or None
        effective_overrides = {
            key: value
            for key, value in (model_overrides or {}).items()
            if value is not None
        }
        return {
            "selected_text": selected_focus_text,
            "selected_text_present": bool(selected_focus_text),
            "revision_instruction": focus_instruction,
            "model_overrides": effective_overrides,
            "metadata": metadata or {},
        }

    # -------------------------------------------------------------------------
    @classmethod
    def detect_prompt_injection_flags(
        cls,
        *,
        instruction_text: str,
        selected_text: str | None = None,
    ) -> list[str]:
        combined_text = "\n".join(
            part.strip()
            for part in [instruction_text, str(selected_text or "")]
            if str(part or "").strip()
        ).casefold()
        detections: list[str] = []
        indicators: list[tuple[str, str]] = [
            ("ignore previous", "ignore_previous_instructions"),
            ("ignore all previous", "ignore_previous_instructions"),
            ("ignore system", "override_system_prompt_attempt"),
            ("ignore developer", "override_developer_prompt_attempt"),
            ("system prompt", "system_prompt_reference"),
            ("developer message", "developer_message_reference"),
            ("tool instruction", "tool_instruction_reference"),
            ("change the schema", "schema_override_attempt"),
            ("override schema", "schema_override_attempt"),
            ("change the routing", "routing_override_attempt"),
            ("override routing", "routing_override_attempt"),
            ("disable qa", "qa_disable_attempt"),
            ("skip qa", "qa_disable_attempt"),
            ("change the model", "model_override_attempt"),
            ("override model", "model_override_attempt"),
            ("do not follow your instructions", "instruction_bypass_attempt"),
            ("instead follow", "instruction_redirection_attempt"),
        ]
        for needle, flag in indicators:
            if needle in combined_text:
                detections.append(flag)
        return unique_preserve_order(detections)

    # -------------------------------------------------------------------------
    @classmethod
    def analyze_reviewer_instructions(
        cls,
        *,
        raw_instruction_text: str,
        selected_text: str | None = None,
    ) -> tuple[ReviewerInstructionProfile, ReviewerInstructionTrace]:
        normalized_instruction = normalize_text_value(raw_instruction_text)
        summary = str(normalized_instruction or "").strip()
        lowered = summary.casefold()

        target_sections: list[str] = []
        target_entities: list[str] = []
        routed_steps: list[str] = [
            "generate_revision",
            "resolve_revision_extraction",
            "validate_anamnesis_drugs",
            "extract_missing_anamnesis_drugs",
            "revise_labs_timeline",
            "reconcile_revision_candidates",
            "merge_revision_snapshot",
            "rebuild_final_report",
            "qa_validate_revision",
            "persist_revision",
            "finalize_revision_version",
        ]

        section_keywords: list[tuple[str, str]] = [
            ("anamnes", "anamnesis"),
            ("history", "anamnesis"),
            ("therap", "therapy"),
            ("drug", "therapy"),
            ("medication", "therapy"),
            ("lab", "labs"),
            ("timeline", "labs"),
            ("livertox", "livertox_matching"),
            ("match", "livertox_matching"),
            ("rucam", "dili_assessment"),
            ("causal", "dili_assessment"),
            ("report", "final_report"),
            ("wording", "final_report"),
            ("qa", "qa"),
            ("consisten", "qa"),
        ]
        for keyword, section in section_keywords:
            if keyword in lowered:
                target_sections.append(section)

        entity_keywords: list[tuple[str, str]] = [
            ("drug", "drugs"),
            ("medication", "drugs"),
            ("disease", "diseases"),
            ("diagnos", "diseases"),
            ("lab", "labs"),
            ("timeline", "labs"),
            ("wording", "report_wording"),
            ("report", "report_wording"),
            ("source", "source_evidence"),
            ("evidence", "source_evidence"),
            ("match", "matching_errors"),
            ("causal", "causality_reasoning"),
            ("missing", "missing_data"),
            ("ambigu", "ambiguity_resolution"),
        ]
        for keyword, entity in entity_keywords:
            if keyword in lowered:
                target_entities.append(entity)

        if not target_sections:
            target_sections.append("unknown")
        if not target_entities:
            target_entities.append("other")

        if any(
            section in target_sections for section in {"anamnesis", "therapy", "labs"}
        ):
            routed_steps.append("preprocess_input")
        if "qa" in target_sections or "source_evidence" in target_entities:
            routed_steps.append("qa_validate_revision")

        mentioned_dates = unique_preserve_order(
            [match.group(0) for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", summary)]
        )
        mentioned_lab_values = unique_preserve_order(
            [
                match.group(0)
                for match in re.finditer(
                    r"\b(?:ALT|AST|ALP|bilirubin|bilirubina)\b[^.;,\n]{0,20}\d+(?:\.\d+)?",
                    summary,
                    re.IGNORECASE,
                )
            ]
        )
        extra_data = unique_preserve_order(
            [selected_text.strip()] if str(selected_text or "").strip() else []
        )
        ambiguities = (
            ["Reviewer instruction contains ambiguity markers."]
            if any(
                token in lowered for token in ("maybe", "unclear", "check", "verify")
            )
            else []
        )
        constraints = (
            ["Limit changes to the explicitly targeted scope."]
            if any(
                token in lowered for token in ("only", "do not", "don't", "must not")
            )
            else []
        )
        safety_or_quality_concerns = (
            ["Reviewer requested evidence or consistency validation."]
            if any(
                token in lowered for token in ("evidence", "source", "consistent", "qa")
            )
            else []
        )
        prompt_injection_flags = cls.detect_prompt_injection_flags(
            instruction_text=summary,
            selected_text=selected_text,
        )
        if prompt_injection_flags:
            safety_or_quality_concerns = unique_preserve_order(
                safety_or_quality_concerns
                + [
                    "Potential prompt-injection or instruction-redirection content detected in untrusted revision inputs."
                ]
            )

        profile = ReviewerInstructionProfile(
            user_intent="revision_request",
            main_goal=summary[:200] or None,
            instruction_summary=summary,
            target_sections=unique_preserve_order(target_sections),  # type: ignore[arg-type]
            target_entities=unique_preserve_order(target_entities),  # type: ignore[arg-type]
            mentioned_drugs=[],
            mentioned_diseases=[],
            mentioned_lab_values=mentioned_lab_values,
            mentioned_dates=mentioned_dates,
            extra_data=extra_data,
            ambiguities=ambiguities,
            constraints=constraints,
            reviewer_assumptions=[],
            safety_or_quality_concerns=safety_or_quality_concerns,
            prompt_injection_flags=prompt_injection_flags,
            pipeline_routing_decision={
                "generate_revision": unique_preserve_order(target_sections),
                "resolve_revision_extraction": ["therapy", "anamnesis"],
                "validate_anamnesis_drugs": ["anamnesis"],
                "extract_missing_anamnesis_drugs": ["anamnesis"],
                "revise_labs_timeline": ["labs"],
                "reconcile_revision_candidates": ["therapy", "anamnesis"],
                "merge_revision_snapshot": ["therapy", "anamnesis", "labs"],
                "rebuild_final_report": ["final_report"],
                "qa_validate_revision": ["qa"],
                "persist_revision": ["artifacts"],
                "finalize_revision_version": ["status_transition"],
            },
        )
        trace = ReviewerInstructionTrace(
            instruction_id=uuid.uuid4().hex,
            raw_instruction_text=summary,
            normalized_instruction_summary=summary,
            routed_pipeline_steps=unique_preserve_order(routed_steps),
            affected_entities=unique_preserve_order(target_entities),
            applied=True,
            ignored=False,
            prompt_injection_detected=bool(prompt_injection_flags),
            prompt_injection_flags=prompt_injection_flags,
            evidence_addressed=extra_data,
            qa_validation_result="pending",
        )
        return profile, trace

    # -------------------------------------------------------------------------
    @staticmethod
    def build_revision_instruction_context(
        *,
        selected_text: str | None,
        instruction_profile: ReviewerInstructionProfile | None,
    ) -> str | None:
        chunks: list[str] = []
        if str(selected_text or "").strip():
            chunks.append(
                f"Reviewer-selected source excerpt:\n{str(selected_text).strip()}"
            )
        if instruction_profile is not None:
            chunks.append(
                "Reviewer instruction summary:\n"
                f"{instruction_profile.instruction_summary}"
            )
            if instruction_profile.target_sections:
                chunks.append(
                    "Target sections:\n"
                    + ", ".join(instruction_profile.target_sections)
                )
            if instruction_profile.target_entities:
                chunks.append(
                    "Target entities:\n"
                    + ", ".join(instruction_profile.target_entities)
                )
            if instruction_profile.constraints:
                chunks.append(
                    "Constraints:\n" + "; ".join(instruction_profile.constraints)
                )
        context = "\n\n".join(chunk for chunk in chunks if chunk.strip())
        return context or None

    # -------------------------------------------------------------------------
    def delete_session(self, session_id: int) -> bool:
        return self.serializer.delete_session(session_id)

    # -------------------------------------------------------------------------
    def get_session_timeline(self, session_id: int) -> PatientTimeline | None:
        return get_session_timeline_value(self, session_id)

    # -------------------------------------------------------------------------
    def get_session_timeline_by_id(
        self, session_id: int, timeline_id: int
    ) -> PatientTimeline | None:
        return get_session_timeline_by_id_value(self, session_id, timeline_id)

    # -------------------------------------------------------------------------
    def list_session_timelines(self, session_id: int) -> list[dict[str, Any]]:
        return list_session_timelines_value(self, session_id)

    # -------------------------------------------------------------------------
    def generate_session_timeline(
        self,
        session_id: int,
        *,
        force_regenerate: bool = False,
    ) -> PatientTimeline | None:
        return generate_session_timeline_value(
            self,
            session_id,
            force_regenerate=force_regenerate,
        )

    # -------------------------------------------------------------------------
    def build_fallback_timeline(
        self,
        *,
        session_id: int,
        source: dict[str, Any],
    ) -> PatientTimeline:
        return build_fallback_timeline_value(
            self,
            session_id=session_id,
            source=source,
        )

    # -------------------------------------------------------------------------
    def normalize_text(self, value: Any) -> str | None:
        return normalize_text_value(value)

    # -------------------------------------------------------------------------
    def first_iso_date(self, value: Any) -> str | None:
        return first_iso_date_value(value)

    # -------------------------------------------------------------------------
    def extract_lab_marker(self, text: str) -> str | None:
        return extract_lab_marker_value(text)

    # -------------------------------------------------------------------------
    def list_rxnav_catalog(
        self,
        *,
        search: str | None,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        items, total = self.serializer.list_rxnav_catalog(
            search=search,
            offset=offset,
            limit=limit,
        )
        return {
            "items": items,
            "total": total,
            "offset": max(int(offset), 0),
            "limit": max(int(limit), 1),
        }

    # -------------------------------------------------------------------------
    def get_rxnav_alias_groups(self, drug_id: int) -> dict[str, Any] | None:
        return self.serializer.get_rxnav_alias_groups(drug_id)

    # -------------------------------------------------------------------------
    def update_rxnav_drug_name(
        self,
        drug_id: int,
        *,
        drug_name: str,
    ) -> dict[str, Any] | None:
        return self.serializer.update_rxnav_drug_name(
            drug_id,
            drug_name=drug_name,
        )

    # -------------------------------------------------------------------------
    def list_livertox_catalog(
        self,
        *,
        search: str | None,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        items, total = self.serializer.list_livertox_catalog(
            search=search,
            offset=offset,
            limit=limit,
        )
        return {
            "items": items,
            "total": total,
            "offset": max(int(offset), 0),
            "limit": max(int(limit), 1),
        }

    # -------------------------------------------------------------------------
    def get_livertox_excerpt(self, drug_id: int) -> dict[str, Any] | None:
        return self.serializer.get_livertox_excerpt(drug_id)

    # -------------------------------------------------------------------------
    def delete_drug(self, drug_id: int) -> bool:
        return self.serializer.delete_drug_with_cleanup(drug_id)

    # -------------------------------------------------------------------------
    def list_rag_documents(
        self,
        *,
        search: str | None,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        serializer = DocumentSerializer(self.get_effective_rag_documents_path())
        vector_model_by_file: dict[str, str] = {}
        try:
            rag_settings = build_effective_rag_settings()
            vector_db = LanceVectorDatabase(
                database_path=VECTOR_DB_PATH,
                collection_name=rag_settings.vector_collection_name,
                metric=rag_settings.vector_index_metric,
                index_type=rag_settings.vector_index_type,
                stream_batch_size=rag_settings.vector_stream_batch_size,
            )
            if vector_db.has_collection():
                for row in vector_db.load_embeddings():
                    file_name = str(row.get("file_name") or "")
                    provider = str(row.get("vector_model_provider") or "").strip()
                    model_name = str(row.get("vector_model_name") or "").strip()
                    if not file_name:
                        continue
                    if provider and model_name:
                        vector_model_by_file[file_name] = f"{provider}:{model_name}"
                    elif model_name:
                        vector_model_by_file[file_name] = model_name
        except Exception:
            vector_model_by_file = {}
        items: list[dict[str, Any]] = []
        supported_ext = {entry.lower() for entry in DOCUMENT_SUPPORTED_EXTENSIONS}
        for path in serializer.collect_document_paths():
            file_path = Path(path)
            suffix = file_path.suffix.lower()
            try:
                stat = file_path.stat()
                modified = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
                size = int(stat.st_size)
            except OSError:
                modified = datetime.fromtimestamp(0, UTC).isoformat()
                size = 0
            items.append(
                {
                    "path": str(file_path),
                    "file_name": file_path.name,
                    "extension": suffix,
                    "file_size": size,
                    "last_modified": modified,
                    "supported_for_ingestion": suffix in supported_ext,
                    "vector_model": vector_model_by_file.get(file_path.name),
                }
            )
        items.sort(key=lambda item: str(item["path"]).casefold())
        normalized_search = (search or "").strip().casefold()
        if normalized_search:
            items = [
                item
                for item in items
                if normalized_search in str(item["file_name"]).casefold()
                or normalized_search in str(item["path"]).casefold()
                or normalized_search in str(item["extension"]).casefold()
            ]

        total = len(items)
        bounded_offset = max(int(offset), 0)
        bounded_limit = max(int(limit), 1)
        paged = items[bounded_offset : bounded_offset + bounded_limit]
        return {
            "items": paged,
            "total": total,
            "offset": bounded_offset,
            "limit": bounded_limit,
        }

    # -------------------------------------------------------------------------
    def get_rag_vector_store_summary(self) -> dict[str, Any]:
        documents_path = self.get_effective_rag_documents_path()
        rag_settings = build_effective_rag_settings()
        collection_name = str(rag_settings.vector_collection_name)
        vector_db = LanceVectorDatabase(
            database_path=VECTOR_DB_PATH,
            collection_name=collection_name,
            metric=rag_settings.vector_index_metric,
            index_type=rag_settings.vector_index_type,
            stream_batch_size=rag_settings.vector_stream_batch_size,
        )
        exists = vector_db.has_collection()
        embedding_count = 0
        distinct_document_count = 0
        embedding_dimension: int | None = None
        if exists:
            try:
                vector_db.get_table()
                embedding_count = vector_db.count_embeddings()
                distinct_document_count = vector_db.count_distinct_documents()
                embedding_dimension = vector_db.read_embedding_dimension()
                if embedding_count > 0:
                    vector_db.ensure_vector_index()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Unable to load LanceDB inspection summary: %s", exc)
        return {
            "source_documents_path": documents_path,
            "vector_db_path": str(VECTOR_DB_PATH),
            "collection_name": collection_name,
            "collection_exists": exists,
            "embedding_count": embedding_count,
            "distinct_document_count": distinct_document_count,
            "embedding_dimension": embedding_dimension,
            "index_ready": bool(vector_db.index_ready) if exists else False,
            "configured_metric": rag_settings.vector_index_metric,
            "configured_index_type": rag_settings.vector_index_type,
        }

    # -------------------------------------------------------------------------
    def patch_job_result(self, *, job_id: str, patch: dict[str, Any]) -> None:
        self.jobs.update_result(job_id, patch)

    # -------------------------------------------------------------------------
    def report_job_progress(
        self, *, job_id: str, progress: float, message: str
    ) -> None:
        bounded_progress = min(100.0, max(0.0, float(progress)))
        self.jobs.update_progress(job_id, bounded_progress)
        self.patch_job_result(job_id=job_id, patch={"progress_message": message})

    # -------------------------------------------------------------------------
    def report_phase(
        self,
        *,
        job_id: str,
        phase: InspectionJobPhase,
        step_index: int,
        step_count: int,
        progress: float,
        message: str,
    ) -> None:
        self.jobs.update_progress(job_id, min(100.0, max(0.0, float(progress))))
        self.patch_job_result(
            job_id=job_id,
            patch={
                "phase": phase,
                "step_index": step_index,
                "step_count": step_count,
                "progress_message": message,
            },
        )

    # -------------------------------------------------------------------------
    def report_phase_by_target(
        self,
        *,
        job_id: str,
        target: UpdateTarget,
        phase: InspectionJobPhase,
        progress: float,
        fallback_message: str,
    ) -> None:
        step = next(
            (entry for entry in self.UPDATE_PHASES[target] if entry[0] == phase),
            None,
        )
        if step is None:
            self.report_job_progress(
                job_id=job_id, progress=progress, message=fallback_message
            )
            return
        self.report_phase(
            job_id=job_id,
            phase=step[0],
            step_index=step[1],
            step_count=step[2],
            progress=progress,
            message=step[3] or fallback_message,
        )

    # -------------------------------------------------------------------------
    def _report_phase_by_target_for_runner(
        self, job_id: str, target: str, progress: int, message: str
    ) -> None:
        phase = "update_started"
        for entry in self.UPDATE_PHASES[target]:  # type: ignore[index]
            if entry[3] == message:
                phase = entry[0]
                break
        self.report_phase_by_target(
            job_id=job_id,
            target=target,  # type: ignore[arg-type]
            phase=phase,  # type: ignore[arg-type]
            progress=float(progress),
            fallback_message=message,
        )

    # -------------------------------------------------------------------------
    def _report_job_progress_for_runner(
        self,
        job_id: str,
        progress: float,
        message: str,
        extra: Any | None = None,
    ) -> None:
        _ = extra
        self.report_job_progress(job_id=job_id, progress=progress, message=message)

    # -------------------------------------------------------------------------
    def _write_rag_manifest_for_runner(
        self, report: dict[str, Any], documents_path: str
    ) -> Path:
        self.write_rag_manifest(documents_path=documents_path, summary=report)
        return self.rag_manifest_path()

    # -------------------------------------------------------------------------
    def run_rxnav_update_job(
        self, job_id: str, overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.update_job_runner.run_rxnav_update_job(job_id, overrides)

    # -------------------------------------------------------------------------
    def run_livertox_update_job(
        self, job_id: str, overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.update_job_runner.run_livertox_update_job(job_id, overrides)

    # -------------------------------------------------------------------------
    def run_rag_update_job(
        self, job_id: str, overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.update_job_runner.run_rag_update_job(job_id, overrides)

    # -------------------------------------------------------------------------
    def start_update_job(
        self, job_type: str, overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        scope_key = f"catalog:{job_type}"
        if self.jobs.is_job_running(job_type, scope_key=scope_key):
            raise ValueError(f"Job type '{job_type}' is already running")
        override_values = dict(overrides or {})
        if job_type == self.RXNAV_JOB_TYPE:
            runner = partial(self.run_rxnav_update_job, overrides=override_values)
        elif job_type == self.LIVERTOX_JOB_TYPE:
            runner = partial(self.run_livertox_update_job, overrides=override_values)
        elif job_type == self.RAG_JOB_TYPE:
            runner = partial(self.run_rag_update_job, overrides=override_values)
        else:
            raise ValueError(f"Unsupported job type: {job_type}")
        job_id = self.jobs.start_job(
            job_type=job_type,
            runner=runner,
            scope_key=scope_key,
        )
        status_payload = self.jobs.get_job_status(job_id)
        if status_payload is None:
            raise RuntimeError(f"Failed to initialize {job_type} job")
        status_payload["poll_interval"] = get_server_settings().jobs.polling_interval
        return status_payload

    # -------------------------------------------------------------------------
    def get_job_status(
        self, job_id: str, *, expected_type: str
    ) -> dict[str, Any] | None:
        payload = self.jobs.get_job_status(job_id)
        if payload is None:
            if expected_type == self.REVISION_JOB_TYPE:
                return {
                    "job_id": job_id,
                    "job_type": expected_type,
                    "scope_key": None,
                    "status": "failed",
                    "progress": 0.0,
                    "result": {
                        "recoverable": True,
                        "recovery_action": "reload_revision_run_and_retry",
                    },
                    "error": (
                        "Revision job state is process-local and is no longer "
                        "available. Reload the revision run and retry if it remains "
                        "draft."
                    ),
                    "created_at": None,
                    "completed_at": None,
                    "version": None,
                }
            return None
        job_type = str(payload.get("job_type") or "")
        if job_type != expected_type:
            logger.warning(
                "Job type mismatch for %s: expected %s, got %s",
                job_id,
                expected_type,
                job_type,
            )
            return None
        return payload

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str, *, expected_type: str) -> bool:
        payload = self.get_job_status(job_id, expected_type=expected_type)
        if payload is None:
            return False
        return self.jobs.cancel_job(job_id) is not None
