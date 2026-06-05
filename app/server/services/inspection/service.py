from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from common.constants import DOCUMENT_SUPPORTED_EXTENSIONS
from common.paths import DOCS_PATH, VECTOR_DB_PATH
from common.utils.logger import logger
from configurations.llm_configs import LLMRuntimeConfig
from configurations.startup import get_server_settings
from domain.clinical.entities import ClinicalSessionRequest
from domain.inspection import InspectionJobPhase
from domain.patient_timeline import PatientTimeline
from repositories.serialization.data import (
    DataSerializer,
    DocumentSerializer,
)
from repositories.vectors import LanceVectorDatabase
from services.clinical.timeline import PatientTimelineExtractor
from services.retrieval.settings import build_effective_rag_settings
from services.inspection.normalization import (
    extract_lab_marker as extract_lab_marker_value,
)
from services.inspection.revision_helpers import (
    build_revision_section_validation as build_revision_section_validation_value,
)
from services.inspection.revision_helpers import (
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
from services.runtime.jobs import JobManager
from services.session.factory import build_clinical_session_service
from services.text.normalization import normalize_drug_query_name
from services.text.vocabulary import (
    deactivate_text_normalization_term_payload,
    invalidate_text_normalization_snapshot,
    list_text_normalization_term_payloads,
    upsert_text_normalization_term_payload,
)

PhaseStep = tuple[InspectionJobPhase, int, int, str]
UpdateTarget = Literal["rxnav", "livertox", "rag"]


class DataInspectionService:
    RXNAV_JOB_TYPE = "rxnav_update"
    LIVERTOX_JOB_TYPE = "livertox_update"
    RAG_JOB_TYPE = "rag_update"
    REVISION_JOB_TYPE = "session_revision"
    RAG_MANIFEST_FILE_NAME = "rag_index_manifest.json"
    REVISION_STEP_SEQUENCE: list[tuple[str, str]] = [
        ("prepare_runtime", "Preparing revision runtime"),
        ("preprocess_input", "Preprocessing source clinical text"),
        ("generate_revision", "Generating revised clinical session"),
        ("persist_revision", "Persisting revision artifacts"),
    ]
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

    # -------------------------------------------------------------------------
    def load_runtime_config(self) -> dict[str, Any]:
        return get_server_settings().model_dump()

    def rag_manifest_path(self) -> Path:
        return VECTOR_DB_PATH / self.RAG_MANIFEST_FILE_NAME

    def read_rag_manifest(self) -> dict[str, Any]:
        manifest_path = self.rag_manifest_path()
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def write_rag_manifest(
        self,
        *,
        documents_path: str,
        summary: dict[str, Any],
    ) -> None:
        manifest_path = self.rag_manifest_path()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents_path": documents_path,
            "documents": int(summary.get("documents", 0) or 0),
            "chunks": int(summary.get("chunks", 0) or 0),
            "supported_files": int(summary.get("supported_files", 0) or 0),
            "loaded_documents": int(summary.get("loaded_documents", 0) or 0),
            "built_at": datetime.now(UTC).isoformat(),
        }
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_effective_rag_documents_path(self) -> str:
        manifest = self.read_rag_manifest()
        manifest_path = str(manifest.get("documents_path") or "").strip()
        if manifest_path:
            return manifest_path
        config = self.load_runtime_config()
        rag_cfg = config.get("rag", {}) if isinstance(config, dict) else {}
        return str(rag_cfg.get("documents_path", DOCS_PATH))

    def list_reference_catalog_runtime_observations(
        self, category: str | None = None
    ) -> list[dict[str, Any]]:
        return list_text_normalization_term_payloads(category=category)

    def upsert_reference_catalog_runtime_observation(
        self,
        *,
        category: str,
        term: str,
        replacement: str | None,
        source: str,
        is_active: bool,
    ) -> dict[str, Any]:
        payload = upsert_text_normalization_term_payload(
            category=category,
            term=term,
            replacement=replacement,
            source=source,
            is_active=is_active,
        )
        invalidate_text_normalization_snapshot()
        return payload

    def deactivate_reference_catalog_runtime_observation(
        self, *, category: str, term: str
    ) -> bool:
        updated = deactivate_text_normalization_term_payload(
            category=category,
            term=term,
        )
        if updated:
            invalidate_text_normalization_snapshot()
        return updated

    # -------------------------------------------------------------------------
    def build_update_config_response(self, target: UpdateTarget) -> dict[str, Any]:
        config = self.load_runtime_config()
        settings = get_server_settings()
        if target == "rxnav":
            source = config.get("runtime", {})
            defaults = {
                "rxnav_request_timeout": float(
                    source.get(
                        "rxnav_request_timeout",
                        settings.runtime.rxnav_request_timeout,
                    )
                ),
                "rxnav_max_concurrency": int(
                    source.get(
                        "rxnav_max_concurrency",
                        settings.runtime.rxnav_max_concurrency,
                    )
                ),
            }
            allowed_fields = list(defaults.keys())
        elif target == "livertox":
            source = config.get("runtime", {})
            defaults = {
                "livertox_monograph_max_workers": int(
                    source.get(
                        "livertox_monograph_max_workers",
                        settings.runtime.livertox_monograph_max_workers,
                    )
                ),
                "livertox_archive": str(
                    source.get(
                        "livertox_archive",
                        settings.runtime.livertox_archive,
                    )
                ),
                "redownload": False,
            }
            allowed_fields = list(defaults.keys())
        else:
            rag_settings = build_effective_rag_settings()
            defaults = {}
            allowed_fields = []
            summary = {
                "chunk_size": int(rag_settings.chunk_size),
                "chunk_overlap": int(rag_settings.chunk_overlap),
                "embedding_batch_size": int(rag_settings.embedding_batch_size),
                "vector_stream_batch_size": int(rag_settings.vector_stream_batch_size),
                "embedding_max_workers": int(rag_settings.embedding_max_workers),
                "embedding_backend": rag_settings.embedding_backend,
                "ollama_embedding_model": rag_settings.ollama_embedding_model,
                "hf_embedding_model": rag_settings.hf_embedding_model,
                "cloud_provider": rag_settings.cloud_provider,
                "cloud_embedding_model": rag_settings.cloud_embedding_model,
                "use_cloud_embeddings": bool(rag_settings.use_cloud_embeddings),
                "reset_vector_collection": bool(rag_settings.reset_vector_collection),
            }
            return {
                "target": target,
                "defaults": defaults,
                "allowed_fields": allowed_fields,
                "summary": summary,
                "read_only": True,
            }

        return {
            "target": target,
            "defaults": defaults,
            "allowed_fields": allowed_fields,
            "summary": {},
            "read_only": False,
        }

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
    def update_session(
        self,
        session_id: int,
        *,
        session_text: str | None,
        report_text: str | None = None,
        edited_fields: list[str] | None = None,
        reviewer_note: str | None = None,
        edited_by: str | None = None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        resolved_report_text = str(report_text or "").strip() or None
        if resolved_report_text is None:
            legacy_report_text = str(session_text or "").strip()
            resolved_report_text = legacy_report_text or None
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
            key: value for key, value in (model_overrides or {}).items() if value is not None
        }
        return {
            "selected_text": selected_focus_text,
            "selected_text_present": bool(selected_focus_text),
            "revision_instruction": focus_instruction,
            "model_overrides": effective_overrides,
            "metadata": metadata or {},
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_revision_runtime_overrides(
        *,
        effective_overrides: dict[str, Any],
    ) -> dict[str, object]:
        runtime_overrides: dict[str, object] = {}
        if "clinical_model" in effective_overrides:
            runtime_overrides["clinical_model"] = effective_overrides["clinical_model"]
        if "text_extraction_model" in effective_overrides:
            runtime_overrides["text_extraction_model"] = effective_overrides[
                "text_extraction_model"
            ]
        if "use_cloud_services" in effective_overrides:
            runtime_overrides["use_cloud_models"] = effective_overrides[
                "use_cloud_services"
            ]
        if "provider" in effective_overrides:
            runtime_overrides["cloud_provider"] = effective_overrides["provider"]
        if "cloud_model" in effective_overrides:
            runtime_overrides["cloud_model"] = effective_overrides["cloud_model"]
        if "ollama_temperature" in effective_overrides:
            runtime_overrides["ollama_temperature"] = effective_overrides[
                "ollama_temperature"
            ]
        if "cloud_temperature" in effective_overrides:
            runtime_overrides["cloud_temperature"] = effective_overrides[
                "cloud_temperature"
            ]
        if "ollama_reasoning" in effective_overrides:
            runtime_overrides["ollama_reasoning"] = effective_overrides[
                "ollama_reasoning"
            ]
        return runtime_overrides

    # -------------------------------------------------------------------------
    @staticmethod
    def _revision_run_actor_source(metadata: dict[str, Any]) -> str:
        return (
            "manual_entry"
            if str((metadata or {}).get("reviewer") or "").strip()
            else "unknown"
        )

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
    @staticmethod
    def _derive_revision_qa_outcome(
        result_payload: dict[str, Any],
    ) -> tuple[str, str]:
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
        effective_overrides = run_configuration["model_overrides"]
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

    # -------------------------------------------------------------------------
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
            raise ValueError("Revision pipeline run is missing its target version shell")
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
        revision_mode = (
            "instruction_guided" if revision_instruction else "default"
        )
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
        selected_focus_text = str(selected_text or "").strip() or None
        focus_instruction = str(revision_instruction or "").strip() or None
        revision_focus_context = self.build_revision_focus_context(
            selected_text=selected_focus_text,
            revision_instruction=focus_instruction,
        )
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
                request = ClinicalSessionRequest(
                    name=session_detail.get("patient_name"),
                    visit_date=session_detail.get("visit_date"),
                    clinical_input=source_text,
                    use_rag=revision_use_rag,
                )
                preprocess_started_at = datetime.now(UTC)
                preprocess_step = self._record_revision_step_start(
                    pipeline_run_id=pipeline_run_id,
                    step_name="preprocess_input",
                    input_summary={
                        "session_id": int(session_detail["session_id"]),
                        "source_text_length": len(source_text),
                        "selected_text_length": len(selected_focus_text or ""),
                        "use_rag": revision_use_rag,
                    },
                    input_payload={
                        "source_text": source_text,
                        "selected_text": selected_focus_text,
                        "revision_instruction": focus_instruction,
                    },
                )
                try:
                    preprocessed_request, section_extraction = asyncio.run(
                        clinical_service.preprocess_unified_input(request)
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
                        "patient_name_present": bool(
                            str(getattr(preprocessed_request, "name", "") or "").strip()
                        ),
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
                        "session_version": int(version),
                        "focused_selection": bool(selected_focus_text),
                        "revision_mode": revision_mode,
                        "patient_payload_keys": sorted(
                            patient_payload.keys()
                            if isinstance(patient_payload, dict)
                            else []
                        ),
                    },
                    input_payload={
                        "patient_payload": patient_payload,
                        "section_extraction": section_extraction,
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
                                "use_rag": revision_use_rag,
                                "revision_mode": True,
                                "focused_selection": bool(selected_focus_text),
                                "revision_instruction": focus_instruction,
                                "model_overrides": effective_overrides,
                                "revised_from_session_id": session_detail.get("session_id"),
                            },
                            original_session_text=source_text,
                            revision_focus_context=revision_focus_context,
                            progress_callback=lambda stage, progress: (
                                self.report_job_progress(
                                    job_id=job_id or "",
                                    progress=progress,
                                    message=f"Revision: {stage}",
                                )
                                if job_id
                                else None
                            ),
                            stop_check=None,
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
                        "persisted_session_id": result_payload.get("session_id"),
                        "has_report": bool(
                            str(result_payload.get("report") or "").strip()
                        ),
                        "matched_drug_count": len(
                            result_payload.get("matched_drugs") or []
                        ),
                    },
                )
            revision_audit = self.build_revision_audit(
                source_detail=session_detail,
                result_payload=result_payload,
                selected_text=selected_focus_text,
                revision_instruction=focus_instruction,
                effective_overrides=effective_overrides,
            )
            result_payload["revision_audit"] = revision_audit
            persisted_session_id = result_payload.get("session_id")
            persist_started_at = datetime.now(UTC)
            persist_step = self._record_revision_step_start(
                pipeline_run_id=pipeline_run_id,
                step_name="persist_revision",
                input_summary={
                    "pipeline_run_id": pipeline_run_id,
                    "persisted_session_id": persisted_session_id,
                    "target_revision_version_id": int(target_revision_version_id),
                },
                input_payload={
                    "persisted_session_id": persisted_session_id,
                    "target_revision_version_id": int(target_revision_version_id),
                },
            )
            if isinstance(persisted_session_id, int):
                version_status, llm_qa_status = self._derive_revision_qa_outcome(
                    result_payload
                )
                self.serializer.upsert_session_result_payload(
                    persisted_session_id,
                    result_payload,
                )
                self.serializer.finalize_revision_version(
                    pipeline_run_id=pipeline_run_id,
                    persisted_session_id=persisted_session_id,
                    model_configuration=run_configuration,
                    version_status=version_status,
                    llm_qa_status=llm_qa_status,
                    clinical_review_status="not_reviewed",
                )
                self.serializer.persist_revision_artifacts(
                    pipeline_run_id=pipeline_run_id,
                    revision_version_id=int(target_revision_version_id),
                    result_payload=result_payload,
                )
            elapsed_ms = int((datetime.now(UTC) - run_started_at).total_seconds() * 1000)
            if isinstance(persisted_session_id, int):
                self.serializer.create_or_update_revision_run(
                    pipeline_run_id=pipeline_run_id,
                    session_id=int(session_detail["session_id"]),
                    root_session_id=root_session_id,
                    source_version_id=int(source_version_id),
                    target_revision_version_id=int(target_revision_version_id),
                    revision_mode=revision_mode,
                    revision_kind="llm_assisted_revision",
                    configuration=run_configuration,
                    reviewer_note=str(metadata.get("revision_note") or "").strip() or None,
                    status="completed",
                    initiated_by=str(metadata.get("reviewer") or "").strip() or None,
                    actor_source=actor_source,
                    actor_confidence="unverified",
                    completed_at=datetime.now(UTC),
                    trace_id=pipeline_run_id,
                    latency_ms=elapsed_ms,
                )
                self._record_revision_step_success(
                    pipeline_run_id=pipeline_run_id,
                    step_name="persist_revision",
                    attempt_number=int(persist_step["attempt_number"]),
                    started_at=persist_started_at,
                    output_summary={
                        "persisted_session_id": persisted_session_id,
                        "revision_version_finalized": True,
                    },
                    output_payload={
                        "persisted_session_id": persisted_session_id,
                        "target_revision_version_id": int(target_revision_version_id),
                    },
                )
            else:
                self.serializer.create_or_update_revision_run(
                    pipeline_run_id=pipeline_run_id,
                    session_id=int(session_detail["session_id"]),
                    root_session_id=root_session_id,
                    source_version_id=int(source_version_id),
                    target_revision_version_id=int(target_revision_version_id),
                    revision_mode=revision_mode,
                    revision_kind="llm_assisted_revision",
                    configuration=run_configuration,
                    reviewer_note=str(metadata.get("revision_note") or "").strip() or None,
                    status="failed",
                    initiated_by=str(metadata.get("reviewer") or "").strip() or None,
                    actor_source=actor_source,
                    actor_confidence="unverified",
                    completed_at=datetime.now(UTC),
                    error={"message": "Revision completed without a persisted session record."},
                    trace_id=pipeline_run_id,
                    latency_ms=elapsed_ms,
                )
                self.serializer.fail_revision_step(
                    pipeline_run_id=pipeline_run_id,
                    step_name="persist_revision",
                    attempt_number=int(persist_step["attempt_number"]),
                    error={"message": "Revision completed without a persisted session record."},
                    latency_ms=int(
                        (datetime.now(UTC) - persist_started_at).total_seconds() * 1000
                    ),
                )
        except Exception as exc:
            self.serializer.create_or_update_revision_run(
                pipeline_run_id=pipeline_run_id,
                session_id=int(session_detail["session_id"]),
                root_session_id=root_session_id,
                source_version_id=int(source_version_id),
                target_revision_version_id=int(target_revision_version_id),
                revision_mode=revision_mode,
                revision_kind="llm_assisted_revision",
                configuration=run_configuration,
                reviewer_note=str(metadata.get("revision_note") or "").strip() or None,
                status="failed",
                initiated_by=str(metadata.get("reviewer") or "").strip() or None,
                actor_source=actor_source,
                actor_confidence="unverified",
                completed_at=datetime.now(UTC),
                error={"message": str(exc)[:500]},
                trace_id=pipeline_run_id,
                latency_ms=int((datetime.now(UTC) - run_started_at).total_seconds() * 1000),
            )
            raise
        return {
            "session_id": result_payload.get("session_id"),
            "version": version,
            "original_session_id": root_session_id,
            "pipeline_run_id": pipeline_run_id,
            "result_payload": result_payload,
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def build_revision_focus_context(
        *,
        selected_text: str | None,
        revision_instruction: str | None,
    ) -> str | None:
        chunks: list[str] = []
        if selected_text:
            chunks.append(
                "Selected excerpt to scrutinize during this second pass:\n"
                f"{selected_text}"
            )
        if revision_instruction:
            chunks.append(f"User revision instruction:\n{revision_instruction}")
        return "\n\n".join(chunks) if chunks else None

    # -------------------------------------------------------------------------
    def delete_session(self, session_id: int) -> bool:
        return self.serializer.delete_session(session_id)

    # -------------------------------------------------------------------------
    def get_session_timeline(self, session_id: int) -> PatientTimeline | None:
        return get_session_timeline_value(self, session_id)

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
            "vector_db_path": VECTOR_DB_PATH,
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
        if self.jobs.is_job_running(job_type):
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
        job_id = self.jobs.start_job(job_type=job_type, runner=runner)
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
