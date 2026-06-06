from __future__ import annotations

import asyncio
import difflib
import json
import re
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
from domain.inspection import (
    InspectionJobPhase,
    ReviewerInstructionProfile,
    ReviewerInstructionTrace,
)
from domain.patient_timeline import PatientTimeline
from repositories.serialization.data import (
    DataSerializer,
    DocumentSerializer,
)
from repositories.vectors import LanceVectorDatabase
from services.clinical.timeline import PatientTimelineExtractor
from services.clinical.revision.qa import (
    RevisionQaValidationPayload,
    build_revision_qa_validation_payload,
)
from services.clinical.revision.report_builder import (
    RevisionFinalReportPayload,
    build_revision_final_report_payload,
)
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


def _append_derived_revision_entity(
    *,
    derived: list[dict[str, Any]],
    session_detail: dict[str, Any],
    version_summary: dict[str, Any],
    revision_version_id: int,
    source_version_id: Any,
    pipeline_run_id: str,
    entity_type: str,
    source_section: str,
    original_entity_id: str,
    revised_name: str | None,
    payload: dict[str, Any],
    entity_revision_status: str = "active",
    requires_human_review: bool = False,
    step_name: str = "persisted_session_result",
) -> None:
    normalized_name = normalize_text_value(revised_name)
    derived.append(
        {
            "revision_version_id": revision_version_id,
            "source_version_id": int(source_version_id)
            if source_version_id is not None
            else None,
            "pipeline_run_id": pipeline_run_id,
            "step_name": step_name,
            "entity_type": entity_type,
            "entity_revision_status": entity_revision_status,
            "source_section": source_section,
            "original_entity_id": original_entity_id,
            "original_name": revised_name,
            "revised_name": revised_name,
            "normalized_name": normalized_name or None,
            "requires_human_review": requires_human_review,
            "human_review_status": (
                "required" if requires_human_review else "not_required"
            ),
            "payload": payload,
            "schema_name": "revision_entity",
            "schema_version": "1",
            "prompt_version": None,
            "parser_version": None,
            "model_provider": None,
            "model_name": None,
            "input_hash": None,
            "output_hash": None,
            "created_at": session_detail.get("session_timestamp")
            or version_summary.get("updated_at"),
            "superseded_at": None,
        }
    )


class DataInspectionService:
    RXNAV_JOB_TYPE = "rxnav_update"
    LIVERTOX_JOB_TYPE = "livertox_update"
    RAG_JOB_TYPE = "rag_update"
    REVISION_JOB_TYPE = "session_revision"
    RAG_MANIFEST_FILE_NAME = "rag_index_manifest.json"
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
    def _extract_version_report_text(self, detail: dict[str, Any]) -> str:
        session_detail = detail.get("session")
        if not isinstance(session_detail, dict):
            return ""
        report_text = (
            session_detail.get("official_report_text")
            or session_detail.get("report")
            or (session_detail.get("result_payload") or {}).get("report")
            or ""
        )
        return str(report_text).strip()

    # -------------------------------------------------------------------------
    def _resolve_version_comparison_entities(
        self,
        *,
        version_id: int,
        detail: dict[str, Any],
    ) -> list[dict[str, Any]]:
        persisted_entities = self.list_revision_entities(
            revision_version_id=version_id,
        )
        if persisted_entities:
            return persisted_entities
        return self._derive_entities_from_version_detail(detail)

    # -------------------------------------------------------------------------
    def _derive_entities_from_version_detail(
        self,
        detail: dict[str, Any],
    ) -> list[dict[str, Any]]:
        session_detail = detail.get("session")
        version_summary = detail.get("version")
        if not isinstance(session_detail, dict) or not isinstance(version_summary, dict):
            return []
        result_payload = session_detail.get("result_payload")
        if not isinstance(result_payload, dict):
            result_payload = {}

        revision_version_id = int(version_summary.get("version_id") or 0)
        source_version_id = version_summary.get("source_version_id")
        pipeline_run_id = str(version_summary.get("pipeline_run_id") or "")
        derived: list[dict[str, Any]] = []

        structured_case = result_payload.get("structured_case")
        if isinstance(structured_case, dict):
            for section_name, source_section in (
                ("therapy_drugs", "therapy"),
                ("anamnesis_drugs", "anamnesis"),
            ):
                entries = structured_case.get(section_name)
                if not isinstance(entries, list):
                    continue
                for index, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        continue
                    revised_name = str(
                        entry.get("name") or entry.get("drug_name") or ""
                    ).strip() or None
                    _append_derived_revision_entity(
                        derived=derived,
                        session_detail=session_detail,
                        version_summary=version_summary,
                        revision_version_id=revision_version_id,
                        source_version_id=source_version_id,
                        pipeline_run_id=pipeline_run_id,
                        entity_type="drug",
                        source_section=source_section,
                        original_entity_id=f"{section_name}:{index}",
                        revised_name=revised_name,
                        payload=entry,
                        requires_human_review=not bool(revised_name),
                    )
            diseases = structured_case.get("anamnesis_diseases")
            if isinstance(diseases, list):
                for index, entry in enumerate(diseases):
                    if not isinstance(entry, dict):
                        continue
                    revised_name = str(entry.get("name") or "").strip() or None
                    _append_derived_revision_entity(
                        derived=derived,
                        session_detail=session_detail,
                        version_summary=version_summary,
                        revision_version_id=revision_version_id,
                        source_version_id=source_version_id,
                        pipeline_run_id=pipeline_run_id,
                        entity_type="disease",
                        source_section="anamnesis",
                        original_entity_id=f"anamnesis_diseases:{index}",
                        revised_name=revised_name,
                        payload=entry,
                        requires_human_review=not bool(revised_name),
                    )

        lab_timeline = result_payload.get("lab_timeline")
        if isinstance(lab_timeline, list):
            for index, entry in enumerate(lab_timeline):
                if not isinstance(entry, dict):
                    continue
                revised_name = str(entry.get("marker_name") or "").strip() or None
                _append_derived_revision_entity(
                    derived=derived,
                    session_detail=session_detail,
                    version_summary=version_summary,
                    revision_version_id=revision_version_id,
                    source_version_id=source_version_id,
                    pipeline_run_id=pipeline_run_id,
                    entity_type="lab_timeline_entry",
                    source_section="laboratory_analysis",
                    original_entity_id=f"lab_timeline:{index}",
                    revised_name=revised_name,
                    payload=entry,
                    requires_human_review=not bool(revised_name),
                )

        matched_drugs = result_payload.get("matched_drugs")
        if isinstance(matched_drugs, list):
            for index, entry in enumerate(matched_drugs):
                if not isinstance(entry, dict):
                    continue
                revised_name = str(
                    entry.get("matched_drug_name") or entry.get("raw_drug_name") or ""
                ).strip() or None
                _append_derived_revision_entity(
                    derived=derived,
                    session_detail=session_detail,
                    version_summary=version_summary,
                    revision_version_id=revision_version_id,
                    source_version_id=source_version_id,
                    pipeline_run_id=pipeline_run_id,
                    entity_type="livertox_match",
                    source_section="therapy",
                    original_entity_id=f"matched_drug:{index}",
                    revised_name=revised_name,
                    payload=entry,
                    entity_revision_status=str(entry.get("match_status") or "active"),
                    requires_human_review=bool(entry.get("requires_human_review")),
                )

        rucam_assessments = result_payload.get("rucam_assessments")
        if isinstance(rucam_assessments, list):
            for index, entry in enumerate(rucam_assessments):
                if not isinstance(entry, dict):
                    continue
                revised_name = str(entry.get("drug_name") or "").strip() or None
                _append_derived_revision_entity(
                    derived=derived,
                    session_detail=session_detail,
                    version_summary=version_summary,
                    revision_version_id=revision_version_id,
                    source_version_id=source_version_id,
                    pipeline_run_id=pipeline_run_id,
                    entity_type="dili_assessment",
                    source_section="therapy",
                    original_entity_id=f"rucam_assessment:{index}",
                    revised_name=revised_name,
                    payload=entry,
                    requires_human_review=bool(entry.get("requires_human_review")),
                )
        return derived

    # -------------------------------------------------------------------------
    @staticmethod
    def _comparison_entity_key(entity: dict[str, Any]) -> tuple[str, str, str]:
        entity_type = str(entity.get("entity_type") or "").strip()
        normalized_name = str(
            entity.get("normalized_name")
            or entity.get("revised_name")
            or entity.get("original_name")
            or entity.get("original_entity_id")
            or ""
        ).strip()
        source_section = str(entity.get("source_section") or "").strip()
        return entity_type, normalized_name, source_section

    # -------------------------------------------------------------------------
    @classmethod
    def _build_entity_diff_item(
        cls,
        *,
        change_type: str,
        left_entity: dict[str, Any] | None,
        right_entity: dict[str, Any] | None,
    ) -> dict[str, Any]:
        reference = right_entity if right_entity is not None else left_entity or {}
        entity_type = str(reference.get("entity_type") or "").strip()
        normalized_name = str(reference.get("normalized_name") or "").strip() or None
        source_section = str(reference.get("source_section") or "").strip() or None
        revised_name = str(reference.get("revised_name") or normalized_name or "").strip()
        if not revised_name:
            revised_name = entity_type or "entity"
        summary = f"{revised_name} ({entity_type or 'entity'})"
        return {
            "entity_type": entity_type,
            "normalized_name": normalized_name,
            "source_section": source_section,
            "change_type": change_type,
            "summary": summary,
            "requires_human_review": bool(reference.get("requires_human_review")),
            "left_entity": left_entity,
            "right_entity": right_entity,
        }

    # -------------------------------------------------------------------------
    @classmethod
    def _build_version_entity_diff(
        cls,
        *,
        left_entities: list[dict[str, Any]],
        right_entities: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        left_map = {cls._comparison_entity_key(item): item for item in left_entities}
        right_map = {cls._comparison_entity_key(item): item for item in right_entities}

        added_entities: list[dict[str, Any]] = []
        removed_entities: list[dict[str, Any]] = []
        corrected_entities: list[dict[str, Any]] = []
        replaced_entities: list[dict[str, Any]] = []
        unresolved_entities: list[dict[str, Any]] = []
        unchanged_entities: list[dict[str, Any]] = []

        for key in sorted(set(left_map) | set(right_map)):
            left_entity = left_map.get(key)
            right_entity = right_map.get(key)
            if left_entity is None and right_entity is not None:
                added_entities.append(
                    cls._build_entity_diff_item(
                        change_type="added",
                        left_entity=None,
                        right_entity=right_entity,
                    )
                )
                continue
            if right_entity is None and left_entity is not None:
                removed_entities.append(
                    cls._build_entity_diff_item(
                        change_type="removed",
                        left_entity=left_entity,
                        right_entity=None,
                    )
                )
                continue
            if left_entity is None or right_entity is None:
                continue

            left_payload = left_entity.get("payload") if isinstance(left_entity, dict) else None
            right_payload = right_entity.get("payload") if isinstance(right_entity, dict) else None
            payload_changed = left_payload != right_payload
            right_status = str(right_entity.get("entity_revision_status") or "").strip().casefold()
            if right_entity.get("requires_human_review"):
                unresolved_entities.append(
                    cls._build_entity_diff_item(
                        change_type="unresolved",
                        left_entity=left_entity,
                        right_entity=right_entity,
                    )
                )
            elif payload_changed and "replace" in right_status:
                replaced_entities.append(
                    cls._build_entity_diff_item(
                        change_type="replaced",
                        left_entity=left_entity,
                        right_entity=right_entity,
                    )
                )
            elif payload_changed:
                corrected_entities.append(
                    cls._build_entity_diff_item(
                        change_type="corrected",
                        left_entity=left_entity,
                        right_entity=right_entity,
                    )
                )
            else:
                unchanged_entities.append(
                    cls._build_entity_diff_item(
                        change_type="unchanged",
                        left_entity=left_entity,
                        right_entity=right_entity,
                    )
                )

        return {
            "added_entities": added_entities,
            "removed_entities": removed_entities,
            "corrected_entities": corrected_entities,
            "replaced_entities": replaced_entities,
            "unresolved_entities": unresolved_entities,
            "unchanged_entities": unchanged_entities,
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_report_text_diff(
        *,
        left_text: str,
        right_text: str,
    ) -> dict[str, Any]:
        left_lines = left_text.splitlines()
        right_lines = right_text.splitlines()
        diff_lines = list(
            difflib.unified_diff(
                left_lines,
                right_lines,
                fromfile="left_version",
                tofile="right_version",
                lineterm="",
                n=2,
            )
        )
        return {
            "changed": left_text != right_text,
            "left_character_count": len(left_text),
            "right_character_count": len(right_text),
            "left_line_count": len(left_lines),
            "right_line_count": len(right_lines),
            "similarity_ratio": round(
                difflib.SequenceMatcher(a=left_text, b=right_text).ratio(),
                4,
            ),
            "diff_lines": diff_lines[:80],
        }

    # -------------------------------------------------------------------------
    def _resolve_qa_payload(self, detail: dict[str, Any]) -> dict[str, Any]:
        version = detail.get("version")
        if not isinstance(version, dict):
            return {}
        version_id = int(version.get("version_id") or 0)
        for artifact in self.list_revision_artifacts(revision_version_id=version_id):
            if str(artifact.get("artifact_key") or "").strip() == "revision_qa_validation":
                payload = artifact.get("payload")
                if isinstance(payload, dict):
                    return payload
        session_detail = detail.get("session")
        if not isinstance(session_detail, dict):
            return {}
        result_payload = session_detail.get("result_payload")
        if not isinstance(result_payload, dict):
            return {}
        revision_payload = result_payload.get("revision")
        if isinstance(revision_payload, dict):
            qa_validation = revision_payload.get("qa_validation")
            if isinstance(qa_validation, dict):
                return qa_validation
        return {}

    # -------------------------------------------------------------------------
    def _build_revision_qa_summary(
        self,
        *,
        left_detail: dict[str, Any],
        right_detail: dict[str, Any],
    ) -> dict[str, Any]:
        left_version = left_detail.get("version") or {}
        right_version = right_detail.get("version") or {}
        left_payload = self._resolve_qa_payload(left_detail)
        right_payload = self._resolve_qa_payload(right_detail)
        left_warnings = [
            str(item).strip()
            for item in (left_payload.get("warnings") or [])
            if str(item).strip()
        ]
        right_warnings = [
            str(item).strip()
            for item in (right_payload.get("warnings") or [])
            if str(item).strip()
        ]
        left_blocking = [
            str(item).strip()
            for item in (left_payload.get("blocking_issues") or [])
            if str(item).strip()
        ]
        right_blocking = [
            str(item).strip()
            for item in (right_payload.get("blocking_issues") or [])
            if str(item).strip()
        ]
        left_finding_count = int(
            left_payload.get("finding_count")
            or len(left_warnings) + len(left_blocking)
        )
        right_finding_count = int(
            right_payload.get("finding_count")
            or len(right_warnings) + len(right_blocking)
        )
        return {
            "left_llm_qa_status": str(left_version.get("llm_qa_status") or "not_run"),
            "right_llm_qa_status": str(
                right_version.get("llm_qa_status") or "not_run"
            ),
            "left_clinical_review_status": str(
                left_version.get("clinical_review_status") or "not_reviewed"
            ),
            "right_clinical_review_status": str(
                right_version.get("clinical_review_status") or "not_reviewed"
            ),
            "left_version_status": str(left_version.get("version_status") or ""),
            "right_version_status": str(right_version.get("version_status") or ""),
            "left_warning_count": len(left_warnings),
            "right_warning_count": len(right_warnings),
            "left_blocking_issue_count": len(left_blocking),
            "right_blocking_issue_count": len(right_blocking),
            "left_finding_count": left_finding_count,
            "right_finding_count": right_finding_count,
            "manual_review_required": bool(left_payload.get("manual_review_required"))
            or bool(right_payload.get("manual_review_required"))
            or bool(left_blocking)
            or bool(right_blocking),
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
    def _unique_preserve_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            cleaned = str(value or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique.append(cleaned)
        return unique

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
        return cls._unique_preserve_order(detections)

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

        if any(section in target_sections for section in {"anamnesis", "therapy", "labs"}):
            routed_steps.append("preprocess_input")
        if "qa" in target_sections or "source_evidence" in target_entities:
            routed_steps.append("qa_validate_revision")

        mentioned_dates = cls._unique_preserve_order(
            [
                match.group(0)
                for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", summary)
            ]
        )
        mentioned_lab_values = cls._unique_preserve_order(
            [
                match.group(0)
                for match in re.finditer(
                    r"\b(?:ALT|AST|ALP|bilirubin|bilirubina)\b[^.;,\n]{0,20}\d+(?:\.\d+)?",
                    summary,
                    re.IGNORECASE,
                )
            ]
        )
        extra_data = cls._unique_preserve_order(
            [selected_text.strip()] if str(selected_text or "").strip() else []
        )
        ambiguities = (
            ["Reviewer instruction contains ambiguity markers."]
            if any(token in lowered for token in ("maybe", "unclear", "check", "verify"))
            else []
        )
        constraints = (
            ["Limit changes to the explicitly targeted scope."]
            if any(token in lowered for token in ("only", "do not", "don't", "must not"))
            else []
        )
        safety_or_quality_concerns = (
            ["Reviewer requested evidence or consistency validation."]
            if any(token in lowered for token in ("evidence", "source", "consistent", "qa"))
            else []
        )
        prompt_injection_flags = cls.detect_prompt_injection_flags(
            instruction_text=summary,
            selected_text=selected_text,
        )
        if prompt_injection_flags:
            safety_or_quality_concerns = cls._unique_preserve_order(
                safety_or_quality_concerns
                + [
                    "Potential prompt-injection or instruction-redirection content detected in untrusted revision inputs."
                ]
            )

        profile = ReviewerInstructionProfile(
            user_intent="revision_request",
            main_goal=summary[:200] or None,
            instruction_summary=summary,
            target_sections=cls._unique_preserve_order(target_sections),  # type: ignore[arg-type]
            target_entities=cls._unique_preserve_order(target_entities),  # type: ignore[arg-type]
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
                "generate_revision": cls._unique_preserve_order(target_sections),
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
            routed_pipeline_steps=cls._unique_preserve_order(routed_steps),
            affected_entities=cls._unique_preserve_order(target_entities),
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
            chunks.append(f"Reviewer-selected source excerpt:\n{str(selected_text).strip()}")
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
    @staticmethod
    def build_revision_livertox_decisions(
        *,
        matched_drugs: list[Any],
        source_matched_drugs: list[Any] | None = None,
        instruction_profile: ReviewerInstructionProfile | None,
    ) -> list[dict[str, Any]]:
        challenged_matching = bool(
            instruction_profile
            and "matching_errors" in instruction_profile.target_entities
        )
        source_match_lookup: dict[str, dict[str, Any]] = {}
        for item in source_matched_drugs or []:
            if not isinstance(item, dict):
                continue
            drug_name = str(
                item.get("matched_drug_name")
                or item.get("raw_drug_name")
                or item.get("drug_name")
                or ""
            ).strip()
            normalized = normalize_drug_query_name(drug_name)
            if normalized:
                source_match_lookup[normalized] = item
        decisions: list[dict[str, Any]] = []
        for index, item in enumerate(matched_drugs):
            if not isinstance(item, dict):
                decisions.append(
                    {
                        "decision_id": f"livertox:{index}",
                        "drug_name": str(item).strip() or f"drug-{index + 1}",
                        "decision": "requires_human_review",
                        "decision_reason": "Matched-drug payload is not structured.",
                        "match_status": "unknown",
                        "match_confidence": None,
                        "requires_human_review": True,
                        "source": "none",
                        "previous_match_found": False,
                        "provenance": {"source_version_match": None},
                    }
                )
                continue
            drug_name = str(
                item.get("matched_drug_name")
                or item.get("raw_drug_name")
                or item.get("drug_name")
                or f"drug-{index + 1}"
            ).strip()
            match_status = str(item.get("match_status") or "unknown").strip().lower()
            raw_confidence = item.get("match_confidence")
            try:
                match_confidence = (
                    float(raw_confidence) if raw_confidence is not None else None
                )
            except (TypeError, ValueError):
                match_confidence = None
            normalized_drug_name = normalize_drug_query_name(drug_name)
            previous_match = (
                source_match_lookup.get(normalized_drug_name or "")
                if normalized_drug_name
                else None
            )
            previous_match_found = isinstance(previous_match, dict)
            previous_match_name = str(
                (previous_match or {}).get("matched_drug_name")
                or (previous_match or {}).get("raw_drug_name")
                or (previous_match or {}).get("drug_name")
                or ""
            ).strip()
            previous_match_confidence = None
            try:
                if previous_match is not None:
                    previous_confidence_raw = previous_match.get("match_confidence")
                    previous_match_confidence = (
                        float(previous_confidence_raw)
                        if previous_confidence_raw is not None
                        else None
                    )
            except (TypeError, ValueError):
                previous_match_confidence = None
            same_match_name = bool(
                previous_match_name
                and previous_match_name.casefold() == drug_name.casefold()
            )
            if challenged_matching and previous_match_found:
                decision = "llm_assisted_resolved_match"
                reason = "Reviewer instruction challenged the previous source-version match."
                requires_human_review = False
                decision_source = "llm_fallback"
            elif (
                previous_match_found
                and previous_match_confidence is not None
                and previous_match_confidence >= 0.95
                and same_match_name
            ):
                decision = "reused_high_confidence_previous_match"
                reason = "High-confidence previous source-version match remains valid."
                requires_human_review = False
                decision_source = "previous_version"
            elif match_status in {"matched_with_excerpt", "matched"} and (
                match_confidence is not None and match_confidence >= 0.95
            ):
                decision = "deterministic_new_match"
                reason = "Revision produced a high-confidence structured LiverTox match."
                requires_human_review = False
                decision_source = "deterministic"
            elif match_status in {"missing_match", "ambiguous_match", "missing"}:
                decision = "no_reliable_match_found"
                reason = "No reliable prior LiverTox match is available."
                requires_human_review = True
                decision_source = "none"
            else:
                decision = "llm_assisted_resolved_match"
                reason = "Revision required a refreshed LiverTox decision."
                requires_human_review = False
                decision_source = "llm_fallback"
            decisions.append(
                {
                    "decision_id": f"livertox:{index}",
                    "drug_name": drug_name,
                    "normalized_drug_name": normalized_drug_name,
                    "decision": decision,
                    "decision_reason": reason,
                    "match_status": match_status,
                    "match_confidence": match_confidence,
                    "requires_human_review": requires_human_review,
                    "reviewer_challenged": challenged_matching,
                    "source": decision_source,
                    "previous_match_found": previous_match_found,
                    "previous_match_confidence": previous_match_confidence,
                    "payload": item,
                    "provenance": {
                        "source_version_match": previous_match,
                        "current_revision_match": item,
                    },
                }
            )
        return decisions

    # -------------------------------------------------------------------------
    @staticmethod
    def build_revised_dili_assessments(
        *,
        rucam_assessments: list[Any],
        matched_drugs: list[Any],
        source_rucam_assessments: list[Any] | None = None,
        revision_version_id: int,
        source_version_id: int,
        instruction_profile: ReviewerInstructionProfile | None,
    ) -> list[dict[str, Any]]:
        matched_lookup: dict[str, dict[str, Any]] = {}
        for item in matched_drugs:
            if not isinstance(item, dict):
                continue
            drug_name = str(
                item.get("matched_drug_name")
                or item.get("raw_drug_name")
                or item.get("drug_name")
                or ""
            ).strip()
            normalized = normalize_drug_query_name(drug_name)
            if normalized:
                matched_lookup[normalized] = item
        previous_assessment_lookup: dict[str, dict[str, Any]] = {}
        for item in source_rucam_assessments or []:
            if not isinstance(item, dict):
                continue
            drug_name = str(item.get("drug_name") or "").strip()
            normalized = normalize_drug_query_name(drug_name)
            if normalized:
                previous_assessment_lookup[normalized] = item
        assessments: list[dict[str, Any]] = []
        for index, item in enumerate(rucam_assessments):
            if not isinstance(item, dict):
                continue
            drug_name = str(item.get("drug_name") or f"drug-{index + 1}").strip()
            normalized = normalize_drug_query_name(drug_name)
            matched_row = matched_lookup.get(normalized or "")
            previous_assessment = previous_assessment_lookup.get(normalized or "")
            total_score = item.get("total_score")
            confidence = "moderate"
            if isinstance(total_score, (int, float)):
                if float(total_score) >= 9:
                    confidence = "high"
                elif float(total_score) <= 3:
                    confidence = "low"
            unresolved_questions = []
            if matched_row is None:
                unresolved_questions.append(
                    "No reliable LiverTox match is available for this revised drug."
                )
            if instruction_profile and "causality_reasoning" in instruction_profile.target_entities:
                unresolved_questions.append(
                    "Reviewer explicitly requested reassessment of causality reasoning."
                )
            changes_from_previous_version: list[str] = []
            previous_causality = str(
                (previous_assessment or {}).get("causality_category")
                or (previous_assessment or {}).get("causality_assessment")
                or ""
            ).strip()
            current_causality = str(
                item.get("causality_category")
                or item.get("causality_assessment")
                or "unresolved"
            )
            if previous_causality and previous_causality != current_causality:
                changes_from_previous_version.append(
                    f"Causality changed from {previous_causality} to {current_causality}."
                )
            previous_score = (previous_assessment or {}).get("total_score")
            if (
                isinstance(previous_score, (int, float))
                and isinstance(total_score, (int, float))
                and float(previous_score) != float(total_score)
            ):
                changes_from_previous_version.append(
                    f"Total score changed from {float(previous_score):g} to {float(total_score):g}."
                )
            if previous_assessment and not changes_from_previous_version:
                changes_from_previous_version.append(
                    "Previous source-version assessment was reviewed and retained."
                )
            assessments.append(
                {
                    "drug_id": item.get("drug_id"),
                    "revised_drug_entry_id": f"revised-drug:{index}",
                    "revision_version_id": revision_version_id,
                    "source_version_id": source_version_id,
                    "assessment_version": "1",
                    "drug_name": drug_name,
                    "causality_assessment": str(
                        item.get("causality_category")
                        or item.get("causality_assessment")
                        or "unresolved"
                    ),
                    "confidence": confidence,
                    "evidence_for": [],
                    "evidence_against": [],
                    "lab_support": [],
                    "temporal_support": [],
                    "alternative_causes": [],
                    "livertox_support": [str(matched_row.get("matched_drug_name"))]
                    if isinstance(matched_row, dict)
                    and str(matched_row.get("matched_drug_name") or "").strip()
                    else [],
                    "changes_from_previous_version": changes_from_previous_version,
                    "unresolved_questions": unresolved_questions,
                    "requires_human_review": bool(unresolved_questions),
                    "previous_assessment_present": bool(previous_assessment),
                    "provenance": {
                        "source_version_assessment": previous_assessment,
                        "current_revision_assessment": item,
                    },
                }
            )
        return assessments

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

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
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
                "anamnesis_drug_count": len(
                    payload.get("anamnesis_drug_names") or []
                ),
                "analysis_drug_count": len(payload.get("analysis_drug_names") or []),
                "rucam_assessment_count": int(
                    payload.get("rucam_assessment_count") or 0
                ),
            }
        return {"status": payload.get("status")}

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
                    "has_section_extraction": bool(
                        session_detail.get("sections")
                    ),
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
                        "target_entity_count": len(
                            instruction_profile.target_entities
                        ),
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
                                "source_deterministic_extraction": (
                                    source_deterministic_extraction
                                    if isinstance(source_deterministic_extraction, dict)
                                    else None
                                ),
                                "source_structured_case": (
                                    source_structured_case
                                    if isinstance(source_structured_case, dict)
                                    else None
                                ),
                                "source_lab_timeline": (
                                    source_lab_timeline
                                    if isinstance(source_lab_timeline, list)
                                    else None
                                ),
                                "source_onset_context": (
                                    source_onset_context
                                    if isinstance(source_onset_context, dict)
                                    else None
                                ),
                                "source_matched_drugs": (
                                    source_matched_drugs
                                    if isinstance(source_matched_drugs, list)
                                    else None
                                ),
                                "source_rucam_assessments": (
                                    source_rucam_assessments
                                    if isinstance(source_rucam_assessments, list)
                                    else None
                                ),
                                "source_official_report_text": session_detail.get(
                                    "official_report_text"
                                ),
                                "source_version_id": int(source_version_id),
                                "target_revision_version_id": int(
                                    target_revision_version_id
                                ),
                                "pipeline_run_id": pipeline_run_id,
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
            revision_entity_pipeline = self._get_revision_entity_pipeline(result_payload)
            for entity_stage_name in (
                "resolve_revision_extraction",
                "validate_anamnesis_drugs",
                "extract_missing_anamnesis_drugs",
                "revise_labs_timeline",
                "reconcile_revision_candidates",
                "merge_revision_snapshot",
            ):
                entity_stage_payload = revision_entity_pipeline.get(entity_stage_name)
                entity_stage_started_at = datetime.now(UTC)
                entity_stage_step = self._record_revision_step_start(
                    pipeline_run_id=pipeline_run_id,
                    step_name=entity_stage_name,
                    input_summary={
                        "generated_revision_present": True,
                        "entity_stage_available": isinstance(
                            entity_stage_payload, dict
                        ),
                    },
                    input_payload={
                        "available": isinstance(entity_stage_payload, dict),
                    },
                )
                if isinstance(entity_stage_payload, dict):
                    self._record_revision_step_success(
                        pipeline_run_id=pipeline_run_id,
                        step_name=entity_stage_name,
                        attempt_number=int(entity_stage_step["attempt_number"]),
                        started_at=entity_stage_started_at,
                        output_summary=self._summarize_revision_entity_stage_payload(
                            entity_stage_name,
                            entity_stage_payload,
                        ),
                        output_payload=entity_stage_payload,
                    )
                else:
                    self._record_revision_step_success(
                        pipeline_run_id=pipeline_run_id,
                        step_name=entity_stage_name,
                        attempt_number=int(entity_stage_step["attempt_number"]),
                        started_at=entity_stage_started_at,
                        output_summary={
                            "skipped": True,
                            "reason": "Revision entity stage payload was unavailable.",
                        },
                        output_payload={"skipped": True},
                    )
            matched_drugs_payload = (
                result_payload.get("matched_drugs")
                if isinstance(result_payload.get("matched_drugs"), list)
                else []
            )
            resolve_matches_started_at = datetime.now(UTC)
            resolve_matches_step = self._record_revision_step_start(
                pipeline_run_id=pipeline_run_id,
                step_name="resolve_livertox_matches",
                input_summary={
                    "matched_drug_count": len(matched_drugs_payload),
                    "reviewer_challenged_matching": bool(
                        instruction_profile
                        and "matching_errors" in instruction_profile.target_entities
                    ),
                },
                input_payload={"matched_drugs": matched_drugs_payload},
            )
            livertox_revision_decisions = self.build_revision_livertox_decisions(
                matched_drugs=matched_drugs_payload,
                source_matched_drugs=source_matched_drugs,
                instruction_profile=instruction_profile,
            )
            self._record_revision_step_success(
                pipeline_run_id=pipeline_run_id,
                step_name="resolve_livertox_matches",
                attempt_number=int(resolve_matches_step["attempt_number"]),
                started_at=resolve_matches_started_at,
                output_summary={
                    "decision_count": len(livertox_revision_decisions),
                    "reused_count": sum(
                        1
                        for item in livertox_revision_decisions
                        if item.get("decision") == "reuse"
                    ),
                    "rerun_count": sum(
                        1
                        for item in livertox_revision_decisions
                        if item.get("decision") == "rerun"
                    ),
                    "requires_human_review_count": sum(
                        1
                        for item in livertox_revision_decisions
                        if bool(item.get("requires_human_review"))
                    ),
                },
                output_payload={"decisions": livertox_revision_decisions},
            )
            rucam_assessment_payload = (
                result_payload.get("rucam_assessments")
                if isinstance(result_payload.get("rucam_assessments"), list)
                else []
            )
            rerun_assessments_started_at = datetime.now(UTC)
            rerun_assessments_step = self._record_revision_step_start(
                pipeline_run_id=pipeline_run_id,
                step_name="rerun_dili_assessments",
                input_summary={
                    "assessment_count": len(rucam_assessment_payload),
                    "match_decision_count": len(livertox_revision_decisions),
                },
                input_payload={"rucam_assessments": rucam_assessment_payload},
            )
            revised_dili_assessments = self.build_revised_dili_assessments(
                rucam_assessments=rucam_assessment_payload,
                matched_drugs=matched_drugs_payload,
                source_rucam_assessments=source_rucam_assessments,
                revision_version_id=int(target_revision_version_id),
                source_version_id=int(source_version_id),
                instruction_profile=instruction_profile,
            )
            self._record_revision_step_success(
                pipeline_run_id=pipeline_run_id,
                step_name="rerun_dili_assessments",
                attempt_number=int(rerun_assessments_step["attempt_number"]),
                started_at=rerun_assessments_started_at,
                output_summary={
                    "assessment_count": len(revised_dili_assessments),
                    "requires_human_review_count": sum(
                        1
                        for item in revised_dili_assessments
                        if bool(item.get("requires_human_review"))
                    ),
                },
                output_payload={"assessments": revised_dili_assessments},
            )
            revision_audit = self.build_revision_audit(
                source_detail=session_detail,
                result_payload=result_payload,
                selected_text=selected_focus_text,
                revision_instruction=focus_instruction,
                effective_overrides=effective_overrides,
            )
            revision_payload = result_payload.get("revision")
            if not isinstance(revision_payload, dict):
                revision_payload = {}
                result_payload["revision"] = revision_payload
            if instruction_profile is not None:
                revision_payload["instruction_profile"] = instruction_profile.model_dump()
            if instruction_trace is not None:
                revision_payload["instruction_trace"] = instruction_trace.model_dump()
            revision_payload["livertox_revision_decisions"] = livertox_revision_decisions
            revision_payload["revised_dili_assessments"] = revised_dili_assessments
            rebuild_report_started_at = datetime.now(UTC)
            rebuild_report_step = self._record_revision_step_start(
                pipeline_run_id=pipeline_run_id,
                step_name="rebuild_final_report",
                input_summary={
                    "report_present": bool(str(result_payload.get("report") or "").strip()),
                    "selected_text_present": bool(selected_focus_text),
                    "instruction_profile_present": instruction_profile is not None,
                },
                input_payload={
                    "report_present": bool(str(result_payload.get("report") or "").strip()),
                    "selected_text": selected_focus_text,
                },
            )
            final_report_payload = build_revision_final_report_payload(
                result_payload=result_payload,
                selected_text=selected_focus_text,
                instruction_profile=instruction_profile,
            )
            revision_payload["final_report_rebuild"] = final_report_payload.model_dump()
            self._record_revision_step_success(
                pipeline_run_id=pipeline_run_id,
                step_name="rebuild_final_report",
                attempt_number=int(rebuild_report_step["attempt_number"]),
                started_at=rebuild_report_started_at,
                output_summary={
                    "report_present": final_report_payload.report_present,
                    "report_character_count": final_report_payload.report_character_count,
                    "warning_count": len(final_report_payload.warnings),
                },
                output_payload=final_report_payload.model_dump(),
            )
            qa_started_at = datetime.now(UTC)
            qa_step = self._record_revision_step_start(
                pipeline_run_id=pipeline_run_id,
                step_name="qa_validate_revision",
                input_summary={
                    "blocking_issue_count": len(result_payload.get("blocking_issues") or []),
                    "manual_review_required": bool(
                        result_payload.get("manual_review_required")
                    ),
                    "report_present": final_report_payload.report_present,
                },
                input_payload={
                    "report_present": final_report_payload.report_present,
                    "reviewer_instruction_summary": (
                        instruction_profile.instruction_summary
                        if instruction_profile is not None
                        else None
                    ),
                },
            )
            qa_validation_payload = build_revision_qa_validation_payload(
                result_payload=result_payload,
                instruction_profile=instruction_profile,
                final_report_payload=final_report_payload,
            )
            revision_payload["qa_validation"] = qa_validation_payload.model_dump()
            if instruction_trace is not None:
                instruction_trace.qa_validation_result = qa_validation_payload.status
                revision_payload["instruction_trace"] = instruction_trace.model_dump()
            self._record_revision_step_success(
                pipeline_run_id=pipeline_run_id,
                step_name="qa_validate_revision",
                attempt_number=int(qa_step["attempt_number"]),
                started_at=qa_started_at,
                output_summary={
                    "status": qa_validation_payload.status,
                    "version_status": qa_validation_payload.version_status,
                    "finding_count": qa_validation_payload.finding_count,
                },
                output_payload=qa_validation_payload.model_dump(),
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
                try:
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
                except Exception as exc:
                    self._record_revision_step_failure(
                        pipeline_run_id=pipeline_run_id,
                        step_name="persist_revision",
                        attempt_number=int(persist_step["attempt_number"]),
                        started_at=persist_started_at,
                        exc=exc,
                    )
                    raise
            elapsed_ms = int((datetime.now(UTC) - run_started_at).total_seconds() * 1000)
            if isinstance(persisted_session_id, int):
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
                finalize_started_at = datetime.now(UTC)
                finalize_step = self._record_revision_step_start(
                    pipeline_run_id=pipeline_run_id,
                    step_name="finalize_revision_version",
                    input_summary={
                        "persisted_session_id": persisted_session_id,
                        "target_revision_version_id": int(target_revision_version_id),
                    },
                    input_payload={
                        "persisted_session_id": persisted_session_id,
                        "target_revision_version_id": int(target_revision_version_id),
                    },
                )
                try:
                    version_status, llm_qa_status = self._derive_revision_qa_outcome(
                        result_payload
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
                except Exception as exc:
                    self._record_revision_step_failure(
                        pipeline_run_id=pipeline_run_id,
                        step_name="finalize_revision_version",
                        attempt_number=int(finalize_step["attempt_number"]),
                        started_at=finalize_started_at,
                        exc=exc,
                    )
                    raise
                self._record_revision_step_success(
                    pipeline_run_id=pipeline_run_id,
                    step_name="finalize_revision_version",
                    attempt_number=int(finalize_step["attempt_number"]),
                    started_at=finalize_started_at,
                    output_summary={
                        "persisted_session_id": persisted_session_id,
                        "version_status": version_status,
                        "llm_qa_status": llm_qa_status,
                    },
                    output_payload={
                        "persisted_session_id": persisted_session_id,
                        "target_revision_version_id": int(target_revision_version_id),
                        "version_status": version_status,
                        "llm_qa_status": llm_qa_status,
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
