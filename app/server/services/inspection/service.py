from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from common.constants import (
    DOCS_PATH,
    DOCUMENT_SUPPORTED_EXTENSIONS,
    VECTOR_DB_PATH,
)
from common.utils.logger import logger
from configurations.startup import get_server_settings
from domain.clinical.entities import ClinicalSessionRequest
from domain.inspection import InspectionJobPhase
from domain.patient_timeline import PatientTimeline
from repositories.serialization.data import (
    DataSerializer,
    DocumentSerializer,
)
from repositories.serialization.model_configs import ModelConfigSerializer
from repositories.vectors import LanceVectorDatabase
from services.clinical.timeline import PatientTimelineExtractor
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
        return Path(VECTOR_DB_PATH) / self.RAG_MANIFEST_FILE_NAME

    def read_rag_manifest(self) -> dict[str, Any]:
        manifest_path = self.rag_manifest_path()
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
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
            source = config.get("rag", {})
            defaults = {
                "documents_path": str(source.get("documents_path", DOCS_PATH)),
                "chunk_size": int(
                    source.get("chunk_size", settings.rag.chunk_size)
                ),
                "chunk_overlap": int(
                    source.get("chunk_overlap", settings.rag.chunk_overlap)
                ),
                "embedding_batch_size": int(
                    source.get(
                        "embedding_batch_size", settings.rag.embedding_batch_size
                    )
                ),
                "vector_stream_batch_size": int(
                    source.get(
                        "vector_stream_batch_size",
                        settings.rag.vector_stream_batch_size,
                    )
                ),
                "embedding_max_workers": int(
                    source.get(
                        "embedding_max_workers",
                        settings.rag.embedding_max_workers,
                    )
                ),
                "embedding_backend": str(
                    source.get(
                        "embedding_backend", settings.rag.embedding_backend
                    )
                ),
                "ollama_embedding_model": str(
                    source.get(
                        "ollama_embedding_model",
                        settings.rag.ollama_embedding_model,
                    )
                ),
                "hf_embedding_model": str(
                    source.get(
                        "hf_embedding_model", settings.rag.hf_embedding_model
                    )
                ),
                "cloud_provider": str(
                    source.get("cloud_provider", settings.rag.cloud_provider)
                ),
                "cloud_embedding_model": str(
                    source.get(
                        "cloud_embedding_model",
                        settings.rag.cloud_embedding_model,
                    )
                ),
                "use_cloud_embeddings": bool(
                    source.get(
                        "use_cloud_embeddings", settings.rag.use_cloud_embeddings
                    )
                ),
                "reset_vector_collection": bool(
                    source.get(
                        "reset_vector_collection",
                        settings.rag.reset_vector_collection,
                    )
                ),
            }
            allowed_fields = list(defaults.keys())

        return {
            "target": target,
            "defaults": defaults,
            "allowed_fields": allowed_fields,
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
    def update_session(
        self,
        session_id: int,
        *,
        session_text: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return self.serializer.update_session_text_and_metadata(
            session_id,
            session_text=session_text,
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
        source_payload = (
            source_detail.get("result_payload")
            if isinstance(source_detail.get("result_payload"), dict)
            else {}
        )
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
        source_sections = (
            source_detail.get("sections")
            if isinstance(source_detail.get("sections"), dict)
            else {}
        )
        extracted_sections = section_extraction if isinstance(section_extraction, dict) else {}
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
    @staticmethod
    def _resolve_override_value(
        overrides: dict[str, Any],
        key: str,
        fallback: Any,
    ) -> Any:
        return overrides[key] if key in overrides else fallback

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
        job_id = self.jobs.start_job(
            job_type=self.REVISION_JOB_TYPE,
            runner=self.run_revision_job,
            kwargs={
                "job_id": None,
                "session_detail": detail,
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
        return status_payload

    # -------------------------------------------------------------------------
    def run_revision_job(
        self,
        *,
        job_id: str | None,
        session_detail: dict[str, Any],
        root_session_id: int,
        version: int,
        selected_text: str | None,
        revision_instruction: str | None,
        model_overrides: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
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
        config_serializer = ModelConfigSerializer()
        previous_snapshot = config_serializer.load_snapshot()
        effective_overrides = {
            key: value for key, value in (model_overrides or {}).items() if value is not None
        }

        try:
            if effective_overrides:
                config_serializer.save_snapshot(
                    clinical_model=self._resolve_override_value(effective_overrides, "clinical_model", previous_snapshot.clinical_model),
                    text_extraction_model=self._resolve_override_value(effective_overrides, "text_extraction_model", previous_snapshot.text_extraction_model),
                    use_cloud_models=self._resolve_override_value(effective_overrides, "use_cloud_services", previous_snapshot.use_cloud_models),
                    cloud_provider=self._resolve_override_value(effective_overrides, "provider", previous_snapshot.cloud_provider),
                    cloud_model=self._resolve_override_value(effective_overrides, "cloud_model", previous_snapshot.cloud_model),
                    ollama_temperature=self._resolve_override_value(effective_overrides, "ollama_temperature", previous_snapshot.ollama_temperature),
                    cloud_temperature=self._resolve_override_value(effective_overrides, "cloud_temperature", previous_snapshot.cloud_temperature),
                    ollama_reasoning=self._resolve_override_value(effective_overrides, "ollama_reasoning", previous_snapshot.ollama_reasoning),
                )
            clinical_service.apply_persisted_runtime_configuration()
            request = ClinicalSessionRequest(
                name=session_detail.get("patient_name"),
                visit_date=session_detail.get("visit_date"),
                clinical_input=source_text,
                use_rag=True,
            )
            preprocessed_request, section_extraction = asyncio.run(
                clinical_service.preprocess_unified_input(request)
            )
            patient_payload = clinical_service.build_patient_payload(preprocessed_request)
            result_payload = asyncio.run(
                clinical_service.process_single_patient(
                    patient_payload,
                    section_extraction=section_extraction,
                    session_version=version,
                    original_session_id=root_session_id,
                    session_metadata={
                        **metadata,
                        "revision_mode": True,
                        "focused_selection": bool(selected_focus_text),
                        "revision_instruction": focus_instruction,
                        "model_overrides": effective_overrides,
                        "revised_from_session_id": session_detail.get("session_id"),
                    },
                    original_session_text=source_text,
                    revision_focus_context=revision_focus_context,
                    progress_callback=lambda stage, progress: self.report_job_progress(
                        job_id=job_id or "",
                        progress=progress,
                        message=f"Revision: {stage}",
                    ) if job_id else None,
                    stop_check=None,
                )
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
            if isinstance(persisted_session_id, int):
                self.serializer.upsert_session_result_payload(
                    persisted_session_id,
                    result_payload,
                )
        finally:
            if effective_overrides:
                config_serializer.save_snapshot(
                    clinical_model=previous_snapshot.clinical_model,
                    text_extraction_model=previous_snapshot.text_extraction_model,
                    use_cloud_models=previous_snapshot.use_cloud_models,
                    cloud_provider=previous_snapshot.cloud_provider,
                    cloud_model=previous_snapshot.cloud_model,
                    ollama_temperature=previous_snapshot.ollama_temperature,
                    cloud_temperature=previous_snapshot.cloud_temperature,
                    ollama_reasoning=previous_snapshot.ollama_reasoning,
                )
        return {
            "session_id": result_payload.get("session_id"),
            "version": version,
            "original_session_id": root_session_id,
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
            chunks.append(
                "User revision instruction:\n"
                f"{revision_instruction}"
            )
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
        config = self.load_runtime_config()
        rag_cfg = config.get("rag", {}) if isinstance(config, dict) else {}
        documents_path = self.get_effective_rag_documents_path()
        settings = get_server_settings()
        collection_name = str(
            rag_cfg.get(
                "vector_collection_name", settings.rag.vector_collection_name
            )
        )
        vector_db = LanceVectorDatabase(
            database_path=VECTOR_DB_PATH,
            collection_name=collection_name,
            metric=settings.rag.vector_index_metric,
            index_type=settings.rag.vector_index_type,
            stream_batch_size=settings.rag.vector_stream_batch_size,
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
            "configured_metric": settings.rag.vector_index_metric,
            "configured_index_type": settings.rag.vector_index_type,
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

