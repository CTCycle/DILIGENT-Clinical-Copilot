from __future__ import annotations
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from common.constants import DOCUMENT_SUPPORTED_EXTENSIONS
from common.paths import VECTOR_DB_PATH
from common.embedding.manifest import read_active_collection_name
from common.embedding.config import CANONICAL_EMBEDDING_CONFIG
from common.utils.logger import logger
from configurations.startup import get_server_settings
from domain.inspection import InspectionJobPhase
from repositories.serialization.data import DataSerializer
from repositories.serialization.document_serializer import DocumentSerializer
from repositories.vectors import LanceVectorDatabase
from services.retrieval.settings import build_effective_rag_settings
from services.clinical.timeline import PatientTimelineExtractor
from services.inspection.normalization import (
    extract_lab_marker as extract_lab_marker_value,
)
from services.inspection.normalization import (
    first_iso_date as first_iso_date_value,
)
from services.inspection.normalization import (
    normalize_text as normalize_text_value,
)
from services.inspection.timeline import InspectionTimelineMixin
from services.inspection.update_jobs import DataInspectionUpdateJobRunner
from services.inspection.update_config import InspectionUpdateConfigMixin
from services.inspection.revision_scaffold import InspectionRevisionScaffoldMixin
from services.inspection.revision_agent import RevisionAgentRunner
from services.runtime.jobs import JobManager

PhaseStep = tuple[InspectionJobPhase, int, int, str]
UpdateTarget = Literal["rxnav", "livertox", "rag"]

###############################################################################
class DataInspectionService(
    InspectionUpdateConfigMixin,
    InspectionRevisionScaffoldMixin,
    InspectionTimelineMixin,
):
    RXNAV_JOB_TYPE = "rxnav_update"
    LIVERTOX_JOB_TYPE = "livertox_update"
    RAG_JOB_TYPE = "rag_update"
    REVISION_JOB_TYPE = "session_revision"
    SESSION_TIMELINE_JOB_TYPE = "session_timeline"
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
        self.revision_agent_runner = RevisionAgentRunner(serializer=self.serializer)

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
    def delete_session(self, session_id: int) -> bool:
        return self.serializer.delete_session(session_id)

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
                database_path=str(VECTOR_DB_PATH),
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
        collection_name = read_active_collection_name(
            str(rag_settings.vector_collection_name)
        )
        vector_db = LanceVectorDatabase(
            database_path=str(VECTOR_DB_PATH),
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
            "embedding_model": CANONICAL_EMBEDDING_CONFIG.model_id,
            "embedding_revision": CANONICAL_EMBEDDING_CONFIG.revision,
            "index_status": str(
                self.read_rag_manifest().get("status") or "reindex_required"
            ),
            "embedding_fingerprint": self.read_rag_manifest().get(
                "embedding_fingerprint"
            ),
            "built_at": self.read_rag_manifest().get("built_at"),
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
    def run_session_timeline_job(
        self,
        session_id: int,
        *,
        force_regenerate: bool,
        model_overrides: Any = None,
        job_id: str,
    ) -> dict[str, Any]:
        def report_progress(progress: float, message: str) -> None:
            self.jobs.update_progress(job_id, progress)
            self.jobs.update_result(
                job_id,
                {"session_id": int(session_id), "progress_message": message},
            )

        timeline = self.generate_session_timeline(
            session_id,
            force_regenerate=force_regenerate,
            model_overrides=model_overrides,
            progress_callback=report_progress,
        )
        if timeline is None:
            raise RuntimeError("Session not found.")
        return {
            "session_id": int(session_id),
            "timeline_id": timeline.timeline_id,
            "progress_message": "Timeline saved.",
        }

    # -------------------------------------------------------------------------
    def start_session_timeline_job(
        self,
        session_id: int,
        *,
        force_regenerate: bool = False,
        model_overrides: Any = None,
    ) -> dict[str, Any]:
        safe_session_id = int(session_id)
        if self.serializer.get_session_timeline_source(safe_session_id) is None:
            raise KeyError(safe_session_id)
        scope_key = f"session_timeline:{safe_session_id}"
        if self.jobs.is_job_running(self.SESSION_TIMELINE_JOB_TYPE, scope_key=scope_key):
            raise ValueError("Timeline regeneration is already in progress for this session.")
        runner = partial(
            self.run_session_timeline_job,
            safe_session_id,
            force_regenerate=force_regenerate,
            model_overrides=model_overrides,
        )
        job_id = self.jobs.start_job(
            job_type=self.SESSION_TIMELINE_JOB_TYPE,
            runner=runner,
            scope_key=scope_key,
        )
        payload = self.jobs.get_job_status(job_id)
        if payload is None:
            raise RuntimeError("Failed to initialize session timeline job")
        payload["poll_interval"] = get_server_settings().jobs.polling_interval
        return payload

    # -------------------------------------------------------------------------
    def get_session_timeline_job_status(
        self, session_id: int, job_id: str
    ) -> dict[str, Any] | None:
        payload = self.get_job_status(job_id, expected_type=self.SESSION_TIMELINE_JOB_TYPE)
        if payload is None:
            return None
        result = payload.get("result")
        if isinstance(result, dict) and result.get("session_id") is not None:
            if int(result["session_id"]) != int(session_id):
                return None
        return payload

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
    def list_update_jobs(self) -> list[dict[str, Any]]:
        supported_types = {
            self.RXNAV_JOB_TYPE,
            self.LIVERTOX_JOB_TYPE,
            self.RAG_JOB_TYPE,
        }
        latest_by_type: dict[str, dict[str, Any]] = {}
        for payload in self.jobs.list_jobs():
            job_type = str(payload.get("job_type") or "")
            if job_type not in supported_types:
                continue
            current = latest_by_type.get(job_type)
            incoming_key = (
                float(payload.get("created_at") or 0),
                int(payload.get("version") or 0),
            )
            current_key = (
                float(current.get("created_at") or 0),
                int(current.get("version") or 0),
            ) if current else (-1.0, -1)
            if incoming_key > current_key:
                latest_by_type[job_type] = payload
        return list(latest_by_type.values())

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str, *, expected_type: str) -> bool:
        payload = self.get_job_status(job_id, expected_type=expected_type)
        if payload is None:
            return False
        return self.jobs.cancel_job(job_id) is not None
