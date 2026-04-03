from __future__ import annotations

from datetime import date, datetime, UTC
from functools import partial
from pathlib import Path
from typing import Any, Literal

from DILIGENT.server.common.constants import (
    ARCHIVES_PATH,
    CONFIGURATIONS_FILE,
    DOCS_PATH,
    DOCUMENT_SUPPORTED_EXTENSIONS,
    VECTOR_DB_PATH,
)
from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.configurations import load_configuration_data, server_settings
from DILIGENT.server.domain.inspection import InspectionJobPhase
from DILIGENT.server.repositories.serialization.data import DataSerializer, DocumentSerializer
from DILIGENT.server.repositories.vectors import LanceVectorDatabase
from DILIGENT.server.services.jobs import JobManager, job_manager
from DILIGENT.server.services.updater.embeddings import RagEmbeddingUpdater
from DILIGENT.server.services.updater.livertox import LiverToxUpdater
from DILIGENT.server.services.updater.rxnav import RxNavClient, RxNavDrugCatalogBuilder

PhaseStep = tuple[InspectionJobPhase, int, int, str]
UpdateTarget = Literal["rxnav", "livertox", "rag"]


###############################################################################
class DataInspectionProgressReporter:
    def __init__(self, *, service: "DataInspectionService", job_id: str) -> None:
        self.service = service
        self.job_id = job_id

    # -------------------------------------------------------------------------
    def __call__(self, progress: float, message: str) -> None:
        self.service.report_job_progress(
            job_id=self.job_id,
            progress=progress,
            message=message,
        )


###############################################################################
class DataInspectionService:
    RXNAV_JOB_TYPE = "rxnav_update"
    LIVERTOX_JOB_TYPE = "livertox_update"
    RAG_JOB_TYPE = "rag_update"
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
        jobs: JobManager = job_manager,
    ) -> None:
        self.serializer = serializer or DataSerializer()
        self.jobs = jobs

    # -------------------------------------------------------------------------
    def load_runtime_config(self) -> dict[str, Any]:
        payload = load_configuration_data(CONFIGURATIONS_FILE)
        return payload if isinstance(payload, dict) else {}

    # -------------------------------------------------------------------------
    def build_update_config_response(self, target: UpdateTarget) -> dict[str, Any]:
        config = self.load_runtime_config()
        if target == "rxnav":
            source = config.get("external_data", {})
            defaults = {
                "rxnav_request_timeout": float(
                    source.get(
                        "rxnav_request_timeout",
                        server_settings.external_data.rxnav_request_timeout,
                    )
                ),
                "rxnav_max_concurrency": int(
                    source.get(
                        "rxnav_max_concurrency",
                        server_settings.external_data.rxnav_max_concurrency,
                    )
                ),
            }
            allowed_fields = list(defaults.keys())
        elif target == "livertox":
            source = config.get("external_data", {})
            defaults = {
                "livertox_monograph_max_workers": int(
                    source.get(
                        "livertox_monograph_max_workers",
                        server_settings.external_data.livertox_monograph_max_workers,
                    )
                ),
                "livertox_archive": str(
                    source.get("livertox_archive", server_settings.external_data.livertox_archive)
                ),
                "redownload": False,
            }
            allowed_fields = list(defaults.keys())
        else:
            source = config.get("rag", {})
            defaults = {
                "chunk_size": int(source.get("chunk_size", server_settings.rag.chunk_size)),
                "chunk_overlap": int(
                    source.get("chunk_overlap", server_settings.rag.chunk_overlap)
                ),
                "embedding_batch_size": int(
                    source.get("embedding_batch_size", server_settings.rag.embedding_batch_size)
                ),
                "vector_stream_batch_size": int(
                    source.get(
                        "vector_stream_batch_size",
                        server_settings.rag.vector_stream_batch_size,
                    )
                ),
                "embedding_max_workers": int(
                    source.get("embedding_max_workers", server_settings.rag.embedding_max_workers)
                ),
                "embedding_backend": str(
                    source.get("embedding_backend", server_settings.rag.embedding_backend)
                ),
                "ollama_embedding_model": str(
                    source.get(
                        "ollama_embedding_model",
                        server_settings.rag.ollama_embedding_model,
                    )
                ),
                "hf_embedding_model": str(
                    source.get("hf_embedding_model", server_settings.rag.hf_embedding_model)
                ),
                "cloud_provider": str(
                    source.get("cloud_provider", server_settings.rag.cloud_provider)
                ),
                "cloud_embedding_model": str(
                    source.get(
                        "cloud_embedding_model",
                        server_settings.rag.cloud_embedding_model,
                    )
                ),
                "use_cloud_embeddings": bool(
                    source.get("use_cloud_embeddings", server_settings.rag.use_cloud_embeddings)
                ),
                "reset_vector_collection": bool(
                    source.get(
                        "reset_vector_collection",
                        server_settings.rag.reset_vector_collection,
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
    def get_session_report(self, session_id: int) -> str | None:
        return self.serializer.get_session_report(session_id)

    # -------------------------------------------------------------------------
    def delete_session(self, session_id: int) -> bool:
        return self.serializer.delete_session(session_id)

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
    def list_rag_documents(self) -> dict[str, Any]:
        serializer = DocumentSerializer(DOCS_PATH)
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
        return {"items": items, "total": len(items)}

    # -------------------------------------------------------------------------
    def get_rag_vector_store_summary(self) -> dict[str, Any]:
        config = self.load_runtime_config()
        rag_cfg = config.get("rag", {}) if isinstance(config, dict) else {}
        collection_name = str(
            rag_cfg.get("vector_collection_name", server_settings.rag.vector_collection_name)
        )
        vector_db = LanceVectorDatabase(
            database_path=VECTOR_DB_PATH,
            collection_name=collection_name,
            metric=server_settings.rag.vector_index_metric,
            index_type=server_settings.rag.vector_index_type,
            stream_batch_size=server_settings.rag.vector_stream_batch_size,
        )
        exists = vector_db.has_collection()
        if exists:
            try:
                vector_db.get_table()
                vector_db.ensure_vector_index()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Unable to load LanceDB inspection summary: %s", exc)
        return {
            "vector_db_path": VECTOR_DB_PATH,
            "collection_name": collection_name,
            "collection_exists": exists,
            "embedding_count": vector_db.count_embeddings() if exists else 0,
            "distinct_document_count": vector_db.count_distinct_documents() if exists else 0,
            "embedding_dimension": vector_db.read_embedding_dimension() if exists else None,
            "index_ready": bool(vector_db.index_ready) if exists else False,
            "configured_metric": server_settings.rag.vector_index_metric,
            "configured_index_type": server_settings.rag.vector_index_type,
        }

    # -------------------------------------------------------------------------
    def patch_job_result(self, *, job_id: str, patch: dict[str, Any]) -> None:
        self.jobs.update_result(job_id, patch)

    # -------------------------------------------------------------------------
    def report_job_progress(self, *, job_id: str, progress: float, message: str) -> None:
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
            self.report_job_progress(job_id=job_id, progress=progress, message=fallback_message)
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
    def run_rxnav_update_job(
        self, job_id: str, overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        stop_check = partial(self.jobs.should_stop, job_id)
        progress_callback = DataInspectionProgressReporter(service=self, job_id=job_id)
        override_values = overrides or {}
        self.report_phase_by_target(
            job_id=job_id,
            target="rxnav",
            phase="configuration_accepted",
            progress=1.0,
            fallback_message="Configuration accepted",
        )
        if stop_check():
            return {}
        self.report_phase_by_target(
            job_id=job_id,
            target="rxnav",
            phase="update_started",
            progress=4.0,
            fallback_message="RxNav update started",
        )
        self.report_phase_by_target(
            job_id=job_id,
            target="rxnav",
            phase="source_data_loading",
            progress=10.0,
            fallback_message="Downloading source catalog data",
        )
        rx_client = RxNavClient(
            request_timeout=override_values.get("rxnav_request_timeout"),
            max_concurrency=override_values.get("rxnav_max_concurrency"),
        )
        builder = RxNavDrugCatalogBuilder(
            serializer=self.serializer,
            rx_client=rx_client,
        )
        self.report_phase_by_target(
            job_id=job_id,
            target="rxnav",
            phase="processing_extraction",
            progress=20.0,
            fallback_message="Processing aliases and synonyms",
        )
        result = builder.update_drug_catalog(
            progress_callback=progress_callback,
            should_stop=stop_check,
        )
        self.report_phase_by_target(
            job_id=job_id,
            target="rxnav",
            phase="persistence_indexing",
            progress=88.0,
            fallback_message="Persisting catalog updates",
        )
        self.report_phase_by_target(
            job_id=job_id,
            target="rxnav",
            phase="finalization",
            progress=96.0,
            fallback_message="Finalizing update",
        )
        self.report_phase_by_target(
            job_id=job_id,
            target="rxnav",
            phase="completed",
            progress=100.0,
            fallback_message="Completed",
        )
        return {"summary": result}

    # -------------------------------------------------------------------------
    def run_livertox_update_job(
        self, job_id: str, overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        stop_check = partial(self.jobs.should_stop, job_id)
        progress_callback = DataInspectionProgressReporter(service=self, job_id=job_id)
        override_values = overrides or {}
        self.report_phase_by_target(
            job_id=job_id,
            target="livertox",
            phase="configuration_accepted",
            progress=1.0,
            fallback_message="Configuration accepted",
        )
        if stop_check():
            return {}
        self.report_phase_by_target(
            job_id=job_id,
            target="livertox",
            phase="update_started",
            progress=4.0,
            fallback_message="LiverTox update started",
        )
        updater = LiverToxUpdater(
            ARCHIVES_PATH,
            redownload=bool(override_values.get("redownload", False)),
            serializer=self.serializer,
            archive_name=override_values.get("livertox_archive"),
            monograph_max_workers=override_values.get("livertox_monograph_max_workers"),
        )
        self.report_phase_by_target(
            job_id=job_id,
            target="livertox",
            phase="source_data_loading",
            progress=10.0,
            fallback_message="Loading source archive",
        )
        result = updater.update_from_livertox(
            progress_callback=progress_callback,
            should_stop=stop_check,
        )
        self.report_phase_by_target(
            job_id=job_id,
            target="livertox",
            phase="persistence_indexing",
            progress=88.0,
            fallback_message="Persisting extracted data",
        )
        self.report_phase_by_target(
            job_id=job_id,
            target="livertox",
            phase="finalization",
            progress=96.0,
            fallback_message="Finalizing update",
        )
        self.report_phase_by_target(
            job_id=job_id,
            target="livertox",
            phase="completed",
            progress=100.0,
            fallback_message="Completed",
        )
        return {"summary": result}

    # -------------------------------------------------------------------------
    def run_rag_update_job(
        self, job_id: str, overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        stop_check = partial(self.jobs.should_stop, job_id)
        override_values = overrides or {}
        self.report_phase_by_target(
            job_id=job_id,
            target="rag",
            phase="configuration_accepted",
            progress=1.0,
            fallback_message="Configuration accepted",
        )
        if stop_check():
            return {}
        self.report_phase_by_target(
            job_id=job_id,
            target="rag",
            phase="update_started",
            progress=4.0,
            fallback_message="RAG update started",
        )
        updater = RagEmbeddingUpdater(
            use_cloud_embeddings=override_values.get("use_cloud_embeddings"),
            cloud_provider=override_values.get("cloud_provider"),
            cloud_embedding_model=override_values.get("cloud_embedding_model"),
            chunk_size=override_values.get("chunk_size"),
            chunk_overlap=override_values.get("chunk_overlap"),
            embedding_batch_size=override_values.get("embedding_batch_size"),
            vector_stream_batch_size=override_values.get("vector_stream_batch_size"),
            embedding_max_workers=override_values.get("embedding_max_workers"),
            embedding_backend=override_values.get("embedding_backend"),
            ollama_embedding_model=override_values.get("ollama_embedding_model"),
            hf_embedding_model=override_values.get("hf_embedding_model"),
            reset_vector_collection=override_values.get("reset_vector_collection"),
        )
        self.report_phase_by_target(
            job_id=job_id,
            target="rag",
            phase="source_data_loading",
            progress=12.0,
            fallback_message="Loading source documents",
        )
        updater.prepare_vector_database()
        if stop_check():
            return {}
        self.report_phase_by_target(
            job_id=job_id,
            target="rag",
            phase="processing_extraction",
            progress=30.0,
            fallback_message="Generating embeddings",
        )
        result = updater.refresh_embeddings()
        self.report_phase_by_target(
            job_id=job_id,
            target="rag",
            phase="persistence_indexing",
            progress=90.0,
            fallback_message="Persisting embeddings and index",
        )
        self.report_phase_by_target(
            job_id=job_id,
            target="rag",
            phase="finalization",
            progress=96.0,
            fallback_message="Finalizing update",
        )
        self.report_phase_by_target(
            job_id=job_id,
            target="rag",
            phase="completed",
            progress=100.0,
            fallback_message="Completed",
        )
        backend = "cloud" if bool(override_values.get("use_cloud_embeddings")) else "local"
        result_with_backend = {**result, "backend": backend}
        return {"summary": result_with_backend}

    # -------------------------------------------------------------------------
    def start_update_job(self, job_type: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
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
        return status_payload

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str, *, expected_type: str) -> dict[str, Any] | None:
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
        return self.jobs.cancel_job(job_id)
