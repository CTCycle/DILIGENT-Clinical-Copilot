from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, Literal

from common.constants import ARCHIVES_PATH
from repositories.serialization.data import DataSerializer
from services.runtime.jobs import JobManager
from services.updater.embeddings import RagEmbeddingUpdater
from services.updater.livertox_core import LiverToxUpdater
from services.updater.rxnav_builder import RxNavDrugCatalogBuilder
from services.updater.rxnav_client import RxNavClient

UpdateTarget = Literal["rxnav", "livertox", "rag"]


class DataInspectionProgressReporter:
    def __init__(
        self,
        jobs: JobManager,
        job_id: str,
        base_progress: float,
        scale: float,
    ) -> None:
        self.jobs = jobs
        self.job_id = job_id
        self.base_progress = float(base_progress)
        self.scale = float(scale)

    def __call__(self, progress: float, message: str) -> None:
        self.emit(progress, message)

    def emit(self, progress: float, message: str) -> None:
        bounded = min(100.0, max(0.0, self.base_progress + float(progress) * self.scale))
        self.jobs.update_progress(self.job_id, bounded)
        payload = self.jobs.get_job_status(self.job_id) or {}
        result = dict(payload.get("result") or {})
        result["progress_message"] = message
        self.jobs.update_result(self.job_id, result)


class DataInspectionUpdateJobRunner:
    def __init__(
        self,
        *,
        serializer: DataSerializer,
        jobs: JobManager,
        report_phase_by_target: Callable[[str, str, int, str], None],
        report_job_progress: Callable[[str, float, str, Mapping[str, object] | None], None],
        write_rag_manifest: Callable[[dict[str, Any], str], Path],
    ) -> None:
        self.serializer = serializer
        self.jobs = jobs
        self.report_phase_by_target = report_phase_by_target
        self.report_job_progress = report_job_progress
        self.write_rag_manifest = write_rag_manifest

    def run_rxnav_update_job(
        self, job_id: str, overrides: Mapping[str, object] | None = None
    ) -> dict[str, Any]:
        stop_check = partial(self.jobs.should_stop, job_id)
        progress_callback = DataInspectionProgressReporter(self.jobs, job_id, 20.0, 0.68)
        override_values = dict(overrides or {})
        self.report_phase_by_target(job_id, "rxnav", 1, "Configuration accepted")
        if stop_check():
            return {}
        self.report_phase_by_target(job_id, "rxnav", 4, "RxNav update started")
        self.report_phase_by_target(job_id, "rxnav", 10, "Downloading source catalog data")
        rx_client = RxNavClient(
            request_timeout=override_values.get("rxnav_request_timeout"),
            max_concurrency=override_values.get("rxnav_max_concurrency"),
        )
        builder = RxNavDrugCatalogBuilder(serializer=self.serializer, rx_client=rx_client)
        self.report_phase_by_target(job_id, "rxnav", 20, "Processing aliases and synonyms")
        result = builder.update_drug_catalog(progress_callback=progress_callback, should_stop=stop_check)
        self.report_phase_by_target(job_id, "rxnav", 88, "Persisting catalog updates")
        self.report_phase_by_target(job_id, "rxnav", 96, "Finalizing update")
        self.report_phase_by_target(job_id, "rxnav", 100, "Completed")
        return {"summary": result}

    def run_livertox_update_job(
        self, job_id: str, overrides: Mapping[str, object] | None = None
    ) -> dict[str, Any]:
        stop_check = partial(self.jobs.should_stop, job_id)
        progress_callback = DataInspectionProgressReporter(self.jobs, job_id, 20.0, 0.68)
        override_values = dict(overrides or {})
        self.report_phase_by_target(job_id, "livertox", 1, "Configuration accepted")
        if stop_check():
            return {}
        self.report_phase_by_target(job_id, "livertox", 4, "LiverTox update started")
        updater = LiverToxUpdater(
            ARCHIVES_PATH,
            redownload=bool(override_values.get("redownload", False)),
            serializer=self.serializer,
            archive_name=override_values.get("livertox_archive"),
            monograph_max_workers=override_values.get("livertox_monograph_max_workers"),
        )
        self.report_phase_by_target(job_id, "livertox", 10, "Loading source archive")
        result = updater.update_from_livertox(progress_callback=progress_callback, should_stop=stop_check)
        self.report_phase_by_target(job_id, "livertox", 88, "Persisting extracted data")
        self.report_phase_by_target(job_id, "livertox", 96, "Finalizing update")
        self.report_phase_by_target(job_id, "livertox", 100, "Completed")
        return {"summary": result}

    def run_rag_update_job(
        self, job_id: str, overrides: Mapping[str, object] | None = None
    ) -> dict[str, Any]:
        stop_check = partial(self.jobs.should_stop, job_id)
        override_values = dict(overrides or {})
        progress_callback = DataInspectionProgressReporter(self.jobs, job_id, 30.0, 0.60)
        self.report_phase_by_target(job_id, "rag", 1, "Configuration accepted")
        if stop_check():
            return {}
        self.report_phase_by_target(job_id, "rag", 4, "RAG update started")
        updater = RagEmbeddingUpdater(
            documents_path=override_values.get("documents_path"),
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
            progress_callback=progress_callback,
        )
        self.report_phase_by_target(job_id, "rag", 12, "Loading source documents")
        updater.prepare_vector_database()
        if stop_check():
            return {}
        self.report_phase_by_target(job_id, "rag", 30, "Generating embeddings")
        result = updater.refresh_embeddings()
        documents_count = int(result.get("documents", 0) or 0)
        chunks_count = int(result.get("chunks", 0) or 0)
        supported_files = int(result.get("supported_files", 0) or 0)
        if chunks_count <= 0:
            sample_paths = result.get("sample_supported_paths", [])
            sample_details = ""
            if isinstance(sample_paths, list) and sample_paths:
                rendered = ", ".join(str(entry) for entry in sample_paths[:3])
                sample_details = f" Sample files: {rendered}."
            if supported_files > 0:
                raise ValueError(
                    f"RAG update produced zero chunks from {supported_files} supported files. "
                    "Verify document text extraction support and source contents."
                    f"{sample_details}"
                )
            raise ValueError("RAG update found zero supported files in the selected folder.")
        self._write_rag_manifest(result, updater.documents_path)
        self.report_phase_by_target(job_id, "rag", 90, "Persisting embeddings and index")
        self.report_phase_by_target(job_id, "rag", 96, "Finalizing update")
        self.report_phase_by_target(job_id, "rag", 100, "Completed")
        backend = "cloud" if bool(override_values.get("use_cloud_embeddings")) else "local"
        model_spec = getattr(getattr(updater, "serializer", None), "model_spec", None)
        vector_model = None
        if model_spec is not None:
            provider = str(getattr(model_spec, "provider", "") or "").strip()
            model_name = str(getattr(model_spec, "model_name", "") or "").strip()
            vector_model = f"{provider}:{model_name}" if provider and model_name else model_name or None
        return {
            "summary": {
                **result,
                "backend": backend,
                "vector_model": vector_model,
                "documents": documents_count,
                "chunks": chunks_count,
                "supported_files": supported_files,
            }
        }

    def _write_rag_manifest(self, report: dict[str, Any], documents_path: str) -> Path:
        return self.write_rag_manifest(report, documents_path)
