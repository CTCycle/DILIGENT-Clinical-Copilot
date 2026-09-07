from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, Literal

from common.paths import ARCHIVES_PATH
from repositories.drug_catalog_repository import DrugCatalogRepository
from repositories.knowledge_repository import KnowledgeRepository
from services.runtime.jobs import JobManager
from services.updater.embeddings import RagEmbeddingUpdater
from services.updater.livertox_core import LiverToxUpdater
from services.updater.rxnav_builder import RxNavDrugCatalogBuilder
from services.updater.rxnav_client import RxNavClient

UpdateTarget = Literal["rxnav", "livertox", "rag"]

###############################################################################
def _override_float(values: Mapping[str, object], key: str) -> float | None:
    value = values.get(key)
    return float(value) if isinstance(value, int | float) else None

###############################################################################
def _override_int(values: Mapping[str, object], key: str) -> int | None:
    value = values.get(key)
    return int(value) if isinstance(value, int | float) else None

###############################################################################
def _override_str(values: Mapping[str, object], key: str) -> str | None:
    value = values.get(key)
    return value if isinstance(value, str) else None

###############################################################################
def _override_bool(values: Mapping[str, object], key: str) -> bool | None:
    value = values.get(key)
    return value if isinstance(value, bool) else None

###############################################################################
class DataInspectionProgressReporter:

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def __call__(self, progress: float, message: str) -> None:
        self.emit(progress, message)

    # -------------------------------------------------------------------------
    def emit(self, progress: float, message: str) -> None:
        bounded = min(
            100.0, max(0.0, self.base_progress + float(progress) * self.scale)
        )
        self.jobs.update_progress(self.job_id, bounded)
        payload = self.jobs.get_job_status(self.job_id) or {}
        result = dict(payload.get("result") or {})
        result["progress_message"] = message
        self.jobs.update_result(self.job_id, result)

###############################################################################
class DataInspectionUpdateJobRunner:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        drug_catalog_repository: DrugCatalogRepository,
        knowledge_repository: KnowledgeRepository,
        jobs: JobManager,
        report_phase_by_target: Callable[[str, str, int, str], None],
        report_job_progress: Callable[
            [str, float, str, Mapping[str, object] | None], None
        ],
        write_rag_manifest: Callable[[dict[str, Any], str], Path],
    ) -> None:
        self.drug_catalog_repository = drug_catalog_repository
        self.knowledge_repository = knowledge_repository
        self.jobs = jobs
        self.report_phase_by_target = report_phase_by_target
        self.report_job_progress = report_job_progress
        self.write_rag_manifest = write_rag_manifest

    # -------------------------------------------------------------------------
    def run_rxnav_update_job(
        self, job_id: str, overrides: Mapping[str, object] | None = None
    ) -> dict[str, Any]:
        stop_check = partial(self.jobs.should_stop, job_id)
        progress_callback = DataInspectionProgressReporter(
            self.jobs, job_id, 20.0, 0.68
        )
        override_values = dict(overrides or {})
        self.report_phase_by_target(job_id, "rxnav", 1, "Configuration accepted")
        if stop_check():
            return {}
        self.report_phase_by_target(job_id, "rxnav", 4, "RxNav update started")
        self.report_phase_by_target(
            job_id, "rxnav", 10, "Downloading source catalog data"
        )
        rx_client = RxNavClient(
            request_timeout=_override_float(override_values, "rxnav_request_timeout"),
            max_concurrency=_override_int(override_values, "rxnav_max_concurrency"),
        )
        builder = RxNavDrugCatalogBuilder(
            drug_catalog_repository=self.drug_catalog_repository, rx_client=rx_client
        )
        self.report_phase_by_target(
            job_id, "rxnav", 20, "Processing aliases and synonyms"
        )
        result = builder.update_drug_catalog(
            progress_callback=progress_callback, should_stop=stop_check
        )
        self.report_phase_by_target(job_id, "rxnav", 88, "Persisting catalog updates")
        self.report_phase_by_target(job_id, "rxnav", 96, "Finalizing update")
        self.report_phase_by_target(job_id, "rxnav", 100, "Completed")
        return {"summary": result}

    # -------------------------------------------------------------------------
    def run_livertox_update_job(
        self, job_id: str, overrides: Mapping[str, object] | None = None
    ) -> dict[str, Any]:
        stop_check = partial(self.jobs.should_stop, job_id)
        progress_callback = DataInspectionProgressReporter(
            self.jobs, job_id, 20.0, 0.68
        )
        override_values = dict(overrides or {})
        self.report_phase_by_target(job_id, "livertox", 1, "Configuration accepted")
        if stop_check():
            return {}
        self.report_phase_by_target(job_id, "livertox", 4, "LiverTox update started")
        updater = LiverToxUpdater(
            str(ARCHIVES_PATH),
            redownload=bool(_override_bool(override_values, "redownload") or False),
            knowledge_repository=self.knowledge_repository,
            archive_name=_override_str(override_values, "livertox_archive"),
            monograph_max_workers=_override_int(
                override_values, "livertox_monograph_max_workers"
            ),
        )
        self.report_phase_by_target(job_id, "livertox", 10, "Loading source archive")
        result = updater.update_from_livertox(
            progress_callback=progress_callback, should_stop=stop_check
        )
        self.report_phase_by_target(job_id, "livertox", 88, "Persisting extracted data")
        self.report_phase_by_target(job_id, "livertox", 96, "Finalizing update")
        self.report_phase_by_target(job_id, "livertox", 100, "Completed")
        return {"summary": result}

    # -------------------------------------------------------------------------
    def run_rag_update_job(
        self, job_id: str, overrides: Mapping[str, object] | None = None
    ) -> dict[str, Any]:
        stop_check = partial(self.jobs.should_stop, job_id)
        override_values = dict(overrides or {})
        progress_callback = DataInspectionProgressReporter(
            self.jobs, job_id, 30.0, 0.60
        )
        self.report_phase_by_target(job_id, "rag", 1, "Configuration accepted")
        if stop_check():
            return {}
        self.report_phase_by_target(job_id, "rag", 4, "RAG update started")
        updater = RagEmbeddingUpdater(
            documents_path=_override_str(override_values, "documents_path"),
            chunk_size=_override_int(override_values, "chunk_size"),
            chunk_overlap=_override_int(override_values, "chunk_overlap"),
            embedding_batch_size=_override_int(override_values, "embedding_batch_size"),
            vector_stream_batch_size=_override_int(
                override_values, "vector_stream_batch_size"
            ),
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
            raise ValueError(
                "RAG update found zero supported files in the selected folder."
            )
        self._write_rag_manifest(result, updater.documents_path)
        self.report_phase_by_target(
            job_id, "rag", 90, "Persisting embeddings and index"
        )
        self.report_phase_by_target(job_id, "rag", 96, "Finalizing update")
        self.report_phase_by_target(job_id, "rag", 100, "Completed")
        backend = "onnxruntime"
        model_spec = getattr(getattr(updater, "serializer", None), "model_spec", None)
        vector_model = None
        if model_spec is not None:
            provider = str(getattr(model_spec, "provider", "") or "").strip()
            model_name = str(getattr(model_spec, "model_name", "") or "").strip()
            vector_model = (
                f"{provider}:{model_name}"
                if provider and model_name
                else model_name or None
            )
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

    # -------------------------------------------------------------------------
    def _write_rag_manifest(self, report: dict[str, Any], documents_path: str) -> Path:
        return self.write_rag_manifest(report, documents_path)
