from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, Literal

from common.paths import ARCHIVES_PATH
from repositories.dilirank_repository import DiliRankRepository
from repositories.drug_catalog_repository import DrugCatalogRepository
from repositories.knowledge_repository import KnowledgeRepository
from services.runtime.jobs import JobManager
from services.updater.dilirank import DiliRankUpdater
from services.updater.embeddings import RagEmbeddingUpdater
from services.updater.livertox_core import LiverToxUpdater
from services.updater.rxnav_builder import RxNavDrugCatalogBuilder
from services.updater.rxnav_client import RxNavClient

UpdateTarget = Literal["rxnav", "livertox", "dilirank", "rag"]

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
        dilirank_repository: DiliRankRepository | None,
        jobs: JobManager,
        report_phase_by_target: Callable[[str, str, int, str], None],
        report_job_progress: Callable[
            [str, float, str, Mapping[str, object] | None], None
        ],
        write_rag_manifest: Callable[[dict[str, Any], str], Path],
    ) -> None:
        self.drug_catalog_repository = drug_catalog_repository
        self.knowledge_repository = knowledge_repository
        self.dilirank_repository = dilirank_repository
        self.jobs = jobs
        self.report_phase_by_target = report_phase_by_target
        self.report_job_progress = report_job_progress
        self.write_rag_manifest = write_rag_manifest

    # -------------------------------------------------------------------------
    def run_rxnav_update_job(
        self,
        job_id: str,
        overrides: Mapping[str, object] | None = None,
        *,
        progress_callback: Callable[[float, str], None] | None = None,
        report_phase: bool = True,
    ) -> dict[str, Any]:
        stop_check = partial(self.jobs.should_stop, job_id)
        progress_callback = progress_callback or DataInspectionProgressReporter(
            self.jobs, job_id, 20.0, 0.68
        )
        phase_reporter: Callable[[str, str, int, str], None] = (
            self.report_phase_by_target
            if report_phase
            else lambda *_args: None
        )
        override_values = dict(overrides or {})
        phase_reporter(job_id, "rxnav", 1, "Configuration accepted")
        if stop_check():
            return {}
        phase_reporter(job_id, "rxnav", 4, "RxNav update started")
        phase_reporter(
            job_id, "rxnav", 10, "Downloading source catalog data"
        )
        rx_client = RxNavClient(
            request_timeout=_override_float(override_values, "rxnav_request_timeout"),
            max_concurrency=_override_int(override_values, "rxnav_max_concurrency"),
        )
        builder = RxNavDrugCatalogBuilder(
            drug_catalog_repository=self.drug_catalog_repository, rx_client=rx_client
        )
        phase_reporter(
            job_id, "rxnav", 20, "Processing aliases and synonyms"
        )
        result = builder.update_drug_catalog(
            progress_callback=progress_callback, should_stop=stop_check
        )
        phase_reporter(job_id, "rxnav", 88, "Persisting catalog updates")
        phase_reporter(job_id, "rxnav", 96, "Finalizing update")
        phase_reporter(job_id, "rxnav", 100, "Completed")
        return {"summary": result}

    # -------------------------------------------------------------------------
    def run_livertox_update_job(
        self,
        job_id: str,
        overrides: Mapping[str, object] | None = None,
        *,
        progress_callback: Callable[[float, str], None] | None = None,
        report_phase: bool = True,
    ) -> dict[str, Any]:
        stop_check = partial(self.jobs.should_stop, job_id)
        progress_callback = progress_callback or DataInspectionProgressReporter(
            self.jobs, job_id, 20.0, 0.68
        )
        phase_reporter: Callable[[str, str, int, str], None] = (
            self.report_phase_by_target
            if report_phase
            else lambda *_args: None
        )
        override_values = dict(overrides or {})
        phase_reporter(job_id, "livertox", 1, "Configuration accepted")
        if stop_check():
            return {}
        phase_reporter(job_id, "livertox", 4, "LiverTox update started")
        updater = LiverToxUpdater(
            str(ARCHIVES_PATH),
            redownload=bool(_override_bool(override_values, "redownload") or False),
            knowledge_repository=self.knowledge_repository,
            archive_name=_override_str(override_values, "livertox_archive"),
            monograph_max_workers=_override_int(
                override_values, "livertox_monograph_max_workers"
            ),
        )
        phase_reporter(job_id, "livertox", 10, "Loading source archive")
        result = updater.update_from_livertox(
            progress_callback=progress_callback, should_stop=stop_check
        )
        phase_reporter(job_id, "livertox", 88, "Persisting extracted data")
        phase_reporter(job_id, "livertox", 96, "Finalizing update")
        phase_reporter(job_id, "livertox", 100, "Completed")
        return {"summary": result}

    # -------------------------------------------------------------------------
    def run_dilirank_update_job(
        self,
        job_id: str,
        overrides: Mapping[str, object] | None = None,
        *,
        progress_callback: Callable[[float, str], None] | None = None,
        report_phase: bool = True,
    ) -> dict[str, Any]:
        if self.dilirank_repository is None:
            raise RuntimeError("DILIrank repository is unavailable.")
        stop_check = partial(self.jobs.should_stop, job_id)
        progress_callback = progress_callback or DataInspectionProgressReporter(
            self.jobs, job_id, 10.0, 0.80
        )
        phase_reporter: Callable[[str, str, int, str], None] = (
            self.report_phase_by_target
            if report_phase
            else lambda *_args: None
        )
        override_values = dict(overrides or {})
        phase_reporter(job_id, "dilirank", 1, "Configuration accepted")
        if stop_check():
            return {}
        phase_reporter(job_id, "dilirank", 4, "DILIrank update started")
        phase_reporter(
            job_id, "dilirank", 10, "Loading FDA DILIrank 2.0 source"
        )
        updater = DiliRankUpdater(
            repository=self.dilirank_repository,
            redownload=bool(_override_bool(override_values, "redownload") or False),
        )
        phase_reporter(
            job_id, "dilirank", 20, "Validating and linking DILIrank records"
        )
        result = updater.update_from_fda(
            progress_callback=progress_callback,
            should_stop=stop_check,
        )
        phase_reporter(
            job_id, "dilirank", 90, "Persisting DILIrank snapshot"
        )
        phase_reporter(job_id, "dilirank", 96, "Finalizing update")
        phase_reporter(job_id, "dilirank", 100, "Completed")
        return {"summary": result}

    # -------------------------------------------------------------------------
    def run_structured_sources_update_job(
        self, job_id: str, overrides: Mapping[str, object] | None = None
    ) -> dict[str, Any]:
        """Run the dependent structured sources in one cancellable job."""

        targets = ("rxnav", "livertox", "dilirank")
        override_values = dict(overrides or {})
        source_state: dict[str, dict[str, Any]] = {
            target: {
                "status": "pending",
                "progress": 0.0,
                "message": "Waiting to start.",
                "error": None,
            }
            for target in targets
        }
        summaries: dict[str, Any] = {}

        def patch_job(progress: float, message: str) -> None:
            self.jobs.update_progress(job_id, progress)
            self.jobs.update_result(
                job_id,
                {
                    "sources": {
                        target: dict(payload)
                        for target, payload in source_state.items()
                    },
                    "summaries": dict(summaries),
                    "progress_message": message,
                },
            )

        def child_overrides(target: str) -> dict[str, object]:
            payload = override_values.get(target)
            return dict(payload) if isinstance(payload, Mapping) else {}

        def mark_cancelled(start_index: int) -> None:
            for target in targets[start_index:]:
                if source_state[target]["status"] == "pending":
                    source_state[target].update(
                        status="cancelled",
                        message="Cancellation requested.",
                    )

        patch_job(0.0, "Structured source updates queued.")
        for index, target in enumerate(targets):
            if self.jobs.should_stop(job_id):
                mark_cancelled(index)
                patch_job(
                    (index / len(targets)) * 100.0,
                    "Cancellation requested for the structured source updates.",
                )
                return {"sources": source_state, "summaries": summaries}

            source_state[target].update(
                status="running",
                message=f"{target.capitalize()} update running.",
            )
            patch_job(
                (index / len(targets)) * 100.0,
                f"Running {target.capitalize()} update.",
            )

            def report_child_progress(progress: float, message: str) -> None:
                bounded = min(100.0, max(0.0, float(progress)))
                source_state[target].update(
                    progress=bounded,
                    message=message,
                )
                patch_job(((index + bounded / 100.0) / len(targets)) * 100.0, message)

            try:
                if target == "rxnav":
                    child_result = self.run_rxnav_update_job(
                        job_id,
                        child_overrides(target),
                        progress_callback=report_child_progress,
                        report_phase=False,
                    )
                elif target == "livertox":
                    child_result = self.run_livertox_update_job(
                        job_id,
                        child_overrides(target),
                        progress_callback=report_child_progress,
                        report_phase=False,
                    )
                else:
                    child_result = self.run_dilirank_update_job(
                        job_id,
                        child_overrides(target),
                        progress_callback=report_child_progress,
                        report_phase=False,
                    )
            except Exception:
                source_state[target].update(
                    status="failed",
                    message=f"{target.capitalize()} update failed.",
                    error=f"{target.capitalize()} update failed.",
                )
                for remaining in targets[index + 1 :]:
                    source_state[remaining].update(
                        status="failed",
                        message="Skipped because an earlier source update failed.",
                        error="Skipped because an earlier source update failed.",
                    )
                patch_job(
                    ((index + source_state[target]["progress"] / 100.0) / len(targets))
                    * 100.0,
                    f"{target.capitalize()} update failed.",
                )
                raise

            if self.jobs.should_stop(job_id):
                source_state[target].update(
                    status="cancelled",
                    message="Cancellation requested.",
                )
                mark_cancelled(index + 1)
                patch_job(
                    ((index + source_state[target]["progress"] / 100.0) / len(targets))
                    * 100.0,
                    "Cancellation requested for the structured source updates.",
                )
                return {"sources": source_state, "summaries": summaries}

            source_state[target].update(
                status="completed",
                progress=100.0,
                message=f"{target.capitalize()} update completed.",
            )
            if isinstance(child_result.get("summary"), Mapping):
                summaries[target] = dict(child_result["summary"])
            patch_job(
                ((index + 1) / len(targets)) * 100.0,
                f"{target.capitalize()} update completed.",
            )

        return {"sources": source_state, "summaries": summaries}

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
