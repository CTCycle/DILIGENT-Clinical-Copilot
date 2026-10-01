# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import services.inspection.update_jobs as update_jobs_module
from services.inspection.update_jobs import DataInspectionUpdateJobRunner
from services.runtime.jobs import JobManager
from services.runtime.state import JobState
from services.updater.embeddings import RagEmbeddingUpdater


###############################################################################
def build_runner(
    jobs: JobManager,
    manifest_calls: list[tuple[dict[str, Any], str]],
) -> DataInspectionUpdateJobRunner:
    return DataInspectionUpdateJobRunner(
        drug_catalog_repository=object(),  # type: ignore[arg-type]
        knowledge_repository=object(),  # type: ignore[arg-type]
        dilirank_repository=object(),  # type: ignore[arg-type]
        jobs=jobs,
        report_phase_by_target=lambda *_args: None,
        report_job_progress=lambda *_args, **_kwargs: None,
        write_rag_manifest=lambda report, path: (
            manifest_calls.append((report, path))
            or Path(path) / "rag_index_manifest.json"
        ),
    )


###############################################################################
def make_job(jobs: JobManager, job_id: str = "rag-job") -> str:
    jobs.jobs[job_id] = JobState(
        job_id=job_id,
        job_type="rag_update",
        status="running",
        scope_key="catalog:rag_update",
    )
    return job_id


###############################################################################
class _FakeRagEmbeddingUpdater:
    result: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    def __init__(self, documents_path: str | None = None, **_: object) -> None:
        self.documents_path = str(documents_path or "")

    # -------------------------------------------------------------------------
    def prepare_vector_database(self) -> None:
        return None

    # -------------------------------------------------------------------------
    def refresh_embeddings(self) -> dict[str, Any]:
        return dict(self.result)


###############################################################################
@pytest.mark.parametrize(
    ("summary", "expected_message"),
    [
        (
            {
                "documents": 0,
                "chunks": 0,
                "supported_files": 0,
                "loaded_documents": 0,
                "sample_supported_paths": [],
            },
            "RAG update found zero supported files",
        ),
        (
            {
                "documents": 0,
                "chunks": 0,
                "supported_files": 2,
                "loaded_documents": 0,
                "sample_supported_paths": ["C:/rag/empty.txt"],
            },
            "RAG update produced zero chunks from 2 supported files",
        ),
    ],
)
def test_rag_update_fails_closed_without_replacing_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    summary: dict[str, Any],
    expected_message: str,
) -> None:
    jobs = JobManager()
    job_id = make_job(jobs)
    manifest_calls: list[tuple[dict[str, Any], str]] = []
    runner = build_runner(jobs, manifest_calls)
    _FakeRagEmbeddingUpdater.result = summary
    monkeypatch.setattr(
        update_jobs_module,
        "RagEmbeddingUpdater",
        _FakeRagEmbeddingUpdater,
    )

    with pytest.raises(ValueError, match=expected_message):
        runner.run_rag_update_job(job_id, {"documents_path": str(tmp_path)})

    assert manifest_calls == []


###############################################################################
def test_rag_update_writes_manifest_only_after_chunks_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs = JobManager()
    job_id = make_job(jobs)
    manifest_calls: list[tuple[dict[str, Any], str]] = []
    runner = build_runner(jobs, manifest_calls)
    _FakeRagEmbeddingUpdater.result = {
        "documents": 1,
        "chunks": 2,
        "supported_files": 1,
        "physical_supported_files": 2,
        "unique_supported_files": 1,
        "unique_ingested_documents": 1,
        "duplicate_file_count": 1,
        "loaded_documents": 1,
        "source_manifest_hash": "duplicate-fixture-hash",
        "sample_supported_paths": ["C:/rag/guide.txt"],
    }
    monkeypatch.setattr(
        update_jobs_module,
        "RagEmbeddingUpdater",
        _FakeRagEmbeddingUpdater,
    )

    result = runner.run_rag_update_job(job_id, {"documents_path": str(tmp_path)})

    assert result["summary"]["chunks"] == 2
    assert len(manifest_calls) == 1
    manifest_summary, manifest_path = manifest_calls[0]
    assert manifest_summary["chunks"] == 2
    assert manifest_summary["physical_supported_files"] == 2
    assert manifest_summary["unique_ingested_documents"] == 1
    assert manifest_summary["duplicate_file_count"] == 1
    assert manifest_summary["source_manifest_hash"] == "duplicate-fixture-hash"
    assert "backend" not in manifest_summary
    assert manifest_path == str(tmp_path)


###############################################################################
@pytest.mark.parametrize(
    "path_factory",
    [
        lambda tmp_path: tmp_path / "missing",
        lambda tmp_path: tmp_path / "file.txt",
    ],
)
def test_rag_updater_rejects_missing_or_non_directory_paths(
    tmp_path: Path,
    path_factory: Any,
) -> None:
    path = path_factory(tmp_path)
    if path.name == "file.txt":
        path.write_text("not a folder", encoding="utf-8")

    with pytest.raises(ValueError, match="does not exist or is not a directory"):
        RagEmbeddingUpdater(documents_path=path)
