from __future__ import annotations

import time
from typing import Any

from services.inspection import update_jobs as update_jobs_module
from services.inspection.service import DataInspectionService
from services.rag.vector_serializer import VectorSerializer
from services.runtime.jobs import JobManager
from repository_fixtures import build_repository_graph

###############################################################################
def test_rag_job_surfaces_incremental_serializer_progress(monkeypatch) -> None:
    captured_documents_path: list[str | None] = []

    ###############################################################################
    class FakeRagEmbeddingUpdater:

        # -------------------------------------------------------------------------
        def __init__(self, **kwargs: Any) -> None:
            captured_documents_path.append(kwargs.get("documents_path"))
            self.documents_path = r"C:\rag"
            self.progress_callback = kwargs["progress_callback"]

        # -------------------------------------------------------------------------
        def prepare_vector_database(self) -> None:
            return None

        # -------------------------------------------------------------------------
        def refresh_embeddings(self) -> dict[str, int]:
            self.progress_callback(52.0, "Embedded and persisted batch 2/4")
            time.sleep(0.05)
            return {
                "documents": 2,
                "chunks": 4,
                "supported_files": 2,
                "loaded_documents": 2,
            }

    graph = build_repository_graph()
    service = DataInspectionService(
        clinical_session_repository=graph.clinical_session_repository,
        drug_catalog_repository=graph.drug_catalog_repository,
        knowledge_repository=graph.knowledge_repository,
        session_timeline_repository=graph.session_timeline_repository,
        session_revision_repository=graph.session_revision_repository,
        jobs=JobManager(),
    )
    monkeypatch.setattr(
        update_jobs_module,
        "RagEmbeddingUpdater",
        FakeRagEmbeddingUpdater,
    )
    monkeypatch.setattr(service, "write_rag_manifest", lambda **_: None)
    monkeypatch.setattr(
        service,
        "get_effective_rag_documents_path",
        lambda: r"C:\persisted\rag",
    )

    payload = service.start_update_job(service.RAG_JOB_TYPE)
    job_id = str(payload["job_id"])

    deadline = time.time() + 2
    observed_messages: list[str] = []
    while time.time() < deadline:
        status = service.jobs.get_job_status(job_id)
        assert status is not None
        result = status.get("result") or {}
        message = result.get("progress_message")
        if isinstance(message, str):
            observed_messages.append(message)
        if status["status"] == "completed":
            break
        time.sleep(0.01)

    assert "Embedded and persisted batch 2/4" in observed_messages
    assert captured_documents_path == [r"C:\persisted\rag"]

###############################################################################
def test_batch_progress_scales_through_embedding_window() -> None:
    events: list[tuple[float, str]] = []
    serializer = object.__new__(VectorSerializer)
    serializer.progress_callback = lambda progress, message: events.append(
        (progress, message)
    )

    serializer.report_batch_progress(completed_batches=1, total_batches=4)
    serializer.report_batch_progress(completed_batches=4, total_batches=4)

    assert events == [
        (44.5, "Embedded and persisted batch 1/4"),
        (88.0, "Embedded and persisted batch 4/4"),
    ]
