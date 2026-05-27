from __future__ import annotations

import time
from typing import Any

from services.inspection import update_jobs as update_jobs_module
from services.inspection.service import DataInspectionService
from services.runtime.jobs import JobManager
from repositories.serialization.data import DataSerializer


def test_rag_job_surfaces_incremental_serializer_progress(monkeypatch) -> None:
    class FakeRagEmbeddingUpdater:
        def __init__(self, **kwargs: Any) -> None:
            self.documents_path = r"C:\rag"
            self.progress_callback = kwargs["progress_callback"]

        def prepare_vector_database(self) -> None:
            return None

        def refresh_embeddings(self) -> dict[str, int]:
            self.progress_callback(52.0, "Embedded and persisted batch 2/4")
            time.sleep(0.05)
            return {
                "documents": 2,
                "chunks": 4,
                "supported_files": 2,
                "loaded_documents": 2,
            }

    service = DataInspectionService(serializer=DataSerializer(), jobs=JobManager())
    monkeypatch.setattr(
        update_jobs_module,
        "RagEmbeddingUpdater",
        FakeRagEmbeddingUpdater,
    )
    monkeypatch.setattr(service, "write_rag_manifest", lambda **_: None)

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
