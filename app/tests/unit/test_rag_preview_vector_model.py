from __future__ import annotations

from services.inspection.service import DataInspectionService
from services.runtime.jobs import JobManager


def test_rag_preview_includes_vector_model(monkeypatch) -> None:
    service = DataInspectionService(jobs=JobManager())

    monkeypatch.setattr(
        "services.inspection.service.DocumentSerializer.collect_document_paths",
        lambda self: [r"C:\docs\doc1.txt"],
    )
    monkeypatch.setattr(
        service,
        "get_effective_rag_documents_path",
        lambda: r"C:\docs",
    )

    class FakeVectorDb:
        def __init__(self, **kwargs):
            _ = kwargs

        def has_collection(self) -> bool:
            return True

        def load_embeddings(self):
            return [
                {
                    "file_name": "doc1.txt",
                    "vector_model_provider": "ollama",
                    "vector_model_name": "nomic-embed-text",
                }
            ]

    monkeypatch.setattr("services.inspection.service.LanceVectorDatabase", FakeVectorDb)
    payload = service.list_rag_documents(search=None, offset=0, limit=10)
    assert payload["items"][0]["vector_model"] == "ollama:nomic-embed-text"
