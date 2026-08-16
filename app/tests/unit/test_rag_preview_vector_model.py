from __future__ import annotations

from services.inspection.service import DataInspectionService
from services.runtime.jobs import JobManager
from repository_fixtures import build_repository_graph

###############################################################################
def test_rag_preview_includes_vector_model(monkeypatch, tmp_path) -> None:
    graph = build_repository_graph()
    service = DataInspectionService(
        clinical_session_repository=graph.clinical_session_repository,
        drug_catalog_repository=graph.drug_catalog_repository,
        knowledge_repository=graph.knowledge_repository,
        session_timeline_repository=graph.session_timeline_repository,
        session_revision_repository=graph.session_revision_repository,
        jobs=JobManager(),
    )

    documents_path = tmp_path / "docs"
    document_path = documents_path / "doc1.txt"
    monkeypatch.setattr(
        "services.inspection.service.DocumentSerializer.collect_document_paths",
        lambda self: [str(document_path)],
    )
    monkeypatch.setattr(
        service,
        "get_effective_rag_documents_path",
        lambda: str(documents_path),
    )

    ###############################################################################
    class FakeVectorDb:

        # -------------------------------------------------------------------------
        def __init__(self, **kwargs):
            _ = kwargs

        # -------------------------------------------------------------------------
        def has_collection(self) -> bool:
            return True

        # -------------------------------------------------------------------------
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
