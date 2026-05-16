from __future__ import annotations

from pathlib import Path

from services.inspection.service import DataInspectionService


def test_effective_rag_documents_path_prefers_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service = object.__new__(DataInspectionService)
    manifest_path = tmp_path / DataInspectionService.RAG_MANIFEST_FILE_NAME
    manifest_path.write_text(
        '{"documents_path": "C:\\\\external\\\\rag"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(service, "rag_manifest_path", lambda: manifest_path)
    monkeypatch.setattr(
        service,
        "load_runtime_config",
        lambda: {"rag": {"documents_path": "C:\\\\configured\\\\rag"}},
    )

    assert service.get_effective_rag_documents_path() == r"C:\external\rag"


def test_write_rag_manifest_persists_latest_successful_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service = object.__new__(DataInspectionService)
    manifest_path = tmp_path / DataInspectionService.RAG_MANIFEST_FILE_NAME
    monkeypatch.setattr(service, "rag_manifest_path", lambda: manifest_path)

    service.write_rag_manifest(
        documents_path=r"C:\external\rag",
        summary={
            "documents": 99,
            "chunks": 3477,
            "supported_files": 99,
            "loaded_documents": 547,
        },
    )

    payload = service.read_rag_manifest()
    assert payload["documents_path"] == r"C:\external\rag"
    assert payload["documents"] == 99
    assert payload["chunks"] == 3477
