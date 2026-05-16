from __future__ import annotations

from pathlib import Path

from domain.documents import Document
from repositories.serialization.data import DocumentChunker, DocumentSerializer


def test_textual_document_metadata_uses_heading_title_fallback(tmp_path: Path) -> None:
    file_path = tmp_path / "study.txt"
    file_path.write_text("HEPATOTOXICITY OVERVIEW\n\nBody text.", encoding="utf-8")
    serializer = DocumentSerializer(str(tmp_path))

    documents = serializer.load_textual_file(str(file_path), ".txt")

    assert len(documents) == 1
    metadata = documents[0].metadata
    assert metadata["file_name"] == "study.txt"
    assert metadata["document_title"] == "HEPATOTOXICITY OVERVIEW"
    assert metadata["content_type"] == "txt"


def test_structure_aware_chunking_preserves_heading_metadata() -> None:
    chunker = DocumentChunker(chunk_size=40, chunk_overlap=5)
    document = Document(
        page_content=(
            "INTRODUCTION\n\n"
            "Short opening paragraph.\n\n"
            "METHODS\n\n"
            "This paragraph is intentionally longer than the configured chunk size."
        ),
        metadata={"document_id": "doc-1", "file_name": "study.txt"},
    )

    chunks = chunker.chunk_documents([document])

    assert chunks
    assert chunks[0].metadata["section_title"] == "INTRODUCTION"
    assert chunks[-1].metadata["section_title"] == "METHODS"
    assert chunks[-1].metadata["heading_path"] == "METHODS"
    assert all("chunk_index" in chunk.metadata for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)
