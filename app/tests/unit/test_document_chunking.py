from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import uuid

from domain.documents import Document
from repositories.serialization.document_chunker import DocumentChunker
from repositories.serialization.document_serializer import DocumentSerializer


TEST_TMP_ROOT = Path("QA/tmp/test-document-chunking")

###############################################################################
def make_workspace_temp_dir() -> Path:
    temp_dir = TEST_TMP_ROOT / uuid.uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir

###############################################################################
def test_textual_document_metadata_uses_heading_title_fallback() -> None:
    tmp_path = make_workspace_temp_dir()
    try:
        file_path = tmp_path / "study.txt"
        file_path.write_text("HEPATOTOXICITY OVERVIEW\n\nBody text.", encoding="utf-8")
        serializer = DocumentSerializer(str(tmp_path))

        documents = serializer.load_textual_file(str(file_path), ".txt")

        assert len(documents) == 1
        metadata = documents[0].metadata
        assert metadata["file_name"] == "study.txt"
        assert metadata["document_title"] == "HEPATOTOXICITY OVERVIEW"
        assert metadata["content_type"] == "txt"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

###############################################################################
def test_document_serializer_accepts_path_objects_and_collects_relative_ids() -> None:
    tmp_path = make_workspace_temp_dir()
    try:
        nested_dir = tmp_path / "nested"
        nested_dir.mkdir()
        file_path = nested_dir / "study.txt"
        file_path.write_text("TITLE\n\nBody text.", encoding="utf-8")

        serializer = DocumentSerializer(tmp_path)

        assert serializer.collect_document_paths() == [str(file_path)]
        expected_relative = Path("nested") / "study.txt"
        expected_id = hashlib.sha256(str(expected_relative).encode("utf-8")).hexdigest()
        assert serializer.compute_document_id(file_path) == expected_id
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

###############################################################################
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
