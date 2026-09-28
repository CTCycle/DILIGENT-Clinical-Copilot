from __future__ import annotations

import hashlib
from pathlib import Path

from domain.documents import Document
from repositories.serialization.document_chunker import DocumentChunker
from repositories.serialization.document_serializer import DocumentSerializer


###############################################################################
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

###############################################################################
def test_document_serializer_accepts_path_objects_and_collects_relative_ids(tmp_path: Path) -> None:
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    file_path = nested_dir / "study.txt"
    file_path.write_text("TITLE\n\nBody text.", encoding="utf-8")

    serializer = DocumentSerializer(tmp_path)

    assert serializer.collect_document_paths() == [str(file_path)]
    expected_relative = Path("nested") / "study.txt"
    expected_id = hashlib.sha256(str(expected_relative).encode("utf-8")).hexdigest()
    assert serializer.compute_document_id(file_path) == expected_id

###############################################################################
def test_document_serializer_lists_all_files_but_ingests_only_supported_formats(
    tmp_path: Path,
) -> None:
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    supported_path = nested_dir / "study.TXT"
    supported_path.write_text("TITLE\n\nBody text.", encoding="utf-8")
    legacy_word_path = nested_dir / "legacy.doc"
    legacy_word_path.write_bytes(b"legacy binary Word document")
    ignored_path = nested_dir / "image.png"
    ignored_path.write_bytes(b"not a document")
    serializer = DocumentSerializer(tmp_path)

    assert set(serializer.collect_file_paths()) == {
        str(supported_path),
        str(legacy_word_path),
        str(ignored_path),
    }
    assert serializer.collect_document_paths() == [str(supported_path)]
    assert serializer.build_listing_metadata(legacy_word_path)[
        "supported_for_ingestion"
    ] is False
    assert [document.metadata["file_name"] for document in serializer.load_documents()] == [
        "study.TXT"
    ]

###############################################################################
def test_document_serializer_returns_no_documents_for_empty_or_unsupported_folders(
    tmp_path: Path,
) -> None:
    empty_serializer = DocumentSerializer(tmp_path / "empty")
    (tmp_path / "empty").mkdir()

    assert empty_serializer.collect_file_paths() == []
    assert empty_serializer.collect_document_paths() == []
    assert empty_serializer.load_documents() == []

    unsupported_path = tmp_path / "unsupported" / "legacy.doc"
    unsupported_path.parent.mkdir()
    unsupported_path.write_bytes(b"legacy binary Word document")
    (unsupported_path.parent / "image.png").write_bytes(b"not a document")
    unsupported_serializer = DocumentSerializer(unsupported_path.parent)

    assert set(unsupported_serializer.collect_file_paths()) == {
        str(unsupported_path),
        str(unsupported_path.parent / "image.png"),
    }
    assert unsupported_serializer.collect_document_paths() == []
    assert unsupported_serializer.load_documents() == []

###############################################################################
def test_document_serializer_ignores_empty_and_malformed_supported_files(
    tmp_path: Path,
) -> None:
    empty_text = tmp_path / "empty.txt"
    empty_text.write_text("", encoding="utf-8")
    malformed_docx = tmp_path / "malformed.docx"
    malformed_docx.write_bytes(b"not a zip archive")
    serializer = DocumentSerializer(tmp_path)

    assert set(serializer.collect_document_paths()) == {
        str(empty_text),
        str(malformed_docx),
    }
    assert serializer.load_documents() == []

###############################################################################
def test_duplicate_content_keeps_distinct_relative_document_ids(tmp_path: Path) -> None:
    first_path = tmp_path / "first" / "guide.txt"
    second_path = tmp_path / "second" / "guide.txt"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_path.write_text("Identical source content.", encoding="utf-8")
    second_path.write_text("Identical source content.", encoding="utf-8")
    serializer = DocumentSerializer(tmp_path)
    collected_paths = serializer.collect_document_paths()

    assert len(collected_paths) == 2
    assert {Path(path).name for path in collected_paths} == {"guide.txt"}
    assert len({serializer.compute_document_id(path) for path in collected_paths}) == 2

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
