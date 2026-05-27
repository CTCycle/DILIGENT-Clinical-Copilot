from __future__ import annotations

from services.retrieval.chunking import SmartDocumentChunker


def test_chunk_ids_and_uid_are_stable() -> None:
    chunker = SmartDocumentChunker(target_chars=40, max_chars=80)
    text = "Heading.\nThis is sentence one. This is sentence two. This is sentence three."
    chunks_a = chunker.chunk_document(
        text=text,
        file_name="doc.txt",
        relative_path="docs/doc.txt",
        content_type="txt",
    )
    chunks_b = chunker.chunk_document(
        text=text,
        file_name="doc.txt",
        relative_path="docs/doc.txt",
        content_type="txt",
    )
    assert chunks_a
    assert chunks_a[0].chunk_index.startswith("doc.txt::p1-1::l")
    assert [c.chunk_uid for c in chunks_a] == [c.chunk_uid for c in chunks_b]


def test_page_and_line_metadata_present() -> None:
    chunker = SmartDocumentChunker(target_chars=30, max_chars=60)
    text = "TITLE\nline2\nline3\nline4\n"
    chunks = chunker.chunk_document(
        text=text,
        file_name="sample.txt",
        relative_path="docs/sample.txt",
        content_type="txt",
        page_texts=["TITLE\nline2\n", "line3\nline4\n"],
    )
    assert len(chunks) >= 2
    assert chunks[0].metadata["page_start"] == 1
    assert chunks[1].metadata["page_start"] == 2
    assert int(chunks[0].metadata["line_start"]) >= 1
    assert int(chunks[0].metadata["line_end"]) >= int(chunks[0].metadata["line_start"])
