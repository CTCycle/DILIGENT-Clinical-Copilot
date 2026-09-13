from __future__ import annotations

from common.embedding.manifest import build_embedding_index_manifest


###############################################################################
def _manifest(*, chunk_size: int | None = None, chunk_overlap: int | None = None):
    return build_embedding_index_manifest(
        generation_id="generation-1",
        collection_name="documents_generation_1",
        documents_path="assets/docs",
        document_count=2,
        chunk_count=4,
        source_manifest_hash="source-hash",
        libraries={"lancedb": "1.0.0"},
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

###############################################################################
def test_manifest_records_effective_non_default_chunking_and_fingerprints_it() -> None:
    default_manifest = _manifest()
    custom_manifest = _manifest(chunk_size=333, chunk_overlap=44)

    assert custom_manifest.chunking == {
        "algorithm": "token_window_v1",
        "target_tokens": 333,
        "overlap_tokens": 44,
    }
    assert custom_manifest.embedding_fingerprint != (
        default_manifest.embedding_fingerprint
    )
