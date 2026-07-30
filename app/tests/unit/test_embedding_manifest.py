from __future__ import annotations

from common.embedding.config import CANONICAL_EMBEDDING_CONFIG
from common.embedding.manifest import (
    assert_manifest_compatible,
    build_embedding_index_manifest,
)

###############################################################################
def test_manifest_contains_strict_canonical_metadata() -> None:
    manifest = build_embedding_index_manifest(
        generation_id="generation-1",
        collection_name="documents__generation-1",
        documents_path="C:/documents",
        document_count=2,
        chunk_count=4,
        source_manifest_hash="source-hash",
        libraries={"torch": "2.13.0", "sentence_transformers": "5.6.0"},
    ).to_dict()

    assert manifest["manifest_version"] == 2
    assert manifest["status"] == "ready"
    assert manifest["model"]["revision"] == CANONICAL_EMBEDDING_CONFIG.revision
    assert manifest["chunking"]["target_tokens"] == 512
    assert manifest["embedding_fingerprint"] != CANONICAL_EMBEDDING_CONFIG.fingerprint

###############################################################################
def test_mismatched_manifest_is_rejected() -> None:
    manifest = build_embedding_index_manifest(
        generation_id="generation-1",
        collection_name="documents__generation-1",
        documents_path="C:/documents",
        document_count=0,
        chunk_count=0,
        source_manifest_hash="source-hash",
        libraries={},
    ).to_dict()

    try:
        assert_manifest_compatible(manifest, expected_fingerprint="wrong")
    except ValueError as exc:
        assert "fingerprint" in str(exc)
    else:
        raise AssertionError("incompatible manifest must be rejected")
