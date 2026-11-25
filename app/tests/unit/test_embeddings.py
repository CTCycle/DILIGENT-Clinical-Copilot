# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from common.embedding.manifest import build_embedding_index_manifest
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import services.retrieval.embedding_runtime as embedding_runtime_module
from common.embedding.config import CANONICAL_EMBEDDING_CONFIG
from services.retrieval.embedding_runtime import REQUIRED_SNAPSHOT_FILES, EmbeddingRuntime, EmbeddingRuntimeUnavailable, EmbeddingVectorValidationError


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


###############################################################################
class FakeTokenizer:

    # -------------------------------------------------------------------------
    def encode(self, text: str, *, add_special_tokens: bool = True):
        return SimpleNamespace(ids=[len(text) + 1, 2, 3])

    # -------------------------------------------------------------------------
    def decode(self, ids, *, skip_special_tokens: bool = True) -> str:
        return "decoded " + " ".join(map(str, ids))


###############################################################################
class FakeSession:

    # -------------------------------------------------------------------------
    def __init__(self, output=None, *, token_type_ids: bool = False):
        self.output = (
            output
            if output is not None
            else np.tile(np.eye(384, dtype=np.float32)[None, :3, :], (1, 1, 1))
        )
        self._inputs = [
            SimpleNamespace(name="input_ids", type="tensor(int64)"),
            SimpleNamespace(name="attention_mask", type="tensor(int64)"),
        ]
        if token_type_ids:
            self._inputs.append(
                SimpleNamespace(name="token_type_ids", type="tensor(int64)")
            )
        self._outputs = [SimpleNamespace(name="last_hidden_state")]
        self.calls = []

    # -------------------------------------------------------------------------
    def get_inputs(self):
        return self._inputs

    # -------------------------------------------------------------------------
    def get_outputs(self):
        return self._outputs

    # -------------------------------------------------------------------------
    def get_providers(self):
        return ["CPUExecutionProvider"]

    # -------------------------------------------------------------------------
    def run(self, names, inputs):
        self.calls.append(inputs)
        batch = inputs["input_ids"].shape[0]
        return [np.tile(self.output, (batch, 1, 1))]


###############################################################################
def _runtime(tmp_path: Path, *, session=None, batch_size=2, config=None):
    config = config or CANONICAL_EMBEDDING_CONFIG
    snapshot = tmp_path / config.revision
    artifact = snapshot / config.artifact_path
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"verified artifact")
    for relative in REQUIRED_SNAPSHOT_FILES - {config.artifact_path}:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(b"verified artifact").hexdigest()
    config = replace(config, artifact_sha256=digest)
    session = session or FakeSession()
    return EmbeddingRuntime(
        config=config,
        cache_directory=tmp_path,
        batch_size=batch_size,
        tokenizer_factory=lambda path: FakeTokenizer(),
        session_factory=lambda path, **kwargs: session,
    ), session


###############################################################################
def test_runtime_is_lazy_reuses_session_and_exposes_chunking_adapter(
    tmp_path: Path,
) -> None:
    runtime, session = _runtime(tmp_path)
    assert runtime.loaded is False
    assert runtime.status()["cache_status"] == "available"
    vectors = runtime.embed_documents(["document", "second", "third"])
    assert len(vectors) == 3 and len(vectors[0]) == 384
    assert runtime.loaded is True
    assert runtime.get_tokenizer().encode("hépatotoxicité")
    assert runtime.get_tokenizer().decode([1, 2]) == "decoded 1 2"
    assert len(session.calls) == 2


###############################################################################
def test_offline_runtime_rejects_incomplete_cache(tmp_path: Path) -> None:
    runtime = EmbeddingRuntime(cache_directory=tmp_path, offline_mode=True)
    with pytest.raises(EmbeddingRuntimeUnavailable, match="incomplete"):
        runtime.embed_documents(["document"])


###############################################################################
def test_runtime_rejects_corrupted_artifact(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    artifact = (
        tmp_path
        / CANONICAL_EMBEDDING_CONFIG.revision
        / CANONICAL_EMBEDDING_CONFIG.artifact_path
    )
    artifact.write_bytes(b"corrupt")
    with pytest.raises(EmbeddingRuntimeUnavailable, match="SHA-256"):
        runtime.embed_documents(["document"])


###############################################################################
def test_runtime_rejects_wrong_dimension(tmp_path: Path) -> None:
    output = np.zeros((1, 3, 12), dtype=np.float32)
    runtime, _ = _runtime(tmp_path, session=FakeSession(output))
    with pytest.raises(EmbeddingVectorValidationError, match="dimension"):
        runtime.embed_documents(["document"])


###############################################################################
def test_runtime_close_releases_session_and_can_be_recreated(tmp_path: Path) -> None:
    runtime, _ = _runtime(tmp_path)
    runtime.embed_queries(["query"])
    runtime.close()
    assert runtime.loaded is False
    with pytest.raises(Exception, match="shutting down"):
        runtime.embed_queries(["query"])


###############################################################################
def test_cached_accessor_reuses_and_close_clears_runtime(monkeypatch) -> None:
    created = []


    ###############################################################################
    class FakeRuntime:

        # -------------------------------------------------------------------------
        def __init__(self, **kwargs):
            created.append(self)
            self.closed = False

        # -------------------------------------------------------------------------
        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        embedding_runtime_module,
        "EmbeddingRuntime",
        FakeRuntime,
    )
    monkeypatch.setattr(
        embedding_runtime_module,
        "build_effective_rag_settings",
        lambda: SimpleNamespace(embedding_offline_mode=True, embedding_batch_size=4),
    )
    embedding_runtime_module.get_embedding_runtime.cache_clear()
    try:
        first = embedding_runtime_module.get_embedding_runtime()
        assert embedding_runtime_module.get_embedding_runtime() is first
        assert len(created) == 1

        embedding_runtime_module.close_embedding_runtime()
        assert first.closed is True
        second = embedding_runtime_module.get_embedding_runtime()
        assert second is not first
        assert len(created) == 2
    finally:
        embedding_runtime_module.get_embedding_runtime.cache_clear()


###############################################################################
def test_close_before_first_accessor_use_is_noop(monkeypatch) -> None:
    monkeypatch.setattr(
        embedding_runtime_module,
        "build_effective_rag_settings",
        lambda: pytest.fail("runtime should not be constructed"),
    )
    embedding_runtime_module.get_embedding_runtime.cache_clear()
    embedding_runtime_module.close_embedding_runtime()


###############################################################################
def test_vector_metadata_checks_read_only_projected_columns(tmp_path: Path) -> None:
    import lancedb
    from repositories.vectors import LanceVectorDatabase

    path = str(tmp_path / "vectors")
    lancedb.connect(path).create_table(
        "docs",
        data=[
            {"document_id": "a", "embedding_fingerprint": "fp", "vector_model_signature": "sig", "vector": [0.1, 0.2]},
            {"document_id": "b", "embedding_fingerprint": "fp", "vector_model_signature": "sig", "vector": [0.3, 0.4]},
        ],
    )
    database = LanceVectorDatabase(path, "docs")

    assert database.read_columns(["document_id"]) == [{"document_id": "a"}, {"document_id": "b"}]
    assert database.count_embeddings() == 2
    assert database.count_distinct_documents() == 2
    database.assert_embedding_fingerprint_matches("fp")
    with pytest.raises(ValueError):
        database.assert_embedding_fingerprint_matches("other")
