from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from common.embedding.config import CANONICAL_EMBEDDING_CONFIG
from services.retrieval.embedding_runtime import (
    EmbeddingRuntime,
    EmbeddingRuntimeUnavailable,
    EmbeddingVectorValidationError,
    REQUIRED_SNAPSHOT_FILES,
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
    assert (
        runtime.status()["cache_status"] == "dependency_missing"
        or runtime.status()["cache_status"] == "available"
    )
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
