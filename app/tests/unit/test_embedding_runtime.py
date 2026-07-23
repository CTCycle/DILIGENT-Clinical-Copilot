from __future__ import annotations

from pathlib import Path

import pytest

from common.embedding.config import CANONICAL_EMBEDDING_CONFIG
from services.retrieval.embedding_runtime import (
    EmbeddingRuntime,
    EmbeddingRuntimeUnavailable,
    EmbeddingVectorValidationError,
    REQUIRED_SNAPSHOT_FILES,
)


def _complete_snapshot(root: Path) -> Path:
    snapshot = root / CANONICAL_EMBEDDING_CONFIG.revision
    for relative_path in REQUIRED_SNAPSHOT_FILES:
        target = snapshot / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    return snapshot


class _FakeModel:
    max_seq_length = 0

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        self.calls.append({"texts": texts, **kwargs})
        return [[1.0 / 2**0.5, 1.0 / 2**0.5] + [0.0] * 382 for _ in texts]


def test_runtime_is_lazy_and_reuses_one_model(tmp_path: Path) -> None:
    snapshot = _complete_snapshot(tmp_path)
    model = _FakeModel()
    created = 0

    def factory(path: str, **kwargs: object) -> _FakeModel:
        nonlocal created
        assert path == str(snapshot)
        assert kwargs["trust_remote_code"] is False
        created += 1
        return model

    runtime = EmbeddingRuntime(
        cache_directory=tmp_path,
        device="cpu",
        model_factory=factory,
    )
    assert runtime.loaded is False
    assert runtime.status()["cache_status"] == "available"
    assert runtime.embed_documents(["document"])[0][0] > 0
    assert runtime.embed_queries(["query"])[0][0] > 0
    assert created == 1
    assert model.max_seq_length == 8192
    assert [call["texts"] for call in model.calls] == [["document"], ["query"]]


def test_offline_runtime_rejects_incomplete_cache(tmp_path: Path) -> None:
    runtime = EmbeddingRuntime(cache_directory=tmp_path, offline_mode=True, device="cpu")

    with pytest.raises(EmbeddingRuntimeUnavailable, match="incomplete"):
        runtime.embed_documents(["document"])


def test_runtime_rejects_wrong_dimension(tmp_path: Path) -> None:
    snapshot = _complete_snapshot(tmp_path)

    class WrongModel:
        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            _ = texts, kwargs
            return [[1.0]]

    runtime = EmbeddingRuntime(
        cache_directory=tmp_path,
        device="cpu",
        model_factory=lambda path, **kwargs: WrongModel(),
    )
    assert snapshot.is_dir()
    with pytest.raises(EmbeddingVectorValidationError, match="dimension"):
        runtime.embed_documents(["document"])
