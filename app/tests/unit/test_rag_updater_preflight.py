from __future__ import annotations

import pytest

from services.updater.embeddings import RagEmbeddingUpdater


class _FakeVectorDatabase:
    def __init__(self) -> None:
        self.clear_calls = 0
        self.initialize_calls = 0
        self.get_table_calls = 0

    def clear_collection(self) -> None:
        self.clear_calls += 1

    def initialize(self) -> None:
        self.initialize_calls += 1

    def get_table(self) -> None:
        self.get_table_calls += 1


def _build_updater() -> tuple[RagEmbeddingUpdater, _FakeVectorDatabase]:
    updater = object.__new__(RagEmbeddingUpdater)
    updater.reset_vector_collection = True
    fake_db = _FakeVectorDatabase()
    updater.vector_database = fake_db
    return updater, fake_db


def test_prepare_vector_database_skips_clear_when_preflight_fails() -> None:
    updater, fake_db = _build_updater()

    def _fail_preflight() -> None:
        raise RuntimeError("embedding backend unavailable")

    updater.validate_embedding_backend = _fail_preflight  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="embedding backend unavailable"):
        updater.prepare_vector_database()

    assert fake_db.clear_calls == 0
    assert fake_db.initialize_calls == 0
    assert fake_db.get_table_calls == 0


def test_prepare_vector_database_clears_after_preflight_success() -> None:
    updater, fake_db = _build_updater()
    updater.validate_embedding_backend = lambda: None  # type: ignore[method-assign]

    updater.prepare_vector_database()

    assert fake_db.clear_calls == 1
    assert fake_db.initialize_calls == 1
    assert fake_db.get_table_calls == 1
