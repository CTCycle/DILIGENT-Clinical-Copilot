from __future__ import annotations

import pytest

from common.embedding.config import CANONICAL_EMBEDDING_CONFIG
from repositories.vectors import LanceVectorDatabase


def test_former_english_fingerprint_is_rejected(monkeypatch) -> None:
    database = object.__new__(LanceVectorDatabase)
    monkeypatch.setattr(database, "has_collection", lambda: True)
    monkeypatch.setattr(
        database,
        "load_embeddings",
        lambda: [{"embedding_fingerprint": "former-english-model-fingerprint"}],
    )
    with pytest.raises(ValueError, match="incompatible embedding fingerprints"):
        database.assert_embedding_fingerprint_matches(CANONICAL_EMBEDDING_CONFIG.fingerprint)
