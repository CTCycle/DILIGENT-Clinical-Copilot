from __future__ import annotations

from services.retrieval.embedding_model import build_embedding_model_signature
from services.retrieval.embeddings import EmbeddingModelMismatchError, SimilaritySearch


def test_embedding_model_signature_is_deterministic() -> None:
    a = build_embedding_model_signature("ollama", "nomic-embed-text", 768, "local")
    b = build_embedding_model_signature("ollama", "nomic-embed-text", 768, "local")
    assert a == b


def test_similarity_search_raises_on_model_mismatch() -> None:
    class VectorDb:
        def initialize(self) -> None:
            return None

        def get_table(self):
            return None

        def assert_query_model_matches_index(self, active_signature: str) -> None:
            _ = active_signature
            raise ValueError("mismatch")

    class Emb:
        def resolve_active_embedding_model_spec(self):
            return type("Spec", (), {"signature": "active"})()

        def embed_texts(self, texts):
            _ = texts
            return [[0.1, 0.2]]

    search = SimilaritySearch(vector_database=VectorDb(), embedding_generator=Emb())  # type: ignore[arg-type]
    try:
        search.search("query", top_k=1)
        assert False, "expected mismatch error"
    except EmbeddingModelMismatchError:
        assert True
