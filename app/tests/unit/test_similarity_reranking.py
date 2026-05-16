from __future__ import annotations

from typing import Any

from services.retrieval.embeddings import SimilaritySearch


###############################################################################
class SearchTableStub:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.limit_value: int | None = None
        self.last_query_type: str | None = None

    # -------------------------------------------------------------------------
    def search(
        self,
        query: list[float] | str,
        *,
        query_type: str | None = None,
    ) -> SearchTableStub:
        _ = query
        self.last_query_type = query_type
        return self

    # -------------------------------------------------------------------------
    def limit(self, limit: int) -> SearchTableStub:
        self.limit_value = int(limit)
        return self

    # -------------------------------------------------------------------------
    def to_list(self) -> list[dict[str, Any]]:
        if self.limit_value is None:
            return list(self.rows)
        return list(self.rows[: self.limit_value])


###############################################################################
class VectorDatabaseStub:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.table = SearchTableStub(rows)

    # -------------------------------------------------------------------------
    def initialize(self) -> None:
        return None

    # -------------------------------------------------------------------------
    def get_table(self) -> SearchTableStub:
        return self.table


###############################################################################
class EmbeddingGeneratorStub:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    # -------------------------------------------------------------------------
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


###############################################################################
###############################################################################
class CrossEncoderStub:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        lookup = {"alpha": 0.1, "beta": 0.8, "gamma": 0.9}
        return [lookup[text] for _, text in pairs]


###############################################################################
class FailingCrossEncoderStub:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        _ = pairs
        raise RuntimeError("synthetic rerank failure")


# -----------------------------------------------------------------------------
def sample_rows() -> list[dict[str, Any]]:
    return [
        {
            "document_id": "doc-a",
            "chunk_id": "a1",
            "text": "alpha",
            "source": "test",
            "metadata": {},
            "_distance": 0.01,
        },
        {
            "document_id": "doc-b",
            "chunk_id": "b1",
            "text": "beta",
            "source": "test",
            "metadata": {},
            "_distance": 0.02,
        },
        {
            "document_id": "doc-c",
            "chunk_id": "c1",
            "text": "gamma",
            "source": "test",
            "metadata": {},
            "_distance": 0.03,
        },
    ]


# -----------------------------------------------------------------------------
def sample_vectors() -> dict[str, list[float]]:
    return {
        "q": [1.0, 0.0],
        "alpha": [0.2, 0.98],
        "beta": [0.98, 0.2],
        "gamma": [0.8, 0.1],
    }


# -----------------------------------------------------------------------------
def test_search_with_reranking_reorders_and_trims_results() -> None:
    search = SimilaritySearch(
        vector_database=VectorDatabaseStub(sample_rows()),
        embedding_generator=EmbeddingGeneratorStub(sample_vectors()),
        default_top_k=3,
    )
    search.reranker = CrossEncoderStub()  # type: ignore[assignment]

    results = search.search_with_reranking(
        "q",
        candidate_k=3,
        final_top_n=2,
        use_reranking=True,
    )

    assert [item.get("text") for item in results] == ["gamma", "beta"]
    assert len(results) == 2
    assert all("rerank_score" in item for item in results)


# -----------------------------------------------------------------------------
def test_search_with_reranking_skips_reorder_when_disabled() -> None:
    search = SimilaritySearch(
        vector_database=VectorDatabaseStub(sample_rows()),
        embedding_generator=EmbeddingGeneratorStub(sample_vectors()),
        default_top_k=3,
    )

    results = search.search_with_reranking(
        "q",
        candidate_k=3,
        final_top_n=2,
        use_reranking=False,
    )

    assert [item.get("text") for item in results] == ["alpha", "beta"]
    assert all("rerank_score" not in item for item in results)


# -----------------------------------------------------------------------------
def test_search_with_reranking_enforces_candidate_floor_against_top_n() -> None:
    vector_db = VectorDatabaseStub(sample_rows())
    search = SimilaritySearch(
        vector_database=vector_db,
        embedding_generator=EmbeddingGeneratorStub(sample_vectors()),
        default_top_k=1,
    )

    results = search.search_with_reranking(
        "q",
        candidate_k=1,
        final_top_n=3,
        use_reranking=False,
    )

    assert len(results) == 3
    assert vector_db.table.limit_value == 3


# -----------------------------------------------------------------------------
def test_search_with_reranking_falls_back_when_reranker_fails() -> None:
    search = SimilaritySearch(
        vector_database=VectorDatabaseStub(sample_rows()),
        embedding_generator=EmbeddingGeneratorStub(sample_vectors()),
        default_top_k=3,
    )
    search.reranker = FailingCrossEncoderStub()  # type: ignore[assignment]

    results = search.search_with_reranking(
        "q",
        candidate_k=3,
        final_top_n=2,
        use_reranking=True,
    )

    assert [item.get("text") for item in results] == ["alpha", "beta"]
    assert all("rerank_score" not in item for item in results)

