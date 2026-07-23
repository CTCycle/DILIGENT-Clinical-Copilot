from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("DILIGENT_RUN_EMBEDDING_INTEGRATION") != "1",
    reason="set DILIGENT_RUN_EMBEDDING_INTEGRATION=1 to run the pinned model integration",
)


###############################################################################
def test_pinned_multilingual_runtime_contract() -> None:
    assert importlib.util.find_spec("torch") is None
    assert importlib.util.find_spec("sentence_transformers") is None
    from common.embedding.config import CANONICAL_EMBEDDING_CONFIG
    from services.retrieval.embedding_runtime import get_embedding_runtime

    fixture = Path(__file__).parents[1] / "fixtures" / "rag_multilingual_retrieval.json"
    import json

    data = json.loads(fixture.read_text(encoding="utf-8"))
    runtime = get_embedding_runtime()
    documents = data["documents"]
    vectors = runtime.embed_documents([item["text"] for item in documents])
    assert all(
        len(vector) == CANONICAL_EMBEDDING_CONFIG.dimension for vector in vectors
    )
    assert all(
        abs(sum(value * value for value in vector) - 1.0) < 1e-3 for vector in vectors
    )
    queries = data["queries"]
    query_vectors = runtime.embed_queries([item["text"] for item in queries])
    scores = [
        [
            sum(a * b for a, b in zip(query, document, strict=True))
            for document in vectors
        ]
        for query in query_vectors
    ]
    ids = [item["id"] for item in documents]
    for query, row in zip(queries, scores, strict=True):
        assert (
            ids.index(query["positive"])
            in sorted(range(len(row)), key=row.__getitem__, reverse=True)[:3]
        )
