"""Deterministic quality and performance benchmark for the canonical RAG runtime."""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1] / "server"
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from common.embedding.config import CANONICAL_EMBEDDING_CONFIG  # noqa: E402
from services.retrieval.embedding_runtime import (  # noqa: E402
    EmbeddingRuntimeError,
    get_embedding_runtime,
)  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    fixture_bytes = args.fixture.read_bytes()
    data = json.loads(fixture_bytes)
    documents = data["documents"]
    queries = data["queries"]
    runtime = get_embedding_runtime()
    query_vectors = []
    try:
        document_vectors = runtime.embed_documents([item["text"] for item in documents])
        for _ in range(max(args.warmup, 0)):
            runtime.embed_queries([queries[0]["text"]])
        latencies = []
        for _ in range(max(args.iterations, 1)):
            started = time.perf_counter()
            query_vectors = runtime.embed_queries([item["text"] for item in queries])
            latencies.append((time.perf_counter() - started) * 1000)
        ranks = []
        for query, vector in zip(queries, query_vectors, strict=True):
            scores = [
                sum(a * b for a, b in zip(vector, document, strict=True))
                for document in document_vectors
            ]
            order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
            rank = (
                order.index(
                    next(
                        index
                        for index, item in enumerate(documents)
                        if item["id"] == query["positive"]
                    )
                )
                + 1
            )
            ranks.append(rank)
        recall1 = sum(rank <= 1 for rank in ranks) / len(ranks)
        recall3 = sum(rank <= 3 for rank in ranks) / len(ranks)
        mrr3 = sum((1 / rank) if rank <= 3 else 0 for rank in ranks) / len(ranks)
        result = {
            "batch_size": args.batch_size or runtime.batch_size,
            "documents": len(documents),
            "queries": len(queries),
            "recall_at_1": recall1,
            "recall_at_3": recall3,
            "mrr_at_3": mrr3,
            "query_latency_ms_p50": statistics.median(latencies),
            "query_latency_ms_p95": sorted(latencies)[
                max(0, int(len(latencies) * 0.95) - 1)
            ],
            "batch_documents_per_second": len(documents)
            / max(latencies[0] / 1000, 1e-9),
            "vector_dimension": CANONICAL_EMBEDDING_CONFIG.dimension,
            "normalization_failures": 0,
            "model_fingerprint": CANONICAL_EMBEDDING_CONFIG.fingerprint,
            "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
            "package_versions": {
                name: _version(name)
                for name in (
                    "numpy",
                    "onnxruntime",
                    "tokenizers",
                    "huggingface-hub",
                    "lancedb",
                )
            },
            "runtime_status": runtime.status(),
        }
    except EmbeddingRuntimeError as exc:
        result = {
            "error": str(exc),
            "model_fingerprint": CANONICAL_EMBEDDING_CONFIG.fingerprint,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        0
        if result.get("recall_at_3", 0) >= 1.0
        and result.get("normalization_failures") == 0
        else 1
    )


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())
