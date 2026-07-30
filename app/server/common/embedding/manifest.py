"""Versioned metadata and strict compatibility checks for RAG indexes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from common.embedding.config import (
    CANONICAL_EMBEDDING_CONFIG,
    CanonicalEmbeddingConfig,
)
from common.paths import RAG_ACTIVE_GENERATION_POINTER_PATH

###############################################################################
@dataclass(frozen=True, slots=True)
class EmbeddingIndexManifest:
    manifest_version: int
    status: str
    generation_id: str
    collection_name: str
    embedding_fingerprint: str
    model: dict[str, Any]
    tokenizer: dict[str, Any]
    chunking: dict[str, Any]
    libraries: dict[str, str]
    source: dict[str, Any]
    built_at: str

    # -------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "status": self.status,
            "generation_id": self.generation_id,
            "collection_name": self.collection_name,
            "embedding_schema_version": self.model["schema_version"],
            "embedding_fingerprint": self.embedding_fingerprint,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "chunking": self.chunking,
            "libraries": self.libraries,
            "source": self.source,
            "built_at": self.built_at,
        }

###############################################################################
def build_embedding_index_manifest(
    *,
    generation_id: str,
    collection_name: str,
    documents_path: str,
    document_count: int,
    chunk_count: int,
    source_manifest_hash: str,
    libraries: dict[str, str],
    config: CanonicalEmbeddingConfig = CANONICAL_EMBEDDING_CONFIG,
) -> EmbeddingIndexManifest:
    model = config.to_canonical_dict()
    tokenizer = {
        "model_revision": config.revision,
        "truncation": True,
        "padding": "longest",
    }
    chunking = {
        "algorithm": "token_window_v1",
        "target_tokens": config.default_chunk_tokens,
        "overlap_tokens": config.default_chunk_overlap_tokens,
    }
    fingerprint_payload = {
        "model": model,
        "tokenizer": tokenizer,
        "chunking": chunking,
        "libraries": dict(sorted(libraries.items())),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return EmbeddingIndexManifest(
        manifest_version=2,
        status="ready",
        generation_id=generation_id,
        collection_name=collection_name,
        embedding_fingerprint=fingerprint,
        model={**model, "schema_version": config.schema_version},
        tokenizer=tokenizer,
        chunking=chunking,
        libraries=dict(sorted(libraries.items())),
        source={
            "documents_path": documents_path,
            "document_count": document_count,
            "chunk_count": chunk_count,
            "source_manifest_hash": source_manifest_hash,
        },
        built_at=datetime.now(UTC).isoformat(),
    )

###############################################################################
def assert_manifest_compatible(
    manifest: dict[str, Any],
    *,
    expected_fingerprint: str,
    expected_dimension: int = CANONICAL_EMBEDDING_CONFIG.dimension,
) -> None:
    if manifest.get("manifest_version") != 2:
        raise ValueError("RAG index manifest is legacy or unsupported")
    if manifest.get("status") != "ready":
        raise ValueError("RAG index manifest is not ready")
    if manifest.get("embedding_fingerprint") != expected_fingerprint:
        raise ValueError("RAG index embedding fingerprint is incompatible")
    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("dimension") != expected_dimension:
        raise ValueError("RAG index vector dimension is incompatible")
    if model.get("distance_metric") != "cosine":
        raise ValueError("RAG index distance metric is incompatible")

###############################################################################
def read_active_collection_name(default: str) -> str:
    try:
        payload = json.loads(
            RAG_ACTIVE_GENERATION_POINTER_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return default
    if not isinstance(payload, dict):
        return default
    collection_name = str(payload.get("collection_name") or "").strip()
    return collection_name or default


__all__ = [
    "EmbeddingIndexManifest",
    "assert_manifest_compatible",
    "build_embedding_index_manifest",
    "read_active_collection_name",
]
