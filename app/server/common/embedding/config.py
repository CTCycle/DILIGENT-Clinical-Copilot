"""Immutable semantic contract for DILIGENT's local RAG embeddings.

Operational concerns such as device and cache location are deliberately kept
outside this module.  Changing any value in this contract changes the vector
space and therefore requires a new index generation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

###############################################################################
@dataclass(frozen=True, slots=True)
class CanonicalEmbeddingConfig:
    schema_version: int
    model_id: str
    revision: str
    runtime_backend: str
    artifact_path: str
    artifact_sha256: str
    quantization: str
    execution_provider: str
    dimension: int
    pooling: str
    normalize: bool
    query_prefix: str
    document_prefix: str
    distance_metric: str
    maximum_model_tokens: int
    maximum_query_tokens: int
    default_chunk_tokens: int
    default_chunk_overlap_tokens: int
    output_dtype: str
    trust_remote_code: bool

    # -------------------------------------------------------------------------
    def to_canonical_dict(self) -> dict[str, Any]:
        """Return the stable, JSON-compatible semantic contract."""

        return asdict(self)

    # -------------------------------------------------------------------------
    def to_canonical_json(self) -> str:
        """Serialize the contract deterministically for hashing and manifests."""

        return json.dumps(
            self.to_canonical_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    # -------------------------------------------------------------------------
    @property
    def fingerprint(self) -> str:
        """Return the SHA-256 identity of this complete semantic contract."""

        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()


CANONICAL_EMBEDDING_CONFIG = CanonicalEmbeddingConfig(
    schema_version=2,
    model_id="ibm-granite/granite-embedding-97m-multilingual-r2",
    revision="c61e626a6255c490879d0af885078b61929d51f6",
    runtime_backend="onnxruntime",
    artifact_path="onnx/model_quint8_avx2.onnx",
    artifact_sha256="a6022dd8220ea6f6595562a1328ee216f4a94faa55362f2f4747c80f1e78772e",
    quantization="uint8-avx2",
    execution_provider="CPUExecutionProvider",
    dimension=384,
    pooling="cls",
    normalize=True,
    query_prefix="",
    document_prefix="",
    distance_metric="cosine",
    maximum_model_tokens=32768,
    maximum_query_tokens=1024,
    default_chunk_tokens=512,
    default_chunk_overlap_tokens=64,
    output_dtype="float32",
    trust_remote_code=False,
)

###############################################################################
def canonical_embedding_fingerprint(
    config: CanonicalEmbeddingConfig = CANONICAL_EMBEDDING_CONFIG,
) -> str:
    """Return a canonical fingerprint without requiring callers to know its fields."""

    return config.fingerprint


__all__ = [
    "CANONICAL_EMBEDDING_CONFIG",
    "CanonicalEmbeddingConfig",
    "canonical_embedding_fingerprint",
]
