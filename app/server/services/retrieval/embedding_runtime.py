"""Lazy, strict ONNX Runtime for the canonical RAG embedding model."""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy
import onnxruntime
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from common.embedding.config import CANONICAL_EMBEDDING_CONFIG, CanonicalEmbeddingConfig
from common.paths import EMBEDDING_MODELS_PATH
from services.retrieval.settings import build_effective_rag_settings

REQUIRED_SNAPSHOT_FILES = frozenset(
    {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        CANONICAL_EMBEDDING_CONFIG.artifact_path,
    }
)


###############################################################################
class EmbeddingRuntimeError(RuntimeError):
    """Base error for unavailable or invalid embedding runtime state."""


###############################################################################
class EmbeddingRuntimeUnavailable(EmbeddingRuntimeError):
    """Raised when dependencies, the snapshot, or the ONNX contract is invalid."""


###############################################################################
class EmbeddingVectorValidationError(EmbeddingRuntimeError):
    """Raised when inference violates the canonical vector contract."""


SnapshotDownloader = Callable[..., str | Path]
SessionFactory = Callable[..., Any]
TokenizerFactory = Callable[..., Any]


###############################################################################
class _ChunkingTokenizer:
    # -------------------------------------------------------------------------
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    # -------------------------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        encoded = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        ids = getattr(encoded, "ids", encoded)
        return [int(value) for value in ids]

    # -------------------------------------------------------------------------
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return str(
            self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)
        )


###############################################################################
class EmbeddingRuntime:
    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        config: CanonicalEmbeddingConfig = CANONICAL_EMBEDDING_CONFIG,
        cache_directory: Path = EMBEDDING_MODELS_PATH,
        offline_mode: bool = False,
        batch_size: int = 64,
        snapshot_downloader: SnapshotDownloader | None = None,
        session_factory: SessionFactory | None = None,
        tokenizer_factory: TokenizerFactory | None = None,
    ) -> None:
        self.config = config
        self.cache_directory = Path(cache_directory)
        self.offline_mode = bool(offline_mode)
        self.batch_size = max(int(batch_size), 1)
        self._snapshot_downloader = snapshot_downloader
        self._session_factory = session_factory or _default_session_factory
        self._tokenizer_factory = tokenizer_factory or _default_tokenizer_factory
        self._session: Any | None = None
        self._inference_tokenizer: Any | None = None
        self._chunking_tokenizer: _ChunkingTokenizer | None = None
        self._verified_artifact: tuple[str, int, int] | None = None
        self._lock = threading.RLock()
        self._closed = False

    # -------------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._session is not None

    # -------------------------------------------------------------------------
    def status(self) -> dict[str, object]:
        snapshot = self._cached_snapshot_path()
        cache_status = "missing"
        if self._has_required_files(snapshot):
            try:
                self._verify_artifact(snapshot / self.config.artifact_path)
                cache_status = "available"
            except EmbeddingRuntimeUnavailable:
                cache_status = "invalid"
        return {
            "model_id": self.config.model_id,
            "model_revision": self.config.revision,
            "backend": self.config.runtime_backend,
            "artifact_path": self.config.artifact_path,
            "execution_provider": self.config.execution_provider,
            "cache_status": cache_status,
            "loaded": self.loaded,
        }

    # -------------------------------------------------------------------------
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(
            texts, self.config.document_prefix, self.config.maximum_model_tokens
        )

    # -------------------------------------------------------------------------
    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(
            texts, self.config.query_prefix, self.config.maximum_query_tokens
        )

    # -------------------------------------------------------------------------
    def get_tokenizer(self) -> Any:
        with self._lock:
            self._ensure_loaded()
            assert self._chunking_tokenizer is not None
            return self._chunking_tokenizer

    # -------------------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._session = None
            self._inference_tokenizer = None
            self._chunking_tokenizer = None

    # -------------------------------------------------------------------------
    def _encode(
        self, texts: Sequence[str], prefix: str, limit: int
    ) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            if self._closed:
                raise EmbeddingRuntimeError("Embedding runtime is shutting down")
            session, tokenizer = self._ensure_loaded()
            prepared = [f"{prefix}{text or ' '}" for text in texts]
            all_vectors: list[list[float]] = []
            for start in range(0, len(prepared), self.batch_size):
                batch = prepared[start : start + self.batch_size]
                encodings = [
                    tokenizer.encode(text, add_special_tokens=True) for text in batch
                ]
                ids = [list(getattr(item, "ids", item))[:limit] for item in encodings]
                masks = [[1] * len(row) for row in ids]
                width = max((len(row) for row in ids), default=1)
                input_ids = [[*row, *([0] * (width - len(row)))] for row in ids]
                attention = [[*row, *([0] * (width - len(row)))] for row in masks]
                inputs = {
                    "input_ids": numpy.asarray(input_ids, dtype=numpy.int64),
                    "attention_mask": numpy.asarray(attention, dtype=numpy.int64),
                }
                declared = {str(item.name): item for item in session.get_inputs()}
                if "token_type_ids" in declared:
                    inputs["token_type_ids"] = numpy.zeros_like(
                        inputs["input_ids"], dtype=numpy.int64
                    )
                unknown = [name for name in declared if name not in inputs]
                if unknown:
                    raise EmbeddingRuntimeUnavailable(
                        f"ONNX model declares unsupported mandatory inputs: {unknown}"
                    )
                outputs = session.run(None, inputs)
                hidden = self._select_hidden(session, outputs, numpy)
                if (
                    getattr(hidden, "ndim", None) != 3
                    or hidden.shape[-1] != self.config.dimension
                ):
                    raise EmbeddingVectorValidationError(
                        "ONNX output must be rank-3 with dimension 384"
                    )
                vectors = hidden[:, 0, :].astype(numpy.float32)
                if self.config.normalize:
                    norms = numpy.linalg.norm(vectors, axis=1, keepdims=True)
                    if numpy.any(norms <= 0) or not numpy.all(numpy.isfinite(norms)):
                        raise EmbeddingVectorValidationError(
                            "Embedding contains a zero or invalid norm"
                        )
                    vectors = vectors / norms
                all_vectors.extend(self._validate_vectors(vectors.tolist(), len(batch)))
            return all_vectors

    # -------------------------------------------------------------------------
    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._session is not None and self._inference_tokenizer is not None:
            return self._session, self._inference_tokenizer
        snapshot = self._resolve_snapshot()
        artifact = snapshot / self.config.artifact_path
        self._verify_artifact(artifact)
        tokenizer_factory = self._tokenizer_factory
        session_factory = self._session_factory
        assert tokenizer_factory is not None
        assert session_factory is not None
        try:
            tokenizer = tokenizer_factory(str(snapshot / "tokenizer.json"))
            session = session_factory(
                str(artifact), providers=[self.config.execution_provider]
            )
        except EmbeddingRuntimeError:
            raise
        except Exception as exc:
            raise EmbeddingRuntimeUnavailable(
                f"Failed to load multilingual Granite ONNX runtime: {exc}"
            ) from exc
        self._validate_session(session)
        self._inference_tokenizer = tokenizer
        self._chunking_tokenizer = _ChunkingTokenizer(tokenizer)
        self._session = session
        return session, tokenizer

    # -------------------------------------------------------------------------
    def _resolve_snapshot(self) -> Path:
        cached = self._cached_snapshot_path()
        if self._has_required_files(cached):
            return cached
        if self.offline_mode:
            raise EmbeddingRuntimeUnavailable(
                "Canonical embedding cache is incomplete in offline mode"
            )
        downloader = self._snapshot_downloader or _default_snapshot_downloader
        try:
            result = downloader(
                repo_id=self.config.model_id,
                revision=self.config.revision,
                local_dir=str(cached),
                allow_patterns=sorted(REQUIRED_SNAPSHOT_FILES),
                local_files_only=False,
            )
        except Exception as exc:
            raise EmbeddingRuntimeUnavailable(
                f"Unable to download embedding snapshot at revision {self.config.revision}"
            ) from exc
        resolved = Path(result) if result else cached
        if resolved != cached and self._has_required_files(resolved):
            return resolved
        if not self._has_required_files(cached):
            raise EmbeddingRuntimeUnavailable(
                "Downloaded embedding snapshot is incomplete"
            )
        return cached

    # -------------------------------------------------------------------------
    def _cached_snapshot_path(self) -> Path:
        return self.cache_directory / self.config.revision

    # -------------------------------------------------------------------------
    @staticmethod
    def _has_required_files(snapshot: Path) -> bool:
        return snapshot.is_dir() and all(
            (snapshot / item).is_file() for item in REQUIRED_SNAPSHOT_FILES
        )

    # -------------------------------------------------------------------------
    def _verify_artifact(self, artifact: Path) -> None:
        stat = artifact.stat()
        marker = (str(artifact), stat.st_size, stat.st_mtime_ns)
        if self._verified_artifact == marker:
            return
        digest = hashlib.sha256()
        with artifact.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != self.config.artifact_sha256:
            raise EmbeddingRuntimeUnavailable(
                "Canonical ONNX artifact SHA-256 does not match the pinned digest"
            )
        self._verified_artifact = marker

    # -------------------------------------------------------------------------
    @staticmethod
    def _select_hidden(session: Any, outputs: Sequence[Any], numpy: Any) -> Any:
        metadata = list(session.get_outputs())
        for index, item in enumerate(metadata):
            if str(getattr(item, "name", "")) == "last_hidden_state":
                return outputs[index]
        candidates = [
            value
            for value in outputs
            if getattr(value, "ndim", None) == 3 and value.shape[-1] == 384
        ]
        if len(candidates) != 1:
            raise EmbeddingRuntimeUnavailable(
                "Unable to identify the canonical ONNX hidden-state output"
            )
        return candidates[0]

    # -------------------------------------------------------------------------
    def _validate_session(self, session: Any) -> None:
        providers = (
            list(session.get_providers()) if hasattr(session, "get_providers") else []
        )
        if providers and self.config.execution_provider not in providers:
            raise EmbeddingRuntimeUnavailable("CPUExecutionProvider is unavailable")
        names = {str(item.name) for item in session.get_inputs()}
        if not {"input_ids", "attention_mask"}.issubset(names):
            raise EmbeddingRuntimeUnavailable(
                "ONNX model must declare input_ids and attention_mask"
            )

    # -------------------------------------------------------------------------
    def _validate_vectors(self, rows: Any, expected: int) -> list[list[float]]:
        if not isinstance(rows, list) or len(rows) != expected:
            raise EmbeddingVectorValidationError(
                "Embedding count does not match inputs"
            )
        result = []
        for row in rows:
            if not isinstance(row, list) or len(row) != self.config.dimension:
                raise EmbeddingVectorValidationError(
                    f"Embedding dimension must be {self.config.dimension}"
                )
            vector = [float(value) for value in row]
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingVectorValidationError(
                    "Embedding contains non-finite values"
                )
            if (
                self.config.normalize
                and abs(math.sqrt(sum(value * value for value in vector)) - 1.0) > 1e-3
            ):
                raise EmbeddingVectorValidationError(
                    "Normalized embedding has invalid norm"
                )
            result.append(vector)
        return result


###############################################################################
def _default_snapshot_downloader(**kwargs: object) -> str:
    return str(cast(Any, snapshot_download)(**kwargs))


###############################################################################
def _default_tokenizer_factory(path: str) -> Any:
    return Tokenizer.from_file(path)


###############################################################################
def _default_session_factory(path: str, **kwargs: object) -> Any:
    return cast(Any, onnxruntime.InferenceSession)(path, **kwargs)


###############################################################################
@lru_cache(maxsize=1)
def get_embedding_runtime() -> EmbeddingRuntime:
    rag = build_effective_rag_settings()
    return EmbeddingRuntime(
        offline_mode=rag.embedding_offline_mode,
        batch_size=rag.embedding_batch_size,
    )


###############################################################################
def close_embedding_runtime() -> None:
    if get_embedding_runtime.cache_info().currsize == 0:
        return
    get_embedding_runtime().close()
    get_embedding_runtime.cache_clear()


__all__ = [
    "EmbeddingRuntime",
    "EmbeddingRuntimeError",
    "EmbeddingRuntimeUnavailable",
    "EmbeddingVectorValidationError",
    "REQUIRED_SNAPSHOT_FILES",
    "close_embedding_runtime",
    "get_embedding_runtime",
]
