"""Lazy, process-local Sentence Transformers runtime for canonical RAG."""

from __future__ import annotations

import importlib
import math
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from common.embedding.config import (
    CANONICAL_EMBEDDING_CONFIG,
    CanonicalEmbeddingConfig,
)
from common.paths import EMBEDDING_MODELS_PATH

REQUIRED_SNAPSHOT_FILES = frozenset(
    {
        "model.safetensors",
        "config.json",
        "modules.json",
        "sentence_bert_config.json",
        "1_Pooling/config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
)


class EmbeddingRuntimeError(RuntimeError):
    """Base error for unavailable or invalid embedding runtime state."""


class EmbeddingRuntimeUnavailable(EmbeddingRuntimeError):
    """Raised when the local inference dependencies or model are unavailable."""


class EmbeddingVectorValidationError(EmbeddingRuntimeError):
    """Raised when an inference result violates the canonical vector contract."""


SnapshotDownloader = Callable[..., str | Path]
ModelFactory = Callable[..., Any]


class EmbeddingRuntime:
    """Load one canonical model lazily and serialize access to its encoder."""

    def __init__(
        self,
        *,
        config: CanonicalEmbeddingConfig = CANONICAL_EMBEDDING_CONFIG,
        cache_directory: Path = EMBEDDING_MODELS_PATH,
        device: str = "auto",
        offline_mode: bool = False,
        batch_size: int = 16,
        model_factory: ModelFactory | None = None,
        snapshot_downloader: SnapshotDownloader | None = None,
    ) -> None:
        self.config = config
        self.cache_directory = Path(cache_directory)
        self.requested_device = device.strip().lower() or "auto"
        self.offline_mode = offline_mode
        self.batch_size = max(int(batch_size), 1)
        self._model_factory = model_factory
        self._snapshot_downloader = snapshot_downloader
        self._model: Any | None = None
        self._resolved_device: str | None = None
        self._lock = threading.RLock()
        self._closed = False

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def resolved_device(self) -> str | None:
        return self._resolved_device

    def status(self) -> dict[str, object]:
        snapshot = self._cached_snapshot_path()
        dependencies_available = (
            self._model_factory is not None
            or (
            self._optional_module("sentence_transformers") is not None
            and self._optional_module("torch") is not None
            )
        )
        return {
            "model_id": self.config.model_id,
            "model_revision": self.config.revision,
            "device": self._resolved_device or self.requested_device,
            "cache_status": (
                "missing"
                if not self._has_required_files(snapshot)
                else "available" if dependencies_available else "dependency_missing"
            ),
            "loaded": self.loaded,
        }

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, prefix=self.config.document_prefix)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, prefix=self.config.query_prefix)

    def get_tokenizer(self) -> Any:
        with self._lock:
            return getattr(self._ensure_model(), "tokenizer", None)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._model = None
            if self._resolved_device == "cuda":
                torch = self._optional_module("torch")
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _encode(self, texts: Sequence[str], *, prefix: str) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            if self._closed:
                raise EmbeddingRuntimeError("Embedding runtime is shutting down")
            model = self._ensure_model()
            prepared = [f"{prefix}{text or ' '}" for text in texts]
            try:
                values = model.encode(
                    prepared,
                    batch_size=self.batch_size,
                    normalize_embeddings=self.config.normalize,
                    convert_to_numpy=True,
                    precision=self.config.output_dtype,
                    show_progress_bar=False,
                )
            except TypeError:
                values = model.encode(
                    prepared,
                    batch_size=self.batch_size,
                    normalize_embeddings=self.config.normalize,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            return self._validate_vectors(values, expected_count=len(prepared))

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        self._resolved_device = self._resolve_device()
        snapshot_path = self._resolve_snapshot()
        factory = self._model_factory or self._default_model_factory
        torch = self._optional_module("torch")
        torch_dtype = getattr(torch, "float32", "float32")
        try:
            self._model = factory(
                str(snapshot_path),
                device=self._resolved_device,
                model_kwargs={"torch_dtype": torch_dtype},
                trust_remote_code=self.config.trust_remote_code,
            )
        except EmbeddingRuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingRuntimeUnavailable(
                f"Failed to load embedding model {self.config.model_id}"
            ) from exc
        if hasattr(self._model, "max_seq_length"):
            self._model.max_seq_length = self.config.maximum_model_tokens
        return self._model

    def _resolve_snapshot(self) -> Path:
        cached = self._cached_snapshot_path()
        if self._has_required_files(cached):
            return cached
        if self.offline_mode:
            raise EmbeddingRuntimeUnavailable(
                "Canonical embedding cache is incomplete in offline mode"
            )
        downloader = self._snapshot_downloader or self._default_snapshot_downloader
        try:
            snapshot = Path(
                downloader(
                    repo_id=self.config.model_id,
                    revision=self.config.revision,
                    cache_dir=str(self.cache_directory),
                    local_files_only=False,
                    allow_patterns=sorted(REQUIRED_SNAPSHOT_FILES),
                    ignore_patterns=["*.bin", "*.onnx", "*.ot", "*.msgpack"],
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingRuntimeUnavailable(
                f"Unable to download embedding snapshot at revision {self.config.revision}"
            ) from exc
        if not self._has_required_files(snapshot):
            raise EmbeddingRuntimeUnavailable(
                "Downloaded embedding snapshot is incomplete"
            )
        return snapshot

    def _cached_snapshot_path(self) -> Path:
        return self.cache_directory / self.config.revision

    @staticmethod
    def _has_required_files(snapshot: Path) -> bool:
        return snapshot.is_dir() and all(
            (snapshot / relative_path).is_file()
            for relative_path in REQUIRED_SNAPSHOT_FILES
        )

    def _resolve_device(self) -> str:
        if self.requested_device not in {"auto", "cpu", "cuda", "mps"}:
            raise EmbeddingRuntimeUnavailable(
                f"Unsupported embedding device: {self.requested_device}"
            )
        if self.requested_device == "cpu":
            return "cpu"
        torch = self._optional_module("torch")
        if torch is None:
            if self.requested_device == "auto":
                return "cpu"
            raise EmbeddingRuntimeUnavailable("PyTorch is not installed")
        if self.requested_device == "cuda":
            if not torch.cuda.is_available():
                raise EmbeddingRuntimeUnavailable("CUDA was requested but is unavailable")
            return "cuda"
        if self.requested_device == "mps":
            if not torch.backends.mps.is_available():
                raise EmbeddingRuntimeUnavailable("MPS was requested but is unavailable")
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _optional_module(name: str) -> Any | None:
        try:
            return importlib.import_module(name)
        except ImportError:
            return None

    @staticmethod
    def _default_snapshot_downloader(**kwargs: object) -> str:
        module = importlib.import_module("huggingface_hub")
        snapshot_download = getattr(module, "snapshot_download")
        return str(snapshot_download(**kwargs))

    @staticmethod
    def _default_model_factory(model_path: str, **kwargs: object) -> Any:
        module = importlib.import_module("sentence_transformers")
        sentence_transformer = getattr(module, "SentenceTransformer")
        return sentence_transformer(model_path, **kwargs)

    def _validate_vectors(self, values: Any, *, expected_count: int) -> list[list[float]]:
        rows = values.tolist() if hasattr(values, "tolist") else values
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise EmbeddingVectorValidationError("Embedding count does not match inputs")
        result: list[list[float]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) != self.config.dimension:
                raise EmbeddingVectorValidationError(
                    f"Embedding dimension must be {self.config.dimension}"
                )
            vector = [float(value) for value in row]
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingVectorValidationError("Embedding contains non-finite values")
            norm = math.sqrt(sum(value * value for value in vector))
            if self.config.normalize and abs(norm - 1.0) > 1e-3:
                raise EmbeddingVectorValidationError("Normalized embedding has invalid norm")
            result.append(vector)
        return result


_EMBEDDING_RUNTIME: EmbeddingRuntime | None = None
_EMBEDDING_RUNTIME_LOCK = threading.Lock()


def get_embedding_runtime() -> EmbeddingRuntime:
    """Return the process singleton without loading the model."""

    global _EMBEDDING_RUNTIME
    with _EMBEDDING_RUNTIME_LOCK:
        if _EMBEDDING_RUNTIME is None:
            _EMBEDDING_RUNTIME = EmbeddingRuntime()
        return _EMBEDDING_RUNTIME


def close_embedding_runtime() -> None:
    """Release the process singleton during application shutdown."""

    global _EMBEDDING_RUNTIME
    with _EMBEDDING_RUNTIME_LOCK:
        if _EMBEDDING_RUNTIME is not None:
            _EMBEDDING_RUNTIME.close()
            _EMBEDDING_RUNTIME = None


__all__ = [
    "EmbeddingRuntime",
    "EmbeddingRuntimeError",
    "EmbeddingRuntimeUnavailable",
    "EmbeddingVectorValidationError",
    "REQUIRED_SNAPSHOT_FILES",
    "close_embedding_runtime",
    "get_embedding_runtime",
]
