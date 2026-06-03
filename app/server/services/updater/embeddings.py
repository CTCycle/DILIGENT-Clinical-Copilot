from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from common.constants import DOCS_PATH, VECTOR_DB_PATH
from common.utils.logger import logger
from configurations.startup import get_server_settings
from repositories.serialization.vectors import VectorSerializer
from repositories.vectors import LanceVectorDatabase
from services.retrieval.settings import build_effective_rag_settings


###############################################################################
class RagEmbeddingUpdater:
    def __init__(
        self,
        documents_path: str | None = None,
        use_cloud_embeddings: bool | None = None,
        cloud_provider: str | None = None,
        cloud_embedding_model: str | None = None,
        vector_collection_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        embedding_batch_size: int | None = None,
        vector_stream_batch_size: int | None = None,
        embedding_max_workers: int | None = None,
        embedding_backend: str | None = None,
        ollama_embedding_model: str | None = None,
        hf_embedding_model: str | None = None,
        reset_vector_collection: bool | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        rag_settings = build_effective_rag_settings(
            {
                key: value
                for key, value in {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "embedding_batch_size": embedding_batch_size,
                    "vector_stream_batch_size": vector_stream_batch_size,
                    "embedding_max_workers": embedding_max_workers,
                    "embedding_backend": embedding_backend,
                    "ollama_embedding_model": ollama_embedding_model,
                    "hf_embedding_model": hf_embedding_model,
                    "cloud_provider": cloud_provider,
                    "cloud_embedding_model": cloud_embedding_model,
                    "use_cloud_embeddings": use_cloud_embeddings,
                    "reset_vector_collection": reset_vector_collection,
                }.items()
                if value is not None
            }
        )
        self.documents_path = documents_path or DOCS_PATH
        resolved_documents_path = Path(self.documents_path)
        if not resolved_documents_path.is_absolute():
            raise ValueError("RAG documents_path must be an absolute path.")
        if not resolved_documents_path.exists() or not resolved_documents_path.is_dir():
            raise ValueError("RAG documents_path does not exist or is not a directory.")
        self.documents_path = str(resolved_documents_path)
        self.use_cloud_embeddings = rag_settings.use_cloud_embeddings
        resolved_provider = rag_settings.cloud_provider
        resolved_model = rag_settings.cloud_embedding_model
        self.vector_collection_name = (
            vector_collection_name or rag_settings.vector_collection_name
        )
        self.chunk_size = int(rag_settings.chunk_size)
        self.chunk_overlap = int(rag_settings.chunk_overlap)
        self.embedding_batch_size = int(rag_settings.embedding_batch_size)
        self.vector_stream_batch_size = int(rag_settings.vector_stream_batch_size)
        self.embedding_max_workers = int(rag_settings.embedding_max_workers)
        self.embedding_backend = rag_settings.embedding_backend
        self.ollama_embedding_model = rag_settings.ollama_embedding_model
        self.hf_embedding_model = rag_settings.hf_embedding_model
        self.reset_vector_collection = bool(rag_settings.reset_vector_collection)
        self.vector_database = LanceVectorDatabase(
            database_path=VECTOR_DB_PATH,
            collection_name=self.vector_collection_name,
            metric=rag_settings.vector_index_metric,
            index_type=rag_settings.vector_index_type,
            stream_batch_size=self.vector_stream_batch_size,
        )
        self.serializer = VectorSerializer(
            documents_path=self.documents_path,
            vector_database=self.vector_database,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedding_batch_size=self.embedding_batch_size,
            embedding_workers=self.embedding_max_workers,
            embedding_backend=self.embedding_backend,
            ollama_base_url=rag_settings.ollama_base_url,
            ollama_model=self.ollama_embedding_model,
            hf_model=self.hf_embedding_model,
            use_cloud_embeddings=self.use_cloud_embeddings,
            cloud_provider=resolved_provider,
            cloud_embedding_model=resolved_model,
            progress_callback=progress_callback,
        )

    # -------------------------------------------------------------------------
    def prepare_vector_database(self) -> None:
        self.validate_embedding_backend()
        if self.reset_vector_collection:
            self.vector_database.clear_collection()
        self.vector_database.initialize()
        self.vector_database.get_table()

    # -------------------------------------------------------------------------
    def validate_embedding_backend(self) -> None:
        self.serializer.embedding_generator.embed_texts(
            ["RAG embedding backend readiness check."]
        )

    # -------------------------------------------------------------------------
    def refresh_embeddings(self) -> dict[str, int]:
        summary = self.serializer.serialize()
        backend_label = "cloud" if self.use_cloud_embeddings else "local"
        logger.info(
            "RAG embeddings refreshed using %s backend (%d documents, %d chunks)",
            backend_label,
            summary.get("documents", 0),
            summary.get("chunks", 0),
        )
        return summary


__all__ = ["RagEmbeddingUpdater"]
