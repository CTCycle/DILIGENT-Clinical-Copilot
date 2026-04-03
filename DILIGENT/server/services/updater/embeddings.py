from __future__ import annotations

import os

from DILIGENT.server.configurations.bootstrap import server_settings
from DILIGENT.server.common.constants import DOCS_PATH, VECTOR_DB_PATH
from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.repositories.serialization.data import VectorSerializer
from DILIGENT.server.repositories.vectors import LanceVectorDatabase

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
    ) -> None:
        self.documents_path = documents_path or DOCS_PATH
        default_use_cloud = server_settings.rag.use_cloud_embeddings
        self.use_cloud_embeddings = (
            default_use_cloud if use_cloud_embeddings is None else use_cloud_embeddings
        )
        resolved_provider = cloud_provider or server_settings.rag.cloud_provider
        resolved_model = cloud_embedding_model or server_settings.rag.cloud_embedding_model
        self.vector_collection_name = (
            vector_collection_name or server_settings.rag.vector_collection_name
        )
        self.chunk_size = int(chunk_size if chunk_size is not None else server_settings.rag.chunk_size)
        self.chunk_overlap = int(
            chunk_overlap if chunk_overlap is not None else server_settings.rag.chunk_overlap
        )
        self.embedding_batch_size = int(
            embedding_batch_size
            if embedding_batch_size is not None
            else server_settings.rag.embedding_batch_size
        )
        self.vector_stream_batch_size = int(
            vector_stream_batch_size
            if vector_stream_batch_size is not None
            else server_settings.rag.vector_stream_batch_size
        )
        self.embedding_max_workers = int(
            embedding_max_workers
            if embedding_max_workers is not None
            else server_settings.rag.embedding_max_workers
        )
        self.embedding_backend = (
            embedding_backend or server_settings.rag.embedding_backend
        )
        self.ollama_embedding_model = (
            ollama_embedding_model or server_settings.rag.ollama_embedding_model
        )
        self.hf_embedding_model = hf_embedding_model or server_settings.rag.hf_embedding_model
        self.reset_vector_collection = (
            server_settings.rag.reset_vector_collection
            if reset_vector_collection is None
            else bool(reset_vector_collection)
        )
        os.makedirs(self.documents_path, exist_ok=True)
        self.vector_database = LanceVectorDatabase(
            database_path=VECTOR_DB_PATH,
            collection_name=self.vector_collection_name,
            metric=server_settings.rag.vector_index_metric,
            index_type=server_settings.rag.vector_index_type,
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
            ollama_base_url=server_settings.rag.ollama_base_url,
            ollama_model=self.ollama_embedding_model,
            hf_model=self.hf_embedding_model,
            use_cloud_embeddings=self.use_cloud_embeddings,
            cloud_provider=resolved_provider,
            cloud_embedding_model=resolved_model,
        )

    # -------------------------------------------------------------------------
    def prepare_vector_database(self) -> None:
        if self.reset_vector_collection:
            self.vector_database.clear_collection()
        self.vector_database.initialize()
        self.vector_database.get_table()

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
