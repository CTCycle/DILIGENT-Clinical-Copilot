from __future__ import annotations

import os

from DILIGENT.server.utils.configurations import server_settings
from DILIGENT.server.utils.constants import DOCS_PATH, VECTOR_DB_PATH
from DILIGENT.server.utils.logger import logger
from DILIGENT.server.utils.repository.serializer import VectorSerializer
from DILIGENT.server.utils.repository.vectors import LanceVectorDatabase

###############################################################################
class RagEmbeddingUpdater:
    def __init__(
        self,
        documents_path: str | None = None,
        use_cloud_embeddings: bool | None = None,
        cloud_provider: str | None = None,
        cloud_embedding_model: str | None = None,
        reset_collection: bool | None = None,
    ) -> None:
        self.documents_path = documents_path or DOCS_PATH
        default_use_cloud = server_settings.rag.use_cloud_embeddings
        self.use_cloud_embeddings = (
            default_use_cloud if use_cloud_embeddings is None else use_cloud_embeddings
        )
        resolved_provider = cloud_provider or server_settings.rag.cloud_provider
        resolved_model = cloud_embedding_model or server_settings.rag.cloud_embedding_model
        self.reset_collection = (
            server_settings.rag.reset_vector_collection if reset_collection is None else reset_collection
        )
        os.makedirs(self.documents_path, exist_ok=True)
        self.vector_database = LanceVectorDatabase(
            database_path=VECTOR_DB_PATH,
            collection_name=server_settings.rag.vector_collection_name,
            metric=server_settings.rag.vector_index_metric,
            index_type=server_settings.rag.vector_index_type,
            stream_batch_size=server_settings.rag.vector_stream_batch_size,
        )
        self.serializer = VectorSerializer(
            documents_path=self.documents_path,
            vector_database=self.vector_database,
            chunk_size=server_settings.rag.chunk_size,
            chunk_overlap=server_settings.rag.chunk_overlap,
            embedding_batch_size=server_settings.rag.embedding_batch_size,
            embedding_workers=server_settings.rag.embedding_max_workers,
            embedding_backend=server_settings.rag.embedding_backend,
            ollama_base_url=server_settings.rag.ollama_base_url,
            ollama_model=server_settings.rag.ollama_embedding_model,
            hf_model=server_settings.rag.hf_embedding_model,
            use_cloud_embeddings=self.use_cloud_embeddings,
            cloud_provider=resolved_provider,
            cloud_embedding_model=resolved_model,
        )

    # -------------------------------------------------------------------------
    def prepare_vector_database(self, reset_collection: bool | None = None) -> None:
        should_reset = self.reset_collection if reset_collection is None else reset_collection
        self.vector_database.initialize(False)
        self.vector_database.get_table()
        if should_reset != self.reset_collection:
            self.reset_collection = should_reset

    # -------------------------------------------------------------------------
    def refresh_embeddings(self, reset_collection: bool | None = None) -> dict[str, int]:
        should_reset = self.reset_collection if reset_collection is None else reset_collection
        summary = self.serializer.serialize(reset_collection=should_reset)
        backend_label = "cloud" if self.use_cloud_embeddings else "local"
        logger.info(
            "RAG embeddings refreshed using %s backend (%d documents, %d chunks)",
            backend_label,
            summary.get("documents", 0),
            summary.get("chunks", 0),
        )
        return summary


__all__ = ["RagEmbeddingUpdater"]
