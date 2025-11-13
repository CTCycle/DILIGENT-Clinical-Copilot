from __future__ import annotations

import os

from DILIGENT.src.packages.configurations import get_configurations
from DILIGENT.src.packages.constants import DOCS_PATH, VECTOR_DB_PATH
from DILIGENT.src.packages.logger import logger
from DILIGENT.src.packages.utils.repository.serializer import VectorSerializer
from DILIGENT.src.packages.utils.repository.vectors import LanceVectorDatabase

CONFIG = get_configurations()
RAG_SETTINGS = CONFIG.rag


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
        default_use_cloud = RAG_SETTINGS.use_cloud_embeddings
        self.use_cloud_embeddings = (
            default_use_cloud if use_cloud_embeddings is None else use_cloud_embeddings
        )
        resolved_provider = cloud_provider or RAG_SETTINGS.cloud_provider
        resolved_model = cloud_embedding_model or RAG_SETTINGS.cloud_embedding_model
        self.reset_collection = (
            RAG_SETTINGS.reset_vector_collection if reset_collection is None else reset_collection
        )
        os.makedirs(self.documents_path, exist_ok=True)
        self.vector_database = LanceVectorDatabase(
            database_path=VECTOR_DB_PATH,
            collection_name=RAG_SETTINGS.vector_collection_name,
            metric=RAG_SETTINGS.vector_index_metric,
            index_type=RAG_SETTINGS.vector_index_type,
        )
        self.serializer = VectorSerializer(
            documents_path=self.documents_path,
            vector_database=self.vector_database,
            chunk_size=RAG_SETTINGS.chunk_size,
            chunk_overlap=RAG_SETTINGS.chunk_overlap,
            embedding_backend=RAG_SETTINGS.embedding_backend,
            ollama_base_url=RAG_SETTINGS.ollama_base_url,
            ollama_model=RAG_SETTINGS.ollama_embedding_model,
            hf_model=RAG_SETTINGS.hf_embedding_model,
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
