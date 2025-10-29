from __future__ import annotations

import os

from DILIGENT.app.configurations import (
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_CLOUD_MODEL,
    RAG_CLOUD_PROVIDER,
    RAG_EMBEDDING_BACKEND,
    RAG_HF_EMBEDDING_MODEL,
    RAG_OLLAMA_BASE_URL,
    RAG_OLLAMA_EMBEDDING_MODEL,
    RAG_RESET_VECTOR_COLLECTION,
    RAG_USE_CLOUD_EMBEDDINGS,
    RAG_VECTOR_INDEX_METRIC,
    RAG_VECTOR_INDEX_TYPE,
    VECTOR_COLLECTION_NAME,
)
from DILIGENT.app.constants import DOCS_PATH, VECTOR_DB_PATH
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.serializer import VectorSerializer
from DILIGENT.app.utils.repository.vectors import LanceVectorDatabase


###############################################################################
class RagEmbeddingUpdater:
    def __init__(
        self,
        documents_path: str | None = None,
        use_cloud_embeddings: bool | None = None,
        cloud_provider: str | None = None,
        cloud_model: str | None = None,
        reset_collection: bool | None = None,
    ) -> None:
        self.documents_path = documents_path or DOCS_PATH
        default_use_cloud = RAG_USE_CLOUD_EMBEDDINGS
        self.use_cloud_embeddings = (
            default_use_cloud if use_cloud_embeddings is None else use_cloud_embeddings
        )
        resolved_provider = cloud_provider or RAG_CLOUD_PROVIDER
        resolved_model = cloud_model or RAG_CLOUD_MODEL
        self.reset_collection = (
            RAG_RESET_VECTOR_COLLECTION if reset_collection is None else reset_collection
        )
        os.makedirs(self.documents_path, exist_ok=True)
        self.vector_database = LanceVectorDatabase(
            database_path=VECTOR_DB_PATH,
            collection_name=VECTOR_COLLECTION_NAME,
            metric=RAG_VECTOR_INDEX_METRIC,
            index_type=RAG_VECTOR_INDEX_TYPE,
        )
        self.serializer = VectorSerializer(
            documents_path=self.documents_path,
            vector_database=self.vector_database,
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
            embedding_backend=RAG_EMBEDDING_BACKEND,
            ollama_base_url=RAG_OLLAMA_BASE_URL,
            ollama_model=RAG_OLLAMA_EMBEDDING_MODEL,
            hf_model=RAG_HF_EMBEDDING_MODEL,
            use_cloud_embeddings=self.use_cloud_embeddings,
            cloud_provider=resolved_provider,
            cloud_model=resolved_model,
        )

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
