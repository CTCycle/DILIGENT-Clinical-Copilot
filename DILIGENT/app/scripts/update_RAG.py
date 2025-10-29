from __future__ import annotations

from DILIGENT.app.configurations import (
    RAG_CLOUD_EMBEDDING_MODEL,
    RAG_CLOUD_PROVIDER,
    RAG_USE_CLOUD_EMBEDDINGS,
)
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.updater.embeddings import RagEmbeddingUpdater

USE_CLOUD_EMBEDDINGS = RAG_USE_CLOUD_EMBEDDINGS
CLOUD_PROVIDER = RAG_CLOUD_PROVIDER
CLOUD_EMBEDDING_MODEL = RAG_CLOUD_EMBEDDING_MODEL

###############################################################################
if __name__ == "__main__":
    updater = RagEmbeddingUpdater(
        use_cloud_embeddings=USE_CLOUD_EMBEDDINGS,
        cloud_provider=CLOUD_PROVIDER,
        cloud_embedding_model=CLOUD_EMBEDDING_MODEL,
    )
    summary = updater.refresh_embeddings()
    logger.info(
        "RAG pipeline update completed (%d documents, %d chunks)",
        summary.get("documents", 0),
        summary.get("chunks", 0),
    )
