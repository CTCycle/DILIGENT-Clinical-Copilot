from __future__ import annotations

from DILIGENT.src.packages.configurations import configurations
from DILIGENT.src.packages.logger import logger
from DILIGENT.src.packages.utils.updater.embeddings import RagEmbeddingUpdater

USE_CLOUD_EMBEDDINGS = configurations.rag.use_cloud_embeddings
CLOUD_PROVIDER = configurations.rag.cloud_provider
CLOUD_EMBEDDING_MODEL = configurations.rag.cloud_embedding_model

###############################################################################
if __name__ == "__main__":
    updater = RagEmbeddingUpdater(
        use_cloud_embeddings=USE_CLOUD_EMBEDDINGS,
        cloud_provider=CLOUD_PROVIDER,
        cloud_embedding_model=CLOUD_EMBEDDING_MODEL,
    )
    updater.prepare_vector_database(reset_collection=False)
    summary = updater.refresh_embeddings()
    logger.info(
        "RAG pipeline update completed (%d documents, %d chunks)",
        summary.get("documents", 0),
        summary.get("chunks", 0),
    )
