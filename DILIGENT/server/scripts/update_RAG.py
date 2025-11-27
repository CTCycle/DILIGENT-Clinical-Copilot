from __future__ import annotations

from DILIGENT.server.packages.configurations import server_settings
from DILIGENT.server.packages.logger import logger
from DILIGENT.server.packages.utils.updater.embeddings import RagEmbeddingUpdater


###############################################################################
if __name__ == "__main__":
    updater = RagEmbeddingUpdater(
        use_cloud_embeddings=server_settings.rag.use_cloud_embeddings,
        cloud_provider=server_settings.rag.cloud_provider,
        cloud_embedding_model=server_settings.rag.cloud_embedding_model,
    )
    updater.prepare_vector_database(reset_collection=False)
    summary = updater.refresh_embeddings()
    logger.info(
        "RAG pipeline update completed (%d documents, %d chunks)",
        summary.get("documents", 0),
        summary.get("chunks", 0),
    )
