from __future__ import annotations

import os

from DILIGENT.app.configurations import DEFAULT_CLOUD_MODEL, DEFAULT_CLOUD_PROVIDER
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.updater.embeddings import RagEmbeddingUpdater

USE_CLOUD_EMBEDDINGS = os.environ.get("USE_CLOUD_EMBEDDINGS", "0").lower() in {
    "1",
    "true",
    "yes",
}
CLOUD_PROVIDER = os.environ.get("RAG_CLOUD_PROVIDER", DEFAULT_CLOUD_PROVIDER)
CLOUD_MODEL = os.environ.get("RAG_CLOUD_MODEL", DEFAULT_CLOUD_MODEL)

###############################################################################
if __name__ == "__main__":
    updater = RagEmbeddingUpdater(
        use_cloud_embeddings=USE_CLOUD_EMBEDDINGS,
        cloud_provider=CLOUD_PROVIDER,
        cloud_model=CLOUD_MODEL,
    )
    summary = updater.refresh_embeddings()
    logger.info(
        "RAG pipeline update completed (%d documents, %d chunks)",
        summary.get("documents", 0),
        summary.get("chunks", 0),
    )
