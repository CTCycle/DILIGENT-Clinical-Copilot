from __future__ import annotations

from DILIGENT.app.constants import (
    DOCS_PATH,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_EMBEDDING_BACKEND,
    RAG_HF_EMBEDDING_MODEL,
    RAG_OLLAMA_BASE_URL,
    RAG_OLLAMA_EMBEDDING_MODEL,
    RAG_RESET_VECTOR_COLLECTION,
    RAG_VECTOR_INDEX_METRIC,
    RAG_VECTOR_INDEX_TYPE,
    VECTOR_COLLECTION_NAME,
    VECTOR_DB_PATH,
)
from DILIGENT.app.utils.services.updater import VectorStoreUpdater

###############################################################################
if __name__ == "__main__":
    updater = VectorStoreUpdater(
        DOCS_PATH,
        VECTOR_DB_PATH,
        VECTOR_COLLECTION_NAME,
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        embedding_backend=RAG_EMBEDDING_BACKEND,
        ollama_base_url=RAG_OLLAMA_BASE_URL,
        ollama_model=RAG_OLLAMA_EMBEDDING_MODEL,
        huggingface_model=RAG_HF_EMBEDDING_MODEL,
        index_metric=RAG_VECTOR_INDEX_METRIC,
        index_type=RAG_VECTOR_INDEX_TYPE,
        reset_collection=RAG_RESET_VECTOR_COLLECTION,
    )
    updater.run()
