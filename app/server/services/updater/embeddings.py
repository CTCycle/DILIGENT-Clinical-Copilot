from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from common.paths import DOCS_PATH, VECTOR_DB_PATH
from common.utils.logger import logger
from services.rag.vector_serializer import VectorSerializer
from repositories.vectors import LanceVectorDatabase
from services.retrieval.settings import build_effective_rag_settings

###############################################################################
class RagEmbeddingUpdater:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        documents_path: str | None = None,
        vector_collection_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        embedding_batch_size: int | None = None,
        vector_stream_batch_size: int | None = None,
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
                }.items()
                if value is not None
            }
        )
        resolved_documents_path = Path(documents_path) if documents_path else DOCS_PATH
        if not resolved_documents_path.is_absolute():
            raise ValueError("RAG documents_path must be an absolute path.")
        if not resolved_documents_path.exists() or not resolved_documents_path.is_dir():
            raise ValueError("RAG documents_path does not exist or is not a directory.")
        self.documents_path = str(resolved_documents_path)
        self.generation_id = str(uuid4())
        self.vector_collection_name = vector_collection_name or (
            f"documents__build_{self.generation_id}"
        )
        self.chunk_size = int(rag_settings.chunk_size)
        self.chunk_overlap = int(rag_settings.chunk_overlap)
        self.embedding_batch_size = int(rag_settings.embedding_batch_size)
        self.vector_stream_batch_size = int(rag_settings.vector_stream_batch_size)
        self.vector_database = LanceVectorDatabase(
            database_path=str(VECTOR_DB_PATH),
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
            progress_callback=progress_callback,
        )

    # -------------------------------------------------------------------------
    def prepare_vector_database(self) -> None:
        self.validate_embedding_backend()
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
        summary["generation_id"] = self.generation_id
        summary["collection_name"] = self.vector_collection_name
        logger.info(
            "RAG embeddings refreshed using the multilingual Granite ONNX runtime (%d documents, %d chunks)",
            summary.get("documents", 0),
            summary.get("chunks", 0),
        )
        return summary


__all__ = ["RagEmbeddingUpdater"]
