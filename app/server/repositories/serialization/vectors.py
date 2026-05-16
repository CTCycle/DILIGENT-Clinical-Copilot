from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from typing import Any

from configurations.startup import server_settings
from domain.documents import Document
from common.constants import DEFAULT_EMBEDDING_BATCH_SIZE
from common.utils.logger import logger
from repositories.serialization.data import DocumentChunker, DocumentSerializer
from repositories.vectors import LanceVectorDatabase
from services.retrieval.embeddings import EmbeddingGenerator


###############################################################################
class VectorSerializer:
    def __init__(
        self,
        documents_path: str,
        vector_database: LanceVectorDatabase,
        chunk_size: int,
        chunk_overlap: int,
        embedding_backend: str,
        ollama_base_url: str | None = None,
        ollama_model: str | None = None,
        hf_model: str | None = None,
        use_cloud_embeddings: bool = False,
        cloud_provider: str | None = None,
        cloud_embedding_model: str | None = None,
        embedding_batch_size: int | None = None,
        embedding_workers: int | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        if not isinstance(vector_database, LanceVectorDatabase):
            raise TypeError("vector_database must be a LanceVectorDatabase instance")
        self.vector_database = vector_database
        self.documents_path = documents_path
        self.document_serializer = DocumentSerializer(documents_path)
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.embedding_generator = EmbeddingGenerator(
            backend=embedding_backend,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            use_cloud_embeddings=use_cloud_embeddings,
            cloud_provider=cloud_provider,
            cloud_embedding_model=cloud_embedding_model,
        )
        resolved_batch_size = (
            DEFAULT_EMBEDDING_BATCH_SIZE
            if embedding_batch_size is None
            else embedding_batch_size
        )
        self.embedding_batch_size = max(int(resolved_batch_size), 1)
        resolved_workers = (
            server_settings.rag.embedding_max_workers
            if embedding_workers is None
            else embedding_workers
        )
        self.embedding_workers = max(int(resolved_workers), 1)
        self.progress_callback = progress_callback

    # -------------------------------------------------------------------------
    def serialize(self) -> dict[str, Any]:
        self.report_progress(14.0, "Discovering supported RAG files")
        self.vector_database.initialize()
        self.vector_database.get_table()
        available_paths = self.document_serializer.collect_document_paths()
        total_supported_files = len(available_paths)
        diagnostic_paths = [path for path in available_paths[:5]]
        self.report_progress(
            18.0,
            f"Loading text from {total_supported_files} supported files",
        )
        documents = self.document_serializer.load_documents()
        if not documents:
            logger.warning("No documents available for embedding serialization")
            return {
                "documents": 0,
                "chunks": 0,
                "supported_files": total_supported_files,
                "loaded_documents": 0,
                "sample_supported_paths": diagnostic_paths,
            }
        self.report_progress(
            24.0,
            f"Loaded {len(documents)} source document parts; chunking content",
        )
        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            logger.warning("Document chunking resulted in zero chunks")
            return {
                "documents": 0,
                "chunks": 0,
                "supported_files": total_supported_files,
                "loaded_documents": len(documents),
                "sample_supported_paths": diagnostic_paths,
            }
        self.report_progress(
            30.0,
            f"Prepared {len(chunks)} chunks for embedding",
        )
        batch_size = self.embedding_batch_size
        total_records = 0
        document_ids: set[str] = set()
        batch_iter: list[list[Document]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            if batch:
                batch_iter.append(batch)
        total_batches = len(batch_iter)
        completed_batches = 0
        with ThreadPoolExecutor(max_workers=self.embedding_workers) as executor:
            futures = [
                executor.submit(self._embed_chunk_batch, batch_chunks)
                for batch_chunks in batch_iter
            ]
            for future in as_completed(futures):
                records, batch_ids = future.result()
                completed_batches += 1
                document_ids.update(batch_ids)
                if not records:
                    self.report_batch_progress(
                        completed_batches=completed_batches,
                        total_batches=total_batches,
                    )
                    continue
                self.vector_database.upsert_embeddings(records)
                total_records += len(records)
                self.report_batch_progress(
                    completed_batches=completed_batches,
                    total_batches=total_batches,
                )
        if total_records:
            self.report_progress(
                88.0,
                "Finalizing full-text and scalar indices",
            )
            table = self.vector_database.get_table()
            self.vector_database.ensure_full_text_index(table)
            self.vector_database.ensure_scalar_indices(table)
        logger.info(
            "Serialized %d documents into %d vector chunks",
            len(document_ids),
            total_records,
        )
        return {
            "documents": len(document_ids),
            "chunks": total_records,
            "supported_files": total_supported_files,
            "loaded_documents": len(documents),
            "sample_supported_paths": diagnostic_paths,
        }

    # -------------------------------------------------------------------------
    def report_progress(self, progress: float, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(progress, message)

    # -------------------------------------------------------------------------
    def report_batch_progress(
        self,
        *,
        completed_batches: int,
        total_batches: int,
    ) -> None:
        if total_batches <= 0:
            return
        bounded_completed = min(max(completed_batches, 0), total_batches)
        progress = 30.0 + (58.0 * bounded_completed / total_batches)
        self.report_progress(
            progress,
            f"Embedded and persisted batch {bounded_completed}/{total_batches}",
        )

    # -------------------------------------------------------------------------
    def _embed_chunk_batch(
        self, batch_chunks: list[Document]
    ) -> tuple[list[dict[str, Any]], set[str]]:
        if not batch_chunks:
            return [], set()
        texts = [chunk.page_content for chunk in batch_chunks]
        embeddings = self.embedding_generator.embed_texts(texts)
        if len(embeddings) != len(batch_chunks):
            raise RuntimeError("Embedding count does not match chunk count")
        document_ids = {
            str(chunk.metadata.get("document_id"))
            for chunk in batch_chunks
            if chunk.metadata.get("document_id")
        }
        records = self.build_records(batch_chunks, embeddings)
        return records, document_ids

    # -------------------------------------------------------------------------
    def build_records(
        self,
        chunks: list[Document],
        embeddings: list[list[float]],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings):
            document_id = str(chunk.metadata.get("document_id", ""))
            chunk_index = chunk.metadata.get("chunk_index")
            chunk_id = (
                f"{document_id}:{chunk_index}"
                if chunk_index is not None
                else document_id
            )
            metadata = self.serialize_metadata(chunk.metadata)
            records.append(
                {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "text": chunk.page_content,
                    "embedding": embedding,
                    "source": metadata.get("source", ""),
                    "file_name": metadata.get("file_name", ""),
                    "document_title": metadata.get("document_title", ""),
                    "page_number": metadata.get("page_number"),
                    "section_title": metadata.get("section_title", ""),
                    "heading_path": metadata.get("heading_path", ""),
                    "content_type": metadata.get("content_type", ""),
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                }
            )
        return records

    # -------------------------------------------------------------------------
    def serialize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for key, value in metadata.items():
            if key == "chunk_index":
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized


__all__ = ["VectorSerializer"]
