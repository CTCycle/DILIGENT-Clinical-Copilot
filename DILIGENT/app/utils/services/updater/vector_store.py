from __future__ import annotations

import asyncio
import json
import os
import uuid
import xml.etree.ElementTree as ElementTree
import zipfile
from collections import defaultdict
from typing import Any

import numpy as np
import pyarrow as pa
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from DILIGENT.app.api.models.providers import OllamaClient, OllamaError
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.serializer import VectorSerializer


###############################################################################
class DocxDocumentLoader:
    DOCUMENT_XML = "word/document.xml"
    WORD_NAMESPACE = (
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    )

    def __init__(self, path: str) -> None:
        self.path = path

    ############################################################################
    def load(self) -> list[Document]:
        content = self.read_document_text()
        if not content:
            return []
        return [Document(page_content=content, metadata={"source": self.path})]

    ############################################################################
    def read_document_text(self) -> str:
        try:
            with zipfile.ZipFile(self.path) as archive:
                xml_bytes = archive.read(self.DOCUMENT_XML)
        except KeyError as exc:
            raise ValueError(
                "DOCX archive does not contain document.xml"
            ) from exc
        except zipfile.BadZipFile as exc:
            raise ValueError("DOCX archive is corrupted") from exc
        try:
            root = ElementTree.fromstring(xml_bytes)
        except ElementTree.ParseError as exc:
            raise ValueError("DOCX document.xml could not be parsed") from exc
        paragraphs: list[str] = []
        for paragraph in root.iter(f"{self.WORD_NAMESPACE}p"):
            segments: list[str] = []
            for element in paragraph.iter():
                if element.tag == f"{self.WORD_NAMESPACE}t":
                    segments.append(element.text or "")
                elif element.tag == f"{self.WORD_NAMESPACE}tab":
                    segments.append("\t")
                elif element.tag == f"{self.WORD_NAMESPACE}br":
                    segments.append("\n")
            combined = "".join(segments).strip()
            if combined:
                paragraphs.append(combined)
        return "\n".join(paragraphs)


###############################################################################
class VectorStoreUpdater:
    SUPPORTED_EXTENSIONS = {".txt", ".docx", ".pdf", ".xml", ".md"}

    def __init__(
        self,
        documents_path: str,
        database_path: str,
        collection_name: str,
        *,
        chunk_size: int,
        chunk_overlap: int,
        embedding_backend: str,
        ollama_base_url: str,
        ollama_model: str,
        huggingface_model: str,
        index_metric: str,
        index_type: str,
        reset_collection: bool,
    ) -> None:
        self.documents_path = documents_path
        self.database_path = database_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_backend = embedding_backend.lower().strip()
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.huggingface_model = huggingface_model
        self.index_metric = index_metric
        self.index_type = index_type
        self.reset_collection = reset_collection
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )
        self.embedder: Embeddings | None = None
        self.serializer = VectorSerializer(
            database_path=self.database_path,
            collection_name=self.collection_name,
            index_metric=self.index_metric,
            index_type=self.index_type,
            reset_collection=self.reset_collection,
        )

    ############################################################################
    def run(self) -> None:
        documents = self.load_documents()
        if not documents:
            logger.warning(
                "No documents were discovered in %s", self.documents_path
            )
            return
        chunks = self.chunk_documents(documents)
        prepared = [chunk for chunk in chunks if chunk.page_content.strip()]
        if not prepared:
            logger.warning("Document processing produced no usable chunks")
            return
        embedder = self.get_embedder()
        embedding_matrix = self.embed_chunks(embedder, prepared)
        self.persist_embeddings(prepared, embedding_matrix)
        logger.info(
            "Vector store update completed with %s chunks", len(prepared)
        )

    ############################################################################
    def load_documents(self) -> list[Document]:
        documents: list[Document] = []
        for path in self.iter_document_paths():
            try:
                loaded = self.load_file(path)
            except Exception as exc:  # noqa: BLE001 - log precise failure
                logger.error("Unable to load %s: %s", path, exc)
                continue
            for doc in loaded:
                doc.metadata["source"] = path
                documents.append(doc)
        logger.info("Loaded %s source documents", len(documents))
        return documents

    ############################################################################
    def iter_document_paths(self) -> list[str]:
        collected: list[str] = []
        for root, _dirs, files in os.walk(self.documents_path):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext not in self.SUPPORTED_EXTENSIONS:
                    continue
                collected.append(os.path.join(root, name))
        return sorted(collected)

    ############################################################################
    def load_file(self, path: str) -> list[Document]:
        ext = os.path.splitext(path)[1].lower()
        if ext in {".txt", ".md", ".xml"}:
            loader = TextLoader(path, autodetect_encoding=True)
        elif ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext == ".docx":
            loader = DocxDocumentLoader(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        return loader.load()

    ############################################################################
    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        chunks = self.text_splitter.split_documents(documents)
        logger.info(
            "Generated %s chunks from %s documents",
            len(chunks),
            len(documents),
        )
        return chunks

    ############################################################################
    def get_embedder(self) -> Embeddings:
        if self.embedder is not None:
            return self.embedder
        backend = self.embedding_backend
        if backend == "ollama":
            try:
                self.embedder = self.create_ollama_embedder()
                return self.embedder
            except Exception as exc:  # noqa: BLE001 - fallback handling
                logger.warning(
                    "Falling back to HuggingFace embeddings after Ollama error: %s",
                    exc,
                )
        if backend in {"huggingface", "ollama"}:
            self.embedder = self.create_huggingface_embedder()
            return self.embedder
        raise ValueError(f"Unknown embedding backend: {self.embedding_backend}")

    ############################################################################
    def create_ollama_embedder(self) -> Embeddings:
        try:
            asyncio.run(self.ensure_ollama_model())
        except OllamaError:
            raise
        except Exception as exc:  # noqa: BLE001 - convert to runtime error
            raise RuntimeError(
                f"Unable to verify Ollama model: {exc}"
            ) from exc
        return OllamaEmbeddings(
            model=self.ollama_model,
            base_url=self.ollama_base_url,
        )

    ############################################################################
    async def ensure_ollama_model(self) -> None:
        async with OllamaClient(
            base_url=self.ollama_base_url,
            default_model=self.ollama_model,
        ) as client:
            await client.check_model_availability(self.ollama_model)

    ############################################################################
    def create_huggingface_embedder(self) -> Embeddings:
        return HuggingFaceEmbeddings(model_name=self.huggingface_model)

    ############################################################################
    def embed_chunks(
        self, embedder: Embeddings, chunks: list[Document]
    ) -> np.ndarray:
        contents = [chunk.page_content for chunk in chunks]
        embeddings = embedder.embed_documents(contents)
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise RuntimeError(
                "Embedding backend returned invalid shape"
            )
        if matrix.shape[0] != len(chunks):
            raise RuntimeError(
                "Embedding count mismatch between chunks and vectors"
            )
        return matrix

    ############################################################################
    def persist_embeddings(self, chunks: list[Document], matrix: np.ndarray) -> None:
        data_columns = self.prepare_columns(chunks)
        vector_length = matrix.shape[1]
        embeddings_array = pa.array(
            matrix.tolist(),
            type=pa.list_(pa.float32(), list_size=vector_length),
        )
        arrow_table = pa.table(
            {
                "id": pa.array(data_columns["ids"]),
                "document_id": pa.array(data_columns["document_ids"]),
                "source": pa.array(data_columns["sources"]),
                "chunk_index": pa.array(
                    data_columns["chunk_indices"],
                    type=pa.int32(),
                ),
                "content": pa.array(data_columns["contents"]),
                "metadata": pa.array(data_columns["metadata"]),
                "embedding": embeddings_array,
            }
        )
        self.serializer.save_embeddings(arrow_table)

    ############################################################################
    def prepare_columns(self, chunks: list[Document]) -> dict[str, list[Any]]:
        ids: list[str] = []
        document_ids: list[str] = []
        sources: list[str] = []
        chunk_indices: list[int] = []
        contents: list[str] = []
        metadata_entries: list[str] = []
        per_document_counter: dict[str, int] = defaultdict(int)
        documents_root = os.path.abspath(self.documents_path)
        for chunk in chunks:
            source_path = str(chunk.metadata.get("source") or "")
            if source_path:
                absolute_source = os.path.abspath(source_path)
            else:
                absolute_source = documents_root
            relative_source = os.path.relpath(absolute_source, documents_root)
            normalized_source = relative_source.replace(os.sep, "/")
            document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, normalized_source))
            position = per_document_counter[document_id]
            per_document_counter[document_id] = position + 1
            ids.append(str(uuid.uuid4()))
            document_ids.append(document_id)
            sources.append(normalized_source)
            chunk_indices.append(position)
            contents.append(chunk.page_content)
            metadata_entries.append(
                json.dumps(chunk.metadata, sort_keys=True, default=str)
            )
        return {
            "ids": ids,
            "document_ids": document_ids,
            "sources": sources,
            "chunk_indices": chunk_indices,
            "contents": contents,
            "metadata": metadata_entries,
        }

__all__ = ["DocxDocumentLoader", "VectorStoreUpdater"]
