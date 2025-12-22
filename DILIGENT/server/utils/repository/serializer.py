from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterator, cast
from xml.etree import ElementTree

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from DILIGENT.server.utils.configurations import server_settings
from DILIGENT.server.utils.constants import (
    CLINICAL_SESSION_COLUMNS,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DRUG_NAME_ALLOWED_PATTERN,
    DOCUMENT_SUPPORTED_EXTENSIONS,
    DRUGS_CATALOG_COLUMNS,
    LIVERTOX_COLUMNS,
    LIVERTOX_MASTER_COLUMNS,
    LIVERTOX_OPTIONAL_COLUMNS,
    LIVERTOX_REQUIRED_COLUMNS,
    TEXT_FILE_FALLBACK_ENCODINGS,
)
from DILIGENT.server.utils.logger import logger
from DILIGENT.server.database.database import database
from DILIGENT.server.utils.services.text.normalization import coerce_text
from DILIGENT.server.utils.repository.vectors import LanceVectorDatabase
from DILIGENT.server.utils.services.retrieval.embeddings import EmbeddingGenerator

###############################################################################
class DataSerializer:
    def __init__(self) -> None:
        pass

    # -------------------------------------------------------------------------
    def save_clinical_session(self, session_data: dict[str, Any]) -> None:
        frame = pd.DataFrame([session_data])
        if frame.empty:
            logger.warning("Skipping clinical session save; payload is empty")
            return
        frame = frame.reindex(columns=CLINICAL_SESSION_COLUMNS)
        frame = frame.where(pd.notnull(frame), cast(Any, None))
        existing = database.load_from_database("CLINICAL_SESSIONS")
        if existing.empty:
            database.save_into_database(frame, "CLINICAL_SESSIONS")
            return
        target_columns = existing.columns.tolist()
        normalized_frame = frame.reindex(columns=target_columns)
        combined = pd.concat([existing, normalized_frame], ignore_index=True)
        combined = combined.where(pd.notnull(combined), cast(Any, None))
        database.save_into_database(combined, "CLINICAL_SESSIONS")

    # -----------------------------------------------------------------------------
    def save_livertox_records(self, records: pd.DataFrame) -> None:
        frame = records.copy()
        if "drug_name" in frame.columns:
            frame = frame.drop_duplicates(subset=["drug_name"], keep="first")
        frame = frame.reindex(columns=LIVERTOX_COLUMNS)
        frame = frame.where(pd.notnull(frame), cast(Any, None))
        database.save_into_database(frame, "LIVERTOX_DATA")
   
    # -----------------------------------------------------------------------------
    def upsert_drugs_catalog_records(
        self, records: pd.DataFrame | list[dict[str, Any]]
    ) -> None:
        if isinstance(records, pd.DataFrame):
            frame = records.copy()
        else:
            frame = pd.DataFrame(records)
        frame = frame.reindex(columns=DRUGS_CATALOG_COLUMNS)
        if frame.empty:
            return
        frame = frame.where(pd.notnull(frame), cast(Any, None))
        frame["brand_names"] = frame["brand_names"].apply(self.serialize_brand_names)
        frame["synonyms"] = frame["synonyms"].apply(self.serialize_string_list)
        database.upsert_into_database(frame, "DRUGS_CATALOG")

    # -----------------------------------------------------------------------------
    def sanitize_livertox_records(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame(columns=LIVERTOX_REQUIRED_COLUMNS)
        for column in LIVERTOX_REQUIRED_COLUMNS:
            if column not in df.columns:
                df[column] = None
        df = df[LIVERTOX_REQUIRED_COLUMNS]
        drop_columns = [
            column
            for column in LIVERTOX_REQUIRED_COLUMNS
            if column not in LIVERTOX_OPTIONAL_COLUMNS
        ]
        df = df.dropna(subset=drop_columns)
        df["drug_name"] = df["drug_name"].apply(coerce_text)
        df = df[df["drug_name"].notna()]
        df = df[df["drug_name"].apply(self.is_valid_drug_name)]
        df["excerpt"] = df["excerpt"].apply(coerce_text)
        df = df[df["excerpt"].notna()]
        df["nbk_id"] = df["nbk_id"].apply(coerce_text)
        df["synonyms"] = df["synonyms"].apply(coerce_text)
        df = df.drop_duplicates(subset=["nbk_id", "drug_name"], keep="first")
        return df.reset_index(drop=True)

    # -----------------------------------------------------------------------------
    def is_valid_drug_name(self, value: str) -> bool:
        normalized = value.strip()
        min_length = server_settings.ingestion.drug_name_min_length
        max_length = server_settings.ingestion.drug_name_max_length
        max_tokens = server_settings.ingestion.drug_name_max_tokens
        if len(normalized) < min_length or len(normalized) > max_length:
            return False
        if len(normalized.split()) > max_tokens:
            return False
        if not re.fullmatch(DRUG_NAME_ALLOWED_PATTERN, normalized):
            return False
        return True

    # -----------------------------------------------------------------------------
    def serialize_string_list(self, value: Any) -> str:
        parsed_values = self._parse_serialized_list_input(value)
        normalized = self._normalize_list_items(parsed_values)
        return json.dumps(normalized, ensure_ascii=False)

    # -----------------------------------------------------------------------------
    def serialize_brand_names(self, value: Any) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            ok, parsed = self._try_json_loads(stripped)
            if not ok:
                return self.normalize_list_item(stripped)
            return self.serialize_brand_names(parsed)
        if isinstance(value, list):
            normalized = self._normalize_unique_list_items(value)
            if not normalized:
                return None
            if len(normalized) == 1:
                return normalized[0]
            return ", ".join(normalized)
        if pd.isna(value) or value is None:
            return None
        normalized_item = self.normalize_list_item(value)
        return normalized_item

    # -----------------------------------------------------------------------------
    def normalize_list_item(self, value: Any) -> str | None:
        return coerce_text(value)

    # -----------------------------------------------------------------------------
    def deserialize_string_list(self, value: Any) -> list[str]:
        parsed_values = self._parse_deserialized_list_input(value)
        return self._normalize_list_items(parsed_values)

    # -----------------------------------------------------------------------------
    def _try_json_loads(self, value: str) -> tuple[bool, Any]:
        try:
            return True, json.loads(value)
        except json.JSONDecodeError:
            return False, None

    # -----------------------------------------------------------------------------
    def _parse_serialized_list_input(self, value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            ok, parsed = self._try_json_loads(stripped)
            if ok and isinstance(parsed, list):
                return parsed
            return [stripped]
        if pd.isna(value) or value is None:
            return []
        return [value]

    # -----------------------------------------------------------------------------
    def _parse_deserialized_list_input(self, value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            ok, parsed = self._try_json_loads(stripped)
            if not ok:
                return [stripped]
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        if pd.isna(value) or value is None:
            return []
        return [value]

    # -----------------------------------------------------------------------------
    def _normalize_list_items(self, values: list[Any]) -> list[str]:
        normalized: list[str] = []
        for item in values:
            normalized_item = self.normalize_list_item(item)
            if normalized_item is not None:
                normalized.append(normalized_item)
        return normalized

    # -----------------------------------------------------------------------------
    def _normalize_unique_list_items(self, values: list[Any]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in values:
            normalized_item = self.normalize_list_item(item)
            if normalized_item is None:
                continue
            key = normalized_item.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(normalized_item)
        return normalized

    # -----------------------------------------------------------------------------
    def get_livertox_records(self) -> pd.DataFrame:
        frame = database.load_from_database("LIVERTOX_DATA")
        if frame.empty:
            return pd.DataFrame(columns=LIVERTOX_COLUMNS)
        return frame.reindex(columns=LIVERTOX_COLUMNS)

    # -----------------------------------------------------------------------------
    def get_livertox_master_list(self) -> pd.DataFrame:
        frame = database.load_from_database("LIVERTOX_DATA")
        if frame.empty:
            return pd.DataFrame(columns=LIVERTOX_MASTER_COLUMNS)
        available = [column for column in LIVERTOX_MASTER_COLUMNS if column in frame.columns]
        if not available:
            return pd.DataFrame(columns=["drug_name"])
        return frame.reindex(columns=available).dropna(subset=["drug_name"]).reset_index(
            drop=True
        )

    # -----------------------------------------------------------------------------
    def get_drugs_catalog(self) -> pd.DataFrame:
        frame = database.load_from_database("DRUGS_CATALOG")
        if frame.empty:
            return pd.DataFrame(columns=DRUGS_CATALOG_COLUMNS)
        return frame.reindex(columns=DRUGS_CATALOG_COLUMNS)

    # -----------------------------------------------------------------------------
    def stream_drugs_catalog(
        self, page_size: int | None = None
    ) -> Iterator[pd.DataFrame]:
        chunk_size = (
            server_settings.database.select_page_size
            if page_size is None
            else max(int(page_size), 1)
        )
        yield from database.stream_rows("DRUGS_CATALOG", chunk_size)

    # -----------------------------------------------------------------------------
    def normalize_string(self, value: Any) -> str | None:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized if normalized else None
        if pd.isna(value):
            return None
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized if normalized else None

    # -----------------------------------------------------------------------------
    def normalize_flag(self, value: Any) -> int | None:
        normalized = self.normalize_string(value)
        if normalized is None:
            return None
        lowered = normalized.lower()
        if lowered in {"1", "y", "yes", "true"}:
            return 1
        if lowered in {"0", "n", "no", "false"}:
            return 0
        if lowered == "2":
            return 0
        try:
            numeric = int(normalized)
        except (TypeError, ValueError):
            return None
        return 1 if numeric != 0 else 0

    # -----------------------------------------------------------------------------
    def normalize_date(self, value: Any) -> str | None:
        normalized = self.normalize_string(value)
        if not normalized:
            return None
        parsed = pd.to_datetime(normalized, errors="coerce", utc=True)
        if pd.isna(parsed):
            return normalized
        return parsed.date().isoformat()

    # -----------------------------------------------------------------------------
    def join_values(self, values: set[str]) -> str | None:
        if not values:
            return None
        return "; ".join(sorted(values))



###############################################################################
class DocumentSerializer:
    SUPPORTED_EXTENSIONS = DOCUMENT_SUPPORTED_EXTENSIONS

    def __init__(self, documents_path: str) -> None:
        self.documents_path = documents_path

    # -------------------------------------------------------------------------
    def collect_document_paths(self) -> list[str]:
        collected: list[str] = []
        for root, _, files in os.walk(self.documents_path):
            for name in files:
                extension = os.path.splitext(name)[1].lower()
                if extension in self.SUPPORTED_EXTENSIONS:
                    collected.append(os.path.join(root, name))
                else:
                    logger.debug("Skipping unsupported document '%s'", name)
        collected.sort()
        return collected

    # -------------------------------------------------------------------------
    def load_documents(self) -> list[Document]:
        documents: list[Document] = []
        for file_path in self.collect_document_paths():
            extension = os.path.splitext(file_path)[1].lower()
            if extension == ".pdf":
                documents.extend(self.load_pdf(file_path))
            elif extension == ".docx":
                documents.extend(self.load_docx(file_path))
            elif extension == ".doc":
                logger.warning(
                    "Legacy Word document '%s' is not supported; skipping", file_path
                )
            elif extension in {".txt", ".xml"}:
                documents.extend(self.load_textual_file(file_path, extension))
        return documents

    # -------------------------------------------------------------------------
    def load_pdf(self, file_path: str) -> list[Document]:
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load PDF '%s': %s", file_path, exc)
            return []
        metadata = self.build_metadata(file_path)
        for index, document in enumerate(pages, start=1):
            document.metadata.update(metadata)
            document.metadata["page_number"] = index
        return pages

    # -------------------------------------------------------------------------
    def load_docx(self, file_path: str) -> list[Document]:
        try:
            with zipfile.ZipFile(file_path) as archive:
                xml_content = archive.read("word/document.xml")
        except (KeyError, zipfile.BadZipFile, OSError) as exc:
            logger.error("Unable to read DOCX '%s': %s", file_path, exc)
            return []
        try:
            tree = ElementTree.fromstring(xml_content)
        except ElementTree.ParseError as exc:
            logger.error("Failed to parse DOCX '%s': %s", file_path, exc)
            return []
        namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paragraphs: list[str] = []
        for paragraph in tree.iter(f"{namespace}p"):
            texts = [
                node.text
                for node in paragraph.iter(f"{namespace}t")
                if node.text and node.text.strip()
            ]
            if texts:
                paragraphs.append("".join(texts))
        content = "\n".join(paragraphs).strip()
        if not content:
            return []
        document = Document(page_content=content, metadata=self.build_metadata(file_path))
        return [document]

    # -------------------------------------------------------------------------
    def load_textual_file(self, file_path: str, extension: str) -> list[Document]:
        text = self.read_text_content(file_path, extension)
        if not text:
            return []
        document = Document(page_content=text, metadata=self.build_metadata(file_path))
        return [document]

    # -------------------------------------------------------------------------
    def read_text_content(self, file_path: str, extension: str) -> str:
        if extension == ".xml":
            return self.read_xml_content(file_path)
        for encoding in TEXT_FILE_FALLBACK_ENCODINGS:
            try:
                with open(file_path, "r", encoding=encoding) as handle:
                    text = handle.read()
            except (OSError, UnicodeDecodeError):
                continue
            return text.strip()
        logger.error("Failed to read text file '%s'", file_path)
        return ""

    # -------------------------------------------------------------------------
    def read_xml_content(self, file_path: str) -> str:
        try:
            tree = ElementTree.parse(file_path)
            root = tree.getroot()
            text = " ".join(segment.strip() for segment in root.itertext())
            return text.strip()
        except (OSError, ElementTree.ParseError) as exc:
            logger.error("Failed to parse XML '%s': %s", file_path, exc)
        return ""

    # -------------------------------------------------------------------------
    def build_metadata(self, file_path: str) -> dict[str, Any]:
        document_id = self.compute_document_id(file_path)
        return {
            "document_id": document_id,
            "source": file_path,
            "file_name": os.path.basename(file_path),
        }

    # -------------------------------------------------------------------------
    def compute_document_id(self, file_path: str) -> str:
        relative_path = os.path.relpath(file_path, self.documents_path)
        return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()


###############################################################################
class DocumentChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = max(chunk_size, 1)
        self.chunk_overlap = max(chunk_overlap, 0)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )

    # -------------------------------------------------------------------------
    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        chunks = self.splitter.split_documents(documents)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
        return chunks


###############################################################################
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
            DEFAULT_EMBEDDING_BATCH_SIZE if embedding_batch_size is None else embedding_batch_size
        )
        self.embedding_batch_size = max(int(resolved_batch_size), 1)
        resolved_workers = (
            server_settings.rag.embedding_max_workers
            if embedding_workers is None
            else embedding_workers
        )
        self.embedding_workers = max(int(resolved_workers), 1)

    # -------------------------------------------------------------------------
    def serialize(self, reset_collection: bool = False) -> dict[str, int]:
        self.vector_database.initialize(False)
        self.vector_database.get_table()
        documents = self.document_serializer.load_documents()
        if not documents:
            logger.warning("No documents available for embedding serialization")
            return {"documents": 0, "chunks": 0}
        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            logger.warning("Document chunking resulted in zero chunks")
            return {"documents": 0, "chunks": 0}
        reset_pending = bool(reset_collection)
        reset_applied = False
        batch_size = self.embedding_batch_size
        total_records = 0
        document_ids: set[str] = set()
        batch_iter: list[list[Document]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            if batch:
                batch_iter.append(batch)
        with ThreadPoolExecutor(max_workers=self.embedding_workers) as executor:
            futures = [
                executor.submit(self._embed_chunk_batch, batch_chunks)
                for batch_chunks in batch_iter
            ]
            for future in as_completed(futures):
                records, batch_ids = future.result()
                document_ids.update(batch_ids)
                if not records:
                    continue
                if reset_pending and not reset_applied:
                    self.vector_database.initialize(True)
                    reset_applied = True
                self.vector_database.upsert_embeddings(records)
                total_records += len(records)
        logger.info(
            "Serialized %d documents into %d vector chunks",
            len(document_ids),
            total_records,
        )
        return {"documents": len(document_ids), "chunks": total_records}

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
                f"{document_id}:{chunk_index}" if chunk_index is not None else document_id
            )
            metadata = self.serialize_metadata(chunk.metadata)
            records.append(
                {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "text": chunk.page_content,
                    "embedding": embedding,
                    "source": metadata.get("source", ""),
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


    
