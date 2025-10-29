from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from typing import Any
from xml.etree import ElementTree

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.exc import SQLAlchemyError

from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.database import ClinicalSession, database
from DILIGENT.app.utils.repository.vectors import LanceVectorDatabase
from DILIGENT.app.utils.services.embeddings import EmbeddingGenerator


# [DATA SERIALIZATION]
###############################################################################
LIVERTOX_COLUMNS = [
    "drug_name",
    "nbk_id",
    "excerpt",
    "likelihood_score",
    "last_update",
    "reference_count",
    "year_approved",
    "agent_classification",
    "primary_classification",
    "secondary_classification",
    "include_in_livertox",
    "source_url",
    "source_last_modified",
]

FDA_COLUMNS = [
    "report_id",
    "case_version",
    "receipt_date",
    "occur_country",
    "patient_age",
    "patient_age_unit",
    "patient_sex",
    "reaction_terms",
    "all_reactions",
    "suspect_products",
    "suspect_product_count",
    "serious",
    "seriousness_death",
    "seriousness_lifethreatening",
    "seriousness_hospitalization",
    "seriousness_disabling",
    "seriousness_congenital_anom",
    "seriousness_other",
]

HEPATOTOXIC_MEDDRA_TERMS = {
    "hepatotoxicity",
    "drug induced liver injury",
    "drug-induced liver injury",
    "liver injury",
    "hepatic failure",
    "acute hepatic failure",
    "hepatitis cholestatic",
    "cholestasis",
    "liver disorder",
    "liver function test increased",
    "alanine aminotransferase increased",
    "aspartate aminotransferase increased",
    "alkaline phosphatase increased",
    "blood bilirubin increased",
}


###############################################################################
class DataSerializer:
    def __init__(self) -> None:
        pass

    # -------------------------------------------------------------------------
    def record_clinical_session(self, session_data: dict[str, Any]) -> None:
        session = database.Session()
        try:
            session.add(ClinicalSession(**session_data))
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            logger.error("Failed to record clinical session: %s", exc)
            raise exc
        finally:
            session.close()

    # -----------------------------------------------------------------------------
    def save_livertox_records(self, records: pd.DataFrame) -> None:
        frame = records.copy()
        if "drug_name" in frame.columns:
            frame = frame.drop_duplicates(subset=["drug_name"], keep="first")
        frame = frame.reindex(columns=LIVERTOX_COLUMNS)
        frame = frame.where(pd.notnull(frame), None)
        database.save_into_database(frame, "LIVERTOX_DATA")
   
    # -----------------------------------------------------------------------------
    def upsert_drugs_catalog_records(
        self, records: pd.DataFrame | list[dict[str, Any]]
    ) -> None:
        if isinstance(records, pd.DataFrame):
            frame = records.copy()
        else:
            frame = pd.DataFrame(records)
        frame = frame.reindex(
            columns=[
                "rxcui",
                "raw_name",
                "term_type",
                "name",
                "brand_names",
                "synonyms",
            ]
        )
        if frame.empty:
            return
        frame = frame.where(pd.notnull(frame), None)
        frame["brand_names"] = frame["brand_names"].apply(self.serialize_brand_names)
        frame["synonyms"] = frame["synonyms"].apply(self.serialize_string_list)
        database.upsert_into_database(frame, "DRUGS_CATALOG")

    # -----------------------------------------------------------------------------
    def sanitize_livertox_records(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        required_columns = [
            "nbk_id",
            "drug_name",
            "excerpt",
            "synonyms",
        ]
        if df.empty:
            return pd.DataFrame(columns=required_columns)
        for column in required_columns:
            if column not in df.columns:
                df[column] = None
        df = df[required_columns]
        allowed_missing = {"nbk_id", "synonyms"}
        drop_columns = [col for col in required_columns if col not in allowed_missing]
        df = df.dropna(subset=drop_columns)
        df["drug_name"] = df["drug_name"].astype(str).str.strip()
        df = df[df["drug_name"].apply(self.is_valid_drug_name)]
        df["excerpt"] = df["excerpt"].astype(str).str.strip()
        df = df[df["excerpt"] != ""]
        df["nbk_id"] = df["nbk_id"].apply(
            lambda value: str(value).strip() if pd.notna(value) else None
        )
        df["synonyms"] = df["synonyms"].apply(
            lambda value: (
                str(value).strip() if pd.notna(value) and str(value).strip() else None
            )
        )
        df = df.drop_duplicates(subset=["nbk_id", "drug_name"], keep="first")
        return df.reset_index(drop=True)

    # -----------------------------------------------------------------------------
    def is_valid_drug_name(self, value: str) -> bool:
        normalized = value.strip()
        if len(normalized) < 3 or len(normalized) > 200:
            return False
        if len(normalized.split()) > 8:
            return False
        if not re.fullmatch(r"[A-Za-z0-9\s\-/(),'+\.]+", normalized):
            return False
        return True

    # -----------------------------------------------------------------------------
    def serialize_string_list(self, value: Any) -> str:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return "[]"
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed_values = [stripped]
            else:
                if isinstance(parsed, list):
                    parsed_values = parsed
                else:
                    parsed_values = [stripped]
        elif isinstance(value, list):
            parsed_values = value
        elif pd.isna(value) or value is None:
            parsed_values = []
        else:
            parsed_values = [value]
        normalized: list[str] = []
        for item in parsed_values:
            normalized_item = self.normalize_list_item(item)
            if normalized_item is not None:
                normalized.append(normalized_item)
        return json.dumps(normalized, ensure_ascii=False)

    # -----------------------------------------------------------------------------
    def serialize_brand_names(self, value: Any) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return self.normalize_list_item(stripped)
            return self.serialize_brand_names(parsed)
        if isinstance(value, list):
            normalized: list[str] = []
            seen: set[str] = set()
            for item in value:
                normalized_item = self.normalize_list_item(item)
                if normalized_item is None:
                    continue
                key = normalized_item.casefold()
                if key in seen:
                    continue
                seen.add(key)
                normalized.append(normalized_item)
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
        if isinstance(value, str):
            normalized = value.strip()
            return normalized if normalized else None
        if pd.isna(value) or value is None:
            return None
        normalized = str(value).strip()
        return normalized if normalized else None

    # -----------------------------------------------------------------------------
    def deserialize_string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            parsed_values = value
        elif isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                normalized_item = self.normalize_list_item(stripped)
                return [normalized_item] if normalized_item is not None else []
            if isinstance(parsed, list):
                parsed_values = parsed
            else:
                normalized_item = self.normalize_list_item(parsed)
                return [normalized_item] if normalized_item is not None else []
        elif pd.isna(value) or value is None:
            return []
        else:
            parsed_values = [value]
        normalized: list[str] = []
        for item in parsed_values:
            normalized_item = self.normalize_list_item(item)
            if normalized_item is not None:
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
            return pd.DataFrame(
                columns=[
                    "drug_name",
                    "likelihood_score",
                    "last_update",
                    "reference_count",
                    "year_approved",
                    "agent_classification",
                    "primary_classification",
                    "secondary_classification",
                    "include_in_livertox",
                    "source_url",
                    "source_last_modified",
                ]
            )
        alias_columns = [
            "drug_name",
            "likelihood_score",
            "last_update",
            "reference_count",
            "year_approved",
            "agent_classification",
            "primary_classification",
            "secondary_classification",
            "include_in_livertox",
            "source_url",
            "source_last_modified",
        ]
        available = [column for column in alias_columns if column in frame.columns]
        if not available:
            return pd.DataFrame(columns=["drug_name"])
        return frame.reindex(columns=available).dropna(subset=["drug_name"]).reset_index(
            drop=True
        )

    # -----------------------------------------------------------------------------
    def get_drugs_catalog(self) -> pd.DataFrame:
        frame = database.load_from_database("DRUGS_CATALOG")
        columns = [
            "rxcui",
            "raw_name",
            "term_type",
            "name",
            "brand_names",
            "synonyms",
        ]
        if frame.empty:
            return pd.DataFrame(columns=columns)
        return frame.reindex(columns=columns)

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
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".xml", ".docx", ".doc"}

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
        encodings = ["utf-8", "utf-16", "latin-1", "iso-8859-1"]
        for encoding in encodings:
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
            hf_model=hf_model,
            use_cloud_embeddings=use_cloud_embeddings,
            cloud_provider=cloud_provider,
            cloud_embedding_model=cloud_embedding_model,
        )

    # -------------------------------------------------------------------------
    def serialize(self, reset_collection: bool = False) -> dict[str, int]:
        documents = self.document_serializer.load_documents()
        if not documents:
            logger.warning("No documents available for embedding serialization")
            self.vector_database.initialize(reset_collection)
            return {"documents": 0, "chunks": 0}
        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            logger.warning("Document chunking resulted in zero chunks")
            self.vector_database.initialize(reset_collection)
            return {"documents": 0, "chunks": 0}
        embeddings = self.embedding_generator.embed_texts(
            [chunk.page_content for chunk in chunks]
        )
        if len(embeddings) != len(chunks):
            raise RuntimeError("Embedding count does not match chunk count")
        records = self.build_records(chunks, embeddings)
        self.vector_database.initialize(reset_collection)
        if records:
            self.vector_database.upsert_embeddings(records)
        document_ids = {
            str(chunk.metadata.get("document_id"))
            for chunk in chunks
            if chunk.metadata.get("document_id")
        }
        logger.info(
            "Serialized %d documents into %d vector chunks",
            len(document_ids),
            len(records),
        )
        return {"documents": len(document_ids), "chunks": len(records)}

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


    
