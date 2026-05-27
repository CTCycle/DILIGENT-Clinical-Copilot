from __future__ import annotations

import hashlib
import os
import re
import zipfile
from datetime import date
from typing import Any, Iterator
from xml.etree import ElementTree

import pandas as pd
from pypdf import PdfReader
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from common.constants import (
    DOCUMENT_SUPPORTED_EXTENSIONS,
    TEXT_FILE_FALLBACK_ENCODINGS,
)
from common.utils.logger import logger
from domain.documents import Document
from repositories.database.session import (
    resolve_engine,
    resolve_session_factory,
)
from repositories.schemas.models import (
    Drug,
    DrugAlias,
    LiverToxMonograph,
    Patient,
)
from repositories.serialization import (
    evidence_aliases,
    evidence_data,
    fda_data,
    session_result_data,
)


class _RepositorySerializationService:
    def __init__(
        self,
        *,
        engine: Engine | None = None,
        session_factory: sessionmaker | None = None,
    ) -> None:
        self.engine = resolve_engine(engine)
        self.session_factory = resolve_session_factory(
            engine=self.engine,
            session_factory=session_factory,
        )

    # -------------------------------------------------------------------------
    def save_clinical_session(self, session_data: dict[str, Any]) -> int | None:
        return session_result_data.save_clinical_session(self, session_data)

    # -----------------------------------------------------------------------------
    def ensure_session_result_table(self) -> None:
        return session_result_data.ensure_session_result_table(self)

    # -----------------------------------------------------------------------------
    def normalize_session_status(self, value: Any) -> str:
        return session_result_data.normalize_session_status(self, value)

    # -----------------------------------------------------------------------------
    def persist_patient(
        self, db_session: Session, session_data: dict[str, Any]
    ) -> Patient:
        return session_result_data.persist_patient(self, db_session, session_data)

    # -----------------------------------------------------------------------------
    def decode_patient_image(self, value: Any) -> bytes | None:
        return session_result_data.decode_patient_image(self, value)

    # -----------------------------------------------------------------------------
    def save_livertox_records(self, records: pd.DataFrame) -> None:
        return evidence_data.save_livertox_records(self, records)

    # -----------------------------------------------------------------------------
    def prepare_livertox_rows(self, records: pd.DataFrame) -> list[dict[str, Any]]:
        return evidence_data.prepare_livertox_rows(self, records)

    # -----------------------------------------------------------------------------
    def livertox_row_sort_key(self, row: dict[str, Any]) -> tuple[str, ...]:
        return evidence_data.livertox_row_sort_key(self, row)

    # -----------------------------------------------------------------------------
    def to_sortable_text(self, value: Any) -> str:
        return evidence_data.to_sortable_text(self, value)

    # -----------------------------------------------------------------------------
    def upsert_livertox_monograph(
        self,
        *,
        db_session: Session,
        drug_id: int,
        row: dict[str, Any],
    ) -> None:
        return evidence_data.upsert_livertox_monograph(self, db_session=db_session, drug_id=drug_id, row=row)

    # -----------------------------------------------------------------------------
    def try_assign_livertox_nbk_id(
        self,
        db_session: Session,
        *,
        drug: Drug,
        livertox_nbk_id: str,
    ) -> None:
        return evidence_data.try_assign_livertox_nbk_id(self, db_session, drug=drug, livertox_nbk_id=livertox_nbk_id)

    # -----------------------------------------------------------------------------
    def build_livertox_monograph_key(self, row: dict[str, Any]) -> str:
        return evidence_data.build_livertox_monograph_key(self, row)

    # -----------------------------------------------------------------------------
    def upsert_drugs_catalog_records(
        self,
        records: pd.DataFrame | list[dict[str, Any]],
        *,
        commit_interval: int | None = None,
        curated_aliases_by_canonical: dict[str, list[tuple[str, str]]] | None = None,
    ) -> None:
        return fda_data.upsert_drugs_catalog_records(self, records, commit_interval=commit_interval, curated_aliases_by_canonical=curated_aliases_by_canonical)

    # -----------------------------------------------------------------------------
    def resolve_commit_interval(self, override: int | None) -> int:
        return fda_data.resolve_commit_interval(self, override)

    # -----------------------------------------------------------------------------
    def prepare_rxnav_rows(
        self,
        records: pd.DataFrame | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return fda_data.prepare_rxnav_rows(self, records)

    # -----------------------------------------------------------------------------
    def prepare_rxnav_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        return fda_data.prepare_rxnav_row(self, row)

    # -----------------------------------------------------------------------------
    def rxnav_row_sort_key(self, row: dict[str, Any]) -> tuple[str, ...]:
        return fda_data.rxnav_row_sort_key(self, row)

    # -----------------------------------------------------------------------------
    def sanitize_livertox_records(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        return fda_data.sanitize_livertox_records(self, records)

    # -----------------------------------------------------------------------------
    def is_valid_drug_name(self, value: str) -> bool:
        return fda_data.is_valid_drug_name(self, value)

    # -----------------------------------------------------------------------------
    def get_livertox_records(self) -> pd.DataFrame:
        return evidence_data.get_livertox_records(self)

    # -----------------------------------------------------------------------------
    def get_livertox_master_list(self) -> pd.DataFrame:
        return evidence_data.get_livertox_master_list(self)

    # -----------------------------------------------------------------------------
    def get_drugs_catalog(self) -> pd.DataFrame:
        return evidence_data.get_drugs_catalog(self)

    # -----------------------------------------------------------------------------
    def stream_drugs_catalog(
        self, page_size: int | None = None
    ) -> Iterator[pd.DataFrame]:
        return evidence_data.stream_drugs_catalog(self, page_size)

    # -----------------------------------------------------------------------------
    def build_search_pattern(self, search: str | None) -> str | None:
        return evidence_data.build_search_pattern(self, search)

    # -----------------------------------------------------------------------------
    def list_sessions(
        self,
        *,
        search: str | None,
        status_filter: str | None,
        date_mode: str | None,
        filter_date: date | None,
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        return session_result_data.list_sessions(self, search=search, status_filter=status_filter, date_mode=date_mode, filter_date=filter_date, offset=offset, limit=limit)

    # -----------------------------------------------------------------------------
    def parse_session_result_payload(
        self, payload_json: str | None
    ) -> dict[str, Any] | None:
        return session_result_data.parse_session_result_payload(self, payload_json)

    # -----------------------------------------------------------------------------
    def get_session_result_payload(self, session_id: int) -> dict[str, Any] | None:
        return session_result_data.get_session_result_payload(self, session_id)

    # -----------------------------------------------------------------------------
    def get_session_detail(self, session_id: int) -> dict[str, Any] | None:
        return session_result_data.get_session_detail(self, session_id)

    # -----------------------------------------------------------------------------
    def build_session_text_from_sections(self, sections: dict[str, str]) -> str:
        return session_result_data.build_session_text_from_sections(self, sections)

    # -----------------------------------------------------------------------------
    def upsert_session_result_payload(
        self, session_id: int, payload: dict[str, Any]
    ) -> bool:
        return session_result_data.upsert_session_result_payload(self, session_id, payload)

    # -----------------------------------------------------------------------------
    def update_session_text_and_metadata(
        self,
        session_id: int,
        *,
        session_text: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return session_result_data.update_session_text_and_metadata(
            self,
            session_id,
            session_text=session_text,
            metadata=metadata,
        )

    # -----------------------------------------------------------------------------
    def get_next_session_version(self, original_session_id: int) -> int:
        return session_result_data.get_next_session_version(self, original_session_id)

    # -----------------------------------------------------------------------------
    def get_session_timeline_source(self, session_id: int) -> dict[str, Any] | None:
        return session_result_data.get_session_timeline_source(self, session_id)

    # -----------------------------------------------------------------------------
    def delete_session(self, session_id: int) -> bool:
        return session_result_data.delete_session(self, session_id)

    # -----------------------------------------------------------------------------
    def list_rxnav_catalog(
        self,
        *,
        search: str | None,
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        return evidence_data.list_rxnav_catalog(self, search=search, offset=offset, limit=limit)

    # -----------------------------------------------------------------------------
    def get_rxnav_alias_groups(self, drug_id: int) -> dict[str, Any] | None:
        return evidence_data.get_rxnav_alias_groups(self, drug_id)

    # -----------------------------------------------------------------------------
    def list_livertox_catalog(
        self,
        *,
        search: str | None,
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        return evidence_data.list_livertox_catalog(self, search=search, offset=offset, limit=limit)

    # -----------------------------------------------------------------------------
    def get_livertox_excerpt(self, drug_id: int) -> dict[str, Any] | None:
        return evidence_data.get_livertox_excerpt(self, drug_id)

    # -----------------------------------------------------------------------------
    def get_drug_knowledge_bundle(self, drug_id: int) -> dict[str, Any]:
        return evidence_data.get_drug_knowledge_bundle(self, drug_id)

    # -----------------------------------------------------------------------------
    def delete_drug_with_cleanup(self, drug_id: int) -> bool:
        return evidence_data.delete_drug_with_cleanup(self, drug_id)

    # -----------------------------------------------------------------------------
    def normalize_string(self, value: Any) -> str | None:
        return session_result_data.normalize_string(self, value)

    # -----------------------------------------------------------------------------
    def normalize_flag(self, value: Any) -> int | None:
        return session_result_data.normalize_flag(self, value)

    # -----------------------------------------------------------------------------
    def normalize_date(self, value: Any) -> str | None:
        return session_result_data.normalize_date(self, value)

    # -----------------------------------------------------------------------------
    def normalize_date_value(self, value: Any) -> date | None:
        return session_result_data.normalize_date_value(self, value)

    # -----------------------------------------------------------------------------
    def join_values(self, values: set[str]) -> str | None:
        return session_result_data.join_values(self, values)

    # -----------------------------------------------------------------------------
    def to_int(self, value: Any) -> int | None:
        return session_result_data.to_int(self, value)

    # -----------------------------------------------------------------------------
    def to_float(self, value: Any) -> float | None:
        return session_result_data.to_float(self, value)

    # -----------------------------------------------------------------------------
    def parse_datetime(self, value: Any) -> Any:
        return session_result_data.parse_datetime(self, value)

    # -----------------------------------------------------------------------------
    def persist_session_sections(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        return session_result_data.persist_session_sections(self, db_session, session_id, session_data)

    # -----------------------------------------------------------------------------
    def persist_session_labs(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        return session_result_data.persist_session_labs(self, db_session, session_id, session_data)

    # -----------------------------------------------------------------------------
    def persist_session_drugs(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        return session_result_data.persist_session_drugs(self, db_session, session_id, session_data)

    # -----------------------------------------------------------------------------
    def resolve_drug_id_from_match_cache(
        self,
        db_session: Session,
        *,
        normalized_drug_key: str,
    ) -> int | None:
        return evidence_data.resolve_drug_id_from_match_cache(self, db_session, normalized_drug_key=normalized_drug_key)

    # -----------------------------------------------------------------------------
    def upsert_high_confidence_kb_match_cache(
        self,
        db_session: Session,
        *,
        raw_drug_name: str,
        raw_drug_name_norm: str,
        normalized_drug_key: str,
        drug_id: int | None,
        rxnorm_rxcui: str | None,
        livertox_nbk_id: str | None,
        source: str,
        confidence: float | None,
        evidence: dict[str, Any],
        ambiguous: bool,
    ) -> None:
        return evidence_data.upsert_high_confidence_kb_match_cache(self, db_session, raw_drug_name=raw_drug_name, raw_drug_name_norm=raw_drug_name_norm, normalized_drug_key=normalized_drug_key, drug_id=drug_id, rxnorm_rxcui=rxnorm_rxcui, livertox_nbk_id=livertox_nbk_id, source=source, confidence=confidence, evidence=evidence, ambiguous=ambiguous)

    # -----------------------------------------------------------------------------
    def persist_session_result_payload(
        self, db_session: Session, session_id: int, session_data: dict[str, Any]
    ) -> None:
        return session_result_data.persist_session_result_payload(self, db_session, session_id, session_data)

    # -----------------------------------------------------------------------------
    def serialize_json_payload(self, payload: Any) -> str | None:
        return session_result_data.serialize_json_payload(self, payload)

    # -----------------------------------------------------------------------------
    def resolve_drug_id(
        self,
        db_session: Session,
        *,
        matched_drug_name: str | None,
        rxcui: str | None,
        nbk_id: str | None,
    ) -> int | None:
        return evidence_aliases.resolve_drug_id(self, db_session, matched_drug_name=matched_drug_name, rxcui=rxcui, nbk_id=nbk_id)

    # -----------------------------------------------------------------------------
    def ensure_drug(
        self,
        db_session: Session,
        *,
        canonical_name: str,
        canonical_name_norm: str,
        rxnorm_rxcui: str | None,
        livertox_nbk_id: str | None,
        rxnav_last_update: str | None = None,
        use_livertox_nbk_lookup: bool = True,
    ) -> Drug:
        return evidence_aliases.ensure_drug(self, db_session, canonical_name=canonical_name, canonical_name_norm=canonical_name_norm, rxnorm_rxcui=rxnorm_rxcui, livertox_nbk_id=livertox_nbk_id, rxnav_last_update=rxnav_last_update, use_livertox_nbk_lookup=use_livertox_nbk_lookup)

    # -----------------------------------------------------------------------------
    def assign_primary_rxcui_if_missing(
        self,
        *,
        drug: Drug,
        incoming_rxcui: str | None,
    ) -> None:
        return evidence_aliases.assign_primary_rxcui_if_missing(self, drug=drug, incoming_rxcui=incoming_rxcui)

    # -----------------------------------------------------------------------------
    def assign_identifier_if_consistent(
        self,
        *,
        drug: Drug,
        field_name: str,
        incoming_value: str | None,
    ) -> None:
        return evidence_aliases.assign_identifier_if_consistent(self, drug=drug, field_name=field_name, incoming_value=incoming_value)

    # -----------------------------------------------------------------------------
    def upsert_drug_rxcui(
        self,
        db_session: Session,
        *,
        drug_id: int,
        rxcui: str | None,
    ) -> None:
        return evidence_aliases.upsert_drug_rxcui(self, db_session, drug_id=drug_id, rxcui=rxcui)

    # -----------------------------------------------------------------------------
    def get_drug_by_rxcui(
        self,
        db_session: Session,
        rxcui: str | None,
    ) -> Drug | None:
        return evidence_aliases.get_drug_by_rxcui(self, db_session, rxcui)

    # -----------------------------------------------------------------------------
    def get_drug_by_canonical_name_norm(
        self,
        db_session: Session,
        canonical_name_norm: str | None,
    ) -> Drug | None:
        return evidence_aliases.get_drug_by_canonical_name_norm(self, db_session, canonical_name_norm)

    # -----------------------------------------------------------------------------
    def get_drug_alias_by_norm(
        self,
        db_session: Session,
        alias_norm: str | None,
    ) -> DrugAlias | None:
        return evidence_aliases.get_drug_alias_by_norm(self, db_session, alias_norm)

    # -----------------------------------------------------------------------------
    def get_monograph_by_drug_id(
        self,
        db_session: Session,
        drug_id: int,
    ) -> LiverToxMonograph | None:
        return evidence_aliases.get_monograph_by_drug_id(self, db_session, drug_id)

    # -----------------------------------------------------------------------------
    def get_monograph_by_key(
        self,
        db_session: Session,
        monograph_key: str,
    ) -> LiverToxMonograph | None:
        return evidence_aliases.get_monograph_by_key(self, db_session, monograph_key)

    # -----------------------------------------------------------------------------
    def upsert_drug_alias(
        self,
        db_session: Session,
        *,
        drug_id: int,
        alias: str,
        alias_kind: str,
        source: str,
        term_type: str | None,
    ) -> None:
        return evidence_aliases.upsert_drug_alias(self, db_session, drug_id=drug_id, alias=alias, alias_kind=alias_kind, source=source, term_type=term_type)

    # -----------------------------------------------------------------------------
    def persist_livertox_aliases(
        self, db_session: Session, drug_id: int, row: dict[str, Any]
    ) -> None:
        return evidence_aliases.persist_livertox_aliases(self, db_session, drug_id, row)

    # -----------------------------------------------------------------------------
    def extract_text_candidates(self, value: Any) -> list[str]:
        return evidence_aliases.extract_text_candidates(self, value)

    # -----------------------------------------------------------------------------
    def extract_synonym_candidates(self, value: Any) -> list[str]:
        return evidence_aliases.extract_synonym_candidates(self, value)

    # -----------------------------------------------------------------------------
    def unique_text(self, values: list[str]) -> list[str]:
        return evidence_aliases.unique_text(self, values)

    # -----------------------------------------------------------------------------
    def build_alias_lookup_by_kind(
        self, aliases_frame: pd.DataFrame
    ) -> dict[int, dict[str, set[str]]]:
        return evidence_aliases.build_alias_lookup_by_kind(self, aliases_frame)

    # -----------------------------------------------------------------------------
    def group_aliases_by_kind(self, aliases: list[DrugAlias]) -> dict[str, set[str]]:
        return evidence_aliases.group_aliases_by_kind(self, aliases)

    # -----------------------------------------------------------------------------
    def alias_values_for_kind(self, aliases: pd.DataFrame, alias_kind: str) -> set[str]:
        return evidence_aliases.alias_values_for_kind(self, aliases, alias_kind)

    # -----------------------------------------------------------------------------
    def alias_model_values_for_kind(
        self,
        aliases: list[DrugAlias],
        alias_kind: str,
    ) -> set[str]:
        return evidence_aliases.alias_model_values_for_kind(self, aliases, alias_kind)

    # -----------------------------------------------------------------------------
    def first_alias_value(self, aliases: pd.DataFrame, alias_kind: str) -> str | None:
        return evidence_aliases.first_alias_value(self, aliases, alias_kind)

    # -----------------------------------------------------------------------------
    def first_alias_term_type(self, aliases: pd.DataFrame) -> str | None:
        return evidence_aliases.first_alias_term_type(self, aliases)

    # -----------------------------------------------------------------------------
    def first_alias_model_value(
        self,
        aliases: list[DrugAlias],
        alias_kind: str,
    ) -> str | None:
        return evidence_aliases.first_alias_model_value(self, aliases, alias_kind)

    # -----------------------------------------------------------------------------
    def first_alias_model_term_type(self, aliases: list[DrugAlias]) -> str | None:
        return evidence_aliases.first_alias_model_term_type(self, aliases)


###############################################################################
class DataSerializer:
    def __init__(
        self,
        *,
        engine: Engine | None = None,
        session_factory: sessionmaker | None = None,
    ) -> None:
        self.service = _RepositorySerializationService(
            engine=engine,
            session_factory=session_factory,
        )

    # -------------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(self.service, name)


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
                    "Unsupported .doc Word document '%s' is not supported; skipping",
                    file_path,
                )
            elif extension in {".txt", ".xml"}:
                documents.extend(self.load_textual_file(file_path, extension))
        return documents

    # -------------------------------------------------------------------------
    def load_pdf(self, file_path: str) -> list[Document]:
        try:
            reader = PdfReader(file_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load PDF '%s': %s", file_path, exc)
            return []

        metadata = self.build_metadata(
            file_path,
            content_type="pdf",
            document_title=self.resolve_pdf_title(reader, file_path),
        )
        pages: list[Document] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to extract text from '%s' page %d: %s",
                    file_path,
                    index,
                    exc,
                )
                continue
            content = text.strip()
            if not content:
                continue
            page_metadata = dict(metadata)
            page_metadata["page_number"] = index
            pages.append(Document(page_content=content, metadata=page_metadata))
        return pages

    # -------------------------------------------------------------------------
    def load_docx(self, file_path: str) -> list[Document]:
        try:
            with zipfile.ZipFile(file_path) as archive:
                xml_content = archive.read("word/document.xml")
                title = self.resolve_docx_title(archive, file_path)
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
        metadata = self.build_metadata(
            file_path,
            content_type="docx",
            document_title=title or self.extract_first_heading(content),
        )
        document = Document(page_content=content, metadata=metadata)
        return [document]

    # -------------------------------------------------------------------------
    def load_textual_file(self, file_path: str, extension: str) -> list[Document]:
        text = self.read_text_content(file_path, extension)
        if not text:
            return []
        document = Document(
            page_content=text,
            metadata=self.build_metadata(
                file_path,
                content_type=extension.lstrip("."),
                document_title=self.extract_first_heading(text),
            ),
        )
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
    def build_metadata(
        self,
        file_path: str,
        *,
        content_type: str,
        document_title: str | None = None,
    ) -> dict[str, Any]:
        document_id = self.compute_document_id(file_path)
        resolved_title = self.normalize_title(document_title) or os.path.splitext(
            os.path.basename(file_path)
        )[0]
        return {
            "document_id": document_id,
            "source": file_path,
            "file_name": os.path.basename(file_path),
            "document_title": resolved_title,
            "content_type": content_type,
        }

    # -------------------------------------------------------------------------
    def compute_document_id(self, file_path: str) -> str:
        relative_path = os.path.relpath(file_path, self.documents_path)
        return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    def resolve_pdf_title(self, reader: PdfReader, file_path: str) -> str:
        raw_title = getattr(getattr(reader, "metadata", None), "title", None)
        normalized = self.normalize_title(raw_title)
        if normalized:
            return normalized
        for page in reader.pages[:2]:
            try:
                candidate = self.extract_first_heading(page.extract_text() or "")
            except Exception:  # noqa: BLE001
                candidate = None
            if candidate:
                return candidate
        return os.path.splitext(os.path.basename(file_path))[0]

    # -------------------------------------------------------------------------
    def resolve_docx_title(self, archive: zipfile.ZipFile, file_path: str) -> str:
        try:
            core_xml = archive.read("docProps/core.xml")
            tree = ElementTree.fromstring(core_xml)
        except (KeyError, ElementTree.ParseError):
            return os.path.splitext(os.path.basename(file_path))[0]
        namespaces = {"dc": "http://purl.org/dc/elements/1.1/"}
        node = tree.find("dc:title", namespaces)
        return self.normalize_title(node.text if node is not None else None) or os.path.splitext(
            os.path.basename(file_path)
        )[0]

    # -------------------------------------------------------------------------
    def extract_first_heading(self, text: str) -> str | None:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if self.is_heading_line(line):
                return self.normalize_title(line)
        return None

    # -------------------------------------------------------------------------
    def is_heading_line(self, line: str) -> bool:
        if len(line) > 120:
            return False
        if line.startswith("#"):
            return True
        if re.match(r"^\d+(\.\d+)*\s+\S+", line):
            return True
        words = line.split()
        return 1 <= len(words) <= 12 and line == line.upper()

    # -------------------------------------------------------------------------
    def normalize_title(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return re.sub(r"\s+", " ", text)


###############################################################################
class DocumentChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = max(chunk_size, 1)
        self.chunk_overlap = max(chunk_overlap, 0)

    # -------------------------------------------------------------------------
    def split_text(self, content: str) -> list[tuple[str, int, str | None, str | None]]:
        text = content.strip()
        if not text:
            return []
        sections = self.split_sections(text)
        chunks: list[tuple[str, int, str | None, str | None]] = []
        for section_text, section_start, section_title, heading_path in sections:
            for chunk_text, relative_start in self.split_section(section_text):
                chunks.append(
                    (
                        chunk_text,
                        section_start + relative_start,
                        section_title,
                        heading_path,
                    )
                )
        return chunks

    # -------------------------------------------------------------------------
    def split_sections(self, text: str) -> list[tuple[str, int, str | None, str | None]]:
        sections: list[tuple[str, int, str | None, str | None]] = []
        heading_stack: list[str] = []
        current_lines: list[str] = []
        current_start = 0
        current_title: str | None = None
        offset = 0
        for raw_line in text.splitlines(keepends=True):
            line = raw_line.strip()
            if self.is_heading_line(line):
                if current_lines:
                    sections.append(
                        (
                            "".join(current_lines).strip(),
                            current_start,
                            current_title,
                            " > ".join(heading_stack) or None,
                        )
                    )
                    current_lines = []
                heading = self.normalize_heading(line)
                heading_stack = [heading]
                current_title = heading
                current_start = offset
            current_lines.append(raw_line)
            offset += len(raw_line)
        if current_lines:
            sections.append(
                (
                    "".join(current_lines).strip(),
                    current_start,
                    current_title,
                    " > ".join(heading_stack) or None,
                )
            )
        return sections

    # -------------------------------------------------------------------------
    def split_section(self, section_text: str) -> list[tuple[str, int]]:
        if len(section_text) <= self.chunk_size:
            return [(section_text, 0)]
        paragraphs = re.split(r"(\n\s*\n)", section_text)
        chunks: list[tuple[str, int]] = []
        buffer = ""
        buffer_start = 0
        cursor = 0
        for part in paragraphs:
            if not part:
                continue
            candidate = f"{buffer}{part}"
            if buffer and len(candidate) > self.chunk_size:
                chunks.extend(self.split_oversized_text(buffer, buffer_start))
                buffer = part.lstrip()
                buffer_start = cursor + (len(part) - len(part.lstrip()))
            else:
                if not buffer:
                    buffer_start = cursor
                buffer = candidate
            cursor += len(part)
        if buffer.strip():
            chunks.extend(self.split_oversized_text(buffer, buffer_start))
        return chunks

    # -------------------------------------------------------------------------
    def split_oversized_text(self, text: str, start_offset: int) -> list[tuple[str, int]]:
        normalized = text.strip()
        if len(normalized) <= self.chunk_size:
            return [(normalized, start_offset)]
        step = max(self.chunk_size - self.chunk_overlap, 1)
        chunks: list[tuple[str, int]] = []
        start = 0
        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            chunk_text = normalized[start:end].strip()
            if chunk_text:
                chunks.append((chunk_text, start_offset + start))
            if end >= len(normalized):
                break
            start += step
        return chunks

    # -------------------------------------------------------------------------
    def is_heading_line(self, line: str) -> bool:
        if not line or len(line) > 120:
            return False
        if line.startswith("#"):
            return True
        if re.match(r"^\d+(\.\d+)*\s+\S+", line):
            return True
        words = line.split()
        return 1 <= len(words) <= 12 and line == line.upper()

    # -------------------------------------------------------------------------
    def normalize_heading(self, line: str) -> str:
        stripped = line.lstrip("#").strip()
        return re.sub(r"\s+", " ", stripped)

    # -------------------------------------------------------------------------
    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        chunks: list[Document] = []
        for document in documents:
            metadata = dict(document.metadata)
            for chunk_text, start_index, section_title, heading_path in self.split_text(
                document.page_content
            ):
                chunk_metadata = dict(metadata)
                chunk_metadata["start_index"] = start_index
                chunk_metadata["section_title"] = section_title
                chunk_metadata["heading_path"] = heading_path
                chunks.append(
                    Document(page_content=chunk_text, metadata=chunk_metadata)
                )
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
        return chunks


