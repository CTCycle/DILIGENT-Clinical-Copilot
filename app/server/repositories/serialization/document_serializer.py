from __future__ import annotations

import hashlib
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from pypdf import PdfReader

from common.constants import (
    DOCUMENT_SUPPORTED_EXTENSIONS,
    TEXT_FILE_FALLBACK_ENCODINGS,
)
from common.utils.logger import logger
from domain.documents import Document

###############################################################################
class DocumentSerializer:
    SUPPORTED_EXTENSIONS = DOCUMENT_SUPPORTED_EXTENSIONS

    # -------------------------------------------------------------------------
    def __init__(self, documents_path: str | Path) -> None:
        self.documents_path = Path(documents_path)

    # -------------------------------------------------------------------------
    def collect_document_paths(self) -> list[str]:
        collected: list[str] = []
        for candidate in self.documents_path.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                collected.append(str(candidate))
            else:
                logger.debug("Skipping unsupported document '%s'", candidate.name)
        collected.sort()
        return collected

    # -------------------------------------------------------------------------
    def load_documents(self) -> list[Document]:
        documents: list[Document] = []
        for file_path in self.collect_document_paths():
            extension = Path(file_path).suffix.lower()
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
        metadata["total_pages"] = len(reader.pages)
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
        path = Path(file_path)
        for encoding in TEXT_FILE_FALLBACK_ENCODINGS:
            try:
                with path.open("r", encoding=encoding) as handle:
                    text = handle.read()
            except OSError, UnicodeDecodeError:
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
        file_path: str | Path,
        *,
        content_type: str,
        document_title: str | None = None,
    ) -> dict[str, Any]:
        path = Path(file_path)
        document_id = self.compute_document_id(file_path)
        resolved_title = self.normalize_title(document_title) or path.stem
        return {
            "document_id": document_id,
            "source": str(path),
            "file_name": path.name,
            "document_title": resolved_title,
            "content_type": content_type,
            "source_relative_path": str(
                path.resolve().relative_to(self.documents_path.resolve())
            ).replace("\\", "/"),
            "source_file_size": path.stat().st_size if path.exists() else 0,
            "source_last_modified": (
                datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                if path.exists()
                else None
            ),
            "total_pages": 1,
        }

    # -------------------------------------------------------------------------
    def compute_document_id(self, file_path: str | Path) -> str:
        relative_path = (
            Path(file_path).resolve().relative_to(self.documents_path.resolve())
        )
        return hashlib.sha256(str(relative_path).encode("utf-8")).hexdigest()

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
        return Path(file_path).stem

    # -------------------------------------------------------------------------
    def resolve_docx_title(self, archive: zipfile.ZipFile, file_path: str) -> str:
        try:
            core_xml = archive.read("docProps/core.xml")
            tree = ElementTree.fromstring(core_xml)
        except KeyError, ElementTree.ParseError:
            return Path(file_path).stem
        namespaces = {"dc": "http://purl.org/dc/elements/1.1/"}
        node = tree.find("dc:title", namespaces)
        return (
            self.normalize_title(node.text if node is not None else None)
            or Path(file_path).stem
        )

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
