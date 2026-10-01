# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import hashlib
import re
import zipfile
from datetime import UTC, datetime
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
    def collect_file_paths(self) -> list[str]:
        collected: list[str] = []
        for candidate in self.documents_path.rglob("*"):
            if not candidate.is_file():
                continue
            collected.append(str(candidate))
        collected.sort()
        return collected

    # -------------------------------------------------------------------------
    def collect_document_paths(self) -> list[str]:
        collected: list[str] = []
        for path in self.collect_file_paths():
            if Path(path).suffix.lower() in self.SUPPORTED_EXTENSIONS:
                collected.append(path)
            else:
                logger.debug("Skipping unsupported document '%s'", Path(path).name)
        return collected

    # -------------------------------------------------------------------------
    def build_content_deduplication_index(
        self,
        file_paths: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return deterministic duplicate/canonical metadata for supported files.

        Duplicate identity is intentionally byte-based.  Relative paths only
        choose the canonical representative and remain the stable provenance
        key; they never participate in the content fingerprint itself.
        """
        paths = [
            str(Path(path).resolve())
            for path in (file_paths or self.collect_document_paths())
        ]
        ordered_paths = sorted(
            set(paths),
            key=lambda path: (
                self.normalized_relative_path(path),
                self.relative_path(path),
            ),
        )
        grouped_paths: dict[str, list[str]] = {}
        fingerprints: dict[str, str] = {}
        for path in ordered_paths:
            try:
                fingerprint = self.compute_content_fingerprint(path)
            except OSError as exc:
                logger.warning(
                    "Unable to fingerprint supported RAG file '%s': %s", path, exc
                )
                fingerprint = ""
            fingerprints[path] = fingerprint
            grouping_key = fingerprint or f"unreadable:{self.relative_path(path)}"
            grouped_paths.setdefault(grouping_key, []).append(path)

        index: dict[str, dict[str, Any]] = {}
        for group in grouped_paths.values():
            ordered_group = sorted(
                group,
                key=lambda path: (
                    self.normalized_relative_path(path),
                    self.relative_path(path),
                ),
            )
            canonical_path = ordered_group[0]
            relative_paths = [self.relative_path(path) for path in ordered_group]
            canonical_relative_path = self.relative_path(canonical_path)
            alias_paths = [
                relative_path
                for relative_path in relative_paths
                if relative_path != canonical_relative_path
            ]
            for path in ordered_group:
                relative_path = self.relative_path(path)
                is_duplicate = path != canonical_path
                index[path] = {
                    "content_fingerprint": fingerprints[path] or None,
                    "canonical_source_path": canonical_path,
                    "canonical_source_relative_path": canonical_relative_path,
                    "is_canonical_source": not is_duplicate,
                    "is_duplicate": is_duplicate,
                    "duplicate_of": canonical_relative_path if is_duplicate else None,
                    "duplicate_source_paths": relative_paths,
                    "duplicate_alias_paths": alias_paths,
                    "source_relative_path": relative_path,
                }
        return index

    # -------------------------------------------------------------------------
    def compute_content_fingerprint(self, file_path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(file_path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    # -------------------------------------------------------------------------
    def canonical_document_paths(
        self,
        file_paths: list[str] | None = None,
        *,
        deduplication_index: dict[str, dict[str, Any]] | None = None,
    ) -> list[str]:
        paths = file_paths or self.collect_document_paths()
        index = deduplication_index or self.build_content_deduplication_index(paths)
        canonical_paths = {
            str(metadata["canonical_source_path"])
            for path, metadata in index.items()
            if path in {str(Path(candidate).resolve()) for candidate in paths}
        }
        return sorted(
            canonical_paths,
            key=lambda path: (
                self.normalized_relative_path(path),
                self.relative_path(path),
            ),
        )

    # -------------------------------------------------------------------------
    def build_listing_metadata(
        self,
        file_path: str | Path,
        *,
        deduplication_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = Path(file_path)
        absolute_path = str(path.resolve())
        try:
            stat = path.stat()
            file_size = int(stat.st_size)
            modified = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
        except OSError:
            file_size = 0
            modified = datetime.fromtimestamp(0, UTC).isoformat()
        extension = path.suffix.lower()
        metadata: dict[str, Any] = {
            "path": str(path),
            "source_relative_path": self.relative_path(path),
            "file_name": path.name,
            "extension": extension,
            "file_size": file_size,
            "last_modified": modified,
            "supported_for_ingestion": extension in self.SUPPORTED_EXTENSIONS,
        }
        if extension in self.SUPPORTED_EXTENSIONS:
            deduplication_metadata = deduplication_metadata or (
                self.build_content_deduplication_index([absolute_path]).get(
                    absolute_path, {}
                )
            )
            metadata.update(
                {
                    "content_fingerprint": deduplication_metadata.get(
                        "content_fingerprint"
                    ),
                    "canonical_source_relative_path": deduplication_metadata.get(
                        "canonical_source_relative_path"
                    ),
                    "is_canonical_source": bool(
                        deduplication_metadata.get("is_canonical_source", False)
                    ),
                    "is_duplicate": bool(
                        deduplication_metadata.get("is_duplicate", False)
                    ),
                    "duplicate_of": deduplication_metadata.get("duplicate_of"),
                    "duplicate_source_paths": list(
                        deduplication_metadata.get("duplicate_source_paths", [])
                    ),
                    "duplicate_alias_paths": list(
                        deduplication_metadata.get("duplicate_alias_paths", [])
                    ),
                }
            )
        else:
            metadata.update(
                {
                    "content_fingerprint": None,
                    "canonical_source_relative_path": None,
                    "is_canonical_source": False,
                    "is_duplicate": False,
                    "duplicate_of": None,
                    "duplicate_source_paths": [],
                    "duplicate_alias_paths": [],
                }
            )
        return metadata

    # -------------------------------------------------------------------------
    def load_documents(
        self,
        *,
        deduplication_index: dict[str, dict[str, Any]] | None = None,
    ) -> list[Document]:
        documents: list[Document] = []
        available_paths = self.collect_document_paths()
        index = deduplication_index or self.build_content_deduplication_index(
            available_paths
        )
        for file_path in self.canonical_document_paths(
            available_paths, deduplication_index=index
        ):
            metadata = index.get(str(Path(file_path).resolve()), {})
            extension = Path(file_path).suffix.lower()
            if extension == ".pdf":
                documents.extend(
                    self.load_pdf(file_path, deduplication_metadata=metadata)
                )
            elif extension == ".docx":
                documents.extend(
                    self.load_docx(file_path, deduplication_metadata=metadata)
                )
            elif extension in {".txt", ".xml"}:
                documents.extend(
                    self.load_textual_file(
                        file_path,
                        extension,
                        deduplication_metadata=metadata,
                    )
                )
        return documents

    # -------------------------------------------------------------------------
    def load_pdf(
        self,
        file_path: str,
        *,
        deduplication_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        try:
            reader = PdfReader(file_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load PDF '%s': %s", file_path, exc)
            return []

        metadata = self.build_metadata(
            file_path,
            content_type="pdf",
            document_title=self.resolve_pdf_title(reader, file_path),
            deduplication_metadata=deduplication_metadata,
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
    def load_docx(
        self,
        file_path: str,
        *,
        deduplication_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
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
            deduplication_metadata=deduplication_metadata,
        )
        document = Document(page_content=content, metadata=metadata)
        return [document]

    # -------------------------------------------------------------------------
    def load_textual_file(
        self,
        file_path: str,
        extension: str,
        *,
        deduplication_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        text = self.read_text_content(file_path, extension)
        if not text:
            return []
        document = Document(
            page_content=text,
            metadata=self.build_metadata(
                file_path,
                content_type=extension.lstrip("."),
                document_title=self.extract_first_heading(text),
                deduplication_metadata=deduplication_metadata,
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
        deduplication_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = Path(file_path)
        document_id = self.compute_document_id(file_path)
        resolved_title = self.normalize_title(document_title) or path.stem
        metadata: dict[str, Any] = {
            "document_id": document_id,
            "source": str(path),
            "file_name": path.name,
            "document_title": resolved_title,
            "content_type": content_type,
            "source_relative_path": self.relative_path(path),
            "source_file_size": path.stat().st_size if path.exists() else 0,
            "source_last_modified": (
                datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                if path.exists()
                else None
            ),
            "total_pages": 1,
        }
        if deduplication_metadata:
            metadata.update(
                {
                    key: value
                    for key, value in deduplication_metadata.items()
                    if key != "canonical_source_path"
                }
            )
        return metadata

    # -------------------------------------------------------------------------
    def compute_document_id(self, file_path: str | Path) -> str:
        relative_path = Path(file_path).resolve().relative_to(
            self.documents_path.resolve()
        )
        return hashlib.sha256(str(relative_path).encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    def relative_path(self, file_path: str | Path) -> str:
        relative = Path(file_path).resolve().relative_to(self.documents_path.resolve())
        return str(relative).replace("\\", "/")

    # -------------------------------------------------------------------------
    def normalized_relative_path(self, file_path: str | Path) -> str:
        return self.relative_path(file_path).casefold()

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
