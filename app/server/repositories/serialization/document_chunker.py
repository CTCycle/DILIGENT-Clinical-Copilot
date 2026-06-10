from __future__ import annotations

import re

from common.utils.chunking import SmartDocumentChunker
from domain.documents import Document


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
    def split_sections(
        self, text: str
    ) -> list[tuple[str, int, str | None, str | None]]:
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
    def split_oversized_text(
        self, text: str, start_offset: int
    ) -> list[tuple[str, int]]:
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
        smart_chunker = SmartDocumentChunker(
            target_chars=self.chunk_size,
            max_chars=max(self.chunk_size, self.chunk_size + self.chunk_overlap),
            overlap_chars=self.chunk_overlap,
        )
        chunks: list[Document] = []
        for document in documents:
            metadata = dict(document.metadata)
            source = str(metadata.get("source") or "")
            file_name = str(metadata.get("file_name") or "")
            relative_path = file_name
            if source and file_name and source.endswith(file_name):
                relative_path = source.replace("\\", "/")
            smart_chunks = smart_chunker.chunk_document(
                text=document.page_content,
                file_name=file_name or "document",
                relative_path=relative_path or "document",
                content_type=str(metadata.get("content_type") or ""),
                page_texts=[document.page_content],
            )
            for chunk in smart_chunks:
                chunk_metadata = dict(metadata)
                chunk_metadata.update(chunk.metadata)
                char_start = chunk.metadata.get("char_start", 0)
                chunk_metadata["start_index"] = (
                    int(char_start) if isinstance(char_start, int | float | str) else 0
                )
                chunk_metadata["section_title"] = chunk.metadata.get("section_heading")
                chunk_metadata["heading_path"] = chunk.metadata.get("section_heading")
                chunks.append(
                    Document(page_content=chunk.text, metadata=chunk_metadata)
                )
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
        return chunks
