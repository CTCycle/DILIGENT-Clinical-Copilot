from __future__ import annotations

import hashlib
import re
from datetime import datetime, UTC
from typing import NamedTuple

from services.retrieval.seed_terms import SeedTermCatalog, detect_seed_matches


class TextLineIndex(NamedTuple):
    text: str

    def line_for_offset(self, offset: int) -> int:
        bounded = max(0, min(offset, len(self.text)))
        return self.text.count("\n", 0, bounded) + 1


class ChunkSourceSpan(NamedTuple):
    page_start: int
    page_end: int
    line_start: int
    line_end: int
    char_start: int
    char_end: int


class SmartChunk(NamedTuple):
    text: str
    chunk_uid: str
    chunk_index: str
    chunk_index_number: int
    metadata: dict[str, object]


class SmartDocumentChunker:
    def __init__(
        self,
        *,
        target_chars: int = 1600,
        max_chars: int = 2400,
        overlap_chars: int = 200,
        min_chars: int = 250,
        seed_catalog: SeedTermCatalog | None = None,
    ) -> None:
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chars = min_chars
        self.seed_catalog = seed_catalog

    def chunk_document(
        self,
        text: str,
        file_name: str,
        relative_path: str,
        content_type: str,
        page_texts: list[str] | None = None,
        page_start_number: int = 1,
        total_pages: int | None = None,
    ) -> list[SmartChunk]:
        source_text = text or ""
        line_index = TextLineIndex(source_text)
        chunks: list[SmartChunk] = []
        cursor = 0
        page_blocks = page_texts or [source_text]
        resolved_total_pages = total_pages or len(page_blocks)
        ordinal = 1
        for local_page_idx, page_text in enumerate(page_blocks):
            page_idx = page_start_number + local_page_idx
            if not page_text.strip():
                cursor += len(page_text)
                continue
            for piece, section_heading, heading_path in self._split_piece(page_text):
                local_start = page_text.find(piece)
                if local_start < 0:
                    local_start = 0
                char_start = cursor + local_start
                char_end = char_start + len(piece)
                span = ChunkSourceSpan(
                    page_start=page_idx,
                    page_end=page_idx,
                    line_start=line_index.line_for_offset(char_start),
                    line_end=line_index.line_for_offset(char_end),
                    char_start=char_start,
                    char_end=char_end,
                )
                section_slug = self._slug(section_heading or "section")
                chunk_index = (
                    f"{file_name}::p{span.page_start}-{span.page_end}"
                    f"::{section_slug}::c{ordinal}"
                )
                normalized_hash = hashlib.sha256(
                    piece.strip().encode("utf-8")
                ).hexdigest()
                chunk_uid = hashlib.sha256(
                    f"{relative_path}|{chunk_index}|{normalized_hash}".encode("utf-8")
                ).hexdigest()
                metadata: dict[str, object] = {
                    "chunk_uid": chunk_uid,
                    "chunk_index": chunk_index,
                    "source_file_name": file_name,
                    "source_relative_path": relative_path,
                    "content_type": content_type,
                    "page_reference": f"p{span.page_start}",
                    "page_number": span.page_start,
                    "page_start": span.page_start,
                    "page_end": span.page_end,
                    "total_pages": resolved_total_pages,
                    "line_start": span.line_start,
                    "line_end": span.line_end,
                    "char_start": span.char_start,
                    "char_end": span.char_end,
                    "chunk_ordinal": ordinal,
                    "chunking_strategy": "clinical_recursive_page_heading_paragraph_sentence_v2",
                    "chunk_char_count": len(piece),
                    "chunk_token_estimate": max(1, len(piece) // 4),
                    "section_heading": section_heading,
                    "section_title": section_heading,
                    "heading_path": heading_path,
                    "document_title": file_name.rsplit(".", 1)[0],
                    "created_at": datetime.now(UTC).isoformat(),
                }
                if self.seed_catalog is not None:
                    seed = detect_seed_matches(piece, self.seed_catalog)
                    metadata["seed_matched_keywords"] = seed["matched_keywords"]
                    metadata["seed_matched_stopwords"] = seed["matched_stopwords"]
                    metadata["seed_matched_terms"] = seed["matched_terms"]
                    metadata["seed_matched_term_groups"] = seed["matched_term_groups"]
                    metadata["seed_matched_term_counts"] = seed["matched_term_counts"]
                chunks.append(
                    SmartChunk(
                        text=piece,
                        chunk_uid=chunk_uid,
                        chunk_index=chunk_index,
                        chunk_index_number=ordinal,
                        metadata=metadata,
                    )
                )
                ordinal += 1
            cursor += len(page_text)
        return chunks

    def _extract_heading(self, text: str) -> str | None:
        first = (text.splitlines()[0] if text.splitlines() else "").strip()
        if not first:
            return None
        if len(first) <= 120 and (first.startswith("#") or first == first.upper()):
            return re.sub(r"^[#\s]+", "", first)
        return None

    def _split_piece(self, text: str) -> list[tuple[str, str | None, str | None]]:
        cleaned = text.strip()
        if not cleaned:
            return []
        sections = self._split_sections(cleaned)
        chunks: list[tuple[str, str | None, str | None]] = []
        for section_text, heading, heading_path in sections:
            for chunk in self._recursive_split(section_text):
                chunks.append((chunk, heading, heading_path))
        return self._merge_small_chunks(chunks)

    def _split_sections(self, text: str) -> list[tuple[str, str | None, str | None]]:
        sections: list[tuple[str, str | None, str | None]] = []
        heading_stack: list[str] = []
        current_lines: list[str] = []
        current_heading: str | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if self._is_heading_line(line) and current_lines:
                sections.append(
                    (
                        "\n".join(current_lines).strip(),
                        current_heading,
                        " > ".join(heading_stack) or current_heading,
                    )
                )
                current_lines = []
            if self._is_heading_line(line):
                current_heading = self._normalize_heading(line)
                heading_stack = [current_heading]
            current_lines.append(raw_line)
        if current_lines:
            sections.append(
                (
                    "\n".join(current_lines).strip(),
                    current_heading or self._extract_heading(text),
                    " > ".join(heading_stack) or current_heading,
                )
            )
        return sections

    def _recursive_split(self, text: str) -> list[str]:
        normalized = text.strip()
        if len(normalized) <= self.max_chars:
            return [normalized]
        return self._split_by_separators(
            normalized,
            separators=("\n\n", "\n- ", "\n* ", "\n", ". ", "; ", ", ", " "),
        )

    def _split_by_separators(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self.max_chars:
            return [text.strip()]
        if not separators:
            return [
                text[start : start + self.max_chars].strip()
                for start in range(0, len(text), max(self.max_chars - self.overlap_chars, 1))
                if text[start : start + self.max_chars].strip()
            ]
        separator = separators[0]
        parts = text.split(separator)
        if len(parts) == 1:
            return self._split_by_separators(text, separators[1:])
        chunks: list[str] = []
        buffer = ""
        for index, part in enumerate(parts):
            segment = part if index == 0 else f"{separator}{part}"
            candidate = f"{buffer}{segment}" if buffer else segment.lstrip()
            if len(candidate) <= self.target_chars:
                buffer = candidate
                continue
            if buffer:
                chunks.extend(self._split_by_separators(buffer.strip(), separators[1:]))
            buffer = segment.lstrip()
        if buffer.strip():
            chunks.extend(self._split_by_separators(buffer.strip(), separators[1:]))
        return chunks

    def _merge_small_chunks(
        self, chunks: list[tuple[str, str | None, str | None]]
    ) -> list[tuple[str, str | None, str | None]]:
        merged: list[tuple[str, str | None, str | None]] = []
        for text, heading, heading_path in chunks:
            if (
                merged
                and len(text) < self.min_chars
                and len(merged[-1][0]) + len(text) + 2 <= self.max_chars
                and merged[-1][1] == heading
            ):
                prev_text, prev_heading, prev_path = merged[-1]
                merged[-1] = (f"{prev_text}\n\n{text}", prev_heading, prev_path)
            else:
                merged.append((text, heading, heading_path))
        return merged

    @staticmethod
    def _is_heading_line(line: str) -> bool:
        if not line or len(line) > 140:
            return False
        if line.startswith("#"):
            return True
        if re.match(r"^\d+(?:\.\d+)*[.)]?\s+\S+", line):
            return True
        words = line.split()
        return 1 <= len(words) <= 12 and line == line.upper()

    @staticmethod
    def _normalize_heading(line: str) -> str:
        return re.sub(r"\s+", " ", line.lstrip("#").strip())

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
        return slug[:48] or "section"
