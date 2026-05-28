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
    ) -> list[SmartChunk]:
        source_text = text or ""
        line_index = TextLineIndex(source_text)
        chunks: list[SmartChunk] = []
        cursor = 0
        page_blocks = page_texts or [source_text]
        ordinal = 1
        for page_idx, page_text in enumerate(page_blocks, start=1):
            if not page_text.strip():
                cursor += len(page_text)
                continue
            for piece in self._split_piece(page_text):
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
                chunk_index = f"{file_name}::p{span.page_start}-{span.page_end}::l{span.line_start}-{span.line_end}::c{ordinal}"
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
                    "page_start": span.page_start,
                    "page_end": span.page_end,
                    "line_start": span.line_start,
                    "line_end": span.line_end,
                    "char_start": span.char_start,
                    "char_end": span.char_end,
                    "chunk_ordinal": ordinal,
                    "chunking_strategy": "smart_page_heading_paragraph_sentence_v1",
                    "chunk_char_count": len(piece),
                    "chunk_token_estimate": max(1, len(piece) // 4),
                    "section_heading": self._extract_heading(piece),
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

    def _split_piece(self, text: str) -> list[str]:
        cleaned = text.strip()
        if len(cleaned) <= self.max_chars:
            return [cleaned]
        sentences = [
            s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", cleaned) if s.strip()
        ]
        if not sentences:
            return [cleaned[: self.max_chars]]
        result: list[str] = []
        buffer = ""
        for sentence in sentences:
            candidate = (buffer + " " + sentence).strip() if buffer else sentence
            if len(candidate) <= self.target_chars:
                buffer = candidate
                continue
            if buffer:
                result.append(buffer)
            buffer = sentence
        if buffer:
            result.append(buffer)
        return result
