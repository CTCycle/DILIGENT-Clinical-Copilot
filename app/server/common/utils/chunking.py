# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from typing import NamedTuple, cast

from common.utils.seed_terms import SeedTermCatalog, detect_seed_matches


###############################################################################
class SmartChunk(NamedTuple):
    text: str
    chunk_uid: str
    chunk_index: str
    chunk_index_number: int
    metadata: dict[str, object]


###############################################################################
class TokenWindowDocumentChunker:
    """Chunk documents by the canonical model tokenizer, not characters."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        tokenizer_provider: Callable[[], object],
        target_tokens: int = 512,
        overlap_tokens: int = 64,
        seed_catalog: SeedTermCatalog | None = None,
    ) -> None:
        self.tokenizer_provider = tokenizer_provider
        self.target_tokens = max(target_tokens, 1)
        self.overlap_tokens = max(min(overlap_tokens, self.target_tokens - 1), 0)
        self.seed_catalog = seed_catalog

    # -------------------------------------------------------------------------
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
        tokenizer = self.tokenizer_provider()
        encode = getattr(tokenizer, "encode", None)
        decode = getattr(tokenizer, "decode", None)
        if not callable(encode) or not callable(decode):
            raise TypeError("Canonical tokenizer must provide encode and decode")
        source_text = text or ""
        token_ids = list(
            cast(Iterable[int], encode(source_text, add_special_tokens=False))
        )
        if not token_ids:
            return []
        step = max(self.target_tokens - self.overlap_tokens, 1)
        chunks: list[SmartChunk] = []
        for ordinal, start in enumerate(range(0, len(token_ids), step), start=1):
            end = min(start + self.target_tokens, len(token_ids))
            piece = str(decode(token_ids[start:end], skip_special_tokens=True)).strip()
            if not piece:
                continue
            normalized_hash = hashlib.sha256(piece.encode("utf-8")).hexdigest()
            chunk_index = f"{file_name}::token-window::c{ordinal}"
            chunk_uid = hashlib.sha256(
                f"{relative_path}|{chunk_index}|{normalized_hash}".encode("utf-8")
            ).hexdigest()
            metadata: dict[str, object] = {
                "chunk_uid": chunk_uid,
                "chunk_index": chunk_index,
                "source_file_name": file_name,
                "source_relative_path": relative_path,
                "content_type": content_type,
                "page_reference": f"p{page_start_number}",
                "page_number": page_start_number,
                "page_start": page_start_number,
                "page_end": page_start_number,
                "total_pages": total_pages or len(page_texts or [source_text]),
                "chunk_ordinal": ordinal,
                "chunking_strategy": "token_window_v1",
                "chunk_token_count": end - start,
                "chunk_token_start": start,
                "chunk_token_end": end,
                "document_title": file_name.rsplit(".", 1)[0],
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
            if end >= len(token_ids):
                break
        return chunks

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_heading(line: str) -> str:
        return re.sub(r"\s+", " ", line.lstrip("#").strip())

    # -------------------------------------------------------------------------
    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
        return slug[:48] or "section"
