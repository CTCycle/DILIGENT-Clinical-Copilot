from __future__ import annotations

import re

from domain.clinical.robustness import (
    NormalizedDocument,
    NormalizedDocumentBlock,
    SourceSpan,
    SpanMapping,
)

ADMIN_RE = re.compile(
    r"\b(patient|paziente|date|data|page|pagina|address|telefono|phone|email|id)\b",
    re.IGNORECASE,
)
BIBLIOGRAPHY_RE = re.compile(
    r"\b(references|bibliography|bibliografia|doi:|pubmed|et al\.|journal|livertox)\b",
    re.IGNORECASE,
)


class DocumentNormalizer:
    MAX_BLOCK_TEXT_CHARS = 5000

    def normalize(self, raw_text: str) -> NormalizedDocument:
        clean_text = self._normalize_whitespace(raw_text)
        blocks = self._build_blocks(raw_text)
        return NormalizedDocument(
            raw_text=raw_text,
            clean_text=clean_text,
            span_mappings=[
                SpanMapping(
                    raw_start=0,
                    raw_end=len(raw_text),
                    clean_start=0,
                    clean_end=len(clean_text),
                )
            ],
            blocks=blocks,
        )

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)

    def _build_blocks(self, raw_text: str) -> list[NormalizedDocumentBlock]:
        blocks: list[NormalizedDocumentBlock] = []
        cursor = 0
        pending_lines: list[str] = []
        pending_start = 0
        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            line_start = cursor
            line_end = cursor + len(line)
            cursor = line_end + 1
            if line.strip():
                if not pending_lines:
                    pending_start = line_start
                pending_lines.append(line)
                continue
            if pending_lines:
                blocks.append(
                    self._make_block(
                        index=len(blocks),
                        lines=pending_lines,
                        start_char=pending_start,
                        end_char=line_start,
                        start_line=line_number - len(pending_lines),
                        end_line=line_number - 1,
                    )
                )
                pending_lines = []
        if pending_lines:
            line_count = len(raw_text.splitlines())
            blocks.append(
                self._make_block(
                    index=len(blocks),
                    lines=pending_lines,
                    start_char=pending_start,
                    end_char=len(raw_text),
                    start_line=max(1, line_count - len(pending_lines) + 1),
                    end_line=max(1, line_count),
                )
            )
        if not blocks and raw_text.strip():
            blocks.append(
                self._make_block(
                    index=0,
                    lines=[raw_text.strip()],
                    start_char=0,
                    end_char=len(raw_text),
                    start_line=1,
                    end_line=1,
                )
            )
        return blocks

    @staticmethod
    def _make_block(
        *,
        index: int,
        lines: list[str],
        start_char: int,
        end_char: int,
        start_line: int,
        end_line: int,
    ) -> NormalizedDocumentBlock:
        text = "\n".join(lines).strip()
        if len(text) > DocumentNormalizer.MAX_BLOCK_TEXT_CHARS:
            text = text[: DocumentNormalizer.MAX_BLOCK_TEXT_CHARS].rstrip()
        block_type = "clinical_content"
        confidence = 0.75
        if BIBLIOGRAPHY_RE.search(text):
            block_type = "bibliography"
            confidence = 0.85
        elif ADMIN_RE.search(text) and len(text.split()) < 35:
            block_type = "administrative"
            confidence = 0.75
        span = SourceSpan(
            span_id=f"block-{index + 1}",
            start_line=start_line,
            end_line=end_line,
            start_char=start_char,
            end_char=end_char,
            text=text,
        )
        return NormalizedDocumentBlock(
            block_id=f"block-{index + 1}",
            block_type=block_type,
            text=text,
            confidence=confidence,
            source_spans=[span],
        )
