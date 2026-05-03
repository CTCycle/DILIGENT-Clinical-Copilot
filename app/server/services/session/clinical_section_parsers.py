from __future__ import annotations

import re
from collections.abc import Mapping
from typing import NamedTuple

from domain.clinical.sections import ClinicalSectionKey, SECTION_KEYS


WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_PUNCTUATION_RE = re.compile(r"[.!?]")


class SectionFragmentSlice(NamedTuple):
    section: ClinicalSectionKey
    start: int
    end: int


class SectionBoundary(NamedTuple):
    section: ClinicalSectionKey
    heading_line_index: int
    content_start_line_index: int


def normalize_section_title(value: str) -> str:
    normalized = re.sub(r"[\*\_`~\[\]\(\)]", " ", value or "")
    normalized = re.sub(r"[^A-Za-z0-9 ]+", " ", normalized)
    return WHITESPACE_RE.sub(" ", normalized).strip().lower()


def strip_heading_decoration(line: str) -> str:
    value = line.strip()
    value = re.sub(r"^#{1,6}\s+", "", value)
    value = re.sub(r"^[-*]\s+", "", value)
    value = re.sub(r"^\d+\s*[\.)\-]\s*", "", value)
    value = re.sub(r"\s*:\s*$", "", value)
    value = value.strip()
    if re.match(r"^\*\*[^*]+\*\*$", value):
        value = value[2:-2].strip()
    return value


def line_offsets(source_text: str) -> list[tuple[int, int, int]]:
    lines = source_text.splitlines(keepends=True)
    offsets: list[tuple[int, int, int]] = []
    cursor = 0
    for line in lines:
        line_start = cursor
        line_end = line_start + len(line.rstrip("\r\n"))
        cursor = line_start + len(line)
        offsets.append((line_start, line_end, cursor))
    return offsets


class PlainTextSectionParser:
    confidence = 0.93

    def __init__(self, section_title_aliases: Mapping[str, frozenset[str]]) -> None:
        self._alias_to_section = self._build_alias_index(section_title_aliases)

    def _build_alias_index(
        self,
        section_title_aliases: Mapping[str, frozenset[str]],
    ) -> dict[str, ClinicalSectionKey]:
        alias_to_section: dict[str, ClinicalSectionKey] = {}
        for section_key in SECTION_KEYS:
            aliases = section_title_aliases.get(section_key, frozenset())
            for alias in aliases:
                normalized = normalize_section_title(alias)
                if normalized:
                    alias_to_section[normalized] = section_key
        return alias_to_section

    def _match_heading(self, line: str) -> ClinicalSectionKey | None:
        if not line.strip() or len(line) > 120:
            return None

        cleaned = strip_heading_decoration(line)
        if not cleaned:
            return None

        if len(SENTENCE_PUNCTUATION_RE.findall(cleaned)) > 1:
            return None

        normalized = normalize_section_title(cleaned)
        if not normalized:
            return None
        return self._alias_to_section.get(normalized)

    def __call__(self, source_text: str) -> list[SectionFragmentSlice] | None:
        if not self._alias_to_section:
            return None

        lines = source_text.splitlines(keepends=True)
        offsets = line_offsets(source_text)

        boundaries: list[SectionBoundary] = []
        for index, line in enumerate(lines):
            section = self._match_heading(line)
            if section is None:
                continue
            boundaries.append(
                SectionBoundary(
                    section=section,
                    heading_line_index=index,
                    content_start_line_index=index + 1,
                )
            )

        if not boundaries:
            return None

        fragments: list[SectionFragmentSlice] = []
        for index, boundary in enumerate(boundaries):
            content_start_line = boundary.content_start_line_index
            next_heading_line = (
                boundaries[index + 1].heading_line_index
                if index + 1 < len(boundaries)
                else len(lines)
            )
            if content_start_line >= len(lines):
                continue

            start = offsets[content_start_line][0]
            end = (
                offsets[next_heading_line - 1][2]
                if next_heading_line > 0
                else start
            )
            if end <= start:
                continue

            while start < end and source_text[start] in "\r\n":
                start += 1
            while end > start and source_text[end - 1] in "\r\n":
                end -= 1

            if start >= end or not source_text[start:end].strip():
                continue

            fragments.append(
                SectionFragmentSlice(section=boundary.section, start=start, end=end)
            )

        required = set(SECTION_KEYS)
        present = {fragment.section for fragment in fragments}
        if not required.issubset(present):
            return None
        return fragments


__all__ = [
    "PlainTextSectionParser",
    "SectionBoundary",
    "SectionFragmentSlice",
    "normalize_section_title",
]
