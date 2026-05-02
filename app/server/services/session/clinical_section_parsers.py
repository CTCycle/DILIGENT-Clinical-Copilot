from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from domain.clinical.sections import (
    SECTION_KEY_BY_INDEX,
    SECTION_KEY_BY_JSON_KEY,
    SECTION_KEY_BY_LABEL,
    SECTION_KEY_BY_XML_TAG,
    ClinicalSectionKey,
)


@dataclass(frozen=True)
class SectionFragmentSlice:
    section: ClinicalSectionKey
    start: int
    end: int


class SectionParser(Protocol):
    confidence: float

    def __call__(self, source_text: str) -> list[SectionFragmentSlice] | None:
        ...


WHITESPACE_RE = re.compile(r"\s+")


def normalize_label(value: str) -> str:
    normalized = re.sub(r"[\*\_`~\[\]\(\)]", " ", value or "")
    normalized = re.sub(r"[^A-Za-z0-9 ]+", " ", normalized)
    return WHITESPACE_RE.sub(" ", normalized).strip().lower()


def line_body_start(source_text: str, line_end: int) -> int:
    if line_end < len(source_text) and source_text[line_end : line_end + 2] == "\r\n":
        return line_end + 2
    if line_end < len(source_text) and source_text[line_end] in "\r\n":
        return line_end + 1
    return line_end


def contains_all_sections(fragments: list[SectionFragmentSlice]) -> bool:
    return set(SECTION_KEY_BY_INDEX.values()).issubset(
        {fragment.section for fragment in fragments}
    )


class MarkdownSectionParser:
    confidence = 0.94
    header_re = re.compile(r"^ {0,3}#{1,6}\s+(?P<label>.+?)\s*$", re.MULTILINE)

    def __call__(self, source_text: str) -> list[SectionFragmentSlice] | None:
        markers = []
        for match in self.header_re.finditer(source_text):
            label = normalize_label(match.group("label"))
            section = SECTION_KEY_BY_LABEL.get(label)
            if section is None:
                continue
            markers.append((match.start(), match.end(), section))

        if not markers:
            return None

        fragments: list[SectionFragmentSlice] = []
        for index, (_, header_end, section) in enumerate(markers):
            body_start = line_body_start(source_text, header_end)
            body_end = markers[index + 1][0] if index + 1 < len(markers) else len(source_text)
            fragments.append(SectionFragmentSlice(section=section, start=body_start, end=body_end))

        return fragments if contains_all_sections(fragments) else None


class XmlLikeSectionParser:
    confidence = 0.97
    tag_re = re.compile(
        r"<(?P<tag>[A-Za-z_][A-Za-z0-9_\-]*)>(?P<body>.*?)</(?P=tag)>",
        re.DOTALL,
    )

    def __call__(self, source_text: str) -> list[SectionFragmentSlice] | None:
        fragments: list[SectionFragmentSlice] = []
        for match in self.tag_re.finditer(source_text):
            tag = normalize_label(match.group("tag")).replace(" ", "_")
            section = SECTION_KEY_BY_XML_TAG.get(tag)
            if section is None:
                continue
            fragments.append(
                SectionFragmentSlice(
                    section=section,
                    start=match.start("body"),
                    end=match.end("body"),
                )
            )

        return fragments if contains_all_sections(fragments) else None


class ColonSectionParser:
    confidence = 0.90
    header_re = re.compile(
        r"^(?P<label>[A-Za-z][A-Za-z0-9_ ]{2,60})\s*:\s*(?P<tail>.*)$",
        re.MULTILINE,
    )

    def __call__(self, source_text: str) -> list[SectionFragmentSlice] | None:
        markers = []
        for match in self.header_re.finditer(source_text):
            label = normalize_label(match.group("label"))
            section = SECTION_KEY_BY_LABEL.get(label)
            if section is None:
                continue
            body_start = match.start("tail")
            markers.append((match.start(), body_start, section))

        if not markers:
            return None

        fragments: list[SectionFragmentSlice] = []
        for index, (_, body_start, section) in enumerate(markers):
            body_end = markers[index + 1][0] if index + 1 < len(markers) else len(source_text)
            fragments.append(SectionFragmentSlice(section=section, start=body_start, end=body_end))

        return fragments if contains_all_sections(fragments) else None


class IndexedSectionParser:
    confidence = 0.88
    header_re = re.compile(
        r"^\s*(?P<index>[123])[\)\.\-]\s*(?P<label>[A-Za-z][A-Za-z0-9_ ]{2,60})?\s*:?\s*$",
        re.MULTILINE,
    )

    def __call__(self, source_text: str) -> list[SectionFragmentSlice] | None:
        markers = []
        seen_numeric_only: set[int] = set()

        for match in self.header_re.finditer(source_text):
            index = int(match.group("index"))
            indexed_section = SECTION_KEY_BY_INDEX[index]
            raw_label = match.group("label")

            if raw_label is None or not raw_label.strip():
                if index in seen_numeric_only:
                    return None
                seen_numeric_only.add(index)
                section = indexed_section
            else:
                label = normalize_label(raw_label)
                label_section = SECTION_KEY_BY_LABEL.get(label)
                if label_section is None:
                    section = indexed_section
                elif label_section != indexed_section:
                    return None
                else:
                    section = label_section

            markers.append((match.start(), match.end(), section))

        if not markers:
            return None

        fragments: list[SectionFragmentSlice] = []
        for marker_index, (_, header_end, section) in enumerate(markers):
            body_start = line_body_start(source_text, header_end)
            body_end = markers[marker_index + 1][0] if marker_index + 1 < len(markers) else len(source_text)
            fragments.append(SectionFragmentSlice(section=section, start=body_start, end=body_end))

        return fragments if contains_all_sections(fragments) else None


class JsonLikeSectionParser:
    confidence = 0.98

    def __call__(self, source_text: str) -> list[SectionFragmentSlice] | None:
        try:
            parsed = json.loads(source_text)
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, dict):
            return None

        fragments: list[SectionFragmentSlice] = []
        cursor = 0

        for raw_key, raw_value in parsed.items():
            section = SECTION_KEY_BY_JSON_KEY.get(str(raw_key))
            if section is None:
                continue

            values: list[str]
            if isinstance(raw_value, str):
                values = [raw_value]
            elif isinstance(raw_value, list) and all(isinstance(item, str) for item in raw_value):
                values = list(raw_value)
            else:
                return None

            for value in values:
                if not value:
                    return None
                start = source_text.find(value, cursor)
                if start < 0:
                    return None
                end = start + len(value)
                fragments.append(SectionFragmentSlice(section=section, start=start, end=end))
                cursor = end

        return fragments if contains_all_sections(fragments) else None


DETERMINISTIC_SECTION_PARSERS: tuple[SectionParser, ...] = (
    JsonLikeSectionParser(),
    XmlLikeSectionParser(),
    MarkdownSectionParser(),
    IndexedSectionParser(),
    ColonSectionParser(),
)
