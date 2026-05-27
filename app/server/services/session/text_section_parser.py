from __future__ import annotations

import re
from typing import NamedTuple

from domain.clinical.entities import (
    ClinicalSectionExtractionResult,
    ClinicalSectionLineRange,
)
from domain.clinical.sections import ClinicalSectionKey, SECTION_DISPLAY_NAMES, SECTION_KEYS


class ParsedTextSection(NamedTuple):
    key: str
    title: str
    text: str
    start_line: int
    end_line: int


class InitialTextSectionParseResult(NamedTuple):
    sections: dict[str, ParsedTextSection]
    missing_required_sections: list[str]
    malformed_sections: list[str]


_HEADING_KEYWORDS: dict[ClinicalSectionKey, tuple[str, ...]] = {
    "anamnesis": ("anamnesis",),
    "drugs": ("drugs", "therapy", "current therapy"),
    "laboratory_analysis": ("laboratory analysis", "lab analysis", "laboratory", "labs"),
}

_REQUIRED_KEYS: tuple[ClinicalSectionKey, ...] = (
    "anamnesis",
    "drugs",
    "laboratory_analysis",
)


def _normalize_heading(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"^[#\-\*\s]+", "", cleaned)
    cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned)
    cleaned = cleaned.rstrip(":")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.casefold()


def _resolve_section_key(line: str) -> ClinicalSectionKey | None:
    normalized = _normalize_heading(line)
    for key, aliases in _HEADING_KEYWORDS.items():
        if normalized in aliases:
            return key
    return None


def parse_initial_text_sections(raw_text: str) -> InitialTextSectionParseResult:
    normalized = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    heading_positions: list[tuple[ClinicalSectionKey, int, str]] = []
    malformed_sections: list[str] = []
    seen_required: set[str] = set()
    for index, line in enumerate(lines, start=1):
        key = _resolve_section_key(line)
        if key is None:
            continue
        if key in seen_required:
            malformed_sections.append(f"duplicate:{key}")
            continue
        seen_required.add(key)
        heading_positions.append((key, index, line.strip()))

    parsed: dict[str, ParsedTextSection] = {}
    for idx, (key, start_line, title) in enumerate(heading_positions):
        next_start = heading_positions[idx + 1][1] if idx + 1 < len(heading_positions) else len(lines) + 1
        section_lines = lines[start_line: next_start - 1]
        text = "\n".join(section_lines).strip()
        if not text and key in _REQUIRED_KEYS:
            malformed_sections.append(f"empty:{key}")
            continue
        end_line = max(start_line, next_start - 1)
        parsed[key] = ParsedTextSection(
            key=key,
            title=title or SECTION_DISPLAY_NAMES.get(key, key),
            text=text,
            start_line=start_line,
            end_line=end_line,
        )

    missing = [key for key in _REQUIRED_KEYS if key not in parsed]
    return InitialTextSectionParseResult(
        sections=parsed,
        missing_required_sections=missing,
        malformed_sections=malformed_sections,
    )


def build_section_extraction_from_initial_text(
    parse_result: InitialTextSectionParseResult,
    source_text: str,
) -> ClinicalSectionExtractionResult:
    line_ranges: dict[ClinicalSectionKey, list[ClinicalSectionLineRange]] = {}
    metadata: dict[str, object] = {
        "parser": "deterministic_initial_text_sections_v1",
        "source_line_ranges": {},
    }
    for key in SECTION_KEYS:
        section = parse_result.sections.get(key)
        if section is None:
            continue
        line_ranges[key] = [
            ClinicalSectionLineRange(start_line=section.start_line, end_line=section.end_line)
        ]
        metadata["source_line_ranges"][key] = {
            "start_line": section.start_line,
            "end_line": section.end_line,
        }

    return ClinicalSectionExtractionResult(
        source_text=source_text,
        anamnesis=parse_result.sections.get("anamnesis", ParsedTextSection("anamnesis", "Anamnesis", "", 1, 1)).text,
        drugs=parse_result.sections.get("drugs", ParsedTextSection("drugs", "Drugs", "", 1, 1)).text,
        laboratory_analysis=parse_result.sections.get("laboratory_analysis", ParsedTextSection("laboratory_analysis", "Laboratory analysis", "", 1, 1)).text,
        line_ranges=line_ranges,
        confidence=1.0,
        metadata=metadata,
    )
