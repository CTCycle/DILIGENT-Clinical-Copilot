from __future__ import annotations

from typing import NamedTuple

from domain.clinical.entities import (
    ClinicalSectionExtractionResult,
    ClinicalSectionLineRange,
)
from domain.clinical.sections import (
    ClinicalSectionKey,
    SECTION_DISPLAY_NAMES,
    SECTION_KEYS,
)
from services.session.clinical_section_parsers import (
    parse_required_dili_sections,
)


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


_CANONICAL_TO_PAYLOAD_KEY: dict[str, ClinicalSectionKey] = {
    "anamnesis": "anamnesis",
    "therapy": "drugs",
    "laboratory_history": "laboratory_analysis",
}


def _map_canonical_key(key: str) -> ClinicalSectionKey | None:
    return _CANONICAL_TO_PAYLOAD_KEY.get(key)


def _map_missing_keys(keys: list[str]) -> list[str]:
    return [mapped for key in keys if (mapped := _map_canonical_key(key)) is not None]


def _map_malformed_issue(issue: str) -> str:
    prefix, _, canonical_key = issue.partition(":")
    payload_key = _map_canonical_key(canonical_key)
    if not prefix or not payload_key:
        return issue
    return f"{prefix}:{payload_key}"


def parse_initial_text_sections(raw_text: str) -> InitialTextSectionParseResult:
    source_text = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    parse_result = parse_required_dili_sections(source_text)

    parsed: dict[str, ParsedTextSection] = {}
    for canonical_key, section in parse_result.sections.items():
        payload_key = _map_canonical_key(canonical_key)
        if payload_key is None:
            continue
        text = section.text.strip()
        parsed[payload_key] = ParsedTextSection(
            key=payload_key,
            title=section.raw_heading
            or SECTION_DISPLAY_NAMES.get(payload_key, payload_key),
            text=text,
            start_line=section.line_start,
            end_line=section.line_end,
        )

    return InitialTextSectionParseResult(
        sections=parsed,
        missing_required_sections=_map_missing_keys(
            parse_result.missing_required_sections
        ),
        malformed_sections=[
            _map_malformed_issue(issue) for issue in parse_result.malformed_sections
        ],
    )


def build_section_extraction_from_initial_text(
    parse_result: InitialTextSectionParseResult,
    source_text: str,
) -> ClinicalSectionExtractionResult:
    line_ranges: dict[ClinicalSectionKey, list[ClinicalSectionLineRange]] = {}
    metadata: dict[str, object] = {
        "parser": "deterministic_initial_text_sections_v2",
        "source_line_ranges": {},
    }
    for key in SECTION_KEYS:
        section = parse_result.sections.get(key)
        if section is None:
            continue
        line_ranges[key] = [
            ClinicalSectionLineRange(
                start_line=section.start_line, end_line=section.end_line
            )
        ]
        metadata["source_line_ranges"][key] = {
            "start_line": section.start_line,
            "end_line": section.end_line,
        }

    return ClinicalSectionExtractionResult(
        source_text=source_text,
        anamnesis=parse_result.sections.get(
            "anamnesis",
            ParsedTextSection("anamnesis", "Anamnesis", "", 1, 1),
        ).text,
        drugs=parse_result.sections.get(
            "drugs",
            ParsedTextSection("drugs", "Therapy", "", 1, 1),
        ).text,
        laboratory_analysis=parse_result.sections.get(
            "laboratory_analysis",
            ParsedTextSection("laboratory_analysis", "Laboratory analysis", "", 1, 1),
        ).text,
        line_ranges=line_ranges,
        confidence=1.0,
        metadata=metadata,
    )
