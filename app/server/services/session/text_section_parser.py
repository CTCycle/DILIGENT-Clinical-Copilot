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
from services.session.audit import build_source_hash


###############################################################################
class ParsedTextSection(NamedTuple):
    key: str
    title: str
    text: str
    start_line: int
    end_line: int
    body_start: int = 0
    body_end: int = 0
    canonical_key: str = ""
    normalized_heading: str = ""
    match_strategy: str = ""
    confidence_score: float = 0.0
    verbatim_coherent: bool = False
    requires_review: bool = False


###############################################################################
def _line_number_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


###############################################################################
def _section_requires_review(match_strategy: str, confidence_score: float) -> bool:
    return (
        match_strategy
        in {
            "content_inference",
            "fallback_assignment",
        }
        or confidence_score < 0.85
    )


###############################################################################
class InitialTextSectionParseResult(NamedTuple):
    sections: dict[str, ParsedTextSection]
    missing_required_sections: list[str]
    malformed_sections: list[str]


###############################################################################
def _aggregate_section_confidence(
    parse_result: InitialTextSectionParseResult,
) -> float:
    confidences = [
        section.confidence_score
        for key, section in parse_result.sections.items()
        if key in SECTION_KEYS
    ]
    if not confidences:
        return 0.0
    return max(0.0, min(1.0, min(confidences)))


###############################################################################
def _review_required_sections(
    parse_result: InitialTextSectionParseResult,
) -> list[str]:
    return [
        key
        for key, section in parse_result.sections.items()
        if key in SECTION_KEYS and section.requires_review
    ]


_CANONICAL_TO_PAYLOAD_KEY: dict[str, ClinicalSectionKey] = {
    "anamnesis": "anamnesis",
    "therapy": "drugs",
    "laboratory_history": "laboratory_analysis",
}


###############################################################################
def _map_canonical_key(key: str) -> ClinicalSectionKey | None:
    return _CANONICAL_TO_PAYLOAD_KEY.get(key)


###############################################################################
def _map_missing_keys(keys: list[str]) -> list[str]:
    return [mapped for key in keys if (mapped := _map_canonical_key(key)) is not None]


###############################################################################
def _map_malformed_issue(issue: str) -> str:
    parts = issue.split(":", 2)
    prefix = parts[0] if parts else ""
    canonical_key = parts[1] if len(parts) > 1 else ""
    payload_key = _map_canonical_key(canonical_key)
    if not prefix or not payload_key:
        return issue
    return f"{prefix}:{payload_key}"


###############################################################################
def parse_initial_text_sections(raw_text: str) -> InitialTextSectionParseResult:
    source_text = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    parse_result = parse_required_dili_sections(source_text)

    parsed: dict[str, ParsedTextSection] = {}
    for canonical_key, section in parse_result.sections.items():
        payload_key = _map_canonical_key(canonical_key)
        if payload_key is None:
            continue
        requires_review = _section_requires_review(
            section.match_strategy,
            section.confidence_score,
        )
        parsed[payload_key] = ParsedTextSection(
            key=payload_key,
            title=section.raw_heading
            or SECTION_DISPLAY_NAMES.get(payload_key, payload_key),
            text=section.text,
            start_line=section.line_start,
            end_line=section.line_end,
            body_start=section.body_start,
            body_end=section.body_end,
            canonical_key=section.canonical_key,
            normalized_heading=section.normalized_heading,
            match_strategy=section.match_strategy,
            confidence_score=section.confidence_score,
            verbatim_coherent=section.verbatim_coherent,
            requires_review=requires_review,
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


###############################################################################
def build_section_extraction_from_initial_text(
    parse_result: InitialTextSectionParseResult,
    source_text: str,
) -> ClinicalSectionExtractionResult:
    line_ranges: dict[ClinicalSectionKey, list[ClinicalSectionLineRange]] = {}
    source_line_ranges: dict[str, dict[str, int]] = {}
    metadata: dict[str, object] = {
        "parser": "deterministic_initial_text_sections_v2",
        "source_hash": build_source_hash(source_text),
        "source_line_ranges": source_line_ranges,
        "sections": {},
    }
    section_metadata = metadata["sections"]
    assert isinstance(section_metadata, dict)
    for key in SECTION_KEYS:
        section = parse_result.sections.get(key)
        if section is None:
            continue
        line_ranges[key] = [
            ClinicalSectionLineRange(
                start_line=section.start_line, end_line=section.end_line
            )
        ]
        source_line_ranges[key] = {
            "start_line": section.start_line,
            "end_line": section.end_line,
        }
        body_line_start = _line_number_for_offset(source_text, section.body_start)
        body_line_end = _line_number_for_offset(
            source_text,
            max(section.body_start, section.body_end - 1),
        )
        section_metadata[key] = {
            "canonical_key": section.canonical_key,
            "payload_key": section.key,
            "raw_heading": section.title,
            "normalized_heading": section.normalized_heading,
            "match_strategy": section.match_strategy,
            "confidence_score": section.confidence_score,
            "heading_line_span": [section.start_line, section.end_line],
            "body_line_span": [body_line_start, body_line_end],
            "char_span": [section.body_start, section.body_end],
            "verbatim_coherent": section.verbatim_coherent,
            "requires_review": section.requires_review,
        }
    review_required_sections = _review_required_sections(parse_result)
    metadata["requires_review"] = bool(review_required_sections)
    metadata["requires_review_sections"] = review_required_sections

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
        confidence=_aggregate_section_confidence(parse_result),
        metadata=metadata,
    )
