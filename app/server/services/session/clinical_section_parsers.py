from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import unicodedata
from typing import Literal

from domain.clinical.sections import ClinicalSectionKey

REQUIRED_DILI_SECTION_KEYS = ("anamnesis", "therapy", "laboratory_history")
CANONICAL_TO_CLINICAL_KEY: Mapping[str, ClinicalSectionKey] = {
    "anamnesis": "anamnesis",
    "therapy": "drugs",
    "laboratory_history": "laboratory_analysis",
}
MIN_TYPO_TOKEN_LEN = 7
MIN_TYPO_SIMILARITY = 0.88
HEADING_PREFIX_RE = re.compile(r"^(?:#{1,6}\s*|\d+\s*[.):\-]\s*|[-*+]\s*|(?:[ivxlcdm]+)\s*[.):\-]\s*)", re.IGNORECASE)
HEADING_SUFFIX_RE = re.compile(r"[\s:;.,\-_/|]+$")
NON_ALNUM_RE = re.compile(r"[^0-9a-zA-ZÀ-ÖØ-öø-ÿ ]+")
WS_RE = re.compile(r"\s+")

SECTION_HEADING_PROFILES: dict[str, dict[str, set[str]]] = {
    "anamnesis": {
        "required_any": {
            "anamnesis",
            "anamnesi",
            "history",
            "clinical history",
            "medical history",
            "patient history",
            "case history",
            "storia clinica",
            "storia anamnestica",
            "anamnesi patologica",
        },
        "supporting": {
            "precedenti",
            "sintomi",
            "diagnosi",
            "clinical conditions",
            "patient background",
        },
    },
    "therapy": {
        "required_any": {
            "therapy",
            "therapies",
            "treatment",
            "treatments",
            "medication",
            "medications",
            "drug therapy",
            "current therapy",
            "current medications",
            "terapia",
            "terapie",
            "trattamento",
            "farmaci",
            "terapia farmacologica",
            "terapie in corso",
        },
        "supporting": {
            "dose",
            "dosage",
            "route",
            "suspension",
            "somministrazione",
            "posologia",
            "sospensione",
        },
    },
    "laboratory_history": {
        "required_any": {
            "laboratory history",
            "laboratory",
            "labs",
            "lab history",
            "laboratory tests",
            "blood tests",
            "biochemistry",
            "liver tests",
            "esami di laboratorio",
            "laboratorio",
            "storia laboratoristica",
            "esami ematochimici",
            "biochimica",
        },
        "supporting": {
            "alt",
            "alp",
            "ast",
            "ggt",
            "bilirubin",
            "bilirubina",
            "inr",
            "uln",
            "limite superiore",
        },
    },
}


@dataclass(frozen=True)
class SectionHeadingMatch:
    canonical_key: str
    raw_heading: str
    normalized_heading: str
    score: float
    strategy: Literal["exact", "phrase", "token_containment", "typo_tolerant"]
    line_start: int
    line_end: int
    body_start: int | None = None
    body_end: int | None = None


@dataclass(frozen=True)
class ClinicalRawSection:
    canonical_key: str
    raw_heading: str
    normalized_heading: str
    match_strategy: Literal["exact", "phrase", "token_containment", "typo_tolerant"]
    confidence_score: float
    line_start: int
    line_end: int
    body_start: int
    body_end: int
    text: str
    verbatim_coherent: bool


def normalize_heading_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    text = text.strip()
    text = HEADING_PREFIX_RE.sub("", text)
    text = HEADING_SUFFIX_RE.sub("", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text.casefold()


def tokenize_heading(value: str) -> tuple[str, ...]:
    normalized = normalize_heading_text(value)
    if not normalized:
        return ()
    return tuple(token for token in normalized.split(" ") if token)


def is_structural_heading_line(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped:
        return False
    if len(stripped) > 120:
        return False
    if stripped.endswith((".", "!", "?")):
        return False
    if "  " in stripped:
        stripped = WS_RE.sub(" ", stripped)
    normalized = normalize_heading_text(stripped)
    if not normalized:
        return False
    if len(normalized.split(" ")) > 8:
        return False
    return bool(
        re.match(
            r"^(?:#{1,6}\s+[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ0-9]+){0,7}\s*:?\s*|[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ0-9]+){0,7}\s*:?\s*)$",
            stripped,
        )
    )


def _token_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def _classify_against_profile(
    normalized_heading: str,
    profile: dict[str, set[str]],
) -> tuple[float, Literal["exact", "phrase", "token_containment", "typo_tolerant"]] | None:
    required = profile["required_any"]
    tokens = set(tokenize_heading(normalized_heading))
    if normalized_heading in required:
        return (1.0, "exact")
    for phrase in required:
        if phrase in normalized_heading:
            return (0.95, "phrase")
    for phrase in required:
        phrase_tokens = set(phrase.split(" "))
        if phrase_tokens and phrase_tokens.issubset(tokens):
            return (0.9, "token_containment")
    for phrase in required:
        for token in phrase.split(" "):
            if len(token) < MIN_TYPO_TOKEN_LEN:
                continue
            if any(
                len(candidate) >= MIN_TYPO_TOKEN_LEN and _token_similarity(candidate, token) >= MIN_TYPO_SIMILARITY
                for candidate in tokens
            ):
                return (0.88, "typo_tolerant")
    return None


def classify_dili_heading(raw_heading: str, *, line_start: int, line_end: int) -> SectionHeadingMatch | None:
    normalized = normalize_heading_text(raw_heading)
    if not normalized:
        return None

    candidates: list[SectionHeadingMatch] = []
    for canonical_key, profile in SECTION_HEADING_PROFILES.items():
        match = _classify_against_profile(normalized, profile)
        if match is None:
            continue
        score, strategy = match
        candidates.append(
            SectionHeadingMatch(
                canonical_key=canonical_key,
                raw_heading=raw_heading.strip(),
                normalized_heading=normalized,
                score=score,
                strategy=strategy,
                line_start=line_start,
                line_end=line_end,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.score, reverse=True)
    if len(candidates) >= 2 and abs(candidates[0].score - candidates[1].score) <= 0.02:
        return None
    return candidates[0]


def find_dili_section_headings(raw_text: str) -> list[SectionHeadingMatch]:
    matches: list[SectionHeadingMatch] = []
    offset = 0
    for line_number, raw_line in enumerate(raw_text.splitlines(keepends=True), start=1):
        line = raw_line.rstrip("\r\n")
        if is_structural_heading_line(line):
            match = classify_dili_heading(line, line_start=line_number, line_end=line_number)
            if match is not None:
                matches.append(match)
        offset += len(raw_line)
    return matches


def resolve_heading_collisions(matches: Sequence[SectionHeadingMatch]) -> list[SectionHeadingMatch]:
    by_line: dict[int, list[SectionHeadingMatch]] = {}
    for match in matches:
        by_line.setdefault(match.line_start, []).append(match)
    resolved: list[SectionHeadingMatch] = []
    for _, line_matches in sorted(by_line.items()):
        line_matches.sort(key=lambda item: item.score, reverse=True)
        top = line_matches[0]
        if len(line_matches) > 1 and abs(top.score - line_matches[1].score) <= 0.02:
            continue
        resolved.append(top)
    return resolved


def _line_char_offsets(raw_text: str) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for line in raw_text.splitlines(keepends=True):
        start = cursor
        cursor += len(line)
        offsets.append((start, cursor))
    if not offsets:
        offsets.append((0, 0))
    return offsets


def extract_required_dili_sections(raw_text: str) -> dict[str, ClinicalRawSection]:
    headings = resolve_heading_collisions(find_dili_section_headings(raw_text))
    offsets = _line_char_offsets(raw_text)
    sections: dict[str, ClinicalRawSection] = {}

    for index, heading in enumerate(headings):
        if heading.line_start - 1 >= len(offsets):
            continue
        heading_line_start, heading_line_end = offsets[heading.line_start - 1]
        body_start = heading_line_end
        if index + 1 < len(headings):
            next_heading_line_start, _ = offsets[headings[index + 1].line_start - 1]
            body_end = next_heading_line_start
        else:
            body_end = len(raw_text)
        content = raw_text[body_start:body_end].strip("\r\n")
        if not content:
            continue
        coherent = verify_verbatim_section_coherence(raw_text, ClinicalRawSection(
            canonical_key=heading.canonical_key,
            raw_heading=heading.raw_heading,
            normalized_heading=heading.normalized_heading,
            match_strategy=heading.strategy,
            confidence_score=heading.score,
            line_start=heading.line_start,
            line_end=heading.line_end,
            body_start=body_start,
            body_end=body_end,
            text=content,
            verbatim_coherent=True,
        ))
        if heading.canonical_key in sections:
            raise ValueError(f"Duplicate section heading for '{heading.canonical_key}'.")
        sections[heading.canonical_key] = ClinicalRawSection(
            canonical_key=heading.canonical_key,
            raw_heading=heading.raw_heading,
            normalized_heading=heading.normalized_heading,
            match_strategy=heading.strategy,
            confidence_score=heading.score,
            line_start=heading.line_start,
            line_end=heading.line_end,
            body_start=body_start,
            body_end=body_end,
            text=content,
            verbatim_coherent=coherent,
        )
    return sections


def verify_verbatim_section_coherence(raw_text: str, section: ClinicalRawSection) -> bool:
    if section.body_start < 0 or section.body_end < section.body_start:
        return False
    if section.body_end > len(raw_text):
        return False
    span_text = raw_text[section.body_start:section.body_end].strip("\r\n")
    return span_text == section.text


def missing_required_section_names(sections: Mapping[str, ClinicalRawSection]) -> list[str]:
    return [key for key in REQUIRED_DILI_SECTION_KEYS if key not in sections or not sections[key].text.strip()]


# Compatibility wrappers for current callers/tests.
def find_section_markers(text: str) -> list[SectionHeadingMatch]:
    return find_dili_section_headings(text)


def extract_sections_from_markers(text: str, _markers: list[SectionHeadingMatch]) -> dict[ClinicalSectionKey, str] | None:
    try:
        sections = extract_required_dili_sections(text)
    except ValueError:
        return None
    if missing_required_section_names(sections):
        return None
    return {
        "anamnesis": sections["anamnesis"].text,
        "drugs": sections["therapy"].text,
        "laboratory_analysis": sections["laboratory_history"].text,
    }


def validate_sections_against_source(text: str, sections: dict[ClinicalSectionKey, str]) -> bool:
    for key in ("anamnesis", "drugs", "laboratory_analysis"):
        value = (sections.get(key) or "").strip()
        if not value or value not in text:
            return False
    return True
