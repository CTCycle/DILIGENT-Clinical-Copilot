from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Literal

from domain.clinical.sections import ClinicalSectionKey
from services.catalogs.runtime import get_reference_catalog_snapshot

REQUIRED_DILI_SECTION_KEYS = ("anamnesis", "therapy", "laboratory_history")
CANONICAL_TO_CLINICAL_KEY: Mapping[str, ClinicalSectionKey] = {
    "anamnesis": "anamnesis",
    "therapy": "drugs",
    "laboratory_history": "laboratory_analysis",
}
MIN_TYPO_TOKEN_LEN = 7
MIN_TYPO_SIMILARITY = 0.88
MIN_TYPO_SHARED_PREFIX_LEN = 4
HEADING_PREFIX_RE = re.compile(r"^(?:#{1,6}\s*|\d+\s*[.):\-]\s*|[-*+]\s*|(?:[ivxlcdm]+)\s*[.):\-]\s*)", re.IGNORECASE)
HEADING_SUFFIX_RE = re.compile(r"[\s:;.,\-_/|]+$")
NON_ALNUM_RE = re.compile(r"[^0-9a-zA-ZÀ-ÖØ-öø-ÿ ]+")
WS_RE = re.compile(r"\s+")
MARKDOWN_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S")
THERAPY_SCHEDULE_RE = re.compile(r"\b\d+(?:[.,]\d+)?(?:-\d+(?:[.,]\d+)?){2,3}\b")
THERAPY_DOSAGE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|ug|ml|units?|ui|iu)\b",
    re.IGNORECASE,
)
LAB_VALUE_RE = re.compile(
    r"\b(?:alt|ast|alp|ggt|bilirubin|bilirubina|inr|bilir)\b[\s:=<>-]*\d",
    re.IGNORECASE,
)
DATE_VALUE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b")
GENERIC_HEADING_SEMANTIC_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "anamnesis": ("anamn", "histor", "clinic", "present", "complain"),
    "therapy": ("therap", "terapi", "medicat", "pharmac", "farmac", "drug", "treat"),
    "laboratory_history": ("labor", "analys", "analy", "blood", "biochem", "chem", "test"),
}
SEMANTIC_PREFIX_STOPWORDS = frozenset({"medical", "patient", "physician", "document", "report"})


SectionMatchStrategy = Literal[
    "exact",
    "phrase",
    "token_containment",
    "typo_tolerant",
    "semantic_tokens",
    "content_inference",
    "fallback_assignment",
]

def _section_profiles_from_catalog() -> dict[str, dict[str, set[str]]]:
    snapshot = get_reference_catalog_snapshot()
    profiles: dict[str, dict[str, set[str]]] = {
        "anamnesis": {"required_any": set(), "supporting": set()},
        "therapy": {"required_any": set(), "supporting": set()},
        "laboratory_history": {"required_any": set(), "supporting": set()},
    }
    category_map: dict[str, tuple[str, str]] = {
        "anamnesis_required": ("anamnesis", "required_any"),
        "anamnesis_supporting": ("anamnesis", "supporting"),
        "therapy_required": ("therapy", "required_any"),
        "therapy_supporting": ("therapy", "supporting"),
        "laboratory_history_required": ("laboratory_history", "required_any"),
        "laboratory_history_supporting": ("laboratory_history", "supporting"),
    }
    for category, (canonical_key, bucket) in category_map.items():
        values = snapshot.values("clinical_extraction", "section_heading_patterns", key=category)
        profiles[canonical_key][bucket].update(
            normalize_heading_text(value) for value in values if normalize_heading_text(value)
        )

    mapped: dict[str, str] = {
        "anamnesis": "anamnesis",
        "drugs": "therapy",
        "laboratory_analysis": "laboratory_history",
    }
    for backend_key, canonical_key in mapped.items():
        profiles[canonical_key]["required_any"].add(
            normalize_heading_text(backend_key)
        )
        profiles[canonical_key]["required_any"].add(
            normalize_heading_text(canonical_key.replace("_", " "))
        )
        aliases = snapshot.values(
            "clinical_extraction",
            "section_aliases",
            key=backend_key,
        )
        if aliases:
            profiles[canonical_key]["required_any"].update(
                normalize_heading_text(alias) for alias in aliases if normalize_heading_text(alias)
            )
    return profiles


@dataclass(frozen=True)
class SectionHeadingMatch:
    canonical_key: str
    raw_heading: str
    normalized_heading: str
    score: float
    strategy: SectionMatchStrategy
    line_start: int
    line_end: int
    body_start: int | None = None
    body_end: int | None = None


@dataclass(frozen=True)
class ClinicalRawSection:
    canonical_key: str
    raw_heading: str
    normalized_heading: str
    match_strategy: SectionMatchStrategy
    confidence_score: float
    line_start: int
    line_end: int
    body_start: int
    body_end: int
    text: str
    verbatim_coherent: bool


@dataclass(frozen=True)
class HeadingScanResult:
    section_headings: list[SectionHeadingMatch]
    boundary_line_starts: set[int]


@dataclass(frozen=True)
class HeadingBoundary:
    raw_heading: str
    line_start: int
    line_end: int
    is_markdown_heading: bool


@dataclass(frozen=True)
class ParsedDiliSectionsResult:
    sections: dict[str, ClinicalRawSection]
    missing_required_sections: list[str]
    malformed_sections: list[str]


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


def _looks_like_typo(candidate: str, token: str) -> bool:
    prefix_len = min(MIN_TYPO_SHARED_PREFIX_LEN, len(candidate), len(token))
    return (
        candidate[:prefix_len] == token[:prefix_len]
        and _token_similarity(candidate, token) >= MIN_TYPO_SIMILARITY
    )


def _phrase_matches_with_typo(tokens: set[str], phrase: str) -> bool:
    phrase_tokens = tuple(token for token in phrase.split(" ") if token)
    if not phrase_tokens:
        return False

    found_typo = False
    for phrase_token in phrase_tokens:
        if phrase_token in tokens:
            continue
        if len(phrase_token) < MIN_TYPO_TOKEN_LEN:
            return False
        if not any(
            len(candidate) >= MIN_TYPO_TOKEN_LEN
            and candidate != phrase_token
            and _looks_like_typo(candidate, phrase_token)
            for candidate in tokens
        ):
            return False
        found_typo = True
    return found_typo


def _classify_against_profile(
    normalized_heading: str,
    profile: dict[str, set[str]],
) -> tuple[float, SectionMatchStrategy] | None:
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
        if _phrase_matches_with_typo(tokens, phrase):
            return (0.88, "typo_tolerant")
    return None


def _semantic_prefix_score(normalized_heading: str) -> tuple[str, float] | None:
    tokens = set(tokenize_heading(normalized_heading))
    if not tokens:
        return None
    best_key = ""
    best_score = 0.0
    for canonical_key, prefixes in _semantic_heading_prefixes().items():
        matched_prefixes = {
            prefix
            for token in tokens
            for prefix in prefixes
            if len(prefix) >= 5 and token.startswith(prefix)
        }
        if not matched_prefixes:
            continue
        score = 0.72 + min(0.16, 0.06 * len(matched_prefixes))
        if score > best_score:
            best_key = canonical_key
            best_score = score
    if not best_key:
        return None
    return (best_key, best_score)


@lru_cache(maxsize=1)
def _semantic_heading_prefixes() -> dict[str, tuple[str, ...]]:
    profiles = _section_profiles_from_catalog()
    prefixes: dict[str, set[str]] = {
        key: set(values)
        for key, values in GENERIC_HEADING_SEMANTIC_PREFIXES.items()
    }
    for canonical_key, profile in profiles.items():
        for phrase in profile["required_any"]:
            for token in tokenize_heading(phrase):
                if len(token) >= 5 and token not in SEMANTIC_PREFIX_STOPWORDS:
                    prefixes.setdefault(canonical_key, set()).add(token[: min(len(token), 8)])
    return {key: tuple(sorted(values)) for key, values in prefixes.items()}


def classify_dili_heading(raw_heading: str, *, line_start: int, line_end: int) -> SectionHeadingMatch | None:
    normalized = normalize_heading_text(raw_heading)
    if not normalized:
        return None

    candidates: list[SectionHeadingMatch] = []
    profiles = _section_profiles_from_catalog()
    for canonical_key, profile in profiles.items():
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
    semantic_match = _semantic_prefix_score(normalized)
    if semantic_match is not None:
        semantic_key, semantic_score = semantic_match
        candidates.append(
            SectionHeadingMatch(
                canonical_key=semantic_key,
                raw_heading=raw_heading.strip(),
                normalized_heading=normalized,
                score=semantic_score,
                strategy="semantic_tokens",
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


def is_markdown_heading_line(line: str) -> bool:
    return bool(MARKDOWN_HEADING_RE.match(line or ""))


def _accept_heading_match(boundary: HeadingBoundary, match: SectionHeadingMatch) -> bool:
    if boundary.is_markdown_heading:
        return True
    if match.strategy == "exact":
        return True
    stripped = boundary.raw_heading.strip()
    if stripped.endswith(":"):
        return True
    words = [token for token in stripped.split() if any(char.isalpha() for char in token)]
    if not words:
        return False
    is_upper = all(word.upper() == word for word in words)
    is_title = all((not word[0].isalpha()) or word[0].isupper() for word in words)
    return is_upper or is_title


def _iter_heading_boundaries(raw_text: str) -> list[HeadingBoundary]:
    boundaries: list[HeadingBoundary] = []
    for line_number, raw_line in enumerate(raw_text.splitlines(keepends=True), start=1):
        line = raw_line.rstrip("\r\n")
        is_markdown_heading = is_markdown_heading_line(line)
        if not is_markdown_heading and not is_structural_heading_line(line):
            continue
        boundaries.append(
            HeadingBoundary(
                raw_heading=line,
                line_start=line_number,
                line_end=line_number,
                is_markdown_heading=is_markdown_heading,
            )
        )
    return boundaries


def _selected_heading_boundaries(raw_text: str) -> list[HeadingBoundary]:
    boundaries = _iter_heading_boundaries(raw_text)
    markdown = [boundary for boundary in boundaries if boundary.is_markdown_heading]
    if markdown:
        return markdown
    selected: list[HeadingBoundary] = []
    for boundary in boundaries:
        match = classify_dili_heading(
            boundary.raw_heading,
            line_start=boundary.line_start,
            line_end=boundary.line_end,
        )
        if match is not None and _accept_heading_match(boundary, match):
            selected.append(boundary)
            continue
        if boundary.raw_heading.strip().endswith(":") and len(tokenize_heading(boundary.raw_heading)) >= 2:
            selected.append(boundary)
    return selected


def scan_dili_section_headings(raw_text: str) -> HeadingScanResult:
    boundaries = _iter_heading_boundaries(raw_text)
    matches: list[SectionHeadingMatch] = []
    markdown_matches: list[SectionHeadingMatch] = []
    markdown_boundary_lines = {
        boundary.line_start for boundary in boundaries if boundary.is_markdown_heading
    }
    for boundary in boundaries:
        match = classify_dili_heading(
            boundary.raw_heading,
            line_start=boundary.line_start,
            line_end=boundary.line_end,
        )
        if match is not None and _accept_heading_match(boundary, match):
            matches.append(match)
            if boundary.is_markdown_heading:
                markdown_matches.append(match)
    markdown_keys = {match.canonical_key for match in markdown_matches}
    if all(key in markdown_keys for key in REQUIRED_DILI_SECTION_KEYS):
        return HeadingScanResult(
            section_headings=markdown_matches,
            boundary_line_starts=markdown_boundary_lines,
        )
    return HeadingScanResult(
        section_headings=matches,
        boundary_line_starts={boundary.line_start for boundary in _selected_heading_boundaries(raw_text)},
    )


def find_dili_section_headings(raw_text: str) -> list[SectionHeadingMatch]:
    return scan_dili_section_headings(raw_text).section_headings


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


@lru_cache(maxsize=1)
def _therapy_signal_re() -> re.Pattern[str]:
    snapshot = get_reference_catalog_snapshot()
    values: set[str] = set()
    for category in (
        "drug_dosage_units",
        "drug_route_terms",
        "drug_frequency_terms",
        "drug_metadata_labels",
    ):
        values.update(value for value in snapshot.values("clinical_extraction", category) if value)
    escaped = sorted({re.escape(value) for value in values if len(value.strip()) >= 2})
    if not escaped:
        return re.compile(r"$^")
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _laboratory_signal_re() -> re.Pattern[str]:
    snapshot = get_reference_catalog_snapshot()
    values: set[str] = set()
    for category in (
        "laboratory_markers",
        "laboratory_units",
        "laboratory_uln_labels",
        "lab_measurement_indicators",
        "lab_marker_indicators",
    ):
        values.update(value for value in snapshot.values("clinical_extraction", category) if value)
    escaped = sorted({re.escape(value) for value in values if len(value.strip()) >= 2})
    if not escaped:
        return re.compile(r"$^")
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


def _score_section_content(raw_heading: str, content: str) -> dict[str, float]:
    normalized_heading = normalize_heading_text(raw_heading)
    scores = {
        "anamnesis": 0.0,
        "therapy": 0.0,
        "laboratory_history": 0.0,
    }
    semantic_heading = _semantic_prefix_score(normalized_heading)
    if semantic_heading is not None:
        semantic_key, semantic_score = semantic_heading
        scores[semantic_key] += semantic_score * 0.35

    therapy_hits = len(_therapy_signal_re().findall(content))
    lab_hits = len(_laboratory_signal_re().findall(content))
    schedule_hits = len(THERAPY_SCHEDULE_RE.findall(content))
    dosage_hits = len(THERAPY_DOSAGE_RE.findall(content))
    dated_hits = len(DATE_VALUE_RE.findall(content))
    lab_value_hits = len(LAB_VALUE_RE.findall(content))
    line_count = max(1, len([line for line in content.splitlines() if line.strip()]))

    scores["therapy"] += min(0.92, therapy_hits * 0.12 + schedule_hits * 0.24 + dosage_hits * 0.18)
    scores["laboratory_history"] += min(0.94, lab_hits * 0.08 + lab_value_hits * 0.28 + dated_hits * 0.04)

    if scores["therapy"] < 0.35 and scores["laboratory_history"] < 0.35:
        scores["anamnesis"] += 0.42
        if line_count >= 3:
            scores["anamnesis"] += 0.08
    if lab_value_hits == 0 and therapy_hits == 0 and line_count >= 4:
        scores["anamnesis"] += 0.08
    return scores


def _infer_section_match(
    raw_heading: str,
    content: str,
    *,
    allowed_keys: set[str],
    line_start: int,
    line_end: int,
) -> SectionHeadingMatch | None:
    scores = _score_section_content(raw_heading, content)
    ranked = sorted(
        ((key, score) for key, score in scores.items() if key in allowed_keys),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        return None
    best_key, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score < 0.55:
        return None
    if best_score - second_score < 0.12:
        return None
    return SectionHeadingMatch(
        canonical_key=best_key,
        raw_heading=raw_heading.strip(),
        normalized_heading=normalize_heading_text(raw_heading),
        score=min(0.89, best_score),
        strategy="content_inference",
        line_start=line_start,
        line_end=line_end,
    )


def parse_required_dili_sections(raw_text: str) -> ParsedDiliSectionsResult:
    scan_result = scan_dili_section_headings(raw_text)
    headings = resolve_heading_collisions(scan_result.section_headings)
    heading_lookup = {
        (heading.line_start, heading.line_end): heading
        for heading in headings
    }
    boundaries = _selected_heading_boundaries(raw_text)
    offsets = _line_char_offsets(raw_text)
    sections: dict[str, ClinicalRawSection] = {}
    unresolved_boundaries: list[tuple[HeadingBoundary, int, int, str]] = []
    malformed_sections: list[str] = []

    for boundary in boundaries:
        if boundary.line_start - 1 >= len(offsets):
            continue
        _, heading_line_end = offsets[boundary.line_start - 1]
        body_start = heading_line_end
        boundary_line = _next_boundary_line_start(
            boundary.line_start,
            scan_result.boundary_line_starts,
        )
        if boundary_line is not None and boundary_line - 1 < len(offsets):
            next_heading_line_start, _ = offsets[boundary_line - 1]
            body_end = next_heading_line_start
        else:
            body_end = len(raw_text)
        heading = heading_lookup.get((boundary.line_start, boundary.line_end))
        content = raw_text[body_start:body_end].strip("\r\n")
        if not content:
            if heading is not None:
                malformed_sections.append(f"empty:{heading.canonical_key}")
            continue
        if heading is None:
            unresolved_boundaries.append((boundary, body_start, body_end, content))
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
            malformed_sections.append(f"duplicate:{heading.canonical_key}")
            continue
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

    missing_keys = {
        key for key in REQUIRED_DILI_SECTION_KEYS if key not in sections
    }
    if unresolved_boundaries and missing_keys:
        inferred: list[tuple[SectionHeadingMatch, int, int, str]] = []
        for boundary, body_start, body_end, content in unresolved_boundaries:
            match = _infer_section_match(
                boundary.raw_heading,
                content,
                allowed_keys=missing_keys,
                line_start=boundary.line_start,
                line_end=boundary.line_end,
            )
            if match is not None:
                inferred.append((match, body_start, body_end, content))
        inferred.sort(key=lambda item: item[0].score, reverse=True)
        used_lines: set[int] = set()
        for match, body_start, body_end, content in inferred:
            if match.canonical_key in sections or match.line_start in used_lines:
                continue
            coherent = verify_verbatim_section_coherence(raw_text, ClinicalRawSection(
                canonical_key=match.canonical_key,
                raw_heading=match.raw_heading,
                normalized_heading=match.normalized_heading,
                match_strategy=match.strategy,
                confidence_score=match.score,
                line_start=match.line_start,
                line_end=match.line_end,
                body_start=body_start,
                body_end=body_end,
                text=content,
                verbatim_coherent=True,
            ))
            sections[match.canonical_key] = ClinicalRawSection(
                canonical_key=match.canonical_key,
                raw_heading=match.raw_heading,
                normalized_heading=match.normalized_heading,
                match_strategy=match.strategy,
                confidence_score=match.score,
                line_start=match.line_start,
                line_end=match.line_end,
                body_start=body_start,
                body_end=body_end,
                text=content,
                verbatim_coherent=coherent,
            )
            used_lines.add(match.line_start)

    remaining_missing = [key for key in REQUIRED_DILI_SECTION_KEYS if key not in sections]
    remaining_unresolved = [
        (boundary, body_start, body_end, content)
        for boundary, body_start, body_end, content in unresolved_boundaries
        if boundary.line_start not in {section.line_start for section in sections.values()}
    ]
    if len(remaining_missing) == 1 and len(remaining_unresolved) == 1:
        missing_key = remaining_missing[0]
        boundary, body_start, body_end, content = remaining_unresolved[0]
        sections[missing_key] = ClinicalRawSection(
            canonical_key=missing_key,
            raw_heading=boundary.raw_heading.strip(),
            normalized_heading=normalize_heading_text(boundary.raw_heading),
            match_strategy="fallback_assignment",
            confidence_score=0.56,
            line_start=boundary.line_start,
            line_end=boundary.line_end,
            body_start=body_start,
            body_end=body_end,
            text=content,
            verbatim_coherent=verify_verbatim_section_coherence(
                raw_text,
                ClinicalRawSection(
                    canonical_key=missing_key,
                    raw_heading=boundary.raw_heading.strip(),
                    normalized_heading=normalize_heading_text(boundary.raw_heading),
                    match_strategy="fallback_assignment",
                    confidence_score=0.56,
                    line_start=boundary.line_start,
                    line_end=boundary.line_end,
                    body_start=body_start,
                    body_end=body_end,
                    text=content,
                    verbatim_coherent=True,
                ),
            ),
        )
    return ParsedDiliSectionsResult(
        sections=sections,
        missing_required_sections=missing_required_section_names(sections),
        malformed_sections=sorted(set(malformed_sections)),
    )


def extract_required_dili_sections(raw_text: str) -> dict[str, ClinicalRawSection]:
    result = parse_required_dili_sections(raw_text)
    duplicate_errors = [
        issue for issue in result.malformed_sections if issue.startswith("duplicate:")
    ]
    if duplicate_errors:
        raise ValueError(
            f"Duplicate section heading for '{duplicate_errors[0].split(':', 1)[1]}'."
        )
    return result.sections


def _next_boundary_line_start(current_line_start: int, boundary_line_starts: set[int]) -> int | None:
    for line_start in sorted(boundary_line_starts):
        if line_start > current_line_start:
            return line_start
    return None


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
