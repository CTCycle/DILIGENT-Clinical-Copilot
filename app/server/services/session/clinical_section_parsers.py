from __future__ import annotations

from dataclasses import dataclass
import re

from domain.clinical.sections import ClinicalSectionKey


NORMALIZE_RE = re.compile(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9 ]+")
WS_RE = re.compile(r"\s+")
HSEP_HEADING_RE = re.compile(
    r"(?m)^[ \t]*---[ \t]*\r?\n[ \t]*(?P<title>[^\r\n]{1,100}?)[ \t]*\r?\n[ \t]*---[ \t]*$"
)
MARKER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?m)^[ \t]*#{1,6}[ \t]+(?P<title>[^\r\n]{1,100})$"),
    re.compile(r"(?m)^[ \t]*\d+[ \t]*[.):][ \t]*(?P<title>[^\r\n]{1,100})$"),
    re.compile(r"(?m)^[ \t]*(?:I|II|III|IV|V|VI|VII|VIII|IX|X)[ \t]*[.):][ \t]*(?P<title>[^\r\n]{1,100})$", re.IGNORECASE),
    re.compile(r"(?m)^[ \t]*(?P<title>[A-Za-zÀ-ÖØ-öø-ÿ][^\r\n:]{0,98})[ \t]*:[ \t]*$"),
    HSEP_HEADING_RE,
)

ANAMNESIS_ALIASES = {
    "anamnesis",
    "anamnesi",
    "history",
    "clinical history",
}
THERAPY_ALIASES = {
    "therapy",
    "therapies",
    "current therapy",
    "drugs",
    "current drugs",
    "current medications",
    "medications",
    "farmaci",
    "terapia",
}
LAB_ALIASES = {
    "lab analysis",
    "laboratory analysis",
    "laboratory",
    "labs",
    "lab",
    "analisi",
    "esami",
    "esami di laboratorio",
}


@dataclass(frozen=True)
class SectionMarker:
    key: ClinicalSectionKey
    title: str
    marker_start: int
    marker_end: int


def normalize_section_title(title: str) -> str | None:
    normalized = NORMALIZE_RE.sub(" ", title or "")
    normalized = WS_RE.sub(" ", normalized).strip().lower()
    if not normalized:
        return None
    return normalized


def _map_title_to_key(title: str) -> ClinicalSectionKey | None:
    normalized = normalize_section_title(title)
    if normalized is None:
        return None
    if normalized in ANAMNESIS_ALIASES:
        return "anamnesis"
    if normalized in THERAPY_ALIASES:
        return "drugs"
    if normalized in LAB_ALIASES:
        return "laboratory_analysis"
    return None


def find_section_markers(text: str) -> list[SectionMarker]:
    markers: list[SectionMarker] = []
    for pattern in MARKER_PATTERNS:
        for match in pattern.finditer(text):
            title = (match.group("title") or "").strip()
            key = _map_title_to_key(title)
            if key is None:
                continue
            markers.append(
                SectionMarker(
                    key=key,
                    title=title,
                    marker_start=match.start(),
                    marker_end=match.end(),
                )
            )
    # Also support standalone heading lines without markdown/colon markers,
    # e.g. "Anamnesis" followed by section content on next lines.
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.strip()
        line_key = _map_title_to_key(line)
        if (
            line_key is not None
            and line
            and not line.startswith(("-", "*", "+"))
            and ":" not in line
        ):
            marker_start = offset + raw_line.find(line)
            marker_end = marker_start + len(line)
            markers.append(
                SectionMarker(
                    key=line_key,
                    title=line,
                    marker_start=marker_start,
                    marker_end=marker_end,
                )
            )
        offset += len(raw_line)
    unique: dict[tuple[int, int, ClinicalSectionKey], SectionMarker] = {}
    for marker in markers:
        unique[(marker.marker_start, marker.marker_end, marker.key)] = marker
    return sorted(unique.values(), key=lambda marker: marker.marker_start)


def extract_sections_from_markers(text: str, markers: list[SectionMarker]) -> dict[ClinicalSectionKey, str] | None:
    if not markers:
        return None
    extracted: dict[ClinicalSectionKey, str] = {}
    ordered = sorted(markers, key=lambda item: item.marker_start)
    for index, marker in enumerate(ordered):
        body_start = marker.marker_end
        body_end = ordered[index + 1].marker_start if index + 1 < len(ordered) else len(text)
        content = text[body_start:body_end].strip()
        if not content:
            continue
        existing = extracted.get(marker.key)
        extracted[marker.key] = f"{existing}\n\n{content}" if existing else content
    if not extracted.get("anamnesis") or not extracted.get("drugs") or not extracted.get("laboratory_analysis"):
        return None
    return extracted


def validate_sections_against_source(text: str, sections: dict[ClinicalSectionKey, str]) -> bool:
    normalized_source = WS_RE.sub(" ", text).strip()
    for key in ("anamnesis", "drugs", "laboratory_analysis"):
        value = (sections.get(key) or "").strip()
        if not value:
            return False
        if value in text:
            continue
        normalized_value = WS_RE.sub(" ", value).strip()
        if not normalized_value or normalized_value not in normalized_source:
            return False
    return True
