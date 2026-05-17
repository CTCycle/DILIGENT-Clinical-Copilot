from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from domain.clinical.sections import ClinicalSectionKey


NORMALIZE_RE = re.compile(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9 ]+")
WS_RE = re.compile(r"\s+")
EMPHASIS_EDGE_RE = re.compile(r"^[*_`~\s]+|[*_`~\s]+$")
ROMAN_NUMERAL_RE = re.compile(r"^(?:[ivxlcdm]+)$", re.IGNORECASE)
LAB_CUE_RE = re.compile(
    r"\b(?:alat|asat|alt|ast|alp|ggt|bilirub|egfr|clcr|creat|cr\b|u/l|umol|mg/dl|labor)\b",
    re.IGNORECASE,
)
DRUG_CUE_RE = re.compile(
    r"\b(?:per os|s\.?c\.?|i\.?v\.?|sospes[oa]|terapia|farmac|medic|dose|dosaggio|dal\s+\d{1,2}[./-]\d{1,2})\b",
    re.IGNORECASE,
)
DRUG_SCHEDULE_RE = re.compile(r"\b\d+(?:[.,]\d+)?-\d+(?:[.,]\d+)?-\d+(?:[.,]\d+)?-\d+(?:[.,]\d+)?\b")
ANAMNESIS_CUE_RE = re.compile(
    r"\b(?:paziente|anamnes|storia clinica|noto per|ricoverat|comorbid|diagnosi)\b",
    re.IGNORECASE,
)
MIN_FUZZY_SIMILARITY = 0.82
MAX_FUZZY_EDIT_DISTANCE_SHORT = 1
MAX_FUZZY_EDIT_DISTANCE_LONG = 2
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
    "anamnesi rilevante",
    "history",
    "clinical history",
    "medical history",
    "storia clinica",
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
    "terapia farmacologica",
    "terapia in corso",
    "farmacoterapia",
    "drug exposure",
    "drug exposure information",
}
LAB_ALIASES = {
    "lab analysis",
    "laboratory analysis",
    "laboratory",
    "labs",
    "lab",
    "laboratorio",
    "analisi di laboratorio",
    "analisi",
    "esami",
    "esami di laboratorio",
    "laboratory data",
    "dati di laboratorio",
}


@dataclass(frozen=True)
class SectionMarker:
    key: ClinicalSectionKey
    title: str
    marker_start: int
    marker_end: int


def normalize_section_title(title: str) -> str | None:
    raw = (title or "").strip()
    if not raw:
        return None
    raw = EMPHASIS_EDGE_RE.sub("", raw)
    normalized = NORMALIZE_RE.sub(" ", raw)
    normalized = WS_RE.sub(" ", normalized).strip().lower()
    if not normalized:
        return None
    return normalized


def _title_alias_match(normalized: str, aliases: set[str]) -> bool:
    if normalized in aliases:
        return True
    word_count = len(normalized.split(" "))
    strict_single_word_aliases = {"terapia", "therapy"}
    for alias in aliases:
        alias_word_count = len(alias.split(" "))
        if alias_word_count >= 2 or alias not in strict_single_word_aliases:
            if normalized.startswith(f"{alias} "):
                return True
            if normalized.endswith(f" {alias}"):
                return True
            if word_count <= 6 and f" {alias} " in normalized:
                return True
        if _is_near_title_match(normalized, alias):
            return True
    return False


def _is_near_title_match(candidate: str, alias: str) -> bool:
    if not candidate or not alias:
        return False
    if candidate == alias:
        return True
    edit_distance = _levenshtein_distance(candidate, alias)
    max_distance = (
        MAX_FUZZY_EDIT_DISTANCE_SHORT
        if max(len(candidate), len(alias)) <= 12
        else MAX_FUZZY_EDIT_DISTANCE_LONG
    )
    if edit_distance <= max_distance:
        return True
    similarity = _normalized_similarity(candidate, alias)
    if similarity >= MIN_FUZZY_SIMILARITY:
        return True

    # Token-level fallback for small differences in multi-word headings.
    cand_tokens = candidate.split(" ")
    alias_tokens = alias.split(" ")
    if len(cand_tokens) != len(alias_tokens):
        return False
    for cand_token, alias_token in zip(cand_tokens, alias_tokens, strict=False):
        token_distance = _levenshtein_distance(cand_token, alias_token)
        if token_distance > 1 and _normalized_similarity(cand_token, alias_token) < 0.8:
            return False
    return True


def _normalized_similarity(left: str, right: str) -> float:
    max_len = max(len(left), len(right))
    if max_len == 0:
        return 1.0
    distance = _levenshtein_distance(left, right)
    return 1.0 - (distance / max_len)


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left

    previous_row = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current_row = [i]
        for j, right_char in enumerate(right, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (left_char != right_char)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _map_title_to_key(title: str) -> ClinicalSectionKey | None:
    normalized = normalize_section_title(title)
    if normalized is None:
        return None
    if _title_alias_match(normalized, ANAMNESIS_ALIASES):
        return "anamnesis"
    if _title_alias_match(normalized, THERAPY_ALIASES):
        return "drugs"
    if _title_alias_match(normalized, LAB_ALIASES):
        return "laboratory_analysis"
    return None


def _is_probable_heading_line(raw_line: str) -> bool:
    stripped = raw_line.strip()
    if not stripped:
        return False
    if stripped.endswith((".", ";", "!", "?")):
        return False
    normalized = normalize_section_title(stripped)
    if normalized is None:
        return False
    words = normalized.split(" ")
    if len(words) > 12:
        return False
    if any(ROMAN_NUMERAL_RE.match(word) for word in words):
        return False
    # Allow numeric artifacts (e.g., patient IDs) only when the line still maps to
    # a known section title alias.
    if any(char.isdigit() for char in normalized) and _map_title_to_key(stripped) is None:
        return False
    return True


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
        if ":" in line and not line.endswith(":"):
            offset += len(raw_line)
            continue
        line_key = _map_title_to_key(line)
        if (
            line_key is not None
            and _is_probable_heading_line(raw_line)
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
    extracted: dict[ClinicalSectionKey, str] = _extract_inline_labeled_sections(text)
    ordered = sorted(markers, key=lambda item: item.marker_start)
    if ordered:
        for index, marker in enumerate(ordered):
            body_start = marker.marker_end
            body_end = ordered[index + 1].marker_start if index + 1 < len(ordered) else len(text)
            content = text[body_start:body_end].strip()
            if not content:
                continue
            existing = extracted.get(marker.key)
            extracted[marker.key] = f"{existing}\n\n{content}" if existing else content

    if _has_all_sections(extracted):
        return extracted

    inferred = _infer_sections_from_cues(text)
    if inferred is None:
        return None
    return inferred


def _has_all_sections(sections: dict[ClinicalSectionKey, str]) -> bool:
    return bool(
        (sections.get("anamnesis") or "").strip()
        and (sections.get("drugs") or "").strip()
        and (sections.get("laboratory_analysis") or "").strip()
    )


def _extract_inline_labeled_sections(text: str) -> dict[ClinicalSectionKey, str]:
    extracted: dict[ClinicalSectionKey, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        heading_part, body_part = line.split(":", 1)
        key = _map_title_to_key(heading_part.strip())
        body = body_part.strip()
        if key is None or not body:
            continue
        existing = extracted.get(key)
        extracted[key] = f"{existing}\n\n{body}" if existing else body
    return extracted


def _line_scores(line: str) -> tuple[int, int, int]:
    lab = 0
    drugs = 0
    anamnesis = 0
    lowered = line.strip().lower()
    if not lowered:
        return (lab, drugs, anamnesis)
    if LAB_CUE_RE.search(lowered):
        lab += 3
    if DRUG_CUE_RE.search(lowered):
        drugs += 3
    if DRUG_SCHEDULE_RE.search(lowered):
        drugs += 3
    if ANAMNESIS_CUE_RE.search(lowered):
        anamnesis += 2
    if ":" in lowered and any(token in lowered for token in ("alat", "asat", "alt", "ast", "alp", "ggt")):
        lab += 2
    return (lab, drugs, anamnesis)


def _infer_sections_from_cues(text: str) -> dict[ClinicalSectionKey, str] | None:
    lines = [line.rstrip() for line in text.splitlines()]
    if not lines:
        return None

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line)
            continue
        if current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    if not blocks:
        return None

    sections: dict[ClinicalSectionKey, str] = {}
    for block in blocks:
        block_text = "\n".join(block).strip()
        if not block_text:
            continue
        lab_score = drugs_score = anamnesis_score = 0
        for line in block:
            l_score, d_score, a_score = _line_scores(line)
            lab_score += l_score
            drugs_score += d_score
            anamnesis_score += a_score
        best_score = max(lab_score, drugs_score, anamnesis_score)
        if best_score <= 0:
            key: ClinicalSectionKey = "anamnesis"
        elif best_score == lab_score:
            key = "laboratory_analysis"
        elif best_score == drugs_score:
            key = "drugs"
        else:
            key = "anamnesis"
        existing = sections.get(key)
        sections[key] = f"{existing}\n\n{block_text}" if existing else block_text

    if _has_all_sections(sections):
        return sections

    line_buckets: dict[ClinicalSectionKey, list[str]] = {
        "anamnesis": [],
        "drugs": [],
        "laboratory_analysis": [],
    }
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lab_score, drugs_score, anamnesis_score = _line_scores(stripped)
        if lab_score == drugs_score == anamnesis_score == 0:
            line_buckets["anamnesis"].append(stripped)
            continue
        if lab_score >= drugs_score and lab_score >= anamnesis_score:
            line_buckets["laboratory_analysis"].append(stripped)
        elif drugs_score >= lab_score and drugs_score >= anamnesis_score:
            line_buckets["drugs"].append(stripped)
        else:
            line_buckets["anamnesis"].append(stripped)

    for key in ("anamnesis", "drugs", "laboratory_analysis"):
        if (sections.get(key) or "").strip():
            continue
        if line_buckets[key]:
            sections[key] = "\n".join(line_buckets[key]).strip()

    if not _has_all_sections(sections):
        return None
    return sections


def validate_sections_against_source(text: str, sections: dict[ClinicalSectionKey, str]) -> bool:
    normalized_source = WS_RE.sub(" ", text).strip()
    canonical_source = _canonicalize_text(normalized_source)
    for key in ("anamnesis", "drugs", "laboratory_analysis"):
        value = (sections.get(key) or "").strip()
        if not value:
            return False
        if value in text:
            continue
        normalized_value = WS_RE.sub(" ", value).strip()
        if not normalized_value:
            return False
        if normalized_value in normalized_source:
            continue
        canonical_value = _canonicalize_text(normalized_value)
        if not canonical_value or canonical_value not in canonical_source:
            return False
    return True


def _canonicalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9\s]+", " ", normalized)
    normalized = WS_RE.sub(" ", normalized).strip().lower()
    return re.sub(r"(?<=\d)\s+(?=\d)", "", normalized)
