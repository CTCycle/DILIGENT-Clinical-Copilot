# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import partial
import re

from common.utils.clinical_dates import (
    NUMERIC_DATE_TOKEN_RE,
    DateOrder,
    case_date_order,
    parse_clinical_date,
)


_DATE_TOKEN_RE = re.compile(
    r"^(?P<year>\d{4})(?:-(?P<month>\d{2})(?:-(?P<day>\d{2}))?)?$"
)
_ISO_TOKEN_RE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
_MONTH_YEAR_TOKEN_RE = re.compile(
    r"(?<![\d./-])(?P<month>0?[1-9]|1[0-2])[./](?P<year>\d{4})\b(?![./-]\d)"
)
_YEAR_TOKEN_RE = re.compile(r"(?<![\d./-])\d{4}\b(?![./-]\d)")
_YEAR_DATE_CONTEXT_RE = re.compile(
    r"\b(?:in|during|since|from|until|on|dated|year)\s*$", re.IGNORECASE
)
_UNIT_SUFFIX_RE = re.compile(
    r"\s*(?:mg|mcg|µg|ug|g|kg|ml|l|dl|iu|ui|u|mmol|µmol|umol|nmol|meq|%|/)(?![a-z])",
    re.IGNORECASE,
)
_TRAILING_PUNCTUATION = " \t\r\n.,;:!?)]}'\""
_MIN_YEAR = 1900
_MAX_YEAR = 2100
_MONTH_NAMES: dict[str, int] = {
    name: month
    for month, names in enumerate(
        (
            ("january", "jan", "gennaio", "januar", "janvier"),
            ("february", "feb", "febbraio", "februar", "février", "fevrier"),
            ("march", "mar", "marzo", "märz", "maerz", "mars"),
            ("april", "apr", "aprile", "avril"),
            ("may", "maggio", "mai"),
            ("june", "jun", "giugno", "juni", "juin"),
            ("july", "jul", "luglio", "juli", "juillet"),
            ("august", "aug", "agosto", "août", "aout"),
            ("september", "sep", "sept", "settembre", "septembre"),
            ("october", "oct", "ottobre", "oktober", "octobre"),
            ("november", "nov", "novembre"),
            ("december", "dec", "dicembre", "dezember", "décembre", "decembre"),
        ),
        start=1,
    )
    for name in names
}
_MONTH_NAME_PATTERN = "|".join(
    sorted((re.escape(name) for name in _MONTH_NAMES), key=len, reverse=True)
)
_NAMED_DAY_FIRST_RE = re.compile(
    rf"\b(?:(?P<day>\d{{1,2}})\.?\s+)?(?P<month>{_MONTH_NAME_PATTERN})\.?,?\s+(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
_NAMED_MONTH_FIRST_RE = re.compile(
    rf"\b(?P<month>{_MONTH_NAME_PATTERN})\.?\s+(?P<day>\d{{1,2}}),?\s+(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)


###############################################################################
@dataclass(frozen=True)
class TimelineDateInterval:
    value: str
    end_value: str | None
    start_day: int
    end_day: int
    precision: str


###############################################################################
def normalize_timeline_interval(
    value: str | None,
    end_value: str | None = None,
) -> TimelineDateInterval | None:
    """Parse a canonical day/month/year token and an optional inclusive end."""
    if not value or not isinstance(value, str):
        return None
    match = _DATE_TOKEN_RE.fullmatch(value.strip())
    if match is None:
        return None
    year = int(match.group("year"))
    month_text = match.group("month")
    day_text = match.group("day")
    try:
        if month_text is None:
            start = date(year, 1, 1)
            end = date(year, 12, 31)
            precision = "year"
        elif day_text is None:
            start = date(year, int(month_text), 1)
            end = (
                date(year, int(month_text) + 1, 1) - timedelta(days=1)
                if int(month_text) < 12
                else date(year, 12, 31)
            )
            precision = "month"
        else:
            start = date(year, int(month_text), int(day_text))
            end = start
            precision = "day"
    except ValueError:
        return None

    normalized_end: str | None = None
    if end_value:
        end_interval = normalize_timeline_interval(end_value)
        if end_interval is None or end_interval.start_day < start.toordinal():
            return TimelineDateInterval(
                value=value.strip(),
                end_value=None,
                start_day=start.toordinal(),
                end_day=end.toordinal(),
                precision=precision,
            )
        end = date.fromordinal(end_interval.end_day)
        normalized_end = end_interval.value
    return TimelineDateInterval(
        value=value.strip(),
        end_value=normalized_end,
        start_day=start.toordinal(),
        end_day=end.toordinal(),
        precision=precision,
    )


###############################################################################
@dataclass(frozen=True)
class EvidenceDate:
    """A calendar date stated in source text, with the verbatim token."""

    text: str
    interval: TimelineDateInterval


###############################################################################
def _is_followed_by_unit(text: str, end: int) -> bool:
    return bool(_UNIT_SUFFIX_RE.match(text, end))


###############################################################################
def _named_date_interval(match: re.Match[str]) -> TimelineDateInterval | None:
    month = _MONTH_NAMES.get(match.group("month").casefold())
    year = int(match.group("year"))
    if month is None or not _MIN_YEAR <= year <= _MAX_YEAR:
        return None
    day = match.group("day")
    token = f"{year:04d}-{month:02d}" + (f"-{int(day):02d}" if day else "")
    return normalize_timeline_interval(token)


###############################################################################
def _claim_match(
    found: list[tuple[int, EvidenceDate]],
    taken: list[tuple[int, int]],
    match: re.Match[str],
    interval: TimelineDateInterval | None,
) -> None:
    if interval is None:
        return
    if any(start < match.end() and match.start() < end for start, end in taken):
        return
    taken.append((match.start(), match.end()))
    found.append((match.start(), EvidenceDate(match.group(0), interval)))


###############################################################################
def find_evidence_dates(
    text: str | None,
    order: DateOrder = "day_first",
) -> list[EvidenceDate]:
    """Return the distinct calendar dates explicitly stated in source text.

    Recognizes ISO day/month tokens, numeric day/month/year tokens in the
    record's date order, numeric month/year tokens, and bare years only when
    the surrounding words mark them as a date. Doses and laboratory values
    (a number followed by a unit) are never read as dates.
    """
    if not text:
        return []
    found: list[tuple[int, EvidenceDate]] = []
    taken: list[tuple[int, int]] = []

    claim = partial(_claim_match, found, taken)

    for match in _ISO_TOKEN_RE.finditer(text):
        claim(match, normalize_timeline_interval(match.group(0)))
    for pattern in (_NAMED_MONTH_FIRST_RE, _NAMED_DAY_FIRST_RE):
        for match in pattern.finditer(text):
            claim(match, _named_date_interval(match))
    for match in NUMERIC_DATE_TOKEN_RE.finditer(text):
        parsed = parse_clinical_date(match.group(0), order)
        claim(match, normalize_timeline_interval(parsed.isoformat()) if parsed else None)
    for match in _MONTH_YEAR_TOKEN_RE.finditer(text):
        if _is_followed_by_unit(text, match.end()):
            continue
        month, year = int(match.group("month")), int(match.group("year"))
        if _MIN_YEAR <= year <= _MAX_YEAR:
            claim(match, normalize_timeline_interval(f"{year:04d}-{month:02d}"))
    for match in _YEAR_TOKEN_RE.finditer(text):
        year = int(match.group(0))
        if not _MIN_YEAR <= year <= _MAX_YEAR or _is_followed_by_unit(text, match.end()):
            continue
        prefix = text[: match.start()].rstrip()
        suffix = text[match.end() :].strip(_TRAILING_PUNCTUATION)
        if (not prefix and not suffix) or _YEAR_DATE_CONTEXT_RE.search(prefix):
            claim(match, normalize_timeline_interval(match.group(0)))

    unique: dict[str, EvidenceDate] = {}
    for _, item in sorted(found, key=lambda pair: pair[0]):
        unique.setdefault(item.interval.value, item)
    return list(unique.values())


###############################################################################
def extract_single_explicit_timeline_date(
    text: str | None,
    order: DateOrder | None = None,
) -> EvidenceDate | None:
    """Return the one unambiguous calendar date stated in source text."""
    dates = find_evidence_dates(text, order or case_date_order(text))
    return dates[0] if len(dates) == 1 else None


###############################################################################
def timeline_date_sort_key(value: str | None) -> tuple[int, str]:
    interval = normalize_timeline_interval(value)
    return (
        (interval.start_day, interval.value) if interval else (date.max.toordinal(), "")
    )
