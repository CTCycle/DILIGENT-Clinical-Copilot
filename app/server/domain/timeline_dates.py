from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import re


_DATE_TOKEN_RE = re.compile(
    r"^(?P<year>\d{4})(?:-(?P<month>\d{2})(?:-(?P<day>\d{2}))?)?$"
)
_DATE_TOKEN_IN_TEXT_RE = re.compile(r"\b\d{4}(?:-\d{2}(?:-\d{2})?)?\b")
_YEAR_DATE_CONTEXT_RE = re.compile(
    r"\b(?:in|during|since|from|on|dated|year)\s*$", re.IGNORECASE
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
def extract_single_explicit_timeline_date(
    text: str | None,
) -> TimelineDateInterval | None:
    """Return one unambiguous, valid ISO date token from source text."""
    if not text:
        return None

    tokens: set[str] = set()
    for match in _DATE_TOKEN_IN_TEXT_RE.finditer(text):
        token = match.group(0)
        if len(token) == 4:
            prefix = text[: match.start()].rstrip()
            suffix = text[match.end() :].strip(" \t\r\n.,;:!?)]}'\"")
            whole_field = not prefix and not suffix
            contextual_year = bool(_YEAR_DATE_CONTEXT_RE.search(prefix)) and not suffix
            if not whole_field and not contextual_year:
                continue
        tokens.add(token)

    if len(tokens) != 1:
        return None
    return normalize_timeline_interval(tokens.pop())

###############################################################################
def timeline_date_sort_key(value: str | None) -> tuple[int, str]:
    interval = normalize_timeline_interval(value)
    return (
        (interval.start_day, interval.value) if interval else (date.max.toordinal(), "")
    )
