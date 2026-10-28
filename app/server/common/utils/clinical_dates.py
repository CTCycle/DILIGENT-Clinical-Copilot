# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import re
from datetime import date
from typing import Literal

DateOrder = Literal["day_first", "month_first"]

ISO_DATE_RE = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$")
NUMERIC_DATE_RE = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})$")
NUMERIC_DATE_TOKEN_RE = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})\b")


###############################################################################
def detect_date_order(*texts: str | None) -> DateOrder | None:
    """Infer the numeric date convention from unambiguous tokens in the record.

    A token such as 20/05/2024 proves day-first and 05/20/2024 proves
    month-first. Returns None when the record gives no evidence or conflicting
    evidence.
    """
    day_first = month_first = False
    for text in texts:
        for first, second, _ in NUMERIC_DATE_TOKEN_RE.findall(text or ""):
            first_value, second_value = int(first), int(second)
            if first_value > 12 >= second_value >= 1:
                day_first = True
            elif second_value > 12 >= first_value >= 1:
                month_first = True
    if day_first == month_first:
        return None
    return "day_first" if day_first else "month_first"


###############################################################################
def case_date_order(*texts: str | None) -> DateOrder:
    """Date order to apply to a clinical record; day-first unless the record proves otherwise."""
    return detect_date_order(*texts) or "day_first"


###############################################################################
def has_ambiguous_date_order(*texts: str | None) -> bool:
    """True when day/month order matters for some token but the record does not settle it."""
    if detect_date_order(*texts) is not None:
        return False
    return any(
        first != second and int(first) <= 12 and int(second) <= 12
        for text in texts
        for first, second, _ in NUMERIC_DATE_TOKEN_RE.findall(text or "")
    )


###############################################################################
def parse_clinical_date(value: str | None, order: DateOrder = "day_first") -> date | None:
    """Parse ISO or numeric day/month dates with an explicit day/month order.

    The requested order is applied to ambiguous tokens; a token that is only
    valid in the other order (for example 05/20/2024 read day-first) is parsed
    in that order because it is unambiguous.
    """
    if not value:
        return None
    text = str(value).strip()
    iso_match = ISO_DATE_RE.match(text)
    if iso_match:
        return _safe_date(*(int(part) for part in iso_match.groups()))
    numeric_match = NUMERIC_DATE_RE.match(text)
    if numeric_match is None:
        return None
    first, second, year_text = numeric_match.groups()
    year = int(year_text) + (2000 if len(year_text) == 2 else 0)
    day, month = (int(first), int(second))
    if order == "month_first":
        day, month = month, day
    return _safe_date(year, month, day) or _safe_date(year, day, month)


###############################################################################
def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None
