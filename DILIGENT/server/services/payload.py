from __future__ import annotations

import html
import re
import unicodedata
from typing import Any
from datetime import datetime, date


HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"[ \t]+")
DRUG_BULLET_PREFIX_RE = re.compile(r"^[\-\u2022\u2023\u2043\*\u25A0\u25AA\u25CF\u25E6\u2219\u00B7]+\s*")
DRUG_ALLOWED_SYMBOLS_RE = re.compile(r"[^\w\s.,;:/()\-+%[\]'\"<>=]")
ANAMNESIS_ALLOWED_SYMBOLS_RE = re.compile(r"[^\w\s.,;:/()\-+%[\]'\"<>=]")


# HELPERS
###############################################################################
def sanitize_field(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


# -----------------------------------------------------------------------------
def strip_html(value: str) -> str:
    unescaped = html.unescape(value)
    return HTML_TAG_RE.sub(" ", unescaped)


# -----------------------------------------------------------------------------
def sanitize_drug_line(value: str) -> str:
    stripped_html = strip_html(value)
    without_bullet = DRUG_BULLET_PREFIX_RE.sub("", stripped_html)
    without_marks = without_bullet.replace("!", " ").replace("?", " ")
    without_symbols = DRUG_ALLOWED_SYMBOLS_RE.sub(" ", without_marks)
    compact = MULTISPACE_RE.sub(" ", without_symbols).strip(" \t,;:-")
    return compact


# -----------------------------------------------------------------------------
def sanitize_drugs_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for raw_line in normalized.split("\n"):
        cleaned = sanitize_drug_line(raw_line)
        if cleaned:
            lines.append(cleaned)
    if not lines:
        return None
    return "\n".join(lines)


# -----------------------------------------------------------------------------
def sanitize_anamnesis_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    stripped_html = strip_html(normalized)
    without_marks = stripped_html.replace("!", " ").replace("?", " ")
    without_symbols = ANAMNESIS_ALLOWED_SYMBOLS_RE.sub(" ", without_marks)
    cleaned_lines: list[str] = []
    for raw_line in without_symbols.split("\n"):
        compact = MULTISPACE_RE.sub(" ", raw_line).strip()
        if compact:
            cleaned_lines.append(compact)
    if not cleaned_lines:
        return None
    return "\n".join(cleaned_lines)

# -----------------------------------------------------------------------------
def normalize_visit_date(
    value: datetime | date | dict[str, Any] | str | None,
) -> date | None:
    if value is None:
        return None

    if isinstance(value, dict):
        day_raw = value.get("day")
        month_raw = value.get("month")
        year_raw = value.get("year")

        if not isinstance(day_raw, (str, int)):
            return None
        if not isinstance(month_raw, (str, int)):
            return None
        if not isinstance(year_raw, (str, int)):
            return None

        try:
            day = int(day_raw)
            month = int(month_raw)
            year = int(year_raw)
            normalized = date(year, month, day)
        except (TypeError, ValueError):
            return None

    elif isinstance(value, datetime):
        normalized = value.date()

    elif isinstance(value, date):
        normalized = value

    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed_datetime = datetime.fromisoformat(stripped)
        except ValueError:
            try:
                normalized = date.fromisoformat(stripped)
            except ValueError:
                return None
        else:
            normalized = parsed_datetime.date()
    else:
        return None

    today = date.today()
    if normalized > today:
        return today
    return normalized

##############################################################################
def sanitize_dili_payload(
    *,
    patient_name: str | None,
    visit_date: datetime | date | dict[str, Any] | str | None,
    anamnesis: str | None,   
    drugs: str | None,
    alt: str | None,
    alt_max: str | None,
    alp: str | None,
    alp_max: str | None,
    use_rag: bool,
    use_web_search: bool = False,
) -> dict[str, Any]:
    normalized_visit_date = normalize_visit_date(visit_date)

    return {
        "name": sanitize_field(patient_name),
        "visit_date": (
            {
                "day": normalized_visit_date.day,
                "month": normalized_visit_date.month,
                "year": normalized_visit_date.year,
            }
            if normalized_visit_date
            else None
        ),
        "anamnesis": sanitize_anamnesis_text(anamnesis),
        "drugs": sanitize_drugs_text(drugs),
        "alt": sanitize_field(alt),
        "alt_max": sanitize_field(alt_max),
        "alp": sanitize_field(alp),
        "alp_max": sanitize_field(alp_max),
        "use_rag": bool(use_rag),
        "use_web_search": bool(use_web_search),
    }


##############################################################################
class PayloadSanitizationService:
    # -------------------------------------------------------------------------
    @staticmethod
    def sanitize_dili_payload(
        *,
        patient_name: str | None,
        visit_date: datetime | date | dict[str, Any] | str | None,
        anamnesis: str | None,
        drugs: str | None,
        alt: str | None,
        alt_max: str | None,
        alp: str | None,
        alp_max: str | None,
        use_rag: bool,
        use_web_search: bool = False,
    ) -> dict[str, Any]:
        return sanitize_dili_payload(
            patient_name=patient_name,
            visit_date=visit_date,
            anamnesis=anamnesis,
            drugs=drugs,
            alt=alt,
            alt_max=alt_max,
            alp=alp,
            alp_max=alp_max,
            use_rag=use_rag,
            use_web_search=use_web_search,
        )
