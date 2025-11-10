from __future__ import annotations

from typing import Any
from datetime import datetime, date
from fastapi import HTTPException, status


# HELPERS
###############################################################################
def sanitize_field(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None

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
) -> dict[str, Any]:
    if anamnesis is None or drugs is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Both anamnesis and drugs fields are required.",
        )

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
        "anamnesis": sanitize_field(anamnesis),
        "drugs": sanitize_field(drugs),
        "alt": sanitize_field(alt),
        "alt_max": sanitize_field(alt_max),
        "alp": sanitize_field(alp),
        "alp_max": sanitize_field(alp_max),
        "use_rag": bool(use_rag),
    }
