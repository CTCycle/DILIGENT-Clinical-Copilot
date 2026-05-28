# ruff: noqa: F405
from datetime import datetime

from services.clinical.hepatox_assessment import *  # noqa: F403

# Extracted from the facade helper module; functions intentionally accept the facade instance.

def evaluate_suspension(
    self, entry: DrugEntry, visit_date: date | None
) -> DrugSuspensionContext:
    start_reported = bool(entry.therapy_start_status) or bool(
        entry.therapy_start_date
    )
    start_date = self.parse_start_date(entry.therapy_start_date, visit_date)
    start_interval_days: int | None = None
    if start_reported and start_date is not None and visit_date is not None:
        start_interval_days = (visit_date - start_date).days
    start_note = self.format_start_note(
        start_reported=start_reported,
        start_date=start_date,
        start_interval_days=start_interval_days,
        visit_date=visit_date,
    )

    suspended = bool(entry.suspension_status)
    parsed_date = self.parse_suspension_date(entry.suspension_date, visit_date)
    interval_days: int | None = None
    if not suspended:
        # No suspension means we track exposure but keep contextual notes
        if entry.source == "anamnesis" or bool(entry.historical_flag):
            exposure_note = (
                "Historical mention from anamnesis without explicit active regimen; "
                "treat current exposure as uncertain unless confirmed in therapy."
            )
        else:
            exposure_note = "Active therapy; no suspension reported."
        combined_note = " ".join(
            part
            for part in (
                start_note,
                exposure_note,
            )
            if part
        )
        return DrugSuspensionContext(
            suspended=False,
            suspension_date=None,
            excluded=False,
            note=combined_note or None,
            interval_days=None,
            start_reported=start_reported,
            start_date=start_date,
            start_interval_days=start_interval_days,
            start_note=start_note,
        )

    if parsed_date is None:
        suspension_note = (
            "Suspension reported without a reliable date; drug kept in analysis."
        )
    elif visit_date is None:
        suspension_note = f"Suspended on {parsed_date.isoformat()}, but visit date missing; drug kept in analysis."
    else:
        interval_days = (visit_date - parsed_date).days
        if interval_days < 0:
            suspension_note = f"Suspended on {parsed_date.isoformat()} ({abs(interval_days)} days after the visit); treat as ongoing exposure."
        elif interval_days == 0:
            suspension_note = f"Suspended on {parsed_date.isoformat()} (same day as the visit); residual exposure is expected."
        else:
            suspension_note = f"Suspended on {parsed_date.isoformat()} ({interval_days} days before the visit); compare this latency with LiverTox guidance."

    combined_note = " ".join(part for part in (start_note, suspension_note) if part)
    return DrugSuspensionContext(
        suspended=suspended,
        suspension_date=parsed_date,
        excluded=False,
        note=combined_note or None,
        interval_days=interval_days,
        start_reported=start_reported,
        start_date=start_date,
        start_interval_days=start_interval_days,
        start_note=start_note,
    )

def parse_timeline_date(
    self, raw_date: str | None, visit_date: date | None
) -> date | None:
    if raw_date is None:
        return None
    text = str(raw_date).strip()
    if not text:
        return None
    normalized = text.replace("/", "-").replace(".", "-").replace(",", "-")
    tokens = [token for token in normalized.split("-") if token]
    candidates: list[str] = []
    if visit_date is not None and len(tokens) == 2:
        day, month = tokens
        candidates.extend(
            [
                f"{day.zfill(2)}-{month.zfill(2)}-{visit_date.year}",
                f"{month.zfill(2)}-{day.zfill(2)}-{visit_date.year}",
                f"{visit_date.year}-{month.zfill(2)}-{day.zfill(2)}",
            ]
        )
    candidates.append("-".join(tokens))
    candidates.append(text)
    candidates.append(normalized)
    checked: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in checked:
            continue
        checked.add(candidate)
        parsed = self.try_parse_date(candidate)
        if parsed is not None:
            return parsed
    return None

def parse_suspension_date(
    self, raw_date: str | None, visit_date: date | None
) -> date | None:
    return self.parse_timeline_date(raw_date, visit_date)

def parse_start_date(
    self, raw_date: str | None, visit_date: date | None
) -> date | None:
    return self.parse_timeline_date(raw_date, visit_date)

def humanize_interval(self, days: int) -> str:
    if days <= 1:
        return "1 day"
    if days < 14:
        return f"{days} days"
    weeks = days / 7
    if days < 60:
        rounded_weeks = round(weeks, 1)
        return f"{rounded_weeks:g} weeks"
    months = days / 30.4375
    if days < 365:
        rounded_months = round(months, 1)
        return f"{rounded_months:g} months"
    years = days / 365.25
    rounded_years = round(years, 1)
    return f"{rounded_years:g} years"

def try_parse_date(value: str) -> date | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    iso_candidate = cleaned.replace(".", "-").replace("/", "-")
    try:
        return date.fromisoformat(iso_candidate)
    except ValueError:
        pass
    for fmt in ("%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d", "%d.%m.%Y", "%Y.%m.%d"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None

def resolve_livertox_score(self, metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return NOT_AVAILABLE_TEXT
    score = metadata.get("likelihood_score")
    if score is None:
        return NOT_AVAILABLE_TEXT
    text = str(score).strip()
    if not text or text.lower() == "nan":
        return NOT_AVAILABLE_TEXT
    return text.upper() if text.isalpha() else text
