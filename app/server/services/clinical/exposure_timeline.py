from __future__ import annotations

from datetime import date, datetime

from domain.clinical.entities import DrugEntry, DrugSuspensionContext


###############################################################################
class ExposureTimelineService:
    # -------------------------------------------------------------------------
    def evaluate_suspension(
        self, entry: DrugEntry, visit_date: date | None
    ) -> DrugSuspensionContext:
        start_reported = bool(entry.therapy_start_status) or bool(
            entry.therapy_start_date
        )
        start_date = self.parse_start_date(entry.therapy_start_date, visit_date)
        start_interval_days = (
            (visit_date - start_date).days
            if start_reported and start_date is not None and visit_date is not None
            else None
        )
        start_note = self.format_start_note(
            start_reported=start_reported,
            start_date=start_date,
            start_interval_days=start_interval_days,
            visit_date=visit_date,
        )
        suspended = bool(entry.suspension_status)
        parsed_date = self.parse_suspension_date(entry.suspension_date, visit_date)
        if not suspended:
            exposure_note = (
                "Historical mention from anamnesis without explicit active regimen; treat current exposure as uncertain unless confirmed in therapy."
                if entry.source == "anamnesis" or bool(entry.historical_flag)
                else "Active therapy; no suspension reported."
            )
            return DrugSuspensionContext(
                suspended=False,
                excluded=False,
                note=" ".join(part for part in (start_note, exposure_note) if part)
                or None,
                start_reported=start_reported,
                start_date=start_date,
                start_interval_days=start_interval_days,
                start_note=start_note,
            )
        interval_days: int | None = None
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
        return DrugSuspensionContext(
            suspended=suspended,
            excluded=False,
            suspension_date=parsed_date,
            note=" ".join(part for part in (start_note, suspension_note) if part)
            or None,
            interval_days=interval_days,
            start_reported=start_reported,
            start_date=start_date,
            start_interval_days=start_interval_days,
            start_note=start_note,
        )

    # -------------------------------------------------------------------------
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
        candidates.extend(["-".join(tokens), text, normalized])
        checked: set[str] = set()
        for candidate in candidates:
            if candidate and candidate not in checked:
                checked.add(candidate)
                parsed = self.try_parse_date(candidate)
                if parsed is not None:
                    return parsed
        return None

    # -------------------------------------------------------------------------
    def parse_suspension_date(
        self, raw_date: str | None, visit_date: date | None
    ) -> date | None:
        return self.parse_timeline_date(raw_date, visit_date)

    # -------------------------------------------------------------------------
    def parse_start_date(
        self, raw_date: str | None, visit_date: date | None
    ) -> date | None:
        return self.parse_timeline_date(raw_date, visit_date)

    # -------------------------------------------------------------------------
    def format_start_note(
        self,
        *,
        start_reported: bool,
        start_date: date | None,
        start_interval_days: int | None,
        visit_date: date | None,
    ) -> str:
        if not start_reported:
            return "Therapy start was not documented; assume chronic exposure unless another source clarifies the onset."
        if start_date is None:
            return "Therapy start was reported but no reliable date could be parsed from the notes."
        if visit_date is None or start_interval_days is None:
            return f"Therapy started on {start_date.isoformat()}, but the visit date was unavailable for latency comparisons."
        if start_interval_days < 0:
            return f"Therapy was documented to start on {start_date.isoformat()}, {self.humanize_interval(abs(start_interval_days))} after the visit; verify this discrepancy manually."
        if start_interval_days == 0:
            return f"Therapy started on {start_date.isoformat()}, coinciding with the clinical visit."
        return f"Therapy started on {start_date.isoformat()}, roughly {self.humanize_interval(start_interval_days)} before the visit."

    # -------------------------------------------------------------------------
    @staticmethod
    def humanize_interval(days: int) -> str:
        if days <= 1:
            return "1 day"
        if days < 14:
            return f"{days} days"
        weeks = days / 7
        if days < 60:
            return f"{round(weeks, 1):g} weeks"
        months = days / 30.4375
        if days < 365:
            return f"{round(months, 1):g} months"
        return f"{round(days / 365.25, 1):g} years"

    # -------------------------------------------------------------------------
    @staticmethod
    def try_parse_date(value: str) -> date | None:
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return date.fromisoformat(cleaned.replace(".", "-").replace("/", "-"))
        except ValueError:
            pass
        for fmt in ("%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d", "%d.%m.%Y", "%Y.%m.%d"):
            try:
                return datetime.strptime(cleaned, fmt).date()
            except ValueError:
                continue
        return None

    # -------------------------------------------------------------------------
    def format_suspension_prompt(self, suspension: DrugSuspensionContext) -> str:
        if not suspension.suspended:
            return "Active therapy; no suspension reported."
        if suspension.suspension_date is None:
            return "Reported as suspended without a reliable date; evaluate latency with the LiverTox excerpt."
        if suspension.interval_days is None:
            return f"Suspended on {suspension.suspension_date.isoformat()}, but the interval relative to the visit is unclear; rely on LiverTox latency guidance."
        if suspension.interval_days < 0:
            return f"Suspended on {suspension.suspension_date.isoformat()} ({abs(suspension.interval_days)} days after the visit); treat as ongoing exposure."
        if suspension.interval_days == 0:
            return f"Suspended on {suspension.suspension_date.isoformat()} (same day as the visit); residual exposure is expected."
        return f"Suspended on {suspension.suspension_date.isoformat()} ({suspension.interval_days} days before the visit); compare with LiverTox latency guidance."

    # -------------------------------------------------------------------------
    def format_start_prompt(self, suspension: DrugSuspensionContext) -> str:
        if suspension.start_note:
            return suspension.start_note
        if suspension.start_reported:
            return "Therapy start was reported, but no reliable date was available."
        return "No therapy start information was detected; treat the exposure window as chronic unless contradicted."

    # -------------------------------------------------------------------------
    @staticmethod
    def format_visit_date_anchor(visit_date: date | None) -> str:
        return "Not provided." if visit_date is None else visit_date.isoformat()
