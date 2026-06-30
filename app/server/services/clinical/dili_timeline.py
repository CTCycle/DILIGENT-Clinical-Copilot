from __future__ import annotations

import re
from datetime import date, datetime

from domain.clinical.dili import ClinicalEvidenceQuote, DiliTimeline, DiliTimelineEvent
from domain.clinical.entities import ClinicalLabEntry, DrugEntry, PatientLabTimeline

DATE_RE = re.compile(r"\b(\d{4}[-/.]\d{2}[-/.]\d{2}|\d{2}[-/.]\d{2}[-/.]\d{4})\b")
DOSE_CHANGE_RE = re.compile(
    r"\b(increase(?:d)?|decrease(?:d)?|reduce(?:d)?|taper(?:ed)?|dose change)\b",
    re.IGNORECASE,
)
RESTART_RE = re.compile(
    r"\b(restart(?:ed)?|resum(?:ed|ption)|reintroduc(?:ed|tion))\b",
    re.IGNORECASE,
)
RECHALLENGE_RE = re.compile(r"\b(rechallenge|re[- ]expos(?:e|ure)|re-exposure)\b", re.IGNORECASE)
SYMPTOM_RE = re.compile(
    r"\b(jaundice|icterus|pruritus|fatigue|nausea|vomiting|abdominal pain|dark urine|itch(?:ing)?)\b",
    re.IGNORECASE,
)
JAUNDICE_RE = re.compile(r"\b(jaundice|icterus|bilirubin)\b", re.IGNORECASE)


###############################################################################
class DiliTimelineEngine:

    # -------------------------------------------------------------------------
    def build(self, drugs: list[DrugEntry], labs: PatientLabTimeline) -> DiliTimeline:
        events: list[DiliTimelineEvent] = []
        missing: list[str] = []

        for drug in drugs:
            events.extend(self._drug_events(drug, missing))

        dated_labs = self._dated_labs(labs)
        events.extend(self._lab_events(dated_labs))
        first_symptom = self._first_event_date(events, "symptom_onset")
        bilirubin_event = self._first_event_date(events, "jaundice_or_bilirubin_rise")
        first_abnormal = self._first_abnormal_liver_test_date(dated_labs)
        peak_dates = self._peak_dates(dated_labs)
        dechallenge_status, recovery_date, last_abnormal_date = self._dechallenge(
            drugs,
            dated_labs,
        )

        if first_symptom is None:
            missing.append("first_symptom_date")
        if bilirubin_event is None:
            missing.append("jaundice_or_bilirubin_timing")
        if first_abnormal is None:
            missing.append("first_abnormal_liver_test_date")

        events.sort(key=self._event_sort_key)
        return DiliTimeline(
            events=events,
            first_abnormal_liver_test_date=first_abnormal,
            first_symptom_date=first_symptom,
            jaundice_or_bilirubin_rise_date=bilirubin_event or peak_dates.get("BILIRUBIN"),
            peak_dates=peak_dates,
            dechallenge_status=dechallenge_status,
            recovery_date=recovery_date,
            last_abnormal_date=last_abnormal_date,
            missing_fields=sorted(set(missing)),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def parse_date(value: str | None) -> date | None:
        if not value:
            return None
        normalized = str(value).strip().replace("/", "-").replace(".", "-")
        try:
            return date.fromisoformat(normalized)
        except ValueError:
            pass
        for fmt in ("%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(normalized, fmt).date()
            except ValueError:
                continue
        return None

    # -------------------------------------------------------------------------
    def _drug_events(
        self,
        drug: DrugEntry,
        missing: list[str],
    ) -> list[DiliTimelineEvent]:
        events: list[DiliTimelineEvent] = []
        evidence = (drug.evidence or "").strip()

        for event_type, event_date in (
            ("drug_start", drug.therapy_start_date),
            ("drug_stop", drug.suspension_date),
        ):
            events.append(
                DiliTimelineEvent(
                    event_type=event_type,
                    event_date=event_date,
                    drug_name=drug.name,
                    evidence=self._quote(
                        claim=event_type.replace("_", " "),
                        quote=evidence or drug.name,
                        source_section=drug.source,
                        event_date=event_date,
                    ),
                )
            )
            if event_date is None:
                missing.append(f"{drug.name}:{event_type}_date")

        derived_date = self._extract_date_from_text(evidence)
        if evidence and DOSE_CHANGE_RE.search(evidence):
            events.append(
                DiliTimelineEvent(
                    event_type="dose_change",
                    event_date=derived_date,
                    drug_name=drug.name,
                    evidence=self._quote(
                        claim="dose change",
                        quote=evidence,
                        source_section=drug.source,
                        event_date=derived_date,
                    ),
                )
            )
        if evidence and RESTART_RE.search(evidence):
            events.append(
                DiliTimelineEvent(
                    event_type="drug_restart",
                    event_date=derived_date,
                    drug_name=drug.name,
                    evidence=self._quote(
                        claim="drug restart",
                        quote=evidence,
                        source_section=drug.source,
                        event_date=derived_date,
                    ),
                )
            )
        if evidence and RECHALLENGE_RE.search(evidence):
            events.append(
                DiliTimelineEvent(
                    event_type="drug_rechallenge",
                    event_date=derived_date,
                    drug_name=drug.name,
                    evidence=self._quote(
                        claim="rechallenge",
                        quote=evidence,
                        source_section=drug.source,
                        event_date=derived_date,
                    ),
                )
            )
        if evidence and SYMPTOM_RE.search(evidence):
            events.append(
                DiliTimelineEvent(
                    event_type="symptom_onset",
                    event_date=derived_date,
                    drug_name=drug.name,
                    evidence=self._quote(
                        claim="symptom onset",
                        quote=evidence,
                        source_section=drug.source,
                        event_date=derived_date,
                    ),
                )
            )
        if evidence and JAUNDICE_RE.search(evidence):
            events.append(
                DiliTimelineEvent(
                    event_type="jaundice_or_bilirubin_rise",
                    event_date=derived_date,
                    drug_name=drug.name,
                    evidence=self._quote(
                        claim="jaundice or bilirubin rise",
                        quote=evidence,
                        source_section=drug.source,
                        event_date=derived_date,
                    ),
                )
            )
        return events

    # -------------------------------------------------------------------------
    def _dated_labs(self, labs: PatientLabTimeline) -> list[ClinicalLabEntry]:
        dated = [
            item for item in labs.entries if item.sample_date and item.value is not None
        ]
        dated.sort(key=lambda item: self.parse_date(item.sample_date) or date.max)
        return dated

    # -------------------------------------------------------------------------
    def _lab_events(self, labs: list[ClinicalLabEntry]) -> list[DiliTimelineEvent]:
        events: list[DiliTimelineEvent] = []
        for lab in labs:
            marker = lab.marker_name.upper()
            events.append(
                DiliTimelineEvent(
                    event_type="laboratory_result",
                    event_date=lab.sample_date,
                    marker=marker,
                    value=lab.value,
                    uln=lab.upper_limit_normal,
                    evidence=self._quote(
                        claim=f"{marker} result",
                        quote=lab.evidence or f"{marker} {lab.value}",
                        source_section=lab.source,
                        event_date=lab.sample_date,
                    ),
                )
            )
            multiple = self._multiple(lab)
            if marker in {"ALT", "AST", "ALP", "BILIRUBIN", "TBIL"} and multiple is not None and multiple > 1:
                events.append(
                    DiliTimelineEvent(
                        event_type="abnormal_liver_test",
                        event_date=lab.sample_date,
                        marker=marker,
                        value=lab.value,
                        uln=lab.upper_limit_normal,
                        evidence=self._quote(
                            claim=f"abnormal {marker}",
                            quote=lab.evidence or f"{marker} {lab.value}",
                            source_section=lab.source,
                            event_date=lab.sample_date,
                        ),
                    )
                )
            if marker in {"BILIRUBIN", "TBIL"} and multiple is not None and multiple >= 2:
                events.append(
                    DiliTimelineEvent(
                        event_type="jaundice_or_bilirubin_rise",
                        event_date=lab.sample_date,
                        marker=marker,
                        value=lab.value,
                        uln=lab.upper_limit_normal,
                        evidence=self._quote(
                            claim="bilirubin rise",
                            quote=lab.evidence or f"{marker} {lab.value}",
                            source_section=lab.source,
                            event_date=lab.sample_date,
                        ),
                    )
                )
        return events

    # -------------------------------------------------------------------------
    def _first_abnormal_liver_test_date(self, labs: list[ClinicalLabEntry]) -> str | None:
        for lab in labs:
            multiple = self._multiple(lab)
            if multiple is not None and multiple > 1:
                return lab.sample_date
        return labs[0].sample_date if labs else None

    # -------------------------------------------------------------------------
    def _peak_dates(self, labs: list[ClinicalLabEntry]) -> dict[str, str | None]:
        peak_dates: dict[str, str | None] = {}
        for marker in ("ALT", "AST", "ALP", "GGT", "BILIRUBIN", "TBIL", "INR"):
            candidates = [
                item
                for item in labs
                if item.marker_name.upper() == marker and item.value is not None
            ]
            peak_dates[marker if marker != "TBIL" else "BILIRUBIN"] = (
                max(candidates, key=lambda item: float(item.value or 0)).sample_date
                if candidates
                else None
            )
        return peak_dates

    # -------------------------------------------------------------------------
    def _dechallenge(
        self,
        drugs: list[DrugEntry],
        labs: list[ClinicalLabEntry],
    ) -> tuple[str, str | None, str | None]:
        stop_dates = sorted(
            self.parse_date(item.suspension_date)
            for item in drugs
            if self.parse_date(item.suspension_date) is not None
        )
        if not stop_dates:
            return "no_follow_up", None, labs[-1].sample_date if labs else None

        stop_date = stop_dates[-1]
        alt_like = [
            item
            for item in labs
            if item.marker_name.upper() in {"ALT", "AST"} and self.parse_date(item.sample_date) is not None
        ]
        if len(alt_like) < 2:
            return "insufficient_interval", None, labs[-1].sample_date if labs else None

        after_stop = [
            item
            for item in alt_like
            if (self.parse_date(item.sample_date) or date.min) >= stop_date
        ]
        if len(after_stop) < 2:
            return "insufficient_interval", None, labs[-1].sample_date if labs else None

        baseline_multiples = [
            self._multiple(item) or 0.0
            for item in alt_like
            if (self.parse_date(item.sample_date) or date.min) <= stop_date
        ]
        baseline_multiple = max(baseline_multiples) if baseline_multiples else 0.0
        first_multiple = self._multiple(after_stop[0]) or 0.0
        last_multiple = self._multiple(after_stop[-1]) or 0.0
        last_date = after_stop[-1].sample_date
        if last_multiple > first_multiple * 1.2:
            return "worsening_after_stop", None, last_date
        if last_multiple <= 1.0:
            return "resolved_to_baseline", last_date, last_date
        if baseline_multiple > 0 and last_multiple <= baseline_multiple:
            return "resolved_to_baseline", last_date, last_date
        days_followed = (
            (self.parse_date(last_date) or stop_date) - stop_date
        ).days
        if days_followed >= 180 and last_multiple > 1.0:
            return "chronic_or_persistent", None, last_date
        if last_multiple < first_multiple * 0.5:
            return "improving_after_stop", None, last_date
        return "stable_abnormality", None, last_date

    # -------------------------------------------------------------------------
    @staticmethod
    def _extract_date_from_text(text: str) -> str | None:
        match = DATE_RE.search(text or "")
        return match.group(1).replace("/", "-").replace(".", "-") if match else None

    # -------------------------------------------------------------------------
    @staticmethod
    def _multiple(lab: ClinicalLabEntry) -> float | None:
        if lab.value is None or not lab.upper_limit_normal:
            return None
        if float(lab.upper_limit_normal) <= 0:
            return None
        return float(lab.value) / float(lab.upper_limit_normal)

    # -------------------------------------------------------------------------
    @staticmethod
    def _quote(
        *,
        claim: str,
        quote: str | None,
        source_section: str | None,
        event_date: str | None,
    ) -> ClinicalEvidenceQuote:
        return ClinicalEvidenceQuote(
            claim=claim,
            quote=quote,
            source_section=source_section,
            event_date=event_date,
            source_kind="patient_record" if quote else "missing",
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _event_sort_key(event: DiliTimelineEvent) -> tuple[date, str]:
        parsed = DiliTimelineEngine.parse_date(event.event_date) or date.max
        return parsed, event.event_type

    # -------------------------------------------------------------------------
    @staticmethod
    def _first_event_date(events: list[DiliTimelineEvent], event_type: str) -> str | None:
        dated = [
            event
            for event in events
            if event.event_type == event_type and DiliTimelineEngine.parse_date(event.event_date)
        ]
        if not dated:
            return None
        dated.sort(key=lambda event: DiliTimelineEngine.parse_date(event.event_date) or date.max)
        return dated[0].event_date
