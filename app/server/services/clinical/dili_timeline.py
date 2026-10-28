# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from common.utils.clinical_dates import parse_clinical_date
from common.utils.clinical_safety import contains_explicitly_negative_rechallenge
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
RECHALLENGE_RE = re.compile(
    r"\b(rechallenge|re[- ]expos(?:e|ure)|re-exposure)\b", re.IGNORECASE
)
SYMPTOM_RE = re.compile(
    r"\b(jaundice|icterus|pruritus|fatigue|nausea|vomiting|abdominal pain|dark urine|itch(?:ing)?)\b",
    re.IGNORECASE,
)
JAUNDICE_RE = re.compile(r"\b(jaundice|icterus|bilirubin)\b", re.IGNORECASE)
DechallengeStatus = Literal[
    "no_follow_up",
    "improving_after_stop",
    "worsening_after_stop",
    "stable_abnormality",
    "resolved_to_baseline",
    "chronic_or_persistent",
    "insufficient_interval",
]

###############################################################################
class DiliTimelineEngine:

    # -------------------------------------------------------------------------
    def build(
        self,
        drugs: list[DrugEntry],
        labs: PatientLabTimeline,
        source_text: str | None = None,
        primary_pattern: str = "indeterminate",
    ) -> DiliTimeline:
        events: list[DiliTimelineEvent] = []
        missing: list[str] = []

        for drug in drugs:
            events.extend(self._drug_events(drug, missing))
        events.extend(self._source_events(source_text))

        dated_labs = self._dated_labs(labs)
        events.extend(self._lab_events(dated_labs))
        first_symptom = self._first_event_date(events, "symptom_onset")
        bilirubin_event = self._first_event_date(events, "jaundice_or_bilirubin_rise")
        first_abnormal = self._first_abnormal_liver_test_date(dated_labs)
        peak_dates = self._peak_dates(dated_labs)
        dechallenge_status, recovery_date, last_abnormal_date = self._dechallenge(
            drugs,
            dated_labs,
            primary_pattern,
        )
        jaundice_date = bilirubin_event or peak_dates.get("BILIRUBIN")

        if first_symptom is None:
            missing.append("first_symptom_date")
        if jaundice_date is None:
            missing.append("jaundice_or_bilirubin_timing")
        if first_abnormal is None:
            missing.append("first_abnormal_liver_test_date")

        events.sort(key=self._event_sort_key)
        return DiliTimeline(
            events=events,
            first_abnormal_liver_test_date=first_abnormal,
            first_symptom_date=first_symptom,
            jaundice_or_bilirubin_rise_date=jaundice_date,
            peak_dates=peak_dates,
            dechallenge_status=dechallenge_status,
            recovery_date=recovery_date,
            last_abnormal_date=last_abnormal_date,
            missing_fields=sorted(set(missing)),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def parse_date(value: str | None) -> date | None:
        return parse_clinical_date(value)

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
        if (
            evidence
            and RECHALLENGE_RE.search(evidence)
            and not contains_explicitly_negative_rechallenge(evidence)
        ):
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
        symptom_match = SYMPTOM_RE.search(evidence)
        if symptom_match:
            symptom_date = self._date_near_position(evidence, symptom_match.start())
            events.append(
                DiliTimelineEvent(
                    event_type="symptom_onset",
                    event_date=symptom_date,
                    drug_name=drug.name,
                    evidence=self._quote(
                        claim="symptom onset",
                        quote=evidence,
                        source_section=drug.source,
                        event_date=symptom_date,
                    ),
                )
            )
        jaundice_match = JAUNDICE_RE.search(evidence)
        if jaundice_match:
            jaundice_date = self._date_near_position(evidence, jaundice_match.start())
            events.append(
                DiliTimelineEvent(
                    event_type="jaundice_or_bilirubin_rise",
                    event_date=jaundice_date,
                    drug_name=drug.name,
                    evidence=self._quote(
                        claim="jaundice or bilirubin rise",
                        quote=evidence,
                        source_section=drug.source,
                        event_date=jaundice_date,
                    ),
                )
            )
        return events

    # -------------------------------------------------------------------------
    def _source_events(self, source_text: str | None) -> list[DiliTimelineEvent]:
        events: list[DiliTimelineEvent] = []
        fragments = re.split(r"(?<=[.!?;])\s+|\r?\n+", str(source_text or ""))
        for fragment in fragments:
            fragment = fragment.strip()
            if not fragment:
                continue
            symptom_match = SYMPTOM_RE.search(fragment)
            if symptom_match:
                event_date = self._date_near_position(fragment, symptom_match.start())
                events.append(
                    DiliTimelineEvent(
                        event_type="symptom_onset",
                        event_date=event_date,
                        evidence=self._quote(
                            claim="symptom onset",
                            quote=fragment,
                            source_section="anamnesis",
                            event_date=event_date,
                        ),
                    )
                )
            jaundice_match = JAUNDICE_RE.search(fragment)
            if jaundice_match:
                event_date = self._date_near_position(fragment, jaundice_match.start())
                events.append(
                    DiliTimelineEvent(
                        event_type="jaundice_or_bilirubin_rise",
                        event_date=event_date,
                        evidence=self._quote(
                            claim="jaundice or bilirubin rise",
                            quote=fragment,
                            source_section="anamnesis",
                            event_date=event_date,
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
            if (
                marker in {"ALT", "AST", "ALP", "BILIRUBIN", "TBIL"}
                and multiple is not None
                and multiple > 1
            ):
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
            if (
                marker in {"BILIRUBIN", "TBIL"}
                and multiple is not None
                and multiple >= 2
            ):
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
    def _first_abnormal_liver_test_date(
        self, labs: list[ClinicalLabEntry]
    ) -> str | None:
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
        primary_pattern: str,
    ) -> tuple[DechallengeStatus, str | None, str | None]:
        stop_dates = sorted(
            parsed_date
            for item in drugs
            for parsed_date in [self.parse_date(item.suspension_date)]
            if parsed_date is not None
        )
        start_dates = [
            parsed_date
            for item in drugs
            for parsed_date in [self.parse_date(item.therapy_start_date)]
            if parsed_date is not None
        ]
        exposure_start = min(start_dates) if start_dates else None
        series = self.dechallenge_series(labs, primary_pattern)
        reference = self._dechallenge_reference(series, exposure_start)
        abnormal_dates = [
            point_date for point_date, multiple, _ in series if multiple > reference
        ]
        last_abnormal_date = abnormal_dates[-1].isoformat() if abnormal_dates else None
        if not stop_dates:
            return "no_follow_up", None, last_abnormal_date

        # The case-level dechallenge follows the first discontinuation made once
        # injury was present; when every stop preceded detection, the latest stop
        # is the closest withdrawal to onset.
        onset = abnormal_dates[0] if abnormal_dates else None
        stops_after_onset = [item for item in stop_dates if onset and item >= onset]
        stop_date = stops_after_onset[0] if stops_after_onset else stop_dates[-1]
        status, recovery_date = self.assess_dechallenge(
            series,
            stop_date=stop_date,
            exposure_start=exposure_start,
            cholestatic=primary_pattern in {"cholestatic", "mixed"},
        )
        return status, recovery_date, last_abnormal_date

    # -------------------------------------------------------------------------
    @classmethod
    def dechallenge_series(
        cls, labs: list[ClinicalLabEntry], primary_pattern: str
    ) -> list[tuple[date, float, str]]:
        """Dated ULN multiples of the RUCAM course marker (ALT, or ALP when cholestatic/mixed)."""
        markers = (
            ("ALP",)
            if primary_pattern in {"cholestatic", "mixed"}
            else ("ALT", "AST")
        )
        for marker in markers:
            series = sorted(
                (point_date, multiple, item.sample_date or "")
                for item in labs
                if item.marker_name.upper() == marker
                for point_date in [cls.parse_date(item.sample_date)]
                for multiple in [cls._multiple(item)]
                if point_date is not None and multiple is not None
            )
            if series:
                return series
        return []

    # -------------------------------------------------------------------------
    @staticmethod
    def _dechallenge_reference(
        series: list[tuple[date, float, str]], exposure_start: date | None
    ) -> float:
        """ULN, or the abnormal pre-treatment baseline multiple when one exists."""
        if exposure_start is None:
            return 1.0
        baseline = [multiple for point_date, multiple, _ in series if point_date < exposure_start]
        return max(1.0, baseline[-1]) if baseline else 1.0

    # -------------------------------------------------------------------------
    @classmethod
    def assess_dechallenge(
        cls,
        series: list[tuple[date, float, str]],
        *,
        stop_date: date,
        exposure_start: date | None,
        cholestatic: bool,
    ) -> tuple[DechallengeStatus, str | None]:
        """Course after withdrawal following the updated RUCAM criteria.

        The decline is measured as the fraction of the peak excess over the
        reference (ULN or abnormal baseline) and must reach 50% within 30 days
        (hepatocellular) or 180 days (cholestatic/mixed) after discontinuation.
        Resolution requires a return to the reference, never a fall below an
        earlier abnormal value.
        """
        reference = cls._dechallenge_reference(series, exposure_start)
        episode = [
            point
            for point in series
            if exposure_start is None or point[0] >= exposure_start
        ]
        if not episode:
            return "insufficient_interval", None
        peak_date, peak, _ = max(episode, key=lambda point: point[1])
        if peak <= reference:
            return "insufficient_interval", None
        follow_up = [
            point for point in episode if point[0] >= stop_date and point[0] > peak_date
        ]
        if not follow_up:
            return "insufficient_interval", None

        last_date, last_multiple, last_raw = follow_up[-1]
        if last_multiple <= reference:
            recovery = last_raw
            for _point_date, multiple, raw in reversed(follow_up):
                if multiple > reference:
                    break
                recovery = raw
            return "resolved_to_baseline", recovery
        if last_multiple > follow_up[0][1] * 1.2:
            return "worsening_after_stop", None

        window_days = 180 if cholestatic else 30
        excess = peak - reference
        if any(
            (peak - multiple) / excess >= 0.5
            and (point_date - stop_date).days <= window_days
            for point_date, multiple, _ in follow_up
        ):
            return "improving_after_stop", None
        if (last_date - stop_date).days > window_days:
            return "chronic_or_persistent", None
        return "stable_abnormality", None

    # -------------------------------------------------------------------------
    @staticmethod
    def _extract_date_from_text(text: str) -> str | None:
        match = DATE_RE.search(text or "")
        return match.group(1).replace("/", "-").replace(".", "-") if match else None

    # -------------------------------------------------------------------------
    @staticmethod
    def _date_near_position(text: str, position: int) -> str | None:
        matches = list(DATE_RE.finditer(text))
        if not matches:
            return None
        match = min(matches, key=lambda item: abs(item.start() - position))
        return match.group(1).replace("/", "-").replace(".", "-")

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
    def _first_event_date(
        events: list[DiliTimelineEvent], event_type: str
    ) -> str | None:
        dated = [
            event
            for event in events
            if event.event_type == event_type
            and DiliTimelineEngine.parse_date(event.event_date)
        ]
        if not dated:
            return None
        dated.sort(
            key=lambda event: (
                DiliTimelineEngine.parse_date(event.event_date) or date.max
            )
        )
        return dated[0].event_date
