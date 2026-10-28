# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import date

from common.constants import (
    DILI_ALKALINE_PHOSPHATASE_QUALIFYING_MULTIPLE,
    DILI_AMINOTRANSFERASE_QUALIFYING_MULTIPLE,
    DILI_COMBINED_AMINOTRANSFERASE_MULTIPLE,
    DILI_COMBINED_BILIRUBIN_MULTIPLE,
)
from domain.clinical.dili import (
    ClinicalEvidenceQuote,
    DiliCaseQualification,
)
from domain.clinical.entities import ClinicalLabEntry, DrugEntry, PatientLabTimeline
from services.clinical.dili_severity import DiliSeverityGrader
from services.clinical.dili_timeline import DiliTimelineEngine

AMINOTRANSFERASE_MARKERS = {"ALT", "AST"}
BILIRUBIN_MARKERS = {"BILIRUBIN", "TBIL"}

###############################################################################
class DiliCaseQualificationEngine:
    """Qualify the liver-injury episode before patient-drug causality synthesis.

    Biochemical criteria follow EASL 2019: ALT >=5x ULN, ALP >=2x ULN, or
    ALT >=3x ULN with total bilirubin >2x ULN.
    """

    # -------------------------------------------------------------------------
    def assess(
        self,
        *,
        labs: PatientLabTimeline,
        drugs: list[DrugEntry],
    ) -> DiliCaseQualification:
        dated = [
            entry
            for entry in labs.entries
            if DiliTimelineEngine.parse_date(entry.sample_date) is not None
            and entry.value is not None
        ]
        dated.sort(
            key=lambda item: DiliTimelineEngine.parse_date(item.sample_date) or date.max
        )
        baseline_date, baseline_entries = self._baseline(dated, drugs)
        baseline_multiples = {
            marker: self._multiple(self._latest_marker(baseline_entries, {marker}))
            for marker in ("ALT", "AST", "ALP", "BILIRUBIN")
        }
        baseline_abnormal_values = [
            value for value in baseline_multiples.values() if value is not None
        ]
        baseline_abnormal = (
            any(value > 1.0 for value in baseline_abnormal_values)
            if baseline_abnormal_values
            else None
        )

        qualifying: list[str] = []
        confirmation_notes: list[str] = []
        evidence: list[ClinicalEvidenceQuote] = []
        enzyme_thresholds = (
            (
                AMINOTRANSFERASE_MARKERS,
                DILI_AMINOTRANSFERASE_QUALIFYING_MULTIPLE,
                "ALT/AST >=5x reference",
            ),
            (
                {"ALP"},
                DILI_ALKALINE_PHOSPHATASE_QUALIFYING_MULTIPLE,
                "ALP >=2x reference",
            ),
        )
        for markers, threshold, label in enzyme_thresholds:
            hits = self._threshold_hits(
                dated,
                markers=markers,
                threshold=threshold,
                baseline_entries=baseline_entries,
            )
            if not hits:
                continue
            qualifying.append(label)
            if len({hit_date for hit_date, _ in hits}) < 2:
                confirmation_notes.append(
                    f"{label} on a single panel; repeat testing is recommended to confirm the episode"
                )
            evidence.extend(self._quote(label, entry) for _, entry in hits)

        combined_criterion, combined_evidence = self._combined_bilirubin_criterion(
            dated,
            baseline_entries=baseline_entries,
        )
        if combined_criterion:
            qualifying.append(combined_criterion)
        evidence.extend(combined_evidence)

        inr_note, inr_evidence = self._inr_note(
            dated,
            baseline_entries=baseline_entries,
        )
        evidence.extend(inr_evidence)

        enzyme_entries = [
            entry
            for entry in dated
            if entry.marker_name.upper() in AMINOTRANSFERASE_MARKERS | {"ALP"}
        ]
        unreferenced = sorted(
            {
                entry.marker_name.upper()
                for entry in enzyme_entries
                if self._reference_value(entry, baseline_entries) is None
            }
        )
        if qualifying:
            status = "meets_typical_detection_criteria"
        elif not enzyme_entries or unreferenced:
            status = "insufficient_data"
        else:
            status = "below_typical_detection_criteria"

        rationale = [
            "Case qualification is separate from drug causality and does not diagnose DILI.",
            "EASL 2019 biochemical criteria: ALT >=5x ULN, ALP >=2x ULN, or ALT >=3x ULN with total bilirubin >2x ULN.",
            "For abnormal pretreatment results, the pretreatment value is used as the reference for that marker when available.",
        ]
        if baseline_date is None:
            rationale.append(
                "No dated pretreatment laboratory panel before the earliest documented exposure start was available."
            )
        if not qualifying and unreferenced:
            rationale.append(
                "No upper limit of normal or pretreatment reference is available for "
                f"{', '.join(unreferenced)}, so the thresholds cannot be assessed."
            )
        if inr_note:
            rationale.append(inr_note)

        return DiliCaseQualification(
            status=status,
            qualifying_criteria=qualifying,
            pending_confirmation=confirmation_notes,
            baseline_date=baseline_date.isoformat() if baseline_date else None,
            baseline_abnormal=baseline_abnormal,
            baseline_multiples=baseline_multiples,
            rationale=rationale,
            evidence=self._dedupe_evidence(evidence)[:8],
        )

    # -------------------------------------------------------------------------
    def _baseline(
        self,
        dated_labs: list[ClinicalLabEntry],
        drugs: list[DrugEntry],
    ) -> tuple[date | None, list[ClinicalLabEntry]]:
        starts = [
            parsed
            for drug in drugs
            for parsed in [DiliTimelineEngine.parse_date(drug.therapy_start_date)]
            if parsed is not None
        ]
        if not starts:
            return None, []
        earliest_start = min(starts)
        candidates = [
            entry
            for entry in dated_labs
            if (DiliTimelineEngine.parse_date(entry.sample_date) or date.max)
            < earliest_start
        ]
        if not candidates:
            return None, []
        baseline_date = max(
            DiliTimelineEngine.parse_date(entry.sample_date) or date.min
            for entry in candidates
        )
        return baseline_date, [
            entry
            for entry in candidates
            if DiliTimelineEngine.parse_date(entry.sample_date) == baseline_date
        ]

    # -------------------------------------------------------------------------
    def _threshold_hits(
        self,
        dated_labs: list[ClinicalLabEntry],
        *,
        markers: set[str],
        threshold: float,
        baseline_entries: list[ClinicalLabEntry],
    ) -> list[tuple[date, ClinicalLabEntry]]:
        hits: list[tuple[date, ClinicalLabEntry]] = []
        for entry in dated_labs:
            if entry.marker_name.upper() not in markers or entry.value is None:
                continue
            reference = self._reference_value(entry, baseline_entries)
            if reference is None or reference <= 0:
                continue
            parsed = DiliTimelineEngine.parse_date(entry.sample_date)
            if parsed is not None and float(entry.value) / reference >= threshold:
                hits.append((parsed, entry))
        return hits

    # -------------------------------------------------------------------------
    def _combined_bilirubin_criterion(
        self,
        dated_labs: list[ClinicalLabEntry],
        *,
        baseline_entries: list[ClinicalLabEntry],
    ) -> tuple[str | None, list[ClinicalEvidenceQuote]]:
        label = "ALT/AST >=3x reference with total bilirubin >2x reference"
        aminotransferase_dates = [
            hit_date
            for hit_date, _ in self._threshold_hits(
                dated_labs,
                markers=AMINOTRANSFERASE_MARKERS,
                threshold=DILI_COMBINED_AMINOTRANSFERASE_MULTIPLE,
                baseline_entries=baseline_entries,
            )
        ]
        for entry in dated_labs:
            if entry.marker_name.upper() not in BILIRUBIN_MARKERS:
                continue
            multiple = self._bilirubin_multiple(entry, baseline_entries)
            sample_date = DiliTimelineEngine.parse_date(entry.sample_date)
            if (
                multiple is None
                or multiple <= DILI_COMBINED_BILIRUBIN_MULTIPLE
                or sample_date is None
            ):
                continue
            if any(abs((sample_date - day).days) <= 1 for day in aminotransferase_dates):
                return label, [self._quote(label, entry)]
        return None, []

    # -------------------------------------------------------------------------
    def _bilirubin_multiple(
        self,
        entry: ClinicalLabEntry,
        baseline_entries: list[ClinicalLabEntry],
    ) -> float | None:
        """Bilirubin multiple with value and reference both in mg/dL.

        µmol/L is divided by 17.1, so a value and ULN reported in different
        units still compare correctly.
        """
        value = DiliSeverityGrader.bilirubin_mg_dl(entry)
        baseline = self._latest_marker(baseline_entries, BILIRUBIN_MARKERS)
        baseline_multiple = self._multiple(baseline)
        if baseline_multiple is not None and baseline_multiple > 1.0:
            reference = DiliSeverityGrader.bilirubin_mg_dl(baseline)
        else:
            reference = DiliSeverityGrader.bilirubin_mg_dl(
                ClinicalLabEntry(
                    marker_name=entry.marker_name,
                    value=entry.upper_limit_normal,
                    upper_limit_normal=entry.upper_limit_normal,
                )
            )
        if value is None or reference is None or reference <= 0:
            return None
        return value / reference

    # -------------------------------------------------------------------------
    def _inr_note(
        self,
        dated_labs: list[ClinicalLabEntry],
        *,
        baseline_entries: list[ClinicalLabEntry],
    ) -> tuple[str | None, list[ClinicalEvidenceQuote]]:
        """INR >1.5 is severity information, not an EASL case-definition criterion."""
        label = (
            "INR >1.5 with elevated liver enzymes "
            "(supporting severity information, not a qualifying criterion)"
        )
        for entry in dated_labs:
            if entry.marker_name.upper() != "INR" or entry.value is None:
                continue
            sample_date = DiliTimelineEngine.parse_date(entry.sample_date)
            if float(entry.value) <= 1.5 or sample_date is None:
                continue
            if self._has_elevated_enzyme_near_date(
                dated_labs,
                sample_date,
                baseline_entries=baseline_entries,
            ):
                return label, [self._quote(label, entry)]
        return None, []

    # -------------------------------------------------------------------------
    def _has_elevated_enzyme_near_date(
        self,
        dated_labs: list[ClinicalLabEntry],
        target_date: date,
        *,
        baseline_entries: list[ClinicalLabEntry],
    ) -> bool:
        for entry in dated_labs:
            if entry.marker_name.upper() not in AMINOTRANSFERASE_MARKERS | {"ALP"}:
                continue
            parsed = DiliTimelineEngine.parse_date(entry.sample_date)
            if parsed is None or abs((parsed - target_date).days) > 1:
                continue
            reference = self._reference_value(entry, baseline_entries)
            if reference and entry.value is not None and float(entry.value) > reference:
                return True
        return False

    # -------------------------------------------------------------------------
    def _reference_value(
        self,
        entry: ClinicalLabEntry,
        baseline_entries: list[ClinicalLabEntry],
    ) -> float | None:
        baseline = self._latest_marker(baseline_entries, {entry.marker_name.upper()})
        if baseline is not None and baseline.value is not None:
            baseline_multiple = self._multiple(baseline)
            if baseline_multiple is not None and baseline_multiple > 1.0:
                return float(baseline.value)
        if entry.upper_limit_normal and float(entry.upper_limit_normal) > 0:
            return float(entry.upper_limit_normal)
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _latest_marker(
        entries: list[ClinicalLabEntry], markers: set[str]
    ) -> ClinicalLabEntry | None:
        matches = [entry for entry in entries if entry.marker_name.upper() in markers]
        return matches[-1] if matches else None

    # -------------------------------------------------------------------------
    @staticmethod
    def _multiple(entry: ClinicalLabEntry | None) -> float | None:
        if (
            entry is None
            or entry.value is None
            or entry.upper_limit_normal is None
            or float(entry.upper_limit_normal) <= 0
        ):
            return None
        return float(entry.value) / float(entry.upper_limit_normal)

    # -------------------------------------------------------------------------
    @staticmethod
    def _quote(claim: str, entry: ClinicalLabEntry) -> ClinicalEvidenceQuote:
        return ClinicalEvidenceQuote(
            claim=claim,
            quote=entry.evidence or f"{entry.marker_name} {entry.value} {entry.unit or ''}".strip(),
            source_section=entry.source,
            event_date=entry.sample_date,
            source_kind="patient_record",
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _dedupe_evidence(
        evidence: list[ClinicalEvidenceQuote],
    ) -> list[ClinicalEvidenceQuote]:
        unique: list[ClinicalEvidenceQuote] = []
        seen: set[tuple[str, str | None, str | None]] = set()
        for item in evidence:
            key = (item.claim, item.quote, item.event_date)
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique
