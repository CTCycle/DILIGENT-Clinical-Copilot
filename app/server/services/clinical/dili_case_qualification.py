from __future__ import annotations

from datetime import date

from domain.clinical.dili import (
    ClinicalEvidenceQuote,
    DiliCaseQualification,
)
from domain.clinical.entities import ClinicalLabEntry, DrugEntry, PatientLabTimeline
from services.clinical.dili_timeline import DiliTimelineEngine

AMINOTRANSFERASE_MARKERS = {"ALT", "AST"}
BILIRUBIN_MARKERS = {"BILIRUBIN", "TBIL"}


###############################################################################
class DiliCaseQualificationEngine:
    """Qualify the liver-injury episode before patient-drug causality synthesis."""

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
        pending: list[str] = []
        evidence: list[ClinicalEvidenceQuote] = []

        aminotransferase_qualified, aminotransferase_pending, amino_evidence = (
            self._repeated_threshold(
                dated,
                markers=AMINOTRANSFERASE_MARKERS,
                threshold=5.0,
                baseline_entries=baseline_entries,
                criterion_label="ALT/AST >=5x reference on measurements at least 24 hours apart",
            )
        )
        if aminotransferase_qualified:
            qualifying.append(
                "ALT/AST >=5x reference on measurements at least 24 hours apart"
            )
        elif aminotransferase_pending:
            pending.append(
                "A single ALT/AST measurement reaches the 5x reference threshold but repeat confirmation is unavailable"
            )
        evidence.extend(amino_evidence)

        alp_qualified, alp_pending, alp_evidence = self._repeated_threshold(
            dated,
            markers={"ALP"},
            threshold=2.0,
            baseline_entries=baseline_entries,
            criterion_label="ALP >=2x reference on measurements at least 24 hours apart",
        )
        if alp_qualified:
            qualifying.append("ALP >=2x reference on measurements at least 24 hours apart")
        elif alp_pending:
            pending.append(
                "A single ALP measurement reaches the 2x reference threshold but repeat confirmation is unavailable"
            )
        evidence.extend(alp_evidence)

        bilirubin_criterion, bilirubin_evidence = self._bilirubin_criterion(
            dated,
            baseline_entries=baseline_entries,
        )
        if bilirubin_criterion:
            qualifying.append(bilirubin_criterion)
        evidence.extend(bilirubin_evidence)

        inr_criterion, inr_evidence = self._inr_criterion(
            dated,
            baseline_entries=baseline_entries,
        )
        if inr_criterion:
            qualifying.append(inr_criterion)
        evidence.extend(inr_evidence)

        assessable = any(
            entry.marker_name.upper()
            in AMINOTRANSFERASE_MARKERS | {"ALP", "INR"} | BILIRUBIN_MARKERS
            for entry in dated
        )
        if qualifying:
            status = "meets_typical_detection_criteria"
        elif not assessable or pending:
            status = "insufficient_data"
        else:
            status = "below_typical_detection_criteria"

        rationale = [
            "Case qualification is separate from drug causality and does not diagnose DILI.",
            "For abnormal pretreatment results, the pretreatment value is used as the reference for that marker when available.",
        ]
        if baseline_date is None:
            rationale.append(
                "No dated pretreatment laboratory panel before the earliest documented exposure start was available."
            )
        if pending:
            rationale.append(
                "Single enzyme-threshold measurements remain provisional until repeat confirmation is documented."
            )

        return DiliCaseQualification(
            status=status,
            qualifying_criteria=qualifying,
            pending_confirmation=pending,
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
    def _repeated_threshold(
        self,
        dated_labs: list[ClinicalLabEntry],
        *,
        markers: set[str],
        threshold: float,
        baseline_entries: list[ClinicalLabEntry],
        criterion_label: str,
    ) -> tuple[bool, bool, list[ClinicalEvidenceQuote]]:
        threshold_hits: list[tuple[date, ClinicalLabEntry]] = []
        for entry in dated_labs:
            if entry.marker_name.upper() not in markers:
                continue
            reference = self._reference_value(entry, baseline_entries)
            if reference is None or reference <= 0 or entry.value is None:
                continue
            if float(entry.value) / reference >= threshold:
                parsed = DiliTimelineEngine.parse_date(entry.sample_date)
                if parsed is not None:
                    threshold_hits.append((parsed, entry))

        repeated = any(
            (later_date - earlier_date).days >= 1
            for index, (earlier_date, _) in enumerate(threshold_hits)
            for later_date, _ in threshold_hits[index + 1 :]
        )
        evidence = [self._quote(criterion_label, entry) for _, entry in threshold_hits]
        return repeated, bool(threshold_hits) and not repeated, evidence

    # -------------------------------------------------------------------------
    def _bilirubin_criterion(
        self,
        dated_labs: list[ClinicalLabEntry],
        *,
        baseline_entries: list[ClinicalLabEntry],
    ) -> tuple[str | None, list[ClinicalEvidenceQuote]]:
        evidence: list[ClinicalEvidenceQuote] = []
        for entry in dated_labs:
            if entry.marker_name.upper() not in BILIRUBIN_MARKERS:
                continue
            unit = str(entry.unit or "").strip().casefold().replace(" ", "")
            if "mg/dl" not in unit or entry.value is None or float(entry.value) <= 2.5:
                continue
            sample_date = DiliTimelineEngine.parse_date(entry.sample_date)
            if sample_date is None:
                continue
            if self._has_elevated_enzyme_near_date(
                dated_labs,
                sample_date,
                baseline_entries=baseline_entries,
            ):
                evidence.append(
                    self._quote(
                        "Total bilirubin >2.5 mg/dL with elevated liver enzymes",
                        entry,
                    )
                )
                return (
                    "Total bilirubin >2.5 mg/dL with elevated liver enzymes",
                    evidence,
                )
        return None, evidence

    # -------------------------------------------------------------------------
    def _inr_criterion(
        self,
        dated_labs: list[ClinicalLabEntry],
        *,
        baseline_entries: list[ClinicalLabEntry],
    ) -> tuple[str | None, list[ClinicalEvidenceQuote]]:
        evidence: list[ClinicalEvidenceQuote] = []
        for entry in dated_labs:
            if entry.marker_name.upper() != "INR" or entry.value is None:
                continue
            if float(entry.value) <= 1.5:
                continue
            sample_date = DiliTimelineEngine.parse_date(entry.sample_date)
            if sample_date is None:
                continue
            if self._has_elevated_enzyme_near_date(
                dated_labs,
                sample_date,
                baseline_entries=baseline_entries,
            ):
                evidence.append(
                    self._quote("INR >1.5 with elevated liver enzymes", entry)
                )
                return "INR >1.5 with elevated liver enzymes", evidence
        return None, evidence

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
