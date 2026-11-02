# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import re
from datetime import date

from domain.clinical.dili import (
    ClinicalEvidenceQuote,
    DiliDifferentialAssessment,
    DiliHysLawAssessment,
    DiliTimeline,
)
from domain.clinical.entities import DrugEntry, PatientLabTimeline
from services.clinical.dili_timeline import DiliTimelineEngine

# ALP is taken from the qualifying aminotransferase panel or the nearest
# sample within this many days (FDA 2009: no initial cholestasis).
ONSET_ALP_WINDOW_DAYS = 7

# A qualifying bilirubin value pairs with an aminotransferase elevation when it
# is dated on the same day or up to this many days after the aminotransferase
# onset. Follows EASL 2019 / FDA 2009 guidance (bilirubin may rise weeks after
# the aminotransferase rise); the exact window still needs clinician review.
BILIRUBIN_LAG_WINDOW_DAYS = 30

AMINOTRANSFERASE_MARKERS = {"ALT", "AST"}
BILIRUBIN_MARKERS = {"BILIRUBIN", "TBIL"}
AMINOTRANSFERASE_MULTIPLE = 3.0
BILIRUBIN_MULTIPLE = 2.0

# Word-bounded, contextual terms: "atrial fibrillation" or "trial of diet"
# must not mark a patient as a clinical-trial case.
CLINICAL_TRIAL_PATTERN = re.compile(
    r"\b(clinical\s+trials?|trial\s+(drug|medication|participants?|participation)"
    r"|(enrolled|participat\w+)\s+in\s+(a|an|the)\s+(\w+\s+){0,2}trial"
    r"|study\s+(arm|drug)|investigational|randomi[sz]ed)\b",
    re.IGNORECASE,
)

# (sample date or None when missing/unparseable, multiple of the reference)
LabPoint = tuple[date | None, float]
EpisodePair = tuple[tuple[date, float], tuple[date, float]]


###############################################################################
class HysLawDetector:

    # -------------------------------------------------------------------------
    def assess(
        self,
        *,
        labs: PatientLabTimeline,
        differential: DiliDifferentialAssessment,
        timeline: DiliTimeline,
        drugs: list[DrugEntry],
        source_text: str,
    ) -> DiliHysLawAssessment:
        baseline_alt = self._first_multiple(labs, AMINOTRANSFERASE_MARKERS)
        baseline_bili = self._first_multiple(labs, BILIRUBIN_MARKERS)
        peak_alt = self._peak_multiple(labs, AMINOTRANSFERASE_MARKERS)
        peak_bili = self._peak_multiple(labs, BILIRUBIN_MARKERS)
        exposure_start = self._exposure_start(drugs)
        amino_points = self._qualifying_points(
            labs, AMINOTRANSFERASE_MARKERS, AMINOTRANSFERASE_MULTIPLE, exposure_start
        )
        bili_points = self._qualifying_points(
            labs, BILIRUBIN_MARKERS, BILIRUBIN_MULTIPLE, exposure_start, strict=True
        )
        pair = self._episode_pair(amino_points, bili_points)
        onset_date = pair[0][0] if pair else self._first_date(amino_points)
        initial_alp = self._onset_alp_multiple(labs, onset_date)

        amino_met = bool(amino_points) if peak_alt is not None else None
        bilirubin_met = bool(bili_points) if peak_bili is not None else None
        same_episode = self._same_episode(pair, amino_points, bili_points)
        initial_cholestasis = initial_alp >= 2 if initial_alp is not None else None
        cholestasis_excluded = (
            (initial_cholestasis is False) if initial_cholestasis is not None else None
        )
        alternatives = differential.all_major_causes_excluded
        compatible_exposures = [
            drug.name
            for drug in drugs
            if drug.name
            and drug.therapy_start_date
            and timeline.first_abnormal_liver_test_date
            and self._plausible_latency(
                drug.therapy_start_date, timeline.first_abnormal_liver_test_date
            )
        ]
        exposure_timing_compatible = bool(compatible_exposures) if drugs else None
        signal_context = (
            "clinical_trial_signal"
            if CLINICAL_TRIAL_PATTERN.search(source_text)
            else "individual_patient_risk_flag"
        )

        # ALP >= 2x ULN at onset marks cholestatic injury, which FDA 2009
        # excludes from Hy's Law; "possible" is kept for unresolved exclusion,
        # exposure timing or missing onset ALP only.
        if None in (amino_met, bilirubin_met, same_episode):
            status = "not_assessable"
        elif not (amino_met and bilirubin_met and same_episode) or initial_cholestasis:
            status = "not_met"
        elif cholestasis_excluded and alternatives and exposure_timing_compatible:
            status = "meets_criteria"
        else:
            status = "possible"

        return DiliHysLawAssessment(
            status=status,
            aminotransferase_threshold_met=amino_met,
            bilirubin_threshold_met=bilirubin_met,
            cholestasis_excluded=cholestasis_excluded,
            alternative_causes_excluded=alternatives,
            exposure_timing_compatible=exposure_timing_compatible,
            same_episode=same_episode,
            baseline_aminotransferase_multiple=baseline_alt,
            baseline_bilirubin_multiple=baseline_bili,
            initial_cholestasis_present=initial_cholestasis,
            compatible_exposures=compatible_exposures,
            signal_context=signal_context,
            evidence=self._build_evidence(timeline),
            rationale=[
                f"Peak ALT/AST multiple: {peak_alt if peak_alt is not None else 'unavailable'}",
                f"Peak bilirubin multiple: {peak_bili if peak_bili is not None else 'unavailable'}",
                f"ALP multiple at the qualifying ALT/AST rise: {initial_alp if initial_alp is not None else 'unavailable'}",
                self._pair_rationale(pair, same_episode, exposure_start),
                "Hy's Law is a risk signal, not a standalone DILI diagnosis.",
            ],
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _first_multiple(labs: PatientLabTimeline, markers: set[str]) -> float | None:
        candidates = []
        for lab in labs.entries:
            if (
                lab.marker_name.upper() not in markers
                or lab.value is None
                or not lab.upper_limit_normal
            ):
                continue
            parsed = DiliTimelineEngine.parse_date(lab.sample_date)
            if parsed is None:
                continue
            candidates.append(
                (parsed, float(lab.value) / float(lab.upper_limit_normal))
            )
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    # -------------------------------------------------------------------------
    @staticmethod
    def _peak_multiple(labs: PatientLabTimeline, markers: set[str]) -> float | None:
        candidates = [
            float(lab.value) / float(lab.upper_limit_normal)
            for lab in labs.entries
            if lab.marker_name.upper() in markers
            and lab.value is not None
            and lab.upper_limit_normal
            and float(lab.upper_limit_normal) > 0
        ]
        return max(candidates) if candidates else None

    # -------------------------------------------------------------------------
    @staticmethod
    def _exposure_start(drugs: list[DrugEntry]) -> date | None:
        starts = [
            parsed
            for drug in drugs
            for parsed in [DiliTimelineEngine.parse_date(drug.therapy_start_date)]
            if parsed is not None
        ]
        return min(starts) if starts else None

    # -------------------------------------------------------------------------
    @staticmethod
    def _qualifying_points(
        labs: PatientLabTimeline,
        markers: set[str],
        multiple: float,
        exposure_start: date | None,
        strict: bool = False,
    ) -> list[LabPoint]:
        """Post-exposure (or undated) values above `multiple` x reference.

        The reference is the ULN, or the latest pre-exposure multiple when that
        baseline is abnormal (FDA 2009 DILI guidance, EASL 2019). It is resolved
        per marker; without a dated pre-exposure value the ULN applies.
        """
        series: dict[str, list[LabPoint]] = {}
        for lab in labs.entries:
            if (
                lab.marker_name.upper() not in markers
                or lab.value is None
                or not lab.upper_limit_normal
                or float(lab.upper_limit_normal) <= 0
            ):
                continue
            series.setdefault(lab.marker_name.upper(), []).append(
                (
                    DiliTimelineEngine.parse_date(lab.sample_date),
                    float(lab.value) / float(lab.upper_limit_normal),
                )
            )
        qualifying: list[LabPoint] = []
        for points in series.values():
            pre_exposure = [
                (point_date, value)
                for point_date, value in points
                if point_date is not None
                and exposure_start is not None
                and point_date < exposure_start
            ]
            reference = max(1.0, max(pre_exposure)[1]) if pre_exposure else 1.0
            threshold = multiple * reference
            qualifying.extend(
                (point_date, value)
                for point_date, value in points
                if (point_date, value) not in pre_exposure
                and (value > threshold if strict else value >= threshold)
            )
        return qualifying

    # -------------------------------------------------------------------------
    @staticmethod
    def _first_date(points: list[LabPoint]) -> date | None:
        return min((item[0] for item in points if item[0] is not None), default=None)

    # -------------------------------------------------------------------------
    @staticmethod
    def _episode_pair(
        amino_points: list[LabPoint], bili_points: list[LabPoint]
    ) -> EpisodePair | None:
        """Earliest bilirubin dated 0..BILIRUBIN_LAG_WINDOW_DAYS after an ALT/AST rise."""
        candidates = [
            ((amino_date, amino_value), (bili_date, bili_value))
            for amino_date, amino_value in amino_points
            if amino_date is not None
            for bili_date, bili_value in bili_points
            if bili_date is not None
            and 0 <= (bili_date - amino_date).days <= BILIRUBIN_LAG_WINDOW_DAYS
        ]
        return min(candidates, key=lambda item: (item[0][0], item[1][0])) if candidates else None

    # -------------------------------------------------------------------------
    @staticmethod
    def _same_episode(
        pair: EpisodePair | None,
        amino_points: list[LabPoint],
        bili_points: list[LabPoint],
    ) -> bool | None:
        """True when paired; None when a missing date leaves the timing unresolved."""
        if pair is not None:
            return True
        if amino_points and bili_points:
            undated = any(item[0] is None for item in (*amino_points, *bili_points))
            return None if undated else False
        return False

    # -------------------------------------------------------------------------
    @staticmethod
    def _pair_rationale(
        pair: EpisodePair | None, same_episode: bool | None, exposure_start: date | None
    ) -> str:
        reference = (
            "ULN, or the abnormal pre-exposure baseline when one is dated"
            if exposure_start is not None
            else "ULN (no exposure start date to define a baseline)"
        )
        if pair is not None:
            (amino_date, amino_value), (bili_date, bili_value) = pair
            return (
                f"Evaluated pair: ALT/AST {amino_value:.2f}x on {amino_date.isoformat()}, "
                f"bilirubin {bili_value:.2f}x on {bili_date.isoformat()} "
                f"({(bili_date - amino_date).days} days later; accepted window 0-"
                f"{BILIRUBIN_LAG_WINDOW_DAYS} days, pending clinician review). "
                f"Reference: {reference}."
            )
        if same_episode is None:
            return (
                "Episode timing is indeterminate: a qualifying ALT/AST or bilirubin "
                f"date is missing or unparseable. Reference: {reference}."
            )
        return (
            "No qualifying bilirubin dated 0-"
            f"{BILIRUBIN_LAG_WINDOW_DAYS} days after a qualifying ALT/AST elevation. "
            f"Reference: {reference}."
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _onset_alp_multiple(
        labs: PatientLabTimeline, onset_date: date | None
    ) -> float | None:
        if onset_date is None:
            return None
        candidates = [
            (abs((parsed - onset_date).days), float(lab.value) / float(lab.upper_limit_normal))
            for lab in labs.entries
            if lab.marker_name.upper() == "ALP"
            and lab.value is not None
            and lab.upper_limit_normal
            for parsed in [DiliTimelineEngine.parse_date(lab.sample_date)]
            if parsed is not None
            and abs((parsed - onset_date).days) <= ONSET_ALP_WINDOW_DAYS
        ]
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    # -------------------------------------------------------------------------
    @staticmethod
    def _plausible_latency(start_date: str, injury_date: str) -> bool:
        parsed_start = DiliTimelineEngine.parse_date(start_date)
        parsed_injury = DiliTimelineEngine.parse_date(injury_date)
        if parsed_start is None or parsed_injury is None:
            return False
        delta = (parsed_injury - parsed_start).days
        return 1 <= delta <= 365

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_evidence(timeline: DiliTimeline) -> list[ClinicalEvidenceQuote]:
        relevant = []
        for event in timeline.events:
            if (
                event.event_type
                in {
                    "abnormal_liver_test",
                    "jaundice_or_bilirubin_rise",
                    "drug_start",
                    "drug_stop",
                }
                and event.evidence is not None
            ):
                relevant.append(event.evidence)
        return relevant[:6]
