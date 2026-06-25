from __future__ import annotations

from domain.clinical.dili import (
    ClinicalEvidenceQuote,
    DiliDifferentialAssessment,
    DiliHysLawAssessment,
    DiliTimeline,
)
from domain.clinical.entities import DrugEntry, PatientLabTimeline
from services.clinical.dili_timeline import DiliTimelineEngine


class HysLawDetector:
    def assess(
        self,
        *,
        labs: PatientLabTimeline,
        differential: DiliDifferentialAssessment,
        timeline: DiliTimeline,
        drugs: list[DrugEntry],
        source_text: str,
    ) -> DiliHysLawAssessment:
        baseline_alt = self._first_multiple(labs, {"ALT", "AST"})
        baseline_bili = self._first_multiple(labs, {"BILIRUBIN", "TBIL"})
        peak_alt = self._peak_multiple(labs, {"ALT", "AST"})
        peak_bili = self._peak_multiple(labs, {"BILIRUBIN", "TBIL"})
        initial_alp = self._first_multiple(labs, {"ALP"})

        amino_met = peak_alt >= 3 if peak_alt is not None else None
        bilirubin_met = peak_bili > 2 if peak_bili is not None else None
        same_episode = self._same_episode(labs, timeline)
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
            and self._plausible_latency(drug.therapy_start_date, timeline.first_abnormal_liver_test_date)
        ]
        exposure_timing_compatible = bool(compatible_exposures) if drugs else None
        signal_context = (
            "clinical_trial_signal"
            if any(token in source_text.lower() for token in ("trial", "study arm", "randomized"))
            else "individual_patient_risk_flag"
        )

        if None in (amino_met, bilirubin_met, same_episode):
            status = "not_assessable"
        elif amino_met and bilirubin_met and same_episode and cholestasis_excluded and alternatives and exposure_timing_compatible:
            status = "meets_criteria"
        elif amino_met and bilirubin_met and same_episode:
            status = "possible"
        else:
            status = "not_met"

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
                f"Initial ALP multiple: {initial_alp if initial_alp is not None else 'unavailable'}",
                "Hy's Law is a risk signal, not a standalone DILI diagnosis.",
            ],
        )

    @staticmethod
    def _first_multiple(labs: PatientLabTimeline, markers: set[str]) -> float | None:
        candidates = []
        for lab in labs.entries:
            if lab.marker_name.upper() not in markers or lab.value is None or not lab.upper_limit_normal:
                continue
            parsed = DiliTimelineEngine.parse_date(lab.sample_date)
            if parsed is None:
                continue
            candidates.append((parsed, float(lab.value) / float(lab.upper_limit_normal)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

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

    @staticmethod
    def _same_episode(labs: PatientLabTimeline, timeline: DiliTimeline) -> bool | None:
        first_alt_date = None
        first_bili_date = timeline.jaundice_or_bilirubin_rise_date
        for lab in labs.entries:
            marker = lab.marker_name.upper()
            if marker not in {"ALT", "AST"} or lab.value is None or not lab.upper_limit_normal:
                continue
            if float(lab.value) / float(lab.upper_limit_normal) < 3:
                continue
            first_alt_date = lab.sample_date
            break
        parsed_alt = DiliTimelineEngine.parse_date(first_alt_date)
        parsed_bili = DiliTimelineEngine.parse_date(first_bili_date)
        if parsed_alt is None or parsed_bili is None:
            return None
        return abs((parsed_bili - parsed_alt).days) <= 14

    @staticmethod
    def _plausible_latency(start_date: str, injury_date: str) -> bool:
        parsed_start = DiliTimelineEngine.parse_date(start_date)
        parsed_injury = DiliTimelineEngine.parse_date(injury_date)
        if parsed_start is None or parsed_injury is None:
            return False
        delta = (parsed_injury - parsed_start).days
        return 1 <= delta <= 365

    @staticmethod
    def _build_evidence(timeline: DiliTimeline) -> list[ClinicalEvidenceQuote]:
        relevant = []
        for event in timeline.events:
            if event.event_type in {
                "abnormal_liver_test",
                "jaundice_or_bilirubin_rise",
                "drug_start",
                "drug_stop",
            } and event.evidence is not None:
                relevant.append(event.evidence)
        return relevant[:6]
