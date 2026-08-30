from __future__ import annotations

from typing import Literal

from domain.clinical.dili import ClinicalEvidenceQuote, DiliInjuryPattern
from domain.clinical.entities import ClinicalLabEntry, PatientLabTimeline


###############################################################################
class DiliPatternEngine:
    DEFAULT_ULN = {"ALT": 40.0, "ALP": 120.0}

    # -------------------------------------------------------------------------
    @staticmethod
    def _value(entry: ClinicalLabEntry) -> float | None:
        return float(entry.value) if entry.value is not None else None

    # -------------------------------------------------------------------------
    @classmethod
    def _best_entry(
        cls,
        entries: list[ClinicalLabEntry],
        marker_names: set[str],
    ) -> ClinicalLabEntry | None:
        selected: ClinicalLabEntry | None = None
        selected_multiple: float | None = None
        for entry in entries:
            if entry.marker_name.upper() not in marker_names:
                continue
            value = cls._value(entry)
            uln = cls._uln(entry)
            if value is None or uln is None or uln <= 0:
                current_multiple = None
            else:
                current_multiple = value / uln
            if selected is None:
                selected = entry
                selected_multiple = current_multiple
                continue
            if selected_multiple is None and current_multiple is not None:
                selected = entry
                selected_multiple = current_multiple
                continue
            if (
                selected_multiple is not None
                and current_multiple is not None
                and current_multiple > selected_multiple
            ):
                selected = entry
                selected_multiple = current_multiple
        return selected

    # -------------------------------------------------------------------------
    @classmethod
    def _uln(cls, entry: ClinicalLabEntry) -> float | None:
        if entry.upper_limit_normal and entry.upper_limit_normal > 0:
            return float(entry.upper_limit_normal)
        return cls.DEFAULT_ULN.get(entry.marker_name.upper())

    # -------------------------------------------------------------------------
    @staticmethod
    def classify(
        r_ratio: float | None,
    ) -> Literal["hepatocellular", "cholestatic", "mixed", "indeterminate"]:
        if r_ratio is None:
            return "indeterminate"
        if r_ratio >= 5:
            return "hepatocellular"
        if r_ratio <= 2:
            return "cholestatic"
        return "mixed"

    # -------------------------------------------------------------------------
    def assess(self, timeline: PatientLabTimeline) -> list[DiliInjuryPattern]:
        buckets: dict[str, list[ClinicalLabEntry]] = {}
        for entry in timeline.entries:
            buckets.setdefault(entry.sample_date or "undated", []).append(entry)
        calculated: list[DiliInjuryPattern] = []
        for sample_date, entries in buckets.items():
            alt = self._best_entry(entries, {"ALT"})
            alp = self._best_entry(entries, {"ALP"})
            if alt is None or alp is None:
                continue
            alt_value, alp_value = self._value(alt), self._value(alp)
            alt_uln, alp_uln = self._uln(alt), self._uln(alp)
            ratio = None
            if (
                alt_value is not None
                and alp_value is not None
                and alt_uln is not None
                and alp_uln is not None
                and alp_value != 0
            ):
                ratio = (alt_value / alt_uln) / (alp_value / alp_uln)
            calculated.append(
                DiliInjuryPattern(
                    assessment_point="first_qualifying",
                    alt=alt_value,
                    alt_uln=alt_uln,
                    alp=alp_value,
                    alp_uln=alp_uln,
                    r_ratio=ratio,
                    pattern=self.classify(ratio),
                    pattern_source="calculated" if ratio is not None else "unavailable",
                    sample_date=None if sample_date == "undated" else sample_date,
                    evidence=[
                        ClinicalEvidenceQuote(
                            claim="R-ratio input",
                            quote=alt.evidence or alp.evidence,
                            source_section="laboratory_analysis",
                            event_date=None
                            if sample_date == "undated"
                            else sample_date,
                            source_kind="calculated",
                        )
                    ],
                )
            )
        if not calculated:
            return [
                DiliInjuryPattern(
                    assessment_point="first_qualifying",
                    pattern="indeterminate",
                    pattern_source="unavailable",
                )
            ]
        calculated.sort(key=lambda item: item.sample_date or "9999")
        first = calculated[0]
        peak = max(calculated, key=lambda item: item.alt or -1)
        first.assessment_point = "first_qualifying"
        peak_payload = peak.model_copy(deep=True)
        peak_payload.assessment_point = "peak"
        # Consumers treat the first pattern as the clinical injury phenotype.
        # Put the peak ALT assessment first so a normal baseline cannot become
        # the primary DILI classification; retain the chronological first pair
        # as the secondary audit point.
        return [peak_payload, first]
