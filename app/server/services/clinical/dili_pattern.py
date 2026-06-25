from __future__ import annotations

from domain.clinical.dili import ClinicalEvidenceQuote, DiliInjuryPattern
from domain.clinical.entities import ClinicalLabEntry, PatientLabTimeline


class DiliPatternEngine:
    DEFAULT_ULN = {"ALT": 40.0, "ALP": 120.0}

    @staticmethod
    def _value(entry: ClinicalLabEntry) -> float | None:
        return float(entry.value) if entry.value is not None else None

    @classmethod
    def _uln(cls, entry: ClinicalLabEntry) -> float | None:
        if entry.upper_limit_normal and entry.upper_limit_normal > 0:
            return float(entry.upper_limit_normal)
        return cls.DEFAULT_ULN.get(entry.marker_name.upper())

    @staticmethod
    def classify(r_ratio: float | None) -> str:
        if r_ratio is None:
            return "indeterminate"
        if r_ratio >= 5:
            return "hepatocellular"
        if r_ratio <= 2:
            return "cholestatic"
        return "mixed"

    def assess(self, timeline: PatientLabTimeline) -> list[DiliInjuryPattern]:
        buckets: dict[str, list[ClinicalLabEntry]] = {}
        for entry in timeline.entries:
            buckets.setdefault(entry.sample_date or "undated", []).append(entry)
        calculated: list[DiliInjuryPattern] = []
        for sample_date, entries in buckets.items():
            alt = next((item for item in entries if item.marker_name.upper() == "ALT"), None)
            alp = next((item for item in entries if item.marker_name.upper() == "ALP"), None)
            if alt is None or alp is None:
                continue
            alt_value, alp_value = self._value(alt), self._value(alp)
            alt_uln, alp_uln = self._uln(alt), self._uln(alp)
            ratio = None
            if None not in (alt_value, alp_value, alt_uln, alp_uln) and alp_value:
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
                            event_date=None if sample_date == "undated" else sample_date,
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
        return [first, peak_payload]
