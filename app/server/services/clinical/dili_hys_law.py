from __future__ import annotations

from domain.clinical.dili import DiliDifferentialAssessment, DiliHysLawAssessment
from domain.clinical.entities import PatientLabTimeline


class HysLawDetector:
    def assess(
        self,
        labs: PatientLabTimeline,
        differential: DiliDifferentialAssessment,
    ) -> DiliHysLawAssessment:
        multiples: dict[str, float] = {}
        for lab in labs.entries:
            marker = lab.marker_name.upper()
            if lab.value is None or not lab.upper_limit_normal:
                continue
            multiples[marker] = max(
                multiples.get(marker, 0.0),
                float(lab.value) / float(lab.upper_limit_normal),
            )
        amino = max(multiples.get("ALT", 0), multiples.get("AST", 0))
        bilirubin = multiples.get("BILIRUBIN", multiples.get("TBIL", 0))
        alp = multiples.get("ALP")
        amino_met = amino >= 3 if amino else None
        bilirubin_met = bilirubin > 2 if bilirubin else None
        cholestasis_excluded = alp < 2 if alp is not None else None
        alternatives = differential.all_major_causes_excluded
        if None in (amino_met, bilirubin_met, cholestasis_excluded):
            status = "possible" if amino_met and bilirubin_met else "not_assessable"
        elif amino_met and bilirubin_met and cholestasis_excluded and alternatives:
            status = "meets_criteria"
        elif amino_met and bilirubin_met:
            status = "possible"
        else:
            status = "not_met"
        return DiliHysLawAssessment(
            status=status,
            aminotransferase_threshold_met=amino_met,
            bilirubin_threshold_met=bilirubin_met,
            cholestasis_excluded=cholestasis_excluded,
            alternative_causes_excluded=alternatives,
            rationale=[
                f"Peak ALT/AST multiple: {amino or 'unavailable'}",
                f"Peak bilirubin multiple: {bilirubin or 'unavailable'}",
                "Hy's Law is a risk signal, not a standalone DILI diagnosis.",
            ],
        )
