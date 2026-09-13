from __future__ import annotations

from typing import Literal

from domain.clinical.dili import ClinicalEvidenceQuote, DiliInjuryPattern
from domain.clinical.entities import ClinicalLabEntry, PatientLabTimeline
from services.clinical.pattern_analyzer import (
    HepatotoxicityPatternAnalyzer,
    HepatotoxicityPatternCalculator,
)

###############################################################################
class DiliPatternEngine:

    # -------------------------------------------------------------------------
    @staticmethod
    def classify(
        r_ratio: float | None,
    ) -> Literal["hepatocellular", "cholestatic", "mixed", "indeterminate"]:
        return HepatotoxicityPatternCalculator.classify_r_score(r_ratio)

    # -------------------------------------------------------------------------
    def assess(self, timeline: PatientLabTimeline) -> list[DiliInjuryPattern]:
        analyzer = HepatotoxicityPatternAnalyzer()
        buckets = analyzer.group_entries_by_date(timeline.entries)

        calculated: list[DiliInjuryPattern] = []

        def add_bucket(
            sample_date: str | None, entries: list[ClinicalLabEntry]
        ) -> None:
            pair = analyzer.build_anchor_from_bucket(entries)
            if pair is None:
                return
            score = analyzer.calculator.calculate(
                alt_value=pair["alt_value"],
                alt_uln=pair["alt_uln"],
                alp_value=pair["alp_value"],
                alp_uln=pair["alp_uln"],
            )
            alt = analyzer.pick_best_entry(entries, {"ALT"})
            alp = analyzer.pick_best_entry(entries, {"ALP"})
            calculated.append(
                DiliInjuryPattern(
                    assessment_point="first_qualifying",
                    alt=pair["alt_value"],
                    alt_uln=pair["alt_uln"],
                    alp=pair["alp_value"],
                    alp_uln=pair["alp_uln"],
                    r_ratio=score.r_score,
                    pattern=self.classify(score.r_score),
                    pattern_source="calculated",
                    sample_date=sample_date,
                    evidence=[
                        ClinicalEvidenceQuote(
                            claim="R-ratio input",
                            quote=(alt.evidence if alt is not None else None)
                            or (alp.evidence if alp is not None else None),
                            source_section="laboratory_analysis",
                            event_date=sample_date,
                            source_kind="calculated",
                        )
                    ],
                )
            )

        for sample_date in sorted(buckets, key=analyzer._date_sort_key):
            add_bucket(sample_date, buckets[sample_date])
        add_bucket(
            None,
            [entry for entry in timeline.entries if not entry.sample_date],
        )

        assessable = [item for item in calculated if item.r_ratio is not None]
        if not assessable:
            return [
                DiliInjuryPattern(
                    assessment_point="first_qualifying",
                    pattern="indeterminate",
                    pattern_source="unavailable",
                )
            ]

        chronological = [item for item in assessable if item.sample_date]
        qualifying = [
            item
            for item in chronological
            if self._is_qualifying_pair(item)
        ]
        if not qualifying:
            qualifying = [
                item
                for item in assessable
                if not item.sample_date and self._is_qualifying_pair(item)
            ]
        if not qualifying:
            return [
                DiliInjuryPattern(
                    assessment_point="first_qualifying",
                    pattern="indeterminate",
                    pattern_source="unavailable",
                )
            ]

        first = qualifying[0]
        first_payload = first.model_copy(deep=True)
        first_payload.assessment_point = "first_qualifying"

        peak = max(
            assessable,
            key=lambda item: (
                (item.alt / item.alt_uln)
                if item.alt is not None and item.alt_uln is not None and item.alt_uln > 0
                else -1.0
            ),
        )
        peak_payload = peak.model_copy(deep=True)
        peak_payload.assessment_point = "peak"
        return [first_payload, peak_payload]

    # -------------------------------------------------------------------------
    @staticmethod
    def _is_qualifying_pair(pattern: DiliInjuryPattern) -> bool:
        if pattern.alt is None or pattern.alt_uln is None:
            return False
        if pattern.alp is None or pattern.alp_uln is None:
            return False
        return HepatotoxicityPatternAnalyzer.is_qualifying_pair(
            pattern.alt / pattern.alt_uln,
            pattern.alp / pattern.alp_uln,
        )
