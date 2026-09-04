from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

from domain.clinical.dili import ClinicalEvidenceQuote, DiliInjuryPattern
from domain.clinical.entities import ClinicalLabEntry, PatientLabTimeline


###############################################################################
class DiliPatternEngine:
    # -------------------------------------------------------------------------
    @staticmethod
    def _value(entry: ClinicalLabEntry) -> float | None:
        if entry.value is not None:
            return float(entry.value)
        raw = str(entry.value_text or "").replace(",", ".")
        match = re.search(r"[-+]?\d*\.?\d+", raw)
        return float(match.group()) if match else None

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
            current_multiple = (
                value / uln if value is not None and uln is not None and uln > 0 else None
            )
            if selected is None:
                selected = entry
                selected_multiple = current_multiple
                continue
            if selected_multiple is None and current_multiple is not None:
                selected = entry
                selected_multiple = current_multiple
            elif (
                selected_multiple is not None
                and current_multiple is not None
                and current_multiple > selected_multiple
            ):
                selected = entry
                selected_multiple = current_multiple
        return selected

    # -------------------------------------------------------------------------
    @staticmethod
    def _uln(entry: ClinicalLabEntry) -> float | None:
        if entry.upper_limit_normal and entry.upper_limit_normal > 0:
            return float(entry.upper_limit_normal)
        raw = str(entry.upper_limit_text or "").replace(",", ".")
        match = re.search(r"[-+]?\d*\.?\d+", raw)
        if not match:
            return None
        parsed = float(match.group())
        return parsed if parsed > 0 else None

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
                and alt_uln > 0
                and alp_uln > 0
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
                            event_date=None if sample_date == "undated" else sample_date,
                            source_kind="calculated",
                        )
                    ],
                )
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

        dated = [item for item in assessable if item.sample_date]
        chronological = sorted(dated, key=lambda item: self._date_sort_key(item.sample_date))
        abnormal = [
            item
            for item in chronological
            if self._is_abnormal_pair(item)
        ]
        first = abnormal[0] if abnormal else (chronological[0] if chronological else assessable[0])
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
    def _is_abnormal_pair(pattern: DiliInjuryPattern) -> bool:
        alt_multiple = (
            pattern.alt / pattern.alt_uln
            if pattern.alt is not None and pattern.alt_uln is not None and pattern.alt_uln > 0
            else None
        )
        alp_multiple = (
            pattern.alp / pattern.alp_uln
            if pattern.alp is not None and pattern.alp_uln is not None and pattern.alp_uln > 0
            else None
        )
        return bool(
            (alt_multiple is not None and alt_multiple > 1.0)
            or (alp_multiple is not None and alp_multiple > 1.0)
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _date_sort_key(value: str | None) -> date:
        if not value:
            return date.max
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
        return date.max
