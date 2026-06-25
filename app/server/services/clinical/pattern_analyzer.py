from __future__ import annotations

import re

from common.constants import (
    DEFAULT_DILI_CLASSIFICATION,
    R_SCORE_CHOLESTATIC_THRESHOLD,
    R_SCORE_HEPATOCELLULAR_THRESHOLD,
)
from domain.clinical.entities import (
    ClinicalLabEntry,
    HepatotoxicityPatternAssessment,
    HepatotoxicityPatternScore,
    PatientLabTimeline,
    PipelineIssue,
)

NOT_AVAILABLE_TEXT = "N/A"

###############################################################################
class HepatotoxicityPatternCalculator:

    # -------------------------------------------------------------------------
    def calculate(
        self,
        *,
        alt_value: float | None,
        alt_uln: float | None,
        alp_value: float | None,
        alp_uln: float | None,
    ) -> HepatotoxicityPatternScore:
        alt_multiple = self.safe_ratio(alt_value, alt_uln)
        alp_multiple = self.safe_ratio(alp_value, alp_uln)

        r_score = None
        if alt_multiple is not None and alp_multiple not in (None, 0.0):
            r_score = alt_multiple / alp_multiple

        classification = DEFAULT_DILI_CLASSIFICATION
        if r_score is not None:
            if r_score >= R_SCORE_HEPATOCELLULAR_THRESHOLD:
                classification = "hepatocellular"
            elif r_score <= R_SCORE_CHOLESTATIC_THRESHOLD:
                classification = "cholestatic"
            else:
                classification = "mixed"

        return HepatotoxicityPatternScore(
            alt_multiple=alt_multiple,
            alp_multiple=alp_multiple,
            r_score=r_score,
            classification=classification,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def safe_ratio(value: float | None, reference: float | None) -> float | None:
        if value is None or reference is None:
            return None
        if reference == 0:
            return None
        return value / reference

###############################################################################
class HepatotoxicityPatternAnalyzer:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.r_score: float | None = None
        self.calculator = HepatotoxicityPatternCalculator()

    # -------------------------------------------------------------------------
    def calculate_hepatotoxicity_pattern(
        self, lab_timeline: PatientLabTimeline
    ) -> HepatotoxicityPatternScore:
        anchor = self.select_anchor_pair(lab_timeline)
        if anchor is None:
            score = HepatotoxicityPatternScore(
                alt_multiple=None,
                alp_multiple=None,
                r_score=None,
                classification=DEFAULT_DILI_CLASSIFICATION,
            )
            self.r_score = None
            return score
        score = self.calculator.calculate(
            alt_value=anchor["alt_value"],
            alt_uln=anchor["alt_uln"],
            alp_value=anchor["alp_value"],
            alp_uln=anchor["alp_uln"],
        )
        self.r_score = score.r_score
        return score

    # -------------------------------------------------------------------------
    def assess_payload(
        self,
        lab_timeline: PatientLabTimeline,
    ) -> HepatotoxicityPatternAssessment:
        score = self.calculate_hepatotoxicity_pattern(lab_timeline)
        if score.r_score is None:
            issue = PipelineIssue(
                severity="warning",
                code="missing_hepatotoxicity_inputs",
                message=(
                    "Laboratory data are insufficient for a numeric R ratio "
                    "(ideally ALT/AST, ALP, and bilirubin). Continuing with "
                    "indeterminate pattern and reduced confidence."
                ),
                field="laboratory_analysis",
            )
            self.r_score = None
            return HepatotoxicityPatternAssessment(
                score=score,
                status="undetermined_due_to_missing_labs",
                issues=[issue],
            )
        self.r_score = score.r_score
        return HepatotoxicityPatternAssessment(
            score=score,
            status="ok",
            issues=[],
        )

    # -------------------------------------------------------------------------
    def select_anchor_pair(
        self, lab_timeline: PatientLabTimeline
    ) -> dict[str, float] | None:
        dated_candidates = self.group_entries_by_date(lab_timeline.entries)
        for sample_date in sorted(dated_candidates):
            bucket = dated_candidates[sample_date]
            pair = self.build_anchor_from_bucket(bucket)
            if pair is not None:
                return pair
        undated = self.build_anchor_from_bucket(lab_timeline.entries)
        return undated

    # -------------------------------------------------------------------------
    def group_entries_by_date(
        self,
        entries: list[ClinicalLabEntry],
    ) -> dict[str, list[ClinicalLabEntry]]:
        grouped: dict[str, list[ClinicalLabEntry]] = {}
        for entry in entries:
            if not entry.sample_date:
                continue
            grouped.setdefault(entry.sample_date, []).append(entry)
        return grouped

    # -------------------------------------------------------------------------
    def build_anchor_from_bucket(
        self,
        entries: list[ClinicalLabEntry],
    ) -> dict[str, float] | None:
        alt_like = self.pick_best_entry(entries, {"ALT", "AST"})
        alp = self.pick_best_entry(entries, {"ALP"})
        if alt_like is None or alp is None:
            return None
        alt_value = self.parse_entry_value(alt_like)
        alp_value = self.parse_entry_value(alp)
        if alt_value is None or alp_value is None:
            return None
        alt_uln = self.resolve_uln(alt_like, fallback=40.0)
        alp_uln = self.resolve_uln(alp, fallback=120.0)
        if alt_uln <= 0 or alp_uln <= 0:
            return None
        return {
            "alt_value": alt_value,
            "alt_uln": alt_uln,
            "alp_value": alp_value,
            "alp_uln": alp_uln,
        }

    # -------------------------------------------------------------------------
    def pick_best_entry(
        self,
        entries: list[ClinicalLabEntry],
        marker_names: set[str],
    ) -> ClinicalLabEntry | None:
        selected: ClinicalLabEntry | None = None
        for entry in entries:
            if entry.marker_name.upper() not in marker_names:
                continue
            if selected is None:
                selected = entry
                continue
            selected_value = self.parse_entry_value(selected)
            current_value = self.parse_entry_value(entry)
            if selected_value is None and current_value is not None:
                selected = entry
            elif (
                current_value is not None
                and selected_value is not None
                and current_value > selected_value
            ):
                selected = entry
        return selected

    # -------------------------------------------------------------------------
    def parse_entry_value(self, entry: ClinicalLabEntry) -> float | None:
        if entry.value is not None:
            return float(entry.value)
        return self.parse_marker_value(entry.value_text)

    # -------------------------------------------------------------------------
    def resolve_uln(self, entry: ClinicalLabEntry, *, fallback: float) -> float:
        if entry.upper_limit_normal is not None and entry.upper_limit_normal > 0:
            return float(entry.upper_limit_normal)
        parsed = self.parse_marker_value(entry.upper_limit_text)
        if parsed is not None and parsed > 0:
            return parsed
        return fallback

    # -------------------------------------------------------------------------
    def parse_marker_value(self, raw: str | None) -> float | None:
        if raw is None:
            return None
        normalized = raw.replace(",", ".")
        match = re.search(r"[-+]?\d*\.?\d+", normalized)
        if not match:
            return None
        try:
            return float(match.group())
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    def safe_ratio(self, value: float | None, reference: float | None) -> float | None:
        return self.calculator.safe_ratio(value, reference)

    # -------------------------------------------------------------------------
    def stringify_scores(
        self, pattern_score: HepatotoxicityPatternScore | None
    ) -> dict[str, str]:
        if not pattern_score:
            return {}

        mapping = {
            "alt_multiple": (pattern_score.alt_multiple, "{:.2f}x ULN"),
            "alp_multiple": (pattern_score.alp_multiple, "{:.2f}x ULN"),
            "r_score": (pattern_score.r_score, "{:.2f}"),
        }

        return {
            key: fmt.format(val) if val is not None else NOT_AVAILABLE_TEXT
            for key, (val, fmt) in mapping.items()
        }


###############################################################################
