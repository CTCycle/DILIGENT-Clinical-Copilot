from __future__ import annotations

import re

from domain.clinical.dili import DiliInjuryPattern, DiliPhenotypeAssessment


class DiliPhenotypeClassifier:
    def assess(self, patterns: list[DiliInjuryPattern], source_text: str) -> DiliPhenotypeAssessment:
        lowered = source_text.lower()
        primary_pattern = patterns[0].pattern if patterns else "indeterminate"
        candidates: list[str] = []
        if "encephalopathy" in lowered or "acute liver failure" in lowered:
            candidates.append("acute_liver_failure")
        if re.search(r"rash|eosinoph|fever", lowered):
            candidates.append("immunoallergic_hepatitis")
        if re.search(r"\bana\b|smooth muscle|autoimmune|igg", lowered):
            candidates.append("autoimmune_like_hepatitis")
        if "lactic acidosis" in lowered or "microvesicular" in lowered:
            candidates.append("acute_fatty_liver_with_lactic_acidosis")
        if re.search(r"persistent|chronic|six months|6 months", lowered):
            candidates.append("chronic_hepatitis")
        if primary_pattern == "hepatocellular":
            candidates.append("acute_hepatitis")
        elif primary_pattern in {"cholestatic", "mixed"}:
            candidates.append("cholestatic_hepatitis")
        return DiliPhenotypeAssessment(
            candidates=list(dict.fromkeys(candidates)),
            primary_candidate=candidates[0] if candidates else None,
            deterministic_basis=[f"R-ratio pattern: {primary_pattern}"],
            missing_data=[
                name
                for name in ("imaging", "biopsy", "autoimmune_markers", "encephalopathy")
                if name.replace("_", " ") not in lowered
            ],
            requires_review=True,
        )
