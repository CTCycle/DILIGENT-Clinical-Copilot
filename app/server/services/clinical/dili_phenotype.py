from __future__ import annotations

import re

from domain.clinical.dili import DiliInjuryPattern, DiliPhenotypeAssessment


###############################################################################
class DiliPhenotypeClassifier:

    # -------------------------------------------------------------------------
    def assess(self, patterns: list[DiliInjuryPattern], source_text: str) -> DiliPhenotypeAssessment:
        lowered = source_text.lower()
        primary_pattern = patterns[0].pattern if patterns else "indeterminate"
        candidates: list[str] = []
        basis: list[str] = [f"R-ratio pattern: {primary_pattern}"]

        if "acute liver failure" in lowered or "encephalopathy" in lowered:
            candidates.append("acute_liver_failure")
            basis.append("acute liver failure or encephalopathy terms present")
        if re.search(r"rash|eosinoph|fever", lowered):
            candidates.append("immunoallergic_hepatitis")
            basis.append("hypersensitivity features present")
        if re.search(r"\bana\b|smooth muscle|autoimmune|igg", lowered):
            candidates.append("autoimmune_like_hepatitis")
            basis.append("autoimmune features present")
        if re.search(r"microvesicular|lactic acidosis|steatos", lowered):
            candidates.append("microvesicular_steatosis_or_lactic_acidosis")
            basis.append("steatotic or lactic-acidosis features present")
        if re.search(r"vanishing bile duct|ductopen", lowered):
            candidates.append("vanishing_bile_duct_syndrome")
            basis.append("ductopenia-related features present")
        if re.search(r"persistent|chronic|6 months|six months", lowered):
            if primary_pattern in {"cholestatic", "mixed"}:
                candidates.append("chronic_cholestatic_injury")
            else:
                candidates.append("chronic_hepatitis_like_injury")
            basis.append("persistent injury terms present")

        if primary_pattern == "hepatocellular":
            candidates.append("acute_hepatocellular_injury")
        elif primary_pattern == "cholestatic":
            candidates.extend(["acute_cholestatic_injury", "bland_cholestasis"])
        elif primary_pattern == "mixed":
            candidates.append("mixed_hepatocellular_cholestatic_injury")

        unique_candidates = list(dict.fromkeys(candidates))
        missing_data = [
            name
            for name in ("biopsy", "imaging", "autoimmune markers", "follow-up over 6 months")
            if name not in lowered
        ]
        return DiliPhenotypeAssessment(
            candidates=unique_candidates,
            primary_candidate=unique_candidates[0] if unique_candidates else None,
            deterministic_basis=basis,
            missing_data=missing_data,
            requires_review=True,
        )
