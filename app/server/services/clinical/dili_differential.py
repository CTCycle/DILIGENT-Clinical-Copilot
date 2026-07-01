from __future__ import annotations

import re
from typing import cast

from domain.clinical.dili import (
    ClinicalEvidenceQuote,
    DiliCompetingCause,
    DiliDifferentialAssessment,
    EvidenceStatus,
)

CAUSE_PATTERNS: dict[str, tuple[str, ...]] = {
    "viral_hepatitis_a_b_c_d_e": (
        "hepatitis a",
        "hepatitis b",
        "hepatitis c",
        "hav",
        "hbv",
        "hcv",
        "hdv",
        "hev",
    ),
    "ebv_cmv_hsv": ("ebv", "cmv", "hsv", "epstein-barr", "cytomegal"),
    "autoimmune_hepatitis": (
        "autoimmune hepatitis",
        "ana",
        "asma",
        "igg",
        "smooth muscle",
    ),
    "alcoholic_hepatitis": ("alcohol", "ethanol", "etoh"),
    "masld_mash_nash": ("masld", "mash", "nash", "fatty liver", "steatos"),
    "biliary_obstruction_gallstones": (
        "biliary",
        "gallstone",
        "choledo",
        "obstruction",
        "cholangitis",
    ),
    "ischemic_hypoxic": ("ischemi", "hypoxi", "anoxi", "shock liver"),
    "sepsis_shock_cardiac_failure": (
        "sepsis",
        "septic",
        "shock",
        "cardiac failure",
        "heart failure",
    ),
    "overdose_or_toxin": (
        "overdose",
        "acetaminophen",
        "paracetamol",
        "toxin",
        "poison",
    ),
    "supplement_otc_recreational_occupational": (
        "herbal",
        "supplement",
        "otc",
        "recreational",
        "occupational",
    ),
    "pre_existing_chronic_liver_disease": (
        "cirrhos",
        "chronic liver",
        "portal hypertension",
        "hepatitis",
        "fibrosis",
    ),
}

EXCLUDED_RE = re.compile(
    r"\b(negative|excluded|ruled out|without evidence|not detected|serology negative)\b",
    re.IGNORECASE,
)
PRESENT_RE = re.compile(
    r"\b(positive|present|history of|known|documented|consistent with|diagnosed)\b",
    re.IGNORECASE,
)
UNKNOWN_RE = re.compile(
    r"\b(unclear|pending|awaiting|possible|cannot exclude|not assessed yet)\b",
    re.IGNORECASE,
)

###############################################################################
class DiliDifferentialEngine:

    # -------------------------------------------------------------------------
    def assess(self, source_text: str) -> DiliDifferentialAssessment:
        lowered = source_text.lower()
        causes: list[DiliCompetingCause] = []
        unresolved: list[str] = []
        for cause, phrases in CAUSE_PATTERNS.items():
            status, rationale, evidence = self._assess_cause(
                cause, phrases, lowered, source_text
            )
            causes.append(
                DiliCompetingCause(
                    cause=cause,
                    status=status,
                    rationale=rationale,
                    evidence=evidence,
                )
            )
            if status != "excluded":
                unresolved.append(cause)
        return DiliDifferentialAssessment(
            causes=causes,
            all_major_causes_excluded=not unresolved,
            unresolved_causes=unresolved,
        )

    # -------------------------------------------------------------------------
    def _assess_cause(
        self,
        cause: str,
        phrases: tuple[str, ...],
        lowered: str,
        source_text: str,
    ) -> tuple[EvidenceStatus, str, list[ClinicalEvidenceQuote]]:
        matched_phrase = next((phrase for phrase in phrases if phrase in lowered), None)
        if matched_phrase is None:
            return (
                "missing_data",
                "No explicit evaluation for this mandatory competing cause was found.",
                [self._quote(cause, None)],
            )

        idx = lowered.index(matched_phrase)
        window = source_text[max(0, idx - 120) : idx + 180].strip()
        window_lower = window.lower()
        if EXCLUDED_RE.search(window_lower):
            status = "excluded"
            rationale = "Explicit exclusion or negative workup is documented."
        elif PRESENT_RE.search(window_lower):
            status = "not_excluded"
            rationale = "The cause is documented or clinically present, so it remains competing."
        elif UNKNOWN_RE.search(window_lower):
            status = "unknown"
            rationale = "The source mentions this cause but leaves the workup or result unresolved."
        else:
            status = "unknown"
            rationale = "The source mentions this cause without a clear exclusion or confirmation."
        return (
            cast(EvidenceStatus, status),
            rationale,
            [self._quote(cause, window or None)],
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _quote(cause: str, quote: str | None) -> ClinicalEvidenceQuote:
        return ClinicalEvidenceQuote(
            claim=f"Competing cause assessment: {cause}",
            quote=quote,
            source_section="clinical_record",
            source_kind="patient_record" if quote else "missing",
        )
