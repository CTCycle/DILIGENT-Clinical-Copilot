# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import re
from typing import cast

from domain.clinical.dili import (
    ClinicalEvidenceQuote,
    DiliCompetingCause,
    DiliDifferentialAssessment,
    EvidenceStatus,
)

# Each phrase is a regular expression matched case-insensitively at word
# boundaries; stems carry an explicit `\w*` suffix.
CAUSE_PATTERNS: dict[str, tuple[str, ...]] = {
    "viral_hepatitis_a_b_c_d_e": (
        r"hepatitis [a-e]",
        r"hav",
        r"hbv",
        r"hcv",
        r"hdv",
        r"hev",
        r"hbsag",
        r"anti-hbc",
    ),
    "ebv_cmv_hsv": (r"ebv", r"cmv", r"hsv", r"epstein-barr", r"cytomegal\w*"),
    "autoimmune_hepatitis": (
        r"autoimmune\w*",
        r"ana",
        r"asma",
        r"igg",
        r"smooth muscle",
    ),
    "alcoholic_hepatitis": (r"alcohol\w*", r"ethanol", r"etoh"),
    "masld_mash_nash": (r"masld", r"mash", r"nash", r"fatty liver", r"steatos\w*"),
    "biliary_obstruction_gallstones": (
        r"biliary",
        r"bile ducts?",
        r"gallstones?",
        r"choledoch\w*",
        r"cholangitis",
    ),
    "ischemic_hypoxic": (r"ischa?emi\w*", r"hypoxi\w*", r"anoxi\w*", r"shock liver"),
    "sepsis_shock_cardiac_failure": (
        r"sepsis",
        r"septic",
        r"shock",
        r"cardiac failure",
        r"heart failure",
    ),
    "overdose_or_toxin": (
        r"overdose",
        r"acetaminophen",
        r"paracetamol",
        r"toxins?",
        r"poison\w*",
    ),
    "supplement_otc_recreational_occupational": (
        r"herbal",
        r"supplements?",
        r"otc",
        r"recreational",
        r"occupational",
    ),
    "pre_existing_chronic_liver_disease": (
        r"cirrhos\w*",
        r"chronic liver",
        r"portal hypertension",
        r"chronic hepatitis",
        r"fibrosis",
    ),
}

BASELINE_CONTEXT_CAUSES = {
    "masld_mash_nash",
    "pre_existing_chronic_liver_disease",
}

# Clauses are split on sentence punctuation, semicolons, newlines and commas
# that are not decimal separators, so each laboratory finding is judged alone.
CLAUSE_SPLIT_RE = re.compile(r"(?<!\d)\.(?!\d)|[;\n]|,(?!\d)")
NEGATION_CUE_RE = re.compile(
    r"\b(no|not|without|denies|denied|negative for|absence of|free of)\b",
    re.IGNORECASE,
)
NEGATIVE_RESULT_RE = re.compile(
    r"\b(negative|not detected|undetectable|excluded|ruled out|normal|"
    r"unremarkable|none)\b",
    re.IGNORECASE,
)
POSITIVE_RE = re.compile(
    r"\b(positive|detected|present|confirmed|diagnosed|consistent with|"
    r"elevated|dilated|ongoing|history of|known|documented)\b"
    r"|\d+(?:\.\d+)?\s*(?:g|grams?|units?|drinks?)\s*(?:/|per)\s*(?:day|week)\b",
    re.IGNORECASE,
)
UNKNOWN_RE = re.compile(
    r"\b(unclear|pending|awaiting|possible|cannot exclude|cannot be excluded|"
    r"not excluded|not ruled out|cannot rule out|not assessed yet)\b",
    re.IGNORECASE,
)
STATUS_PRIORITY: dict[str, int] = {"not_excluded": 2, "unknown": 1, "excluded": 0}

###############################################################################
class DiliDifferentialEngine:

    # -------------------------------------------------------------------------
    def assess(self, source_text: str) -> DiliDifferentialAssessment:
        causes: list[DiliCompetingCause] = []
        unresolved: list[str] = []
        for cause, phrases in CAUSE_PATTERNS.items():
            status, rationale, evidence = self._assess_cause(
                cause, phrases, source_text
            )
            causes.append(
                DiliCompetingCause(
                    cause=cause,
                    status=status,
                    rationale=rationale,
                    evidence=evidence,
                )
            )
            if self._blocks_exclusion(cause, status):
                unresolved.append(cause)
        return DiliDifferentialAssessment(
            causes=causes,
            all_major_causes_excluded=not unresolved,
            unresolved_causes=unresolved,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _blocks_exclusion(cause: str, status: EvidenceStatus) -> bool:
        if cause in BASELINE_CONTEXT_CAUSES:
            return status in {"not_excluded", "unknown"}
        return status != "excluded"

    # -------------------------------------------------------------------------
    def _assess_cause(
        self,
        cause: str,
        phrases: tuple[str, ...],
        source_text: str,
    ) -> tuple[EvidenceStatus, str, list[ClinicalEvidenceQuote]]:
        phrase_re = re.compile(r"\b(?:" + "|".join(phrases) + r")\b", re.IGNORECASE)
        assessed: list[tuple[str, str]] = []
        for clause in CLAUSE_SPLIT_RE.split(source_text):
            match = phrase_re.search(clause)
            if match is None:
                continue
            assessed.append((self._clause_status(clause, match.start()), clause.strip()))

        if not assessed:
            if cause in BASELINE_CONTEXT_CAUSES:
                return (
                    "missing_data",
                    "No explicit baseline documentation for this chronic liver context was found; absence is not treated as exclusion.",
                    [self._quote(cause, None)],
                )
            return (
                "missing_data",
                "No explicit evaluation for this major competing cause was found.",
                [self._quote(cause, None)],
            )

        status = max((item[0] for item in assessed), key=STATUS_PRIORITY.__getitem__)
        if status == "excluded":
            rationale = "Explicit exclusion or negative workup is documented."
        elif status == "not_excluded":
            rationale = (
                "The condition is documented or clinically present and must be interpreted as baseline context and/or a competing explanation."
                if cause in BASELINE_CONTEXT_CAUSES
                else "The cause is documented or clinically present, so it remains competing."
            )
        else:
            rationale = "The source mentions this cause but leaves the workup or result unresolved."
        quote = "; ".join(clause for item_status, clause in assessed if item_status == status)
        return (
            cast(EvidenceStatus, status),
            rationale,
            [self._quote(cause, quote[:500] or None)],
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _clause_status(clause: str, phrase_start: int) -> str:
        if NEGATION_CUE_RE.search(clause[max(0, phrase_start - 40) : phrase_start]):
            return "excluded"
        uncertain = UNKNOWN_RE.search(clause)
        resolved_text = UNKNOWN_RE.sub(" ", clause)
        negative = NEGATIVE_RESULT_RE.search(resolved_text)
        positive = POSITIVE_RE.search(NEGATIVE_RESULT_RE.sub(" ", resolved_text))
        if positive:
            return "not_excluded"
        if uncertain:
            return "unknown"
        if negative:
            return "excluded"
        return "unknown"

    # -------------------------------------------------------------------------
    @staticmethod
    def _quote(cause: str, quote: str | None) -> ClinicalEvidenceQuote:
        return ClinicalEvidenceQuote(
            claim=f"Competing cause assessment: {cause}",
            quote=quote,
            source_section="clinical_record",
            source_kind="patient_record" if quote else "missing",
        )
