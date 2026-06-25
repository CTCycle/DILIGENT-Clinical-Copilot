from __future__ import annotations

import re

from domain.clinical.dili import (
    ClinicalEvidenceQuote,
    DiliCompetingCause,
    DiliDifferentialAssessment,
)


class DiliDifferentialEngine:
    CAUSES = {
        "viral_hepatitis_a_b_c_d_e": r"hepatitis\s*[abcde]|hav|hbv|hcv|hdv|hev",
        "ebv_cmv_hsv": r"\bebv\b|\bcmv\b|\bhsv\b|epstein.barr|cytomegal",
        "autoimmune_hepatitis": r"autoimmune|ana\b|asma\b|igg\b",
        "alcoholic_hepatitis": r"alcohol|ethanol",
        "masld_mash_nash": r"masld|mash|nash|fatty liver|steatos",
        "biliary_obstruction_gallstones": r"biliary|gallstone|choledo|obstruction",
        "ischemic_hypoxic": r"ischemi|hypoxi|anoxi",
        "sepsis_shock_cardiac_failure": r"sepsis|septic|shock|cardiac failure|heart failure",
        "overdose_or_toxin": r"overdose|acetaminophen|paracetamol|toxin",
        "supplement_otc_recreational_occupational": r"herbal|supplement|otc|recreational|occupational",
        "pre_existing_chronic_liver_disease": r"cirrhos|chronic liver|portal hypertension",
    }
    EXCLUSION = re.compile(r"\b(no|negative|excluded|ruled out|without)\b", re.I)

    def assess(self, source_text: str) -> DiliDifferentialAssessment:
        causes: list[DiliCompetingCause] = []
        for cause, pattern in self.CAUSES.items():
            match = re.search(pattern, source_text, re.I)
            if match is None:
                status, rationale, quote = "missing_data", "No documented evaluation.", None
            else:
                window = source_text[max(0, match.start() - 60) : match.end() + 100]
                excluded = bool(self.EXCLUSION.search(window))
                status = "excluded" if excluded else "not_excluded"
                rationale = (
                    "Source documents exclusion/negative evaluation."
                    if excluded
                    else "Cause is mentioned but not documented as excluded."
                )
                quote = window.strip()
            causes.append(
                DiliCompetingCause(
                    cause=cause,
                    status=status,
                    rationale=rationale,
                    evidence=[
                        ClinicalEvidenceQuote(
                            claim=f"Competing cause: {cause}",
                            quote=quote,
                            source_section="clinical_record",
                            source_kind="patient_record" if quote else "missing",
                        )
                    ],
                )
            )
        unresolved = [item.cause for item in causes if item.status != "excluded"]
        return DiliDifferentialAssessment(
            causes=causes,
            all_major_causes_excluded=not unresolved,
            unresolved_causes=unresolved,
        )
