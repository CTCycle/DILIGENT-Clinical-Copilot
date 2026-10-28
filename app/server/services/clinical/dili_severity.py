# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import re

from domain.clinical.dili import ClinicalEvidenceQuote, DiliSeverityAssessment
from domain.clinical.entities import ClinicalLabEntry, PatientLabTimeline
from services.clinical.dili_differential import NEGATION_CUE_RE

SYMPTOM_RE = re.compile(
    r"\b(jaundice|pruritus|fatigue|nausea|vomiting|abdominal pain|dark urine|itch(?:ing)?)\b",
    re.IGNORECASE,
)
# DILIN severity scale (Fontana 2009, as reproduced in LiverTox).
BILIRUBIN_THRESHOLD_MG_DL = 2.5
INR_THRESHOLD = 1.5
BILIRUBIN_UMOL_PER_MG_DL = 17.1

SENTENCE_SPLIT_RE = re.compile(r"(?<!\d)\.(?!\d)|[;\n!?]")
HOSPITALIZATION_RE = re.compile(
    r"\b(hospitali[sz]\w*|admitted|admission|inpatient)\b", re.IGNORECASE
)
DEATH_RE = re.compile(r"\b(died|death|deceased|fatal\w*|passed away)\b", re.IGNORECASE)
TRANSPLANT_RE = re.compile(r"\btransplant\w*\b", re.IGNORECASE)
DECOMPENSATION_RE = re.compile(
    r"\b(encephalopathy|ascites|liver failure|(?:multi-?)?organ failure|"
    r"renal failure|kidney failure|respiratory failure)\b",
    re.IGNORECASE,
)
LIVER_CONTEXT_RE = re.compile(
    r"\b(liver|hepat\w*|jaundice|icter\w*|dili|bilirubin|transaminases?|"
    r"aminotransferases?|alt|coagulopathy|encephalopathy|ascites|drug-induced)\b",
    re.IGNORECASE,
)
# Cues placing an event outside the current patient episode: another person,
# a past history, or a hypothetical or planned event.
FAMILY_CONTEXT_RE = re.compile(
    r"\b(father|mother|brother|sister|son|daughter|parents?|siblings?|family|"
    r"grand(?:father|mother|parent)s?|uncle|aunt|wife|husband|spouse|cousin)\b",
    re.IGNORECASE,
)
NON_EPISODE_PREFIX_RE = re.compile(
    r"\b(history of|previous|prior|status post|s/p|risk of|potentially|"
    r"(?-i:may)|might|could|would|if|prevent\w*|listed for|listing for|"
    r"evaluat\w*|referr\w*|candidate|consider\w*)\b",
    re.IGNORECASE,
)

###############################################################################
class DiliSeverityGrader:

    # -------------------------------------------------------------------------
    def assess(
        self, labs: PatientLabTimeline, source_text: str
    ) -> DiliSeverityAssessment:
        lowered = source_text.lower()
        sentences = [item for item in SENTENCE_SPLIT_RE.split(source_text) if item.strip()]
        bilirubin_peak = self._peak_bilirubin_mg_dl(labs)
        inr_peak = self._peak_value(labs, {"INR"})
        symptom_flag = (
            "S"
            if SYMPTOM_RE.search(lowered)
            else "A"
            if source_text.strip()
            else "unknown"
        )
        evidence = self._supporting_evidence(labs)
        bilirubin_high = (
            bilirubin_peak is not None and bilirubin_peak >= BILIRUBIN_THRESHOLD_MG_DL
        )
        inr_high = inr_peak is not None and inr_peak >= INR_THRESHOLD
        death_or_transplant = self._documented_event(
            sentences, DEATH_RE
        ) or self._documented_event(sentences, TRANSPLANT_RE, require_liver_context=True)
        decompensation = self._documented_event(sentences, DECOMPENSATION_RE)
        hospitalization = self._documented_event(
            sentences, HOSPITALIZATION_RE, require_liver_context=True
        )

        if death_or_transplant:
            grade = "5_fatal_or_transplant"
            rationale = [f"Death or liver transplant is documented: {death_or_transplant}"]
        elif bilirubin_high and (inr_high or decompensation):
            grade = "4_severe"
            rationale = [
                "Total bilirubin >= 2.5 mg/dL with hepatic decompensation or organ "
                f"failure ({decompensation or 'INR >= 1.5'})."
            ]
        elif bilirubin_high and hospitalization:
            grade = "3_moderate_severe"
            rationale = [
                "Total bilirubin >= 2.5 mg/dL with liver-related hospitalization: "
                f"{hospitalization}"
            ]
        elif bilirubin_high or inr_high:
            grade = "2_moderate"
            rationale = ["Total bilirubin >= 2.5 mg/dL or INR >= 1.5."]
        elif labs.entries:
            grade = "1_mild"
            rationale = [
                "Total bilirubin < 2.5 mg/dL and INR < 1.5 where reported."
                if bilirubin_peak is not None or inr_peak is not None
                else "No bilirubin or INR is reported; the grade may be underestimated."
            ]
        else:
            grade = "unassessable"
            rationale = ["No laboratory evidence was available for severity grading."]
        return DiliSeverityAssessment(
            grade=grade,
            symptom_flag=symptom_flag,
            evidence=evidence,
            rationale=rationale,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _documented_event(
        sentences: list[str],
        event_re: re.Pattern[str],
        *,
        require_liver_context: bool = False,
    ) -> str | None:
        """Return the first sentence asserting the event for the current episode."""
        for sentence in sentences:
            if FAMILY_CONTEXT_RE.search(sentence):
                continue
            if require_liver_context and not LIVER_CONTEXT_RE.search(sentence):
                continue
            for match in event_re.finditer(sentence):
                clause_prefix = sentence[: match.start()].rsplit(",", 1)[-1][-40:]
                if NEGATION_CUE_RE.search(clause_prefix) or NON_EPISODE_PREFIX_RE.search(
                    clause_prefix
                ):
                    continue
                return sentence.strip()
        return None

    # -------------------------------------------------------------------------
    @classmethod
    def _peak_bilirubin_mg_dl(cls, labs: PatientLabTimeline) -> float | None:
        values = [
            converted
            for item in labs.entries
            if item.marker_name.upper() in {"BILIRUBIN", "TBIL"}
            for converted in [cls._bilirubin_mg_dl(item)]
            if converted is not None
        ]
        return max(values) if values else None

    # -------------------------------------------------------------------------
    @staticmethod
    def _bilirubin_mg_dl(item: ClinicalLabEntry) -> float | None:
        """Total bilirubin in mg/dL; µmol/L is divided by 17.1.

        Without a unit, the ULN magnitude identifies it (about 1.2 mg/dL versus
        about 20 µmol/L).
        """
        if item.value is None:
            return None
        unit = (item.unit or "").casefold()
        uln = float(item.upper_limit_normal or 0)
        if "mg/dl" in unit:
            return float(item.value)
        if "mol/l" in unit:
            return float(item.value) / BILIRUBIN_UMOL_PER_MG_DL
        if unit or uln <= 0:
            return None
        if uln <= 3:
            return float(item.value)
        if uln >= 5:
            return float(item.value) / BILIRUBIN_UMOL_PER_MG_DL
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _peak_value(labs: PatientLabTimeline, markers: set[str]) -> float | None:
        values = [
            float(item.value)
            for item in labs.entries
            if item.marker_name.upper() in markers and item.value is not None
        ]
        return max(values) if values else None

    # -------------------------------------------------------------------------
    @staticmethod
    def _supporting_evidence(labs: PatientLabTimeline) -> list[ClinicalEvidenceQuote]:
        evidence: list[ClinicalEvidenceQuote] = []
        for lab in labs.entries[:4]:
            evidence.append(
                ClinicalEvidenceQuote(
                    claim=f"severity marker {lab.marker_name}",
                    quote=lab.evidence or f"{lab.marker_name} {lab.value}",
                    source_section=lab.source,
                    event_date=lab.sample_date,
                    source_kind="patient_record"
                    if lab.value is not None
                    else "missing",
                )
            )
        return evidence
