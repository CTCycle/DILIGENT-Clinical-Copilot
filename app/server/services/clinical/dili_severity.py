from __future__ import annotations

import re

from domain.clinical.dili import ClinicalEvidenceQuote, DiliSeverityAssessment
from domain.clinical.entities import ClinicalLabEntry, PatientLabTimeline

SYMPTOM_RE = re.compile(
    r"\b(jaundice|pruritus|fatigue|nausea|vomiting|abdominal pain|dark urine|itch(?:ing)?)\b",
    re.IGNORECASE,
)


class DiliSeverityGrader:
    def assess(self, labs: PatientLabTimeline, source_text: str) -> DiliSeverityAssessment:
        lowered = source_text.lower()
        bilirubin_peak = self._peak_multiple(labs, {"BILIRUBIN", "TBIL"})
        inr_peak = self._peak_value(labs, {"INR"})
        symptom_flag = "S" if SYMPTOM_RE.search(lowered) else "A" if source_text.strip() else "unknown"
        evidence = self._supporting_evidence(labs)

        if re.search(r"transplant|death|died|fatal", lowered):
            grade = "5_fatal_or_transplant"
            rationale = ["Death or transplant language is documented."]
        elif re.search(r"encephalopathy|acute liver failure|multi-organ failure|organ failure", lowered):
            grade = "4_severe"
            rationale = ["Severe liver failure features are documented."]
        elif re.search(r"hospitali[sz]|admitted", lowered) or (inr_peak is not None and inr_peak >= 1.5):
            grade = "3_moderate_severe"
            rationale = ["Hospitalization or INR >= 1.5 suggests moderate-severe injury."]
        elif bilirubin_peak is not None and bilirubin_peak >= 2.5:
            grade = "2_moderate"
            rationale = ["Peak bilirubin is at least 2.5 x ULN without severe failure features."]
        elif labs.entries:
            grade = "1_mild"
            rationale = ["Liver injury is present without bilirubin or severe failure escalation."]
        else:
            grade = "unassessable"
            rationale = ["No laboratory evidence was available for severity grading."]
        return DiliSeverityAssessment(
            grade=grade,
            symptom_flag=symptom_flag,
            evidence=evidence,
            rationale=rationale,
        )

    @staticmethod
    def _peak_multiple(labs: PatientLabTimeline, markers: set[str]) -> float | None:
        values = [
            float(item.value) / float(item.upper_limit_normal)
            for item in labs.entries
            if item.marker_name.upper() in markers
            and item.value is not None
            and item.upper_limit_normal
            and float(item.upper_limit_normal) > 0
        ]
        return max(values) if values else None

    @staticmethod
    def _peak_value(labs: PatientLabTimeline, markers: set[str]) -> float | None:
        values = [
            float(item.value)
            for item in labs.entries
            if item.marker_name.upper() in markers and item.value is not None
        ]
        return max(values) if values else None

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
                    source_kind="patient_record" if lab.value is not None else "missing",
                )
            )
        return evidence
