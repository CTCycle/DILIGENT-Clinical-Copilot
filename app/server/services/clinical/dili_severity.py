from __future__ import annotations

import re

from domain.clinical.dili import DiliSeverityAssessment
from domain.clinical.entities import PatientLabTimeline


class DiliSeverityGrader:
    def assess(self, labs: PatientLabTimeline, source_text: str) -> DiliSeverityAssessment:
        lowered = source_text.lower()
        symptomatic = bool(re.search(r"jaundice|fatigue|nausea|pruritus|abdominal pain", lowered))
        if re.search(r"transplant|death|died|fatal", lowered):
            grade = "5_fatal_or_transplant"
        elif re.search(r"encephalopathy|organ failure|acute liver failure", lowered):
            grade = "4_severe"
        elif re.search(r"hospitali[sz]|inr\s*[>:]?\s*1\.[5-9]|coagulopathy", lowered):
            grade = "3_moderate_severe"
        else:
            bilirubin = [
                item for item in labs.entries if item.marker_name.upper() in {"BILIRUBIN", "TBIL"}
            ]
            if any(
                item.value is not None
                and item.upper_limit_normal
                and item.value / item.upper_limit_normal >= 2.5
                for item in bilirubin
            ):
                grade = "2_moderate"
            elif labs.entries:
                grade = "1_mild"
            else:
                grade = "unassessable"
        return DiliSeverityAssessment(
            grade=grade,
            symptom_flag="S" if symptomatic else "A" if source_text.strip() else "unknown",
            rationale=["Deterministic LiverTox/DILIN-style severity screen."],
        )
