from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from DILIGENT.server.domain.clinical.entities import (
    DrugClinicalAssessment,
    HepatotoxicityPatternAssessment,
    LiverInjuryOnsetContext,
    PatientDiseaseContext,
    PatientDrugs,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
)


@dataclass(frozen=True)
class SessionStructuredInputs:
    patient_name: str | None
    visit_date: date | None
    therapy_drugs: PatientDrugs
    anamnesis_drugs: PatientDrugs
    combined_drugs: PatientDrugs
    disease_context: PatientDiseaseContext
    lab_timeline: PatientLabTimeline
    onset_context: LiverInjuryOnsetContext | None
    pattern: HepatotoxicityPatternAssessment
    rucam: PatientRucamAssessmentBundle


@dataclass(frozen=True)
class SessionKnowledgeBundle:
    matched_drugs: list[DrugClinicalAssessment]
    matched_drugs_payload: list[dict[str, Any]]
    clinical_context: str

