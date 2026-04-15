from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Any

from DILIGENT.server.api import session as session_module
from DILIGENT.server.domain.clinical import (
    ClinicalLabEntry,
    ClinicalPipelineValidationError,
    DrugEntry,
    DrugRucamAssessment,
    HepatotoxicityPatternScore,
    LiverInjuryOnsetContext,
    PatientData,
    PatientDiseaseContext,
    PatientDrugs,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
    RucamComponentAssessment,
)
from DILIGENT.server.services.clinical.preparation import HepatoxPreparedInputs


class FakeSerializer:
    def __init__(self) -> None:
        self.saved_payload: dict[str, Any] | None = None

    def save_clinical_session(self, payload: dict[str, Any]) -> None:
        self.saved_payload = payload


class FakeInputPreparator:
    async def prepare_inputs(self, *args: Any, **kwargs: Any) -> HepatoxPreparedInputs:
        _ = args
        _ = kwargs
        return HepatoxPreparedInputs(
            resolved_drugs={
                "drug-a": {
                    "matched_livertox_row": {"drug_name": "Drug A", "likelihood_score": "B"},
                    "match_confidence": 0.9,
                    "match_reason": "exact",
                    "match_status": "matched",
                    "match_notes": [],
                    "match_candidates": [],
                    "chosen_candidate": "Drug A",
                    "rejected_candidates": [],
                    "missing_livertox": False,
                    "ambiguous_match": False,
                    "origins": ["therapy"],
                    "raw_mentions": ["Drug A"],
                    "regimen_group_ids": [],
                    "regimen_components": [],
                }
            },
            pattern_prompt="Pattern summary.",
            clinical_context="Clinical context.",
        )


class FakeHepatoxConsultation:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _ = args
        _ = kwargs
        self.llm_model = "fake-clinical-model"

    async def run_analysis(self, **kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        return {"final_report": "Final report body."}


@dataclass
class FakeLabExtractor:
    fail: bool = False

    async def extract_from_payload(
        self,
        payload: PatientData,
        *,
        progress_callback: Any | None = None,
    ) -> tuple[PatientLabTimeline, LiverInjuryOnsetContext | None]:
        _ = payload
        _ = progress_callback
        if self.fail:
            raise RuntimeError("lab extraction unavailable")
        return (
            PatientLabTimeline(
                entries=[
                    ClinicalLabEntry(
                        marker_name="ALT",
                        value=300.0,
                        upper_limit_normal=40.0,
                        sample_date="2025-01-10",
                        source="laboratory_analysis",
                    ),
                    ClinicalLabEntry(
                        marker_name="ALP",
                        value=130.0,
                        upper_limit_normal=120.0,
                        sample_date="2025-01-10",
                        source="laboratory_analysis",
                    )
                ]
            ),
            LiverInjuryOnsetContext(
                onset_date="2025-01-10",
                onset_basis="first_abnormal_lab",
                evidence="First abnormal ALT on 2025-01-10.",
            ),
        )


@dataclass
class FakeRucamEstimator:
    fail: bool = False

    def estimate(self, **kwargs: Any) -> PatientRucamAssessmentBundle:
        _ = kwargs
        if self.fail:
            raise RuntimeError("rucam unavailable")
        return PatientRucamAssessmentBundle(
            entries=[
                DrugRucamAssessment(
                    drug_name="Drug A",
                    injury_type_for_rucam="hepatocellular",
                    total_score=7,
                    causality_category="probable",
                    confidence="moderate",
                    components=[
                        RucamComponentAssessment(
                            component_key="time_to_onset",
                            label="Time to onset",
                            score=2,
                            status="scored",
                        )
                    ],
                    limitations=["Sparse rechallenge data"],
                    summary="Estimated RUCAM summary.",
                )
            ]
        )


class FakeDrugsParser:
    model = "fake-parser-model"

    def clean_text(self, text: str) -> str:
        return text

    async def extract_drugs_from_therapy(self, text: str, **kwargs: Any) -> PatientDrugs:
        _ = text
        _ = kwargs
        return PatientDrugs(
            entries=[
                DrugEntry(
                    name="Drug A",
                    source="therapy",
                    therapy_start_status=True,
                    therapy_start_date="2024-12-20",
                    temporal_classification="temporal_known",
                )
            ]
        )

    async def extract_drugs_from_anamnesis(self, text: str | None, **kwargs: Any) -> PatientDrugs:
        _ = text
        _ = kwargs
        return PatientDrugs(entries=[])


class FakeDiseaseExtractor:
    async def extract_diseases_from_anamnesis(self, text: str | None, **kwargs: Any) -> PatientDiseaseContext:
        _ = text
        _ = kwargs
        return PatientDiseaseContext(entries=[])


def _build_payload() -> PatientData:
    return PatientData(
        name="Patient",
        visit_date=date(2025, 1, 20),
        anamnesis="ALT rose and then declined.",
        drugs="Drug A 50 mg",
        laboratory_analysis="Lab 2025-01-10: ALT 300 U/L (ULN 40), ALP 130 U/L (ULN 120)",
    )


def test_session_persists_rucam_bundle_and_per_drug_rucam(monkeypatch) -> None:
    fake_serializer = FakeSerializer()
    endpoint = session_module.endpoint
    endpoint.serializer = fake_serializer
    endpoint.drugs_parser = FakeDrugsParser()
    endpoint.disease_extractor = FakeDiseaseExtractor()
    endpoint.lab_extractor = FakeLabExtractor(fail=False)
    endpoint.rucam_estimator = FakeRucamEstimator(fail=False)
    monkeypatch.setattr(session_module, "input_preparator", FakeInputPreparator())
    monkeypatch.setattr(session_module, "HepatoxConsultation", FakeHepatoxConsultation)

    result = asyncio.run(endpoint.process_single_patient(_build_payload()))

    assert "rucam_assessments" in result
    assert result["rucam_assessments"]
    assert "lab_timeline" in result
    assert "onset_context" in result
    assert result["matched_drugs"][0]["rucam"] is not None


def test_session_lab_or_rucam_failure_degrades_to_warnings(monkeypatch) -> None:
    fake_serializer = FakeSerializer()
    endpoint = session_module.endpoint
    endpoint.serializer = fake_serializer
    endpoint.drugs_parser = FakeDrugsParser()
    endpoint.disease_extractor = FakeDiseaseExtractor()
    endpoint.lab_extractor = FakeLabExtractor(fail=True)
    endpoint.rucam_estimator = FakeRucamEstimator(fail=True)
    monkeypatch.setattr(session_module, "input_preparator", FakeInputPreparator())
    monkeypatch.setattr(session_module, "HepatoxConsultation", FakeHepatoxConsultation)

    try:
        asyncio.run(endpoint.process_single_patient(_build_payload()))
        assert False, "Expected validation error for missing hepatotoxicity inputs."
    except ClinicalPipelineValidationError as exc:
        assert any(issue.code == "missing_hepatotoxicity_inputs" for issue in exc.issues)
