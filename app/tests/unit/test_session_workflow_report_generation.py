from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace
from typing import Any

from domain.clinical.entities import (
    ClinicalLabEntry,
    DrugEntry,
    HepatotoxicityPatternScore,
    PatientData,
    PatientDrugs,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
)
from services.session.document_normalizer import DocumentNormalizer
from services.session.session_workflow import process_single_patient_workflow


class FakePatternAnalyzer:
    def stringify_scores(self, pattern_score: HepatotoxicityPatternScore) -> dict[str, str]:
        return {"r_score": f"{pattern_score.r_score:.2f}" if pattern_score.r_score else "Not available"}


class FakeDrugsParser:
    model = "test-parser"

    def clean_text(self, text: str) -> str:
        return text


class FakeSerializer:
    def save_clinical_session(self, payload: dict[str, Any]) -> int | None:
        self.saved_payload = payload
        return None


class FakeClinicalService:
    JOB_TYPE = "clinical"

    def __init__(self) -> None:
        self.drugs_parser = FakeDrugsParser()
        self.pattern_analyzer = FakePatternAnalyzer()
        self.serializer = FakeSerializer()

    def run_stop_check(self, stop_check: Any) -> None:
        if stop_check is not None:
            stop_check()

    def emit_progress(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs

    def build_validation_bundle_for_payload(self, payload: PatientData) -> object:
        _ = payload
        return object()

    def ensure_submission_requirements(self, payload: PatientData) -> None:
        _ = payload

    async def extract_therapy_drugs(self, **kwargs: Any) -> PatientDrugs:
        _ = kwargs
        return PatientDrugs(
            entries=[DrugEntry(name="Paracetamolo", source="therapy")]
        )

    async def extract_anamnesis_drugs(self, **kwargs: Any) -> PatientDrugs:
        _ = kwargs
        return PatientDrugs(entries=[])

    async def extract_disease_context(self, **kwargs: Any) -> SimpleNamespace:
        _ = kwargs
        return SimpleNamespace(entries=[])

    async def extract_lab_timeline(self, **kwargs: Any) -> tuple[PatientLabTimeline, None]:
        _ = kwargs
        return (
            PatientLabTimeline(
                entries=[
                    ClinicalLabEntry(marker_name="ALT", value=100.0, unit="U/L"),
                    ClinicalLabEntry(marker_name="ALP", value=120.0, unit="U/L"),
                ]
            ),
            None,
        )

    def assess_pattern(self, **kwargs: Any) -> SimpleNamespace:
        _ = kwargs
        return SimpleNamespace(
            status="ok",
            score=HepatotoxicityPatternScore(
                classification="mixed",
                r_score=2.5,
            ),
        )

    def estimate_rucam(self, **kwargs: Any) -> PatientRucamAssessmentBundle:
        _ = kwargs
        return PatientRucamAssessmentBundle(entries=[])

    def build_structured_clinical_context(self, *args: Any, **kwargs: Any) -> str:
        _ = args, kwargs
        return "Contesto clinico strutturato."

    def build_rag_query(self, **kwargs: Any) -> dict[str, str]:
        _ = kwargs
        return {}

    async def run_livertox_lookup(self, **kwargs: Any) -> None:
        _ = kwargs
        return None

    def reestimate_rucam_with_livertox(
        self,
        *,
        rucam_bundle: PatientRucamAssessmentBundle,
        **kwargs: Any,
    ) -> PatientRucamAssessmentBundle:
        _ = kwargs
        return rucam_bundle

    async def run_consultation(self, **kwargs: Any) -> tuple[SimpleNamespace, str]:
        _ = kwargs
        return (
            SimpleNamespace(llm_model="test-clinical-model"),
            "Relazione narrativa con discussione farmacologica e sintesi finale.",
        )

    def _normalized_resolved_drug_map(self, prepared_inputs: Any) -> dict[str, Any]:
        _ = prepared_inputs
        return {}

    def _normalized_rucam_map(
        self,
        rucam_bundle: PatientRucamAssessmentBundle,
    ) -> dict[str, Any]:
        _ = rucam_bundle
        return {}

    def serialize_pipeline_issues(self, issues: list[Any]) -> list[dict[str, Any]]:
        serialized = []
        for issue in issues:
            if hasattr(issue, "model_dump"):
                serialized.append(issue.model_dump())
            else:
                serialized.append(dict(issue))
        return serialized


def test_workflow_keeps_narrative_report_and_stores_audit_report() -> None:
    payload = PatientData(
        name="Mario Rossi",
        visit_date=date(2025, 5, 20),
        anamnesis="Paziente con ittero.",
        drugs="Paracetamolo 1-0-0-0",
        laboratory_analysis="ALT 100 U/L, ALP 120 U/L.",
    )

    result = asyncio.run(
        process_single_patient_workflow(
            FakeClinicalService(),
            payload,
            normalized_document=DocumentNormalizer().normalize(
                "Paziente con ittero.\nParacetamolo 1-0-0-0\nALT 100 U/L, ALP 120 U/L."
            ),
            report_mode="faithful_only",
        ),
    )

    assert (
        result["final_report"]
        == "Relazione narrativa con discussione farmacologica e sintesi finale."
    )
    assert result["final_report"] in result["report"]
    assert "## Report Clinico" in result["pipeline_artifacts"]["generated_report"]
    assert "### Esposizione ai Farmaci" in result["pipeline_artifacts"]["generated_report"]
    assert result["pipeline_artifacts"]["generated_report"] != result["final_report"]
