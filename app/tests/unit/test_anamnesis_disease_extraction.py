from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace
from typing import Any

from domain.clinical import (
    DiseaseContextEntry,
    DrugEntry,
    HepatotoxicityPatternScore,
    LiverInjuryOnsetContext,
    PatientData,
    PatientDiseaseContext,
    PatientDrugs,
    PatientLabTimeline,
)
from services.clinical.disease import DiseaseExtractor
from services.session.session_service import ClinicalSessionService


class FakeDiseaseClient:
    def __init__(self, responses: list[PatientDiseaseContext]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    async def llm_structured_call(self, **kwargs: Any) -> PatientDiseaseContext:
        self.call_count += 1
        schema = kwargs.get("schema", PatientDiseaseContext)
        if self.responses:
            return self.responses.pop(0)
        return schema(entries=[])


class FlakyDiseaseClient:
    def __init__(self, *, failures_before_success: int) -> None:
        self.failures_before_success = max(failures_before_success, 0)
        self.call_count = 0

    async def llm_structured_call(self, **kwargs: Any) -> PatientDiseaseContext:
        self.call_count += 1
        if self.call_count <= self.failures_before_success:
            raise RuntimeError(
                "HTTP 429: Please try again in 0.01s. rate_limit_exceeded"
            )
        return PatientDiseaseContext(
            entries=[
                DiseaseContextEntry(
                    name="Steatosi epatica",
                    chronic=True,
                    hepatic_related=True,
                )
            ]
        )


def test_extract_diseases_from_anamnesis_deduplicates_and_keeps_rich_entry(
    monkeypatch,
) -> None:
    client = FakeDiseaseClient(
        [
            PatientDiseaseContext(
                entries=[
                    DiseaseContextEntry(
                        name="Steatosi epatica",
                        chronic=None,
                        hepatic_related=None,
                    )
                ]
            ),
            PatientDiseaseContext(
                entries=[
                    DiseaseContextEntry(
                        name="steatosi epatica",
                        occurrence_time="2021",
                        timeline="diagnosed in 2021, persistent thereafter",
                        severity="moderate",
                        diagnosis_status="confirmed",
                        symptoms="fatigue",
                        clinical_context="chronic liver disease follow-up",
                        chronic=True,
                        hepatic_related=True,
                        evidence="Steatosi epatica cronica documentata dal 2021.",
                    ),
                    DiseaseContextEntry(
                        name="Ipertensione arteriosa",
                        chronic=True,
                        hepatic_related=False,
                    ),
                ]
            ),
        ]
    )
    extractor = DiseaseExtractor(client=client)
    monkeypatch.setattr(
        "services.clinical.disease.extract_deterministic_diseases",
        lambda _text: SimpleNamespace(
            context=PatientDiseaseContext(entries=[]),
            unresolved_lines=["Anamnesis line."] * 300,
        ),
    )
    anamnesis = "\n".join(["Anamnesis line."] * 300)

    parsed = asyncio.run(extractor.extract_diseases_from_anamnesis(anamnesis))

    assert client.call_count >= 2
    assert len(parsed.entries) == 2
    steatosis = parsed.entries[0]
    assert steatosis.name.lower() == "steatosi epatica"
    assert steatosis.occurrence_time == "2021"
    assert steatosis.timeline is not None
    assert steatosis.severity == "moderate"
    assert steatosis.diagnosis_status == "confirmed"
    assert steatosis.symptoms == "fatigue"
    assert steatosis.clinical_context == "chronic liver disease follow-up"
    assert steatosis.chronic is True
    assert steatosis.hepatic_related is True


def test_extract_diseases_from_anamnesis_retries_transient_failures(
    monkeypatch,
) -> None:
    client = FlakyDiseaseClient(failures_before_success=1)
    extractor = DiseaseExtractor(client=client)
    extractor.extraction_retry_attempts = 2
    monkeypatch.setattr(
        "services.clinical.disease.extract_deterministic_diseases",
        lambda _text: SimpleNamespace(
            context=PatientDiseaseContext(entries=[]),
            unresolved_lines=["Steatosi epatica cronica."],
        ),
    )

    parsed = asyncio.run(
        extractor.extract_diseases_from_anamnesis("Steatosi epatica cronica.")
    )

    assert client.call_count == 2
    assert [entry.name for entry in parsed.entries] == ["Steatosi epatica"]


def test_build_structured_clinical_context_includes_disease_timeline() -> None:
    payload = PatientData(
        name="Patient A",
        visit_date=date(2025, 4, 14),
        anamnesis="History of steatosis and hypertension.",
        drugs="Ursodeoxycholic acid 300 mg",
        laboratory_analysis="ALT 100 U/L (ULN 50) on 2025-04-14; ALP 120 U/L (ULN 100)",
    )
    therapy_drugs = PatientDrugs(entries=[DrugEntry(name="Ursodeoxycholic acid")])
    anamnesis_drugs = PatientDrugs(entries=[DrugEntry(name="Metformin")])
    disease_context = PatientDiseaseContext(
        entries=[
            DiseaseContextEntry(
                name="Steatosis",
                occurrence_time="2023",
                chronic=True,
                hepatic_related=True,
                evidence="Known hepatic steatosis since 2023.",
            )
        ]
    )
    pattern_score = HepatotoxicityPatternScore(
        alt_multiple=2.0,
        alp_multiple=1.2,
        r_score=1.67,
        classification="mixed",
    )

    context = ClinicalSessionService.build_structured_clinical_context(
        payload,
        therapy_drugs=therapy_drugs,
        anamnesis_drugs=anamnesis_drugs,
        disease_context=disease_context,
        lab_timeline=PatientLabTimeline(entries=[]),
        onset_context=LiverInjuryOnsetContext(onset_basis="unknown"),
        pattern_score=pattern_score,
    )

    assert "# Disease Timeline" in context
    assert "Steatosis | time=2023 | chronic=yes | hepatic=yes" in context
    assert "# Visit Date" in context
    assert "# Onset Anchor" in context
    assert "# Pattern" in context
    assert "2025-04-14" in context
    assert "class=mixed | R=1.67" in context
