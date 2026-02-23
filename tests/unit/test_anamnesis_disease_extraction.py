from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

from DILIGENT.server.entities.clinical import (
    DiseaseContextEntry,
    DrugEntry,
    HepatotoxicityPatternScore,
    PatientData,
    PatientDiseaseContext,
    PatientDrugs,
)
from DILIGENT.server.routes.session import ClinicalSessionEndpoint
from DILIGENT.server.services.clinical.disease import DiseaseExtractor


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


def test_extract_diseases_from_anamnesis_deduplicates_and_keeps_rich_entry() -> None:
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
    anamnesis = "\n".join(["Anamnesis line."] * 300)

    parsed = asyncio.run(extractor.extract_diseases_from_anamnesis(anamnesis))

    assert client.call_count >= 2
    assert len(parsed.entries) == 2
    steatosis = parsed.entries[0]
    assert steatosis.name.lower() == "steatosi epatica"
    assert steatosis.occurrence_time == "2021"
    assert steatosis.chronic is True
    assert steatosis.hepatic_related is True


def test_build_structured_clinical_context_includes_disease_timeline() -> None:
    payload = PatientData(
        name="Patient A",
        visit_date=date(2025, 4, 14),
        anamnesis="History of steatosis and hypertension.",
        drugs="Ursodeoxycholic acid 300 mg",
        alt="100",
        alt_max="50",
        alp="120",
        alp_max="100",
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

    context = ClinicalSessionEndpoint.build_structured_clinical_context(
        payload,
        therapy_drugs=therapy_drugs,
        anamnesis_drugs=anamnesis_drugs,
        disease_context=disease_context,
        pattern_score=pattern_score,
    )

    assert "# Structured Disease Timeline (from Anamnesis)" in context
    assert "Steatosis | occurrence: 2023 | chronic: yes | hepatic-related: yes" in context
    assert "# Visit Date Anchor" in context
    assert "2025-04-14" in context
