from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from DILIGENT.server.domain.clinical import DrugEntry, PatientDrugs
from DILIGENT.server.services.clinical.parser import DrugsParser


class FakeStructuredClient:
    def __init__(self, responses: Sequence[PatientDrugs]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    async def llm_structured_call(self, **kwargs: Any) -> PatientDrugs:
        self.call_count += 1
        schema = kwargs.get("schema", PatientDrugs)
        if self.responses:
            return self.responses.pop(0)
        return schema(entries=[])


def test_extract_drugs_from_anamnesis_sets_historical_tags() -> None:
    fake_client = FakeStructuredClient(
        [
            PatientDrugs(
                entries=[
                    DrugEntry(name="Aspirin", dosage="100 mg"),
                    DrugEntry(name="Legacy Name\nmultiline", dosage="200 mg"),
                ]
            )
        ]
    )
    parser = DrugsParser(client=fake_client)

    parsed = asyncio.run(
        parser.extract_drugs_from_anamnesis("Patient previously used aspirin.")
    )

    assert len(parsed.entries) == 1
    entry = parsed.entries[0]
    assert entry.name == "Aspirin"
    assert entry.source == "anamnesis"
    assert entry.historical_flag is True
    assert entry.temporal_classification == "temporal_uncertain"


def test_extract_drugs_from_anamnesis_empty_result_is_allowed() -> None:
    parser = DrugsParser(client=FakeStructuredClient([PatientDrugs(entries=[])]))

    parsed = asyncio.run(
        parser.extract_drugs_from_anamnesis("No pharmacological therapy in history.")
    )

    assert parsed.entries == []


def test_extract_drugs_from_anamnesis_rule_fallback_recovers_drug_lines() -> None:
    parser = DrugsParser(client=FakeStructuredClient([PatientDrugs(entries=[])]))
    anamnesis = "Xanax 0,5 mg cpr sospesa dal 10/02/2024"

    parsed = asyncio.run(parser.extract_drugs_from_anamnesis(anamnesis))

    assert len(parsed.entries) == 1
    entry = parsed.entries[0]
    assert entry.name == "Xanax"
    assert entry.dosage is not None
    assert entry.dosage.startswith("0,5 mg")
    assert entry.suspension_status is True
    assert entry.source == "anamnesis"
    assert entry.historical_flag is True


def test_extract_drugs_from_anamnesis_chunks_long_input() -> None:
    client = FakeStructuredClient(
        [
            PatientDrugs(entries=[DrugEntry(name="Aspirin")]),
            PatientDrugs(entries=[DrugEntry(name="Metformin")]),
        ]
    )
    parser = DrugsParser(client=client)
    long_text = "\n".join(
        [
            "Anamnesis line without medication details."
            for _ in range(80)
        ]
    )

    parsed = asyncio.run(parser.extract_drugs_from_anamnesis(long_text))

    assert client.call_count >= 2
    assert [entry.name for entry in parsed.entries] == ["Aspirin", "Metformin"]
