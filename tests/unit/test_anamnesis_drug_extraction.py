from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from DILIGENT.server.entities.clinical import DrugEntry, PatientDrugs
from DILIGENT.server.services.clinical.parser import DrugsParser


class FakeStructuredClient:
    def __init__(self, responses: Sequence[PatientDrugs]) -> None:
        self.responses = list(responses)

    async def llm_structured_call(self, **kwargs: Any) -> PatientDrugs:
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
