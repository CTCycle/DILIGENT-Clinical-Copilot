from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from domain.clinical import DrugEntry, PatientDrugs
from services.clinical.parser import DrugsParser

class RecordingStructuredClient:
    def __init__(self, response: PatientDrugs) -> None:
        self.response = response
        self.user_prompts: list[str] = []
        self.call_count = 0

    async def llm_structured_call(self, **kwargs: Any) -> PatientDrugs:
        self.call_count += 1
        self.user_prompts.append(str(kwargs.get("user_prompt", "")))
        return self.response
    

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


class AlwaysFailingStructuredClient:
    async def llm_structured_call(self, **kwargs: Any) -> PatientDrugs:
        raise RuntimeError("simulated llm failure")


def test_extract_drugs_from_anamnesis_sends_full_context_to_llm() -> None:
    client = RecordingStructuredClient(
        PatientDrugs(entries=[DrugEntry(name="Co-Amoxicillina")])
    )
    parser = DrugsParser(client=client)
    anamnesis = """
    Paziente di 68 anni nota per:
    Dal 17.03.2023 al 30.06.2023 Carboplatino e Paclitaxel, con aggiunta di Bevacizumab dal
    secondo ciclo.
    Dal 28.12.2023 al 17.05.2024: Chemioterapia di seconda linea con Carboplatino e Caelyx,
    eseguiti 6 cicli.
    Dal 06.07.2024 al 16.07.2024: Terapia con Olaparib, sospeso per PD in sede peritoneale.
    Dal 10.01.2025 Protocollo con Gemcitabina + Bevacizumab.
    Nozione di terapia antibiotica con Co-Amoxicillina 1-0-1 dal 18.02 prescritta per 5 giorni.
    """

    parsed = asyncio.run(parser.extract_drugs_from_anamnesis(anamnesis))

    assert client.call_count >= 1
    combined_prompts = "\n".join(client.user_prompts)
    assert "Carboplatino e Paclitaxel" in combined_prompts
    assert "secondo ciclo." in combined_prompts
    assert "Co-Amoxicillina 1-0-1" in combined_prompts
    assert [entry.name for entry in parsed.entries if entry.name == "Co-Amoxicillina"]


def test_extract_drugs_from_anamnesis_sets_historical_tags() -> None:
    fake_client = FakeStructuredClient(
        [
            PatientDrugs(
                entries=[
                    DrugEntry(name="Aspirin", dosage="100 mg"),
                    DrugEntry(name="Historical Name\nmultiline", dosage="200 mg"),
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
        ["Anamnesis line without medication details." for _ in range(80)]
    )

    parsed = asyncio.run(parser.extract_drugs_from_anamnesis(long_text))

    assert client.call_count >= 2
    assert [entry.name for entry in parsed.entries] == ["Aspirin", "Metformin"]


def test_extract_drugs_from_anamnesis_filters_non_drug_fragments() -> None:
    fake_client = FakeStructuredClient(
        [
            PatientDrugs(
                entries=[
                    DrugEntry(name="Pemetrexed"),
                    DrugEntry(name="Benziodiazepine"),
                    DrugEntry(name="In riserva"),
                    DrugEntry(name="il lunedi"),
                    DrugEntry(name="Paziente femmina"),
                    DrugEntry(name="Dopo"),
                    DrugEntry(name="Dal"),
                    DrugEntry(name="entrambi e il"),
                    DrugEntry(
                        name="Nozione di terapia antibiotica con Co-Amoxicillina"
                    ),
                    DrugEntry(name="rialzo a"),
                    DrugEntry(name="ulteriore ciclo (originariamente previsto il"),
                ]
            )
        ]
    )
    parser = DrugsParser(client=fake_client)

    parsed = asyncio.run(
        parser.extract_drugs_from_anamnesis("History includes oncology treatment.")
    )

    assert [entry.name for entry in parsed.entries] == [
        "Pemetrexed",
        "Benziodiazepine",
        "Co-Amoxicillina",
    ]


def test_extract_drugs_from_anamnesis_llm_failure_uses_rule_fallback() -> None:
    parser = DrugsParser(client=AlwaysFailingStructuredClient())
    anamnesis = "Xanax 0,5 mg cpr sospesa dal 10/02/2024"

    parsed = asyncio.run(parser.extract_drugs_from_anamnesis(anamnesis))

    assert [entry.name for entry in parsed.entries] == ["Xanax"]
    assert parsed.entries[0].historical_flag is True
    assert parsed.entries[0].source == "anamnesis"
