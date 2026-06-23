from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from domain.clinical import DrugEntry, PatientDrugs
from services.clinical.parser import DrugsParser

###############################################################################
class RecordingStructuredClient:

    # -------------------------------------------------------------------------
    def __init__(self, response: PatientDrugs) -> None:
        self.response = response
        self.user_prompts: list[str] = []
        self.call_count = 0

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, **kwargs: Any) -> PatientDrugs:
        self.call_count += 1
        self.user_prompts.append(str(kwargs.get("user_prompt", "")))
        return self.response

###############################################################################
class FakeStructuredClient:

    # -------------------------------------------------------------------------
    def __init__(self, responses: Sequence[PatientDrugs]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, **kwargs: Any) -> PatientDrugs:
        self.call_count += 1
        schema = kwargs.get("schema", PatientDrugs)
        if self.responses:
            return self.responses.pop(0)
        return schema(entries=[])

###############################################################################
class RecordingSequenceStructuredClient:

    # -------------------------------------------------------------------------
    def __init__(self, responses: Sequence[PatientDrugs]) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.user_prompts: list[str] = []

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, **kwargs: Any) -> PatientDrugs:
        self.call_count += 1
        self.user_prompts.append(str(kwargs.get("user_prompt", "")))
        schema = kwargs.get("schema", PatientDrugs)
        if self.responses:
            return self.responses.pop(0)
        return schema(entries=[])

###############################################################################
class AlwaysFailingStructuredClient:

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, **kwargs: Any) -> PatientDrugs:
        raise RuntimeError("simulated llm failure")

###############################################################################
def test_extract_drugs_from_anamnesis_sends_full_context_to_llm() -> None:
    client = RecordingStructuredClient(
        PatientDrugs(entries=[DrugEntry(name="Trialmycin")])
    )
    parser = DrugsParser(client=client)
    anamnesis = """
    Patient enrolled in a synthetic extraction trial:
    From 17.03.2023 to 30.06.2023 Alphaquel and Betamab, with addition of Gammacin from
    the second cycle.
    From 28.12.2023 to 17.05.2024: protocol with Deltatrex and Epsilix,
    eseguiti 6 cicli.
    From 06.07.2024 to 16.07.2024: therapy with Zetapar, suspended later.
    From 10.01.2025 protocol with Trialmycin + Omegavir.
    Rescue therapy with Trialmycin 1-0-1 from 18.02 for 5 days.
    """

    parsed = asyncio.run(parser.extract_drugs_from_anamnesis(anamnesis))

    assert client.call_count >= 1
    combined_prompts = "\n".join(client.user_prompts)
    assert "Alphaquel and Betamab" in combined_prompts
    assert "the second cycle." in combined_prompts
    assert "Trialmycin 1-0-1" in combined_prompts
    assert [entry.name for entry in parsed.entries if entry.name == "Trialmycin"]

###############################################################################
def test_extract_drugs_from_anamnesis_sends_complete_anamnesis_to_llm() -> None:
    client = RecordingStructuredClient(
        PatientDrugs(entries=[DrugEntry(name="Trialmycin")])
    )
    parser = DrugsParser(client=client)
    anamnesis = """
    Patient enrolled in a synthetic extraction trial:
    From 17.03.2023 to 30.06.2023 Alphaquel and Betamab, with addition of Gammacin from
    the second cycle.
    From 28.12.2023 to 17.05.2024: protocol with Deltatrex and Epsilix,
    eseguiti 6 cicli.
    From 06.07.2024 to 16.07.2024: therapy with Zetapar, suspended later.
    From 10.01.2025 protocol with Trialmycin + Omegavir.
    Rescue therapy with Trialmycin 1-0-1 from 18.02 for 5 days.
    """

    parsed = asyncio.run(parser.extract_drugs_from_anamnesis(anamnesis))

    assert client.call_count >= 1
    combined_prompt = "\n".join(client.user_prompts)
    assert "Alphaquel and Betamab" in combined_prompt
    assert "the second cycle." in combined_prompt
    assert "Trialmycin 1-0-1" in combined_prompt
    assert [entry.name for entry in parsed.entries if entry.name == "Trialmycin"]

###############################################################################
def test_extract_drugs_from_anamnesis_sets_historical_tags() -> None:
    fake_client = FakeStructuredClient(
        [
            PatientDrugs(
                entries=[
                    DrugEntry(name="Trialmed", dosage="100 mg", evidence="Trialmed"),
                    DrugEntry(name="Historical Name\nmultiline", dosage="200 mg"),
                ]
            )
        ]
    )
    parser = DrugsParser(client=fake_client)

    parsed = asyncio.run(
        parser.extract_drugs_from_anamnesis("Patient previously used Trialmed.")
    )

    assert len(parsed.entries) == 1
    entry = parsed.entries[0]
    assert entry.name == "Trialmed"
    assert entry.source == "anamnesis"
    assert entry.historical_flag is True
    assert entry.temporal_classification == "temporal_uncertain"

###############################################################################
def test_extract_drugs_from_anamnesis_empty_result_is_allowed() -> None:
    parser = DrugsParser(client=FakeStructuredClient([PatientDrugs(entries=[])]))

    parsed = asyncio.run(
        parser.extract_drugs_from_anamnesis("No pharmacological therapy in history.")
    )

    assert parsed.entries == []

###############################################################################
def test_extract_drugs_from_anamnesis_rule_fallback_recovers_drug_lines() -> None:
    parser = DrugsParser(client=FakeStructuredClient([PatientDrugs(entries=[])]))
    anamnesis = "Sleepmed 0,5 mg cpr sospesa dal 10/02/2024"

    parsed = asyncio.run(parser.extract_drugs_from_anamnesis(anamnesis))

    assert len(parsed.entries) == 1
    entry = parsed.entries[0]
    assert entry.name == "Sleepmed"
    assert entry.dosage is not None
    assert entry.dosage.startswith("0,5 mg")
    assert entry.suspension_status is True
    assert entry.source == "anamnesis"
    assert entry.historical_flag is True

###############################################################################
def test_extract_drugs_from_anamnesis_sends_long_input_as_single_chunk() -> None:
    client = FakeStructuredClient(
        [
            PatientDrugs(entries=[]),
        ]
    )
    parser = DrugsParser(client=client)
    long_text = "\n".join(
        ["Anamnesis line without medication details." for _ in range(80)]
    )

    parsed = asyncio.run(parser.extract_drugs_from_anamnesis(long_text))

    assert client.call_count == 2
    assert parsed.entries == []

###############################################################################
def test_extract_drugs_from_anamnesis_filters_non_drug_fragments() -> None:
    fake_client = FakeStructuredClient(
        [
            PatientDrugs(
                entries=[
                    DrugEntry(name="Trialmed"),
                    DrugEntry(name="Sedation class"),
                    DrugEntry(name="In riserva"),
                    DrugEntry(name="il lunedi"),
                    DrugEntry(name="Paziente femmina"),
                    DrugEntry(name="Dopo"),
                    DrugEntry(name="Dal"),
                    DrugEntry(name="entrambi e il"),
                    DrugEntry(name="Rescuecin", evidence="Rescuecin"),
                    DrugEntry(name="rialzo a"),
                    DrugEntry(name="ulteriore ciclo (originariamente previsto il"),
                ]
            )
        ]
    )
    parser = DrugsParser(client=fake_client)

    parsed = asyncio.run(
        parser.extract_drugs_from_anamnesis(
            "History includes oncology treatment. "
            "Rescue therapy with Rescuecin."
        )
    )

    assert [entry.name for entry in parsed.entries] == [
        "Rescuecin",
    ]

###############################################################################
def test_extract_drugs_from_anamnesis_rejects_grounded_non_medication_entities() -> None:
    client = RecordingSequenceStructuredClient(
        [
            PatientDrugs(
                entries=[
                    DrugEntry(
                        name="Tumorstage",
                        evidence="Tumorstage descriptor",
                    )
                ]
            ),
            PatientDrugs(entries=[]),
        ]
    )
    parser = DrugsParser(client=client)

    parsed = asyncio.run(
        parser.extract_drugs_from_anamnesis(
            "Tumorstage descriptor in synthetic follow-up note."
        )
    )

    assert client.call_count == 1
    assert parsed.entries == []

###############################################################################
def test_extract_drugs_from_anamnesis_accepts_medication_syntax_without_dose() -> None:
    client = FakeStructuredClient(
        [
            PatientDrugs(
                entries=[
                    DrugEntry(
                        name="Narramed",
                        evidence="Therapy with Narramed",
                    )
                ]
            )
        ]
    )
    parser = DrugsParser(client=client)

    parsed = asyncio.run(
        parser.extract_drugs_from_anamnesis(
            "In 2024 therapy with Narramed, later suspended."
        )
    )

    assert [entry.name for entry in parsed.entries] == ["Narramed"]
    assert parsed.entries[0].source_span is not None

###############################################################################
def test_extract_drugs_from_anamnesis_llm_failure_uses_rule_fallback() -> None:
    parser = DrugsParser(client=AlwaysFailingStructuredClient())
    anamnesis = "Sleepmed 0,5 mg cpr sospesa dal 10/02/2024"

    parsed = asyncio.run(parser.extract_drugs_from_anamnesis(anamnesis))

    assert [entry.name for entry in parsed.entries] == ["Sleepmed"]
    assert parsed.entries[0].historical_flag is True
    assert parsed.entries[0].source == "anamnesis"

###############################################################################
def test_extract_drugs_from_therapy_uses_llm_before_rule_fallback() -> None:
    client = FakeStructuredClient(
        [PatientDrugs(entries=[DrugEntry(name="Cardiomed", dosage="100 mg")])]
    )
    parser = DrugsParser(client=client)

    parsed = asyncio.run(parser.extract_drugs_from_therapy("Cardiomed 100 mg cpr\n1-0-0-0"))

    assert client.call_count == 1
    assert [entry.name for entry in parsed.entries] == ["Cardiomed"]
    assert parsed.entries[0].source == "therapy"
    assert parsed.entries[0].historical_flag is False

###############################################################################
def test_extract_drugs_retries_semantically_invalid_llm_output() -> None:
    client = RecordingSequenceStructuredClient(
        [
            PatientDrugs(entries=[DrugEntry(name="Artifact descriptor")]),
            PatientDrugs(entries=[DrugEntry(name="Retrymed")]),
        ]
    )
    parser = DrugsParser(client=client)

    parsed = asyncio.run(
        parser.extract_drugs_from_anamnesis(
            "Synthetic finding: non-medication descriptor.\n"
            "Rescue therapy with Retrymed 1000 mg x2."
        )
    )

    assert client.call_count == 2
    assert "Previous wrong output" in client.user_prompts[1]
    assert "Artifact descriptor" in client.user_prompts[1]
    assert [entry.name for entry in parsed.entries] == ["Retrymed"]

###############################################################################
def test_extract_drugs_from_therapy_falls_back_after_llm_failure() -> None:
    parser = DrugsParser(client=AlwaysFailingStructuredClient())

    parsed = asyncio.run(parser.extract_drugs_from_therapy("Cardiomed 100 mg cpr\n1-0-0-0"))

    assert [entry.name for entry in parsed.entries] == ["Cardiomed"]
