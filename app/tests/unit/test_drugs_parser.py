from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from domain.clinical import DrugEntry, PatientDrugs
from domain.clinical.extractor_contracts import LocalDrugEntryDraft, LocalPatientDrugs
from services.clinical.drug_blocks import isolate_drug_blocks
from services.clinical.parser import DrugsParser

###############################################################################
class RecordingCorpusStructuredClient:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.user_prompts: list[str] = []

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, **kwargs: Any) -> PatientDrugs:
        self.user_prompts.append(str(kwargs["user_prompt"]))
        return PatientDrugs(
            entries=[
                DrugEntry(
                    name="Cardiomed",
                    dosage="100 mg",
                    evidence="Cardiomed 100 mg cpr\n1-0-0-0",
                )
            ]
        )

###############################################################################
class RecordingLocalStructuredClient:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.schemas: list[type[Any]] = []

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, **kwargs: Any) -> LocalPatientDrugs:
        self.schemas.append(kwargs["schema"])
        return LocalPatientDrugs(
            entries=[
                LocalDrugEntryDraft(
                    name="Ciproflax",
                    dosage="500 mg",
                    evidence="Ciproflax 500 mg dal 24.01.",
                )
            ]
        )

###############################################################################
class MultilineStructuredClient:

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, **kwargs: Any) -> PatientDrugs:
        source = str(kwargs["user_prompt"])
        if "patient anamnesis" in source:
            return PatientDrugs(
                entries=[
                    DrugEntry(
                        name="Bactrim",
                        evidence="Bactrim: terapia iniziata il 04.02.2025",
                    ),
                    DrugEntry(name="interrotta il", evidence="interrotta il"),
                    DrugEntry(name="ricevuta il", evidence="ricevuta il"),
                    DrugEntry(name="termine", evidence="termine"),
                ]
            )
        return PatientDrugs(
            entries=[
                DrugEntry(name="Bactrim", evidence="Bactrim forte 800/160 mg"),
                DrugEntry(name="Huo Ma Ren", evidence="Huo Ma Ren 30"),
                DrugEntry(name="Hou", evidence="Hou Po 6"),
                DrugEntry(name="Hou Po", evidence="Hou Po 6"),
            ]
        )

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
class RecordingLocalSectionClient:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.schemas: list[type[Any]] = []

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, **kwargs: Any) -> LocalPatientDrugs:
        self.schemas.append(kwargs["schema"])
        return LocalPatientDrugs(
            entries=[
                LocalDrugEntryDraft(
                    name="Pembrolizumab",
                    evidence="Pembrolizumab",
                    current_status="past",
                )
            ]
        )

###############################################################################
class AlwaysFailingStructuredClient:

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, **kwargs: Any) -> PatientDrugs:
        raise RuntimeError("simulated llm failure")

###############################################################################
def test_therapy_extraction_uses_complete_multiline_corpus_once() -> None:
    client = RecordingCorpusStructuredClient()
    parser = DrugsParser(client=client)
    source = "Cardiomed 100 mg cpr\n1-0-0-0"

    parsed = asyncio.run(parser.extract_drugs_from_therapy(source))

    assert [entry.name for entry in parsed.entries] == ["Cardiomed"]
    assert len(client.user_prompts) == 1
    assert source in client.user_prompts[0]

###############################################################################
def test_multiline_corpus_rejects_truncated_and_status_fragments() -> None:
    parser = DrugsParser(client=MultilineStructuredClient())
    therapy = """
    Bactrim forte 800/160 mg cpr
    1-0-0-1 per os
    Huo Ma Ren 30
    Hou Po 6
    0-0-0-0
    """
    anamnesis = """
    Bactrim: terapia iniziata il 04.02.2025 e
    interrotta il 09.02.2025.
    Ultima dose ricevuta il 31.01.2025.
    Terapia a lungo termine.
    """

    therapy_result = asyncio.run(parser.extract_drugs_from_therapy(therapy))
    anamnesis_result = asyncio.run(parser.extract_drugs_from_anamnesis(anamnesis))

    therapy_names = [entry.name for entry in therapy_result.entries]
    anamnesis_names = [entry.name for entry in anamnesis_result.entries]
    assert "Bactrim" in therapy_names
    assert "Huo Ma Ren" in therapy_names
    assert "Hou Po" in therapy_names
    assert "Hou" not in therapy_names
    assert anamnesis_names == ["Bactrim"]

###############################################################################
def test_whole_section_extraction_uses_local_compact_schema_for_ollama() -> None:
    client = RecordingLocalStructuredClient()
    parser = DrugsParser(client=client)
    parser.forced_provider = "ollama"

    parsed = asyncio.run(
        parser.llm_extract_drugs_from_section(
            "Ciproflax 500 mg dal 24.01.",
            source="therapy",
            historical_flag=False,
        )
    )

    assert client.schemas == [LocalPatientDrugs]
    assert len(parsed.entries) == 1
    assert parsed.entries[0].name == "Ciproflax"
    assert parsed.entries[0].dosage == "500 mg"

###############################################################################
def test_extract_drugs_from_therapy_parses_schedule_route_and_dates() -> None:
    parser = DrugsParser(client=object())
    therapy_text = """
    Acetaminophen 500 mg 1 - 0 - 0 - 0 po started from 01/01/2024
    Ceftriaxone 1 g iv sospesa dal 03.01.2024
    """

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert len(parsed.entries) == 2

    first = parsed.entries[0]
    assert first.name == "Acetaminophen"
    assert first.dosage == "500 mg"
    assert first.route == "oral"
    assert first.administration_pattern == "1-0-0-0"
    assert first.daytime_administration == [1.0, 0.0, 0.0, 0.0]
    assert first.therapy_start_status is True
    assert first.therapy_start_date == "2024-01-01"
    assert first.temporal_classification == "temporal_known"
    assert first.source == "therapy"
    assert first.historical_flag is False

    second = parsed.entries[1]
    assert second.name == "Ceftriaxone"
    assert second.route == "iv"
    assert second.suspension_status is True
    assert second.suspension_date == "2024-01-03"
    assert second.temporal_classification == "temporal_known"

###############################################################################
def test_extract_drugs_from_therapy_missing_schedule_remains_parseable() -> None:
    parser = DrugsParser(client=object())
    therapy_text = "Pantoprazole 40 mg"

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert len(parsed.entries) == 1
    entry = parsed.entries[0]
    assert entry.name == "Pantoprazole"
    assert entry.dosage == "40 mg"
    assert entry.administration_pattern is None
    assert entry.daytime_administration == []
    assert entry.temporal_classification == "temporal_uncertain"
    assert entry.source == "therapy"
    assert entry.historical_flag is False

###############################################################################
def test_extract_drugs_from_therapy_supports_decimal_schedule_padding() -> None:
    parser = DrugsParser(client=object())
    therapy_text = "Prednisone 25 mg 0,5-0-0"

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert len(parsed.entries) == 1
    entry = parsed.entries[0]
    assert entry.name == "Prednisone"
    assert entry.administration_pattern == "0.5-0-0"
    assert entry.daytime_administration == [0.5, 0.0, 0.0, 0.0]
    assert entry.temporal_classification == "temporal_known"

###############################################################################
def test_extract_drugs_from_therapy_detects_ongoing_vs_suspended() -> None:
    parser = DrugsParser(client=object())
    therapy_text = """
    Ursodeoxycholic acid 250 mg sospesa dal 10/02/2024
    Metformin 500 mg started on 11-02-2024
    """

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert len(parsed.entries) == 2

    suspended = parsed.entries[0]
    ongoing = parsed.entries[1]

    assert suspended.suspension_status is True
    assert suspended.suspension_date == "2024-02-10"
    assert suspended.temporal_classification == "temporal_known"

    assert ongoing.suspension_status is None
    assert ongoing.therapy_start_status is True
    assert ongoing.therapy_start_date == "2024-02-11"
    assert ongoing.temporal_classification == "temporal_known"

###############################################################################
def test_extract_drugs_from_therapy_strips_temporal_tail_from_name() -> None:
    parser = DrugsParser(client=object())
    therapy_text = """
    Nivolumab EV, ultima somministrazione 12 giorni prima del picco enzimatico
    Ipilimumab EV, ultima somministrazione 12 giorni prima del picco enzimatico
    Trastuzumab deruxtecan EV (linea precedente, sospeso 6 settimane fa)
    """

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert [entry.name for entry in parsed.entries] == [
        "Nivolumab",
        "Ipilimumab",
        "Trastuzumab deruxtecan",
    ]
    assert [entry.route for entry in parsed.entries] == ["iv", "iv", "iv"]

###############################################################################
def test_extract_drugs_from_therapy_does_not_parse_iso_dates_as_schedule() -> None:
    parser = DrugsParser(client=object())
    therapy_text = (
        "Piperacillina/tazobactam 4.5 g EV q8h, iniziata 2026-02-10, sospesa 2026-02-16"
    )

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert len(parsed.entries) == 1
    entry = parsed.entries[0]
    assert entry.name == "Piperacillina/tazobactam"
    assert entry.dosage == "4.5 g EV q8h"
    assert entry.route == "iv"
    assert entry.administration_pattern is None
    assert entry.daytime_administration == []
    assert entry.therapy_start_status is True
    assert entry.therapy_start_date == "2026-02-10"
    assert entry.suspension_status is True
    assert entry.suspension_date == "2026-02-16"

###############################################################################
def test_fragment_guard_rejects_multiline_status_and_truncated_compound_names() -> None:
    parser = DrugsParser(client=object())

    assert (
        parser.normalize_entry(
            DrugEntry(name="interrotta il"),
            source="anamnesis",
            historical_flag=True,
        )
        is None
    )
    assert (
        parser.normalize_entry(
            DrugEntry(name="ricevuta il"),
            source="anamnesis",
            historical_flag=True,
        )
        is None
    )
    assert (
        parser.normalize_entry(
            DrugEntry(name="termine"),
            source="anamnesis",
            historical_flag=True,
        )
        is None
    )
    assert (
        parser.attach_source_grounding(
            DrugEntry(name="Hou"),
            source_text="Hou Po 6\nDa Huang 15",
            historical_flag=False,
            require_medication_syntax=False,
        )
        is None
    )

###############################################################################
def test_extract_drugs_from_therapy_skips_non_assumed_drug_line() -> None:
    parser = DrugsParser(client=object())
    therapy_text = """
    Esomeprazolo 40 mg PO 1 volta/die, terapia cronica (>12 mesi)
    Farmaci non assunti: paracetamolo ad alto dosaggio, antibiotici recenti
    """

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert [entry.name for entry in parsed.entries] == ["Esomeprazolo"]

###############################################################################
def test_extract_drugs_from_therapy_keeps_continuation_lines_with_drug_blocks() -> None:
    parser = DrugsParser(client=object())
    therapy_text = """
    ■Amlodipin axapharm cpr 5 mg 0-0-1-0 per os
    ■Prednison 20 mg cpr [cpr] 2-0-0-0 per os
     Dal 15.01.2025 40 mg (inizio terapia il 6-7 gennaio, alla dose di 60 mg/die) - Peso della paziente
    51.60 kg
    ■Diovan 80 mg cpr [cpr] 0-0-0-0 per os
     se PAS>o= 100 mmHg
    ■Domperidon axapharm lingual cpr orodisp 10 mg 0-0-0-0 per os
     In riserva: se nausea, vomito
    """

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert [entry.name for entry in parsed.entries] == [
        "Amlodipin axapharm",
        "Prednison",
        "Diovan",
        "Domperidon axapharm lingual cpr orodisp",
    ]

###############################################################################
def test_extract_drugs_from_therapy_uses_rules_before_llm_for_structured_blocks() -> (
    None
):
    parser = DrugsParser(client=object())
    therapy_text = """
    ■Fortecortin 4 mg cpr
    [cpr]
    1-0-0-0
    15.03 - 20.03

    ■De-Ursil 150 mg caps
    [caps]
    1-0-1-0
    per os
    dal 21.03

    ■Pantozol 40 mg cpr
    [cpr]
    1-0-0-0
    per os
    dal 06.02
    """

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert [entry.name for entry in parsed.entries] == [
        "Fortecortin",
        "De-Ursil",
        "Pantozol",
    ]

###############################################################################
def test_extract_drugs_from_therapy_splits_reserve_drugs_without_bullets() -> None:
    parser = DrugsParser(client=object())
    therapy_text = """
    Imodium lingual 2 mg cpr orodisp
    [cpr]
    0-0-0-0
    per os
    In riserva:

    Dafalgan 1 g cpr
    [cpr]
    0-0-0-0
    In riserva:

    Rivotril 2,5mg/ml gtt orali 10 ml
    [mg]
    0-0-0-0
    In riserva:
    """

    parsed = asyncio.run(parser.extract_drugs_from_therapy(therapy_text))

    assert [entry.name for entry in parsed.entries] == [
        "Imodium lingual",
        "Dafalgan",
        "Rivotril",
    ]

###############################################################################
def test_extract_drugs_from_therapy_empty_input_is_safe() -> None:
    parser = DrugsParser(client=object())

    parsed = asyncio.run(parser.extract_drugs_from_therapy(""))

    assert parsed.entries == []

###############################################################################
def test_normalize_entry_filters_non_drug_fragments() -> None:
    parser = DrugsParser(client=object())

    assert (
        parser.normalize_entry(
            DrugEntry(name="In riserva"),
            source="therapy",
            historical_flag=False,
        )
        is None
    )
    assert (
        parser.normalize_entry(
            DrugEntry(name="Paziente femmina"),
            source="anamnesis",
            historical_flag=True,
        )
        is None
    )
    assert (
        parser.normalize_entry(
            DrugEntry(name="Dopo"),
            source="anamnesis",
            historical_flag=True,
        )
        is None
    )
    assert (
        parser.normalize_entry(
            DrugEntry(name="il lunedi"),
            source="therapy",
            historical_flag=False,
        )
        is None
    )
    assert (
        parser.normalize_entry(
            DrugEntry(name="ulteriore ciclo (originariamente previsto il"),
            source="therapy",
            historical_flag=False,
        )
        is None
    )
    kept = parser.normalize_entry(
        DrugEntry(name="Pemetrexed"),
        source="therapy",
        historical_flag=False,
    )
    assert kept is not None
    assert kept.name == "Pemetrexed"

###############################################################################
def test_post_process_llm_entry_splits_dosage_from_temporal_details() -> None:
    parser = DrugsParser(client=object())
    raw_line = (
        "Boswellia serrata estratto secco 1 cps BID, iniziata circa 6 settimane "
        "prima dell'ittero, sospesa alla comparsa sintomi"
    )
    entry = DrugEntry(
        name="Boswellia serrata estratto secco",
        dosage=(
            "1 cps BID, iniziata circa 6 settimane prima dell'ittero, "
            "sospesa alla comparsa sintomi"
        ),
    )

    parsed = parser.post_process_llm_entry(
        entry,
        raw_line=raw_line,
        source="therapy",
        historical_flag=False,
    )

    assert parsed is not None
    assert parsed.dosage == "1 cps BID"
    assert parsed.therapy_start_status is True
    assert parsed.therapy_start_date == "circa 6 settimane prima dell'ittero"
    assert parsed.suspension_status is True
    assert parsed.suspension_date == "alla comparsa sintomi"
    assert parsed.temporal_classification == "temporal_known"


###############################################################################
# ── Anamnesis-specific extraction tests (from test_anamnesis_drug_extraction.py) ─

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

###############################################################################
def test_extract_drugs_from_anamnesis_uses_local_compact_schema_for_ollama() -> None:
    client = RecordingLocalSectionClient()
    parser = DrugsParser(client=client)
    parser.forced_provider = "ollama"

    parsed = asyncio.run(
        parser.extract_drugs_from_anamnesis(
            "Terapia precedente con Pembrolizumab sospesa successivamente."
        )
    )

    assert LocalPatientDrugs in client.schemas
    assert len(parsed.entries) == 1
    entry = parsed.entries[0]
    assert entry.name == "Pembrolizumab"
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
            "History includes oncology treatment. Rescue therapy with Rescuecin."
        )
    )

    assert [entry.name for entry in parsed.entries] == [
        "Rescuecin",
    ]

###############################################################################
def test_extract_drugs_from_anamnesis_rejects_grounded_non_medication_entities() -> (
    None
):
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

    parsed = asyncio.run(
        parser.extract_drugs_from_therapy("Cardiomed 100 mg cpr\n1-0-0-0")
    )

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

    parsed = asyncio.run(
        parser.extract_drugs_from_therapy("Cardiomed 100 mg cpr\n1-0-0-0")
    )

    assert [entry.name for entry in parsed.entries] == ["Cardiomed"]


###############################################################################
# ── Drug-block isolation tests (from test_drug_block_isolation.py) ────────────

###############################################################################
def test_bullet_list_blocks() -> None:
    text = "- Esomeprazolo 20 mg\n- Boswellia serrata 1 cps"
    blocks = isolate_drug_blocks(text)
    assert len(blocks) == 2

###############################################################################
def test_wrapped_bullet_continuation_attached() -> None:
    text = "- Esomeprazolo 20 mg\n  al mattino\n- Bromelina 1 cps"
    blocks = isolate_drug_blocks(text)
    assert "al mattino" in blocks[0].text

###############################################################################
def test_free_prose_returns_single_block() -> None:
    text = "Paziente in terapia cronica senza dettagli posologici specifici."
    blocks = isolate_drug_blocks(text)
    assert len(blocks) == 1

###############################################################################
def test_sentence_style_therapy_list_splits_into_blocks() -> None:
    text = (
        "Bactrim forte started 2024-01-01. "
        "Nitrofurantoin (Furadantin retard) started 2024-01-02. "
        "Ceftriaxone started 2024-01-03."
    )
    blocks = isolate_drug_blocks(text)
    assert [block.text for block in blocks] == [
        "Bactrim forte started 2024-01-01.",
        "Nitrofurantoin (Furadantin retard) started 2024-01-02.",
        "Ceftriaxone started 2024-01-03.",
    ]

###############################################################################
def test_overlong_block_is_truncated_at_sentence_boundary() -> None:
    text = (
        "Nitrofurantoin 100 mg twice daily. "
        "Paziente nota per abuso di etile con follow-up clinico."
    )
    blocks = isolate_drug_blocks(text)
    assert [block.text for block in blocks] == ["Nitrofurantoin 100 mg twice daily"]


###############################################################################
# ── Drug traceability tests (from test_anamnesis_extraction_traceability.py) ──

###############################################################################
def test_drug_llm_post_processing_downgrades_ungrounded_evidence() -> None:
    parser = DrugsParser()
    result = parser.post_process_llm_entry(
        DrugEntry(name="ImaginaryDrug", evidence="not in source"),
        raw_line="Patient denies medication use.",
        source="anamnesis",
        historical_flag=True,
    )

    assert result is not None
    assert result.confidence == "low"
    assert result.attribution == "unclear"


###############################################################################
# ── Deterministic anamnesis drug extraction tests (from test_deterministic_anamnesis_extraction.py) ──

###############################################################################
def test_deterministic_anamnesis_regimen_extraction_captures_oncology_history() -> None:
    parser = DrugsParser(client=object())
    text = (
        "Dal 17.03.2023 al 30.06.2023 Carboplatino e Paclitaxel, con aggiunta di Bevacizumab dal secondo ciclo.\n"
        "Dal 28.12.2023 al 17.05.2024: Chemioterapia di seconda linea con Carboplatino e Caelyx, eseguiti 6 cicli.\n"
        "Dal 06.07.2024 al 16.07.2024: Terapia con Olaparib, sospeso per PD in sede peritoneale.\n"
        "Dal 10.01.2025 Protocollo con Gemcitabina + Bevacizumab, ultima somministrazione il 27.02."
    )

    result = parser.extract_drugs_from_anamnesis_deterministic(text)
    names = [entry.name for entry in result.entries]

    assert "Carboplatino" in names
    assert "Paclitaxel" in names
    assert "Bevacizumab" in names
    assert "Caelyx" in names
    assert "Olaparib" in names
    assert "Gemcitabina" in names
    assert result.regimen_lines

###############################################################################
def test_deterministic_anamnesis_ignores_iso_date_in_symptom_sentence() -> None:
    parser = DrugsParser(client=object())

    result = parser.extract_drugs_from_anamnesis_deterministic(
        "Adult with fatigue and mild nausea beginning 2026-07-24. "
        "No known chronic liver disease."
    )

    assert result.entries == []
