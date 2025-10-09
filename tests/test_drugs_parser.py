import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from DILIGENT.app.api.schemas.clinical import DrugEntry, PatientDrugs
from DILIGENT.app.utils.services.parser import DrugsParser


class FakeLLMClient:
    def __init__(self, parser: DrugsParser) -> None:
        self.parser = parser

    async def llm_structured_call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: type[PatientDrugs],
        temperature: float,
        use_json_mode: bool,
        max_repair_attempts: int,
    ) -> PatientDrugs:
        _ = (model, system_prompt, temperature, use_json_mode, max_repair_attempts)
        entries = []
        for line in user_prompt.split("\n"):
            parsed = self.parser._parse_line(line)  # noqa: SLF001
            if parsed is not None:
                entries.append(parsed)
        return schema(entries=entries)

    async def close(self) -> None:
        return None


def build_parser() -> DrugsParser:
    parser = DrugsParser()
    parser.client = FakeLLMClient(parser)
    parser.client_provider = "injected"
    return parser


def test_parse_drug_start_date_after_schedule():
    parser = build_parser()
    text = "Metformina 850 mg 1-0-1 dal 12/05/2023"
    result = parser.parse_drug_list(text)
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.therapy_start_status is True
    assert entry.therapy_start_date == "2023-05-12"


def test_parse_drug_start_date_before_schedule():
    parser = build_parser()
    text = "Dal 01/02/2024 Metformina 1-1-1"
    result = parser.parse_drug_list(text)
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.therapy_start_status is True
    assert entry.therapy_start_date == "2024-02-01"


def test_suspension_does_not_trigger_start_detection():
    parser = build_parser()
    text = "Amoxicillina 500 mg 1-0-1 sospeso dal 03/03/2023"
    result = parser.parse_drug_list(text)
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.suspension_date == "2023-03-03"
    assert entry.therapy_start_status is None
    assert entry.therapy_start_date is None


def test_drug_entry_drops_partial_daytime_schedule():
    entry = DrugEntry(
        name="Test",
        dosage=None,
        administration_mode=None,
        daytime_administration=[1.0],
    )
    assert entry.daytime_administration == []


def test_drug_entry_trims_long_daytime_schedule():
    entry = DrugEntry(
        name="Test",
        dosage=None,
        administration_mode=None,
        daytime_administration=[1.0, 0.0, 2.0, 1.0, 3.0],
    )
    assert entry.daytime_administration == [1.0, 0.0, 2.0, 1.0]


def test_multiline_suspension_and_start_metadata():
    parser = build_parser()
    text = (
        "Seresta 15 mg cpr [cpr] 1-1-0-2 per os\n"
        "dal 07.08\n"
        "Quviviq 50 mg cpr [cpr] 0-0-0-1 per os\n"
        "sospeso dal 04.08"
    )
    result = parser.parse_drug_list(text)
    assert len(result.entries) == 2
    seresta, quviviq = result.entries
    assert seresta.therapy_start_status is True
    assert seresta.therapy_start_date == "07.08"
    assert quviviq.suspension_status is True
    assert quviviq.suspension_date == "04.08"
