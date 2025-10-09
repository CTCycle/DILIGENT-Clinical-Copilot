import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from DILIGENT.app.utils.services.parser import DrugsParser


def test_parse_drug_start_date_after_schedule():
    parser = DrugsParser()
    text = "Metformina 850 mg 1-0-1 dal 12/05/2023"
    result = parser.parse_drug_list(text)
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.therapy_start_status is True
    assert entry.therapy_start_date == "2023-05-12"


def test_parse_drug_start_date_before_schedule():
    parser = DrugsParser()
    text = "Dal 01/02/2024 Metformina 1-1-1"
    result = parser.parse_drug_list(text)
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.therapy_start_status is True
    assert entry.therapy_start_date == "2024-02-01"


def test_suspension_does_not_trigger_start_detection():
    parser = DrugsParser()
    text = "Amoxicillina 500 mg 1-0-1 sospeso dal 03/03/2023"
    result = parser.parse_drug_list(text)
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.suspension_date == "2023-03-03"
    assert entry.therapy_start_status is None
    assert entry.therapy_start_date is None
