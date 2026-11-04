# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest
from domain.clinical.entities import PatientData
from pydantic import ValidationError


###############################################################################
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024-02-29", date(2024, 2, 29)),
        ("2025-03-10T23:30:00", date(2025, 3, 10)),
        ({"day": 5, "month": "3", "year": 2025}, date(2025, 3, 5)),
        (date.today().isoformat(), date.today()),
        ("  ", None),
        ({"day": "", "month": "", "year": ""}, None),
        (None, None),
    ],
)
def test_valid_visit_dates_are_preserved(raw: Any, expected: date | None) -> None:
    assert PatientData(visit_date=raw).visit_date == expected


###############################################################################
@pytest.mark.parametrize(
    "raw",
    [
        "2025-02-29",
        "not-a-date",
        "2025-13-45",
        {"day": 31, "month": 2, "year": 2025},
        {"day": "x", "month": 1, "year": 2025},
        12345,
    ],
)
def test_malformed_visit_dates_are_rejected(raw: Any) -> None:
    with pytest.raises(ValidationError, match="not a valid date"):
        PatientData(visit_date=raw)


###############################################################################
def test_future_visit_date_is_rejected_instead_of_replaced() -> None:
    tomorrow = date.today() + timedelta(days=1)
    with pytest.raises(ValidationError, match="must not be in the future"):
        PatientData(visit_date=tomorrow)
