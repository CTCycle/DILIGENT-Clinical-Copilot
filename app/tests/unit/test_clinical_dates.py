# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import date

from common.utils.clinical_dates import (
    case_date_order,
    has_ambiguous_date_order,
    parse_clinical_date,
)
from services.clinical.labs import ClinicalLabExtractor

###############################################################################
def test_unambiguous_token_sets_month_first_order_for_the_record() -> None:
    text = "05/06/2024: ALT 400 U/L. 05/20/2024: ALT 150 U/L."
    order = case_date_order(text)
    assert order == "month_first"
    assert parse_clinical_date("05/06/2024", order) == date(2024, 5, 6)
    assert not has_ambiguous_date_order(text)

###############################################################################
def test_unsettled_day_month_order_is_flagged() -> None:
    assert has_ambiguous_date_order("03/05/2024: ALT 300 U/L")
    assert not has_ambiguous_date_order("2024-03-05: ALT 300 U/L")

###############################################################################
def test_two_digit_years_and_yearless_dates_are_normalized() -> None:
    extractor = ClinicalLabExtractor(client=None)
    assert parse_clinical_date("15-03-24") == date(2024, 3, 15)
    assert (
        extractor.normalize_date_with_visit_year("12/12", date(2024, 1, 10))
        == "2023-12-12"
    )
