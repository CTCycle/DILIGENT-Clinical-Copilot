# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import pytest
from pydantic import ValidationError

from domain.clinical.drafts import ClinicalDraftForm, ClinicalDraftWrite


def test_draft_contract_preserves_incomplete_and_multiline_input() -> None:
    request = ClinicalDraftWrite.model_validate(
        {
            "expected_revision": 0,
            "expected_draft_key": None,
            "form": {
                "patient_name": "  Synthetic patient  ",
                "visit_date": None,
                "clinical_input": "  First line\n\n Second line  ",
                "use_rag": True,
            },
        }
    )

    assert request.form.patient_name == "  Synthetic patient  "
    assert request.form.clinical_input == "  First line\n\n Second line  "
    assert request.form.visit_date is None


@pytest.mark.parametrize(
    "form",
    [
        {"patient_name": "Synthetic", "unexpected": "value"},
        {"visit_date": "not-a-date"},
        {"patient_image_data_url": "data:image/png;base64,synthetic"},
        {"clinical_input": "x" * 100_001},
        {"patient_name": "x" * 201},
    ],
)
def test_draft_form_rejects_invalid_or_unapproved_fields(form: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ClinicalDraftForm.model_validate(form)


def test_draft_write_rejects_extra_fields_and_negative_revision() -> None:
    with pytest.raises(ValidationError):
        ClinicalDraftWrite.model_validate(
            {
                "expected_revision": 0,
                "expected_draft_key": None,
                "form": {},
                "patient_image": "not accepted",
            }
        )
    with pytest.raises(ValidationError):
        ClinicalDraftWrite.model_validate(
            {"expected_revision": -1, "expected_draft_key": None, "form": {}}
        )
