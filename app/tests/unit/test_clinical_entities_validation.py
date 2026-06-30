from __future__ import annotations

import pytest
from pydantic import ValidationError

from domain.clinical.entities import ClinicalSessionRequest, PatientData


def test_future_patient_visit_date_fails_validation() -> None:
    with pytest.raises(ValidationError, match="visit_date cannot be in the future"):
        PatientData.model_validate({"visit_date": "2999-01-01"})


def test_future_session_visit_date_fails_validation() -> None:
    with pytest.raises(ValidationError, match="visit_date cannot be in the future"):
        ClinicalSessionRequest.model_validate(
            {
                "visit_date": "2999-01-01",
                "clinical_input": "clinical input",
                "selected_model_providers": ["openai"],
            }
        )
