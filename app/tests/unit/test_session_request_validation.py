from __future__ import annotations

import pytest
from domain.clinical.entities import ClinicalSessionRequest
from fastapi import HTTPException
from services.session.request_validation import (
    validate_clinical_session_request,
)


def _words(count: int) -> str:
    return " ".join(f"w{i}" for i in range(count))


def _request(**overrides: object) -> ClinicalSessionRequest:
    payload = {
        "visit_date": "2025-01-01",
        "clinical_input": _words(60),
        "selected_model_providers": ["openai"],
    }
    payload.update(overrides)
    return ClinicalSessionRequest.model_validate(payload)


def test_missing_visit_date_fails() -> None:
    with pytest.raises(HTTPException):
        validate_clinical_session_request(_request(visit_date=None))


def test_empty_clinical_input_fails() -> None:
    with pytest.raises(HTTPException):
        validate_clinical_session_request(_request(clinical_input=""))


def test_fifty_nine_words_fails() -> None:
    with pytest.raises(HTTPException):
        validate_clinical_session_request(_request(clinical_input=_words(59)))


def test_sixty_words_pass() -> None:
    validate_clinical_session_request(_request(clinical_input=_words(60)))


def test_empty_provider_list_fails() -> None:
    with pytest.raises(HTTPException):
        validate_clinical_session_request(_request(selected_model_providers=[]))
