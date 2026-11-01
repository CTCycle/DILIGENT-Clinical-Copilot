# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""
E2E tests for the clinical session API endpoints.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from playwright.sync_api import APIRequestContext

###############################################################################
def repeated_words(count: int) -> str:
    return " ".join(f"clinicalword{i}" for i in range(count))

###############################################################################
def build_minimal_payload() -> dict:
    return {
        "name": "Test Patient",
        "visit_date": "2024-01-15",
        "clinical_input": (
            "## Anamnesis\nPatient with jaundice and fatigue.\n\n"
            "## Therapy\nZetamycin 10 mg 1-0-0-0\n\n"
            "## Laboratory Analysis\n"
            "Lab 2024-01-15: ALT 100 U/L (ULN 50), ALP 200 U/L (ULN 100), bilirubin 2.0 mg/dL. "
            f"{repeated_words(45)}"
        ),
        "evidence_sources": ["livertox", "dilirank"],
        "selected_model_providers": ["openai"],
    }

###############################################################################
@pytest.fixture
def draft_session_id(api_context: APIRequestContext) -> Iterator[int]:
    created = api_context.post(
        "/api/clinical/sessions",
        data={
            "patient_name": "Clinical API Draft",
            "visit_date": "2024-01-15",
            "clinical_input": "Draft text for launch validation.",
        },
    )
    assert created.status == 201, created.text()
    session_id = created.json()["session_id"]
    yield session_id
    api_context.delete(f"/api/inspection/sessions/{session_id}")

###############################################################################
def _assert_launch_rejected(
    api_context: APIRequestContext, session_id: int, payload: dict, message: str
) -> None:
    response = api_context.post(f"/api/clinical/sessions/{session_id}/launch", data=payload)
    assert response.status == 422, response.text()
    assert message in response.text()
    status = api_context.get(f"/api/clinical/sessions/{session_id}/status").json()
    assert status["status"] == "draft"

###############################################################################
def test_clinical_requires_sections(api_context: APIRequestContext, draft_session_id: int):
    _assert_launch_rejected(
        api_context, draft_session_id, {"name": "Test"}, "Clinical input is required."
    )

###############################################################################
def test_clinical_validate_input_returns_deterministic_diagnostics(
    api_context: APIRequestContext,
):
    response = api_context.post(
        "/api/clinical/validate-input", data=build_minimal_payload()
    )
    assert response.status == 200
    payload = response.json()
    assert "deterministic_diagnostics" in payload
    codes = {issue["code"] for issue in payload["blocking_issues"]}
    assert not codes & {"livertox_catalog_empty", "rxnav_catalog_empty"}
    assert payload["deterministic_diagnostics"]["therapy"]["drug_count"] >= 1

###############################################################################
def test_clinical_accepts_visit_date_dict(api_context: APIRequestContext):
    payload = build_minimal_payload()
    payload["visit_date"] = {"day": 15, "month": 1, "year": 2024}
    response = api_context.post("/api/clinical/validate-input", data=payload)
    assert response.status == 200
    assert "deterministic_diagnostics" in response.json()

###############################################################################
def test_clinical_rejects_empty_provider_selection(
    api_context: APIRequestContext, draft_session_id: int
):
    payload = build_minimal_payload()
    payload["selected_model_providers"] = []

    _assert_launch_rejected(
        api_context, draft_session_id, payload, "At least one model provider must be selected."
    )

###############################################################################
def test_clinical_requires_visit_date(api_context: APIRequestContext, draft_session_id: int):
    payload = build_minimal_payload()
    payload["visit_date"] = None

    _assert_launch_rejected(api_context, draft_session_id, payload, "Visit date is required.")
