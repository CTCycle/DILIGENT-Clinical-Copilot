"""
E2E tests for the clinical session API endpoint.
"""
from __future__ import annotations

from playwright.sync_api import APIRequestContext


def build_minimal_payload() -> dict:
    return {
        "name": "Test Patient",
        "visit_date": "2024-01-15",
        "drugs": "Zetamycin 10 mg 1-0-0-0",
        "alt": "100",
        "alt_max": "50",
        "alp": "200",
        "alp_max": "100",
        "use_rag": False,
        "has_hepatic_diseases": False,
    }


def test_clinical_requires_sections(api_context: APIRequestContext):
    response = api_context.post("/clinical", data={"name": "Test"})
    assert response.status == 422

    payload = response.json()
    details = payload.get("detail", [])
    assert any(
        "clinical section" in str(item.get("msg", "")).lower()
        for item in details
    )


def test_clinical_rejects_blank_sections(api_context: APIRequestContext):
    response = api_context.post(
        "/clinical",
        data={"anamnesis": "  \n", "drugs": "   "},
    )
    assert response.status == 422


def test_clinical_accepts_minimal_payload(api_context: APIRequestContext):
    response = api_context.post("/clinical", data=build_minimal_payload())
    assert response.status == 202

    text = response.text()
    assert "# Clinical Visit Summary" in text
    assert "## Hepato-toxicity Pattern" in text
    assert "ALT multiple" in text
    assert "ALP multiple" in text
    assert "R-score" in text
    assert "Classification" in text
    assert "Zetamycin" in text
    assert "2.00x ULN" in text
    assert "1.00" in text


def test_clinical_accepts_visit_date_dict(api_context: APIRequestContext):
    payload = build_minimal_payload()
    payload["visit_date"] = {"day": 15, "month": 1, "year": 2024}
    response = api_context.post("/clinical", data=payload)
    assert response.status == 202

    text = response.text()
    assert "Visit date:" in text
    assert "Visit date: Not provided" not in text
