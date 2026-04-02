"""
E2E tests for the clinical session API endpoint.
"""
from __future__ import annotations

from playwright.sync_api import APIRequestContext


def build_minimal_payload() -> dict:
    return {
        "name": "Test Patient",
        "visit_date": "2024-01-15",
        "anamnesis": "Patient with jaundice and fatigue.",
        "drugs": "Zetamycin 10 mg 1-0-0-0",
        "laboratory_analysis": "Lab 2024-01-15: ALT 100 U/L (ULN 50), ALP 200 U/L (ULN 100), bilirubin 2.0 mg/dL",
        "use_rag": False,
        "has_hepatic_diseases": False,
    }


def extract_issue_codes(payload: dict) -> set[str]:
    detail = payload.get("detail", [])
    if not isinstance(detail, list):
        return set()
    codes: set[str] = set()
    for item in detail:
        if isinstance(item, dict):
            code = item.get("code")
            if isinstance(code, str):
                codes.add(code)
    return codes


def test_clinical_requires_sections(api_context: APIRequestContext):
    response = api_context.post("/clinical", data={"name": "Test"})
    assert response.status == 422

    codes = extract_issue_codes(response.json())
    assert "missing_anamnesis" in codes
    assert "missing_visit_date" in codes


def test_clinical_rejects_blank_sections(api_context: APIRequestContext):
    response = api_context.post(
        "/clinical",
        data={"anamnesis": "  \n", "drugs": "   ", "visit_date": None},
    )
    assert response.status == 422


def test_clinical_accepts_minimal_payload(api_context: APIRequestContext):
    response = api_context.post("/clinical", data=build_minimal_payload())
    assert response.status == 202

    text = response.text()
    assert "# Clinical Visit Summary" in text or "# Sintesi Visita Clinica" in text
    assert "## Hepatotoxicity Pattern" in text or "## Pattern di Epatotossicità" in text
    assert "R-score" in text
    assert "Classification" in text
    assert "Zetamycin" in text


def test_clinical_accepts_visit_date_dict(api_context: APIRequestContext):
    payload = build_minimal_payload()
    payload["visit_date"] = {"day": 15, "month": 1, "year": 2024}
    response = api_context.post("/clinical", data=payload)
    assert response.status == 202

    text = response.text()
    assert "Visit date:" in text
    assert "Visit date: Not provided" not in text


def test_clinical_requires_therapy_drugs_even_with_anamnesis(
    api_context: APIRequestContext,
):
    payload = build_minimal_payload()
    payload["anamnesis"] = "History mentions aspirin."
    payload["drugs"] = "   "

    response = api_context.post("/clinical", data=payload)
    assert response.status == 422

    codes = extract_issue_codes(response.json())
    assert "missing_timed_drug" in codes


def test_clinical_missing_labs_blocks_by_default(api_context: APIRequestContext):
    payload = build_minimal_payload()
    payload["laboratory_analysis"] = "Labs unavailable."

    response = api_context.post("/clinical", data=payload)
    assert response.status == 422

    codes = extract_issue_codes(response.json())
    assert "missing_hepatotoxicity_inputs" in codes


def test_clinical_missing_visit_date_blocks(api_context: APIRequestContext):
    payload = build_minimal_payload()
    payload["visit_date"] = None

    response = api_context.post("/clinical", data=payload)
    assert response.status == 422
    codes = extract_issue_codes(response.json())
    assert "missing_visit_date" in codes


def test_clinical_missing_anamnesis_blocks(api_context: APIRequestContext):
    payload = build_minimal_payload()
    payload["anamnesis"] = None

    response = api_context.post("/clinical", data=payload)
    assert response.status == 422
    codes = extract_issue_codes(response.json())
    assert "missing_anamnesis" in codes


def test_clinical_missing_timing_blocks(api_context: APIRequestContext):
    payload = build_minimal_payload()
    payload["drugs"] = "Zetamycin"

    response = api_context.post("/clinical", data=payload)
    assert response.status == 422
    codes = extract_issue_codes(response.json())
    assert "missing_timed_drug" in codes


def test_clinical_report_includes_estimated_rucam(api_context: APIRequestContext):
    payload = build_minimal_payload()
    payload["anamnesis"] = (
        "ALT 320 U/L (ULN 40) on 2025-01-10, ALT 150 U/L on 2025-01-20. "
        "Jaundice started on 2025-01-11."
    )
    payload["laboratory_analysis"] = (
        "Lab 2025-01-10: ALT 320 U/L (ULN 40), ALP 140 U/L (ULN 120), bilirubin 2.1 mg/dL\n"
        "Lab 2025-01-20: ALT 150 U/L (ULN 40), ALP 120 U/L (ULN 120)"
    )

    response = api_context.post("/clinical", data=payload)
    assert response.status == 202
    text = response.text()
    assert "RUCAM" in text


def test_clinical_endpoint_stable_without_usable_longitudinal_labs(
    api_context: APIRequestContext,
):
    payload = build_minimal_payload()
    payload["anamnesis"] = "Patient reports fatigue but no explicit longitudinal labs."
    payload["laboratory_analysis"] = "No measurable values."

    response = api_context.post("/clinical", data=payload)
    assert response.status == 422
    codes = extract_issue_codes(response.json())
    assert "missing_hepatotoxicity_inputs" in codes


def test_clinical_preserves_italian_output_language(api_context: APIRequestContext):
    payload = {
        "name": "Mario Rossi",
        "visit_date": "2025-01-15",
        "anamnesis": "Paziente con ittero e dolore addominale.",
        "drugs": "Paracetamolo sospeso dal 2025-01-11",
        "laboratory_analysis": (
            "Laboratorio 2025-01-10: ALT 320 U/L (ULN 40), ALP 140 U/L (ULN 120), bilirubina 2.1 mg/dL"
        ),
        "use_rag": False,
        "has_hepatic_diseases": False,
    }
    response = api_context.post("/clinical", data=payload)
    assert response.status == 202
    text = response.text()
    assert "# Sintesi Visita Clinica" in text
