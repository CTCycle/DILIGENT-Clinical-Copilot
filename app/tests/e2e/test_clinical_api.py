"""
E2E tests for the clinical session API endpoints.
"""

from __future__ import annotations

import time

from playwright.sync_api import APIRequestContext


def build_minimal_payload() -> dict:
    return {
        "name": "Test Patient",
        "visit_date": "2024-01-15",
        "clinical_input": (
            "## Anamnesis\nPatient with jaundice and fatigue.\n\n"
            "## Therapy\nZetamycin 10 mg 1-0-0-0\n\n"
            "## Laboratory Analysis\n"
            "Lab 2024-01-15: ALT 100 U/L (ULN 50), ALP 200 U/L (ULN 100), bilirubin 2.0 mg/dL"
        ),
        "use_rag": False,
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


def wait_for_job_completion(api_context: APIRequestContext, job_id: str) -> dict:
    for _ in range(180):
        response = api_context.get(f"/api/clinical/jobs/{job_id}")
        assert response.status == 200
        payload = response.json()
        status = payload.get("status")
        if status in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.25)
    raise AssertionError("Clinical job did not finish in time")


def test_clinical_requires_sections(api_context: APIRequestContext):
    response = api_context.post("/api/clinical/jobs", data={"name": "Test"})
    assert response.status == 422
    assert "Clinical input is required." in response.text()


def test_clinical_rejects_blank_sections(api_context: APIRequestContext):
    response = api_context.post(
        "/api/clinical/jobs",
        data={"clinical_input": "  \n", "visit_date": None},
    )
    assert response.status == 422


def test_clinical_job_accepts_minimal_payload(api_context: APIRequestContext):
    response = api_context.post("/api/clinical/jobs", data=build_minimal_payload())
    assert response.status == 202
    job_id = response.json()["job_id"]

    job_payload = wait_for_job_completion(api_context, job_id)
    assert job_payload["status"] == "completed"
    report = str((job_payload.get("result") or {}).get("report", ""))
    assert "R-score" in report
    assert "Zetamycin" in report


def test_clinical_accepts_visit_date_dict(api_context: APIRequestContext):
    payload = build_minimal_payload()
    payload["visit_date"] = {"day": 15, "month": 1, "year": 2024}
    response = api_context.post("/api/clinical/jobs", data=payload)
    assert response.status == 202


def test_clinical_requires_therapy_drugs_even_with_anamnesis(
    api_context: APIRequestContext,
):
    payload = build_minimal_payload()
    payload["clinical_input"] = "## Anamnesis\nHistory mentions aspirin."

    response = api_context.post("/api/clinical/jobs", data=payload)
    assert response.status == 422

    codes = extract_issue_codes(response.json())
    assert "missing_timed_drug" in codes


def test_clinical_missing_labs_blocks_by_default(api_context: APIRequestContext):
    payload = build_minimal_payload()
    payload["clinical_input"] = (
        "## Anamnesis\nPatient with jaundice.\n\n"
        "## Therapy\nZetamycin 10 mg 1-0-0-0\n\n"
        "## Laboratory Analysis\nLabs unavailable."
    )

    response = api_context.post("/api/clinical/jobs", data=payload)
    assert response.status == 422

    codes = extract_issue_codes(response.json())
    assert "missing_hepatotoxicity_inputs" in codes
