"""
E2E tests for the clinical session API endpoints.
"""

from __future__ import annotations

import time

import pytest
from playwright.sync_api import APIRequestContext


def repeated_words(count: int) -> str:
    return " ".join(f"clinicalword{i}" for i in range(count))


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
        "use_rag": False,
        "selected_model_providers": ["openai"],
    }


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


@pytest.fixture(autouse=True)
def reset_runtime_to_local(api_context: APIRequestContext) -> None:
    latest_job = api_context.get("/api/clinical/jobs/latest")
    if latest_job.status == 200:
        payload = latest_job.json()
        job_id = payload.get("job_id")
        status = payload.get("status")
        if job_id and status in {"pending", "running"}:
            api_context.delete(f"/api/clinical/jobs/{job_id}")
    response = api_context.put(
        "/api/model-config",
        data={"use_cloud_services": False, "cloud_model": "gpt-4.1-mini"},
    )
    assert response.status == 200


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

    status_response = api_context.get(f"/api/clinical/jobs/{job_id}")
    assert status_response.status == 200
    assert status_response.json()["job_id"] == job_id
    api_context.delete(f"/api/clinical/jobs/{job_id}")
    wait_for_job_completion(api_context, job_id)


def test_clinical_accepts_visit_date_dict(api_context: APIRequestContext):
    payload = build_minimal_payload()
    payload["visit_date"] = {"day": 15, "month": 1, "year": 2024}
    response = api_context.post("/api/clinical/jobs", data=payload)
    assert response.status == 202
    job_id = response.json()["job_id"]
    api_context.delete(f"/api/clinical/jobs/{job_id}")
    wait_for_job_completion(api_context, job_id)


def test_clinical_rejects_empty_provider_selection(api_context: APIRequestContext):
    payload = build_minimal_payload()
    payload["selected_model_providers"] = []

    response = api_context.post("/api/clinical/jobs", data=payload)
    assert response.status == 422
    assert "At least one model provider must be selected." in response.text()


def test_clinical_requires_visit_date(api_context: APIRequestContext):
    payload = build_minimal_payload()
    payload["visit_date"] = None

    response = api_context.post("/api/clinical/jobs", data=payload)
    assert response.status == 422
    assert "Visit date is required." in response.text()
