from __future__ import annotations

import time
from typing import Any

from fastapi.testclient import TestClient

import app as server_app_module
from services.llm.runtime_config import LLMRuntimeConfig
from domain.clinical.robustness import ClinicalInputPreflightResult
from services.runtime.jobs import JobManager
from services.session.session_service import ClinicalSessionService

###############################################################################
def get_route_service(route_path: str) -> Any:
    for route in server_app_module.app.routes:
        if getattr(route, "path", "").endswith(route_path):
            endpoint_owner = getattr(route.endpoint, "__self__", None)
            if endpoint_owner is not None:
                return endpoint_owner.service
    raise AssertionError(f"Route not found: {route_path}")

###############################################################################
def dummy_clinical_payload() -> dict[str, Any]:
    filler = (
        "The patient reports fatigue, pruritus, dark urine, mild nausea, and "
        "recent appetite reduction without alcohol excess, viral prodrome, "
        "travel exposure, hypotension, biliary colic, or known chronic liver "
        "disease. Symptoms began gradually after medication exposure and are "
        "documented here only as mock endpoint test content."
    )
    return {
        "name": "Endpoint Dummy",
        "visit_date": "2026-06-19",
        "clinical_input": (
            "## Anamnesis\n"
            f"{filler}\n\n"
            "## Therapy\n"
            "Amoxicillin 500 mg three times daily for seven days.\n"
            "Ibuprofen 400 mg as needed for fever.\n"
            "Atorvastatin 20 mg nightly as chronic therapy.\n\n"
            "## Laboratory Analysis\n"
            "ALT 220 U/L with ULN 50, ALP 180 U/L with ULN 120, total "
            "bilirubin 2.4 mg/dL, INR 1.1, and hepatic pattern mixed."
        ),
        "use_rag": False,
        "selected_model_providers": ["ollama"],
    }

###############################################################################
def wait_for_terminal_status(client: TestClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 3.0
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/clinical/jobs/{job_id}")
        assert response.status_code == 200
        last_payload = response.json()
        if last_payload["status"] in {"completed", "failed", "cancelled"}:
            return last_payload
        time.sleep(0.05)
    raise AssertionError(f"Clinical job did not finish: {last_payload}")

###############################################################################
def test_clinical_jobs_endpoint_completes_dummy_three_drug_assessment(
    monkeypatch,
) -> None:
    service = get_route_service("/clinical/jobs")
    monkeypatch.setattr(service, "job_manager", JobManager())
    monkeypatch.setattr(
        ClinicalSessionService,
        "apply_persisted_runtime_configuration",
        lambda self: None,
    )
    monkeypatch.setattr(
        ClinicalSessionService,
        "ensure_submission_requirements",
        lambda self, payload: None,
    )
    monkeypatch.setattr(
        LLMRuntimeConfig,
        "is_cloud_enabled",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        ClinicalSessionService,
        "validate_clinical_input",
        lambda self, request: ClinicalInputPreflightResult(
            ready=True,
            deterministic_diagnostics={
                "therapy": {
                    "drug_count": 3,
                    "drugs": ["Amoxicillin", "Ibuprofen", "Atorvastatin"],
                }
            },
        ),
    )
    monkeypatch.setattr(
        service.serializer,
        "list_livertox_catalog",
        lambda search, offset, limit: ([{"drug_name": "Amoxicillin"}], 1),
    )
    monkeypatch.setattr(
        service.serializer,
        "list_rxnav_catalog",
        lambda search, offset, limit: ([{"drug_name": "Amoxicillin"}], 1),
    )

    async def fake_process_single_patient(
        self: ClinicalSessionService,
        payload: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        source_text = "\n".join(
            [
                payload.anamnesis or "",
                payload.drugs or "",
                payload.laboratory_analysis or "",
            ]
        )
        assert source_text.count("Amoxicillin") == 1
        assert source_text.count("Ibuprofen") == 1
        assert source_text.count("Atorvastatin") == 1
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback("clinical", 16.0, "drugs.extracting")
            progress_callback("clinical", 82.0, "retrieval.evidence")
            progress_callback("clinical", 94.0, "report.generating")
            progress_callback("clinical", 99.0, "session.saving")
        return {
            "detected_drugs": ["Amoxicillin", "Ibuprofen", "Atorvastatin"],
            "matched_drugs": [],
            "issues": [],
            "final_report": "Dummy endpoint assessment completed.",
            "runtime_settings": {
                "use_cloud_services": False,
                "text_extraction_model": "dummy-text-model",
                "clinical_model": "dummy-clinical-model",
            },
        }

    monkeypatch.setattr(
        ClinicalSessionService,
        "process_single_patient",
        fake_process_single_patient,
    )

    with TestClient(server_app_module.app, raise_server_exceptions=False) as client:
        start_response = client.post("/api/clinical/jobs", json=dummy_clinical_payload())
        assert start_response.status_code == 202, start_response.text
        job_id = start_response.json()["job_id"]

        terminal = wait_for_terminal_status(client, job_id)

    assert terminal["status"] == "completed"
    assert terminal["progress"] == 100.0
    result = terminal["result"]
    assert result is not None
    assert result["progress_stage"] == "completed"
    assert result["progress_message"] == "Clinical analysis completed."
    assert result["progress_stage"] != "clinical"
    assert result["detected_drugs"] == ["Amoxicillin", "Ibuprofen", "Atorvastatin"]
    assert len(result["detected_drugs"]) == 3
