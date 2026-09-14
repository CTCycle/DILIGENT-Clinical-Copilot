"""Hosted live-provider E2E coverage for the release gate.

The test is opt-in because it sends synthetic clinical text to the configured
provider and requires a real repository secret. The workflow gives it a fresh
SQLite database, so its temporary key, catalog rows, and session are isolated
from developer data.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import pytest
from playwright.sync_api import APIRequestContext, Page

from repositories.context import RepositoryContext
from repositories.drug_catalog_repository import DrugCatalogRepository
from repositories.knowledge_repository import KnowledgeRepository

LIVE_PROVIDER_ENABLED = os.getenv("DILIGENT_LIVE_PROVIDER_E2E", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PROVIDER = "opencode_go"
ACCESS_KEY_PROVIDER = "opencode"
MODEL = "deepseek-v4-flash"

pytestmark = pytest.mark.skipif(
    not LIVE_PROVIDER_ENABLED,
    reason="Set DILIGENT_LIVE_PROVIDER_E2E=1 to run the live provider gate.",
)

###############################################################################
def _seed_release_gate_catalogs() -> None:
    """Provide the minimum public catalog evidence required by clinical preflight."""

    context = RepositoryContext.create()
    DrugCatalogRepository(context).upsert_drugs_catalog_records(
        [
            {
                "rxcui": "723",
                "raw_name": "amoxicillin",
                "term_type": "IN",
                "name": "amoxicillin",
                "brand_names": "Amoxil",
                "synonyms": "amoxicillin",
            }
        ]
    )
    KnowledgeRepository(context).save_livertox_records(
        pd.DataFrame(
            [
                {
                    "drug_name": "amoxicillin",
                    "nbk_id": "NBK548517",
                    "synonyms": "amoxicillin",
                    "excerpt": "Synthetic release-gate catalog evidence for amoxicillin.",
                    "include_in_livertox": True,
                    "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK548517/",
                }
            ]
        )
    )

###############################################################################
def _clinical_input() -> str:
    return (
        "## Anamnesis\n"
        "Synthetic release-gate subject reports persistent fatigue, nausea, and "
        "mild right upper quadrant discomfort. There is no reported viral illness, "
        "alcohol exposure, or chronic liver disease in this fictional record. "
        "The subject is an adult and has no documented prior reaction to this drug.\n\n"
        "## Therapy\n"
        "Amoxicillin 500 mg twice daily was started on 2026-09-01 and stopped on "
        "2026-09-08 after symptoms developed. No other new medicine is reported.\n\n"
        "## Laboratory history\n"
        "On 2026-09-08 ALT was 210 U/L with ULN 40 U/L, AST was 170 U/L, ALP was "
        "130 U/L with ULN 120 U/L, and total bilirubin was 2.1 mg/dL."
    )

###############################################################################
def _wait_for_job(
    api_context: APIRequestContext,
    job_id: str,
    *,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = api_context.get(
            f"/api/clinical/jobs/{job_id}?_={time.monotonic_ns()}"
        )
        assert response.status == 200, response.text()
        last_payload = response.json()
        if last_payload.get("status") in {"completed", "failed", "cancelled"}:
            return last_payload
        time.sleep(2)
    pytest.fail(
        "Live provider clinical job did not reach a terminal state within "
        f"{timeout_seconds}s: status={last_payload.get('status')}"
    )

###############################################################################
def _wait_for_browser_job_submission(
    page: Page,
    *,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def capture(response: Any) -> None:
        if (
            response.request.method == "POST"
            and response.url.rstrip("/").endswith("/api/clinical/jobs")
        ):
            captured["response"] = response

    page.on("response", capture)
    try:
        page.get_by_role("button", name="Run DILI analysis").click()
        deadline = time.monotonic() + timeout_seconds
        dialog = page.get_by_role("dialog")
        while "response" not in captured and time.monotonic() < deadline:
            if dialog.is_visible():
                continue_button = dialog.get_by_role(
                    "button", name="Continue with limitations"
                )
                if not continue_button.is_visible():
                    pytest.fail(
                        "Live provider browser preflight blocked the synthetic case: "
                        + dialog.inner_text()
                    )
                continue_button.click()
            page.wait_for_timeout(250)
        if "response" not in captured:
            pytest.fail("Browser did not submit the live provider clinical job.")
        response = captured["response"]
        assert response.status == 202, response.text()
        return response.json()
    finally:
        page.remove_listener("response", capture)

###############################################################################
def test_live_opencode_go_provider_and_browser_clinical_flow(
    page: Page,
    base_url: str,
    api_context: APIRequestContext,
) -> None:
    provider_key = os.getenv("OPENCODE_GO_API_KEY", "").strip()
    if not provider_key:
        pytest.fail(
            "DILIGENT_LIVE_PROVIDER_E2E=1 requires the OPENCODE_GO_API_KEY secret."
        )

    _seed_release_gate_catalogs()
    created_key_id: int | None = None
    job_id: str | None = None
    session_id: int | None = None

    try:
        create_key = api_context.post(
            "/api/access-keys",
            data={"provider": ACCESS_KEY_PROVIDER, "access_key": provider_key},
        )
        assert create_key.status == 201, create_key.text()
        created_key = create_key.json()
        created_key_id = created_key.get("id")
        assert isinstance(created_key_id, int)
        assert "access_key" not in created_key

        activate_key = api_context.put(
            f"/api/access-keys/{created_key_id}/activate?provider={ACCESS_KEY_PROVIDER}"
        )
        assert activate_key.status == 200, activate_key.text()

        config = api_context.put(
            "/api/model-config",
            data={
                "use_cloud_services": True,
                "llm_provider": PROVIDER,
                "cloud_model": MODEL,
                "text_extraction_model": MODEL,
                "clinical_model": MODEL,
                "revision_model": MODEL,
                "timeline_model": MODEL,
                "reasoning_level": "off",
            },
        )
        assert config.status == 200, config.text()
        config_payload = config.json()
        assert config_payload["llm_provider"] == PROVIDER
        assert config_payload["clinical_model"] == MODEL

        connectivity = api_context.post(
            "/api/model-config/connectivity-check",
            data={"provider": PROVIDER, "model": MODEL},
        )
        assert connectivity.status == 200, connectivity.text()
        connectivity_payload = connectivity.json()
        assert connectivity_payload == {
            "provider": PROVIDER,
            "model": MODEL,
            "ok": True,
            "response_preview": connectivity_payload.get("response_preview"),
            "error": None,
        }
        assert connectivity_payload["response_preview"]

        page.goto(base_url)
        page.get_by_label("Clinical Input").fill(_clinical_input())
        page.get_by_label("Patient Name").fill("Synthetic Release Gate Subject")
        page.get_by_label("Visit Date").fill("2026-09-08")
        start_payload = _wait_for_browser_job_submission(page)
        job_id = start_payload.get("job_id")
        assert isinstance(job_id, str) and job_id
        assert start_payload.get("job_type") == "clinical"

        final_payload = _wait_for_job(api_context, job_id)
        assert final_payload.get("status") == "completed", final_payload.get("error")
        result = final_payload.get("result")
        assert isinstance(result, dict)
        assert isinstance(result.get("session_id"), int)
        session_id = result["session_id"]
        assert result.get("report") or result.get("final_report")

        page.wait_for_function(
            """() => {
                const report = document.querySelector('#rendered-report-content');
                return Boolean(report && report.textContent?.trim().length);
            }""",
            timeout=120000,
        )
        report_text = page.locator("#rendered-report-content").inner_text()
        assert report_text.strip()

        session_response = api_context.get(f"/api/inspection/sessions/{session_id}")
        assert session_response.status == 200, session_response.text()
        assert session_response.json().get("session_id") == session_id
    finally:
        if job_id is not None and session_id is None:
            api_context.delete(f"/api/clinical/jobs/{job_id}")
        if session_id is not None:
            api_context.delete(f"/api/inspection/sessions/{session_id}")
        if created_key_id is not None:
            api_context.delete(
                f"/api/access-keys/{created_key_id}?provider={ACCESS_KEY_PROVIDER}"
            )
