# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Local-Ollama E2E coverage of a completed DILI assessment.

Opt-in lane (``run_browser_e2e.ps1 -Suite LocalOllama``). It needs a running
Ollama with the model named by ``DILIGENT_LOCAL_OLLAMA_MODEL`` (default
``qwen3.5:2b``). Synthetic clinical text never leaves the machine, and the
lane runs against the isolated seeded database.
"""

from __future__ import annotations

import os
import re
import time

from playwright.sync_api import APIRequestContext, Page, expect

MODEL = os.getenv("DILIGENT_LOCAL_OLLAMA_MODEL", "qwen3.5:2b")
JOB_TIMEOUT_SECONDS = 1500


###############################################################################
def _clinical_input() -> str:
    return (
        "## Anamnesis\n"
        "Synthetic subject reports persistent fatigue, nausea, and mild right upper "
        "quadrant discomfort. There is no reported viral illness, alcohol exposure, "
        "or chronic liver disease in this fictional record. The subject is an adult "
        "with no documented prior reaction to this drug.\n\n"
        "## Therapy\n"
        "Amoxicillin 500 mg twice daily was started on 2026-09-01 and stopped on "
        "2026-09-08 after symptoms developed. No other new medicine is reported.\n\n"
        "## Laboratory history\n"
        "On 2026-09-08 ALT was 210 U/L with ULN 40 U/L, AST was 170 U/L, ALP was "
        "130 U/L with ULN 120 U/L, and total bilirubin was 2.1 mg/dL."
    )


###############################################################################
def _save_agent(
    api_context: APIRequestContext, agent: dict, configuration: dict
) -> dict:
    response = api_context.put(
        f"/api/agents/{agent['id']}",
        data={
            "expected_revision": agent["current_revision"],
            "name": agent["name"],
            "description": agent["description"],
            "configuration": configuration,
        },
    )
    assert response.status == 200, response.text()
    return response.json()


###############################################################################
def test_completed_assessment_on_local_ollama(
    page: Page,
    base_url: str,
    api_context: APIRequestContext,
) -> None:
    refresh = api_context.post("/api/model-config/catalogs/ollama/refresh")
    assert refresh.status == 200, refresh.text()
    listing = api_context.get("/api/agents?kind=dili")
    assert listing.status == 200, listing.text()
    agent = next(item for item in listing.json()["agents"] if item["is_default"])
    original = agent["revision"]["configuration"]
    local = {"provider": "ollama", "model": MODEL}
    configuration = {
        **original,
        "models": {
            **original["models"],
            "extraction": local,
            "clinical": local,
            "timeline": local,
        },
        "reasoning_level": "off",
    }
    updated = _save_agent(api_context, agent, configuration)
    try:
        page.goto(base_url)
        page.get_by_label("Clinical Input").fill(_clinical_input())
        page.get_by_label("Patient Name").fill("Synthetic Local Ollama Subject")
        page.get_by_label("Visit Date").fill("2026-09-08")

        page.get_by_role("button", name="Run DILI analysis").click()
        dialog = page.get_by_role("dialog", name="Prepare analysis")
        next_button = dialog.get_by_role("button", name="Next")
        for _ in range(3):
            expect(next_button).to_be_enabled(timeout=60000)
            next_button.click()
        with page.expect_response(
            lambda response: response.request.method == "POST"
            and response.url.rstrip("/").endswith("/launch")
        ) as started:
            dialog.get_by_role("button", name="Start analysis").click()
        assert started.value.status == 202, started.value.text()
        session_id = started.value.json()["session_id"]
        expect(page).to_have_url(re.compile(rf"/report/{session_id}$"))

        deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
        state: dict = {}
        while time.monotonic() < deadline:
            response = api_context.get(f"/api/clinical/sessions/{session_id}/status")
            assert response.status == 200, response.text()
            state = response.json()
            if state["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(3)
        assert state.get("status") == "completed", state.get("error") or state.get("status")
        assert state["has_report"] is True

        expect(page.locator("app-clinical-report-viewer")).not_to_be_empty(timeout=60000)

        session = api_context.get(f"/api/inspection/sessions/{session_id}")
        assert session.status == 200, session.text()
        assert session.json()["session_id"] == session_id
    finally:
        _save_agent(api_context, updated, original)
