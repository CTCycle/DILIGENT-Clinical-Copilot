# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Agent Studio browser flows: lifecycle management and launching with a non-default agent."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator

import pytest
from playwright.sync_api import APIRequestContext, Page, expect

AGENT_NAME_PREFIX = "E2E agent "


###############################################################################
def _agents(api_context: APIRequestContext, kind: str = "dili") -> list[dict]:
    response = api_context.get("/api/agents", params={"include_archived": "true"})
    assert response.status == 200, response.text()
    return [item for item in response.json()["agents"] if item["kind"] == kind]


###############################################################################
def _default_agent(api_context: APIRequestContext, kind: str = "dili") -> dict:
    return next(
        item for item in _agents(api_context, kind) if item["is_default"] and not item["is_archived"]
    )


###############################################################################
@pytest.fixture
def studio_cleanup(api_context: APIRequestContext) -> Iterator[str]:
    """Restore the original default DILI agent and archive agents the test created."""
    original_default = _default_agent(api_context)["id"]
    suffix = uuid.uuid4().hex[:6]
    yield suffix
    restored = api_context.put(f"/api/agents/{original_default}/default")
    assert restored.status == 200, restored.text()
    for item in _agents(api_context):
        if item["name"].startswith(AGENT_NAME_PREFIX) and suffix in item["name"]:
            if not item["is_archived"]:
                api_context.post(f"/api/agents/{item['id']}/archive")


###############################################################################
def _create_agent(page: Page, name: str) -> None:
    page.get_by_role("button", name="New DILI agent").click()
    page.get_by_label("Name for the new DILI agent").fill(name)
    page.get_by_role("button", name="Create agent").click()
    expect(page.get_by_role("heading", name=name)).to_be_visible()


###############################################################################
def _agent_row(page: Page, name: str):
    return page.locator(".agent-row").filter(has=page.get_by_role("button", name=name, exact=True))


###############################################################################
def test_agent_lifecycle_create_edit_duplicate_default_archive_and_reload(
    page: Page,
    base_url: str,
    api_context: APIRequestContext,
    studio_cleanup: str,
) -> None:
    page.set_default_timeout(15000)
    name = f"{AGENT_NAME_PREFIX}{studio_cleanup}"
    page.goto(f"{base_url}/agents")
    expect(page.get_by_role("tab", name="My Agents")).to_have_attribute("aria-selected", "true")

    # Create: the new agent opens in the editor at revision 1.
    _create_agent(page, name)
    expect(page.get_by_text("Active revision 1")).to_be_visible()

    # Edit: saving produces a new immutable revision.
    focus = f"Prioritise hepatocellular pattern {studio_cleanup}"
    page.get_by_label("Assessment focus").fill(focus)
    expect(page.get_by_text("Unsaved changes")).to_be_visible()
    with page.expect_response(
        lambda r: r.request.method == "PUT" and "/api/agents/" in r.url and not r.url.endswith("/default")
    ) as saved:
        page.get_by_role("button", name="Save agent").click()
    assert saved.value.status == 200, saved.value.text()
    expect(page.get_by_text("Active revision 2")).to_be_visible()
    created = next(item for item in _agents(api_context) if item["name"] == name)
    assert created["current_revision"] == 2
    assert created["revision"]["configuration"]["instructions"]["assessment_focus"] == focus

    # Duplicate, then make the duplicate the default.
    page.get_by_role("tab", name="My Agents").click()
    page.get_by_role("button", name=f"Duplicate {name}", exact=True).click()
    duplicate_name = next(
        item["name"]
        for item in _agents(api_context)
        if item["name"].startswith(name) and item["name"] != name
    )
    page.get_by_role("tab", name="My Agents").click()
    page.get_by_role("button", name=f"Set {duplicate_name} as default", exact=True).click()
    expect(_agent_row(page, duplicate_name).locator(".chip.is-default")).to_be_visible()

    # The previous default is no longer archive-protected; the current default is.
    expect(page.get_by_role("button", name=f"Archive {duplicate_name}", exact=True)).to_be_disabled()
    page.get_by_role("button", name=f"Archive {name}", exact=True).click()
    page.get_by_role("button", name="Archive agent").click()
    expect(page.get_by_role("button", name=name, exact=True)).to_have_count(0)

    # Reload: state persists (archived agent hidden, duplicate still default).
    page.reload()
    expect(page.get_by_role("button", name=duplicate_name, exact=True)).to_be_visible()
    expect(page.get_by_role("button", name=name, exact=True)).to_have_count(0)
    page.get_by_label("Show archived").check()
    expect(_agent_row(page, name).get_by_text("Archived", exact=True)).to_be_visible()
    assert _default_agent(api_context)["name"] == duplicate_name


###############################################################################
def test_launch_with_non_default_agent_records_its_runtime_provenance(
    page: Page,
    base_url: str,
    api_context: APIRequestContext,
    studio_cleanup: str,
) -> None:
    page.set_default_timeout(15000)
    agent_name = f"{AGENT_NAME_PREFIX}{studio_cleanup}"
    patient = f"Agent Launch {studio_cleanup}"
    default_agent = _default_agent(api_context)
    page.goto(f"{base_url}/agents")
    _create_agent(page, agent_name)
    agent = next(item for item in _agents(api_context) if item["name"] == agent_name)
    assert agent["id"] != default_agent["id"] and not agent["is_default"]

    session_id = None
    try:
        page.get_by_role("tab", name="DILI Agent").click()
        page.get_by_label("Patient Name").fill(patient)
        page.get_by_label("Visit Date").fill("2026-10-23")
        page.get_by_label("Clinical Input").fill(
            "## Anamnesis\n"
            + " ".join(["Patient reports persistent fatigue nausea and abdominal discomfort."] * 12)
            + "\n\n## Therapy\nAmoxicillin 500 mg BID, started on 2026-04-13, stopped on 2026-04-20.\n\n"
            + "## Laboratory history\nALT 210 U/L; AST 180 U/L; ALP 130 U/L."
        )
        page.get_by_role("button", name="Run DILI analysis").click()
        dialog = page.get_by_role("dialog", name="Prepare analysis")
        next_button = dialog.get_by_role("button", name="Next")
        expect(next_button).to_be_enabled(timeout=30000)
        next_button.click()

        # Agent step: the default is preselected; choose the new agent and re-check.
        expect(dialog.locator("#wizard-agent")).to_have_value(default_agent["id"])
        dialog.locator("#wizard-agent").select_option(agent["id"])
        expect(dialog.get_by_test_id("agent-summary")).to_contain_text(agent_name)
        expect(dialog.get_by_test_id("agent-summary")).to_contain_text("Revision 1")
        expect(next_button).to_be_enabled(timeout=30000)
        next_button.click()
        expect(dialog.locator(".wizard-review")).to_contain_text(agent_name)

        with page.expect_response(
            lambda r: r.request.method == "POST" and r.url.endswith("/launch")
        ) as launched:
            dialog.get_by_role("button", name="Start analysis").click()
        assert launched.value.status == 202, launched.value.text()
        assert launched.value.request.post_data_json["agent_id"] == agent["id"]
        assert launched.value.request.post_data_json["agent_revision"] == 1
        session_id = launched.value.json()["session_id"]

        detail = api_context.get(f"/api/inspection/sessions/{session_id}").json()
        provenance = detail["metadata"]["agent"]
        assert provenance["agent_id"] == agent["id"]
        assert provenance["agent_revision"] == 1
        assert provenance["agent_name"] == agent_name
    finally:
        if session_id is not None:
            api_context.post(f"/api/clinical/sessions/{session_id}/cancel")
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                status = api_context.get(f"/api/clinical/sessions/{session_id}/status").json()
                if status["status"] not in {"queued", "running"}:
                    break
                time.sleep(1)
            api_context.delete(f"/api/inspection/sessions/{session_id}")
