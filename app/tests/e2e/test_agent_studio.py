# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Agent Studio browser flows: card management, the shared editor and non-default agent launches."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator

import pytest
from playwright.sync_api import APIRequestContext, Locator, Page, expect

AGENT_NAME_PREFIX = "E2E agent "


###############################################################################
def _agents(api_context: APIRequestContext) -> list[dict]:
    response = api_context.get("/api/agents", params={"include_archived": "true"})
    assert response.status == 200, response.text()
    return response.json()["agents"]


###############################################################################
def _default_agent(api_context: APIRequestContext) -> dict:
    return next(
        item for item in _agents(api_context) if item["is_default"] and not item["is_archived"]
    )


###############################################################################
@pytest.fixture
def studio_cleanup(api_context: APIRequestContext) -> Iterator[str]:
    """Restore the original default agent and delete agents the test created."""
    original_default = _default_agent(api_context)["id"]
    suffix = uuid.uuid4().hex[:6]
    yield suffix
    restored = api_context.put(f"/api/agents/{original_default}/default")
    assert restored.status == 200, restored.text()
    for item in _agents(api_context):
        if item["name"].startswith(AGENT_NAME_PREFIX) and suffix in item["name"]:
            api_context.put(
                f"/api/agents/{item['id']}/lifecycle",
                data={"is_locked": False, "is_archived": True},
            )
            api_context.delete(f"/api/agents/{item['id']}")


###############################################################################
def _create_agent(page: Page, name: str) -> None:
    page.get_by_role("link", name="New agent").click()
    expect(page.get_by_role("heading", name="New agent", level=1)).to_be_visible()
    page.get_by_label("Name", exact=True).fill(name)
    page.get_by_role("button", name="Create agent").click()
    expect(page.get_by_role("heading", name=name, level=1)).to_be_visible()


###############################################################################
def _card(page: Page, name: str) -> Locator:
    return page.get_by_test_id("agent-card").filter(
        has=page.get_by_role("link", name=name, exact=True)
    )


###############################################################################
def test_agent_lifecycle_create_edit_lock_archive_delete_and_reload(
    page: Page,
    base_url: str,
    api_context: APIRequestContext,
    studio_cleanup: str,
) -> None:
    page.set_default_timeout(15000)
    name = f"{AGENT_NAME_PREFIX}{studio_cleanup}"
    page.goto(f"{base_url}/agents")
    expect(page.get_by_role("tab", name="My Agents")).to_have_attribute("aria-selected", "true")

    # Create: one agent with all three capabilities, active and unlocked.
    _create_agent(page, name)
    created = next(item for item in _agents(api_context) if item["name"] == name)
    assert created["capabilities"] == ["assessment", "revision", "timeline"]
    assert created["is_active"] and not created["is_locked"] and not created["is_default"]

    # Edit one capability: saving produces a new immutable revision.
    focus = f"Dates first {studio_cleanup}"
    page.get_by_label("Timeline focus").fill(focus)
    expect(page.get_by_text("Unsaved changes")).to_be_visible()
    with page.expect_response(
        lambda r: r.request.method == "PUT" and r.url.endswith(f"/api/agents/{created['id']}")
    ) as saved:
        page.get_by_role("button", name="Save agent").click()
    assert saved.value.status == 200, saved.value.text()
    persisted = next(item for item in _agents(api_context) if item["name"] == name)
    assert persisted["current_revision"] == 2
    assert persisted["revision"]["configuration"]["timeline"]["instructions"]["timeline_focus"] == focus

    # Disable one capability independently; its values stay stored.
    page.get_by_role("switch", name="Disable Timeline").uncheck()
    page.get_by_role("button", name="Save agent").click()
    expect(page.get_by_text("Saved as revision 3")).to_be_visible()
    disabled = next(item for item in _agents(api_context) if item["name"] == name)
    timeline = disabled["revision"]["configuration"]["timeline"]
    assert disabled["capabilities"] == ["assessment", "revision"]
    assert timeline["enabled"] is False and timeline["instructions"]["timeline_focus"] == focus

    # Lock from the profile card: configuration becomes read-only, unlock stays reachable.
    page.get_by_role("switch", name="Locked").check()
    expect(page.get_by_text("This agent is locked")).to_be_visible()
    expect(page.get_by_label("Name", exact=True)).to_be_disabled()
    expect(page.get_by_role("switch", name="Locked")).to_be_enabled()
    page.get_by_role("switch", name="Locked").uncheck()
    expect(page.get_by_label("Name", exact=True)).to_be_enabled()

    # Back on the grid: the card shows its status, and delete needs confirmation.
    page.get_by_role("link", name="Back to My Agents").click()
    card = _card(page, name)
    expect(card).to_contain_text("Active")
    card.get_by_role("button", name=f"Lock {name}").click()
    expect(card).to_contain_text("Locked")
    delete = card.get_by_role("button", name=f"Delete {name}")
    expect(delete).to_be_disabled()
    card.get_by_role("button", name=f"Unlock {name}").click()
    card.get_by_role("button", name=f"Archive {name}").click()
    expect(_card(page, name)).to_have_count(0)

    # Reload: the archived agent stays hidden until requested, then can be restored.
    page.reload()
    expect(_card(page, name)).to_have_count(0)
    page.get_by_label("Show archived").check()
    expect(_card(page, name)).to_contain_text("Archived")
    _card(page, name).get_by_role("button", name=f"Restore {name}").click()
    expect(_card(page, name)).not_to_contain_text("Archived")

    # Permanent delete asks for confirmation first.
    _card(page, name).get_by_role("button", name=f"Delete {name}").click()
    page.get_by_role("dialog").get_by_role("button", name="Delete agent").click()
    expect(_card(page, name)).to_have_count(0)
    assert all(item["name"] != name for item in _agents(api_context))


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
        assert provenance["agent_capability"] == "assessment"
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
