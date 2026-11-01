# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import json
import re
from collections.abc import Iterator

import pytest
from playwright.sync_api import APIRequestContext, Page, Route, expect

LIFECYCLE_FILTERS = ["All", "Draft", "Queued", "Running", "Completed", "Cancelled", "Failed"]
ANY_STATUS = re.compile(r"^(Queued|Running|Completed|Cancelled|Failed)$")

###############################################################################
def _fill_case(page: Page, patient_name: str, clinical_input: str) -> None:
    page.get_by_label("Patient Name").fill(patient_name)
    page.get_by_label("Visit Date").fill("2026-10-23")
    page.get_by_label("Clinical Input").fill(clinical_input)

###############################################################################
def _mock_ready_preflight(page: Page) -> None:
    page.route(
        "**/api/clinical/validate-input",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "ready": True,
                    "blocking_issues": [],
                    "non_blocking_issues": [],
                    "runtime_settings": {},
                    "extraction_quality": {},
                    "deterministic_diagnostics": {},
                    "rag_readiness": None,
                }
            ),
        ),
    )

###############################################################################
def _start_from_wizard(page: Page) -> None:
    page.get_by_role("button", name="Run DILI analysis").click()
    dialog = page.get_by_role("dialog", name="Prepare analysis")
    expect(dialog).to_be_visible()
    for _ in range(3):
        next_button = dialog.get_by_role("button", name="Next")
        expect(next_button).to_be_enabled()
        next_button.click()
    dialog.get_by_role("button", name="Start analysis").click()

###############################################################################
def _sessions_named(api_context: APIRequestContext, name: str) -> list[dict]:
    response = api_context.get(
        "/api/inspection/sessions", params={"search": name, "limit": "100"}
    )
    assert response.status == 200, response.text()
    return [item for item in response.json()["items"] if item["patient_name"] == name]

###############################################################################
def _discard_sessions(api_context: APIRequestContext, name: str) -> None:
    for item in _sessions_named(api_context, name):
        if item["status"] in {"queued", "running"}:
            api_context.post(f"/api/clinical/sessions/{item['session_id']}/cancel")
    for item in _sessions_named(api_context, name):
        api_context.delete(f"/api/inspection/sessions/{item['session_id']}")

###############################################################################
@pytest.fixture
def case_name(api_context: APIRequestContext, request: pytest.FixtureRequest) -> Iterator[str]:
    name = f"Lifecycle {request.node.name}"
    _discard_sessions(api_context, name)
    yield name
    _discard_sessions(api_context, name)

###############################################################################
def test_draft_save_reopen_edit_and_launch_without_duplicates(
    page: Page,
    base_url: str,
    api_context: APIRequestContext,
    case_name: str,
) -> None:
    _mock_ready_preflight(page)
    page.goto(base_url)
    _fill_case(page, case_name, "First draft text")
    with page.expect_response(
        lambda response: response.request.method == "POST"
        and response.url.endswith("/api/clinical/sessions")
    ) as created:
        page.get_by_role("button", name="Save as draft").click()
    session_id = created.value.json()["session_id"]
    expect(page.locator(".dili-draft-status")).to_contain_text("Saved")
    expect(page).to_have_url(re.compile(rf"\?session={session_id}$"))

    stored = api_context.get(f"/api/clinical/sessions/{session_id}/input").json()
    assert stored["status"] == "draft"
    assert stored["clinical_input"] == "First draft text"

    # A refresh restores the draft from the URL.
    page.reload()
    expect(page.get_by_label("Patient Name")).to_have_value(case_name)
    expect(page.get_by_label("Clinical Input")).to_have_value("First draft text")

    # Editing and saving again updates the same session.
    page.get_by_label("Clinical Input").fill("Edited draft text")
    expect(page.locator(".dili-draft-status")).to_contain_text("Unsaved changes")
    with page.expect_response(
        lambda response: response.request.method == "PUT"
        and response.url.endswith(f"/api/clinical/sessions/{session_id}/input")
        and response.status == 200
    ):
        page.get_by_role("button", name="Save as draft").click()
    drafts = _sessions_named(api_context, case_name)
    assert [item["session_id"] for item in drafts] == [session_id]
    assert drafts[0]["status"] == "draft"

    # Launching the draft promotes it instead of creating a second session.
    _start_from_wizard(page)
    expect(page).to_have_url(re.compile(rf"/report/{session_id}$"))
    expect(page.get_by_test_id("session-status")).to_have_text(ANY_STATUS)
    launched = _sessions_named(api_context, case_name)
    assert [item["session_id"] for item in launched] == [session_id]
    assert launched[0]["status"] != "draft"

###############################################################################
def test_launch_redirects_to_report_and_resets_the_form(
    page: Page,
    base_url: str,
    case_name: str,
) -> None:
    _mock_ready_preflight(page)
    page.goto(base_url)
    _fill_case(page, case_name, "Text for an immediate launch")

    _start_from_wizard(page)

    expect(page).to_have_url(re.compile(r"/report/\d+$"))
    expect(page.get_by_test_id("session-status")).to_have_text(ANY_STATUS)
    expect(page.get_by_role("heading", level=1)).to_contain_text(case_name)

    # The creation page is immediately free for another session.
    page.go_back()
    expect(page.get_by_label("Patient Name")).to_have_value("")
    expect(page.get_by_label("Visit Date")).to_have_value("")
    expect(page.get_by_label("Clinical Input")).to_have_value("")
    expect(page.get_by_role("button", name="Run DILI analysis")).to_be_enabled()

###############################################################################
def test_sessions_list_filters_by_lifecycle_status_and_keeps_context_on_refresh(
    page: Page,
    base_url: str,
    api_context: APIRequestContext,
    case_name: str,
) -> None:
    created = api_context.post(
        "/api/clinical/sessions",
        data={
            "patient_name": case_name,
            "visit_date": "2026-10-23",
            "clinical_input": "Draft for the list",
        },
    )
    assert created.status == 201, created.text()
    session_id = created.json()["session_id"]

    page.goto(f"{base_url}/clinical-sessions")
    status_nav = page.get_by_role("navigation", name="Session status")
    expect(status_nav.get_by_role("button")).to_have_text(LIFECYCLE_FILTERS)

    status_nav.get_by_role("button", name="Draft").click()
    row = page.locator(".clinical-session-row", has_text=case_name)
    expect(row).to_have_count(1)
    expect(row).to_contain_text("Draft")

    row.click()
    expect(page).to_have_url(re.compile(rf"session={session_id}"))
    expect(page.get_by_role("link", name="Edit draft")).to_be_visible()

    page.get_by_role("button", name="Timeline").click()
    expect(page).to_have_url(re.compile(r"tab=timeline"))
    page.reload()
    expect(page.get_by_role("heading", name=case_name)).to_be_visible()
    expect(page.get_by_role("button", name="Generate Timeline")).to_be_visible()

    status_nav.get_by_role("button", name="Failed").click()
    expect(page.locator(".clinical-session-row", has_text=case_name)).to_have_count(0)

###############################################################################
def test_cancel_queued_or_running_session_from_the_report(
    page: Page,
    base_url: str,
    api_context: APIRequestContext,
    case_name: str,
) -> None:
    _mock_ready_preflight(page)
    page.goto(base_url)
    _fill_case(page, case_name, "Text for a cancelled launch")
    _start_from_wizard(page)
    expect(page).to_have_url(re.compile(r"/report/\d+$"))

    cancel = page.get_by_role("button", name="Cancel analysis")
    if cancel.count() == 0:
        pytest.skip("The analysis already finished before it could be cancelled.")
    cancel.click()
    expect(page.get_by_text("Cancel this analysis?")).to_be_visible()
    page.get_by_role("button", name="Yes, cancel").click()

    expect(
        page.get_by_text(re.compile(r"Stopping after the current step|cancelled", re.IGNORECASE)).first
    ).to_be_visible()
