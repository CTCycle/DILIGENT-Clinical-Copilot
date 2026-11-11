# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from playwright.sync_api import Page, Route, expect


###############################################################################
def _fill_valid_fields(page: Page) -> None:
    page.get_by_label("Clinical Input").fill(
        "## Anamnesis\n"
        + " ".join(["Patient reports fatigue and nausea after therapy."] * 12)
        + "\n\n## Therapy\nAmoxicillin 500 mg twice daily from 2026-07-01.\n\n"
        + "## Laboratory analysis\nALT 210 U/L, ALP 160 U/L, bilirubin 2.1 mg/dL."
    )
    page.get_by_label("Visit Date").fill("2026-07-17")


###############################################################################
def _open_wizard(page: Page):
    page.get_by_role("button", name="Run DILI analysis").click()
    dialog = page.get_by_role("dialog", name="Prepare analysis")
    expect(dialog).to_be_visible()
    return dialog


###############################################################################
def _issue(severity: str, code: str, field: str) -> dict:
    return {
        "severity": severity,
        "code": code,
        "message": f"{code} message",
        "field": field,
        "title": f"{code} title",
        "description": f"{code} description",
        "affected_section": field.replace("_", " ").title(),
        "consequence": f"{code} consequence",
        "continuation_allowed": severity == "non_blocking",
    }


###############################################################################
def _preflight_body(blocking: list[dict], warnings: list[dict]) -> str:
    return json.dumps(
        {
            "ready": not blocking,
            "blocking_issues": blocking,
            "non_blocking_issues": warnings,
            "runtime_settings": {},
            "extraction_quality": {},
            "deterministic_diagnostics": {},
            "rag_readiness": None,
        }
    )


###############################################################################
def _session_state(status: str, session_id: int = 9, **overrides) -> dict:
    state = {
        "session_id": session_id,
        "status": status,
        "patient_name": None,
        "visit_date": "2026-07-17",
        "created_at": "2026-07-17T08:00:00Z",
        "updated_at": "2026-07-17T08:00:00Z",
        "queued_at": None,
        "started_at": None,
        "finished_at": None,
        "queue_position": 1 if status == "queued" else None,
        "progress": None,
        "progress_stage": None,
        "progress_message": None,
        "stop_requested": False,
        "error": None,
        "has_report": False,
        "version": 1,
        "poll_interval": 0.2,
    }
    state.update(overrides)
    return state


###############################################################################
def _json(route: Route, body: dict, status: int = 200) -> None:
    route.fulfill(status=status, content_type="application/json", body=json.dumps(body))


###############################################################################
def test_blocking_preflight_modal_prevents_job_submission(
    page: Page,
    base_url: str,
) -> None:
    submission_count = 0

    def count_sessions(route: Route) -> None:
        nonlocal submission_count
        if route.request.method == "POST":
            submission_count += 1
            route.abort()
        else:
            route.continue_()

    page.route("**/api/clinical/sessions**", count_sessions)
    try:
        page.goto(base_url)
        sessions_before = page.request.get(f"{base_url}/api/inspection/sessions").json()
        _fill_valid_fields(page)
        page.get_by_label("Visit Date").fill("")
        with page.expect_response("**/api/clinical/validate-input") as response:
            dialog = _open_wizard(page)
        body = response.value.json()
        codes = [issue["code"] for issue in body["blocking_issues"]]
        assert codes.count("visit_date_missing") == 1
        assert "missing_visit_date" not in codes

        expect(
            dialog.locator(".wizard-issue h3").filter(has_text="Visit date is missing")
        ).to_have_count(1)
        expect(dialog.locator(".wizard-issue")).to_have_count(
            len(body["blocking_issues"]) + len(body["non_blocking_issues"])
        )
        expect(dialog.get_by_role("button", name="Next")).to_be_disabled()
        expect(dialog.get_by_role("button", name="Agent")).to_be_disabled()
        assert submission_count == 0
        assert (
            page.request.get(f"{base_url}/api/inspection/sessions").json()
            == sessions_before
        )
    finally:
        page.unroute("**/api/clinical/sessions**", count_sessions)


###############################################################################
def test_warning_acceptance_launches_one_session_and_opens_its_report(
    page: Page,
    base_url: str,
) -> None:
    created = 0
    launches = 0

    def handle_sessions(route: Route) -> None:
        nonlocal created, launches
        request = route.request
        path = urlparse(request.url).path.removeprefix("/api/clinical/sessions")
        if request.method == "POST" and path == "":
            created += 1
            _json(route, _session_state("draft"), 201)
        elif request.method == "POST" and path.endswith("/launch"):
            launches += 1
            _json(route, _session_state("queued"), 202)
        elif path.endswith("/status"):
            _json(route, _session_state("queued"))
        elif path.endswith("/input"):
            _json(
                route,
                {
                    "session_id": 9,
                    "status": "queued",
                    "patient_name": None,
                    "visit_date": "2026-07-17",
                    "clinical_input": "Original input",
                    "patient_image_base64": None,
                    "updated_at": "2026-07-17T08:00:00Z",
                },
            )
        else:
            route.continue_()

    page.route(
        "**/api/clinical/validate-input",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=_preflight_body(
                [],
                [
                    _issue("non_blocking", "timed_drug_feasibility_failed", "drugs"),
                    _issue(
                        "non_blocking", "anamnesis_disease_context_sparse", "anamnesis"
                    ),
                ],
            ),
        ),
    )
    page.route("**/api/clinical/sessions**", handle_sessions)
    try:
        page.goto(base_url)
        _fill_valid_fields(page)
        dialog = _open_wizard(page)
        expect(dialog.locator(".wizard-issue")).to_have_count(2)
        for _ in range(2):
            next_button = dialog.get_by_role("button", name="Next")
            expect(next_button).to_be_enabled()
            next_button.click()
        expect(dialog).to_contain_text("Ready with 2 warnings")
        dialog.get_by_role("button", name="Start analysis").click()

        expect(page).to_have_url(re.compile(r"/report/9$"))
        expect(page.get_by_text("Waiting for capacity")).to_be_visible()
        assert created == 1
        assert launches == 1

        page.get_by_role("tab", name="DILI Agent").click()
        expect(page.get_by_label("Clinical Input")).to_have_value("")
        expect(page.get_by_label("Visit Date")).to_have_value("")
    finally:
        page.unroute("**/api/clinical/validate-input")
        page.unroute("**/api/clinical/sessions**", handle_sessions)


###############################################################################
def test_long_preflight_list_scrolls_without_hiding_header_or_actions(
    page: Page,
    base_url: str,
) -> None:
    page.set_viewport_size({"width": 1280, "height": 700})
    warnings = [
        _issue("non_blocking", f"warning_{index}", "clinical_input")
        for index in range(20)
    ]
    page.route(
        "**/api/clinical/validate-input",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=_preflight_body([], warnings),
        ),
    )
    try:
        page.goto(base_url)
        dialog = _open_wizard(page)
        header = dialog.locator(".modal-header")
        footer = dialog.locator(".modal-footer")
        issue_list = dialog.locator('[data-testid="preflight-issue-list"]')
        expect(dialog).to_be_visible()
        expect(header).to_be_visible()
        expect(footer).to_be_visible()
        expect(issue_list.locator(".wizard-issue")).to_have_count(20)

        dimensions = dialog.evaluate(
            "element => ({width: element.getBoundingClientRect().width, height: element.getBoundingClientRect().height, viewportWidth: window.innerWidth, viewportHeight: window.innerHeight})"
        )
        scroll = issue_list.evaluate(
            "element => ({clientHeight: element.clientHeight, scrollHeight: element.scrollHeight})"
        )
        assert dimensions["width"] < dimensions["viewportWidth"]
        assert dimensions["height"] < dimensions["viewportHeight"]
        assert scroll["scrollHeight"] > scroll["clientHeight"]

        issue_list.evaluate("element => { element.scrollTop = element.scrollHeight; }")
        expect(header).to_be_visible()
        expect(footer).to_be_visible()
    finally:
        page.unroute("**/api/clinical/validate-input")


###############################################################################
def test_focus_is_trapped_inside_warning_modal(
    page: Page,
    base_url: str,
) -> None:
    page.route(
        "**/api/clinical/validate-input",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=_preflight_body(
                [],
                [_issue("non_blocking", "missing_timing", "drugs")],
            ),
        ),
    )
    try:
        page.goto(base_url)
        dialog = _open_wizard(page)
        close_button = dialog.get_by_role("button", name="Cancel and return to the workspace")
        next_button = dialog.get_by_role("button", name="Next")
        expect(next_button).to_be_enabled()

        close_button.focus()
        page.keyboard.press("Shift+Tab")
        expect(next_button).to_be_focused()

        page.keyboard.press("Tab")
        expect(close_button).to_be_focused()
    finally:
        page.unroute("**/api/clinical/validate-input")


###############################################################################
def test_escape_cancels_wizard_and_fix_in_input_returns_to_field(
    page: Page,
    base_url: str,
) -> None:
    page.route(
        "**/api/clinical/validate-input",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=_preflight_body(
                [_issue("blocking", "visit_date_missing", "visit_date")],
                [],
            ),
        ),
    )
    try:
        page.goto(base_url)
        _fill_valid_fields(page)
        original_input = page.get_by_label("Clinical Input").input_value()
        dialog = _open_wizard(page)
        expect(dialog.locator(".wizard-issue")).to_have_count(1)

        page.keyboard.press("Escape")
        expect(page.get_by_role("dialog")).to_have_count(0)
        expect(page.get_by_label("Clinical Input")).to_have_value(original_input)

        dialog = _open_wizard(page)
        dialog.get_by_role("button", name="Fix in input").click()
        expect(page.get_by_role("dialog")).to_have_count(0)
        expect(page.get_by_label("Visit Date")).to_be_focused()
    finally:
        page.unroute("**/api/clinical/validate-input")


###############################################################################
def test_seeded_knowledge_base_lets_unmocked_preflight_reach_review(
    page: Page,
    base_url: str,
) -> None:
    jobs_before = page.request.get(f"{base_url}/api/inspection/jobs").json()
    page.goto(base_url)
    _fill_valid_fields(page)
    dialog = _open_wizard(page)

    expect(dialog.locator(".wizard-badge.is-blocking")).to_have_count(0)
    for _ in range(2):
        next_button = dialog.get_by_role("button", name="Next")
        expect(next_button).to_be_enabled()
        next_button.click()
    expect(dialog.get_by_role("button", name="Start analysis")).to_be_enabled()
    expect(dialog).not_to_contain_text("blocking issue")
    page.keyboard.press("Escape")
    assert page.request.get(f"{base_url}/api/inspection/jobs").json() == jobs_before
