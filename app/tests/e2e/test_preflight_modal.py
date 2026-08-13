from __future__ import annotations

import json

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
def test_blocking_preflight_modal_prevents_job_submission(
    page: Page,
    base_url: str,
) -> None:
    submission_count = 0

    def count_jobs(route: Route) -> None:
        nonlocal submission_count
        submission_count += 1
        route.abort()

    page.route(
        "**/api/clinical/validate-input",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=_preflight_body(
                [
                    _issue("blocking", "clinical_input_missing", "clinical_input"),
                    _issue("blocking", "visit_date_missing", "visit_date"),
                ],
                [],
            ),
        ),
    )
    page.route("**/api/clinical/jobs", count_jobs)
    try:
        page.goto(base_url)
        page.get_by_role("button", name="Run DILI analysis").click()

        dialog = page.get_by_role("dialog")
        expect(dialog).to_be_visible()
        expect(dialog).to_contain_text("Cannot start analysis")
        expect(dialog.locator(".preflight-issue")).to_have_count(2)
        expect(dialog.get_by_role("button", name="Continue with limitations")).to_have_count(0)
        assert submission_count == 0
    finally:
        page.unroute("**/api/clinical/validate-input")
        page.unroute("**/api/clinical/jobs", count_jobs)

###############################################################################
def test_warning_acceptance_starts_one_job_and_preserves_form(
    page: Page,
    base_url: str,
) -> None:
    submission_count = 0

    def start_job(route: Route) -> None:
        nonlocal submission_count
        submission_count += 1
        route.fulfill(
            status=202,
            content_type="application/json",
            body=json.dumps(
                {
                    "job_id": "preflight-warning",
                    "job_type": "clinical",
                    "status": "running",
                    "message": "Clinical analysis started",
                    "poll_interval": 0.2,
                }
            ),
        )

    page.route(
        "**/api/clinical/validate-input",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=_preflight_body(
                [],
                [
                    _issue("non_blocking", "timed_drug_feasibility_failed", "drugs"),
                    _issue("non_blocking", "anamnesis_disease_context_sparse", "anamnesis"),
                ],
            ),
        ),
    )
    page.route("**/api/clinical/jobs", start_job)
    page.route(
        "**/api/clinical/jobs/preflight-warning**",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "job_id": "preflight-warning",
                    "job_type": "clinical",
                    "status": "completed",
                    "progress": 100,
                    "result": {"report": "Pre-flight warning run completed."},
                    "error": None,
                    "created_at": 1,
                    "completed_at": 2,
                    "version": 1,
                }
            ),
        ),
    )
    try:
        page.goto(base_url)
        _fill_valid_fields(page)
        original_input = page.get_by_label("Clinical Input").input_value()
        page.get_by_role("button", name="Run DILI analysis").click()

        dialog = page.get_by_role("dialog")
        expect(dialog).to_contain_text("Review before continuing")
        continue_button = dialog.get_by_role("button", name="Continue with limitations")
        continue_button.click()
        page.wait_for_timeout(50)

        expect(page.locator(".inspection-excerpt-text")).to_contain_text(
            "Pre-flight warning run completed."
        )
        assert submission_count == 1
        expect(page.get_by_label("Clinical Input")).to_have_value(original_input)
    finally:
        page.unroute("**/api/clinical/validate-input")
        page.unroute("**/api/clinical/jobs")
        page.unroute("**/api/clinical/jobs/preflight-warning**")

###############################################################################
def test_long_preflight_list_scrolls_without_hiding_header_or_actions(
    page: Page,
    base_url: str,
) -> None:
    page.set_viewport_size({"width": 420, "height": 700})
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
        page.get_by_role("button", name="Run DILI analysis").click()

        dialog = page.get_by_role("dialog")
        header = dialog.locator(".modal-header")
        footer = dialog.locator(".modal-footer")
        issue_list = dialog.locator('[data-testid="preflight-issue-list"]')
        expect(dialog).to_be_visible()
        expect(header).to_be_visible()
        expect(footer).to_be_visible()
        expect(issue_list.locator(".preflight-issue")).to_have_count(20)

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
        page.get_by_role("button", name="Run DILI analysis").click()

        dialog = page.get_by_role("dialog")
        close_button = dialog.get_by_role("button", name="Return to clinical input")
        continue_button = dialog.get_by_role(
            "button", name="Continue with limitations"
        )
        expect(dialog).to_be_visible()

        close_button.focus()
        page.keyboard.press("Shift+Tab")
        expect(continue_button).to_be_focused()

        page.keyboard.press("Tab")
        expect(close_button).to_be_focused()
    finally:
        page.unroute("**/api/clinical/validate-input")

###############################################################################
def test_escape_does_not_close_blocking_modal(
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
        page.get_by_role("button", name="Run DILI analysis").click()
        expect(page.get_by_role("dialog")).to_be_visible()

        page.keyboard.press("Escape")

        expect(page.get_by_role("dialog")).to_be_visible()
        page.get_by_role("button", name="Return to input").click()
        expect(page.get_by_role("dialog")).to_have_count(0)
        expect(page.get_by_label("Visit Date")).to_be_focused()
    finally:
        page.unroute("**/api/clinical/validate-input")
