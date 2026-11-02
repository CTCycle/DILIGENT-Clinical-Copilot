# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""
E2E tests for UI navigation and key UI workflows.
"""

import json
import re
from urllib.parse import urlparse

from playwright.sync_api import Page, Route, expect


###############################################################################
def _fill_required_dili_fields(page: Page) -> None:
    page.get_by_label("Clinical Input").fill(
        "## Anamnesis\n"
        + " ".join(
            ["Patient reports persistent fatigue nausea and abdominal discomfort."] * 12
        )
        + "\n\n## Therapy\nAmoxicillin 500 mg BID, started on 2026-04-13, stopped on 2026-04-20.\n\n"
        + "## Laboratory history\nALT 210 U/L; AST 180 U/L; ALP 130 U/L."
    )
    page.get_by_label("Patient Name").fill("Marco Rossi")
    page.get_by_label("Visit Date").fill("2026-04-20")


###############################################################################
def _build_variant_heading_payload() -> dict:
    return {
        "name": "Marco Rossi",
        "visit_date": {"day": 20, "month": 4, "year": 2026},
        "clinical_input": (
            "## Clinical history\n"
            + " ".join(
                ["Patient reports persistent fatigue nausea and abdominal discomfort."]
                * 12
            )
            + "\n\n## Current medications\nAmoxicillin 500 mg BID, started on 2026-04-13, stopped on 2026-04-20.\n\n"
            + "## Laboratory tests\nALT 210 U/L; AST 180 U/L; ALP 130 U/L."
        ),
        "selected_model_providers": ["openai"],
        "patient_image_base64": None,
        "evidence_sources": ["livertox", "dilirank"],
    }


###############################################################################
def _mock_ready_preflight(page: Page) -> None:
    page.route(
        "**/api/clinical/validate-input",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=(
                '{"ready":true,"blocking_issues":[],"non_blocking_issues":[],'
                '"runtime_settings":{},"extraction_quality":{},'
                '"deterministic_diagnostics":{},'
                '"rag_readiness":{"requested":false,"available":true,'
                '"backend":"disabled","model":null,"reason_code":null,"message":null}}'
            ),
        ),
    )


###############################################################################
def _start_analysis_from_wizard(page: Page) -> None:
    page.get_by_role("button", name="Run DILI analysis").click()
    dialog = page.get_by_role("dialog", name="Prepare analysis")
    for _ in range(3):
        next_button = dialog.get_by_role("button", name="Next")
        expect(next_button).to_be_enabled(timeout=10000)
        next_button.click()
    dialog.get_by_role("button", name="Start analysis").click()


###############################################################################
def test_dilu_agent_page_loads(page: Page, base_url: str):
    page.goto(base_url)

    expect(page.get_by_label("Clinical Input")).to_be_visible()
    expect(page.get_by_label("Patient Name")).to_be_visible()
    expect(page.get_by_role("button", name="Run DILI analysis")).to_be_visible()


###############################################################################
def test_home_initial_load_has_no_console_errors_or_failed_requests(
    page: Page, base_url: str
):
    console_errors: list[str] = []
    failed_requests: list[str] = []

    page.on(
        "console",
        lambda message: (
            console_errors.append(message.text) if message.type == "error" else None
        ),
    )
    page.on(
        "requestfailed",
        lambda request: failed_requests.append(
            f"{request.method} {request.url} :: {request.failure}"
        ),
    )

    page.goto(base_url)
    page.wait_for_timeout(1500)

    assert console_errors == []
    assert failed_requests == []


###############################################################################
def test_model_config_initial_load_has_no_console_errors_or_failed_requests(
    page: Page, base_url: str
):
    console_errors: list[str] = []
    failed_requests: list[str] = []

    page.on(
        "console",
        lambda message: (
            console_errors.append(message.text) if message.type == "error" else None
        ),
    )
    page.on(
        "requestfailed",
        lambda request: failed_requests.append(
            f"{request.method} {request.url} :: {request.failure}"
        ),
    )

    page.goto(f"{base_url}/settings/models")
    page.wait_for_timeout(1500)

    assert console_errors == []
    assert failed_requests == []


###############################################################################
def test_data_inspection_initial_load_has_no_console_errors_or_failed_requests(
    page: Page, base_url: str
):
    console_errors: list[str] = []
    failed_requests: list[str] = []

    page.on(
        "console",
        lambda message: (
            console_errors.append(message.text) if message.type == "error" else None
        ),
    )
    page.on(
        "requestfailed",
        lambda request: failed_requests.append(
            f"{request.method} {request.url} :: {request.failure}"
        ),
    )

    page.goto(f"{base_url}/data")
    page.wait_for_timeout(1500)

    assert console_errors == []
    assert failed_requests == []


###############################################################################
def test_clinical_sessions_initial_load_has_no_console_errors_or_failed_requests(
    page: Page, base_url: str
):
    console_errors: list[str] = []
    failed_requests: list[str] = []

    page.on(
        "console",
        lambda message: (
            console_errors.append(message.text) if message.type == "error" else None
        ),
    )
    page.on(
        "requestfailed",
        lambda request: failed_requests.append(
            f"{request.method} {request.url} :: {request.failure}"
        ),
    )

    page.goto(f"{base_url}/clinical-sessions")
    page.wait_for_timeout(1500)

    assert console_errors == []
    assert failed_requests == []


###############################################################################
def test_timetable_initial_load_has_no_console_errors_or_failed_requests(
    page: Page, base_url: str, api_base_url: str
):
    sessions_response = page.request.get(
        f"{api_base_url}/api/inspection/sessions",
        params={"offset": 0, "limit": 100},
    )
    assert sessions_response.status == 200
    sessions_payload = sessions_response.json()
    items = (
        sessions_payload.get("items") if isinstance(sessions_payload, dict) else None
    )
    session_id: int | None = None
    if isinstance(items, list):
        for item in items:
            candidate_id = item.get("session_id") if isinstance(item, dict) else None
            if not isinstance(candidate_id, int) or candidate_id <= 0:
                continue
            if not item.get("has_timeline"):
                continue
            timelines_response = page.request.get(
                f"{api_base_url}/api/inspection/sessions/{candidate_id}/timelines"
            )
            if timelines_response.status != 200:
                continue
            timelines_payload = timelines_response.json()
            timeline_items = (
                timelines_payload.get("items", [])
                if isinstance(timelines_payload, dict)
                else []
            )
            latest = (
                timeline_items[0]
                if isinstance(timeline_items, list) and timeline_items
                else None
            )
            timeline_id = (
                latest.get("timeline_id") if isinstance(latest, dict) else None
            )
            if isinstance(timeline_id, int) and timeline_id > 0:
                timeline_response = page.request.get(
                    f"{api_base_url}/api/inspection/sessions/{candidate_id}/timelines/{timeline_id}"
                )
            else:
                timeline_response = None
            if timeline_response is not None and timeline_response.status == 200:
                session_id = candidate_id
                break
    assert session_id is not None, (
        "The browser regression seed must provide a persisted session with a "
        "generated timetable."
    )

    console_errors: list[str] = []
    failed_requests: list[str] = []

    page.on(
        "console",
        lambda message: (
            console_errors.append(message.text) if message.type == "error" else None
        ),
    )
    page.on(
        "requestfailed",
        lambda request: failed_requests.append(
            f"{request.method} {request.url} :: {request.failure}"
        ),
    )

    page.goto(f"{base_url}/sessions/{session_id}/timetable")
    page.wait_for_timeout(1500)

    assert console_errors == []
    assert failed_requests == []


###############################################################################
def test_keyboard_navigation_reaches_primary_tabs(page: Page, base_url: str):
    page.goto(base_url)

    # Move focus using keyboard only until a top-level tab receives focus.
    for _ in range(24):
        page.keyboard.press("Tab")
        focused_role = page.evaluate(
            "() => document.activeElement?.getAttribute('role') || ''"
        )
        focused_text = page.evaluate(
            "() => (document.activeElement?.textContent || '').replace(/\\s+/g, ' ').trim()"
        )
        if focused_role == "tab" and focused_text in (
            "DILI Agent",
            "Clinical Sessions",
            "Knowledge",
            "Settings",
        ):
            return
    raise AssertionError(
        "Keyboard tab traversal did not reach a primary navigation tab."
    )


###############################################################################
def test_primary_tab_arrow_navigation_moves_focus_with_selection(
    page: Page, base_url: str
):
    page.goto(base_url)

    page.focus('[role="tab"][aria-selected="true"]')
    page.keyboard.press("ArrowRight")

    clinical_sessions = page.get_by_role("tab", name="Clinical Sessions")
    expect(clinical_sessions).to_have_attribute("aria-selected", "true")
    expect(clinical_sessions).to_be_focused()


###############################################################################
def test_home_form_labels_are_associated_with_inputs(page: Page, base_url: str):
    page.goto(base_url)

    patient_name = page.get_by_label("Patient Name")
    visit_date = page.get_by_label("Visit Date")
    clinical_input = page.get_by_label("Clinical Input")

    expect(patient_name).to_be_visible()
    expect(visit_date).to_be_visible()
    expect(clinical_input).to_be_visible()

    # Basic accessibility operability: controls must be focusable and reachable.
    patient_name.click()
    expect(patient_name).to_be_focused()

    visit_date.click()
    expect(visit_date).to_be_focused()


###############################################################################
def test_keyboard_tab_traversal_reaches_home_form_controls(page: Page, base_url: str):
    page.goto(base_url)

    clinical_input = page.locator("#clinical-input")
    patient_name = page.locator("#patient-name")
    visit_date = page.locator("#visit-date")

    page.focus("#clinical-input")
    expect(clinical_input).to_be_focused()

    page.focus("#patient-name")
    expect(patient_name).to_be_focused()

    page.keyboard.press("Tab")
    expect(visit_date).to_be_focused()

    page.keyboard.press("Shift+Tab")
    expect(patient_name).to_be_focused()


###############################################################################
def test_model_config_navigation(page: Page, base_url: str):
    page.goto(base_url)

    model_button = page.get_by_role("tab", name="Settings")
    expect(model_button).to_be_visible()
    model_button.click()
    page.get_by_role("link", name="Models").click()

    expect(page).to_have_url(re.compile(r"/settings/models/?$"))
    expect(page.get_by_role("heading", name="Runtime Source")).to_be_visible()


###############################################################################
def test_primary_navigation_restores_scroll_position(page: Page, base_url: str):
    page.goto(f"{base_url}/settings/models")
    page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
    page.get_by_role("tab", name="DILI Agent").click()

    expect(page).to_have_url(re.compile(r"/?$"))
    expect(page.get_by_label("Clinical Input")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollTop") < 1


###############################################################################
def test_data_inspection_navigation(page: Page, base_url: str):
    page.goto(base_url)

    data_button = page.get_by_role("tab", name="Knowledge")
    expect(data_button).to_be_visible()
    data_button.click()
    expect(page).to_have_url(re.compile(r"/data/?$"))


###############################################################################
def test_dili_form_state_resets_after_refresh(page: Page, base_url: str):
    page.goto(base_url)

    page.get_by_label("Patient Name").fill("Persisted Natural")
    page.get_by_label("Visit Date").fill("2026-05-02")
    page.get_by_label("Clinical Input").fill(
        " ".join(["Persistent form content for reload validation."] * 16)
    )
    page.reload()

    expect(page.get_by_label("Patient Name")).to_have_value("")
    expect(page.get_by_label("Visit Date")).to_have_value("")
    expect(page.get_by_label("Clinical Input")).to_have_value("")


###############################################################################
def _session_state(status: str, session_id: int = 9, **overrides) -> dict:
    state = {
        "session_id": session_id,
        "status": status,
        "patient_name": "Marco Rossi",
        "visit_date": "2026-04-20",
        "created_at": "2026-04-20T08:00:00Z",
        "updated_at": "2026-04-20T08:00:00Z",
        "queued_at": None,
        "started_at": "2026-04-20T08:00:01Z" if status != "draft" else None,
        "finished_at": None,
        "queue_position": None,
        "progress": None,
        "progress_stage": None,
        "progress_message": None,
        "stop_requested": False,
        "error": None,
        "has_report": False,
        "version": 1,
        "poll_interval": 0.5,
    }
    state.update(overrides)
    return state


###############################################################################
def _json_route(route: Route, body: dict, status: int = 200) -> None:
    route.fulfill(status=status, content_type="application/json", body=json.dumps(body))


###############################################################################
class _SessionMock:
    """Routes the session lifecycle endpoints for a single fake session."""

    # -------------------------------------------------------------------------
    def __init__(self, page: Page, status_for_call, launch_status: int = 202) -> None:
        self.page = page
        self.status_for_call = status_for_call
        self.launch_status = launch_status
        self.creates = 0
        self.launches = 0
        self.status_calls = 0
        self.launch_bodies: list[dict] = []
        page.route("**/api/clinical/sessions**", self._handle)
        page.route(
            "**/api/inspection/sessions/9",
            lambda route: _json_route(
                route,
                {
                    "session_id": 9,
                    "patient_name": "Marco Rossi",
                    "visit_date": "2026-04-20",
                    "session_timestamp": "2026-04-20T08:00:00Z",
                    "version": 1,
                    "original_session_id": None,
                    "status": "completed",
                    "text_extraction_model": "m",
                    "clinical_model": "m",
                    "metadata": {},
                    "sections": {},
                    "session_text": "",
                    "source_clinical_text": "",
                    "result_payload": {},
                    "report": "Recovered report body.",
                    "official_report_text": "Recovered report body.",
                    "manual_edit_history": [],
                },
            ),
        )

    # -------------------------------------------------------------------------
    def _handle(self, route: Route) -> None:
        request = route.request
        path = urlparse(request.url).path.removeprefix("/api/clinical/sessions")
        if request.method == "POST" and path == "":
            self.creates += 1
            _json_route(route, _session_state("draft"), 201)
        elif request.method == "POST" and path.endswith("/launch"):
            self.launches += 1
            self.launch_bodies.append(json.loads(request.post_data or "{}"))
            if self.launch_status == 202:
                _json_route(route, _session_state("queued", queue_position=1), 202)
            else:
                _json_route(
                    route,
                    {"detail": "Another operation is already running."},
                    self.launch_status,
                )
        elif path.endswith("/status"):
            self.status_calls += 1
            _json_route(route, self.status_for_call(self.status_calls))
        elif path.endswith("/input"):
            _json_route(
                route,
                {
                    "session_id": 9,
                    "status": "running",
                    "patient_name": "Marco Rossi",
                    "visit_date": "2026-04-20",
                    "clinical_input": "Original clinical input",
                    "patient_image_base64": None,
                    "updated_at": "2026-04-20T08:00:00Z",
                },
            )
        else:
            route.continue_()

    # -------------------------------------------------------------------------
    def close(self) -> None:
        self.page.unroute("**/api/clinical/sessions**", self._handle)
        self.page.unroute("**/api/inspection/sessions/9")
        self.page.unroute("**/api/clinical/validate-input")


###############################################################################
def test_dili_run_burst_click_submits_single_session(page: Page, base_url: str):
    page.goto(base_url)
    _fill_required_dili_fields(page)
    run_button = page.get_by_role("button", name="Run DILI analysis")
    expect(run_button).to_be_enabled(timeout=10000)

    _mock_ready_preflight(page)
    mock = _SessionMock(page, lambda _call: _session_state("queued", queue_position=1))
    try:
        run_button.click()
        dialog = page.get_by_role("dialog", name="Prepare analysis")
        for _ in range(3):
            next_button = dialog.get_by_role("button", name="Next")
            expect(next_button).to_be_enabled(timeout=10000)
            next_button.click()
        expect(dialog.get_by_role("button", name="Start analysis")).to_be_enabled()
        page.evaluate(
            """() => {
                const startButton = [...document.querySelectorAll('.analysis-wizard-modal button')]
                    .find((button) => button.textContent?.trim() === 'Start analysis');
                for (let clickIndex = 0; clickIndex < 10; clickIndex += 1) {
                    startButton?.click();
                }
            }"""
        )
        expect(page).to_have_url(re.compile(r"/report/9$"))
        page.wait_for_timeout(800)
        assert mock.creates == 1
        assert mock.launches == 1
    finally:
        mock.close()


###############################################################################
def test_report_progress_is_restored_after_navigation_and_refresh(
    page: Page, base_url: str
):
    def status_for_call(call: int) -> dict:
        if call <= 2:
            return _session_state(
                "running",
                progress=18,
                progress_stage="drugs.extracting",
                progress_message="Extracting therapy and medication data",
                version=call,
            )
        if call <= 12:
            return _session_state(
                "running",
                progress=62,
                progress_stage="retrieval.evidence",
                progress_message="Gathering grounded evidence",
                version=call,
            )
        return _session_state("completed", has_report=True, progress=100, version=call)

    _mock_ready_preflight(page)
    mock = _SessionMock(page, status_for_call)
    try:
        page.goto(base_url)
        _fill_required_dili_fields(page)
        _start_analysis_from_wizard(page)
        expect(page).to_have_url(re.compile(r"/report/9$"))
        expect(page.locator(".analysis-progress-detail").first).to_contain_text(
            "Extracting therapy and medication data"
        )

        page.get_by_role("tab", name="Settings").click()
        expect(page).to_have_url(re.compile(r"/settings/"))
        page.go_back()
        expect(page).to_have_url(re.compile(r"/report/9$"))
        expect(page.locator(".analysis-progress-detail").first).to_contain_text(
            "Gathering grounded evidence", timeout=15000
        )

        page.reload()
        expect(page.get_by_text("Recovered report body.")).to_be_visible(timeout=30000)
        expect(page.get_by_test_id("session-status")).to_have_text("Completed")
        assert mock.launches == 1
    finally:
        mock.close()


###############################################################################
def test_dili_submit_accepts_variant_section_headings(
    page: Page, base_url: str
) -> None:
    page.goto(base_url)
    payload = _build_variant_heading_payload()
    page.get_by_label("Patient Name").fill(payload["name"])
    page.get_by_label("Visit Date").fill("2026-04-20")
    page.get_by_label("Clinical Input").fill(payload["clinical_input"])

    run_button = page.get_by_role("button", name="Run DILI analysis")
    expect(run_button).to_be_enabled(timeout=10000)

    _mock_ready_preflight(page)
    mock = _SessionMock(page, lambda _call: _session_state("queued", queue_position=1))
    try:
        _start_analysis_from_wizard(page)
        expect(page).to_have_url(re.compile(r"/report/9$"))
        assert mock.launches == 1
        assert mock.launch_bodies[0]["clinical_input"] == payload["clinical_input"]
    finally:
        mock.close()


###############################################################################
def test_dili_run_conflict_surfaces_clear_error_message(page: Page, base_url: str):
    _mock_ready_preflight(page)
    mock = _SessionMock(page, lambda _call: _session_state("draft"), launch_status=409)
    try:
        page.goto(base_url)
        _fill_required_dili_fields(page)
        _start_analysis_from_wizard(page)
        expect(page.get_by_role("alert").filter(has_text="Another operation")).to_contain_text(
            "Another operation is already running. Please wait and retry."
        )
        expect(
            page.get_by_role("dialog", name="Prepare analysis").get_by_role(
                "button", name="Start analysis"
            )
        ).to_be_enabled(timeout=10000)
        expect(page).not_to_have_url(re.compile(r"/report/"))
        expect(page.get_by_label("Patient Name")).to_have_value("Marco Rossi")
    finally:
        mock.close()


###############################################################################
def test_timetable_route_load_does_not_autogenerate_timeline(
    page: Page, base_url: str, api_base_url: str
):
    sessions_response = page.request.get(
        f"{api_base_url}/api/inspection/sessions",
        params={"offset": 0, "limit": 100},
    )
    assert sessions_response.status == 200
    sessions_payload = sessions_response.json()
    items = (
        sessions_payload.get("items") if isinstance(sessions_payload, dict) else None
    )
    assert isinstance(items, list) and items, (
        "The browser regression seed must provide a persisted session."
    )
    session_id = items[0].get("session_id")
    assert isinstance(session_id, int) and session_id > 0

    timeline_post_count = 0

    def count_timeline_posts(route: Route) -> None:
        nonlocal timeline_post_count
        request = route.request
        if (
            request.method == "POST"
            and "/api/inspection/sessions/" in request.url
            and request.url.endswith("/timelines")
        ):
            timeline_post_count += 1
        route.continue_()

    page.route("**/api/inspection/sessions/*/timelines", count_timeline_posts)
    page.goto(f"{base_url}/sessions/{session_id}/timetable")
    page.wait_for_timeout(1200)
    page.unroute("**/api/inspection/sessions/*/timelines", count_timeline_posts)

    assert timeline_post_count == 0


###############################################################################
def test_timetable_invalid_session_id_shows_validation_error(page: Page, base_url: str):
    page.goto(f"{base_url}/sessions/0/timetable")
    expect(page.locator(".error-note")).to_contain_text("Invalid session id.")


###############################################################################
def test_timetable_collapsible_chronology_and_horizontal_view(
    page: Page, base_url: str
):
    timeline = {
        "timeline_id": 91,
        "session_id": 42,
        "generated_at": "2026-07-22T08:00:00Z",
        "generation_status": "llm_generated",
        "source_model": "test-model",
        "source_kind": "local",
        "model_provider": "ollama",
        "events": [
            {
                "event_id": "december",
                "title": "December baseline",
                "description": "Baseline",
                "event_type": "disease",
                "timing_type": "explicit_date",
                "event_date": "2024-12-01",
                "event_date_end": None,
                "date_precision": "day",
                "date_certainty": "explicit",
                "uncertainty_reason": None,
                "relative_time": None,
                "extracted_timing_text": "December 1",
                "source_evidence": "Baseline on 2024-12-01.",
                "linked_patient_event_ids": [],
                "source": "fixture",
                "confidence": 0.9,
                "confidence_rationale": "Explicit",
                "sort_order": 0,
            },
            *[
                {
                    "event_id": f"therapy-{index}",
                    "title": f"Therapy {index}",
                    "description": "Therapy",
                    "event_type": "therapy",
                    "timing_type": "explicit_date",
                    "event_date": "2025-01-05",
                    "event_date_end": None,
                    "date_precision": "day",
                    "date_certainty": "explicit",
                    "uncertainty_reason": None,
                    "relative_time": None,
                    "extracted_timing_text": "January 5",
                    "source_evidence": "Therapy on 2025-01-05.",
                    "linked_patient_event_ids": [],
                    "source": "fixture",
                    "confidence": 0.9,
                    "confidence_rationale": "Explicit",
                    "sort_order": index,
                }
                for index in range(1, 5)
            ],
            {
                "event_id": "february",
                "title": "February endpoint",
                "description": "Endpoint",
                "event_type": "lab",
                "timing_type": "explicit_date",
                "event_date": "2025-02-01",
                "event_date_end": None,
                "date_precision": "day",
                "date_certainty": "explicit",
                "uncertainty_reason": None,
                "relative_time": None,
                "extracted_timing_text": "February 1",
                "source_evidence": "Endpoint on 2025-02-01.",
                "linked_patient_event_ids": [],
                "source": "fixture",
                "confidence": 0.9,
                "confidence_rationale": "Explicit",
                "sort_order": 5,
            },
        ],
    }
    preview = {
        "timeline_id": 91,
        "session_id": 42,
        "generated_at": timeline["generated_at"],
        "generation_status": "llm_generated",
        "source_model": "test-model",
        "source_kind": "local",
        "model_provider": "ollama",
        "event_count": 6,
        "start_date": "2024-12-01",
        "end_date": "2025-02-01",
        "source_evidence_event_count": 6,
        "missing_evidence_event_count": 0,
        "uncertain_event_count": 0,
        "undated_event_count": 0,
    }

    page.route(
        "**/api/inspection/sessions/42/timelines",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"items":[' + json.dumps(preview) + "]}",
        ),
    )
    page.route(
        "**/api/inspection/sessions/42/timelines/91",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(timeline)
        ),
    )

    page.goto(f"{base_url}/sessions/42/timetable")
    expect(page.get_by_role("heading", name="Patient timeline")).to_be_visible()
    chronology = page.get_by_role("region", name="Clinical chronology")
    expect(chronology).to_be_visible()
    expect(chronology).to_contain_text("December 2024 – February 2025")
    january_group = page.get_by_role("region", name="5 Jan 2025")
    expect(january_group).to_be_visible()
    expect(january_group).to_contain_text("4 events")
    therapy_row = january_group.get_by_role("button", name="Therapy 1", exact=False)
    expect(therapy_row).to_have_attribute("aria-expanded", "false")
    expect(january_group).not_to_contain_text("Therapy on 2025-01-05.")

    therapy_row.click()
    expect(therapy_row).to_have_attribute("aria-expanded", "true")
    expect(january_group).to_contain_text("Therapy on 2025-01-05.")

    page.get_by_role("button", name="Timeline", exact=True).click()
    track = page.get_by_role("region", name="Horizontal clinical timeline")
    expect(track).to_be_visible()
    expect(track).to_contain_text("1 Dec 2024")
    expect(track).to_contain_text("+35 d")
    track.get_by_role("button", name="February endpoint").click()
    expect(chronology).to_contain_text("Endpoint on 2025-02-01.")
    page.keyboard.press("Escape")
    expect(chronology).to_contain_text("Select an event on the timeline")


###############################################################################
def test_home_form_state_persists_across_back_forward_navigation(
    page: Page, base_url: str
):
    page.goto(base_url)
    page.get_by_label("Patient Name").fill("Back Forward Persist")
    page.get_by_label("Visit Date").fill("2026-05-24")
    page.get_by_label("Clinical Input").fill(
        "Navigation persistence verification input."
    )

    page.get_by_role("tab", name="Settings").click()
    page.get_by_role("link", name="Models").click()
    expect(page).to_have_url(re.compile(r"/settings/models/?$"))

    page.go_back()
    expect(page).to_have_url(re.compile(r"/settings/general/?$"))
    page.go_back()
    expect(page).to_have_url(re.compile(r"/?$"))
    expect(page.get_by_label("Patient Name")).to_have_value("Back Forward Persist")
    expect(page.get_by_label("Visit Date")).to_have_value("2026-05-24")
    expect(page.get_by_label("Clinical Input")).to_have_value(
        "Navigation persistence verification input."
    )

    page.go_forward()
    expect(page).to_have_url(re.compile(r"/settings/general/?$"))
    page.go_forward()
    expect(page).to_have_url(re.compile(r"/settings/models/?$"))


###############################################################################
def test_clinical_sessions_row_selection_loads_matching_detail(
    page: Page, base_url: str, api_base_url: str
):
    sessions_response = page.request.get(
        f"{api_base_url}/api/inspection/sessions",
        params={"offset": 0, "limit": 100},
    )
    assert sessions_response.status == 200
    sessions_payload = sessions_response.json()
    items = (
        sessions_payload.get("items") if isinstance(sessions_payload, dict) else None
    )
    assert isinstance(items, list) and items, (
        "The browser regression seed must provide a persisted session."
    )

    detail_request_urls: list[str] = []

    def capture_detail_request(route: Route) -> None:
        detail_request_urls.append(route.request.url)
        route.continue_()

    page.route("**/api/inspection/sessions/*", capture_detail_request)
    try:
        page.goto(f"{base_url}/clinical-sessions")

        session_rows = page.locator(".clinical-session-row")
        expect(session_rows.first).to_be_visible(timeout=15000)
        row_count = session_rows.count()
        target_row = session_rows.nth(1) if row_count >= 2 else session_rows.first

        row_label = target_row.inner_text()
        session_id_match = re.search(r"#(\d+)", row_label)
        assert session_id_match is not None, (
            f"Unable to parse session id from row label: {row_label!r}"
        )
        target_session_id = session_id_match.group(1)

        target_row.click()
        expect(page.locator(".clinical-session-toolbar p")).to_contain_text(
            f"Session {target_session_id}"
        )

        assert any(
            re.search(rf"/api/inspection/sessions/{target_session_id}$", url)
            for url in detail_request_urls
        ), f"No detail request captured for selected session id {target_session_id}."
    finally:
        page.unroute("**/api/inspection/sessions/*", capture_detail_request)
