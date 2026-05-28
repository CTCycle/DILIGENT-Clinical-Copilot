"""
E2E tests for UI navigation and key UI workflows.
"""

import re

import pytest
from playwright.sync_api import Page, Route, expect


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


def _build_clinical_job_payload() -> dict:
    return {
        "name": "Marco Rossi",
        "visit_date": {"day": 20, "month": 4, "year": 2026},
        "clinical_input": (
            "## Anamnesis\n"
            + " ".join(
                ["Patient reports persistent fatigue nausea and abdominal discomfort."]
                * 12
            )
            + "\n\n## Therapy\nAmoxicillin 500 mg BID, started on 2026-04-13, stopped on 2026-04-20.\n\n"
            + "## Laboratory history\nALT 210 U/L; AST 180 U/L; ALP 130 U/L."
        ),
        "selected_model_providers": ["openai"],
        "patient_image_base64": None,
        "use_rag": False,
    }


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
        "use_rag": False,
    }


def test_dilu_agent_page_loads(page: Page, base_url: str):
    page.goto(base_url)

    expect(page.get_by_text("DILI Assessment")).to_be_visible()
    expect(page.get_by_label("Clinical Input")).to_be_visible()
    expect(page.get_by_label("Patient Name")).to_be_visible()
    expect(page.get_by_role("button", name="Run DILI analysis")).to_be_visible()


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

    page.goto(f"{base_url}/model-config")
    page.wait_for_timeout(1500)

    assert console_errors == []
    assert failed_requests == []


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


def test_timetable_initial_load_has_no_console_errors_or_failed_requests(
    page: Page, base_url: str, api_base_url: str
):
    sessions_response = page.request.get(
        f"{api_base_url}/api/inspection/sessions",
        params={"offset": 0, "limit": 1},
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
            timeline_response = page.request.get(
                f"{api_base_url}/api/inspection/sessions/{candidate_id}/timeline"
            )
            if timeline_response.status == 200:
                session_id = candidate_id
                break
    if session_id is None:
        pytest.skip("No persisted session with a generated timetable is available.")

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
            "Data Inspection",
            "Model Configurations",
        ):
            return
    raise AssertionError(
        "Keyboard tab traversal did not reach a primary navigation tab."
    )


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


def test_model_config_navigation(page: Page, base_url: str):
    page.goto(base_url)

    model_button = page.get_by_role("tab", name="Model Configurations")
    expect(model_button).to_be_visible()
    model_button.click()

    expect(page).to_have_url(re.compile(r"/model-config/?$"))
    expect(page.get_by_role("heading", name="Model Configurations")).to_be_visible()


def test_data_inspection_navigation(page: Page, base_url: str):
    page.goto(base_url)

    data_button = page.get_by_role("tab", name="Data Inspection")
    expect(data_button).to_be_visible()
    data_button.click()
    expect(page).to_have_url(re.compile(r"/data/?$"))


def test_dili_form_state_restores_after_refresh(page: Page, base_url: str):
    page.goto(base_url)

    page.get_by_label("Patient Name").fill("Persisted Natural")
    page.get_by_label("Visit Date").fill("2026-05-02")
    page.get_by_label("Clinical Input").fill(
        " ".join(["Persistent form content for reload validation."] * 16)
    )
    page.reload()

    expect(page.get_by_label("Patient Name")).to_have_value("Persisted Natural")
    expect(page.get_by_label("Visit Date")).to_have_value("2026-05-02")
    expect(page.get_by_label("Clinical Input")).to_have_value(
        re.compile(r"Persistent form content for reload validation\.")
    )


def test_dili_run_burst_click_submits_single_job(
    page: Page, base_url: str, api_base_url: str
):
    runtime_reset = page.request.put(
        f"{api_base_url}/api/model-config",
        data={"use_cloud_services": False},
    )
    assert runtime_reset.status == 200

    page.goto(base_url)
    _fill_required_dili_fields(page)

    run_button = page.get_by_role("button", name="Run DILI analysis")
    expect(run_button).to_be_enabled(timeout=10000)

    submission_count = 0

    def count_submissions(route: Route) -> None:
        nonlocal submission_count
        if route.request.method == "POST":
            submission_count += 1
            route.fulfill(
                status=202,
                content_type="application/json",
                body=(
                    '{"job_id":"burst-test","job_type":"clinical","status":"running",'
                    '"message":"Clinical analysis started","poll_interval":0.2}'
                ),
            )
            return
        route.continue_()

    page.route("**/api/clinical/jobs", count_submissions)
    page.route(
        "**/api/clinical/jobs/burst-test**",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=(
                '{"job_id":"burst-test","job_type":"clinical","status":"completed",'
                '"progress":1,"result":{"report":"Burst test completed."},'
                '"error":null,"created_at":1,"completed_at":2,"version":1}'
            ),
        ),
    )
    page.evaluate(
        """() => {
            const runButton = document.querySelector('button.btn.btn-primary');
            for (let clickIndex = 0; clickIndex < 10; clickIndex += 1) {
                runButton?.click();
            }
        }"""
    )
    page.wait_for_timeout(1200)
    page.unroute("**/api/clinical/jobs/burst-test**")
    page.unroute("**/api/clinical/jobs", count_submissions)

    assert submission_count == 1


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
    submission_count = 0

    def submit_variant(route: Route) -> None:
        nonlocal submission_count
        if route.request.method == "POST":
            submission_count += 1
            route.fulfill(
                status=202,
                content_type="application/json",
                body=(
                    '{"job_id":"variant-headings","job_type":"clinical","status":"running",'
                    '"message":"Clinical analysis started","poll_interval":0.2}'
                ),
            )
            return
        route.continue_()

    page.route("**/api/clinical/jobs", submit_variant)
    page.route(
        "**/api/clinical/jobs/variant-headings**",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=(
                '{"job_id":"variant-headings","job_type":"clinical","status":"completed",'
                '"progress":1,"result":{"report":"Variant heading test completed."},'
                '"error":null,"created_at":1,"completed_at":2,"version":1}'
            ),
        ),
    )
    run_button.click()
    page.wait_for_timeout(1200)
    page.unroute("**/api/clinical/jobs/variant-headings**")
    page.unroute("**/api/clinical/jobs", submit_variant)

    assert submission_count == 1


def test_dili_run_conflict_surfaces_clear_error_message(
    page: Page, base_url: str, api_base_url: str
):
    runtime_reset = page.request.put(
        f"{api_base_url}/api/model-config",
        data={"use_cloud_services": True, "cloud_model": "gpt-4.1-mini"},
    )
    assert runtime_reset.status == 200

    def mock_conflict(route: Route) -> None:
        route.fulfill(
            status=409,
            content_type="application/json",
            body='{"detail":"Another operation is already running."}',
        )

    page.route("**/api/clinical/jobs", mock_conflict)
    try:
        page.goto(base_url)
        _fill_required_dili_fields(page)
        run_button = page.get_by_role("button", name="Run DILI analysis")
        expect(run_button).to_be_enabled(timeout=10000)
        run_button.click()
        expect(page.locator(".inspection-excerpt-text")).to_contain_text(
            "Another operation is already running. Please wait and retry."
        )
        expect(run_button).to_be_enabled(timeout=10000)
    finally:
        page.unroute("**/api/clinical/jobs", mock_conflict)


def test_model_config_runtime_toggle_enables_save_and_submits_put(
    page: Page, base_url: str, api_base_url: str
):
    runtime_seed = page.request.put(
        f"{api_base_url}/api/model-config",
        data={"use_cloud_services": True, "cloud_model": "gpt-4.1-mini"},
    )
    assert runtime_seed.status == 200

    page.goto(base_url)
    page.get_by_role("tab", name="Model Configurations").click()
    expect(page).to_have_url(re.compile(r"/model-config/?$"))

    runtime_options = page.locator("label.model-config-runtime-option")
    local_option = runtime_options.nth(0)
    cloud_option = runtime_options.nth(1)
    save_button = page.get_by_role("button", name="Save Configuration")

    # Toggle to local mode and verify the save action eventually becomes available.
    local_option.click()
    expect(local_option).to_have_class(re.compile(r"\bis-active\b"))
    expect(save_button).to_be_enabled(timeout=10000)

    # Return to cloud mode and tweak temperature to produce a deterministic save payload.
    cloud_option.click()
    expect(cloud_option).to_have_class(re.compile(r"\bis-active\b"))
    temperature_input = page.locator("#global-temperature-input")
    current_temperature_raw = temperature_input.input_value().strip()
    try:
        current_temperature = float(current_temperature_raw)
    except ValueError:
        current_temperature = 0.7
    next_temperature = 1.99 if abs(current_temperature - 1.99) > 1e-6 else 1.98
    temperature_input.fill(f"{next_temperature:.2f}")
    expect(save_button).to_be_enabled(timeout=15000)

    with page.expect_response(
        lambda response: (
            response.url.endswith("/api/model-config")
            and response.request.method == "PUT"
            and response.status == 200
        )
    ):
        save_button.click()


def test_timetable_route_load_does_not_autogenerate_timeline(
    page: Page, base_url: str, api_base_url: str
):
    sessions_response = page.request.get(
        f"{api_base_url}/api/inspection/sessions",
        params={"offset": 0, "limit": 1},
    )
    assert sessions_response.status == 200
    sessions_payload = sessions_response.json()
    items = (
        sessions_payload.get("items") if isinstance(sessions_payload, dict) else None
    )
    assert isinstance(items, list) and items, (
        "No sessions available for timetable route test."
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
            and request.url.endswith("/timeline")
        ):
            timeline_post_count += 1
        route.continue_()

    page.route("**/api/inspection/sessions/*/timeline", count_timeline_posts)
    page.goto(f"{base_url}/sessions/{session_id}/timetable")
    page.wait_for_timeout(1200)
    page.unroute("**/api/inspection/sessions/*/timeline", count_timeline_posts)

    assert timeline_post_count == 0


def test_timetable_invalid_session_id_shows_validation_error(page: Page, base_url: str):
    page.goto(f"{base_url}/sessions/0/timetable")
    expect(page.locator(".error-note")).to_contain_text("Invalid session id.")


def test_home_form_state_persists_across_back_forward_navigation(
    page: Page, base_url: str
):
    page.goto(base_url)
    page.get_by_label("Patient Name").fill("Back Forward Persist")
    page.get_by_label("Visit Date").fill("2026-05-24")
    page.get_by_label("Clinical Input").fill(
        "Navigation persistence verification input."
    )

    page.get_by_role("tab", name="Model Configurations").click()
    expect(page).to_have_url(re.compile(r"/model-config/?$"))

    page.go_back()
    expect(page).to_have_url(re.compile(r"/?$"))
    expect(page.get_by_label("Patient Name")).to_have_value("Back Forward Persist")
    expect(page.get_by_label("Visit Date")).to_have_value("2026-05-24")
    expect(page.get_by_label("Clinical Input")).to_have_value(
        "Navigation persistence verification input."
    )

    page.go_forward()
    expect(page).to_have_url(re.compile(r"/model-config/?$"))


def test_clinical_sessions_row_selection_loads_matching_detail(
    page: Page, base_url: str
):
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
        session_id_match = re.search(r"Session\s+(\d+)", row_label)
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
