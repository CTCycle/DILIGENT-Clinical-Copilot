"""
E2E tests for UI navigation and key UI workflows.
"""

import re

from playwright.sync_api import Page, Route, expect


def _fill_required_dili_fields(page: Page) -> None:
    page.get_by_label("Clinical Input").fill(
        "## Anamnesis\nAcute abdominal discomfort with fatigue.\n\n"
        "## Therapy\nAmoxicillin 500 mg BID, started on 2026-04-13, stopped on 2026-04-20.\n\n"
        "## Laboratory Analysis\nALT 210 U/L; AST 180 U/L; ALP 130 U/L."
    )
    page.get_by_label("Patient Name").fill("Marco Rossi")
    page.get_by_label("Visit Date").fill("2026-04-20")


def test_dilu_agent_page_loads(page: Page, base_url: str):
    page.goto(base_url)

    expect(page.get_by_text("DILI Assessment")).to_be_visible()
    expect(page.get_by_label("Clinical Input")).to_be_visible()
    expect(page.get_by_label("Patient Name")).to_be_visible()
    expect(page.get_by_role("button", name="Run DILI analysis")).to_be_visible()


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


def test_dili_report_state_restores_after_refresh(page: Page, base_url: str):
    page.goto(base_url)
    page.evaluate(
        """() => {
            sessionStorage.setItem("dili-agent-state-v1", JSON.stringify({
                settings: {
                    useCloudServices: false,
                    provider: "openai",
                    cloudModel: "gpt-5.4-mini",
                    textExtractionModel: "qwen3:1.7b",
                    clinicalModel: "gpt-oss:20b",
                    temperature: 0.7,
                    reasoning: false
                },
                form: {
                    patientName: "Persisted User",
                    visitDate: "2026-04-20",
                    patientImageDataUrl: null,
                    clinicalInput: "Persisted clinical input",
                    useRag: false,
                    useWebSearch: false
                },
                message: "Persisted clinical report body",
                jobStatus: "completed",
                isExpanded: false
            }));
        }"""
    )
    page.reload()

    expect(page.locator(".inspection-excerpt-text")).to_contain_text(
        "Persisted clinical report body"
    )
    expect(page.get_by_label("Patient Name")).to_have_value("Persisted User")
    expect(page.get_by_role("button", name="Download markdown")).to_be_enabled()


def test_dili_run_burst_click_submits_single_job(page: Page, base_url: str):
    page.goto(base_url)
    _fill_required_dili_fields(page)

    submission_count = 0

    def count_submissions(route: Route) -> None:
        nonlocal submission_count
        if route.request.method == "POST":
            submission_count += 1
        route.continue_()

    page.route("**/api/clinical/jobs", count_submissions)
    page.evaluate(
        """() => {
            const runButton = Array.from(document.querySelectorAll("button")).find(
                (button) => button.textContent?.includes("Run DILI analysis")
            );
            for (let clickIndex = 0; clickIndex < 10; clickIndex += 1) {
                runButton?.click();
            }
        }"""
    )
    page.wait_for_timeout(1200)
    page.unroute("**/api/clinical/jobs", count_submissions)

    assert submission_count == 1
