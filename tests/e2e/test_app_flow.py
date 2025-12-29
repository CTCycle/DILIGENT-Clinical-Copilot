"""
E2E tests for UI navigation and key UI workflows.
"""
import re

from playwright.sync_api import Page, expect


def test_dilu_agent_page_loads(page: Page, base_url: str):
    page.goto(base_url)

    expect(
        page.get_by_role("heading", name="Drug-Induced Liver Injury analysis")
    ).to_be_visible()
    expect(page.get_by_label("Anamnesis")).to_be_visible()
    expect(page.get_by_label("Current Drugs")).to_be_visible()
    expect(page.get_by_label("Patient Name")).to_be_visible()
    expect(page.get_by_role("button", name="Run DILI analysis")).to_be_visible()


def test_model_config_modal_opens_and_closes(page: Page, base_url: str):
    page.goto(base_url)

    page.get_by_role("button", name="Model Configurations").click()
    modal = page.locator(".modal-container")
    expect(modal).to_be_visible()
    expect(modal.get_by_text("Model Configurations")).to_be_visible()

    page.get_by_role("button", name="Close configuration modal").click()
    expect(page.locator(".modal-container")).to_have_count(0)


def test_navigation_to_database_browser(page: Page, base_url: str):
    page.goto(base_url)

    page.get_by_role("button", name="Database Browser").click()
    expect(page.get_by_role("heading", name="Database Browser")).to_be_visible()
    expect(page.locator("#table-select")).to_be_visible()
    expect(page.locator(".stats-panel")).to_be_visible()


def test_database_browser_refresh_loads_table(page: Page, base_url: str):
    page.goto(base_url)
    page.get_by_role("button", name="Database Browser").click()

    with page.expect_response(re.compile(r".*/browser/sessions.*")) as response_info:
        page.locator("button.refresh-btn").click()
    response = response_info.value
    assert response.ok

    expect(page.locator(".data-table-container, .data-table-empty")).to_be_visible()


def test_database_browser_switch_table(page: Page, base_url: str):
    page.goto(base_url)
    page.get_by_role("button", name="Database Browser").click()

    table_select = page.locator("#table-select")
    with page.expect_response(re.compile(r".*/browser/livertox.*")):
        table_select.select_option("livertox")

    expect(page.locator(".stat-table-name")).to_contain_text("LiverTox Data")
