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


def test_model_config_navigation(page: Page, base_url: str):
    page.goto(base_url)

    model_button = page.get_by_role("button", name="Model Configurations")
    expect(model_button).to_be_visible()
    model_button.click()

    expect(page).to_have_url(re.compile(r"/model-config/?$"))
    expect(page.get_by_role("heading", name="Model Configurations")).to_be_visible()


def test_data_inspection_navigation(page: Page, base_url: str):
    page.goto(base_url)

    data_button = page.get_by_role("button", name="Data Inspection")
    expect(data_button).to_be_visible()
    data_button.click()
    expect(page).to_have_url(re.compile(r"/data/?$"))
