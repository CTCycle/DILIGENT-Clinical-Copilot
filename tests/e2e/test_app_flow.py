"""
E2E tests for UI navigation and key UI workflows.
"""
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
