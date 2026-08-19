from __future__ import annotations

import os
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect


GUIDANCE_STORAGE_KEY = "diligent-guidance-state-v1"

###############################################################################
def _reset_guidance(page: Page) -> None:
    page.evaluate(f"localStorage.removeItem('{GUIDANCE_STORAGE_KEY}')")
    page.reload()

###############################################################################
def test_dili_first_use_callout_is_dismissible_and_persists(
    page: Page,
    base_url: str,
) -> None:
    page.goto(base_url)
    _reset_guidance(page)

    callout = page.get_by_role("note", name="Get started with DILI Agent")
    expect(callout).to_be_visible()
    expect(callout.get_by_role("button", name="Show me")).to_be_visible()
    expect(callout.get_by_role("button", name="Open Configurations")).to_be_visible()

    callout.get_by_role("button", name="Dismiss this tip").click()
    expect(page.get_by_role("note", name="Get started with DILI Agent")).to_have_count(0)

    page.reload()
    expect(page.get_by_role("note", name="Get started with DILI Agent")).to_have_count(0)

###############################################################################
def test_help_reopens_tips_and_restores_focus(
    page: Page,
    base_url: str,
) -> None:
    page.goto(base_url)
    help_button = page.get_by_role("button", name="Open Help and Tips")
    help_button.click()

    dialog = page.get_by_role("dialog", name="Tips & Tricks")
    expect(dialog).to_be_visible()
    expect(dialog).to_contain_text("Run a first assessment")
    dialog.get_by_role("button", name="Close Tips & Tricks").click()
    expect(dialog).to_have_count(0)
    expect(help_button).to_be_focused()

###############################################################################
def test_dili_walkthrough_supports_back_skip_and_finish(
    page: Page,
    base_url: str,
) -> None:
    page.goto(base_url)
    _reset_guidance(page)
    callout = page.get_by_role("note", name="Get started with DILI Agent")
    callout.get_by_role("button", name="Show me").click()

    tour = page.locator('[role="dialog"][aria-label="Run a DILI assessment"]')
    expect(tour).to_have_attribute("aria-label", "Run a DILI assessment")
    expect(tour).to_contain_text("1 of 3")
    expect(tour).to_contain_text("Start with the clinical input")
    tour.get_by_role("button", name="Next").click()
    expect(tour).to_contain_text("2 of 3")
    tour.get_by_role("button", name="Back").click()
    expect(tour).to_contain_text("1 of 3")
    tour.get_by_role("button", name="Next").click()
    tour.get_by_role("button", name="Skip walkthrough").click()
    expect(page.locator('[role="dialog"][aria-label="Run a DILI assessment"]')).to_have_count(0)

    # Help is a manual restart path after the first-use tip has been seen.
    page.get_by_role("button", name="Open Help and Tips").click()
    page.get_by_role("dialog", name="Tips & Tricks").get_by_role("button", name="Show me").click()
    tour = page.locator('[role="dialog"][aria-label="Run a DILI assessment"]')
    expect(tour).to_contain_text("1 of 3")
    tour.get_by_role("button", name="Next").click()
    tour.get_by_role("button", name="Next").click()
    tour.get_by_role("button", name="Finish").click()
    expect(page.locator('[role="dialog"][aria-label="Run a DILI assessment"]')).to_have_count(0)

###############################################################################
def test_model_configuration_help_popovers_are_keyboard_dismissible(
    page: Page,
    base_url: str,
) -> None:
    page.goto(f"{base_url}/model-config")
    runtime_help = page.get_by_role("button", name="Explain runtime source")
    runtime_help.click()
    runtime_panel = page.get_by_role("dialog", name="Choose where new work runs")
    expect(runtime_panel).to_be_visible()
    page.keyboard.press("Escape")
    expect(runtime_panel).to_have_count(0)
    expect(runtime_help).to_be_focused()

    rag_help = page.get_by_role("button", name="Explain RAG settings")
    rag_help.click()
    expect(page.get_by_role("dialog", name="What RAG settings control")).to_be_visible()

###############################################################################
def test_capture_guidance_screenshots_when_requested(
    page: Page,
    base_url: str,
) -> None:
    capture_dir = os.getenv("GUIDANCE_CAPTURE_DIR")
    if not capture_dir:
        pytest.skip("Set GUIDANCE_CAPTURE_DIR to capture visual QA screenshots.")
    output = Path(capture_dir)
    output.mkdir(parents=True, exist_ok=True)

    page.goto(base_url)
    _reset_guidance(page)
    page.get_by_role("note", name="Get started with DILI Agent").screenshot(path=output / "01-dili-first-use.png")
    page.get_by_role("button", name="Open Help and Tips").click()
    page.get_by_role("dialog", name="Tips & Tricks").screenshot(path=output / "02-help-tips.png")
    page.get_by_role("dialog", name="Tips & Tricks").get_by_role("button", name="Show me").click()
    tour = page.locator('[role="dialog"][aria-label="Run a DILI assessment"]')
    tour.screenshot(path=output / "03-tour-step-1.png")
    tour.get_by_role("button", name="Next").click()
    tour.screenshot(path=output / "04-tour-step-2.png")
    tour.get_by_role("button", name="Next").click()
    tour.screenshot(path=output / "05-tour-step-3.png")
    tour.get_by_role("button", name="Skip walkthrough").click()

    page.goto(f"{base_url}/model-config")
    page.get_by_role("button", name="Explain runtime source").click()
    page.get_by_role("dialog", name="Choose where new work runs").screenshot(path=output / "06-model-runtime-popover.png")
    page.keyboard.press("Escape")
    page.get_by_role("button", name="Explain RAG settings").click()
    page.get_by_role("dialog", name="What RAG settings control").screenshot(path=output / "07-model-rag-popover.png")

    page.set_viewport_size({"width": 420, "height": 860})
    page.emulate_media(color_scheme="dark")
    page.goto(base_url)
    _reset_guidance(page)
    page.get_by_role("note", name="Get started with DILI Agent").screenshot(path=output / "08-dili-mobile-dark.png")

    # These two captures depend on a populated live inspection backend. Keep the
    # optional capture resilient for clean installations with no saved sessions.
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.emulate_media(color_scheme="light")
    page.goto(f"{base_url}/clinical-sessions")
    session_row = page.locator(".clinical-session-row").first
    if session_row.count():
        session_row.click()
        tabs_help = page.get_by_role("button", name="Explain session sections")
        if tabs_help.count():
            tabs_help.click()
            page.get_by_role("dialog", name="How to use these sections").screenshot(path=output / "09-session-tabs-popover.png")
    page.goto(f"{base_url}/sessions/13/timetable/10")
    review_help = page.get_by_role("button", name="Explain timeline review controls")
    if review_help.count():
        review_help.click()
        page.get_by_role("dialog", name="Review the chronology deliberately").screenshot(path=output / "10-timeline-review-popover.png")
