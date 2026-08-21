from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect


def _capture(page: Page, name: str) -> None:
    capture_dir = os.getenv("UI_POLISH_CAPTURE_DIR")
    if not capture_dir:
        return
    output = Path(capture_dir)
    output.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=output / name, full_page=True)


@pytest.mark.parametrize(
    "color_scheme",
    [pytest.param("light", id="light"), pytest.param("dark", id="dark")],
)
def test_reasoning_positions_and_provider_logos(
    page: Page, base_url: str, color_scheme: str
) -> None:
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.emulate_media(color_scheme=color_scheme)
    page.goto(f"{base_url}/model-config")

    reasoning = page.locator(".model-config-reasoning-level").first
    slider = reasoning.locator("input.model-config-reasoning-slider")
    labels = reasoning.locator(".model-config-reasoning-level-labels > span")
    expect(slider).to_be_visible(timeout=15000)
    expect(slider).to_be_enabled(timeout=15000)
    expect(labels).to_have_count(4)

    slider_box = slider.bounding_box()
    assert slider_box is not None
    label_boxes = [labels.nth(index).bounding_box() for index in range(4)]
    assert all(box is not None for box in label_boxes)
    expected_labels = ["Off", "Low", "Medium", "High"]
    for index, (label_box, expected_label) in enumerate(zip(label_boxes, expected_labels)):
        assert label_box is not None
        expected_center = slider_box["x"] + slider_box["width"] * index / 3
        actual_center = label_box["x"] + label_box["width"] / 2
        assert abs(actual_center - expected_center) <= 1.5, (
            f"{expected_label} label is {actual_center - expected_center:.2f}px "
            "away from its slider position"
        )

    original_value = int(slider.input_value())
    for index, expected_label in enumerate(expected_labels):
        slider.focus()
        slider.press("Home")
        for _ in range(index):
            slider.press("ArrowRight")
        expect(slider).to_have_value(str(index))
        expect(slider).to_have_attribute("aria-valuetext", expected_label)

    provider_sources = {
        "OpenAI": "/logos/openai-blossom-light.svg",
        "Google Gemini": "/logos/google-g.svg",
        "DeepSeek": "/logos/deepseek.svg",
        "Anthropic Claude": "/logos/anthropic.svg",
        "OpenCode": "/logos/opencode.svg",
    }
    provider_rows = page.locator(".model-config-cloud-row")
    expect(provider_rows.first).to_be_visible(timeout=15000)
    expect(provider_rows).to_have_count(5)
    for provider_label, source in provider_sources.items():
        row = provider_rows.filter(has_text=provider_label).first
        image = row.locator("img.model-config-provider-logo")
        expect(image).to_be_visible()
        expect(image).to_have_attribute("src", re.compile(f"{re.escape(source)}$"))
        assert image.evaluate(
            "element => element.complete && element.naturalWidth > 0"
        ), f"{provider_label} logo did not load"

    _capture(page, f"model-config-{color_scheme}-1440x1000.png")

    slider.focus()
    slider.press("Home")
    for _ in range(original_value):
        slider.press("ArrowRight")
    page.wait_for_timeout(350)


@pytest.mark.parametrize(
    "viewport",
    [
        pytest.param({"width": 1440, "height": 1000}, id="desktop"),
        pytest.param({"width": 1100, "height": 768}, id="narrow-desktop"),
    ],
)
@pytest.mark.parametrize(
    "color_scheme",
    [pytest.param("light", id="light"), pytest.param("dark", id="dark")],
)
def test_clinical_session_preview_keeps_report_sections_structured(
    page: Page,
    base_url: str,
    api_base_url: str,
    viewport: dict[str, int],
    color_scheme: str,
) -> None:
    sessions_response = page.request.get(
        f"{api_base_url}/api/inspection/sessions",
        params={"offset": 0, "limit": 1},
    )
    assert sessions_response.status == 200
    payload = sessions_response.json()
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        pytest.skip("No persisted sessions in the isolated test database.")

    page.set_viewport_size(viewport)
    page.emulate_media(color_scheme=color_scheme)
    page.goto(f"{base_url}/clinical-sessions")
    session_row = page.locator(".clinical-session-row").first
    expect(session_row).to_be_visible(timeout=15000)
    session_row.click()

    preview = page.locator(".clinical-session-preview")
    expect(preview).to_be_visible(timeout=15000)
    content = preview.locator(".clinical-session-preview-content")
    side = preview.locator(".clinical-session-preview-side")
    expect(content).to_be_visible()
    expect(side).to_be_visible()

    content_style = content.evaluate(
        "element => ({overflow: getComputedStyle(element).overflow, height: getComputedStyle(element).height})"
    )
    assert content_style["overflow"] == "visible"
    assert content_style["height"] != "0px"
    side_style = side.evaluate(
        "element => ({border: getComputedStyle(element).borderInlineStartWidth, overflow: getComputedStyle(element).overflowY})"
    )
    assert side_style["border"] == "1px"
    assert side_style["overflow"] in {"auto", "scroll", "visible"}

    headings = content.locator("h2, h3")
    if headings.count() > 1:
        second_heading_style = headings.nth(1).evaluate(
            "element => ({border: getComputedStyle(element).borderTopWidth, style: getComputedStyle(element).borderTopStyle})"
        )
        assert second_heading_style == {"border": "1px", "style": "solid"}

    _capture(
        page,
        f"clinical-sessions-{color_scheme}-{viewport['width']}x{viewport['height']}.png",
    )


@pytest.mark.parametrize(
    "viewport",
    [
        pytest.param({"width": 1440, "height": 1000}, id="desktop"),
        pytest.param({"width": 1100, "height": 768}, id="narrow-desktop"),
    ],
)
@pytest.mark.parametrize(
    "color_scheme",
    [pytest.param("light", id="light"), pytest.param("dark", id="dark")],
)
def test_clinical_sessions_date_hint_has_single_aligned_rendering(
    page: Page,
    base_url: str,
    viewport: dict[str, int],
    color_scheme: str,
) -> None:
    page.set_viewport_size(viewport)
    page.emulate_media(color_scheme=color_scheme)
    page.goto(f"{base_url}/clinical-sessions")

    shell = page.locator(".clinical-sessions-date-input-shell")
    date_input = page.get_by_label("Session filter date")
    expect(shell).to_be_visible(timeout=15000)
    expect(date_input).to_be_disabled()

    rendering = date_input.evaluate(
        """element => {
          const shell = element.closest('.clinical-sessions-date-input-shell');
          const shellRect = shell.getBoundingClientRect();
          const after = getComputedStyle(shell, '::after');
          const afterRight = parseFloat(after.left) + parseFloat(after.width);
          const color = getComputedStyle(element).color;
          return {
            nativeTextIsHidden: color === 'transparent' || color === 'rgba(0, 0, 0, 0)',
            hintContent: after.content,
            hintFitsShell: afterRight <= shellRect.width,
          };
        }"""
    )
    assert rendering == {
        "nativeTextIsHidden": True,
        "hintContent": '"dd/mm/yyyy"',
        "hintFitsShell": True,
    }

    page.get_by_label("Date filter mode").select_option("exact")
    expect(date_input).to_be_enabled()
    assert date_input.evaluate(
        "element => getComputedStyle(element).color === 'transparent' || "
        "getComputedStyle(element).color === 'rgba(0, 0, 0, 0)'"
    )
