"""Capture desktop UI states for the 2026-09-25 frontend refinement pass."""

from __future__ import annotations

import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


REPO = Path(r"G:\Projects\Repositories\Active projects\DILIGENT Clinical Copilot")
OUTPUT = REPO / "assets" / "QA" / "ui-consistency-refinement-2026-09-25"
BASE_URL = "http://127.0.0.1:9847"

os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(REPO / "runtimes" / "cache" / "playwright"))

settings = {
    "values": {
        "general": {"polling_interval": 5},
        "data": {
            "drug_name_min_length": 2,
            "drug_name_max_length": 100,
            "drug_name_max_tokens": 512,
        },
        "integrations": {
            "livertox_download_timeout": 45,
            "rxnav_request_timeout": 20,
            "rxnav_max_concurrency": 3,
        },
        "advanced": {
            "default_llm_timeout": 120,
            "clinical_llm_timeout": 120,
            "livertox_llm_timeout": 120,
            "minimum_llm_timeout": 5,
            "cloud_llm_timeout_cap": 300,
            "local_llm_timeout_cap": 600,
            "max_excerpt_length": 12000,
        },
    },
    "source": "settings/configurations.json",
    "environment_editable": False,
    "updated_at": "2026-09-25T06:00:00Z",
}
settings["defaults"] = settings["values"]

event = {
    "event_id": "fixture-therapy-1",
    "title": "Therapy 1",
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
    "source_evidence": "Therapy recorded on 2025-01-05.",
    "linked_patient_event_ids": [],
    "source": "fixture",
    "confidence": 0.9,
    "confidence_rationale": "Explicit date",
    "sort_order": 0,
}
timeline = {
    "timeline_id": 91,
    "session_id": 42,
    "generated_at": "2026-07-22T08:00:00Z",
    "generation_status": "llm_generated",
    "source_model": "test-model",
    "source_kind": "local",
    "model_provider": "ollama",
    "events": [event],
}
timeline_preview = {
    "timeline_id": 91,
    "session_id": 42,
    "generated_at": timeline["generated_at"],
    "generation_status": "llm_generated",
    "source_model": "test-model",
    "source_kind": "local",
    "model_provider": "ollama",
    "event_count": 1,
    "start_date": "2025-01-05",
    "end_date": "2025-01-05",
    "source_evidence_event_count": 1,
    "missing_evidence_event_count": 0,
    "uncertain_event_count": 0,
    "undated_event_count": 0,
}


def capture(page, name: str, path: str, width: int, height: int, *, theme: str | None = None):
    page.set_viewport_size({"width": width, "height": height})
    page.goto(f"{BASE_URL}{path}", wait_until="domcontentloaded", timeout=20000)
    page.locator(".minimum-viewport-notice").wait_for(state="attached", timeout=10000)
    if width < 1100:
        page.locator(".minimum-viewport-notice").wait_for(state="visible", timeout=10000)
    else:
        page.locator(".app-main").wait_for(state="visible", timeout=10000)
    page.wait_for_timeout(350)
    if theme == "dark" and page.locator('button[aria-label="Switch to dark theme"]').count():
        page.get_by_role("button", name="Switch to dark theme").click()
    elif theme == "light" and page.locator('button[aria-label="Switch to light theme"]').count():
        page.get_by_role("button", name="Switch to light theme").click()
    page.wait_for_timeout(100)
    image_path = OUTPUT / name
    page.screenshot(path=str(image_path), full_page=False)
    metrics = page.evaluate(
        """() => {
          const rect = (selector) => {
            const element = document.querySelector(selector);
            if (!element) return null;
            const { x, y, width, height } = element.getBoundingClientRect();
            return { x, y, width, height };
          };
          return {
            route: location.pathname,
            theme: document.documentElement.dataset.theme || 'light-default',
            viewport: { width: innerWidth, height: innerHeight },
            document: {
              clientWidth: document.documentElement.clientWidth,
              clientHeight: document.documentElement.clientHeight,
              scrollWidth: document.documentElement.scrollWidth,
              scrollHeight: document.documentElement.scrollHeight,
            },
            header: rect('.app-top-nav'),
            shellContent: rect('.app-main'),
            pageFrame: rect('.settings-page') || rect('.page-container') || rect('.model-config-page'),
            settingsContent: rect('.settings-content'),
            notificationToast: rect('.notification-toast'),
            notice: getComputedStyle(document.querySelector('.minimum-viewport-notice')).display,
            visibleText: document.body.innerText.slice(0, 260),
          };
        }"""
    )
    return {"image": name, **metrics}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1366, "height": 900}, device_scale_factor=1)
        page.route(
            "**/api/settings",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(settings),
            ),
        )
        page.route(
            "**/api/inspection/sessions/42/timelines",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"items": [timeline_preview]}),
            ),
        )
        page.route(
            "**/api/inspection/sessions/42/timelines/91",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(timeline),
            ),
        )

        results.append(capture(page, "dili-agent-1366-light.png", "/", 1366, 900, theme="light"))
        results.append(capture(page, "dili-agent-1366-dark.png", "/", 1366, 900, theme="dark"))
        results.append(capture(page, "dili-agent-1100.png", "/", 1100, 900, theme="light"))
        results.append(capture(page, "narrow-window-1099.png", "/", 1099, 900, theme="light"))

        for name, route in [
            ("clinical-sessions-1366.png", "/clinical-sessions"),
            ("data-inspection-1366.png", "/data"),
            ("patient-timeline-1366.png", "/sessions/42/timetable"),
            ("settings-general-1366.png", "/settings/general"),
            ("settings-models-1366.png", "/settings/models"),
            ("settings-data-1366.png", "/settings/data"),
            ("settings-integrations-1366.png", "/settings/integrations"),
            ("settings-advanced-1366.png", "/settings/advanced"),
        ]:
            results.append(capture(page, name, route, 1366, 900))

        for name, route in [
            ("clinical-sessions-1100.png", "/clinical-sessions"),
            ("data-inspection-1100.png", "/data"),
            ("patient-timeline-1100.png", "/sessions/42/timetable"),
            ("settings-general-1100.png", "/settings/general"),
            ("settings-models-1100.png", "/settings/models"),
            ("settings-data-1100.png", "/settings/data"),
            ("settings-integrations-1100.png", "/settings/integrations"),
            ("settings-advanced-1100.png", "/settings/advanced"),
        ]:
            results.append(capture(page, name, route, 1100, 900))

        results.append(capture(page, "data-inspection-1920.png", "/data", 1920, 1080))
        results.append(capture(page, "patient-timeline-1920.png", "/sessions/42/timetable", 1920, 1080))
        browser.close()

    (OUTPUT / "visual-metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Captured {len(results)} screenshots at the requested desktop viewports.")
    for item in results:
        document = item["document"]
        overflow = (
            document["scrollWidth"] > document["clientWidth"],
            document["scrollHeight"] > document["clientHeight"],
        )
        frame = item["pageFrame"] or {}
        toast = item["notificationToast"]
        toast_position = "toast=none" if toast is None else f"toast=({toast['x']:.0f},{toast['y']:.0f})"
        print(
            f"{item['image']}: {item['route']} {item['viewport']['width']}x{item['viewport']['height']} "
            f"header={item['header']['height']:.0f}px frame=({frame.get('x', 0):.0f},{frame.get('y', 0):.0f}) "
            f"frame-width={frame.get('width', 0):.0f}px document-overflow={overflow} notice={item['notice']} {toast_position}"
        )


if __name__ == "__main__":
    main()
