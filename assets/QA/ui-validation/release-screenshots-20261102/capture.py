"""Capture release screenshots from the running app (isolated, fictional data)."""

from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:7690"
OUT = Path(__file__).resolve().parents[3] / "figures"
CASE = Path(__file__).resolve().parents[2] / "audits/clinical-20261028/cases/C04_polypharmacy_staggered_stops.txt"
OUT.mkdir(exist_ok=True)


def settle(page, ms=1800):
    page.wait_for_timeout(ms)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", args=["--lang=en-GB"])
    page = browser.new_page(viewport={"width": 1440, "height": 900}, color_scheme="light", locale="en-GB")

    # 01 DILI Agent with a populated fictional case
    page.goto(BASE + "/")
    settle(page)
    page.locator("textarea").first.fill(CASE.read_text(encoding="utf-8"))
    page.get_by_placeholder("e.g., Marco Rossi").fill("Anna Moretti")
    page.locator("input[type=date]").first.fill("2026-10-30")
    page.mouse.click(1240, 700)
    settle(page, 800)
    page.screenshot(path=OUT / "01-dili-agent.jpg", type="jpeg", quality=90)

    # 02 Clinical sessions
    page.goto(BASE + "/clinical-sessions?session=1&tab=preview")
    page.wait_for_selector(".clinical-session-row")
    settle(page)
    page.screenshot(path=OUT / "02-clinical-sessions.jpg", type="jpeg", quality=90)

    # 03 report
    page.goto(BASE + "/report/1")
    settle(page, 2500)
    page.get_by_role("heading", name="DILI adjudication summary").evaluate(
        "el => { el.style.scrollMarginTop = '175px'; el.scrollIntoView({block: 'start'}); }"
    )
    settle(page, 800)
    page.screenshot(path=OUT / "03-report.jpg", type="jpeg", quality=90)

    # 04 patient timeline (horizontal view)
    page.goto(BASE + "/sessions/3/timetable")
    settle(page, 2500)
    page.get_by_role("button", name="Timeline", exact=True).click()
    settle(page, 1200)
    page.screenshot(path=OUT / "04-patient-timeline.jpg", type="jpeg", quality=90)

    # 05 model settings
    page.goto(BASE + "/settings/models")
    settle(page, 2500)
    page.get_by_role("button", name="Up to 8B").click()
    settle(page, 800)
    page.screenshot(path=OUT / "05-model-settings.jpg", type="jpeg", quality=90)

    # 06 RAG settings
    page.get_by_role("button", name="Configure").first.click()
    settle(page, 1200)
    page.screenshot(path=OUT / "06-rag-settings.jpg", type="jpeg", quality=90)
    browser.close()
