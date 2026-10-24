# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from playwright.sync_api import APIRequestContext, Browser, Page, expect


def _get_draft(api_context: APIRequestContext) -> dict:
    response = api_context.get("/api/clinical/draft")
    assert response.status == 200, response.text()
    assert response.headers.get("cache-control") == "no-store"
    return response.json()


def _discard_draft(api_context: APIRequestContext) -> dict:
    snapshot = _get_draft(api_context)
    response = api_context.delete(
        "/api/clinical/draft",
        params={
            "expected_revision": str(snapshot["revision"]),
            **(
                {"expected_draft_key": snapshot["draft_key"]}
                if snapshot["draft_key"] is not None
                else {}
            ),
        },
    )
    assert response.status == 200, response.text()
    assert response.headers.get("cache-control") == "no-store"
    return response.json()


def _enter_case(page: Page, patient_name: str, clinical_input: str) -> None:
    page.get_by_label("Patient Name").fill(patient_name)
    page.get_by_label("Visit Date").fill("2026-10-23")
    page.get_by_label("Clinical Input").fill(clinical_input)
    page.get_by_label("Use RAG evidence").check()


def test_draft_http_contract_validation_conflict_and_tombstone(
    api_context: APIRequestContext,
) -> None:
    _discard_draft(api_context)
    initial = _get_draft(api_context)
    form = {
        "patient_name": "Synthetic draft patient",
        "visit_date": None,
        "clinical_input": "  first line\n\n second line  ",
        "use_rag": True,
    }
    saved_response = api_context.put(
        "/api/clinical/draft",
        data={
            "expected_revision": initial["revision"],
            "expected_draft_key": initial["draft_key"],
            "form": form,
        },
    )
    assert saved_response.status == 200, saved_response.text()
    assert saved_response.headers.get("cache-control") == "no-store"
    saved = saved_response.json()
    assert saved["form"] == form
    assert saved["revision"] == initial["revision"] + 1

    invalid = api_context.put(
        "/api/clinical/draft",
        data={
            "expected_revision": saved["revision"],
            "expected_draft_key": saved["draft_key"],
            "form": {
                **form,
                "patient_name": "Sensitive invalid payload sentinel",
                "patient_image_data_url": "data:image/png;base64,not-accepted",
            },
        },
    )
    assert invalid.status == 422
    assert invalid.headers.get("cache-control") == "no-store"
    assert "Sensitive invalid payload sentinel" not in invalid.text()

    stale = api_context.put(
        "/api/clinical/draft",
        data={
            "expected_revision": initial["revision"],
            "expected_draft_key": initial["draft_key"],
            "form": {**form, "clinical_input": "must not overwrite"},
        },
    )
    assert stale.status == 409
    assert stale.headers.get("cache-control") == "no-store"
    assert _get_draft(api_context)["form"] == form

    tombstone = api_context.delete(
        "/api/clinical/draft",
        params={
            "expected_revision": saved["revision"],
            "expected_draft_key": saved["draft_key"],
        },
    )
    assert tombstone.status == 200, tombstone.text()
    assert tombstone.json()["form"] is None
    delayed = api_context.put(
        "/api/clinical/draft",
        data={
            "expected_revision": saved["revision"],
            "expected_draft_key": saved["draft_key"],
            "form": form,
        },
    )
    assert delayed.status == 409


def test_explicit_save_resume_and_discard_recover_without_running_assessment(
    page: Page,
    base_url: str,
    api_context: APIRequestContext,
) -> None:
    tombstone = _discard_draft(api_context)
    assert tombstone["form"] is None
    page.goto(base_url)
    _enter_case(page, "Unsubmitted synthetic patient", "Unsaved text\nwith spacing  ")
    page.reload()
    expect(page.get_by_label("Patient Name")).to_have_value("")
    expect(page.get_by_label("Visit Date")).to_have_value("")
    expect(page.get_by_label("Clinical Input")).to_have_value("")
    assert _get_draft(api_context)["form"] is None

    saved_text = "  Synthetic draft\n\nwith exact spacing  "
    _enter_case(page, "Synthetic recovery patient", saved_text)
    clinical_job_requests: list[str] = []
    page.on(
        "request",
        lambda request: clinical_job_requests.append(request.url)
        if request.method == "POST" and request.url.endswith("/api/clinical/jobs")
        else None,
    )
    with page.expect_response(
        lambda response: response.request.method == "PUT"
        and response.url.endswith("/api/clinical/draft")
    ):
        page.get_by_role("button", name="Save draft").click()
    expect(page.locator(".clinical-draft-status")).to_contain_text("Saved at")
    initial = _get_draft(api_context)

    edited_text = "  Edited after opt-in\nsecond line  "
    with page.expect_response(
        lambda response: response.request.method == "PUT"
        and response.url.endswith("/api/clinical/draft")
        and response.status == 200
    ) as auto_save:
        page.get_by_label("Clinical Input").fill(edited_text)
    expect(page.locator(".clinical-draft-status")).to_contain_text("Saved at")
    updated = auto_save.value.json()
    assert updated["revision"] == initial["revision"] + 1
    assert updated["form"]["clinical_input"] == edited_text

    browser_storage = page.evaluate("JSON.stringify(Object.fromEntries(Object.entries(localStorage)))")
    assert "Synthetic recovery patient" not in browser_storage
    assert edited_text not in browser_storage

    page.reload()
    expect(page.get_by_role("heading", name="Saved draft available")).to_be_visible()
    expect(page.get_by_label("Patient Name")).to_have_value("")
    expect(page.get_by_label("Clinical Input")).to_have_value("")
    page.get_by_role("button", name="Resume").click()
    expect(page.get_by_label("Patient Name")).to_have_value("Synthetic recovery patient")
    expect(page.get_by_label("Visit Date")).to_have_value("2026-10-23")
    expect(page.get_by_label("Clinical Input")).to_have_value(edited_text)
    expect(page.get_by_label("Use RAG evidence")).to_be_checked()
    assert clinical_job_requests == []

    page.get_by_role("button", name="Discard draft").click()
    dialog = page.get_by_role("dialog")
    expect(dialog).to_be_visible()
    dialog.get_by_role("button", name="Discard draft").click()
    expect(page.get_by_label("Patient Name")).to_have_value("")
    page.reload()
    expect(page.get_by_role("heading", name="Saved draft available")).to_have_count(0)
    assert _get_draft(api_context)["form"] is None


def test_stale_browser_tab_cannot_overwrite_newer_draft(
    browser: Browser,
    base_url: str,
    api_context: APIRequestContext,
) -> None:
    _discard_draft(api_context)
    first_context = browser.new_context()
    second_context = browser.new_context()
    try:
        first_page = first_context.new_page()
        second_page = second_context.new_page()
        first_page.goto(base_url)
        _enter_case(first_page, "Synthetic tab A", "First tab contents")
        first_page.get_by_role("button", name="Save draft").click()
        expect(first_page.locator(".clinical-draft-status")).to_contain_text("Saved at")

        second_page.goto(base_url)
        expect(second_page.get_by_role("heading", name="Saved draft available")).to_be_visible()
        second_page.get_by_role("button", name="Resume").click()
        with second_page.expect_response(
            lambda response: response.request.method == "PUT"
            and response.url.endswith("/api/clinical/draft")
            and response.status == 200
        ):
            second_page.get_by_label("Clinical Input").fill("Newer tab contents")
        expect(second_page.locator(".clinical-draft-status")).to_contain_text("Saved at")

        with first_page.expect_response(
            lambda response: response.request.method == "PUT"
            and response.url.endswith("/api/clinical/draft")
            and response.status == 409
        ):
            first_page.get_by_label("Clinical Input").fill("Stale tab attempt")
        expect(first_page.locator(".clinical-draft-status")).to_contain_text(
            "Draft changed or database switched. Reload before saving."
        )
        expect(first_page.get_by_label("Clinical Input")).to_have_value("Stale tab attempt")
        assert _get_draft(api_context)["form"]["clinical_input"] == "Newer tab contents"
    finally:
        _discard_draft(api_context)
        first_context.close()
        second_context.close()
