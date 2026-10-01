# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Rendered access-key management coverage for the Settings → Models flow."""

from __future__ import annotations

import re

from playwright.sync_api import Page, expect

###############################################################################
def test_access_key_modal_lifecycle_is_metadata_only_and_persistent(
    page: Page, base_url: str
) -> None:
    plaintext_key = "sk-proj-ui-lifecycle-20260929"

    page.goto(f"{base_url}/settings/models")
    provider_row = page.locator(".model-config-cloud-row").filter(has_text="OpenAI")
    manage_keys = provider_row.get_by_role("button", name="Manage keys")
    expect(manage_keys).to_be_visible()
    manage_keys.click()

    dialog = page.locator("dialog.access-key-modal")
    expect(dialog).to_be_visible()
    expect(dialog.get_by_role("heading", name="OpenAI Access Keys")).to_be_visible()

    try:
        page.get_by_placeholder("Paste access key").fill(plaintext_key)
        dialog.get_by_role("button", name="Add").click()

        row = dialog.locator(".access-key-row")
        expect(row).to_have_count(1)
        expect(row).not_to_contain_text(plaintext_key)
        assert plaintext_key not in page.locator("body").inner_text()

        dialog.get_by_role("button", name="Show fingerprint").click()
        fingerprint = row.locator(".access-key-fingerprint")
        expect(fingerprint).to_contain_text(re.compile(r"fp: [0-9a-f]{6}\.\.\.[0-9a-f]{4}"))
        expect(fingerprint).not_to_contain_text(plaintext_key)
        assert plaintext_key not in page.locator("body").inner_text()

        dialog.get_by_role("button", name="Activate key").click()
        expect(row).to_have_class(re.compile(r"\bis-active\b"))
        expect(row.get_by_text("Active")).to_be_visible()

        page.reload()
        provider_row = page.locator(".model-config-cloud-row").filter(has_text="OpenAI")
        provider_row.get_by_role("button", name="Manage keys").click()
        dialog = page.locator("dialog.access-key-modal")
        row = dialog.locator(".access-key-row")
        expect(row).to_have_count(1)
        expect(row).to_have_class(re.compile(r"\bis-active\b"))
        expect(row.get_by_text("Active")).to_be_visible()
        assert plaintext_key not in page.locator("body").inner_text()

        dialog.get_by_role("button", name="Delete key").click()
        expect(dialog.get_by_text("No keys stored for this provider.")).to_be_visible()
        assert plaintext_key not in page.locator("body").inner_text()
    finally:
        if dialog.is_visible():
            page.get_by_role("button", name="Close access key modal").click()
