from __future__ import annotations

import time

import pytest
from playwright.sync_api import APIRequestContext


def maybe_skip_for_encryption_config(response) -> None:  # type: ignore[no-untyped-def]
    if response.status < 500:
        return
    detail = response.text()
    if "ACCESS_KEY_ENCRYPTION_KEY" in detail:
        pytest.skip("ACCESS_KEY_ENCRYPTION_KEY is not configured for API e2e run.")


def test_access_keys_crud_returns_metadata_only(api_context: APIRequestContext) -> None:
    provider = "openai"
    plaintext_key = f"sk-test-{int(time.time())}"
    created_id: int | None = None
    try:
        create_response = api_context.post(
            "/access-keys",
            data={"provider": provider, "access_key": plaintext_key},
        )
        maybe_skip_for_encryption_config(create_response)
        assert create_response.status == 201
        created_payload = create_response.json()
        created_id = created_payload.get("id")
        assert isinstance(created_id, int)
        assert created_payload.get("provider") == provider
        assert "access_key" not in created_payload
        assert plaintext_key not in str(created_payload)

        list_response = api_context.get(f"/access-keys?provider={provider}")
        assert list_response.status == 200
        keys_payload = list_response.json()
        assert isinstance(keys_payload, list)
        assert all("access_key" not in row for row in keys_payload)
        assert all(plaintext_key not in str(row) for row in keys_payload)

        activate_response = api_context.put(f"/access-keys/{created_id}/activate")
        assert activate_response.status == 200
        activated_payload = activate_response.json()
        assert activated_payload.get("id") == created_id
        assert activated_payload.get("is_active") is True
        assert "access_key" not in activated_payload
    finally:
        if created_id is not None:
            api_context.delete(f"/access-keys/{created_id}")
