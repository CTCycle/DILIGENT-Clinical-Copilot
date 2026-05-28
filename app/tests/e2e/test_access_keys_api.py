from __future__ import annotations

import time

from playwright.sync_api import APIRequestContext


def test_access_keys_crud_returns_metadata_only(api_context: APIRequestContext) -> None:
    provider = "openai"
    plaintext_key = f"sk-test-{int(time.time())}"
    created_id: int | None = None
    try:
        create_response = api_context.post(
            "/api/access-keys",
            data={"provider": provider, "access_key": plaintext_key},
        )
        assert create_response.status == 201
        created_payload = create_response.json()
        created_id = created_payload.get("id")
        assert isinstance(created_id, int)
        assert created_payload.get("provider") == provider
        assert "access_key" not in created_payload
        assert plaintext_key not in str(created_payload)

        list_response = api_context.get(f"/api/access-keys?provider={provider}")
        assert list_response.status == 200
        keys_payload = list_response.json()
        assert isinstance(keys_payload, list)
        assert all("access_key" not in row for row in keys_payload)
        assert all(plaintext_key not in str(row) for row in keys_payload)

        activate_response = api_context.put(
            f"/api/access-keys/{created_id}/activate?provider={provider}"
        )
        assert activate_response.status == 200
        activated_payload = activate_response.json()
        assert activated_payload.get("id") == created_id
        assert activated_payload.get("is_active") is True
        assert "access_key" not in activated_payload
    finally:
        if created_id is not None:
            api_context.delete(f"/api/access-keys/{created_id}?provider={provider}")


def test_activate_and_delete_require_provider(api_context: APIRequestContext) -> None:
    provider = "openai"
    create_response = api_context.post(
        "/api/access-keys",
        data={"provider": provider, "access_key": f"sk-test-{int(time.time())}"},
    )
    assert create_response.status == 201
    key_id = create_response.json()["id"]
    try:
        activate_response = api_context.put(f"/api/access-keys/{key_id}/activate")
        assert activate_response.status == 422

        delete_response = api_context.delete(f"/api/access-keys/{key_id}")
        assert delete_response.status == 422
    finally:
        api_context.delete(f"/api/access-keys/{key_id}?provider={provider}")


def test_access_key_creation_rejects_short_secret(
    api_context: APIRequestContext,
) -> None:
    response = api_context.post(
        "/api/access-keys",
        data={"provider": "openai", "access_key": "short"},
    )

    assert response.status == 422
