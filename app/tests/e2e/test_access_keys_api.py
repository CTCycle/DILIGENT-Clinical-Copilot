from __future__ import annotations

import re

from playwright.sync_api import APIRequestContext

###############################################################################
def _assert_metadata_only(payload: object, plaintext_keys: tuple[str, ...]) -> None:
    serialized = str(payload)
    assert "access_key" not in serialized
    for plaintext_key in plaintext_keys:
        assert plaintext_key not in serialized

    assert isinstance(payload, dict)
    fingerprint = payload.get("fingerprint")
    assert isinstance(fingerprint, str)
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint)


###############################################################################
def _list_provider_keys(
    api_context: APIRequestContext,
    provider: str,
    plaintext_keys: tuple[str, ...],
) -> list[dict]:
    response = api_context.get(f"/api/access-keys?provider={provider}")
    assert response.status == 200, response.text()
    payload = response.json()
    assert isinstance(payload, list)
    for row in payload:
        _assert_metadata_only(row, plaintext_keys)
    return payload


###############################################################################
def test_access_keys_complete_synthetic_lifecycle(
    api_context: APIRequestContext,
) -> None:
    provider = "openai"
    wrong_provider = "brave"
    plaintext_a = "sk-proj-ci-lifecycle-a-20260929"
    plaintext_b = "sk-proj-ci-lifecycle-b-20260929"
    plaintext_keys = (plaintext_a, plaintext_b)
    created_ids: list[int] = []

    try:
        create_a = api_context.post(
            "/api/access-keys",
            data={"provider": provider, "access_key": plaintext_a},
        )
        assert create_a.status == 201, create_a.text()
        payload_a = create_a.json()
        _assert_metadata_only(payload_a, plaintext_keys)
        assert payload_a.get("provider") == provider
        assert payload_a.get("is_active") is False
        key_a_id = payload_a.get("id")
        assert isinstance(key_a_id, int)
        created_ids.append(key_a_id)
        assert _list_provider_keys(api_context, provider, plaintext_keys) == [payload_a]

        create_b = api_context.post(
            "/api/access-keys",
            data={"provider": provider, "access_key": plaintext_b},
        )
        assert create_b.status == 201, create_b.text()
        payload_b = create_b.json()
        _assert_metadata_only(payload_b, plaintext_keys)
        assert payload_b.get("provider") == provider
        assert payload_b.get("is_active") is False
        key_b_id = payload_b.get("id")
        assert isinstance(key_b_id, int)
        created_ids.append(key_b_id)

        listed = _list_provider_keys(api_context, provider, plaintext_keys)
        assert {row["id"] for row in listed} == {key_a_id, key_b_id}
        assert all(row["is_active"] is False for row in listed)

        activate_a = api_context.put(
            f"/api/access-keys/{key_a_id}/activate?provider={provider}"
        )
        assert activate_a.status == 200, activate_a.text()
        activated_a = activate_a.json()
        _assert_metadata_only(activated_a, plaintext_keys)
        assert activated_a["id"] == key_a_id
        assert activated_a["is_active"] is True

        listed = _list_provider_keys(api_context, provider, plaintext_keys)
        assert sum(row["is_active"] for row in listed) == 1
        assert next(row for row in listed if row["id"] == key_a_id)["is_active"]
        assert not next(row for row in listed if row["id"] == key_b_id)["is_active"]

        activate_b = api_context.put(
            f"/api/access-keys/{key_b_id}/activate?provider={provider}"
        )
        assert activate_b.status == 200, activate_b.text()
        activated_b = activate_b.json()
        _assert_metadata_only(activated_b, plaintext_keys)
        assert activated_b["id"] == key_b_id
        assert activated_b["is_active"] is True

        listed = _list_provider_keys(api_context, provider, plaintext_keys)
        assert sum(row["is_active"] for row in listed) == 1
        assert not next(row for row in listed if row["id"] == key_a_id)["is_active"]
        assert next(row for row in listed if row["id"] == key_b_id)["is_active"]

        wrong_provider_activation = api_context.put(
            f"/api/access-keys/{key_b_id}/activate?provider={wrong_provider}"
        )
        assert wrong_provider_activation.status == 404
        wrong_provider_delete = api_context.delete(
            f"/api/access-keys/{key_b_id}?provider={wrong_provider}"
        )
        assert wrong_provider_delete.status == 404
        assert sum(
            row["is_active"]
            for row in _list_provider_keys(api_context, provider, plaintext_keys)
        ) == 1

        for key_id in (key_a_id, key_b_id):
            delete_response = api_context.delete(
                f"/api/access-keys/{key_id}?provider={provider}"
            )
            assert delete_response.status == 200, delete_response.text()

        assert _list_provider_keys(api_context, provider, plaintext_keys) == []
    finally:
        for key_id in created_ids:
            api_context.delete(f"/api/access-keys/{key_id}?provider={provider}")

###############################################################################
def test_activate_and_delete_require_provider(api_context: APIRequestContext) -> None:
    provider = "openai"
    create_response = api_context.post(
        "/api/access-keys",
        data={
            "provider": provider,
            "access_key": "sk-proj-provider-boundary-20260929",
        },
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

###############################################################################
def test_access_key_creation_rejects_short_secret(
    api_context: APIRequestContext,
) -> None:
    response = api_context.post(
        "/api/access-keys",
        data={"provider": "openai", "access_key": "short"},
    )

    assert response.status == 422
