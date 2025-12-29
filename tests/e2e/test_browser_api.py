"""
E2E tests for database browser API endpoints.
"""
from __future__ import annotations

import pytest
from playwright.sync_api import APIRequestContext


EXPECTED_SESSION_COLUMNS = {
    "id",
    "patient_name",
    "session_timestamp",
    "hepatic_pattern",
    "final_report",
}
EXPECTED_LIVERTOX_COLUMNS = {
    "drug_name",
    "nbk_id",
    "excerpt",
    "likelihood_score",
}
EXPECTED_DRUGS_COLUMNS = {
    "rxcui",
    "raw_name",
    "name",
    "term_type",
}


def fetch_table(
    api_context: APIRequestContext, table: str, offset: int = 0, limit: int = 5
) -> dict:
    response = api_context.get(
        f"/browser/{table}?offset={offset}&limit={limit}"
    )
    assert response.ok, f"Expected 200, got {response.status}"
    return response.json()


def assert_table_payload(payload: dict, expected_columns: set[str], offset: int) -> None:
    assert isinstance(payload.get("columns"), list)
    assert isinstance(payload.get("rows"), list)
    assert isinstance(payload.get("total_rows"), int)
    assert isinstance(payload.get("has_more"), bool)
    assert expected_columns.issubset(set(payload["columns"]))

    rows = payload["rows"]
    total_rows = payload["total_rows"]
    assert total_rows >= len(rows)
    assert payload["has_more"] == ((offset + len(rows)) < total_rows)


def test_sessions_table_contract(api_context: APIRequestContext):
    payload = fetch_table(api_context, "sessions", offset=0, limit=5)
    assert_table_payload(payload, EXPECTED_SESSION_COLUMNS, offset=0)


def test_livertox_table_contract(api_context: APIRequestContext):
    payload = fetch_table(api_context, "livertox", offset=0, limit=5)
    assert_table_payload(payload, EXPECTED_LIVERTOX_COLUMNS, offset=0)


def test_drugs_table_contract(api_context: APIRequestContext):
    payload = fetch_table(api_context, "drugs", offset=0, limit=5)
    assert_table_payload(payload, EXPECTED_DRUGS_COLUMNS, offset=0)


def test_browser_limit_applies(api_context: APIRequestContext):
    payload = fetch_table(api_context, "sessions", offset=0, limit=1)
    assert len(payload["rows"]) <= 1


@pytest.mark.parametrize("offset", [-1, -10])
def test_browser_rejects_negative_offset(
    api_context: APIRequestContext, offset: int
):
    response = api_context.get(f"/browser/sessions?offset={offset}&limit=5")
    assert response.status == 422


@pytest.mark.parametrize("limit", [0, 10001])
def test_browser_rejects_invalid_limit(
    api_context: APIRequestContext, limit: int
):
    response = api_context.get(f"/browser/sessions?offset=0&limit={limit}")
    assert response.status == 422
