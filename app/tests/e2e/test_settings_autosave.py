# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import os
from decimal import Decimal

from playwright.sync_api import Page, expect

OPERATIONAL_FIELDS = {
    "general": [
        "polling_interval",
        "llm_output_token_limit",
        "max_concurrent_clinical_jobs",
    ],
    "data": [
        "drug_name_min_length",
        "drug_name_max_length",
        "drug_name_max_tokens",
        "text_extraction_batch_size",
        "text_extraction_max_concurrency",
        "retrieval_batch_size",
        "retrieval_max_concurrency",
        "clinical_assessment_batch_size",
        "clinical_assessment_max_concurrency",
        "min_best_score",
        "high_confidence_min_score",
        "high_confidence_min_margin",
        "moderate_confidence_min_score",
        "moderate_confidence_min_margin",
    ],
    "integrations": [
        "ncbi_contact_email",
        "livertox_download_timeout",
        "livertox_archive",
        "livertox_yield_interval",
        "livertox_skip_deterministic_ratio",
        "livertox_monograph_max_workers",
        "rxnav_request_timeout",
        "terminology_download_timeout",
        "drugcentral_enabled",
        "drugcentral_host",
        "drugcentral_port",
        "drugcentral_database",
        "drugcentral_user",
        "chembl_enabled",
        "chembl_max_phase",
        "swissmedic_enabled",
    ],
    "matching": [
        "fuzzy_min_query_length",
        "fuzzy_short_name_length",
        "fuzzy_short_max_distance",
        "fuzzy_long_max_distance",
        "fuzzy_min_margin",
        "fuzzy_shortlist_size",
        "resolution_cache_limit",
        "external_fallback_enabled",
        "external_max_requests_per_name",
        "external_negative_cache_days",
    ],
    "advanced": [
        "default_llm_timeout",
        "parser_llm_timeout",
        "disease_llm_timeout",
        "clinical_llm_timeout",
        "livertox_llm_timeout",
        "minimum_llm_timeout",
        "cloud_llm_timeout_cap",
        "local_llm_timeout_cap",
        "ollama_server_start_timeout",
        "max_excerpt_length",
    ],
}

RAG_FIELDS = [
    ("Chunking", "rag-chunk-size", "chunk_size"),
    ("Chunking", "rag-chunk-overlap", "chunk_overlap"),
    ("Embeddings", "rag-batch-size", "embedding_batch_size"),
    ("Embeddings", "rag-embedding-offline", "embedding_offline_mode"),
    ("Index", "rag-collection-name", "vector_collection_name"),
    ("Index", "rag-index-metric", "vector_index_metric"),
    ("Index", "rag-index-type", "vector_index_type"),
    ("Index", "rag-local-filesystem", "allow_local_filesystem_access"),
    ("Index", "rag-stream-batch", "vector_stream_batch_size"),
]


###############################################################################
def _open_shared_resources(page: Page, base_url: str) -> None:
    page.goto(f"{base_url}/settings/providers")


###############################################################################
def _api_state(page: Page, api_base_url: str, endpoint: str) -> dict:
    response = page.request.get(f"{api_base_url}{endpoint}")
    assert response.status == 200
    return response.json()


###############################################################################
def _expect_settings_write(page: Page):
    return page.expect_response(
        lambda response: response.request.method == "PATCH"
        and response.url.endswith("/api/settings")
    )


###############################################################################
def _expect_model_write(page: Page):
    return page.expect_response(
        lambda response: response.request.method == "PUT"
        and response.url.endswith("/api/model-config")
    )


###############################################################################
def _number_target(
    category: str,
    key: str,
    control,
    values: dict,
) -> Decimal:
    current = Decimal(str(values[key]))
    step = Decimal(control.get_attribute("step") or "1")
    minimum = Decimal(control.get_attribute("min") or "-1000000000")
    maximum = Decimal(control.get_attribute("max") or "1000000000")
    directions = (
        (-step, step)
        if key.startswith("moderate_confidence_") or key == "minimum_llm_timeout"
        else (step, -step)
    )
    for delta in directions:
        candidate = current + delta
        if candidate < minimum or candidate > maximum or candidate == current:
            continue
        trial = {**values, key: float(candidate)}
        if category == "data":
            if trial["drug_name_max_length"] < trial["drug_name_min_length"]:
                continue
            if trial["high_confidence_min_score"] < trial["moderate_confidence_min_score"]:
                continue
            if trial["high_confidence_min_margin"] < trial["moderate_confidence_min_margin"]:
                continue
        if category == "advanced":
            if trial["cloud_llm_timeout_cap"] < trial["minimum_llm_timeout"]:
                continue
            if trial["local_llm_timeout_cap"] < trial["minimum_llm_timeout"]:
                continue
        return candidate
    raise AssertionError(f"No valid alternate value found for {category}.{key}")


###############################################################################
def _set_control(control, value) -> None:
    if control.get_attribute("type") == "checkbox":
        control.set_checked(value)
    else:
        control.fill(str(value))


###############################################################################
def _assert_control_value(control, value) -> None:
    if control.get_attribute("type") == "checkbox":
        assert control.is_checked() is value
        return
    actual = control.input_value()
    if control.get_attribute("type") == "number":
        assert Decimal(actual) == Decimal(str(value))
    else:
        assert actual == str(value)


###############################################################################
def test_every_operational_setting_autosaves_and_survives_reload(
    page: Page,
    base_url: str,
    api_base_url: str,
) -> None:
    original = _api_state(page, api_base_url, "/api/settings")["values"]
    page.set_default_timeout(15000)

    for category, fields in OPERATIONAL_FIELDS.items():
        page.goto(f"{base_url}/settings/{category}")
        rows = page.locator(".operational-setting-row")
        expect(rows).to_have_count(len(fields))
        controls = {
            row.locator("input, select").get_attribute("name"): row.locator("input, select")
            for row in rows.all()
        }
        assert set(controls) == set(fields)
        initial_category = dict(original[category])
        expected = dict(initial_category)

        for key in fields:
            control = controls[key]
            input_type = control.get_attribute("type")
            if input_type == "checkbox":
                candidate = not expected[key]
                expected_value = candidate
            elif input_type == "email":
                candidate = "settings-autosave@example.org"
                expected_value = candidate
            elif input_type == "text":
                candidate = f"{expected[key]}.autosave"
                expected_value = candidate
            elif input_type == "number":
                numeric = _number_target(category, key, control, expected)
                candidate = format(numeric.normalize(), "f")
                expected_value = float(numeric)
            else:
                raise AssertionError(f"Unreviewed operational control type: {input_type}")

            with _expect_settings_write(page) as response_info:
                _set_control(control, candidate)
            response = response_info.value
            assert response.status == 200, response.text()
            expected[key] = expected_value
            assert response.json()["values"][category][key] == expected_value
            assert _api_state(page, api_base_url, "/api/settings")["values"][category][key] == expected_value

        page.reload()
        live_state = _api_state(page, api_base_url, "/api/settings")["values"][category]
        for key in fields:
            _assert_control_value(page.locator(f'[name="{key}"]'), expected[key])
            assert live_state[key] == expected[key]

        with _expect_settings_write(page) as response_info:
            for key in fields:
                control = page.locator(f'[name="{key}"]')
                _set_control(control, initial_category[key])
        restored = response_info.value
        assert restored.status == 200, restored.text()
        assert restored.json()["values"][category] == initial_category
        assert _api_state(page, api_base_url, "/api/settings")["values"][category] == initial_category


###############################################################################
def test_rapid_edits_merge_and_failed_writes_are_flagged(
    page: Page,
    base_url: str,
    api_base_url: str,
) -> None:
    page.set_default_timeout(15000)
    page.goto(f"{base_url}/settings/data")
    original = _api_state(page, api_base_url, "/api/settings")["values"]["data"]
    first_key = "drug_name_max_tokens"
    second_key = "text_extraction_batch_size"
    first_value = original[first_key] + 1
    second_value = original[second_key] + 1

    with _expect_settings_write(page) as response_info:
        page.locator(f'[name="{first_key}"]').fill(str(first_value))
        page.locator(f'[name="{second_key}"]').fill(str(second_value))
    rapid_response = response_info.value
    assert rapid_response.status == 200, rapid_response.text()
    rapid_values = rapid_response.json()["values"]["data"]
    assert rapid_values[first_key] == first_value
    assert rapid_values[second_key] == second_value
    page.reload()
    assert _api_state(page, api_base_url, "/api/settings")["values"]["data"][first_key] == first_value
    assert _api_state(page, api_base_url, "/api/settings")["values"]["data"][second_key] == second_value

    with _expect_settings_write(page) as response_info:
        page.locator(f'[name="{first_key}"]').fill(str(original[first_key]))
        page.locator(f'[name="{second_key}"]').fill(str(original[second_key]))
    restored = response_info.value
    assert restored.status == 200, restored.text()

    page.goto(f"{base_url}/settings/general")
    original_polling = _api_state(page, api_base_url, "/api/settings")["values"]["general"]["polling_interval"]
    candidate = original_polling + 0.5

    def reject_settings_patch(route) -> None:
        if route.request.method == "PATCH":
            route.fulfill(
                status=503,
                content_type="application/json",
                body='{"detail":"controlled settings write failure"}',
            )
        else:
            route.continue_()

    page.route("**/api/settings", reject_settings_patch)
    with page.expect_response(
        lambda response: response.request.method == "PATCH"
        and response.url.endswith("/api/settings")
    ) as failed_info:
        page.locator('[name="polling_interval"]').fill(str(candidate))
    failed_response = failed_info.value
    assert failed_response.status == 503
    expect(page.get_by_role("alert")).to_contain_text("A required service is unavailable")
    assert _api_state(page, api_base_url, "/api/settings")["values"]["general"]["polling_interval"] == original_polling
    page.unroute("**/api/settings", reject_settings_patch)
    page.reload()
    assert Decimal(page.locator('[name="polling_interval"]').input_value()) == Decimal(str(original_polling))


###############################################################################
def _rag_candidate(control, key: str, current: dict):
    input_type = control.get_attribute("type")
    if input_type == "checkbox":
        return not current[key]
    if input_type == "number":
        value = Decimal(str(current[key]))
        step = Decimal(control.get_attribute("step") or "1")
        minimum = Decimal(control.get_attribute("min") or "-1000000000")
        maximum = Decimal(control.get_attribute("max") or "1000000000")
        directions = (-step, step) if key == "chunk_overlap" else (step, -step)
        for delta in directions:
            candidate = value + delta
            if candidate < minimum or candidate > maximum:
                continue
            if key == "retrieval_selected_count" and candidate > Decimal(str(current["retrieval_candidate_count"])):
                continue
            if key == "chunk_overlap" and candidate >= Decimal(str(current["chunk_size"])):
                continue
            return candidate
        raise AssertionError(f"No valid alternate RAG value found for {key}")
    if control.evaluate("element => element.tagName.toLowerCase()") == "select":
        return next(
            option.get_attribute("value")
            for option in control.locator("option").all()
            if option.get_attribute("value") != current[key]
        )
    return f"{current[key]}-autosave"


###############################################################################
def _set_rag_value(control, key: str, value) -> None:
    if control.get_attribute("type") == "checkbox":
        control.set_checked(bool(value))
    elif control.evaluate("element => element.tagName.toLowerCase()") == "select":
        control.select_option(str(value))
    else:
        control.fill(str(value))


###############################################################################
def _rag_control(page: Page, control_id: str):
    return page.locator(f"#{control_id}")


###############################################################################
def _save_rag_value(page: Page, api_base_url: str, control, key: str, value):
    with _expect_model_write(page) as response_info:
        _set_rag_value(control, key, value)
    response = response_info.value
    assert response.status == 200, response.text()
    expect(page.locator(".model-config-rag-modal-footer")).to_contain_text("All changes saved")
    saved = _api_state(page, api_base_url, "/api/model-config")["rag_settings"]
    assert saved[key] == value


###############################################################################
def test_shared_index_settings_autosave_without_a_save_button(
    page: Page,
    base_url: str,
    api_base_url: str,
) -> None:
    page.set_default_timeout(15000)
    _open_shared_resources(page, base_url)
    expect(page.get_by_role("button", name="Configure RAG settings")).to_be_visible()
    assert page.get_by_role("button", name="Save Configuration").count() == 0
    assert page.get_by_role("button", name="Save settings").count() == 0

    original = dict(_api_state(page, api_base_url, "/api/model-config")["rag_settings"])
    # Retrieval and reranking choices belong to agents, not to the shared index.
    assert "use_reranking" not in original and "use_hybrid_search" not in original
    expected = dict(original)
    page.get_by_role("button", name="Configure RAG settings").click()

    # Dependent numeric and text controls first, then the switches that can disable them.
    ordered = sorted(
        RAG_FIELDS,
        key=lambda item: item[2] in {"embedding_offline_mode", "allow_local_filesystem_access"},
    )
    for tab, control_id, key in ordered:
        page.get_by_role("button", name=tab).click()
        control = _rag_control(page, control_id)
        candidate = _rag_candidate(control, key, expected)
        value = float(candidate) if isinstance(candidate, Decimal) else candidate
        _save_rag_value(page, api_base_url, control, key, value)
        expected[key] = value

    page.reload()
    page.get_by_role("button", name="Configure RAG settings").click()
    for tab, control_id, key in RAG_FIELDS:
        page.get_by_role("button", name=tab).click()
        control = _rag_control(page, control_id)
        actual = control.is_checked() if control.get_attribute("type") == "checkbox" else control.input_value()
        if isinstance(expected[key], bool):
            assert actual is expected[key]
        elif isinstance(expected[key], (int, float)):
            assert Decimal(actual) == Decimal(str(expected[key]))
        else:
            assert actual == expected[key]

    # Restore the original shared settings, one write per changed field.
    for tab, control_id, key in ordered[::-1]:
        page.get_by_role("button", name=tab).click()
        control = _rag_control(page, control_id)
        with _expect_model_write(page) as response_info:
            _set_rag_value(control, key, original[key])
        assert response_info.value.status == 200, response_info.value.text()
        expect(page.locator(".model-config-rag-modal-footer")).to_contain_text("All changes saved")
    assert _api_state(page, api_base_url, "/api/model-config")["rag_settings"] == original
    page.get_by_role("button", name="Done").click()
    expect(page.get_by_role("dialog", name="RAG settings")).to_be_hidden()


###############################################################################
def test_shared_model_config_writes_carry_only_shared_keys(
    page: Page,
    base_url: str,
    api_base_url: str,
) -> None:
    page.set_default_timeout(15000)
    _open_shared_resources(page, base_url)
    original = _api_state(page, api_base_url, "/api/model-config")
    page.get_by_role("button", name="Configure RAG settings").click()
    page.get_by_role("button", name="Index").click()
    control = _rag_control(page, "rag-collection-name")
    with _expect_model_write(page) as response_info:
        control.fill(f"{original['rag_settings']['vector_collection_name']}-keys")
    request_body = response_info.value.request.post_data_json
    assert set(request_body) <= {"rag_settings"}
    assert set(request_body["rag_settings"]) <= set(original["rag_settings"])
    with _expect_model_write(page) as response_info:
        control.fill(original["rag_settings"]["vector_collection_name"])
    assert response_info.value.status == 200
    page.get_by_role("button", name="Done").click()

    ollama_url = os.environ.get("OLLAMA_URL")
    assert ollama_url, "Full browser E2E supplies an isolated fake Ollama endpoint."
    with page.expect_response(
        lambda response: response.request.method == "POST"
        and response.url.endswith("/api/model-config/catalogs/ollama/refresh")
    ) as refresh_info:
        page.get_by_role("button", name="Refresh model catalog").click()
    assert refresh_info.value.status == 200, refresh_info.value.text()


###############################################################################
def test_database_edits_wait_for_confirmation_and_failed_switch_rolls_back(
    page: Page,
    base_url: str,
    api_base_url: str,
) -> None:
    page.goto(f"{base_url}/settings/database")
    initial = _api_state(page, api_base_url, "/api/database/configuration")
    expect(page.get_by_role("button", name="Save and switch")).to_be_visible()

    if initial["configuration"]["backend"] == "sqlite":
        page.locator("#sqlite-path").fill("C:/settings-autosave-must-not-apply.db")
        assert _api_state(page, api_base_url, "/api/database/configuration") == initial
        page.reload()

    page.get_by_label("Database backend").select_option("postgresql")
    page.locator("#postgres-host").fill("127.0.0.1")
    page.locator("#postgres-port").fill("1")
    page.locator("#postgres-database").fill("diligent_settings_test")
    page.locator("#postgres-username").fill("settings_test")

    with page.expect_response(
        lambda response: response.request.method == "POST"
        and response.url.endswith("/api/database/test")
    ) as test_info:
        page.get_by_role("button", name="Test connection").click()
    connection_test = test_info.value
    assert connection_test.status == 200
    assert connection_test.json()["reachable"] is False
    assert _api_state(page, api_base_url, "/api/database/configuration") == initial

    with page.expect_response(
        lambda response: response.request.method == "PUT"
        and response.url.endswith("/api/database/configuration")
    ) as switch_info:
        page.get_by_role("button", name="Save and switch").click()
    failed_switch = switch_info.value
    assert failed_switch.status == 503
    assert _api_state(page, api_base_url, "/api/database/configuration") == initial
    # The previous database stays active and readable after the failed switch.
    sessions_read = page.request.get(f"{api_base_url}/api/inspection/sessions?limit=1")
    assert sessions_read.status == 200, sessions_read.text()
    expect(page.get_by_role("alert")).to_contain_text("A required service is unavailable")
