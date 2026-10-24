# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import os
from decimal import Decimal

from playwright.sync_api import Page, expect

OPERATIONAL_FIELDS = {
    "general": ["polling_interval"],
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
        "rxnav_max_concurrency",
    ],
    "matching": [
        "direct_confidence",
        "master_confidence",
        "synonym_confidence",
        "min_confidence",
        "normalization_cache_limit",
        "match_cache_limit",
        "alias_cache_limit",
        "token_min_length",
        "catalog_index_limit",
        "spelling_confidence",
        "spelling_min_query_length",
        "spelling_short_name_length",
        "spelling_short_max_distance",
        "spelling_long_max_distance",
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
    ("Retrieval", "rag-candidate-count", "retrieval_candidate_count"),
    ("Retrieval", "rag-selected-count", "retrieval_selected_count"),
    ("Retrieval", "rag-vector-weight", "hybrid_vector_weight"),
    ("Retrieval", "rag-text-weight", "hybrid_text_weight"),
    ("Chunking", "rag-chunk-size", "chunk_size"),
    ("Chunking", "rag-chunk-overlap", "chunk_overlap"),
    ("Embeddings", "rag-batch-size", "embedding_batch_size"),
    ("Embeddings", "rag-embedding-offline", "embedding_offline_mode"),
    ("Ranking", "rag-reranker-model", "reranker_model"),
    ("Index", "rag-collection-name", "vector_collection_name"),
    ("Index", "rag-index-metric", "vector_index_metric"),
    ("Index", "rag-index-type", "vector_index_type"),
    ("Index", "rag-local-filesystem", "allow_local_filesystem_access"),
    ("Index", "rag-stream-batch", "vector_stream_batch_size"),
]


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
def _assert_control_value(control, value) -> None:
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
            if input_type == "email":
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
                control.fill(candidate)
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
                control.fill(str(initial_category[key]))
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
def test_model_and_rag_settings_autosave_without_a_save_button(
    page: Page,
    base_url: str,
    api_base_url: str,
) -> None:
    page.set_default_timeout(15000)
    page.goto(f"{base_url}/settings/models")
    expect(page.get_by_role("button", name="Configure RAG settings")).to_be_visible()
    assert page.get_by_role("button", name="Save Configuration").count() == 0
    assert page.get_by_role("button", name="Save settings").count() == 0

    initial_model = _api_state(page, api_base_url, "/api/model-config")
    original = dict(initial_model["rag_settings"])
    expected = dict(original)
    page.get_by_role("button", name="Configure RAG settings").click()

    # Dependent numeric and text controls are exercised before switches that can disable them.
    editable_fields = [
        item for item in RAG_FIELDS if item[2] not in {"embedding_offline_mode", "allow_local_filesystem_access"}
    ]
    for tab, control_id, key in editable_fields:
        page.get_by_role("button", name=tab).click()
        control = _rag_control(page, control_id)
        candidate = _rag_candidate(control, key, expected)
        if isinstance(candidate, Decimal):
            value = float(candidate)
        else:
            value = candidate
        _save_rag_value(page, api_base_url, control, key, value)
        expected[key] = value

    # Exercise the remaining independent toggles after their dependent controls.
    for tab, control_id, key in [
        ("Retrieval", "rag-hybrid-toggle", "use_hybrid_search"),
        ("Retrieval", "rag-rerank-toggle", "use_reranking"),
        ("Embeddings", "rag-embedding-offline", "embedding_offline_mode"),
        ("Index", "rag-local-filesystem", "allow_local_filesystem_access"),
    ]:
        page.get_by_role("button", name=tab).click()
        candidate = not expected[key]
        _save_rag_value(page, api_base_url, _rag_control(page, control_id), key, candidate)
        expected[key] = candidate

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

    # Re-enable dependent controls, restore their values, then restore the switches.
    page.get_by_role("button", name="Retrieval").click()
    for control_id in ("rag-hybrid-toggle", "rag-rerank-toggle"):
        toggle = _rag_control(page, control_id)
        if not toggle.is_checked():
            toggle.check()
    with _expect_model_write(page) as response_info:
        for key in ("retrieval_candidate_count", "retrieval_selected_count", "hybrid_vector_weight", "hybrid_text_weight"):
            _set_rag_value(
                _rag_control(page, next(control_id for _, control_id, field in RAG_FIELDS if field == key)),
                key,
                original[key],
            )
    response = response_info.value
    assert response.status == 200, response.text()

    for tab, keys in {
        "Chunking": ["chunk_size", "chunk_overlap"],
        "Embeddings": ["embedding_batch_size", "embedding_offline_mode"],
        "Ranking": ["reranker_model"],
        "Index": ["vector_collection_name", "vector_index_metric", "vector_index_type", "allow_local_filesystem_access", "vector_stream_batch_size"],
    }.items():
        page.get_by_role("button", name=tab).click()
        with _expect_model_write(page) as response_info:
            for key in keys:
                control_id = next(control_id for _, control_id, field in RAG_FIELDS if field == key)
                _set_rag_value(_rag_control(page, control_id), key, original[key])
        response = response_info.value
        assert response.status == 200, response.text()

    page.get_by_role("button", name="Retrieval").click()
    toggles_to_restore = [
        ("rag-hybrid-toggle", "use_hybrid_search"),
        ("rag-rerank-toggle", "use_reranking"),
    ]
    toggles_to_restore = [
        (control_id, key)
        for control_id, key in toggles_to_restore
        if _rag_control(page, control_id).is_checked() is not original[key]
    ]
    if toggles_to_restore:
        with _expect_model_write(page) as response_info:
            for control_id, key in toggles_to_restore:
                _set_rag_value(_rag_control(page, control_id), key, original[key])
        response = response_info.value
        assert response.status == 200, response.text()
    # Multi-field restores save in sequence; wait for the last write before reading state.
    expect(page.locator(".model-config-rag-modal-footer")).to_contain_text("All changes saved")
    assert _api_state(page, api_base_url, "/api/model-config")["rag_settings"] == original
    page.get_by_role("button", name="Done").click()
    # The modal dialog keeps the page inert until it closes, so key presses would be lost.
    expect(page.get_by_role("dialog", name="RAG settings")).to_be_hidden()

    # Reasoning is a continuous input; the latest value autosaves and is restored after reload.
    original_level = initial_model["reasoning_level"]
    slider = page.get_by_label("Enable reasoning")
    original_slider = int(slider.input_value())
    target_slider = original_slider + 1 if original_slider < 3 else original_slider - 1
    with _expect_model_write(page) as response_info:
        slider.press("ArrowRight" if target_slider > original_slider else "ArrowLeft")
    response = response_info.value
    assert response.status == 200, response.text()
    expected_level = ["off", "low", "medium", "high"][target_slider]
    assert _api_state(page, api_base_url, "/api/model-config")["reasoning_level"] == expected_level
    page.reload()
    slider = page.get_by_label("Enable reasoning")
    assert int(slider.input_value()) == target_slider
    with _expect_model_write(page) as response_info:
        slider.press("ArrowLeft" if target_slider > original_slider else "ArrowRight")
    restored = response_info.value
    assert restored.status == 200, restored.text()
    assert _api_state(page, api_base_url, "/api/model-config")["reasoning_level"] == original_level


###############################################################################
def test_each_local_model_role_autosaves_and_survives_reload(
    page: Page,
    base_url: str,
    api_base_url: str,
) -> None:
    page.set_default_timeout(15000)
    page.goto(f"{base_url}/settings/models")
    original = _api_state(page, api_base_url, "/api/model-config")
    alternate_model = "qwen3.5:4b"
    ollama_url = os.environ.get("OLLAMA_URL")
    assert ollama_url, "Full browser E2E supplies an isolated fake Ollama endpoint."

    pulled = page.request.post(f"{ollama_url.rstrip('/')}/api/pull", data={"name": alternate_model})
    assert pulled.status == 200, pulled.text()
    with page.expect_response(
        lambda response: response.request.method == "POST"
        and response.url.endswith("/api/model-config/catalogs/ollama/refresh")
    ) as refresh_info:
        page.get_by_role("button", name="Refresh model catalog").click()
    refreshed = refresh_info.value
    assert refreshed.status == 200, refreshed.text()
    expect(page.get_by_role("button", name=f"Set {alternate_model} as clinical model")).to_be_visible()

    roles = {
        "clinical_model": "clinical model",
        "text_extraction_model": "text extraction model",
        "revision_model": "Revision model",
        "timeline_model": "Timeline model",
    }
    for field, role_label in roles.items():
        with _expect_model_write(page) as response_info:
            page.get_by_role("button", name=f"Set {alternate_model} as {role_label}").click()
        response = response_info.value
        assert response.status == 200, response.text()
        assert response.json()[field] == alternate_model
        page.reload()
        assert _api_state(page, api_base_url, "/api/model-config")[field] == alternate_model
        expect(page.get_by_role("button", name=f"{role_label.capitalize()} selected")).to_be_visible()

        original_model = original[field]
        with _expect_model_write(page) as restore_info:
            page.get_by_role("button", name=f"Set {original_model} as {role_label}").click()
        restored = restore_info.value
        assert restored.status == 200, restored.text()
        assert _api_state(page, api_base_url, "/api/model-config")[field] == original_model


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
    draft_read = page.request.get(f"{api_base_url}/api/clinical/draft")
    assert draft_read.status == 200, draft_read.text()
    assert draft_read.headers.get("cache-control") == "no-store"
    expect(page.get_by_role("alert")).to_contain_text("A required service is unavailable")
