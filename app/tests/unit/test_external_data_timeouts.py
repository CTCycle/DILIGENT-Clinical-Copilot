from __future__ import annotations

from configurations.management import (
    EnvironmentSnapshot,
    build_settings_payload_from_json,
)


def _env() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        ollama_url=None,
        ollama_host="localhost",
        ollama_port=11434,
    )


def test_runtime_timeouts_respect_values_and_minimums() -> None:
    payload = build_settings_payload_from_json(
        {
            "runtime": {
                "default_llm_timeout": 120.0,
                "parser_llm_timeout": 45.0,
                "disease_llm_timeout": 55.0,
                "clinical_llm_timeout": 240.0,
                "livertox_llm_timeout": 600.0,
                "ollama_server_start_timeout": 22.0,
            }
        },
        _env(),
    )
    settings = payload["runtime"]

    assert settings["default_llm_timeout"] == 120.0
    assert settings["parser_llm_timeout"] == 120.0
    assert settings["disease_llm_timeout"] == 120.0
    assert settings["clinical_llm_timeout"] == 120.0
    assert settings["livertox_llm_timeout"] == 600.0
    assert settings["ollama_server_start_timeout"] == 22.0


def test_runtime_timeouts_floor_to_positive_values() -> None:
    payload = build_settings_payload_from_json(
        {
            "runtime": {
                "default_llm_timeout": -100.0,
                "parser_llm_timeout": 0.0,
                "disease_llm_timeout": -1.0,
                "clinical_llm_timeout": 0.0,
                "livertox_llm_timeout": -5.0,
                "ollama_server_start_timeout": 0.0,
            }
        },
        _env(),
    )
    settings = payload["runtime"]

    assert settings["default_llm_timeout"] == 1.0
    assert settings["parser_llm_timeout"] == 1.0
    assert settings["disease_llm_timeout"] == 1.0
    assert settings["clinical_llm_timeout"] == 1.0
    assert settings["livertox_llm_timeout"] == 1.0
    assert settings["ollama_server_start_timeout"] == 1.0


def test_runtime_timeouts_allow_long_clinical_budget_without_legacy_cap() -> None:
    payload = build_settings_payload_from_json(
        {
            "runtime": {
                "default_llm_timeout": 7200.0,
                "clinical_llm_timeout": 9000.0,
            }
        },
        _env(),
    )
    settings = payload["runtime"]

    assert settings["default_llm_timeout"] == 7200.0
    assert settings["clinical_llm_timeout"] == 7200.0
