from __future__ import annotations

from DILIGENT.server.configurations.server import build_external_data_settings


# -----------------------------------------------------------------------------
def test_external_data_timeouts_respect_json_overrides() -> None:
    settings = build_external_data_settings(
        {
            "default_llm_timeout": 120.0,
            "parser_llm_timeout": 45.0,
            "disease_llm_timeout": 55.0,
            "clinical_llm_timeout": 240.0,
            "livertox_llm_timeout": 600.0,
            "ollama_server_start_timeout": 22.0,
        },
        fallback_timeout=30.0,
    )

    assert settings.default_llm_timeout == 120.0
    assert settings.parser_llm_timeout == 45.0
    assert settings.disease_llm_timeout == 55.0
    assert settings.clinical_llm_timeout == 240.0
    assert settings.livertox_llm_timeout == 600.0
    assert settings.ollama_server_start_timeout == 22.0


# -----------------------------------------------------------------------------
def test_external_data_timeouts_apply_safe_minimum() -> None:
    settings = build_external_data_settings(
        {
            "default_llm_timeout": 0.0,
            "parser_llm_timeout": 0.0,
            "disease_llm_timeout": 0.0,
            "clinical_llm_timeout": 0.0,
            "livertox_llm_timeout": 0.0,
            "ollama_server_start_timeout": 0.0,
        },
        fallback_timeout=30.0,
    )

    assert settings.default_llm_timeout == 1.0
    assert settings.parser_llm_timeout == 1.0
    assert settings.disease_llm_timeout == 1.0
    assert settings.clinical_llm_timeout == 1.0
    assert settings.livertox_llm_timeout == 1.0
    assert settings.ollama_server_start_timeout == 1.0
