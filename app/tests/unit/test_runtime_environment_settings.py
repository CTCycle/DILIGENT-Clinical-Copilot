from __future__ import annotations

import json
import os

from common import paths
from configurations import environment
from configurations.startup import (
    get_server_settings,
    initialize_environment,
    reset_app_settings_cache,
)

###############################################################################
def test_initialize_environment_preserves_process_environment_precedence(
    tmp_path, monkeypatch
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "DILIGENT_TAURI_MODE=true\nFASTAPI_HOST=0.0.0.0\n", encoding="utf-8"
    )
    monkeypatch.setattr(paths, "ENV_FILE_PATH", dotenv_path)
    monkeypatch.setenv("FASTAPI_HOST", "127.0.0.1")
    environment.reset_environment_bootstrap_for_tests()

    initialize_environment()

    assert environment.get_dotenv_injected_keys()
    assert "DILIGENT_TAURI_MODE" in environment.get_dotenv_injected_keys()
    assert os.environ.get("FASTAPI_HOST") == "127.0.0.1"

###############################################################################
def test_initialize_environment_creates_env_from_example_when_missing(
    tmp_path, monkeypatch
) -> None:
    dotenv_path = tmp_path / ".env"
    example_path = tmp_path / ".env.example"
    example_path.write_text(
        "DILIGENT_TAURI_MODE=true\nFASTAPI_HOST=127.0.0.1\n", encoding="utf-8"
    )
    monkeypatch.setattr(paths, "ENV_FILE_PATH", dotenv_path)
    monkeypatch.setattr(paths, "ENV_EXAMPLE_PATH", example_path)
    environment.reset_environment_bootstrap_for_tests()

    initialize_environment()

    assert dotenv_path.read_text(encoding="utf-8") == example_path.read_text(
        encoding="utf-8"
    )
    assert os.environ.get("DILIGENT_TAURI_MODE") == "true"

###############################################################################
def test_ui_owned_env_keys_do_not_override_json_runtime_defaults(
    monkeypatch, tmp_path
) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(paths, "CONFIGURATIONS_FILE", config_path)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    reset_app_settings_cache()
    settings = get_server_settings()
    assert settings.llm_defaults.llm_provider == "openai"

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    reset_app_settings_cache()
