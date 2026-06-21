from __future__ import annotations

import json
import os

from common import constants
from common import paths
from configurations import environment
from configurations.startup import (
    get_server_settings,
    reset_app_settings_cache,
)
from configurations.management import build_settings_payload_from_json
from domain.settings.environment import EnvironmentSnapshot

###############################################################################
def test_initialize_environment_loads_dotenv_with_override_precedence(
    tmp_path, monkeypatch
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "DILIGENT_TAURI_MODE=true\nFASTAPI_HOST=0.0.0.0\n", encoding="utf-8"
    )
    monkeypatch.setattr(paths, "ENV_FILE_PATH", dotenv_path)
    monkeypatch.setenv("FASTAPI_HOST", "127.0.0.1")
    environment.reset_environment_bootstrap_for_tests()

    environment.initialize_environment()

    assert environment.get_dotenv_injected_keys()
    assert "DILIGENT_TAURI_MODE" in environment.get_dotenv_injected_keys()
    assert os.environ.get("FASTAPI_HOST") == "0.0.0.0"

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

###############################################################################
def test_ui_owned_json_keys_are_ignored(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(
        json.dumps({"llm_defaults": {"cloud_model": "gpt-5-mini"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "CONFIGURATIONS_FILE", config_path)

    reset_app_settings_cache()

###############################################################################
def test_deployment_mode_defaults_to_local_single_user() -> None:
    payload = build_settings_payload_from_json(
        {},
        EnvironmentSnapshot(
            ollama_url=None,
            ollama_host=None,
            ollama_port=None,
        ),
    )
    assert payload["deployment"]["mode"] == "local_single_user"
