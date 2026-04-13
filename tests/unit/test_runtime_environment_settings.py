from __future__ import annotations

import json
import os

from DILIGENT.server.common import constants
from DILIGENT.server.configurations import environment_bootstrap as environment
from DILIGENT.server.configurations.bootstrap import get_app_settings, reset_app_settings_cache


def test_initialize_environment_loads_dotenv_with_override_precedence(tmp_path, monkeypatch) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("DILIGENT_TAURI_MODE=true\nFASTAPI_HOST=0.0.0.0\n", encoding="utf-8")
    monkeypatch.setattr(constants, "ENV_FILE_PATH", str(dotenv_path))
    monkeypatch.setenv("FASTAPI_HOST", "127.0.0.1")
    environment.reset_environment_bootstrap_for_tests()

    environment.initialize_environment()

    assert environment.get_dotenv_injected_keys()
    assert "DILIGENT_TAURI_MODE" in environment.get_dotenv_injected_keys()
    assert os.environ.get("FASTAPI_HOST") == "0.0.0.0"


def test_ui_owned_env_keys_do_not_override_json_runtime_defaults(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(constants, "CONFIGURATIONS_FILE", str(config_path))
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    reset_app_settings_cache()
    settings = get_app_settings()
    assert settings.llm_defaults.llm_provider == "openai"

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    reset_app_settings_cache()


def test_ui_owned_json_keys_are_ignored(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(
        json.dumps({"llm_defaults": {"cloud_model": "gpt-5-mini"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(constants, "CONFIGURATIONS_FILE", str(config_path))

    reset_app_settings_cache()
    settings = get_app_settings()
    assert settings.llm_defaults.cloud_model == constants.OPENAI_CLOUD_MODELS[0]

    reset_app_settings_cache()
