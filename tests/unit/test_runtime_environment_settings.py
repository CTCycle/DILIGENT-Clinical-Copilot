from __future__ import annotations

import json

import pytest

from DILIGENT.server.common import constants
from DILIGENT.server.configurations import environment, sources
from DILIGENT.server.configurations.bootstrap import get_app_settings, reset_app_settings_cache


def test_initialize_environment_loads_dotenv_without_overriding_existing(tmp_path, monkeypatch) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("DILIGENT_TAURI_MODE=true\nFASTAPI_HOST=0.0.0.0\n", encoding="utf-8")
    monkeypatch.setattr(environment, "ENV_FILE_PATH", str(dotenv_path))
    monkeypatch.setenv("FASTAPI_HOST", "127.0.0.1")
    monkeypatch.setattr(environment, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.setattr(environment, "_DOTENV_INJECTED_KEYS", set())

    environment.initialize_environment()

    assert environment.get_dotenv_injected_keys()
    assert "DILIGENT_TAURI_MODE" in environment.get_dotenv_injected_keys()
    assert "FASTAPI_HOST" not in environment.get_dotenv_injected_keys()


def test_ui_owned_env_keys_are_rejected(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(constants, "CONFIGURATIONS_FILE", str(config_path))
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    reset_app_settings_cache()
    with pytest.raises(RuntimeError, match="UI-owned runtime keys"):
        get_app_settings()

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    reset_app_settings_cache()


def test_ui_owned_json_keys_are_rejected(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(
        json.dumps({"llm_defaults": {"cloud_model": "gpt-5-mini"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(constants, "CONFIGURATIONS_FILE", str(config_path))
    monkeypatch.setattr(sources, "_read_dotenv", lambda: {})

    reset_app_settings_cache()
    with pytest.raises(RuntimeError, match="UI-owned runtime keys"):
        get_app_settings()

    reset_app_settings_cache()
