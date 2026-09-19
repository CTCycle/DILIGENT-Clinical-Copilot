from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from common.exceptions import ServiceValidationError
from domain.settings.runtime_ui import RuntimeSettingsUpdateRequest
from services.settings.runtime import RuntimeSettingsService


###############################################################################
def _write_configuration(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "jobs": {"polling_interval": 1},
                "rag": {
                    "vector_collection_name": "documents",
                    "allow_local_filesystem_access": True,
                },
                "runtime": {
                    "default_llm_timeout": 3600.0,
                    "clinical_llm_timeout": 3600.0,
                    "livertox_llm_timeout": 3600.0,
                    "minimum_llm_timeout": 5.0,
                    "cloud_llm_timeout_cap": 1800.0,
                    "local_llm_timeout_cap": 45.0,
                    "livertox_download_timeout": 30.0,
                    "livertox_archive": "custom-livertox.tar.gz",
                    "livertox_yield_interval": 25,
                    "livertox_skip_deterministic_ratio": 0.8,
                    "livertox_monograph_max_workers": 4,
                    "max_excerpt_length": 8000,
                    "rxnav_request_timeout": 12.0,
                    "rxnav_max_concurrency": 10,
                },
                "ingestion": {
                    "drug_name_min_length": 3,
                    "drug_name_max_length": 200,
                    "drug_name_max_tokens": 8,
                },
                "session_pipeline": {
                    "text_extraction_batch_size": 4,
                    "text_extraction_max_concurrency": 2,
                },
                "clinical_language_detection": {"min_best_score": 2.0},
                "drugs_matcher": {"direct_confidence": 1.0},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


###############################################################################
def test_runtime_settings_state_exposes_only_supported_json_settings(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configurations.json"
    _write_configuration(config_path)
    service = RuntimeSettingsService(config_path=config_path)

    state = service.get_state()

    assert state.source == "settings/configurations.json"
    assert state.environment_editable is False
    assert state.values.general.polling_interval == 1.0
    assert state.values.integrations.rxnav_max_concurrency == 10
    assert set(state.values.model_dump()) == {
        "general",
        "data",
        "integrations",
        "advanced",
    }
    serialized = json.dumps(state.model_dump(mode="json"))
    assert "DATABASE_URL" not in serialized
    assert "password" not in serialized.lower()
    assert "livertox_archive" not in serialized


###############################################################################
def test_runtime_settings_partial_update_persists_and_preserves_static_keys(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configurations.json"
    _write_configuration(config_path)
    service = RuntimeSettingsService(config_path=config_path)

    updated = service.update_state(
        RuntimeSettingsUpdateRequest.model_validate(
            {
                "general": {"polling_interval": 2.5},
                "integrations": {
                    "rxnav_request_timeout": 20.0,
                    "rxnav_max_concurrency": 12,
                },
            }
        )
    )

    assert updated.values.general.polling_interval == 2.5
    assert updated.values.integrations.rxnav_request_timeout == 20.0
    assert updated.values.integrations.rxnav_max_concurrency == 12
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["jobs"]["polling_interval"] == 2.5
    assert persisted["runtime"]["livertox_archive"] == "custom-livertox.tar.gz"
    assert persisted["clinical_language_detection"] == {"min_best_score": 2.0}
    assert persisted["session_pipeline"]["text_extraction_batch_size"] == 4


###############################################################################
def test_runtime_settings_reload_observes_persisted_update(tmp_path: Path) -> None:
    config_path = tmp_path / "configurations.json"
    _write_configuration(config_path)
    service = RuntimeSettingsService(config_path=config_path)
    service.update_state(
        RuntimeSettingsUpdateRequest.model_validate(
            {"data": {"drug_name_max_tokens": 11}}
        )
    )

    reloaded = RuntimeSettingsService(config_path=config_path).get_state()

    assert reloaded.values.data.drug_name_max_tokens == 11


###############################################################################
def test_runtime_settings_reset_restores_checked_in_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "configurations.json"
    _write_configuration(config_path)
    service = RuntimeSettingsService(config_path=config_path)
    service.update_state(
        RuntimeSettingsUpdateRequest.model_validate(
            {"advanced": {"local_llm_timeout_cap": 120.0}}
        )
    )

    reset = service.reset_category("advanced")

    assert reset.values.advanced == reset.defaults.advanced
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["runtime"]["local_llm_timeout_cap"] == 45.0
    assert persisted["runtime"]["livertox_archive"] == "custom-livertox.tar.gz"


###############################################################################
def test_runtime_settings_reject_unknown_environment_fields() -> None:
    with pytest.raises(ValidationError):
        RuntimeSettingsUpdateRequest.model_validate(
            {
                "general": {
                    "polling_interval": 2.0,
                    "DATABASE_URL": "postgresql://example.invalid/db",
                }
            }
        )


###############################################################################
def test_runtime_settings_reject_incompatible_timeout_caps(tmp_path: Path) -> None:
    config_path = tmp_path / "configurations.json"
    _write_configuration(config_path)
    service = RuntimeSettingsService(config_path=config_path)

    with pytest.raises(ServiceValidationError, match="Local LLM timeout cap"):
        service.update_state(
            RuntimeSettingsUpdateRequest.model_validate(
                {
                    "advanced": {
                        "minimum_llm_timeout": 60.0,
                        "local_llm_timeout_cap": 30.0,
                    }
                }
            )
        )
