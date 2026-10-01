# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

import services.settings.runtime as runtime_module
from common.exceptions import ServiceValidationError
from configurations.management import (
    build_default_application_configuration_payload,
    environment_snapshot_from_os_env,
)
from configurations.startup import ConfigurationManager
from domain.settings.runtime_ui import RuntimeSettingsUpdateRequest
from services.settings.runtime import RuntimeSettingsService


class MemoryApplicationConfigurationSerializer:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = dict(payload)
        self.updated_at = datetime.now(UTC)

    def load(self) -> dict[str, Any]:
        return dict(self.payload)

    def load_with_metadata(self) -> tuple[dict[str, Any], datetime]:
        return dict(self.payload), self.updated_at

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payload = dict(payload)
        self.updated_at = datetime.now(UTC)
        return dict(self.payload)


def _service(monkeypatch) -> tuple[RuntimeSettingsService, MemoryApplicationConfigurationSerializer]:  # type: ignore[no-untyped-def]
    payload = build_default_application_configuration_payload(
        environment_snapshot_from_os_env()
    )
    payload["runtime"]["livertox_archive"] = "custom-livertox.tar.gz"
    serializer = MemoryApplicationConfigurationSerializer(payload)
    manager = ConfigurationManager(persisted_payload=payload)
    monkeypatch.setattr(runtime_module, "get_server_settings", lambda: manager.server_settings)
    monkeypatch.setattr(runtime_module, "get_configuration_manager", lambda: manager)
    return RuntimeSettingsService(serializer=serializer), serializer


def test_runtime_settings_state_exposes_complete_database_settings(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    service, _ = _service(monkeypatch)

    state = service.get_state()

    assert state.source == "database"
    assert state.environment_editable is False
    assert state.values.general.polling_interval == 1.0
    assert state.values.data.retrieval_batch_size == 8
    assert state.values.integrations.livertox_archive == "custom-livertox.tar.gz"
    assert state.values.matching.spelling_short_name_length == 8
    assert state.values.advanced.parser_llm_timeout == 3600.0
    assert set(state.values.model_dump()) == {
        "general",
        "data",
        "integrations",
        "matching",
        "advanced",
    }


def test_runtime_settings_partial_update_persists_all_mapped_blocks(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    service, serializer = _service(monkeypatch)

    updated = service.update_state(
        RuntimeSettingsUpdateRequest.model_validate(
            {
                "general": {"polling_interval": 2.5},
                "data": {
                    "retrieval_batch_size": 12,
                    "min_best_score": 3.0,
                },
                "integrations": {
                    "rxnav_request_timeout": 20.0,
                    "rxnav_max_concurrency": 12,
                },
                "matching": {"spelling_short_name_length": 10},
            }
        )
    )

    assert updated.values.general.polling_interval == 2.5
    assert updated.values.data.retrieval_batch_size == 12
    assert updated.values.data.min_best_score == 3.0
    assert updated.values.integrations.rxnav_request_timeout == 20.0
    assert updated.values.matching.spelling_short_name_length == 10
    assert serializer.payload["jobs"]["polling_interval"] == 2.5
    assert serializer.payload["session_pipeline"]["retrieval_batch_size"] == 12
    assert serializer.payload["clinical_language_detection"]["min_best_score"] == 3.0
    assert serializer.payload["runtime"]["livertox_archive"] == "custom-livertox.tar.gz"
    assert serializer.payload["drugs_matcher"]["spelling_short_name_length"] == 10


def test_runtime_settings_persist_reload_and_reset_ncbi_contact(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    service, serializer = _service(monkeypatch)
    updated = service.update_state(
        RuntimeSettingsUpdateRequest.model_validate(
            {"integrations": {"ncbi_contact_email": " developer@example.org "}}
        )
    )

    assert updated.values.integrations.ncbi_contact_email == "developer@example.org"
    assert serializer.payload["runtime"]["ncbi_contact_email"] == "developer@example.org"

    reloaded_manager = ConfigurationManager(persisted_payload=serializer.payload)
    monkeypatch.setattr(runtime_module, "get_server_settings", lambda: reloaded_manager.server_settings)
    monkeypatch.setattr(runtime_module, "get_configuration_manager", lambda: reloaded_manager)
    reloaded_service = RuntimeSettingsService(serializer=serializer)
    reloaded = reloaded_service.get_state()
    assert reloaded.values.integrations.ncbi_contact_email == "developer@example.org"

    reset = reloaded_service.reset_category("integrations")
    assert reset.values.integrations.ncbi_contact_email == "clinical-copilot@pharmagent.local"
    assert serializer.payload["runtime"]["ncbi_contact_email"] == "clinical-copilot@pharmagent.local"


def test_runtime_settings_old_payload_uses_ncbi_default(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    service, serializer = _service(monkeypatch)
    serializer.payload["runtime"].pop("ncbi_contact_email", None)
    manager = ConfigurationManager(persisted_payload=serializer.payload)
    monkeypatch.setattr(runtime_module, "get_server_settings", lambda: manager.server_settings)

    state = RuntimeSettingsService(serializer=serializer).get_state()

    assert state.values.integrations.ncbi_contact_email == "clinical-copilot@pharmagent.local"


def test_runtime_settings_reload_observes_persisted_update(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    service, serializer = _service(monkeypatch)
    service.update_state(
        RuntimeSettingsUpdateRequest.model_validate(
            {"data": {"drug_name_max_tokens": 11}}
        )
    )

    reloaded_manager = ConfigurationManager(persisted_payload=serializer.payload)
    monkeypatch.setattr(runtime_module, "get_server_settings", lambda: reloaded_manager.server_settings)
    reloaded = RuntimeSettingsService(serializer=serializer).get_state()

    assert reloaded.values.data.drug_name_max_tokens == 11


def test_runtime_settings_reset_restores_typed_defaults(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    service, serializer = _service(monkeypatch)
    service.update_state(
        RuntimeSettingsUpdateRequest.model_validate(
            {"advanced": {"local_llm_timeout_cap": 120.0}}
        )
    )

    reset = service.reset_category("advanced")

    assert reset.values.advanced == reset.defaults.advanced
    assert serializer.payload["runtime"]["local_llm_timeout_cap"] == 45.0
    assert serializer.payload["runtime"]["livertox_archive"] == "custom-livertox.tar.gz"


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


def test_runtime_settings_reject_incompatible_timeout_caps(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    service, _ = _service(monkeypatch)

    with pytest.raises(ServiceValidationError, match="Local LLM timeout cap"):
        service.update_state(
            RuntimeSettingsUpdateRequest.model_validate(
                {
                    "advanced": {
                        "minimum_llm_timeout": 60.0,
                        "cloud_llm_timeout_cap": 60.0,
                        "local_llm_timeout_cap": 30.0,
                    }
                }
            )
        )
