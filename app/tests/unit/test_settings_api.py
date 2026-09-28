from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

import services.settings.runtime as runtime_module
from api.error_handling import register_error_handling
from api.settings import SettingsEndpoint
from configurations.management import (
    build_default_application_configuration_payload,
    environment_snapshot_from_os_env,
)
from configurations.startup import ConfigurationManager
from services.settings.runtime import RuntimeSettingsService


class MemoryApplicationConfigurationSerializer:
    def __init__(self) -> None:
        self.payload = build_default_application_configuration_payload(
            environment_snapshot_from_os_env()
        )
        self.updated_at = datetime.now(UTC)

    def load(self) -> dict[str, Any]:
        return dict(self.payload)

    def load_with_metadata(self) -> tuple[dict[str, Any], datetime]:
        return dict(self.payload), self.updated_at

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payload = dict(payload)
        self.updated_at = datetime.now(UTC)
        return dict(self.payload)


def _client(monkeypatch) -> tuple[TestClient, MemoryApplicationConfigurationSerializer]:  # type: ignore[no-untyped-def]
    serializer = MemoryApplicationConfigurationSerializer()
    manager = ConfigurationManager(persisted_payload=serializer.payload)
    monkeypatch.setattr(runtime_module, "get_server_settings", lambda: manager.server_settings)
    monkeypatch.setattr(runtime_module, "get_configuration_manager", lambda: manager)
    router = APIRouter(prefix="/settings", tags=["settings-test"])
    SettingsEndpoint(
        router=router,
        service=RuntimeSettingsService(serializer=serializer),
    ).add_routes()
    application = FastAPI()
    register_error_handling(application)
    application.include_router(router, prefix="/api")
    return TestClient(application), serializer


def test_settings_api_get_patch_and_reset(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client, serializer = _client(monkeypatch)

    response = client.get("/api/settings")
    assert response.status_code == 200
    assert response.headers["cache-control"].startswith("no-store")
    assert response.json()["source"] == "database"
    assert response.json()["environment_editable"] is False
    assert "matching" in response.json()["values"]
    assert response.json()["values"]["data"]["clinical_assessment_batch_size"] == 2

    response = client.patch(
        "/api/settings",
        json={
            "general": {"polling_interval": 3.0},
            "data": {"retrieval_batch_size": 10},
            "matching": {"spelling_confidence": 0.8},
        },
    )
    assert response.status_code == 200
    assert response.json()["values"]["general"]["polling_interval"] == 3.0
    assert response.json()["values"]["data"]["retrieval_batch_size"] == 10
    assert response.json()["values"]["matching"]["spelling_confidence"] == 0.8
    assert serializer.payload["jobs"]["polling_interval"] == 3.0
    assert serializer.payload["session_pipeline"]["retrieval_batch_size"] == 10

    response = client.post("/api/settings/reset/general")
    assert response.status_code == 200
    assert response.json()["values"]["general"]["polling_interval"] == 1.0


def test_settings_api_rejects_unknown_or_invalid_fields(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client, _ = _client(monkeypatch)

    unknown = client.patch(
        "/api/settings",
        json={"general": {"polling_interval": 1.0, "DATABASE_URL": "hidden"}},
    )
    assert unknown.status_code == 422

    invalid = client.patch(
        "/api/settings",
        json={"integrations": {"rxnav_max_concurrency": 0}},
    )
    assert invalid.status_code == 422

    environment_field = client.patch(
        "/api/settings",
        json={"advanced": {"DATABASE_URL": "hidden"}},
    )
    assert environment_field.status_code == 422
