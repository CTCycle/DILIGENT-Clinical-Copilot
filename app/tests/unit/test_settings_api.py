from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from api.error_handling import register_error_handling
from api.settings import SettingsEndpoint
from services.settings.runtime import RuntimeSettingsService


def _client(tmp_path: Path) -> tuple[TestClient, Path]:
    config_path = tmp_path / "configurations.json"
    config_path.write_text(
        json.dumps(
            {
                "jobs": {"polling_interval": 1},
                "runtime": {
                    "default_llm_timeout": 3600.0,
                    "clinical_llm_timeout": 3600.0,
                    "livertox_llm_timeout": 3600.0,
                    "minimum_llm_timeout": 5.0,
                    "cloud_llm_timeout_cap": 1800.0,
                    "local_llm_timeout_cap": 45.0,
                    "livertox_download_timeout": 30.0,
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
            }
        ),
        encoding="utf-8",
    )
    router = APIRouter(prefix="/settings", tags=["settings-test"])
    SettingsEndpoint(
        router=router,
        service=RuntimeSettingsService(config_path=config_path),
    ).add_routes()
    application = FastAPI()
    register_error_handling(application)
    application.include_router(router, prefix="/api")
    return TestClient(application), config_path


def test_settings_api_get_patch_and_reset(tmp_path: Path) -> None:
    client, config_path = _client(tmp_path)

    response = client.get("/api/settings")
    assert response.status_code == 200
    assert response.headers["cache-control"].startswith("no-store")
    assert response.json()["environment_editable"] is False

    response = client.patch(
        "/api/settings",
        json={"general": {"polling_interval": 3.0}},
    )
    assert response.status_code == 200
    assert response.json()["values"]["general"]["polling_interval"] == 3.0
    assert json.loads(config_path.read_text(encoding="utf-8"))["jobs"] == {
        "polling_interval": 3.0
    }

    response = client.post("/api/settings/reset/general")
    assert response.status_code == 200
    assert response.json()["values"]["general"]["polling_interval"] == 1.0


def test_settings_api_rejects_unknown_or_invalid_fields(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)

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
