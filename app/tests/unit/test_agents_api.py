# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from api.agents import AgentsEndpoint
from api.error_handling import register_error_handling
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from repositories.schemas.base import Base
from repository_fixtures import build_repository_graph
from services.agents import AgentService
from sqlalchemy import create_engine
from test_agents import (
    _FakeAccessKeys,
    _FakeModelConfigService,
    dili_payload,
    revision_payload,
)


###############################################################################
@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'agents.db'}", future=True)
    Base.metadata.create_all(engine)
    service = AgentService(
        build_repository_graph(engine=engine).agent_repository,
        model_config_service=cast(Any, _FakeModelConfigService()),
        access_key_service=cast(Any, _FakeAccessKeys(active=False)),
    )
    router = APIRouter(prefix="/agents", tags=["agents-test"])
    AgentsEndpoint(router=router, service=service).add_routes()
    application = FastAPI()
    register_error_handling(application)
    application.include_router(router, prefix="/api")
    return TestClient(application)


# -----------------------------------------------------------------------------
def _create(client: TestClient, name: str = "Precise") -> dict[str, Any]:
    response = client.post(
        "/api/agents", json={"name": name, "configuration": dili_payload()}
    )
    assert response.status_code == 201, response.text
    return response.json()


###############################################################################
def test_create_get_and_list_agents(client: TestClient) -> None:
    created = _create(client)
    assert created["current_revision"] == 1 and created["is_default"] is True

    assert client.get(f"/api/agents/{created['id']}").json()["name"] == "Precise"
    listed = client.get("/api/agents", params={"kind": "dili"}).json()["agents"]
    assert [item["id"] for item in listed] == [created["id"]]
    assert client.get("/api/agents", params={"kind": "revision"}).json() == {
        "agents": []
    }
    assert client.get("/api/agents/missing").status_code == 404


###############################################################################
def test_update_creates_revisions_and_rejects_stale_or_foreign_config(
    client: TestClient,
) -> None:
    created = _create(client)
    url = f"/api/agents/{created['id']}"
    edited = dili_payload(instructions={"assessment_focus": "Chronology first."})
    body = {"expected_revision": 1, "name": "Precise", "configuration": edited}

    updated = client.put(url, json=body)
    assert updated.status_code == 200
    assert updated.json()["current_revision"] == 2
    assert client.put(url, json=body).status_code == 409

    wrong_kind = {**body, "expected_revision": 2, "configuration": revision_payload()}
    assert client.put(url, json=wrong_kind).status_code == 422
    assert (
        client.put(
            "/api/agents/missing", json={**body, "expected_revision": 1}
        ).status_code
        == 404
    )

    revisions = client.get(f"{url}/revisions").json()["revisions"]
    assert [item["revision_number"] for item in revisions] == [2, 1]


###############################################################################
def test_invalid_request_bodies_are_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/agents",
        json={
            "name": "Bad",
            "configuration": revision_payload(
                retrieval={
                    "retrieval_candidate_count": 4,
                    "retrieval_selected_count": 2,
                }
            ),
        },
    )
    assert response.status_code == 422


###############################################################################
def test_default_archive_duplicate_and_validate(client: TestClient) -> None:
    default = _create(client)
    assert client.post(f"/api/agents/{default['id']}/archive").status_code == 409

    duplicate = client.post(f"/api/agents/{default['id']}/duplicate")
    assert duplicate.status_code == 201
    copy = duplicate.json()
    assert copy["is_default"] is False and copy["id"] != default["id"]
    named = client.post(f"/api/agents/{default['id']}/duplicate", json={"name": "Mine"})
    assert named.json()["name"] == "Mine"

    assert client.get(f"/api/agents/{copy['id']}/revisions").status_code == 200
    validation = client.post(f"/api/agents/{copy['id']}/validate")
    assert validation.status_code == 200 and "valid" in validation.json()

    promoted = client.put(f"/api/agents/{copy['id']}/default").json()
    assert promoted["is_default"] is True
    archived = client.post(f"/api/agents/{default['id']}/archive").json()
    assert archived["is_archived"] is True
    ids = [a["id"] for a in client.get("/api/agents").json()["agents"]]
    assert default["id"] not in ids
    ids_all = [
        a["id"]
        for a in client.get("/api/agents", params={"include_archived": True}).json()[
            "agents"
        ]
    ]
    assert default["id"] in ids_all
