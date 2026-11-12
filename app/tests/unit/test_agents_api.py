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
    agent_payload,
    assessment_payload,
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
def _create(
    client: TestClient, name: str = "Precise", payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    response = client.post(
        "/api/agents", json={"name": name, "configuration": payload or agent_payload()}
    )
    assert response.status_code == 201, response.text
    return response.json()


###############################################################################
def test_create_get_and_list_agents(client: TestClient) -> None:
    created = _create(client)
    assert created["current_revision"] == 1 and created["is_default"] is True
    assert created["capabilities"] == ["assessment", "revision", "timeline"]
    assert created["is_active"] is True and created["is_locked"] is False
    assert created["editable"] is True and created["reference_count"] == 0

    assert client.get(f"/api/agents/{created['id']}").json()["name"] == "Precise"
    listed = client.get("/api/agents").json()["agents"]
    assert [item["id"] for item in listed] == [created["id"]]
    assert client.get("/api/agents/missing").status_code == 404


###############################################################################
def test_selectable_for_filters_by_capability_and_lifecycle(
    client: TestClient,
) -> None:
    full = _create(client, "Full")
    partial_payload = agent_payload(timeline={"enabled": False})
    partial = _create(client, "No timeline", partial_payload)

    def ids(capability: str) -> list[str]:
        response = client.get("/api/agents", params={"selectable_for": capability})
        return [item["id"] for item in response.json()["agents"]]

    assert ids("assessment") == [full["id"], partial["id"]]
    assert ids("timeline") == [full["id"]]
    client.put(f"/api/agents/{partial['id']}/lifecycle", json={"is_active": False})
    assert ids("assessment") == [full["id"]]
    assert client.get("/api/agents", params={"selectable_for": "x"}).status_code == 422


###############################################################################
def test_update_creates_revisions_and_rejects_stale_or_non_editable_saves(
    client: TestClient,
) -> None:
    created = _create(client)
    url = f"/api/agents/{created['id']}"
    edited = assessment_payload(instructions={"custom_instructions": "Chronology first."})
    body = {"expected_revision": 1, "name": "Precise", "configuration": edited}

    updated = client.put(url, json=body)
    assert updated.status_code == 200
    assert updated.json()["current_revision"] == 2
    assert client.put(url, json=body).status_code == 409
    assert (
        client.put(
            "/api/agents/missing", json={**body, "expected_revision": 1}
        ).status_code
        == 404
    )

    locked = client.put(f"{url}/lifecycle", json={"is_locked": True}).json()
    assert locked["editable"] is False and locked["current_revision"] == 2
    assert client.put(url, json={**body, "expected_revision": 2}).status_code == 409
    client.put(f"{url}/lifecycle", json={"is_locked": False})
    assert client.put(url, json={**body, "expected_revision": 2}).status_code == 200

    revisions = client.get(f"{url}/revisions").json()["revisions"]
    assert [item["revision_number"] for item in revisions] == [3, 2, 1]


###############################################################################
def test_invalid_request_bodies_are_rejected(client: TestClient) -> None:
    bad_retrieval = agent_payload(
        assessment={
            "retrieval": {
                "retrieval_candidate_count": 4,
                "retrieval_selected_count": 5,
            }
        }
    )
    response = client.post(
        "/api/agents", json={"name": "Bad", "configuration": bad_retrieval}
    )
    assert response.status_code == 422
    missing = agent_payload()
    del missing["timeline"]
    assert (
        client.post(
            "/api/agents", json={"name": "Bad", "configuration": missing}
        ).status_code
        == 422
    )
    default = _create(client)
    assert (
        client.put(f"/api/agents/{default['id']}/lifecycle", json={}).status_code == 422
    )


###############################################################################
def test_default_lifecycle_duplicate_and_delete(client: TestClient) -> None:
    default = _create(client)
    url = f"/api/agents/{default['id']}"
    assert client.put(f"{url}/lifecycle", json={"is_archived": True}).status_code == 409
    assert client.put(f"{url}/lifecycle", json={"is_active": False}).status_code == 409
    assert client.delete(url).status_code == 409

    duplicate = client.post(f"{url}/duplicate")
    assert duplicate.status_code == 201
    copy = duplicate.json()
    assert copy["is_default"] is False and copy["id"] != default["id"]
    named = client.post(f"{url}/duplicate", json={"name": "Mine"})
    assert named.json()["name"] == "Mine"

    promoted = client.put(f"/api/agents/{copy['id']}/default").json()
    assert promoted["is_default"] is True
    archived = client.put(f"{url}/lifecycle", json={"is_archived": True}).json()
    assert archived["is_archived"] is True and archived["editable"] is False
    ids = [a["id"] for a in client.get("/api/agents").json()["agents"]]
    assert default["id"] not in ids
    ids_all = [
        a["id"]
        for a in client.get("/api/agents", params={"include_archived": True}).json()[
            "agents"
        ]
    ]
    assert default["id"] in ids_all
    # An archived agent can be restored and cannot become the default directly.
    assert client.put(f"{url}/default").status_code == 409
    restored = client.put(f"{url}/lifecycle", json={"is_archived": False}).json()
    assert restored["editable"] is True

    # Unreferenced, unlocked, non-default agents can be deleted permanently.
    client.put(f"/api/agents/{named.json()['id']}/lifecycle", json={"is_locked": True})
    assert client.delete(f"/api/agents/{named.json()['id']}").status_code == 409
    client.put(f"/api/agents/{named.json()['id']}/lifecycle", json={"is_locked": False})
    assert client.delete(f"/api/agents/{named.json()['id']}").status_code == 204
    assert client.get(f"/api/agents/{named.json()['id']}").status_code == 404
