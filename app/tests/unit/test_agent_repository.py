# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from repositories.agent_repository import (
    AgentArchivedError,
    AgentDefaultArchiveError,
    AgentNotFoundError,
    AgentRepository,
    AgentRevisionConflictError,
)
from repositories.schemas.base import Base
from repositories.schemas.configuration import AgentDefinition
from repository_fixtures import build_repository_graph
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError


###############################################################################
@pytest.fixture
def repository(tmp_path: Path) -> AgentRepository:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'agents.db'}", future=True)
    Base.metadata.create_all(engine)
    return build_repository_graph(engine=engine).agent_repository


# -----------------------------------------------------------------------------
def _create(
    repository: AgentRepository,
    name: str = "Agent",
    kind: str = "dili",
    configuration: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return repository.create_agent(
        kind=kind,
        name=name,
        description="",
        configuration=configuration or {"v": 1},
        configuration_hash="a" * 64,
        schema_version=1,
        **kwargs,
    )


# -----------------------------------------------------------------------------
def _revise(
    repository: AgentRepository,
    agent_id: str,
    expected: int,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    return repository.create_revision(
        agent_id=agent_id,
        expected_revision=expected,
        name="Renamed",
        description="d",
        configuration=configuration,
        configuration_hash="b" * 64,
        schema_version=1,
    )


###############################################################################
def test_create_agent_writes_definition_and_revision_one(
    repository: AgentRepository,
) -> None:
    created = _create(repository)
    fetched = repository.get_agent(created["id"])
    assert fetched is not None
    assert fetched["current_revision"] == 1
    assert fetched["revision"]["revision_number"] == 1
    assert fetched["revision"]["configuration"] == {"v": 1}
    assert repository.count_agents() == 1


# -----------------------------------------------------------------------------
def test_update_creates_new_revision_and_keeps_previous_unchanged(
    repository: AgentRepository,
) -> None:
    agent = _create(repository)
    updated = _revise(repository, agent["id"], 1, {"v": 2})
    assert updated["current_revision"] == 2
    assert updated["name"] == "Renamed"
    revisions = repository.list_revisions(agent["id"])
    assert [row["revision_number"] for row in revisions] == [2, 1]
    assert revisions[1]["configuration"] == {"v": 1}
    assert revisions[1]["configuration_hash"] == "a" * 64


# -----------------------------------------------------------------------------
def test_stale_expected_revision_is_rejected(repository: AgentRepository) -> None:
    agent = _create(repository)
    _revise(repository, agent["id"], 1, {"v": 2})
    with pytest.raises(AgentRevisionConflictError):
        _revise(repository, agent["id"], 1, {"v": 3})
    assert len(repository.list_revisions(agent["id"])) == 2
    with pytest.raises(AgentNotFoundError):
        _revise(repository, "missing", 1, {"v": 3})


# -----------------------------------------------------------------------------
def test_single_default_per_kind(repository: AgentRepository) -> None:
    first = _create(repository, "First", is_default=True)
    second = _create(repository, "Second", is_default=True)
    other_kind = _create(repository, "Rev", kind="revision", is_default=True)
    assert repository.get_agent(first["id"])["is_default"] is False
    assert repository.get_default("dili")["id"] == second["id"]
    assert repository.get_default("revision")["id"] == other_kind["id"]
    repository.set_default(first["id"])
    assert repository.get_default("dili")["id"] == first["id"]
    assert repository.get_agent(second["id"])["is_default"] is False


# -----------------------------------------------------------------------------
def test_database_rejects_two_defaults_of_one_kind(
    repository: AgentRepository,
) -> None:
    _create(repository, "First", is_default=True)
    second = _create(repository, "Second")
    with pytest.raises(IntegrityError):
        with repository.session_factory.begin() as db_session:
            row = db_session.get(AgentDefinition, second["id"])
            row.is_default = True
            row.default_kind = "dili"


# -----------------------------------------------------------------------------
def test_archive_hides_agent_and_default_cannot_be_archived(
    repository: AgentRepository,
) -> None:
    default = _create(repository, "Default", is_default=True)
    spare = _create(repository, "Spare")
    with pytest.raises(AgentDefaultArchiveError):
        repository.archive_agent(default["id"])
    repository.archive_agent(spare["id"])
    assert [a["id"] for a in repository.list_agents()] == [default["id"]]
    assert len(repository.list_agents(include_archived=True)) == 2
    with pytest.raises(AgentArchivedError):
        repository.set_default(spare["id"])
    with pytest.raises(AgentArchivedError):
        _revise(repository, spare["id"], 1, {"v": 2})


# -----------------------------------------------------------------------------
def test_get_revision_returns_kind_and_name(repository: AgentRepository) -> None:
    agent = _create(repository, "Reviser", kind="revision")
    revision = repository.get_revision(agent["revision"]["id"])
    assert revision is not None
    assert revision["kind"] == "revision"
    assert revision["agent_name"] == "Reviser"
    assert repository.get_revision(9999) is None
