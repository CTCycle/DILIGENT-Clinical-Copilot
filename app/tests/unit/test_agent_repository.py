# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from repositories.agent_repository import (
    AgentDefaultRetireError,
    AgentDeleteBlockedError,
    AgentInactiveError,
    AgentNotEditableError,
    AgentNotFoundError,
    AgentRepository,
    AgentRevisionConflictError,
)
from repositories.schemas.base import Base
from repositories.schemas.clinical import ClinicalSession
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
    configuration: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return repository.create_agent(
        name=name,
        description="",
        configuration=configuration or {"v": 1},
        configuration_hash="a" * 64,
        schema_version=2,
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
        schema_version=2,
    )


###############################################################################
def test_create_agent_is_active_unlocked_with_revision_one(
    repository: AgentRepository,
) -> None:
    created = _create(repository)
    fetched = repository.get_agent(created["id"])
    assert fetched is not None
    assert (fetched["is_active"], fetched["is_locked"], fetched["is_archived"]) == (
        True,
        False,
        False,
    )
    assert fetched["current_revision"] == 1
    assert fetched["revision"]["configuration"] == {"v": 1}
    assert repository.count_agents() == 1


# -----------------------------------------------------------------------------
def test_update_creates_new_revision_and_keeps_previous_unchanged(
    repository: AgentRepository,
) -> None:
    agent = _create(repository)
    updated = _revise(repository, agent["id"], 1, {"v": 2})
    assert updated["current_revision"] == 2
    revisions = repository.list_revisions(agent["id"])
    assert [row["revision_number"] for row in revisions] == [2, 1]
    assert revisions[1]["configuration"] == {"v": 1}


# -----------------------------------------------------------------------------
def test_stale_expected_revision_is_rejected(repository: AgentRepository) -> None:
    agent = _create(repository)
    _revise(repository, agent["id"], 1, {"v": 2})
    with pytest.raises(AgentRevisionConflictError):
        _revise(repository, agent["id"], 1, {"v": 3})
    with pytest.raises(AgentNotFoundError):
        _revise(repository, "missing", 1, {"v": 3})


###############################################################################
def test_single_global_default(repository: AgentRepository) -> None:
    first = _create(repository, "First", is_default=True)
    second = _create(repository, "Second", is_default=True)
    assert repository.get_agent(first["id"])["is_default"] is False
    assert repository.get_default()["id"] == second["id"]
    repository.set_default(first["id"])
    assert repository.get_default()["id"] == first["id"]


# -----------------------------------------------------------------------------
def test_database_rejects_two_defaults(repository: AgentRepository) -> None:
    _create(repository, "First", is_default=True)
    second = _create(repository, "Second")
    with pytest.raises(IntegrityError):
        with repository.session_factory.begin() as db_session:
            db_session.get(AgentDefinition, second["id"]).is_default = True


###############################################################################
def test_configuration_is_editable_only_when_active_unlocked_and_not_archived(
    repository: AgentRepository,
) -> None:
    agent = _create(repository, "Spare")
    agent_id = agent["id"]
    for flags in (
        {"is_locked": True},
        {"is_locked": False, "is_active": False},
        {"is_active": True, "is_archived": True},
    ):
        repository.set_lifecycle(agent_id, **flags)
        with pytest.raises(AgentNotEditableError):
            _revise(repository, agent_id, 1, {"v": 2})
    # Every restrictive state can be left again: no impossible state.
    repository.set_lifecycle(agent_id, is_archived=False)
    assert _revise(repository, agent_id, 1, {"v": 2})["current_revision"] == 2


# -----------------------------------------------------------------------------
def test_lifecycle_changes_do_not_create_revisions(
    repository: AgentRepository,
) -> None:
    agent = _create(repository)
    changed = repository.set_lifecycle(agent["id"], is_locked=True, is_active=False)
    assert changed["current_revision"] == 1
    assert len(repository.list_revisions(agent["id"])) == 1


# -----------------------------------------------------------------------------
def test_default_cannot_be_deactivated_or_archived_and_inactive_cannot_default(
    repository: AgentRepository,
) -> None:
    default = _create(repository, "Default", is_default=True)
    spare = _create(repository, "Spare")
    for flags in ({"is_active": False}, {"is_archived": True}):
        with pytest.raises(AgentDefaultRetireError):
            repository.set_lifecycle(default["id"], **flags)
    repository.set_lifecycle(spare["id"], is_archived=True)
    assert [a["id"] for a in repository.list_agents()] == [default["id"]]
    assert len(repository.list_agents(include_archived=True)) == 2
    with pytest.raises(AgentInactiveError):
        repository.set_default(spare["id"])


###############################################################################
def test_delete_is_blocked_for_default_locked_and_referenced_agents(
    repository: AgentRepository,
) -> None:
    default = _create(repository, "Default", is_default=True)
    locked = _create(repository, "Locked")
    repository.set_lifecycle(locked["id"], is_locked=True)
    referenced = _create(repository, "Referenced")
    free = _create(repository, "Free")
    with pytest.raises(AgentDeleteBlockedError):
        repository.delete_agent(default["id"])
    with pytest.raises(AgentDeleteBlockedError):
        repository.delete_agent(locked["id"])

    with repository.session_factory.begin() as db_session:
        db_session.add(
            ClinicalSession(
                patient_name="P",
                agent_revision_id=referenced["revision"]["id"],
            )
        )
    assert repository.get_agent(referenced["id"])["reference_count"] == 1
    with pytest.raises(AgentDeleteBlockedError, match="archive"):
        repository.delete_agent(referenced["id"])

    repository.delete_agent(free["id"])
    assert repository.get_agent(free["id"]) is None
    assert repository.get_revision(free["revision"]["id"]) is None
    with pytest.raises(AgentNotFoundError):
        repository.delete_agent(free["id"])


# -----------------------------------------------------------------------------
def test_get_revision_returns_agent_name(repository: AgentRepository) -> None:
    agent = _create(repository, "Reviser")
    revision = repository.get_revision(agent["revision"]["id"])
    assert revision is not None
    assert revision["agent_name"] == "Reviser"
    assert repository.get_revision(9999) is None
