# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from repositories.context import RepositoryContext
from repositories.database.session import unit_of_work
from repositories.schemas.configuration import AgentDefinition, AgentRevision


###############################################################################
class AgentNotFoundError(LookupError):
    """The requested agent or agent revision does not exist."""


###############################################################################
class AgentArchivedError(RuntimeError):
    """The operation is not allowed on an archived agent."""


###############################################################################
class AgentRevisionConflictError(RuntimeError):
    """The save was based on an agent revision that is no longer current."""


###############################################################################
class AgentDefaultArchiveError(RuntimeError):
    """A default agent cannot be archived until another becomes the default."""


###############################################################################
class AgentRepository:

    # -------------------------------------------------------------------------
    def __init__(self, context: RepositoryContext) -> None:
        self.context = context
        self.engine = context.engine
        self.session_factory = context.session_factory

    # -------------------------------------------------------------------------
    @staticmethod
    def _revision_dict(row: AgentRevision) -> dict[str, Any]:
        return {
            "id": int(row.id),
            "agent_id": row.agent_id,
            "revision_number": int(row.revision_number),
            "schema_version": int(row.schema_version),
            "configuration": row.configuration_json,
            "configuration_hash": row.configuration_hash,
            "created_at": row.created_at,
        }

    # -------------------------------------------------------------------------
    def _agent_dict(
        self, definition: AgentDefinition, revision: AgentRevision
    ) -> dict[str, Any]:
        return {
            "id": definition.id,
            "kind": definition.kind,
            "name": definition.name,
            "description": definition.description,
            "current_revision": int(definition.current_revision),
            "is_default": bool(definition.is_default),
            "is_archived": bool(definition.is_archived),
            "created_at": definition.created_at,
            "updated_at": definition.updated_at,
            "revision": self._revision_dict(revision),
        }

    # -------------------------------------------------------------------------
    def _load_agents(
        self, db_session: Session, *conditions: Any
    ) -> list[dict[str, Any]]:
        rows = db_session.execute(
            select(AgentDefinition, AgentRevision)
            .join(
                AgentRevision,
                (AgentRevision.agent_id == AgentDefinition.id)
                & (AgentRevision.revision_number == AgentDefinition.current_revision),
            )
            .where(*conditions)
            .order_by(AgentDefinition.created_at.asc(), AgentDefinition.id.asc())
        ).all()
        return [self._agent_dict(definition, revision) for definition, revision in rows]

    # -------------------------------------------------------------------------
    @staticmethod
    def _clear_default(db_session: Session, kind: str, *, keep_id: str) -> None:
        db_session.execute(
            update(AgentDefinition)
            .where(
                AgentDefinition.default_kind == kind,
                AgentDefinition.id != keep_id,
            )
            .values(is_default=False, default_kind=None)
        )

    # -------------------------------------------------------------------------
    def list_agents(
        self, kind: str | None = None, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        conditions: list[Any] = []
        if kind is not None:
            conditions.append(AgentDefinition.kind == kind)
        if not include_archived:
            conditions.append(AgentDefinition.is_archived.is_(False))
        with self.session_factory() as db_session:
            return self._load_agents(db_session, *conditions)

    # -------------------------------------------------------------------------
    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            agents = self._load_agents(db_session, AgentDefinition.id == agent_id)
            return agents[0] if agents else None

    # -------------------------------------------------------------------------
    def get_default(self, kind: str) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            agents = self._load_agents(
                db_session,
                AgentDefinition.default_kind == kind,
                AgentDefinition.is_archived.is_(False),
            )
            return agents[0] if agents else None

    # -------------------------------------------------------------------------
    def count_agents(self) -> int:
        with self.session_factory() as db_session:
            return int(
                db_session.execute(
                    select(func.count()).select_from(AgentDefinition)
                ).scalar_one()
            )

    # -------------------------------------------------------------------------
    def create_agent(
        self,
        *,
        kind: str,
        name: str,
        description: str,
        configuration: dict[str, Any],
        configuration_hash: str,
        schema_version: int,
        is_default: bool = False,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        agent_id = uuid4().hex
        with unit_of_work(session_factory=self.session_factory, immediate=True) as db:
            if is_default:
                self._clear_default(db, kind, keep_id=agent_id)
            definition = AgentDefinition(
                id=agent_id,
                kind=kind,
                name=name,
                description=description,
                current_revision=1,
                is_default=is_default,
                default_kind=kind if is_default else None,
                is_archived=False,
                created_at=now,
                updated_at=now,
            )
            revision = AgentRevision(
                agent_id=agent_id,
                revision_number=1,
                schema_version=schema_version,
                configuration_json=configuration,
                configuration_hash=configuration_hash,
                created_at=now,
            )
            db.add(definition)
            db.flush()
            db.add(revision)
            db.flush()
            return self._agent_dict(definition, revision)

    # -------------------------------------------------------------------------
    def create_revision(
        self,
        *,
        agent_id: str,
        expected_revision: int,
        name: str,
        description: str,
        configuration: dict[str, Any],
        configuration_hash: str,
        schema_version: int,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        with unit_of_work(session_factory=self.session_factory, immediate=True) as db:
            definition = db.get(AgentDefinition, agent_id)
            if definition is None:
                raise AgentNotFoundError(agent_id)
            if definition.is_archived:
                raise AgentArchivedError(agent_id)
            if int(definition.current_revision) != int(expected_revision):
                raise AgentRevisionConflictError(agent_id)
            next_number = int(expected_revision) + 1
            # Optimistic guard: another writer that advanced the revision
            # between our read and this UPDATE matches no row.
            claimed = db.execute(
                update(AgentDefinition)
                .where(
                    AgentDefinition.id == agent_id,
                    AgentDefinition.current_revision == int(expected_revision),
                    AgentDefinition.is_archived.is_(False),
                )
                .values(
                    name=name,
                    description=description,
                    current_revision=next_number,
                    updated_at=now,
                )
            )
            if claimed.rowcount != 1:
                raise AgentRevisionConflictError(agent_id)
            revision = AgentRevision(
                agent_id=agent_id,
                revision_number=next_number,
                schema_version=schema_version,
                configuration_json=configuration,
                configuration_hash=configuration_hash,
                created_at=now,
            )
            db.add(revision)
            db.flush()
            db.refresh(definition)
            return self._agent_dict(definition, revision)

    # -------------------------------------------------------------------------
    def set_default(self, agent_id: str) -> dict[str, Any]:
        with unit_of_work(session_factory=self.session_factory, immediate=True) as db:
            definition = db.get(AgentDefinition, agent_id)
            if definition is None:
                raise AgentNotFoundError(agent_id)
            if definition.is_archived:
                raise AgentArchivedError(agent_id)
            self._clear_default(db, definition.kind, keep_id=agent_id)
            definition.is_default = True
            definition.default_kind = definition.kind
            db.flush()
            agents = self._load_agents(db, AgentDefinition.id == agent_id)
            return agents[0]

    # -------------------------------------------------------------------------
    def archive_agent(self, agent_id: str) -> dict[str, Any]:
        with unit_of_work(session_factory=self.session_factory, immediate=True) as db:
            definition = db.get(AgentDefinition, agent_id)
            if definition is None:
                raise AgentNotFoundError(agent_id)
            if definition.is_default:
                raise AgentDefaultArchiveError(agent_id)
            definition.is_archived = True
            db.flush()
            agents = self._load_agents(db, AgentDefinition.id == agent_id)
            return agents[0]

    # -------------------------------------------------------------------------
    def list_revisions(self, agent_id: str) -> list[dict[str, Any]]:
        with self.session_factory() as db_session:
            rows = (
                db_session.execute(
                    select(AgentRevision)
                    .where(AgentRevision.agent_id == agent_id)
                    .order_by(AgentRevision.revision_number.desc())
                )
                .scalars()
                .all()
            )
            return [self._revision_dict(row) for row in rows]

    # -------------------------------------------------------------------------
    def get_revision(self, revision_id: int) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            row = db_session.execute(
                select(AgentRevision, AgentDefinition.kind, AgentDefinition.name)
                .join(AgentDefinition, AgentDefinition.id == AgentRevision.agent_id)
                .where(AgentRevision.id == int(revision_id))
            ).first()
            if row is None:
                return None
            revision, kind, name = row
            return {**self._revision_dict(revision), "kind": kind, "agent_name": name}
