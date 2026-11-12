# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from repositories.context import RepositoryContext
from repositories.database.session import unit_of_work
from repositories.schemas.clinical import (
    ClinicalSession,
    ClinicalSessionRevisionRun,
    ClinicalSessionTimeline,
)
from repositories.schemas.configuration import AgentDefinition, AgentRevision


###############################################################################
class AgentNotFoundError(LookupError):
    """The requested agent or agent revision does not exist."""


###############################################################################
class AgentNotEditableError(RuntimeError):
    """The agent is inactive, locked or archived, so its configuration is frozen."""


###############################################################################
class AgentInactiveError(RuntimeError):
    """The operation requires an active, non-archived agent."""


###############################################################################
class AgentRevisionConflictError(RuntimeError):
    """The save was based on an agent revision that is no longer current."""


###############################################################################
class AgentDefaultRetireError(RuntimeError):
    """A default agent cannot be deactivated or archived while it is the default."""


###############################################################################
class AgentDeleteBlockedError(RuntimeError):
    """The agent is the default, locked or referenced by persisted runs."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


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
        self,
        definition: AgentDefinition,
        revision: AgentRevision,
        reference_count: int = 0,
    ) -> dict[str, Any]:
        return {
            "id": definition.id,
            "name": definition.name,
            "description": definition.description,
            "current_revision": int(definition.current_revision),
            "is_default": bool(definition.is_default),
            "is_active": bool(definition.is_active),
            "is_locked": bool(definition.is_locked),
            "is_archived": bool(definition.is_archived),
            "reference_count": reference_count,
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
        return [
            self._agent_dict(
                definition,
                revision,
                self._reference_count(db_session, definition.id),
            )
            for definition, revision in rows
        ]

    # -------------------------------------------------------------------------
    @staticmethod
    def _reference_count(db_session: Session, agent_id: str) -> int:
        """Count persisted runs that point at any revision of the agent."""

        revision_ids = select(AgentRevision.id).where(
            AgentRevision.agent_id == agent_id
        )
        total = 0
        for model in (ClinicalSession, ClinicalSessionRevisionRun, ClinicalSessionTimeline):
            total += int(
                db_session.execute(
                    select(func.count())
                    .select_from(model)
                    .where(model.agent_revision_id.in_(revision_ids))
                ).scalar_one()
            )
        return total

    # -------------------------------------------------------------------------
    @staticmethod
    def _clear_default(db_session: Session, *, keep_id: str) -> None:
        db_session.execute(
            update(AgentDefinition)
            .where(AgentDefinition.id != keep_id, AgentDefinition.is_default.is_(True))
            .values(is_default=False)
        )

    # -------------------------------------------------------------------------
    def list_agents(self, include_archived: bool = False) -> list[dict[str, Any]]:
        conditions: list[Any] = []
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
    def get_default(self) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            agents = self._load_agents(
                db_session,
                AgentDefinition.is_default.is_(True),
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
                self._clear_default(db, keep_id=agent_id)
            definition = AgentDefinition(
                id=agent_id,
                name=name,
                description=description,
                current_revision=1,
                is_default=is_default,
                is_active=True,
                is_locked=False,
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
            if (
                not definition.is_active
                or definition.is_locked
                or definition.is_archived
            ):
                raise AgentNotEditableError(agent_id)
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
                    AgentDefinition.is_active.is_(True),
                    AgentDefinition.is_locked.is_(False),
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
            if not definition.is_active or definition.is_archived:
                raise AgentInactiveError(agent_id)
            self._clear_default(db, keep_id=agent_id)
            definition.is_default = True
            db.flush()
            return self._load_agents(db, AgentDefinition.id == agent_id)[0]

    # -------------------------------------------------------------------------
    def set_lifecycle(
        self,
        agent_id: str,
        *,
        is_active: bool | None = None,
        is_locked: bool | None = None,
        is_archived: bool | None = None,
    ) -> dict[str, Any]:
        """Change lifecycle flags.

        Never blocked by lock, inactivity or archive, so a restrictive state
        can always be left. Only the default agent is protected from becoming
        inactive or archived.
        """

        with unit_of_work(session_factory=self.session_factory, immediate=True) as db:
            definition = db.get(AgentDefinition, agent_id)
            if definition is None:
                raise AgentNotFoundError(agent_id)
            next_active = definition.is_active if is_active is None else is_active
            next_archived = (
                definition.is_archived if is_archived is None else is_archived
            )
            if definition.is_default and (not next_active or next_archived):
                raise AgentDefaultRetireError(agent_id)
            definition.is_active = next_active
            definition.is_archived = next_archived
            if is_locked is not None:
                definition.is_locked = is_locked
            definition.updated_at = datetime.now(UTC)
            db.flush()
            return self._load_agents(db, AgentDefinition.id == agent_id)[0]

    # -------------------------------------------------------------------------
    def delete_agent(self, agent_id: str) -> None:
        with unit_of_work(session_factory=self.session_factory, immediate=True) as db:
            definition = db.get(AgentDefinition, agent_id)
            if definition is None:
                raise AgentNotFoundError(agent_id)
            if definition.is_default:
                raise AgentDeleteBlockedError("The default agent cannot be deleted.")
            if definition.is_locked:
                raise AgentDeleteBlockedError("Unlock the agent before deleting it.")
            count = self._reference_count(db, agent_id)
            if count:
                raise AgentDeleteBlockedError(
                    f"The agent is referenced by {count} saved run(s); archive it instead."
                )
            db.execute(delete(AgentRevision).where(AgentRevision.agent_id == agent_id))
            db.delete(definition)

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
                select(AgentRevision, AgentDefinition.name)
                .join(AgentDefinition, AgentDefinition.id == AgentRevision.agent_id)
                .where(AgentRevision.id == int(revision_id))
            ).first()
            if row is None:
                return None
            revision, name = row
            return {**self._revision_dict(revision), "agent_name": name}
