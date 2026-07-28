from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from repositories.database.session import resolve_engine, resolve_session_factory


@dataclass(frozen=True, slots=True)
class RepositoryContext:
    """Immutable database handles shared by one repository composition root."""

    engine: Engine
    session_factory: sessionmaker[Session]

    @classmethod
    def create(
        cls,
        *,
        engine: Engine | None = None,
        session_factory: sessionmaker[Session] | None = None,
    ) -> RepositoryContext:
        resolved_engine = resolve_engine(engine)
        return cls(
            engine=resolved_engine,
            session_factory=resolve_session_factory(
                engine=resolved_engine,
                session_factory=session_factory,
            ),
        )
