from __future__ import annotations

import json
from typing import Any, Literal, overload

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from repositories.database.session import resolve_engine, resolve_session_factory, unit_of_work
from repositories.database.upsert import (
    insert_application_configuration_if_missing,
    upsert_application_configuration,
)
from repositories.schemas.configuration import ApplicationConfiguration

###############################################################################
class ApplicationConfigurationSerializer:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        engine: Engine | None = None,
        session_factory: sessionmaker | None = None,
    ) -> None:
        self.engine = resolve_engine(engine)
        self.session_factory = resolve_session_factory(
            engine=self.engine,
            session_factory=session_factory,
            expire_on_commit=False,
        )

    # -------------------------------------------------------------------------
    def load(self) -> dict[str, Any] | None:
        with self.session_factory() as db_session:
            row = db_session.get(ApplicationConfiguration, 1)
            if row is None:
                return None
            return dict(row.payload)

    # -------------------------------------------------------------------------
    @overload
    def save(
        self,
        payload: dict[str, Any],
        *,
        return_metadata: Literal[False] = False,
    ) -> dict[str, Any]: ...

    # -------------------------------------------------------------------------
    @overload
    def save(
        self,
        payload: dict[str, Any],
        *,
        return_metadata: Literal[True],
    ) -> tuple[dict[str, Any], Any]: ...

    # -------------------------------------------------------------------------
    def save(
        self,
        payload: dict[str, Any],
        *,
        return_metadata: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any], Any]:
        json_safe_payload = json.loads(json.dumps(payload, default=str))
        with unit_of_work(session_factory=self.session_factory) as db_session:
            row = upsert_application_configuration(
                db_session,
                payload=json_safe_payload,
            )
            saved_payload = dict(row.payload)
            if return_metadata:
                return saved_payload, row.updated_at
            return saved_payload

    # -------------------------------------------------------------------------
    def save_if_missing(self, payload: dict[str, Any]) -> bool:
        """Seed the singleton only when initialization has no saved state."""
        json_safe_payload = json.loads(json.dumps(payload, default=str))
        with unit_of_work(session_factory=self.session_factory) as db_session:
            return insert_application_configuration_if_missing(
                db_session,
                payload=json_safe_payload,
            )
