from __future__ import annotations

import json
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from repositories.database.session import resolve_engine, resolve_session_factory, unit_of_work
from repositories.database.upsert import upsert_application_configuration
from repositories.schemas.models import ApplicationConfiguration


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
    def save(
        self,
        payload: dict[str, Any],
        *,
        schema_version: int = 1,
    ) -> dict[str, Any]:
        json_safe_payload = json.loads(json.dumps(payload, default=str))
        with unit_of_work(session_factory=self.session_factory) as db_session:
            row = upsert_application_configuration(
                db_session,
                payload=json_safe_payload,
                schema_version=schema_version,
            )
            return dict(row.payload)
