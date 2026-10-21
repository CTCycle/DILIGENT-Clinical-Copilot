# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import json
from copy import deepcopy
from threading import RLock
from typing import Any, Callable, Literal, overload

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from repositories.database.session import (
    resolve_engine,
    resolve_session_factory,
    unit_of_work,
)
from repositories.database.upsert import (
    insert_application_configuration_if_missing,
    upsert_application_configuration,
)
from repositories.schemas.configuration import ApplicationConfiguration

###############################################################################
class ApplicationConfigurationSerializer:
    _write_lock = RLock()

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
        payload, _ = self.load_with_metadata()
        return payload

    # -------------------------------------------------------------------------
    def load_with_metadata(self) -> tuple[dict[str, Any] | None, Any]:
        with self.session_factory() as db_session:
            row = db_session.get(ApplicationConfiguration, 1)
            if row is None:
                return None, None
            return dict(row.payload), row.updated_at

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
        with self._write_lock:
            with unit_of_work(session_factory=self.session_factory) as db_session:
                row = upsert_application_configuration(
                    db_session,
                    payload=json_safe_payload,
                )
                saved_payload = dict(row.payload)
                updated_at = row.updated_at
        if return_metadata:
            return saved_payload, updated_at
        return saved_payload

    # -------------------------------------------------------------------------
    @overload
    def update_payload(
        self,
        update: Callable[[dict[str, Any]], None],
        *,
        return_metadata: Literal[False] = False,
    ) -> dict[str, Any]: ...

    # -------------------------------------------------------------------------
    @overload
    def update_payload(
        self,
        update: Callable[[dict[str, Any]], None],
        *,
        return_metadata: Literal[True],
    ) -> tuple[dict[str, Any], Any]: ...

    # -------------------------------------------------------------------------
    def update_payload(
        self,
        update: Callable[[dict[str, Any]], None],
        *,
        return_metadata: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any], Any]:
        """Atomically merge a settings update into the latest singleton payload."""
        with self._write_lock:
            with unit_of_work(session_factory=self.session_factory) as db_session:
                row = db_session.get(
                    ApplicationConfiguration,
                    1,
                    with_for_update=True,
                )
                payload = deepcopy(row.payload) if row is not None else {}
                if not isinstance(payload, dict):
                    raise ValueError("Application configuration payload must be an object")
                update(payload)
                json_safe_payload = json.loads(json.dumps(payload, default=str))
                saved_row = upsert_application_configuration(
                    db_session,
                    payload=json_safe_payload,
                )
                # The singleton row may already be present in this session's
                # identity map from the read above. Refresh after the upsert so
                # callers receive the committed payload and revision timestamp,
                # rather than the pre-update ORM object.
                db_session.refresh(saved_row)
                saved_payload = dict(saved_row.payload)
                updated_at = saved_row.updated_at
        if return_metadata:
            return saved_payload, updated_at
        return saved_payload

    # -------------------------------------------------------------------------
    def save_if_missing(self, payload: dict[str, Any]) -> bool:
        """Seed the singleton only when initialization has no saved state."""
        json_safe_payload = json.loads(json.dumps(payload, default=str))
        with self._write_lock:
            with unit_of_work(session_factory=self.session_factory) as db_session:
                return insert_application_configuration_if_missing(
                    db_session,
                    payload=json_safe_payload,
                )
