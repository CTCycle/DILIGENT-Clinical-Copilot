# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from functools import lru_cache
from threading import Condition, RLock
from time import monotonic

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from configurations.database_config import (
    DatabaseConfiguration,
    database_settings,
    load_database_configuration,
    public_database_configuration,
    save_database_configuration,
)
from configurations.startup import (
    reload_persisted_settings,
    reset_app_settings_cache,
)
from domain.settings.database import (
    DatabaseConfigurationResponse,
    DatabaseConnectionTestResponse,
    DatabaseSwitchResponse,
)
from repositories.database.initializer import ensure_database_ready
from repositories.database.postgres import PostgresRepository
from repositories.database.session import get_default_repository
from repositories.database.sqlite import SQLiteRepository
from services.runtime.jobs import get_job_manager


class DatabaseSwitchBusyError(RuntimeError):
    pass


class DatabaseActivationError(RuntimeError):
    pass


class DatabaseRequestGate:
    """Reject new API work and drain active requests before pool replacement."""

    def __init__(self) -> None:
        self._condition = Condition(RLock())
        self._switching = False
        self._active_requests = 0

    def enter_request(self) -> bool:
        with self._condition:
            if self._switching:
                return False
            self._active_requests += 1
            return True

    def leave_request(self) -> None:
        with self._condition:
            self._active_requests = max(0, self._active_requests - 1)
            if self._active_requests == 0:
                self._condition.notify_all()

    def begin_switch(self) -> None:
        with self._condition:
            self._switching = True

    def wait_for_idle(self, timeout: float) -> bool:
        deadline = monotonic() + timeout
        with self._condition:
            while self._active_requests:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def end_switch(self) -> None:
        with self._condition:
            self._switching = False
            self._condition.notify_all()


class DatabaseRuntimeManager:
    def __init__(self) -> None:
        self.request_gate = DatabaseRequestGate()
        self._switch_lock = RLock()

    def get_state(self) -> dict[str, object]:
        configuration = load_database_configuration()
        return {
            "configuration": DatabaseConfigurationResponse.model_validate(
                public_database_configuration(configuration)
            ),
            "active_backend": configuration.backend,
        }

    def test_configuration(
        self, configuration: DatabaseConfiguration
    ) -> DatabaseConnectionTestResponse:
        repository = None
        try:
            repository = _create_repository(configuration)
            with repository.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            schema_ready = inspect(repository.engine).has_table("alembic_version")
            return DatabaseConnectionTestResponse(
                reachable=True,
                schema_ready=schema_ready,
                message=(
                    "Connection succeeded and the application schema is ready."
                    if schema_ready
                    else "Connection succeeded. The application schema will be initialized when this configuration is applied."
                ),
            )
        except (SQLAlchemyError, OSError, ValueError) as exc:
            del exc
            return DatabaseConnectionTestResponse(
                reachable=False,
                schema_ready=False,
                message=(
                    "Connection failed. Check the database path, server availability, "
                    "host, port, database name, credentials, and SSL settings."
                ),
            )
        finally:
            if repository is not None:
                repository.engine.dispose()

    def switch_configuration(
        self, configuration: DatabaseConfiguration
    ) -> DatabaseSwitchResponse:
        candidate_settings = database_settings(configuration)
        with self._switch_lock:
            previous_configuration = load_database_configuration()
            previous_repository = get_default_repository()
            job_manager = get_job_manager()
            if not job_manager.begin_database_switch():
                raise DatabaseSwitchBusyError(
                    "A background job is pending or running. Wait for it to finish and retry."
                )
            self.request_gate.begin_switch()
            candidate_repository = None
            try:
                if not self.request_gate.wait_for_idle(timeout=60.0):
                    raise DatabaseActivationError(
                        "Active requests did not finish in time. No database change was applied."
                    )

                # Initialization runs against the candidate only. It creates a
                # missing PostgreSQL database when permitted, applies migrations,
                # and seeds defaults without copying current application data.
                ensure_database_ready(candidate_settings)
                probe = self.test_configuration(configuration)
                if not probe.reachable or not probe.schema_ready:
                    raise DatabaseActivationError(
                        probe.message
                        if not probe.reachable
                        else "The application schema could not be verified after initialization."
                    )

                save_database_configuration(configuration)
                get_default_repository.cache_clear()
                reset_app_settings_cache()
                candidate_repository = get_default_repository()
                reload_persisted_settings()
                _rebind_database_services()
                _invalidate_database_snapshots()
                previous_repository.engine.dispose()
                return DatabaseSwitchResponse(
                    active_backend=configuration.backend,
                    message=(
                        "Database switched. Application settings were loaded from the selected database."
                    ),
                )
            except Exception as exc:
                try:
                    if candidate_repository is None:
                        try:
                            candidate_repository = get_default_repository()
                        except Exception:
                            candidate_repository = None
                    save_database_configuration(previous_configuration)
                    get_default_repository.cache_clear()
                    reset_app_settings_cache()
                    rollback_repository = get_default_repository()
                    reload_persisted_settings()
                    _rebind_database_services()
                    _invalidate_database_snapshots()
                    if rollback_repository.engine is not previous_repository.engine:
                        previous_repository.engine.dispose()
                    if (
                        candidate_repository is not None
                        and candidate_repository.engine is not rollback_repository.engine
                        and candidate_repository.engine is not previous_repository.engine
                    ):
                        candidate_repository.engine.dispose()
                except Exception as rollback_error:
                    raise DatabaseActivationError(
                        "Database activation failed and automatic rollback could not fully restore runtime services. Restart the application to recover the saved configuration."
                    ) from rollback_error
                if isinstance(exc, DatabaseActivationError):
                    raise
                raise DatabaseActivationError(
                    "Database activation failed. The previous database remains active and the saved configuration was restored."
                ) from exc
            finally:
                self.request_gate.end_switch()
                job_manager.end_database_switch()


def _create_repository(configuration: DatabaseConfiguration):
    settings = database_settings(configuration)
    repository_type = SQLiteRepository if settings.backend == "sqlite" else PostgresRepository
    return repository_type(settings)


def _rebind_database_services() -> None:
    from api import (
        access_keys,
        data_inspection,
        model_config,
        session,
        settings,
        system_status,
    )

    for module in (
        access_keys,
        data_inspection,
        model_config,
        session,
        settings,
        system_status,
    ):
        module.rebind_database_services()


def _invalidate_database_snapshots() -> None:
    from services.catalogs.runtime import reload_reference_catalog_snapshot
    from services.text.vocabulary import invalidate_text_normalization_snapshot

    reload_reference_catalog_snapshot()
    invalidate_text_normalization_snapshot()


@lru_cache(maxsize=1)
def get_database_runtime_manager() -> DatabaseRuntimeManager:
    return DatabaseRuntimeManager()
