from __future__ import annotations

from configurations.startup import get_server_settings
from domain.settings.configuration import ServerSettings
from services.catalogs.runtime import get_reference_catalog_snapshot
from services.llm.model_config import ModelConfigService


###############################################################################
def run_startup_validations(settings: ServerSettings | None = None) -> None:
    resolved_settings = settings or get_server_settings()

    catalog_snapshot = get_reference_catalog_snapshot()
    if not catalog_snapshot.entries_by_scope:
        raise RuntimeError("Reference catalogs must be available at startup.")

    try:
        ModelConfigService().load_current_snapshot()
    except Exception as exc:
        raise RuntimeError(
            "Persisted model configuration could not be resolved."
        ) from exc

    if resolved_settings.database.backend == "sqlite":
        return

    if not (resolved_settings.database.engine or "").strip():
        raise RuntimeError("External database engine must be configured.")
    if not (resolved_settings.database.host or "").strip():
        raise RuntimeError("External database host must be configured.")
    if not (resolved_settings.database.database_name or "").strip():
        raise RuntimeError("External database name must be configured.")
    if not (resolved_settings.database.username or "").strip():
        raise RuntimeError("External database username must be configured.")
