from __future__ import annotations

from common.paths import CLIENT_INDEX_FILE_PATH
from configurations.startup import get_server_settings, tauri_mode_enabled
from domain.settings.configuration import ServerSettings
from services.catalogs.runtime import get_reference_catalog_snapshot
from services.llm.model_config import ModelConfigService

###############################################################################
def run_startup_validations(settings: ServerSettings | None = None) -> None:
    resolved_settings = settings or get_server_settings()
    if tauri_mode_enabled() and not CLIENT_INDEX_FILE_PATH.is_file():
        raise RuntimeError(
            f"Tauri mode requires a packaged client build at {CLIENT_INDEX_FILE_PATH}."
        )

    catalog_snapshot = get_reference_catalog_snapshot()
    if not catalog_snapshot.entries_by_scope:
        raise RuntimeError("Reference catalogs must be available at startup.")

    model_snapshot = ModelConfigService().ensure_defaults()
    if not (model_snapshot.clinical_model or "").strip():
        raise RuntimeError("Clinical model defaults could not be resolved.")
    if not (model_snapshot.text_extraction_model or "").strip():
        raise RuntimeError("Text extraction model defaults could not be resolved.")

    if resolved_settings.database.embedded_database:
        return

    if not (resolved_settings.database.engine or "").strip():
        raise RuntimeError("External database engine must be configured.")
    if not (resolved_settings.database.host or "").strip():
        raise RuntimeError("External database host must be configured.")
    if not (resolved_settings.database.database_name or "").strip():
        raise RuntimeError("External database name must be configured.")
    if not (resolved_settings.database.username or "").strip():
        raise RuntimeError("External database username must be configured.")
