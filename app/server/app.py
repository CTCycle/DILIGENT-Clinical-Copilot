# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.access_keys import router as access_keys_router
from api.data_inspection import router as data_inspection_router
from api.desktop import router as desktop_router
from api.database import router as database_router
from api.error_handling import register_error_handling
from api.health import router as health_router
from api.jobs import router as jobs_router
from api.model_config import router as model_config_router
from api.ollama import router as ollama_router
from api.session import router as session_router
from api.settings import router as settings_router
from api.system_status import router as system_status_router
from common.constants import (
    FASTAPI_API_PREFIX,
    FASTAPI_ASSETS_ENDPOINT,
    FASTAPI_DESCRIPTION,
    FASTAPI_DOCS_URL,
    FASTAPI_OPENAPI_URL,
    FASTAPI_REDOC_URL,
    FASTAPI_ROOT_ENDPOINT,
    FASTAPI_SPA_FALLBACK_ENDPOINT,
    FASTAPI_TITLE,
)
from common.paths import (
    CLIENT_ASSETS_PATH,
    CLIENT_DIST_PATH,
    CLIENT_INDEX_FILE_PATH,
)
from common.security.desktop import DesktopSecurityMiddleware
from middleware.database_switch import DatabaseSwitchMiddleware
from common.version import resolve_application_version
from configurations.network import get_cors_allow_origins
from configurations.startup import (
    get_server_settings,
    initialize_settings,
    reload_persisted_settings,
)
from common.utils.logger import logger
from repositories.context import RepositoryContext
from repositories.database.initializer import ensure_database_ready
from services.catalogs.runtime import initialize_reference_catalog_provider
from services.inspection.revision_recovery import (
    fail_revision_runs_interrupted_by_shutdown,
    reconcile_interrupted_revision_jobs,
)
from services.retrieval.embedding_runtime import close_embedding_runtime
from services.runtime.desktop import get_desktop_runtime_service
from services.runtime.jobs import get_job_manager
from services.startup_validation import run_startup_validations

_CLIENT_ASSET_SUFFIXES = {
    ".css",
    ".eot",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".map",
    ".mjs",
    ".png",
    ".svg",
    ".ttf",
    ".txt",
    ".wasm",
    ".webmanifest",
    ".webp",
    ".woff",
    ".woff2",
}

###############################################################################
def _client_build_available() -> bool:
    return CLIENT_INDEX_FILE_PATH.is_file()

###############################################################################
def _resolve_client_file(full_path: str) -> Path | None:
    client_root = CLIENT_DIST_PATH.resolve()
    normalized_path = full_path.replace("\\", "/")
    requested_path = (client_root / normalized_path).resolve()
    if not requested_path.is_relative_to(client_root):
        raise HTTPException(status_code=404, detail="Not found")
    if requested_path.is_file():
        return requested_path
    return None

###############################################################################
def serve_client_root() -> FileResponse:
    return FileResponse(CLIENT_INDEX_FILE_PATH)

###############################################################################
def serve_client_path(full_path: str) -> FileResponse:
    normalized_path = full_path.replace("\\", "/")
    if normalized_path == "api" or normalized_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    client_file = _resolve_client_file(full_path)
    if client_file is not None:
        return FileResponse(client_file)
    if Path(normalized_path).suffix.casefold() in _CLIENT_ASSET_SUFFIXES:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(CLIENT_INDEX_FILE_PATH)

###############################################################################
def redirect_root_to_docs() -> RedirectResponse:
    return RedirectResponse(FASTAPI_DOCS_URL)

###############################################################################
async def _release_application_resources() -> None:
    jobs_finished = await asyncio.to_thread(get_job_manager().shutdown, 5.0)
    if not jobs_finished:
        logger.warning("Background jobs did not stop before shutdown; failing interrupted runs")
        try:
            await asyncio.to_thread(fail_revision_runs_interrupted_by_shutdown)
        except Exception:
            logger.exception("Could not mark interrupted revision runs during shutdown")
    close_embedding_runtime()
    try:
        RepositoryContext.create().engine.dispose()
    except Exception:
        logger.exception("Could not dispose the database engine during shutdown")

###############################################################################
@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    bootstrap_settings = get_server_settings()
    ensure_database_ready(bootstrap_settings.database)
    settings = reload_persisted_settings()
    job_manager = get_job_manager()
    job_manager.begin_startup()
    reconcile_interrupted_revision_jobs(job_manager)
    initialize_reference_catalog_provider()
    run_startup_validations(settings)
    application.state.server_settings = settings
    try:
        yield
    finally:
        await _release_application_resources()

###############################################################################
def create_app() -> FastAPI:
    initialize_settings()
    application = FastAPI(
        title=FASTAPI_TITLE,
        version=resolve_application_version(),
        description=FASTAPI_DESCRIPTION,
        docs_url=FASTAPI_DOCS_URL,
        redoc_url=FASTAPI_REDOC_URL,
        openapi_url=FASTAPI_OPENAPI_URL,
        lifespan=app_lifespan,
    )
    desktop_runtime = get_desktop_runtime_service()
    application.state.desktop_runtime = desktop_runtime
    register_error_handling(application)
    application.add_middleware(
        DesktopSecurityMiddleware,
        security=desktop_runtime.security,
    )
    application.add_middleware(DatabaseSwitchMiddleware)
    cors_allow_origins = (
        ("http://tauri.localhost",)
        if desktop_runtime.security.enabled
        else get_cors_allow_origins()
    )
    if cors_allow_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_allow_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "X-DILIGENT-Desktop-Session"],
        )
    for router in (
        session_router,
        data_inspection_router,
        health_router,
        jobs_router,
        ollama_router,
        model_config_router,
        settings_router,
        database_router,
        access_keys_router,
        system_status_router,
        desktop_router,
    ):
        application.include_router(router, prefix=FASTAPI_API_PREFIX)

    if _client_build_available():
        if CLIENT_ASSETS_PATH.is_dir():
            application.mount(
                FASTAPI_ASSETS_ENDPOINT,
                StaticFiles(directory=CLIENT_ASSETS_PATH),
                name="assets",
            )
        application.add_api_route(
            FASTAPI_ROOT_ENDPOINT,
            serve_client_root,
            methods=["GET"],
            include_in_schema=False,
        )
        application.add_api_route(
            FASTAPI_SPA_FALLBACK_ENDPOINT,
            serve_client_path,
            methods=["GET"],
            include_in_schema=False,
        )
    else:
        application.add_api_route(
            FASTAPI_ROOT_ENDPOINT,
            redirect_root_to_docs,
            methods=["GET"],
            include_in_schema=False,
        )
    return application


app = create_app()
