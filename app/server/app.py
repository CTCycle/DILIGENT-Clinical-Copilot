from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.access_keys import router as access_keys_router
from api.data_inspection import router as data_inspection_router
from api.error_handling import register_error_handling
from api.health import router as health_router
from api.model_config import router as model_config_router
from api.ollama import router as ollama_router
from api.session import router as session_router
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
    FASTAPI_VERSION,
)
from common.paths import (
    CLIENT_ASSETS_PATH,
    CLIENT_DIST_PATH,
    CLIENT_INDEX_FILE_PATH,
)
from configurations.startup import (
    get_server_settings,
    initialize_settings,
)
from repositories.database.initializer import initialize_database
from services.startup_validation import run_startup_validations
from services.catalogs.runtime import initialize_reference_catalog_provider
from services.retrieval.embedding_runtime import close_embedding_runtime

###############################################################################
def _client_build_available() -> bool:
    return CLIENT_INDEX_FILE_PATH.is_file()

###############################################################################
def _resolve_client_file(full_path: str) -> Path | None:
    client_root = CLIENT_DIST_PATH.resolve()
    requested_path = (client_root / full_path).resolve()

    if not requested_path.is_relative_to(client_root):
        return None

    if requested_path.is_file():
        return requested_path

    return None

###############################################################################
def serve_client_root() -> FileResponse:
    return FileResponse(CLIENT_INDEX_FILE_PATH)

###############################################################################
def serve_client_path(full_path: str) -> FileResponse:
    client_file = _resolve_client_file(full_path)
    if client_file is not None:
        return FileResponse(client_file)
    return FileResponse(CLIENT_INDEX_FILE_PATH)

###############################################################################
def redirect_root_to_docs() -> RedirectResponse:
    return RedirectResponse(FASTAPI_DOCS_URL)

###############################################################################
@asynccontextmanager
async def app_lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = get_server_settings()

    initialize_database(
        drop_existing=False,
        seed_catalogs=True,
        force_reseed_catalogs=False,
    )
    initialize_reference_catalog_provider()
    run_startup_validations(settings)

    application.state.server_settings = settings
    try:
        yield
    finally:
        close_embedding_runtime()

###############################################################################
def create_app() -> FastAPI:
    initialize_settings()

    application = FastAPI(
        title=FASTAPI_TITLE,
        version=FASTAPI_VERSION,
        description=FASTAPI_DESCRIPTION,
        docs_url=FASTAPI_DOCS_URL,
        redoc_url=FASTAPI_REDOC_URL,
        openapi_url=FASTAPI_OPENAPI_URL,
        lifespan=app_lifespan,
    )
    register_error_handling(application)

    for router in (
        session_router,
        data_inspection_router,
        health_router,
        ollama_router,
        model_config_router,
        access_keys_router,
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
