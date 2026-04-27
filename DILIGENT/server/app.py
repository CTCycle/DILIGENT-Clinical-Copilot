from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from DILIGENT.server.configurations.startup import initialize_settings
from DILIGENT.server.configurations.startup import tauri_mode_enabled
from DILIGENT.server.common.constants import (
    FASTAPI_DESCRIPTION,
    FASTAPI_DOCS_URL,
    FASTAPI_OPENAPI_URL,
    FASTAPI_REDOC_URL,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from DILIGENT.server.api.access_keys import router as access_keys_router
from DILIGENT.server.api.data_inspection import router as data_inspection_router
from DILIGENT.server.api.model_config import router as model_config_router
from DILIGENT.server.configurations.model_runtime import sync_runtime_model_config
from DILIGENT.server.api.session import router as session_router
from DILIGENT.server.api.ollama import router as ollama_router
from DILIGENT.server.api.research import router as research_router
from DILIGENT.server.api.root import RootEndpoint
from DILIGENT.server.api.error_handling import register_error_handling
from DILIGENT.server.repositories.database.initializer import initialize_database


###############################################################################
@asynccontextmanager
async def app_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    initialize_database()
    sync_runtime_model_config()
    yield


###############################################################################
def create_app() -> FastAPI:
    initialize_settings()

    fastapi_app = FastAPI(
        title=FASTAPI_TITLE,
        version=FASTAPI_VERSION,
        description=FASTAPI_DESCRIPTION,
        docs_url=FASTAPI_DOCS_URL,
        redoc_url=FASTAPI_REDOC_URL,
        openapi_url=FASTAPI_OPENAPI_URL,
        lifespan=app_lifespan,
    )
    register_error_handling(fastapi_app)

    for router in (
        session_router,
        data_inspection_router,
        ollama_router,
        model_config_router,
        access_keys_router,
        research_router,
    ):
        fastapi_app.include_router(router, prefix="/api")

    root_endpoint = RootEndpoint(
        app=fastapi_app,
        tauri_mode=tauri_mode_enabled(),
    )
    root_endpoint.add_routes()
    return fastapi_app


app = create_app()

