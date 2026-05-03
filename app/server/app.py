from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from configurations.startup import initialize_settings
from configurations.startup import tauri_mode_enabled
from common.constants import (
    FASTAPI_DESCRIPTION,
    FASTAPI_DOCS_URL,
    FASTAPI_OPENAPI_URL,
    FASTAPI_REDOC_URL,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
)
from api.access_keys import router as access_keys_router
from api.data_inspection import router as data_inspection_router
from api.health import router as health_router
from api.model_config import router as model_config_router
from api.session import router as session_router
from api.ollama import router as ollama_router
from api.research import router as research_router
from api.root import RootEndpoint
from api.error_handling import register_error_handling
from repositories.database.initializer import initialize_database
from services.llm.model_config import ModelConfigService


###############################################################################
@asynccontextmanager
async def app_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    initialize_database()
    ModelConfigService().ensure_defaults()
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
        health_router,
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

