from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

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
initialize_settings()

app = FastAPI(
    title=FASTAPI_TITLE,
    version=FASTAPI_VERSION,
    description=FASTAPI_DESCRIPTION,
    docs_url=FASTAPI_DOCS_URL,
    redoc_url=FASTAPI_REDOC_URL,
    openapi_url=FASTAPI_OPENAPI_URL,
)
register_error_handling(app)

routers = [
    session_router,
    data_inspection_router,
    ollama_router,
    model_config_router,
    access_keys_router,
    research_router,
]

for router in routers:
    app.include_router(router, prefix="/api")


@app.on_event("startup")
async def initialize_database_on_startup() -> None:
    initialize_database()
    sync_runtime_model_config()

root_endpoint = RootEndpoint(
    app=app,
    tauri_mode=tauri_mode_enabled(),
)
root_endpoint.add_routes()

