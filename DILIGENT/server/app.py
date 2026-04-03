from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI

from DILIGENT.server.configurations.bootstrap import server_settings
from DILIGENT.server.configurations.runtime import cloud_mode_enabled, tauri_mode_enabled
from DILIGENT.server.api.access_keys import router as access_keys_router
from DILIGENT.server.api.data_inspection import router as data_inspection_router
from DILIGENT.server.api.model_config import router as model_config_router
from DILIGENT.server.api.session import router as session_router
from DILIGENT.server.api.ollama import router as ollama_router
from DILIGENT.server.api.research import router as research_router
from DILIGENT.server.api.root import RootEndpoint
from DILIGENT.server.api.error_handling import register_error_handling


###############################################################################
cloud_mode = cloud_mode_enabled()
app = FastAPI(
    title=server_settings.fastapi.title,
    version=server_settings.fastapi.version,
    description=server_settings.fastapi.description,
    docs_url=None if cloud_mode else "/docs",
    redoc_url=None if cloud_mode else "/redoc",
    openapi_url=None if cloud_mode else "/openapi.json",
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
    if not cloud_mode:
        app.include_router(router)
    app.include_router(router, prefix="/api", include_in_schema=False)

root_endpoint = RootEndpoint(
    app=app,
    cloud_mode=cloud_mode,
    tauri_mode=tauri_mode_enabled(),
)
root_endpoint.add_routes()
