from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from DILIGENT.common.utils.variables import env_variables  # noqa: F401
from DILIGENT.server.routes.session import router as session_router
from DILIGENT.server.routes.ollama import router as ollama_router
from DILIGENT.server.routes.modelconfig import router as modelconfig_router
from DILIGENT.server.routes.accesskeys import router as accesskeys_router
from DILIGENT.server.routes.research import router as research_router
from DILIGENT.server.configurations import server_settings


def tauri_mode_enabled() -> bool:
    value = os.getenv("DILIGENT_TAURI_MODE", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_client_dist_path() -> str:
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_path, "client", "dist")


def packaged_client_available() -> bool:
    return tauri_mode_enabled() and os.path.isdir(get_client_dist_path())


###############################################################################
app = FastAPI(
    title=server_settings.fastapi.title,
    version=server_settings.fastapi.version,
    description=server_settings.fastapi.description,
)

routers = [
    session_router,
    ollama_router,
    modelconfig_router,
    accesskeys_router,
    research_router,
]

for router in routers:
    app.include_router(router)
    app.include_router(router, prefix="/api", include_in_schema=False)


if packaged_client_available():
    client_dist_path = get_client_dist_path()
    assets_path = os.path.join(client_dist_path, "assets")

    if os.path.isdir(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="spa-assets")

    @app.get("/", include_in_schema=False)
    def serve_spa_root() -> FileResponse:
        return FileResponse(os.path.join(client_dist_path, "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa_entrypoint(full_path: str) -> FileResponse:
        requested_path = os.path.join(client_dist_path, full_path)
        if os.path.isfile(requested_path):
            return FileResponse(requested_path)
        return FileResponse(os.path.join(client_dist_path, "index.html"))

else:

    @app.get("/")
    def redirect_to_docs() -> RedirectResponse:
        return RedirectResponse(url="/docs")
