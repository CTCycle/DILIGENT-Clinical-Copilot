from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from DILIGENT.server.packages.variables import env_variables
from DILIGENT.server.endpoints.session import router as session_router
from DILIGENT.server.endpoints.ollama import router as ollama_router
from DILIGENT.server.packages.configurations import server_settings


###############################################################################
root_path = os.getenv("FASTAPI_ROOT_PATH", "").strip()

app = FastAPI(
    title=server_settings.fastapi.title,
    version=server_settings.fastapi.version,
    description=server_settings.fastapi.description,
    root_path=root_path or "",
)

app.include_router(session_router)
app.include_router(ollama_router)

@app.get("/")
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")
