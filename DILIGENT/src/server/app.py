from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from DILIGENT.src.packages.variables import env_variables
from DILIGENT.src.server.endpoints.session import router as session_router
from DILIGENT.src.server.endpoints.ollama import router as ollama_router
from DILIGENT.src.packages.configurations import configurations


###############################################################################
fastapi_settings = configurations.server.fastapi
ui_settings = configurations.client.ui
app = FastAPI(
    title=fastapi_settings.title,
    version=fastapi_settings.version,
    description=fastapi_settings.description,
)

app.include_router(session_router)
app.include_router(ollama_router)

@app.get("/")
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")