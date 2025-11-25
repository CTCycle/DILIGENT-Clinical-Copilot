from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from DILIGENT.src.packages.variables import env_variables
from DILIGENT.src.server.endpoints.session import router as session_router
from DILIGENT.src.server.endpoints.ollama import router as ollama_router
from DILIGENT.src.packages.configurations import server_settings


###############################################################################
app = FastAPI(
    title=server_settings.fastapi.title,
    version=server_settings.fastapi.version,
    description=server_settings.fastapi.description,
)

app.include_router(session_router)
app.include_router(ollama_router)

@app.get("/")
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")