from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from nicegui import ui

from DILIGENT.src.app.server.endpoints.session import router as session_router
from DILIGENT.src.app.server.endpoints.ollama import router as ollama_router
from DILIGENT.src.app.client.interface import create_interface
from DILIGENT.src.packages.configurations import configurations
from DILIGENT.src.packages.logger import logger
from DILIGENT.src.packages.utils.repository.database import database
from DILIGENT.src.packages.variables import env_variables

###############################################################################
# initialize the database if it has not been created
if database.requires_sqlite_initialization():
    logger.info("Database not found, creating instance and making all tables")
    database.initialize_database()
    logger.info("DILIGENT database has been initialized successfully.")

app = FastAPI(
    title=configurations.backend.title,
    version=configurations.backend.version,
    description=configurations.backend.description,
)

app.include_router(session_router)
app.include_router(ollama_router)

create_interface()
ui.run_with(
    app,
    mount_path=configurations.ui_runtime.mount_path,
    title=configurations.ui_runtime.title,
    show_welcome_message=configurations.ui_runtime.show_welcome_message,
    reconnect_timeout=configurations.ui_runtime.reconnect_timeout,
)

@app.get("/")
def redirect_to_ui() -> RedirectResponse:
    return RedirectResponse(url=configurations.ui_runtime.redirect_path)
