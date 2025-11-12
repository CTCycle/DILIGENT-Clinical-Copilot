from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from nicegui import ui

from DILIGENT.src.app.backend.endpoints.session import router as session_router
from DILIGENT.src.app.backend.endpoints.ollama import router as ollama_router
from DILIGENT.src.app.frontend.interface import create_interface
from DILIGENT.src.packages.configurations import BACKEND_SETTINGS, UI_RUNTIME_SETTINGS
from DILIGENT.src.packages.logger import logger
from DILIGENT.src.packages.utils.repository.database import database

###############################################################################
# initialize the database if it has not been created
if not os.path.exists(database.db_path):
    logger.info("Database not found, creating instance and making all tables")
    database.initialize_database()
    logger.info("DILIGENT database has been initialized successfully.")

app = FastAPI(
    title=BACKEND_SETTINGS.title,
    version=BACKEND_SETTINGS.version,
    description=BACKEND_SETTINGS.description,
)

app.include_router(session_router)
app.include_router(ollama_router)

create_interface()
ui.run_with(
    app,
    mount_path=UI_RUNTIME_SETTINGS.mount_path,
    title=UI_RUNTIME_SETTINGS.title,
    show_welcome_message=UI_RUNTIME_SETTINGS.show_welcome_message,
    reconnect_timeout=UI_RUNTIME_SETTINGS.reconnect_timeout,
)

@app.get("/")
def redirect_to_ui() -> RedirectResponse:
    return RedirectResponse(url=UI_RUNTIME_SETTINGS.redirect_path)
