from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from nicegui import ui

from DILIGENT.src.app.backend.endpoints.session import router as session_router
from DILIGENT.src.app.backend.endpoints.ollama import router as ollama_router
from DILIGENT.src.app.frontend.interface import create_interface
from DILIGENT.src.packages.configurations import get_configurations
from DILIGENT.src.packages.logger import logger
from DILIGENT.src.packages.utils.repository.database import database

CONFIG = get_configurations()

###############################################################################
# initialize the database if it has not been created
if not os.path.exists(database.db_path):
    logger.info("Database not found, creating instance and making all tables")
    database.initialize_database()
    logger.info("DILIGENT database has been initialized successfully.")

app = FastAPI(
    title=CONFIG.backend.title,
    version=CONFIG.backend.version,
    description=CONFIG.backend.description,
)

app.include_router(session_router)
app.include_router(ollama_router)

create_interface()
ui.run_with(
    app,
    mount_path=CONFIG.ui_runtime.mount_path,
    title=CONFIG.ui_runtime.title,
    show_welcome_message=CONFIG.ui_runtime.show_welcome_message,
    reconnect_timeout=CONFIG.ui_runtime.reconnect_timeout,
)

@app.get("/")
def redirect_to_ui() -> RedirectResponse:
    return RedirectResponse(url=CONFIG.ui_runtime.redirect_path)
