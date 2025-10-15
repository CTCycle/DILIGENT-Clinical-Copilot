from __future__ import annotations

from DILIGENT.app.variables import EnvironmentVariables

EV = EnvironmentVariables()

import os

import gradio as gr
from fastapi import FastAPI

from DILIGENT.app.api.endpoints.session import router as report_router
from DILIGENT.app.api.endpoints.ollama import router as models_router
from DILIGENT.app.client.main import create_interface
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.database import database

###############################################################################
# initialize the database if it has not been created
if not os.path.exists(database.db_path):
    logger.info("Database not found, creating instance and making all tables")
    database.initialize_database()
    logger.info("DILIGENT database has been initialized successfully.")

app = FastAPI(
    title="LLM Backend",
    version="0.1.0",
    description="Minimal FastAPI bootstrap with chat, embeddings, and a placeholder endpoint.",
)

app.include_router(report_router)
app.include_router(models_router)

ui_app = create_interface()
app = gr.mount_gradio_app(app, ui_app, path="/ui", root_path="/ui")
