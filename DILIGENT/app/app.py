from __future__ import annotations

import os

import gradio as gr
from fastapi import FastAPI
from nicegui import ui

from DILIGENT.app.variables import EnvironmentVariables
from DILIGENT.app.api.endpoints.session import router as report_router
from DILIGENT.app.api.endpoints.ollama import router as models_router
from DILIGENT.app.client.interface import create_interface
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.database import database

EV = EnvironmentVariables()

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
# Serve the UI under /ui while keeping the FastAPI routes available
app = gr.mount_gradio_app(app, ui_app, path="/ui", root_path="/ui")
