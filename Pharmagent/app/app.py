from __future__ import annotations

from Pharmagent.app.variables import EnvironmentVariables

EV = EnvironmentVariables()

import os

from fastapi import FastAPI

from Pharmagent.app.api.endpoints.agent import router as report_router
from Pharmagent.app.api.endpoints.ollama import router as models_router
from Pharmagent.app.api.endpoints.pharmacology import router as pharma_router
from Pharmagent.app.logger import logger
from Pharmagent.app.utils.database.sqlite import database

###############################################################################
# initialize the database if it has not been created
if not os.path.exists(database.db_path):
    logger.info("Database not found, creating instance and making all tables")
    database.initialize_database()
    logger.info("Pharmagent database has been initialized successfully.")

app = FastAPI(
    title="LLM Backend",
    version="0.1.0",
    description="Minimal FastAPI bootstrap with chat, embeddings, and a placeholder endpoint.",
)

# Enable CORS later if you need a browser-based frontend
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173", "http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.include_router(report_router)
app.include_router(models_router)
app.include_router(pharma_router)
