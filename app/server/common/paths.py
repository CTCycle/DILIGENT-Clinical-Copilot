from __future__ import annotations

import os
from pathlib import Path

from common.runtime_layout import resolve_runtime_layout

RUNTIME_LAYOUT = resolve_runtime_layout()
APP_DIR = RUNTIME_LAYOUT.application_root
ROOT_DIR = RUNTIME_LAYOUT.runtime_root
SETTINGS_PATH = RUNTIME_LAYOUT.settings_root
IMMUTABLE_RESOURCES_PATH = RUNTIME_LAYOUT.immutable_resources_root
RESOURCES_PATH = RUNTIME_LAYOUT.mutable_resources_root
TOOLS_PATH = IMMUTABLE_RESOURCES_PATH / "tools"
SOURCES_PATH = RESOURCES_PATH / "sources"
EMBEDDING_MODELS_PATH = RESOURCES_PATH / "models" / "embeddings"
ARCHIVES_PATH = SOURCES_PATH / "archives"
DOCS_PATH = SOURCES_PATH / "documents"
LOGS_PATH = RESOURCES_PATH / "logs"
VECTOR_DB_PATH = SOURCES_PATH / "vectors"
RAG_ACTIVE_GENERATION_POINTER_PATH = VECTOR_DB_PATH / "active_generation.json"
CATALOGS_PATH = IMMUTABLE_RESOURCES_PATH / "catalogs"
RXNAV_CURATED_ALIASES_PATH = SOURCES_PATH / "rxnav_curated_aliases.json"
ENV_FILE_PATH = SETTINGS_PATH / ".env"
ENV_EXAMPLE_PATH = RUNTIME_LAYOUT.settings_template_root / ".env.example"
CONFIGURATIONS_FILE = SETTINGS_PATH / "configurations.json"
DATABASE_FILE_PATH = Path(
    os.getenv("DILIGENT_SQLITE_PATH") or str(RESOURCES_PATH / "database.db")
)
CLIENT_DIST_PATH = APP_DIR / "client" / "dist" / "browser"
CLIENT_ASSETS_PATH = CLIENT_DIST_PATH / "assets"
CLIENT_INDEX_FILE_PATH = CLIENT_DIST_PATH / "index.html"

__all__ = [
    "APP_DIR",
    "ROOT_DIR",
    "SETTINGS_PATH",
    "RUNTIME_LAYOUT",
    "IMMUTABLE_RESOURCES_PATH",
    "RESOURCES_PATH",
    "TOOLS_PATH",
    "SOURCES_PATH",
    "EMBEDDING_MODELS_PATH",
    "ARCHIVES_PATH",
    "DOCS_PATH",
    "LOGS_PATH",
    "VECTOR_DB_PATH",
    "RAG_ACTIVE_GENERATION_POINTER_PATH",
    "CATALOGS_PATH",
    "RXNAV_CURATED_ALIASES_PATH",
    "ENV_FILE_PATH",
    "ENV_EXAMPLE_PATH",
    "CONFIGURATIONS_FILE",
    "DATABASE_FILE_PATH",
    "CLIENT_DIST_PATH",
    "CLIENT_ASSETS_PATH",
    "CLIENT_INDEX_FILE_PATH",
]
