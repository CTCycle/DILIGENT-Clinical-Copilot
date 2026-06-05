from __future__ import annotations

from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = Path(__file__).resolve().parents[3]
SETTINGS_PATH = ROOT_DIR / "settings"
RESOURCES_PATH = APP_DIR / "resources"
MODELS_PATH = RESOURCES_PATH / "models"
SOURCES_PATH = RESOURCES_PATH / "sources"
ARCHIVES_PATH = SOURCES_PATH / "archives"
DOCS_PATH = SOURCES_PATH / "documents"
LOGS_PATH = RESOURCES_PATH / "logs"
VECTOR_DB_PATH = SOURCES_PATH / "vectors"
CATALOGS_PATH = RESOURCES_PATH / "catalogs"
RXNAV_CURATED_ALIASES_PATH = SOURCES_PATH / "rxnav_curated_aliases.json"
ENV_FILE_PATH = SETTINGS_PATH / ".env"
CONFIGURATIONS_FILE = SETTINGS_PATH / "configurations.json"
DATABASE_FILE_PATH = RESOURCES_PATH / "database.db"
CLIENT_DIST_PATH = APP_DIR / "client" / "dist" / "browser"
CLIENT_ASSETS_PATH = CLIENT_DIST_PATH / "assets"
CLIENT_INDEX_FILE_PATH = CLIENT_DIST_PATH / "index.html"

__all__ = [
    "APP_DIR",
    "ROOT_DIR",
    "SETTINGS_PATH",
    "RESOURCES_PATH",
    "MODELS_PATH",
    "SOURCES_PATH",
    "ARCHIVES_PATH",
    "DOCS_PATH",
    "LOGS_PATH",
    "VECTOR_DB_PATH",
    "CATALOGS_PATH",
    "RXNAV_CURATED_ALIASES_PATH",
    "ENV_FILE_PATH",
    "CONFIGURATIONS_FILE",
    "DATABASE_FILE_PATH",
    "CLIENT_DIST_PATH",
    "CLIENT_ASSETS_PATH",
    "CLIENT_INDEX_FILE_PATH",
]
