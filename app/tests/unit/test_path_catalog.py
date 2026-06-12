from __future__ import annotations

from common import constants, paths


###############################################################################
def test_common_constants_no_longer_exposes_shared_paths() -> None:
    forbidden_names = {
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
    }

    assert all(not hasattr(constants, name) for name in forbidden_names)
    assert paths.CONFIGURATIONS_FILE.name == "configurations.json"
    assert paths.CLIENT_INDEX_FILE_PATH.name == "index.html"
