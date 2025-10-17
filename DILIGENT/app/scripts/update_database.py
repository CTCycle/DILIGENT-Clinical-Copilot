from __future__ import annotations

from DILIGENT.app.constants import SOURCES_PATH
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.database import database
from DILIGENT.app.utils.updater.livertox import LiverToxUpdater

REDOWNLOAD = True  # Force a fresh pull of LiverTox archives instead of reusing local files

###############################################################################
if __name__ == "__main__":
    # Instantiate the updater with the configured sources directory
    updater = LiverToxUpdater(
        SOURCES_PATH,
        redownload=REDOWNLOAD,
    )
    # Ensure the SQLite schema exists before attempting to write updates
    database.initialize_database()
    logger.info("Running LiverTox updater")
    result = updater.update_from_livertox()
    logger.info("LiverTox updater summary: %s", result)
    

