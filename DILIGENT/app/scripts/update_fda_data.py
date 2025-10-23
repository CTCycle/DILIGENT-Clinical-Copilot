from __future__ import annotations

from DILIGENT.app.constants import SOURCES_PATH
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.database import database
from DILIGENT.app.utils.updater.fda import FdaUpdater

REDOWNLOAD = True

###############################################################################
if __name__ == "__main__":
    database.initialize_database()
    updater = FdaUpdater(SOURCES_PATH, redownload=REDOWNLOAD)
    logger.info("Running FDA adverse event updater")
    result = updater.update_from_fda()
    logger.info("FDA adverse event updater summary: %s", result)
