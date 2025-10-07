from __future__ import annotations

from DILIGENT.app.constants import SOURCES_PATH
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.database.sqlite import database
from DILIGENT.app.utils.services.updater import LiverToxUpdater

REDOWNLOAD = True

###############################################################################
if __name__ == "__main__":
    updater = LiverToxUpdater(
        SOURCES_PATH,
        redownload=REDOWNLOAD,
    )
    database.initialize_database()
    logger.info("Running LiverTox updater")
    result = updater.run()
    logger.info("LiverTox updater summary: %s", result)
    

