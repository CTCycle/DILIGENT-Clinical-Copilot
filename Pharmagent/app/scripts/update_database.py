from __future__ import annotations

from Pharmagent.app.constants import SOURCES_PATH
from Pharmagent.app.logger import logger
from Pharmagent.app.utils.services.livertox import LiverToxUpdater

REDOWNLOAD = False
CONVERT_TO_DATAFRAME = False

###############################################################################
if __name__ == "__main__":
    updater = LiverToxUpdater(
        SOURCES_PATH,
        redownload=REDOWNLOAD,
    )
    logger.info("Running LiverTox updater")
    result = updater.run()
    logger.info("LiverTox updater summary: %s", result)
    

