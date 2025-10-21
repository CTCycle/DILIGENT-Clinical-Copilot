from __future__ import annotations

from DILIGENT.app.constants import SOURCES_PATH
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.database import database
from DILIGENT.app.utils.updater.fda import FdaUpdater

REDOWNLOAD = True

###############################################################################
def run(redownload: bool = REDOWNLOAD, initialize_db: bool = True) -> None:
    updater = FdaUpdater(
        SOURCES_PATH,
        redownload=redownload,
    )
    if initialize_db:
        database.initialize_database()
    logger.info("Running FDA adverse event updater")
    result = updater.update_from_fda_approvals()
    logger.info("FDA adverse event updater summary: %s", result)


###############################################################################
if __name__ == "__main__":
    run()
