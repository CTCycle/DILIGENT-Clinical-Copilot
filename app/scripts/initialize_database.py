from __future__ import annotations

import time

from common.utils.logger import logger
from repositories.database.initializer import initialize_database

###############################################################################
if __name__ == "__main__":
    start = time.perf_counter()
    logger.info("Starting database initialization")
    initialize_database()
    elapsed = time.perf_counter() - start
    logger.info("Database initialization completed in %.2f seconds", elapsed)

